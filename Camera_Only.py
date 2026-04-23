#!/usr/bin/env python3
"""
Dual-camera lane follower: **left + right** cameras, white tape on each side; steer to stay
centered, **Forward Half** when |error| is small.

**--lane-detector fitline** (default): HLS mask, ROI, ``cv2.fitLine``.
**--lane-detector histogram**: column-sum peak + hysteresis.

If only one side sees a line, the vehicle creeps forward until too close, then turns away until
the other camera can see the other line.

Use ``--http-preview-port`` for browser preview. The panel is always two rows
(LEFT then RIGHT). Camera mapping uses the proven default: stream opened as ``--right-index``
is assigned to LEFT lane processing, and ``--left-index`` to RIGHT.

Debug: telemetry each loop and optional MJPEG HTTP preview.
"""

import argparse
import signal
import socketserver
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import List, Optional, Tuple

import cv2
import numpy as np
import serial

# ---------------------------
# TM4C serial settings
# ---------------------------
PORT = "COM3" if sys.platform == "win32" else "/dev/ttyACM0"
BAUD = 19200
EOL = "\r"  # TM4C parser ends on c == 13
BOOT_WAIT = 2.0
CHAR_DELAY = 0.01

CMD_FWD = "Forward Half"
CMD_LEFT = "Left Half"
CMD_RIGHT = "Right Half"
CMD_STOP = "Stop"

# Global refs used by signal-based emergency shutdown.
GLOBAL_MC = None
GLOBAL_SER = None
GLOBAL_CAP = None
GLOBAL_CAPS = {}
SHUTDOWN_IN_PROGRESS = False

# ---------------------------
# Camera settings
# ---------------------------
LEFT_CAM_INDEX = 2
RIGHT_CAM_INDEX = 0
# V4L2 is Linux; DirectShow is typical on Windows.
CAM_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
MIRROR_FRAME = False

# Opening two USB cameras on Linux/Jetson: the 2nd often fails without a pause; some
# systems need retries or a different VideoCapture API. Tune via CLI if needed.
CAM_SECOND_OPEN_DELAY_SEC = 0.85

# Default capture (MJPEG); lower res/FPS helps two cameras on one USB hub.
CAM_FRAME_WIDTH = 960
CAM_FRAME_HEIGHT = 540
CAM_FRAME_FPS = 30

# ---------------------------
# White-lane mask settings
# ---------------------------
# Full-frame ROI (used only if narrow_right_roi disabled)
ROI_VERTICES_RATIO = dict(
    bottom_left=(0.05, 0.98),
    top_left=(0.22, 0.45),
    top_right=(0.78, 0.60),
    bottom_right=(0.95, 0.98),
)

# Right-lane search window: ignore left side of image so floor / far lane marks are not picked.
RIGHT_LINE_ROI_X0_RATIO = 0.28
RIGHT_LINE_ROI_Y0_RATIO = 0.42
# Bottom band (within ROI) used for column histogram — where the tape crosses the road plane
LINE_HIST_BAND_H_RATIO = 0.14
# Peak must stand out from median column sum (noise rejection)
HIST_PEAK_MIN_PROMINENCE_FRAC = 0.35
HIST_ABS_MIN_PEAK = 120

HSV_S_MAX = 70
HSV_V_MIN = 170
LAB_A_ABS_MAX = 20
LAB_B_ABS_MAX = 20
Y_MIN = 160
CR_ABS_MAX = 14
CB_ABS_MAX = 14

OPEN_K = (3, 3)
CLOSE_K = (11, 11)

# style (fitLine + HLS mask + trapezoid ROI) ---
PART3_MIN_AREA = 300
PART3_MIN_HEIGHT = 40
PART3_MIN_ASPECT_H_OVER_W = 1.2
PART3_HLS_L_MIN = 170
PART3_HLS_S_MAX = 50

# ---------------------------
# Right-lane avoid tuning (histogram + hysteresis)
# ---------------------------
# x_norm = line_x / frame_width. High = stripe near right edge = you are close to the tape.
# Default matches camera_part3 TARGET_X_RATIO_RIGHT (0.65); histogram mode often uses ~0.52.
RIGHT_LANE_TARGET_RATIO_DEFAULT = 0.90
# For left-camera preview / future left-lane follow (camera_part3 TARGET_X_RATIO_LEFT).
LEFT_LANE_TARGET_RATIO_DEFAULT = 0.10
# Hysteresis: enter corrective left turn when stripe is this far right; exit when back in band.
TURN_LEFT_ENTER_RATIO_DEFAULT = 0.58
TURN_LEFT_EXIT_RATIO_DEFAULT = 0.48
# EMA on line x (pixels) to reduce twitchy steering
# Stabilize lane x from contour jitter (wrong blob / frame-to-frame jumps).
LINE_X_EMA_ALPHA = 0.38
LINE_X_MAX_JUMP_FRAC = 0.14

# Camera dropout tolerance:
# Use latest good frame for short hiccups; then try reopen (USB/V4L2 often recovers).
MAX_STALE_FRAMES_PER_CAM = 18
MAX_CONSECUTIVE_DUAL_MISS = 4
CAM_RECOVER_REOPEN_DELAY_SEC = 0.65
MAX_CAM_RECOVERIES_PER_RUN = 15

# ---------------------------
# Dual-camera center (between left + right lane lines)
# ---------------------------
# e = (nR - tgtR) - (nL - tgtL) with n = line_x / width. e > 0 => too far right in lane => steer LEFT.
CENTER_DEADBAND = 0.20 # decrease to stop going straight earlier
DUAL_KP = 0.90 # reduce if the vehicle snaps too hard when it starts turning
DUAL_KD = 0.80 # if it swings past and rebounds, increase to dampen the swing
DUAL_U_THRESH = 0.7 # increase to trigger a left/right turn sooner
# Single-side: creep forward until line norm exceeds these, then turn away from that line.
SINGLE_LEFT_DANGER_RATIO = 0.40
SINGLE_RIGHT_DANGER_RATIO = 0.50
# When not BOTH and steering wants Left/Right: turn briefly, then Stop so cameras can update.
BLIND_PULSE_TURN_SEC = 0.8
BLIND_PULSE_WAIT_SEC = 2.0

# White mask settings 
def white_mask(bgr: np.ndarray, hsv_v_min: int = HSV_V_MIN) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    h, s, v = cv2.split(hsv)
    _, a, b = cv2.split(lab)
    y, cr, cb = cv2.split(ycc)

    m_hsv = (s <= HSV_S_MAX) & (v >= hsv_v_min)
    m_lab = (np.abs(a.astype(np.int16) - 128) <= LAB_A_ABS_MAX) & (
        np.abs(b.astype(np.int16) - 128) <= LAB_B_ABS_MAX
    )
    m_ycc = (y >= Y_MIN) & (np.abs(cr.astype(np.int16) - 128) <= CR_ABS_MAX) & (
        np.abs(cb.astype(np.int16) - 128) <= CB_ABS_MAX
    )

    return ((m_hsv & m_lab & m_ycc).astype(np.uint8)) * 255

# Morphological cleaning 
def morph_clean(mask: np.ndarray) -> np.ndarray:
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m

# Draw lane reference markers   
def draw_lane_reference_markers(
    overlay: np.ndarray,
    mask_dbg: np.ndarray,
    w: int,
    y_top: int,
    y_bottom: int,
    tx_enter: int,
    tx_exit: int,
    tx_tgt: int,
    target_ratio: float,
    green_thickness: int,
    goal_band_ratio: float,
) -> None:
    """
    Red/orange: thin hysteresis guides. Green: thick target line. Optional dark-green vertical band
    ± ``goal_band_ratio`` of frame width (visual only; helps see tolerance when lane width varies).
    """
    cv2.line(overlay, (tx_enter, y_top), (tx_enter, y_bottom), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(overlay, (tx_exit, y_top), (tx_exit, y_bottom), (0, 165, 255), 1, cv2.LINE_AA)
    if goal_band_ratio > 0.0:
        x_lo = int(np.clip((target_ratio - goal_band_ratio) * w, 0, w - 1))
        x_hi = int(np.clip((target_ratio + goal_band_ratio) * w, 0, w - 1))
        if x_hi > x_lo:
            sub = overlay[y_top : y_bottom + 1, x_lo : x_hi + 1]
            tint = np.zeros_like(sub)
            tint[:] = (0, 55, 0)
            cv2.addWeighted(sub, 0.76, tint, 0.24, 0, sub)
            subm = mask_dbg[y_top : y_bottom + 1, x_lo : x_hi + 1]
            tint_m = np.zeros_like(subm)
            tint_m[:] = (0, 55, 0)
            cv2.addWeighted(subm, 0.76, tint_m, 0.24, 0, subm)
    gt = max(1, int(green_thickness))
    cv2.line(overlay, (tx_tgt, y_top), (tx_tgt, y_bottom), (0, 255, 0), gt, cv2.LINE_AA)
    cv2.line(mask_dbg, (tx_enter, y_top), (tx_enter, y_bottom), (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(mask_dbg, (tx_exit, y_top), (tx_exit, y_bottom), (0, 165, 255), 1, cv2.LINE_AA)
    cv2.line(mask_dbg, (tx_tgt, y_top), (tx_tgt, y_bottom), (0, 255, 0), gt, cv2.LINE_AA)


def region_selection_binary(mask_u8: np.ndarray) -> np.ndarray: 
    """Apply trapezoid ROI to a 2D binary mask (same geometry as camera_part3)."""
    h, w = mask_u8.shape[:2]
    poly = np.zeros_like(mask_u8)
    bl = (
        int(w * ROI_VERTICES_RATIO["bottom_left"][0]),
        int(h * ROI_VERTICES_RATIO["bottom_left"][1]),
    )
    tl = (
        int(w * ROI_VERTICES_RATIO["top_left"][0]),
        int(h * ROI_VERTICES_RATIO["top_left"][1]),
    )
    tr = (
        int(w * ROI_VERTICES_RATIO["top_right"][0]),
        int(h * ROI_VERTICES_RATIO["top_right"][1]),
    )
    br = (
        int(w * ROI_VERTICES_RATIO["bottom_right"][0]),
        int(h * ROI_VERTICES_RATIO["bottom_right"][1]),
    )
    pts = np.array([[bl, tl, tr, br]], dtype=np.int32)
    cv2.fillPoly(poly, pts, 255)
    return cv2.bitwise_and(mask_u8, poly)


def white_mask_hls( # HLS lightness + low saturation picks white tape.
    bgr: np.ndarray, l_min: int = PART3_HLS_L_MIN, s_max: int = PART3_HLS_S_MAX
) -> np.ndarray:
    """camera_part3: HLS lightness + low saturation picks white tape."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    _, l_ch, s_ch = cv2.split(hls)
    return ((l_ch > l_min) & (s_ch < s_max)).astype(np.uint8) * 255


def detect_line_x_fitline( 
    frame_bgr: np.ndarray,
    l_min: int = PART3_HLS_L_MIN,
    s_max: int = PART3_HLS_S_MAX,
    lane_roi_side: str = "right",
) -> Tuple[Optional[float], np.ndarray, str]:
    """
    camera_part3 lane detection: full-frame HLS mask -> trapezoid ROI -> optional **left/right
    half-plane gate** (same geometry as histogram mode) -> contours -> merge tall fragments ->
    cv2.fitLine -> x at bottom row.

    ``lane_roi_side`` must match which stripe each camera should track: *left* keeps only
    mask pixels in the left search band so the LEFT camera does not latch the opposite lane line;
    *right* keeps the right band for the RIGHT camera.
    """
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1

    full_mask = white_mask_hls(frame_bgr, l_min=l_min, s_max=s_max)
    mask_white = region_selection_binary(full_mask)
    mask_white = morph_clean(mask_white)
    x0, x1, y0, y1 = _lane_search_roi_slice(h, w, lane_roi_side)
    side_gate = np.zeros_like(mask_white)
    side_gate[y0:y1, x0:x1] = 255
    mask_white = cv2.bitwise_and(mask_white, side_gate)

    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lane_points: List[np.ndarray] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PART3_MIN_AREA:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, cy), (rw, rh), _ = rect
        height = max(rw, rh)
        width = min(rw, rh)
        aspect = height / max(1.0, width)
        if height > PART3_MIN_HEIGHT and aspect > PART3_MIN_ASPECT_H_OVER_W:
            lane_points.append(cnt)

    line_x_bottom: Optional[float] = None
    if len(lane_points) > 0:
        all_pts = np.vstack(lane_points)
        if len(all_pts) >= 2:
            vx, vy, x0, y0 = cv2.fitLine(all_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(vx[0])
            vy = float(vy[0])
            x0 = float(x0[0])
            y0 = float(y0[0])
            if abs(vy) > 1e-6:
                line_x_bottom = float(x0 + (vx / vy) * (y_bottom - y0))
                # Extrapolated fits can land outside the image when the tape curves at the edge.
                line_x_bottom = float(np.clip(line_x_bottom, 0.0, float(w - 1)))

    tag = "fitline_ok" if line_x_bottom is not None else "fitline_none"
    return line_x_bottom, mask_white, tag

# Horizontal band for column histogram: ``right`` = right lane line; ``left`` = left lane line.
def _lane_search_roi_slice(
    h: int, w: int, roi_side: str = "right"
) -> Tuple[int, int, int, int]:
    """Horizontal band for column histogram: ``right`` = right lane line; ``left`` = left lane line."""
    y0 = int(h * RIGHT_LINE_ROI_Y0_RATIO)
    y1 = h
    if roi_side == "left":
        x0 = 0
        x1 = int(w * (1.0 - RIGHT_LINE_ROI_X0_RATIO))
    else:
        x0 = int(w * RIGHT_LINE_ROI_X0_RATIO)
        x1 = w
    return x0, x1, y0, y1

# Lane stripe x from column sum in a left or right ROI.
def detect_line_x_histogram(
    frame_bgr: np.ndarray,
    hsv_v_min: int = HSV_V_MIN,
    roi_side: str = "right",
) -> Tuple[Optional[float], np.ndarray, str]:
    """
    Lane stripe x from column sum in a left or right ROI.
    Returns (line_x in full-image coords, binary mask uint8 full size, reason tag).
    """
    h, w = frame_bgr.shape[:2]
    x0, x1, y0, y1 = _lane_search_roi_slice(h, w, roi_side)
    roi_view = frame_bgr[y0:y1, x0:x1]
    if roi_view.size == 0:
        return None, np.zeros((h, w), dtype=np.uint8), "empty_roi"

    wm = white_mask(roi_view, hsv_v_min=hsv_v_min)
    wm = morph_clean(wm)

    full_bin = np.zeros((h, w), dtype=np.uint8)
    full_bin[y0:y1, x0:x1] = wm

    band_h = max(4, int(h * LINE_HIST_BAND_H_RATIO))
    y_lo = max(y0, y1 - band_h)
    strip = full_bin[y_lo:y1, x0:x1]
    if strip.size == 0:
        return None, full_bin, "empty_strip"

    col_sum = strip.sum(axis=0).astype(np.float64)
    if col_sum.size < 3:
        return None, full_bin, "narrow"

    peak_val = float(col_sum.max())
    med = float(np.median(col_sum))
    prom = peak_val - med
    if peak_val < HIST_ABS_MIN_PEAK:
        return None, full_bin, "weak_peak"
    if prom < HIST_PEAK_MIN_PROMINENCE_FRAC * max(peak_val, 1.0):
        return None, full_bin, "low_prominence"

    x_rel = int(np.argmax(col_sum))
    pw = max(3, int(0.02 * (x1 - x0)))
    a = max(0, x_rel - pw)
    b = min(col_sum.size, x_rel + pw + 1)
    wts = col_sum[a:b]
    idxs = np.arange(a, b, dtype=np.float64)
    s = float(wts.sum())
    if s < 1e-6:
        x_f = float(x_rel)
    else:
        x_f = float(np.dot(idxs, wts) / s)
    line_x = float(np.clip(x0 + x_f, 0.0, float(w - 1)))
    return line_x, full_bin, "ok"

# Open serial port and set DTR and RTS (DTR: Data Terminal Ready, RTS: Request to Send)
def open_serial(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(
        port,
        baudrate=baud,
        timeout=0.5,
        write_timeout=1.0,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
    )
    ser.setDTR(True)
    ser.setRTS(True)
    time.sleep(BOOT_WAIT)
    return ser

# Send line typewriter (Send a single command string, wait a bit, then send it again)
def send_line_typewriter(ser: serial.Serial, text: str): 
    data = (text + EOL).encode("ascii", errors="ignore")
    for b in data:
        ser.write(bytes([b]))
        ser.flush()
        time.sleep(CHAR_DELAY)

# MotorCommander class (Send commands to the motor controller)
class MotorCommander:
    def __init__(self, ser: serial.Serial, min_cmd_interval: float = 0.08):
        self.ser = ser
        self.last_cmd = None
        self.last_send_time = 0.0
        self.min_cmd_interval = min_cmd_interval

    def send(self, cmd: str, force: bool = False):
        now = time.time()
        same_cmd = cmd == self.last_cmd
        too_soon = (now - self.last_send_time) < self.min_cmd_interval
        if force or (not same_cmd) or (not too_soon):
            send_line_typewriter(self.ser, cmd)
            self.last_cmd = cmd
            self.last_send_time = now

# Best-effort redundant stop path: 1) high-level forced command 2) raw serial fallback (twice)
def force_stop_motors(mc, ser):
    # Best-effort redundant stop path:
    # 1) high-level forced command
    # 2) raw serial fallback (twice)
    try:
        if mc is not None and getattr(mc, "ser", None) is not None and mc.ser.is_open:
            mc.send(CMD_STOP, force=True)
    except Exception as e:
        print(f"[WARN] mc force-stop failed: {e}")

    try:
        if ser is not None and ser.is_open:
            send_line_typewriter(ser, CMD_STOP)
            time.sleep(0.05)
            send_line_typewriter(ser, CMD_STOP)
            time.sleep(0.05)
            ser.flush()
            time.sleep(0.12)
    except Exception as e:
        print(f"[WARN] raw serial stop failed: {e}")

# Emergency shutdown (Ctrl+C)
def emergency_shutdown(signum, frame):
    del frame
    global SHUTDOWN_IN_PROGRESS
    if SHUTDOWN_IN_PROGRESS:
        raise SystemExit(0)

    SHUTDOWN_IN_PROGRESS = True
    print(f"\n[SIGNAL] Caught signal {signum}, attempting safe stop...")

    try: # Try to stop the motors
        force_stop_motors(GLOBAL_MC, GLOBAL_SER)
    except Exception:
        pass

    try: # Try to release the cameras
        for side_cap in GLOBAL_CAPS.values():
            if side_cap is not None:
                side_cap.release()
    except Exception:
        pass

    try: # Try to destroy the windows
        cv2.destroyAllWindows()
    except Exception:
        pass

    raise SystemExit(0)

# Backend order to try when opening /dev/videoN (Linux often needs V4L2 then fallback).
def _video_capture_backends():
    """Backend order to try when opening /dev/videoN (Linux often needs V4L2 then fallback)."""
    if sys.platform == "win32":
        return [CAM_BACKEND]
    out = [cv2.CAP_V4L2]
    cap_any = getattr(cv2, "CAP_ANY", None)
    if cap_any is not None:
        out.append(cap_any)
    out.append(None)
    seen = set()
    uniq = []
    for b in out:
        key = id(b) if b is None else b
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)
    return uniq

# Open a camera by index (Linux often needs V4L2 then fallback).
def open_camera(
    index: int,
    label: str,
    width: int = CAM_FRAME_WIDTH,
    height: int = CAM_FRAME_HEIGHT,
    fps: int = CAM_FRAME_FPS,
) -> Optional[cv2.VideoCapture]:
    for api in _video_capture_backends(): # Try to open the camera with the available backends
        try:
            if api is None: # If the backend is None, use the default VideoCapture
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, api) # If the backend is not None, use the backend
        except Exception: 
            continue
        time.sleep(0.12)
        if not cap.isOpened(): # If the camera is not opened, try to release it
            try:
                cap.release()
            except Exception:
                pass
            continue

        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
            ret, _ = cap.read()
            if not ret: # If the camera is not read, try to release it
                cap.release()
                continue
        except Exception: # If the camera is not read, try to release it
            try:
                cap.release()
            except Exception:
                pass
            continue

        api_tag = "default" if api is None else str(api)
        print( # Print the camera information
            f"[CAM] Using {label} camera index={index} (backend={api_tag}) "
            f"{width}x{height}@{fps}"
        )
        return cap

    print(f"[CAM] Failed to open {label} camera at index {index} (all backends)")
    return None

# 
def print_camera_troubleshooting():
    print(
        "[CAM] Second camera not available. Common causes on Jetson/Linux:\n"
        "  • USB bandwidth: use a powered USB3 hub, shorter cables, or lower resolution in code.\n"
        "  • Device nodes: run  v4l2-ctl --list-devices  and  ls -la /dev/video*\n"
        "    (some indices are metadata, not capture — try --right-index explicitly).\n"
        "  • Unplug/replug both cameras; try  --camera-second-delay 1.5  or higher.\n"
        "  • Another process may hold the device: close other camera apps."
    )

# Capture dimensions for LEFT or RIGHT; right may use smaller WxH to save USB bandwidth.
def capture_dims(args, side: str) -> Tuple[int, int, int]:
    """(width, height, fps) for both LEFT and RIGHT streams."""
    del side
    return args.cam_width, args.cam_height, int(args.cam_fps)

# Unique indices
def uniq_indices(indices: List[int]) -> List[int]:
    seen = set()
    out = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out

# Open a side camera with fallback (Open a camera by index (Linux often needs V4L2 then fallback).)
def open_side_camera_with_fallback(
    side: str, # LEFT or RIGHT
    preferred_index: int, # The preferred index to open
    scan_indices: List[int], # The indices to scan
    avoid_index: Optional[int] = None, # Avoid the index of the other camera
    width: int = CAM_FRAME_WIDTH, # The width of the camera
    height: int = CAM_FRAME_HEIGHT, # The height of the camera
    fps: int = CAM_FRAME_FPS,
) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]: # Return the camera and the index
    candidates = [preferred_index] + scan_indices
    candidates = uniq_indices(candidates) 
    if avoid_index is not None:
        candidates = [i for i in candidates if i != avoid_index]
    # Try to open the camera with the available indices
    for idx in candidates:
        cap = open_camera(idx, side, width, height, fps)
        if cap is not None: # If the camera is opened, return the camera and the index
            return cap, idx
    return None, None # If the camera is not opened, return None

# Initialize the dual lane cameras (Open RIGHT index first, then LEFT index (pause between))
def initialize_dual_lane_cameras(args, scan_indices: List[int]):
    """
    Open RIGHT index first, then LEFT index (pause between) for dual-lane centering. Both required.

    The VideoCapture opened with ``--right-index`` is wired to **LEFT** lane detection and
    ``--left-index`` to **RIGHT**. This matches the physical camera layout used on this robot.
    """
    rw, rh, rf = capture_dims(args, "RIGHT")
    cap_r, idx_r = open_side_camera_with_fallback(
        "RIGHT",
        preferred_index=args.right_index,
        scan_indices=scan_indices,
        avoid_index=None,
        width=rw,
        height=rh,
        fps=rf,
    )
    if cap_r is None:
        return None, None
    time.sleep(CAM_SECOND_OPEN_DELAY_SEC)
    lw, lh, lf = args.cam_width, args.cam_height, int(args.cam_fps)
    cap_l, idx_l = open_side_camera_with_fallback(
        "LEFT",
        preferred_index=args.left_index,
        scan_indices=scan_indices,
        avoid_index=idx_r,
        width=lw,
        height=lh,
        fps=lf,
    )
    if cap_l is None:
        try:
            cap_r.release()
        except Exception:
            pass
        return None, None
    return {"LEFT": cap_r, "RIGHT": cap_l}, {"LEFT": idx_r, "RIGHT": idx_l}

# Classify the lane visibility (BOTH: both lines are visible, LEFT_ONLY: only left line is visible, RIGHT_ONLY: only right line is visible, NONE: no lines are visible)
def classify_lane_visibility(
    line_L: Optional[float], line_R: Optional[float],
) -> str:
    hl = line_L is not None
    hr = line_R is not None
    if hl and hr:
        return "BOTH"
    if hl:
        return "LEFT_ONLY"
    if hr:
        return "RIGHT_ONLY"
    return "NONE"

# Center error for both lanes (Positive => vehicle too far right in lane => steer left to correct.)
def center_error_both(nL: float, nR: float, tgtL: float, tgtR: float) -> float:
    """Positive => vehicle too far right in lane => steer left to correct."""
    return (nR - tgtR) - (nL - tgtL)

# Choose the steering command (BOTH: PD-like u on center error; |e| small => forward. ONE SIDE: forward until visible line norm exceeds danger ratio, then turn away from that line; creep forward otherwise until the other camera picks up the other line. NONE: stop (dual-center main loop overrides: coast last Left/Right until a line returns).)
def choose_steering_dual_center(
    vis: str,
    e: float, # Center error
    de: float, # Center error derivative    
    nL: Optional[float], # Line norm for left lane
    nR: Optional[float], # Line norm for right lane
    danger_L: float, # Danger ratio for left lane
    danger_R: float, # Danger ratio for right lane
    center_db: float, # Center deadband
    u_thresh: float, # Threshold for PD-like u
) -> Tuple[str, str]: # Return the steering command and the logic
    """
    BOTH: PD-like u on center error; |e| small => forward.
    ONE SIDE: forward until visible line norm exceeds danger ratio, then turn away from that line;
    creep forward otherwise until the other camera picks up the other line.
    NONE: stop (dual-center main loop overrides: coast last Left/Right until a line returns).
    """
    if vis == "BOTH": # If both lines are visible, use PD-like u on center error
        u = DUAL_KP * e + DUAL_KD * de
        u = float(np.clip(u, -1.0, 1.0))
        if abs(e) < center_db: # If the center error is small, go forward
            return CMD_FWD, "CENTER_FWD"
        if u > u_thresh: # If the u is positive, steer left
            return CMD_LEFT, "DUAL_L"
        if u < -u_thresh: # If the u is negative, steer right   
            return CMD_RIGHT, "DUAL_R"
        return CMD_FWD, "CENTER_FWD"
    if vis == "LEFT_ONLY": # If only the left line is visible, creep forward until the visible line norm exceeds danger ratio
        assert nL is not None
        if nL >= danger_L: # If the visible line norm exceeds danger ratio, turn away from the left line
            return CMD_RIGHT, "AWAY_LEFT_LINE"
        return CMD_FWD, "CREEP_LEFT"
    if vis == "RIGHT_ONLY": # If only the right line is visible, creep forward until the visible line norm exceeds danger ratio
        assert nR is not None
        if nR >= danger_R: # If the visible line norm exceeds danger ratio, turn away from the right line
            return CMD_LEFT, "AWAY_RIGHT_LINE"
        return CMD_FWD, "CREEP_RIGHT"
    return CMD_STOP, "LOST" # If no lines are visible, stop

# Apply the blind turn pulse (If ``vis != BOTH`` and the controller wants Left/Right, alternate: turn for ``turn_sec``, then Stop for ``wait_sec``, repeating until BOTH is visible again. Reduces over-correction when vision lags the vehicle.)
def apply_blind_turn_pulse(
    st: "ControllerState",
    vis: str, # Lane visibility
    desired_cmd: str, # Desired command
    logic: str,
    now_mono: float, # Current time in seconds
    turn_sec: float, # Time to turn in seconds
    wait_sec: float, # Time to wait in seconds
    enabled: bool,
) -> Tuple[str, str]:
    """
    If ``vis != BOTH`` and the controller wants Left/Right, alternate: turn for ``turn_sec``,
    then Stop for ``wait_sec``, repeating until BOTH is visible again. Reduces over-correction when
    vision lags the vehicle.
    """
    if not enabled or vis == "BOTH": # If the lane visibility is BOTH, stop the blind turn pulse
        st.blind_pulse_until = 0.0
        return desired_cmd, logic

    if desired_cmd not in (CMD_LEFT, CMD_RIGHT): # If the desired command is not Left or Right, stop the blind turn pulse
        st.blind_pulse_until = 0.0
        return desired_cmd, logic

    if st.blind_pulse_until <= 0.0 or now_mono >= st.blind_pulse_until: # If the blind turn pulse is not active, start the blind turn pulse
        if st.blind_pulse_until <= 0.0:
            st.blind_in_turn_phase = True
        else: # If the blind turn pulse is active, toggle the turn phase
            st.blind_in_turn_phase = not st.blind_in_turn_phase
        dur = turn_sec if st.blind_in_turn_phase else wait_sec
        st.blind_pulse_until = now_mono + float(dur)

    if st.blind_in_turn_phase:
        return desired_cmd, f"{logic}|PULSE_TURN"
    return CMD_STOP, f"{logic}|PULSE_WAIT"

# Try to recover one camera (Release and reopen one VideoCapture after V4L2 stops delivering frames (errno 19, stale reads). Tries the last known index first, then scan_indices (same as cold start).)
def try_recover_one_camera(
    caps: dict,
    used_idxs: dict,
    side: str,
    scan_indices: List[int],
    args,
) -> bool:
    """
    Release and reopen one VideoCapture after V4L2 stops delivering frames (errno 19, stale reads).
    Tries the last known index first, then scan_indices (same as cold start).
    """
    global GLOBAL_CAPS
    other = "RIGHT" if side == "LEFT" else "LEFT"
    prior_idx = used_idxs[side]
    other_idx = used_idxs.get(other)
    print(
        f"[CAM] Recover: reopening {side} camera (last index {prior_idx}, other={other_idx})..."
    )
    old = caps.get(side)
    if old is not None:
        try:
            old.release()
        except Exception:
            pass
    caps[side] = None
    time.sleep(CAM_RECOVER_REOPEN_DELAY_SEC)

    w, h, fps = capture_dims(args, side)
    cap = open_camera(prior_idx, f"{side}_recover", w, h, fps)
    if cap is None:
        cap, idx = open_side_camera_with_fallback(
            side,
            preferred_index=prior_idx,
            scan_indices=scan_indices,
            avoid_index=other_idx,
            width=w,
            height=h,
            fps=fps,
        )
        if cap is not None:
            used_idxs[side] = idx
    if cap is None:
        print("[CAM] Recover failed: could not reopen this camera.")
        return False

    caps[side] = cap
    GLOBAL_CAPS[side] = cap
    print(f"[CAM] Recover OK: {side} now on index={used_idxs[side]}")
    return True

# Process a frame for histogram lane detection (Left or right ROI + histogram stripe finder. Overlay draws search ROI, reference lines, and detected x. err_norm is (x - target) / (w/2) for telemetry.)
def process_frame_histogram(
    frame_bgr: np.ndarray,
    target_ratio: float, # Target ratio
    turn_enter_ratio: float, # Turn enter ratio
    turn_exit_ratio: float, # Turn exit ratio
    hsv_v_min: int = HSV_V_MIN, # HSV V min
    lane_roi_side: str = "right", # Lane ROI side
    green_line_thickness: int = 5, # Green line thickness
    target_goal_band_ratio: float = 0.0, # Target goal band ratio
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float, str]: # Return the overlay, mask, line x, error norm, and detection tag
    """
    Left or right ROI + histogram stripe finder. Overlay draws search ROI, reference lines,
    and detected x. err_norm is (x - target) / (w/2) for telemetry.
    """
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top = int(h * ROI_VERTICES_RATIO["top_left"][1])
    x0, x1, y0, y1 = _lane_search_roi_slice(h, w, lane_roi_side)

    line_x_bottom, bin_mask, det_tag = detect_line_x_histogram(
        frame_bgr, hsv_v_min=hsv_v_min, roi_side=lane_roi_side
    )
    overlay = frame_bgr.copy()
    mask_dbg = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 2)
    cv2.rectangle(mask_dbg, (x0, y0), (x1 - 1, y1 - 1), (255, 200, 0), 2)

    tx_enter = int(np.clip(turn_enter_ratio * w, 0, w - 1))
    tx_exit = int(np.clip(turn_exit_ratio * w, 0, w - 1))
    tx_tgt = int(np.clip(target_ratio * w, 0, w - 1))
    draw_lane_reference_markers(
        overlay,
        mask_dbg,
        w,
        y_top,
        y_bottom,
        tx_enter,
        tx_exit,
        tx_tgt,
        target_ratio,
        green_line_thickness,
        target_goal_band_ratio,
    )

    if line_x_bottom is not None:
        ix = int(np.clip(line_x_bottom, 0, w - 1))
        cv2.circle(overlay, (ix, y_bottom - 6), 7, (0, 0, 255), -1)
        cv2.circle(mask_dbg, (ix, y_bottom - 6), 7, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            det_tag,
            (x0 + 4, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    target_x = target_ratio * w
    if line_x_bottom is None:
        err_norm = 0.0
    else:
        err_px = float(line_x_bottom) - target_x
        err_norm = float(np.clip(err_px / (w / 2), -1.0, 1.0))

    return overlay, mask_dbg, line_x_bottom, err_norm, det_tag

# Process a frame for fitline lane detection (HLS mask + fitLine x + same reference lines as histogram mode.)
def process_frame_fitline(
    frame_bgr: np.ndarray,
    target_ratio: float, # Target ratio
    turn_enter_ratio: float, # Turn enter ratio
    turn_exit_ratio: float, # Turn exit ratio
    hls_l_min: int = PART3_HLS_L_MIN, # HLS L min
    hls_s_max: int = PART3_HLS_S_MAX, # HLS S max
    lane_roi_side: str = "right", # Lane ROI side
    green_line_thickness: int = 5, # Green line thickness
    target_goal_band_ratio: float = 0.0, # Target goal band ratio
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float, str]: # Return the overlay, mask, line x, error norm, and detection tag
    """camera_part3 visualization: HLS mask + fitLine x + same reference lines as histogram mode."""
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top = int(h * ROI_VERTICES_RATIO["top_left"][1])

    line_x_bottom, mask_white, det_tag = detect_line_x_fitline(
        frame_bgr,
        l_min=hls_l_min,
        s_max=hls_s_max,
        lane_roi_side=lane_roi_side,
    )
    overlay = frame_bgr.copy()
    mask_dbg = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)

    rx0, rx1, ry0, ry1 = _lane_search_roi_slice(h, w, lane_roi_side)
    cv2.rectangle(overlay, (rx0, ry0), (rx1 - 1, ry1 - 1), (255, 200, 0), 2)
    cv2.rectangle(mask_dbg, (rx0, ry0), (rx1 - 1, ry1 - 1), (255, 200, 0), 2)

    tx_enter = int(np.clip(turn_enter_ratio * w, 0, w - 1))
    tx_exit = int(np.clip(turn_exit_ratio * w, 0, w - 1))
    tx_tgt = int(np.clip(target_ratio * w, 0, w - 1))
    draw_lane_reference_markers(
        overlay,
        mask_dbg,
        w,
        y_top,
        y_bottom,
        tx_enter,
        tx_exit,
        tx_tgt,
        target_ratio,
        green_line_thickness,
        target_goal_band_ratio,
    )

    cv2.line(overlay, (0, y_top), (w - 1, y_top), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(mask_dbg, (0, y_top), (w - 1, y_top), (255, 0, 0), 1, cv2.LINE_AA)

    if line_x_bottom is not None:
        ix = int(np.clip(line_x_bottom, 0, w - 1))
        cv2.circle(overlay, (ix, y_bottom - 6), 7, (0, 0, 255), -1)
        cv2.circle(mask_dbg, (ix, y_bottom - 6), 7, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            det_tag,
            (10, y_top + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    target_x = target_ratio * w
    if line_x_bottom is None:
        err_norm = 0.0
    else:
        err_px = float(line_x_bottom) - target_x
        err_norm = float(np.clip(err_px / (w / 2), -1.0, 1.0))

    return overlay, mask_dbg, line_x_bottom, err_norm, det_tag

# Process a frame for lane detection (Fitline or histogram mode.)
def process_frame(
    frame_bgr: np.ndarray,
    lane_detector: str,
    target_ratio: float,
    turn_enter_ratio: float,
    turn_exit_ratio: float,
    hsv_v_min: int = HSV_V_MIN,
    hls_l_min: int = PART3_HLS_L_MIN,
    hls_s_max: int = PART3_HLS_S_MAX,
    lane_roi_side: str = "right",
    green_line_thickness: int = 5,
    target_goal_band_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float, str]:
    if lane_detector == "fitline":
        return process_frame_fitline(
            frame_bgr,
            target_ratio,
            turn_enter_ratio,
            turn_exit_ratio,
            hls_l_min=hls_l_min,
            hls_s_max=hls_s_max,
            lane_roi_side=lane_roi_side,
            green_line_thickness=green_line_thickness,
            target_goal_band_ratio=target_goal_band_ratio,
        )
    return process_frame_histogram(
        frame_bgr,
        target_ratio,
        turn_enter_ratio,
        turn_exit_ratio,
        hsv_v_min=hsv_v_min,
        lane_roi_side=lane_roi_side,
        green_line_thickness=green_line_thickness,
        target_goal_band_ratio=target_goal_band_ratio,
    )

# Smooth the line x (EMA on bottom line x with per-frame jump limit (reduces zig-zag when the detector locks onto different parts of the lane or noise).)
def smooth_line_x_ema(
    prev: Optional[float], # Previous line x
    meas: Optional[float], # Measured line x
    w: int, # Width
    alpha: float = LINE_X_EMA_ALPHA,
    max_jump_frac: float = LINE_X_MAX_JUMP_FRAC,
) -> Optional[float]:
    """
    EMA on bottom line x with per-frame jump limit (reduces zig-zag when the
    detector locks onto different parts of the lane or noise).
    """
    if meas is None:
        return None
    m = float(meas)
    if prev is None:
        return m
    max_jump = max_jump_frac * float(w)
    prev_f = float(prev)
    m = prev_f + float(np.clip(m - prev_f, -max_jump, max_jump))
    return alpha * m + (1.0 - alpha) * prev_f

# Controller state (Last command, line x EMA, center error, lost both frames, read phase, last lateral command, blind pulse until, blind in turn phase.)
@dataclass
class ControllerState:
    last_cmd: str = CMD_STOP
    line_right_ema: Optional[float] = None
    line_left_ema: Optional[float] = None
    center_prev_e: float = 0.0
    lost_both_frames: int = 0
    read_phase: int = 0
    last_lateral_cmd: Optional[str] = None  # CMD_LEFT or CMD_RIGHT; used when vis==NONE (dual-center)
    blind_pulse_until: float = 0.0
    blind_in_turn_phase: bool = True

# Latest JPEG for HTTP MJPEG clients (thread-safe).
class MjpegPreviewState:
    """Latest JPEG for HTTP MJPEG clients (thread-safe)."""

    __slots__ = ("lock", "jpeg_bytes")

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes: Optional[bytes] = None

    def set_jpeg(self, data: bytes):
        with self.lock:
            self.jpeg_bytes = data

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.jpeg_bytes

# Make a MJPEG handler (HTTP handler for MJPEG preview.)
def make_mjpeg_handler(preview: MjpegPreviewState):
    boundary = b"--jpgboundary"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/view"):
                body = (
                    "<!DOCTYPE html><html><head><meta charset=utf-8>"
                    "<title>CameraOnly preview</title></head><body>"
                    "<p>CameraOnly live (MJPEG). Use SSH: "
                    "<code>ssh -L 8765:127.0.0.1:8765 user@jetson</code> "
                    "then open <code>http://127.0.0.1:8765/</code></p>"
                    '<img src="/stream" style="max-width:100%;height:auto;" />'
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=jpgboundary"
            )
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            while True:
                jpg = preview.get_jpeg()
                if jpg:
                    try:
                        self.wfile.write(
                            boundary
                            + b"\r\nContent-Type: image/jpeg\r\n"
                            + b"Content-Length: "
                            + str(len(jpg)).encode("ascii")
                            + b"\r\n\r\n"
                            + jpg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        break
                time.sleep(0.03)

        def log_message(self, format, *log_args):
            pass

    return Handler

# Start a MJPEG server (Start a thread to serve MJPEG preview on the given host and port.)
def start_mjpeg_server(
    preview: MjpegPreviewState, host: str, port: int
) -> Optional[socketserver.TCPServer]:
    if port <= 0:
        return None
    handler = make_mjpeg_handler(preview)
    server = socketserver.ThreadingTCPServer((host, port), handler)
    server.daemon_threads = True
    server.allow_reuse_address = True
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(
        f"[HTTP] MJPEG preview on http://{host}:{port}/  (stream: /stream) "
        f"— from PC: ssh -L {port}:127.0.0.1:{port} user@jetson"
    )
    return server

# Apply steer inversion (Invert the steering command if the invert flag is set.)
def apply_steer_inversion(cmd: str, invert: bool) -> str:
    if not invert:
        return cmd
    if cmd == CMD_LEFT:
        return CMD_RIGHT
    if cmd == CMD_RIGHT:
        return CMD_LEFT
    return cmd

# Build a single camera debug panel (Build a single camera debug panel: raw|mask|overlay with caption.)
def build_single_cam_debug_panel(
    frame: np.ndarray,
    mask_dbg: np.ndarray,
    overlay: np.ndarray,
    caption: str,
    thumb_w: int = 480,
    thumb_h: int = 270,
) -> np.ndarray:
    raw = cv2.resize(frame, (thumb_w, thumb_h))
    mask = cv2.resize(mask_dbg, (thumb_w, thumb_h))
    ov = cv2.resize(overlay, (thumb_w, thumb_h))
    row = np.hstack((raw, mask, ov))
    cv2.putText(
        row,
        caption,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )
    return row

# Build a dual camera debug panel (Two rows: LEFT (top) then RIGHT (bottom), each raw|mask|overlay — best for browser scrolling.)
def build_dual_cam_debug_panel(
    top_frame: np.ndarray,
    top_mask_dbg: np.ndarray,
    top_overlay: np.ndarray,
    top_caption: str,
    bot_frame: np.ndarray,
    bot_mask_dbg: np.ndarray,
    bot_overlay: np.ndarray,
    bot_caption: str,
    thumb_w: int = 480,
    thumb_h: int = 270,
) -> np.ndarray:
    """Two rows: LEFT (top) then RIGHT (bottom), each raw|mask|overlay — best for browser scrolling."""
    row1 = build_single_cam_debug_panel(
        top_frame, top_mask_dbg, top_overlay, top_caption, thumb_w, thumb_h
    )
    row2 = build_single_cam_debug_panel(
        bot_frame, bot_mask_dbg, bot_overlay, bot_caption, thumb_w, thumb_h
    )
    w = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] != w:
        row1 = cv2.resize(row1, (w, int(row1.shape[0] * w / row1.shape[1])))
    if row2.shape[1] != w:
        row2 = cv2.resize(row2, (w, int(row2.shape[0] * w / row2.shape[1])))
    return np.vstack((row1, row2))

# Build a dual camera debug panel (One row: [LEFT raw|mask|ov] [RIGHT raw|mask|ov] so both cameras fit in a browser without vertical scrolling (MJPEG / remote viewing).)
def main():
    ap = argparse.ArgumentParser(
        description="Dual-camera lane centering: fitLine (default) or histogram lane detection"
    )
    ap.add_argument("--port", default=PORT, help="TM4C serial port")
    ap.add_argument("--baud", type=int, default=BAUD, help="TM4C serial baud")
    ap.add_argument(
        "--right-index",
        type=int,
        default=RIGHT_CAM_INDEX,
        help="OpenCV index for the RIGHT-side lane camera",
    )
    ap.add_argument(
        "--left-index",
        type=int,
        default=LEFT_CAM_INDEX,
        help="OpenCV index for the LEFT-side lane camera",
    )
    ap.add_argument(
        "--left-lane-target",
        type=float,
        default=LEFT_LANE_TARGET_RATIO_DEFAULT,
        help="Target column ratio for LEFT lane overlay / steering (camera_part3 used ~0.43 for left lane)",
    )
    ap.add_argument(
        "--http-preview-port",
        type=int,
        default=0,
        metavar="PORT",
        help="Serve MJPEG debug panel on 0.0.0.0:PORT (0=off). "
        "On PC: ssh -L PORT:127.0.0.1:PORT user@jetson then open http://127.0.0.1:PORT/",
    )
    ap.add_argument(
        "--http-bind",
        default="0.0.0.0",
        help="Bind address for --http-preview-port (default all interfaces)",
    )
    ap.add_argument(
        "--center-deadband",
        type=float,
        default=CENTER_DEADBAND,
        help="dual-center: |error| below this => Forward Half (between both lines).",
    )
    ap.add_argument(
        "--green-line-thickness",
        type=int,
        default=5,
        metavar="PX",
        help="Overlay only: thickness (pixels) of the green target line; red/orange stay thin.",
    )
    ap.add_argument(
        "--target-goal-band",
        type=float,
        default=0.0,
        metavar="RATIO",
        help="Overlay only: semi-transparent green vertical band ± this fraction of frame width around "
        "each green target (try 0.05–0.12 when lane width varies; 0 disables). Does not change steering.",
    )
    ap.add_argument(
        "--dual-u-thresh",
        type=float,
        default=DUAL_U_THRESH,
        help="dual-center: PD output magnitude to command Left/Right Half.",
    )
    ap.add_argument(
        "--single-left-danger",
        type=float,
        default=SINGLE_LEFT_DANGER_RATIO,
        help="dual-center LEFT-only: line x/W above this => turn away (Right Half).",
    )
    ap.add_argument(
        "--single-right-danger",
        type=float,
        default=SINGLE_RIGHT_DANGER_RATIO,
        help="dual-center RIGHT-only: line x/W above this => turn away (Left Half).",
    )
    ap.add_argument(
        "--blind-pulse-turn-sec",
        type=float,
        default=BLIND_PULSE_TURN_SEC,
        metavar="SEC",
        help="dual-center: when not BOTH and steering wants Left/Right, turn for this long, then "
        f"hold Stop for --blind-pulse-wait-sec (default {BLIND_PULSE_WAIT_SEC}s). Set 0 with "
        "--no-blind-turn-pulse to disable.",
    )
    ap.add_argument(
        "--blind-pulse-wait-sec",
        type=float,
        default=BLIND_PULSE_WAIT_SEC,
        metavar="SEC",
        help="dual-center: pause duration (Stop) between turn pulses while lane is not BOTH.",
    )
    ap.add_argument(
        "--blind-turn-pulse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dual-center: enable turn/wait pulsing when vis is not BOTH (default on). "
        "Use --no-blind-turn-pulse for continuous turn like before.",
    )
    ap.add_argument(
        "--invert-steer",
        action="store_true",
        help="Swap Left/Right motor commands if the robot turns the wrong way for +e/-e",
    )
    ap.add_argument(
        "--lane-detector",
        choices=("fitline", "histogram"),
        default="fitline",
        help="fitline=camera_part3 HLS+fitLine; histogram=column peak+hysteresis",
    )
    ap.add_argument(
        "--right-lane-target",
        type=float,
        default=RIGHT_LANE_TARGET_RATIO_DEFAULT,
        help="Target column ratio for RIGHT lane overlay / steering (camera_part3 right lane used 0.65).",
    )
    ap.add_argument(
        "--turn-left-enter",
        type=float,
        default=TURN_LEFT_ENTER_RATIO_DEFAULT,
        help="If stripe center is farther right than this fraction of frame width, command Left Half "
        "(too close to tape). Typical 0.54–0.64.",
    )
    ap.add_argument(
        "--turn-left-exit",
        type=float,
        default=TURN_LEFT_EXIT_RATIO_DEFAULT,
        help="Hysteresis: leave Left Half when stripe moves left of this fraction (must be < --turn-left-enter).",
    )
    ap.add_argument(
        "--cam-width",
        type=int,
        default=CAM_FRAME_WIDTH,
        help="Capture width (see also --right-cam-width)",
    )
    ap.add_argument("--cam-height", type=int, default=CAM_FRAME_HEIGHT)
    ap.add_argument(
        "--cam-fps",
        type=int,
        default=CAM_FRAME_FPS,
        help="Lower values reduce USB bandwidth (e.g. 10-15 on a shared hub)",
    )
    ap.add_argument(
        "--hsv-v-min",
        type=int,
        default=HSV_V_MIN,
        help="histogram mode: minimum V in HSV for white tape",
    )
    ap.add_argument(
        "--hls-l-min",
        type=int,
        default=PART3_HLS_L_MIN,
        help="fitline mode: HLS L minimum (camera_part3 used 170)",
    )
    ap.add_argument(
        "--hls-s-max",
        type=int,
        default=PART3_HLS_S_MAX,
        help="fitline mode: HLS S maximum (camera_part3 used 50)",
    )
    args = ap.parse_args()
    args.green_line_thickness = max(1, min(41, int(args.green_line_thickness)))
    if args.target_goal_band < 0.0:
        args.target_goal_band = 0.0
    if args.target_goal_band > 0.45:
        print("[WARN] --target-goal-band very large; capping display at 0.45")
        args.target_goal_band = 0.45
    lo, hi = sorted((args.turn_left_enter, args.turn_left_exit))
    if lo != args.turn_left_exit or hi != args.turn_left_enter:
        print("[WARN] Using --turn-left-exit < --turn-left-enter; values were normalized.")
    args.turn_left_exit = lo
    args.turn_left_enter = hi if hi > lo + 1e-6 else lo + 0.06
    scan_indices = [0, 1, 2, 3, 4, 5, 6]

    ser = None
    mc = None
    cap = None
    caps = {}
    http_server = None
    preview_state = MjpegPreviewState()

    st = ControllerState()

    try:
        ser = open_serial(args.port, args.baud)
        mc = MotorCommander(ser)
        global GLOBAL_MC, GLOBAL_SER, GLOBAL_CAP, GLOBAL_CAPS
        GLOBAL_MC = mc
        GLOBAL_SER = ser
        mc.send(CMD_STOP, force=True)
        time.sleep(0.2)

        signal.signal(signal.SIGINT, emergency_shutdown)   # Ctrl+C to exit

        caps, used_idxs = initialize_dual_lane_cameras(args, scan_indices)
        if caps is None:
            raise RuntimeError(
                "Could not open BOTH cameras (USB bandwidth or wrong --left-index / --right-index)."
            )
        GLOBAL_CAPS = caps
        print(
            f"[CAM] dual-center: LEFT index={used_idxs['LEFT']} "
            f"RIGHT index={used_idxs['RIGHT']} detector={args.lane_detector} "
            "(fixed LR mapping)"
        )
        if args.invert_steer:
            print("[CAM] --invert-steer: Left/Right serial commands swapped at output")

        http_server = start_mjpeg_server(
            preview_state, args.http_bind, args.http_preview_port
        )

        while True:
            force_stop_motors(mc, ser)
            begin_msg = "Left + right cameras ready. Type 'begin' then press Enter to start: "
            user_cmd = input(begin_msg).strip().lower()
            if user_cmd == "begin":
                break
            print("[INFO] Waiting for 'begin'...")

        st.line_right_ema = None
        st.read_phase = 0
        cap = caps["RIGHT"]
        GLOBAL_CAP = cap

        prev_t = time.time()

        st.line_left_ema = None
        st.lost_both_frames = 0
        st.center_prev_e = 0.0
        st.last_lateral_cmd = None
        st.blind_pulse_until = 0.0
        st.blind_in_turn_phase = True
        frame_cache_left = None
        frame_cache_right = None
        stale_left = 0
        stale_right = 0
        dual_miss_streak = 0
        cam_recoveries_done = 0

        while True:
            if st.read_phase % 2 == 0:
                ret_r, frame_right_new = caps["RIGHT"].read()
                ret_l, frame_left_new = caps["LEFT"].read()
            else:
                ret_l, frame_left_new = caps["LEFT"].read()
                ret_r, frame_right_new = caps["RIGHT"].read()
            st.read_phase += 1
            GLOBAL_CAP = caps["RIGHT"]

            if ret_r:
                frame_cache_right = frame_right_new
                stale_right = 0
            else:
                stale_right += 1
            if ret_l:
                frame_cache_left = frame_left_new
                stale_left = 0
            else:
                stale_left += 1

            if not ret_r and not ret_l:
                dual_miss_streak += 1
            else:
                dual_miss_streak = 0

            if frame_cache_right is None or frame_cache_left is None:
                force_stop_motors(mc, ser)
                print("[CAM] Waiting for initial frames from both cameras...")
                time.sleep(0.01)
                continue

            dropout = (
                stale_left > MAX_STALE_FRAMES_PER_CAM
                or stale_right > MAX_STALE_FRAMES_PER_CAM
                or dual_miss_streak > MAX_CONSECUTIVE_DUAL_MISS
            )
            if dropout:
                print(
                    "[CAM] Camera dropout "
                    f"(stale_L={stale_left}, stale_R={stale_right}, dual_miss={dual_miss_streak})"
                )
                recover_side = (
                    "RIGHT"
                    if dual_miss_streak > MAX_CONSECUTIVE_DUAL_MISS
                    else ("RIGHT" if stale_right > stale_left else "LEFT")
                )
                if cam_recoveries_done < MAX_CAM_RECOVERIES_PER_RUN and try_recover_one_camera(
                    caps, used_idxs, recover_side, scan_indices, args
                ):
                    cam_recoveries_done += 1
                    stale_left = stale_right = 0
                    dual_miss_streak = 0
                    st.line_left_ema = None
                    st.line_right_ema = None
                    st.center_prev_e = 0.0
                    st.last_lateral_cmd = None
                    st.blind_pulse_until = 0.0
                    st.blind_in_turn_phase = True
                    frame_cache_left = frame_cache_right = None
                    force_stop_motors(mc, ser)
                    print("[CAM] Recovered; waiting for fresh frames from both cameras.")
                    continue
                print("[CAM] Giving up camera recovery -> STOP")
                mc.send(CMD_STOP)
                break

            frame_right = frame_cache_right.copy()
            frame_left = frame_cache_left.copy()
            if MIRROR_FRAME:
                frame_right = cv2.flip(frame_right, 1)
                frame_left = cv2.flip(frame_left, 1)

            tgtL = float(args.left_lane_target)
            tgtR = float(args.right_lane_target)

            left_overlay, left_mask_dbg, line_L, err_L, det_L = process_frame(
                frame_left,
                args.lane_detector,
                tgtL,
                float(args.turn_left_enter),
                float(args.turn_left_exit),
                hsv_v_min=int(args.hsv_v_min),
                hls_l_min=int(args.hls_l_min),
                hls_s_max=int(args.hls_s_max),
                lane_roi_side="left",
                green_line_thickness=int(args.green_line_thickness),
                target_goal_band_ratio=float(args.target_goal_band),
            )
            right_overlay, right_mask_dbg, line_R, err_R, det_R = process_frame(
                frame_right,
                args.lane_detector,
                tgtR,
                float(args.turn_left_enter),
                float(args.turn_left_exit),
                hsv_v_min=int(args.hsv_v_min),
                hls_l_min=int(args.hls_l_min),
                hls_s_max=int(args.hls_s_max),
                lane_roi_side="right",
                green_line_thickness=int(args.green_line_thickness),
                target_goal_band_ratio=float(args.target_goal_band),
            )

            wL = frame_left.shape[1]
            wR = frame_right.shape[1]
            st.line_left_ema = smooth_line_x_ema(st.line_left_ema, line_L, wL)
            st.line_right_ema = smooth_line_x_ema(st.line_right_ema, line_R, wR)

            nL = st.line_left_ema / wL if st.line_left_ema is not None else None
            nR = st.line_right_ema / wR if st.line_right_ema is not None else None

            vis = classify_lane_visibility(line_L, line_R)

            if vis == "BOTH" and nL is not None and nR is not None:
                e = center_error_both(nL, nR, tgtL, tgtR)
                de = e - st.center_prev_e
                st.center_prev_e = e
            else:
                e = 0.0
                de = 0.0

            cmd, logic = choose_steering_dual_center(
                vis,
                e,
                de,
                nL,
                nR,
                float(args.single_left_danger),
                float(args.single_right_danger),
                float(args.center_deadband),
                float(args.dual_u_thresh),
            )

            if vis == "NONE":
                st.lost_both_frames += 1
                if st.last_lateral_cmd in (CMD_LEFT, CMD_RIGHT):
                    cmd = st.last_lateral_cmd
                    logic = (
                        "LOST_COAST_L"
                        if cmd == CMD_LEFT
                        else "LOST_COAST_R"
                    )
                else:
                    cmd = CMD_FWD
                    logic = "LOST_BOTH_FWD_SEARCH"
            else:
                st.lost_both_frames = 0

            desired_cmd = cmd
            desired_logic = logic
            if desired_cmd in (CMD_LEFT, CMD_RIGHT):
                st.last_lateral_cmd = desired_cmd

            now_mono = time.monotonic()
            pulse_on = bool(args.blind_turn_pulse) and float(args.blind_pulse_turn_sec) > 0.0
            cmd, logic = apply_blind_turn_pulse(
                st,
                vis,
                desired_cmd,
                desired_logic,
                now_mono,
                float(args.blind_pulse_turn_sec),
                float(args.blind_pulse_wait_sec),
                pulse_on,
            )

            cmd_motor = apply_steer_inversion(cmd, args.invert_steer)
            mc.send(cmd_motor)
            st.last_cmd = cmd_motor

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now

            print(
                f"[TELEM][DUAL] cmd={cmd_motor} vis={vis} e={e:+.4f} {logic} "
                f"Lx={line_L} Rx={line_R} detL={det_L} detR={det_R} "
                f"lost_both={st.lost_both_frames} fps={fps:.1f}"
            )

            caption = (
                f"DUAL {vis} {logic} e={e:+.3f} OUT={cmd_motor} | "
                f"L={det_L} R={det_R}"
            )
            dual_stack = build_dual_cam_debug_panel(
                frame_left,
                left_mask_dbg,
                left_overlay,
                f"LEFT | {det_L}",
                frame_right,
                right_mask_dbg,
                right_overlay,
                f"RIGHT | {det_R} | {caption}",
            )
            if args.http_preview_port > 0:
                ok, jpg = cv2.imencode(
                    ".jpg", dual_stack, [cv2.IMWRITE_JPEG_QUALITY, 72]
                )
                if ok:
                    preview_state.set_jpeg(jpg.tobytes())
            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C pressed")
    finally:
        try:
            force_stop_motors(mc, ser)
        except Exception:
            pass
        try:
            for side_cap in caps.values():
                if side_cap is not None:
                    side_cap.release()
        except Exception:
            pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        GLOBAL_MC = None
        GLOBAL_SER = None
        GLOBAL_CAP = None
        GLOBAL_CAPS = {}
        if http_server is not None:
            try:
                http_server.shutdown()
                http_server.server_close()
            except Exception:
                pass
        print("[INFO] Clean exit")


if __name__ == "__main__":
    main()

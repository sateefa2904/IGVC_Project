#!/usr/bin/env python3
"""
Camera-only lane follower (default **--control-mode dual-center**): **left + right** cameras,
white tape on each side; steer to stay centered, **Forward Half** when |error| is small.

Use ``--control-mode right-only`` for the previous **single RIGHT camera** behavior
(fitline PD or histogram avoid-right-stripe).

**--lane-detector fitline** (default): HLS mask, ROI, ``cv2.fitLine`` (and PD in right-only mode).
**--lane-detector histogram**: column-sum peak + hysteresis.

In dual-center, if only one side sees a line, the vehicle creeps forward until too close, then
turns away until the other camera can see the other line.

Use ``--show``, ``--http-preview-port``, or ``--preview-stills`` for debug. Dual preview: two rows
(LEFT then RIGHT) or ``--dual-preview-layout wide``. By default ``--swap-lr-cameras`` maps USB
``--right-index`` to the LEFT lane view and ``--left-index`` to the RIGHT (see help); use
``--no-swap-lr-cameras`` if previews already match the vehicle. ``--http-single-cam-preview`` is only for
right-only mode (dual-center needs both streams).

Debug: telemetry each loop, optional OpenCV window, optional MJPEG HTTP preview, optional stills.
"""

import argparse
import os
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
# After sending Stop on exit: USB-UART buffers may still be draining; flush() does not always
# wait for the last stop bit. Settle before closing the port so the TM4C sees a full "Stop\r".
STOP_SHUTDOWN_BURST_TIMES = 2
STOP_SHUTDOWN_SETTLE_SEC = 0.35

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
LEFT_CAM_INDEX = 0
RIGHT_CAM_INDEX = 2
# V4L2 is Linux; DirectShow is typical on Windows.
CAM_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
MIRROR_FRAME = False

# Opening two USB cameras on Linux/Jetson: the 2nd often fails without a pause; some
# systems need retries or a different VideoCapture API. Tune via CLI if needed.
CAM_SECOND_OPEN_DELAY_SEC = 0.85
CAM_OPEN_RETRIES_PER_INDEX = 5
CAM_RETRY_GAP_SEC = 0.35

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
    top_left=(0.22, 0.60),
    top_right=(0.78, 0.60),
    bottom_right=(0.95, 0.98),
)

# Right-lane search window: ignore left side of image so floor / far lane marks are not picked.
RIGHT_LINE_ROI_X0_RATIO = 0.28
RIGHT_LINE_ROI_X1_RATIO = 0.99
RIGHT_LINE_ROI_Y0_RATIO = 0.42
RIGHT_LINE_ROI_Y1_RATIO = 0.99
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

# --- camera_part3.py style (fitLine + HLS mask + trapezoid ROI) ---
PART3_MIN_AREA = 300
PART3_MIN_HEIGHT = 40
PART3_MIN_WIDTH = 4
PART3_MIN_ASPECT_H_OVER_W = 1.2
PART3_HLS_L_MIN = 170
PART3_HLS_S_MAX = 50
# PD steering (same as camera_part3 SimpleLineTracker)
PART3_KP = 0.9
PART3_KD = 0.2
PART3_TURN_LEFT_THRESH = -0.20
PART3_TURN_RIGHT_THRESH = 0.20

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
CENTER_DEADBAND = 0.07
DUAL_KP = 1.05
DUAL_KD = 0.18
DUAL_U_THRESH = 0.18
# Single-side: creep forward until line norm exceeds these, then turn away from that line.
SINGLE_LEFT_DANGER_RATIO = 0.52
SINGLE_RIGHT_DANGER_RATIO = 0.52
# When not BOTH and steering wants Left/Right: turn briefly, then Stop so cameras can update.
BLIND_PULSE_TURN_SEC = 1.0
BLIND_PULSE_WAIT_SEC = 2.0


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


def morph_clean(mask: np.ndarray) -> np.ndarray:
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m


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


def white_mask_hls(
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


class SimpleLineTracker:
    """camera_part3 PD on normalized error vs target column."""

    def __init__(self, w: int, target_x_ratio: float):
        self.w = w
        self.target_x_ratio = target_x_ratio
        self.prev_err = 0.0

    def control_from_x(self, line_x_bottom: Optional[float]) -> Tuple[float, float]:
        target_x = self.target_x_ratio * self.w
        if line_x_bottom is None:
            err_norm = 0.0
        else:
            err_px = float(line_x_bottom) - target_x
            err_norm = float(np.clip(err_px / (self.w / 2), -1.0, 1.0))
        d = err_norm - self.prev_err
        self.prev_err = err_norm
        u = PART3_KP * err_norm + PART3_KD * d
        u = float(np.clip(u, -1.0, 1.0))
        return u, err_norm


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


def choose_avoid_right_stripe(
    x_norm: float,
    steer_state: str,
    enter_ratio: float,
    exit_ratio: float,
) -> Tuple[str, str]:
    """
    x_norm = line_x / width. Stripe far right (high x_norm) => too close to tape => turn left.
    Hysteresis avoids oscillating between FWD and LEFT at the threshold.
    """
    if steer_state == "LEFT":
        if x_norm <= exit_ratio:
            return CMD_FWD, "STRAIGHT"
        return CMD_LEFT, "LEFT"
    if x_norm >= enter_ratio:
        return CMD_LEFT, "LEFT"
    return CMD_FWD, "STRAIGHT"


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


def send_line_typewriter(ser: serial.Serial, text: str):
    data = (text + EOL).encode("ascii", errors="ignore")
    for b in data:
        ser.write(bytes([b]))
        ser.flush()
        time.sleep(CHAR_DELAY)


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


def emergency_shutdown(signum, frame):
    del frame
    global SHUTDOWN_IN_PROGRESS
    if SHUTDOWN_IN_PROGRESS:
        raise SystemExit(0)

    SHUTDOWN_IN_PROGRESS = True
    print(f"\n[SIGNAL] Caught signal {signum}, attempting safe stop...")

    try:
        force_stop_motors(GLOBAL_MC, GLOBAL_SER)
    except Exception:
        pass

    try:
        for side_cap in GLOBAL_CAPS.values():
            if side_cap is not None:
                side_cap.release()
    except Exception:
        pass

    # Leave GLOBAL_SER open here so a final force_stop in main's `finally` can run; closing too
    # early can truncate the typewriter "Stop\r" and leave the MCU with a half-line / wrong tower.

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    raise SystemExit(0)


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


def open_camera(
    index: int,
    label: str,
    width: int = CAM_FRAME_WIDTH,
    height: int = CAM_FRAME_HEIGHT,
    fps: int = CAM_FRAME_FPS,
) -> Optional[cv2.VideoCapture]:
    for api in _video_capture_backends():
        try:
            if api is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, api)
        except Exception:
            continue
        time.sleep(0.12)
        if not cap.isOpened():
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
            if not ret:
                cap.release()
                continue
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            continue

        api_tag = "default" if api is None else str(api)
        print(
            f"[CAM] Using {label} camera index={index} (backend={api_tag}) "
            f"{width}x{height}@{fps}"
        )
        return cap

    print(f"[CAM] Failed to open {label} camera at index {index} (all backends)")
    return None


def print_camera_troubleshooting():
    print(
        "[CAM] Second camera not available. Common causes on Jetson/Linux:\n"
        "  • USB bandwidth: use a powered USB3 hub, shorter cables, or lower resolution in code.\n"
        "  • Device nodes: run  v4l2-ctl --list-devices  and  ls -la /dev/video*\n"
        "    (some indices are metadata, not capture — try --right-index explicitly).\n"
        "  • Unplug/replug both cameras; try  --camera-second-delay 1.5  or higher.\n"
        "  • Another process may hold the device: close other camera apps."
    )


def capture_dims(args, side: str) -> Tuple[int, int, int]:
    """(width, height, fps) for LEFT or RIGHT; right may use smaller WxH to save USB bandwidth."""
    w, h = args.cam_width, args.cam_height
    if (
        side == "RIGHT"
        and getattr(args, "right_cam_width", 0) > 0
        and getattr(args, "right_cam_height", 0) > 0
    ):
        w, h = args.right_cam_width, args.right_cam_height
    return w, h, int(args.cam_fps)


def parse_indices_csv(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out


def uniq_indices(indices: List[int]) -> List[int]:
    seen = set()
    out = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def open_side_camera_with_fallback(
    side: str,
    preferred_index: int,
    scan_indices: List[int],
    avoid_index: Optional[int] = None,
    width: int = CAM_FRAME_WIDTH,
    height: int = CAM_FRAME_HEIGHT,
    fps: int = CAM_FRAME_FPS,
) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    candidates = [preferred_index] + scan_indices
    candidates = uniq_indices(candidates)
    if avoid_index is not None:
        candidates = [i for i in candidates if i != avoid_index]

    for idx in candidates:
        cap = open_camera(idx, side, width, height, fps)
        if cap is not None:
            return cap, idx
    return None, None


def initialize_right_lane_camera(args, scan_indices):
    """Open a single capture device: the RIGHT-side lane camera."""
    rw, rh, rf = capture_dims(args, "RIGHT")
    cap, idx = open_side_camera_with_fallback(
        "RIGHT",
        preferred_index=args.right_index,
        scan_indices=scan_indices,
        avoid_index=None,
        width=rw,
        height=rh,
        fps=rf,
    )
    if cap is None:
        return None, None
    return {"RIGHT": cap}, {"RIGHT": idx}


def initialize_optional_left_preview_camera(
    args, scan_indices: List[int], avoid_index: Optional[int]
) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """Second USB camera for debug only (same resolution as --cam-width/--cam-height)."""
    w, h, fps = args.cam_width, args.cam_height, int(args.cam_fps)
    return open_side_camera_with_fallback(
        "LEFT",
        preferred_index=args.left_index,
        scan_indices=scan_indices,
        avoid_index=avoid_index,
        width=w,
        height=h,
        fps=fps,
    )


def initialize_dual_lane_cameras(args, scan_indices: List[int]):
    """
    Open RIGHT index first, then LEFT index (pause between) for dual-lane centering. Both required.

    By default (``--swap-lr-cameras``, on) the VideoCapture opened with ``--right-index`` is wired to
    **LEFT** lane detection and ``--left-index`` to **RIGHT**, because USB enumeration order often
    does not match physical mounting. Disable with ``--no-swap-lr-cameras`` if your indices already
    match left/right lenses.
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
    if args.swap_lr_cameras:
        return {"LEFT": cap_r, "RIGHT": cap_l}, {"LEFT": idx_r, "RIGHT": idx_l}
    return {"LEFT": cap_l, "RIGHT": cap_r}, {"LEFT": idx_l, "RIGHT": idx_r}


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


def center_error_both(nL: float, nR: float, tgtL: float, tgtR: float) -> float:
    """Positive => vehicle too far right in lane => steer left to correct."""
    return (nR - tgtR) - (nL - tgtL)


def choose_steering_dual_center(
    vis: str,
    e: float,
    de: float,
    nL: Optional[float],
    nR: Optional[float],
    danger_L: float,
    danger_R: float,
    center_db: float,
    u_thresh: float,
) -> Tuple[str, str]:
    """
    BOTH: PD-like u on center error; |e| small => forward.
    ONE SIDE: forward until visible line norm exceeds danger ratio, then turn away from that line;
    creep forward otherwise until the other camera picks up the other line.
    NONE: stop (dual-center main loop overrides: coast last Left/Right until a line returns).
    """
    if vis == "BOTH":
        u = DUAL_KP * e + DUAL_KD * de
        u = float(np.clip(u, -1.0, 1.0))
        if abs(e) < center_db:
            return CMD_FWD, "CENTER_FWD"
        if u > u_thresh:
            return CMD_LEFT, "DUAL_L"
        if u < -u_thresh:
            return CMD_RIGHT, "DUAL_R"
        return CMD_FWD, "CENTER_FWD"
    if vis == "LEFT_ONLY":
        assert nL is not None
        if nL >= danger_L:
            return CMD_RIGHT, "AWAY_LEFT_LINE"
        return CMD_FWD, "CREEP_LEFT"
    if vis == "RIGHT_ONLY":
        assert nR is not None
        if nR >= danger_R:
            return CMD_LEFT, "AWAY_RIGHT_LINE"
        return CMD_FWD, "CREEP_RIGHT"
    return CMD_STOP, "LOST"


def apply_blind_turn_pulse(
    st: "ControllerState",
    vis: str,
    desired_cmd: str,
    logic: str,
    now_mono: float,
    turn_sec: float,
    wait_sec: float,
    enabled: bool,
) -> Tuple[str, str]:
    """
    If ``vis != BOTH`` and the controller wants Left/Right, alternate: turn for ``turn_sec``,
    then Stop for ``wait_sec``, repeating until BOTH is visible again. Reduces over-correction when
    vision lags the vehicle.
    """
    if not enabled or vis == "BOTH":
        st.blind_pulse_until = 0.0
        return desired_cmd, logic

    if desired_cmd not in (CMD_LEFT, CMD_RIGHT):
        st.blind_pulse_until = 0.0
        return desired_cmd, logic

    if st.blind_pulse_until <= 0.0 or now_mono >= st.blind_pulse_until:
        if st.blind_pulse_until <= 0.0:
            st.blind_in_turn_phase = True
        else:
            st.blind_in_turn_phase = not st.blind_in_turn_phase
        dur = turn_sec if st.blind_in_turn_phase else wait_sec
        st.blind_pulse_until = now_mono + float(dur)

    if st.blind_in_turn_phase:
        return desired_cmd, f"{logic}|PULSE_TURN"
    return CMD_STOP, f"{logic}|PULSE_WAIT"


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


def process_frame_histogram(
    frame_bgr: np.ndarray,
    target_ratio: float,
    turn_enter_ratio: float,
    turn_exit_ratio: float,
    hsv_v_min: int = HSV_V_MIN,
    lane_roi_side: str = "right",
    green_line_thickness: int = 5,
    target_goal_band_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float, str]:
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


def process_frame_fitline(
    frame_bgr: np.ndarray,
    target_ratio: float,
    turn_enter_ratio: float,
    turn_exit_ratio: float,
    hls_l_min: int = PART3_HLS_L_MIN,
    hls_s_max: int = PART3_HLS_S_MAX,
    lane_roi_side: str = "right",
    green_line_thickness: int = 5,
    target_goal_band_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], float, str]:
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


def choose_part3_pd_cmd(
    pd_u: float, right_lane_semantics: bool = True
) -> Tuple[str, str]:
    """
    Map PD ``pd_u`` to discrete drive commands.

    **Default (right_lane_semantics=True):** matches this program's right-lane goal — positive
    ``pd_u`` means the stripe is to the **right** of the target column (too close to the tape)
    ⇒ **Left Half**. Negative ⇒ **Right Half**. This is the opposite motor pairing from raw
    ``camera_part3.py`` lane-follow, which was written for a different vehicle/convention.

    Use ``right_lane_semantics=False`` (``--legacy-part3-pd-steering``) to restore camera_part3
    Left/Right pairing if your hardware was tuned for that script.
    """
    if right_lane_semantics:
        if pd_u > PART3_TURN_RIGHT_THRESH:
            return CMD_LEFT, "PD_L"
        if pd_u < PART3_TURN_LEFT_THRESH:
            return CMD_RIGHT, "PD_R"
    else:
        if pd_u > PART3_TURN_RIGHT_THRESH:
            return CMD_RIGHT, "PD_R"
        if pd_u < PART3_TURN_LEFT_THRESH:
            return CMD_LEFT, "PD_L"
    return CMD_FWD, "PD_FWD"


def smooth_line_x_ema(
    prev: Optional[float],
    meas: Optional[float],
    w: int,
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


@dataclass
class ControllerState:
    steer_state: str = "STRAIGHT"
    lost_frames: int = 0
    last_cmd: str = CMD_STOP
    line_right_ema: Optional[float] = None
    line_left_ema: Optional[float] = None
    center_prev_e: float = 0.0
    lost_both_frames: int = 0
    read_phase: int = 0
    last_lateral_cmd: Optional[str] = None  # CMD_LEFT or CMD_RIGHT; used when vis==NONE (dual-center)
    blind_pulse_until: float = 0.0
    blind_in_turn_phase: bool = True


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


def apply_steer_inversion(cmd: str, invert: bool) -> str:
    if not invert:
        return cmd
    if cmd == CMD_LEFT:
        return CMD_RIGHT
    if cmd == CMD_RIGHT:
        return CMD_LEFT
    return cmd


def build_debug_panel(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    left_mask_dbg: np.ndarray,
    right_mask_dbg: np.ndarray,
    left_overlay: np.ndarray,
    right_overlay: np.ndarray,
    caption: str,
    thumb_w: int = 320,
    thumb_h: int = 180,
) -> np.ndarray:
    left_raw = cv2.resize(frame_left, (thumb_w, thumb_h))
    left_mask = cv2.resize(left_mask_dbg, (thumb_w, thumb_h))
    left_ov = cv2.resize(left_overlay, (thumb_w, thumb_h))
    right_raw = cv2.resize(frame_right, (thumb_w, thumb_h))
    right_mask = cv2.resize(right_mask_dbg, (thumb_w, thumb_h))
    right_ov = cv2.resize(right_overlay, (thumb_w, thumb_h))
    stacked = np.vstack(
        (
            np.hstack((left_raw, left_mask, left_ov)),
            np.hstack((right_raw, right_mask, right_ov)),
        )
    )
    cv2.putText(
        stacked,
        caption,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )
    return stacked


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


def build_dual_cam_debug_panel_side_by_side(
    left_frame: np.ndarray,
    left_mask_dbg: np.ndarray,
    left_overlay: np.ndarray,
    left_caption: str,
    right_frame: np.ndarray,
    right_mask_dbg: np.ndarray,
    right_overlay: np.ndarray,
    right_caption: str,
    thumb_w: int = 384,
    thumb_h: int = 216,
) -> np.ndarray:
    """
    One row: [LEFT raw|mask|ov] [RIGHT raw|mask|ov] so both cameras fit in a browser
    without vertical scrolling (MJPEG / remote viewing).
    """
    row_l = build_single_cam_debug_panel(
        left_frame, left_mask_dbg, left_overlay, left_caption, thumb_w, thumb_h
    )
    row_r = build_single_cam_debug_panel(
        right_frame, right_mask_dbg, right_overlay, right_caption, thumb_w, thumb_h
    )
    h = max(row_l.shape[0], row_r.shape[0])
    if row_l.shape[0] != h:
        row_l = cv2.resize(row_l, (int(row_l.shape[1] * h / row_l.shape[0]), h))
    if row_r.shape[0] != h:
        row_r = cv2.resize(row_r, (int(row_r.shape[1] * h / row_r.shape[0]), h))
    return np.hstack((row_l, row_r))


def save_debug_images(
    out_dir: str, frame: np.ndarray, mask_dbg: np.ndarray, overlay: np.ndarray, tag: str
):
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_{tag}_input.jpg"), frame)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_{tag}_mask.jpg"), mask_dbg)
    cv2.imwrite(os.path.join(out_dir, f"{ts}_{tag}_overlay.jpg"), overlay)


def write_preview_stills(
    dir_path: str,
    frame_bgr: np.ndarray,
    overlay: np.ndarray,
    mask_dbg: np.ndarray,
    caption: str,
    left_bundle: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = None,
    dual_stack: Optional[np.ndarray] = None,
) -> None:
    """
    Fixed filenames overwritten each call — easy to SCP or open from a network share to see
    what the vehicle sees (same idea as input.jpg / output.jpg / mask in other scripts).
    Writes: input.jpg, overlay.jpg, mask.jpg, stack.jpg for the **RIGHT** (driving) camera.
    If ``left_bundle`` is set, also writes left_input.jpg, …, left_stack.jpg.
    If ``dual_stack`` is set, writes stack_dual.jpg (LEFT+RIGHT combined).
    """
    os.makedirs(dir_path, exist_ok=True)
    cv2.imwrite(os.path.join(dir_path, "input.jpg"), frame_bgr)
    cv2.imwrite(os.path.join(dir_path, "overlay.jpg"), overlay)
    cv2.imwrite(os.path.join(dir_path, "mask.jpg"), mask_dbg)
    stacked = build_single_cam_debug_panel(frame_bgr, mask_dbg, overlay, caption)
    cv2.imwrite(os.path.join(dir_path, "stack.jpg"), stacked)
    if left_bundle is not None:
        lf, lo, lm, lcap = left_bundle
        cv2.imwrite(os.path.join(dir_path, "left_input.jpg"), lf)
        cv2.imwrite(os.path.join(dir_path, "left_overlay.jpg"), lo)
        cv2.imwrite(os.path.join(dir_path, "left_mask.jpg"), lm)
        cv2.imwrite(
            os.path.join(dir_path, "left_stack.jpg"),
            build_single_cam_debug_panel(lf, lm, lo, lcap),
        )
    if dual_stack is not None:
        cv2.imwrite(os.path.join(dir_path, "stack_dual.jpg"), dual_stack)


def main():
    ap = argparse.ArgumentParser(
        description="Right-camera lane follow: camera_part3 fitLine+PD (default) or histogram mode"
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
        "--preview-left-camera",
        action="store_true",
        help="Open LEFT at --left-index for dual preview. Implied when using --http-preview-port "
        "unless --http-single-cam-preview. Steering always uses RIGHT only.",
    )
    ap.add_argument(
        "--left-index",
        type=int,
        default=LEFT_CAM_INDEX,
        help="OpenCV index for optional LEFT preview camera",
    )
    ap.add_argument(
        "--left-lane-target",
        type=float,
        default=LEFT_LANE_TARGET_RATIO_DEFAULT,
        help="Target column ratio for LEFT preview overlays (camera_part3 used ~0.43 for left lane)",
    )
    ap.add_argument(
        "--scan-indices",
        default="0,1,2,3,4,5,6",
        help="Fallback camera indices to probe when a side camera fails",
    )
    ap.add_argument("--show", action="store_true", help="Show OpenCV debug windows")
    ap.add_argument(
        "--save-debug", action="store_true", help="Save debug images on keypress 's'"
    )
    ap.add_argument(
        "--debug-dir", default="./camera_debug", help="Folder for saved debug images"
    )
    ap.add_argument(
        "--preview-stills",
        action="store_true",
        help="Continuously overwrite input.jpg, overlay.jpg, mask.jpg, stack.jpg in "
        "--preview-stills-dir (see --preview-stills-every). For remote inspection without a display.",
    )
    ap.add_argument(
        "--preview-stills-dir",
        default="./vehicle_preview",
        help="Directory for --preview-stills fixed filenames",
    )
    ap.add_argument(
        "--preview-stills-every",
        type=int,
        default=1,
        metavar="N",
        help="Write preview stills every N frames (default 1; use 5–15 to reduce USB/disk load)",
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
        "--http-single-cam-preview",
        action="store_true",
        help="With --http-preview-port: do not open the LEFT USB camera; stream RIGHT strip only. "
        "By default, HTTP preview tries to open LEFT+RIGHT (same as --preview-left-camera).",
    )
    ap.add_argument(
        "--dual-preview-layout",
        choices=("rows", "wide"),
        default="rows",
        help="When both cameras are open: rows=LEFT on first line, RIGHT below (default); "
        "wide=single line of six tiles (LEFT|RIGHT side by side).",
    )
    ap.add_argument(
        "--control-mode",
        choices=("dual-center", "right-only"),
        default="dual-center",
        help="dual-center: both cameras, steer between lane lines; forward when centered. "
        "right-only: previous single RIGHT-camera modes (fitline/histogram).",
    )
    ap.add_argument(
        "--swap-lr-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dual-center only: USB indices (--left-index / --right-index) often do not match which "
        "lens is physically on the left vs right of the vehicle. Default ON: assign the stream "
        "opened as --right-index to LEFT lane processing and --left-index to RIGHT. "
        "Use --no-swap-lr-cameras if your preview labels already match reality.",
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
        help="fitline=camera_part3 HLS+fitLine+PD; histogram=column peak+hysteresis",
    )
    ap.add_argument(
        "--legacy-part3-pd-steering",
        action="store_true",
        help="fitline mode only: use camera_part3 PD→motor sign (may fight right-lane logic). "
        "Omit this flag so +error steers Left Half away from the right stripe.",
    )
    ap.add_argument(
        "--right-lane-target",
        type=float,
        default=RIGHT_LANE_TARGET_RATIO_DEFAULT,
        help="Green line on overlay / PD target column (camera_part3 right lane used 0.65).",
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
        "--right-cam-width",
        type=int,
        default=0,
        help="If >0 with --right-cam-height, right camera only uses this resolution (saves USB)",
    )
    ap.add_argument("--right-cam-height", type=int, default=0)
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
    if args.control_mode == "dual-center" and args.http_single_cam_preview:
        print(
            "[ERROR] dual-center mode needs both cameras — remove --http-single-cam-preview "
            "or use --control-mode right-only.",
            file=sys.stderr,
        )
        sys.exit(2)
    lo, hi = sorted((args.turn_left_enter, args.turn_left_exit))
    if lo != args.turn_left_exit or hi != args.turn_left_enter:
        print("[WARN] Using --turn-left-exit < --turn-left-enter; values were normalized.")
    args.turn_left_exit = lo
    args.turn_left_enter = hi if hi > lo + 1e-6 else lo + 0.06
    scan_indices = parse_indices_csv(args.scan_indices)

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

        signal.signal(signal.SIGINT, emergency_shutdown)   # Ctrl+C
        signal.signal(signal.SIGTERM, emergency_shutdown)  # kill
        signal.signal(signal.SIGTSTP, emergency_shutdown)  # Ctrl+Z

        open_left_preview = False
        if args.control_mode == "dual-center":
            caps, used_idxs = initialize_dual_lane_cameras(args, scan_indices)
            if caps is None:
                raise RuntimeError(
                    "dual-center mode could not open BOTH cameras (USB bandwidth or wrong --left-index / --right-index)."
                )
            GLOBAL_CAPS = caps
            print(
                f"[CAM] dual-center: LEFT index={used_idxs['LEFT']} "
                f"RIGHT index={used_idxs['RIGHT']} detector={args.lane_detector} "
                f"(swap_lr_cameras={'on' if args.swap_lr_cameras else 'off'})"
            )
        else:
            caps, used_idxs = initialize_right_lane_camera(args, scan_indices)
            if caps is None:
                raise RuntimeError("Could not open RIGHT lane camera at startup")
            GLOBAL_CAPS = caps
            print(
                f"[CAM] RIGHT camera index={used_idxs['RIGHT']} "
                f"(detector={args.lane_detector})"
            )
            open_left_preview = args.preview_left_camera or (
                args.http_preview_port > 0 and not args.http_single_cam_preview
            )
            if open_left_preview:
                time.sleep(CAM_SECOND_OPEN_DELAY_SEC)
                lcap, lidx = initialize_optional_left_preview_camera(
                    args, scan_indices, used_idxs.get("RIGHT")
                )
                if lcap is None:
                    print(
                        "[CAM] Dual preview: could not open LEFT camera (continuing RIGHT-only). "
                        "Try --http-single-cam-preview if USB bandwidth is the issue."
                    )
                    print_camera_troubleshooting()
                else:
                    caps["LEFT"] = lcap
                    used_idxs["LEFT"] = lidx
                    print(
                        f"[CAM] LEFT camera index={lidx} (preview only; steering uses RIGHT). "
                        f"HTTP dual layout: --dual-preview-layout {args.dual_preview_layout} (rows=stacked)."
                    )
            elif args.http_preview_port > 0 and args.http_single_cam_preview:
                print("[CAM] HTTP preview: RIGHT camera only (--http-single-cam-preview).")
        if args.preview_stills:
            extra = ""
            if open_left_preview or args.control_mode == "dual-center":
                extra = " + left_*.jpg stack_dual.jpg (if LEFT opens)"
            print(
                f"[CAM] --preview-stills -> {os.path.abspath(args.preview_stills_dir)}/"
                f" input.jpg … stack.jpg{extra} "
                f"(every {max(1, args.preview_stills_every)} frame(s))"
            )
        if (
            args.control_mode == "right-only"
            and args.lane_detector == "fitline"
            and not args.legacy_part3_pd_steering
        ):
            print(
                "[CAM] fitline PD: right-lane semantics (+pd_u -> Left Half away from stripe). "
                "Use --legacy-part3-pd-steering only if you matched raw camera_part3 wiring."
            )
        if args.invert_steer:
            print("[CAM] --invert-steer: Left/Right serial commands swapped at output")

        http_server = start_mjpeg_server(
            preview_state, args.http_bind, args.http_preview_port
        )

        while True:
            force_stop_motors(mc, ser)
            begin_msg = (
                "Left + right cameras ready. Type 'begin' then press Enter to start: "
                if args.control_mode == "dual-center"
                else "Right camera is ready. Type 'begin' then press Enter to start: "
            )
            user_cmd = input(begin_msg).strip().lower()
            if user_cmd == "begin":
                break
            print("[INFO] Waiting for 'begin'...")

        st.steer_state = "STRAIGHT"
        st.line_right_ema = None
        st.lost_frames = 0
        st.read_phase = 0
        cap = caps["RIGHT"]
        GLOBAL_CAP = cap

        prev_t = time.time()
        if args.show:
            try:
                cv2.namedWindow("CameraOnly Debug", cv2.WINDOW_NORMAL)
            except cv2.error as e:
                print(f"[WARN] OpenCV display unavailable, disabling --show: {e}")
                args.show = False

        if args.control_mode == "dual-center":
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
                dual_stack = None
                if args.dual_preview_layout == "wide":
                    dual_stack = build_dual_cam_debug_panel_side_by_side(
                        frame_left,
                        left_mask_dbg,
                        left_overlay,
                        f"LEFT | {det_L}",
                        frame_right,
                        right_mask_dbg,
                        right_overlay,
                        f"RIGHT | {det_R} | {caption}",
                    )
                else:
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

                if args.preview_stills and (
                    st.read_phase % max(1, args.preview_stills_every) == 0
                ):
                    try:
                        write_preview_stills(
                            args.preview_stills_dir,
                            frame_right,
                            right_overlay,
                            right_mask_dbg,
                            caption,
                            left_bundle=(
                                frame_left,
                                left_overlay,
                                left_mask_dbg,
                                f"LEFT | {det_L}",
                            ),
                            dual_stack=dual_stack,
                        )
                    except cv2.error as ex:
                        print(f"[WARN] preview stills write failed: {ex}")

                stacked = None
                if args.show or args.http_preview_port > 0:
                    stacked = dual_stack
                    if args.http_preview_port > 0 and stacked is not None:
                        ok, jpg = cv2.imencode(
                            ".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 72]
                        )
                        if ok:
                            preview_state.set_jpeg(jpg.tobytes())

                if args.show and stacked is not None:
                    try:
                        cv2.imshow("CameraOnly Debug", stacked)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q") or k == 27:
                            print("[INFO] Quit requested")
                            break
                        if k == ord("s") and args.save_debug:
                            save_debug_images(
                                args.debug_dir,
                                frame_right,
                                right_mask_dbg,
                                right_overlay,
                                "RIGHT",
                            )
                            save_debug_images(
                                args.debug_dir,
                                frame_left,
                                left_mask_dbg,
                                left_overlay,
                                "LEFT",
                            )
                            print(f"[DEBUG] Saved snapshot to {args.debug_dir}")
                    except cv2.error:
                        pass
                else:
                    time.sleep(0.002)

        else:
            pd_tracker: Optional[SimpleLineTracker] = None
    
            frame_cache_right = None
            stale_right = 0
            miss_streak = 0
            cam_recoveries_done = 0
    
            while True:
                ret_r, frame_right_new = caps["RIGHT"].read()
                st.read_phase += 1
                GLOBAL_CAP = caps["RIGHT"]
    
                if ret_r:
                    frame_cache_right = frame_right_new
                    stale_right = 0
                    miss_streak = 0
                else:
                    stale_right += 1
                    miss_streak += 1
    
                if frame_cache_right is None:
                    force_stop_motors(mc, ser)
                    print("[CAM] Waiting for initial frame from RIGHT camera...")
                    time.sleep(0.01)
                    continue
    
                dropout = stale_right > MAX_STALE_FRAMES_PER_CAM or miss_streak > MAX_CONSECUTIVE_DUAL_MISS
                if dropout:
                    print(
                        "[CAM] Camera dropout detected "
                        f"(stale_right={stale_right}, miss_streak={miss_streak})"
                    )
    
                    if cam_recoveries_done < MAX_CAM_RECOVERIES_PER_RUN and try_recover_one_camera(
                        caps, used_idxs, "RIGHT", scan_indices, args
                    ):
                        cam_recoveries_done += 1
                        stale_right = 0
                        miss_streak = 0
                        st.line_right_ema = None
                        frame_cache_right = None
                        pd_tracker = None
                        force_stop_motors(mc, ser)
                        print("[CAM] Recovered stream; waiting for fresh frame.")
                        continue
    
                    print(
                        "[CAM] Giving up (recovery failed or max recoveries "
                        f"{MAX_CAM_RECOVERIES_PER_RUN}) -> STOP"
                    )
                    mc.send(CMD_STOP)
                    break
    
                frame_right = frame_cache_right.copy()
    
                if MIRROR_FRAME:
                    frame_right = cv2.flip(frame_right, 1)
    
                target_ratio = float(args.right_lane_target)
                w_right_px = frame_right.shape[1]
                if args.lane_detector == "fitline":
                    if (
                        pd_tracker is None
                        or pd_tracker.w != w_right_px
                        or abs(pd_tracker.target_x_ratio - target_ratio) > 1e-6
                    ):
                        pd_tracker = SimpleLineTracker(w_right_px, target_ratio)
    
                right_overlay, right_mask_dbg, line_right, err_right, det_tag = process_frame(
                    frame_right,
                    args.lane_detector,
                    target_ratio,
                    float(args.turn_left_enter),
                    float(args.turn_left_exit),
                    hsv_v_min=int(args.hsv_v_min),
                    hls_l_min=int(args.hls_l_min),
                    hls_s_max=int(args.hls_s_max),
                    lane_roi_side="right",
                    green_line_thickness=int(args.green_line_thickness),
                    target_goal_band_ratio=float(args.target_goal_band),
                )
    
                left_bundle: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]] = None
                if caps.get("LEFT") is not None:
                    ret_l, frame_left_new = caps["LEFT"].read()
                    if ret_l:
                        frame_left = frame_left_new.copy()
                        if MIRROR_FRAME:
                            frame_left = cv2.flip(frame_left, 1)
                        left_t = float(args.left_lane_target)
                        lo, lm, _, _, ldet = process_frame(
                            frame_left,
                            args.lane_detector,
                            left_t,
                            float(args.turn_left_enter),
                            float(args.turn_left_exit),
                            hsv_v_min=int(args.hsv_v_min),
                            hls_l_min=int(args.hls_l_min),
                            hls_s_max=int(args.hls_s_max),
                            lane_roi_side="left",
                            green_line_thickness=int(args.green_line_thickness),
                            target_goal_band_ratio=float(args.target_goal_band),
                        )
                        left_bundle = (
                            frame_left,
                            lo,
                            lm,
                            f"LEFT preview | {args.lane_detector} | {ldet} | no steer | tgt={left_t:.2f}",
                        )
    
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev_t))
                prev_t = now
    
                pd_u = 0.0
                logic = ""
    
                if line_right is None:
                    st.lost_frames += 1
                    st.line_right_ema = None
                    st.steer_state = "STRAIGHT"
                    if pd_tracker is not None:
                        pd_tracker.prev_err = 0.0
                    cmd = CMD_STOP
                    x_norm = float("nan")
                elif args.lane_detector == "fitline":
                    st.lost_frames = 0
                    assert pd_tracker is not None
                    pd_u, _ = pd_tracker.control_from_x(line_right)
                    cmd, logic = choose_part3_pd_cmd(
                        pd_u, right_lane_semantics=not args.legacy_part3_pd_steering
                    )
                    x_norm = float(line_right) / float(w_right_px)
                else:
                    st.lost_frames = 0
                    st.line_right_ema = smooth_line_x_ema(
                        st.line_right_ema, line_right, w_right_px
                    )
                    x_norm = float(st.line_right_ema) / float(w_right_px)
                    cmd, st.steer_state = choose_avoid_right_stripe(
                        x_norm,
                        st.steer_state,
                        float(args.turn_left_enter),
                        float(args.turn_left_exit),
                    )
                    logic = st.steer_state
    
                cmd_motor = apply_steer_inversion(cmd, args.invert_steer)
                mc.send(cmd_motor)
                st.last_cmd = cmd_motor
    
                xn_s = "nan" if line_right is None else f"{x_norm:.3f}"
                sm_s = (
                    "nan"
                    if st.line_right_ema is None
                    else str(int(st.line_right_ema))
                )
                extra = f" pd_u={pd_u:+.3f} {logic}" if args.lane_detector == "fitline" else f" {logic}"
                print(
                    f"[TELEM] cmd={cmd_motor} mode={args.lane_detector} det={det_tag} x_norm={xn_s} "
                    f"Rx={None if line_right is None else int(line_right)} smR={sm_s} "
                    f"e_tgt={err_right:+.3f}{extra} lost={st.lost_frames} staleR={stale_right} fps={fps:.1f}"
                )
    
                caption = (
                    f"{args.lane_detector} OUT={cmd_motor} | {cmd} | det={det_tag} | x/w={xn_s}"
                    + (
                        f" | pd={pd_u:+.2f}"
                        if args.lane_detector == "fitline"
                        else f" | enter>{args.turn_left_enter:.2f} exit<{args.turn_left_exit:.2f}"
                    )
                )
                dual_stack = None
                if left_bundle is not None:
                    lf, lo, lm, lcap = left_bundle
                    rcap = f"RIGHT (steer) | {caption}"
                    if args.dual_preview_layout == "wide":
                        dual_stack = build_dual_cam_debug_panel_side_by_side(
                            lf,
                            lm,
                            lo,
                            lcap,
                            frame_right,
                            right_mask_dbg,
                            right_overlay,
                            rcap,
                        )
                    else:
                        dual_stack = build_dual_cam_debug_panel(
                            lf,
                            lm,
                            lo,
                            lcap,
                            frame_right,
                            right_mask_dbg,
                            right_overlay,
                            rcap,
                        )
    
                if args.preview_stills and (
                    st.read_phase % max(1, args.preview_stills_every) == 0
                ):
                    try:
                        write_preview_stills(
                            args.preview_stills_dir,
                            frame_right,
                            right_overlay,
                            right_mask_dbg,
                            caption,
                            left_bundle=left_bundle,
                            dual_stack=dual_stack,
                        )
                    except cv2.error as e:
                        print(f"[WARN] preview stills write failed: {e}")
    
                stacked = None
                if args.show or args.http_preview_port > 0:
                    if dual_stack is not None:
                        stacked = dual_stack
                    else:
                        stacked = build_single_cam_debug_panel(
                            frame_right,
                            right_mask_dbg,
                            right_overlay,
                            caption,
                        )
                    if args.http_preview_port > 0:
                        ok, jpg = cv2.imencode(
                            ".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 72]
                        )
                        if ok:
                            preview_state.set_jpeg(jpg.tobytes())
    
                if args.show and stacked is not None:
                    try:
                        cv2.imshow("CameraOnly Debug", stacked)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q") or k == 27:
                            print("[INFO] Quit requested")
                            break
                        if k == ord("s") and args.save_debug:
                            save_debug_images(
                                args.debug_dir,
                                frame_right,
                                right_mask_dbg,
                                right_overlay,
                                "RIGHT",
                            )
                            print(f"[DEBUG] Saved snapshot to {args.debug_dir}")
                    except cv2.error:
                        pass
                else:
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

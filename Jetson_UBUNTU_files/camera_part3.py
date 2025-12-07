#!/usr/bin/env python3
"""
Integrated LIDAR + Lane-following with three main modes:

1. Camera+Lidar mode (start & normal driving):
   - Follow left or right lane.
   - LIDAR is always running for obstacle detection & safety.

2. Camera-only lane alignment phase:
   - When switching lanes or after corridor, we turn on the camera on
     the new side and go FORWARD until lane 'turn' ≈ 0.
   - Only then we start normal lane-following steering.

3. LIDAR-only mode:
   - When obstacles exist on BOTH sides and only the middle is free.
   - Cameras are turned off.
   - LIDAR keeps us centered between left/right obstacles, then we go
     forward for some time, turn back toward the origin side, switch
     that camera back on, and re-align to the lane (turn ≈ 0).
"""

import time
import math
import numpy as np
import cv2
import serial
from rplidar import RPLidar, RPLidarException

# =========================
# Serial / driving config
# =========================

PORT = "/dev/ttyACM0"   # change if needed
BAUD = 19200
EOL = "\r"              # TM4C parser ends on CR (13)
DTR = True
RTS = True
CHAR_DELAY = 0.01       # 10 ms between bytes
BOOT_WAIT = 2.0         # seconds after opening port

CMD_UP     = "Forward Half"
CMD_DOWN   = "Backward Half"
CMD_LEFT   = "Left Half"
CMD_RIGHT  = "Right Half"
CMD_AUTO   = "Auto"
CMD_MANUAL = "Manual"
STOP_CMD   = "Stop"

# For deciding steering commands from lane PD output
TURN_LEFT_THRESH  = -0.20
TURN_RIGHT_THRESH =  0.20

# =========================
# LIDAR config / thresholds
# =========================

LIDAR_PORT   = "/dev/ttyUSB0"
LIDAR_BAUD   = 1_000_000
LIDAR_TO     = 1.0
FRONT_DEG    = 0
ANGLE_SIGN_D = -1    # -1 if your LIDAR mount is mirrored

LOOK_ANGLES  = [-50, 0, +50]
ANG_TOL      = 5

FORWARD_SECTORS = {
    'FR': (-60.0, -10.0),
    'F' : (-15.0, +15.0),
    'FL': (+10.0, +60.0),
}

THRESH_MAIN        = 762   # main "open" threshold (mm)
BACKOFF_CLEAR_ANY  = 850
NEAR_OBS_TRIP      = 300   # too close -> back up
NEAR_OBS_RELEASE   = 500   # safe again
POST_NEAR_CLEAR    = 672

NO_DATA_STOP_SEC   = 0.5   # stop if no valid LIDAR data for this time

PRINT_INF_AS = -1

# Corridor / middle-road tuning
CORRIDOR_BALANCE_EPS_MM     = 20.0  # |dL - dR| <= this is "centered"
CORRIDOR_FORWARD_TIME       = 4.0   # go forward in middle (seconds)
CORRIDOR_EXIT_TURN_TIME     = 1.5   # turn back toward origin lane (seconds)

# Lane change turning from one lane to the other
LANE_CHANGE_TURN_TIME       = 1.5   # seconds (both L->R and R->L)

# Alignment threshold: when we consider lane 'turn' "0"
ALIGN_TURN_EPS              = 0.05  # |turn| <= this => aligned

# =========================
# Lane / camera tunables
# =========================

ROI_VERTICES_RATIO = dict(
    bottom_left = (0.05, 0.98),
    top_left    = (0.22, 0.60),
    top_right   = (0.78, 0.60),
    bottom_right= (0.95, 0.98),
)

HSV_S_MAX = 70
HSV_V_MIN = 170
HSV_H_ANY = (0, 180)

LAB_A_ABS_MAX = 20
LAB_B_ABS_MAX = 20

Y_MIN = 160
CR_ABS_MAX = 14
CB_ABS_MAX = 14

USE_OTSU_BACKUP = True
OTSU_RATIO = 0.60

OPEN_K  = (3, 3)
CLOSE_K = (11, 11)

MIN_AREA             = 300
MIN_HEIGHT           = 40
MIN_WIDTH            = 4
MIN_ASPECT_H_OVER_W  = 1.2
ANGLE_TOL_DEG        = 90

TARGET_X_RATIO_LEFT  = 0.43
TARGET_X_RATIO_RIGHT = 0.65

# PD gains
Kp = 0.9
Kd = 0.2

TEST_TURN_SCALE = 1.0

LANE_COLOR   = (255, 255, 255)
TARGET_COLOR = (0, 255, 0)
TEXT_COLOR   = (0, 255, 255)

MIRROR_FRAME = False

USE_DISPLAY        = False
SHOW_STACKED_DEBUG = False

LEFT_CAM_INDEX  = 2
RIGHT_CAM_INDEX = 0
CAM_BACKEND     = cv2.CAP_V4L2   # or cv2.CAP_ANY

# lane-loss tolerance (we'll use simple stop on long loss)
LOST_FRAMES_THRESH = 5

# =========================
# Serial helpers
# =========================

def open_serial():
    print("Opening serial:", PORT, "@", BAUD)
    ser = serial.Serial(
        PORT, BAUD,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.5,
        write_timeout=1.0,
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
    )
    ser.setDTR(DTR)
    ser.setRTS(RTS)
    time.sleep(BOOT_WAIT)
    return ser

def send_line_typewriter(ser: serial.Serial, text: str):
    data = (text + EOL).encode("ascii", errors="ignore")
    print("TX:", repr(text + EOL), list(data))
    for b in data:
        ser.write(bytes([b]))
        ser.flush()
        time.sleep(CHAR_DELAY)

class MotorCommander:
    def __init__(self, ser):
        self.ser = ser
        self.last_cmd = None

    def send(self, cmd: str):
        if cmd != self.last_cmd:
            send_line_typewriter(self.ser, cmd)
            self.last_cmd = cmd

# =========================
# LIDAR helpers
# =========================

def wrap180(a):
    return (a + 180) % 360 - 180

def rel_to_front(a_abs, angle_sign, front_deg):
    return angle_sign * wrap180(a_abs - front_deg)

def fmt_mm(x):
    return int(x) if x < math.inf else PRINT_INF_AS

def open_lidar(port, baud, timeout):
    lid = RPLidar(port, baudrate=baud, timeout=timeout)
    time.sleep(0.1)
    try:
        lid.start_motor()
    except Exception:
        pass
    # drain garbage
    try:
        if hasattr(lid, 'clean_input'):
            lid.clean_input()
        elif hasattr(lid, 'clear_input'):
            lid.clear_input()
        else:
            try:
                _ = lid._serial.read(4096)
            except:
                pass
    except:
        pass
    try:
        _ = lid.get_info()
    except:
        pass
    try:
        _ = lid.get_health()
    except:
        pass
    return lid

def iter_scans_standard(lidar):
    for scan in lidar.iter_scans(max_buf_meas=8192, min_len=40):
        yield scan

def bins_from_scan(scan, angle_sign, front_deg):
    bins = {a: math.inf for a in LOOK_ANGLES}
    for q, ang, dist in scan:
        if dist <= 0:
            continue
        rel = rel_to_front(ang, angle_sign, front_deg)
        for tgt in LOOK_ANGLES:
            if abs(rel - tgt) <= ANG_TOL and dist < bins[tgt]:
                bins[tgt] = dist
    return bins

def sectors_from_scan(scan, angle_sign, front_deg):
    sectors = {name: math.inf for name in FORWARD_SECTORS}
    for q, ang, dist in scan:
        if dist <= 0:
            continue
        rel = rel_to_front(ang, angle_sign, front_deg)
        for name, (lo, hi) in FORWARD_SECTORS.items():
            if lo <= rel <= hi and dist < sectors[name]:
                sectors[name] = dist
    return sectors

def combine_min(*vals):
    finite = [v for v in vals if not math.isinf(v)]
    return min(finite) if finite else math.inf

class LidarState:
    def __init__(self, angle_sign, front_deg, no_data_stop_sec=NO_DATA_STOP_SEC):
        self.angle_sign = angle_sign
        self.front_deg  = front_deg
        self.no_data_stop_sec = no_data_stop_sec

        self.last_valid_time = time.time()
        self.dL = self.dC = self.dR = math.inf
        self.dFL = self.dF = self.dFR = math.inf

    def update_from_scan(self, scan):
        bins = bins_from_scan(scan, self.angle_sign, self.front_deg)
        dR_raw, dC_raw, dL_raw = bins[-50], bins[0], bins[+50]

        sectors = sectors_from_scan(scan, self.angle_sign, self.front_deg)
        dFR = sectors['FR']
        dF  = sectors['F']
        dFL = sectors['FL']

        dR = combine_min(dR_raw, dFR)
        dC = combine_min(dC_raw, dF)
        dL = combine_min(dL_raw, dFL)

        now = time.time()
        have_any = not (math.isinf(dL) and math.isinf(dC) and math.isinf(dR))
        if have_any:
            self.last_valid_time = now

        self.dL, self.dC, self.dR = dL, dC, dR
        self.dFL, self.dF, self.dFR = dFL, dF, dFR
        return have_any

    def no_data_too_long(self):
        return (time.time() - self.last_valid_time) > self.no_data_stop_sec

# =========================
# Lane / camera helpers
# =========================

def region_selection(image):
    mask = np.zeros_like(image)
    h, w = image.shape[:2]
    bl = (int(w*ROI_VERTICES_RATIO['bottom_left'][0]),  int(h*ROI_VERTICES_RATIO['bottom_left'][1]))
    tl = (int(w*ROI_VERTICES_RATIO['top_left'][0]),     int(h*ROI_VERTICES_RATIO['top_left'][1]))
    tr = (int(w*ROI_VERTICES_RATIO['top_right'][0]),    int(h*ROI_VERTICES_RATIO['top_right'][1]))
    br = (int(w*ROI_VERTICES_RATIO['bottom_right'][0]), int(h*ROI_VERTICES_RATIO['bottom_right'][1]))
    pts = np.array([[bl, tl, tr, br]], dtype=np.int32)
    color = 255 if len(image.shape) == 2 else (255,) * image.shape[2]
    cv2.fillPoly(mask, pts, color)
    return cv2.bitwise_and(image, mask)

def white_mask(bgr):
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ycc  = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    h, s, v   = cv2.split(hsv)
    L, a, b   = cv2.split(lab)
    Y, Cr, Cb = cv2.split(ycc)

    m_hsv = (s <= HSV_S_MAX) & (v >= HSV_V_MIN)
    m_hsv &= (h >= HSV_H_ANY[0]) & (h <= HSV_H_ANY[1])

    m_lab = (np.abs(a.astype(np.int16)-128) <= LAB_A_ABS_MAX) & \
            (np.abs(b.astype(np.int16)-128) <= LAB_B_ABS_MAX)

    m_y   = (Y >= Y_MIN)
    m_cr  = (np.abs(Cr.astype(np.int16)-128) <= CR_ABS_MAX)
    m_cb  = (np.abs(Cb.astype(np.int16)-128) <= CB_ABS_MAX)
    m_ycc = (m_y & m_cr & m_cb)

    mask = (m_hsv & m_lab & m_ycc).astype(np.uint8) * 255

    if USE_OTSU_BACKUP:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thr = int(thr * OTSU_RATIO)
        _, m_gray = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, m_gray)

    return mask

def morph_clean(mask):
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m

def rect_angle_from_vertical(rect):
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    edges = []
    for i in range(4):
        p0 = box[i]; p1 = box[(i+1) % 4]
        v = p1 - p0
        edges.append((np.linalg.norm(v), v))
    edges.sort(key=lambda t: t[0], reverse=True)
    vx, vy = np.abs(edges[0][1][0]), np.abs(edges[0][1][1])
    if vx == 0 and vy == 0:
        return 90.0
    ang_deg = np.degrees(np.arctan2(vx, vy))
    return float(ang_deg)

def score_component(rect):
    (cx, cy), (w, h), _ = rect
    area = max(1.0, w*h)
    height_bonus = max(w, h)**2
    bottom_bias = cy
    return area + 0.3*height_bonus + 0.8*bottom_bias

class SimpleLineTracker:
    def __init__(self, w, target_x_ratio: float):
        self.w = w
        self.target_x_ratio = target_x_ratio
        self.prev_err = 0.0

    def control_from_x(self, line_x_bottom):
        target_x = self.target_x_ratio * self.w

        if line_x_bottom is None:
            err_norm = 0.0
        else:
            err_px = line_x_bottom - target_x
            err_norm = float(np.clip(err_px / (self.w/2), -1.0, 1.0))

        d = err_norm - self.prev_err
        self.prev_err = err_norm
        u = Kp * err_norm + Kd * d
        u = float(np.clip(u, -1.0, 1.0))
        return u, err_norm

def apply_steering(norm_error):
    turn = float(np.clip(norm_error, -1.0, 1.0)) * TEST_TURN_SCALE
    forward = 0.6 * (1 - 0.5*abs(turn))
    print(f"[CTRL] forward={forward:.2f} turn={turn:.2f}")
    return forward, turn

def process_frame(frame_bgr, tracker: SimpleLineTracker):
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top    = int(h * ROI_VERTICES_RATIO['top_left'][1])

    roi = region_selection(frame_bgr)
    mask_white = white_mask(roi)
    mask_white = morph_clean(mask_white)

    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect
        height = max(rw, rh); width = min(rw, rh)
        if height < MIN_HEIGHT or width < MIN_WIDTH:
            continue
        aspect = height / max(1.0, width)
        if aspect < MIN_ASPECT_H_OVER_W:
            continue
        score = score_component(rect)
        if score > best_score:
            best_score = score
            best_rect = rect

    out = frame_bgr.copy()
    mask_debug = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)

    line_x_bottom = None
    angle_deg = None

    if best_rect is not None:
        box = cv2.boxPoints(best_rect).astype(np.int32)
        cv2.drawContours(out, [box], 0, LANE_COLOR, 2)
        cv2.drawContours(mask_debug, [box], 0, (0, 255, 255), 2)

        ys = box[:,1]
        low = box[np.argsort(ys)][-2:]
        line_x_bottom = float(np.mean(low[:,0]))
        cv2.circle(out, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)
        cv2.circle(mask_debug, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)

        angle_deg = rect_angle_from_vertical(best_rect)

    cv2.line(out, (0, y_top), (w-1, y_top), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(mask_debug, (0, y_top), (w-1, y_top), (255, 0, 0), 1, cv2.LINE_AA)

    target_x = int(tracker.target_x_ratio * w)
    cv2.line(out, (target_x, y_top), (target_x, y_bottom), TARGET_COLOR, 2, cv2.LINE_AA)
    cv2.line(mask_debug, (target_x, y_top), (target_x, y_bottom), TARGET_COLOR, 2, cv2.LINE_AA)

    u, err_norm = tracker.control_from_x(line_x_bottom)
    forward, turn = apply_steering(u)

    print(f"[TELEM] LineX={line_x_bottom}  ErrNorm={err_norm:+.2f}  AngleAbsDeg={angle_deg}")

    return out, mask_debug, forward, turn, (angle_deg if angle_deg is not None else 0.0), line_x_bottom, err_norm

def open_camera_by_index(index: int, label: str):
    print(f"[CAM] Trying to open {label} camera at index {index}...")
    cap = cv2.VideoCapture(index, CAM_BACKEND)
    time.sleep(0.3)
    if not cap.isOpened():
        print(f"[CAM] Failed to open {label} camera at index {index}")
        cap.release()
        return None
    ret, frame = cap.read()
    if not ret:
        print(f"[CAM] {label} opened but could not read frame")
        cap.release()
        return None
    print(f"[CAM] Using {label} camera at index {index}")
    return cap

def tune_camera_for_speed(cap, label: str):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] {label} tuned to ~{w:.0f}x{h:.0f} @ {fps:.1f} FPS")

def open_side_camera(side: str):
    if side == "LEFT":
        index = LEFT_CAM_INDEX
        label = "LEFT"
        target_ratio = TARGET_X_RATIO_LEFT
    else:
        index = RIGHT_CAM_INDEX
        label = "RIGHT"
        target_ratio = TARGET_X_RATIO_RIGHT

    cap = open_camera_by_index(index, label)
    if cap is None:
        return None, None

    tune_camera_for_speed(cap, label)
    ret, frame = cap.read()
    if not ret:
        print(f"[CAM] {label}: could not read initial frame")
        cap.release()
        return None, None

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    tracker = SimpleLineTracker(w, target_ratio)
    return cap, tracker

# =========================
# Main integrated state machine
# =========================

def run_integrated():
    ser = open_serial()
    mc  = MotorCommander(ser)
    mc.send(STOP_CMD)
    time.sleep(0.2)
    mc.send(CMD_AUTO)

    # LIDAR
    angle_sign = ANGLE_SIGN_D
    front_deg  = FRONT_DEG
    lidar = open_lidar(LIDAR_PORT, LIDAR_BAUD, LIDAR_TO)
    scans = iter_scans_standard(lidar)
    lidar_state = LidarState(angle_sign, front_deg, NO_DATA_STOP_SEC)
    try:
        _ = next(scans)
    except Exception:
        pass

    # Cameras
    active_side = "LEFT"
    cap, tracker = open_side_camera(active_side)
    if cap is None:
        print("[STATE] Could not open LEFT camera, exiting.")
        return

    state = "CAM_LANE_LEFT"   # starting mode
    state_start_time = time.time()
    corridor_origin = None     # 'LEFT' or 'RIGHT' when in middle

    lost_frames = 0
    prev_time   = time.time()

    if USE_DISPLAY:
        cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)

    try:
        while True:
            # ---- LIDAR step ----
            try:
                scan = next(scans)
            except (StopIteration, RPLidarException, OSError) as e:
                print("[WARN] LIDAR frame issue -> reopen:", e)
                try:
                    lidar.stop(); lidar.stop_motor(); lidar.disconnect()
                except Exception:
                    pass
                time.sleep(0.2)
                lidar = open_lidar(LIDAR_PORT, LIDAR_BAUD, LIDAR_TO)
                scans = iter_scans_standard(lidar)
                try:
                    _ = next(scans)
                except Exception:
                    pass
                continue

            have_any = lidar_state.update_from_scan(scan)
            dL, dC, dR = lidar_state.dL, lidar_state.dC, lidar_state.dR
            dFL, dF, dFR = lidar_state.dFL, lidar_state.dF, lidar_state.dFR

            print(
                f"L={fmt_mm(dL)}  C={fmt_mm(dC)}  R={fmt_mm(dR)}"
                f"  |  FL={fmt_mm(dFL)}  F={fmt_mm(dF)}  FR={fmt_mm(dFR)}",
                end="  "
            )

            if lidar_state.no_data_too_long():
                mc.send(STOP_CMD)
                print("-> STOP (no LIDAR data)")
                time.sleep(0.05)
                continue

            # Near obstacle override: immediate back up
            if min(dL, dC, dR) < NEAR_OBS_TRIP:
                mc.send(CMD_DOWN)
                print("-> BACKOFF (NEAR obstacle)")
                time.sleep(0.05)
                continue

            # Side open/blocked
            openL = dL >= THRESH_MAIN
            openC = dC >= THRESH_MAIN
            openR = dR >= THRESH_MAIN
            left_blocked  = not openL
            right_blocked = not openR

            # ---- Camera handling ----
            frame = None
            out = None
            mask_debug = None
            forward = 0.0
            turn = 0.0
            angle_deg = 0.0
            line_x_bottom = None
            err_norm = 0.0
            lane_visible = False

            cam_state_uses_camera = state.startswith("CAM_LANE") or \
                                    state.startswith("CHANGE_ALIGN") or \
                                    state.startswith("ALIGN_AFTER_CORRIDOR")

            if cam_state_uses_camera:
                if cap is None or tracker is None:
                    print(f"[STATE] {state}: camera/tracker missing, stopping.")
                    mc.send(STOP_CMD)
                    break

                ret, frame = cap.read()
                if not ret:
                    print(f"[CAM] {active_side}: frame grab failed, stopping.")
                    mc.send(STOP_CMD)
                    break

                if MIRROR_FRAME:
                    frame = cv2.flip(frame, 1)

                out, mask_debug, forward, turn, angle_deg, line_x_bottom, err_norm = process_frame(frame, tracker)
                lane_visible = (line_x_bottom is not None)

                if lane_visible:
                    lost_frames = 0
                else:
                    lost_frames += 1
            else:
                # No camera in this state
                lost_frames = 0

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_time))
            prev_time = now

            cmd = STOP_CMD  # default each iteration

            # ===== STATE MACHINE =====

            # 1) Corridor detection (both sides blocked) -> LIDAR-only mode
            if left_blocked and right_blocked and not state.startswith("LIDAR_MIDDLE"):
                # Enter corridor from current "origin"
                corridor_origin = "LEFT" if active_side == "LEFT" else "RIGHT"
                print(f"[STATE] Entering LIDAR_MIDDLE from {corridor_origin}")
                # Turn off any camera
                if cap is not None:
                    cap.release()
                    cap = None
                    tracker = None
                state = "LIDAR_MIDDLE_BALANCE"
                state_start_time = now

            # ----- LIDAR middle corridor states -----
            if state == "LIDAR_MIDDLE_BALANCE":
                # Try to center between left/right obstacles
                if math.isinf(dL) or math.isinf(dR):
                    # Not enough info, just stop for safety
                    cmd = STOP_CMD
                else:
                    diff = dL - dR  # >0: farther from left, closer to right
                    if abs(diff) <= CORRIDOR_BALANCE_EPS_MM and openC:
                        # Centered enough -> next phase
                        print("[STATE] Balanced in middle -> LIDAR_MIDDLE_FORWARD")
                        state = "LIDAR_MIDDLE_FORWARD"
                        state_start_time = now
                        cmd = CMD_UP
                    else:
                        # Move away from closer side
                        if diff > CORRIDOR_BALANCE_EPS_MM:
                            # closer to right -> steer left
                            cmd = CMD_LEFT
                        elif diff < -CORRIDOR_BALANCE_EPS_MM:
                            # closer to left -> steer right
                            cmd = CMD_RIGHT
                        else:
                            cmd = CMD_UP

            elif state == "LIDAR_MIDDLE_FORWARD":
                elapsed = now - state_start_time
                if elapsed < CORRIDOR_FORWARD_TIME and openC:
                    # just go forward down the middle
                    cmd = CMD_UP
                else:
                    # corridor forward complete or blocked; go to exit turn
                    print(f"[STATE] LIDAR_MIDDLE_FORWARD done ({elapsed:.2f}s) -> LIDAR_MIDDLE_EXIT_TURN")
                    state = "LIDAR_MIDDLE_EXIT_TURN"
                    state_start_time = now
                    cmd = CMD_LEFT if corridor_origin == "LEFT" else CMD_RIGHT

            elif state == "LIDAR_MIDDLE_EXIT_TURN":
                elapsed = now - state_start_time
                exit_cmd = CMD_LEFT if corridor_origin == "LEFT" else CMD_RIGHT
                if elapsed < CORRIDOR_EXIT_TURN_TIME:
                    cmd = exit_cmd
                else:
                    # Done turning, open camera for origin side and align
                    side = corridor_origin
                    print(f"[STATE] LIDAR_MIDDLE_EXIT complete -> ALIGN_AFTER_CORRIDOR_{side}")
                    active_side = side
                    cap, tracker = open_side_camera(active_side)
                    if cap is None:
                        print(f"[STATE] Could not open {active_side} camera after corridor, stopping.")
                        mc.send(STOP_CMD)
                        break
                    state = f"ALIGN_AFTER_CORRIDOR_{side}"
                    state_start_time = now
                    cmd = CMD_UP

            # ----- Align after corridor (camera on) -----
            elif state == "ALIGN_AFTER_CORRIDOR_LEFT" or state == "ALIGN_AFTER_CORRIDOR_RIGHT":
                if lane_visible:
                    # We ignore the sign of turn; go straight until |turn| ~ 0
                    if abs(turn) <= ALIGN_TURN_EPS:
                        # Aligned: enter normal lane mode
                        if "LEFT" in state:
                            print("[STATE] ALIGN_AFTER_CORRIDOR_LEFT complete -> CAM_LANE_LEFT")
                            state = "CAM_LANE_LEFT"
                        else:
                            print("[STATE] ALIGN_AFTER_CORRIDOR_RIGHT complete -> CAM_LANE_RIGHT")
                            state = "CAM_LANE_RIGHT"
                        state_start_time = now
                        cmd = CMD_UP
                    else:
                        cmd = CMD_UP
                else:
                    # lane not visible yet -> still go forward
                    cmd = CMD_UP

            # ----- Lane-change: L2R turn phase -----
            elif state == "CHANGE_L2R_TURN":
                elapsed = now - state_start_time
                # During this turn, still check corridor: handled above
                if elapsed < LANE_CHANGE_TURN_TIME:
                    cmd = CMD_RIGHT
                else:
                    # Done initial turn, open RIGHT camera and go to align phase
                    print("[STATE] CHANGE_L2R_TURN done -> CHANGE_ALIGN_TO_RIGHT")
                    active_side = "RIGHT"
                    cap, tracker = open_side_camera(active_side)
                    if cap is None:
                        print("[STATE] Could not open RIGHT camera for change, stopping.")
                        mc.send(STOP_CMD)
                        break
                    state = "CHANGE_ALIGN_TO_RIGHT"
                    state_start_time = now
                    cmd = CMD_UP

            # ----- Lane-change: R2L turn phase -----
            elif state == "CHANGE_R2L_TURN":
                elapsed = now - state_start_time
                if elapsed < LANE_CHANGE_TURN_TIME:
                    cmd = CMD_LEFT
                else:
                    print("[STATE] CHANGE_R2L_TURN done -> CHANGE_ALIGN_TO_LEFT")
                    active_side = "LEFT"
                    cap, tracker = open_side_camera(active_side)
                    if cap is None:
                        print("[STATE] Could not open LEFT camera for change, stopping.")
                        mc.send(STOP_CMD)
                        break
                    state = "CHANGE_ALIGN_TO_LEFT"
                    state_start_time = now
                    cmd = CMD_UP

            # ----- Align after lane-change -----
            elif state == "CHANGE_ALIGN_TO_RIGHT" or state == "CHANGE_ALIGN_TO_LEFT":
                if lane_visible:
                    if abs(turn) <= ALIGN_TURN_EPS:
                        # Done aligning, return to normal lane mode
                        if "RIGHT" in state:
                            print("[STATE] CHANGE_ALIGN_TO_RIGHT complete -> CAM_LANE_RIGHT")
                            state = "CAM_LANE_RIGHT"
                        else:
                            print("[STATE] CHANGE_ALIGN_TO_LEFT complete -> CAM_LANE_LEFT")
                            state = "CAM_LANE_LEFT"
                        state_start_time = now
                        cmd = CMD_UP
                    else:
                        # Still aligning: just go forward
                        cmd = CMD_UP
                else:
                    cmd = CMD_UP

            # ----- Normal lane-follow states -----
            elif state == "CAM_LANE_LEFT" or state == "CAM_LANE_RIGHT":
                # Lane-loss simple behavior: if lost for too long, stop
                if lane_visible:
                    # Check for case 1 / case 3: obstacle on current side only
                    if state == "CAM_LANE_LEFT":
                        if left_blocked and not right_blocked:
                            # Case 1: obstacle on left, right is open
                            print("[STATE] Obstacle on LEFT side -> CHANGE_L2R_TURN")
                            # Turn off left camera
                            if cap is not None:
                                cap.release()
                                cap = None
                                tracker = None
                            state = "CHANGE_L2R_TURN"
                            state_start_time = now
                            cmd = CMD_RIGHT
                        else:
                            # No special obstacle case -> standard lane-follow
                            if turn > TURN_RIGHT_THRESH:
                                cmd = CMD_RIGHT
                            elif turn < TURN_LEFT_THRESH:
                                cmd = CMD_LEFT
                            else:
                                cmd = CMD_UP
                    else:  # CAM_LANE_RIGHT
                        if right_blocked and not left_blocked:
                            # Case 3: obstacle on right, left is open
                            print("[STATE] Obstacle on RIGHT side -> CHANGE_R2L_TURN")
                            if cap is not None:
                                cap.release()
                                cap = None
                                tracker = None
                            state = "CHANGE_R2L_TURN"
                            state_start_time = now
                            cmd = CMD_LEFT
                        else:
                            if turn > TURN_RIGHT_THRESH:
                                cmd = CMD_RIGHT
                            elif turn < TURN_LEFT_THRESH:
                                cmd = CMD_LEFT
                            else:
                                cmd = CMD_UP
                else:
                    # Lane temporarily missing
                    if lost_frames >= LOST_FRAMES_THRESH:
                        print(f"[STATE] Lane lost in {state} for {lost_frames} frames -> STOP")
                        cmd = STOP_CMD
                    else:
                        cmd = STOP_CMD

            else:
                # Unknown state -> safe stop
                print(f"[STATE] Unknown state '{state}', stopping.")
                cmd = STOP_CMD

            mc.send(cmd)
            print(f"-> state={state} cmd={cmd}")

            # Display if enabled
            if USE_DISPLAY and frame is not None:
                tl_text = f"{active_side} | state={state} cmd={cmd}"
                if SHOW_STACKED_DEBUG and out is not None and mask_debug is not None:
                    # simple 2x2 debug view
                    h = 300
                    raw_r = cv2.resize(frame, (int(frame.shape[1]*h/frame.shape[0]), h))
                    md_r  = cv2.resize(mask_debug, (raw_r.shape[1], h))
                    out_r = cv2.resize(out, (raw_r.shape[1], h))
                    debug_view = np.vstack((np.hstack((raw_r, md_r)), np.hstack((out_r, out_r*0))))
                    cv2.putText(debug_view, tl_text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
                else:
                    debug_view = out if out is not None else frame.copy()
                    cv2.putText(debug_view, tl_text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)
                cv2.putText(debug_view, f"FPS: {fps:.1f}", (10, debug_view.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.imshow("Lane Debug", debug_view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[STATE] Quit requested.")
                    break
            else:
                time.sleep(0.001)

    finally:
        try:
            mc.send(STOP_CMD)
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        if USE_DISPLAY:
            cv2.destroyAllWindows()
        try:
            lidar.stop(); lidar.stop_motor(); lidar.disconnect()
        except Exception:
            pass
        ser.close()
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    run_integrated()

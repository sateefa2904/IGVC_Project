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

Added safety hooks:
- Stop sign placeholder
- Pedestrian stop placeholder (blue dot)

"""
# =========================================================
# WARNING / CURRENT DEBUG STATUS
# [SAteefa 3/22]
# camera_part3.py is now the main integrated controller for:
#   1) camera lane-following
#   2) lidar obstacle handling
#   3) future stop-sign / pedestrian stop hooks
#
# Current testing priority:
#   Phase 1 = verify camera-only lane following works
#   Phase 2 = verify lidar-only obstacle behavior works
#   Phase 3 = verify full camera + lidar integration works
#
# NOTE:
# Stop-sign / pedestrian code is added into the file,
# but should remain disabled until base driving behavior is verified.
# =========================================================


import time
import math
import numpy as np
import cv2
import serial
from rplidar import RPLidar, RPLidarException
import os
import sys
import fcntl
import atexit
import signal
from ultralytics import YOLO
from multiprocessing import Process, Queue, Manager
from multiprocessing import Value, Array
from multiprocessing import shared_memory
import subprocess
# =========================
# Debug / integration toggles
# =========================
# [SAteefa 3/21 Added: top-level mode switch so we can test one subsystem at a time]
# CAMERA_ONLY = test only lane-following + TM4C output
# LIDAR_ONLY  = test only obstacle / corridor logic
# FULL        = full integrated behavior

 #CAMERA_ONLY, LIDAR_ONLY, FULL



GLOBAL_MC = None
GLOBAL_SERIAL_LOCK = None
GLOBAL_SER = None

# =========================
# Serial / driving config
# =========================

SERIAL_LOCK_PATH = "/tmp/tm4c_serial.lock" # DEBUG Phase 1: [SAteefa 3/21: added to enforce one-way pathway and prevent interleaving]
PORT = "/dev/ttyACM0"   # change if needed
BAUD = 19200
EOL = "\r"              # TM4C parser ends on carriage returnn
DTR = True
RTS = True
CHAR_DELAY = 0.005       # 10 ms between bytes
BOOT_WAIT = 2.0         # wait after opening serial for TM4C boot

CMD_UP     = "Forward Half"
CMD_DOWN   = "Backward Half"
CMD_LEFT   = "Left Half"
CMD_RIGHT  = "Right Half"
CMD_AUTO   = "Auto"
CMD_MANUAL = "Manual"
STOP_CMD   = "Stop"

# For deciding steering commands from lane PD output
#TURN_LEFT_THRESH  = -0.20
#TURN_RIGHT_THRESH =  0.20

##new to avoid wheel spasm:
# [SAteefa 3/22 Added: hysteresis thresholds to reduce left/right command chatter]
# LEFT_ENTER_THRESH  = -0.24
# LEFT_EXIT_THRESH   = -0.12
# RIGHT_ENTER_THRESH =  0.24
# RIGHT_EXIT_THRESH  =  0.12

LEFT_ENTER_THRESH  = -0.10
LEFT_EXIT_THRESH   = -0.05
RIGHT_ENTER_THRESH =  0.10
RIGHT_EXIT_THRESH  =  0.05


# =========================
# LIDAR config / thresholds
# =========================

LIDAR_PORT   = "/dev/ttyUSB0"
LIDAR_BAUD   = 1_000_000
LIDAR_TO     = 1.0
FRONT_DEG    = 0
ANGLE_SIGN_D = -1    # -1 if your LIDAR mount is mirrored

LOOK_ANGLES  = [-50, 0, +50]
ANG_TOL      = 10

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

USE_OTSU_BACKUP = False
OTSU_RATIO = 0.60

OPEN_K  = (3, 3)
CLOSE_K = (11, 11)

MIN_AREA             = 80 #old: 100
MIN_HEIGHT           = 15 #old: 20
MIN_WIDTH            = 2
MIN_ASPECT_H_OVER_W  = 0.6 #old: 1.0
ANGLE_TOL_DEG        = 90


#### changing ratios to avoid spasing
TARGET_X_RATIO_LEFT = 0.32 # old 0.43
TARGET_X_RATIO_RIGHT = 0.68 # old 0.65

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
SAVE_DEBUG_IMAGES = True



RIGHT_CAM_INDEX = "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.2:1.0-video-index0"
LEFT_CAM_INDEX = "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.3:1.0-video-index0"
CAM_BACKEND     = cv2.CAP_V4L2   # or cv2.CAP_ANY

# lane-loss tolerance (we'll use simple stop on long loss)
LOST_FRAMES_THRESH = 5

# =========================
# Stop sign / pedestrian draft config
# =========================

STOP_SIGN_HOLD_SEC = 2.0  # [SAteefa 3/21 Added: placeholder timed stop for future stop-sign integration]



# =========================
# Serial helpers
# =========================

# DEBUG Phase 1: [SAteefa 3.21] Helper function for open_serial

#[SAteefa 3/21 Added: lock-file helper to ensure only one script owns the TM4c serial port at a time]
# Prevents multiple scripts from writing to TM4C at the same time by using an exclusive lock file
class SerialPortLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path   #path to the lock file used to cooredinate port ownership
        self.fd = None              #file descriptor for the lock file
    def acquire(self):
        #[SAteefa 3/21 Added: open lock file in a+ mode so we can both write our PID and read existing owner's PID]
        #Using "w" would erase contents and would not let us read back the current owner properly
        self.fd = open(self.lock_path, "a+")
        try:
            #[SAteefa 3/21 Added: request a non-blocking exclusive lock]
            # If this succeeds, this script becomes the only allowed owner of the TM4c serial resource
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            #[SAteefa 3/21 Added: clear old contents, then store this process ID in the lock file]
            # This helps identify which process currently owns the serial port.
            self.fd.seek(0)
            self.fd.truncate()
            self.fd.write(str(os.getpid()))
            self.fd.flush()


            print(f"[LOCK] Acquired TM4C serial lock: {self.lock_path}")
        except BlockingIOError:
            # [SAteefa 3/21 Added: if another process already owns the lock, read its PID for a clearer error message]
            self.fd.seek(0)
            owner = self.fd.read().strip()


            raise RuntimeError(
                f"TM4c serial port is already owned by another process"
                f"{' (PID ' + owner + ')' if owner else ''}. "
                f"Stop the other script before running this one."
            )
    def release(self):
        if self.fd is not None:
            try:
                #[SAteefa 3/21 ADDED: clear the stored PID and release the exclusive lock during shutdown.]
                #This prevents ownership from being left behind after the script exits
                self.fd.seek(0)
                self.fd.truncate()
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                print(f"[LOCK] Released TM4C serial lock : {self.lock_path}")
            except Exception:
                pass
            try:
                #close the lock-file handle after unlocking
                self.fd.close()
            except Exception:
                pass
            self.fd = None


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


# [SAteefa 3/22 Added: TM4C command sender]
# Sends commands while:
#   - suppressing exact duplicates
#   - reducing rapid serial spam/jitter
#   - allowing force=True for critical commands like STOP and AUTO
class MotorCommander:
    def __init__(self,ser,min_cmd_interval=0.08):
        self.ser = ser              #serial connection to TM4C
        self.last_cmd = None        #remembers the last command sent
        self.last_send_time = 0.0   # [SAteefa 3/21 added: remembers when the last command was sent]
        self.min_cmd_interval = min_cmd_interval # [SAteefa 3/21 added: minimum delay between non-forced command sends]
    def send(self, cmd:str, force:bool=False):
        now = time.time()

        same_cmd = (cmd == self.last_cmd)
        too_soon = (now - self.last_send_time) < self.min_cmd_interval

        # send immediately if forced
        if force:
            send_line_typewriter(self.ser, cmd)
            self.last_cmd = cmd
            self.last_send_time = now
            return

        # send if command changed, or if same command but refresh interval has passed
        if (not same_cmd) or (not too_soon):
            send_line_typewriter(self.ser, cmd)
            self.last_cmd = cmd
            self.last_send_time = now

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
            except Exception:
                pass
    except Exception:
        pass

    try:
        _ = lid.get_info()
    except Exception:
        pass
    try:
        _ = lid.get_health()
    except Exception:
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

# [SAteefa 3/22 Added: hysteresis-based discrete steering decision]
# Keeps the controller from flipping between LEFT and FORWARD on tiny frame-to-frame noise.
def choose_cmd_with_hysteresis(turn, steer_state):
    if steer_state == "LEFT":
        if turn > LEFT_EXIT_THRESH:
            steer_state = "STRAIGHT"
    elif steer_state == "RIGHT":
        if turn < RIGHT_EXIT_THRESH:
            steer_state = "STRAIGHT"
    else:
        if turn < LEFT_ENTER_THRESH:
            steer_state = "LEFT"
        elif turn > RIGHT_ENTER_THRESH:
            steer_state = "RIGHT"

    if steer_state == "LEFT":
        return CMD_LEFT, steer_state
    elif steer_state == "RIGHT":
        return CMD_RIGHT, steer_state
    else:
        return CMD_UP, steer_state

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
    if SAVE_DEBUG_IMAGES:
        cv2.imwrite("output_mask.jpg", mask_white)
        cv2.imwrite("output.jpg", out)
        cv2.imwrite("input.jpg", frame_bgr)
        cv2.imwrite("mask.jpg", mask_debug)


    return out, mask_debug, forward, turn, (angle_deg if angle_deg is not None else 0.0), line_x_bottom, err_norm

def open_camera_by_index(label: str):
    path = LEFT_CAM_INDEX if label == "LEFT" else RIGHT_CAM_INDEX
    print(f"[CAM] Trying to open {label} camera at index {path}...")
    cap = cv2.VideoCapture(path, CAM_BACKEND)
    time.sleep(0.3)
    if not cap.isOpened():
        print(f"[CAM] Failed to open {label} camera at index {path}")
        cap.release()
        return None
    ret, frame = cap.read()
    if not ret:
        print(f"[CAM] {label} opened but could not read frame")
        cap.release()
        return None
    print(f"[CAM] Using {label} camera at index {path}")
    return cap

def tune_camera_for_speed(cap, label: str):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320) #previous 960
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320) #previous 540
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] {label} tuned to ~{w:.0f}x{h:.0f} @ {fps:.1f} FPS")

def open_side_camera(side: str):
    cap = open_camera_by_index(side)
    
    if cap is None:
        return None
    
    tune_camera_for_speed(cap, side)
    
    return cap



def read_camera_step(cap, tracker, active_side):
    # [SAteefa 3/21 Added: isolated camera read/process logic to make lane debugging cleaner]
    if cap is None or tracker is None:
        return None, None, None, None, None, None, None, None, False

    ret, frame = cap.read()
    
    if not ret:
        print(f"[CAM] {active_side}: frame grab failed.")
        return None, None, None, None, None, None, None, None, False

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)
    #cv2.imshow('Camera View', frame)
   

    out, mask_debug, forward, turn, angle_deg, line_x_bottom, err_norm = process_frame(frame, tracker)
    lane_visible = (line_x_bottom is not None)

    return frame, out, mask_debug, forward, turn, angle_deg, line_x_bottom, err_norm, lane_visible

# =========================
# Stop sign / pedestrian hooks
# =========================
# [SAteefa 3/22 Warning: phase-2 only]
# These hooks are intentionally added now so the architecture is ready,
# but detection logic still need edits before use.


class StopEventState:
    # [SAteefa 3/21 Added: helper state for timed stop-sign holding behavior]
    def __init__(self):
        self.active = False
        self.end_time = 0.0


def handle_stop_sign(stop_detected, stop_state, now):
    # [SAteefa 3/21 Added: placeholder timed stop-sign behavior]
    if stop_state.active:
        if now < stop_state.end_time:
            return True
        stop_state.active = False

    if stop_detected:
        stop_state.active = True
        stop_state.end_time = now + STOP_SIGN_HOLD_SEC
        return True

    return False

# =========================
# Boot / cleanup helper functions
# =========================

# def emergency_shutdown(signum):
#     print(f"\n[SIGNAL] Caught signal {signum}, attempting safe stop...")

#     try:
#         if GLOBAL_MC is not None:
#             GLOBAL_MC.send(STOP_CMD, force=True)
#             time.sleep(0.2)
#     except Exception as e:
#         print("[SIGNAL] Failed to send stop:", e)

#     try:
#         if GLOBAL_SER is not None:
#             GLOBAL_SER.close()
#     except Exception:
#         pass

#     try:
#         if GLOBAL_SERIAL_LOCK is not None:
#             GLOBAL_SERIAL_LOCK.release()
#     except Exception:
#         pass

#     raise SystemExit(0)

def initialize_system():
    # [SAteefa 3/21 Added: modular startup helper so all debug phases share the same safe initialization]
    serial_lock = SerialPortLock(SERIAL_LOCK_PATH)
    serial_lock.acquire()
    atexit.register(serial_lock.release)

    ser = open_serial()
    mc = MotorCommander(ser)

    global GLOBAL_MC, GLOBAL_SERIAL_LOCK, GLOBAL_SER
    GLOBAL_MC = mc
    GLOBAL_SERIAL_LOCK = serial_lock
    GLOBAL_SER = ser

    # signal.signal(signal.SIGINT, emergency_shutdown)   # Ctrl+C
    # signal.signal(signal.SIGTERM, emergency_shutdown)  # kill
    # signal.signal(signal.SIGTSTP, emergency_shutdown)  # Ctrl+Z


    # [SAteefa 3/21 Added: startup banner to show which script and hardware ports are active]
    print("=" * 60)
    print("[BOOT] camera_part3.py starting")
    print(f"[BOOT] DEBUG_MODE : FULL")
    print(f"[BOOT] TM4C port  : {PORT}")
    print(f"[BOOT] LIDAR port : {LIDAR_PORT}")
    print(f"[BOOT] Active side starts on: LEFT")
    print("=" * 60)

    mc.send(STOP_CMD, force=True)
    time.sleep(0.2)
    mc.send(CMD_AUTO, force=True)

    return serial_lock, ser, mc

#Isaac Elizarraraz [3/30/2026]
#This function is passed a frame and returns a list of detected objects

def detect_objects(frame,detector:YOLO):
    detections = []
    results = detector(frame, imgsz=320, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = detector.names[cls]

            if label == 'stop sign':
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "bbox": (x1,y1,x2,y2),
                    "confidence": float(box.conf[0])
                })
            # elif label == 'new object':
            #     print('no detection')
            #     #Isaac Elizarraraz [3/30/2026]
            #     #the code to implement another object is the same assuming the model is trained for it
            #     #for now only the stop sign is being tested
            #     #pedestrian and potholes have to be added later
    return detections


#Isaac Elizarraraz [3/30/2026]
#helper function for object detection handling
#measures the distance from an object by the size of its bounding box
#the measurement may need to be calibrated based on the cameras focal length
def measure_distance(bbox,focal_length,real_height):
    """
    focal length is a pre-calibrated focal langth from our camera. Needs to be confirmed, currently a placeholder
    real_height is the real height of the object being detected
    distance is returned in meters
    """
    x1,y1,x2,y2 = bbox
    height = abs(y2-y1)

    if height <= 0: return None

    distance = (real_height * focal_length)/(height)
    return distance

def cleanup_system(mc=None, ser=None, serial_lock=None):
    try:
        if mc is not None:
            mc.send(STOP_CMD, force=True) #stop the robot before shutting anything down
    except Exception:
        pass

    if USE_DISPLAY:
        try:
            cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print("[WARN] OpenCV display unavailable, disabling USE_DISPLAY:", e)
            globals()["USE_DISPLAY"] = False

    try:
        if ser is not None:
            ser.close() #close the TM4c serial connection
    except Exception:
        pass

    try:
        if serial_lock is not None:
            serial_lock.release() # [SAteefa 3/21 Added: release the serial-port ownership lock so another script can use TM4C later]
    except Exception:
        pass
 

    print("[INFO] Clean exit.") #confirmation that shutdown completed

def show_debug_view(frame, out, mask_debug, active_side, state, cmd, fps):
    if not USE_DISPLAY or frame is None:
        return False

    tl_text = f"{active_side} | state={state} cmd={cmd}"

    if SHOW_STACKED_DEBUG and out is not None and mask_debug is not None:
        h = 300
        raw_r = cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h))
        md_r = cv2.resize(mask_debug, (raw_r.shape[1], h))
        out_r = cv2.resize(out, (raw_r.shape[1], h))
        debug_view = np.vstack((np.hstack((raw_r, md_r)), np.hstack((out_r, out_r * 0))))
        cv2.putText(debug_view, tl_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    else:
        debug_view = out if out is not None else frame.copy()
        cv2.putText(debug_view, tl_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

    cv2.putText(debug_view, f"FPS: {fps:.1f}", (10, debug_view.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Lane Debug", debug_view)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q') or key == 27


# =========================
# Full integrated state machine
# =========================

# [SAteefa 3/22 Added: main integrated control loop]
# This function is the single owner of TM4C output.
# It combines:
#   - serial/TM4C setup
#   - lidar updates
#   - camera lane detection
#   - state-machine transitions
#   - final motor command selection
#
# Because this script is the integrated controller, no additional
# driving script should write to the TM4C serial port at the same time.

def read_lidar_step(lidar, scans, lidar_state):
    # [SAteefa 3/21 Added: isolated LIDAR read/reopen logic to make failures easier to debug]
    try:
        scan = next(scans)
    except (StopIteration, RPLidarException, OSError) as e:
        print("[WARN] LIDAR frame issue -> reopen:", type(e).__name__, repr(e))
        try:
            lidar.clean_input()
            time.sleep(0.1)
            scan = next(scans)
            # lidar.stop()
            # lidar.stop_motor()
            # lidar.disconnect()
        except Exception:
            pass


        lidar = open_lidar(LIDAR_PORT, LIDAR_BAUD, LIDAR_TO)
        
        scans = iter_scans_standard(lidar)

        try:
            _ = next(scans)
        except Exception:
            pass

        return lidar, scans, lidar_state, False

    lidar_state.update_from_scan(scan)
    return lidar, scans, lidar_state, True

def camera_worker(shared,shm_name):
    current_side = "LEFT"
    cap = open_side_camera(current_side)
    if cap is None: 
        print("[CAM] failed to start")
        return
    
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((320,320,3), dtype=np.uint8, buffer=shm.buf)
    print("[CAMERA PROCESS] started")

    while True:
        # Check if the main loop wants a different camera
        requested_side = "LEFT" if shared["active_camera"].value == 0 else "RIGHT"
        
        if requested_side != current_side:
            print(f"[CAM] Switching to {requested_side}")
            cap.release() # Must release the old hardware handle!
            cap = open_side_camera(requested_side)
            current_side = requested_side
            
        ret, img = cap.read()

        if not ret:
            continue
        
        img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        
        if MIRROR_FRAME:
            img = cv2.flip(img, 1)
            
        #cv2.imwrite("camera_worker.jpg", img)
        frame[:] = img
        

def lane_worker(shared,shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((320,320,3), dtype=np.uint8, buffer=shm.buf)
    last_active_side = shared["active_camera"].value
    target_ratio = TARGET_X_RATIO_LEFT if shared["active_camera"].value == 0 else TARGET_X_RATIO_RIGHT
    tracker = SimpleLineTracker(320,target_ratio)
    #h, w = frame.shape[:2]

    print("[LANE PROCESS] started")
    while True:

        img = frame.copy()  # optional safety
        #cv2.imwrite("lane_worker.jpg", img)

        if img is None:
            time.sleep(0.01)
            continue
        
        if last_active_side != shared["active_camera"].value:
            last_active_side = shared["active_camera"].value
            target_ratio = TARGET_X_RATIO_LEFT if shared["active_camera"].value == 0 else TARGET_X_RATIO_RIGHT    
            tracker.target_x_ratio = target_ratio


        try:
            _, _, forward, turn, _, line_x_bottom, _ = process_frame(img, tracker)

            shared["lane_turn"].value = turn
            shared["lane_visible"].value = (line_x_bottom is not None)

        except Exception as e:
            print("[LANE ERROR]", e)
      
#Isaac Elizarraraz [3/31/2026]
#thread worker that finds objects in current frame
def yolo_worker(shared,shm_name):

    detector = YOLO("./yolov8n.pt")
    detector.to("cpu")
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((320,320,3), dtype=np.uint8, buffer=shm.buf)
    stop_detected = False
    real_height = 0.3048
    focal_length = 650
    stop_frame_counter = 0
    stop_latched = False
    
    print("[YOLO PROCESS] started")
    while True:
        stop_detected = False
        img = frame.copy()  # optional safety
        #cv2.imwrite("lane_worker.jpg", img)

        if img is None:
            time.sleep(0.02)
            continue

        try:

            detections = detect_objects(img, detector)
            min_distance = float("inf") #keep track of how far away the stop sign is. car stops 2 meters away
            if detections:
                shared["det_time"].value = time.time()

            for d in detections:
                if d["label"] == "stop sign":
                    stop_detected = True
                    stop_frame_counter += 1
                    
                    distance = measure_distance(
                            d["bbox"],
                            focal_length,
                            real_height
                        )
                    min_distance = min(min_distance,distance)
                    #print(f"STOP SIGN IS {distance} METERS AWAY")
                    continue
            
            if not stop_detected:
                stop_frame_counter = 0
                
            if stop_frame_counter >= 3 and min_distance <= 5:
                if not stop_latched:
                    shared["stop_event"].value = True
                    stop_latched = True
            else:
                shared["stop_event"].value = False
                if stop_frame_counter == 0:
                    stop_latched = False
            time.sleep(0.02)
            
        except Exception as e:
            print("[YOLO ERROR]", e)
            
def lidar_worker(shared):

    lidar = open_lidar(LIDAR_PORT, LIDAR_BAUD, LIDAR_TO)
    scans = iter_scans_standard(lidar)
    lidar_state = LidarState(ANGLE_SIGN_D, FRONT_DEG, NO_DATA_STOP_SEC)
    print("[LIDAR PROCESS] started")
    
    bad_streak = 0
    BAD_LIMIT = 5
    GRACE_LIMIT = 10
    grace_frames = GRACE_LIMIT

    while True:

        try:
            scan = next(scans)
            bad_streak = 0

        except (StopIteration, RPLidarException, OSError) as e:

            if grace_frames > 0:
                grace_frames -= 1
                continue

            bad_streak += 1

            if bad_streak >= BAD_LIMIT:

                print(f"[WARN] lidar scan error -> reconnect")

                bad_streak = 0

                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except Exception:
                    pass

                time.sleep(0.25)

                lidar = open_lidar(LIDAR_PORT, LIDAR_BAUD, LIDAR_TO)
                scans = iter_scans_standard(lidar)

                grace_frames = GRACE_LIMIT

            continue
        
        lidar_state.update_from_scan(scan)

        shared["dL"].value = lidar_state.dL
        shared["dC"].value = lidar_state.dC
        shared["dR"].value = lidar_state.dR

        shared["dFL"].value = lidar_state.dFL
        shared["dF"].value  = lidar_state.dF
        shared["dFR"].value = lidar_state.dFR
        
        

def run_full_integrated(mc, shared):

    active_side = "LEFT"

    state = "CAM_LANE_LEFT"
    state_start_time = time.time()
    corridor_origin = None

    lost_frames = 0
    steer_state = "STRAIGHT"

    last_completed_stop = time.time()
    stop_state_end_time = 0
    try:
        while True:

            now = time.time()

            # =========================
            # Read shared sensor data
            # =========================

            dL = shared["dL"].value
            dC = shared["dC"].value
            dR = shared["dR"].value

            dFL = shared["dFL"].value
            dF  = shared["dF"].value
            dFR = shared["dFR"].value

            lane_visible = shared["lane_visible"].value
            turn = shared["lane_turn"].value
            
            print(
                f"L={fmt_mm(dL)}  C={fmt_mm(dC)}  R={fmt_mm(dR)}"
                f"  |  FL={fmt_mm(dFL)}  F={fmt_mm(dF)}  FR={fmt_mm(dFR)}",
                end="  "
            )

            # =========================
            # Stop sign logic
            # =========================

            stop_event = shared["stop_event"].value
            
            if now < stop_state_end_time:
                mc.send(STOP_CMD)
                continue
            # finished stopping → update cooldown
            if stop_state_end_time != 0 and now >= stop_state_end_time:
                last_completed_stop = now
                stop_state_end_time = 0
                print("[STOP] Completed stop")

            # trigger new stop
            if stop_event and (now - last_completed_stop > 10):
                print("[STOP] Triggered")
                stop_state_end_time = now + 8
                shared["stop_event"].value = False  # consume event
                mc.send(STOP_CMD, force=True)
                continue
                
            
            # =========================
            # LIDAR safety overrides
            # =========================

            if min(dL, dC, dR) < NEAR_OBS_TRIP:
                mc.send(CMD_DOWN)
                print("-> BACKOFF (NEAR obstacle)")
                time.sleep(0.05)
                continue

            openL = dL >= THRESH_MAIN
            openC = dC >= THRESH_MAIN
            openR = dR >= THRESH_MAIN

            left_blocked = not openL
            right_blocked = not openR

            cmd = STOP_CMD

            # =========================
            # STATE MACHINE
            # =========================

            if left_blocked and right_blocked and not state.startswith("LIDAR_MIDDLE"):

                corridor_origin = active_side

                print(f"[STATE] Entering LIDAR_MIDDLE from {corridor_origin}")

                steer_state = "STRAIGHT"
                state = "LIDAR_MIDDLE_BALANCE"
                state_start_time = now


            if state == "LIDAR_MIDDLE_BALANCE":

                if math.isinf(dL) or math.isinf(dR):
                    cmd = STOP_CMD

                else:

                    diff = dL - dR

                    if abs(diff) <= CORRIDOR_BALANCE_EPS_MM and openC:

                        print("[STATE] Balanced -> LIDAR_MIDDLE_FORWARD")

                        state = "LIDAR_MIDDLE_FORWARD"
                        state_start_time = now
                        cmd = CMD_UP

                    else:

                        if diff > CORRIDOR_BALANCE_EPS_MM:
                            cmd = CMD_LEFT
                        elif diff < -CORRIDOR_BALANCE_EPS_MM:
                            cmd = CMD_RIGHT
                        else:
                            cmd = CMD_UP


            elif state == "LIDAR_MIDDLE_FORWARD":

                elapsed = now - state_start_time

                if elapsed < CORRIDOR_FORWARD_TIME and openC:
                    cmd = CMD_UP

                else:

                    print("[STATE] -> LIDAR_MIDDLE_EXIT_TURN")

                    state = "LIDAR_MIDDLE_EXIT_TURN"
                    state_start_time = now

                    cmd = CMD_LEFT if corridor_origin == "LEFT" else CMD_RIGHT


            elif state == "LIDAR_MIDDLE_EXIT_TURN":

                elapsed = now - state_start_time

                exit_cmd = CMD_LEFT if corridor_origin == "LEFT" else CMD_RIGHT

                if elapsed < CORRIDOR_EXIT_TURN_TIME:
                    cmd = exit_cmd

                else:

                    active_side = corridor_origin

                    state = f"CAM_LANE_{active_side}"
                    state_start_time = now
                    cmd = CMD_UP


            elif state.startswith("CAM_LANE"):

                if lane_visible:
                    lost_frames = 0

                    if state == "CAM_LANE_LEFT":

                        if left_blocked and not right_blocked:

                            print("[STATE] Obstacle LEFT -> CHANGE_L2R")

                            steer_state = "STRAIGHT"
                            state = "CHANGE_L2R_TURN"
                            state_start_time = now
                            cmd = CMD_RIGHT

                        else:

                            cmd, steer_state = choose_cmd_with_hysteresis(turn, steer_state)

                    else:

                        if right_blocked and not left_blocked:

                            print("[STATE] Obstacle RIGHT -> CHANGE_R2L")

                            steer_state = "STRAIGHT"
                            state = "CHANGE_R2L_TURN"
                            state_start_time = now
                            cmd = CMD_LEFT

                        else:

                            cmd, steer_state = choose_cmd_with_hysteresis(turn, steer_state)

                else:

                    lost_frames += 1

                    if lost_frames >= LOST_FRAMES_THRESH:

                        print("[STATE] Lane lost -> STOP")

                        steer_state = "STRAIGHT"
                        cmd = STOP_CMD

                    else:

                        cmd = STOP_CMD


            mc.send(cmd)

            print(f"-> state={state} cmd={cmd}")

            time.sleep(0.005)

    finally:
        mc.send(STOP_CMD, force=True)
        


def kill_leftover_processes():
    # This command finds any other Python scripts running this file and kills them
    current_pid = os.getpid()
    cmd = "pgrep -f integrated_v4.py"
    try:
        pids = subprocess.check_output(cmd, shell=True).decode().split()
        for pid in pids:
            if int(pid) != current_pid:
                print(f"[BOOT] Killing ghost process: {pid}")
                os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass # No other processes found

# =========================
# Main entry point
# =========================

#Isaac Elizarraz [3/30/2026]
#added pretrained yolov8 model to detect stop signs

def run_integrated():


    serial_lock = None
    ser = None
    mc = None

    frame_shape = (320,320,3)
    frame_size = np.prod(frame_shape)

    shared = {}
    shm = shared_memory.SharedMemory(create=True, size=frame_size) #shared frame memory
    shared_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
    
    # LIDAR
    shared["dL"] = Value('d', float('inf'))
    shared["dC"] = Value('d', float('inf'))
    shared["dR"] = Value('d', float('inf'))

    shared["dFL"] = Value('d', float('inf'))
    shared["dF"]  = Value('d', float('inf'))
    shared["dFR"] = Value('d', float('inf'))
    
    # camera
    shared["active_camera"] = Value('i', 0) # 0 = LEFT, 1 = RIGHT
    
    # lane tracking
    shared["lane_turn"] = Value('d', 0.0)
    shared["lane_visible"] = Value('b', False)
    
    # YOLO detection
    shared["stop_detected"] = Value('b', False)
    shared["stop_event"] = Value('b', False)
    shared["det_time"] = Value('d', 0.0)

    camera_proc = Process(
        target=camera_worker,
        args=(shared,shm.name),
        daemon=True
    )
    
    lidar_proc = Process(
        target=lidar_worker,
        args=(shared,),
        daemon=True
    )

    yolo_proc = Process(
        target=yolo_worker,
        args=(shared,shm.name),
        daemon=True
    )

    lane_proc = Process(
        target=lane_worker,
        args=(shared,shm.name),
        daemon=True
    )

    camera_proc.start()
    lidar_proc.start()
    yolo_proc.start()
    lane_proc.start()
    
    

    print("[MAIN] sensor processes started")
    print("[BOOT] Waiting for sensors to stabilize...")
    start_wait = time.time()
    sensors_ready = False
    
    while time.time() - start_wait < 10:  # 10-second timeout
        # Check if all sensors have reported at least one valid value
        lidar_ok = not math.isinf(shared["dC"].value)
        cam_ok   = (shared["det_time"].value > 0) # YOLO has processed one frame
        
        if lidar_ok and cam_ok:
            sensors_ready = True
            print(f"[BOOT] All systems GO! (Stabilized in {time.time()-start_wait:.2f}s)")
            break
        
        print(f"  > Waiting... LIDAR: {'OK' if lidar_ok else '...'}, YOLO: {'OK' if cam_ok else '...'}")
        time.sleep(0.5)

    if not sensors_ready:
        print("[FATAL] Sensor timeout. Check hardware connections.")
        # Cleanup and exit here


    try:
        serial_lock, ser, mc = initialize_system()

        run_full_integrated(mc, shared)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C")
    finally:
        print("[MAIN] Shutting down workers...")
        for p in [lidar_proc, yolo_proc, lane_proc, camera_proc]:
            if p.is_alive():
                p.terminate()

        for p in [lidar_proc, yolo_proc, lane_proc, camera_proc]:
            p.join()
            try:
                shm.close()   # Close this process's view
                shm.unlink()  # Permanently destroy the segment from the OS
                print("[MAIN] SharedMemory unlinked successfully.")
            except Exception as e:
                print(f"[MAIN] Error unlinking SHM: {e}")
        cleanup_system(mc=mc, ser=ser, serial_lock=serial_lock)
        

  
        
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_integrated()
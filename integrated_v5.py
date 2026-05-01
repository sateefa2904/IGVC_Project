"""
Autonomous Vehicle Integration System
=====================================

This module serves as the primary entry point for the self-driving ground vehicle.
It utilizes Python's `multiprocessing` library to run parallel workers for:
- Camera capture and lane tracking (OpenCV)
- Object detection (Ultralytics YOLOv8)
- Obstacle avoidance (RPLidar)
- Motor control and telemetry (Serial to TM4C)

The workers communicate via shared memory and thread-safe multiprocessing Values.
"""

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
from http.server import BaseHTTPRequestHandler
import threading
import socketserver
from typing import List, Optional, Tuple
import multiprocessing as mp

# =========================
# Serial / driving config
# =========================

SERIAL_LOCK_PATH = "/tmp/tm4c_serial.lock" # DEBUG Phase 1: [SAteefa 3/21: added to enforce one-way pathway and prevent interleaving]
PORT = "/dev/ttyACM0"   # change if needed
BAUD = 19200
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

LIDAR_PORT   = "/dev/ttyUSB0"
LIDAR_BAUD   = 1_000_000
LIDAR_TO     = 1.0

LEFT_CAM_INDEX = "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.1.1:1.0-video-index0"
RIGHT_CAM_INDEX = "/dev/v4l/by-path/platform-3610000.usb-usb-0:1.2:1.0-video-index0"
CAM_BACKEND     = cv2.CAP_V4L2   # or cv2.CAP_ANY


USE_DISPLAY        = False
SHOW_STACKED_DEBUG = False



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
    """
    Initializes and opens the serial connection to the TM4C microcontroller.

    Returns:
        serial.Serial: The active serial connection object.
    """
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
    """
    Sends a string command over serial, character by character, to avoid buffer overflow.

    Args:
        ser (serial.Serial): The active serial connection.
        text (str): The command string to send (e.g., 'Forward Half').
    """
    EOL = "\r" 
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
    """
    Manages high-level motor commands to the TM4C. 
    Prevents serial spam by suppressing exact duplicates and enforcing a minimum send interval.

    Attributes:
        ser (serial.Serial): The serial connection object.
        min_cmd_interval (float): Minimum seconds between identical commands.
    """
    def __init__(self,ser,min_cmd_interval=0.08):
        self.ser = ser              #serial connection to TM4C
        self.last_cmd = None        #remembers the last command sent
        self.last_send_time = 0.0   # [SAteefa 3/21 added: remembers when the last command was sent]
        self.min_cmd_interval = min_cmd_interval # [SAteefa 3/21 added: minimum delay between non-forced command sends]
    def send(self, cmd:str, force:bool=False):
        """
        Evaluates and sends a command to the motors.

        Args:
            cmd (str): The movement command string.
            force (bool): If True, bypasses timing restrictions and sends immediately.
        """
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
    """Normalizes an angle to be within -180 and +180 degrees."""
    return (a + 180) % 360 - 180

def rel_to_front(a_abs, angle_sign, front_deg):
    """Calculates the relative angle of a LIDAR point based on the physical mount orientation."""
    return angle_sign * wrap180(a_abs - front_deg)

def fmt_mm(x):
    """Formats a millimeter distance for console printing, hiding infinite values."""
    PRINT_INF_AS = -1
    
    return int(x) if x < math.inf else PRINT_INF_AS

def open_lidar(port, baud, timeout):
    """
    Connects to the RPLidar sensor and clears its initial buffers.

    Args:
        port (str): The USB port (e.g., '/dev/ttyUSB0').
        baud (int): Baudrate for the lidar connection.
        timeout (float): Connection timeout in seconds.

    Returns:
        RPLidar: The initialized Lidar object.
    """
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
    """Generator that yields standard scans from the RPLidar."""
    for scan in lidar.iter_scans(max_buf_meas=8192, min_len=40):
        yield scan

def bins_from_scan(scan, angle_sign, front_deg):
    """Groups lidar points into direct angles (-50, 0, 50 degrees)."""
    LOOK_ANGLES  = [-50, 0, +50]
    ANG_TOL      = 10
    
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
    """Groups lidar points into wide frontal sectors for object avoidance."""
    FORWARD_SECTORS = {
    'FR': (-60.0, -10.0),
    'F' : (-15.0, +15.0),
    'FL': (+10.0, +60.0),
    }
    
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
    """Returns the minimum value from a list, ignoring mathematical infinity."""
    finite = [v for v in vals if not math.isinf(v)]
    return min(finite) if finite else math.inf

class LidarState:
    """Maintains the current spatial state of the vehicle's surroundings using Lidar data."""
    def __init__(self, angle_sign, front_deg, no_data_stop_sec):
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
    

class MjpegPreviewState:
    def __init__(self, lock):
        self.lock = lock  # This will be a Manager.Lock()
        self.jpeg_bytes = None

    def set_jpeg(self, data: bytes):
        with self.lock:
            self.jpeg_bytes = data

    def get_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.jpeg_bytes

def make_mjpeg_handler(preview_bundle):
    preview_state, preview_lock = preview_bundle
    boundary = b"--jpgboundary"
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/view"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><img src='/stream' style='width:100%'></body></html>")
                return
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=jpgboundary")
                self.end_headers()
                while True:
                    with preview_lock:
                        jpg = preview_state.jpeg_bytes

                    if jpg is None:
                        time.sleep(0.1)
                        continue
                    try:
                        self.wfile.write(boundary + b"\r\nContent-Type: image/jpeg\r\n" +
                                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                                    jpg + b"\r\n")
                        self.wfile.flush()
                    except (ConnectionResetError, BrokenPipeError):
                        break
                    time.sleep(0.05) # Cap at ~20 FPS to save CPU
        def log_message(self, format, *log_args): pass
    return Handler

def start_mjpeg_server(preview, host="0.0.0.0", port=8765):
    server = socketserver.ThreadingTCPServer((host, port), make_mjpeg_handler(preview))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[HTTP] Preview live at http://localhost:{port}")
    
def kill_port(port=8765):
    """Kill any process still holding the preview server port from a previous run."""
    try:
        result = subprocess.check_output(
            f"fuser {port}/tcp 2>/dev/null", shell=True
        ).decode().strip()
        if result:
            for pid in result.split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"[BOOT] Killed stale process {pid} holding port {port}")
                except Exception:
                    pass
            time.sleep(0.5)  # give OS a moment to release the port
    except subprocess.CalledProcessError:
        pass  # fuser returns non-zero if nothing is holding the port — that's fine
# =========================
# Lane / camera helpers
# =========================

# Add roi_dict as the second parameter
def region_selection(image, roi_dict):
    mask = np.zeros_like(image)
    h, w = image.shape[:2]
    
    # Use the passed dictionary instead of the hardcoded global
    bl = (int(w*roi_dict['bottom_left'][0]),  int(h*roi_dict['bottom_left'][1]))
    tl = (int(w*roi_dict['top_left'][0]),     int(h*roi_dict['top_left'][1]))
    tr = (int(w*roi_dict['top_right'][0]),    int(h*roi_dict['top_right'][1]))
    br = (int(w*roi_dict['bottom_right'][0]), int(h*roi_dict['bottom_right'][1]))
    
    pts = np.array([[bl, tl, tr, br]], dtype=np.int32)
    color = 255 if len(image.shape) == 2 else (255,) * image.shape[2]
    cv2.fillPoly(mask, pts, color)
    return cv2.bitwise_and(image, mask)

def white_mask(bgr):
    HSV_S_MAX = 70
    HSV_V_MIN = 200
    HSV_H_ANY = (0, 180)
    LAB_A_ABS_MAX = 20
    LAB_B_ABS_MAX = 20
    Y_MIN = 210
    CR_ABS_MAX = 14
    CB_ABS_MAX = 14
    USE_OTSU_BACKUP = False
    OTSU_RATIO = 0.60

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
    OPEN_K  = (3, 3)
    CLOSE_K = (11, 11)

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
    """
    A Proportional-Derivative (PD) controller that calculates steering error 
    based on a lane line's position on the screen.
    """
    def __init__(self, w, target_x_ratio: float):
        self.w = w
        self.target_x_ratio = target_x_ratio
        self.prev_err = 0.0

    def control_from_x(self, line_x_bottom):
        """
        Calculates the steering output needed to center the line.

        Args:
            line_x_bottom (float): The current X coordinate of the tracked line.

        Returns:
            tuple: (Steering output command, Normalized Error)
        """
        # PD gains
        KP = 0.9
        KD = 0.2
        target_x = self.target_x_ratio * self.w

        if line_x_bottom is None:
            err_norm = 0.0
        else:
            err_px = line_x_bottom - target_x
            err_norm = float(np.clip(err_px / (self.w/2), -1.0, 1.0))

        d = err_norm - self.prev_err
        self.prev_err = err_norm
        u = KP * err_norm + KD * d
        u = float(np.clip(u, -1.0, 1.0))
        return u, err_norm

def apply_steering(norm_error):
    turn_scale = 1.0
    turn = float(np.clip(norm_error, -1.0, 1.0)) * turn_scale
    forward = 0.6 * (1 - 0.5*abs(turn))
    print(f"[CTRL] forward={forward:.2f} turn={turn:.2f}")
    return forward, turn

# [SAteefa 3/22 Added: hysteresis-based discrete steering decision]
# Keeps the controller from flipping between LEFT and FORWARD on tiny frame-to-frame noise.
def choose_cmd_with_hysteresis(turn, steer_state):
    """
    Translates a fractional turn value into a discrete TM4C hardware command.
    Applies hysteresis thresholds to prevent the robot from wiggling on straightaways.

    Args:
        turn (float): Normalized turn desire (-1.0 to 1.0).
        steer_state (str): The current driving state ("LEFT", "RIGHT", "STRAIGHT").

    Returns:
        tuple: (The specific string command, The new state)
    """
    LEFT_ENTER_THRESH  = -0.05
    LEFT_EXIT_THRESH   = -0.02

    RIGHT_ENTER_THRESH =  0.05
    RIGHT_EXIT_THRESH  =  0.02

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

def process_frame(frame_bgr, tracker: SimpleLineTracker, roi_dict):
    LANE_COLOR   = (255, 255, 255)
    TARGET_COLOR = (0, 255, 0)
    MIN_AREA             = 80 #old: 80
    MIN_HEIGHT           = 15 #old: 15
    MIN_WIDTH            = 2
    MIN_ASPECT_H_OVER_W  = 1 #old: 0.6,1.0


    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top    = int(h * roi_dict['top_left'][1]) # Use roi_dict here

    roi = region_selection(frame_bgr, roi_dict)
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

        # --- OLD NEAR-SIGHTED MATH ---
        # ys = box[:,1]
        # low = box[np.argsort(ys)][-2:]
        # line_x_bottom = float(np.mean(low[:,0]))
        # cv2.circle(out, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)
        # cv2.circle(mask_debug, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)

        # --- NEW LOOK-AHEAD MATH ---
        # best_rect format is: ((cx, cy), (width, height), angle)
        (cx, cy) = best_rect[0]
        
        # We still call it line_x_bottom for variable compatibility, 
        # but it is now tracking the center of the line!
        line_x_bottom = float(cx) 
        
        # Draw the red dot in the center of the bounding box so you can see it working!
        cv2.circle(out, (int(cx), int(cy)), 5, (0,0,255), -1)
        cv2.circle(mask_debug, (int(cx), int(cy)), 5, (0,0,255), -1)
        
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480) #previous 960
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270) #previous 540
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] {label} tuned to ~{w:.0f}x{h:.0f} @ {fps:.1f} FPS")


def initialize_system():
    # [SAteefa 3/21 Added: modular startup helper so all debug phases share the same safe initialization]
    serial_lock = SerialPortLock(SERIAL_LOCK_PATH)
    serial_lock.acquire()
    atexit.register(serial_lock.release)

    ser = open_serial()
    mc = MotorCommander(ser)

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

# =========================
# YOLO Helpers
# =========================

def detect_objects(frame,detector:YOLO):
    """
    Runs YOLOv8 inference on a given video frame to find specific objects.

    Args:
        frame (np.ndarray): The OpenCV BGR image matrix.
        detector (YOLO): The instantiated Ultralytics YOLO model.

    Returns:
        list: A list of dictionaries containing detected labels, bounding boxes, and confidence scores.
    """
    detections = []
    results = detector(frame, verbose=False)

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
            elif label == 'person':
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "bbox": (x1,y1,x2,y2),
                    "confidence": float(box.conf[0])
                })

    return detections


#Isaac Elizarraraz [3/30/2026]
#helper function for object detection handling
#measures the distance from an object by the size of its bounding box
#the measurement may need to be calibrated based on the cameras focal length
def measure_distance(bbox,focal_length,real_height):
    """
    Estimates distance to an object using the pinhole camera geometry model.

    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        focal_length (float): Calibrated camera focal length.
        real_height (float): Known physical height of the object in meters.

    Returns:
        float: Estimated distance in meters, or None if calculation fails.
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

# =========================
# Multiprocessing Workers
# =========================

def camera_worker(shared, shm_name_L, shm_name_R):
    """
    Background process that continuously captures frames from two USB cameras 
    and writes them into high-speed shared memory for the lane and YOLO workers.

    This function initializes the physical camera hardware via V4L2, tunes the 
    capture resolution and framerate to prevent bottlenecking, and maintains 
    a tight loop to pull the latest frames.

    Args:
        shared (dict): Shared multiprocessing variables dictionary.
        shm_name_L (str): The shared memory reference name for the left camera buffer.
        shm_name_R (str): The shared memory reference name for the right camera buffer.
    """
    cap_L = open_camera_by_index("LEFT")
    cap_R = open_camera_by_index("RIGHT")
    time.sleep(1)
    if cap_L is None or cap_R is None: 
        print("[CAM] failed to start")
        return
    tune_camera_for_speed(cap_L, "LEFT")
    tune_camera_for_speed(cap_R, "RIGHT")
    shm_L = shared_memory.SharedMemory(name=shm_name_L)
    shm_R = shared_memory.SharedMemory(name=shm_name_R)
    
    buf_L = np.ndarray((270, 480, 3), dtype=np.uint8, buffer=shm_L.buf)
    buf_R = np.ndarray((270, 480, 3), dtype=np.uint8, buffer=shm_R.buf)
    
    print("[CAMERA PROCESS] started")

    while True:
            
        retL, imgL = cap_L.read()
        retR, imgR = cap_R.read()

        if retL:
            buf_L[:] = cv2.resize(imgL, (480, 270), interpolation=cv2.INTER_NEAREST)
        if retR:
            buf_R[:] = cv2.resize(imgR, (480, 270), interpolation=cv2.INTER_NEAREST)
        time.sleep(0.01)
        
def lane_worker(shared, shm_name_L, shm_name_R, preview_bundle):
    """
    Background process responsible for executing the computer vision pipeline 
    and Proportional-Derivative (PD) steering control.

    Reads high-speed camera frames from shared memory, applies region-of-interest (ROI) 
    masking to prevent field-of-view cross-talk, and calculates the required steering 
    angle to keep the vehicle centered. Features a dual-camera primary mode and a 
    robust single-camera fallback mode to navigate sharp curves. 

    Args:
        shared (dict): Shared multiprocessing dictionary to output the calculated 
                       steering commands ('lane_turn', 'lane_visible_L', 'lane_visible_R').
        shm_name_L (str): Shared memory reference name for the left camera buffer.
        shm_name_R (str): Shared memory reference name for the right camera buffer.
        preview_bundle (tuple): A tuple containing (preview_state, preview_lock) used 
                                to safely push debug JPG frames to the MJPEG HTTP server.
    """
    CENTER_DEADBAND = 0.1
    DUAL_KP = 1.0 #0.6 #0.9 
    DUAL_KD = 0.4
    TARGET_X_RATIO_LEFT = 0.35 # old 0.43
    TARGET_X_RATIO_RIGHT = 0.65 # old 0.65
    # The Left camera is not allowed to look past 60% of the screen (Right edge blocked)
    ROI_LEFT_CAM = dict(
        bottom_left  = (0.00, 0.95),
        top_left     = (0.00, 0.35),
        top_right    = (0.60, 0.35), 
        bottom_right = (0.60, 0.95),
    )

    # The Right camera is not allowed to look past 40% of the screen (Left edge blocked)
    ROI_RIGHT_CAM = dict(
        bottom_left  = (0.40, 0.95), 
        top_left     = (0.40, 0.35),
        top_right    = (1.00, 0.35),
        bottom_right = (1.00, 0.95),
    )

    preview_state, preview_lock = preview_bundle
    shm_L = shared_memory.SharedMemory(name=shm_name_L)
    shm_R = shared_memory.SharedMemory(name=shm_name_R)

    frame_L = np.ndarray((270, 480, 3), dtype=np.uint8, buffer=shm_L.buf)
    frame_R = np.ndarray((270, 480, 3), dtype=np.uint8, buffer=shm_R.buf)

    tracker_L = SimpleLineTracker(480, TARGET_X_RATIO_LEFT)
    tracker_R = SimpleLineTracker(480, TARGET_X_RATIO_RIGHT)

    prev_e  = 0.0
   
    print("[LANE PROCESS] started")
    while True:
        img_L = frame_L.copy()
        img_R = frame_R.copy()

        try:
            out_L, mask_L, _, _, _, line_x_L, _ = process_frame(img_L, tracker_L, ROI_LEFT_CAM)
            out_R, mask_R, _, _, _, line_x_R, _ = process_frame(img_R, tracker_R, ROI_RIGHT_CAM
                                                                )
            nL = line_x_L / 480.0 if line_x_L is not None else None
            nR = line_x_R / 480.0 if line_x_R is not None else None

            # ------------------------------------------------
            # BOTH cameras: standard dual PD control
            # ------------------------------------------------
            if nL is not None and nR is not None:

                current_center = (nL + nR) / 2.0
                
                # FIX: Swapped so steering matches the error polarity!
                error = current_center - 0.5 
                
                de    = error - prev_e
                prev_e = error

                turn_val = DUAL_KP * error + DUAL_KD * de

                turn_val  = float(np.clip(turn_val, -1.0, 1.0))

                if abs(error) < CENTER_DEADBAND:
                    turn_val = 0.0
                shared["lane_turn"].value      = turn_val 
                shared["lane_visible_L"].value = True
                shared["lane_visible_R"].value = True

            # ------------------------------------------------
            # ONE camera: direct error → turn, no hold/blend
            # ------------------------------------------------
            elif nL is not None or nR is not None:

                if nL is not None:
                    err   = nL - TARGET_X_RATIO_LEFT
                    
                    # FIX: Removed the negative sign
                    turn_val = float(np.clip(err * DUAL_KP, -0.8, 0.8)) 
                    if abs(err) < CENTER_DEADBAND:
                        turn_val = 0.0
                    shared["lane_turn"].value = turn_val
                    shared["lane_visible_L"].value = True
                    shared["lane_visible_R"].value = False

                else:
                    err   = nR - TARGET_X_RATIO_RIGHT
                    
                    # FIX: Removed the negative sign
                    turn_val = float(np.clip(err * DUAL_KP, -0.8, 0.8)) 
                    if abs(err) < CENTER_DEADBAND:
                        turn_val = 0.0
                    shared["lane_turn"].value = turn_val
                    shared["lane_visible_L"].value = False
                    shared["lane_visible_R"].value = True

            # Debug overlay
            debug_view = np.hstack((out_L, out_R, mask_L, mask_R))
            cv2.putText(debug_view, f"Turn: {shared['lane_turn'].value:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cam_status = f"L:{'OK' if nL is not None else '--'}  R:{'OK' if nR is not None else '--'}"
            cv2.putText(debug_view, cam_status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            ok, jpg = cv2.imencode(".jpg", debug_view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with preview_lock:
                    preview_state.jpeg_bytes = jpg.tobytes()

        except Exception as e:
            print("[LANE ERROR]", e)

        time.sleep(0.03)
#Isaac Elizarraraz [3/31/2026]
#thread worker that finds objects in current frame
def yolo_worker(shared,shm_name):
    """
    Background process that continuously reads shared memory frames and runs 
    the YOLOv8 object detection model to trigger stop or pedestrian events.

    Args:
        shared (dict): Shared multiprocessing variables dictionary.
        shm_name (str): The shared memory reference name for the camera frame.
    """

    detector = YOLO("./yolov8n.pt")
    detector.to("cpu") # ideally this should be gpu!!
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((270,480,3), dtype=np.uint8, buffer=shm.buf)
    
    #constants
    real_height = 0.3048
    focal_length = 650
    
    #stop event variables
    stop_detected = False
    stop_latched = False
    stop_frame_counter = 0
    
    #pedestrian event variables
    person_detected = False
    person_latch = False
    person_frame_counter = 0

    
    print("[YOLO PROCESS] started")
    while True:
        stop_detected = False
        person_detected = False
        img = frame.copy()  # optional safety
        #cv2.imwrite("lane_worker.jpg", img)

        if img is None:
            time.sleep(0.02)
            continue

        try:

            detections = detect_objects(img, detector)
            min_stop_distance = float("inf") #keep track of how far away the stop sign is. car stops 2 meters away
            min_person_distance = float("inf")
            
            if detections:
                shared["det_time"].value = time.time()

            for d in detections:  
                distance = measure_distance(
                            d["bbox"],
                            focal_length,
                            real_height
                        )
                if d["label"] == "stop sign":
                    stop_detected = True
                    stop_frame_counter += 1
                    min_stop_distance = min(min_stop_distance,distance)
                    
                elif d["label"] == "person":
                    person_detected = True
                    person_frame_counter += 1
                    min_person_distance = min(min_person_distance,distance)
                    
            #stop event logic
            if not stop_detected:
                stop_frame_counter = 0
                
            if stop_frame_counter >= 3 and min_stop_distance <= 5:
                if not stop_latched:
                    shared["stop_event"].value = True
                    stop_latched = True
            else:
                shared["stop_event"].value = False
                if stop_frame_counter == 0:
                    stop_latched = False
                    
            #pedestrian event logic
            if not person_detected:
                person_frame_counter = 0
                
            if person_frame_counter >= 2 and min_person_distance <= 4:
                if not person_latch:
                    shared["pedestrian_event"].value = True
                    person_latch = True
            else:
                shared["pedestrian_event"].value = False
                if person_frame_counter == 0:
                    person_latch = False
                    
            time.sleep(0.02)
            
        except Exception as e:
            print("[YOLO ERROR]", e)
            
def lidar_worker(shared):
    """
    Background process that maintains a continuous serial connection to the RPLidar.

    This worker reads raw point cloud data from the spinning lidar, groups the 
    points into distinct front-facing sectors (Left, Center, Right), and updates 
    the shared memory dictionary. It includes robust error handling to automatically 
    power-cycle and reconnect to the lidar if the data stream drops.

    Args:
        shared (dict): Shared multiprocessing dictionary where the processed 
                       distance values ('dL', 'dC', 'dR', 'dFL', 'dF', 'dFR') are stored.
    """
    FRONT_DEG    = 0
    ANGLE_SIGN_D = -1    # -1 if your LIDAR mount is mirrored
    NO_DATA_STOP_SEC   = 0.5   # stop if no valid LIDAR data for this time

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
    """
    The main control loop of the robot. 
    Reads telemetry from the Lidar, Camera, and YOLO workers, performs 
    Vector Blending for obstacle avoidance, and dispatches serial commands to the TM4C.

    Args:
        mc (MotorCommander): The initialized MotorCommander object.
        shared (dict): Shared multiprocessing variables dictionary.
    """

    last_completed_stop = time.time()
    stop_state_end_time = 0
    steer_state = "STRAIGHT"  # Hysteresis state: "STRAIGHT", "LEFT", or "RIGHT"

    # Tuned constants (moved here so they're easy to adjust)
    AVOID_ZONE   = 1200.0  # mm — was 800, reduced to avoid fighting the lane controller
    MAX_REPULSION = 0.8   # was 0.8

    try:
        while True:

            now = time.time()

            # =========================
            # Read shared sensor data
            # =========================
            dL  = shared["dL"].value
            dC  = shared["dC"].value
            dR  = shared["dR"].value
            dFL = shared["dFL"].value
            dF  = shared["dF"].value
            dFR = shared["dFR"].value

            lane_visible = shared["lane_visible_L"].value or shared["lane_visible_R"].value

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
                time.sleep(0.05)
                continue

            if stop_state_end_time != 0 and now >= stop_state_end_time:
                last_completed_stop = now
                stop_state_end_time = 0
                print("[STOP] Completed stop")

            if stop_event and (now - last_completed_stop > 10):
                print("[STOP] Triggered")
                stop_state_end_time = now + 8
                shared["stop_event"].value = False
                mc.send(STOP_CMD, force=True)
                time.sleep(0.05)
                continue

            # =========================
            # 1. Lane turn value from camera
            # =========================
            lane_turn = shared["lane_turn"].value

            # =========================
            # 2. LIDAR obstacle repulsion
            # =========================
            left_dist  = min(dL, dFL)
            right_dist = min(dR, dFR)
            center_dist = min(dC, dF)

            obs_turn = 0.0

            if left_dist < AVOID_ZONE:
                push = (AVOID_ZONE - left_dist) / AVOID_ZONE
                obs_turn += push * MAX_REPULSION   # obstacle left → push right

            if right_dist < AVOID_ZONE:
                push = (AVOID_ZONE - right_dist) / AVOID_ZONE
                obs_turn -= push * MAX_REPULSION   # obstacle right → push left

            if center_dist < AVOID_ZONE:
                push = (AVOID_ZONE - center_dist) / AVOID_ZONE
                if left_dist > right_dist:
                    obs_turn -= push * MAX_REPULSION  # more room on left → swerve left
                else:
                    obs_turn += push * MAX_REPULSION  # more room on right → swerve right

            # =========================
            # 3. Blend lane + obstacle
            # =========================
            final_turn = float(np.clip(lane_turn + obs_turn, -1.0, 1.0))

            # =========================
            # 4. Translate float → discrete TM4C command via hysteresis
            # =========================
            cmd, steer_state = choose_cmd_with_hysteresis(final_turn, steer_state)

 

            mc.send(cmd)

            # =========================
            # Telemetry
            # =========================
            print(
                f"Lane: {lane_turn:+.2f} | Obs: {obs_turn:+.2f} | "
                f"Final: {final_turn:+.2f} | Steer: {steer_state} | CMD: {cmd}"
            )

            time.sleep(0.05)   # 20 Hz — was 0.005 (200 Hz), which caused command spam

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

    frame_shape = (270,480,3)
    frame_size = np.prod(frame_shape)

    shared = {}
    shm = shared_memory.SharedMemory(create=True, size=frame_size) #shared frame memory
    shm_L = shared_memory.SharedMemory(create=True, size=frame_size, name="shm_left")
    shm_R = shared_memory.SharedMemory(create=True, size=frame_size, name="shm_right")
    
    # LIDAR
    shared["dL"] = Value('d', float('inf'))
    shared["dC"] = Value('d', float('inf'))
    shared["dR"] = Value('d', float('inf'))

    shared["dFL"] = Value('d', float('inf'))
    shared["dF"]  = Value('d', float('inf'))
    shared["dFR"] = Value('d', float('inf'))
    
    # camera
    shared["active_camera"] = Value('i', 0) # 0 = LEFT, 1 = RIGHT
    shared["lane_turn"] = Value('d', 0.0)
    shared["lane_visible_L"] = Value('b', False)
    shared["lane_visible_R"] = Value('b', False)
    shared["line_x_L"] = Value('d', 0.0)
    shared["line_x_R"] = Value('d', 0.0)
    
    # YOLO detection
    shared["stop_detected"] = Value('b', False)
    shared["stop_event"] = Value('b', False)
    shared["pedestrian_event"] = Value('b', False)
    shared["det_time"] = Value('d', 0.0)
    
    manager = Manager()
    
    # 2. Initialize the preview state with a Manager Lock
    # This proxy object CAN be pickled and passed to processes
    preview_lock = manager.Lock()
    preview_state = manager.Namespace()
    preview_state.jpeg_bytes = None
    
    preview_lock = manager.Lock()
    
    # 3. Start the HTTP server (still runs in the main process thread)
    start_mjpeg_server((preview_state, preview_lock), host="0.0.0.0", port=8765)

    camera_proc = Process(
        target=camera_worker,
        args=(shared,shm_L.name, shm_R.name),
        daemon=True
    )
    
    lidar_proc = Process(
        target=lidar_worker,
        args=(shared,),
        daemon=True
    )

    yolo_proc = Process(
        target=yolo_worker,
        args=(shared,shm_L.name),
        daemon=True
    )

    lane_proc = Process(
        target=lane_worker,
        args=(shared,shm_L.name, shm_R.name, (preview_state, preview_lock)),
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
        try:
            if mc is not None:
                # Use force=True to bypass the 0.08s interval check
                mc.send(STOP_CMD, force=True) 
        except Exception as e:
            print(f"[WARN] Failed to send stop: {e}")
            
        print("[MAIN] Shutting down workers...")
        for p in [lidar_proc, yolo_proc, lane_proc, camera_proc]:
            if p.is_alive():
                p.terminate()

        for p in [lidar_proc, yolo_proc, lane_proc, camera_proc]:
            p.join()
                
        for s in [shm,shm_L, shm_R]:
            try:
                s.close()
                s.unlink()
            except Exception:
                pass
        kill_port(8765)
        cleanup_system(mc=mc, ser=ser, serial_lock=serial_lock)
        

  

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_integrated()
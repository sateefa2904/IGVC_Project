import os
import time
import numpy as np
import cv2
import serial
import threading  # for timer-like behavior

# =========================
# Serial / driving config
# =========================

# On Jetson, your TM4C will typically show up as /dev/ttyACM0 or /dev/ttyUSB0
PORT = "/dev/ttyACM0"   # <-- CHANGE ME if needed
BAUD = 19200
EOL = "\r\n"       # CRLF like PuTTY/miniterm
DTR = True
RTS = True
CHAR_DELAY = 0.01  # 10 ms between bytes (typewriter style)
BOOT_WAIT = 2.0    # seconds after opening port before first send

CMD_UP     = "Forward Half"
CMD_DOWN   = "Backward Half"
CMD_LEFT   = "Left Half"
CMD_RIGHT  = "Right Half"
CMD_AUTO   = "Auto"
CMD_MANUAL = "Manual"
STOP_CMD   = "Stop"     # TM4C parser should handle this

# Turn thresholds
TURN_LEFT_THRESH  = -0.20
TURN_RIGHT_THRESH =  0.20

# =========================
# Camera / lane tunables
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
TOP_K_COMPONENTS     = 8

# Per-camera targets
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

# =========================
# Display / debug toggles
# =========================

# Set this to False when running headless on Jetson (no monitor / no GUI)
USE_DISPLAY       = False
SHOW_STACKED_DEBUG = False   # True = heavy 4-panel debug; False = just overlay

# =========================
# Camera indices / backend
# =========================

# On Ubuntu/Jetson, cameras are /dev/video0, /dev/video1, ...
# These indices map to those devices.
LEFT_CAM_INDEX  = 2   # <-- adjust after checking which cam is which
RIGHT_CAM_INDEX = 0   # <-- adjust as needed

# Use V4L2 backend on Linux, or CAP_ANY if V4L2 causes trouble
CAM_BACKEND = cv2.CAP_V4L2   # or cv2.CAP_ANY

# =========================
# Lane-loss / timing tunables
# =========================

LOST_FRAMES_THRESH = 5          # frames with no lane before we say "lane ended"

# Transition timing (your chosen values)
TURN_TIME_RIGHT      = 2.2      # seconds turning RIGHT after left lane ends
GO_TIME_AFTER_RIGHT  = 4      # seconds going FORWARD after that right turn
TURN_TIME_LEFT       = 2.15      # seconds turning LEFT after right lane ends
GO_TIME_AFTER_LEFT   = 5      # seconds going FORWARD after that left turn
CAMERA_SETTLE_TIME   = 6      # seconds waiting after opening a camera

# =========================
# Serial helpers
# =========================

def open_serial():
    print("Opening serial:", PORT, "@", BAUD)
    ser = serial.Serial(PORT, BAUD,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=0.5, write_timeout=1.0,
                        rtscts=False, dsrdtr=False, xonxoff=False)
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

def schedule_forward_after_delay(delay_s: float, send_cmd):
    """
    Soft 'timer ISR' in Python:
      - After delay_s seconds, call send_cmd(CMD_UP) once.
    Runs in a background thread so main loop can keep doing work.
    """
    def worker():
        time.sleep(delay_s)
        print(f"[TIMER] {delay_s:.2f}s elapsed -> sending Forward Half")
        send_cmd(CMD_UP)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def schedule_stop_after_delay(delay_s: float, send_cmd):
    """
    After delay_s seconds, call send_cmd(STOP_CMD) once.
    """
    def worker():
        time.sleep(delay_s)
        print(f"[TIMER] {delay_s:.2f}s elapsed -> sending STOP")
        send_cmd(STOP_CMD)
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# =========================
# Camera / debug helpers
# =========================

def stack_debug_views(raw, mask, overlay, shown_turn, shown_angle, tl_text=""):
    if len(mask.shape) == 2:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bgr = mask

    h = 300
    raw_r     = cv2.resize(raw,     (int(raw.shape[1] * h/raw.shape[0]), h))
    mask_r    = cv2.resize(mask_bgr,(raw_r.shape[1], h))
    overlay_r = cv2.resize(overlay, (raw_r.shape[1], h))

    steer_panel = np.zeros_like(overlay_r)
    cv2.putText(steer_panel, f"Turn:  {shown_turn:+.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 3)
    cv2.putText(steer_panel, f"Angle: {shown_angle:5.1f} deg", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 3)
    cv2.putText(steer_panel, "(Raw | Mask+Track | Overlay)", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    if tl_text:
        cv2.putText(raw_r, tl_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)

    top = np.hstack((raw_r, mask_r))
    bottom = np.hstack((overlay_r, steer_panel))
    return np.vstack((top, bottom))

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
    ang_deg = np.degrees(np.arctan2(vx, vy))  # 0 = vertical, 90 = horizontal
    return float(ang_deg)

def score_component(rect):
    (cx, cy), (w, h), _ = rect
    area = max(1.0, w*h)
    height_bonus = max(w, h)**2
    bottom_bias = cy
    return area + 0.3*height_bonus + 0.8*bottom_bias

# =========================
# Control helpers
# =========================

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

# =========================
# Frame processing (single camera)
# =========================

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

    cv2.putText(out, f"LineX: {('%.1f'%line_x_bottom) if line_x_bottom is not None else 'None'}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Err(norm): {err_norm:+.2f}  Turn: {turn:+.2f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Angle(|vert|): {('%.1f'%angle_deg if angle_deg is not None else 'NA')} deg",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    print(f"[TELEM] LineX={line_x_bottom}  ErrNorm={err_norm:+.2f}  AngleAbsDeg={angle_deg}")

    return out, mask_debug, forward, turn, (angle_deg if angle_deg is not None else 0.0), line_x_bottom

# =========================
# Camera helpers
# =========================

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
        print(f"[CAM] {label} camera at index {index} opened but could not read frame")
        cap.release()
        return None
    print(f"[CAM] Using {label} camera at index {index}")
    return cap

def tune_camera_for_speed(cap, label: str):
    # USB cams on Jetson usually like MJPG at 640x480 or 960x540
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[CAM] {label} tuned to ~{w:.0f}x{h:.0f} @ {fps:.1f} FPS")

def open_side_camera(side: str):
    """Open LEFT or RIGHT camera and create its tracker."""
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
# Main: LEFT → RIGHT → LEFT with threaded transitions
# =========================

def run_hardcoded_left_right_left():
    ser = open_serial()
    last_cmd = None

    def send_cmd(cmd: str):
        nonlocal last_cmd
        if cmd != last_cmd:
            send_line_typewriter(ser, cmd)
            last_cmd = cmd

    send_cmd(STOP_CMD)
    time.sleep(0.2)
    send_cmd(CMD_AUTO)

    if USE_DISPLAY:
        cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)

    prev = time.time()

    try:
        # ===== PHASE 1: LEFT CAMERA =====
        active_side = "LEFT"
        cap, tracker = open_side_camera(active_side)
        if cap is None:
            print("[STATE] Could not open LEFT camera, exiting.")
            return

        print("[STATE] >>> PHASE 1: Following LEFT lane")
        lost_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] LEFT: frame grab failed, ending LEFT phase.")
                break

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)

            out, mask_debug, forward, turn, angle_deg, line_x_bottom = process_frame(frame, tracker)

            if line_x_bottom is None:
                lost_frames += 1
                cmd = STOP_CMD
            else:
                lost_frames = 0
                if turn > TURN_RIGHT_THRESH:
                    cmd = CMD_RIGHT
                elif turn < TURN_LEFT_THRESH:
                    cmd = CMD_LEFT
                else:
                    cmd = CMD_UP

            send_cmd(cmd)

            if lost_frames >= LOST_FRAMES_THRESH:
                print(f"[STATE] LEFT lane lost for {lost_frames} frames -> ending LEFT phase.")
                send_cmd(STOP_CMD)
                break

            # FPS + HUD
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now

            if USE_DISPLAY:
                tl_text = f"LEFT cam | cmd={cmd}"
                if SHOW_STACKED_DEBUG:
                    debug_view = stack_debug_views(frame, mask_debug, out, turn, angle_deg, tl_text)
                else:
                    debug_view = out.copy()
                    cv2.putText(debug_view, tl_text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)

                cv2.putText(debug_view, f"FPS: {fps:.1f}", (10, debug_view.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Lane Debug", debug_view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[STATE] Quit requested during LEFT phase.")
                    return
            else:
                # Small sleep to avoid maxing CPU if no display
                time.sleep(0.001)

        # Close LEFT camera after Phase 1
        if cap is not None:
            cap.release()
            cap = None

        # ===== TRANSITION 1: LEFT -> RIGHT (threaded turn/forward/stop, open RIGHT early) =====
        print("[STATE] >>> Transition 1: START (LEFT -> RIGHT)")
        # Start turning RIGHT immediately
        send_cmd(CMD_RIGHT)
        # Schedule Forward, then Stop in background threads
        schedule_forward_after_delay(TURN_TIME_RIGHT, send_cmd)
        schedule_stop_after_delay(TURN_TIME_RIGHT + GO_TIME_AFTER_RIGHT, send_cmd)

        print("[STATE] >>> Transition 1: opening RIGHT camera (no lane processing yet)")
        cap, tracker = open_side_camera("RIGHT")
        if cap is None:
            print("[STATE] Could not open RIGHT camera, exiting.")
            return

        # Optional settle (you set this to 0.0 => no wait)
        time.sleep(CAMERA_SETTLE_TIME)
        print("[STATE] >>> Transition 1 complete, entering RIGHT lane follow")

        # ===== PHASE 2: RIGHT CAMERA =====
        print("[STATE] >>> PHASE 2: Following RIGHT lane")
        lost_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] RIGHT: frame grab failed, ending RIGHT phase.")
                break

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)

            out, mask_debug, forward, turn, angle_deg, line_x_bottom = process_frame(frame, tracker)

            if line_x_bottom is None:
                lost_frames += 1
                cmd = STOP_CMD
            else:
                lost_frames = 0
                if turn > TURN_RIGHT_THRESH:
                    cmd = CMD_RIGHT
                elif turn < TURN_LEFT_THRESH:
                    cmd = CMD_LEFT
                else:
                    cmd = CMD_UP

            send_cmd(cmd)

            if lost_frames >= LOST_FRAMES_THRESH:
                print(f"[STATE] RIGHT lane lost for {lost_frames} frames -> ending RIGHT phase.")
                send_cmd(STOP_CMD)
                break

            # FPS + HUD
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now

            if USE_DISPLAY:
                tl_text = f"RIGHT cam | cmd={cmd}"
                if SHOW_STACKED_DEBUG:
                    debug_view = stack_debug_views(frame, mask_debug, out, turn, angle_deg, tl_text)
                else:
                    debug_view = out.copy()
                    cv2.putText(debug_view, tl_text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)

                cv2.putText(debug_view, f"FPS: {fps:.1f}", (10, debug_view.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Lane Debug", debug_view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[STATE] Quit requested during RIGHT phase.")
                    return
            else:
                time.sleep(0.001)

        # Close RIGHT camera after Phase 2
        if cap is not None:
            cap.release()
            cap = None

        # ===== TRANSITION 2: RIGHT -> LEFT (threaded turn/forward/stop, open LEFT early) =====
        print("[STATE] >>> Transition 2: START (RIGHT -> LEFT)")
        send_cmd(CMD_LEFT)
        schedule_forward_after_delay(TURN_TIME_LEFT, send_cmd)
        schedule_stop_after_delay(TURN_TIME_LEFT + GO_TIME_AFTER_LEFT, send_cmd)

        print("[STATE] >>> Transition 2: opening LEFT camera (no lane processing yet)")
        cap, tracker = open_side_camera("LEFT")
        if cap is None:
            print("[STATE] Could not reopen LEFT camera, exiting.")
            return

        time.sleep(CAMERA_SETTLE_TIME)
        print("[STATE] >>> Transition 2 complete, entering final LEFT lane follow")

        # ===== PHASE 3: LEFT CAMERA (FINAL) =====
        print("[STATE] >>> PHASE 3: Following LEFT lane (final)")
        lost_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] LEFT: frame grab failed, ending final LEFT phase.")
                break

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)

            out, mask_debug, forward, turn, angle_deg, line_x_bottom = process_frame(frame, tracker)

            if line_x_bottom is None:
                lost_frames += 1
                cmd = STOP_CMD
            else:
                lost_frames = 0
                if turn > TURN_RIGHT_THRESH:
                    cmd = CMD_RIGHT
                elif turn < TURN_LEFT_THRESH:
                    cmd = CMD_LEFT
                else:
                    cmd = CMD_UP

            send_cmd(cmd)

            if lost_frames >= LOST_FRAMES_THRESH:
                print(f"[STATE] Final LEFT lane lost for {lost_frames} frames -> stopping program.")
                send_cmd(STOP_CMD)
                break

            # FPS + HUD
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now

            if USE_DISPLAY:
                tl_text = f"LEFT cam (final) | cmd={cmd}"
                if SHOW_STACKED_DEBUG:
                    debug_view = stack_debug_views(frame, mask_debug, out, turn, angle_deg, tl_text)
                else:
                    debug_view = out.copy()
                    cv2.putText(debug_view, tl_text, (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2)

                cv2.putText(debug_view, f"FPS: {fps:.1f}", (10, debug_view.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Lane Debug", debug_view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[STATE] Quit requested during final LEFT phase.")
                    return
            else:
                time.sleep(0.001)

    finally:
        try:
            send_cmd(STOP_CMD)
        except Exception:
            pass
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
        except Exception:
            pass
        if USE_DISPLAY:
            cv2.destroyAllWindows()
        ser.close()

# Entry
if __name__ == "__main__":
    run_hardcoded_left_right_left()

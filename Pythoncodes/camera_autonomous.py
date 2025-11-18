import os
import time
import numpy as np
import cv2
import serial

# =========================
# Serial / driving config
# =========================

PORT = "COM6"
BAUD = 19200
EOL = "\r\n"       # CRLF like PuTTY/miniterm
DTR = True
RTS = True
CHAR_DELAY = 0.01  # 10 ms between bytes (typewriter style)
BOOT_WAIT = 2.0    # seconds after opening port before first send

CMD_UP    = "Forward Half"
CMD_DOWN  = "Backward Half"
CMD_LEFT  = "Left Half"
CMD_RIGHT = "Right Half"
CMD_AUTO  = "Auto"
CMD_MANUAL = "Manual"
STOP_CMD  = "Stop"     # <- TM4C parser should handle this

# Turn thresholds
TURN_LEFT_THRESH  = -0.15
TURN_RIGHT_THRESH =  0.15

# =========================
# Camera / lane tunables
# =========================

# ROI trapezoid focusing on floor in front of robot
ROI_VERTICES_RATIO = dict(
    bottom_left = (0.05, 0.98),
    top_left    = (0.22, 0.60),
    top_right   = (0.78, 0.60),
    bottom_right= (0.95, 0.98),
)

# White mask thresholds (tuned for white duct tape indoors)
HSV_S_MAX = 70
HSV_V_MIN = 170
HSV_H_ANY = (0, 180)

LAB_A_ABS_MAX = 20
LAB_B_ABS_MAX = 20

# YCrCb reinforcement (helps with LED/indoor casts)
Y_MIN = 160
CR_ABS_MAX = 14
CB_ABS_MAX = 14

# Optional OTSU brightness backup (keeps brightest region in ROI)
USE_OTSU_BACKUP = True
OTSU_RATIO = 0.60

# Morphology
OPEN_K  = (3, 3)
CLOSE_K = (11, 11)

# Component filters (relaxed a bit to catch more curved / short segments)
MIN_AREA             = 300     # was 600
MIN_HEIGHT           = 40      # was 70
MIN_WIDTH            = 4       # was 5
MIN_ASPECT_H_OVER_W  = 1.2     # was 1.4
ANGLE_TOL_DEG        = 90      # effectively disabled
TOP_K_COMPONENTS     = 8

# Where we want the line to be (fraction of image width)
# 0.2 = left side, 0.5 = center, 0.8 = right side
TARGET_X_RATIO = 0.43   # you said 0.43 matches your setup

# PD control gains (normalized error in [-1, 1])
Kp = 0.9
Kd = 0.2

# Limit turn authority (now full range so we can cross ±0.6)
TEST_TURN_SCALE = 1.0

# Drawing
LANE_COLOR   = (255, 255, 255)
TARGET_COLOR = (0, 255, 0)
TEXT_COLOR   = (0, 255, 255)

MIRROR_FRAME = False   # robot view

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

# =========================
# Camera / debug helpers
# =========================

def stack_debug_views(raw, mask, overlay, shown_turn, shown_angle, tl_text=""):
    # mask may already be BGR; handle both grayscale/BGR
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
    """Detect white tape: HSV ∧ Lab, reinforced by YCrCb and optional Otsu."""
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
    """|angle from vertical| in degrees for minAreaRect (for filtering & display)."""
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    edges = []
    for i in range(4):
        p0 = box[i]; p1 = box[(i+1) % 4]
        v = p1 - p0
        edges.append((np.linalg.norm(v), v))
    # Take the longest edge as the "direction" of the line
    edges.sort(key=lambda t: t[0], reverse=True)
    vx, vy = np.abs(edges[0][1][0]), np.abs(edges[0][1][1])
    if vx == 0 and vy == 0:
        return 90.0
    # angle from vertical: arctan(|dx| / |dy|)
    ang_deg = np.degrees(np.arctan2(vx, vy))  # 0 = vertical, 90 = horizontal
    return float(ang_deg)

def score_component(rect):
    """Score: prefer big, tall, and near bottom."""
    (cx, cy), (w, h), _ = rect
    area = max(1.0, w*h)
    height_bonus = max(w, h)**2
    bottom_bias = cy
    return area + 0.3*height_bonus + 0.8*bottom_bias

# =========================
# Control helpers
# =========================

class SimpleLineTracker:
    def __init__(self, w):
        self.w = w
        self.prev_err = 0.0

    def control_from_x(self, line_x_bottom):
        target_x = TARGET_X_RATIO * self.w

        if line_x_bottom is None:
            # No line → go straight (or you could search)
            err_norm = 0.0
        else:
            # Pixel error: positive if line is to the RIGHT of target
            err_px = line_x_bottom - target_x
            # Normalize by half screen width
            err_norm = float(np.clip(err_px / (self.w/2), -1.0, 1.0))

        # PD control on normalized error
        d = err_norm - self.prev_err
        self.prev_err = err_norm
        u = Kp * err_norm + Kd * d
        u = float(np.clip(u, -1.0, 1.0))
        return u, err_norm

def apply_steering(norm_error):
    """Map norm_error (-1..1) to turn command (no serial here)."""
    turn = float(np.clip(norm_error, -1.0, 1.0)) * TEST_TURN_SCALE
    forward = 0.6 * (1 - 0.5*abs(turn))  # we still compute forward for debug
    print(f"[CTRL] forward={forward:.2f} turn={turn:.2f}")
    return forward, turn

# =========================
# Frame processing (single-line + positive angle)
# =========================

def process_frame(frame_bgr, tracker: SimpleLineTracker):
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top    = int(h * ROI_VERTICES_RATIO['top_left'][1])

    # ROI + white mask
    roi = region_selection(frame_bgr)
    mask_white = white_mask(roi)
    mask_white = morph_clean(mask_white)

    # Find candidate contours
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
        # Draw the line rect and compute bottom x
        box = cv2.boxPoints(best_rect).astype(np.int32)
        cv2.drawContours(out, [box], 0, LANE_COLOR, 2)
        cv2.drawContours(mask_debug, [box], 0, (0, 255, 255), 2)  # tracking on mask view too

        ys = box[:,1]
        low = box[np.argsort(ys)][-2:]
        line_x_bottom = float(np.mean(low[:,0]))
        cv2.circle(out, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)
        cv2.circle(mask_debug, (int(line_x_bottom), y_bottom-4), 5, (0,0,255), -1)

        # Positive angle from vertical (0..90 deg)
        angle_deg = rect_angle_from_vertical(best_rect)

    # Draw ROI top line
    cv2.line(out, (0, y_top), (w-1, y_top), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(mask_debug, (0, y_top), (w-1, y_top), (255, 0, 0), 1, cv2.LINE_AA)

    # Draw desired target position
    target_x = int(TARGET_X_RATIO * w)
    cv2.line(out, (target_x, y_top), (target_x, y_bottom), TARGET_COLOR, 2, cv2.LINE_AA)
    cv2.line(mask_debug, (target_x, y_top), (target_x, y_bottom), TARGET_COLOR, 2, cv2.LINE_AA)

    # Control
    u, err_norm = tracker.control_from_x(line_x_bottom)
    forward, turn = apply_steering(u)

    # HUD text
    cv2.putText(out, f"LineX: {('%.1f'%line_x_bottom) if line_x_bottom is not None else 'None'}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Err(norm): {err_norm:+.2f}  Turn: {turn:+.2f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Angle(|vert|): {('%.1f'%angle_deg if angle_deg is not None else 'NA')} deg",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # Telemetry
    print(f"[TELEM] LineX={line_x_bottom}  ErrNorm={err_norm:+.2f}  AngleAbsDeg={angle_deg}")

    # line_x_bottom == None is how we know "no lane" in the main loop
    return out, mask_debug, forward, turn, (angle_deg if angle_deg is not None else 0.0), line_x_bottom

# =========================
# Camera helpers (Windows)
# =========================

def open_windows_camera():
    preferred_indices = [1, 0, 2, 3]
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        for idx in preferred_indices:
            cap = cv2.VideoCapture(idx, backend)
            time.sleep(0.3)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if ret:
                print(f"[CAM] Using index {idx} with backend {backend}")
                return cap
            cap.release()
    return None

def open_default_camera():
    if os.name == "nt":
        cap = open_windows_camera()
        if cap is None:
            print("[CAM] Could not open any camera on Windows.")
        return cap

    # Non-Windows (fallback)
    print("[CAM] Non-Windows: trying AVFOUNDATION 0")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        print("[CAM] AVFOUNDATION failed, trying default 0")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        print("[CAM] Could not open any camera.")
        return None
    return cap

# =========================
# Main loop: lane + serial
# =========================

def run_webcam_and_drive():
    # ---- Serial setup ----
    ser = open_serial()
    last_cmd = None

    def send_cmd(cmd: str):
        nonlocal last_cmd
        if cmd != last_cmd:
            send_line_typewriter(ser, cmd)
            last_cmd = cmd

    # Start safe, then Auto mode
    send_cmd(STOP_CMD)
    time.sleep(0.2)
    send_cmd(CMD_AUTO)

    # ---- Camera setup ----
    cap = open_default_camera()
    if cap is None:
        ser.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        ser.close()
        return

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    tracker = SimpleLineTracker(w)
    prev = time.time()

    cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, exiting.")
                break

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)

            result, mask_debug, forward, turn, angle_deg, line_x_bottom = process_frame(frame, tracker)

            # Decide command from turn value and lane presence
            if line_x_bottom is None:
                cmd = STOP_CMD
            else:
                if turn > TURN_RIGHT_THRESH:
                    cmd = CMD_RIGHT
                elif turn < TURN_LEFT_THRESH:
                    cmd = CMD_LEFT
                else:
                    cmd = CMD_UP

            send_cmd(cmd)

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev))
            prev = now
            cv2.putText(result, f"FPS: {fps:.1f}", (10, result.shape[0]-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            tl_text = f"{'MIRROR ' if MIRROR_FRAME else ''}Lane-follow Auto; ESC/q to quit"
            debug_view = stack_debug_views(frame, mask_debug, result, turn, angle_deg, tl_text)
            cv2.imshow("Lane Debug", debug_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        # On exit, stop robot and clean up
        try:
            send_cmd(STOP_CMD)
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        ser.close()

# Entry
if __name__ == "__main__":
    run_webcam_and_drive()

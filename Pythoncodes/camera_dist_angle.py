import os
import time
import numpy as np
import cv2
from collections import deque

# =========================
# Tunables (start here)
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

# Component filters (scale with your resolution)
MIN_AREA             = 600
MIN_HEIGHT           = 70
MIN_WIDTH            = 5
MIN_ASPECT_H_OVER_W  = 1.4
ANGLE_TOL_DEG        = 40
TOP_K_COMPONENTS     = 8

# Temporal smoothing
EMA_ALPHA_CENTER = 0.25
EMA_ALPHA_WIDTH  = 0.20

# Lane width (pixels at bottom)
LANE_WIDTH_INIT = 320
LANE_WIDTH_MIN  = 110
LANE_WIDTH_MAX  = 600

# ===== Real-world calibration =====
# Measure this with a tape measure on your course!
LANE_WIDTH_REAL_M = 0.60   # meters between left & right white lines

# Steering gains (normalized output in [-1, +1])
Kp = 0.9
Kd = 0.2

# Limit the turn authority during testing (0..1) and auto-slow on turns
TEST_TURN_SCALE = 0.5

# Angle-align (single-line mode) gain
ANGLE_GAIN = 0.02   # try 0.01..0.05

# Drawing
LANE_COLOR = (255, 255, 255)
MID_COLOR  = (0, 220, 0)
TEXT_COLOR = (0, 255, 255)
LANE_THICK = 6
MID_THICK  = 4

# Debug / behavior toggles
MIRROR_FRAME = False    # WINDOWS / robot view: no mirroring
DECAY_WHEN_LOST = True  # kept for safety (mode=none forces straight anyway)
DECAY_ALPHA = 0.1

# =========================
# Helpers
# =========================

def stack_debug_views(raw, mask, overlay, shown_turn, tl_text=""):
    # Single composite window (2x2)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h = 300
    raw_r     = cv2.resize(raw,     (int(raw.shape[1] * h/raw.shape[0]), h))
    mask_r    = cv2.resize(mask_bgr,(raw_r.shape[1], h))
    overlay_r = cv2.resize(overlay, (raw_r.shape[1], h))
    steer_panel = np.zeros_like(overlay_r)
    cv2.putText(steer_panel, f"Motor turn: {shown_turn:+.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 3)
    cv2.putText(steer_panel, "(Raw | Mask | Overlay)", (10, 160),
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
    """Detect white tape: HSV ∧ Lab, reinforced by YCrCb and OTSU brightness."""
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
    """
    Robustly compute |angle from vertical| in degrees for a minAreaRect.
    We find the longer edge vector and measure its angle to the +Y axis.
    """
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

def rect_signed_angle_deg(rect):
    """Signed angle of the long edge relative to vertical; right-lean positive."""
    box = cv2.boxPoints(rect).astype(np.float32)
    edges = []
    for i in range(4):
        p0 = box[i]; p1 = box[(i+1) % 4]
        v = p1 - p0
        edges.append((np.linalg.norm(v), v))
    edges.sort(key=lambda t: t[0], reverse=True)
    vx, vy = edges[0][1]
    ang = np.degrees(np.arctan2(vx, vy))  # signed: right-lean positive
    return float(ang)

def score_component(rect):
    (cx, cy), (w, h), _ = rect
    area = max(1.0, w*h)
    height_bonus = max(w, h)**2
    bottom_bias = cy
    return area + 0.3*height_bonus + 0.8*bottom_bias

# =========================
# Tracking & Control
# =========================

class LaneTracker:
    """
    Tracks an estimate of lane width + center at the bottom of the image.

    Handles:
      - two visible lines  -> use both
      - one visible line   -> infer center using lane_width_ema
      - no visible lines   -> mark lost (steer straight)
    """
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.center_bottom_ema = w/2
        self.lane_width_ema = LANE_WIDTH_INIT
        self.prev_error = 0.0
        self.lost_frames = 0
        self.mode = "none"  # "none" | "one-left" | "one-right" | "two"

    def update_from_two(self, xl, xr):
        if xl is None or xr is None:
            return False
        if xl > xr:
            xl, xr = xr, xl
        width  = np.clip(xr - xl, LANE_WIDTH_MIN, LANE_WIDTH_MAX)
        center = 0.5*(xl + xr)
        self.lane_width_ema    = (1-EMA_ALPHA_WIDTH)*self.lane_width_ema + EMA_ALPHA_WIDTH*width
        self.center_bottom_ema = (1-EMA_ALPHA_CENTER)*self.center_bottom_ema + EMA_ALPHA_CENTER*center
        self.lost_frames = 0
        self.mode = "two"
        return True

    def update_from_one(self, side_xb, side: str):
        if side_xb is None:
            return False
        if side == 'left':
            center = side_xb + self.lane_width_ema/2
            self.mode = "one-left"
        else:
            center = side_xb - self.lane_width_ema/2
            self.mode = "one-right"
        self.center_bottom_ema = (1-EMA_ALPHA_CENTER)*self.center_bottom_ema + EMA_ALPHA_CENTER*center
        self.lost_frames = 0
        return True

    def mark_lost(self):
        self.lost_frames += 1
        self.mode = "none"

    def center_error_only(self):
        """
        Return normalized center-following error in [-1,1] (no angle term).

        err = -1  -> camera sits on left line
        err = +1  -> camera sits on right line
        err =  0  -> camera at lane center
        """
        center_img = self.w/2
        half_w = max(1.0, self.lane_width_ema/2)
        err_px = (self.center_bottom_ema - center_img) / half_w
        err_px = float(np.clip(err_px, -1.0, 1.0))
        # PD
        d = err_px - self.prev_error
        self.prev_error = err_px
        u = Kp*err_px + Kd*d
        u = float(np.clip(u, -1.0, 1.0))
        # If no lines -> force straight
        if self.mode == "none":
            u = 0.0
        return u, err_px  # also return raw center error

def draw_lane_and_mid(out, left_rect, right_rect, y_top, y_bottom):
    def bottom_x(rect):
        if rect is None:
            return None
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(out, [box], 0, LANE_COLOR, 2)
        ys = box[:,1]
        low = box[np.argsort(ys)][-2:]
        return float(np.mean(low[:,0]))
    xl = bottom_x(left_rect)
    xr = bottom_x(right_rect)
    if xl is not None and xr is not None:
        xm = int(0.5*(xl + xr))
        cv2.line(out, (xm, y_top), (xm, y_bottom), MID_COLOR, MID_THICK, cv2.LINE_AA)
    xc = int(out.shape[1]//2)
    cv2.circle(out, (xc, y_bottom-4), 4, (0, 255, 255), -1)
    return xl, xr

def apply_steering(norm_error):
    """Map norm_error (-1..1) to motor commands. Returns (forward, turn)."""
    turn = float(np.clip(norm_error, -1.0, 1.0)) * TEST_TURN_SCALE
    forward = 0.6 * (1 - 0.5*abs(turn))  # slow down when turning
    print(f"[CTRL] forward={forward:.2f} turn={turn:.2f}")
    return forward, turn

# =========================
# Frame processor
# =========================

def frame_processor(frame_bgr, tracker: LaneTracker):
    h, w = frame_bgr.shape[:2]
    y_bottom = h - 1
    y_top    = int(h * ROI_VERTICES_RATIO['top_left'][1])

    # 1) ROI + white mask
    roi = region_selection(frame_bgr)
    mask_white = white_mask(roi)
    mask_white = morph_clean(mask_white)

    # 2) Contours -> candidates with geometry checks
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
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
        ang_from_vert = rect_angle_from_vertical(rect)  # robust angle
        if ang_from_vert > ANGLE_TOL_DEG:
            continue
        score = score_component(rect)
        candidates.append((score, rect))

    candidates.sort(key=lambda t: t[0], reverse=True)
    rects = [r for _, r in candidates[:TOP_K_COMPONENTS]]

    # 3) Choose left/right by bottom x; prefer pair that straddles center
    out = frame_bgr.copy()
    bottoms = []
    for r in rects:
        box = cv2.boxPoints(r).astype(np.int32)
        ys = box[:,1]
        low = box[np.argsort(ys)][-2:]
        xb = float(np.mean(low[:,0]))
        bottoms.append((xb, r))
    bottoms.sort(key=lambda t: t[0])

    center_x = w/2
    left_rect = right_rect = None
    pair = None
    for i in range(len(bottoms)-1):
        xl, rl = bottoms[i]
        xr, rr = bottoms[i+1]
        if xl < center_x < xr:
            pair = (rl, rr)
            break
    if pair is None and len(bottoms) >= 2:
        # two closest to center
        near = sorted(bottoms, key=lambda t: abs(t[0]-center_x))[:2]
        if near[0][0] <= near[1][0]:
            pair = (near[0][1], near[1][1])
        else:
            pair = (near[1][1], near[0][1])

    if pair is not None:
        left_rect, right_rect = pair

    # 4) Draw & measure
    xl, xr = draw_lane_and_mid(out, left_rect, right_rect, y_top, y_bottom)

    # 5) Update tracker states explicitly for 2/1/0 line cases
    updated = False
    if xl is not None and xr is not None:
        updated = tracker.update_from_two(xl, xr)
    elif xl is not None:
        updated = tracker.update_from_one(xl, 'left')
    elif xr is not None:
        updated = tracker.update_from_one(xr, 'right')
    if not updated:
        tracker.mark_lost()

    # 6) Angle-align (single-line mode): add a heading term
    left_ang  = rect_signed_angle_deg(left_rect)  if left_rect  is not None else None
    right_ang = rect_signed_angle_deg(right_rect) if right_rect is not None else None

    heading_term = 0.0
    if tracker.mode == "one-left" and left_ang is not None:
        heading_term = np.clip(left_ang, -40, 40) * ANGLE_GAIN
    elif tracker.mode == "one-right" and right_ang is not None:
        heading_term = np.clip(right_ang, -40, 40) * ANGLE_GAIN

    # Base center-following error + heading alignment (still normalized -1..1)
    u_center, err_center = tracker.center_error_only()
    steer_err  = float(np.clip(u_center + heading_term, -1.0, 1.0))

    # ---------- Real-world distances & heading ----------
    # Lateral offset from lane center (meters, signed):
    # err_center = -1 (on left line), +1 (on right line), 0 (center)
    if tracker.mode == "none":
        lateral_offset_m = 0.0
    else:
        lateral_offset_m = err_center * (LANE_WIDTH_REAL_M / 2.0)

    # Distance from camera center to the *visible* line (nearest one)
    center_img = w/2
    line_offset_px = None
    if tracker.mode in ("one-left", "two") and xl is not None:
        line_offset_px = xl - center_img
    if tracker.mode in ("one-right", "two") and xr is not None:
        # choose closer of the two if both exist
        cand = xr - center_img
        if line_offset_px is None or abs(cand) < abs(line_offset_px):
            line_offset_px = cand

    if line_offset_px is None or tracker.lane_width_ema <= 1.0:
        line_offset_m = 0.0
    else:
        # pixel -> meter scale based on lane width
        meters_per_px = LANE_WIDTH_REAL_M / tracker.lane_width_ema
        line_offset_m = line_offset_px * meters_per_px

    # Approx lane heading in degrees (0 = straight, +ve = lane rotated to the right)
    lane_heading_deg = 0.0
    if tracker.mode == "two":
        hs = []
        if left_ang is not None:
            hs.append(left_ang)
        if right_ang is not None:
            hs.append(right_ang)
        if hs:
            lane_heading_deg = float(np.mean(hs))
    elif tracker.mode == "one-left" and left_ang is not None:
        lane_heading_deg = float(left_ang)
    elif tracker.mode == "one-right" and right_ang is not None:
        lane_heading_deg = float(right_ang)

    # HUD + telemetry
    center_ema = tracker.center_bottom_ema
    width_ema  = tracker.lane_width_ema

    # Distance from image center to nearest visible line (pixels, signed) for display
    dist_px = line_offset_px

    cv2.putText(out, f"XL:{str(round(xl,1)) if xl is not None else 'None'}  "
                     f"XR:{str(round(xr,1)) if xr is not None else 'None'}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Mode:{tracker.mode}  Ctr:{center_ema:.0f}  W:{width_ema:.0f}  "
                     f"Dist(px):{('%.0f'%dist_px) if dist_px is not None else 'NA'}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(out, f"Err:{steer_err:+.2f}  Lat(m):{lateral_offset_m:+.2f}  Line(m):{line_offset_m:+.2f}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if abs(steer_err)<0.2 else (0,140,255), 2)
    cv2.putText(out, f"Heading(deg):{lane_heading_deg:+.1f}", (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # Console telemetry
    print(f"[TELEM] XL={xl} XR={xr} MODE={tracker.mode} CENTER={center_ema:.1f} "
          f"WIDTHpx={width_ema:.1f} ERR={steer_err:+.2f} ErrCenter={err_center:+.2f} "
          f"LatM={lateral_offset_m:+.3f} LineM={line_offset_m:+.3f} "
          f"HeadingDeg={lane_heading_deg:+.2f} DistPx={dist_px}")

    return out, steer_err, mask_white, xl, xr, lateral_offset_m, line_offset_m, lane_heading_deg

# =========================
# Camera opening (Windows-friendly)
# =========================

def open_windows_camera():
    """
    Try to find a working camera on Windows, preferring external (C920) first.
    Returns an opened cv2.VideoCapture or None.
    """
    preferred_indices = [1, 0, 2, 3]   # try 1 first (often external), then 0,2,3
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
    """
    Cross-platform camera open. On Windows uses open_windows_camera().
    On Mac/Linux falls back to camera 0 with AVFOUNDATION / default.
    """
    if os.name == "nt":   # Windows
        cap = open_windows_camera()
        if cap is None:
            print("[CAM] Could not open any camera on Windows.")
        return cap

    # ---- non-Windows (old Mac behavior) ----
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
# Main loop
# =========================

def run_webcam():
    cap = open_default_camera()
    if cap is None:
        return

    # Request a reasonable size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        cap.release()
        return

    if MIRROR_FRAME:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    tracker = LaneTracker(w, h)
    prev = time.time()

    cv2.namedWindow("Lane Debug", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed, exiting.")
            break

        if MIRROR_FRAME:
            frame = cv2.flip(frame, 1)

        result, steer_err, mask_white, xl, xr, lateral_offset_m, line_offset_m, lane_heading_deg = \
            frame_processor(frame, tracker)

        # Convert steering to motor commands
        forward, motor_turn = apply_steering(steer_err)

        # Example: these are the two key values you'll send to the robot:
        #   lateral_offset_m  -> how far left/right from lane center
        #   lane_heading_deg  -> how much robot heading differs from lane
        # print(f"[NAV] lat={lateral_offset_m:+.3f} m, heading={lane_heading_deg:+.2f} deg")

        # FPS stamp on overlay
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        cv2.putText(result, f"FPS: {fps:.1f}", (10, result.shape[0]-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        tl_text = f"{'MIRROR ' if MIRROR_FRAME else ''}ROI trapezoid; ESC/q to quit"
        debug_view = stack_debug_views(frame, mask_white, result, motor_turn, tl_text)
        cv2.imshow("Lane Debug", debug_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry
if __name__ == "__main__":
    run_webcam()

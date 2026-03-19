import time, math, serial, cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

# ===============
# CONFIGURATION 
# ===============
LIDAR_PORT   = "/dev/ttyUSB0"
MOTOR_PORT   = "/dev/ttyACM0" 
LIDAR_BAUD   = 1_000_000
MOTOR_BAUD   = 19200

# Behavior Thresholds (mm)
THRESH_MAIN       = 762   # Obstacle detected if closer than this
NEAR_OBS_TRIP     = 300   # Emergency backup threshold
NEAR_OBS_RELEASE  = 500   # Safe distance to stop backing up

# Commands
CMD_FWD   = "Forward Half"
CMD_REV   = "Backward Half"
CMD_LEFT  = "Left Half"
CMD_RIGHT = "Right Half"
CMD_STOP  = "Stop"

# Vision ROI & Alignment
ROI_VERTICES_RATIO = {'bottom_left': (0.05, 0.98), 'top_left': (0.22, 0.60), 
                      'top_right': (0.78, 0.60), 'bottom_right': (0.95, 0.98)}
TURN_LEFT_THRESH   = -0.20
TURN_RIGHT_THRESH  = 0.20

# PD gains
KP = 0.9
KD = 0.2

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
        u = KP * err_norm + KD * d
        u = float(np.clip(u, -1.0, 1.0))
        return u, err_norm
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

#detecting the white lane with a white mask
def white_mask(bgr):
    """Refined version of white_mask using conditional logic for speed."""
    # 1. Start with HSV (The fastest/most restrictive filter for white)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) #
    h, s, v = cv2.split(hsv) #
    
    # Check for low saturation and high value (characteristic of white)
    m_hsv = (s <= HSV_S_MAX) & (v >= HSV_V_MIN) #
    m_hsv &= (h >= HSV_H_ANY[0]) & (h <= HSV_H_ANY[1]) #

    # --- LAZY STEP 1: If HSV sees nothing, don't bother with the rest ---
    if not np.any(m_hsv):
        return np.zeros(bgr.shape[:2], dtype=np.uint8)

    # 2. Only check YCrCb if HSV found potential pixels
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb) #
    Y, Cr, Cb = cv2.split(ycc) #

    m_y   = (Y >= Y_MIN) #
    m_cr  = (np.abs(Cr.astype(np.int16)-128) <= CR_ABS_MAX) #
    m_cb  = (np.abs(Cb.astype(np.int16)-128) <= CB_ABS_MAX) #
    
    # Combine HSV and YCC (I've removed LAB for speed as it's often redundant)
    mask = m_hsv & m_y & m_cr & m_cb 

    # --- LAZY STEP 2: Only run the heavy Otsu backup if we still have pixels ---
    if USE_OTSU_BACKUP and np.any(mask): #
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) #
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #
        thr = int(thr * OTSU_RATIO) #
        _, m_gray = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY) #
        mask &= (m_gray > 0) #

    return mask.astype(np.uint8) * 255

#cleans white noise from scan
def morph_clean(mask):
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, OPEN_K)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m

#calculates angles
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

#decides which white object is most likely the actual lane.
def score_component(rect):
    (cx, cy), (w, h), _ = rect
    area = max(1.0, w*h)
    height_bonus = max(w, h)**2
    bottom_bias = cy
    return area + 0.3*height_bonus + 0.8*bottom_bias

def apply_steering(norm_error):
    turn = float(np.clip(norm_error, -1.0, 1.0)) * TEST_TURN_SCALE
    forward = 0.6 * (1 - 0.5*abs(turn))
    print(f"[CTRL] forward={forward:.2f} turn={turn:.2f}")
    return forward, turn

#very heavy, needs major inprovement
def process_frame(frame_bgr, tracker: SimpleLineTracker):
   # 1. DOWNSAMPLE: Immediately reduce pixel count by 75%
    frame_small = cv2.resize(frame_bgr, (480, 270))
    h, w = frame_small.shape[:2]
    
    # 2. ROI: Crop the image to the driving area before heavy math
    roi = region_selection(frame_small)
    
    # 3. LAZY MASK: Skip expensive LAB/Otsu if no white is found
    mask_white = white_mask(roi)
    mask_white = morph_clean(mask_white)
    
    # 4. CONTOURS: Find the lane
    contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_score = -1e9

    for cnt in contours:
        if cv2.contourArea(cnt) < (MIN_AREA / 4): continue # Scale area for 50% resize
        
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect
        if max(rw, rh) < (MIN_HEIGHT / 2): continue # Scale height for 50% resize
        
        score = score_component(rect)
        if score > best_score:
            best_score = score
            best_rect = rect

    # 5. TRACKING: Calculate steering
    line_x_bottom = None
    if best_rect is not None:
        box = cv2.boxPoints(best_rect).astype(np.int32)
        ys = box[:,1]
        low = box[np.argsort(ys)][-2:]
        line_x_bottom = float(np.mean(low[:,0]))

    u, err_norm = tracker.control_from_x(line_x_bottom)
    forward, turn = apply_steering(u)

    return None, None, forward, turn, 0.0, line_x_bottom, err_norm




class UnifiedIGV:
    def __init__(self):
        # 1. Hardware Initialization
        print("[INIT] Opening Serial & LIDAR...")
        self.ser = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=0.5)
        
        
        # 2. Camera Initialization (Persistent Streams)
        print("[INIT] Opening Cameras (0 and 2)...")
        self.cap_L = cv2.VideoCapture(2, cv2.CAP_V4L2)
        self.cap_R = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # 3. State Tracking
        self.active_side = "LEFT" 
        self.last_cmd = None
        self.lidar_bins = {"L": math.inf, "C": math.inf, "R": math.inf}
        
        # 4. Vision Trackers
        self.tracker_L = SimpleLineTracker(480, 0.43) # Target 43% of 480 pixels
        self.tracker_R = SimpleLineTracker(480, 0.65) # Target 65% of 480 pixels
        
    def initializeHardware(self):
        # Initialize variables as None first
        self.lidar, self.camera, self.motors = None, None, None

        try:
            # 1. Lidar Setup
            try:
                self.lidar = RPLidar(LIDAR_PORT)
                
                info = self.lidar.get_info()
                print(f"Info: {info}")
                
                health = self.lidar.get_health()
                print(f"Status: {health[0]}")
            except Exception as e:
                print(f"Lidar failed to start: {e}")

            # 2. Camera Setup
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened(): raise Exception("Camera port busy")
                print("Camera initialized.")
            except Exception as e:
                print(f"Camera failed to start: {e}")

            # 3. Motor Setup
            try:
                motors = MotorController(MOTOR_PORT)
                print("Motors initialized.")
            except Exception as e:
                print(f"Motors failed to start: {e}")



        finally:
            # Granular cleanup: only close what was actually opened
            if lidar: 
                lidar.stop_motor()
                lidar.disconnect()
            if camera: 
             camera.release()
            if motors: 
                motors.stop_all()
            print("All active hardware released.")

    def send_cmd(self, cmd):
        """Typewriter-style serial output to TM4C."""
        if cmd != self.last_cmd:
            print(f"TX: {cmd}")
            data = (cmd + "\r").encode("ascii")
            self.ser.write(data)
            self.ser.flush()
            # for b in data:
            #     self.ser.write(bytes([b]))
            #     self.ser.flush()
            #     time.sleep(0.01)
            self.last_cmd = cmd

    def update_lidar(self, scan):
        """Processes raw scan into simplified distance bins."""
        bins = {a: math.inf for a in [-50, 0, 50]}
        for q, ang, dist in scan:
            if dist <= 0: continue
            # Calculate angle relative to front (mirrored mount)
            rel = -1 * ((ang + 180) % 360 - 180)
            for tgt in bins:
                if abs(rel - tgt) <= 5 and dist < bins[tgt]:
                    bins[tgt] = dist
        self.lidar_bins = {"R": bins[-50], "C": bins[0], "L": bins[50]}

    def get_vision_turn(self):
        """Processes the frame already grabbed by the main loop."""
        # Select the active hardware and tracker
        cap = self.cap_L if self.active_side == "LEFT" else self.cap_R
        tracker = self.tracker_L if self.active_side == "LEFT" else self.tracker_R
        
        # .retrieve() decodes the frame that was .grab()ed in the main loop
        ret, frame = cap.retrieve()
        if not ret: 
            return None
        
        # Standard lane processing logic
        _, _, _, turn, _, line_x, _ = process_frame(frame, tracker)
        return turn if line_x is not None else None

    def decide_action(self):
        """The Priority Arbiter: Checks Safety -> Avoidance -> Navigation."""
        dL, dC, dR = self.lidar_bins["L"], self.lidar_bins["C"], self.lidar_bins["R"]
        
        # PRIORITY 1: Emergency Reflex (Too close to anything)
        if min(dL, dC, dR) < NEAR_OBS_TRIP:
            return CMD_REV

        # PRIORITY 2: Lane Selection (Obstacle blocking current lane)
        if self.active_side == "LEFT" and dL < THRESH_MAIN:
            print("[AUTO] Left blocked! Switching to Right lane.")
            self.active_side = "RIGHT"
            return CMD_RIGHT # Immediate turn away from obstacle
        
        if self.active_side == "RIGHT" and dR < THRESH_MAIN:
            print("[AUTO] Right blocked! Switching to Left lane.")
            self.active_side = "LEFT"
            return CMD_LEFT

        # PRIORITY 3: Lane Following (Standard Vision)
        turn = self.get_vision_turn()
        if turn is None:
            return CMD_STOP
        
        if turn > TURN_RIGHT_THRESH: return CMD_RIGHT
        if turn < TURN_LEFT_THRESH:  return CMD_LEFT
        return CMD_FWD
    
   
    def run(self):
        print("[RUN] System Active. Using Priority Arbiter.")
        try:
            for scan in self.lidar.iter_scans(max_buf_meas=10000):
                #step 1: update lidar
                self.update_lidar(scan)

                #step 2: get information from both cameras and clear buffers
                self.cap_L.grab()
                self.cap_R.grab()

                #decide an action
                cmd = self.decide_action()
                self.send_cmd(cmd)

        except KeyboardInterrupt:
            self.send_cmd(CMD_STOP)
        finally:
            self.lidar.stop(); self.lidar.disconnect()
            self.cap_L.release(); self.cap_R.release()
            self.ser.close()


if __name__ == "__main__":
    # Ensure all your helper functions (process_frame, SimpleLineTracker, etc.) 
    # from camera_part3.py are available in the script or imported.
    bot = UnifiedIGV()
    bot.run()
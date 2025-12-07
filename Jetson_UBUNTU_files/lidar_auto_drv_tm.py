#!/usr/bin/env python3
# RPLidar 3-angle avoidance -> TM4C (word commands)
# Stability-tuned: tolerant to scan hiccups, graceful after command changes.

import time, math, argparse
import serial
from rplidar import RPLidar, RPLidarException

# ---------- Defaults (override via CLI) ----------
LIDAR_PORT   = "/dev/ttyUSB0"
LIDAR_BAUD   = 1_000_000     # confirmed by your probe
LIDAR_TO     = 1.0
FRONT_DEG    = 0
ANGLE_SIGN_D = -1            # -1 for your mirrored mount

MOTOR_PORT   = "/dev/ttyACM2"
MOTOR_BAUD   = 19200
MOTOR_TO     = 0.4
EOL          = "\r"          # CR only (your TM4C ends on c==13)
BOOT_WAIT    = 2.0
CHAR_DELAY   = 0.01          # 10 ms per byte (typewriter)

LOOK_ANGLES  = [-50, 0, +50] # Right, Center, Left
ANG_TOL      = 5

# Behavior thresholds (mm)
THRESH_MAIN        = 762
BACKOFF_CLEAR_ANY  = 850
NEAR_OBS_TRIP      = 300
NEAR_OBS_RELEASE   = 500
POST_NEAR_CLEAR    = 672

# TM4C word commands
CMD_FWD   = "Forward Half"
CMD_REV   = "Backward Half"
CMD_LEFT  = "Left Half"
CMD_RIGHT = "Right Half"
CMD_STOP  = "Stop"

PRINT_INF_AS = -1

# Robustness knobs
BAD_LIMIT       = 10   # reopen lidar only after this many consecutive bad frames
GRACE_FRAMES_TX = 2    # ignore N frames after sending a motion command
SCAN_MIN_LEN    = 30   # accept shorter frames a bit more to avoid churn
SCAN_MAXBUF     = 16384

# ---------- Helpers ----------
def wrap180(a): return (a+180)%360-180
def rel_to_front(a_abs, angle_sign, front_deg): return angle_sign * wrap180(a_abs - front_deg)
def fmt_mm(x): return int(x) if x < math.inf else PRINT_INF_AS

def open_motor(port, baud):
    ser = serial.Serial(
        port=port, baudrate=baud, timeout=MOTOR_TO,
        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
        rtscts=False, dsrdtr=False, xonxoff=False
    )
    # match your Windows arrow script behavior
    ser.dtr = True
    ser.rts = True
    time.sleep(BOOT_WAIT)
    try:
        ser.reset_input_buffer(); ser.reset_output_buffer()
    except Exception:
        pass
    return ser

def send_line_typewriter(ser, text):
    data = (text + EOL).encode("ascii", errors="ignore")
    print("TX:", repr(text + EOL))
    for b in data:
        ser.write(bytes([b])); ser.flush()
        time.sleep(CHAR_DELAY)

def open_lidar(port, baud, timeout):
    lid = RPLidar(port, baudrate=baud, timeout=timeout)
    time.sleep(0.1)
    try: lid.start_motor()
    except Exception: pass
    # drain any garbage quickly
    try:
        if hasattr(lid, 'clean_input'):
            lid.clean_input()
        elif hasattr(lid, 'clear_input'):
            lid.clear_input()
        else:
            try: _ = lid._serial.read(4096)
            except: pass
    except: pass
    # settle
    try: _ = lid.get_info()
    except: pass
    try: _ = lid.get_health()
    except: pass
    return lid

def iter_scans_standard(lidar):
    # STANDARD scans only; bigger buffer, slightly smaller min_len for tolerance
    for scan in lidar.iter_scans(max_buf_meas=SCAN_MAXBUF, min_len=SCAN_MIN_LEN):
        yield scan

def bins_from_scan(scan, angle_sign, front_deg):
    """Compute minima at -50, 0, +50 from one full scan frame."""
    bins = {a: math.inf for a in LOOK_ANGLES}
    for q, ang, dist in scan:
        if dist <= 0:
            continue
        # (Optional) lightly clamp absurdly huge ranges so they don't sway logic
        if dist > 6000:
            dist = 6000
        rel = rel_to_front(ang, angle_sign, front_deg)
        for tgt in LOOK_ANGLES:
            if abs(rel - tgt) <= ANG_TOL and dist < bins[tgt]:
                bins[tgt] = dist
    return bins

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="RPLidar 3-angle avoidance -> TM4C (word protocol, stable).")
    ap.add_argument("--lidar", default=LIDAR_PORT)
    ap.add_argument("--lidar-baud", type=int, default=LIDAR_BAUD)
    ap.add_argument("--motor", default=MOTOR_PORT)
    ap.add_argument("--motor-baud", type=int, default=MOTOR_BAUD)
    ap.add_argument("--mirror", type=int, default=ANGLE_SIGN_D, choices=[-1, 1])
    ap.add_argument("--front", type=int, default=FRONT_DEG)
    args = ap.parse_args()

    angle_sign = args.mirror
    front_deg  = args.front

    # Motor/TM4C
    print(f"[INFO] MOTOR {args.motor}@{args.motor_baud}")
    mot = open_motor(args.motor, args.motor_baud)
    last_sent = None
    grace_frames = 0

    def send_cmd(cmd):
        nonlocal last_sent, grace_frames
        if cmd != last_sent:
            send_line_typewriter(mot, cmd)
            last_sent = cmd
            # give lidar a couple frames to settle after USB/EMI jolts from motors
            grace_frames = GRACE_FRAMES_TX

    send_cmd(CMD_STOP)

    # Lidar
    print(f"[INFO] LIDAR {args.lidar}@{args.lidar_baud} (ANGLE_SIGN={angle_sign})")
    lidar = open_lidar(args.lidar, args.lidar_baud, LIDAR_TO)
    scans = iter_scans_standard(lidar)
    print("CTRL-C to quit.")

    # State machine
    state = None
    clear_targets = set()
    pivot_clear_thresh = THRESH_MAIN
    near_target = None

    bad_streak = 0

    try:
        # skip one settling frame
        try: _ = next(scans)
        except Exception: pass

        while True:
            # --- get one full scan frame ---
            try:
                scan = next(scans)
                bad_streak = 0
            except (StopIteration, RPLidarException, OSError) as e:
                if grace_frames > 0:
                    # in grace, just ignore this hiccup quietly
                    grace_frames -= 1
                    continue
                bad_streak += 1
                if bad_streak >= BAD_LIMIT:
                    print(f"[WARN] lidar: {e if str(e) else 'scan framing error'} -> reopen (streak={bad_streak})")
                    bad_streak = 0
                    try:
                        lidar.stop(); lidar.stop_motor(); lidar.disconnect()
                    except Exception: pass
                    time.sleep(0.25)
                    lidar = open_lidar(args.lidar, args.lidar_baud, LIDAR_TO)
                    scans = iter_scans_standard(lidar)
                    try: _ = next(scans)
                    except Exception: pass
                # whether we reopened or not, skip this iteration
                continue

            if grace_frames > 0:
                grace_frames -= 1
                # don’t decide during grace; just show it
                bins = bins_from_scan(scan, angle_sign, front_deg)
                dR, dC, dL = bins[-50], bins[0], bins[+50]
                print(f"L={fmt_mm(dL)}  C={fmt_mm(dC)}  R={fmt_mm(dR)}  -> (grace)")
                continue

            bins = bins_from_scan(scan, angle_sign, front_deg)
            dR, dC, dL = bins[-50], bins[0], bins[+50]
            print(f"L={fmt_mm(dL)}  C={fmt_mm(dC)}  R={fmt_mm(dR)}", end="  ")

            have_any = not (math.isinf(dL) and math.isinf(dC) and math.isinf(dR))
            if not have_any:
                state, clear_targets, near_target = 'STOP', set(), None
                send_cmd(CMD_STOP)
                print("-> STOP (no data)")
                continue

            openR_main = dR >= THRESH_MAIN
            openC_main = dC >= THRESH_MAIN
            openL_main = dL >= THRESH_MAIN
            all_blocked_main = (not openR_main) and (not openC_main) and (not openL_main)

            # -------- NEAR OBSTACLE --------
            if state != 'BACKOFF_NEAR':
                candidates = []
                if dR < NEAR_OBS_TRIP: candidates.append(('R', dR))
                if dC < NEAR_OBS_TRIP: candidates.append(('C', dC))
                if dL < NEAR_OBS_TRIP: candidates.append(('L', dL))
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    near_target = candidates[0][0]
                    state = 'BACKOFF_NEAR'
                    clear_targets = set()
                    send_cmd(CMD_REV)
                    print(f"-> BACKOFF_NEAR {near_target} {CMD_REV}")
                    continue
            else:
                val = {'R': dR, 'C': dC, 'L': dL}[near_target]
                if val >= NEAR_OBS_RELEASE:
                    state = None
                    clear_targets = set()
                    near_target = None
                    pivot_clear_thresh = POST_NEAR_CLEAR
                else:
                    send_cmd(CMD_REV)
                    print(f"-> BACKOFF_NEAR {near_target} {CMD_REV}")
                    continue

            # -------- Existing states --------
            if state == 'BACKOFF':
                if max(dR, dC, dL) >= BACKOFF_CLEAR_ANY:
                    state, clear_targets = None, set()
                    send_cmd(CMD_STOP)
                else:
                    send_cmd(CMD_REV)
                    print("-> BACKOFF", CMD_REV)
                    continue

            elif state in ('PIVOT_L', 'PIVOT_R'):
                if all_blocked_main:
                    state, clear_targets, near_target = 'BACKOFF', set(), None
                    send_cmd(CMD_REV)
                    print("-> BACKOFF (all blocked during pivot)", CMD_REV)
                    continue

                needR = ('R' in clear_targets) and (dR < pivot_clear_thresh)
                needC = ('C' in clear_targets) and (dC < pivot_clear_thresh)
                needL = ('L' in clear_targets) and (dL < pivot_clear_thresh)
                if not (needR or needC or needL):
                    state, clear_targets = None, set()
                    send_cmd(CMD_STOP)
                else:
                    send_cmd(CMD_LEFT if state == 'PIVOT_L' else CMD_RIGHT)
                    print(f"-> {state} clearing {sorted(clear_targets)}@{pivot_clear_thresh}")
                    continue

            # -------- Fresh decision --------
            if openR_main and openC_main and openL_main:
                state, clear_targets = 'FWD', set()
                pivot_clear_thresh = THRESH_MAIN
                send_cmd(CMD_FWD)
            else:
                if not openC_main:
                    side_open = []
                    if openR_main: side_open.append(('R', dR))
                    if openL_main: side_open.append(('L', dL))
                    if side_open:
                        side_open.sort(key=lambda x: x[1], reverse=True)
                        best = side_open[0][0]
                        if best == 'L':
                            state, clear_targets = 'PIVOT_L', {'C','R'}
                            pivot_clear_thresh = THRESH_MAIN
                            send_cmd(CMD_LEFT)
                        else:
                            state, clear_targets = 'PIVOT_R', {'C','L'}
                            pivot_clear_thresh = THRESH_MAIN
                            send_cmd(CMD_RIGHT)
                    else:
                        state, clear_targets, near_target = 'BACKOFF', set(), None
                        pivot_clear_thresh = THRESH_MAIN
                        send_cmd(CMD_REV)
                else:
                    if (dR < THRESH_MAIN) and (dL >= THRESH_MAIN):
                        state, clear_targets = 'PIVOT_L', {'R'}
                        pivot_clear_thresh = THRESH_MAIN
                        send_cmd(CMD_LEFT)
                    elif (dL < THRESH_MAIN) and (dR >= THRESH_MAIN):
                        state, clear_targets = 'PIVOT_R', {'L'}
                        pivot_clear_thresh = THRESH_MAIN
                        send_cmd(CMD_RIGHT)
                    elif (dL < THRESH_MAIN) and (dR < THRESH_MAIN):
                        if dL >= dR:
                            state, clear_targets = 'PIVOT_L', {'L','R'}
                            pivot_clear_thresh = THRESH_MAIN
                            send_cmd(CMD_LEFT)
                        else:
                            state, clear_targets = 'PIVOT_R', {'L','R'}
                            pivot_clear_thresh = THRESH_MAIN
                            send_cmd(CMD_RIGHT)
                    else:
                        state, clear_targets = 'FWD', set()
                        pivot_clear_thresh = THRESH_MAIN
                        send_cmd(CMD_FWD)

            print(f"-> {state}")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C")
    finally:
        try: send_cmd(CMD_STOP)
        except Exception: pass
        try: mot.close()
        except Exception: pass
        try:
            lidar.stop(); lidar.stop_motor(); lidar.disconnect()
        except Exception: pass
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()

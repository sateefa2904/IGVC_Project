#!/usr/bin/env python3
"""
IGVC integrated launcher:
- camera only
- lidar only
- all (camera + lidar safety override)
"""

import argparse
import math
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import Camera_Only as cam


NEAR_OBS_TRIP_MM = 300.0
OPEN_THRESH_MM = 762.0
BACKOFF_CLEAR_ANY_MM = 850.0
NEAR_OBS_RELEASE_MM = 500.0
POST_NEAR_CLEAR_MM = 672.0

ENC_RE = re.compile(r"\[ENC\]\s+rpmL=([-+]?\d*\.?\d+)\s+rpmR=([-+]?\d*\.?\d+)\s+mph=([-+]?\d*\.?\d+)")

# ---------------------------------------------------------------------------
# Integrated / lane-follow tuning (edit here or override on the CLI)
# ---------------------------------------------------------------------------
# Blind pulse runs when only one lane line is visible: turn for TURN sec, then
# Stop for WAIT sec. Larger turn = stronger correction; larger wait = more
# spacing between pulses (slower average lateral motion).
DEFAULT_BLIND_PULSE_TURN_SEC = 1.25
DEFAULT_BLIND_PULSE_WAIT_SEC = 2.75
# GentleTurnPacer (PWM) for lidar pivots: used heavily in lidar-only; in "all"
# mode it is applied only when lidar overrides the camera command (see
# run_camera_loop) so lane-follow pulses are not chopped into tiny bursts.
DEFAULT_LIDAR_TURN_DUTY = 0.72
DEFAULT_LIDAR_TURN_PERIOD = 0.55


@dataclass
class LidarSnapshot:
    dL: float = math.inf
    dC: float = math.inf
    dR: float = math.inf
    ok: bool = False
    stamp: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IGVC integrated camera/lidar controller")
    ap.add_argument("--port", default=cam.PORT, help="TM4C serial port")
    ap.add_argument("--baud", type=int, default=cam.BAUD, help="TM4C serial baud")
    ap.add_argument("--right-index", type=int, default=cam.RIGHT_CAM_INDEX)
    ap.add_argument("--left-index", type=int, default=cam.LEFT_CAM_INDEX)
    ap.add_argument("--left-lane-target", type=float, default=cam.LEFT_LANE_TARGET_RATIO_DEFAULT)
    ap.add_argument("--right-lane-target", type=float, default=cam.RIGHT_LANE_TARGET_RATIO_DEFAULT)
    ap.add_argument("--http-preview-port", type=int, default=0, metavar="PORT")
    ap.add_argument("--http-bind", default="0.0.0.0")
    ap.add_argument("--center-deadband", type=float, default=cam.CENTER_DEADBAND)
    ap.add_argument("--dual-u-thresh", type=float, default=cam.DUAL_U_THRESH)
    ap.add_argument("--single-left-danger", type=float, default=cam.SINGLE_LEFT_DANGER_RATIO)
    ap.add_argument("--single-right-danger", type=float, default=cam.SINGLE_RIGHT_DANGER_RATIO)
    ap.add_argument(
        "--blind-pulse-turn-sec",
        type=float,
        default=DEFAULT_BLIND_PULSE_TURN_SEC,
        help="Seconds per turn phase when one line visible (lane blind pulse).",
    )
    ap.add_argument(
        "--blind-pulse-wait-sec",
        type=float,
        default=DEFAULT_BLIND_PULSE_WAIT_SEC,
        help="Seconds of Stop between turn phases (lane blind pulse).",
    )
    ap.add_argument("--blind-turn-pulse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--invert-steer", action="store_true")
    ap.add_argument("--lane-detector", choices=("fitline", "histogram"), default="fitline")
    ap.add_argument("--turn-left-enter", type=float, default=cam.TURN_LEFT_ENTER_RATIO_DEFAULT)
    ap.add_argument("--turn-left-exit", type=float, default=cam.TURN_LEFT_EXIT_RATIO_DEFAULT)
    ap.add_argument("--cam-width", type=int, default=cam.CAM_FRAME_WIDTH)
    ap.add_argument("--cam-height", type=int, default=cam.CAM_FRAME_HEIGHT)
    ap.add_argument("--cam-fps", type=int, default=cam.CAM_FRAME_FPS)
    ap.add_argument("--hsv-v-min", type=int, default=cam.HSV_V_MIN)
    ap.add_argument("--hls-l-min", type=int, default=cam.PART3_HLS_L_MIN)
    ap.add_argument("--hls-s-max", type=int, default=cam.PART3_HLS_S_MAX)
    ap.add_argument("--green-line-thickness", type=int, default=5)
    ap.add_argument("--target-goal-band", type=float, default=0.0)

    ap.add_argument("--lidar-port", default="/dev/ttyUSB0")
    ap.add_argument("--lidar-baud", type=int, default=1_000_000)
    ap.add_argument("--lidar-timeout", type=float, default=1.0)
    ap.add_argument("--lidar-front-deg", type=float, default=0.0)
    ap.add_argument("--lidar-angle-sign", type=int, default=-1, choices=(-1, 1))
    ap.add_argument(
        "--lidar-gentle-turn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PWM-style Left/Right pacing for lidar pivots. In 'all' mode, lane-follow uses full "
        "pulses; pacing runs only when lidar overrides steering.",
    )
    ap.add_argument(
        "--lidar-turn-duty",
        type=float,
        default=DEFAULT_LIDAR_TURN_DUTY,
        help="Fraction of each gentle-turn cycle on Left/Right (0..1). In 'all' mode, gentle PWM "
        "applies only when lidar overrides the camera command.",
    )
    ap.add_argument(
        "--lidar-turn-period",
        type=float,
        default=DEFAULT_LIDAR_TURN_PERIOD,
        help="Seconds per full gentle-turn cycle (on+off). Larger = longer on/off segments.",
    )
    ap.add_argument(
        "--lidar-turn-off-cmd",
        choices=("stop", "forward"),
        default="stop",
        help="Command during the 'off' part of a gentle turn: stop (safest) or forward (small creep).",
    )
    return ap


class GentleTurnPacer:
    """PWM-style pacing for lidar pivot commands (firmware only has one Half turn speed)."""

    def __init__(self, enabled: bool, duty: float, period: float, off_cmd: str):
        self.enabled = enabled
        self.duty = float(duty)
        self.period = float(period)
        self.off_cmd = off_cmd

    def pace(self, cmd: str, now_mono: float) -> str:
        if not self.enabled:
            return cmd
        if cmd not in (cam.CMD_LEFT, cam.CMD_RIGHT):
            return cmd
        d = max(0.0, min(1.0, self.duty))
        if d <= 0.0:
            return self.off_cmd
        if d >= 1.0:
            return cmd
        per = max(0.08, self.period)
        if (now_mono % per) / per < d:
            return cmd
        return self.off_cmd


def _rel_to_front(a_abs: float, angle_sign: int, front_deg: float) -> float:
    ang = (a_abs - front_deg + 180.0) % 360.0 - 180.0
    return angle_sign * ang


def lidar_worker(shared, lidar_port: str, lidar_baud: int, lidar_timeout: float, front_deg: float, angle_sign: int):
    try: 
        from rplidar import RPLidar
    except Exception:
        shared["ok"] = False
        return

    look_angles = (-50.0, 0.0, 50.0)
    ang_tol = 10.0
    sectors = {"FR": (-60.0, -10.0), "F": (-15.0, 15.0), "FL": (10.0, 60.0)}
    lidar = None
    bad_streak = 0
    bad_limit = 8
    try:
        lidar = RPLidar(lidar_port, baudrate=lidar_baud, timeout=lidar_timeout)
        time.sleep(0.1)
        try:
            lidar.start_motor()
        except Exception:
            pass
        scans = lidar.iter_scans(max_buf_meas=16384, min_len=30)
        while True:
            try:
                scan = next(scans)
                bad_streak = 0
            except Exception:
                bad_streak += 1
                if bad_streak < bad_limit:
                    continue
                bad_streak = 0
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except Exception:
                    pass
                time.sleep(0.25)
                lidar = RPLidar(lidar_port, baudrate=lidar_baud, timeout=lidar_timeout)
                try:
                    lidar.start_motor()
                except Exception:
                    pass
                scans = lidar.iter_scans(max_buf_meas=16384, min_len=30)
                continue

            bins = {-50.0: math.inf, 0.0: math.inf, 50.0: math.inf}
            sec = {"FR": math.inf, "F": math.inf, "FL": math.inf}
            for _, ang, dist in scan:
                if dist <= 0:
                    continue
                rel = _rel_to_front(ang, angle_sign, front_deg)
                for tgt in look_angles:
                    if abs(rel - tgt) <= ang_tol and dist < bins[tgt]:
                        bins[tgt] = dist
                for name, (lo, hi) in sectors.items():
                    if lo <= rel <= hi and dist < sec[name]:
                        sec[name] = dist

            dR = min(bins[-50.0], sec["FR"])
            dC = min(bins[0.0], sec["F"])
            dL = min(bins[50.0], sec["FL"])

            shared["dR"] = float(dR)
            shared["dC"] = float(dC)
            shared["dL"] = float(dL)
            shared["stamp"] = time.time()
            shared["ok"] = not (math.isinf(dL) and math.isinf(dC) and math.isinf(dR))
    except Exception:
        shared["ok"] = False
    finally:
        if lidar is not None:
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except Exception:
                pass


def pick_mode() -> str:
    while True:
        s = input("\nChoose mode [camera only | lidar only | all | quit]: ").strip().lower()
        if s in ("camera only", "camera", "lidar only", "lidar", "all", "quit", "q"):
            return s
        print("Invalid mode.")


def begin_or_back() -> str:
    while True:
        s = input("Type 'begin' to start or 'back' to choose mode again: ").strip().lower()
        if s in ("begin", "back"):
            return s
        print("Please type begin or back.")


def get_lidar_snapshot(shared) -> LidarSnapshot:
    return LidarSnapshot(
        dL=float(shared.get("dL", math.inf)),
        dC=float(shared.get("dC", math.inf)),
        dR=float(shared.get("dR", math.inf)),
        ok=bool(shared.get("ok", False)),
        stamp=float(shared.get("stamp", 0.0)),
    )


def lidar_only_cmd(ls: LidarSnapshot) -> str:
    if not ls.ok:
        return cam.CMD_STOP
    if min(ls.dL, ls.dC, ls.dR) < NEAR_OBS_TRIP_MM:
        return "Backward Half"
    openL = ls.dL >= OPEN_THRESH_MM
    openC = ls.dC >= OPEN_THRESH_MM
    openR = ls.dR >= OPEN_THRESH_MM
    if openC:
        return cam.CMD_FWD
    if openL and not openR:
        return cam.CMD_LEFT
    if openR and not openL:
        return cam.CMD_RIGHT
    return cam.CMD_STOP


class LidarOnlyController:
    def __init__(self):
        self.state = "IDLE"
        self.clear_targets = set()
        self.pivot_clear_thresh = OPEN_THRESH_MM
        self.near_target: Optional[str] = None

    def decide(self, ls: LidarSnapshot) -> str:
        if not ls.ok:
            self.state = "STOP_NO_DATA"
            return cam.CMD_STOP

        dL, dC, dR = ls.dL, ls.dC, ls.dR
        openL = dL >= OPEN_THRESH_MM
        openC = dC >= OPEN_THRESH_MM
        openR = dR >= OPEN_THRESH_MM
        all_blocked = (not openL) and (not openC) and (not openR)

        if self.state != "BACKOFF_NEAR":
            cands = []
            if dR < NEAR_OBS_TRIP_MM:
                cands.append(("R", dR))
            if dC < NEAR_OBS_TRIP_MM:
                cands.append(("C", dC))
            if dL < NEAR_OBS_TRIP_MM:
                cands.append(("L", dL))
            if cands:
                cands.sort(key=lambda x: x[1])
                self.near_target = cands[0][0]
                self.state = "BACKOFF_NEAR"
                self.clear_targets = set()
                return "Backward Half"
        else:
            val = {"R": dR, "C": dC, "L": dL}[self.near_target]
            if val < NEAR_OBS_RELEASE_MM:
                return "Backward Half"
            self.state = "IDLE"
            self.clear_targets = set()
            self.near_target = None
            self.pivot_clear_thresh = POST_NEAR_CLEAR_MM

        if self.state == "BACKOFF":
            if max(dL, dC, dR) >= BACKOFF_CLEAR_ANY_MM:
                self.state = "IDLE"
                self.clear_targets = set()
                return cam.CMD_STOP
            return "Backward Half"

        if self.state in ("PIVOT_L", "PIVOT_R"):
            if all_blocked:
                self.state = "BACKOFF"
                self.clear_targets = set()
                return "Backward Half"
            needR = ("R" in self.clear_targets) and (dR < self.pivot_clear_thresh)
            needC = ("C" in self.clear_targets) and (dC < self.pivot_clear_thresh)
            needL = ("L" in self.clear_targets) and (dL < self.pivot_clear_thresh)
            if needR or needC or needL:
                return cam.CMD_LEFT if self.state == "PIVOT_L" else cam.CMD_RIGHT
            self.state = "IDLE"
            self.clear_targets = set()
            return cam.CMD_STOP

        if openR and openC and openL:
            self.state = "FWD"
            self.pivot_clear_thresh = OPEN_THRESH_MM
            return cam.CMD_FWD

        if not openC:
            side_open = []
            if openR:
                side_open.append(("R", dR))
            if openL:
                side_open.append(("L", dL))
            if side_open:
                side_open.sort(key=lambda x: x[1], reverse=True)
                best = side_open[0][0]
                if best == "L":
                    self.state, self.clear_targets = "PIVOT_L", {"C", "R"}
                    return cam.CMD_LEFT
                self.state, self.clear_targets = "PIVOT_R", {"C", "L"}
                return cam.CMD_RIGHT
            self.state = "BACKOFF"
            self.clear_targets = set()
            return "Backward Half"

        if (dR < OPEN_THRESH_MM) and (dL >= OPEN_THRESH_MM):
            self.state, self.clear_targets = "PIVOT_L", {"R"}
            return cam.CMD_LEFT
        if (dL < OPEN_THRESH_MM) and (dR >= OPEN_THRESH_MM):
            self.state, self.clear_targets = "PIVOT_R", {"L"}
            return cam.CMD_RIGHT
        if (dL < OPEN_THRESH_MM) and (dR < OPEN_THRESH_MM):
            if dL >= dR:
                self.state, self.clear_targets = "PIVOT_L", {"L", "R"}
                return cam.CMD_LEFT
            self.state, self.clear_targets = "PIVOT_R", {"L", "R"}
            return cam.CMD_RIGHT
        return cam.CMD_FWD


def lidar_override_cmd(base_cmd: str, ls: LidarSnapshot) -> str:
    if not ls.ok:
        return base_cmd
    if min(ls.dL, ls.dC, ls.dR) < NEAR_OBS_TRIP_MM:
        return "Backward Half"
    if ls.dC < OPEN_THRESH_MM:
        if ls.dL > ls.dR:
            return cam.CMD_LEFT
        if ls.dR > ls.dL:
            return cam.CMD_RIGHT
        # Ambiguous tie (left ~= right) used to force Stop, which made the
        # integrated stack freeze whenever camera lost both lines. Keep the
        # camera command so lane-reacquisition behavior can continue.
        return base_cmd
    return base_cmd


def run_camera_loop(
    args,
    lidar_shared=None,
    wait_for_begin: bool = False,
    lidar_turn_pacer: Optional[GentleTurnPacer] = None,
):
    ser = None
    mc = None
    caps = {}
    http_server = None
    preview_state = cam.MjpegPreviewState()
    st = cam.ControllerState()
    scan_indices = [0, 1, 2, 3, 4, 5, 6]
    rx_buf = ""
    enc_rpm_l = None
    enc_rpm_r = None
    enc_mph = None
    try:
        ser = cam.open_serial(args.port, args.baud)
        mc = cam.MotorCommander(ser)
        mc.send(cam.CMD_STOP, force=True)
        time.sleep(0.2)

        caps, used_idxs = cam.initialize_dual_lane_cameras(args, scan_indices)
        if caps is None:
            raise RuntimeError("Could not open both cameras.")
        print(f"[CAM] LEFT={used_idxs['LEFT']} RIGHT={used_idxs['RIGHT']}")
        http_server = cam.start_mjpeg_server(preview_state, args.http_bind, args.http_preview_port)
        if args.http_preview_port > 0:
            # Ensure /stream has a valid first frame even before camera warmup succeeds.
            boot = np.zeros((270, 480, 3), dtype=np.uint8)
            cv2.putText(
                boot,
                "Starting camera preview...",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok_boot, jpg_boot = cv2.imencode(".jpg", boot, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok_boot:
                preview_state.set_jpeg(jpg_boot.tobytes())

        st.line_left_ema = None
        st.line_right_ema = None
        st.center_prev_e = 0.0
        st.last_lateral_cmd = None
        st.blind_pulse_until = 0.0
        st.blind_in_turn_phase = True
        frame_cache_left = None
        frame_cache_right = None
        stale_left = stale_right = dual_miss_streak = 0
        prev_t = time.time()

        # Prime preview immediately so the website is live before motion.
        for _ in range(40):
            ret_r, frame_right_new = caps["RIGHT"].read()
            ret_l, frame_left_new = caps["LEFT"].read()
            if ret_r and ret_l:
                seed_left = frame_left_new
                seed_right = frame_right_new
                if cam.MIRROR_FRAME:
                    seed_left = cv2.flip(seed_left, 1)
                    seed_right = cv2.flip(seed_right, 1)
                seed_stack = cam.build_dual_cam_debug_panel(
                    seed_left,
                    cv2.cvtColor(cam.white_mask(seed_left), cv2.COLOR_GRAY2BGR),
                    seed_left,
                    "LEFT | preview warmup",
                    seed_right,
                    cv2.cvtColor(cam.white_mask(seed_right), cv2.COLOR_GRAY2BGR),
                    seed_right,
                    "RIGHT | preview warmup",
                )
                if args.http_preview_port > 0:
                    ok, jpg = cv2.imencode(".jpg", seed_stack, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ok:
                        preview_state.set_jpeg(jpg.tobytes())
                break
            time.sleep(0.01)

        if wait_for_begin:
            if args.http_preview_port > 0:
                print(f"[HTTP] Preview ready at http://<jetson-ip>:{args.http_preview_port}/")
            while True:
                user_cmd = input("Type 'begin' to move or 'back' to mode menu: ").strip().lower()
                if user_cmd == "begin":
                    break
                if user_cmd == "back":
                    cam.force_stop_motors(mc, ser)
                    return "back"
                print("[INFO] Please type begin or back.")

        while True:
            if st.read_phase % 2 == 0:
                ret_r, r_new = caps["RIGHT"].read()
                ret_l, l_new = caps["LEFT"].read()
            else:
                ret_l, l_new = caps["LEFT"].read()
                ret_r, r_new = caps["RIGHT"].read()
            st.read_phase += 1
            if ret_r:
                frame_cache_right = r_new
                stale_right = 0
            else:
                stale_right += 1
            if ret_l:
                frame_cache_left = l_new
                stale_left = 0
            else:
                stale_left += 1
            dual_miss_streak = dual_miss_streak + 1 if (not ret_r and not ret_l) else 0
            if frame_cache_left is None or frame_cache_right is None:
                mc.send(cam.CMD_STOP)
                time.sleep(0.01)
                continue
            if stale_left > cam.MAX_STALE_FRAMES_PER_CAM or stale_right > cam.MAX_STALE_FRAMES_PER_CAM or dual_miss_streak > cam.MAX_CONSECUTIVE_DUAL_MISS:
                mc.send(cam.CMD_STOP)
                break

            frame_left = frame_cache_left.copy()
            frame_right = frame_cache_right.copy()
            if cam.MIRROR_FRAME:
                frame_left = cv2.flip(frame_left, 1)
                frame_right = cv2.flip(frame_right, 1)

            left_overlay, left_mask_dbg, line_L, _, det_L = cam.process_frame(
                frame_left, args.lane_detector, float(args.left_lane_target),
                float(args.turn_left_enter), float(args.turn_left_exit),
                hsv_v_min=int(args.hsv_v_min), hls_l_min=int(args.hls_l_min), hls_s_max=int(args.hls_s_max),
                lane_roi_side="left", green_line_thickness=int(args.green_line_thickness),
                target_goal_band_ratio=float(args.target_goal_band),
            )
            right_overlay, right_mask_dbg, line_R, _, det_R = cam.process_frame(
                frame_right, args.lane_detector, float(args.right_lane_target),
                float(args.turn_left_enter), float(args.turn_left_exit),
                hsv_v_min=int(args.hsv_v_min), hls_l_min=int(args.hls_l_min), hls_s_max=int(args.hls_s_max),
                lane_roi_side="right", green_line_thickness=int(args.green_line_thickness),
                target_goal_band_ratio=float(args.target_goal_band),
            )
            wL = frame_left.shape[1]
            wR = frame_right.shape[1]
            st.line_left_ema = cam.smooth_line_x_ema(st.line_left_ema, line_L, wL)
            st.line_right_ema = cam.smooth_line_x_ema(st.line_right_ema, line_R, wR)
            nL = st.line_left_ema / wL if st.line_left_ema is not None else None
            nR = st.line_right_ema / wR if st.line_right_ema is not None else None
            vis = cam.classify_lane_visibility(line_L, line_R)
            if vis == "BOTH" and nL is not None and nR is not None:
                e = cam.center_error_both(nL, nR, float(args.left_lane_target), float(args.right_lane_target))
                de = e - st.center_prev_e
                st.center_prev_e = e
            else:
                e = 0.0
                de = 0.0
            cmd, logic = cam.choose_steering_dual_center(
                vis, e, de, nL, nR,
                float(args.single_left_danger), float(args.single_right_danger),
                float(args.center_deadband), float(args.dual_u_thresh),
            )
            if vis == "NONE":
                if st.last_lateral_cmd in (cam.CMD_LEFT, cam.CMD_RIGHT):
                    cmd = st.last_lateral_cmd
            elif cmd in (cam.CMD_LEFT, cam.CMD_RIGHT):
                st.last_lateral_cmd = cmd
            cmd, logic = cam.apply_blind_turn_pulse(
                st, vis, cmd, logic, time.monotonic(),
                float(args.blind_pulse_turn_sec), float(args.blind_pulse_wait_sec),
                bool(args.blind_turn_pulse) and float(args.blind_pulse_turn_sec) > 0.0,
            )

            if lidar_shared is not None:
                lsnap = get_lidar_snapshot(lidar_shared)
                cmd_after_camera = cmd
                cmd = lidar_override_cmd(cmd_after_camera, lsnap)
                # Do not stack GentleTurnPacer on lane-follow Left/Right: that produced tiny
                # effective pulses (duty × period) on top of blind-pulse timing. Pace only when
                # lidar actually overrides the camera command.
                if lidar_turn_pacer is not None and cmd != cmd_after_camera:
                    cmd = lidar_turn_pacer.pace(cmd, time.monotonic())

            cmd_motor = cam.apply_steer_inversion(cmd, args.invert_steer)
            mc.send(cmd_motor)

            # Read encoder telemetry emitted by TM4C over UART0.
            if ser is not None:
                try:
                    waiting = int(getattr(ser, "in_waiting", 0))
                    if waiting > 0:
                        raw = ser.read(waiting).decode("ascii", errors="ignore")
                        rx_buf += raw
                        if len(rx_buf) > 2048:
                            rx_buf = rx_buf[-2048:]
                        while "\n" in rx_buf:
                            line, rx_buf = rx_buf.split("\n", 1)
                            m = ENC_RE.search(line.strip())
                            if m:
                                enc_rpm_l = float(m.group(1))
                                enc_rpm_r = float(m.group(2))
                                enc_mph = float(m.group(3))
                except Exception:
                    pass

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_t)
            prev_t = now
            enc_s = ""
            if enc_mph is not None and enc_rpm_l is not None and enc_rpm_r is not None:
                enc_s = f" mph={enc_mph:.2f} rpmL={enc_rpm_l:.1f} rpmR={enc_rpm_r:.1f}"
            print(f"[TELEM] cmd={cmd_motor} vis={vis} {logic} detL={det_L} detR={det_R} fps={fps:.1f}{enc_s}")

            right_caption = f"RIGHT | {det_R} | {cmd_motor}"
            if enc_mph is not None:
                right_caption += f" | mph={enc_mph:.2f}"
            stack = cam.build_dual_cam_debug_panel(
                frame_left, left_mask_dbg, left_overlay, f"LEFT | {det_L}",
                frame_right, right_mask_dbg, right_overlay, right_caption,
            )
            if args.http_preview_port > 0:
                ok, jpg = cv2.imencode(".jpg", stack, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    preview_state.set_jpeg(jpg.tobytes())
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            cam.force_stop_motors(mc, ser)
        except Exception:
            pass
        for c in caps.values():
            try:
                if c is not None:
                    c.release()
            except Exception:
                pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        if http_server is not None:
            try:
                http_server.shutdown()
                http_server.server_close()
            except Exception:
                pass
        time.sleep(0.1)
    return "done"


def run_lidar_only(args, lidar_shared, lidar_turn_pacer: Optional[GentleTurnPacer] = None):
    ser = None
    mc = None
    ctrl = LidarOnlyController()
    try:
        ser = cam.open_serial(args.port, args.baud)
        mc = cam.MotorCommander(ser)
        mc.send(cam.CMD_STOP, force=True)
        print("[LIDAR] Running lidar-only control.")
        if lidar_turn_pacer is not None and lidar_turn_pacer.enabled:
            print(
                f"[LIDAR] Gentle turn ON: duty={args.lidar_turn_duty} "
                f"period={args.lidar_turn_period}s off={args.lidar_turn_off_cmd}"
            )
        while True:
            ls = get_lidar_snapshot(lidar_shared)
            cmd = ctrl.decide(ls)
            if lidar_turn_pacer is not None:
                cmd = lidar_turn_pacer.pace(cmd, time.monotonic())
            mc.send(cmd)
            print(
                f"[LIDAR] dL={int(ls.dL) if ls.dL < math.inf else -1} "
                f"dC={int(ls.dC) if ls.dC < math.inf else -1} "
                f"dR={int(ls.dR) if ls.dR < math.inf else -1} "
                f"ok={ls.ok} state={ctrl.state} cmd={cmd}"
            )
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cam.force_stop_motors(mc, ser)
        if ser is not None:
            ser.close()


def main():
    mp.set_start_method("spawn", force=True)
    args = build_parser().parse_args()
    args.green_line_thickness = max(1, min(41, int(args.green_line_thickness)))
    lo, hi = sorted((args.turn_left_enter, args.turn_left_exit))
    args.turn_left_exit = lo
    args.turn_left_enter = hi if hi > lo + 1e-6 else lo + 0.06
    turn_off = cam.CMD_STOP if args.lidar_turn_off_cmd == "stop" else cam.CMD_FWD
    lidar_turn_pacer = GentleTurnPacer(
        bool(args.lidar_gentle_turn),
        float(args.lidar_turn_duty),
        float(args.lidar_turn_period),
        turn_off,
    )

    while True:
        mode = pick_mode()
        if mode in ("quit", "q"):
            print("Exiting.")
            return
        if mode in ("camera only", "camera"):
            status = run_camera_loop(args, lidar_shared=None, wait_for_begin=True, lidar_turn_pacer=None)
            if status == "back":
                continue
            continue

        if mode in ("lidar only", "lidar"):
            if begin_or_back() == "back":
                continue
            manager = mp.Manager()
            lidar_shared = manager.dict({"dL": math.inf, "dC": math.inf, "dR": math.inf, "ok": False, "stamp": 0.0})
            lidar_proc = mp.Process(
                target=lidar_worker,
                args=(lidar_shared, args.lidar_port, args.lidar_baud, args.lidar_timeout, args.lidar_front_deg, args.lidar_angle_sign),
                daemon=True,
            )
            lidar_proc.start()
            time.sleep(0.4)
            try:
                deadline = time.time() + 3.0
                while time.time() < deadline and not bool(lidar_shared.get("ok", False)):
                    time.sleep(0.05)
                if not bool(lidar_shared.get("ok", False)):
                    print("[WARN] LIDAR has not produced valid data yet. Check port, mount, and --lidar-angle-sign.")
                run_lidar_only(args, lidar_shared, lidar_turn_pacer)
            finally:
                if lidar_proc.is_alive():
                    lidar_proc.terminate()
                    lidar_proc.join(timeout=1.0)
            continue

        # all mode
        manager = mp.Manager()
        lidar_shared = manager.dict({"dL": math.inf, "dC": math.inf, "dR": math.inf, "ok": False, "stamp": 0.0})
        lidar_proc = mp.Process(
            target=lidar_worker,
            args=(lidar_shared, args.lidar_port, args.lidar_baud, args.lidar_timeout, args.lidar_front_deg, args.lidar_angle_sign),
            daemon=True,
        )
        lidar_proc.start()
        time.sleep(0.4)
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline and not bool(lidar_shared.get("ok", False)):
                time.sleep(0.05)
            if not bool(lidar_shared.get("ok", False)):
                print("[WARN] LIDAR has not produced valid data yet. Check port, mount, and --lidar-angle-sign.")
            status = run_camera_loop(
                args,
                lidar_shared=lidar_shared,
                wait_for_begin=True,
                lidar_turn_pacer=lidar_turn_pacer,
            )
            if status == "back":
                continue
        finally:
            if lidar_proc.is_alive():
                lidar_proc.terminate()
                lidar_proc.join(timeout=1.0)


if __name__ == "__main__":
    main()


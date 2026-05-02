#!/usr/bin/env python3
"""
GPS live plot served over HTTP (MJPEG).

Reads NMEA from a u-blox GPS (or any NMEA serial source), renders a live
matplotlib plot of the track, and serves it as an MJPEG stream so you can
view it in a browser on your laptop via SSH port-forwarding (works with
Cursor Remote-SSH's auto-forwarding too -- same pattern as Camera_Only.py's
--http-preview-port).

On the Jetson:
    python3 gps_http_plot.py --http-port 8766

On your laptop:
    Open Chrome / Firefox to
        http://localhost:8766/
    (Cursor Remote-SSH auto-forwards the port. If using plain ssh, run:
        ssh -L 8766:127.0.0.1:8766 user@jetson
     and then open the same URL.)

The existing laptop-side gps_track_plot.py is left untouched and still
works for direct matplotlib-window testing on Windows.
"""

from __future__ import annotations

import argparse
import glob
import io
import math
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Optional

# Force headless matplotlib backend BEFORE importing pyplot so this runs
# cleanly on a Jetson with no display attached.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import serial  # noqa: E402


# =========================
# Config
# =========================

MAX_POINTS = 500
WINDOW_METERS = 30.0          # visible area around start
SPEED_MOVE_KNOTS = 0.3        # treat as "moving" if speed >= this (knots)

RENDER_INTERVAL_S = 0.2       # ~5 FPS plot renders
JPEG_QUALITY = 80


# =========================
# NMEA helpers
# =========================

def dm_to_deg(dm: float) -> float:
    """Convert ddmm.mmmm (or dddmm.mmmm) to decimal degrees."""
    dd = int(dm / 100.0)
    mm = dm - dd * 100.0
    return dd + mm / 60.0


# =========================
# Shared state (GPS reader thread <-> render thread)
# =========================

class GpsState:
    """Thread-safe container for the latest parsed GPS state and trail."""

    def __init__(self):
        self.lock = threading.Lock()
        self.trail_lats: list[float] = []
        self.trail_lons: list[float] = []
        self.last_lat: Optional[float] = None
        self.last_lon: Optional[float] = None
        self.last_alt: float = 0.0
        self.last_sats: int = 0
        self.last_speed_knots: float = 0.0
        self.center_lat: Optional[float] = None
        self.center_lon: Optional[float] = None
        self.fix_valid: bool = False
        self.last_nmea_time: float = 0.0

    def snapshot(self):
        with self.lock:
            return {
                "trail_lats": list(self.trail_lats),
                "trail_lons": list(self.trail_lons),
                "last_lat": self.last_lat,
                "last_lon": self.last_lon,
                "last_alt": self.last_alt,
                "last_sats": self.last_sats,
                "last_speed_knots": self.last_speed_knots,
                "center_lat": self.center_lat,
                "center_lon": self.center_lon,
                "fix_valid": self.fix_valid,
                "last_nmea_time": self.last_nmea_time,
            }


# =========================
# GPS reader thread
# =========================

def _open_serial(port: str, baud: int) -> serial.Serial:
    return serial.Serial(port=port, baudrate=baud, timeout=0.5)


def run_gps_reader(port: str, baud: int,
                   state: GpsState, stop_event: threading.Event):
    """
    Continuously read NMEA lines and update shared state.

    Resilient to the CDC-ACM "device reports readiness to read but returned no
    data" quirk that u-blox devices often trigger under USB bus pressure on
    Linux/Jetson. We close+reopen the port when this happens instead of
    spinning on the error.
    """
    ser: Optional[serial.Serial] = None
    consecutive_errors = 0
    last_error_msg = ""

    def close_ser():
        nonlocal ser
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
            ser = None

    while not stop_event.is_set():
        # Make sure the port is open
        if ser is None:
            try:
                ser = _open_serial(port, baud)
                if consecutive_errors > 0:
                    print(f"[GPS] Reopened {port} after {consecutive_errors} error(s).")
                consecutive_errors = 0
                last_error_msg = ""
            except Exception as e:
                msg = str(e)
                if msg != last_error_msg:
                    print(f"[GPS] Could not open {port}: {msg}")
                    last_error_msg = msg
                if stop_event.wait(timeout=0.5):
                    break
                continue

        try:
            line_raw = ser.readline().decode(errors="ignore").strip()
            # Got through a read without exception -> consider us "healthy"
            if consecutive_errors > 0:
                consecutive_errors = 0
                last_error_msg = ""
        except Exception as e:
            msg = str(e)
            consecutive_errors += 1
            # Suppress identical repeats; just print the first and then
            # occasional updates so the log isn't swamped.
            if msg != last_error_msg or consecutive_errors % 20 == 0:
                print(f"[GPS] read error ({consecutive_errors}): {msg}")
                last_error_msg = msg
            # Hard-reset the serial port on the CDC-ACM false-ready quirk
            # (message contains "readiness" or "no data"), or after repeated
            # failures in a row.
            if ("readiness" in msg) or ("no data" in msg) or consecutive_errors >= 3:
                close_ser()
                # brief backoff before reopen
                if stop_event.wait(timeout=0.3):
                    break
            else:
                if stop_event.wait(timeout=0.1):
                    break
            continue

        if not line_raw:
            continue

        now = time.time()

        # RMC -> speed over ground
        if "RMC" in line_raw:
            parts = line_raw.split(",")
            if len(parts) >= 8:
                try:
                    status = parts[2]
                    if status == "A" and parts[7]:
                        with state.lock:
                            state.last_speed_knots = float(parts[7])
                            state.last_nmea_time = now
                    else:
                        with state.lock:
                            state.last_speed_knots = 0.0
                            state.last_nmea_time = now
                except ValueError:
                    pass
            continue

        # GGA -> position
        if "GGA" in line_raw:
            parts = line_raw.split(",")
            if len(parts) < 10:
                continue
            try:
                lat_dm = float(parts[2]) if parts[2] else None
                ns = parts[3]
                lon_dm = float(parts[4]) if parts[4] else None
                ew = parts[5]
                fix = int(parts[6]) if parts[6] else 0
                sats = int(parts[7]) if parts[7] else 0
                alt = float(parts[9]) if parts[9] else 0.0
            except ValueError:
                continue

            if fix < 1 or lat_dm is None or lon_dm is None:
                with state.lock:
                    state.fix_valid = False
                    state.last_sats = sats
                    state.last_nmea_time = now
                continue

            lat = dm_to_deg(lat_dm)
            lon = dm_to_deg(lon_dm)
            if ns == "S":
                lat = -lat
            if ew == "W":
                lon = -lon

            with state.lock:
                state.fix_valid = True
                state.last_alt = alt
                state.last_sats = sats
                state.last_nmea_time = now

                if state.last_lat is None or state.last_lon is None:
                    state.last_lat = lat
                    state.last_lon = lon
                    state.trail_lats.append(lat)
                    state.trail_lons.append(lon)
                    state.center_lat = lat
                    state.center_lon = lon
                else:
                    if state.last_speed_knots >= SPEED_MOVE_KNOTS:
                        state.last_lat = lat
                        state.last_lon = lon
                        state.trail_lats.append(lat)
                        state.trail_lons.append(lon)
                        if len(state.trail_lats) > MAX_POINTS:
                            state.trail_lats = state.trail_lats[-MAX_POINTS:]
                            state.trail_lons = state.trail_lons[-MAX_POINTS:]


# =========================
# Rendering thread (matplotlib -> JPEG)
# =========================

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


def run_renderer(state: GpsState, preview: MjpegPreviewState, stop_event: threading.Event):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    track_line, = ax.plot([], [], "-", linewidth=1.5)
    head_point, = ax.plot([], [], "o", markersize=8)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("GPS Track (live, speed-gated)")
    ax.grid(True)
    ax.ticklabel_format(useOffset=False, style="plain")
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        family="monospace", fontsize=9,
                        verticalalignment="top")

    # Placeholder axis ranges until we get a fix
    ax.set_xlim(-0.001, 0.001)
    ax.set_ylim(-0.001, 0.001)

    last_center = (None, None)

    while not stop_event.is_set():
        t0 = time.time()
        snap = state.snapshot()

        trail_lats = snap["trail_lats"]
        trail_lons = snap["trail_lons"]
        last_lat = snap["last_lat"]
        last_lon = snap["last_lon"]
        center_lat = snap["center_lat"]
        center_lon = snap["center_lon"]

        # Set window around first-fix center
        if center_lat is not None and center_lon is not None \
                and last_center != (center_lat, center_lon):
            dlat = WINDOW_METERS / 111000.0
            cos_lat = math.cos(math.radians(center_lat))
            dlon = WINDOW_METERS / (111000.0 * cos_lat if cos_lat != 0 else 1.0)
            ax.set_xlim(center_lon - dlon, center_lon + dlon)
            ax.set_ylim(center_lat - dlat, center_lat + dlat)
            last_center = (center_lat, center_lon)

        track_line.set_data(trail_lons, trail_lats)
        if last_lat is not None and last_lon is not None:
            head_point.set_data([last_lon], [last_lat])
        else:
            head_point.set_data([], [])

        # Status text
        age = time.time() - snap["last_nmea_time"] if snap["last_nmea_time"] else -1
        age_line = f"NMEA age: {age:.1f}s" if age >= 0 else "No NMEA received yet"
        if snap["fix_valid"] and last_lat is not None and last_lon is not None:
            status_line = (
                f"Pts:  {len(trail_lats)}\n"
                f"Lat:  {last_lat:+.6f}\n"
                f"Lon:  {last_lon:+.6f}\n"
                f"Alt:  {snap['last_alt']:.1f} m\n"
                f"Sats: {snap['last_sats']}\n"
                f"Spd:  {snap['last_speed_knots']:.2f} kt\n"
                f"{age_line}"
            )
        else:
            status_line = (
                f"Waiting for GPS fix...\n"
                f"Sats in view: {snap['last_sats']}\n"
                f"{age_line}"
            )
        info_text.set_text(status_line)

        # Render to JPEG bytes
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="jpg",
                        pil_kwargs={"quality": JPEG_QUALITY, "optimize": True})
            preview.set_jpeg(buf.getvalue())
        except Exception as e:
            # Older matplotlib: pil_kwargs may not be supported; fall back.
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="jpg")
                preview.set_jpeg(buf.getvalue())
            except Exception as e2:
                print(f"[RENDER] savefig failed: {e2}")

        # Pace the renderer
        elapsed = time.time() - t0
        sleep_for = max(0.0, RENDER_INTERVAL_S - elapsed)
        if stop_event.wait(timeout=sleep_for):
            break

    plt.close(fig)


# =========================
# HTTP server (MJPEG)  (mirrors Camera_Only.py pattern)
# =========================

def make_mjpeg_handler(preview: MjpegPreviewState):
    boundary = b"--jpgboundary"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/view"):
                body = (
                    "<!DOCTYPE html><html><head><meta charset=utf-8>"
                    "<title>GPS live plot</title></head><body>"
                    "<p>GPS live plot (MJPEG). If accessing via SSH port-forward:"
                    " <code>ssh -L PORT:127.0.0.1:PORT user@jetson</code></p>"
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
                time.sleep(0.05)

        def log_message(self, format, *log_args):
            pass

    return Handler


def start_mjpeg_server(preview: MjpegPreviewState, host: str, port: int):
    if port <= 0:
        return None
    handler = make_mjpeg_handler(preview)
    server = socketserver.ThreadingTCPServer((host, port), handler)
    server.daemon_threads = True
    server.allow_reuse_address = True
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(
        f"[HTTP] GPS MJPEG plot on http://{host}:{port}/  (stream: /stream)\n"
        f"       from PC: ssh -L {port}:127.0.0.1:{port} user@jetson  "
        f"(or use Cursor Remote-SSH auto-forwarding)"
    )
    return server


# =========================
# Device auto-detect
# =========================

def autodetect_gps_port() -> Optional[str]:
    """Try to find a u-blox GPS via /dev/serial/by-id/. Returns None if not found."""
    try:
        candidates = sorted(glob.glob("/dev/serial/by-id/*"))
    except Exception:
        return None
    # Prefer u-blox explicitly
    for p in candidates:
        name = os.path.basename(p).lower()
        if "u-blox" in name or "u_blox" in name or "ublox" in name:
            return p
    return None


# =========================
# Main
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="GPS live plot via HTTP MJPEG (headless, Jetson-friendly)."
    )
    ap.add_argument(
        "--port", default=None,
        help="Serial device (e.g. /dev/ttyACM0, /dev/serial/by-id/...). "
             "If omitted, tries to auto-detect a u-blox device via /dev/serial/by-id/."
    )
    ap.add_argument(
        "--baud", type=int, default=9600,
        help="Serial baud rate (default 9600). u-blox over USB ignores baud but it's fine to set."
    )
    ap.add_argument(
        "--http-port", type=int, default=8766,
        help="HTTP port for MJPEG preview (default 8766). Set to 0 to disable HTTP."
    )
    ap.add_argument(
        "--http-bind", default="0.0.0.0",
        help="Bind address for the HTTP server (default 0.0.0.0 = all interfaces)."
    )
    return ap.parse_args()


def main():
    args = parse_args()

    port = args.port or autodetect_gps_port()
    if not port:
        print("[ERROR] No --port given and could not auto-detect a u-blox device "
              "under /dev/serial/by-id/. Pass --port /dev/ttyACM0 (or the correct path).",
              file=sys.stderr)
        return 2

    print(f"[GPS] Will read from {port} @ {args.baud} baud "
          f"(reader thread handles open/reopen).")

    state = GpsState()
    preview = MjpegPreviewState()
    stop_event = threading.Event()

    reader_th = threading.Thread(
        target=run_gps_reader, args=(port, args.baud, state, stop_event),
        daemon=True,
    )
    renderer_th = threading.Thread(
        target=run_renderer, args=(state, preview, stop_event), daemon=True
    )
    reader_th.start()
    renderer_th.start()

    http_server = start_mjpeg_server(preview, args.http_bind, args.http_port)
    if args.http_port <= 0:
        print("[HTTP] HTTP server disabled (--http-port 0). "
              "Running in data-only mode (no viewer).")

    print("[MAIN] Running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C received, shutting down ...")
    finally:
        stop_event.set()
        if http_server is not None:
            try:
                http_server.shutdown()
                http_server.server_close()
            except Exception:
                pass
        reader_th.join(timeout=1.0)
        renderer_th.join(timeout=2.0)
        print("[MAIN] Clean exit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

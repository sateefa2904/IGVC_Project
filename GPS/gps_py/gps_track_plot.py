import argparse
import serial
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--port", default="COM4")
parser.add_argument("--baud", type=int, default=115200)
args = parser.parse_args()

# ---------- Serial ----------
ser = serial.Serial(
    port=args.port,
    baudrate=args.baud,
    timeout=0.1
)

# ---------- Helpers ----------
def dm_to_deg(dm: float) -> float:
    """Convert ddmm.mmmm (or dddmm.mmmm) to decimal degrees."""
    dd = int(dm / 100.0)
    mm = dm - dd * 100.0
    return dd + mm / 60.0

def distance_m(lat1, lon1, lat2, lon2) -> float:
    """Approx distance in meters between two lat/lon points (Haversine)."""
    R = 6371000.0  # Earth radius [m]
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------- Config ----------
MAX_POINTS = 500
WINDOW_METERS = 30.0          # visible area around start
SPEED_MOVE_KNOTS = 0.3        # treat as "moving" if speed >= this (knots, ~0.15 m/s)

# ---------- Data + plot ----------
trail_lats = []
trail_lons = []

fig, ax = plt.subplots()
track_line, = ax.plot([], [], '-')   # trail
head_point, = ax.plot([], [], 'o')   # current position

ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title("GPS Track (live, speed-gated)")
ax.grid(True)

ax.ticklabel_format(useOffset=False)
ax.ticklabel_format(style='plain')

info_text = ax.text(0.02, 0.95, "Pts: 0", transform=ax.transAxes)

center_lat = None
center_lon = None

# Last known values
last_speed_knots = 0.0
last_lat = None
last_lon = None

def on_key(event):
    """Press 'r' to reset the trail and recenter."""
    global trail_lats, trail_lons, center_lat, center_lon, last_lat, last_lon
    if event.key == 'r':
        trail_lats = []
        trail_lons = []
        center_lat = None
        center_lon = None
        last_lat = None
        last_lon = None
        track_line.set_data([], [])
        head_point.set_data([], [])
        info_text.set_text("Pts: 0")
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

# ---------- Animation callback ----------
def update(frame):
    global trail_lats, trail_lons
    global center_lat, center_lon
    global last_speed_knots, last_lat, last_lon

    latest_lat = None
    latest_lon = None
    latest_alt = None
    latest_sats = None

    # Read all available lines this frame
    while ser.in_waiting > 0:
        line_raw = ser.readline().decode(errors='ignore').strip()
        if not line_raw:
            continue

        # Parse RMC for speed over ground
        if "RMC" in line_raw:
            parts = line_raw.split(',')
            # Expected: $GPRMC, time, status, lat, N/S, lon, E/W, speed_knots, ...
            if len(parts) >= 8:
                try:
                    status = parts[2]  # 'A' = valid
                    if status == 'A' and parts[7]:
                        last_speed_knots = float(parts[7])
                    else:
                        # invalid or missing → treat as 0
                        last_speed_knots = 0.0
                except ValueError:
                    pass
            continue

        # Parse GGA for position
        if "GGA" in line_raw:
            parts = line_raw.split(',')
            if len(parts) < 10:
                continue
            try:
                lat_dm = float(parts[2])
                ns = parts[3]
                lon_dm = float(parts[4])
                ew = parts[5]
                fix = int(parts[6]) if parts[6] else 0
                sats = int(parts[7]) if parts[7] else 0
                alt = float(parts[9]) if parts[9] else 0.0

                if fix < 1:
                    continue

                lat = dm_to_deg(lat_dm)
                lon = dm_to_deg(lon_dm)
                if ns == 'S':
                    lat = -lat
                if ew == 'W':
                    lon = -lon

                latest_lat = lat
                latest_lon = lon
                latest_alt = alt
                latest_sats = sats

            except ValueError:
                continue

    # If we got a valid GGA point this frame
    if latest_lat is not None:
        # First point → always accept as trail start
        if last_lat is None or last_lon is None:
            last_lat = latest_lat
            last_lon = latest_lon
            trail_lats.append(latest_lat)
            trail_lons.append(latest_lon)

            # Initialize fixed view window
            center_lat = latest_lat
            center_lon = latest_lon
            dlat = WINDOW_METERS / 111000.0
            cos_lat = math.cos(math.radians(center_lat))
            dlon = WINDOW_METERS / (111000.0 * cos_lat if cos_lat != 0 else 1.0)
            ax.set_xlim(center_lon - dlon, center_lon + dlon)
            ax.set_ylim(center_lat - dlat, center_lat + dlat)
        else:
            # Decide based on speed
            if last_speed_knots >= SPEED_MOVE_KNOTS:
                # robot is moving → accept new point
                last_lat = latest_lat
                last_lon = latest_lon
                trail_lats.append(latest_lat)
                trail_lons.append(latest_lon)
                if len(trail_lats) > MAX_POINTS:
                    trail_lats = trail_lats[-MAX_POINTS:]
                    trail_lons = trail_lons[-MAX_POINTS:]
            else:
                # robot considered stopped → hold at last accepted position
                latest_lat = last_lat
                latest_lon = last_lon

        # Update plot using last accepted (snapped) point
        track_line.set_data(trail_lons, trail_lats)
        head_point.set_data([last_lon], [last_lat])

        info_text.set_text(
            f"Pts: {len(trail_lats)}  Now: {last_lat:.6f}, {last_lon:.6f}\n"
            f"Alt: {latest_alt:.1f} m  Sats: {latest_sats}  Spd: {last_speed_knots:.2f} kt"
        )

    return track_line, head_point, info_text

ani = FuncAnimation(fig, update, interval=200)  # 5 updates/sec
plt.show()











#C:\Users\rozak\Documents\gps_py
#python gps_track_plot.py --port COM7 --baud 115200

# drive_arrows.py
# Drive TM4C over UART using arrow keys. Press ESC to quit.
import time
import serial
import keyboard  # pip install keyboard

PORT = "COM23"
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
CMD_Manual = "Manual"
STOP_CMD  = "Stop"     # <- add this in your TM4C command parser (strcmp) to stop motors

ARROW_KEYS = ("up", "down", "left", "right")
AUTO_MANUAL_KEYS = ("a", "m")  # 'a' -> Auto, 'm' -> Manual

def open_serial():
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
    # print for debug:
    print("TX:", repr(text + EOL), list(data))
    for b in data:
        ser.write(bytes([b]))
        ser.flush()
        time.sleep(CHAR_DELAY)

def desired_command(keys_down: set[str]) -> str | None:
    # Choose a command based on keys currently held (priority: up, down, left, right)
    if "up" in keys_down:
        return CMD_UP
    if "down" in keys_down:
        return CMD_DOWN
    if "left" in keys_down:
        return CMD_LEFT
    if "right" in keys_down:
        return CMD_RIGHT
    return None  # means STOP

def main():
    print("Opening serial:", PORT, "@", BAUD)
    ser = open_serial()
    print("Ready. Hold arrow keys to drive; release to Stop. Press ESC to quit.")
    print("Also: press 'A' for Auto and 'M' for Manual.")

    keys_down: set[str] = set()
    last_sent: str | None = None

    # Prime with a Stop so we start safe
    send_line_typewriter(ser, STOP_CMD)
    last_sent = STOP_CMD

    try:
        # Use keyboard hook to capture keydown/keyup globally
        def on_event(e: keyboard.KeyboardEvent):
            nonlocal last_sent
            # One-shot Auto/Manual on key press
            if e.event_type == "down" and e.name in AUTO_MANUAL_KEYS:
                if e.name == "a":
                    send_line_typewriter(ser, CMD_AUTO)
                    last_sent = CMD_AUTO
                elif e.name == "m":
                    send_line_typewriter(ser, CMD_Manual)
                    last_sent = CMD_Manual
                return

            if e.name not in ARROW_KEYS and e.name != "esc":
                return
            if e.event_type == "down":
                if e.name == "esc":
                    raise KeyboardInterrupt
                keys_down.add(e.name)
            elif e.event_type == "up":
                if e.name in keys_down:
                    keys_down.remove(e.name)

            # Compute the command we want given current keys held
            cmd = desired_command(keys_down)
            if cmd is None:
                cmd = STOP_CMD

            # Only send when the command actually changes
            if cmd != last_sent:
                send_line_typewriter(ser, cmd)
                last_sent = cmd

        keyboard.hook(on_event)
        keyboard.wait()  # blocks until Ctrl+C or KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nExiting… sending Stop.")
        try:
            send_line_typewriter(ser, STOP_CMD)
        except Exception:
            pass
    finally:
        ser.close()
        keyboard.unhook_all()

if __name__ == "__main__":
    main()

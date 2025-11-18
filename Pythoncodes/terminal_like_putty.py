# terminal_like_putty.py
import time, sys
import serial

PORT = "COM6"
BAUD = 19200
EOL  = "\r\n"        # CRLF like PuTTY/miniterm
DTR  = True
RTS  = True
CHAR_DELAY = 0.01    # 10 ms between bytes (adjust if needed)

print("VERSION: terminal_like_putty v2 (typewriter)")

ser = serial.Serial(PORT, BAUD, bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                    timeout=0.5, write_timeout=1.0,
                    rtscts=False, dsrdtr=False, xonxoff=False)
ser.setDTR(DTR)
ser.setRTS(RTS)

# give the device time just like when you open a terminal before typing
time.sleep(2.0)

def send_line_typewriter(s, line, eol):
    data = (line + eol).encode("ascii", errors="ignore")
    print("TX bytes:", list(data), repr(line + eol))
    for b in data:
        s.write(bytes([b]))
        s.flush()
        time.sleep(CHAR_DELAY)

print(f"Connected to {PORT} @ {BAUD}, DTR={DTR}, RTS={RTS}. Type commands and press Enter. Ctrl+C to exit.")
try:
    while True:
        line = input("> ").strip()
        if not line:
            continue
        send_line_typewriter(ser, line, EOL)
except KeyboardInterrupt:
    print("\nBye.")
finally:
    ser.close()

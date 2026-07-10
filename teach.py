#!/usr/bin/env python3
"""teach_record.py - RoArm-M2-S Teach & Record (Direct Serial, Fixed Camera)
   With live movement detection display."""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "opencv-python",
#     "numpy",
# ]
# ///

import os
import sys

def _ensure_uv():
    if os.environ.get("_UV_SAFE_ENV") == "1":
        return
    os.environ["_UV_SAFE_ENV"] = "1"
    from datetime import datetime, timedelta, timezone
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

# Suppress Qt font warnings
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import json
import time
import math
import threading
import serial
import serial.tools.list_ports
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2


def find_arm_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if "usb" in desc or "ch340" in desc or "cp210" in desc or "ftdi" in desc:
            return p.device
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    if ports:
        return ports[0].device
    return None


class RoArmDirect:
    def __init__(self, port: str = None, baudrate: int = 115200):
        if port is None:
            port = find_arm_port()
        if port is None:
            raise RuntimeError("Kein serieller Port gefunden! --port angeben.")
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=1.0, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()

    def send_cmd(self, cmd: dict) -> str:
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.05)
            response = ""
            deadline = time.time() + 0.5
            while time.time() < deadline:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response = line
                        if '"T":1051' in line or '"T": 1051' in line:
                            return line
                else:
                    time.sleep(0.01)
            return response

    def get_feedback(self) -> dict:
        with self._lock:
            self._ser.reset_input_buffer()
        resp = self.send_cmd({"T": 105})
        if not resp:
            time.sleep(0.1)
            with self._lock:
                while self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if '"T":1051' in line or '"T": 1051' in line:
                        resp = line
                        break
        if not resp:
            return None
        try:
            start = resp.find('{')
            end = resp.rfind('}')
            if start >= 0 and end > start:
                data = json.loads(resp[start:end+1])
                if data.get("T") == 1051 or "b" in data:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def torque_off(self):
        self.send_cmd({"T": 210, "cmd": 0})

    def torque_on(self):
        self.send_cmd({"T": 210, "cmd": 1})

    def move_degrees(self, b, s, e, h, spd=10, acc=10):
        self.send_cmd({"T": 122, "b": b, "s": s, "e": e, "h": h, "spd": spd, "acc": acc})

    def gripper_open(self):
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})

    def gripper_close(self):
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})

    def set_led(self, brightness: int):
        self.send_cmd({"T": 114, "led": brightness})

    def move_init(self):
        self.send_cmd({"T": 100})

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


class TeachRecorder:
    POLL_HZ = 10

    def __init__(self, port=None, camera_index=2, output_dir="teach_recordings",
                 continuous=True, wait_seconds=1.0, move_threshold=1.5):
        self._port = port
        self._camera_index = camera_index
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._continuous = continuous
        self._wait_seconds = wait_seconds
        self._move_threshold = move_threshold  # degrees threshold for movement detection
        self._arm = None
        self._camera = None
        self._recording = False
        self._commands = []
        self._frame_count = 0
        self._session_dir = None
        self._images_dir = None
        self._script_path = None
        self._speed_scale = 1.0
        self._spd = 0
        self._acc = 10
        self._running = False
        self._window_name = "RoArm Teach & Record"
        # Movement detection state
        self._last_pos = None
        self._movement_detected = False
        self._movement_flash_time = 0  # timestamp of last movement detection
        self._movement_delta = {}  # which joints moved and by how much
        self._total_movements = 0
        # Live display log (recent waypoints)
        self._live_log = []  # list of (timestamp, message) for on-screen display
        self._max_log_lines = 8

    def setup(self) -> bool:
        print("=" * 60)
        print("  RoArm-M2-S Teach & Record (Movement Detection)")
        print("=" * 60)

        print("\n  [1/3] Arm verbinden...")
        try:
            self._arm = RoArmDirect(port=self._port)
            print(f"    OK: {self._arm.port}")
        except Exception as e:
            print(f"    FEHLER: {e}")
            return False

        print(f"  [2/3] Kamera {self._camera_index}...")
        self._camera = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            self._camera = cv2.VideoCapture(self._camera_index)
        if self._camera.isOpened():
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(5):
                self._camera.read()
            print(f"    OK: Kamera {self._camera_index}")
        else:
            print(f"    FEHLER: Kamera {self._camera_index} nicht offen!")
            self._camera = None

        print("  [3/3] Torque Lock AUS...")
        self._arm.torque_off()
        time.sleep(0.5)

        fb = self._arm.get_feedback()
        if fb:
            print(f"    Feedback OK: b={fb.get('b',0):.2f} s={fb.get('s',0):.2f} e={fb.get('e',0):.2f} t={fb.get('t',0):.2f}")
        else:
            print("    WARNUNG: Kein Feedback")

        print("\n  Tasten:")
        print("    R       = Aufnahme Start/Stop")
        print("    SPACE   = Manueller Wegpunkt")
        print("    O / C   = Greifer offen / zu")
        print("    1-5     = LED Helligkeit (0=aus)")
        print("    W       = Wartezeit")
        print("    +/-     = Schwellwert anpassen")
        print("    P       = Abspielen")
        print("    Q       = Beenden")
        print(f"\n  Bewegungsschwelle: {self._move_threshold}°\n")
        return True

    def _create_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._output_dir / f"session_{ts}"
        self._images_dir = self._session_dir / "frames"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._script_path = self._session_dir / f"program_{ts}.roarm"
        self._commands = []
        self._frame_count = 0
        self._last_pos = None
        self._total_movements = 0
        self._live_log = []
        print(f"\n  Session: {self._session_dir}")

    def _save_frame(self, frame) -> str:
        if frame is None:
            return ""
        self._frame_count += 1
        filename = f"frame_{self._frame_count:06d}.jpg"
        cv2.imwrite(str(self._images_dir / filename), frame)
        return filename

    def _get_arm_position(self) -> dict:
        fb = self._arm.get_feedback()
        if fb and "b" in fb:
            return {
                "b": round(math.degrees(fb["b"]), 2),
                "s": round(math.degrees(fb["s"]), 2),
                "e": round(math.degrees(fb["e"]), 2),
                "h": round(math.degrees(fb["t"]), 2),
            }
        return None

    def _check_movement(self, pos: dict) -> bool:
        """Check if position has moved beyond threshold from last recorded position."""
        if self._last_pos is None:
            return True  # First position always counts

        self._movement_delta = {}
        moved = False
        for joint in ['b', 's', 'e', 'h']:
            delta = abs(pos[joint] - self._last_pos[joint])
            if delta >= self._move_threshold:
                self._movement_delta[joint] = pos[joint] - self._last_pos[joint]
                moved = True

        return moved

    def _add_log(self, msg: str):
        """Add a message to the live on-screen log."""
        self._live_log.append((time.time(), msg))
        if len(self._live_log) > self._max_log_lines:
            self._live_log.pop(0)

    def _record_waypoint(self, frame=None, force=False):
        pos = self._get_arm_position()
        if pos is None:
            return

        # Movement detection: only record if moved beyond threshold
        if not force and not self._check_movement(pos):
            self._movement_detected = False
            return

        # Movement detected!
        self._movement_detected = True
        self._movement_flash_time = time.time()
        self._total_movements += 1

        # Build movement info string
        if self._movement_delta:
            parts = []
            for joint, delta in self._movement_delta.items():
                direction = "+" if delta > 0 else ""
                parts.append(f"{joint}:{direction}{delta:.1f}°")
            move_info = " | ".join(parts)
        else:
            move_info = "initial position"

        # Record time since recording started
        elapsed = time.time() - self._rec_start_time
        self._commands.append(f"MOVE b={pos['b']} s={pos['s']} e={pos['e']} h={pos['h']} t={elapsed:.3f}")
        if frame is not None:
            fn = self._save_frame(frame)
            if fn:
                self._commands.append(f"FRAME {fn}")

        self._last_pos = pos.copy()

        # Live log and console output
        log_msg = f"#{self._total_movements} [{elapsed:.1f}s] {move_info}"
        self._add_log(log_msg)
        print(f"    ► MOVE #{self._total_movements}: {move_info}")

    def _save_script(self):
        if not self._commands:
            print("    Nichts aufgezeichnet!")
            return
        lines = [
            f"# RoArm-M2-S Teach Recording (Movement Detection)",
            f"# {datetime.now().isoformat()}",
            f"# Frames: {self._frame_count}",
            f"# Movements detected: {self._total_movements}",
            f"# Movement threshold: {self._move_threshold} degrees",
            f"#CONFIG speed_scale={self._speed_scale}",
            f"#CONFIG spd={self._spd}",
            f"#CONFIG acc={self._acc}",
            "",
        ] + self._commands
        with open(self._script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  Gespeichert: {self._script_path}")
        print(f"  {len(self._commands)} Befehle, {self._frame_count} Frames, {self._total_movements} Bewegungen")

    def _draw_overlay(self, disp):
        """Draw the live movement detection overlay on the display frame."""
        now = time.time()
        h, w = disp.shape[:2]

        # Recording status
        if self._recording:
            # Flash effect when movement detected (green flash for 0.3s)
            flash_active = (now - self._movement_flash_time) < 0.3
            if flash_active:
                # Green border flash
                cv2.rectangle(disp, (0, 0), (w-1, h-1), (0, 255, 0), 4)
                status_color = (0, 255, 0)
                status_text = "● MOVE DETECTED"
            else:
                status_color = (0, 0, 255)
                status_text = "● REC"

            cv2.putText(disp, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # Stats line
            cv2.putText(disp, f"Moves:{self._total_movements} | Cmds:{len(self._commands)} | F:{self._frame_count}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Threshold indicator
            cv2.putText(disp, f"Threshold: {self._move_threshold:.1f} deg",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Current position display
            if self._last_pos:
                pos_text = f"Pos: b={self._last_pos['b']:.1f} s={self._last_pos['s']:.1f} e={self._last_pos['e']:.1f} h={self._last_pos['h']:.1f}"
                cv2.putText(disp, pos_text, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

            # Live log panel (bottom-left)
            log_y_start = h - 20 - (len(self._live_log) * 22)
            # Semi-transparent background for log
            if self._live_log:
                overlay = disp.copy()
                cv2.rectangle(overlay, (5, log_y_start - 5), (w - 5, h - 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)

                for i, (log_time, log_msg) in enumerate(self._live_log):
                    age = now - log_time
                    # Fade out old entries
                    alpha = max(0.3, 1.0 - (age / 10.0))
                    color = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
                    y = log_y_start + i * 22
                    cv2.putText(disp, log_msg, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Movement delta indicator (right side, large)
            if flash_active and self._movement_delta:
                delta_y = 140
                for joint, delta in self._movement_delta.items():
                    direction = "↑" if delta > 0 else "↓"
                    bar_color = (0, 255, 0) if delta > 0 else (0, 100, 255)
                    text = f"{joint.upper()} {direction} {abs(delta):.1f}°"
                    cv2.putText(disp, text, (w - 180, delta_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
                    # Visual bar
                    bar_len = min(int(abs(delta) * 3), 100)
                    cv2.rectangle(disp, (w - 185, delta_y + 5), (w - 185 + bar_len, delta_y + 15), bar_color, -1)
                    delta_y += 40

        else:
            cv2.putText(disp, "IDLE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(disp, "Press R to record", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return disp

    def run(self):
        if not self.setup():
            return
        self._running = True
        last_rec = 0
        interval = 1.0 / self.POLL_HZ
        last_key_time = 0

        try:
            while self._running:
                frame = None
                if self._camera:
                    ret, frame = self._camera.read()
                    if not ret:
                        frame = None

                now = time.time()
                if self._recording and self._continuous and now - last_rec >= interval:
                    self._record_waypoint(frame)
                    last_rec = now

                # Display
                if frame is not None:
                    disp = frame.copy()
                    disp = self._draw_overlay(disp)
                    cv2.imshow(self._window_name, disp)
                else:
                    dummy = np.zeros((300, 600, 3), dtype=np.uint8)
                    dummy = self._draw_overlay(dummy)
                    cv2.imshow(self._window_name, dummy)

                key = cv2.waitKey(30) & 0xFF

                # Debounce
                if key != 255:
                    if (time.time() - last_key_time) < 0.3:
                        continue
                    last_key_time = time.time()

                if key == ord('q'):
                    self._running = False
                elif key == ord('r'):
                    if not self._recording:
                        self._create_session()
                        self._rec_start_time = time.time()
                        self._recording = True
                        last_rec = time.time()
                        print("  ● AUFNAHME GESTARTET (Bewegungserkennung aktiv)")
                    else:
                        self._recording = False
                        self._save_script()
                        print("  ■ AUFNAHME GESTOPPT")
                        self._session_dir = None
                elif key == ord(' '):
                    if self._recording:
                        self._record_waypoint(frame, force=True)
                        self._add_log("MANUAL waypoint")
                        print("    + Manueller Waypoint")
                elif key == ord('o'):
                    if self._recording:
                        self._commands.append("GRIPPER OPEN")
                        self._add_log("GRIPPER OPEN")
                    self._arm.gripper_open()
                    print("    Greifer OFFEN")
                elif key == ord('c'):
                    if self._recording:
                        self._commands.append("GRIPPER CLOSE")
                        self._add_log("GRIPPER CLOSE")
                    self._arm.gripper_close()
                    print("    Greifer ZU")
                elif key == ord('0'):
                    if self._recording:
                        self._commands.append("LED 0")
                    self._arm.set_led(0)
                    print("    LED AUS")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    level = (key - ord('0')) * 51
                    if self._recording:
                        self._commands.append(f"LED {level}")
                    self._arm.set_led(level)
                    print(f"    LED {level}")
                elif key == ord('w'):
                    if self._recording:
                        self._commands.append(f"WAIT {self._wait_seconds}")
                        self._add_log(f"WAIT {self._wait_seconds}s")
                        print(f"    WAIT {self._wait_seconds}s")
                elif key == ord('+') or key == ord('='):
                    self._move_threshold = min(20.0, self._move_threshold + 0.5)
                    print(f"    Schwelle: {self._move_threshold:.1f}°")
                elif key == ord('-'):
                    self._move_threshold = max(0.5, self._move_threshold - 0.5)
                    print(f"    Schwelle: {self._move_threshold:.1f}°")
                elif key == ord('p'):
                    if not self._recording:
                        scripts = sorted(self._output_dir.rglob("*.roarm"))
                        if scripts:
                            print(f"  Abspielen: {scripts[-1].name}")
                            os.system(f"python3 play_roarm.py '{scripts[-1]}'")
                            self._arm.torque_off()
                        else:
                            print("    Kein Skript!")

        except KeyboardInterrupt:
            print("\n  [Abbruch]")
        finally:
            self._shutdown()

    def _shutdown(self):
        if self._recording:
            self._recording = False
            self._save_script()
        if self._arm:
            self._arm.torque_on()
            time.sleep(0.3)
            self._arm.set_led(0)
            self._arm.disconnect()
        if self._camera:
            self._camera.release()
        cv2.destroyAllWindows()
        print("  Beendet")


def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Teach & Record (Movement Detection)")
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--camera", type=int, default=2)
    p.add_argument("--output", type=str, default="teach_recordings")
    p.add_argument("--manual", action="store_true")
    p.add_argument("--hz", type=int, default=10)
    p.add_argument("--wait", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=1.5,
                   help="Movement threshold in degrees (default: 1.5)")
    args = p.parse_args()

    rec = TeachRecorder(port=args.port, camera_index=args.camera,
                        output_dir=args.output, continuous=not args.manual,
                        wait_seconds=args.wait, move_threshold=args.threshold)
    rec.POLL_HZ = args.hz
    rec.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
        print("uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

"""
RoArm-M2-S Teach & Record (Direct Serial)
==========================================
Direkte serielle Kommunikation mit dem Arm via JSON-Befehle.
Kein externer Wrapper nötig.

Steuerung (im OpenCV-Fenster):
  R      → Aufnahme starten/stoppen
  SPACE  → Manueller Wegpunkt (bei --manual)
  O      → Greifer öffnen
  C      → Greifer schließen
  L      → LED an (255)
  K      → LED aus (0)
  W      → Wartezeit einfügen
  P      → Letztes Skript abspielen
  Q      → Beenden
"""

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
    """Findet den seriellen Port des RoArm-M2-S."""
    ports = list(serial.tools.list_ports.comports())
    # Bevorzuge USB-Serial-Ports
    for p in ports:
        desc = (p.description or "").lower()
        if "usb" in desc or "ch340" in desc or "cp210" in desc or "ftdi" in desc:
            return p.device
    # Auf Linux: /dev/ttyUSB*
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    if ports:
        return ports[0].device
    return None


class RoArmDirect:
    """Direkte serielle Kommunikation mit dem RoArm-M2-S."""

    def __init__(self, port: str = None, baudrate: int = 115200):
        if port is None:
            port = find_arm_port()
        if port is None:
            raise RuntimeError("Kein serieller Port gefunden! Bitte --port angeben.")
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=1.0, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        # Buffer leeren
        self._ser.reset_input_buffer()

    def send_cmd(self, cmd: dict) -> str:
        """Sendet JSON-Befehl, gibt Antwort zurück."""
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.05)
            # Antwort lesen (bis zu 500ms warten)
            response = ""
            deadline = time.time() + 0.5
            while time.time() < deadline:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response = line
                        # Wenn es JSON mit T:1051 ist, sofort zurück
                        if '"T":1051' in line or '"T": 1051' in line:
                            return line
                else:
                    time.sleep(0.01)
            return response

    def get_feedback(self) -> dict:
        """Holt Servo-Feedback via CMD_SERVO_RAD_FEEDBACK {T:105}."""
        # Buffer leeren
        with self._lock:
            self._ser.reset_input_buffer()

        resp = self.send_cmd({"T": 105})

        # Versuche JSON zu parsen
        # Manchmal kommen mehrere Zeilen, wir suchen die mit T:1051
        if not resp:
            # Nochmal versuchen, mehr lesen
            time.sleep(0.1)
            with self._lock:
                while self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if '"T":1051' in line or '"T": 1051' in line:
                        resp = line
                        break

        if not resp:
            return None

        # Finde JSON in der Antwort
        try:
            # Manchmal ist Müll davor
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
        """Torque Lock ausschalten - Arm frei bewegbar."""
        self.send_cmd({"T": 210, "cmd": 0})

    def torque_on(self):
        """Torque Lock einschalten."""
        self.send_cmd({"T": 210, "cmd": 1})

    def move_radians(self, b: float, s: float, e: float, t: float, spd: int = 0, acc: int = 10):
        """Bewegt alle Gelenke (Radians)."""
        self.send_cmd({"T": 102, "base": b, "shoulder": s, "elbow": e, "hand": t, "spd": spd, "acc": acc})

    def move_degrees(self, b: float, s: float, e: float, h: float, spd: int = 10, acc: int = 10):
        """Bewegt alle Gelenke (Grad)."""
        self.send_cmd({"T": 122, "b": b, "s": s, "e": e, "h": h, "spd": spd, "acc": acc})

    def gripper_open(self):
        """Greifer öffnen (EoAT auf ~1.08 rad = offen)."""
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})

    def gripper_close(self):
        """Greifer schließen (EoAT auf 3.14 rad = geschlossen)."""
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})

    def set_led(self, brightness: int):
        """LED setzen (0-255)."""
        self.send_cmd({"T": 114, "led": brightness})

    def move_init(self):
        """Zur Startposition."""
        self.send_cmd({"T": 100})

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


class TeachRecorder:
    POLL_HZ = 10

    def __init__(self, port: str = None, camera_index: int = 2,
                 output_dir: str = "teach_recordings",
                 continuous: bool = True, wait_seconds: float = 1.0):
        self._port = port
        self._camera_index = camera_index
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._continuous = continuous
        self._wait_seconds = wait_seconds

        self._arm: RoArmDirect = None
        self._camera = None

        self._recording = False
        self._commands: list = []
        self._frame_count = 0
        self._session_dir: Path = None
        self._images_dir: Path = None
        self._script_path: Path = None

        self._speed_scale = 1.0
        self._spd = 0
        self._acc = 10

        self._running = False
        self._window_name = "RoArm Teach & Record"

    def setup(self) -> bool:
        print("=" * 60)
        print("  RoArm-M2-S Teach & Record (Direct Serial)")
        print("=" * 60)

        # Arm verbinden
        print("\n  [1/3] Arm verbinden...")
        try:
            self._arm = RoArmDirect(port=self._port)
            print(f"    OK Verbunden: {self._arm.port}")
        except Exception as e:
            print(f"    FEHLER: {e}")
            return False

        # Kamera - NUR den angegebenen Index verwenden, kein Fallback
        print(f"  [2/3] Kamera Index {self._camera_index}...")
        self._camera = cv2.VideoCapture(self._camera_index)
        if self._camera.isOpened():
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"    OK Kamera {self._camera_index}")
        else:
            print(f"    WARNUNG: Kamera {self._camera_index} nicht verfügbar!")
            print("    Verfügbare Kameras testen...")
            self._camera = None
            for idx in range(5):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    print(f"      Kamera {idx}: verfügbar")
                    test.release()
                else:
                    print(f"      Kamera {idx}: nicht verfügbar")
            print("    Starte ohne Kamera. Nutze --camera <index>")

        # Torque ausschalten
        print("  [3/3] Torque Lock AUS...")
        self._arm.torque_off()
        time.sleep(0.5)

        # Test: Feedback lesen
        print("  [Test] Feedback lesen...")
        fb = self._arm.get_feedback()
        if fb:
            print(f"    OK Feedback: b={fb.get('b',0):.3f} s={fb.get('s',0):.3f} e={fb.get('e',0):.3f} t={fb.get('t',0):.3f}")
        else:
            print("    WARNUNG: Kein Feedback erhalten! Prüfe Verbindung.")
            print("    (Versuche trotzdem weiterzumachen...)")

        print("\n  Steuerung:")
        print("    R     = Aufnahme Start/Stop")
        print("    SPACE = Manueller Wegpunkt")
        print("    O/C   = Greifer öffnen/schließen")
        print("    L/K   = LED an/aus")
        print("    W     = Wartezeit")
        print("    P     = Abspielen")
        print("    Q     = Beenden")
        if self._continuous:
            print(f"    [Kontinuierlich: {self.POLL_HZ} Hz]")
        print()
        return True

    def _create_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._output_dir / f"session_{timestamp}"
        self._images_dir = self._session_dir / "frames"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._script_path = self._session_dir / f"program_{timestamp}.roarm"
        self._commands = []
        self._frame_count = 0
        print(f"\n  Session: {self._session_dir}")

    def _save_frame(self, frame) -> str:
        if frame is None:
            return ""
        self._frame_count += 1
        filename = f"frame_{self._frame_count:06d}.jpg"
        filepath = self._images_dir / filename
        cv2.imwrite(str(filepath), frame)
        return filename

    def _get_arm_position(self) -> dict:
        """Holt aktuelle Gelenkwinkel vom Arm (Radians → Grad)."""
        fb = self._arm.get_feedback()
        if fb and "b" in fb:
            return {
                "b": round(math.degrees(fb["b"]), 2),
                "s": round(math.degrees(fb["s"]), 2),
                "e": round(math.degrees(fb["e"]), 2),
                "h": round(math.degrees(fb["t"]), 2),  # "t" im Feedback = hand/EoAT
            }
        return None

    def _record_waypoint(self, frame=None):
        pos = self._get_arm_position()
        if pos is None:
            print("    !! Konnte Position nicht lesen!")
            return

        cmd = f"MOVE b={pos['b']} s={pos['s']} e={pos['e']} h={pos['h']}"
        self._commands.append(cmd)

        if frame is not None:
            filename = self._save_frame(frame)
            if filename:
                self._commands.append(f"FRAME {filename}")

        print(f"    + Waypoint: B={pos['b']:.1f} S={pos['s']:.1f} E={pos['e']:.1f} H={pos['h']:.1f}")

    def _save_script(self):
        if not self._commands:
            print("    !! Nichts aufgezeichnet!")
            return

        lines = [
            f"# RoArm-M2-S Teach Recording",
            f"# Erstellt: {datetime.now().isoformat()}",
            f"# Frames: {self._frame_count}",
            f"#CONFIG speed_scale={self._speed_scale}",
            f"#CONFIG spd={self._spd}",
            f"#CONFIG acc={self._acc}",
            "",
        ]
        lines.extend(self._commands)

        with open(self._script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n  Gespeichert: {self._script_path}")
        print(f"  {len(self._commands)} Befehle, {self._frame_count} Frames")

    def _play_script(self, script_path: Path = None):
        path = script_path or self._script_path
        if not path or not path.exists():
            print("    !! Kein Skript zum Abspielen!")
            return

        print(f"\n  > Abspielen: {path.name}")
        self._arm.torque_on()
        time.sleep(0.3)

        spd = self._spd
        acc = self._acc
        speed_scale = self._speed_scale

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or (line.startswith("#") and not line.startswith("#CONFIG")):
                continue

            if line.startswith("#CONFIG"):
                _, kv = line.split(" ", 1)
                key, val = kv.split("=")
                if key == "speed_scale":
                    speed_scale = float(val)
                elif key == "spd":
                    spd = int(val)
                elif key == "acc":
                    acc = int(val)
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "MOVE":
                vals = {}
                for p in parts[1:]:
                    k, v = p.split("=")
                    vals[k] = float(v)
                actual_spd = max(1, int(spd * speed_scale)) if spd > 0 else 0
                actual_acc = max(1, int(acc * speed_scale)) if acc > 0 else 0
                self._arm.move_degrees(
                    b=vals.get("b", 0),
                    s=vals.get("s", 0),
                    e=vals.get("e", 90),
                    h=vals.get("h", 180),
                    spd=actual_spd,
                    acc=actual_acc
                )
                time.sleep(0.1 / max(speed_scale, 0.1))

            elif cmd == "GRIPPER":
                action = parts[1] if len(parts) > 1 else "OPEN"
                if action == "OPEN":
                    self._arm.gripper_open()
                else:
                    self._arm.gripper_close()
                time.sleep(0.5)

            elif cmd == "LED":
                brightness = int(parts[1]) if len(parts) > 1 else 0
                self._arm.set_led(brightness)

            elif cmd == "WAIT":
                wait_time = float(parts[1]) if len(parts) > 1 else 1.0
                time.sleep(wait_time / max(speed_scale, 0.1))

            elif cmd == "FRAME":
                pass

        print("  Abspielen beendet")
        self._arm.torque_off()
        time.sleep(0.3)

    def run(self):
        if not self.setup():
            return

        self._running = True
        last_record_time = 0
        record_interval = 1.0 / self.POLL_HZ

        try:
            while self._running:
                frame = None
                if self._camera:
                    self._camera.grab()
                    ret, frame = self._camera.retrieve()
                    if not ret:
                        frame = None

                now = time.time()
                if self._recording and self._continuous:
                    if now - last_record_time >= record_interval:
                        self._record_waypoint(frame)
                        last_record_time = now

                # Anzeige
                if frame is not None:
                    status_color = (0, 0, 255) if self._recording else (200, 200, 200)
                    status_text = "REC" if self._recording else "IDLE"
                    cv2.putText(frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    if self._recording:
                        cv2.putText(frame, f"Cmds: {len(self._commands)}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(self._window_name, frame)
                else:
                    # Ohne Kamera: kleines schwarzes Fenster für Tasteneingabe
                    dummy = np.zeros((100, 400, 3), dtype=np.uint8)
                    status_text = "REC" if self._recording else "IDLE"
                    cv2.putText(dummy, f"{status_text} | Cmds: {len(self._commands)}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(self._window_name, dummy)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    self._running = False
                elif key == ord('r'):
                    if not self._recording:
                        self._create_session()
                        self._recording = True
                        last_record_time = time.time()
                        print("\n  ● AUFNAHME GESTARTET")
                    else:
                        self._recording = False
                        self._save_script()
                        print("  ■ AUFNAHME GESTOPPT")
                elif key == ord(' '):
                    if self._recording:
                        self._record_waypoint(frame)
                elif key == ord('o'):
                    if self._recording:
                        self._commands.append("GRIPPER OPEN")
                        print("    Greifer OFFEN")
                    self._arm.gripper_open()
                elif key == ord('c'):
                    if self._recording:
                        self._commands.append("GRIPPER CLOSE")
                        print("    Greifer ZU")
                    self._arm.gripper_close()
                elif key == ord('l'):
                    if self._recording:
                        self._commands.append("LED 255")
                        print("    LED AN")
                    self._arm.set_led(255)
                elif key == ord('k'):
                    if self._recording:
                        self._commands.append("LED 0")
                        print("    LED AUS")
                    self._arm.set_led(0)
                elif key == ord('w'):
                    if self._recording:
                        self._commands.append(f"WAIT {self._wait_seconds}")
                        print(f"    WAIT {self._wait_seconds}s")
                elif key == ord('p'):
                    if not self._recording:
                        scripts = sorted(self._output_dir.rglob("*.roarm"))
                        if scripts:
                            self._play_script(scripts[-1])
                        else:
                            print("    !! Kein Skript gefunden!")

        except KeyboardInterrupt:
            print("\n  [Abgebrochen]")
        finally:
            self._shutdown()

    def _shutdown(self):
        if self._recording:
            self._recording = False
            self._save_script()
        if self._arm:
            self._arm.torque_on()
            time.sleep(0.3)
            self._arm.move_init()
            time.sleep(2.0)
            self._arm.set_led(0)
            self._arm.disconnect()
        if self._camera:
            self._camera.release()
        cv2.destroyAllWindows()
        print("\n  Beendet")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoArm-M2-S Teach & Record (Direct Serial)")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port (z.B. /dev/ttyUSB0)")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index (default: 2)")
    parser.add_argument("--output", type=str, default="teach_recordings", help="Output-Verzeichnis")
    parser.add_argument("--manual", action="store_true", help="Nur bei SPACE aufzeichnen")
    parser.add_argument("--hz", type=int, default=10, help="Aufnahme-Frequenz (Hz)")
    parser.add_argument("--wait", type=float, default=1.0, help="Standard-Wartezeit (s)")
    args = parser.parse_args()

    recorder = TeachRecorder(
        port=args.port,
        camera_index=args.camera,
        output_dir=args.output,
        continuous=not args.manual,
        wait_seconds=args.wait,
    )
    recorder.POLL_HZ = args.hz
    recorder.run()


if __name__ == "__main__":
    main()

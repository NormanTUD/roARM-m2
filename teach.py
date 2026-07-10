#!/usr/bin/env python3
"""teach_record.py - RoArm-M2-S Teach & Record"""
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
            raise RuntimeError("Kein serieller Port gefunden!")
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0.3, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()

    def _send_raw(self, cmd: dict) -> str:
        msg = json.dumps(cmd, separators=(',', ':'))
        self._ser.write(msg.encode() + b'\n')
        self._ser.flush()
        time.sleep(0.02)
        response = ""
        deadline = time.time() + 0.3
        while time.time() < deadline:
            if self._ser.in_waiting:
                line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    response = line
                    if '"T":1051' in line or '"T": 1051' in line:
                        return line
            else:
                time.sleep(0.005)
        return response

    def send_cmd(self, cmd: dict) -> str:
        with self._lock:
            return self._send_raw(cmd)

    def get_feedback(self) -> dict:
        with self._lock:
            self._ser.reset_input_buffer()
            time.sleep(0.01)
            resp = self._send_raw({"T": 105})
            if not resp:
                time.sleep(0.05)
                resp = self._send_raw({"T": 105})
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
        with self._lock:
            self._ser.reset_input_buffer()
            self._send_raw({"T": 210, "cmd": 0})
            time.sleep(0.05)
            self._send_raw({"T": 210, "cmd": 0})
            time.sleep(0.05)
            for servo_id in range(1, 5):
                self._send_raw({"T": 212, "id": servo_id, "cmd": 0})
                time.sleep(0.02)
            self._send_raw({"T": 10, "cmd": 0})
            time.sleep(0.05)
            self._ser.reset_input_buffer()

    def torque_on(self):
        with self._lock:
            self._ser.reset_input_buffer()
            self._send_raw({"T": 210, "cmd": 1})
            time.sleep(0.05)
            for servo_id in range(1, 5):
                self._send_raw({"T": 212, "id": servo_id, "cmd": 1})
                time.sleep(0.02)
            self._send_raw({"T": 10, "cmd": 1})
            time.sleep(0.05)
            self._ser.reset_input_buffer()

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
        self._move_threshold = move_threshold
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
        self._last_pos = None
        self._movement_detected = False
        self._movement_flash_time = 0
        self._movement_delta = {}
        self._total_movements = 0
        self._live_log = []
        self._max_log_lines = 8
        self._torque_active = True
        self._rec_start_time = 0
        self._last_feedback = None
        self._last_feedback_time = 0

    def setup(self) -> bool:
        print("=" * 60)
        print("  RoArm-M2-S Teach & Record")
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
            print(f"    OK")
        else:
            print(f"    WARNUNG: Kamera nicht offen!")
            self._camera = None

        print("  [3/3] Default-Position...")
        self._arm.torque_on()
        time.sleep(0.3)
        self._arm.move_init()
        time.sleep(2.0)

        fb = self._arm.get_feedback()
        if fb:
            print(f"    Feedback OK: b={fb.get('b',0):.2f} s={fb.get('s',0):.2f} e={fb.get('e',0):.2f} t={fb.get('t',0):.2f}")
            self._last_feedback = fb
        else:
            print("    WARNUNG: Kein Feedback!")

        print("\n    T=Torque  R=Record  O=Open  C=Close  Q=Quit")
        self._torque_active = True
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

    def _save_frame(self, frame) -> str:
        if frame is None:
            return ""
        self._frame_count += 1
        filename = f"frame_{self._frame_count:06d}.jpg"
        cv2.imwrite(str(self._images_dir / filename), frame)
        return filename

    def _poll_feedback(self):
        now = time.time()
        if now - self._last_feedback_time < 0.08:
            return self._last_feedback
        fb = self._arm.get_feedback()
        if fb and "b" in fb:
            self._last_feedback = fb
            self._last_feedback_time = now
        return self._last_feedback

    def _get_arm_position(self) -> dict:
        fb = self._poll_feedback()
        if fb and "b" in fb:
            return {
                "b": round(math.degrees(fb["b"]), 2),
                "s": round(math.degrees(fb["s"]), 2),
                "e": round(math.degrees(fb["e"]), 2),
                "h": round(math.degrees(fb.get("t", fb.get("h", 0))), 2),
            }
        return None

    def _check_movement(self, pos: dict) -> bool:
        if self._last_pos is None:
            return True
        self._movement_delta = {}
        moved = False
        for joint in ['b', 's', 'e', 'h']:
            delta = abs(pos[joint] - self._last_pos[joint])
            if delta >= self._move_threshold:
                self._movement_delta[joint] = pos[joint] - self._last_pos[joint]
                moved = True
        return moved

    def _add_log(self, msg: str):
        self._live_log.append((time.time(), msg))
        if len(self._live_log) > self._max_log_lines:
            self._live_log.pop(0)

    def _record_waypoint(self, frame=None, force=False):
        pos = self._get_arm_position()
        if pos is None:
            return
        if not force and not self._check_movement(pos):
            self._movement_detected = False
            return
        self._movement_detected = True
        self._movement_flash_time = time.time()
        self._total_movements += 1
        elapsed = time.time() - self._rec_start_time
        self._commands.append(f"MOVE b={pos['b']} s={pos['s']} e={pos['e']} h={pos['h']} t={elapsed:.3f}")
        if frame is not None:
            fn = self._save_frame(frame)
            if fn:
                self._commands.append(f"FRAME {fn}")
        if self._movement_delta:
            parts = []
            for joint, delta in self._movement_delta.items():
                direction = "+" if delta > 0 else ""
                parts.append(f"{joint}:{direction}{delta:.1f}")
            move_info = " | ".join(parts)
        else:
            move_info = "initial"
        self._last_pos = pos.copy()
        log_msg = f"#{self._total_movements} [{elapsed:.1f}s] {move_info}"
        self._add_log(log_msg)
        print(f"    WP {log_msg}")

    def _save_script(self):
        if not self._commands:
            print("    Nichts aufgezeichnet!")
            return
        lines = [
            f"# RoArm-M2-S Teach Recording",
            f"# {datetime.now().isoformat()}",
            f"# Movements: {self._total_movements}",
            f"# Threshold: {self._move_threshold} degrees",
            f"#CONFIG speed_scale={self._speed_scale}",
            f"#CONFIG spd={self._spd}",
            f"#CONFIG acc={self._acc}",
            "",
        ] + self._commands
        with open(self._script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  Gespeichert: {self._script_path}")
        print(f"  {len(self._commands)} Befehle, {self._total_movements} Bewegungen")

    def _draw_overlay(self, disp):
        now = time.time()
        h, w = disp.shape[:2]

        if self._torque_active:
            torque_text = "TORQUE: AN (fest)"
            torque_color = (0, 100, 255)
        else:
            torque_text = "TORQUE: AUS (frei)"
            torque_color = (0, 255, 0)
        cv2.putText(disp, torque_text, (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, torque_color, 2)

        pos = self._get_arm_position()
        if pos:
            pos_text = f"b={pos['b']:.1f} s={pos['s']:.1f} e={pos['e']:.1f} h={pos['h']:.1f}"
            cv2.putText(disp, pos_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        if self._recording:
            flash_active = (now - self._movement_flash_time) < 0.3
            if flash_active:
                cv2.rectangle(disp, (0, 0), (w-1, h-1), (0, 255, 0), 4)
                status_color = (0, 255, 0)
                status_text = "REC - MOVE"
            else:
                status_color = (0, 0, 255)
                status_text = "REC"
            cv2.putText(disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            elapsed = now - self._rec_start_time
            cv2.putText(disp, f"Moves:{self._total_movements} | {elapsed:.1f}s | Thr:{self._move_threshold:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self._live_log:
                log_y = h - 50 - (len(self._live_log) * 20)
                for i, (log_time, log_msg) in enumerate(self._live_log):
                    age = now - log_time
                    alpha = max(0.3, 1.0 - (age / 10.0))
                    color = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
                    y = log_y + i * 20
                    cv2.putText(disp, log_msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            cv2.putText(disp, "IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(disp, "T=Torque R=Record O=Open C=Close Q=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return disp

    def run(self):
        if not self.setup():
            return
        self._running = True
        last_rec = 0
        interval = 1.0 / self.POLL_HZ

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

                if frame is not None:
                    disp = frame.copy()
                    disp = self._draw_overlay(disp)
                    cv2.imshow(self._window_name, disp)
                else:
                    dummy = np.zeros((300, 600, 3), dtype=np.uint8)
                    dummy = self._draw_overlay(dummy)
                    cv2.imshow(self._window_name, dummy)

                key = cv2.waitKey(30) & 0xFF

                if key == 255:
                    continue

                if key == ord('q'):
                    self._running = False

                elif key == ord('t'):
                    if self._torque_active:
                        self._arm.torque_off()
                        self._torque_active = False
                        print("    TORQUE AUS")
                        time.sleep(0.2)
                        self._arm._ser.reset_input_buffer()
                    else:
                        self._arm.torque_on()
                        self._torque_active = True
                        print("    TORQUE AN")
                        time.sleep(0.2)
                        self._arm._ser.reset_input_buffer()

                elif key == ord('r'):
                    if not self._recording:
                        if self._torque_active:
                            self._arm.torque_off()
                            self._torque_active = False
                            time.sleep(0.2)
                            self._arm._ser.reset_input_buffer()
                            print("    TORQUE AUS fuer Aufnahme")
                        self._create_session()
                        self._rec_start_time = time.time()
                        self._recording = True
                        self._last_feedback_time = 0
                        last_rec = 0
                        print("  REC gestartet")
                    else:
                        self._recording = False
                        self._save_script()
                        self._session_dir = None

                elif key == ord(' '):
                    if self._recording:
                        self._record_waypoint(frame, force=True)

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

                elif key == ord('w'):
                    if self._recording:
                        self._commands.append(f"WAIT {self._wait_seconds}")
                        self._add_log(f"WAIT {self._wait_seconds}s")

                elif key == ord('+') or key == ord('='):
                    self._move_threshold = min(20.0, self._move_threshold + 0.5)
                    print(f"    Schwelle: {self._move_threshold:.1f}")

                elif key == ord('-'):
                    self._move_threshold = max(0.5, self._move_threshold - 0.5)
                    print(f"    Schwelle: {self._move_threshold:.1f}")

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
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--camera", type=int, default=2)
    p.add_argument("--output", type=str, default="teach_recordings")
    p.add_argument("--manual", action="store_true")
    p.add_argument("--hz", type=int, default=10)
    p.add_argument("--wait", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=1.5)
    args = p.parse_args()
    rec = TeachRecorder(port=args.port, camera_index=args.camera,
                        output_dir=args.output, continuous=not args.manual,
                        wait_seconds=args.wait, move_threshold=args.threshold)
    rec.POLL_HZ = args.hz
    rec.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""teach_record.py - RoArm-M2-S Teach & Record (Maximum Precision Edition)"""
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
from collections import deque
import cv2


# === NORMALIZED START POSITION ===
NORMALIZED_START_POSITION = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

START_POSITION_TOLERANCE = 0.3  # Tighter tolerance


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
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0.15, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()

    def _send_raw(self, cmd: dict) -> str:
        msg = json.dumps(cmd, separators=(',', ':'))
        self._ser.write(msg.encode() + b'\n')
        self._ser.flush()
        time.sleep(0.015)
        response = ""
        deadline = time.time() + 0.2
        while time.time() < deadline:
            if self._ser.in_waiting:
                line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    response = line
                    if '"T":1051' in line or '"T": 1051' in line:
                        return line
            else:
                time.sleep(0.003)
        return response

    def send_cmd(self, cmd: dict) -> str:
        with self._lock:
            return self._send_raw(cmd)

    def get_feedback(self) -> dict:
        with self._lock:
            self._ser.reset_input_buffer()
            time.sleep(0.005)
            resp = self._send_raw({"T": 105})
            if not resp:
                time.sleep(0.03)
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

    def get_feedback_averaged(self, num_readings: int = 10) -> dict:
        """
        Take multiple readings, discard outliers (IQR method), and average.
        This dramatically reduces encoder noise.
        """
        readings = []
        for _ in range(num_readings + 4):  # Take extra to account for discards
            fb = self.get_feedback()
            if fb and "b" in fb:
                readings.append(fb)
            time.sleep(0.008)
        if len(readings) < 3:
            # Fallback: just average what we have
            if not readings:
                return None
            avg = {}
            for key in ["b", "s", "e", "t"]:
                vals = [r[key] for r in readings if key in r]
                if vals:
                    avg[key] = sum(vals) / len(vals)
            return avg if "b" in avg else None

        # IQR-based outlier rejection per joint
        avg = {}
        for key in ["b", "s", "e", "t"]:
            vals = sorted([r[key] for r in readings if key in r])
            if len(vals) < 3:
                if vals:
                    avg[key] = sum(vals) / len(vals)
                continue
            q1_idx = len(vals) // 4
            q3_idx = (3 * len(vals)) // 4
            q1 = vals[q1_idx]
            q3 = vals[q3_idx]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = [v for v in vals if lower <= v <= upper]
            if filtered:
                avg[key] = sum(filtered) / len(filtered)
            else:
                avg[key] = sum(vals) / len(vals)

        if "T" in readings[0]:
            avg["T"] = readings[0]["T"]
        return avg if "b" in avg else None

    def get_feedback_median(self, num_readings: int = 7) -> dict:
        """
        Take multiple readings and return the median for each joint.
        Median is more robust to single-sample spikes than mean.
        """
        readings = []
        for _ in range(num_readings):
            fb = self.get_feedback()
            if fb and "b" in fb:
                readings.append(fb)
            time.sleep(0.008)
        if not readings:
            return None
        avg = {}
        for key in ["b", "s", "e", "t"]:
            vals = sorted([r[key] for r in readings if key in r])
            if vals:
                avg[key] = vals[len(vals) // 2]  # Median
        return avg if "b" in avg else None

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
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    def set_led(self, brightness: int):
        self.send_cmd({"T": 114, "led": brightness})

    def move_init(self):
        self.send_cmd({"T": 100})

    def move_to_position(self, b: float, s: float, e: float, h: float, spd: int = 20, acc: int = 10):
        """Move to an exact joint position (in degrees)."""
        self.send_cmd({
            "T": 122,
            "b": round(b, 4),
            "s": round(s, 4),
            "e": round(e, 4),
            "h": round(h, 4),
            "spd": spd,
            "acc": acc
        })

    def move_to_normalized_start(self):
        """Move to the normalized starting position with iterative correction."""
        pos = NORMALIZED_START_POSITION
        # First move: get close
        self.move_to_position(pos["b"], pos["s"], pos["e"], pos["h"], spd=15, acc=5)
        time.sleep(2.0)
        # Second move: slower, more precise
        self.move_to_position(pos["b"], pos["s"], pos["e"], pos["h"], spd=8, acc=3)
        time.sleep(1.5)

    def verify_position(self, target: dict, tolerance: float = START_POSITION_TOLERANCE) -> tuple:
        """Verify the arm is at the target position within tolerance."""
        fb = self.get_feedback_averaged(num_readings=10)
        if fb is None:
            return False, None, float('inf')
        actual = {
            "b": math.degrees(fb["b"]),
            "s": math.degrees(fb["s"]),
            "e": math.degrees(fb["e"]),
            "h": math.degrees(fb.get("t", fb.get("h", 0))),
        }
        max_error = 0.0
        for joint in ["b", "s", "e", "h"]:
            error = abs(actual[joint] - target[joint])
            max_error = max(max_error, error)
        return max_error <= tolerance, actual, max_error

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing servo encoder readings."""
    def __init__(self, process_noise=0.001, measurement_noise=0.01):
        self.q = process_noise      # Process noise
        self.r = measurement_noise  # Measurement noise
        self.x = None               # State estimate
        self.p = 1.0                # Estimate uncertainty
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.p = self.r
            self.initialized = True
            return self.x

        # Predict
        # (state doesn't change in our model, so predict = current)
        p_pred = self.p + self.q

        # Update
        k = p_pred / (p_pred + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred

        return self.x

    def reset(self):
        self.x = None
        self.p = 1.0
        self.initialized = False


class TeachRecorder:
    POLL_HZ = 50  # Increased to 50Hz for much better temporal resolution

    def __init__(self, port=None, camera_index=2, output_dir="teach_recordings",
                 continuous=True, wait_seconds=1.0, move_threshold=0.15):
        self._port = port
        self._camera_index = camera_index
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._continuous = continuous
        self._wait_seconds = wait_seconds
        self._move_threshold = move_threshold  # Much lower: 0.15 degrees
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
        self._num_avg_readings = 10  # Increased from 3 to 10
        self._start_verified = False
        # Kalman filters for each joint
        self._kalman = {
            "b": KalmanFilter1D(process_noise=0.0005, measurement_noise=0.005),
            "s": KalmanFilter1D(process_noise=0.0005, measurement_noise=0.005),
            "e": KalmanFilter1D(process_noise=0.0005, measurement_noise=0.005),
            "t": KalmanFilter1D(process_noise=0.0005, measurement_noise=0.005),
        }
        # Velocity estimation for predictive recording
        self._velocity_history = {
            "b": deque(maxlen=5),
            "s": deque(maxlen=5),
            "e": deque(maxlen=5),
            "h": deque(maxlen=5),
        }
        self._last_pos_time = 0
        # Deadband hysteresis: require threshold to START moving, but record
        # at lower threshold once movement is detected
        self._is_moving = False
        self._move_start_threshold = move_threshold  # Higher threshold to start
        self._move_continue_threshold = move_threshold * 0.4  # Lower to continue
        self._stillness_count = 0
        self._stillness_required = 3  # Must be still for N polls to stop

    def setup(self) -> bool:
        print("=" * 60)
        print("  RoArm-M2-S Teach & Record (Maximum Precision Edition)")
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

        print("  [3/3] Fahre zur normalisierten Startposition...")
        self._arm.torque_on()
        time.sleep(0.3)

        self._arm.move_to_normalized_start()
        time.sleep(3.0)

        # Verify with tight tolerance
        print("    Verifiziere Startposition...")
        is_ok, actual, max_error = self._arm.verify_position(NORMALIZED_START_POSITION, tolerance=0.3)
        if is_ok:
            print(f"    ✓ Startposition erreicht (max Fehler: {max_error:.4f}°)")
            self._start_verified = True
        else:
            print(f"    ⚠ Startposition nicht exakt erreicht (max Fehler: {max_error:.4f}°)")
            if actual:
                print(f"      Ist:  b={actual['b']:.3f} s={actual['s']:.3f} e={actual['e']:.3f} h={actual['h']:.3f}")
                print(f"      Soll: b={NORMALIZED_START_POSITION['b']:.3f} s={NORMALIZED_START_POSITION['s']:.3f} "
                      f"e={NORMALIZED_START_POSITION['e']:.3f} h={NORMALIZED_START_POSITION['h']:.3f}")
            # Iterative correction
            for attempt in range(3):
                print(f"    Korrekturversuch {attempt + 2}...")
                self._arm.move_to_position(
                    NORMALIZED_START_POSITION["b"],
                    NORMALIZED_START_POSITION["s"],
                    NORMALIZED_START_POSITION["e"],
                    NORMALIZED_START_POSITION["h"],
                    spd=5, acc=2
                )
                time.sleep(2.0)
                is_ok, actual, max_error = self._arm.verify_position(NORMALIZED_START_POSITION, tolerance=0.3)
                if is_ok:
                    print(f"    ✓ Startposition erreicht (max Fehler: {max_error:.4f}°)")
                    self._start_verified = True
                    break
            if not self._start_verified:
                print(f"    ⚠ WARNUNG: Startposition weicht ab um {max_error:.4f}° - Aufnahme trotzdem möglich")

        fb = self._arm.get_feedback()
        if fb:
            print(f"    Feedback OK: b={math.degrees(fb.get('b',0)):.3f}° "
                  f"s={math.degrees(fb.get('s',0)):.3f}° "
                  f"e={math.degrees(fb.get('e',0)):.3f}° "
                  f"t={math.degrees(fb.get('t',0)):.3f}°")
            self._last_feedback = fb
        else:
            print("    WARNUNG: Kein Feedback!")

        print(f"\n    Einstellungen: Poll={self.POLL_HZ}Hz, Schwelle={self._move_threshold}°, "
              f"Avg={self._num_avg_readings}, Kalman=ON")
        print("    T=Torque  R=Record  O=Open  C=Close  Q=Quit")
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
        self._is_moving = False
        self._stillness_count = 0
        # Reset Kalman filters
        for kf in self._kalman.values():
            kf.reset()
        # Reset velocity history
        for key in self._velocity_history:
            self._velocity_history[key].clear()

    def _save_frame(self, frame) -> str:
        if frame is None:
            return ""
        self._frame_count += 1
        filename = f"frame_{self._frame_count:06d}.jpg"
        cv2.imwrite(str(self._images_dir / filename), frame)
        return filename

    def _poll_feedback(self):
        now = time.time()
        min_interval = 1.0 / (self.POLL_HZ * 1.5)
        if now - self._last_feedback_time < min_interval:
            return self._last_feedback
        # Use median for robustness against spikes
        fb = self._arm.get_feedback_median(num_readings=self._num_avg_readings)
        if fb and "b" in fb:
            # Apply Kalman filtering on top of median
            for key in ["b", "s", "e", "t"]:
                if key in fb:
                    fb[key] = self._kalman[key].update(fb[key])
            self._last_feedback = fb
            self._last_feedback_time = now
        return self._last_feedback

    def _get_arm_position(self) -> dict:
        fb = self._poll_feedback()
        if fb and "b" in fb:
            return {
                "b": round(math.degrees(fb["b"]), 5),  # 5 decimal places
                "s": round(math.degrees(fb["s"]), 5),
                "e": round(math.degrees(fb["e"]), 5),
                "h": round(math.degrees(fb.get("t", fb.get("h", 0))), 5),
            }
        return None

    def _check_movement(self, pos: dict) -> bool:
        """
        Hysteresis-based movement detection:
        - Need to exceed _move_start_threshold to START recording movement
        - Once moving, continue recording at _move_continue_threshold
        - Must be still for _stillness_required polls to STOP
        """
        if self._last_pos is None:
            return True

        self._movement_delta = {}
        max_delta = 0.0
        for joint in ['b', 's', 'e', 'h']:
            delta = pos[joint] - self._last_pos[joint]
            abs_delta = abs(delta)
            if abs_delta > max_delta:
                max_delta = abs_delta
            if abs_delta >= self._move_continue_threshold:
                self._movement_delta[joint] = delta

        if self._is_moving:
            # Already moving: use lower threshold, require stillness to stop
            if max_delta < self._move_continue_threshold:
                self._stillness_count += 1
                if self._stillness_count >= self._stillness_required:
                    self._is_moving = False
                    self._stillness_count = 0
                    # Record one final position at the stop point
                    return True
                return False
            else:
                self._stillness_count = 0
                return True
        else:
            # Not moving: use higher threshold to start
            if max_delta >= self._move_start_threshold:
                self._is_moving = True
                self._stillness_count = 0
                return True
            return False

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

        # Update velocity estimation
        now = time.time()
        if self._last_pos and self._last_pos_time > 0:
            dt = now - self._last_pos_time
            if dt > 0:
                for joint in ['b', 's', 'e', 'h']:
                    vel = (pos[joint] - self._last_pos[joint]) / dt
                    self._velocity_history[joint].append(vel)

        # Store with 5 decimal places for maximum precision
        self._commands.append(
            f"MOVE b={pos['b']:.5f} s={pos['s']:.5f} e={pos['e']:.5f} h={pos['h']:.5f} t={elapsed:.5f}"
        )
        if frame is not None:
            fn = self._save_frame(frame)
            if fn:
                self._commands.append(f"FRAME {fn}")
        if self._movement_delta:
            parts = []
            for joint, delta in self._movement_delta.items():
                direction = "+" if delta > 0 else ""
                parts.append(f"{joint}:{direction}{delta:.3f}")
            move_info = " | ".join(parts)
        else:
            move_info = "initial"
        self._last_pos = pos.copy()
        self._last_pos_time = now
        log_msg = f"#{self._total_movements} [{elapsed:.2f}s] {move_info}"
        self._add_log(log_msg)
        print(f"    WP {log_msg}")

    def _save_script(self):
        if not self._commands:
            print("    Nichts aufgezeichnet!")
            return
        lines = [
            f"# RoArm-M2-S Teach Recording (Maximum Precision Edition)",
            f"# {datetime.now().isoformat()}",
            f"# Movements: {self._total_movements}",
            f"# Threshold: {self._move_threshold} degrees (start), {self._move_continue_threshold:.3f} (continue)",
            f"# Poll Rate: {self.POLL_HZ} Hz",
            f"# Averaging: {self._num_avg_readings} readings (median + Kalman)",
            f"# Start Verified: {self._start_verified}",
            f"#CONFIG speed_scale={self._speed_scale}",
            f"#CONFIG spd={self._spd}",
            f"#CONFIG acc={self._acc}",
            f"#CONFIG poll_hz={self.POLL_HZ}",
            f"#CONFIG threshold={self._move_threshold}",
            f"#START_POS b={NORMALIZED_START_POSITION['b']:.5f} s={NORMALIZED_START_POSITION['s']:.5f} "
            f"e={NORMALIZED_START_POSITION['e']:.5f} h={NORMALIZED_START_POSITION['h']:.5f}",
            "",
        ] + self._commands
        with open(self._script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  Gespeichert: {self._script_path}")
        print(f"  {len(self._commands)} Befehle, {self._total_movements} Bewegungen")
        print(f"  Poll: {self.POLL_HZ}Hz, Schwelle: {self._move_threshold}°, Avg: {self._num_avg_readings}")

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

        if self._start_verified:
            cv2.putText(disp, "START: OK", (w - 280, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(disp, "START: UNVERIFIED", (w - 280, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show moving state
        if self._is_moving:
            cv2.putText(disp, "MOVING", (w - 280, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        pos = self._get_arm_position()
        if pos:
            pos_text = f"b={pos['b']:.3f} s={pos['s']:.3f} e={pos['e']:.3f} h={pos['h']:.3f}"
            cv2.putText(disp, pos_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        settings_text = f"Hz:{self.POLL_HZ} Thr:{self._move_threshold:.2f} Avg:{self._num_avg_readings} Kalman:ON"
        cv2.putText(disp, settings_text, (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

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
            cv2.putText(disp, f"Moves:{self._total_movements} | {elapsed:.1f}s | Thr:{self._move_threshold:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self._live_log:
                log_y = h - 70 - (len(self._live_log) * 20)
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

                key = cv2.waitKey(16) & 0xFF  # ~60fps UI for responsiveness

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
                        # Reset Kalman filters when torque changes (arm will move)
                        for kf in self._kalman.values():
                            kf.reset()
                    else:
                        self._arm.torque_on()
                        self._torque_active = True
                        print("    TORQUE AN")
                        time.sleep(0.2)
                        self._arm._ser.reset_input_buffer()
                        for kf in self._kalman.values():
                            kf.reset()

                elif key == ord('r'):
                    if not self._recording:
                        if self._torque_active:
                            self._arm.torque_off()
                            self._torque_active = False
                            time.sleep(0.2)
                            self._arm._ser.reset_input_buffer()
                            print("    TORQUE AUS fuer Aufnahme")
                        # Reset filters for fresh recording
                        for kf in self._kalman.values():
                            kf.reset()
                        self._create_session()
                        self._rec_start_time = time.time()
                        self._recording = True
                        self._last_feedback_time = 0
                        last_rec = 0
                        # Warm up Kalman filters with initial readings
                        print("    Kalibriere Filter...")
                        for _ in range(15):
                            self._poll_feedback()
                            time.sleep(0.02)
                        # Record the initial position immediately
                        time.sleep(0.05)
                        self._record_waypoint(frame, force=True)
                        print("  REC gestartet (Maximum Precision Mode)")
                    else:
                        self._recording = False
                        self._save_script()
                        self._session_dir = None

                elif key == ord(' '):
                    if self._recording:
                        self._record_waypoint(frame, force=True)

                elif key == ord('o'):
                    if self._recording:
                        elapsed = time.time() - self._rec_start_time
                        self._commands.append(f"GRIPPER OPEN t={elapsed:.5f}")
                        self._add_log("GRIPPER OPEN")
                    self._arm.gripper_open()
                    print("    Greifer OFFEN")

                elif key == ord('c'):
                    if self._recording:
                        elapsed = time.time() - self._rec_start_time
                        self._commands.append(f"GRIPPER CLOSE t={elapsed:.5f}")
                        self._add_log("GRIPPER CLOSE")
                    self._arm.gripper_close()
                    print("    Greifer ZU")

                elif key == ord('w'):
                    if self._recording:
                        self._commands.append(f"WAIT {self._wait_seconds}")
                        self._add_log(f"WAIT {self._wait_seconds}s")

                elif key == ord('+') or key == ord('='):
                    self._move_threshold = min(20.0, self._move_threshold + 0.05)
                    self._move_start_threshold = self._move_threshold
                    self._move_continue_threshold = self._move_threshold * 0.4
                    print(f"    Schwelle: {self._move_threshold:.3f}° (continue: {self._move_continue_threshold:.3f}°)")

                elif key == ord('-'):
                    self._move_threshold = max(0.02, self._move_threshold - 0.05)
                    self._move_start_threshold = self._move_threshold
                    self._move_continue_threshold = self._move_threshold * 0.4
                    print(f"    Schwelle: {self._move_threshold:.3f}° (continue: {self._move_continue_threshold:.3f}°)")

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
    p.add_argument("--hz", type=int, default=50)
    p.add_argument("--wait", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--avg", type=int, default=7, help="Number of readings for median filter")
    args = p.parse_args()
    rec = TeachRecorder(port=args.port, camera_index=args.camera,
                        output_dir=args.output, continuous=not args.manual,
                        wait_seconds=args.wait, move_threshold=args.threshold)
    rec.POLL_HZ = args.hz
    rec._num_avg_readings = args.avg
    rec.run()

if __name__ == "__main__":
    main()

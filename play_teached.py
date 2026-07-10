#!/usr/bin/env python3
"""play_teached.py - Plays .roarm script files with smooth interpolated motion (Precision Edition)"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
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
        print("uv not installed.")
        sys.exit(1)

_ensure_uv()

import json
import time
import math
import threading
import serial
import serial.tools.list_ports
import numpy as np
from pathlib import Path


# === NORMALIZED START POSITION ===
NORMALIZED_START_POSITION = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

START_POSITION_TOLERANCE = 0.5


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


class RoArmPlayer:
    def __init__(self, port=None, baudrate=115200):
        if port is None:
            port = find_arm_port()
        if port is None:
            raise RuntimeError("Kein Port gefunden!")
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0.15, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self.port = port
        print(f"  Verbunden: {port}")

    def send(self, cmd: dict):
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.008)

    def send_and_read(self, cmd: dict) -> str:
        with self._lock:
            self._ser.reset_input_buffer()
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

    def get_feedback(self) -> dict:
        resp = self.send_and_read({"T": 105})
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

    def get_feedback_averaged(self, num_readings: int = 3) -> dict:
        readings = []
        for _ in range(num_readings):
            fb = self.get_feedback()
            if fb and "b" in fb:
                readings.append(fb)
            time.sleep(0.01)
        if not readings:
            return None
        avg = {}
        for key in ["b", "s", "e", "t"]:
            vals = [r[key] for r in readings if key in r]
            if vals:
                avg[key] = sum(vals) / len(vals)
        return avg if "b" in avg else None

    def get_position_degrees(self) -> dict:
        fb = self.get_feedback_averaged(num_readings=3)
        if fb and "b" in fb:
            return {
                "b": math.degrees(fb["b"]),
                "s": math.degrees(fb["s"]),
                "e": math.degrees(fb["e"]),
                "h": math.degrees(fb.get("t", fb.get("h", 0))),
            }
        return None

    def verify_position(self, target: dict, tolerance: float = START_POSITION_TOLERANCE) -> tuple:
        actual = self.get_position_degrees()
        if actual is None:
            return False, None, float('inf')
        max_error = 0.0
        for joint in ["b", "s", "e", "h"]:
            error = abs(actual[joint] - target[joint])
            max_error = max(max_error, error)
        return max_error <= tolerance, actual, max_error

    def torque_on(self):
        """Explicitly enable torque on all servos."""
        with self._lock:
            self._ser.reset_input_buffer()
            msg = json.dumps({"T": 210, "cmd": 1}, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.05)
            for servo_id in range(1, 5):
                msg = json.dumps({"T": 212, "id": servo_id, "cmd": 1}, separators=(',', ':'))
                self._ser.write(msg.encode() + b'\n')
                self._ser.flush()
                time.sleep(0.02)
            msg = json.dumps({"T": 10, "cmd": 1}, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.05)
            self._ser.reset_input_buffer()

    def gripper_open(self):
        """Open gripper with safe speed/acc values that don't kill torque."""
        # Use spd=50, acc=20 instead of spd=0, acc=0
        # spd=0 on some firmware versions means "disable servo" not "max speed"
        self.send({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        """Close gripper with safe speed/acc values that don't kill torque."""
        self.send({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


def monotone_cubic_interpolate(times, values, sample_times):
    """
    Fritsch-Carlson monotone cubic interpolation.
    Prevents overshoot between waypoints unlike natural cubic splines.
    """
    if len(times) < 2:
        return np.full_like(sample_times, values[0] if len(values) > 0 else 0.0)

    if len(times) == 2:
        return np.interp(sample_times, times, values)

    n = len(times)
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    # Step 1: Compute secants
    h = np.diff(times)
    h = np.where(h == 0, 1e-10, h)
    delta = np.diff(values) / h

    # Step 2: Initialize tangents
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            m[i] = (delta[i - 1] + delta[i]) / 2.0

    # Step 3: Fritsch-Carlson modification
    for i in range(n - 1):
        if abs(delta[i]) < 1e-12:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            alpha = m[i] / delta[i]
            beta = m[i + 1] / delta[i]
            tau = alpha**2 + beta**2
            if tau > 9.0:
                s = 3.0 / math.sqrt(tau)
                m[i] = s * alpha * delta[i]
                m[i + 1] = s * beta * delta[i]

    # Step 4: Evaluate Hermite basis functions
    result = np.zeros_like(sample_times)
    for idx, t in enumerate(sample_times):
        if t <= times[0]:
            result[idx] = values[0]
            continue
        if t >= times[-1]:
            result[idx] = values[-1]
            continue

        seg = np.searchsorted(times, t, side='right') - 1
        seg = max(0, min(seg, n - 2))

        dt = times[seg + 1] - times[seg]
        if dt < 1e-12:
            result[idx] = values[seg]
            continue

        s = (t - times[seg]) / dt
        s2 = s * s
        s3 = s2 * s

        h00 = 2*s3 - 3*s2 + 1
        h10 = s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 = s3 - s2

        result[idx] = (h00 * values[seg] +
                       h10 * dt * m[seg] +
                       h01 * values[seg + 1] +
                       h11 * dt * m[seg + 1])

    return result


def cubic_interpolate(times, values, sample_times):
    """Natural cubic spline interpolation (fallback)."""
    if len(times) < 2:
        return np.full_like(sample_times, values[0] if len(values) > 0 else 0.0)

    if len(times) == 2:
        return np.interp(sample_times, times, values)

    n = len(times)
    h = np.diff(times)
    h = np.where(h == 0, 1e-6, h)

    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i]) - \
                   (3.0 / h[i - 1]) * (values[i] - values[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2.0 * (times[i + 1] - times[i - 1]) - h[i - 1] * mu[i - 1]
        if abs(l[i]) < 1e-12:
            l[i] = 1e-12
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (values[j + 1] - values[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    result = np.zeros_like(sample_times)
    for idx, t in enumerate(sample_times):
        if t <= times[0]:
            result[idx] = values[0]
            continue
        if t >= times[-1]:
            result[idx] = values[-1]
            continue
        seg = np.searchsorted(times, t, side='right') - 1
        seg = max(0, min(seg, n - 2))
        dt = t - times[seg]
        result[idx] = values[seg] + b[seg] * dt + c[seg] * dt**2 + d[seg] * dt**3

    return result


def parse_script(script_path: str):
    """
    Parse a .roarm script file and extract waypoints and other commands.
    Returns: (waypoints, other_commands, config, start_pos)
    """
    path = Path(script_path)
    waypoints = []
    other_commands = []
    config = {"speed_scale": 1.0, "spd": 0, "acc": 10, "poll_hz": 25, "threshold": 0.5}
    start_pos = None

    with open(path, 'r') as f:
        lines = f.readlines()

    last_t = 0.0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse start position
        if line.startswith("#START_POS"):
            parts = line.split()[1:]
            vals = {}
            for p in parts:
                k, v = p.split("=")
                vals[k] = float(v)
            start_pos = {
                "b": vals.get("b", NORMALIZED_START_POSITION["b"]),
                "s": vals.get("s", NORMALIZED_START_POSITION["s"]),
                "e": vals.get("e", NORMALIZED_START_POSITION["e"]),
                "h": vals.get("h", NORMALIZED_START_POSITION["h"]),
            }
            continue

        if line.startswith("#") and not line.startswith("#CONFIG"):
            continue

        if line.startswith("#CONFIG"):
            _, kv = line.split(" ", 1)
            key, val = kv.split("=")
            if key == "speed_scale":
                config["speed_scale"] = float(val)
            elif key == "spd":
                config["spd"] = int(val)
            elif key == "acc":
                config["acc"] = int(val)
            elif key == "poll_hz":
                config["poll_hz"] = int(val)
            elif key == "threshold":
                config["threshold"] = float(val)
            continue

        parts = line.split()
        cmd = parts[0]

        if cmd == "MOVE":
            vals = {}
            for p in parts[1:]:
                k, v = p.split("=")
                vals[k] = float(v)
            wp = {
                "b": vals.get("b", 0),
                "s": vals.get("s", 0),
                "e": vals.get("e", 90),
                "h": vals.get("h", 180),
                "t": vals.get("t", last_t + 0.1),
            }
            last_t = wp["t"]
            waypoints.append(wp)

        elif cmd == "GRIPPER":
            # Parse gripper command with timestamp
            gripper_cmd = "OPEN"
            gripper_t = last_t
            for p in parts[1:]:
                if p.startswith("t="):
                    gripper_t = float(p.split("=")[1])
                elif p in ("OPEN", "CLOSE"):
                    gripper_cmd = p
            other_commands.append((gripper_t, "GRIPPER", gripper_cmd))

        elif cmd == "LED":
            other_commands.append((last_t, "LED", int(parts[1]) if len(parts) > 1 else 0))
        elif cmd == "WAIT":
            wt = float(parts[1]) if len(parts) > 1 else 1.0
            other_commands.append((last_t, "WAIT", wt))
            last_t += wt
        elif cmd == "FRAME":
            pass

    return waypoints, other_commands, config, start_pos


def play(script_path: str, port: str = None, speed_override: float = None,
         playback_hz: int = 30, use_monotone: bool = True, verify_start: bool = True,
         feedback_correction: bool = False):
    """
    Play a recorded .roarm script with smooth interpolated motion (Precision Edition).
    
    Key fixes:
    - Gripper commands use safe spd/acc values (not 0/0 which kills torque)
    - Re-asserts torque after every gripper command
    - Normalized start position with verification
    - Monotone cubic interpolation (no overshoot)
    - Lower MIN_CHANGE threshold for precision
    """
    path = Path(script_path)
    if not path.exists():
        print(f"  FEHLER: {path} nicht gefunden!")
        return

    waypoints, other_commands, config, start_pos = parse_script(script_path)

    if not waypoints:
        print("  FEHLER: Keine Wegpunkte im Skript!")
        return

    if start_pos is None:
        start_pos = NORMALIZED_START_POSITION.copy()

    speed_scale = speed_override if speed_override is not None else config["speed_scale"]

    print(f"  Abspielen: {path.name}")
    print(f"    {len(waypoints)} Wegpunkte, Speed-Scale: {speed_scale}")
    print(f"    Interpolation: {'Monotone Cubic' if use_monotone else 'Natural Cubic Spline'}")
    print(f"    Playback Hz: {playback_hz}")
    print(f"    Start: b={start_pos['b']:.2f} s={start_pos['s']:.2f} "
          f"e={start_pos['e']:.2f} h={start_pos['h']:.2f}")

    # Connect
    arm = RoArmPlayer(port=port)

    # === Explicitly enable torque on ALL servos ===
    print("  Torque AN...")
    arm.torque_on()
    time.sleep(0.3)

    # === Move to normalized start position ===
    print("  Fahre zur Startposition...")
    arm.send({"T": 122,
              "b": round(start_pos["b"], 4),
              "s": round(start_pos["s"], 4),
              "e": round(start_pos["e"], 4),
              "h": round(start_pos["h"], 4),
              "spd": 15, "acc": 5})
    time.sleep(2.5)

    # === Verify start position ===
    if verify_start:
        print("  Verifiziere Startposition...")
        is_ok, actual, max_error = arm.verify_position(start_pos)
        if is_ok:
            print(f"    ✓ OK (max Fehler: {max_error:.3f}°)")
        else:
            print(f"    ⚠ Abweichung: {max_error:.3f}°")
            if actual:
                print(f"      Ist:  b={actual['b']:.3f} s={actual['s']:.3f} "
                      f"e={actual['e']:.3f} h={actual['h']:.3f}")
            # Second attempt
            arm.send({"T": 122,
                      "b": round(start_pos["b"], 4),
                      "s": round(start_pos["s"], 4),
                      "e": round(start_pos["e"], 4),
                      "h": round(start_pos["h"], 4),
                      "spd": 10, "acc": 3})
            time.sleep(2.0)
            is_ok, actual, max_error = arm.verify_position(start_pos)
            if is_ok:
                print(f"    ✓ OK nach 2. Versuch ({max_error:.3f}°)")
            else:
                print(f"    ⚠ WARNUNG: immer noch {max_error:.3f}° daneben")
                resp = input("    Fortfahren? (j/n): ").strip().lower()
                if resp not in ('j', 'y', ''):
                    arm.close()
                    return

    # Extract arrays
    times = np.array([wp["t"] for wp in waypoints])
    b_vals = np.array([wp["b"] for wp in waypoints])
    s_vals = np.array([wp["s"] for wp in waypoints])
    e_vals = np.array([wp["e"] for wp in waypoints])
    h_vals = np.array([wp["h"] for wp in waypoints])

    total_duration = (times[-1] - times[0]) / max(speed_scale, 0.01)

    dt = 1.0 / playback_hz
    num_samples = max(1, int(total_duration / dt))
    sample_times_original = np.linspace(times[0], times[-1], num_samples)

    # Interpolate
    print("  Interpoliere Trajektorie...")
    interp_func = monotone_cubic_interpolate if use_monotone else cubic_interpolate
    b_interp = interp_func(times, b_vals, sample_times_original)
    s_interp = interp_func(times, s_vals, sample_times_original)
    e_interp = interp_func(times, e_vals, sample_times_original)
    h_interp = interp_func(times, h_vals, sample_times_original)

    # Clamp to recorded range + small margin
    b_interp = np.clip(b_interp, b_vals.min() - 1.0, b_vals.max() + 1.0)
    s_interp = np.clip(s_interp, s_vals.min() - 1.0, s_vals.max() + 1.0)
    e_interp = np.clip(e_interp, e_vals.min() - 1.0, e_vals.max() + 1.0)
    h_interp = np.clip(h_interp, h_vals.min() - 1.0, h_vals.max() + 1.0)

    print(f"    {num_samples} Punkte über {total_duration:.1f}s")

    # Move to first waypoint
    print("  Fahre zum ersten Wegpunkt...")
    arm.send({"T": 122,
              "b": round(float(b_interp[0]), 4),
              "s": round(float(s_interp[0]), 4),
              "e": round(float(e_interp[0]), 4),
              "h": round(float(h_interp[0]), 4),
              "spd": 20, "acc": 10})
    time.sleep(1.5)

    print("  ▶ Wiedergabe läuft...")

    other_commands.sort(key=lambda x: x[0])
    other_cmd_idx = 0

    playback_start = time.time()
    last_b, last_s, last_e, last_h = float(b_interp[0]), float(s_interp[0]), float(e_interp[0]), float(h_interp[0])

    MIN_CHANGE = 0.02  # degrees - much lower for precision

    # Track torque re-assertions
    torque_reassert_needed = False

    for i in range(num_samples):
        target_time = playback_start + (i * dt)

        # Handle other commands at their timestamps
        current_script_time = sample_times_original[i]
        while other_cmd_idx < len(other_commands):
            cmd_time, cmd_type, cmd_data = other_commands[other_cmd_idx]
            if cmd_time <= current_script_time:
                if cmd_type == "GRIPPER":
                    print(f"    [{current_script_time:.1f}s] GRIPPER {cmd_data}")
                    if cmd_data == "OPEN":
                        arm.gripper_open()
                    else:
                        arm.gripper_close()
                    # CRITICAL: Wait for gripper, then RE-ASSERT TORQUE
                    time.sleep(0.5)
                    arm.torque_on()
                    time.sleep(0.1)
                    playback_start += 0.6  # Account for gripper + torque time
                    torque_reassert_needed = False
                elif cmd_type == "LED":
                    arm.send({"T": 114, "led": cmd_data})
                elif cmd_type == "WAIT":
                    scaled_wait = cmd_data / max(speed_scale, 0.1)
                    time.sleep(scaled_wait)
                    playback_start += scaled_wait
                other_cmd_idx += 1
            else:
                break

        # Send position command
        target_b = float(b_interp[i])
        target_s = float(s_interp[i])
        target_e = float(e_interp[i])
        target_h = float(h_interp[i])

        db = abs(target_b - last_b)
        ds = abs(target_s - last_s)
        de = abs(target_e - last_e)
        dh = abs(target_h - last_h)

        if db >= MIN_CHANGE or ds >= MIN_CHANGE or de >= MIN_CHANGE or dh >= MIN_CHANGE:
            arm.send({"T": 122,
                      "b": round(target_b, 4),
                      "s": round(target_s, 4),
                      "e": round(target_e, 4),
                      "h": round(target_h, 4),
                      "spd": 0, "acc": 0})
            last_b, last_s, last_e, last_h = target_b, target_s, target_e, target_h

        # Precise timing
        now = time.time()
        sleep_time = target_time - now
        if sleep_time > 0:
            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.002)
            while time.time() < target_time:
                pass

    # === Done - ensure torque stays on ===
    print("  ✓ Fertig!")
    arm.torque_on()
    time.sleep(0.3)
    arm.send({"T": 114, "led": 0})
    time.sleep(0.5)
    arm.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm .roarm Skript abspielen (Precision Edition)")
    p.add_argument("script", type=str, help=".roarm Datei")
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--speed", type=float, default=None,
                   help="Speed override (0.5=halb, 2.0=doppelt)")
    p.add_argument("--hz", type=int, default=30,
                   help="Playback Hz (default: 30)")
    p.add_argument("--linear", action="store_true",
                   help="Use natural cubic spline instead of monotone")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip start position verification")
    p.add_argument("--feedback", action="store_true",
                   help="Enable feedback correction during playback")
    args = p.parse_args()
    play(args.script, port=args.port, speed_override=args.speed,
         playback_hz=args.hz, use_monotone=not args.linear,
         verify_start=not args.no_verify, feedback_correction=args.feedback)


if __name__ == "__main__":
    main()

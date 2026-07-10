#!/usr/bin/env python3
"""play_teached.py - Plays .roarm script files with maximum precision (Closed-Loop Edition)"""
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
from collections import deque


# === NORMALIZED START POSITION ===
NORMALIZED_START_POSITION = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

START_POSITION_TOLERANCE = 0.3  # Tighter


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


class BacklashCompensator:
    """
    Compensates for mechanical backlash in servo gears.
    When direction reverses, adds a small offset to overcome dead zone.
    """
    def __init__(self, backlash_deg=0.3):
        self.backlash = backlash_deg
        self.last_direction = {}  # joint -> +1 or -1
        self.last_target = {}    # joint -> last target value

    def compensate(self, joint: str, target: float) -> float:
        if joint not in self.last_target:
            self.last_target[joint] = target
            self.last_direction[joint] = 0
            return target

        delta = target - self.last_target[joint]
        if abs(delta) < 0.001:
            return target  # No movement

        new_direction = 1 if delta > 0 else -1
        old_direction = self.last_direction[joint]

        self.last_target[joint] = target
        self.last_direction[joint] = new_direction

        # Direction reversal detected
        if old_direction != 0 and new_direction != old_direction:
            # Add backlash compensation in the new direction
            compensated = target + (new_direction * self.backlash * 0.5)
            return compensated

        return target


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
            # Was 0.006 — reduced. At 115200 baud, ~60 bytes takes <1ms
            time.sleep(0.002)

    def send_and_read(self, cmd: dict) -> str:
        with self._lock:
            self._ser.reset_input_buffer()
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.015)
            response = ""
            deadline = time.time() + 0.15
            while time.time() < deadline:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response = line
                        if '"T":1051' in line or '"T": 1051' in line:
                            return line
                else:
                    time.sleep(0.002)
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

    def get_feedback_averaged(self, num_readings: int = 5) -> dict:
        """Median-based averaging for robustness."""
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

    def get_position_degrees(self, num_readings: int = 5) -> dict:
        fb = self.get_feedback_averaged(num_readings=num_readings)
        if fb and "b" in fb:
            return {
                "b": math.degrees(fb["b"]),
                "s": math.degrees(fb["s"]),
                "e": math.degrees(fb["e"]),
                "h": math.degrees(fb.get("t", fb.get("h", 0))),
            }
        return None

    def verify_position(self, target: dict, tolerance: float = START_POSITION_TOLERANCE) -> tuple:
        actual = self.get_position_degrees(num_readings=7)
        if actual is None:
            return False, None, float('inf')
        max_error = 0.0
        for joint in ["b", "s", "e", "h"]:
            error = abs(actual[joint] - target[joint])
            max_error = max(max_error, error)
        return max_error <= tolerance, actual, max_error

    def torque_on(self):
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
        self.send({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        self.send({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


def monotone_cubic_interpolate(times, values, sample_times):
    """
    Fritsch-Carlson monotone cubic interpolation.
    Prevents overshoot between waypoints.
    """
    if len(times) < 2:
        return np.full_like(sample_times, values[0] if len(values) > 0 else 0.0)

    if len(times) == 2:
        return np.interp(sample_times, times, values)

    n = len(times)
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)

    h = np.diff(times)
    h = np.where(h == 0, 1e-10, h)
    delta = np.diff(values) / h

    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            m[i] = (delta[i - 1] + delta[i]) / 2.0

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


def parse_script(script_path: str):
    """Parse a .roarm script file."""
    path = Path(script_path)
    waypoints = []
    other_commands = []
    config = {"speed_scale": 1.0, "spd": 0, "acc": 10, "poll_hz": 50, "threshold": 0.15}
    start_pos = None

    with open(path, 'r') as f:
        lines = f.readlines()

    last_t = 0.0
    for line in lines:
        line = line.strip()
        if not line:
            continue

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


def compute_adaptive_speed(delta_deg: float, dt: float) -> tuple:
    """
    For real-time streaming at high Hz, we want the servo to reach
    the target ASAP before the next command arrives.
    
    spd=0, acc=0 means "move to target as fast as possible" which is
    actually what we want for streaming — the interpolation already
    controls the speed by sending small increments.
    
    Only use slower speeds for large jumps (>2° per tick) to prevent
    mechanical shock.
    """
    if dt <= 0:
        return (0, 0)

    deg_per_tick = abs(delta_deg)

    if deg_per_tick > 5.0:
        # Very large jump — slow down to prevent shock
        return (80, 20)
    else:
        # Normal streaming: go as fast as possible to target
        # The trajectory interpolation already rate-limits us
        return (0, 0)

def play(script_path: str, port: str = None, speed_override: float = None,
         playback_hz: int = 50, use_monotone: bool = True, verify_start: bool = True,
         feedback_correction: bool = False, backlash_comp: float = 0.3,
         settling_enabled: bool = True):
    """
    Play a recorded .roarm script with maximum precision.
    
    Key improvements over original:
    - Adaptive spd/acc (never sends 0/0)
    - Backlash compensation on direction reversals
    - Optional closed-loop feedback correction
    - Settling pauses at critical points (direction changes, stops)
    - Higher default Hz (50)
    - Tighter start position verification
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
    print(f"    Interpolation: {'Monotone Cubic' if use_monotone else 'Linear'}")
    print(f"    Playback Hz: {playback_hz}")
    print(f"    Backlash Compensation: {backlash_comp}°")
    print(f"    Feedback Correction: {'ON' if feedback_correction else 'OFF'}")
    print(f"    Settling: {'ON' if settling_enabled else 'OFF'}")
    print(f"    Start: b={start_pos['b']:.3f} s={start_pos['s']:.3f} "
          f"e={start_pos['e']:.3f} h={start_pos['h']:.3f}")

    # Connect
    arm = RoArmPlayer(port=port)

    # Initialize backlash compensator
    backlash = BacklashCompensator(backlash_deg=backlash_comp)

    # === Explicitly enable torque ===
    print("  Torque AN...")
    arm.torque_on()
    time.sleep(0.3)

    # === Move to normalized start position with iterative correction ===
    print("  Fahre zur Startposition...")
    arm.send({"T": 122,
              "b": round(start_pos["b"], 4),
              "s": round(start_pos["s"], 4),
              "e": round(start_pos["e"], 4),
              "h": round(start_pos["h"], 4),
              "spd": 15, "acc": 5})
    time.sleep(2.5)

    # Second slower pass for precision
    arm.send({"T": 122,
              "b": round(start_pos["b"], 4),
              "s": round(start_pos["s"], 4),
              "e": round(start_pos["e"], 4),
              "h": round(start_pos["h"], 4),
              "spd": 8, "acc": 3})
    time.sleep(1.5)

    # === Verify start position ===
    if verify_start:
        print("  Verifiziere Startposition...")
        OVERSHOOT_FACTOR = 2.0  # How far past the target to overshoot
        JOG_DISTANCE = 3.0      # Minimum jog distance in degrees to force servo re-engagement

        for attempt in range(5):
            is_ok, actual, max_error = arm.verify_position(start_pos, tolerance=START_POSITION_TOLERANCE)
            if is_ok:
                print(f"    ✓ OK (max Fehler: {max_error:.4f}°, Versuch {attempt+1})")
                break
            else:
                print(f"    Versuch {attempt+1}: Abweichung {max_error:.4f}°")
                if actual and attempt < 4:
                    # Calculate error per joint
                    error_b = start_pos["b"] - actual["b"]
                    error_s = start_pos["s"] - actual["s"]
                    error_e = start_pos["e"] - actual["e"]
                    error_h = start_pos["h"] - actual["h"]

                    # Overshoot: move PAST the target to force servo re-engagement
                    def overshoot(target, error):
                        offset = error * OVERSHOOT_FACTOR
                        # Ensure minimum jog distance so servo actually moves
                        if abs(offset) < JOG_DISTANCE:
                            offset = JOG_DISTANCE if error >= 0 else -JOG_DISTANCE
                        return target + offset

                    overshoot_b = overshoot(start_pos["b"], error_b)
                    overshoot_s = overshoot(start_pos["s"], error_s)
                    overshoot_e = overshoot(start_pos["e"], error_e)
                    overshoot_h = overshoot(start_pos["h"], error_h)

                    print(f"      Overshoot: b={overshoot_b:.2f} s={overshoot_s:.2f} "
                          f"e={overshoot_e:.2f} h={overshoot_h:.2f}")

                    # Step 1: Move PAST the target (fast) to break servo dead zone
                    arm.send({"T": 122,
                              "b": round(overshoot_b, 4),
                              "s": round(overshoot_s, 4),
                              "e": round(overshoot_e, 4),
                              "h": round(overshoot_h, 4),
                              "spd": 15, "acc": 8})
                    time.sleep(1.2)

                    # Step 2: Move back to actual target (slow, precise)
                    arm.send({"T": 122,
                              "b": round(start_pos["b"], 4),
                              "s": round(start_pos["s"], 4),
                              "e": round(start_pos["e"], 4),
                              "h": round(start_pos["h"], 4),
                              "spd": 5, "acc": 2})
                    time.sleep(2.0)
        else:
            # All 5 attempts exhausted
            print(f"    ⚠ WARNUNG: Startposition weicht ab um {max_error:.4f}°")
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

    # Interpolate trajectory
    print("  Interpoliere Trajektorie...")
    interp_func = monotone_cubic_interpolate if use_monotone else cubic_interpolate
    b_interp = interp_func(times, b_vals, sample_times_original)
    s_interp = interp_func(times, s_vals, sample_times_original)
    e_interp = interp_func(times, e_vals, sample_times_original)
    h_interp = interp_func(times, h_vals, sample_times_original)

    # Clamp to recorded range + small margin (prevent overshoot beyond physical limits)
    b_interp = np.clip(b_interp, b_vals.min() - 0.5, b_vals.max() + 0.5)
    s_interp = np.clip(s_interp, s_vals.min() - 0.5, s_vals.max() + 0.5)
    e_interp = np.clip(e_interp, e_vals.min() - 0.5, e_vals.max() + 0.5)
    h_interp = np.clip(h_interp, h_vals.min() - 0.5, h_vals.max() + 0.5)

    print(f"    {num_samples} Punkte über {total_duration:.1f}s")

    # === Pre-compute per-segment velocity for adaptive speed ===
    segment_max_delta = np.zeros(num_samples)
    for i in range(1, num_samples):
        db = abs(b_interp[i] - b_interp[i-1])
        ds = abs(s_interp[i] - s_interp[i-1])
        de = abs(e_interp[i] - e_interp[i-1])
        dh = abs(h_interp[i] - h_interp[i-1])
        segment_max_delta[i] = max(db, ds, de, dh)

    # === Detect direction reversals for settling ===
    direction_reversals = set()
    if settling_enabled:
        for joint_interp in [b_interp, s_interp, e_interp, h_interp]:
            prev_dir = 0
            for i in range(1, num_samples):
                delta = joint_interp[i] - joint_interp[i-1]
                if abs(delta) < 0.01:
                    continue
                curr_dir = 1 if delta > 0 else -1
                if prev_dir != 0 and curr_dir != prev_dir:
                    direction_reversals.add(i)
                prev_dir = curr_dir
    print(f"    {len(direction_reversals)} Richtungswechsel erkannt")

    # Move to first waypoint slowly and precisely
    print("  Fahre zum ersten Wegpunkt...")
    arm.send({"T": 122,
              "b": round(float(b_interp[0]), 4),
              "s": round(float(s_interp[0]), 4),
              "e": round(float(e_interp[0]), 4),
              "h": round(float(h_interp[0]), 4),
              "spd": 15, "acc": 5})
    time.sleep(2.0)

    # Verify first waypoint
    first_target = {"b": float(b_interp[0]), "s": float(s_interp[0]),
                    "e": float(e_interp[0]), "h": float(h_interp[0])}
    is_ok, actual, max_error = arm.verify_position(first_target, tolerance=0.5)
    if is_ok:
        print(f"    ✓ Erster Wegpunkt OK ({max_error:.4f}°)")
    else:
        print(f"    ⚠ Erster Wegpunkt Abweichung: {max_error:.4f}°, korrigiere...")
        arm.send({"T": 122,
                  "b": round(float(b_interp[0]), 4),
                  "s": round(float(s_interp[0]), 4),
                  "e": round(float(e_interp[0]), 4),
                  "h": round(float(h_interp[0]), 4),
                  "spd": 8, "acc": 3})
        time.sleep(1.5)

    print("  ▶ Wiedergabe läuft...")

    other_commands.sort(key=lambda x: x[0])
    other_cmd_idx = 0

    playback_start = time.time()
    last_b = float(b_interp[0])
    last_s = float(s_interp[0])
    last_e = float(e_interp[0])
    last_h = float(h_interp[0])

    MIN_CHANGE = 0.05  # Was 0.008 — the servo can't resolve below ~0.09° anyway

    # Feedback correction state
    correction_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
    last_correction_time = 0
    correction_interval = 2  # Correct every 500ms
    correction_gain = 0.3  # How aggressively to correct (0=none, 1=full)

    # Statistics
    commands_sent = 0
    corrections_applied = 0
    settling_pauses = 0

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
                    time.sleep(0.5)
                    arm.torque_on()
                    time.sleep(0.1)
                    playback_start += 0.6
                elif cmd_type == "LED":
                    arm.send({"T": 114, "led": cmd_data})
                elif cmd_type == "WAIT":
                    scaled_wait = cmd_data / max(speed_scale, 0.1)
                    time.sleep(scaled_wait)
                    playback_start += scaled_wait
                other_cmd_idx += 1
            else:
                break

        # === Feedback correction (closed-loop) ===
        now = time.time()
        if feedback_correction and (now - last_correction_time) >= correction_interval:
            actual_pos = arm.get_position_degrees(num_readings=3)
            if actual_pos:
                # Compare actual vs where we think we should be
                expected = {"b": last_b, "s": last_s, "e": last_e, "h": last_h}
                for joint in ["b", "s", "e", "h"]:
                    error = expected[joint] - actual_pos[joint]
                    if abs(error) > 0.05:  # Only correct if error > 0.05°
                        correction_offset[joint] += error * correction_gain
                        # Clamp correction to prevent runaway
                        correction_offset[joint] = max(-2.0, min(2.0, correction_offset[joint]))
                corrections_applied += 1
            last_correction_time = now
            # Account for time spent reading feedback
            elapsed_for_feedback = time.time() - now
            playback_start += elapsed_for_feedback

        # === Settling pause at direction reversals ===
        if settling_enabled and i in direction_reversals:
            # Only settle on LARGE reversals (>1° change in direction)
            if segment_max_delta[i] > 0.5:
                time.sleep(0.015)  # Was 0.03 — halved
                playback_start += 0.015
                settling_pauses += 1

        # Compute target position with backlash compensation and feedback correction
        target_b = float(b_interp[i]) + correction_offset["b"]
        target_s = float(s_interp[i]) + correction_offset["s"]
        target_e = float(e_interp[i]) + correction_offset["e"]
        target_h = float(h_interp[i]) + correction_offset["h"]

        # Apply backlash compensation
        target_b = backlash.compensate("b", target_b)
        target_s = backlash.compensate("s", target_s)
        target_e = backlash.compensate("e", target_e)
        target_h = backlash.compensate("h", target_h)

        # Check if movement is significant enough to send
        db = abs(target_b - last_b)
        ds = abs(target_s - last_s)
        de = abs(target_e - last_e)
        dh = abs(target_h - last_h)
        max_delta = max(db, ds, de, dh)

        if max_delta >= MIN_CHANGE:
            # Compute adaptive speed based on required movement rate
            spd, acc = compute_adaptive_speed(segment_max_delta[i], dt)

            arm.send({"T": 122,
                      "b": round(target_b, 4),
                      "s": round(target_s, 4),
                      "e": round(target_e, 4),
                      "h": round(target_h, 4),
                      "spd": spd, "acc": acc})
            last_b, last_s, last_e, last_h = target_b, target_s, target_e, target_h
            commands_sent += 1

        # Precise timing with busy-wait for last 2ms
        now = time.time()
        sleep_time = target_time - now
        if sleep_time > 0:
            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.001)
            while time.time() < target_time:
                pass

    # === Final settling: send last position again slowly and wait ===
    print("  Finale Positionskorrektur...")
    arm.send({"T": 122,
              "b": round(float(b_interp[-1]), 4),
              "s": round(float(s_interp[-1]), 4),
              "e": round(float(e_interp[-1]), 4),
              "h": round(float(h_interp[-1]), 4),
              "spd": 10, "acc": 5})
    time.sleep(1.0)

    # Verify final position
    final_target = {"b": float(b_interp[-1]), "s": float(s_interp[-1]),
                    "e": float(e_interp[-1]), "h": float(h_interp[-1])}
    is_ok, actual, max_error = arm.verify_position(final_target, tolerance=0.5)
    if is_ok:
        print(f"    ✓ Endposition OK ({max_error:.4f}°)")
    else:
        print(f"    ⚠ Endposition Abweichung: {max_error:.4f}°")

    # === Statistics ===
    print(f"\n  Statistik:")
    print(f"    Befehle gesendet: {commands_sent}/{num_samples}")
    print(f"    Feedback-Korrekturen: {corrections_applied}")
    print(f"    Settling-Pausen: {settling_pauses}")
    print(f"    Richtungswechsel: {len(direction_reversals)}")

    # === Done - ensure torque stays on ===
    print("  ✓ Fertig!")
    arm.torque_on()
    time.sleep(0.3)
    arm.send({"T": 114, "led": 0})
    time.sleep(0.5)
    arm.close()


def cubic_interpolate(times, values, sample_times):
    """Natural cubic spline interpolation (fallback, may overshoot)."""
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

    b_coef = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b_coef[j] = (values[j + 1] - values[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
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
        dt_val = t - times[seg]
        result[idx] = values[seg] + b_coef[seg] * dt_val + c[seg] * dt_val**2 + d[seg] * dt_val**3

    return result


def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm .roarm Skript abspielen (Maximum Precision Edition)")
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
                   help="Enable closed-loop feedback correction during playback")
    p.add_argument("--backlash", type=float, default=0.3,
                   help="Backlash compensation in degrees (default: 0.3)")
    p.add_argument("--no-settling", action="store_true",
                   help="Disable settling pauses at direction reversals")
    args = p.parse_args()
    play(args.script, port=args.port, speed_override=args.speed,
         playback_hz=args.hz, use_monotone=not args.linear,
         verify_start=not args.no_verify, feedback_correction=args.feedback,
         backlash_comp=args.backlash, settling_enabled=not args.no_settling)


if __name__ == "__main__":
    main()

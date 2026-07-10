#!/usr/bin/env python3
"""play_teached.py - Plays .roarm script files with smooth interpolated motion"""
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
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=1.0, dsrdtr=None)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        print(f"  Verbunden: {port}")

    def send(self, cmd: dict):
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


def cubic_interpolate(times, values, sample_times):
    """
    Attempt cubic spline interpolation. Falls back to linear if not enough points.
    times: array of recorded timestamps
    values: array of recorded joint values
    sample_times: array of desired output timestamps
    Returns: interpolated values at sample_times
    """
    if len(times) < 2:
        return np.full_like(sample_times, values[0] if len(values) > 0 else 0.0)

    if len(times) == 2:
        # Linear interpolation for 2 points
        return np.interp(sample_times, times, values)

    # Natural cubic spline interpolation
    n = len(times)
    h = np.diff(times)
    
    # Avoid division by zero
    h = np.where(h == 0, 1e-6, h)
    
    # Build tridiagonal system for natural cubic spline
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

    # Evaluate spline at sample_times
    result = np.zeros_like(sample_times)
    for idx, t in enumerate(sample_times):
        # Clamp to range
        if t <= times[0]:
            result[idx] = values[0]
            continue
        if t >= times[-1]:
            result[idx] = values[-1]
            continue
        # Find segment
        seg = np.searchsorted(times, t, side='right') - 1
        seg = max(0, min(seg, n - 2))
        dt = t - times[seg]
        result[idx] = values[seg] + b[seg] * dt + c[seg] * dt**2 + d[seg] * dt**3

    return result


def parse_script(script_path: str):
    """
    Parse a .roarm script file and extract waypoints and other commands.
    Returns: (waypoints, other_commands, config)
        waypoints: list of dicts with keys b, s, e, h, t
        other_commands: list of (timestamp, cmd_type, cmd_data) for non-MOVE commands
        config: dict of config values
    """
    path = Path(script_path)
    waypoints = []
    other_commands = []
    config = {"speed_scale": 1.0, "spd": 0, "acc": 10}

    with open(path, 'r') as f:
        lines = f.readlines()

    last_t = 0.0
    for line in lines:
        line = line.strip()
        if not line or (line.startswith("#") and not line.startswith("#CONFIG")):
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
            other_commands.append((last_t, "GRIPPER", parts[1] if len(parts) > 1 else "OPEN"))
        elif cmd == "LED":
            other_commands.append((last_t, "LED", int(parts[1]) if len(parts) > 1 else 0))
        elif cmd == "WAIT":
            wt = float(parts[1]) if len(parts) > 1 else 1.0
            other_commands.append((last_t, "WAIT", wt))
            last_t += wt
        elif cmd == "FRAME":
            pass  # Skip frames during playback

    return waypoints, other_commands, config


def play(script_path: str, port: str = None, speed_override: float = None,
         playback_hz: int = 50):
    """
    Play a recorded .roarm script with smooth cubic-spline interpolation.
    
    The key insight: instead of sending recorded waypoints one-by-one and waiting,
    we interpolate a smooth trajectory at high frequency (50Hz) and stream
    micro-positions to the arm. This produces fluid, continuous motion that
    exactly mirrors how you moved the arm.
    """
    path = Path(script_path)
    if not path.exists():
        print(f"  FEHLER: {path} nicht gefunden!")
        return

    # Parse the script
    waypoints, other_commands, config = parse_script(script_path)

    if not waypoints:
        print("  FEHLER: Keine Wegpunkte im Skript!")
        return

    speed_scale = speed_override if speed_override is not None else config["speed_scale"]

    print(f"  Abspielen: {path.name}")
    print(f"    {len(waypoints)} Wegpunkte, Speed-Scale: {speed_scale}")

    # Connect to arm
    arm = RoArmPlayer(port=port)

    # Torque on
    arm.send({"T": 210, "cmd": 1})
    time.sleep(0.3)

    # Extract time and joint arrays from waypoints
    times = np.array([wp["t"] for wp in waypoints])
    b_vals = np.array([wp["b"] for wp in waypoints])
    s_vals = np.array([wp["s"] for wp in waypoints])
    e_vals = np.array([wp["e"] for wp in waypoints])
    h_vals = np.array([wp["h"] for wp in waypoints])

    # Total duration (scaled)
    total_duration = (times[-1] - times[0]) / max(speed_scale, 0.01)

    # Generate smooth sample times at playback_hz
    dt = 1.0 / playback_hz
    num_samples = max(1, int(total_duration / dt))
    sample_times_scaled = np.linspace(times[0], times[-1], num_samples)

    # Interpolate all joints using cubic splines
    print("  Interpoliere Trajektorie...")
    b_interp = cubic_interpolate(times, b_vals, sample_times_scaled)
    s_interp = cubic_interpolate(times, s_vals, sample_times_scaled)
    e_interp = cubic_interpolate(times, e_vals, sample_times_scaled)
    h_interp = cubic_interpolate(times, h_vals, sample_times_scaled)

    print(f"    {num_samples} interpolierte Punkte über {total_duration:.1f}s")
    print("  Starte Wiedergabe...")

    # Sort other commands by time for insertion during playback
    other_commands.sort(key=lambda x: x[0])
    other_cmd_idx = 0

    # Move to start position first (slowly)
    arm.send({"T": 122,
              "b": round(b_interp[0], 2),
              "s": round(s_interp[0], 2),
              "e": round(e_interp[0], 2),
              "h": round(h_interp[0], 2),
              "spd": 20, "acc": 10})
    time.sleep(1.0)  # Wait to reach start position

    # Stream interpolated positions in real-time
    playback_start = time.time()
    last_b, last_s, last_e, last_h = b_interp[0], s_interp[0], e_interp[0], h_interp[0]
    
    # Minimum change threshold to avoid flooding with identical commands
    MIN_CHANGE = 0.1  # degrees

    for i in range(num_samples):
        # Calculate when this sample should be sent
        target_time = playback_start + (i * dt)

        # Handle other commands (gripper, LED, wait) at their timestamps
        current_script_time = sample_times_scaled[i]
        while other_cmd_idx < len(other_commands):
            cmd_time, cmd_type, cmd_data = other_commands[other_cmd_idx]
            if cmd_time <= current_script_time:
                if cmd_type == "GRIPPER":
                    if cmd_data == "OPEN":
                        arm.send({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})
                    else:
                        arm.send({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})
                    time.sleep(0.3)
                    playback_start += 0.3  # Adjust timeline for gripper delay
                elif cmd_type == "LED":
                    arm.send({"T": 114, "led": cmd_data})
                elif cmd_type == "WAIT":
                    scaled_wait = cmd_data / max(speed_scale, 0.1)
                    time.sleep(scaled_wait)
                    playback_start += scaled_wait
                other_cmd_idx += 1
            else:
                break

        # Only send if position actually changed (avoid flooding serial)
        db = abs(b_interp[i] - last_b)
        ds = abs(s_interp[i] - last_s)
        de = abs(e_interp[i] - last_e)
        dh = abs(h_interp[i] - last_h)

        if db >= MIN_CHANGE or ds >= MIN_CHANGE or de >= MIN_CHANGE or dh >= MIN_CHANGE:
            # Send with spd=0 (maximum speed) and high acc so the arm
            # rushes to each micro-position instantly — since positions are
            # very close together, this creates smooth continuous motion
            arm.send({"T": 122,
                      "b": round(b_interp[i], 2),
                      "s": round(s_interp[i], 2),
                      "e": round(e_interp[i], 2),
                      "h": round(h_interp[i], 2),
                      "spd": 0, "acc": 0})
            last_b, last_s, last_e, last_h = b_interp[i], s_interp[i], e_interp[i], h_interp[i]

        # Precise timing: sleep until target time
        now = time.time()
        sleep_time = target_time - now
        if sleep_time > 0:
            # Use a spin-wait for the last 2ms for precision
            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.002)
            # Spin-wait for remaining time (more precise than sleep)
            while time.time() < target_time:
                pass

    print("  ✓ Fertig!")
    arm.send({"T": 114, "led": 0})
    time.sleep(0.5)
    arm.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm .roarm Skript abspielen (smooth)")
    p.add_argument("script", type=str, help=".roarm Datei")
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--speed", type=float, default=None,
                   help="Speed override (z.B. 0.5=halb, 2.0=doppelt)")
    p.add_argument("--hz", type=int, default=50,
                   help="Playback interpolation rate in Hz (default: 50)")
    args = p.parse_args()
    play(args.script, port=args.port, speed_override=args.speed, playback_hz=args.hz)


if __name__ == "__main__":
    main()

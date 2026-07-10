#!/usr/bin/env python3
"""play_roarm.py - Spielt .roarm Skriptdateien ab"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
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
            time.sleep(0.05)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


def play(script_path: str, port: str = None, speed_override: float = None):
    path = Path(script_path)
    if not path.exists():
        print(f"  FEHLER: {path} nicht gefunden!")
        return

    arm = RoArmPlayer(port=port)

    # Torque on
    arm.send({"T": 210, "cmd": 1})
    time.sleep(0.3)

    # Defaults
    spd = 0
    acc = 10
    speed_scale = 1.0

    with open(path, 'r') as f:
        lines = f.readlines()

    print(f"  Abspielen: {path.name} ({len(lines)} Zeilen)")

    for i, line in enumerate(lines):
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

        # Override speed_scale from CLI
        if speed_override is not None:
            speed_scale = speed_override

        parts = line.split()
        cmd = parts[0]

        if cmd == "MOVE":
            vals = {}
            for p in parts[1:]:
                k, v = p.split("=")
                vals[k] = float(v)
            actual_spd = max(0, int(spd * speed_scale)) if spd > 0 else 0
            actual_acc = max(1, int(acc * speed_scale)) if acc > 0 else 0
            arm.send({"T": 122, "b": vals.get("b", 0), "s": vals.get("s", 0),
                      "e": vals.get("e", 90), "h": vals.get("h", 180),
                      "spd": actual_spd, "acc": actual_acc})
            time.sleep(0.1 / max(speed_scale, 0.1))

        elif cmd == "GRIPPER":
            action = parts[1] if len(parts) > 1 else "OPEN"
            if action == "OPEN":
                arm.send({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})
            else:
                arm.send({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})
            time.sleep(0.5)

        elif cmd == "LED":
            brightness = int(parts[1]) if len(parts) > 1 else 0
            arm.send({"T": 114, "led": brightness})

        elif cmd == "WAIT":
            wt = float(parts[1]) if len(parts) > 1 else 1.0
            time.sleep(wt / max(speed_scale, 0.1))

        elif cmd == "FRAME":
            pass  # Skip during playback

    print("  Fertig!")
    arm.send({"T": 114, "led": 0})
    arm.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm .roarm Skript abspielen")
    p.add_argument("script", type=str, help=".roarm Datei")
    p.add_argument("--port", type=str, default=None)
    p.add_argument("--speed", type=float, default=None,
                   help="Speed override (z.B. 0.5=halb, 2.0=doppelt)")
    args = p.parse_args()
    play(args.script, port=args.port, speed_override=args.speed)

if __name__ == "__main__":
    main()

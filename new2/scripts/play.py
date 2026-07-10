#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
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
        print("uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

"""
DSL Player — Spielt .roarm-Skripte ab (Step-by-Step oder durchgehend).

Usage:
    python3 play.py mein_skript.roarm
    python3 play.py mein_skript.roarm --step
    python3 play.py mein_skript.roarm --model my_yolo.pt
"""

import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from roarm_lib.hardware import RoArmHardware
from roarm_lib.vision import VisionSystem
from roarm_lib.dsl import DSLInterpreter


def main():
    parser = argparse.ArgumentParser(description="RoArm DSL Player")
    parser.add_argument("script", type=str, help=".roarm-Datei zum Abspielen")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--model", type=str, default=None, help="YOLO-Modell (für 'when see')")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index")
    parser.add_argument("--step", action="store_true", help="Step-by-Step Modus")
    parser.add_argument("--dry-run", action="store_true", help="Nur parsen, nicht ausführen")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"✗ Datei nicht gefunden: {script_path}")
        sys.exit(1)

    # Hardware
    hw = RoArmHardware(port=args.port)

    # Vision (optional)
    vision = None
    if args.model:
        vision = VisionSystem(camera_index=args.camera, model_path=args.model)

    # Interpreter
    interp = DSLInterpreter(hw, vision)
    interp.load_file(script_path)

    # Callbacks
    def on_step(cmd, state):
        print(f"  [{cmd.line_number}] {cmd.raw_line}")
        if args.step:
            input("  → Enter für nächsten Schritt...")

    def on_print(msg):
        print(f"  💬 {msg}")

    interp.on_step = on_step
    interp.on_print = on_print

    if args.dry_run:
        print(f"\n  Dry-Run: {script_path.name}")
        print(f"  Funktionen: {list(interp.functions.keys())}")
        print(f"  (Keine Ausführung)")
        return

    # Ausführen
    print(f"\n  ▶ Starte: {script_path.name}")
    print(f"    Modus: {'Step-by-Step' if args.step else 'Durchgehend'}")
    print()

    try:
        interp.run(step_mode=args.step)
    except KeyboardInterrupt:
        print("\n  ⏹ Abgebrochen")
    finally:
        hw.park()
        time.sleep(1.0)
        hw.disconnect()
        if vision:
            vision.release()

    print("  ✓ Fertig")


if __name__ == "__main__":
    main()

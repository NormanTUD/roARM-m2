#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "pyserial",
#     "torch",
#     "ultralytics",
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
DSL Recorder — Aufzeichnung als .roarm-Dateien + JPG-Export.

Steuerung per Pfeiltasten, Funktions-Definition per Knopfdruck.
Kein YOLO-Modell nötig (optional).

Usage:
    python3 record.py
    python3 record.py --model my_yolo.pt
    python3 record.py --output my_recordings --camera 0
"""

import argparse
from pathlib import Path

# Füge parent zum path hinzu für roarm_lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from roarm_lib.recorder import SessionRecorder


def main():
    parser = argparse.ArgumentParser(description="RoArm DSL Recorder")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index")
    parser.add_argument("--model", type=str, default=None,
                        help="YOLO-Modell (optional! Kann auch ohne aufzeichnen)")
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO Confidence")
    parser.add_argument("--output", type=str, default="recordings", help="Output-Verzeichnis")
    parser.add_argument("--save-images-every", type=int, default=10,
                        help="Jeden N-ten Frame als JPG speichern")
    args = parser.parse_args()

    recorder = SessionRecorder(
        output_dir=args.output,
        camera_index=args.camera,
        model_path=args.model,  # None ist OK!
        confidence=args.confidence,
        port=args.port,
        save_images_every_n=args.save_images_every,
    )
    recorder.run()


if __name__ == "__main__":
    main()


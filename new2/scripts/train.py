#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
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
BBox-Policy Trainer — Trainiert ein NN das NUR Bounding Boxes sieht.

Das ist der Kern der Idee:
- YOLO erkennt Objekte → liefert BBoxes (Position, Größe)
- Das NN sieht NUR diese BBoxes + den Arm-State
- Kein rohes Bild → extrem kleiner Input → schnelles Lernen
- Invariant gegenüber Textur, Beleuchtung, Hintergrund

Faserbündel-Analogie:
- Basis-Mannigfaltigkeit = BBox-Raum (wo sind die Objekte?)
- Faser = Raum der nützlichen Bewegungen
- Wenn Position bekannt → Bewegung kann invariant transformiert werden

Usage:
    python3 train.py recordings/ --class-names charger wall_marker
    python3 train.py recordings/ --epochs 300 --chunk-size 15
"""

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from roarm_lib.policy import train_policy


def main():
    parser = argparse.ArgumentParser(description="BBox-Policy Training")
    parser.add_argument("recordings_dir", type=str, help="Recordings-Verzeichnis")
    parser.add_argument("--output", type=str, default="trained_bbox_policy",
                        help="Output-Verzeichnis")
    parser.add_argument("--epochs", type=int, default=200, help="Trainings-Epochen")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch-Größe")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--chunk-size", type=int, default=10, help="Action-Chunk Größe")
    parser.add_argument("--max-objects", type=int, default=5,
                        help="Max. Objekte die das NN sieht")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="YOLO-Klassen (z.B. charger wall_marker table)")

    args = parser.parse_args()

    model = train_policy(
        recordings_dir=args.recordings_dir,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        chunk_size=args.chunk_size,
        max_objects=args.max_objects,
        class_names=args.class_names,
    )

    if model:
        print(f"\n  Nächster Schritt:")
        print(f"    python3 run_policy.py {args.output}/bbox_policy_best.pt")


if __name__ == "__main__":
    main()


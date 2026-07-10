#!/usr/bin/env python3
"""
scripts/annotate.py — Export camera frames as JPGs for YOLO annotation.

Usage:
    # Live capture: take photos while moving the arm
    python scripts/annotate.py --live --output dataset/images/

    # From recording: extract frames from a .roarm recording session
    python scripts/annotate.py --from-recording recordings/frames/recording_20240101/
    
    # Batch: capture N frames at different arm positions
    python scripts/annotate.py --batch 50 --output dataset/images/
"""

import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.vision import VisionPipeline


def live_capture(vision: VisionPipeline, output_dir: Path):
    """Capture frames interactively. Press SPACE to save, Q to quit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(output_dir.glob("*.jpg")))

    print(f"\n  Live Capture Mode")
    print(f"  Output: {output_dir}")
    print(f"  Existing frames: {count}")
    print(f"  SPACE = capture, Q = quit\n")

    while True:
        frame = vision.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Show with count
        display = frame.copy()
        cv2.putText(display, f"Frames: {count} | SPACE=Capture Q=Quit",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("YOLO Annotation Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            filepath = vision.export_frame_as_jpg(
                str(output_dir), prefix="capture"
            )
            if filepath:
                count += 1
                print(f"  ✓ Saved: {filepath}")

    cv2.destroyAllWindows()
    print(f"\n  Done! {count} frames in {output_dir}")
    print(f"  → Now annotate these in your YOLO annotation tool")
    print(f"  → Then train: yolo train data=dataset.yaml model=yolo11n.pt epochs=50")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export frames for YOLO annotation")
    parser.add_argument("--live", action="store_true", help="Live capture mode")
    parser.add_argument("--output", type=str, default="dataset/images/train",
                        help="Output directory for JPGs")
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    vision = VisionPipeline(
        model_path="yolo11n.pt",
        camera_index=args.camera,
    )

    if not vision.available:
        print("  ✗ Camera not available!")
        return

    if args.live:
        live_capture(vision, Path(args.output))

    vision.release()


if __name__ == "__main__":
    main()


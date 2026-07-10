#!/usr/bin/env python3
"""
scripts/replay.py — Step-by-step interpreter for .roarm programs.

Usage:
    python scripts/replay.py programs/pick_bottle.roarm
    python scripts/replay.py programs/pick_bottle.roarm --step  # Single-step mode
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.arm import RoArmController
from lib.vision import VisionPipeline
from lib.dsl import DSLParser, DSLInterpreter


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run .roarm programs")
    parser.add_argument("program", type=str, help="Path to .roarm file")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--step", action="store_true",
                        help="Single-step mode (press Enter for each step)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and print without executing")
    args = parser.parse_args()

    # Parse program
    dsl_parser = DSLParser()
    program = dsl_parser.parse_file(args.program)

    print(f"\n  Program: {args.program}")
    print(f"  Functions: {list(program.functions.keys())}")
    print(f"  Main steps: {len(program.main)}")
    print(f"  Defaults: {program.defaults}")

    if args.dry_run:
        print("\n  [DRY RUN] Program parsed successfully.")
        return

    # Setup hardware
    arm = RoArmController(port=args.port)
    vision = VisionPipeline(model_path=args.model, camera_index=args.camera)

    # Create interpreter
    interpreter = DSLInterpreter(arm, vision, defaults=program.defaults)

    # Callbacks
    def on_step(node, context):
        print(f"  [{context.step_count:3d}] {node.source_line}")
        if args.step:
            input("       Press Enter for next step...")

    def on_detect(detections):
        if detections:
            for d in detections:
                print(f"       → Detected: {d.class_name} ({d.confidence:.2f}) "
                      f"at ({d.cx:.2f}, {d.cy:.2f})")
        else:
            print("       → No detections")

    def on_error(error, node):
        print(f"  ✗ ERROR at line {node.line_number}: {error}")
        print(f"    Source: {node.source_line}")

    interpreter.on_step = on_step
    interpreter.on_detect = on_detect
    interpreter.on_error = on_error

    # Load and run
    interpreter.load(program)

    print(f"\n  ▶ Running program...")
    print(f"  {'─' * 50}")

    try:
        interpreter.run()
        print(f"\n  ✓ Program finished ({interpreter._context.step_count} steps)")
    except KeyboardInterrupt:
        print(f"\n  ■ Interrupted at step {interpreter._context.step_count}")
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
    finally:
        arm.park()
        arm.disconnect()
        vision.release()


if __name__ == "__main__":
    main()


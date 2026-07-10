#!/usr/bin/env python3
"""
scripts/record.py — Record teleoperation into .roarm DSL format.

Features:
- Keyboard teleop (same controls as before)
- Press F5 to start recording a function
- Press F5 again to end function recording (asks for name)
- Outputs .roarm file + JPG frames for YOLO annotation
- YOLO detections are recorded alongside movements
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.arm import RoArmController
from lib.vision import VisionPipeline
from lib.recorder import DSLRecorder


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record robot actions → .roarm DSL")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--output", type=str, default="recordings")
    parser.add_argument("--target", type=str, default=None,
                        help="Target class for YOLO (e.g. 'bottle')")
    args = parser.parse_args()

    # Setup
    arm = RoArmController(port=args.port)
    vision = VisionPipeline(
        model_path=args.model,
        camera_index=args.camera,
    )
    recorder = DSLRecorder(arm, vision, output_dir=args.output)

    print("\n╔══════════════════════════════════════════╗")
    print("║   RoArm DSL Recorder                    ║")
    print("╠══════════════════════════════════════════╣")
    print("║  R      = Start/Stop recording          ║")
    print("║  F5     = Start/End function recording  ║")
    print("║  Arrows = Move base/shoulder            ║")
    print("║  W/S    = Elbow up/down                 ║")
    print("║  A/D    = Hand left/right               ║")
    print("║  O/C    = Gripper open/close            ║")
    print("║  Q      = Quit & save                   ║")
    print("╚══════════════════════════════════════════╝\n")

    arm.home()
    time.sleep(2.0)

    running = True
    window_name = "RoArm Recorder"

    while running:
        # Get frame and detections
        frame = vision.get_frame()
        detections = vision.detect(frame, target_classes=[args.target] if args.target else None)

        # Record step if recording
        if recorder.is_recording:
            arm_state = {
                'base_deg': arm.state.base_deg,
                'shoulder_deg': arm.state.shoulder_deg,
                'elbow_deg': arm.state.elbow_deg,
                'hand_deg': arm.state.hand_deg,
                'gripper_open': arm.state.gripper_open,
            }
            # Build DSL command from current action
            command = f"move base={arm.state.base_deg:.1f} shoulder={arm.state.shoulder_deg:.1f} elbow={arm.state.elbow_deg:.1f} hand={arm.state.hand_deg:.1f}"
            recorder.record_step(
                command=command,
                arm_state=arm_state,
                detections=[{'class': d.class_name, 'cx': d.cx, 'cy': d.cy,
                            'width': d.width, 'height': d.height}
                           for d in detections],
                frame=frame,
            )

        # Annotate frame for display
        if frame is not None:
            # Draw detections
            h, w = frame.shape[:2]
            for det in detections:
                x1 = int((det.cx - det.width/2) * w)
                y1 = int((det.cy - det.height/2) * h)
                x2 = int((det.cx + det.width/2) * w)
                y2 = int((det.cy + det.height/2) * h)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Status bar
            status = "● REC" if recorder.is_recording else "○ IDLE"
            if recorder.is_recording_function:
                status += " [FUNC]"
            cv2.putText(frame, status, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255) if recorder.is_recording else (200, 200, 200), 2)

            cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFFFF

        if key == ord('q'):
            running = False
        elif key == ord('r'):
            if recorder.is_recording:
                recorder.stop_recording()
                filepath = recorder.save()
                print(f"  ✓ Saved: {filepath}")
            else:
                recorder.start_recording()
        elif key == 65474 or key == 0xFFC2:  # F5
            if recorder.is_recording_function:
                # Ask for function name
                name = input("  Function name: ").strip()
                recorder.end_function(name=name if name else None)
            else:
                recorder.start_function()
        # Movement keys (same as before)
        elif key == 65361:  # Left
            arm.move_joints_relative(base=1.0)
        elif key == 65363:  # Right
            arm.move_joints_relative(base=-1.0)
        elif key == 65362:  # Up
            arm.move_joints_relative(shoulder=1.0)
        elif key == 65364:  # Down
            arm.move_joints_relative(shoulder=-1.0)
        elif key == ord('w'):
            arm.move_joints_relative(elbow=-1.0)
        elif key == ord('s'):
            arm.move_joints_relative(elbow=1.0)
        elif key == ord('a'):
            arm.move_joints_relative(hand=-2.0)
        elif key == ord('d'):
            arm.move_joints_relative(hand=2.0)
        elif key == ord('o'):
            arm.gripper_open()
        elif key == ord('c'):
            arm.gripper_close()

        time.sleep(0.02)

    # Cleanup
    if recorder.is_recording:
        recorder.stop_recording()
        recorder.save()

    arm.park()
    arm.disconnect()
    vision.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

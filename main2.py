#!/usr/bin/env python3
"""
RoArm-M2-S Automatic Demo
Automatically finds the serial port, connects, and demonstrates arm capabilities.
"""

from roarm_m2s import RoArmM2S, ArmStatus
import time
import sys


def print_status(status: ArmStatus) -> None:
    """Pretty-print arm status."""
    print(f"  Position:  X={status.x:.1f}mm  Y={status.y:.1f}mm  Z={status.z:.1f}mm")
    print(f"  Joints:    Base={status.base_rad:.3f}rad  Shoulder={status.shoulder_rad:.3f}rad  "
          f"Elbow={status.elbow_rad:.3f}rad  EoAT={status.eoat_rad:.3f}rad")
    print(f"  Torque:    B={status.torque_base}  S={status.torque_shoulder}  "
          f"E={status.torque_elbow}  H={status.torque_hand}")
    print(f"  Voltage:   {status.voltage:.2f}V")


def main():
    print("=" * 60)
    print("  RoArm-M2-S Automatic Control Demo")
    print("=" * 60)
    print()

    # Auto-detect and connect
    with RoArmM2S() as arm:
        print(f"\n{arm}\n")

        # 1. Get initial status
        print("[1] Reading arm status...")
        status = arm.get_status()
        if status:
            print_status(status)
        else:
            print("  (Could not parse status, raw response follows)")
            print(f"  {arm.get_status_raw()}")

        # 2. Move to home/init position
        print("\n[2] Moving to home position...")
        arm.move_to_init()
        print("  Done.")

        # 3. LED flash to confirm connection
        print("\n[3] Flashing LED...")
        for _ in range(3):
            arm.set_led(255)
            time.sleep(0.2)
            arm.set_led(0)
            time.sleep(0.2)
        print("  Done.")

        # 4. Joint angle demo (degrees)
        print("\n[4] Joint angle demo (degrees)...")
        arm.move_joints_degrees(b=30, s=0, e=90, h=180, spd=15, acc=10)
        time.sleep(1)
        arm.move_joints_degrees(b=-30, s=0, e=90, h=180, spd=15, acc=10)
        time.sleep(1)
        arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=15, acc=10)
        time.sleep(0.5)
        print("  Done.")

        # 5. Cartesian movement demo
        print("\n[5] Cartesian movement demo...")
        arm.move_cartesian(x=250, y=0, z=200, t=3.14, spd=0.25)
        time.sleep(1)
        arm.move_cartesian(x=200, y=80, z=150, t=3.14, spd=0.25)
        time.sleep(1)
        arm.move_cartesian(x=200, y=-80, z=150, t=3.14, spd=0.25)
        time.sleep(1)
        print("  Done.")

        # 6. Gripper demo
        print("\n[6] Gripper demo...")
        arm.gripper_open()
        time.sleep(0.8)
        arm.gripper_close()
        time.sleep(0.8)
        arm.gripper_open(amount=2.0)  # Half open
        time.sleep(0.5)
        print("  Done.")

        # 7. Read final status
        print("\n[7] Final status:")
        status = arm.get_status()
        if status:
            print_status(status)

        # 8. Park safely
        print("\n[8] Parking arm...")
        arm.park()
        time.sleep(1)
        print("  Done.")

        print("\n" + "=" * 60)
        print("  Demo complete! Arm parked safely.")
        print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Exiting...")
        sys.exit(0)

#!/usr/bin/env python3
"""torque_off.py - Schaltet Torque aus → Arm ist frei bewegbar ("unfrozen")"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
# ]
# ///

import sys
import time

from bootstrap import ensure_uv
ensure_uv()

from robot import RoArmConnection, find_arm_port

def main():
    port = None
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = find_arm_port()

    if port is None:
        print("❌ Kein serieller Port gefunden!")
        print("   Usage: python3 torque_off.py [/dev/ttyUSB0]")
        sys.exit(1)

    print(f"🔌 Verbinde mit {port}...")
    try:
        arm = RoArmConnection(port)
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        sys.exit(1)

    print("🔓 Schalte Torque AUS (alle Servos)...")
    arm.torque_off()
    time.sleep(0.1)

    print("✅ Torque AUS — Arm ist jetzt frei bewegbar!")
    print("   Du kannst den Arm von Hand bewegen.")

    arm.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n   Abgebrochen.")

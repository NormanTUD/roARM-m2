#!/usr/bin/env python3
"""torque_on.py - Schaltet Torque ein → Arm ist fixiert ("frozen")"""
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
        print("   Usage: python3 torque_on.py [/dev/ttyUSB0]")
        sys.exit(1)

    print(f"🔌 Verbinde mit {port}...")
    try:
        arm = RoArmConnection(port)
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        sys.exit(1)

    print("🔒 Schalte Torque EIN (alle Servos)...")
    arm.torque_on()
    time.sleep(0.1)

    # Optional: Position anzeigen
    pos = arm.read_position_deg()
    if pos:
        print(f"   Position: b={pos['b']:.2f}° s={pos['s']:.2f}° "
              f"e={pos['e']:.2f}° h={pos['h']:.2f}°")

    print("✅ Torque EIN — Arm ist jetzt fixiert (frozen)!")
    print("   Servos halten ihre aktuelle Position.")

    arm.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n   Abgebrochen.")

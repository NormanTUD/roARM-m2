#!/usr/bin/env python3
"""servo_scan.py - Scan and manage servo IDs on the RoArm-M2-S over USB

Uses the firmware's CMD_SCAN_SERVOS (T:504) and CMD_SET_SERVO_ID (T:501)
commands to list all connected servos and optionally change their IDs.

Usage:
    python3 servo_scan.py              # Scan and list all servos
    python3 servo_scan.py --change 1 11  # Change servo ID 1 → 11
    python3 servo_scan.py --port /dev/ttyUSB0  # Specify port manually
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
# ]
# ///

import sys
import json
import time
import argparse
from typing import Optional

from robot import RoArmConnection, find_arm_port, BAUDRATE

# Known RoArm-M2 servo assignments
KNOWN_SERVOS = {
    11: "Base",
    12: "Shoulder (Driving)",
    13: "Shoulder (Driven)",
    14: "Elbow",
    15: "Gripper",
}


def scan_servos(arm: RoArmConnection) -> Optional[dict]:
    """
    Sends CMD_SCAN_SERVOS (T:504) and parses the response.
    
    The firmware scans IDs 0–30, reads position from each,
    and returns a JSON with found servo IDs and positions.
    
    Returns:
        Dict with keys: "count", "servo_0", "servo_1", etc.
        Each servo entry has "id" and "pos".
        Returns None on communication failure.
    """
    # The scan takes a while (5ms delay per ID × 31 IDs ≈ 155ms minimum)
    # Give it extra timeout
    resp = arm.send_cmd({"T": 504}, timeout=5.0)
    
    if not resp:
        return None
    
    try:
        # Find the JSON in the response
        start = resp.find('{')
        end = resp.rfind('}')
        if start >= 0 and end > start:
            data = json.loads(resp[start:end + 1])
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def change_servo_id(arm: RoArmConnection, old_id: int, new_id: int) -> bool:
    """
    Sends CMD_SET_SERVO_ID (T:501) to change a servo's ID.
    
    WARNING: Only connect ONE servo at a time when changing IDs!
    
    The firmware unlocks the servo's EEPROM, writes the new ID,
    and locks it again.
    
    Args:
        old_id: Current servo ID
        new_id: Desired new servo ID (0-253)
        
    Returns:
        True if command was sent successfully.
    """
    if not (0 <= new_id <= 253):
        print(f"  ❌ Invalid new ID: {new_id}. Must be 0–253.")
        return False
    
    if old_id == new_id:
        print(f"  ⚠️  Old ID and new ID are the same ({old_id}). Nothing to do.")
        return False
    
    print(f"  🔧 Changing servo ID: {old_id} → {new_id} ...")
    resp = arm.send_cmd({"T": 501, "raw": old_id, "new": new_id}, timeout=2.0)
    
    if resp:
        print(f"  ✅ Change command sent. Response: {resp}")
        return True
    else:
        print(f"  ❌ No response. Servo ID {old_id} may not exist.")
        return False


def print_servo_table(scan_result: dict):
    """Pretty-prints the scan results as a table."""
    count = scan_result.get("count", 0)
    
    if count == 0:
        print("\n  ⚠️  No servos found!")
        print("     Check:")
        print("     - Is 12V power connected?")
        print("     - Are servo cables plugged in?")
        print("     - Is the servo bus (UART2TTL) working?")
        return
    
    print(f"\n  ✅ Found {count} servo(s):\n")
    print(f"  {'ID':>4}  {'Position':>10}  {'Role':<25}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 25}")
    
    for i in range(count):
        key = f"servo_{i}"
        servo = scan_result.get(key)
        if servo is None:
            continue
        
        servo_id = servo.get("id", "?")
        servo_pos = servo.get("pos", "?")
        role = KNOWN_SERVOS.get(servo_id, "(unknown)")
        
        print(f"  {servo_id:>4}  {servo_pos:>10}  {role:<25}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Scan and manage servo IDs on the RoArm-M2-S over USB"
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "--change", nargs=2, type=int, metavar=("OLD_ID", "NEW_ID"),
        help="Change a servo ID. Only connect ONE servo when doing this!"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After changing ID, rescan to verify the change"
    )
    args = parser.parse_args()
    
    # Find port
    port = args.port or find_arm_port()
    if port is None:
        print("❌ No serial port found!")
        print("   Connect the RoArm-M2-S via USB and try again.")
        print("   Or specify manually: --port /dev/ttyUSB0")
        sys.exit(1)
    
    print(f"🔌 Connecting to {port} @ {BAUDRATE} baud...")
    
    try:
        arm = RoArmConnection(port)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    print(f"   ✓ Connected\n")
    
    try:
        if args.change:
            # Change servo ID mode
            old_id, new_id = args.change
            
            print(f"⚠️  CHANGING SERVO ID: {old_id} → {new_id}")
            print(f"   Make sure only ONE servo is connected!\n")
            
            confirm = input("   Continue? [y/N] ").strip().lower()
            if confirm != 'y':
                print("   Aborted.")
                return
            
            success = change_servo_id(arm, old_id, new_id)
            
            if success and args.verify:
                print("\n  ⏳ Waiting 1.5s then rescanning to verify...")
                time.sleep(1.5)
                result = scan_servos(arm)
                if result:
                    print_servo_table(result)
                else:
                    print("  ⚠️  Could not verify (no scan response)")
        else:
            # Scan mode (default)
            print("🔍 Scanning for connected servos (ID 0–30)...")
            print("   This takes a few seconds...\n")
            
            result = scan_servos(arm)
            
            if result is None:
                print("  ❌ No response from firmware.")
                print("     The device may be busy or not running the correct firmware.")
                print("     Expected firmware with CMD_SCAN_SERVOS (T:504) support.")
            else:
                print_servo_table(result)
                
                # Show info text
                info = result.get("info", "")
                if info:
                    print(f"  ℹ️  Firmware says: {info}")
    
    finally:
        arm.close()
        print("🔌 Disconnected.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n   Interrupted.")
    except OSError as e:
        print(f"\n   ⚠️  Connection error: {e}")
        print(f"   The arm may have been disconnected.")

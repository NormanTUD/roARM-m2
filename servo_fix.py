#!/usr/bin/env python3
"""servo_fix.py - Gripper-Servo ID scannen und reparieren

Scannt den Servo-Bus und ändert die ID des angeschlossenen Servos.
Nur EIN Servo darf angeschlossen sein beim ID-Ändern!

Usage:
    python3 servo_fix.py                    # Nur scannen
    python3 servo_fix.py --set-id 15        # Gefundenen Servo auf ID 15 setzen
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
# ]
# ///

import os
import sys

from bootstrap import ensure_uv
ensure_uv()

import json
import time
import re
import argparse

from robot import RoArmConnection, find_arm_port, BAUDRATE


# Bekannte Servo-Zuordnungen
KNOWN_SERVOS = {
    11: "Base",
    12: "Shoulder (Driving)",
    13: "Shoulder (Driven)",
    14: "Elbow",
    15: "Gripper",
}

# Regex für die Debug-Ausgabe der Firmware
FOUND_SERVO_RE = re.compile(r"Found servo ID:\s*(\d+),\s*pos:\s*(-?\d+)")
SCAN_COMPLETE_RE = re.compile(r"Scan complete\.\s*Found\s*(\d+)\s*servo")
NO_SERVOS_RE = re.compile(r"No servos found")
CHANGE_SUCCEED_RE = re.compile(r"change:\s*(\d+)\s*succeed")
CHANGE_FAILED_RE = re.compile(r"change:\s*(\d+)\s*failed")


def scan_servos(arm: RoArmConnection) -> list[dict]:
    """
    Sendet CMD_SCAN_SERVOS (T:504) und parst die Debug-Ausgabe.
    
    Die Firmware gibt für jeden gefundenen Servo eine Zeile aus:
        Found servo ID: <id>, pos: <pos>
    Und am Ende:
        Scan complete. Found <n> servo(s).
    
    Das JSON geht nur an jsonInfoHttp (für den Webserver),
    NICHT an Serial. Daher parsen wir die Debug-Zeilen.
    """
    # Buffer leeren
    arm._ser.reset_input_buffer()
    time.sleep(0.1)
    
    # Scan-Befehl senden
    msg = json.dumps({"T": 504}, separators=(',', ':'))
    arm._ser.write(msg.encode() + b'\n')
    arm._ser.flush()
    
    print("  ⏳ Warte auf Scan-Ergebnis...")
    
    found_servos = []
    deadline = time.time() + 10.0  # 10s Timeout (31 IDs × 5ms + Overhead)
    scan_done = False
    
    while time.time() < deadline and not scan_done:
        if arm._ser.in_waiting:
            line = arm._ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Debug: Zeige alles was reinkommt
            print(f"  📡 {line}")
            
            # "Found servo ID: X, pos: Y" parsen
            match = FOUND_SERVO_RE.search(line)
            if match:
                servo_id = int(match.group(1))
                servo_pos = int(match.group(2))
                found_servos.append({"id": servo_id, "pos": servo_pos})
                continue
            
            # "Scan complete. Found N servo(s)." → fertig
            if SCAN_COMPLETE_RE.search(line):
                scan_done = True
                continue
            
            # "No servos found" → fertig
            if NO_SERVOS_RE.search(line):
                scan_done = True
                continue
        else:
            time.sleep(0.02)
    
    if not scan_done:
        # Timeout — aber vielleicht haben wir trotzdem Servos gefunden
        print("  ⚠️  Scan-Timeout (kein 'Scan complete' empfangen)")
    
    return found_servos


def change_servo_id(arm: RoArmConnection, old_id: int, new_id: int) -> bool:
    """
    Sendet CMD_SET_SERVO_ID (T:501) um eine Servo-ID zu ändern.
    
    Die Firmware gibt aus:
        "change: <old_id> succeed" oder "change: <old_id> failed"
    
    WICHTIG: Nur EIN Servo darf angeschlossen sein!
    """
    arm._ser.reset_input_buffer()
    time.sleep(0.1)
    
    cmd = {"T": 501, "raw": old_id, "new": new_id}
    msg = json.dumps(cmd, separators=(',', ':'))
    arm._ser.write(msg.encode() + b'\n')
    arm._ser.flush()
    
    # Antwort lesen
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if arm._ser.in_waiting:
            line = arm._ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            print(f"  📡 {line}")
            
            if CHANGE_SUCCEED_RE.search(line):
                return True
            if CHANGE_FAILED_RE.search(line):
                return False
        else:
            time.sleep(0.02)
    
    # Kein klares Ergebnis — prüfen wir per Rescan
    print("  ⚠️  Keine eindeutige Antwort, versuche Verifizierung...")
    return None  # Unbekannt


def print_scan_results(found_servos: list[dict]) -> list[int]:
    """Zeigt die Scan-Ergebnisse als Tabelle."""
    if not found_servos:
        print("\n  ⚠️  Keine Servos gefunden!")
        print("     Prüfe:")
        print("     - Ist 12V Strom angeschlossen?")
        print("     - Ist das Servo-Kabel eingesteckt?")
        print("     - Ist der Servo-Bus (UART2TTL) OK?")
        return []
    
    print(f"\n  ✅ {len(found_servos)} Servo(s) gefunden:\n")
    print(f"  {'ID':>4}  {'Position':>10}  {'Rolle':<25}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 25}")
    
    ids = []
    for servo in found_servos:
        servo_id = servo["id"]
        servo_pos = servo["pos"]
        role = KNOWN_SERVOS.get(servo_id, "(unbekannt)")
        ids.append(servo_id)
        
        print(f"  {servo_id:>4}  {servo_pos:>10}  {role:<25}")
    
    print()
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Servo-IDs scannen und ändern (RoArm-M2-S über USB)"
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="Serieller Port (auto-detect wenn nicht angegeben)"
    )
    parser.add_argument(
        "--set-id", type=int, default=None,
        help="Ändere den gefundenen Servo auf diese ID (z.B. 15 für Gripper)"
    )
    args = parser.parse_args()
    
    # Port finden
    port = args.port or find_arm_port()
    if port is None:
        print("❌ Kein serieller Port gefunden!")
        print("   Verbinde den RoArm-M2-S per USB und versuche es erneut.")
        sys.exit(1)
    
    print(f"🔌 Verbinde mit {port} @ {BAUDRATE} baud...")
    
    try:
        arm = RoArmConnection(port)
    except Exception as e:
        print(f"❌ Verbindung fehlgeschlagen: {e}")
        sys.exit(1)
    
    print(f"   ✓ Verbunden\n")
    
    # Kurz warten damit die Firmware bereit ist
    time.sleep(0.5)
    
    # Erstmal den Input-Buffer leeren (Boot-Meldungen etc.)
    arm._ser.reset_input_buffer()
    time.sleep(0.2)
    
    try:
        # === SCAN ===
        print("🔍 Scanne Servo-Bus (ID 0–30)...")
        print("   Das dauert ein paar Sekunden...\n")
        
        found_servos = scan_servos(arm)
        found_ids = print_scan_results(found_servos)
        
        # === ID ÄNDERN ===
        if args.set_id is not None:
            new_id = args.set_id
            
            if not (0 <= new_id <= 253):
                print(f"  ❌ Ungültige ID: {new_id}. Muss 0–253 sein.")
                sys.exit(1)
            
            if not found_ids:
                print("  ❌ Kein Servo gefunden zum Ändern!")
                sys.exit(1)
            
            if len(found_ids) > 1:
                print(f"  ⚠️  WARNUNG: {len(found_ids)} Servos gefunden!")
                print(f"     Beim ID-Ändern sollte nur EIN Servo angeschlossen sein!")
                print(f"     Gefundene IDs: {found_ids}")
                print()
                choice = input("     Trotzdem fortfahren? Welche ID ändern? > ").strip()
                try:
                    old_id = int(choice)
                    if old_id not in found_ids:
                        print(f"     ID {old_id} nicht in gefundenen Servos!")
                        sys.exit(1)
                except ValueError:
                    print("     Abgebrochen.")
                    sys.exit(0)
            else:
                old_id = found_ids[0]
            
            if old_id == new_id:
                print(f"  ✅ Servo hat bereits ID {new_id}. Nichts zu tun!")
                return
            
            print(f"\n  🔧 ÄNDERE SERVO-ID: {old_id} → {new_id}")
            print(f"     ({KNOWN_SERVOS.get(old_id, 'unbekannt')} → {KNOWN_SERVOS.get(new_id, 'unbekannt')})")
            print()
            
            confirm = input("     Fortfahren? [j/N] ").strip().lower()
            if confirm not in ('j', 'y', 'ja', 'yes'):
                print("     Abgebrochen.")
                return
            
            print()
            success = change_servo_id(arm, old_id, new_id)
            
            if success is True:
                print(f"\n  ✅ ID erfolgreich geändert: {old_id} → {new_id}")
            elif success is False:
                print(f"\n  ❌ ID-Änderung fehlgeschlagen!")
                print(f"     Der Servo mit ID {old_id} antwortet möglicherweise nicht.")
                sys.exit(1)
            
            # Verifizieren
            print(f"\n  🔍 Verifiziere... (Rescan)")
            time.sleep(1.5)
            
            verify_servos = scan_servos(arm)
            verify_ids = print_scan_results(verify_servos)
            
            if new_id in verify_ids:
                print(f"  ✅ Verifiziert! Servo antwortet jetzt auf ID {new_id}")
            elif old_id not in verify_ids and not verify_ids:
                print(f"  ⚠️  Kein Servo im Rescan gefunden.")
                print(f"     Möglicherweise muss der Servo neu gestartet werden.")
            else:
                print(f"  ⚠️  ID {new_id} nicht im Rescan. Gefunden: {verify_ids}")
        
        else:
            # Nur Scan-Modus: Hinweis geben
            if found_ids:
                if 15 not in found_ids:
                    print("  💡 Tipp: Dein Gripper-Servo sollte ID 15 haben.")
                    print(f"     Aktuell gefunden: {found_ids}")
                    if len(found_ids) == 1:
                        print(f"     → python3 servo_fix.py --set-id 15")
                else:
                    print("  ✅ Gripper-Servo (ID 15) ist vorhanden!")
    
    finally:
        arm.close()
        print("\n🔌 Verbindung getrennt.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n   Abgebrochen.")
    except OSError as e:
        print(f"\n   ⚠️  Verbindungsfehler: {e}")
        print(f"   Der Arm wurde möglicherweise getrennt.")

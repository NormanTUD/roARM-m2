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


def scan_servos(arm: RoArmConnection) -> dict | None:
    """
    Sendet CMD_SCAN_SERVOS (T:504).
    
    Die Firmware iteriert über IDs 0–30, liest die Position jedes Servos,
    und gibt ein JSON mit allen gefundenen Servos zurück.
    
    Timeout ist großzügig weil der Scan ~155ms+ dauert (5ms × 31 IDs).
    """
    # Vor dem Scan: Buffer leeren
    arm._ser.reset_input_buffer()
    time.sleep(0.1)
    
    # Scan-Befehl senden
    msg = json.dumps({"T": 504}, separators=(',', ':'))
    arm._ser.write(msg.encode() + b'\n')
    arm._ser.flush()
    
    # Antwort sammeln (Firmware gibt auch Debug-Prints aus)
    print("  ⏳ Warte auf Scan-Ergebnis...")
    
    all_lines = []
    deadline = time.time() + 8.0  # 8s Timeout (großzügig)
    json_result = None
    
    while time.time() < deadline:
        if arm._ser.in_waiting:
            line = arm._ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                all_lines.append(line)
                # Debug-Output anzeigen
                if line.startswith("Found servo") or line.startswith("No servos") or line.startswith("Scan complete"):
                    print(f"  📡 {line}")
                
                # JSON-Antwort erkennen (enthält "T":504 und "count")
                if '"T":504' in line or ('"count"' in line and '"servo_' in line):
                    try:
                        start = line.find('{')
                        end = line.rfind('}')
                        if start >= 0 and end > start:
                            json_result = json.loads(line[start:end + 1])
                            # Kurz weiter lesen für restliche Debug-Ausgaben
                            time.sleep(0.3)
                            while arm._ser.in_waiting:
                                extra = arm._ser.readline().decode('utf-8', errors='ignore').strip()
                                if extra:
                                    all_lines.append(extra)
                                    if extra.startswith("Found") or extra.startswith("Scan"):
                                        print(f"  📡 {extra}")
                            break
                    except json.JSONDecodeError:
                        pass
        else:
            time.sleep(0.02)
    
    # Falls kein sauberes JSON gefunden: Versuche aus allen Zeilen zu parsen
    if json_result is None:
        for line in reversed(all_lines):
            try:
                start = line.find('{')
                end = line.rfind('}')
                if start >= 0 and end > start:
                    candidate = json.loads(line[start:end + 1])
                    if "count" in candidate:
                        json_result = candidate
                        break
            except json.JSONDecodeError:
                continue
    
    return json_result


def change_servo_id(arm: RoArmConnection, old_id: int, new_id: int) -> bool:
    """
    Sendet CMD_SET_SERVO_ID (T:501) um eine Servo-ID zu ändern.
    
    Die Firmware:
    1. Prüft ob der Servo mit old_id antwortet (getFeedback)
    2. Entsperrt das EEPROM (st.unLockEprom)
    3. Schreibt die neue ID (st.writeByte)
    4. Sperrt das EEPROM wieder (st.LockEprom)
    
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
            if line:
                print(f"  📡 {line}")
                if "succeed" in line.lower():
                    return True
                if "failed" in line.lower():
                    return False
        else:
            time.sleep(0.02)
    
    return False


def print_scan_results(result: dict):
    """Zeigt die Scan-Ergebnisse als Tabelle."""
    count = result.get("count", 0)
    
    if count == 0:
        print("\n  ⚠️  Keine Servos gefunden!")
        print("     Prüfe:")
        print("     - Ist 12V Strom angeschlossen?")
        print("     - Ist das Servo-Kabel eingesteckt?")
        print("     - Ist der Servo-Bus (UART2TTL) OK?")
        return []
    
    print(f"\n  ✅ {count} Servo(s) gefunden:\n")
    print(f"  {'ID':>4}  {'Position':>10}  {'Rolle':<25}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 25}")
    
    found_servos = []
    for i in range(count):
        key = f"servo_{i}"
        servo = result.get(key)
        if servo is None:
            continue
        
        servo_id = servo.get("id", "?")
        servo_pos = servo.get("pos", "?")
        role = KNOWN_SERVOS.get(servo_id, "(unbekannt)")
        
        print(f"  {servo_id:>4}  {servo_pos:>10}  {role:<25}")
        found_servos.append(servo_id)
    
    print()
    return found_servos


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
    
    try:
        # === SCAN ===
        print("🔍 Scanne Servo-Bus (ID 0–30)...")
        print("   Das dauert ein paar Sekunden...\n")
        
        result = scan_servos(arm)
        
        if result is None:
            print("  ❌ Keine Antwort von der Firmware.")
            print("     Mögliche Ursachen:")
            print("     - Firmware unterstützt CMD_SCAN_SERVOS (T:504) nicht")
            print("     - Gerät ist beschäftigt")
            print("     - Falscher Port")
            sys.exit(1)
        
        found_servos = print_scan_results(result)
        
        # === ID ÄNDERN ===
        if args.set_id is not None:
            new_id = args.set_id
            
            if not (0 <= new_id <= 253):
                print(f"  ❌ Ungültige ID: {new_id}. Muss 0–253 sein.")
                sys.exit(1)
            
            if not found_servos:
                print("  ❌ Kein Servo gefunden zum Ändern!")
                sys.exit(1)
            
            if len(found_servos) > 1:
                print(f"  ⚠️  WARNUNG: {len(found_servos)} Servos gefunden!")
                print(f"     Beim ID-Ändern sollte nur EIN Servo angeschlossen sein!")
                print(f"     Gefundene IDs: {found_servos}")
                print()
                choice = input("     Trotzdem fortfahren? Welche ID ändern? > ").strip()
                try:
                    old_id = int(choice)
                    if old_id not in found_servos:
                        print(f"     ID {old_id} nicht in gefundenen Servos!")
                        sys.exit(1)
                except ValueError:
                    print("     Abgebrochen.")
                    sys.exit(0)
            else:
                old_id = found_servos[0]
            
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
            
            if success:
                print(f"\n  ✅ ID erfolgreich geändert: {old_id} → {new_id}")
            else:
                print(f"\n  ❌ ID-Änderung fehlgeschlagen!")
                print(f"     Der Servo mit ID {old_id} antwortet möglicherweise nicht.")
                sys.exit(1)
            
            # Verifizieren
            print(f"\n  🔍 Verifiziere... (Rescan)")
            time.sleep(1.5)
            
            verify_result = scan_servos(arm)
            if verify_result:
                verify_servos = print_scan_results(verify_result)
                if new_id in verify_servos:
                    print(f"  ✅ Verifiziert! Servo antwortet jetzt auf ID {new_id}")
                else:
                    print(f"  ⚠️  Servo mit ID {new_id} nicht im Rescan gefunden.")
                    print(f"     Möglicherweise muss der Servo neu gestartet werden.")
        
        else:
            # Nur Scan-Modus: Hinweis geben
            if found_servos:
                print("  ℹ️  Um die ID zu ändern:")
                print(f"     python3 servo_fix.py --set-id 15")
                print()
                if 15 not in found_servos:
                    print("  💡 Tipp: Dein Gripper-Servo sollte ID 15 haben.")
                    print(f"     Aktuell gefunden: {found_servos}")
                    if len(found_servos) == 1:
                        print(f"     → python3 servo_fix.py --set-id 15")
    
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

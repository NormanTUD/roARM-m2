#!/usr/bin/env python3
"""
RoArm-M2-S Controller
- Ohne Argumente: Hardware-Test-Demo
- Mit --target <objekt>: Greift das Objekt per YOLO-Vision (Kamera auf Arm montiert)
- Live-Stream zeigt Detections in Echtzeit
"""

from roarm_m2s import RoArmM2S, ArmStatus
import time
import sys
import argparse


def print_status(status: ArmStatus) -> None:
    print(f"  Position:  X={status.x:.1f}  Y={status.y:.1f}  Z={status.z:.1f} mm")
    print(f"  Joints:    B={status.base_rad:.3f}  S={status.shoulder_rad:.3f}  "
          f"E={status.elbow_rad:.3f}  H={status.eoat_rad:.3f} rad")
    print(f"  Torque:    B={status.torque_base}  S={status.torque_shoulder}  "
          f"E={status.torque_elbow}  H={status.torque_hand}")
    print(f"  Voltage:   {status.voltage:.2f}V")


def run_test_demo(arm: RoArmM2S) -> None:
    """Hardware-Test: Bewegung, LED, Gripper. Kein Vision nötig."""
    print("\n" + "=" * 60)
    print("  HARDWARE TEST DEMO")
    print("=" * 60)

    # Status
    print("\n[1] Status lesen...")
    status = arm.get_status()
    if status:
        print_status(status)
    else:
        print("  (Kein Status empfangen)")

    # Home
    print("\n[2] Home-Position...")
    arm.move_to_init()
    print("  ✓")

    # LED
    print("\n[3] LED-Test...")
    for _ in range(3):
        arm.set_led(255)
        time.sleep(0.15)
        arm.set_led(0)
        time.sleep(0.15)
    print("  ✓")

    # Bewegung
    print("\n[4] Bewegungs-Test...")
    arm.move_joints_degrees(b=25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)
    arm.move_joints_degrees(b=-25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)
    arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.5)
    print("  ✓")

    # Cartesian
    print("\n[5] Cartesian-Test...")
    arm.move_cartesian(200, 0, 150, t=3.14, spd=0.25)
    time.sleep(1.0)
    arm.move_cartesian(200, 50, 150, t=3.14, spd=0.25)
    time.sleep(1.0)
    arm.move_cartesian(200, -50, 150, t=3.14, spd=0.25)
    time.sleep(1.0)
    print("  ✓")

    # Gripper
    print("\n[6] Gripper-Test...")
    arm.gripper_open()
    time.sleep(0.6)
    arm.gripper_close()
    time.sleep(0.6)
    arm.gripper_open()
    time.sleep(0.3)
    print("  ✓")

    # Park
    print("\n[7] Parken...")
    arm.park()
    time.sleep(1.0)
    print("  ✓")

    print("\n  Test-Demo abgeschlossen!")


def eye_in_hand_offset(center_px: tuple, img_resolution: tuple) -> tuple:
    """
    Berechnet den Offset in mm den der Arm fahren muss,
    damit das Objekt in der Bildmitte (= unter dem Greifer) ist.
    
    Kamera ist am End-Effector montiert, schaut nach unten.
    
    Returns:
        (dx_mm, dy_mm) – Offset für den Arm.
        
    KALIBRIERUNG NÖTIG: scale anpassen je nach Kamerahöhe über Objekt!
    """
    img_w, img_h = img_resolution
    px_x, px_y = center_px

    offset_px_x = px_x - (img_w / 2)
    offset_px_y = px_y - (img_h / 2)

    # mm pro Pixel – KALIBRIEREN!
    # Bei 640x480, Kamera ~12-15cm über Tisch: ca. 0.4-0.6 mm/px
    scale = 0.5

    # Kamera-Achsen → Arm-Achsen (Eye-in-Hand, Kamera schaut runter)
    # Pixel-Y (unten) → Arm-X (vorwärts)
    # Pixel-X (rechts) → Arm-Y (links, daher negiert)
    dx_mm = offset_px_y * scale
    dy_mm = -offset_px_x * scale

    return (dx_mm, dy_mm)


def run_grab(arm: RoArmM2S, target_class: str) -> bool:
    """
    Greift ein Objekt per Vision. Eye-in-Hand Konfiguration.
    Zeigt Live-Stream während des gesamten Vorgangs.
    
    Ablauf:
    1. Scan-Position anfahren
    2. Live-Stream starten, Objekt suchen
    3. Über Objekt zentrieren (iterativ)
    4. Absenken und greifen
    5. Anheben und ablegen
    """
    if not arm.vision or not arm.vision.available:
        print("\n[Grab] ✗ Vision nicht verfügbar!")
        print("  → Kamera anschließen")
        print("  → pip install ultralytics opencv-python")
        return False

    print(f"\n{'=' * 60}")
    print(f"  GRAB MODE: Suche '{target_class}'")
    print(f"{'=' * 60}")

    # Live-Stream starten (läuft im Hintergrund)
    arm.start_live_stream(target_classes=[target_class])
    time.sleep(1.0)  # Stream stabilisieren

    # 1. Scan-Position (hoch, nach vorne, Kamera schaut runter)
    print("\n[1] Fahre Scan-Position...")
    scan_x, scan_y, scan_z = 180, 0, 200
    arm.move_cartesian(scan_x, scan_y, scan_z, t=3.14, spd=0.25)
    time.sleep(2.5)

    # 2. Objekt suchen
    print(f"[2] Suche '{target_class}'...")
    found = False
    best_detection = None

    # Mehrere Versuche (Kamera braucht manchmal einen Moment)
    for attempt in range(10):
        detections = arm.vision.detect_objects([target_class])
        if detections:
            best_detection = detections[0]
            found = True
            break
        time.sleep(0.3)

    if not found:
        print(f"  ✗ '{target_class}' nicht gefunden!")
        # Was ist stattdessen sichtbar?
        all_det = arm.vision.detect_objects()
        if all_det:
            classes = set(d['class'] for d in all_det)
            print(f"  Sichtbare Objekte: {', '.join(classes)}")
            print(f"  Tipp: Verwende --target <eines davon>")
        else:
            print("  Keine Objekte erkannt. Beleuchtung/Position prüfen.")
        arm.stop_live_stream()
        return False

    print(f"  ✓ '{best_detection['class']}' gefunden! "
          f"(conf={best_detection['confidence']:.2f}, "
          f"pixel=({best_detection['center_px'][0]:.0f}, {best_detection['center_px'][1]:.0f}))")

    # 3. Iterativ über Objekt zentrieren
    print("\n[3] Zentriere über Objekt...")
    img_res = arm.vision.resolution
    max_iterations = 5
    center_threshold = 20  # Pixel – wenn Objekt so nah an Bildmitte, ist es "zentriert"

    for i in range(max_iterations):
        detections = arm.vision.detect_objects([target_class])
        if not detections:
            print(f"  Iteration {i+1}: Objekt verloren! Warte...")
            time.sleep(0.5)
            continue

        det = detections[0]
        dx_mm, dy_mm = eye_in_hand_offset(det['center_px'], img_res)

        # Check ob schon zentriert
        offset_px_x = det['center_px'][0] - (img_res[0] / 2)
        offset_px_y = det['center_px'][1] - (img_res[1] / 2)
        pixel_dist = (offset_px_x**2 + offset_px_y**2) ** 0.5

        print(f"  Iteration {i+1}: Offset = ({dx_mm:.1f}, {dy_mm:.1f})mm, "
              f"Pixel-Dist = {pixel_dist:.0f}px")

        if pixel_dist < center_threshold:
            print(f"  ✓ Zentriert!")
            break

        # Aktuelle Position holen und korrigieren
        status = arm.get_status()
        if not status:
            print("  ✗ Kann Status nicht lesen!")
            break

        new_x = status.x + dx_mm * 0.7  # Gedämpft (70%) um Überschwingen zu vermeiden
        new_y = status.y + dy_mm * 0.7

        # Sicherheitscheck
        dist = (new_x**2 + new_y**2) ** 0.5
        if dist > 300:
            print(f"  ✗ Außerhalb Reichweite ({dist:.0f}mm)!")
            arm.stop_live_stream()
            return False

        arm.move_cartesian(new_x, new_y, scan_z, t=3.14, spd=0.2)
        time.sleep(1.5)

    # 4. Aktuelle Position für Greifen merken
    status = arm.get_status()
    if not status:
        print("  ✗ Status-Fehler!")
        arm.stop_live_stream()
        return False

    grab_x = status.x
    grab_y = status.y
    grab_z = 75  # Greifhöhe über Tisch – KALIBRIEREN!

    print(f"\n[4] Greife bei X={grab_x:.0f} Y={grab_y:.0f} Z={grab_z}...")

    # Gripper öffnen
    arm.gripper_open()
    time.sleep(0.5)

    # Absenken (langsam)
    print("  → Absenken...")
    arm.move_cartesian(grab_x, grab_y, grab_z + 40, t=3.14, spd=0.15)
    time.sleep(1.5)
    arm.move_cartesian(grab_x, grab_y, grab_z, t=3.14, spd=0.1)
    time.sleep(1.5)

    # Greifen
    print("  → Greifer schließen...")
    arm.gripper_close()
    time.sleep(1.0)

    # 5. Anheben
    print("\n[5] Anheben...")
    arm.move_cartesian(grab_x, grab_y, grab_z + 100, t=3.14, spd=0.2)
    time.sleep(2.0)

    # 6. Ablegen (100mm nach rechts vom Greifpunkt)
    place_y = grab_y - 100
    print(f"[6] Ablegen bei Y={place_y:.0f}...")
    arm.move_cartesian(grab_x, place_y, grab_z + 30, t=3.14, spd=0.2)
    time.sleep(2.0)

    arm.gripper_open()
    time.sleep(0.5)

    # Zurückziehen
    arm.move_cartesian(grab_x, place_y, grab_z + 100, t=3.14, spd=0.25)
    time.sleep(1.0)

    # Parken
    print("\n[7] Parken...")
    arm.park()
    time.sleep(1.0)

    # Live-Stream stoppen
    arm.stop_live_stream()

    print(f"\n  ✓ '{target_class}' erfolgreich gegriffen und abgelegt!")
    return True


def run_scan_only(arm: RoArmM2S) -> None:
    """Fährt in Scan-Position und zeigt Live-Stream mit allen Detections."""
    if not arm.vision or not arm.vision.available:
        print("[Scan] Vision nicht verfügbar.")
        return

    print("\n[Scan] Fahre Scan-Position und zeige Live-Stream...")
    print("       Drücke 'q' im Fenster zum Beenden.")
    arm.move_cartesian(180, 0, 200, t=3.14, spd=0.25)
    time.sleep(2.0)

    arm.start_live_stream()

    # Warte bis User den Stream beendet (q drückt)
    try:
        while arm._live_running:
            time.sleep(0.5)
            # Zeige periodisch was erkannt wird
            dets = arm.get_live_detections()
            if dets:
                classes = set(d['class'] for d in dets)
                # Nur alle 3 Sekunden printen
    except KeyboardInterrupt:
        pass

    arm.stop_live_stream()
    arm.park()


def main():
    parser = argparse.ArgumentParser(
        description="RoArm-M2-S Controller",
        epilog="""
Beispiele:
  python main.py                        # Hardware-Test-Demo
  python main.py --target bottle        # Greift eine Flasche
  python main.py --target cup           # Greift eine Tasse
  python main.py --scan                 # Nur Live-Stream zeigen
  python main.py --target cell\ phone   # Greift ein Handy
  python main.py --target bottle --confidence 0.3  # Niedrigere Schwelle
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=str, default=None,
                        help="Serieller Port (auto-detect wenn leer)")
    parser.add_argument("--target", type=str, default=None,
                        help="Ziel-Objekt zum Greifen (z.B. 'bottle', 'cup', 'cell phone')")
    parser.add_argument("--scan", action="store_true",
                        help="Nur Scan-Position + Live-Stream (kein Greifen)")
    parser.add_argument("--camera", type=int, default=None,
                        help="Kamera-Index (None = auto/fragen wenn mehrere)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO-Modell Pfad (default: yolo11n.pt)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min. Detection-Confidence (default: 0.5)")
    parser.add_argument("--no-vision", action="store_true",
                        help="Vision komplett deaktivieren (nur Test-Demo)")
    args = parser.parse_args()

    # Entscheide Modus
    need_vision = args.target is not None or args.scan
    enable_vision = need_vision and not args.no_vision

    print("=" * 60)
    print("  RoArm-M2-S Controller")
    if args.target:
        print(f"  Modus: GREIFEN → '{args.target}'")
    elif args.scan:
        print(f"  Modus: SCAN (Live-Stream)")
    else:
        print(f"  Modus: TEST-DEMO")
    print("=" * 60)

    # Verbinden
    with RoArmM2S(
        port=args.port,
        enable_vision=enable_vision,
        camera_index=args.camera,
        yolo_model=args.model,
        confidence=args.confidence
    ) as arm:
        print(f"\n  {arm}")

        # Vision-Status
        if enable_vision:
            if arm.vision and arm.vision.available:
                print(f"  [✓] Vision aktiv (Modell: {args.model}, Conf: {args.confidence})")
            else:
                print(f"  [✗] Vision konnte nicht initialisiert werden!")
                if args.target:
                    print("      Kann ohne Vision nicht greifen. Abbruch.")
                    return
        print()

        # ─── Modus ausführen ──────────────────────────────────────────

        if args.target:
            # GRAB-Modus
            success = run_grab(arm, args.target)
            if not success:
                print("\n  Greifen fehlgeschlagen.")
                arm.park()

        elif args.scan:
            # SCAN-Modus
            run_scan_only(arm)

        else:
            # TEST-DEMO (kein Argument)
            run_test_demo(arm)

    print("\n" + "=" * 60)
    print("  Fertig!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\n[FEHLER] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[Abgebrochen]")
        sys.exit(0)

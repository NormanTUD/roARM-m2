#!/usr/bin/env python3
"""
RoArm-M2-S Automatic Demo with Optional YOLO Vision
Automatically finds the serial port, connects, and demonstrates arm capabilities.
If a camera + YOLO model is available, demonstrates vision-based object detection and grasping.
"""

from roarm_m2s import RoArmM2S, ArmStatus
import time
import sys
import argparse


def print_status(status: ArmStatus) -> None:
    """Pretty-print arm status."""
    print(f"  Position:  X={status.x:.1f}mm  Y={status.y:.1f}mm  Z={status.z:.1f}mm")
    print(f"  Joints:    Base={status.base_rad:.3f}rad  Shoulder={status.shoulder_rad:.3f}rad  "
          f"Elbow={status.elbow_rad:.3f}rad  EoAT={status.eoat_rad:.3f}rad")
    print(f"  Torque:    B={status.torque_base}  S={status.torque_shoulder}  "
          f"E={status.torque_elbow}  H={status.torque_hand}")
    print(f"  Voltage:   {status.voltage:.2f}V")


def simple_pixel_to_mm(center_px: tuple) -> tuple:
    """
    Einfache Pixel→mm Umrechnung (muss für dein Setup kalibriert werden!).
    
    Annahmen (Beispiel für Kamera direkt über dem Arm, ~40cm Höhe):
    - Bildmitte = Arm-Basis (0, 0)
    - 1 Pixel ≈ 0.5mm bei 640x480 Auflösung
    - Z ist fix (Tischhöhe)
    
    WICHTIG: Diese Werte musst du für dein Setup anpassen!
    """
    img_width, img_height = 640, 480
    px_x, px_y = center_px

    # Pixel-Offset von Bildmitte
    offset_x = px_x - (img_width / 2)
    offset_y = px_y - (img_height / 2)

    # Skalierung (Pixel → mm) – KALIBRIEREN!
    scale = 0.55  # mm pro Pixel, abhängig von Kamerahöhe

    # Arm-Koordinaten (Kamera schaut von oben)
    arm_x = 200 + (-offset_y * scale)  # Y-Pixel → X-Arm (vorwärts)
    arm_y = -offset_x * scale           # X-Pixel → Y-Arm (links/rechts)
    arm_z = 80                           # Feste Greifhöhe über Tisch

    return (arm_x, arm_y, arm_z)


def run_vision_demo(arm: RoArmM2S, target: str = "cup") -> None:
    """Führt die Vision-Demo durch, wenn Kamera verfügbar."""
    print("\n" + "=" * 60)
    print("  VISION DEMO")
    print("=" * 60)

    if not arm.vision or not arm.vision.available:
        print("  [Vision] Nicht verfügbar – überspringe Vision-Demo.")
        print("  Tipp: Kamera anschließen und 'pip install ultralytics opencv-python' ausführen.")
        return

    print(f"\n[V1] Suche nach Objekten...")
    detections = arm.vision.detect_objects()
    if detections:
        print(f"  {len(detections)} Objekt(e) erkannt:")
        for i, det in enumerate(detections):
            print(f"    [{i+1}] {det['class']} (conf={det['confidence']:.2f}) "
                  f"@ pixel ({det['center_px'][0]:.0f}, {det['center_px'][1]:.0f})")
    else:
        print("  Keine Objekte erkannt.")
        return

    # Suche nach spezifischem Ziel
    print(f"\n[V2] Suche spezifisch nach '{target}'...")
    target_detections = arm.vision.detect_objects(target_classes=[target])

    if not target_detections:
        print(f"  Kein '{target}' gefunden. Verfügbare Objekte:")
        all_det = arm.vision.detect_objects()
        classes_found = set(d['class'] for d in all_det)
        print(f"    {', '.join(classes_found) if classes_found else 'keine'}")
        return

    best = max(target_detections, key=lambda d: d['confidence'])
    print(f"  Bestes Ergebnis: '{best['class']}' (conf={best['confidence']:.2f})")
    print(f"  Pixel-Position: ({best['center_px'][0]:.0f}, {best['center_px'][1]:.0f})")

    # Koordinaten umrechnen
    arm_x, arm_y, arm_z = simple_pixel_to_mm(best['center_px'])
    print(f"  Arm-Koordinaten: X={arm_x:.1f}mm  Y={arm_y:.1f}mm  Z={arm_z:.1f}mm")

    # Sicherheitscheck
    dist = (arm_x**2 + arm_y**2 + arm_z**2) ** 0.5
    if dist > 320:
        print(f"  [WARNUNG] Position außerhalb Reichweite ({dist:.0f}mm) – überspringe Greifen.")
        return

    # Greifen
    print(f"\n[V3] Greife '{target}'...")
    arm.gripper_open()
    time.sleep(0.5)

    # Über Objekt fahren
    print(f"  → Fahre über Objekt...")
    arm.move_cartesian(arm_x, arm_y, arm_z + 60, t=3.14, spd=0.25)
    time.sleep(1.5)

    # Absenken
    print(f"  → Absenken...")
    arm.move_cartesian(arm_x, arm_y, arm_z, t=3.14, spd=0.15)
    time.sleep(1.5)

    # Greifen
    print(f"  → Greifer schließen...")
    arm.gripper_close()
    time.sleep(1.0)

    # Anheben
    print(f"  → Anheben...")
    arm.move_cartesian(arm_x, arm_y, arm_z + 80, t=3.14, spd=0.2)
    time.sleep(1.5)

    # Ablegen (Beispiel: 100mm nach rechts)
    place_y = arm_y - 100
    print(f"  → Ablegen bei Y={place_y:.0f}mm...")
    arm.move_cartesian(arm_x, place_y, arm_z + 20, t=3.14, spd=0.2)
    time.sleep(1.5)

    arm.gripper_open()
    time.sleep(0.5)

    # Zurückziehen
    arm.move_cartesian(arm_x, place_y, arm_z + 80, t=3.14, spd=0.25)
    time.sleep(1.0)

    print("  ✓ Vision-Greifen abgeschlossen!")


def run_live_detection(arm: RoArmM2S, duration: float = 10.0) -> None:
    """Zeigt Live-Erkennung für eine bestimmte Dauer (ohne Greifen)."""
    if not arm.vision or not arm.vision.available:
        return

    print(f"\n[V-Live] Live-Erkennung für {duration:.0f}s (nur Anzeige)...")

    try:
        import cv2
    except ImportError:
        print("  OpenCV nicht verfügbar für Live-Anzeige.")
        return

    start = time.time()
    frame_count = 0

    while time.time() - start < duration:
        frame = arm.vision.get_frame()
        if frame is None:
            break

        detections = arm.vision.detect_objects()
        frame_count += 1

        # Bounding Boxes zeichnen
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("RoArm Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    fps = frame_count / (time.time() - start)
    print(f"  {frame_count} Frames verarbeitet ({fps:.1f} FPS)")


def main():
    # Argument-Parser für flexible Nutzung
    parser = argparse.ArgumentParser(description="RoArm-M2-S Demo mit optionaler YOLO-Vision")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect wenn leer)")
    parser.add_argument("--camera", type=int, default=0, help="Kamera-Index (default: 0)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO-Modell Pfad")
    parser.add_argument("--target", type=str, default="cup", help="Ziel-Objekt für Vision-Greifen")
    parser.add_argument("--no-vision", action="store_true", help="Vision komplett deaktivieren")
    parser.add_argument("--no-movement", action="store_true", help="Nur Vision, kein Arm-Movement")
    parser.add_argument("--live", action="store_true", help="Live-Kamerabild mit Erkennung anzeigen")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min. Confidence für Erkennung")
    args = parser.parse_args()

    print("=" * 60)
    print("  RoArm-M2-S Control Demo")
    print("  mit optionaler YOLO-Objekterkennung")
    print("=" * 60)
    print()

    # Verbinden
    enable_vision = not args.no_vision
    with RoArmM2S(
        port=args.port,
        enable_vision=enable_vision,
        camera_index=args.camera,
        yolo_model=args.model,
        confidence=args.confidence
    ) as arm:
        print(f"\n{arm}\n")

        # Vision-Status anzeigen
        if arm.vision and arm.vision.available:
            print(f"  [✓] Vision aktiv (Kamera {args.camera}, Modell: {args.model})")
            print(f"      Ziel-Objekt: '{args.target}', Confidence: {args.confidence}")
        elif enable_vision:
            print(f"  [–] Vision angefragt aber nicht verfügbar")
            print(f"      (Kamera nicht gefunden oder ultralytics/opencv fehlt)")
        else:
            print(f"  [–] Vision deaktiviert (--no-vision)")

        # ─── Arm-Basis-Demo ───────────────────────────────────────────

        if not args.no_movement:
            # 1. Status lesen
            print("\n[1] Lese Arm-Status...")
            status = arm.get_status()
            if status:
                print_status(status)
            else:
                print("  (Status konnte nicht gelesen werden)")

            # 2. Home-Position
            print("\n[2] Fahre Home-Position...")
            arm.move_to_init()
            print("  ✓ Done.")

            # 3. LED-Bestätigung
            print("\n[3] LED-Flash...")
            for _ in range(3):
                arm.set_led(255)
                time.sleep(0.15)
                arm.set_led(0)
                time.sleep(0.15)
            print("  ✓ Done.")

            # 4. Kurze Bewegungs-Demo
            print("\n[4] Bewegungs-Demo...")
            arm.move_joints_degrees(b=20, s=0, e=90, h=180, spd=15, acc=10)
            time.sleep(0.8)
            arm.move_joints_degrees(b=-20, s=0, e=90, h=180, spd=15, acc=10)
            time.sleep(0.8)
            arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=15, acc=10)
            time.sleep(0.5)
            print("  ✓ Done.")

            # 5. Gripper-Demo
            print("\n[5] Gripper-Demo...")
            arm.gripper_open()
            time.sleep(0.6)
            arm.gripper_close()
            time.sleep(0.6)
            arm.gripper_open()
            time.sleep(0.3)
            print("  ✓ Done.")

        # ─── Vision-Demo ──────────────────────────────────────────────

        if args.live:
            run_live_detection(arm, duration=15.0)

        if enable_vision and arm.vision and arm.vision.available:
            run_vision_demo(arm, target=args.target)

        # ─── Abschluss ───────────────────────────────────────────────

        if not args.no_movement:
            print("\n[Ende] Parke Arm...")
            arm.park()
            time.sleep(1)
            print("  ✓ Done.")

        print("\n" + "=" * 60)
        print("  Demo abgeschlossen!")
        print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[Abgebrochen] Beende...")
        sys.exit(0)

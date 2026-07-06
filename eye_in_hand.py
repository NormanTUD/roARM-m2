"""
Eye-in-Hand Controller für RoArm-M2-S.

Kamera ist am Arm montiert. Diese Klasse:
- Wählt die Kamera aus (mit Live-Preview)
- Zeigt IMMER ein Live-Bild mit Detections (im Main-Thread, kein Qt-Problem)
- Sucht automatisch per 360°-Rotation wenn Objekt nicht sichtbar
- Zentriert iterativ über dem Objekt
- Greift und legt ab

Alles GUI-Rendering passiert im Main-Thread via update()-Calls.
"""

import time
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.live import Live
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from roarm_m2s import RoArmM2S, ArmStatus


# ─── Rich Console ────────────────────────────────────────────────────────────

console = Console() if HAS_RICH else None


def _print(msg: str, style: str = ""):
    if HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _header(title: str):
    if HAS_RICH:
        console.print(Panel(title, style="bold cyan", expand=True))
    else:
        print("=" * 60)
        print(f"  {title}")
        print("=" * 60)


def _success(msg: str):
    _print(f"  [bold green]✓[/bold green] {msg}" if HAS_RICH else f"  ✓ {msg}")


def _error(msg: str):
    _print(f"  [bold red]✗[/bold red] {msg}" if HAS_RICH else f"  ✗ {msg}")


def _info(msg: str):
    _print(f"  [dim]{msg}[/dim]" if HAS_RICH else f"  {msg}")


def _step(num: int, msg: str):
    _print(f"\n[bold yellow]\\[{num}][/bold yellow] {msg}" if HAS_RICH else f"\n[{num}] {msg}")


# ─── Kalibrierung ────────────────────────────────────────────────────────────

@dataclass
class Calibration:
    """Eye-in-Hand Kalibrierungsparameter. ANPASSEN FÜR DEIN SETUP!"""
    pixel_to_mm: float = 0.5       # mm pro Pixel bei Scan-Höhe
    grab_height: float = 75.0      # mm über Tisch zum Greifen
    scan_height: float = 200.0     # mm Scan-Höhe
    scan_forward: float = 180.0    # mm nach vorne in Scan-Position
    center_threshold_px: float = 20.0  # Pixel-Toleranz für "zentriert"
    damping: float = 0.6           # Dämpfung bei Zentrierung (0-1)
    base_scan_range: Tuple[float, float] = (-80.0, 80.0)  # Grad
    base_scan_step: float = 25.0   # Grad pro Schritt bei 360°-Suche


# ─── Controller ──────────────────────────────────────────────────────────────

class EyeInHandController:
    """
    Steuert den Arm + Kamera als Eye-in-Hand System.
    
    WICHTIG: Alle GUI-Operationen (imshow) laufen im Main-Thread!
    Kein Threading für die Anzeige.
    """

    def __init__(self, arm: RoArmM2S, camera_index: Optional[int] = None,
                 model_path: str = "yolo11n.pt", confidence: float = 0.5,
                 headless: bool = False, calibration: Optional[Calibration] = None):
        self.arm = arm
        self._headless = headless
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._camera_index = None
        self.cal = calibration or Calibration()
        self._window_name = "RoArm Eye-in-Hand"
        self._active = False

        if not HAS_CV2:
            _error("OpenCV nicht installiert: pip install opencv-python")
            return
        if not HAS_YOLO:
            _error("ultralytics nicht installiert: pip install ultralytics")
            return

        # Kamera auswählen und öffnen
        self._camera = self._select_camera(camera_index)
        if self._camera is None:
            _error("Keine Kamera verfügbar!")
            return

        # YOLO laden
        _info(f"Lade YOLO '{model_path}'...")
        try:
            self._model = YOLO(model_path)
            self._model.verbose = False
            # Warmup
            ret, frame = self._camera.read()
            if ret:
                self._model(frame, conf=self._confidence, verbose=False)
            _success(f"YOLO '{model_path}' bereit")
            self._active = True
        except Exception as e:
            _error(f"YOLO Fehler: {e}")
            self._camera.release()
            self._camera = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._camera:
            return (
                int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (640, 480)

    # ─── Kamera-Auswahl ──────────────────────────────────────────────────

    def _find_working_cameras(self) -> Dict[int, str]:
        """Findet Kameras die tatsächlich Frames liefern."""
        cameras = {}
        # Nur typische Indizes, nicht blind 0-7
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    backend = cap.getBackendName()
                    cameras[i] = f"{w}x{h} ({backend})"
                cap.release()
            time.sleep(0.05)
        return cameras

    def _select_camera(self, preferred: Optional[int] = None) -> Optional[cv2.VideoCapture]:
        """Wählt Kamera aus. Bei mehreren: Live-Preview zur Auswahl."""
        _info("Suche Kameras...")
        cameras = self._find_working_cameras()

        if not cameras:
            return None

        if HAS_RICH:
            table = Table(title="Verfügbare Kameras", show_header=True)
            table.add_column("Index", style="cyan")
            table.add_column("Auflösung", style="green")
            for idx, desc in cameras.items():
                table.add_row(str(idx), desc)
            console.print(table)
        else:
            print(f"\n  Kameras gefunden: {len(cameras)}")
            for idx, desc in cameras.items():
                print(f"    [{idx}] {desc}")

        # Wenn explizit angegeben
        if preferred is not None:
            if preferred in cameras:
                selected = preferred
            else:
                _error(f"Kamera {preferred} nicht verfügbar!")
                return None
        elif len(cameras) == 1:
            selected = list(cameras.keys())[0]
            _success(f"Kamera {selected} automatisch gewählt")
        else:
            # Mehrere → Live-Preview zur Auswahl
            selected = self._interactive_camera_select(cameras)
            if selected is None:
                return None

        # Öffnen
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            _error(f"Kamera {selected} konnte nicht geöffnet werden!")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera_index = selected

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _success(f"Kamera {selected} aktiv ({w}x{h})")
        return cap

    def _interactive_camera_select(self, cameras: Dict[int, str]) -> Optional[int]:
        """Zeigt jede Kamera live an. User wählt mit SPACE/ENTER."""
        if self._headless:
            first = list(cameras.keys())[0]
            _info(f"Headless-Modus: Kamera {first} gewählt")
            return first

        _info("Live-Preview: [SPACE/ENTER]=wählen, [N]=nächste, [Q]=abbrechen")

        for idx in cameras:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue

            start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed = time.time() - start
                # Overlay
                cv2.putText(frame, f"Kamera {idx}: {cameras[idx]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE/ENTER=waehlen | N=naechste | Q=abbrechen",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(frame, f"Auto-weiter in {max(0, 5 - elapsed):.0f}s",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

                cv2.imshow("Kamera-Auswahl", frame)
                key = cv2.waitKey(30) & 0xFF

                if key in (ord(' '), 13):  # Space/Enter
                    cap.release()
                    cv2.destroyWindow("Kamera-Auswahl")
                    _success(f"Kamera {idx} gewählt")
                    return idx
                elif key == ord('n'):
                    break
                elif key in (ord('q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

                if elapsed > 5.0:
                    break

            cap.release()

        cv2.destroyAllWindows()

        # Fallback: erste
        first = list(cameras.keys())[0]
        _info(f"Keine Auswahl → Kamera {first}")
        return first

    # ─── Frame + Detection ────────────────────────────────────────────────

    def get_frame(self):
        """Holt aktuellen Frame."""
        if not self._camera:
            return None
        ret, frame = self._camera.read()
        return frame if ret else None

    def detect(self, frame=None, target_classes: List[str] = None) -> List[Dict]:
        """Erkennt Objekte im Frame."""
        if not self._active:
            return []
        if frame is None:
            frame = self.get_frame()
            if frame is None:
                return []

        results = self._model(frame, conf=self._confidence, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if target_classes and cls_name not in target_classes:
                continue

            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'center_px': (cx, cy),
                'size_px': (x2 - x1, y2 - y1),
            })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def _annotate(self, frame, detections: List[Dict], target_classes: List[str] = None,
                  status_text: str = "") -> None:
        """Zeichnet Detections + Info auf Frame (in-place)."""
        if frame is None:
            return

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            is_target = target_classes and det['class'] in target_classes
            color = (0, 0, 255) if is_target else (0, 255, 0)

            if is_target:
                label = f"TARGET: {label}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cx, cy = int(det['center_px'][0]), int(det['center_px'][1])
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Fadenkreuz Bildmitte
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (128, 128, 128),
                       cv2.MARKER_CROSS, 30, 1)

        # Status-Text oben links
        if status_text:
            cv2.putText(frame, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _show(self, frame, wait_ms: int = 1) -> int:
        """Zeigt Frame an. Gibt Taste zurück. NUR MAIN-THREAD!"""
        if self._headless or frame is None:
            return -1
        cv2.imshow(self._window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def _update(self, target_classes: List[str] = None, status_text: str = "") -> Tuple[List[Dict], int]:
        """
        Ein kompletter Zyklus: Frame holen → Detect → Annotate → Show.
        Gibt (detections, key) zurück.
        """
        frame = self.get_frame()
        if frame is None:
            return [], -1

        detections = self.detect(frame, target_classes)
        self._annotate(frame, detections, target_classes, status_text)
        key = self._show(frame)
        return detections, key

    def _update_for(self, seconds: float, target_classes: List[str] = None,
                    status_text: str = "") -> None:
        """Zeigt Live-Preview für eine bestimmte Dauer."""
        start = time.time()
        while time.time() - start < seconds:
            _, key = self._update(target_classes, status_text)
            if key == ord('q'):
                break

    # ─── Pixel → Arm Transform ───────────────────────────────────────────

    def _pixel_offset_mm(self, center_px: Tuple[float, float]) -> Tuple[float, float]:
        """Pixel-Offset von Bildmitte → Arm-Offset in mm."""
        w, h = self.resolution
        offset_x = center_px[0] - (w / 2)
        offset_y = center_px[1] - (h / 2)

        # Kamera schaut runter, am Arm montiert:
        # Pixel-Y (unten im Bild) → Arm muss nach vorne (X+)
        # Pixel-X (rechts im Bild) → Arm muss nach links (Y-)
        dx_mm = offset_y * self.cal.pixel_to_mm
        dy_mm = -offset_x * self.cal.pixel_to_mm
        return (dx_mm, dy_mm)

    def _pixel_dist_from_center(self, center_px: Tuple[float, float]) -> float:
        """Pixel-Distanz von Bildmitte."""
        w, h = self.resolution
        dx = center_px[0] - (w / 2)
        dy = center_px[1] - (h / 2)
        return (dx**2 + dy**2) ** 0.5

    # ─── Status mit Retry ─────────────────────────────────────────────────

    def _get_status(self, retries: int = 3, delay: float = 0.5) -> Optional[ArmStatus]:
        """Holt Arm-Status mit Retries."""
        for i in range(retries):
            status = self.arm.get_status()
            if status:
                return status
            time.sleep(delay)
        return None

    # ─── 360° Suche ──────────────────────────────────────────────────────

    def _search_rotate(self, target_class: str) -> Optional[Dict]:
        """
        Dreht den Arm schrittweise und sucht das Objekt.
        Zeigt dabei Live-Preview.
        """
        _info("Starte Rotations-Suche...")
        start_deg, end_deg = self.cal.base_scan_range
        step = self.cal.base_scan_step
        current = start_deg

        while current <= end_deg:
            # Drehen
            self.arm.move_joints_degrees(
                b=current, s=0, e=45, h=180, spd=30, acc=15
            )

            # Warte + zeige Preview + suche
            start_t = time.time()
            while time.time() - start_t < 1.5:
                detections, key = self._update(
                    [target_class],
                    f"Suche '{target_class}' @ {current:.0f} deg"
                )
                if key == ord('q'):
                    return None
                if detections:
                    _success(f"'{target_class}' gefunden bei Base={current:.0f}°!")
                    return detections[0]

            current += step

        _error(f"'{target_class}' nicht gefunden im Bereich {start_deg}°..{end_deg}°")
        return None

    # ─── Zentrieren ───────────────────────────────────────────────────────

    def _center_over(self, target_class: str, max_iter: int = 8) -> bool:
        """Iterativ über dem Objekt zentrieren."""
        for i in range(max_iter):
            # Mehrere Frames für Stabilität
            detection = None
            for _ in range(5):
                detections, key = self._update(
                    [target_class],
                    f"Zentriere... (Iter {i+1}/{max_iter})"
                )
                if key == ord('q'):
                    return False
                if detections:
                    detection = detections[0]
                    break
                time.sleep(0.1)

            if not detection:
                _info(f"Iter {i+1}: Objekt verloren!")
                time.sleep(0.3)
                continue

            # Distanz prüfen
            pixel_dist = self._pixel_dist_from_center(detection['center_px'])
            if pixel_dist < self.cal.center_threshold_px:
                _success(f"Zentriert! (Dist={pixel_dist:.0f}px)")
                return True

            # Offset berechnen
            dx_mm, dy_mm = self._pixel_offset_mm(detection['center_px'])
            _info(f"Iter {i+1}: Offset=({dx_mm:.1f}, {dy_mm:.1f})mm, Dist={pixel_dist:.0f}px")

            # Status holen
            status = self._get_status()
            if not status:
                _error("Status nicht lesbar!")
                return False

            # Neue Position (gedämpft)
            new_x = status.x + dx_mm * self.cal.damping
            new_y = status.y + dy_mm * self.cal.damping

            # Sicherheitscheck
            dist = (new_x**2 + new_y**2) ** 0.5
            if dist > self.arm.MAX_REACH:
                _error(f"Außerhalb Reichweite ({dist:.0f}mm)!")
                return False

            # Bewegen + Preview während Bewegung
            self.arm.move_cartesian(new_x, new_y, status.z, t=3.14, spd=0.15)
            self._update_for(1.2, [target_class], "Bewege...")

        return False

    # ─── GRAB (Hauptlogik) ────────────────────────────────────────────────

    def grab(self, target_class: str, place_offset_y: float = -100) -> bool:
        """
        Findet und greift ein Objekt. Komplett automatisch mit Live-Preview.
        
        1. Scan-Position
        2. Suche (erst direkt, dann 360°)
        3. Zentrieren
        4. Absenken + Greifen
        5. Anheben + Ablegen
        """
        if not self._active:
            _error("Controller nicht aktiv!")
            return False

        _header(f"GRAB: '{target_class}'")

        # 1. Scan-Position
        _step(1, "Scan-Position...")
        self.arm.move_joints_degrees(b=0, s=0, e=45, h=180, spd=20, acc=10)
        self._update_for(2.5, [target_class], "Fahre Scan-Position...")

        # 2. Suche
        _step(2, f"Suche '{target_class}'...")
        detection = None

        # Erst direkt schauen (mehrere Frames)
        for _ in range(20):
            detections, key = self._update([target_class], f"Suche '{target_class}'...")
            if key == ord('q'):
                self._cleanup()
                return False
            if detections:
                detection = detections[0]
                break
            time.sleep(0.1)

        # Nicht gefunden → 360° Rotation
        if not detection:
            _info("Nicht direkt sichtbar → Rotations-Suche...")
            detection = self._search_rotate(target_class)

        if not detection:
            # Zeige was stattdessen da ist
            all_dets, _ = self._update(status_text="Nichts gefunden!")
            if all_dets:
                classes = set(d['class'] for d in all_dets)
                _info(f"Sichtbar: {', '.join(classes)}")
            self._update_for(2.0, status_text="Objekt nicht gefunden")
            self.arm.park()
            self._cleanup()
            return False

        _success(f"'{detection['class']}' (conf={detection['confidence']:.2f})")

        # 3. Zentrieren
        _step(3, "Zentriere über Objekt...")
        if not self._center_over(target_class):
            _error("Konnte nicht zentrieren!")
            self.arm.park()
            self._cleanup()
            return False

        # 4. Status für Greifposition
        status = self._get_status()
        if not status:
            _error("Status nicht lesbar!")
            self.arm.park()
            self._cleanup()
            return False

        grab_x = status.x
        grab_y = status.y

        _step(4, f"Greife bei X={grab_x:.0f} Y={grab_y:.0f} Z={self.cal.grab_height:.0f}...")

        # Gripper öffnen
        self.arm.gripper_open()
        self._update_for(0.5, [target_class], "Gripper offen")

        # Absenken
        _info("Absenken...")
        self.arm.move_cartesian(grab_x, grab_y, self.cal.grab_height + 50, t=3.14, spd=0.15)
        self._update_for(1.5, [target_class], "Absenken...")

        self.arm.move_cartesian(grab_x, grab_y, self.cal.grab_height, t=3.14, spd=0.1)
        self._update_for(1.5, [target_class], "Greifen...")

        # Greifen
        _info("Greifer schließen...")
        self.arm.gripper_close()
        self._update_for(1.0, status_text="Greife!")

        # 5. Anheben
        _step(5, "Anheben...")
        self.arm.move_cartesian(grab_x, grab_y, self.cal.grab_height + 120, t=3.14, spd=0.2)
        self._update_for(2.0, status_text="Anheben...")

        # 6. Ablegen
        place_y = grab_y + place_offset_y
        _step(6, f"Ablegen bei Y={place_y:.0f}...")
        self.arm.move_cartesian(grab_x, place_y, self.cal.grab_height + 40, t=3.14, spd=0.2)
        self._update_for(2.0, status_text="Ablegen...")

        self.arm.gripper_open()
        self._update_for(0.5, status_text="Abgelegt!")

        # Zurückziehen
        self.arm.move_cartesian(grab_x, place_y, self.cal.grab_height + 120, t=3.14, spd=0.25)
        self._update_for(1.5, status_text="Zurückziehen...")

        # 7. Parken
        _step(7, "Parken...")
        self.arm.park()
        self._update_for(2.0, status_text="Fertig!")

        self._cleanup()
        _success(f"'{target_class}' erfolgreich gegriffen und abgelegt!")
        return True

    def _cleanup(self):
        """Fenster schließen."""
        if not self._headless:
            cv2.destroyAllWindows()

    # ─── Live Scan (nur anzeigen) ─────────────────────────────────────────

    def live_scan(self) -> None:
        """
        Fährt in Scan-Position und zeigt Live-Stream mit allen Detections.
        Alles im Main-Thread – kein Qt-Problem.
        Beenden mit 'q'.
        """
        if not self._active:
            _error("Controller nicht aktiv!")
            return

        _header("LIVE SCAN")
        _info("Fahre Scan-Position...")
        self.arm.move_joints_degrees(b=0, s=0, e=45, h=180, spd=20, acc=10)
        time.sleep(2.0)

        _info("Live-Stream läuft (q=beenden)")
        fps_time = time.time()
        frame_count = 0

        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = self.detect(frame=frame)
            frame_count += 1
            elapsed = time.time() - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            classes_str = ", ".join(set(d['class'] for d in detections)) if detections else "---"
            self._annotate(frame, detections,
                           status_text=f"FPS:{fps:.1f} | {len(detections)} Obj | {classes_str}")

            key = self._show(frame)
            if key == ord('q'):
                break

        self._cleanup()
        self.arm.park()
        _success("Scan beendet.")

    # ─── Shutdown ─────────────────────────────────────────────────────────

    def shutdown(self):
        """Kamera freigeben und Fenster schließen."""
        self._cleanup()
        if self._camera:
            self._camera.release()
            self._camera = None
        self._active = False


# ─── Standalone Test-Demo ─────────────────────────────────────────────────────

def run_test_demo(arm: RoArmM2S) -> None:
    """Hardware-Test ohne Vision."""
    _header("HARDWARE TEST DEMO")

    _step(1, "Status lesen...")
    status = arm.get_status()
    if status:
        if HAS_RICH:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="cyan")
            table.add_column(style="green")
            table.add_row("Position", f"X={status.x:.1f}  Y={status.y:.1f}  Z={status.z:.1f} mm")
            table.add_row("Joints", f"B={status.base_rad:.3f}  S={status.shoulder_rad:.3f}  "
                          f"E={status.elbow_rad:.3f}  H={status.eoat_rad:.3f} rad")
            table.add_row("Spannung", f"{status.voltage:.2f}V")
            console.print(table)
        else:
            print(f"  Pos: X={status.x:.1f} Y={status.y:.1f} Z={status.z:.1f}")
            print(f"  V={status.voltage:.2f}V")
    else:
        _info("(Kein Status empfangen)")

    _step(2, "Home-Position...")
    arm.move_to_init()
    _success("OK")

    _step(3, "LED-Test...")
    for _ in range(3):
        arm.set_led(255)
        time.sleep(0.15)
        arm.set_led(0)
        time.sleep(0.15)
    _success("OK")

    _step(4, "Bewegungs-Test...")
    arm.move_joints_degrees(b=25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)
    arm.move_joints_degrees(b=-25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)
    arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
    _success("OK")

    _step(5, "Gripper-Test...")
    arm.gripper_open()
    time.sleep(0.5)
    arm.gripper_close()
    time.sleep(0.5)
    arm.gripper_open()
    _success("OK")

    _step(6, "Parken...")
    arm.park()
    time.sleep(1.0)
    _success("OK")

    if HAS_RICH:
        console.print(Panel("[bold green]Test-Demo abgeschlossen![/bold green]", style="green"))
    else:
        print("\n  ✓ Test-Demo abgeschlossen!")

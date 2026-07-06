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
from position_tracker import PositionTracker

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
    pixel_to_mm: float = 0.5
    grab_height: float = 75.0
    scan_height: float = 200.0
    scan_forward: float = 180.0
    center_threshold_px: float = 20.0
    damping: float = 0.6
    base_scan_range: Tuple[float, float] = (-90.0, 90.0)  # VOLLER Bereich
    base_scan_step: float = 20.0  # Kleinere Schritte = gründlicher

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

        self._init_tracker()

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
        """Findet Kameras die tatsächlich Frames liefern. Nutzt /dev/video* direkt."""
        import os
        cameras = {}

        # Finde echte Video-Devices über v4l2
        real_devices = []
        for i in range(10):
            dev = f"/dev/video{i}"
            if os.path.exists(dev):
                # Prüfe ob es ein CAPTURE device ist (nicht metadata)
                try:
                    import subprocess
                    result = subprocess.run(
                        ["v4l2-ctl", f"--device={dev}", "--all"],
                        capture_output=True, text=True, timeout=2
                    )
                    if "Video Capture" in result.stdout:
                        real_devices.append(i)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    # v4l2-ctl nicht verfügbar → Fallback
                    real_devices.append(i)

        # Fallback wenn v4l2-ctl nicht da ist: nur 0 und 2 probieren
        if not real_devices:
            real_devices = [0, 2]

        for i in real_devices:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras[i] = f"{w}x{h} (V4L2)"
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
                    status_text: str = "", discard_first: int = 3) -> None:
        """Zeigt Live-Preview für eine bestimmte Dauer. Verwirft erste Frames."""
        # Erste Frames verwerfen (Bewegungs-Artefakte / Kamera-Buffer)
        for _ in range(discard_first):
            self.get_frame()

        start = time.time()
        while time.time() - start < seconds:
            _, key = self._update(target_classes, status_text)
            if key == ord('q'):
                break

    # ─── Pixel → Arm Transform ───────────────────────────────────────────

    def _pixel_offset_mm(self, center_px: Tuple[float, float]) -> Tuple[float, float]:
        w, h = self.resolution
        offset_x = center_px[0] - (w / 2)
        offset_y = center_px[1] - (h / 2)

        # Kamera schaut NACH VORNE (nicht nach unten!):
        # Pixel-X rechts → Arm Y- (nach rechts)
        # Pixel-Y unten  → Arm Z- (nach unten)
        # Arm X (vor/zurück) ändert sich NICHT durch Pixelversatz!
        dy_mm = -offset_x * self.cal.pixel_to_mm
        dz_mm = -offset_y * self.cal.pixel_to_mm
        return (0.0, dy_mm, dz_mm)  # dx=0, nur Y und Z anpassen

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
        Dreht aus NORMALER Position (nicht nach oben geneigt).
        Wartet bis Arm still steht bevor Frames ausgewertet werden.
        """
        _info("Starte Rotations-Suche...")
        start_deg, end_deg = self.cal.base_scan_range
        step = self.cal.base_scan_step
        current = start_deg

        if HAS_RICH:
            console.print(f"  [dim]Bereich: {start_deg:.0f}° → {end_deg:.0f}° "
                          f"(Schritt: {step:.0f}°)[/dim]")

        while current <= end_deg:
            # Drehen – NORMALE Position (s=0, e=90, h=180 = geradeaus schauen)
            # Nicht e=45 (das neigt den Arm nach oben/unten)
            self._move_joints(b=current, s=0, e=90, h=180, spd=25, acc=10)

            # WICHTIG: Warte bis Arm WIRKLICH still steht
            # move_joints_degrees hat intern 2s wait, aber der Arm braucht
            # noch etwas um mechanisch zur Ruhe zu kommen
            time.sleep(0.5)

            # Frames verwerfen die während Bewegung aufgenommen wurden
            for _ in range(5):
                self.get_frame()

            # Jetzt stabile Frames auswerten + anzeigen
            found = False
            for frame_i in range(15):  # ~1.5s bei 10fps
                detections, key = self._update(
                    [target_class],
                    f"Suche '{target_class}' | Base: {current:.0f} deg"
                )
                if key == ord('q'):
                    return None
                if detections:
                    _success(f"'{target_class}' gefunden bei Base={current:.0f}°!")
                    return detections[0]
                time.sleep(0.05)

            current += step

        _error(f"'{target_class}' nicht gefunden ({start_deg:.0f}° bis {end_deg:.0f}°)")
        return None

    # ─── Position Tracker Integration ─────────────────────────────────────
    # Diese Methoden NACH __init__ einfügen (oder __init__ erweitern)

    def _init_tracker(self):
        """Initialisiert den Position-Tracker. Aufrufen am Ende von __init__."""
        self._tracker = PositionTracker()
        # Sync mit initialer Scan-Position (b=0, s=0, e=90, h=180)
        self._tracker.update_from_joints_degrees(b=0, s=0, e=90, h=180)

    def _move_joints(self, b=0, s=0, e=90, h=180, spd=20, acc=10):
        """Bewegt Joints UND trackt Position."""
        self.arm.move_joints_degrees(b=b, s=s, e=e, h=h, spd=spd, acc=acc)
        self._tracker.update_from_joints_degrees(b, s, e, h)

    def _move_cartesian(self, x: float, y: float, z: float, t: float = 3.14, spd: float = 0.25):
        """Bewegt kartesisch UND trackt Position."""
        self.arm.move_cartesian(x, y, z, t=t, spd=spd)
        self._tracker.update_from_cartesian(x, y, z)

    def _get_position(self) -> Tuple[float, float, float]:
        """
        Holt aktuelle Position.
        Versucht zuerst echten Status vom Arm.
        Fallback: Position aus dem Tracker (Forward Kinematics).
        """
        status = self._get_status(retries=2, delay=0.3)
        if status:
            # Sync Tracker mit echtem Feedback
            self._tracker.update_from_cartesian(status.x, status.y, status.z)
            return (status.x, status.y, status.z)
        else:
            _info(f"[Tracker-Fallback] Position: "
                  f"X={self._tracker.pos.x:.1f} "
                  f"Y={self._tracker.pos.y:.1f} "
                  f"Z={self._tracker.pos.z:.1f}")
            return self._tracker.cartesian

    # ─── Zentrieren ───────────────────────────────────────────────────────

    def _center_over(self, target_class: str, max_iter: int = 12) -> bool:
        """
        Zentriert das Objekt in der Bildmitte.
        
        Kamera schaut nach vorne:
        - Objekt rechts im Bild → Base nach rechts drehen (b+)
        - Objekt unten im Bild → Arm nach unten neigen (shoulder/elbow)
        - Objekt zu klein (weit weg) → nach vorne fahren (X+)
        
        Nutzt RELATIVE Joint-Korrekturen statt absoluter Cartesian-Befehle!
        """
        
        # --- Parameter ---
        TARGET_BB_HEIGHT = 200.0
        SMOOTHING_FRAMES = 4
        MIN_MOVE_PX = 8.0
        
        # Umrechnungsfaktoren (KALIBRIEREN!)
        # Wie viel Grad Base-Rotation pro Pixel horizontaler Offset
        DEG_PER_PIXEL_H = 0.08  # ~0.08°/px bei 640px Breite ≈ ±25° FOV
        # Wie viel Grad Shoulder-Neigung pro Pixel vertikaler Offset  
        DEG_PER_PIXEL_V = 0.06
        # Wie viel mm X-Bewegung pro Pixel Größen-Differenz
        MM_PER_PIXEL_DEPTH = 0.15
        
        DAMPING = 0.5
        
        # Aktuelle Joint-Winkel tracken (starten bei Scan-Position)
        cur_base = self._tracker.pos.base_deg
        cur_shoulder = self._tracker.pos.shoulder_deg
        cur_elbow = self._tracker.pos.elbow_deg
        
        prev_bb_height = None
        
        for i in range(max_iter):
            # --- Stabile Detection über mehrere Frames ---
            centers = []
            bb_heights = []
            bb_widths = []
            
            for _ in range(SMOOTHING_FRAMES * 3):
                detections, key = self._update(
                    [target_class],
                    f"Zentriere (Iter {i+1}/{max_iter}) | B={cur_base:.1f} S={cur_shoulder:.1f} E={cur_elbow:.1f}"
                )
                if key == ord('q'):
                    return False
                if detections:
                    det = detections[0]
                    centers.append(det['center_px'])
                    bb_heights.append(det['size_px'][1])
                    bb_widths.append(det['size_px'][0])
                    if len(centers) >= SMOOTHING_FRAMES:
                        break
                time.sleep(0.05)
            
            if len(centers) < 2:
                _info(f"Iter {i+1}: Objekt verloren!")
                time.sleep(0.5)
                continue
            
            # --- Gemittelte Werte ---
            avg_cx = sum(c[0] for c in centers) / len(centers)
            avg_cy = sum(c[1] for c in centers) / len(centers)
            avg_bb_h = sum(bb_heights) / len(bb_heights)
            avg_bb_w = sum(bb_widths) / len(bb_widths)
            
            # --- Pixel-Offset von Bildmitte ---
            w, h = self.resolution
            offset_px_x = avg_cx - (w / 2)   # positiv = rechts im Bild
            offset_px_y = avg_cy - (h / 2)   # positiv = unten im Bild
            pixel_dist = (offset_px_x**2 + offset_px_y**2) ** 0.5
            
            # --- Tiefe über BBox ---
            depth_error = TARGET_BB_HEIGHT - avg_bb_h  # positiv = zu weit weg
            
            _info(f"Iter {i+1}: Offset=({offset_px_x:.0f},{offset_px_y:.0f})px, "
                  f"BBox={avg_bb_w:.0f}x{avg_bb_h:.0f}px, Dist={pixel_dist:.0f}px, "
                  f"Depth={depth_error:+.0f}px")
            
            # --- Prüfe ob fertig ---
            centered = pixel_dist < self.cal.center_threshold_px
            close_enough = abs(depth_error) < 40
            
            if centered and close_enough:
                _success(f"Zentriert! (Dist={pixel_dist:.0f}px, BBox_H={avg_bb_h:.0f}px)")
                return True
            
            # --- RELATIVE Joint-Korrekturen berechnen ---
            
            # Horizontal: Base drehen
            # Objekt rechts im Bild → Base muss nach rechts (positiv) drehen
            d_base = 0.0
            if abs(offset_px_x) > MIN_MOVE_PX:
                d_base = -offset_px_x * DEG_PER_PIXEL_H * DAMPING
            
            # Vertikal: Shoulder anpassen
            # Objekt unten im Bild → Arm muss nach unten schauen → Shoulder erhöhen
            d_shoulder = 0.0
            if abs(offset_px_y) > MIN_MOVE_PX:
                d_shoulder = offset_px_y * DEG_PER_PIXEL_V * DAMPING
            
            # Tiefe: Elbow strecken (Arm nach vorne) oder beugen (zurück)
            d_elbow = 0.0
            if not close_enough:
                # Zu weit weg (depth_error > 0) → Elbow strecken (Winkel verkleinern)
                # Feedback: wenn BBox kleiner wurde, Richtung umkehren
                direction = -1.0 if depth_error > 0 else 1.0  # Elbow kleiner = gestreckter = weiter vorne
                
                if prev_bb_height is not None and hasattr(self, '_last_depth_dir'):
                    bb_change = avg_bb_h - prev_bb_height
                    if self._last_depth_dir * bb_change < 0 and abs(bb_change) > 5:
                        direction = -direction
                        _info("  [Feedback] Tiefe-Richtung korrigiert!")
                
                magnitude = min(abs(depth_error) / TARGET_BB_HEIGHT, 1.0) * 5.0  # max 5° pro Schritt
                d_elbow = direction * magnitude * DAMPING
                self._last_depth_dir = direction
            
            prev_bb_height = avg_bb_h
            
            # --- Neue Winkel ---
            new_base = cur_base + d_base
            new_shoulder = cur_shoulder + d_shoulder
            new_elbow = cur_elbow + d_elbow
            
            # Limits
            new_base = max(-90, min(90, new_base))
            new_shoulder = max(-30, min(60, new_shoulder))
            new_elbow = max(20, min(160, new_elbow))
            
            _info(f"  → dBase={d_base:+.1f}° dShoulder={d_shoulder:+.1f}° dElbow={d_elbow:+.1f}°")
            _info(f"  → Neu: B={new_base:.1f} S={new_shoulder:.1f} E={new_elbow:.1f}")
            
            # --- Bewegen ---
            self._move_joints(b=new_base, s=new_shoulder, e=new_elbow, h=180, spd=15, acc=10)
            
            cur_base = new_base
            cur_shoulder = new_shoulder
            cur_elbow = new_elbow
            
            # Warten + Frames verwerfen
            self._update_for(1.2, [target_class], "Bewege...")
        
        _error(f"Max Iterationen ({max_iter}) erreicht!")
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

        # 1. Normale Scan-Position (geradeaus schauen, NICHT nach oben)
        _step(1, "Scan-Position...")
        self._move_joints(b=0, s=0, e=90, h=180, spd=20, acc=10)
        self._update_for(2.5, [target_class], "Fahre Scan-Position...")

        # 2. Suche
        _step(2, f"Suche '{target_class}'...")
        detection = None

        # Frames verwerfen (Bewegungs-Artefakte)
        for _ in range(5):
            self.get_frame()

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

        # 4. Position für Greifposition (mit Tracker-Fallback)
        grab_x, grab_y, grab_z = self._get_position()
        if grab_x == 0 and grab_y == 0 and grab_z == 0:
            _error("Position unbekannt (Tracker nicht initialisiert)!")
            self.arm.park()
            self._cleanup()
            return False

        _step(4, f"Greife bei X={grab_x:.0f} Y={grab_y:.0f} Z={self.cal.grab_height:.0f}...")

        # Gripper öffnen
        self.arm.gripper_open()
        self._update_for(0.5, [target_class], "Gripper offen")

        # Absenken
        _info("Absenken...")
        self._move_cartesian(grab_x, grab_y, self.cal.grab_height + 50, t=3.14, spd=0.15)
        self._update_for(1.5, [target_class], "Absenken...")

        self._move_cartesian(grab_x, grab_y, self.cal.grab_height, t=3.14, spd=0.1)
        self._update_for(1.5, [target_class], "Greifen...")

        # Greifen
        _info("Greifer schließen...")
        self.arm.gripper_close()
        self._update_for(1.0, status_text="Greife!")

        # 5. Anheben
        _step(5, "Anheben...")
        self._move_cartesian(grab_x, grab_y, self.cal.grab_height + 120, t=3.14, spd=0.2)
        self._update_for(2.0, status_text="Anheben...")

        # 6. Ablegen
        place_y = grab_y + place_offset_y
        _step(6, f"Ablegen bei Y={place_y:.0f}...")
        self._move_cartesian(grab_x, place_y, self.cal.grab_height + 40, t=3.14, spd=0.2)
        self._update_for(2.0, status_text="Ablegen...")

        self.arm.gripper_open()
        self._update_for(0.5, status_text="Abgelegt!")

        # Zurückziehen
        self._move_cartesian(grab_x, place_y, self.cal.grab_height + 120, t=3.14, spd=0.25)
        self._update_for(1.5, status_text="Zurückziehen...")

        # 7. Parken
        _step(7, "Parken...")
        self._move_joints(b=0, s=0, e=90, h=180, spd=15, acc=10)
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

"""
RoArm-M2-S Full Abstraction Library
- Automatische Port-Erkennung
- Eye-in-Hand Vision mit YOLO
- Live-Preview im Main-Thread (kein Qt-Thread-Problem)
- Automatisches 360°-Suchen wenn Objekt nicht sichtbar
- Kamera auf dem Arm montiert
"""

import serial
import serial.tools.list_ports
import json
import time
import math
import sys
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class ArmStatus:
    """Parsed feedback from the arm (T:1051)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    base_rad: float = 0.0
    shoulder_rad: float = 0.0
    elbow_rad: float = 0.0
    eoat_rad: float = 0.0
    torque_base: int = 0
    torque_shoulder: int = 0
    torque_elbow: int = 0
    torque_hand: int = 0
    voltage: float = 0.0


class VisionModule:
    """
    Kamera + YOLO für Objekterkennung.
    Kein GUI hier – das macht der Main-Thread.
    """

    def __init__(self, camera_index: Optional[int] = None, model_path: str = "yolo11n.pt",
                 confidence: float = 0.5):
        self._available = False
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._cv2 = None
        self._camera_index = None

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            print("[Vision] ✗ OpenCV nicht installiert")
            return

        # Kamera finden
        selected_index = self._find_camera(camera_index)
        if selected_index is None:
            print("[Vision] ✗ Keine Kamera gefunden")
            return

        # Kamera öffnen
        cap = cv2.VideoCapture(selected_index)
        if not cap.isOpened():
            print(f"[Vision] ✗ Kamera {selected_index} nicht öffenbar")
            return

        self._camera = cap
        self._camera_index = selected_index
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Vision] ✓ Kamera {selected_index} ({w}x{h})")

        # YOLO laden
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model.verbose = False
            # Warmup
            ret, frame = self._camera.read()
            if ret:
                self._model(frame, conf=self._confidence, verbose=False)
            self._available = True
            print(f"[Vision] ✓ YOLO '{model_path}' bereit")
        except ImportError:
            print("[Vision] ✗ ultralytics nicht installiert")
            self._camera.release()
            self._camera = None
        except Exception as e:
            print(f"[Vision] ✗ Modell-Fehler: {e}")
            self._camera.release()
            self._camera = None

    def _find_camera(self, preferred_index: Optional[int]) -> Optional[int]:
        """Findet eine funktionierende Kamera. Probiert nur V4L2-Devices."""
        cv2 = self._cv2
        if preferred_index is not None:
            cap = cv2.VideoCapture(preferred_index)
            if cap.isOpened():
                cap.release()
                return preferred_index
            return None

        # Nur 0 und 2 probieren (typisch: 0=eingebaut, 2=USB)
        # /dev/video1, video3 etc. sind oft Metadata-Devices
        for i in [0, 2, 1, 4]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"[Vision] Kamera gefunden: Index {i}")
                    return i
        return None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._camera and self._cv2:
            w = int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (640, 480)

    def get_frame(self):
        """Aktuelles Kamerabild holen."""
        if not self._camera:
            return None
        ret, frame = self._camera.read()
        return frame if ret else None

    def detect_objects(self, target_classes: list = None, frame=None) -> list:
        """Erkennt Objekte im Frame."""
        if not self._available:
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
            w, h = x2 - x1, y2 - y1

            if target_classes and cls_name not in target_classes:
                continue

            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'center_px': (cx, cy),
                'size_px': (w, h),
            })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def annotate_frame(self, frame, detections: list, target_classes: list = None,
                       info_text: str = "") -> None:
        """Zeichnet Detections auf den Frame (in-place)."""
        cv2 = self._cv2
        if frame is None:
            return

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            color = (0, 255, 0)

            if target_classes and det['class'] in target_classes:
                color = (0, 0, 255)
                label = f"TARGET: {label}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cx, cy = int(det['center_px'][0]), int(det['center_px'][1])
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Fadenkreuz Bildmitte
        h_f, w_f = frame.shape[:2]
        cv2.drawMarker(frame, (w_f // 2, h_f // 2),
                       (128, 128, 128), cv2.MARKER_CROSS, 30, 1)

        # Info-Text
        if info_text:
            cv2.putText(frame, info_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def release(self):
        if self._camera:
            self._camera.release()
            self._camera = None


class RoArmM2S:
    """
    High-level Abstraction für den Waveshare RoArm-M2-S.
    Eye-in-Hand Konfiguration (Kamera am Arm montiert).
    """

    BAUDRATE = 115200
    ESP32_VID_PIDS = [
        (0x1A86, 0x7523), (0x1A86, 0x55D4), (0x10C4, 0xEA60),
        (0x303A, 0x1001), (0x0403, 0x6001), (0x0403, 0x6015),
    ]
    MAX_REACH = 320
    MIN_Z = -10
    MAX_Z = 350

    # Eye-in-Hand Kalibrierung
    PIXEL_TO_MM_SCALE = 0.5  # mm pro Pixel – KALIBRIEREN!
    GRAB_HEIGHT = 75  # mm über Tisch – KALIBRIEREN!
    SCAN_HEIGHT = 200  # mm Scan-Höhe
    SCAN_FORWARD = 180  # mm nach vorne in Scan-Position

    # Base-Joint Limits (Radians)
    BASE_MIN_RAD = -1.57  # -90°
    BASE_MAX_RAD = 1.57   # +90°

    def __init__(self, port: Optional[str] = None, baudrate: int = BAUDRATE,
                 auto_connect: bool = True, timeout: float = 1.0,
                 enable_vision: bool = False, camera_index: Optional[int] = None,
                 yolo_model: str = "yolo11n.pt", confidence: float = 0.5,
                 headless: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._connected = False
        self._lock = threading.Lock()
        self._headless = headless

        # Vision
        self.vision: Optional[VisionModule] = None
        if enable_vision:
            self.vision = VisionModule(
                camera_index=camera_index,
                model_path=yolo_model,
                confidence=confidence
            )

        if auto_connect:
            self.connect()

    # ─── Connection ───────────────────────────────────────────────────────

    @classmethod
    def find_port(cls) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.vid and p.pid:
                for vid, pid in cls.ESP32_VID_PIDS:
                    if p.vid == vid and p.pid == pid:
                        return p.device
        for p in ports:
            if any(n in p.device.lower() for n in ['ttyusb', 'ttyacm']):
                return p.device
        return None

    def connect(self, port: Optional[str] = None) -> None:
        if port:
            self.port = port
        if not self.port:
            self.port = self.find_port()
            if not self.port:
                raise ConnectionError("RoArm-M2-S nicht gefunden. Port manuell angeben.")

        print(f"[Arm] Verbinde {self.port}...")
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        self.ser.setRTS(True)
        self.ser.setDTR(True)
        time.sleep(2)
        self.ser.reset_input_buffer()
        self._connected = True
        print(f"[Arm] ✓ Verbunden")

    def disconnect(self) -> None:
        if self.vision:
            self.vision.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ser is not None and self.ser.is_open

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
        return False

    # ─── Low-Level Comm ───────────────────────────────────────────────────

    def _send(self, command: dict, wait_time: float = 0.5) -> List[str]:
        if not self.is_connected:
            raise ConnectionError("Nicht verbunden!")
        with self._lock:
            self.ser.write((json.dumps(command) + '\n').encode('utf-8'))
            self.ser.flush()
            time.sleep(wait_time)
            responses = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
            return responses

    def _parse_feedback(self, responses: List[str]) -> Optional[Dict]:
        for line in responses:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    # ─── Movement ─────────────────────────────────────────────────────────

    def move_to_init(self, wait: float = 3.0) -> List[str]:
        return self._send({"T": 100}, wait_time=wait)

    def move_joints_degrees(self, b=0, s=0, e=90, h=180, spd=10, acc=10) -> List[str]:
        return self._send({"T": 122, "b": b, "s": s, "e": e, "h": h, "spd": spd, "acc": acc}, 0.001)

    def move_joints_radians(self, base=0, shoulder=0, elbow=1.57, hand=3.14, spd=0, acc=10) -> List[str]:
        return self._send({"T": 102, "base": base, "shoulder": shoulder,
                           "elbow": elbow, "hand": hand, "spd": spd, "acc": acc}, 2.0)

    def move_single_joint_radians(self, joint: int, rad: float, spd=0, acc=10) -> List[str]:
        return self._send({"T": 101, "joint": joint, "rad": rad, "spd": spd, "acc": acc}, 2.0)

    def move_cartesian(self, x: float, y: float, z: float, t=3.14, spd=0.25) -> List[str]:
        return self._send({"T": 104, "x": x, "y": y, "z": z, "t": t, "spd": spd}, 3.0)

    def move_single_axis(self, axis: int, pos: float, spd=0.25) -> List[str]:
        return self._send({"T": 103, "axis": axis, "pos": pos, "spd": spd}, 3.0)

    # ─── Gripper ──────────────────────────────────────────────────────────

    def gripper_set(self, rad=3.14, spd=0, acc=0) -> List[str]:
        return self._send({"T": 106, "cmd": rad, "spd": spd, "acc": acc}, 1.0)

    def gripper_open(self, amount=1.08) -> List[str]:
        """Öffnet den Gripper. 1.08 = voll offen."""
        return self.gripper_set(rad=amount)

    def gripper_close(self, amount=3.14) -> List[str]:
        """Schließt den Gripper. 3.14 = voll geschlossen."""
        return self.gripper_set(rad=amount)

    def gripper_set_max_torque(self, torque: int = 300) -> List[str]:
        """
        Setzt maximales Gripper-Drehmoment.
        torque: 200 = 20%, 500 = 50%, 1000 = 100% des Servo-Max.
        """
        return self._send({"T": 107, "tor": torque}, 0.5)

    def gripper_close_until_resistance(self, torque_threshold: int = 80,
                                        timeout: float = 4.0,
                                        close_speed: int = 0,
                                        step_rad: float = 0.05) -> bool:
        """
        Schließt den Gripper schrittweise bis Widerstand erkannt wird.

        Strategie: Gripper in kleinen Schritten schließen, nach jedem Schritt
        torH aus dem Feedback prüfen. Wenn |torH| > threshold → Objekt gegriffen.

        Returns:
            True wenn Objekt gegriffen, False bei Timeout (nichts gegriffen).
        """
        import time

        # Aktuelle Position holen
        status = self.get_status()
        if not status:
            # Fallback: einfach schließen
            self.gripper_close()
            time.sleep(1.0)
            return True

        # Aktuelle Hand-Position (Radians)
        current_rad = status.eoat_rad
        target_rad = 3.14  # Voll geschlossen

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Nächsten Schritt berechnen
            current_rad = min(current_rad + step_rad, target_rad)

            # Gripper ein Stück weiter schließen
            self.gripper_set(rad=current_rad, spd=close_speed, acc=0)
            time.sleep(0.3)

            # Feedback holen
            status = self.get_status()
            if status:
                hand_torque = abs(status.torque_hand)
                if hand_torque > torque_threshold:
                    # Widerstand erkannt! Objekt gegriffen.
                    return True

            # Schon voll geschlossen?
            if current_rad >= target_rad - 0.01:
                break

        # Timeout oder voll geschlossen ohne Widerstand
        return False

    # ─── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Optional[ArmStatus]:
        responses = self._send({"T": 105}, wait_time=0.8)
        data = self._parse_feedback(responses)
        if data and data.get("T") == 1051:
            return ArmStatus(
                x=data.get("x", 0), y=data.get("y", 0), z=data.get("z", 0),
                base_rad=data.get("b", 0), shoulder_rad=data.get("s", 0),
                elbow_rad=data.get("e", 0), eoat_rad=data.get("t", 0),
                torque_base=data.get("torB", 0), torque_shoulder=data.get("torS", 0),
                torque_elbow=data.get("torE", 0), torque_hand=data.get("torH", 0),
                voltage=data.get("v", 0) / 100.0,
            )
        return None

    # ─── LED / Torque / Misc ──────────────────────────────────────────────

    def set_led(self, brightness: int = 0) -> List[str]:
        return self._send({"T": 114, "led": brightness}, 0.3)

    def set_torque(self, enable=True) -> List[str]:
        return self._send({"T": 210, "cmd": 1 if enable else 0}, 0.5)

    def park(self) -> List[str]:
        return self.move_joints_degrees(b=0, s=0, e=90, h=180, spd=15, acc=10)

    def home(self) -> List[str]:
        return self.move_to_init()

    # ─── Eye-in-Hand Transform ────────────────────────────────────────────

    def _pixel_to_arm_offset(self, center_px: Tuple[float, float]) -> Tuple[float, float]:
        """
        Pixel-Offset von Bildmitte → Arm-Offset in mm.
        Kamera auf Arm montiert, schaut nach unten.
        """
        img_w, img_h = self.vision.resolution
        offset_px_x = center_px[0] - (img_w / 2)
        offset_px_y = center_px[1] - (img_h / 2)

        # Kamera-Achsen → Arm-Achsen (Eye-in-Hand, Kamera schaut runter)
        dx_mm = offset_px_y * self.PIXEL_TO_MM_SCALE
        dy_mm = -offset_px_x * self.PIXEL_TO_MM_SCALE

        return (dx_mm, dy_mm)

    # ─── GUI Helper (Main-Thread!) ────────────────────────────────────────

    def _show_frame(self, frame, window_name="RoArm Vision") -> int:
        """Zeigt Frame an und gibt gedrückte Taste zurück. NUR IM MAIN-THREAD!"""
        if self._headless or frame is None:
            return -1
        cv2 = self.vision._cv2
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def _destroy_windows(self):
        if not self._headless and self.vision:
            self.vision._cv2.destroyAllWindows()

    # ─── Detect + Show (ein Schritt) ─────────────────────────────────────

    def _detect_and_show(self, target_classes=None, info="") -> list:
        """Holt Frame, detektiert, zeigt an, gibt Detections zurück."""
        if not self.vision or not self.vision.available:
            return []

        frame = self.vision.get_frame()
        if frame is None:
            return []

        detections = self.vision.detect_objects(target_classes, frame=frame)
        self.vision.annotate_frame(frame, detections, target_classes, info)
        self._show_frame(frame)
        return detections

    # ─── 360° Suche ───────────────────────────────────────────────────────

    def _search_360(self, target_class: str, step_degrees: float = 30,
                    frames_per_step: int = 10) -> Optional[dict]:
        """
        Dreht den Arm schrittweise um die Base-Achse (360° bzw. voller Bereich)
        bis das Zielobjekt gefunden wird.
        
        Returns:
            Detection-Dict wenn gefunden, None sonst.
        """
        print(f"[Suche] Starte 360°-Scan nach '{target_class}'...")

        # Voller Bereich: -90° bis +90° (Arm-Limits)
        start_deg = -90
        end_deg = 90
        current_deg = start_deg

        while current_deg <= end_deg:
            # Base drehen, Rest in Scan-Position lassen
            self.move_joints_degrees(b=current_deg, s=0, e=45, h=180, spd=30, acc=15)
            time.sleep(1.0)

            # Mehrere Frames checken (Kamera braucht Moment)
            for _ in range(frames_per_step):
                detections = self._detect_and_show(
                    [target_class],
                    info=f"Suche '{target_class}' @ {current_deg:.0f} deg"
                )
                if detections:
                    print(f"[Suche] ✓ '{target_class}' gefunden bei Base={current_deg:.0f}°!")
                    return detections[0]

                key = self._show_frame(self.vision.get_frame()) if self._headless else -1
                if key == ord('q'):
                    return None
                time.sleep(0.05)

            current_deg += step_degrees

        print(f"[Suche] ✗ '{target_class}' nicht gefunden im gesamten Bereich.")
        return None

    # ─── Grab Object (Hauptlogik) ─────────────────────────────────────────

    def grab_object(self, target_class: str, place_offset_y: float = -100) -> bool:
        """
        Findet und greift ein Objekt. Komplett automatisch.
        
        1. Scan-Position anfahren
        2. Objekt suchen (erst geradeaus, dann 360°-Scan)
        3. Über Objekt zentrieren (iterativ, mit Live-Preview)
        4. Absenken und greifen
        5. Anheben und ablegen
        
        Alles mit Live-Preview im Main-Thread.
        """
        if not self.vision or not self.vision.available:
            print("[Grab] ✗ Vision nicht verfügbar!")
            return False

        print(f"\n{'='*60}")
        print(f"  GRAB: '{target_class}'")
        print(f"{'='*60}")

        # 1. Scan-Position
        print("\n[1] Scan-Position...")
        self.move_joints_degrees(b=0, s=0, e=45, h=180, spd=20, acc=10)
        time.sleep(2.0)

        # 2. Objekt suchen (erst direkt, dann 360°)
        print(f"\n[2] Suche '{target_class}'...")
        detection = None

        # Erst ein paar Frames direkt checken
        for _ in range(15):
            detections = self._detect_and_show(
                [target_class], info=f"Suche '{target_class}'..."
            )
            if detections:
                detection = detections[0]
                break
            time.sleep(0.1)

        # Nicht gefunden → 360° Scan
        if not detection:
            print(f"  Nicht direkt sichtbar → starte Rotation...")
            detection = self._search_360(target_class)

        if not detection:
            # Zeige was stattdessen da ist
            all_det = self._detect_and_show(info="Nichts gefunden")
            if all_det:
                classes = set(d['class'] for d in all_det)
                print(f"  Sichtbar: {', '.join(classes)}")
            self._destroy_windows()
            self.park()
            return False

        print(f"  ✓ '{detection['class']}' (conf={detection['confidence']:.2f})")

        # 3. Zentrieren (iterativ)
        print(f"\n[3] Zentriere über Objekt...")
        centered = self._center_over_object(target_class, max_iter=8)

        if not centered:
            print("  ✗ Konnte nicht zentrieren")
            self._destroy_windows()
            self.park()
            return False

        # 4. Status holen für Greifposition
        status = self.get_status()
        if not status:
            # Retry
            time.sleep(0.5)
            status = self.get_status()
        if not status:
            print("  ✗ Kann Status nicht lesen")
            self._destroy_windows()
            self.park()
            return False

        grab_x = status.x
        grab_y = status.y

        print(f"\n[4] Greife bei X={grab_x:.0f} Y={grab_y:.0f} Z={self.GRAB_HEIGHT}...")

        # Gripper öffnen
        self.gripper_open()
        time.sleep(0.5)

        # Absenken (mit Live-Preview)
        print("  → Absenken...")
        self.move_cartesian(grab_x, grab_y, self.GRAB_HEIGHT + 50, t=3.14, spd=0.15)
        self._show_frames_during_wait(1.5, [target_class], "Absenken...")

        self.move_cartesian(grab_x, grab_y, self.GRAB_HEIGHT, t=3.14, spd=0.1)
        self._show_frames_during_wait(1.5, [target_class], "Greifen...")

        # Greifen
        print("  → Greifer schließen...")
        self.gripper_close()
        time.sleep(0.8)

        # 5. Anheben
        print("\n[5] Anheben...")
        self.move_cartesian(grab_x, grab_y, self.GRAB_HEIGHT + 120, t=3.14, spd=0.2)
        self._show_frames_during_wait(2.0, info="Anheben...")

        # 6. Ablegen
        place_y = grab_y + place_offset_y
        print(f"\n[6] Ablegen bei Y={place_y:.0f}...")
        self.move_cartesian(grab_x, place_y, self.GRAB_HEIGHT + 40, t=3.14, spd=0.2)
        self._show_frames_during_wait(2.0, info="Ablegen...")

        self.gripper_open()
        time.sleep(0.5)

        # Zurückziehen
        self.move_cartesian(grab_x, place_y, self.GRAB_HEIGHT + 120, t=3.14, spd=0.25)
        time.sleep(1.0)

        # Parken
        print("\n[7] Parken...")
        self.park()
        time.sleep(1.0)

        self._destroy_windows()
        print(f"\n  ✓ '{target_class}' erfolgreich gegriffen und abgelegt!")
        return True

    def _center_over_object(self, target_class: str, max_iter: int = 8,
                            threshold_px: float = 20, damping: float = 0.6) -> bool:
        """
        Iterativ über dem Objekt zentrieren.
        Bewegt den Arm so, dass das Objekt in der Bildmitte ist.
        """
        for i in range(max_iter):
            # Mehrere Frames für stabiles Ergebnis
            detection = None
            for _ in range(5):
                detections = self._detect_and_show(
                    [target_class],
                    info=f"Zentriere... (Iter {i+1}/{max_iter})"
                )
                if detections:
                    detection = detections[0]
                    break
                time.sleep(0.1)

            if not detection:
                print(f"  Iteration {i+1}: Objekt verloren!")
                time.sleep(0.3)
                continue

            # Offset berechnen
            img_w, img_h = self.vision.resolution
            offset_px_x = detection['center_px'][0] - (img_w / 2)
            offset_px_y = detection['center_px'][1] - (img_h / 2)
            pixel_dist = (offset_px_x**2 + offset_px_y**2) ** 0.5

            if pixel_dist < threshold_px:
                print(f"  ✓ Zentriert! (Dist={pixel_dist:.0f}px)")
                return True

            # Pixel → mm
            dx_mm, dy_mm = self._pixel_to_arm_offset(detection['center_px'])
            print(f"  Iter {i+1}: Offset=({dx_mm:.1f}, {dy_mm:.1f})mm, Dist={pixel_dist:.0f}px")

            # Status holen
            status = self.get_status()
            if not status:
                time.sleep(0.5)
                status = self.get_status()
            if not status:
                print(f"  ✗ Status nicht lesbar!")
                return False

            # Neue Position berechnen (gedämpft)
            new_x = status.x + dx_mm * damping
            new_y = status.y + dy_mm * damping

            # Sicherheitscheck
            dist = (new_x**2 + new_y**2) ** 0.5
            if dist > self.MAX_REACH:
                print(f"  ✗ Außerhalb Reichweite ({dist:.0f}mm)!")
                return False

            # Bewegen
            self.move_cartesian(new_x, new_y, status.z, t=3.14, spd=0.15)
            time.sleep(1.2)

        # Nach max_iter: prüfe ob nah genug
        return pixel_dist < threshold_px * 2 if 'pixel_dist' in dir() else False

    def _show_frames_during_wait(self, duration: float, target_classes=None,
                                  info: str = "") -> None:
        """Zeigt Live-Frames während einer Wartezeit (Main-Thread)."""
        start = time.time()
        while time.time() - start < duration:
            self._detect_and_show(target_classes, info)
            time.sleep(0.05)

    # ─── Scan Live (nur anzeigen) ─────────────────────────────────────────

    def scan_live(self) -> None:
        """
        Fährt in Scan-Position und zeigt Live-Stream mit Detections.
        Alles im Main-Thread – kein Qt-Problem.
        Beenden mit 'q'.
        """
        if not self.vision or not self.vision.available:
            print("[Scan] Vision nicht verfügbar.")
            return

        print("\n[Scan] Live-Stream (q=beenden)...")
        self.move_joints_degrees(b=0, s=0, e=45, h=180, spd=20, acc=10)
        time.sleep(2.0)

        cv2 = self.vision._cv2
        fps_time = time.time()
        frame_count = 0

        while True:
            frame = self.vision.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = self.vision.detect_objects(frame=frame)
            frame_count += 1
            fps = frame_count / (time.time() - fps_time) if time.time() > fps_time else 0

            self.vision.annotate_frame(frame, detections,
                                       info_text=f"FPS:{fps:.1f} | {len(detections)} Objekte")
            key = self._show_frame(frame)
            if key == ord('q'):
                break

        self._destroy_windows()
        self.park()

    # ─── Test Demo ────────────────────────────────────────────────────────

    def run_test_demo(self) -> None:
        """Hardware-Test ohne Vision."""
        print("\n[Demo] Hardware-Test...")

        print("  [1] Status...")
        status = self.get_status()
        if status:
            print(f"      Pos: X={status.x:.1f} Y={status.y:.1f} Z={status.z:.1f}")
            print(f"      V={status.voltage:.2f}V")

        print("  [2] Home...")
        self.move_to_init()

        print("  [3] LED...")
        for _ in range(3):
            self.set_led(255)
            time.sleep(0.15)
            self.set_led(0)
            time.sleep(0.15)

        print("  [4] Bewegung...")
        self.move_joints_degrees(b=25, s=0, e=90, h=180, spd=20, acc=10)
        time.sleep(0.8)
        self.move_joints_degrees(b=-25, s=0, e=90, h=180, spd=20, acc=10)
        time.sleep(0.8)

        print("  [5] Gripper...")
        self.gripper_open()
        time.sleep(0.5)
        self.gripper_close()
        time.sleep(0.5)
        self.gripper_open()

        print("  [6] Park...")
        self.park()
        time.sleep(1.0)

        print("\n  ✓ Demo fertig!")

    def _send_nowait(self, command: dict):
        """
        Send command without waiting for response. For continuous streaming.
        Uses non-blocking lock to avoid stalling the control loop.
        Periodically drains input buffer to prevent overflow.
        """
        if not self.is_connected:
            return
        # Non-blocking: if something else holds the lock, skip this cycle
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            msg = json.dumps(command, separators=(',', ':')) + '\n'
            self.ser.write(msg.encode('utf-8'))
            self.ser.flush()
            # Drain buffer every time, but don't use reset (can corrupt mid-message)
            while self.ser.in_waiting:
                self.ser.readline()
        finally:
            self._lock.release()


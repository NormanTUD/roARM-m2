"""
RoArm-M2-S Full Abstraction Library
Provides a high-level Pythonic interface for the Waveshare RoArm-M2-S robotic arm.
Supports automatic port detection, UART communication, optional YOLO vision.
"""

import serial
import serial.tools.list_ports
import json
import time
import sys
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union


@dataclass
class ArmStatus:
    """Parsed feedback from CMD_SERVO_RAD_FEEDBACK (T:105)."""
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
    torque_switch_base: bool = True
    torque_switch_shoulder: bool = True
    torque_switch_elbow: bool = True
    torque_switch_hand: bool = True
    voltage: float = 0.0


class VisionModule:
    """
    Optional camera + YOLO integration for object detection and grasping.
    Gracefully degrades if camera or YOLO is not available.
    Supports multiple cameras with interactive selection.
    """

    def __init__(self, camera_index: Optional[int] = None, model_path: str = "yolo11n.pt",
                 confidence: float = 0.5, auto_select: bool = True):
        self._available = False
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._cv2 = None
        self._camera_index = camera_index

        # ─── OpenCV Check ─────────────────────────────────────────────
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            print("[Vision] ✗ OpenCV nicht installiert (pip install opencv-python)")
            return

        # ─── Kamera finden / auswählen ────────────────────────────────
        available_cameras = self._enumerate_cameras()

        if not available_cameras:
            print("[Vision] ✗ Keine Kamera gefunden – Vision deaktiviert.")
            return

        if camera_index is not None:
            # Explizit angegeben
            if camera_index not in available_cameras:
                print(f"[Vision] ✗ Kamera {camera_index} nicht verfügbar.")
                print(f"         Verfügbare Kameras: {list(available_cameras.keys())}")
                return
            selected_index = camera_index
        elif len(available_cameras) == 1:
            # Nur eine Kamera → automatisch nehmen
            selected_index = list(available_cameras.keys())[0]
            print(f"[Vision] ✓ Eine Kamera gefunden (Index {selected_index})")
        elif auto_select:
            # Mehrere Kameras → User fragen
            selected_index = self._ask_user_camera(available_cameras)
            if selected_index is None:
                print("[Vision] ✗ Keine Kamera ausgewählt – Vision deaktiviert.")
                return
        else:
            selected_index = 0

        # ─── Kamera öffnen ────────────────────────────────────────────
        cap = cv2.VideoCapture(selected_index)
        if not cap.isOpened():
            print(f"[Vision] ✗ Kamera {selected_index} konnte nicht geöffnet werden.")
            return

        self._camera = cap
        self._camera_index = selected_index
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Vision] ✓ Kamera {selected_index} aktiv ({actual_w}x{actual_h})")

        # ─── YOLO Check ───────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            # Warmup-Inference
            ret, frame = self._camera.read()
            if ret:
                self._model(frame, conf=self._confidence, verbose=False)
            self._available = True
            print(f"[Vision] ✓ YOLO-Modell '{model_path}' geladen und bereit.")
        except ImportError:
            print("[Vision] ✗ ultralytics nicht installiert (pip install ultralytics)")
            self._camera.release()
            self._camera = None
            return
        except Exception as e:
            print(f"[Vision] ✗ Modell-Fehler: {e}")
            self._camera.release()
            self._camera = None
            return

    def _enumerate_cameras(self, max_check: int = 8) -> Dict[int, str]:
        """Prüft welche Kamera-Indizes verfügbar sind."""
        cv2 = self._cv2
        cameras = {}
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName()
                cameras[i] = f"Index {i}: {w}x{h} ({backend})"
                cap.release()
        return cameras

    def _ask_user_camera(self, cameras: Dict[int, str]) -> Optional[int]:
        """Fragt den User welche Kamera benutzt werden soll, mit Live-Preview."""
        cv2 = self._cv2
        print("\n[Vision] Mehrere Kameras gefunden:")
        for idx, desc in cameras.items():
            print(f"  [{idx}] {desc}")

        # Zeige kurze Preview jeder Kamera
        print("\n[Vision] Zeige Preview jeder Kamera (2s pro Kamera)...")
        for idx in cameras:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                start = time.time()
                while time.time() - start < 2.0:
                    ret, frame = cap.read()
                    if ret:
                        # Label draufschreiben
                        label = f"Kamera {idx} - Druecke Taste zum Waehlen, 'n' fuer naechste"
                        cv2.putText(frame, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("Kamera-Auswahl", frame)
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord('n'):
                            break
                        elif key != 255 and key != ord('n'):
                            cap.release()
                            cv2.destroyWindow("Kamera-Auswahl")
                            print(f"[Vision] Kamera {idx} ausgewählt.")
                            return idx
                cap.release()

        cv2.destroyAllWindows()

        # Fallback: Terminal-Eingabe
        try:
            choice = input(f"\n[Vision] Welche Kamera? [{'/'.join(str(k) for k in cameras)}]: ").strip()
            idx = int(choice)
            if idx in cameras:
                return idx
        except (ValueError, EOFError):
            pass

        # Default: erste Kamera
        first = list(cameras.keys())[0]
        print(f"[Vision] Verwende Kamera {first} (default).")
        return first

    @property
    def available(self) -> bool:
        return self._available

    @property
    def resolution(self) -> tuple:
        """Gibt die aktuelle Kamera-Auflösung zurück."""
        if self._camera and self._cv2:
            w = int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    def detect_objects(self, target_classes: list = None, frame=None) -> list:
        """
        Erkennt Objekte im aktuellen Kamerabild (oder übergebenem Frame).
        
        Returns:
            Liste von Dicts mit 'class', 'confidence', 'bbox', 'center_px', 'size_px'.
        """
        if not self._available:
            return []

        if frame is None:
            ret, frame = self._camera.read()
            if not ret:
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

    def detect_closest_to_center(self, target_classes: list = None) -> Optional[dict]:
        """Erkennt das Objekt, das am nächsten zur Bildmitte ist."""
        detections = self.detect_objects(target_classes)
        if not detections:
            return None

        w, h = self.resolution
        img_cx, img_cy = w / 2, h / 2

        def dist_to_center(det):
            dx = det['center_px'][0] - img_cx
            dy = det['center_px'][1] - img_cy
            return (dx**2 + dy**2) ** 0.5

        return min(detections, key=dist_to_center)

    def get_frame(self):
        """Gibt das aktuelle Kamerabild zurück."""
        if not self._available or not self._camera:
            return None
        ret, frame = self._camera.read()
        return frame if ret else None

    def get_annotated_frame(self, target_classes: list = None, frame=None):
        """Gibt Frame mit eingezeichneten Bounding Boxes zurück."""
        if not self._available:
            return None

        if frame is None:
            ret, frame = self._camera.read()
            if not ret:
                return None

        results = self._model(frame, conf=self._confidence, verbose=False)[0]
        annotated = results.plot()
        return annotated

    def release(self):
        """Kamera freigeben."""
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._cv2:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass

    def __del__(self):
        self.release()


class RoArmM2S:
    """
    High-level abstraction for the Waveshare RoArm-M2-S robotic arm.
    
    Usage:
        # Einfach (Test-Demo):
        with RoArmM2S() as arm:
            arm.move_to_init()
        
        # Mit Vision:
        with RoArmM2S(enable_vision=True) as arm:
            arm.grab_object("bottle")
    """

    BAUDRATE = 115200
    ESP32_VID_PIDS = [
        (0x1A86, 0x7523),  # CH340
        (0x1A86, 0x55D4),  # CH9102
        (0x10C4, 0xEA60),  # CP2102
        (0x303A, 0x1001),  # ESP32-S2/S3 native USB
        (0x0403, 0x6001),  # FTDI
        (0x0403, 0x6015),  # FTDI FT231X
    ]

    # Workspace-Grenzen (ungefähr, in mm)
    MAX_REACH = 320
    MIN_Z = -10
    MAX_Z = 350

    def __init__(self, port: Optional[str] = None, baudrate: int = BAUDRATE,
                 auto_connect: bool = True, timeout: float = 1.0,
                 enable_vision: bool = False, camera_index: Optional[int] = None,
                 yolo_model: str = "yolo11n.pt", confidence: float = 0.5):
        """
        Initialize the RoArm-M2-S controller.
        
        Args:
            port: Serial port. Auto-detected if None.
            baudrate: Communication baud rate (default 115200).
            auto_connect: Automatically connect on initialization.
            timeout: Serial read timeout in seconds.
            enable_vision: Try to initialize camera + YOLO (optional).
            camera_index: Which camera to use. None = auto/ask if multiple.
            yolo_model: Path to YOLO model file.
            confidence: Minimum detection confidence.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._connected = False
        self._lock = threading.Lock()

        # Live-Stream Thread
        self._live_thread: Optional[threading.Thread] = None
        self._live_running = False
        self._live_detections: list = []
        self._live_lock = threading.Lock()

        # Vision (optional – graceful degradation)
        self.vision: Optional[VisionModule] = None
        if enable_vision:
            self.vision = VisionModule(
                camera_index=camera_index,
                model_path=yolo_model,
                confidence=confidence
            )

        if auto_connect:
            self.connect()

    # ─── Connection Management ────────────────────────────────────────────

    @classmethod
    def find_port(cls) -> Optional[str]:
        """Automatically detect the serial port of the RoArm-M2-S."""
        ports = serial.tools.list_ports.comports()
        candidates = []

        for p in ports:
            if p.vid is not None and p.pid is not None:
                for vid, pid in cls.ESP32_VID_PIDS:
                    if p.vid == vid and p.pid == pid:
                        candidates.append(p.device)
                        break

        if not candidates:
            for p in ports:
                dev = p.device.lower()
                if any(name in dev for name in ['ttyusb', 'ttyacm', 'ch340', 'cp210', 'wchusbserial']):
                    candidates.append(p.device)

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            for p in ports:
                if p.device in candidates:
                    desc = (p.description or '').lower()
                    if 'esp32' in desc or 'cp210' in desc or 'ch340' in desc:
                        return p.device
            return candidates[0]
        return None

    @classmethod
    def list_ports(cls) -> List[Dict[str, Any]]:
        """List all available serial ports with details."""
        ports = serial.tools.list_ports.comports()
        return [{
            'device': p.device,
            'description': p.description,
            'hwid': p.hwid,
            'vid': hex(p.vid) if p.vid else None,
            'pid': hex(p.pid) if p.pid else None,
            'manufacturer': p.manufacturer,
        } for p in ports]

    def connect(self, port: Optional[str] = None) -> None:
        """Connect to the robotic arm."""
        if port:
            self.port = port

        if not self.port:
            self.port = self.find_port()
            if not self.port:
                raise ConnectionError(
                    "Could not auto-detect RoArm-M2-S. Available ports:\n"
                    + "\n".join(f"  {p['device']}: {p['description']}" for p in self.list_ports())
                    + "\nPlease specify the port manually."
                )

        print(f"[RoArm] Connecting on {self.port} @ {self.baudrate} baud...")
        self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
        self.ser.setRTS(True)
        self.ser.setDTR(True)
        time.sleep(2)
        self._flush_input()
        self._connected = True
        print(f"[RoArm] ✓ Connected on {self.port}")

    def disconnect(self) -> None:
        """Disconnect from the robotic arm and clean up."""
        self.stop_live_stream()
        if self.vision:
            self.vision.release()
        if self.ser and self.ser.is_open:
            self.ser.close()
            self._connected = False
            print("[RoArm] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ser is not None and self.ser.is_open

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ─── Low-Level Communication ──────────────────────────────────────────

    def _flush_input(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()

    def _send(self, command: dict, wait_time: float = 0.5) -> List[str]:
        """Send a JSON command and return response lines."""
        if not self.is_connected:
            raise ConnectionError("Not connected to RoArm-M2-S!")

        with self._lock:
            json_cmd = json.dumps(command) + '\n'
            self.ser.write(json_cmd.encode('utf-8'))
            self.ser.flush()
            time.sleep(wait_time)

            responses = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
            return responses

    def send_raw(self, json_string: str, wait_time: float = 0.5) -> List[str]:
        """Send a raw JSON string command."""
        if not self.is_connected:
            raise ConnectionError("Not connected to RoArm-M2-S!")

        with self._lock:
            self.ser.write((json_string.strip() + '\n').encode('utf-8'))
            self.ser.flush()
            time.sleep(wait_time)

            responses = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
            return responses

    def _parse_feedback(self, responses: List[str]) -> Optional[Dict[str, Any]]:
        """Try to parse a JSON response from the arm."""
        for line in responses:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    # ─── Live Stream ──────────────────────────────────────────────────────

    def start_live_stream(self, target_classes: list = None,
                          window_name: str = "RoArm Vision") -> None:
        """
        Startet einen Live-Stream mit YOLO-Detections in einem separaten Thread.
        Zeigt Bounding Boxes, Klassen und Confidence live an.
        """
        if not self.vision or not self.vision.available:
            print("[Live] Vision nicht verfügbar – kein Live-Stream.")
            return

        if self._live_running:
            print("[Live] Stream läuft bereits.")
            return

        self._live_running = True
        self._live_thread = threading.Thread(
            target=self._live_stream_loop,
            args=(target_classes, window_name),
            daemon=True
        )
        self._live_thread.start()
        print(f"[Live] ✓ Stream gestartet ('{window_name}', q=beenden)")

    def _live_stream_loop(self, target_classes: list, window_name: str) -> None:
        """Interner Live-Stream Loop (läuft in separatem Thread)."""
        cv2 = self.vision._cv2
        fps_start = time.time()
        frame_count = 0

        while self._live_running:
            frame = self.vision.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detection
            detections = self.vision.detect_objects(target_classes, frame=frame)

            # Detections für andere Threads verfügbar machen
            with self._live_lock:
                self._live_detections = detections

            # Annotieren
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                label = f"{det['class']} {det['confidence']:.2f}"
                color = (0, 255, 0)

                # Highlight target
                if target_classes and det['class'] in target_classes:
                    color = (0, 0, 255)
                    label = f">>> {label} <<<"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Mittelpunkt
                cx, cy = int(det['center_px'][0]), int(det['center_px'][1])
                cv2.circle(frame, (cx, cy), 5, color, -1)

            # Info-Overlay
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Objekte: {len(detections)}",
                f"Kamera: {self.vision._camera_index}",
            ]
            if target_classes:
                info_lines.append(f"Ziel: {', '.join(target_classes)}")

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Fadenkreuz (Bildmitte)
            h_frame, w_frame = frame.shape[:2]
            cv2.drawMarker(frame, (w_frame // 2, h_frame // 2),
                           (128, 128, 128), cv2.MARKER_CROSS, 20, 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._live_running = False
                break

        cv2.destroyWindow(window_name)

    def stop_live_stream(self) -> None:
        """Stoppt den Live-Stream."""
        if self._live_running:
            self._live_running = False
            if self._live_thread:
                self._live_thread.join(timeout=3.0)
                self._live_thread = None
            print("[Live] Stream gestoppt.")

    def get_live_detections(self) -> list:
        """Gibt die letzten Detections aus dem Live-Stream zurück (thread-safe)."""
        with self._live_lock:
            return list(self._live_detections)

    # ─── Movement: Reset ──────────────────────────────────────────────────

    def move_to_init(self, wait_time: float = 3.0) -> List[str]:
        """Move all joints to the initial/home position (CMD 100)."""
        return self._send({"T": 100}, wait_time=wait_time)

    # ─── Movement: Joint Angle Control (Degrees) ──────────────────────────

    def move_joints_degrees(self, b: float = 0, s: float = 0, e: float = 90,
                            h: float = 180, spd: float = 10, acc: float = 10) -> List[str]:
        """Move all joints using angles in degrees (CMD 122)."""
        cmd = {"T": 122, "b": b, "s": s, "e": e, "h": h, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    def move_single_joint_degrees(self, joint: int, angle: float,
                                   spd: float = 10, acc: float = 10) -> List[str]:
        """Move a single joint using angle in degrees (CMD 121)."""
        cmd = {"T": 121, "joint": joint, "angle": angle, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    # ─── Movement: Joint Angle Control (Radians) ──────────────────────────

    def move_joints_radians(self, base: float = 0, shoulder: float = 0,
                            elbow: float = 1.57, hand: float = 3.14,
                            spd: int = 0, acc: int = 10) -> List[str]:
        """Move all joints using radians (CMD 102)."""
        cmd = {"T": 102, "base": base, "shoulder": shoulder,
               "elbow": elbow, "hand": hand, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    def move_single_joint_radians(self, joint: int, rad: float,
                                   spd: int = 0, acc: int = 10) -> List[str]:
        """Move a single joint using radians (CMD 101)."""
        cmd = {"T": 101, "joint": joint, "rad": rad, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    # ─── Movement: Cartesian / Inverse Kinematics ─────────────────────────

    def move_cartesian(self, x: float, y: float, z: float,
                       t: float = 3.14, spd: float = 0.25) -> List[str]:
        """Move EoAT to XYZ coordinates using inverse kinematics (CMD 104)."""
        cmd = {"T": 104, "x": x, "y": y, "z": z, "t": t, "spd": spd}
        return self._send(cmd, wait_time=3.0)

    def move_cartesian_safe(self, x: float, y: float, z: float,
                            t: float = 3.14, spd: float = 0.25) -> List[str]:
        """Cartesian move mit Workspace-Grenzen-Check."""
        dist = (x**2 + y**2) ** 0.5
        if dist > self.MAX_REACH:
            raise ValueError(
                f"Position ({x:.0f}, {y:.0f}) außerhalb Reichweite "
                f"(Distanz={dist:.0f}mm > {self.MAX_REACH}mm)")
        if z < self.MIN_Z:
            raise ValueError(f"Z={z:.0f}mm unter Minimum ({self.MIN_Z}mm) – Kollisionsgefahr!")
        if z > self.MAX_Z:
            raise ValueError(f"Z={z:.0f}mm über Maximum ({self.MAX_Z}mm)")
        return self.move_cartesian(x, y, z, t, spd)

    def move_cartesian_direct(self, x: float, y: float, z: float,
                              t: float = 3.14) -> List[str]:
        """Move EoAT directly to XYZ (CMD 1041). Non-blocking."""
        cmd = {"T": 1041, "x": x, "y": y, "z": z, "t": t}
        return self._send(cmd, wait_time=0.1)

    def move_single_axis(self, axis: int, pos: float, spd: float = 0.25) -> List[str]:
        """Move a single axis to a position (CMD 103)."""
        cmd = {"T": 103, "axis": axis, "pos": pos, "spd": spd}
        return self._send(cmd, wait_time=3.0)

    # ─── Movement: Continuous Control ─────────────────────────────────────

    def continuous_move(self, mode: int = 0, axis: int = 1,
                        cmd: int = 0, spd: float = 5) -> List[str]:
        """Continuous movement control (CMD 123)."""
        return self._send({"T": 123, "m": mode, "axis": axis, "cmd": cmd, "spd": spd}, wait_time=0.2)

    def continuous_stop(self, mode: int = 0, axis: int = 1) -> List[str]:
        """Stop continuous movement on the given axis."""
        return self.continuous_move(mode=mode, axis=axis, cmd=0, spd=0)

    # ─── EoAT / Gripper Control ───────────────────────────────────────────

    def gripper_set(self, rad: float = 3.14, spd: int = 0, acc: int = 0) -> List[str]:
        """
        Set gripper/EoAT position in radians (CMD 106).
        
        Args:
            rad: Target angle. ~1.08 = fully open, ~3.14 = fully closed (clamp).
            spd: Speed. 0 = max.
            acc: Acceleration. 0 = max.
        """
        return self._send({"T": 106, "cmd": rad, "spd": spd, "acc": acc}, wait_time=1.0)

    def gripper_open(self, amount: float = 1.08) -> List[str]:
        """Open the gripper (default fully open)."""
        return self.gripper_set(rad=amount)

    def gripper_close(self, amount: float = 3.14) -> List[str]:
        """Close the gripper (default fully closed)."""
        return self.gripper_set(rad=amount)

    def set_gripper_torque(self, torque: int = 200) -> List[str]:
        """
        Set maximum gripper torque (CMD 107).
        Args:
            torque: 200 = 20% max torque, 1000 = 100% max torque.
        """
        return self._send({"T": 107, "tor": torque}, wait_time=0.5)

    def set_eoat_type(self, mode: int = 0) -> List[str]:
        """Set EoAT type (CMD 1). 0 = clamp (default), 1 = wrist."""
        return self._send({"T": 1, "cmd": mode}, wait_time=0.5)

    # ─── Feedback / Status ────────────────────────────────────────────────

    def get_status_raw(self) -> List[str]:
        """Get raw feedback from the arm (CMD 105)."""
        return self._send({"T": 105}, wait_time=0.5)

    def get_status(self) -> Optional[ArmStatus]:
        """Get parsed arm status including position, angles, torque, and voltage."""
        responses = self.get_status_raw()
        data = self._parse_feedback(responses)
        if data and data.get("T") == 1051:
            return ArmStatus(
                x=data.get("x", 0),
                y=data.get("y", 0),
                z=data.get("z", 0),
                base_rad=data.get("b", 0),
                shoulder_rad=data.get("s", 0),
                elbow_rad=data.get("e", 0),
                eoat_rad=data.get("t", 0),
                torque_base=data.get("torB", 0),
                torque_shoulder=data.get("torS", 0),
                torque_elbow=data.get("torE", 0),
                torque_hand=data.get("torH", 0),
                torque_switch_base=bool(data.get("torswitchB", 1)),
                torque_switch_shoulder=bool(data.get("torswitchS", 1)),
                torque_switch_elbow=bool(data.get("torswitchE", 1)),
                torque_switch_hand=bool(data.get("torswitchH", 1)),
                voltage=data.get("v", 0) / 100.0,
            )
        return None

    # ─── Torque Control ───────────────────────────────────────────────────

    def set_torque(self, enable: bool = True) -> List[str]:
        """Enable or disable torque lock (CMD 210)."""
        return self._send({"T": 210, "cmd": 1 if enable else 0}, wait_time=0.5)

    def torque_on(self) -> List[str]:
        return self.set_torque(True)

    def torque_off(self) -> List[str]:
        return self.set_torque(False)

    # ─── Dynamic Adaptation ───────────────────────────────────────────────

    def set_dynamic_adaptation(self, enable: bool = True,
                                b: int = 60, s: int = 110,
                                e: int = 50, h: int = 50) -> List[str]:
        """Enable/disable dynamic force self-adaptation (CMD 112)."""
        if enable:
            return self._send({"T": 112, "mode": 1, "b": b, "s": s, "e": e, "h": h}, wait_time=0.5)
        else:
            return self._send({"T": 112, "mode": 0, "b": 1000, "s": 1000, "e": 1000, "h": 1000}, wait_time=0.5)

    # ─── PID Control ──────────────────────────────────────────────────────

    def set_joint_pid(self, joint: int, p: int = 16, i: int = 0) -> List[str]:
        """Set PID values for a joint (CMD 108)."""
        return self._send({"T": 108, "joint": joint, "p": p, "i": i}, wait_time=0.5)

    def reset_pid(self) -> List[str]:
        """Reset all joint PID values to defaults (CMD 109)."""
        return self._send({"T": 109}, wait_time=0.5)

    # ─── LED Control ──────────────────────────────────────────────────────

    def set_led(self, brightness: int = 0) -> List[str]:
        """Set LED brightness (CMD 114). 0=off, 255=max."""
        return self._send({"T": 114, "led": brightness}, wait_time=0.3)

    # ─── WiFi Configuration ───────────────────────────────────────────────

    def wifi_get_info(self) -> List[str]:
        return self._send({"T": 405}, wait_time=1.0)

    def wifi_set_mode_on_boot(self, mode: int = 3) -> List[str]:
        return self._send({"T": 401, "cmd": mode}, wait_time=0.5)

    def wifi_set_sta(self, ssid: str, password: str) -> List[str]:
        return self._send({"T": 403, "ssid": ssid, "password": password}, wait_time=5.0)

    def wifi_set_ap(self, ssid: str = "RoArm-M2", password: str = "12345678") -> List[str]:
        return self._send({"T": 402, "ssid": ssid, "password": password}, wait_time=2.0)

    def wifi_stop(self) -> List[str]:
        return self._send({"T": 408}, wait_time=0.5)

    # ─── FLASH File System ────────────────────────────────────────────────

    def flash_scan_files(self) -> List[str]:
        return self._send({"T": 200}, wait_time=1.0)

    def flash_create_file(self, name: str, content: str) -> List[str]:
        return self._send({"T": 201, "name": name, "content": content}, wait_time=1.0)

    def flash_read_file(self, name: str) -> List[str]:
        return self._send({"T": 202, "name": name}, wait_time=1.0)

    def flash_delete_file(self, name: str) -> List[str]:
        return self._send({"T": 203, "name": name}, wait_time=1.0)

    # ─── Mission / Step Recording ─────────────────────────────────────────

    def mission_create(self, name: str, intro: str = "") -> List[str]:
        return self._send({"T": 220, "name": name, "intro": intro}, wait_time=1.0)

    def mission_append_current_pos(self, name: str, spd: float = 0.25) -> List[str]:
        return self._send({"T": 223, "name": name, "spd": spd}, wait_time=1.0)

    def mission_append_delay(self, name: str, delay_ms: int = 1000) -> List[str]:
        return self._send({"T": 224, "name": name, "delay": delay_ms}, wait_time=0.5)

    def mission_play(self, name: str, times: int = 1) -> List[str]:
        return self._send({"T": 242, "name": name, "times": times}, wait_time=2.0)

    # ─── ESP-NOW ──────────────────────────────────────────────────────────

    def espnow_get_mac(self) -> List[str]:
        return self._send({"T": 302}, wait_time=0.5)

    def espnow_set_mode(self, mode: int = 3) -> List[str]:
        return self._send({"T": 301, "mode": mode}, wait_time=0.5)

    # ─── Live Stream ──────────────────────────────────────────────────────

    def start_live_stream(self, target_classes: list = None,
                          window_name: str = "RoArm Vision") -> None:
        """Startet Live-Stream mit YOLO-Detections in separatem Thread."""
        if not self.vision or not self.vision.available:
            print("[Live] Vision nicht verfügbar.")
            return
        if self._live_running:
            return

        self._live_running = True
        self._live_target_classes = target_classes
        self._live_thread = threading.Thread(
            target=self._live_stream_loop,
            args=(target_classes, window_name),
            daemon=True
        )
        self._live_thread.start()
        print(f"[Live] ✓ Stream gestartet (q=beenden)")

    def _live_stream_loop(self, target_classes: list, window_name: str) -> None:
        """Interner Live-Stream Loop."""
        cv2 = self.vision._cv2
        fps_time = time.time()
        frame_count = 0

        while self._live_running:
            frame = self.vision.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detection
            detections = self.vision.detect_objects(target_classes, frame=frame)

            with self._live_lock:
                self._live_detections = detections

            # FPS
            frame_count += 1
            elapsed = time.time() - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Annotieren
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

            # Info
            cv2.putText(frame, f"FPS: {fps:.1f} | Objekte: {len(detections)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if target_classes:
                cv2.putText(frame, f"Ziel: {', '.join(target_classes)}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._live_running = False
                break

        cv2.destroyWindow(window_name)

    def stop_live_stream(self) -> None:
        """Stoppt den Live-Stream."""
        if self._live_running:
            self._live_running = False
            if self._live_thread:
                self._live_thread.join(timeout=3.0)
                self._live_thread = None
            print("[Live] Stream gestoppt.")

    def get_live_detections(self) -> list:
        """Gibt die letzten Detections aus dem Live-Stream zurück (thread-safe)."""
        with self._live_lock:
            return list(self._live_detections)

    # ─── Convenience / High-Level Methods ─────────────────────────────────

    def home(self) -> List[str]:
        return self.move_to_init()

    def park(self) -> List[str]:
        return self.move_joints_degrees(b=0, s=0, e=90, h=180, spd=15, acc=10)

    def pick_and_place(self, pick_x: float, pick_y: float, pick_z: float,
                       place_x: float, place_y: float, place_z: float,
                       hover_offset: float = 50, spd: float = 0.25) -> None:
        """Simple pick-and-place using Cartesian coordinates."""
        self.gripper_open()
        time.sleep(0.5)
        self.move_cartesian(pick_x, pick_y, pick_z + hover_offset, spd=spd)
        time.sleep(0.5)
        self.move_cartesian(pick_x, pick_y, pick_z, spd=spd)
        time.sleep(0.5)
        self.gripper_close()
        time.sleep(0.5)
        self.move_cartesian(pick_x, pick_y, pick_z + hover_offset, spd=spd)
        time.sleep(0.5)
        self.move_cartesian(place_x, place_y, place_z + hover_offset, spd=spd)
        time.sleep(0.5)
        self.move_cartesian(place_x, place_y, place_z, spd=spd)
        time.sleep(0.5)
        self.gripper_open()
        time.sleep(0.5)
        self.move_cartesian(place_x, place_y, place_z + hover_offset, spd=spd)

    def grab_object(self, target_class: str = "bottle",
                    place_offset_y: float = -100,
                    pixel_to_mm_func=None) -> bool:
        """
        Erkennt ein Objekt per YOLO und greift es. Eye-in-Hand Konfiguration.
        
        Der Arm fährt erst in eine Scan-Position, erkennt das Objekt,
        zentriert sich darüber, und greift dann.
        
        Args:
            target_class: YOLO-Klasse des Zielobjekts (z.B. "bottle", "cup", "cell phone")
            place_offset_y: Wo ablegen relativ zur Pick-Position (mm)
            pixel_to_mm_func: Custom Kalibrierungsfunktion. None = default.
            
        Returns:
            True wenn erfolgreich gegriffen, False sonst.
        """
        if not self.vision or not self.vision.available:
            print("[Grab] Vision nicht verfügbar.")
            return False

        if pixel_to_mm_func is None:
            pixel_to_mm_func = self._default_eye_in_hand_transform

        print(f"[Grab] Suche '{target_class}'...")

        # 1. Scan-Position (Arm hoch, nach vorne schauen)
        self.move_cartesian(180, 0, 200, t=3.14, spd=0.25)
        time.sleep(2.0)

        # 2. Erkennung
        detections = self.vision.detect_objects([target_class])
        if not detections:
            print(f"[Grab] '{target_class}' nicht gefunden.")
            # Zeige was stattdessen da ist
            all_det = self.vision.detect_objects()
            if all_det:
                classes = set(d['class'] for d in all_det)
                print(f"[Grab] Sichtbare Objekte: {', '.join(classes)}")
            return False

        best = detections[0]  # Bereits nach confidence sortiert
        print(f"[Grab] '{best['class']}' erkannt (conf={best['confidence']:.2f}) "
              f"@ pixel ({best['center_px'][0]:.0f}, {best['center_px'][1]:.0f})")

        # 3. Pixel → Arm-Offset berechnen (Eye-in-Hand)
        dx_mm, dy_mm = pixel_to_mm_func(best['center_px'])

        # 4. Aktuelle Position holen
        status = self.get_status()
        if not status:
            print("[Grab] Kann Arm-Status nicht lesen.")
            return False

        # 5. Über Objekt fahren (aktuelle Position + Offset)
        target_x = status.x + dx_mm
        target_y = status.y + dy_mm
        grab_z = 80  # Greifhöhe – KALIBRIEREN!

        print(f"[Grab] Fahre zu X={target_x:.0f} Y={target_y:.0f}...")

        # Sicherheitscheck
        dist = (target_x**2 + target_y**2) ** 0.5
        if dist > self.MAX_REACH:
            print(f"[Grab] Außerhalb Reichweite ({dist:.0f}mm > {self.MAX_REACH}mm)")
            return False

        # 6. Über Objekt positionieren
        self.move_cartesian(target_x, target_y, grab_z + 80, t=3.14, spd=0.2)
        time.sleep(1.5)

        # 7. Nochmal checken (Kamera ist jetzt näher dran)
        detections2 = self.vision.detect_objects([target_class])
        if detections2:
            best2 = detections2[0]
            dx2, dy2 = pixel_to_mm_func(best2['center_px'])
            # Fein-Korrektur
            if abs(dx2) > 5 or abs(dy2) > 5:
                target_x += dx2 * 0.5  # Gedämpft
                target_y += dy2 * 0.5
                print(f"[Grab] Fein-Korrektur → X={target_x:.0f} Y={target_y:.0f}")
                self.move_cartesian(target_x, target_y, grab_z + 80, t=3.14, spd=0.15)
                time.sleep(1.0)

        # 8. Greifen
        print("[Grab] Greife...")
        self.gripper_open()
        time.sleep(0.5)
        self.move_cartesian(target_x, target_y, grab_z, t=3.14, spd=0.15)
        time.sleep(1.5)
        self.gripper_close()
        time.sleep(1.0)

        # 9. Anheben
        self.move_cartesian(target_x, target_y, grab_z + 100, t=3.14, spd=0.2)
        time.sleep(1.5)

        # 10. Ablegen
        place_y = target_y + place_offset_y
        print(f"[Grab] Ablegen bei Y={place_y:.0f}...")
        self.move_cartesian(target_x, place_y, grab_z + 30, t=3.14, spd=0.2)
        time.sleep(1.5)
        self.gripper_open()
        time.sleep(0.5)

        # 11. Zurückziehen
        self.move_cartesian(target_x, place_y, grab_z + 100, t=3.14, spd=0.25)
        time.sleep(1.0)

        print("[Grab] ✓ Erfolgreich!")
        return True

    def _default_eye_in_hand_transform(self, center_px: tuple) -> tuple:
        """
        Default Pixel→mm Offset für Eye-in-Hand Konfiguration.
        
        Kamera ist am End-Effector montiert, schaut nach unten.
        Bildmitte = direkt unter dem Greifer.
        Offset in Pixel → Offset in mm den der Arm fahren muss.
        
        Returns:
            (dx_mm, dy_mm) – wie weit der Arm sich bewegen muss.
            
        WICHTIG: scale muss für dein Setup kalibriert werden!
        Hängt ab von: Kamerahöhe über Objekt, Brennweite, Auflösung.
        """
        img_w, img_h = self.vision.resolution
        px_x, px_y = center_px

        # Offset von Bildmitte (Pixel)
        offset_px_x = px_x - (img_w / 2)
        offset_px_y = px_y - (img_h / 2)

        # Skalierung: mm pro Pixel
        # Bei 640x480, Kamera ~15cm über Tisch: ca. 0.4-0.6 mm/px
        # KALIBRIEREN für dein Setup!
        scale = 0.5  # mm pro Pixel

        # Kamera-Achsen → Arm-Achsen Mapping
        # Annahme: Kamera X-Achse = Arm Y-Achse (links/rechts)
        #           Kamera Y-Achse = Arm X-Achse (vor/zurück)
        dx_mm = offset_px_y * scale   # Pixel-Y → Arm vorwärts
        dy_mm = -offset_px_x * scale  # Pixel-X → Arm links (invertiert)

        return (dx_mm, dy_mm)

    def scan_and_list_objects(self) -> List[Dict]:
        """
        Fährt in Scan-Position und listet alle sichtbaren Objekte auf.
        Nützlich um zu sehen was YOLO erkennt bevor man greift.
        """
        if not self.vision or not self.vision.available:
            print("[Scan] Vision nicht verfügbar.")
            return []

        # Scan-Position
        self.move_cartesian(180, 0, 220, t=3.14, spd=0.25)
        time.sleep(2.0)

        detections = self.vision.detect_objects()
        if detections:
            print(f"[Scan] {len(detections)} Objekt(e) erkannt:")
            for i, det in enumerate(detections):
                print(f"  [{i+1}] {det['class']} "
                      f"(conf={det['confidence']:.2f}, "
                      f"size={det['size_px'][0]:.0f}x{det['size_px'][1]:.0f}px)")
        else:
            print("[Scan] Keine Objekte erkannt.")
        return detections

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        vision_status = ""
        if self.vision:
            vision_status = f" vision={'active' if self.vision.available else 'inactive'}"
        return f"<RoArmM2S port={self.port} {status}{vision_status}>"


class VisionModule:
    """
    Optional camera + YOLO integration for object detection.
    Supports multiple cameras with interactive selection + live preview.
    Gracefully degrades if dependencies are missing.
    """

    def __init__(self, camera_index: Optional[int] = None, model_path: str = "yolo11n.pt",
                 confidence: float = 0.5):
        self._available = False
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._cv2 = None
        self._camera_index = None

        # ─── OpenCV Check ─────────────────────────────────────────────
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            print("[Vision] ✗ OpenCV nicht installiert (pip install opencv-python)")
            return

        # ─── Kamera finden / auswählen ────────────────────────────────
        available_cameras = self._enumerate_cameras()

        if not available_cameras:
            print("[Vision] ✗ Keine Kamera gefunden – Vision deaktiviert.")
            return

        if camera_index is not None:
            # Explizit angegeben
            if camera_index not in available_cameras:
                print(f"[Vision] ✗ Kamera {camera_index} nicht verfügbar.")
                print(f"         Verfügbar: {list(available_cameras.keys())}")
                return
            selected_index = camera_index
        elif len(available_cameras) == 1:
            # Nur eine → automatisch
            selected_index = list(available_cameras.keys())[0]
            print(f"[Vision] Eine Kamera gefunden (Index {selected_index})")
        else:
            # Mehrere → User fragen mit Live-Preview
            selected_index = self._ask_user_camera(available_cameras)
            if selected_index is None:
                print("[Vision] ✗ Keine Kamera ausgewählt.")
                return

        # ─── Kamera öffnen ────────────────────────────────────────────
        cap = self._cv2.VideoCapture(selected_index)
        if not cap.isOpened():
            print(f"[Vision] ✗ Kamera {selected_index} konnte nicht geöffnet werden.")
            return

        self._camera = cap
        self._camera_index = selected_index
        self._camera.set(self._cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(self._cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_w = int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Vision] ✓ Kamera {selected_index} aktiv ({actual_w}x{actual_h})")

        # ─── YOLO Check ───────────────────────────────────────────────
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            # Warmup
            ret, frame = self._camera.read()
            if ret:
                self._model(frame, conf=self._confidence, verbose=False)
            self._available = True
            print(f"[Vision] ✓ YOLO '{model_path}' geladen und bereit.")
        except ImportError:
            print("[Vision] ✗ ultralytics nicht installiert (pip install ultralytics)")
            self._camera.release()
            self._camera = None
            return
        except Exception as e:
            print(f"[Vision] ✗ Modell-Fehler: {e}")
            self._camera.release()
            self._camera = None
            return

    def _enumerate_cameras(self, max_check: int = 8) -> Dict[int, str]:
        """Prüft welche Kamera-Indizes verfügbar sind."""
        cv2 = self._cv2
        cameras = {}
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend = cap.getBackendName()
                cameras[i] = f"{w}x{h} ({backend})"
                cap.release()
            # Kurze Pause damit nicht alle gleichzeitig geöffnet werden
            time.sleep(0.1)
        return cameras

    def _ask_user_camera(self, cameras: Dict[int, str]) -> Optional[int]:
        """Fragt den User welche Kamera, zeigt Live-Preview jeder Kamera."""
        cv2 = self._cv2

        print(f"\n[Vision] {len(cameras)} Kameras gefunden:")
        for idx, desc in cameras.items():
            print(f"  [{idx}] {desc}")
        print()

        # Live-Preview jeder Kamera
        print("[Vision] Zeige Live-Preview (SPACE=auswählen, N=nächste, Q=abbrechen)...")

        for idx in cameras:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue

            selected = False
            start = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Info auf Frame
                elapsed = time.time() - start
                cv2.putText(frame, f"Kamera {idx}: {cameras[idx]}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE=waehlen | N=naechste | Q=abbrechen",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Auto-weiter in {max(0, 5-elapsed):.0f}s...",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                cv2.imshow("Kamera-Auswahl", frame)
                key = cv2.waitKey(30) & 0xFF

                if key == ord(' ') or key == 13:  # Space oder Enter
                    selected = True
                    break
                elif key == ord('n'):
                    break
                elif key == ord('q') or key == 27:  # Q oder ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

                # Auto-weiter nach 5 Sekunden
                if elapsed > 5.0:
                    break

            cap.release()

            if selected:
                cv2.destroyAllWindows()
                return idx

        cv2.destroyAllWindows()

        # Fallback: Terminal-Eingabe
        try:
            choice = input(f"\n[Vision] Welche Kamera? [{'/'.join(str(k) for k in cameras)}]: ").strip()
            idx = int(choice)
            if idx in cameras:
                return idx
        except (ValueError, EOFError):
            pass

        # Default: erste Kamera
        first = list(cameras.keys())[0]
        print(f"[Vision] Verwende Kamera {first} (default).")
        return first

    @property
    def available(self) -> bool:
        return self._available

    @property
    def resolution(self) -> tuple:
        if self._camera and self._cv2:
            w = int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    def detect_objects(self, target_classes: list = None, frame=None) -> list:
        """Erkennt Objekte. Kann eigenen Frame annehmen (für Live-Stream Thread)."""
        if not self._available:
            return []

        if frame is None:
            ret, frame = self._camera.read()
            if not ret:
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

    def detect_closest_to_center(self, target_classes: list = None) -> Optional[dict]:
        detections = self.detect_objects(target_classes)
        if not detections:
            return None
        w, h = self.resolution
        img_cx, img_cy = w / 2, h / 2

        def dist_to_center(det):
            dx = det['center_px'][0] - img_cx
            dy = det['center_px'][1] - img_cy
            return (dx**2 + dy**2) ** 0.5

        return min(detections, key=dist_to_center)

    def get_frame(self):
        if not self._available or not self._camera:
            return None
        ret, frame = self._camera.read()
        return frame if ret else None

    def get_annotated_frame(self, target_classes: list = None, frame=None):
        if not self._available:
            return None
        if frame is None:
            ret, frame = self._camera.read()
            if not ret:
                return None
        results = self._model(frame, conf=self._confidence, verbose=False)[0]
        return results.plot()

    def release(self):
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._cv2:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass

    def __del__(self):
        self.release()

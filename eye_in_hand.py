"""
Eye-in-Hand Controller für RoArm-M2-S.
Nutzt GrabSequencer für die Greif-Logik.
"""

import time
from typing import Optional, Tuple, List, Dict
from position_tracker import PositionTracker
from grab_sequencer import GrabSequencer, GrabConfig, GrabState

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
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from roarm_m2s import RoArmM2S, ArmStatus


# ─── Logging ─────────────────────────────────────────────────────────────────

console = Console() if HAS_RICH else None

DEBUG = False

def _print(msg: str, style: str = ""):
    if HAS_RICH:
        console.print(msg, style=style)
    else:
        print(msg)

def _header(title: str):
    if HAS_RICH:
        console.print(Panel(title, style="bold cyan", expand=True))
    else:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

def _success(msg: str):
    _print(f"  ✓ {msg}")

def _error(msg: str):
    _print(f"  ✗ {msg}")

def _info(msg: str):
    _print(f"  {msg}")

def _debug(msg: str):
    if DEBUG:
        _print(f"  [DBG] {msg}")


# ─── Arm Interface (für GrabSequencer) ───────────────────────────────────────

class _ArmInterface:
    """Adapter zwischen RoArmM2S und GrabSequencer."""

    def __init__(self, arm: RoArmM2S):
        self._arm = arm
        self._tracker = PositionTracker()
        self._tracker.update_from_joints_degrees(b=0, s=0, e=90, h=180)

    def move_joints(self, b=0, s=0, e=90, h=180, spd=20, acc=10):
        _debug(f"move_joints(b={b:.1f}, s={s:.1f}, e={e:.1f}, h={h:.1f})")
        self._arm.move_joints_degrees(b=b, s=s, e=e, h=h, spd=spd, acc=acc)
        self._tracker.update_from_joints_degrees(b, s, e, h)

    def move_cartesian(self, x: float, y: float, z: float, t: float = 3.14, spd: float = 0.25):
        _debug(f"move_cartesian(x={x:.1f}, y={y:.1f}, z={z:.1f})")
        self._arm.move_cartesian(x, y, z, t=t, spd=spd)
        self._tracker.update_from_cartesian(x, y, z)

    def gripper_open(self):
        _debug("gripper_open()")
        self._arm.gripper_open()

    def gripper_close_until_resistance(self, torque_threshold=60, timeout=4.0, step_rad=0.08) -> bool:
        _debug(f"gripper_close_until_resistance(threshold={torque_threshold})")
        return self._arm.gripper_close_until_resistance(
            torque_threshold=torque_threshold,
            timeout=timeout,
            step_rad=step_rad
        )

    def gripper_set_max_torque(self, torque: int = 300):
        self._arm.gripper_set_max_torque(torque)

    def get_position(self) -> Optional[Tuple[float, float, float]]:
        status = self._arm.get_status()
        if status:
            self._tracker.update_from_cartesian(status.x, status.y, status.z)
            _debug(f"get_position() → ({status.x:.1f}, {status.y:.1f}, {status.z:.1f})")
            return (status.x, status.y, status.z)
        # Fallback
        pos = self._tracker.cartesian
        _debug(f"get_position() → FALLBACK ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        return pos

    def park(self):
        self._arm.park()
        self._tracker.update_from_joints_degrees(b=0, s=0, e=90, h=180)


# ─── Vision Interface (für GrabSequencer) ────────────────────────────────────

class _VisionInterface:
    """Adapter zwischen Kamera/YOLO und GrabSequencer."""

    def __init__(self, camera, model, confidence: float, headless: bool, window_name: str):
        self._camera = camera
        self._model = model
        self._confidence = confidence
        self._headless = headless
        self._window_name = window_name
        self._frame_count = 0
        self._fps_time = time.time()
        self._last_fps = 0.0

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._camera:
            return (
                int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (640, 480)

    def get_frame(self):
        # Buffer flushen: 2x grab() verwirft alte Frames
        for _ in range(2):
            self._camera.grab()
        ret, frame = self._camera.retrieve()
        return frame if ret else None  # ← DAS FEHLTE!

    def detect(self, target_classes: List[str] = None) -> List[Dict]:
        """Detect ohne GUI."""
        frame = self.get_frame()
        if frame is None or not self._model:
            return []
        return self._run_detection(frame, target_classes)

    def update(self, target_classes: List[str] = None, status_text: str = "") -> Tuple[List[Dict], int]:
        """Detect + Annotate + Show. Gibt (detections, key) zurück."""
        frame = self.get_frame()
        if frame is None:
            return [], -1

        detections = self._run_detection(frame, target_classes)

        # FPS berechnen
        self._frame_count += 1
        elapsed = time.time() - self._fps_time
        if elapsed >= 1.0:
            self._last_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_time = time.time()

        # Debug-Info in Status einbauen
        fps_text = f"FPS:{self._last_fps:.1f}"
        if status_text:
            full_status = f"{fps_text} | {status_text}"
        else:
            full_status = fps_text

        # Debug: Detection-Info
        if DEBUG and target_classes and detections:
            det = detections[0]
            cx, cy = det['center_px']
            w, h = self.resolution
            off_x = cx - w/2
            off_y = cy - h/2
            full_status += f" | off=({off_x:.0f},{off_y:.0f})px conf={det['confidence']:.2f}"

        self._annotate(frame, detections, target_classes, full_status)
        key = self._show(frame)
        return detections, key

    def update_for(self, seconds: float, target_classes: List[str] = None,
                   status_text: str = "") -> None:
        """Live-Preview für eine Dauer."""
        start = time.time()
        while time.time() - start < seconds:
            _, key = self.update(target_classes, status_text)
            if key == ord('q'):
                break

    def _run_detection(self, frame, target_classes: List[str] = None) -> List[Dict]:
        if not self._model:
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

        if DEBUG:
            if target_classes:
                all_classes = set()
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    all_classes.add(results.names[cls_id])
                if not detections and all_classes:
                    _debug(f"Kein '{target_classes}' aber sehe: {all_classes}")

        return detections

    def _annotate(self, frame, detections: List[Dict], target_classes: List[str] = None,
                  status_text: str = "") -> None:
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

        h_f, w_f = frame.shape[:2]
        cv2.drawMarker(frame, (w_f // 2, h_f // 2),
                       (128, 128, 128), cv2.MARKER_CROSS, 30, 1)

        if status_text:
            cv2.putText(frame, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _show(self, frame, wait_ms: int = 1) -> int:
        if self._headless or frame is None:
            return -1
        cv2.imshow(self._window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def release(self):
        if self._camera:
            self._camera.release()
            self._camera = None


# ─── EyeInHandController ─────────────────────────────────────────────────────

class EyeInHandController:
    """
    Steuert den Arm + Kamera als Eye-in-Hand System.
    Nutzt GrabSequencer für die Greif-Logik.
    """

    def __init__(self, arm: RoArmM2S, camera_index: Optional[int] = None,
                 model_path: str = "yolo11n.pt", confidence: float = 0.5,
                 headless: bool = False, debug: bool = False):
        global DEBUG
        DEBUG = debug

        self.arm = arm
        self._headless = headless
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._window_name = "RoArm Eye-in-Hand"
        self._active = False
        self._debug = debug

        # Interfaces
        self._arm_iface: Optional[_ArmInterface] = None
        self._vision_iface: Optional[_VisionInterface] = None
        self._sequencer = GrabSequencer(self._arm_iface, self._vision_iface, debug=debug)

        if not HAS_CV2:
            _error("OpenCV nicht installiert: pip install opencv-python")
            return
        if not HAS_YOLO:
            _error("ultralytics nicht installiert: pip install ultralytics")
            return

        # Kamera
        self._camera = self._select_camera(camera_index)
        if self._camera is None:
            _error("Keine Kamera verfügbar!")
            return

        # YOLO
        _info(f"Lade YOLO '{model_path}'...")
        try:
            self._model = YOLO(model_path)
            self._model.verbose = False
            ret, frame = self._camera.read()
            if ret:
                self._model(frame, conf=self._confidence, verbose=False)
            _success(f"YOLO '{model_path}' bereit")
            self._active = True
        except Exception as e:
            _error(f"YOLO Fehler: {e}")
            self._camera.release()
            self._camera = None
            return

        # Interfaces erstellen
        self._arm_iface = _ArmInterface(arm)
        self._vision_iface = _VisionInterface(
            self._camera, self._model, self._confidence,
            self._headless, self._window_name
        )
        self._sequencer = GrabSequencer(self._arm_iface, self._vision_iface, debug=debug)

    @property
    def active(self) -> bool:
        return self._active

    # ─── Kamera-Auswahl ──────────────────────────────────────────────────

    def _find_working_cameras(self) -> Dict[int, str]:
        """Findet Kameras die tatsächlich Frames liefern."""
        import os
        cameras = {}

        real_devices = []
        for i in range(10):
            dev = f"/dev/video{i}"
            if os.path.exists(dev):
                try:
                    import subprocess
                    result = subprocess.run(
                        ["v4l2-ctl", f"--device={dev}", "--all"],
                        capture_output=True, text=True, timeout=2
                    )
                    if "Video Capture" in result.stdout:
                        real_devices.append(i)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    real_devices.append(i)

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
        """Wählt Kamera aus. Höchster Index = externe Kamera."""
        _info("Suche Kameras...")
        cameras = self._find_working_cameras()

        if not cameras:
            return None

        _info(f"Kameras gefunden: {len(cameras)}")
        for idx, desc in cameras.items():
            _info(f"  [{idx}] {desc}")

        if preferred is not None:
            if preferred in cameras:
                selected = preferred
            else:
                _error(f"Kamera {preferred} nicht verfügbar!")
                return None
        elif len(cameras) == 1:
            selected = list(cameras.keys())[0]
        else:
            selected = max(cameras.keys())

        _success(f"Kamera {selected} gewählt")

        cap = cv2.VideoCapture(selected, cv2.CAP_V4L2)
        if not cap.isOpened():
            _error(f"Kamera {selected} konnte nicht geöffnet werden!")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimaler Buffer gegen Ruckeln
        # Optional: FPS limitieren um CPU zu sparen
        cap.set(cv2.CAP_PROP_FPS, 30)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _success(f"Kamera {selected} aktiv ({w}x{h})")
        return cap

    # ─── Grab (delegiert an Sequencer) ───────────────────────────────────

    def grab(self, target_class: str, place_offset_y: float = -100) -> bool:
        """
        Findet und greift ein Objekt. Komplett automatisch mit Live-Preview.
        """
        if not self._active or not self._sequencer:
            _error("Controller nicht aktiv!")
            return False

        _header(f"GRAB: '{target_class}'")

        config = GrabConfig(
            target_class=target_class,
            place_offset_y=place_offset_y,
        )

        self._sequencer.start(target_class, config)

        # Tick-Loop (Main-Thread!)
        last_state = None
        while self._sequencer.running:
            state = self._sequencer.tick()

            # State-Wechsel loggen
            if state != last_state:
                _info(f"State: {state.name}")
                last_state = state

        # Ergebnis
        ctx = self._sequencer.context
        if ctx.success:
            _success(f"'{target_class}' erfolgreich gegriffen und abgelegt!")
            return True
        else:
            _error(f"Fehlgeschlagen: {ctx.error_msg}")
            return False

    # ─── Live Scan ────────────────────────────────────────────────────────

    def live_scan(self) -> None:
        """Fährt in Scan-Position und zeigt Live-Stream. Beenden mit 'q'."""
        if not self._active:
            _error("Controller nicht aktiv!")
            return

        _header("LIVE SCAN")
        self._arm_iface.move_joints(b=0, s=0, e=90, h=180, spd=20, acc=10)
        time.sleep(2.0)

        _info("Live-Stream (q=beenden)...")
        frame_count = 0
        fps_time = time.time()

        while True:
            elapsed = time.time() - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            detections, key = self._vision_iface.update(
                status_text=f"FPS:{fps:.1f} | {frame_count} frames"
            )
            frame_count += 1

            if key == ord('q'):
                break

        self._cleanup()
        self._arm_iface.park()

    # ─── Shutdown ─────────────────────────────────────────────────────────

    def shutdown(self):
        """Alles aufräumen."""
        self._cleanup()
        if self._vision_iface:
            self._vision_iface.release()

    def _cleanup(self):
        """GUI aufräumen."""
        if not self._headless and HAS_CV2:
            cv2.destroyAllWindows()


# ─── Test Demo (ohne Vision) ─────────────────────────────────────────────────

def run_test_demo(arm: RoArmM2S):
    """Hardware-Test ohne Vision."""
    _header("Hardware-Test")

    _info("[1] Status...")
    status = arm.get_status()
    if status:
        _info(f"    Pos: X={status.x:.1f} Y={status.y:.1f} Z={status.z:.1f}")
        _info(f"    V={status.voltage:.2f}V")

    _info("[2] Home...")
    arm.move_to_init()

    _info("[3] LED...")
    for _ in range(3):
        arm.set_led(255)
        time.sleep(0.15)
        arm.set_led(0)
        time.sleep(0.15)

    _info("[4] Bewegung...")
    arm.move_joints_degrees(b=25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)
    arm.move_joints_degrees(b=-25, s=0, e=90, h=180, spd=20, acc=10)
    time.sleep(0.8)

    _info("[5] Gripper...")
    arm.gripper_open()
    time.sleep(0.5)
    arm.gripper_close()
    time.sleep(0.5)
    arm.gripper_open()

    _info("[6] Park...")
    arm.park()
    time.sleep(1.0)

    _success("Demo fertig!")

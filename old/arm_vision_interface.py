"""
Interfaces die der GrabSequencer braucht.
Abstrahiert Arm-Hardware und Vision weg.
"""

from typing import Optional, Tuple, List, Dict
from position_tracker import PositionTracker
from roarm_m2s import RoArmM2S, ArmStatus


class ArmInterface:
    """
    Abstrahiert alle Arm-Operationen für den GrabSequencer.
    Inkl. Position-Tracking als Fallback.
    """
    
    def __init__(self, arm: RoArmM2S):
        self._arm = arm
        self._tracker = PositionTracker()
        self._tracker.update_from_joints_degrees(b=0, s=0, e=90, h=180)
    
    def move_joints(self, b=0, s=0, e=90, h=180, spd=20, acc=10):
        self._arm.move_joints_degrees(b=b, s=s, e=e, h=h, spd=spd, acc=acc)
        self._tracker.update_from_joints_degrees(b, s, e, h)
    
    def move_cartesian(self, x: float, y: float, z: float, t: float = 3.14, spd: float = 0.25):
        self._arm.move_cartesian(x, y, z, t=t, spd=spd)
        self._tracker.update_from_cartesian(x, y, z)
    
    def gripper_open(self):
        self._arm.gripper_open()
    
    def gripper_close_until_resistance(self, torque_threshold=60, timeout=4.0, step_rad=0.08) -> bool:
        return self._arm.gripper_close_until_resistance(
            torque_threshold=torque_threshold,
            timeout=timeout,
            step_rad=step_rad
        )
    
    def gripper_set_max_torque(self, torque: int = 300):
        self._arm.gripper_set_max_torque(torque)
    
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Position mit Fallback auf Tracker."""
        status = self._arm.get_status()
        if status:
            self._tracker.update_from_cartesian(status.x, status.y, status.z)
            return (status.x, status.y, status.z)
        # Fallback
        return self._tracker.cartesian
    
    def park(self):
        self._arm.park()
        self._tracker.update_from_joints_degrees(b=0, s=0, e=90, h=180)


class VisionInterface:
    """
    Abstrahiert Vision-Operationen für den GrabSequencer.
    Kapselt Kamera + YOLO + GUI.
    """
    
    def __init__(self, camera, model, confidence: float, headless: bool, window_name: str):
        self._camera = camera
        self._model = model
        self._confidence = confidence
        self._headless = headless
        self._window_name = window_name
        
        import cv2
        self._cv2 = cv2
    
    @property
    def resolution(self) -> Tuple[int, int]:
        if self._camera:
            return (
                int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        return (640, 480)

    def update(self, target_classes: List[str] = None, status_text: str = "") -> Tuple[List[Dict], int]:
        """
        Ein kompletter Zyklus: Frame holen → Detect → Annotate → Show.
        Gibt (detections, key) zurück.
        """
        frame = self._get_frame()
        if frame is None:
            return [], -1

        detections = self._detect(frame, target_classes)
        self._annotate(frame, detections, target_classes, status_text)
        key = self._show(frame)
        return detections, key

    def update_for(self, seconds: float, target_classes: List[str] = None,
                   status_text: str = "") -> None:
        """Zeigt Live-Preview für eine bestimmte Dauer."""
        start = time.time()
        while time.time() - start < seconds:
            _, key = self.update(target_classes, status_text)
            if key == ord('q'):
                break

    def get_frame(self):
        """Holt aktuellen Frame (grab+retrieve für neuesten)."""
        if not self._camera:
            return None
        self._camera.grab()
        ret, frame = self._camera.retrieve()
        return frame if ret else None

    # Alias für internen Gebrauch
    _get_frame = get_frame

    def _detect(self, frame, target_classes: List[str] = None) -> List[Dict]:
        """Erkennt Objekte im Frame."""
        if not self._model or frame is None:
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
        """Zeichnet Detections + Info auf Frame."""
        import cv2
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
        """Zeigt Frame an. NUR MAIN-THREAD!"""
        import cv2
        if self._headless or frame is None:
            return -1
        cv2.imshow(self._window_name, frame)
        return cv2.waitKey(wait_ms) & 0xFF

    def release(self):
        if self._camera:
            self._camera.release()
            self._camera = None


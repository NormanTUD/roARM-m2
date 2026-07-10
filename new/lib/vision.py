"""
lib/vision.py — YOLO Vision Pipeline

The robot AI only sees bounding boxes (class, position, size).
This dramatically reduces the input space and makes learning faster.

Concept (your fiber bundle idea):
- YOLO extracts objects → normalized bounding boxes
- Position in space is factored out (invariant representation)
- The policy learns movement patterns independent of absolute camera position
- Like rotating a fiber bundle: the base manifold (image space) is separated
  from the fiber (movement space)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time


@dataclass
class BoundingBox:
    """A single detected object — the ONLY visual input the policy sees."""
    class_name: str
    confidence: float
    # Normalized coordinates (0-1, relative to image)
    cx: float          # center x, normalized
    cy: float          # center y, normalized
    width: float       # bbox width, normalized
    height: float      # bbox height, normalized
    # Derived: offset from image center (for centering)
    offset_x: float    # -0.5 to +0.5 (negative = left of center)
    offset_y: float    # -0.5 to +0.5 (negative = above center)
    # Estimated distance (from known object size or calibration)
    estimated_distance_mm: float = 0.0


@dataclass
class VisionState:
    """
    Complete vision state for the policy network.
    This is what the neural network sees — NOT raw pixels.
    
    Encodes as a fixed-size vector for the policy:
    - Up to N target objects (padded with zeros if fewer)
    - Each object: [class_id, cx, cy, w, h, offset_x, offset_y, distance, confidence]
    """
    detections: List[BoundingBox] = field(default_factory=list)
    frame_timestamp: float = 0.0
    
    def to_vector(self, max_objects: int = 3, class_map: Dict[str, int] = None) -> np.ndarray:
        """
        Convert to fixed-size numpy vector for neural network input.
        
        Shape: [max_objects * 9]
        Per object: [class_id, cx, cy, w, h, offset_x, offset_y, distance_norm, confidence]
        
        This is the "genormte Position" — normalized so the network
        learns movement patterns invariant of absolute position.
        """
        vector = np.zeros(max_objects * 9, dtype=np.float32)
        
        for i, det in enumerate(self.detections[:max_objects]):
            offset = i * 9
            class_id = 0
            if class_map and det.class_name in class_map:
                class_id = class_map[det.class_name]
            
            vector[offset + 0] = class_id / max(len(class_map) if class_map else 1, 1)
            vector[offset + 1] = det.cx
            vector[offset + 2] = det.cy
            vector[offset + 3] = det.width
            vector[offset + 4] = det.height
            vector[offset + 5] = det.offset_x
            vector[offset + 6] = det.offset_y
            vector[offset + 7] = min(det.estimated_distance_mm / 1000.0, 1.0)  # normalize to [0,1]
            vector[offset + 8] = det.confidence

        return vector


class VisionPipeline:
    """
    Complete YOLO vision pipeline.
    
    Responsibilities:
    - Run YOLO inference on camera frames
    - Convert raw detections → BoundingBox objects
    - Estimate distance from known object sizes
    - Export frames as JPGs for annotation
    - Record which frames belong to which function (for DSL recording)
    """

    def __init__(self, model_path: str = "yolo_custom.pt",
                 fallback_model: str = "yolo11n.pt",
                 confidence: float = 0.5,
                 camera_index: int = 2,
                 class_map: Dict[str, int] = None,
                 known_object_sizes_mm: Dict[str, float] = None):
        """
        Args:
            model_path: Path to custom-trained YOLO model
            fallback_model: Pretrained model if custom not found
            confidence: Minimum detection confidence
            camera_index: Camera device index
            class_map: Mapping class_name → integer ID for vector encoding
            known_object_sizes_mm: Real-world widths for distance estimation
                                   e.g. {"bottle": 70, "cup": 80}
        """
        self._confidence = confidence
        self._camera = None
        self._model = None
        self._class_map = class_map or {}
        self._known_sizes = known_object_sizes_mm or {}
        self._frame_count = 0
        self._last_frame = None
        self._last_detections: List[BoundingBox] = []

        # Camera focal length (pixels) — calibrate for your camera!
        # For 640x480 with ~60° FOV: focal_px ≈ 554
        self._focal_length_px = 554.0

        self._setup_camera(camera_index)
        self._setup_model(model_path, fallback_model)

    def _setup_camera(self, camera_index: int):
        """Open camera with fallback."""
        import cv2
        self._cv2 = cv2

        for idx in ([camera_index] + [0, 2, 1, 4]):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self._camera = cap
                    self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"[Vision] ✓ Camera {idx}")
                    return
            cap.release()
        print("[Vision] ✗ No camera found")

    def _setup_model(self, model_path: str, fallback: str):
        """Load YOLO model."""
        from pathlib import Path
        try:
            from ultralytics import YOLO

            path = model_path if Path(model_path).exists() else fallback
            self._model = YOLO(path)
            self._model.verbose = False
            print(f"[Vision] ✓ YOLO model: {path}")

            # Warmup
            if self._camera:
                ret, frame = self._camera.read()
                if ret:
                    self._model(frame, conf=self._confidence, verbose=False)
        except ImportError:
            print("[Vision] ✗ ultralytics not installed")
        except Exception as e:
            print(f"[Vision] ✗ Model error: {e}")

    @property
    def available(self) -> bool:
        return self._camera is not None and self._model is not None

    @property
    def resolution(self) -> Tuple[int, int]:
        if self._camera:
            w = int(self._camera.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._camera.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (640, 480)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get current camera frame (raw, no annotations)."""
        if not self._camera:
            return None
        self._camera.grab()
        ret, frame = self._camera.retrieve()
        if ret:
            self._last_frame = frame
            self._frame_count += 1
        return frame if ret else None

    def detect(self, frame: np.ndarray = None,
               target_classes: List[str] = None) -> List[BoundingBox]:
        """
        Run YOLO detection → list of BoundingBox objects.
        
        This is the ONLY visual information the policy network receives.
        Raw pixels never reach the policy.
        """
        if not self.available:
            return []

        if frame is None:
            frame = self.get_frame()
            if frame is None:
                return []

        results = self._model(frame, conf=self._confidence, verbose=False)[0]
        img_h, img_w = frame.shape[:2]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])

            if target_classes and cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Normalized coordinates
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Offset from center
            offset_x = cx - 0.5
            offset_y = cy - 0.5

            # Distance estimation
            distance = self._estimate_distance(cls_name, x2 - x1, img_w)

            detections.append(BoundingBox(
                class_name=cls_name,
                confidence=conf,
                cx=cx, cy=cy,
                width=w, height=h,
                offset_x=offset_x,
                offset_y=offset_y,
                estimated_distance_mm=distance,
            ))

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        self._last_detections = detections
        return detections

    def _estimate_distance(self, class_name: str,
                           bbox_width_px: float, img_width: int) -> float:
        """
        Estimate distance using known object size and pinhole camera model.
        
        distance = (real_width * focal_length) / bbox_width_px
        
        This is the "genormte Position" idea:
        By knowing the real size, we can estimate depth and normalize
        the representation so the policy learns position-invariant movements.
        """
        if class_name not in self._known_sizes:
            return 0.0  # Unknown → 0 (policy ignores)

        real_width_mm = self._known_sizes[class_name]
        if bbox_width_px < 1:
            return 0.0

        distance_mm = (real_width_mm * self._focal_length_px) / bbox_width_px
        return distance_mm

    def get_vision_state(self, target_classes: List[str] = None,
                         max_objects: int = 3) -> VisionState:
        """
        Get complete vision state for the policy network.
        
        Returns a VisionState that can be converted to a fixed-size vector.
        This is the interface between vision and policy.
        """
        frame = self.get_frame()
        detections = self.detect(frame, target_classes)

        return VisionState(
            detections=detections[:max_objects],
            frame_timestamp=time.time(),
        )

    def center_arm_on_target(self, arm, target_class: str,
                             threshold_px: int = 20,
                             max_iter: int = 8,
                             damping: float = 0.6) -> bool:
        """
        Iteratively center the arm over a detected target.
        Uses pixel offset → arm movement mapping.
        
        Args:
            arm: RoArmController instance
            target_class: YOLO class to center on
            threshold_px: Pixel distance considered "centered"
            max_iter: Maximum centering iterations
            damping: Movement damping factor (0-1)
            
        Returns:
            True if successfully centered
        """
        PIXEL_TO_MM = 0.5  # Calibrate for your setup!

        for i in range(max_iter):
            detections = self.detect(target_classes=[target_class])
            if not detections:
                time.sleep(0.2)
                continue

            best = detections[0]
            img_w, img_h = self.resolution

            # Pixel offset from center
            offset_px_x = best.offset_x * img_w
            offset_px_y = best.offset_y * img_h
            dist_px = (offset_px_x**2 + offset_px_y**2) ** 0.5

            if dist_px < threshold_px:
                return True

            # Convert to mm (camera-to-arm transform)
            dx_mm = offset_px_y * PIXEL_TO_MM * damping
            dy_mm = -offset_px_x * PIXEL_TO_MM * damping

            # Move arm
            arm.move_cartesian_relative(dx=dx_mm, dy=dy_mm)
            time.sleep(0.8)

        return False

    # ═══ Frame Export (for YOLO annotation) ══════════════════════════════

    def export_frame_as_jpg(self, output_dir: str, prefix: str = "frame",
                            frame: np.ndarray = None) -> Optional[str]:
        """
        Export current frame as JPG for YOLO annotation software.
        
        Use this to build your custom dataset:
        1. Record a session
        2. Export frames
        3. Annotate in your YOLO annotation tool
        4. Train custom YOLO model
        5. Use custom model for better detection
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return None

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}_{self._frame_count:06d}.jpg"
        filepath = str(Path(output_dir) / filename)
        self._cv2.imwrite(filepath, frame)
        return filepath

    def export_frames_batch(self, output_dir: str, frames: List[np.ndarray],
                            prefix: str = "frame") -> List[str]:
        """Export multiple frames at once (e.g., from a recording session)."""
        paths = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:06d}.jpg"
            filepath = str(Path(output_dir) / filename)
            self._cv2.imwrite(filepath, frame)
            paths.append(filepath)
        return paths

    def release(self):
        """Release camera resources."""
        if self._camera:
            self._camera.release()
            self._camera = None


"""
Vision-System: YOLO-Wrapper der Bounding Boxes liefert.
Kann Bilder als JPG exportieren für Annotation.
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class BoundingBox:
    """Normalisierte Bounding Box (0-1 relativ zum Bild)."""
    x_center: float  # 0-1
    y_center: float  # 0-1
    width: float     # 0-1
    height: float    # 0-1

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Gibt (x1, y1, x2, y2) in Pixeln zurück."""
        cx = self.x_center * img_w
        cy = self.y_center * img_h
        w = self.width * img_w
        h = self.height * img_h
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    def to_flat(self) -> list:
        """Für neuronales Netz: [cx, cy, w, h] normalisiert."""
        return [self.x_center, self.y_center, self.width, self.height]


@dataclass
class Detection:
    """Eine einzelne YOLO-Detection."""
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_flat(self) -> list:
        """Für NN: [cx, cy, w, h, conf] — class wird separat kodiert."""
        return self.bbox.to_flat() + [self.confidence]


class VisionSystem:
    """
    Kamera + optionales YOLO-Modell.
    Liefert nur Bounding Boxes — das rohe Bild geht NICHT ans NN.
    """

    def __init__(self, camera_index: int = 2, model_path: Optional[str] = None,
                 confidence: float = 0.5):
        self._cv2 = None
        self._camera = None
        self._model = None
        self._confidence = confidence
        self._img_w = 640
        self._img_h = 480
        self._class_names: List[str] = []

        # OpenCV
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            raise ImportError("opencv-python benötigt: pip install opencv-python")

        # Kamera öffnen
        self._camera = self._open_camera(camera_index)
        if self._camera:
            self._img_w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._img_h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # YOLO (optional!)
        if model_path:
            self._load_model(model_path)

    def _open_camera(self, index: int):
        cv2 = self._cv2
        for idx in [index, 0, 2, 1, 4]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"[Vision] ✓ Kamera {idx}")
                    return cap
                cap.release()
        return None

    def _load_model(self, model_path: str):
        """Lädt YOLO-Modell. Fehlschlag ist OK — dann halt keine Detections."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model.verbose = False
            # Warmup
            frame = self.get_frame()
            if frame is not None:
                results = self._model(frame, conf=self._confidence, verbose=False)[0]
                self._class_names = list(results.names.values())
            print(f"[Vision] ✓ YOLO '{model_path}' ({len(self._class_names)} Klassen)")
        except Exception as e:
            print(f"[Vision] ⚠ YOLO nicht geladen: {e}")
            self._model = None

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def has_camera(self) -> bool:
        return self._camera is not None and self._camera.isOpened()

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._img_w, self._img_h)

    def get_frame(self):
        """Holt aktuellen Frame (None wenn keine Kamera)."""
        if not self._camera:
            return None
        self._camera.grab()
        ret, frame = self._camera.retrieve()
        return frame if ret else None

    def detect(self, frame=None, target_classes: List[str] = None) -> List[Detection]:
        """
        Führt YOLO-Detection aus.
        Gibt leere Liste zurück wenn kein Modell geladen.
        """
        if not self._model:
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

            if target_classes and cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = BoundingBox(
                x_center=(x1 + x2) / 2 / self._img_w,
                y_center=(y1 + y2) / 2 / self._img_h,
                width=(x2 - x1) / self._img_w,
                height=(y2 - y1) / self._img_h,
            )
            detections.append(Detection(cls_name, conf, bbox))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detections_to_nn_input(self, detections: List[Detection],
                               max_objects: int = 5) -> list:
        """
        Konvertiert Detections in einen flachen Vektor für das NN.
        Feste Größe: max_objects * 5 (cx, cy, w, h, conf).
        Nicht-erkannte Slots werden mit Nullen gefüllt.

        Das ist der EINZIGE visuelle Input den das NN bekommt!
        → Invariant gegenüber Textur, Beleuchtung, Hintergrund
        → Nur Position und Größe der relevanten Objekte
        """
        flat = []
        for i in range(max_objects):
            if i < len(detections):
                flat.extend(detections[i].to_flat())
            else:
                flat.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        return flat

    def save_frame_as_jpg(self, frame, output_path: Path, prefix: str = "frame"):
        """Speichert Frame als JPG für YOLO-Annotation."""
        if frame is None:
            return None
        timestamp = int(time.time() * 1000)
        filename = output_path / f"{prefix}_{timestamp}.jpg"
        self._cv2.imwrite(str(filename), frame)
        return filename

    def draw_detections(self, frame, detections: List[Detection],
                        highlight_classes: List[str] = None):
        """Zeichnet Bounding Boxes auf Frame (nur für Display, nicht für NN!)."""
        cv2 = self._cv2
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_pixel(self._img_w, self._img_h)
            is_target = highlight_classes and det.class_name in highlight_classes
            color = (0, 0, 255) if is_target else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def release(self):
        if self._camera:
            self._camera.release()


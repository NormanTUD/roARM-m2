#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "opencv-python",
#     "pyserial",
#     "ultralytics",
# ]
# ///

import os
import sys

def _ensure_uv():
    if os.environ.get("_UV_SAFE_ENV") == "1":
        return
    os.environ["_UV_SAFE_ENV"] = "1"
    from datetime import datetime, timedelta, timezone
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

"""
BBox-Policy Runner — Führt trainierte BBox-Policy auf dem Arm aus.

Der Roboter sieht NUR Bounding Boxes (via YOLO) und seinen eigenen State.
Kein rohes Bild geht ans NN — nur abstrahierte Positionen.

Usage:
    python3 run_bbox_policy.py trained_bbox_policy/bbox_policy_best.pt --yolo my_model.pt
    python3 run_bbox_policy.py trained_bbox_policy/bbox_policy_best.pt --yolo my_model.pt --step
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from roarm_lib.hardware import RoArmHardware, ArmState, SPEED_PRESETS
from roarm_lib.vision import VisionSystem
from roarm_lib.policy import BBoxPolicy, BBoxObservation

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required!")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class BBoxPolicyRunner:
    """
    Führt eine trainierte BBox-Policy in Echtzeit aus.

    Loop:
    1. Kamera-Frame holen
    2. YOLO → Bounding Boxes
    3. BBoxes + Arm-State → NN-Input (flacher Vektor, ~80 dim)
    4. NN → Action-Chunk
    5. Erste Aktion ausführen
    6. Repeat
    """

    CONTROL_HZ = 30

    def __init__(self, model_path: str, yolo_model: str,
                 port: str = None, camera_index: int = 2,
                 confidence: float = 0.5, speed_scale: float = 1.0):
        self._model_path = model_path
        self._yolo_model = yolo_model
        self._port = port
        self._camera_index = camera_index
        self._confidence = confidence
        self._speed_scale = speed_scale

        self._hw: Optional[RoArmHardware] = None
        self._vision: Optional[VisionSystem] = None
        self._model: Optional[BBoxPolicy] = None
        self._encoder: Optional[BBoxObservation] = None

        self._state = ArmState()
        self._running = False
        self._executing = False
        self._action_queue: List[np.ndarray] = []

        # Config from checkpoint
        self._config = {}
        self._class_names: List[str] = []

    def setup(self) -> bool:
        """Initialisiert alles."""
        print("=" * 50)
        print("  🤖 BBox-Policy Runner")
        print("=" * 50)

        # 1. Modell laden
        print(f"\n  [1/3] Modell laden: {self._model_path}")
        try:
            checkpoint = torch.load(self._model_path, map_location="cpu", weights_only=False)
            self._config = checkpoint["config"]
            self._class_names = self._config.get("class_names", []) or []

            model = BBoxPolicy(
                max_objects=self._config.get("max_objects", 5),
                num_classes=self._config.get("num_classes", 10),
                chunk_size=self._config.get("chunk_size", 10),
                hidden_dim=self._config.get("hidden_dim", 128),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._model = model

            self._encoder = BBoxObservation(
                max_objects=self._config.get("max_objects", 5),
                num_classes=self._config.get("num_classes", 10),
            )

            num_params = sum(p.numel() for p in model.parameters())
            print(f"    ✓ {num_params:,} Parameter")
            print(f"    ✓ Chunk Size: {self._config.get('chunk_size', 10)}")
            print(f"    ✓ Klassen: {self._class_names or '(alle)'}")
        except Exception as e:
            print(f"    ✗ Fehler: {e}")
            return False

        # 2. Hardware
        print(f"\n  [2/3] Hardware verbinden...")
        try:
            self._hw = RoArmHardware(port=self._port)
            print(f"    ✓ Arm verbunden")
        except Exception as e:
            print(f"    ✗ {e}")
            return False

        # 3. Vision (YOLO)
        print(f"\n  [3/3] YOLO laden: {self._yolo_model}")
        try:
            self._vision = VisionSystem(
                camera_index=self._camera_index,
                model_path=self._yolo_model,
                confidence=self._confidence,
            )
            if not self._vision.has_model:
                print(f"    ✗ YOLO-Modell konnte nicht geladen werden!")
                return False
            print(f"    ✓ YOLO bereit ({len(self._vision.class_names)} Klassen)")
        except Exception as e:
            print(f"    ✗ {e}")
            return False

        # Startposition
        self._hw.move_joints(self._state, spd=20, acc=10)
        time.sleep(2.0)

        print(f"\n  ✓ Bereit!")
        print(f"    SPACE = Start/Stop")
        print(f"    R     = Reset")
        print(f"    Q     = Beenden")
        return True

    def run(self):
        """Hauptschleife."""
        if not self.setup():
            return

        self._running = True
        window_name = "BBox Policy Runner"
        control_interval = 1.0 / self.CONTROL_HZ

        try:
            while self._running:
                loop_start = time.time()

                # Frame + Detection
                frame = self._vision.get_frame()
                detections = self._vision.detect(frame) if frame is not None else []

                # Key handling
                if HAS_CV2:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        self._executing = not self._executing
                        if self._executing:
                            self._action_queue = []
                            print("  ▶ Policy gestartet")
                        else:
                            print("  ⏸ Policy pausiert")
                    elif key == ord('r'):
                        self._executing = False
                        self._state = ArmState()
                        self._hw.move_joints(self._state, spd=20, acc=10)
                        self._action_queue = []
                        print("  ↺ Reset")

                # Policy ausführen
                if self._executing:
                    self._step_policy(detections)

                # Display
                if frame is not None and HAS_CV2:
                    self._vision.draw_detections(frame, detections)
                    self._annotate_display(frame, detections)
                    cv2.imshow(window_name, frame)

                # Rate limit
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, control_interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _step_policy(self, detections: list):
        """Einen Policy-Schritt ausführen."""
        if not self._action_queue:
            # Neuen Action-Chunk vorhersagen
            arm_vec = self._state.to_list()
            obs = self._encoder.encode(detections, arm_vec, self._class_names)
            actions = self._model.predict(obs)  # [chunk_size, action_dim]
            self._action_queue = list(actions)

        if self._action_queue:
            action = self._action_queue.pop(0)
            self._apply_action(action)

    def _apply_action(self, action: np.ndarray):
        """Wendet eine Aktion auf den Arm an."""
        # Denormalisieren (action ist normalisiert: base/90, shoulder/60, etc.)
        self._state.base_deg = float(np.clip(action[0] * 90.0, -90, 90))
        self._state.shoulder_deg = float(np.clip(action[1] * 60.0, -30, 60))
        self._state.elbow_deg = float(np.clip(action[2] * 180.0, 0, 180))
        self._state.hand_deg = float(np.clip(action[3] * 270.0, 0, 270))

        if len(action) > 4:
            gripper_val = float(action[4])
            if gripper_val > 0.5 and self._state.gripper_open:
                self._hw.gripper_close()
                self._state.gripper_open = False
            elif gripper_val <= 0.5 and not self._state.gripper_open:
                self._hw.gripper_open()
                self._state.gripper_open = True

        if len(action) > 5:
            led = int(np.clip(action[5] * 255, 0, 255))
            self._hw.set_led(led)
            self._state.led_brightness = led

        self._hw.move_joints(self._state, spd=50, acc=100)

    def _annotate_display(self, frame, detections: list):
        """Status-Overlay."""
        h_f, w_f = frame.shape[:2]

        # Status
        status = "▶ RUNNING" if self._executing else "⏸ PAUSED"
        color = (0, 255, 0) if self._executing else (0, 200, 255)
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Detections count
        det_text = f"Objects: {len(detections)}"
        cv2.putText(frame, det_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Arm state
        s = self._state
        state_text = f"B:{s.base_deg:.0f} S:{s.shoulder_deg:.0f} E:{s.elbow_deg:.0f} H:{s.hand_deg:.0f}"
        cv2.putText(frame, state_text, (10, h_f - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Queue
        queue_text = f"Queue: {len(self._action_queue)}"
        cv2.putText(frame, queue_text, (w_f - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _shutdown(self):
        """Aufräumen."""
        import cv2

        print("\n[PolicyRunner] Shutdown...")
        self._executing = False

        if self._hw:
            self._hw.gripper_open()
            time.sleep(0.3)
            self._hw.park()
            time.sleep(1.5)
            self._hw.set_led(0)
            self._hw.disconnect()

        if self._vision:
            self._vision.release()

        if HAS_CV2:
            cv2.destroyAllWindows()

        print("[PolicyRunner] ✓ Fertig")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 BBox-Policy Runner — Roboter sieht nur Bounding Boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Standard-Ausführung
  python3 run_policy.py trained_bbox_policy/bbox_policy_best.pt --yolo my_model.pt

  # Mit spezifischem Port und Kamera
  python3 run_policy.py model.pt --yolo yolo.pt --port /dev/ttyUSB0 --camera 0

  # Langsamer ausführen
  python3 run_policy.py model.pt --yolo yolo.pt --speed 0.5
        """
    )

    parser.add_argument("model_path", type=str,
                        help="Pfad zum trainierten BBox-Policy Modell (.pt)")
    parser.add_argument("--yolo", type=str, required=True,
                        help="Pfad zum YOLO-Modell (für Objekt-Detection)")
    parser.add_argument("--port", type=str, default=None,
                        help="Serieller Port (auto-detect wenn nicht angegeben)")
    parser.add_argument("--camera", type=int, default=2,
                        help="Kamera-Index (default: 2)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="YOLO Confidence Threshold (default: 0.5)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Geschwindigkeits-Multiplikator (default: 1.0)")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"✗ Modell nicht gefunden: {args.model_path}")
        sys.exit(1)

    if not Path(args.yolo).exists():
        print(f"✗ YOLO-Modell nicht gefunden: {args.yolo}")
        sys.exit(1)

    runner = BBoxPolicyRunner(
        model_path=args.model_path,
        yolo_model=args.yolo,
        port=args.port,
        camera_index=args.camera,
        confidence=args.confidence,
        speed_scale=args.speed,
    )
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen (CTRL-C)")
        sys.exit(0)
    except OSError:
        print("\nOSError — Kabel getrennt?")
        sys.exit(1)

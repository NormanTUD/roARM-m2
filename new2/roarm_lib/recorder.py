"""
Session Recorder — Verbindet Hardware, Vision, DSL-Recorder.
Einziger Entry-Point für das Aufzeichnen.
"""

import time
import threading
from pathlib import Path
from typing import Optional, List, Dict

from .hardware import RoArmHardware, ArmState, SPEED_PRESETS, LIMITS
from .vision import VisionSystem, Detection
from .dsl import DSLRecorder


class SessionRecorder:
    """
    Interaktiver Recorder mit:
    - Pfeiltasten-Steuerung
    - Funktions-Aufzeichnung per Knopfdruck
    - Automatischer JPG-Export
    - DSL-Output
    
    Designed für Flow-Zustand: minimale Ablenkung,
    nur Bewegung + ein Knopf für Funktionen.
    """

    SPEED_LEVELS = [
        {"label": "VERY SLOW", "step": 0.15, "spd": 15, "acc": 30},
        {"label": "SLOW", "step": 0.3, "spd": 30, "acc": 50},
        {"label": "MEDIUM", "step": 0.6, "spd": 50, "acc": 100},
        {"label": "FAST", "step": 1.0, "spd": 80, "acc": 150},
        {"label": "VERY FAST", "step": 1.5, "spd": 100, "acc": 200},
    ]

    def __init__(self, output_dir: str = "recordings",
                 camera_index: int = 2,
                 model_path: Optional[str] = None,
                 confidence: float = 0.5,
                 port: Optional[str] = None,
                 save_images_every_n: int = 10):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Hardware
        self._hw = RoArmHardware(port=port)
        self._vision = VisionSystem(
            camera_index=camera_index,
            model_path=model_path,
            confidence=confidence,
        )
        self._dsl_recorder = DSLRecorder(
            output_dir=self._output_dir / "dsl",
            vision=self._vision,
        )

        # State
        self._state = ArmState()
        self._speed_level = 2  # MEDIUM
        self._running = False
        self._frame_count = 0
        self._save_images_every_n = save_images_every_n

        # Keys
        self._keys_down: set = set()
        self._key_last_seen: Dict[str, float] = {}
        self._key_timeout = 0.35

    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._dsl_recorder.is_recording

    @property
    def is_recording_function(self) -> bool:
        return self._dsl_recorder.is_recording_function

    def setup(self) -> bool:
        """Initialisiert alles."""
        print("=" * 50)
        print("  RoArm-M2-S DSL Recorder")
        print("=" * 50)

        # Arm in Startposition
        self._hw.move_joints(self._state, spd=20, acc=10)
        time.sleep(2.0)
        self._hw.set_led(255)

        print(f"  ✓ Hardware bereit")
        if self._vision.has_camera:
            print(f"  ✓ Kamera bereit ({self._vision.resolution})")
        if self._vision.has_model:
            print(f"  ✓ YOLO bereit ({len(self._vision.class_names)} Klassen)")
        else:
            print(f"  ℹ Kein YOLO-Modell (--model zum Laden)")

        print(f"\n  Steuerung:")
        print(f"    Pfeiltasten  → Base/Shoulder")
        print(f"    W/S          → Elbow")
        print(f"    A/D          → Hand")
        print(f"    O/C          → Gripper")
        print(f"    R            → Recording Start/Stop")
        print(f"    F            → Funktion Start/Stop (fragt nach Name)")
        print(f"    I            → Bild speichern (für YOLO)")
        print(f"    +/-          → Speed")
        print(f"    Q            → Beenden")
        return True

    def run(self):
        """Hauptschleife."""
        import cv2

        if not self.setup():
            return

        self._running = True
        window_name = "RoArm DSL Recorder"

        try:
            while self._running:
                loop_start = time.time()

                # Frame holen
                frame = self._vision.get_frame()

                # Detection (wenn Modell da)
                detections = []
                if self._vision.has_model and frame is not None:
                    detections = self._vision.detect(frame)

                # Keys verarbeiten
                key = cv2.waitKey(1) & 0xFFFF
                if key != 0xFFFF and key != -1:
                    self._process_key(key)

                # Bewegung anwenden
                self._apply_movement()

                # Recording
                if self._dsl_recorder.is_recording or self._dsl_recorder.is_recording_function:
                    action = self._get_current_action()
                    self._dsl_recorder.record_frame(self._state, action, detections)

                    # Bilder speichern
                    self._frame_count += 1
                    if frame is not None and self._frame_count % self._save_images_every_n == 0:
                        self._dsl_recorder.save_image(frame)

                # Display
                if frame is not None:
                    self._annotate_frame(frame, detections)
                    cv2.imshow(window_name, frame)

                # Rate limit
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, (1.0 / 40.0) - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _process_key(self, key: int):
        """Verarbeitet Tastendruck."""
        import cv2

        now = time.time()
        key_low = key & 0xFF

        # Pfeiltasten
        if key == 65361 or key_low == 81:  # Left
            self._keys_down.add("base_left")
            self._key_last_seen["base_left"] = now
        elif key == 65363 or key_low == 83:  # Right
            self._keys_down.add("base_right")
            self._key_last_seen["base_right"] = now
        elif key == 65362 or key_low == 82:  # Up
            self._keys_down.add("shoulder_up")
            self._key_last_seen["shoulder_up"] = now
        elif key == 65364 or key_low == 84:  # Down
            self._keys_down.add("shoulder_down")
            self._key_last_seen["shoulder_down"] = now

        # WASD
        elif key_low == ord('w'):
            self._keys_down.add("elbow_up")
            self._key_last_seen["elbow_up"] = now
        elif key_low == ord('s'):
            self._keys_down.add("elbow_down")
            self._key_last_seen["elbow_down"] = now
        elif key_low == ord('a'):
            self._keys_down.add("hand_left")
            self._key_last_seen["hand_left"] = now
        elif key_low == ord('d'):
            self._keys_down.add("hand_right")
            self._key_last_seen["hand_right"] = now

        # Gripper
        elif key_low == ord('o'):
            self._hw.gripper_open()
            self._state.gripper_open = True
        elif key_low == ord('c'):
            self._hw.gripper_close()
            self._state.gripper_open = False

        # Recording
        elif key_low == ord('r'):
            if self._dsl_recorder.is_recording:
                self._dsl_recorder.stop_recording()
            else:
                self._dsl_recorder.start_recording()

        # Funktion aufzeichnen
        elif key_low == ord('f'):
            if self._dsl_recorder.is_recording_function:
                self._dsl_recorder.stop_function()
            else:
                # Name abfragen (im Terminal)
                name = input("\n  Funktionsname: ").strip()
                if name:
                    self._dsl_recorder.start_function(name)

        # Bild speichern
        elif key_low == ord('i'):
            frame = self._vision.get_frame()
            if frame is not None:
                path = self._dsl_recorder.save_image(frame)
                if path:
                    print(f"  📷 Bild: {path}")

        # Speed
        elif key_low == ord('+') or key_low == ord('='):
            self._speed_level = min(4, self._speed_level + 1)
            print(f"  ⚡ Speed: {self.SPEED_LEVELS[self._speed_level]['label']}")
        elif key_low == ord('-'):
            self._speed_level = max(0, self._speed_level - 1)
            print(f"  ⚡ Speed: {self.SPEED_LEVELS[self._speed_level]['label']}")

        # Quit
        elif key_low == ord('q'):
            self._running = False

    def _apply_movement(self):
        """Wendet gehaltene Tasten auf den Arm an."""
        now = time.time()

        # Expired keys entfernen
        expired = [k for k, t in self._key_last_seen.items() if now - t > self._key_timeout]
        for k in expired:
            self._keys_down.discard(k)
            del self._key_last_seen[k]

        if not self._keys_down:
            return

        spd = self.SPEED_LEVELS[self._speed_level]
        step = spd["step"]

        if "base_left" in self._keys_down:
            self._state.base_deg = min(90, self._state.base_deg + step)
        if "base_right" in self._keys_down:
            self._state.base_deg = max(-90, self._state.base_deg - step)
        if "shoulder_up" in self._keys_down:
            self._state.shoulder_deg = min(60, self._state.shoulder_deg + step)
        if "shoulder_down" in self._keys_down:
            self._state.shoulder_deg = max(-30, self._state.shoulder_deg - step)
        if "elbow_up" in self._keys_down:
            self._state.elbow_deg = max(0, self._state.elbow_deg - step)
        if "elbow_down" in self._keys_down:
            self._state.elbow_deg = min(180, self._state.elbow_deg + step)
        if "hand_left" in self._keys_down:
            self._state.hand_deg = max(0, self._state.hand_deg - step)
        if "hand_right" in self._keys_down:
            self._state.hand_deg = min(270, self._state.hand_deg + step)

        self._hw.move_joints(self._state, spd=spd["spd"], acc=spd["acc"])

    def _get_current_action(self) -> str:
        """Gibt die aktuelle Aktion als String zurück (für DSL-Recording)."""
        active = self._keys_down.copy()
        if not active:
            return ""
        # Priorität: Gripper > Bewegung
        if "gripper_open" in active:
            return "gripper_open"
        if "gripper_close" in active:
            return "gripper_close"
        return next(iter(active))

    def _annotate_frame(self, frame, detections: list):
        """Zeichnet Status-Overlays auf den Frame."""
        if frame is None:
            return

        cv2 = self._vision._cv2
        h_f, w_f = frame.shape[:2]

        # Detections zeichnen
        if detections:
            self._vision.draw_detections(frame, detections)

        # Status-Leiste oben
        state = self._state
        spd_label = self.SPEED_LEVELS[self._speed_level]["label"]
        status = (f"B:{state.base_deg:.0f} S:{state.shoulder_deg:.0f} "
                  f"E:{state.elbow_deg:.0f} H:{state.hand_deg:.0f} "
                  f"G:{'O' if state.gripper_open else 'C'} | {spd_label}")

        # Semi-transparenter Hintergrund
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Recording-Indikator
        if self._dsl_recorder.is_recording:
            blink = int(time.time() * 3) % 2 == 0
            color = (0, 0, 255) if blink else (0, 0, 180)
            cv2.circle(frame, (w_f - 20, 15), 8, color, -1)
            cv2.putText(frame, "REC", (w_f - 55, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Funktions-Recording
        if self._dsl_recorder.is_recording_function:
            name = self._dsl_recorder.current_function_name
            cv2.putText(frame, f"FN: {name}", (w_f - 150, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # Hilfe unten
        help_text = "R=Rec F=Func I=Img +/-=Spd Q=Quit"
        cv2.putText(frame, help_text, (10, h_f - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def _shutdown(self):
        """Aufräumen."""
        import cv2

        if self._dsl_recorder.is_recording:
            self._dsl_recorder.stop_recording()
        if self._dsl_recorder.is_recording_function:
            self._dsl_recorder.stop_function()

        print("\n[Recorder] Shutdown...")
        self._hw.park()
        time.sleep(1.0)
        self._hw.set_led(0)
        self._hw.disconnect()
        self._vision.release()
        cv2.destroyAllWindows()
        print("[Recorder] ✓ Fertig")


#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "pyserial",
#     "torch",
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
Session Recorder — Verbindet Hardware, Vision, DSL-Recorder.
Einziger Entry-Point für das Aufzeichnen.

Architecture matches record_policy.py:
  - Dedicated 50Hz command thread for smooth movement
  - cv2.waitKey drain loop (multiple keys per frame)
  - Persistent key-hold state with timeout
  - Keyboard overlay + joystick widget
  - Non-blocking gripper/LED
"""

import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import sys
from pathlib import Path

# Add the roarm_lib directory to path
_script_dir = Path(__file__).resolve().parent
_lib_dir = _script_dir.parent / "roarm_lib"
if _lib_dir.exists():
    sys.path.insert(0, str(_lib_dir.parent))
else:
    # Try sibling directory structure
    for candidate in [_script_dir.parent, _script_dir]:
        if (candidate / "roarm_lib").exists():
            sys.path.insert(0, str(candidate))
            break

from roarm_lib.hardware import RoArmHardware, ArmState, SPEED_PRESETS, LIMITS
from roarm_lib.vision import VisionSystem, Detection
from roarm_lib.dsl import DSLRecorder


class SessionRecorder:
    """
    Interaktiver Recorder mit:
    - Pfeiltasten-Steuerung (smooth, 50Hz command thread)
    - Funktions-Aufzeichnung per Knopfdruck
    - Automatischer JPG-Export
    - DSL-Output
    - Keyboard overlay + joystick widget
    """

    COMMAND_HZ = 50
    COMMAND_INTERVAL = 1.0 / 50

    SPEED_LEVELS = [
        {"label": "VERY SLOW", "base": 0.15, "shoulder": 0.12, "elbow": 0.12, "hand": 0.2, "spd": 15, "acc": 30},
        {"label": "SLOW", "base": 0.3, "shoulder": 0.25, "elbow": 0.25, "hand": 0.4, "spd": 30, "acc": 50},
        {"label": "MEDIUM", "base": 0.6, "shoulder": 0.5, "elbow": 0.5, "hand": 0.7, "spd": 50, "acc": 100},
        {"label": "FAST", "base": 1.0, "shoulder": 0.8, "elbow": 0.8, "hand": 1.2, "spd": 80, "acc": 150},
        {"label": "VERY FAST", "base": 1.5, "shoulder": 1.2, "elbow": 1.2, "hand": 1.8, "spd": 100, "acc": 200},
    ]

    KEY_HOLD_TIMEOUT = 0.35

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

        # Persistent key state (like record_policy.py)
        self._keys_down: set = set()
        self._key_last_seen: Dict[str, float] = {}

        # Command thread timing
        self._last_cmd_time = 0.0
        self._cmd_thread: Optional[threading.Thread] = None

    @property
    def state(self) -> ArmState:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._dsl_recorder.is_recording

    @property
    def is_recording_function(self) -> bool:
        return self._dsl_recorder.is_recording_function

    @property
    def _current_speed(self) -> dict:
        return self.SPEED_LEVELS[self._speed_level]

    def _get_active_keys(self) -> set:
        """Returns currently held keys, expiring stale ones."""
        now = time.time()
        expired = [k for k, t in self._key_last_seen.items()
                   if now - t >= self.KEY_HOLD_TIMEOUT]
        for k in expired:
            self._keys_down.discard(k)
            del self._key_last_seen[k]
        return set(self._keys_down)

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
        print(f"    F            → Funktion Start/Stop")
        print(f"    I            → Bild speichern (für YOLO)")
        print(f"    +/-          → Speed")
        print(f"    Q            → Beenden")
        return True

    def _command_loop(self):
        """Dedicated thread: sends arm commands at fixed 50Hz rate."""
        while self._running:
            self._apply_movement()
            time.sleep(self.COMMAND_INTERVAL)

    def run(self):
        """Hauptschleife."""
        import cv2

        if not self.setup():
            return

        self._running = True
        self._last_cmd_time = time.time()
        window_name = "RoArm DSL Recorder"

        # Start dedicated command thread (50Hz)
        self._cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._cmd_thread.start()

        try:
            while self._running:
                loop_start = time.time()

                # Frame holen
                frame = self._vision.get_frame()

                # Detection
                detections = []
                if self._vision.has_model and frame is not None:
                    detections = self._vision.detect(frame)

                # Drain ALL pending keys (not just one!)
                for _ in range(50):
                    key = cv2.waitKey(1) & 0xFFFF
                    if key == 0xFFFF or key == -1:
                        break
                    self._process_key(key)

                # Recording
                if self._dsl_recorder.is_recording or self._dsl_recorder.is_recording_function:
                    action = self._get_current_action()
                    self._dsl_recorder.record_frame(self._state, action, detections)

                    self._frame_count += 1
                    if frame is not None and self._frame_count % self._save_images_every_n == 0:
                        self._dsl_recorder.save_image(frame)

                # Display
                if frame is not None:
                    self._annotate_frame(frame, detections)
                    cv2.imshow(window_name, frame)

                # Rate limit ~40 FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, (1.0 / 40.0) - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _process_key(self, key: int):
        """Verarbeitet Tastendruck — updates persistent key state."""
        if key == -1 or key == 0xFFFF:
            return

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

        # Gripper (non-blocking now)
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
            print(f"  ⚡ Speed: {self._current_speed['label']}")
        elif key_low == ord('-'):
            self._speed_level = max(0, self._speed_level - 1)
            print(f"  ⚡ Speed: {self._current_speed['label']}")

        # Quit
        elif key_low == ord('q'):
            self._running = False

    def _apply_movement(self):
        """Velocity-based movement at fixed rate from command thread."""
        now = time.time()
        dt = now - self._last_cmd_time

        if dt < (1.0 / 50):
            return

        # Expire stale keys
        expired = [k for k, t in self._key_last_seen.items() if now - t > self.KEY_HOLD_TIMEOUT]
        for k in expired:
            self._keys_down.discard(k)
            del self._key_last_seen[k]

        if not self._keys_down:
            self._last_cmd_time = now
            return

        # Scale step by elapsed time
        time_factor = dt / (1.0 / 50)
        time_factor = min(time_factor, 3.0)

        spd = self.SPEED_LEVELS[self._speed_level]
        moved = False

        if "base_left" in self._keys_down:
            self._state.base_deg = min(90, self._state.base_deg + spd["base"] * time_factor)
            moved = True
        if "base_right" in self._keys_down:
            self._state.base_deg = max(-90, self._state.base_deg - spd["base"] * time_factor)
            moved = True
        if "shoulder_up" in self._keys_down:
            self._state.shoulder_deg = min(60, self._state.shoulder_deg + spd["shoulder"] * time_factor)
            moved = True
        if "shoulder_down" in self._keys_down:
            self._state.shoulder_deg = max(-30, self._state.shoulder_deg - spd["shoulder"] * time_factor)
            moved = True
        if "elbow_up" in self._keys_down:
            self._state.elbow_deg = max(0, self._state.elbow_deg - spd["elbow"] * time_factor)
            moved = True
        if "elbow_down" in self._keys_down:
            self._state.elbow_deg = min(180, self._state.elbow_deg + spd["elbow"] * time_factor)
            moved = True
        if "hand_left" in self._keys_down:
            self._state.hand_deg = max(0, self._state.hand_deg - spd["hand"] * time_factor)
            moved = True
        if "hand_right" in self._keys_down:
            self._state.hand_deg = min(270, self._state.hand_deg + spd["hand"] * time_factor)
            moved = True

        if moved:
            self._hw.move_joints(self._state, spd=spd["spd"], acc=spd["acc"])

        self._last_cmd_time = now

    def _get_current_action(self) -> str:
        """Gibt die aktuelle Aktion als String zurück."""
        active = self._get_active_keys()
        if not active:
            return ""
        if "gripper_open" in active:
            return "gripper_open"
        if "gripper_close" in active:
            return "gripper_close"
        return next(iter(active))

    def _annotate_frame(self, frame, detections: list):
        """Zeichnet Status-Overlays auf den Frame (with keyboard + joystick)."""
        if frame is None:
            return

        cv2 = self._vision._cv2
        h_f, w_f = frame.shape[:2]

        # Detections zeichnen
        if detections:
            self._vision.draw_detections(frame, detections)

        # Status-Leiste oben (semi-transparent)
        state = self._state
        spd_label = self._current_speed["label"]
        status = (f"B:{state.base_deg:.0f} S:{state.shoulder_deg:.0f} "
                  f"E:{state.elbow_deg:.0f} H:{state.hand_deg:.0f} "
                  f"G:{'O' if state.gripper_open else 'C'} | {spd_label}")

        # Semi-transparenter Hintergrund
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Active keys display
        active = self._get_active_keys()
        if active:
            active_str = " + ".join(sorted(active))
            cv2.putText(frame, f"Keys: {active_str}", (10, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

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

        # Keyboard overlay (bottom left)
        self._draw_keyboard_overlay(frame, active, 10, h_f - 100)

        # Joystick widget (bottom right)
        self._draw_joystick_widget(frame, w_f - 110, h_f - 110, 100)

        # Hilfe unten
        help_text = "R=Rec F=Func I=Img +/-=Spd Q=Quit"
        cv2.putText(frame, help_text, (10, h_f - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def _draw_keyboard_overlay(self, frame, active_keys: set, x: int, y: int):
        """Mini keyboard showing active keys."""
        cv2 = self._vision._cv2
        key_size = 20
        gap = 2

        # Background panel
        panel_w = 200
        panel_h = 65
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (100, 150, 200), 1)

        def draw_key(kx, ky, label, is_active):
            color_bg = (0, 120, 255) if is_active else (40, 40, 40)
            color_text = (255, 255, 255) if is_active else (150, 150, 150)
            cv2.rectangle(frame, (kx, ky), (kx + key_size, ky + key_size), color_bg, -1)
            cv2.rectangle(frame, (kx, ky), (kx + key_size, ky + key_size), (80, 80, 80), 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            tx = kx + (key_size - tw) // 2
            ty = ky + (key_size + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_text, 1)

        # Arrow keys
        bx = x + 5
        by = y + 5
        draw_key(bx + key_size + gap, by, "^", "shoulder_up" in active_keys)
        draw_key(bx, by + key_size + gap, "<", "base_left" in active_keys)
        draw_key(bx + key_size + gap, by + key_size + gap, "v", "shoulder_down" in active_keys)
        draw_key(bx + 2 * (key_size + gap), by + key_size + gap, ">", "base_right" in active_keys)

        # WASD
        wx = x + 75
        wy = y + 5
        draw_key(wx + key_size + gap, wy, "W", "elbow_up" in active_keys)
        draw_key(wx, wy + key_size + gap, "A", "hand_left" in active_keys)
        draw_key(wx + key_size + gap, wy + key_size + gap, "S", "elbow_down" in active_keys)
        draw_key(wx + 2 * (key_size + gap), wy + key_size + gap, "D", "hand_right" in active_keys)

        # O/C
        gx = x + 155
        gy = y + 5 + key_size + gap
        draw_key(gx, gy, "O", False)
        draw_key(gx + key_size + gap, gy, "C", False)

    def _draw_joystick_widget(self, frame, x: int, y: int, size: int):
        """Visual joystick showing arm position."""
        cv2 = self._vision._cv2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + size, y + size), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x, y), (x + size, y + size), (80, 200, 80), 1)

        cx = x + size // 2
        cy = y + size // 2
        radius = size // 2 - 10

        # Axes
        cv2.line(frame, (x + 10, cy), (x + size - 10, cy), (60, 60, 60), 1)
        cv2.line(frame, (cx, y + 10), (cx, y + size - 10), (60, 60, 60), 1)

        # Base → X, Shoulder → Y
        base_norm = max(-1.0, min(1.0, self._state.base_deg / 90.0))
        shoulder_norm = max(0.0, min(1.0, (self._state.shoulder_deg + 30) / 90.0))

        # Punkt-Position
        px = int(cx + base_norm * radius)
        py = int(cy - (shoulder_norm - 0.5) * 2 * radius)

        # Elbow als Farbe (0°=blau, 90°=grün, 180°=rot)
        elbow_norm = self._state.elbow_deg / 180.0
        r = int(elbow_norm * 255)
        g = int((1.0 - abs(elbow_norm - 0.5) * 2) * 255)
        b = int((1.0 - elbow_norm) * 255)
        dot_color = (b, g, r)

        # Punkt zeichnen
        cv2.circle(frame, (px, py), 6, dot_color, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

        # Gripper-Status
        grip_symbol = "O" if self._state.gripper_open else "X"
        grip_color = (0, 255, 0) if self._state.gripper_open else (0, 0, 255)
        cv2.putText(frame, grip_symbol, (x + size - 18, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 2)

        # Label
        cv2.putText(frame, "JOY", (x + 3, y + size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    def _shutdown(self):
        """Aufräumen."""
        import cv2

        self._running = False

        # Wait for command thread to finish
        if self._cmd_thread and self._cmd_thread.is_alive():
            self._cmd_thread.join(timeout=1.0)

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


sr = SessionRecorder()
sr.run()

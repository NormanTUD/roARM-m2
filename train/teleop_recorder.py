#!/usr/bin/env python3
"""
Teleop Recorder – Fernsteuerung des RoArm-M2-S mit Tastatur + Aufzeichnung.

Steuerung:
  Pfeiltasten Links/Rechts  → Base-Rotation (Links = Base dreht links, Rechts = rechts)
  Pfeiltasten Oben/Unten    → Shoulder (Oben = Arm hebt sich / nach oben, Unten = senkt sich)
  W/S                        → Elbow (W = hoch, S = runter)
  A/D                        → Hand/Wrist Rotation (A = links, D = rechts)
  O                          → Gripper öffnen
  C                          → Gripper schließen
  R                          → Aufnahme starten (neue Episode)
  S (nur bei Aufnahme)       → Episode speichern & beenden
  F                          → Episode als fehlgeschlagen markieren & speichern
  Q                          → Beenden

Aufzeichnung:
  - Bounding Boxes (Klasse, Koordinaten, Confidence)
  - Arm-Gelenkwinkel + kartesische Position
  - Aktionen (Tasteneingaben) mit Timestamps
  - Alles relativ zu den BBoxes
"""

import cv2
import json
import time
import os
import argparse
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict

from roarm_m2s import RoArmM2S

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# ─── Datenstrukturen ─────────────────────────────────────────────────────────

@dataclass
class ArmState:
    """Aktueller Zustand des Arms."""
    base_deg: float = 0.0
    shoulder_deg: float = 0.0
    elbow_deg: float = 90.0
    hand_deg: float = 180.0
    gripper_open: bool = True
    # Kartesisch (wenn verfügbar)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Episode:
    """Eine komplette Aufnahme-Episode."""
    episode_id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    frames: List[Dict] = field(default_factory=list)
    target_class: str = ""
    success: bool = False


# ─── Teleop Controller ───────────────────────────────────────────────────────

class TeleopRecorder:
    """
    Fernsteuerung + Aufzeichnung für Behaviour Cloning.
    
    Smooth movement via:
    - Persistent key state tracking (not relying on OS key-repeat)
    - High-frequency command streaming in a separate thread
    - Small position increments at consistent rate
    """

    # ─── Steuerungs-Parameter (FIXED) ────────────────────────────────────
    
    # Command streaming rate (separate from display loop)
    COMMAND_HZ = 50            # 50 commands/sec to servo
    COMMAND_INTERVAL = 1.0 / 50  # 20ms between commands
    
    # Step sizes PER COMMAND (at 50Hz, 0.6°/cmd = 30°/sec movement)
    BASE_STEP = 0.6
    SHOULDER_STEP = 0.5
    ELBOW_STEP = 0.5
    
    # Servo parameters
    # spd for T:122 in degrees mode: higher = faster. 0 might not mean max!
    # Try 50-100 for fast continuous motion
    MOVE_SPEED = 50
    MOVE_ACC = 100
    
    # Limits
    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    # Timing für flüssige Steuerung
    KEY_HOLD_TIMEOUT = 0.35  # Sekunden: Taste gilt als "gehalten" für diese Dauer nach letztem Druck

    def __init__(self, port: str = None, camera_index: int = 2,
                 model_path: str = "yolo11n.pt", confidence: float = 0.5,
                 output_dir: str = "recordings", target_class: str = None):
        self._port = port
        self._camera_index = camera_index
        self._model_path = model_path
        self._confidence = confidence
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._target_class = target_class

        # Hardware
        self._arm: Optional[RoArmM2S] = None
        self._camera: Optional[cv2.VideoCapture] = None
        self._model = None

        # State
        self._arm_state = ArmState()
        self._recording = False
        self._current_episode: Optional[Episode] = None
        self._episode_count = self._count_existing_episodes()
        self._running = False

        # ─── NEW: Persistent key state (press/release tracking) ───
        self._keys_down: set = set()           # Currently held keys
        self._key_last_seen: Dict[str, float] = {}  # Timestamp of last keypress event

        # ─── NEW: Command thread ───
        self._cmd_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0
        self._last_action = ""

        # Window
        self._window_name = "RoArm Teleop"

    def _count_existing_episodes(self) -> int:
        """Zählt bereits vorhandene Episoden im Output-Verzeichnis."""
        count = 0
        for f in self._output_dir.glob("episode_*.json"):
            count += 1
        return count

    def _get_active_keys(self) -> set:
        """
        Returns all keys currently considered 'held'.
        A key is held if it was seen within KEY_HOLD_TIMEOUT.
        This handles OS key-repeat gaps gracefully.
        """
        now = time.time()
        expired = []
        for key, last_time in self._key_last_seen.items():
            if now - last_time >= self.KEY_HOLD_TIMEOUT:
                expired.append(key)
        for key in expired:
            del self._key_last_seen[key]
            self._keys_down.discard(key)
        return set(self._keys_down)  # Return a copy

    # ─── Setup ────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialisiert Hardware."""
        print("=" * 60)
        print("  RoArm-M2-S Teleop Recorder")
        print("=" * 60)

        # Arm verbinden
        print("\n[1] Arm verbinden...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            print("  ✓ Arm verbunden")
        except Exception as e:
            print(f"  ✗ Arm-Fehler: {e}")
            return False

        # Kamera
        print(f"\n[2] Kamera {self._camera_index} öffnen...")
        self._camera = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            for idx in [0, 2, 1, 4]:
                self._camera = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if self._camera.isOpened():
                    print(f"  → Fallback auf Kamera {idx}")
                    self._camera_index = idx
                    break
            else:
                print("  ✗ Keine Kamera gefunden!")
                return False

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  ✓ Kamera {self._camera_index} ({w}x{h})")

        # YOLO
        if HAS_YOLO:
            print(f"\n[3] YOLO '{self._model_path}' laden...")
            try:
                self._model = YOLO(self._model_path)
                self._model.verbose = False
                ret, frame = self._camera.read()
                if ret:
                    self._model(frame, conf=self._confidence, verbose=False)
                print(f"  ✓ YOLO bereit")
            except Exception as e:
                print(f"  ✗ YOLO-Fehler: {e}")
                self._model = None
        else:
            print("\n[3] YOLO nicht verfügbar (pip install ultralytics)")

        # Arm in Startposition
        print("\n[4] Arm → Startposition...")
        self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
        self._arm_state = ArmState(
            base_deg=0, shoulder_deg=0, elbow_deg=90, hand_deg=180, gripper_open=True
        )

        self._arm.set_led(255)

        print("  ✓ Bereit")

        print("\n" + "=" * 60)
        print("  STEUERUNG:")
        print("    ←         Base nach links drehen")
        print("    →         Base nach rechts drehen")
        print("    ↑         Shoulder hoch (Arm hebt sich)")
        print("    ↓         Shoulder runter (Arm senkt sich)")
        print("    W         Elbow hoch")
        print("    S         Elbow runter (ohne Aufnahme) / Save (bei Aufnahme)")
        print("    A         Hand/Wrist nach links")
        print("    D         Hand/Wrist nach rechts")
        print("    O         Gripper öffnen")
        print("    C         Gripper schließen")
        print("    R         Aufnahme starten")
        print("    S         Aufnahme speichern (nur während Aufnahme)")
        print("    F         Aufnahme als fehlgeschlagen speichern")
        print("    Q         Beenden")
        print("=" * 60)
        print(f"\n  Episoden bisher: {self._episode_count}")
        if self._target_class:
            print(f"  Ziel-Objekt: '{self._target_class}'")
        print()

        return True

    # ─── Main Loop ────────────────────────────────────────────────────────

    def _command_loop(self):
        """
        Dedicated thread: sends arm commands at a fixed 50Hz rate,
        completely independent of the display/detection loop.
        """
        while self._running:
            self._apply_movement()
            time.sleep(self.COMMAND_INTERVAL)  # Precise 20ms sleep

    def run(self):
        """Hauptschleife – start command thread, then run display loop."""
        if not self.setup():
            return

        self._running = True
        self._last_cmd_time = time.time()

        # START A REAL COMMAND THREAD
        self._cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._cmd_thread.start()
        fps_time = time.time()
        frame_count = 0
        current_fps = 0.0

        # Detection throttle: don't run YOLO every frame
        detect_every_n = 3  # Run YOLO every 3rd frame
        loop_counter = 0
        last_detections = []

        try:
            while self._running:
                loop_start = time.time()
                loop_counter += 1

                # 1. Frame holen (always, for display)
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                # 2. Detection (throttled to reduce blocking)
                if loop_counter % detect_every_n == 0:
                    last_detections = self._detect(frame)
                detections = last_detections

                # 3. Process ALL pending key events (drain the event queue)
                action = ""
                for _ in range(50):  # Was 10, but at 40fps we might queue more
                    key = cv2.waitKey(1) & 0xFFFF
                    if key == -1 or key == 0xFFFF:
                        break
                    self._process_key(key)


                # 4. Apply movement (at fixed rate, independent of display)
                self._apply_movement()

                # 5. Annotate
                self._annotate_frame(frame, detections, action, current_fps)

                # 6. Record
                if self._recording:
                    self._record_frame(detections, action)

                # 7. Display
                cv2.imshow(self._window_name, frame)

                # FPS
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_time = time.time()

                # Target ~40 FPS for display (commands are sent independently within _apply_movement)
                loop_elapsed = time.time() - loop_start
                sleep_time = max(0.001, (1.0 / 40.0) - loop_elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Abgebrochen]")
        finally:
            self._shutdown()


    # ─── Frame & Detection ────────────────────────────────────────────────

    def _get_frame(self):
        """Holt aktuellen Frame (Buffer-Flush)."""
        if not self._camera:
            return None
        self._camera.grab()
        ret, frame = self._camera.retrieve()
        return frame if ret else None

    def _detect(self, frame) -> List[Dict]:
        """YOLO-Detection auf Frame."""
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

            det = {
                'class': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'center_px': [cx, cy],
                'size_px': [x2 - x1, y2 - y1],
            }

            if self._target_class is None or cls_name == self._target_class:
                detections.append(det)

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    # ─── Tastatur ─────────────────────────────────────────────────────────

    def _process_key(self, key: int) -> str:
        """
        Verarbeitet Tastendruck. Updates persistent key state.
        """
        if key == -1 or key == 0xFFFF:
            return ""

        now = time.time()
        key_low = key & 0xFF
        action = ""

        # ─── Pfeiltasten ───
        if key == 65361 or key_low == 81:  # Left Arrow
            self._keys_down.add("base_left")
            self._key_last_seen["base_left"] = now
            action = "base_left"
        elif key == 65363 or key_low == 83:  # Right Arrow
            self._keys_down.add("base_right")
            self._key_last_seen["base_right"] = now
            action = "base_right"
        elif key == 65362 or key_low == 82:  # Up Arrow
            self._keys_down.add("shoulder_up")
            self._key_last_seen["shoulder_up"] = now
            action = "shoulder_up"
        elif key == 65364 or key_low == 84:  # Down Arrow
            self._keys_down.add("shoulder_down")
            self._key_last_seen["shoulder_down"] = now
            action = "shoulder_down"

        # ─── W/A/S/D ───
        elif key_low == ord('w'):
            self._keys_down.add("elbow_up")
            self._key_last_seen["elbow_up"] = now
            action = "elbow_up"
        elif key_low == ord('s'):
            if self._recording:
                self._stop_recording(success=True)
                action = ""
            else:
                self._keys_down.add("elbow_down")
                self._key_last_seen["elbow_down"] = now
                action = "elbow_down"
        elif key_low == ord('a'):
            self._keys_down.add("hand_left")
            self._key_last_seen["hand_left"] = now
            action = "hand_left"
        elif key_low == ord('d'):
            self._keys_down.add("hand_right")
            self._key_last_seen["hand_right"] = now
            action = "hand_right"

        # ─── Gripper ───
        elif key_low == ord('o'):
            self._gripper_open()
            action = "gripper_open"
        elif key_low == ord('c'):
            self._gripper_close()
            action = "gripper_close"

        # ─── Recording ───
        elif key_low == ord('r'):
            self._start_recording()
        elif key_low == ord('f'):
            self._stop_recording(success=False)

        # ─── Quit ───
        elif key_low == ord('q'):
            self._running = False

        if action:
            self._last_action = action
        return action

    def _apply_movement(self):
        """
        Apply movement at a FIXED RATE regardless of display loop timing.
        This is the key fix: decouple command rate from frame rate.
        """
        now = time.time()
        dt = now - self._last_cmd_time
        
        if dt < self.COMMAND_INTERVAL:
            return
        
        active = self._get_active_keys()
        if not active:
            return

        # Calculate steps proportional to actual elapsed time (velocity-based)
        # This ensures consistent speed regardless of timing jitter
        time_factor = dt / self.COMMAND_INTERVAL  # Normally ~1.0
        time_factor = min(time_factor, 3.0)  # Cap to prevent jumps after stalls

        moved = False
        state = self._arm_state

        if "base_left" in active:
            state.base_deg = min(self.BASE_MAX, state.base_deg + self.BASE_STEP * time_factor)
            moved = True
        if "base_right" in active:
            state.base_deg = max(self.BASE_MIN, state.base_deg - self.BASE_STEP * time_factor)
            moved = True
        if "shoulder_up" in active:
            state.shoulder_deg = min(self.SHOULDER_MAX, state.shoulder_deg + self.SHOULDER_STEP * time_factor)
            moved = True
        if "shoulder_down" in active:
            state.shoulder_deg = max(self.SHOULDER_MIN, state.shoulder_deg - self.SHOULDER_STEP * time_factor)
            moved = True
        if "elbow_up" in active:
            state.elbow_deg = max(self.ELBOW_MIN, state.elbow_deg - self.ELBOW_STEP * time_factor)
            moved = True
        if "elbow_down" in active:
            state.elbow_deg = min(self.ELBOW_MAX, state.elbow_deg + self.ELBOW_STEP * time_factor)
            moved = True

        if moved:
            self._send_arm_command()
        
        self._last_cmd_time = now

    def _send_arm_command(self):
        """
        Send current joint angles to the arm.
        Uses _send_nowait for non-blocking streaming.
        Rounds values to reduce unique command strings (helps firmware parser).
        """
        state = self._arm_state
        cmd = {
            "T": 122,
            "b": round(state.base_deg, 1),
            "s": round(state.shoulder_deg, 1),
            "e": round(state.elbow_deg, 1),
            "h": round(state.hand_deg, 1),
            "spd": self.MOVE_SPEED,
            "acc": self.MOVE_ACC
        }
        self._arm._send_nowait(cmd)


    def _gripper_open(self):
        """Öffnet den Gripper und synchronisiert hand_deg."""
        self._arm_state.gripper_open = True
        GRIPPER_OPEN_DEG = 61.88    # 1.08 rad = open
        self._arm_state.hand_deg = GRIPPER_OPEN_DEG
        self._arm.gripper_open()

    def _gripper_close(self):
        """Schließt den Gripper und synchronisiert hand_deg."""
        self._arm_state.gripper_open = False
        GRIPPER_CLOSED_DEG = 180.0  # 3.14 rad = closed
        self._arm_state.hand_deg = GRIPPER_CLOSED_DEG
        self._arm.gripper_close()

    # ─── Recording ────────────────────────────────────────────────────────

    def _start_recording(self):
        """Startet eine neue Episode."""
        if self._recording:
            print("  [!] Bereits am Aufnehmen!")
            return

        self._episode_count += 1
        self._current_episode = Episode(
            episode_id=self._episode_count,
            start_time=time.time(),
            target_class=self._target_class or "unknown",
            frames=[]
        )
        self._recording = True
        print(f"\n  ● AUFNAHME GESTARTET (Episode {self._episode_count})")

    def _stop_recording(self, success: bool = True):
        """Stoppt und speichert die aktuelle Episode."""
        if not self._recording or not self._current_episode:
            print("  [!] Keine aktive Aufnahme!")
            return

        self._current_episode.end_time = time.time()
        self._current_episode.success = success
        self._recording = False

        filename = self._output_dir / f"episode_{self._current_episode.episode_id:04d}.json"
        duration = self._current_episode.end_time - self._current_episode.start_time
        num_frames = len(self._current_episode.frames)

        data = {
            "episode_id": self._current_episode.episode_id,
            "target_class": self._current_episode.target_class,
            "start_time": self._current_episode.start_time,
            "end_time": self._current_episode.end_time,
            "duration_s": duration,
            "num_frames": num_frames,
            "success": success,
            "frames": self._current_episode.frames,
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        status = "✓ ERFOLG" if success else "✗ FEHLGESCHLAGEN"
        print(f"\n  ■ AUFNAHME GESPEICHERT: {filename}")
        print(f"    {status} | {num_frames} Frames | {duration:.1f}s")

        self._current_episode = None

    def _record_frame(self, detections: List[Dict], action: str):
        """Zeichnet einen Frame auf."""
        if not self._current_episode:
            return

        state = self._arm_state
        now = time.time()

        # Relative Position zum nächsten Target
        rel_to_target = None
        if detections:
            best = detections[0]
            img_w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cx, cy = best['center_px']
            rel_to_target = {
                "target_class": best['class'],
                "target_confidence": best['confidence'],
                "offset_px_x": cx - img_w / 2,
                "offset_px_y": cy - img_h / 2,
                "target_size_px": best['size_px'],
                "target_bbox_normalized": [
                    best['bbox'][0] / img_w,
                    best['bbox'][1] / img_h,
                    best['bbox'][2] / img_w,
                    best['bbox'][3] / img_h,
                ],
            }

        frame_data = {
            "timestamp": now - self._current_episode.start_time,
            "arm_state": {
                "base_deg": state.base_deg,
                "shoulder_deg": state.shoulder_deg,
                "elbow_deg": state.elbow_deg,
                "hand_deg": state.hand_deg,
                "gripper_open": state.gripper_open,
            },
            "detections": detections,
            "action": action,
            "rel_to_target": rel_to_target,
        }

        self._current_episode.frames.append(frame_data)

    # ─── Annotation ──────────────────────────────────────────────────────

    def _annotate_frame(self, frame, detections: List[Dict], action: str, fps: float = 0):
        """Zeichnet Infos auf den Frame."""
        if frame is None:
            return

        h_f, w_f = frame.shape[:2]
        cx_img, cy_img = w_f // 2, h_f // 2

        # Fadenkreuz
        cv2.drawMarker(frame, (cx_img, cy_img), (128, 128, 128), cv2.MARKER_CROSS, 30, 1)

        # Detections
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            is_target = (self._target_class and det['class'] == self._target_class)
            color = (0, 0, 255) if is_target else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            det_cx, det_cy = int(det['center_px'][0]), int(det['center_px'][1])
            cv2.circle(frame, (det_cx, det_cy), 5, color, -1)
            if is_target:
                cv2.line(frame, (det_cx, det_cy), (cx_img, cy_img), (0, 255, 255), 1)
                dist = ((det_cx - cx_img)**2 + (det_cy - cy_img)**2) ** 0.5
                cv2.putText(frame, f"{dist:.0f}px",
                            ((det_cx + cx_img) // 2, (det_cy + cy_img) // 2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # ─── Status-Leiste oben ───
        state = self._arm_state
        status_line = (f"B:{state.base_deg:.0f} S:{state.shoulder_deg:.0f} "
                       f"E:{state.elbow_deg:.0f} H:{state.hand_deg:.0f} "
                       f"G:{'O' if state.gripper_open else 'C'} "
                       f"FPS:{fps:.0f}")
        cv2.putText(frame, status_line, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Aktive Tasten anzeigen (visuelles Feedback)
        active = self._get_active_keys()
        if active:
            active_str = " + ".join(sorted(active))
            cv2.putText(frame, f"Active: {active_str}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Aktuelle Aktion
        if action:
            cv2.putText(frame, f"Action: {action}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ─── Recording-Indikator ───
        if self._recording:
            cv2.circle(frame, (w_f - 30, 25), 10, (0, 0, 255), -1)
            ep = self._current_episode
            if ep:
                elapsed = time.time() - ep.start_time
                rec_text = f"REC {elapsed:.1f}s | {len(ep.frames)} frames"
                cv2.putText(frame, rec_text, (w_f - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Episodes: {self._episode_count} | R=Record S=Save Q=Quit",
                        (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ─── Steuerungs-Overlay ───
        help_y = h_f - 40
        cv2.putText(frame, "Arrows=Base/Shoulder  W/S=Elbow  A/D=Hand  O/C=Gripper",
                    (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # ─── Shutdown ─────────────────────────────────────────────────────────

    def _shutdown(self):
        """Aufräumen."""
        if self._recording:
            self._stop_recording(success=False)

        print("\n[Shutdown]...")
        if self._arm:
            self._arm.park()
            self._arm.set_led(0)
            time.sleep(1.0)
            self._arm.disconnect()
        if self._camera:
            self._camera.release()
        cv2.destroyAllWindows()
        print("  ✓ Fertig")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoArm-M2-S Teleop Recorder")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index (default: 2)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO-Modell")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min. Confidence")
    parser.add_argument("--output", type=str, default="recordings", help="Output-Verzeichnis")
    parser.add_argument("--target", type=str, default=None, help="Ziel-Objekt (z.B. 'bottle')")
    args = parser.parse_args()

    recorder = TeleopRecorder(
        port=args.port,
        camera_index=args.camera,
        model_path=args.model,
        confidence=args.confidence,
        output_dir=args.output,
        target_class=args.target,
    )
    recorder.run()


if __name__ == "__main__":
    main()

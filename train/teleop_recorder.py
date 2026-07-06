#!/usr/bin/env python3
"""
Teleop Recorder – Fernsteuerung des RoArm-M2-S mit Tastatur + Aufzeichnung.

Steuerung:
  Pfeiltasten Links/Rechts  → Base-Rotation
  Pfeiltasten Oben/Unten    → Shoulder (vor/zurück)
  Shift+Oben/Unten          → Elbow (hoch/runter)
  Shift+Links/Rechts        → Hand/Wrist Rotation
  O                          → Gripper öffnen
  C                          → Gripper schließen
  SPACE                      → Episode starten/stoppen (Toggle)
  R                          → Aufnahme starten (neue Episode)
  S                          → Episode speichern & beenden
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
class Detection:
    """Eine einzelne YOLO-Detection."""
    cls: str = ""
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x1, y1, x2, y2
    center_px: Tuple[float, float] = (0, 0)
    size_px: Tuple[float, float] = (0, 0)


@dataclass
class Frame:
    """Ein aufgezeichneter Frame."""
    timestamp: float = 0.0
    arm_state: ArmState = field(default_factory=ArmState)
    detections: List[Dict] = field(default_factory=list)
    action: str = ""  # Welche Taste gerade gedrückt ist
    # Relative Position zu nächster BBox (für Training)
    rel_to_target: Optional[Dict] = None


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
    """

    # Steuerungs-Parameter
    BASE_STEP = 2.0        # Grad pro Tick
    SHOULDER_STEP = 1.5    # Grad pro Tick
    ELBOW_STEP = 1.5       # Grad pro Tick
    HAND_STEP = 3.0        # Grad pro Tick
    MOVE_SPEED = 50        # Servo-Speed
    MOVE_ACC = 20          # Servo-Acceleration

    # Limits
    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    # Key codes (OpenCV)
    KEY_UP = 82
    KEY_DOWN = 84
    KEY_LEFT = 81
    KEY_RIGHT = 83
    # Shift+Arrow (varies by system, common codes)
    KEY_SHIFT_UP = 0  # Will detect via flags
    KEY_SHIFT_DOWN = 1
    KEY_SHIFT_LEFT = 2
    KEY_SHIFT_RIGHT = 3

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

        # Timing
        self._last_move_time = 0.0
        self._move_interval = 0.05  # 50ms zwischen Befehlen (20Hz Steuerung)

        # Aktive Tasten (für "gehalten = bewegt sich")
        self._active_keys = set()

        # Window
        self._window_name = "RoArm Teleop"

    def _count_existing_episodes(self) -> int:
        """Zählt bereits vorhandene Episoden im Output-Verzeichnis."""
        count = 0
        for f in self._output_dir.glob("episode_*.json"):
            count += 1
        return count

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
            # Fallback: andere Indizes probieren
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
                # Warmup
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
        self._arm.gripper_open()
        time.sleep(2.0)
        self._arm_state = ArmState(
            base_deg=0, shoulder_deg=0, elbow_deg=90, hand_deg=180, gripper_open=True
        )
        print("  ✓ Bereit")

        print("\n" + "=" * 60)
        print("  STEUERUNG:")
        print("    ←/→         Base drehen")
        print("    ↑/↓         Shoulder vor/zurück")
        print("    Shift+↑/↓   Elbow hoch/runter")
        print("    Shift+←/→   Hand/Wrist drehen")
        print("    O            Gripper öffnen")
        print("    C            Gripper schließen")
        print("    R            Aufnahme starten")
        print("    S            Aufnahme speichern")
        print("    Q            Beenden")
        print("=" * 60)
        print(f"\n  Episoden bisher: {self._episode_count}")
        if self._target_class:
            print(f"  Ziel-Objekt: '{self._target_class}'")
        print()

        return True

    # ─── Main Loop ────────────────────────────────────────────────────────

    def run(self):
        """Hauptschleife."""
        if not self.setup():
            return

        self._running = True
        fps_time = time.time()
        frame_count = 0

        try:
            while self._running:
                loop_start = time.time()

                # 1. Frame holen
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. Detection
                detections = self._detect(frame)

                # 3. Tastatur verarbeiten
                # OpenCV waitKey gibt nur 1 Taste zurück, aber wir brauchen
                # "gehalten" Erkennung. Lösung: kurzes waitKey + State-Tracking
                key = cv2.waitKey(1) & 0xFFFF  # 16-bit für Sondertasten

                action = self._process_key(key)

                # 4. Arm bewegen (wenn Taste gehalten)
                self._apply_movement()

                # 5. Frame annotieren
                self._annotate_frame(frame, detections, action)

                # 6. Aufzeichnen
                if self._recording:
                    self._record_frame(detections, action)

                # 7. Anzeigen
                cv2.imshow(self._window_name, frame)

                # FPS
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_time = time.time()

                # Loop-Timing (Ziel: ~30 FPS)
                loop_elapsed = time.time() - loop_start
                sleep_time = max(0, (1.0 / 30.0) - loop_elapsed)
                if sleep_time > 0:
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
        # Grab+Retrieve für neuesten Frame
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

            # Filter auf Target-Klasse wenn angegeben
            if self._target_class is None or cls_name == self._target_class:
                detections.append(det)

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    # ─── Tastatur ─────────────────────────────────────────────────────────

    def _process_key(self, key: int) -> str:
        """
        Verarbeitet Tastendruck. Gibt Aktions-String zurück.
        OpenCV Key-Codes (Linux):
          Pfeiltasten: 65361=Left, 65362=Up, 65363=Right, 65364=Down
          Mit Shift: Bit 16 gesetzt → key & 0xFF gleich, aber raw key anders
        """
        if key == -1 or key == 0xFFFF:
            # Keine Taste → alle Bewegungen stoppen
            self._active_keys.clear()
            return ""

        # Normalisieren
        raw_key = key
        key_low = key & 0xFF

        # Shift-Detection: OpenCV gibt bei Shift+Arrow andere Codes
        # Auf Linux/X11: Shift+Up = 65362 mit Modifier
        # Einfacher: Wir nutzen separate Tasten
        # w/a/s/d für Shift-Variante als Alternative

        action = ""

        # ─── Pfeiltasten (Base + Shoulder) ───
        if raw_key == 65361 or key_low == 81:  # Left
            self._active_keys.add("base_left")
            action = "base_left"
        elif raw_key == 65363 or key_low == 83:  # Right
            self._active_keys.add("base_right")
            action = "base_right"
        elif raw_key == 65362 or key_low == 82:  # Up
            self._active_keys.add("shoulder_up")
            action = "shoulder_up"
        elif raw_key == 65364 or key_low == 84:  # Down
            self._active_keys.add("shoulder_down")
            action = "shoulder_down"

        # ─── W/A/S/D für Elbow + Hand (Shift-Alternative) ───
        elif key_low == ord('w'):  # Elbow hoch
            self._active_keys.add("elbow_up")
            action = "elbow_up"
        elif key_low == ord('s') and not self._recording:  # Elbow runter (nur wenn nicht "save")
            self._active_keys.add("elbow_down")
            action = "elbow_down"
        elif key_low == ord('a'):  # Hand links
            self._active_keys.add("hand_left")
            action = "hand_left"
        elif key_low == ord('d'):  # Hand rechts
            self._active_keys.add("hand_right")
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
            action = ""
        elif key_low == ord('s') and self._recording:
            self._stop_recording(success=True)
            action = ""
        elif key_low == ord('f'):  # Failed episode
            self._stop_recording(success=False)
            action = ""

        # ─── Quit ───
        elif key_low == ord('q'):
            self._running = False
            action = ""

        return action

    def _apply_movement(self):
        """Wendet aktive Bewegungen an (solange Taste gehalten)."""
        now = time.time()
        if now - self._last_move_time < self._move_interval:
            return

        if not self._active_keys:
            return

        moved = False
        state = self._arm_state

        for key in list(self._active_keys):
            if key == "base_left":
                state.base_deg = max(self.BASE_MIN, state.base_deg - self.BASE_STEP)
                moved = True
            elif key == "base_right":
                state.base_deg = min(self.BASE_MAX, state.base_deg + self.BASE_STEP)
                moved = True
            elif key == "shoulder_up":
                state.shoulder_deg = min(self.SHOULDER_MAX, state.shoulder_deg + self.SHOULDER_STEP)
                moved = True
            elif key == "shoulder_down":
                state.shoulder_deg = max(self.SHOULDER_MIN, state.shoulder_deg - self.SHOULDER_STEP)
                moved = True
            elif key == "elbow_up":
                state.elbow_deg = min(self.ELBOW_MAX, state.elbow_deg + self.ELBOW_STEP)
                moved = True
            elif key == "elbow_down":
                state.elbow_deg = max(self.ELBOW_MIN, state.elbow_deg - self.ELBOW_STEP)
                moved = True
            elif key == "hand_left":
                state.hand_deg = max(self.HAND_MIN, state.hand_deg - self.HAND_STEP)
                moved = True
            elif key == "hand_right":
                state.hand_deg = min(self.HAND_MAX, state.hand_deg + self.HAND_STEP)
                moved = True

        if moved:
            self._send_arm_command()
            self._last_move_time = now

    def _send_arm_command(self):
        """Sendet aktuelle Gelenkwinkel an den Arm."""
        state = self._arm_state
        self._arm.move_joints_degrees(
            b=state.base_deg,
            s=state.shoulder_deg,
            e=state.elbow_deg,
            h=state.hand_deg,
            spd=self.MOVE_SPEED,
            acc=self.MOVE_ACC
        )

    def _gripper_open(self):
        self._arm.gripper_open()
        self._arm_state.gripper_open = True

    def _gripper_close(self):
        self._arm.gripper_close()
        self._arm_state.gripper_open = False

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

        # Speichern
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

        # Relative Position zum nächsten Target berechnen
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

    def _annotate_frame(self, frame, detections: List[Dict], action: str):
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

            # Zentrum + Linie zur Mitte
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
                       f"G:{'O' if state.gripper_open else 'C'}")
        cv2.putText(frame, status_line, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Aktuelle Aktion
        if action:
            cv2.putText(frame, f"Action: {action}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ─── Recording-Indikator ───
        if self._recording:
            # Roter Punkt + Text
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
        help_y = h_f - 80
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

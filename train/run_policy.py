#!/usr/bin/env python3
"""
Run Policy – Führt die trainierte Greif-Policy autonom aus.

Lädt das trainierte Modell und steuert den Arm basierend auf
YOLO-Detections + aktuellem Arm-Zustand.

Nutzung:
  python run_policy.py --model policy_model.pt
  python run_policy.py --model policy_model.pt --target bottle --episodes 5
  python run_policy.py --model policy_model.pt --camera 2 --confidence 0.4
"""

import cv2
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn

from roarm_m2s import RoArmM2S

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Import der Modell-Klassen und Feature-Extraktion aus train_policy
from train_policy import (
    ACTION_SPACE, NUM_ACTIONS, ACTION_TO_IDX,
    FeatureConfig, extract_frame_features, get_feature_dim,
    GraspPolicyMLP, GraspPolicyLSTM, GraspPolicyTransformer,
)


# ─── Policy Runner ───────────────────────────────────────────────────────────

class PolicyRunner:
    """
    Führt eine trainierte Policy autonom aus.
    Liest Kamera + YOLO → Features → Modell → Aktion → Arm-Befehl.
    """

    # Gleiche Parameter wie im Teleop-Recorder
    BASE_STEP = 2.0
    SHOULDER_STEP = 1.5
    ELBOW_STEP = 1.5
    HAND_STEP = 3.0
    MOVE_SPEED = 50
    MOVE_ACC = 20

    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    def __init__(self, model_path: str, port: str = None,
                 camera_index: int = 2, yolo_model: str = "yolo11n.pt",
                 confidence: float = 0.5, target_class: str = None,
                 max_steps: int = 500, action_interval: float = 0.1,
                 device: str = "auto"):
        self._model_path = model_path
        self._port = port
        self._camera_index = camera_index
        self._yolo_model_path = yolo_model
        self._confidence = confidence
        self._target_class = target_class
        self._max_steps = max_steps
        self._action_interval = action_interval

        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Hardware
        self._arm: Optional[RoArmM2S] = None
        self._camera: Optional[cv2.VideoCapture] = None
        self._yolo_model = None

        # Policy
        self._policy_model: Optional[nn.Module] = None
        self._feature_config: Optional[FeatureConfig] = None

        # State
        self._arm_state = {
            "base_deg": 0.0,
            "shoulder_deg": 0.0,
            "elbow_deg": 90.0,
            "hand_deg": 180.0,
            "gripper_open": True,
        }

        # Sequenz-Buffer (für LSTM/Transformer)
        self._seq_buffer: List[np.ndarray] = []
        self._seq_len: int = 1

        # Stats
        self._step_count = 0
        self._action_counts = {}

    # ─── Setup ────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialisiert alles."""
        print("=" * 60)
        print("  RoArm-M2-S Policy Runner")
        print("=" * 60)

        # 1. Modell laden
        print(f"\n[1] Policy-Modell laden: {self._model_path}")
        if not self._load_model():
            return False

        # 2. Arm verbinden
        print("\n[2] Arm verbinden...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            print("  ✓ Arm verbunden")
        except Exception as e:
            print(f"  ✗ Arm-Fehler: {e}")
            return False

        # 3. Kamera
        print(f"\n[3] Kamera {self._camera_index} öffnen...")
        self._camera = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            for idx in [0, 2, 1, 4]:
                self._camera = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if self._camera.isOpened():
                    self._camera_index = idx
                    break
            else:
                print("  ✗ Keine Kamera!")
                return False

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        print(f"  ✓ Kamera {self._camera_index}")

        # 4. YOLO
        if HAS_YOLO:
            print(f"\n[4] YOLO '{self._yolo_model_path}' laden...")
            try:
                self._yolo_model = YOLO(self._yolo_model_path)
                self._yolo_model.verbose = False
                ret, frame = self._camera.read()
                if ret:
                    self._yolo_model(frame, conf=self._confidence, verbose=False)
                print("  ✓ YOLO bereit")
            except Exception as e:
                print(f"  ✗ YOLO-Fehler: {e}")
                self._yolo_model = None
        else:
            print("\n[4] YOLO nicht verfügbar!")
            return False

        # 5. Arm in Startposition
        print("\n[5] Arm → Startposition...")
        self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
        self._arm.gripper_open()
        time.sleep(2.0)
        print("  ✓ Bereit")

        print("\n" + "=" * 60)
        print("  AUTONOMER MODUS")
        print(f"  Target: {self._target_class or '(alle)'}")
        print(f"  Max Steps: {self._max_steps}")
        print(f"  Action Interval: {self._action_interval}s")
        print("  Q = Abbrechen")
        print("=" * 60 + "\n")

        return True

    def _load_model(self) -> bool:
        """Lädt das trainierte Modell."""
        try:
            checkpoint = torch.load(self._model_path, map_location=self._device,
                                    weights_only=False)
        except Exception as e:
            print(f"  ✗ Kann Modell nicht laden: {e}")
            return False

        # Feature-Config rekonstruieren
        fc_data = checkpoint.get("feature_config", {})
        self._feature_config = FeatureConfig(
            max_detections=fc_data.get("max_detections", 5),
            normalize_bbox=fc_data.get("normalize_bbox", True),
            include_arm_state=fc_data.get("include_arm_state", True),
            include_gripper=fc_data.get("include_gripper", True),
            include_rel_to_target=fc_data.get("include_rel_to_target", True),
        )

        # Modell-Klasse bestimmen
        model_class = checkpoint.get("model_class", "GraspPolicyMLP")
        input_dim = get_feature_dim(self._feature_config)

        if model_class == "GraspPolicyMLP":
            self._policy_model = GraspPolicyMLP(input_dim=input_dim)
        elif model_class == "GraspPolicyLSTM":
            self._policy_model = GraspPolicyLSTM(input_dim=input_dim)
        elif model_class == "GraspPolicyTransformer":
            self._policy_model = GraspPolicyTransformer(input_dim=input_dim)
        else:
            print(f"  ✗ Unbekannte Modell-Klasse: {model_class}")
            return False

        # Weights laden
        self._policy_model.load_state_dict(checkpoint["model_state_dict"])
        self._policy_model.to(self._device)
        self._policy_model.eval()

        val_acc = checkpoint.get("val_acc", 0)
        epoch = checkpoint.get("epoch", 0)
        print(f"  ✓ {model_class} geladen (Epoch {epoch}, Val-Acc: {val_acc:.3f})")
        print(f"  ✓ Input-Dim: {input_dim}, Actions: {NUM_ACTIONS}")

        # Target-Klasse aus Checkpoint übernehmen falls nicht angegeben
        if not self._target_class:
            action_space = checkpoint.get("action_space", ACTION_SPACE)
            print(f"  ℹ Kein Target angegeben, nutze alle Detections")

        return True

    # ─── Main Loop ────────────────────────────────────────────────────────

    def run(self, num_episodes: int = 1) -> Dict:
        """
        Führt die Policy für mehrere Episoden aus.

        Returns:
            Dict mit Statistiken.
        """
        if not self.setup():
            return {"success": False, "error": "Setup fehlgeschlagen"}

        results = []

        try:
            for ep in range(1, num_episodes + 1):
                print(f"\n{'─'*40}")
                print(f"  Episode {ep}/{num_episodes}")
                print(f"{'─'*40}")

                result = self._run_episode()
                results.append(result)

                print(f"  → {result['steps']} Steps, "
                      f"{'ERFOLG' if result['completed'] else 'ABGEBROCHEN'}")

                # Reset für nächste Episode
                if ep < num_episodes:
                    print("  → Reset...")
                    self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
                    self._arm.gripper_open()
                    self._arm_state = {
                        "base_deg": 0.0, "shoulder_deg": 0.0,
                        "elbow_deg": 90.0, "hand_deg": 180.0,
                        "gripper_open": True,
                    }
                    time.sleep(2.0)

        except KeyboardInterrupt:
            print("\n[Abgebrochen]")
        finally:
            self._shutdown()

        # Zusammenfassung
        completed = sum(1 for r in results if r['completed'])
        print(f"\n{'='*60}")
        print(f"  ERGEBNIS: {completed}/{len(results)} Episoden abgeschlossen")
        print(f"  Aktions-Verteilung gesamt:")
        for action, count in sorted(self._action_counts.items(), key=lambda x: -x[1]):
            name = action if action else "(idle)"
            print(f"    {name}: {count}")
        print(f"{'='*60}\n")

        return {
            "episodes": results,
            "completed": completed,
            "total": len(results),
            "action_counts": self._action_counts,
        }

    def _run_episode(self) -> Dict:
        """Führt eine einzelne Episode aus."""
        self._step_count = 0
        self._seq_buffer = []
        episode_start = time.time()
        completed = False
        aborted = False

        while self._step_count < self._max_steps:
            step_start = time.time()

            # 1. Frame holen
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 2. Detection
            detections = self._detect(frame)

            # 3. Features extrahieren
            frame_data = self._build_frame_data(detections)
            features = extract_frame_features(frame_data, self._feature_config)

            # 4. Policy abfragen
            action_idx, action_probs = self._predict_action(features)
            action_name = ACTION_SPACE[action_idx]

            # 5. Aktion ausführen
            self._execute_action(action_name)

            # 6. Stats
            self._step_count += 1
            self._action_counts[action_name] = self._action_counts.get(action_name, 0) + 1

            # 7. Visualisierung
            self._annotate_frame(frame, detections, action_name, action_probs)
            cv2.imshow("Policy Runner", frame)

            # 8. Tastatur (Q = Abbrechen, SPACE = Episode beenden)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                aborted = True
                break
            elif key == ord(' '):
                completed = True
                break

            # 9. Timing
            elapsed = time.time() - step_start
            sleep_time = max(0, self._action_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.time() - episode_start
        return {
            "steps": self._step_count,
            "duration_s": duration,
            "completed": completed,
            "aborted": aborted,
        }

    # ─── Prediction ──────────────────────────────────────────────────────

    def _predict_action(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Fragt die Policy ab."""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self._device)

            # Für Sequenz-Modelle: Buffer nutzen
            if isinstance(self._policy_model, (GraspPolicyLSTM, GraspPolicyTransformer)):
                self._seq_buffer.append(features)
                if len(self._seq_buffer) > self._seq_len:
                    self._seq_buffer = self._seq_buffer[-self._seq_len:]

                seq = np.stack(self._seq_buffer)
                x = torch.FloatTensor(seq).unsqueeze(0).to(self._device)

            logits = self._policy_model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            action_idx = int(logits.argmax(dim=-1).item())

        return action_idx, probs

    # ─── Aktion ausführen ─────────────────────────────────────────────────

    def _execute_action(self, action: str):
        """Führt eine Aktion auf dem Arm aus."""
        state = self._arm_state

        if action == "base_left":
            state["base_deg"] = max(self.BASE_MIN, state["base_deg"] - self.BASE_STEP)
        elif action == "base_right":
            state["base_deg"] = min(self.BASE_MAX, state["base_deg"] + self.BASE_STEP)
        elif action == "shoulder_up":
            state["shoulder_deg"] = min(self.SHOULDER_MAX, state["shoulder_deg"] + self.SHOULDER_STEP)
        elif action == "shoulder_down":
            state["shoulder_deg"] = max(self.SHOULDER_MIN, state["shoulder_deg"] - self.SHOULDER_STEP)
        elif action == "elbow_up":
            state["elbow_deg"] = min(self.ELBOW_MAX, state["elbow_deg"] + self.ELBOW_STEP)
        elif action == "elbow_down":
            state["elbow_deg"] = max(self.ELBOW_MIN, state["elbow_deg"] - self.ELBOW_STEP)
        elif action == "hand_left":
            state["hand_deg"] = max(self.HAND_MIN, state["hand_deg"] - self.HAND_STEP)
        elif action == "hand_right":
            state["hand_deg"] = min(self.HAND_MAX, state["hand_deg"] + self.HAND_STEP)
        elif action == "gripper_open":
            if not state["gripper_open"]:
                self._arm.gripper_open()
                state["gripper_open"] = True
            return  # Kein Joint-Move nötig
        elif action == "gripper_close":
            if state["gripper_open"]:
                self._arm.gripper_close()
                state["gripper_open"] = False
            return  # Kein Joint-Move nötig
        elif action == "":
            return  # Idle

        # Joint-Befehl senden
        self._arm.move_joints_degrees(
            b=state["base_deg"],
            s=state["shoulder_deg"],
            e=state["elbow_deg"],
            h=state["hand_deg"],
            spd=self.MOVE_SPEED,
            acc=self.MOVE_ACC,
        )

    # ─── Hilfsfunktionen ─────────────────────────────────────────────────

    def _get_frame(self):
        if not self._camera:
            return None
        self._camera.grab()
        ret, frame = self._camera.retrieve()
        return frame if ret else None

    def _detect(self, frame) -> List[Dict]:
        if not self._yolo_model or frame is None:
            return []

        results = self._yolo_model(frame, conf=self._confidence, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if self._target_class and cls_name != self._target_class:
                continue

            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'center_px': [cx, cy],
                'size_px': [x2 - x1, y2 - y1],
            })

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def _build_frame_data(self, detections: List[Dict]) -> Dict:
        """Baut ein Frame-Dict wie es extract_frame_features erwartet."""
        state = self._arm_state

        # Relative zum Target
        rel_to_target = None
        if detections:
            best = detections[0]
            cx, cy = best['center_px']
            rel_to_target = {
                "offset_px_x": cx - 320,
                "offset_px_y": cy - 240,
                "target_size_px": best['size_px'],
            }

        return {
            "arm_state": state,
            "detections": detections,
            "rel_to_target": rel_to_target,
        }

    def _annotate_frame(self, frame, detections: List[Dict],
                        action: str, probs: np.ndarray):
        """Zeichnet Policy-Infos auf den Frame."""
        if frame is None:
            return

        h_f, w_f = frame.shape[:2]

        # Detections
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Fadenkreuz
        cv2.drawMarker(frame, (w_f // 2, h_f // 2), (128, 128, 128),
                       cv2.MARKER_CROSS, 30, 1)

        # Status
        state = self._arm_state
        status = (f"B:{state['base_deg']:.0f} S:{state['shoulder_deg']:.0f} "
                  f"E:{state['elbow_deg']:.0f} H:{state['hand_deg']:.0f} "
                  f"G:{'O' if state['gripper_open'] else 'C'}")
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Aktion + Confidence
        if action:
            conf = probs[ACTION_TO_IDX.get(action, 0)]
            action_text = f"ACTION: {action} ({conf:.2f})"
            cv2.putText(frame, action_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Step-Counter
        cv2.putText(frame, f"Step: {self._step_count}/{self._max_steps}",
                    (w_f - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Top-3 Aktionen
        top_indices = np.argsort(probs)[::-1][:3]
        for i, idx in enumerate(top_indices):
            name = ACTION_SPACE[idx] if ACTION_SPACE[idx] else "(idle)"
            text = f"{name}: {probs[idx]:.2f}"
            cv2.putText(frame, text, (w_f - 200, 55 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # AUTONOM-Indikator
        cv2.putText(frame, "AUTONOM", (10, h_f - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ─── Shutdown ─────────────────────────────────────────────────────────

    def _shutdown(self):
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
    parser = argparse.ArgumentParser(description="Run trained grasp policy")
    parser.add_argument("--model", type=str, required=True, help="Pfad zum trainierten Modell (.pt)")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index")
    parser.add_argument("--yolo", type=str, default="yolo11n.pt", help="YOLO-Modell für Detection")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min. Detection Confidence")
    parser.add_argument("--target", type=str, default=None, help="Ziel-Objekt")
    parser.add_argument("--episodes", type=int, default=1, help="Anzahl Episoden")
    parser.add_argument("--max-steps", type=int, default=500, help="Max. Steps pro Episode")
    parser.add_argument("--interval", type=float, default=0.1, help="Sekunden zwischen Aktionen")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    runner = PolicyRunner(
        model_path=args.model,
        port=args.port,
        camera_index=args.camera,
        yolo_model=args.yolo,
        confidence=args.confidence,
        target_class=args.target,
        max_steps=args.max_steps,
        action_interval=args.interval,
        device=args.device,
    )

    runner.run(num_episodes=args.episodes)


if __name__ == "__main__":
    main()

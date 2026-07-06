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
  P                          → Letzte Episode 1:1 abspielen (Replay)
  Q                          → Beenden

Aufzeichnung:
  - Bounding Boxes (Klasse, Koordinaten, Confidence)
  - Arm-Gelenkwinkel + kartesische Position
  - Aktionen (Tasteneingaben) mit Timestamps
  - Alles relativ zu den BBoxes
  - LeRobot-kompatibles Format (HuggingFace Dataset)
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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

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

# ─── LeRobot Dataset Helper ──────────────────────────────────────────────────

class LeRobotSaver:
    """
    Speichert Episoden im LeRobot-kompatiblen Format.
    
    Struktur:
      output_dir/
        meta/
          info.json          # Dataset-Metadaten
          episodes.jsonl     # Episode-Index
          stats.json         # Statistiken
          tasks.jsonl        # Task-Beschreibungen
        data/
          chunk-000/
            episode_000000.parquet   # Frames als Parquet
            episode_000001.parquet
            ...
        videos/              # (optional) Kamera-Videos
          chunk-000/
            observation.images.top/
              episode_000000.mp4
    """

    def __init__(self, output_dir: Path, fps: float = 30.0):
        self._output_dir = output_dir
        self._fps = fps
        self._meta_dir = output_dir / "meta"
        self._data_dir = output_dir / "data" / "chunk-000"
        self._video_dir = output_dir / "videos" / "chunk-000" / "observation.images.top"
        
        # Erstelle Verzeichnisse
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        
        # Lade oder erstelle info.json
        self._info_path = self._meta_dir / "info.json"
        self._episodes_path = self._meta_dir / "episodes.jsonl"
        self._tasks_path = self._meta_dir / "tasks.jsonl"
        self._stats_path = self._meta_dir / "stats.json"
        
        self._total_episodes = 0
        self._total_frames = 0
        self._load_or_create_meta()

    def _load_or_create_meta(self):
        """Lade bestehende Metadaten oder erstelle neue."""
        if self._info_path.exists():
            with open(self._info_path, 'r') as f:
                info = json.load(f)
                self._total_episodes = info.get("total_episodes", 0)
                self._total_frames = info.get("total_frames", 0)
        else:
            self._save_info()
        
        # Tasks-Datei erstellen falls nicht vorhanden
        if not self._tasks_path.exists():
            with open(self._tasks_path, 'w') as f:
                task = {"task_index": 0, "task": "Pick up target object with robot arm"}
                f.write(json.dumps(task) + "\n")

    def _save_info(self):
        """Speichert Dataset-Info."""
        info = {
            "codebase_version": "v2.1",
            "robot_type": "roarm_m2s",
            "total_episodes": self._total_episodes,
            "total_frames": self._total_frames,
            "fps": self._fps,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [4],
                    "names": ["base_deg", "shoulder_deg", "elbow_deg", "hand_deg"]
                },
                "observation.gripper": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["gripper_position"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [5],
                    "names": ["base_deg", "shoulder_deg", "elbow_deg", "hand_deg", "gripper"]
                },
                "observation.images.top": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "video_info": {
                        "video.fps": self._fps,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "has_audio": False
                    }
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["time_s"]
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["episode_index"]
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["frame_index"]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["global_index"]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["task_index"]
                }
            },
            "splits": {
                "train": f"0:{self._total_episodes}"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": 1000
        }
        with open(self._info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def save_episode(self, episode: Episode, frames_images: List[np.ndarray] = None) -> Path:
        """
        Speichert eine Episode im LeRobot-Format.
        
        Args:
            episode: Die aufgezeichnete Episode
            frames_images: Liste von Kamera-Frames (numpy arrays) für Video
            
        Returns:
            Pfad zur gespeicherten Parquet-Datei
        """
        ep_idx = episode.episode_id - 1  # 0-basiert
        
        # ─── Parquet-Daten vorbereiten ───
        records = []
        for i, frame_data in enumerate(episode.frames):
            arm = frame_data["arm_state"]
            action_str = frame_data.get("action", "")
            
            # Nächster State als Action (oder gleicher State wenn letzter Frame)
            if i < len(episode.frames) - 1:
                next_arm = episode.frames[i + 1]["arm_state"]
                action_vec = [
                    next_arm["base_deg"],
                    next_arm["shoulder_deg"],
                    next_arm["elbow_deg"],
                    next_arm["hand_deg"],
                    0.0 if next_arm["gripper_open"] else 1.0
                ]
            else:
                action_vec = [
                    arm["base_deg"],
                    arm["shoulder_deg"],
                    arm["elbow_deg"],
                    arm["hand_deg"],
                    0.0 if arm["gripper_open"] else 1.0
                ]
            
            record = {
                "observation.state": [
                    arm["base_deg"],
                    arm["shoulder_deg"],
                    arm["elbow_deg"],
                    arm["hand_deg"]
                ],
                "observation.gripper": [0.0 if arm["gripper_open"] else 1.0],
                "action": action_vec,
                "timestamp": [frame_data["timestamp"]],
                "episode_index": [ep_idx],
                "frame_index": [i],
                "index": [self._total_frames + i],
                "task_index": [0],
                # Zusätzliche Daten für Replay
                "action_label": action_str,
            }
            
            # Detections als JSON-String speichern
            if frame_data.get("detections"):
                record["detections_json"] = json.dumps(frame_data["detections"])
            else:
                record["detections_json"] = "[]"
            
            if frame_data.get("rel_to_target"):
                record["rel_to_target_json"] = json.dumps(frame_data["rel_to_target"])
            else:
                record["rel_to_target_json"] = "{}"
            
            records.append(record)
        
        # ─── Als Parquet speichern ───
        parquet_path = self._data_dir / f"episode_{ep_idx:06d}.parquet"
        
        if HAS_PARQUET and records:
            # Konvertiere zu spaltenbasiertem Format
            columns = {}
            for key in records[0].keys():
                columns[key] = [r[key] for r in records]
            
            table = pa.table(columns)
            pq.write_table(table, parquet_path)
        else:
            # Fallback: JSON speichern
            json_path = self._data_dir / f"episode_{ep_idx:06d}.json"
            with open(json_path, 'w') as f:
                json.dump(records, f, indent=2)
            parquet_path = json_path
        
        # ─── Video speichern (wenn Frames vorhanden) ───
        if frames_images and len(frames_images) > 0:
            video_path = self._video_dir / f"episode_{ep_idx:06d}.mp4"
            self._save_video(frames_images, video_path)
        
        # ─── Episode-Index aktualisieren ───
        num_frames = len(episode.frames)
        episode_entry = {
            "episode_index": ep_idx,
            "task_index": 0,
            "task": "Pick up target object with robot arm",
            "length": num_frames,
            "target_class": episode.target_class,
            "success": episode.success,
            "duration_s": episode.end_time - episode.start_time
        }
        with open(self._episodes_path, 'a') as f:
            f.write(json.dumps(episode_entry) + "\n")
        
        # ─── Metadaten aktualisieren ───
        self._total_episodes += 1
        self._total_frames += num_frames
        self._save_info()
        self._update_stats(records)
        
        return parquet_path

    def _save_video(self, frames: List[np.ndarray], video_path: Path):
        """Speichert Frames als MP4-Video."""
        if not frames:
            return
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, self._fps, (w, h))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        print(f"    📹 Video gespeichert: {video_path} ({len(frames)} frames)")

    def _update_stats(self, records: List[Dict]):
        """Aktualisiert Statistiken über alle Episoden."""
        if not records:
            return
        
        # Berechne Min/Max/Mean für States und Actions
        states = np.array([r["observation.state"] for r in records])
        actions = np.array([r["action"] for r in records])
        
        stats = {
            "observation.state": {
                "min": states.min(axis=0).tolist(),
                "max": states.max(axis=0).tolist(),
                "mean": states.mean(axis=0).tolist(),
                "std": states.std(axis=0).tolist()
            },
            "action": {
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist()
            }
        }
        
        with open(self._stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

# ─── Replay Engine ───────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Spielt eine aufgezeichnete Episode 1:1 auf dem Arm ab.
    Reproduziert exakt die gleichen Gelenkwinkel mit dem gleichen Timing.
    """

    def __init__(self, arm: RoArmM2S, camera: cv2.VideoCapture = None):
        self._arm = arm
        self._camera = camera
        self._replaying = False

    def replay_episode(self, episode_path: Path, window_name: str = "RoArm Replay") -> bool:
        """
        Spielt eine Episode 1:1 ab – mit exakt den aufgezeichneten spd/acc Werten.
        """
        # Lade Episode
        frames_data = self._load_episode(episode_path)
        if not frames_data:
            print(f"  [!] Keine Frames in {episode_path}")
            return False

        print(f"\n  ▶ REPLAY START: {episode_path.name}")
        print(f"    {len(frames_data)} Frames")

        if frames_data and "timestamp" in frames_data[0]:
            duration = frames_data[-1]["timestamp"]
            if isinstance(duration, list):
                duration = duration[0]
            print(f"    Dauer: {duration:.1f}s")

        self._replaying = True
        start_time = time.time()

        # Tracking für Zustandsänderungen
        last_gripper_state = None
        last_led_brightness = None

        for i, frame_data in enumerate(frames_data):
            if not self._replaying:
                print("  [!] Replay abgebrochen")
                break

            # Timing einhalten
            target_time = self._get_timestamp(frame_data)
            elapsed = time.time() - start_time
            wait_time = target_time - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

            # Gelenkwinkel setzen
            arm_state = self._get_arm_state(frame_data)
            if arm_state:
                # Aufgezeichnete Servo-Parameter verwenden (Fallback für alte Aufnahmen)
                replay_spd = frame_data.get("servo_spd", 100)
                replay_acc = frame_data.get("servo_acc", 200)

                cmd = {
                    "T": 122,
                    "b": round(arm_state["base_deg"], 1),
                    "s": round(arm_state["shoulder_deg"], 1),
                    "e": round(arm_state["elbow_deg"], 1),
                    "h": round(arm_state["hand_deg"], 1),
                    "spd": replay_spd,   # <-- EXAKT wie bei Aufnahme
                    "acc": replay_acc    # <-- EXAKT wie bei Aufnahme
                }
                self._arm._send_nowait(cmd)

                # Gripper NUR bei Zustandsänderung
                gripper_open = arm_state.get("gripper_open", True)
                if gripper_open != last_gripper_state:
                    if gripper_open:
                        self._arm.gripper_open()
                    else:
                        self._arm.gripper_close()
                    last_gripper_state = gripper_open

            # LED wiederherstellen (nur bei Änderung)
            led_brightness = frame_data.get("led_brightness", None)
            if led_brightness is not None and led_brightness != last_led_brightness:
                self._arm.set_led(led_brightness)
                last_led_brightness = led_brightness

            # Live-Anzeige
            if self._camera:
                self._camera.grab()
                ret, frame = self._camera.retrieve()
                if ret:
                    self._annotate_replay_frame(frame, frame_data, i, len(frames_data))
                    cv2.imshow(window_name, frame)

            # Abbruch mit Q
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._replaying = False
                print("  [!] Replay durch Benutzer abgebrochen")
                break

        self._replaying = False
        actual_duration = time.time() - start_time
        print(f"  ■ REPLAY BEENDET ({actual_duration:.1f}s)")
        return True

    def _load_episode(self, path: Path) -> List[Dict]:
        """Lädt Episode aus JSON oder Parquet."""
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            # Unterstütze beide Formate (alt: mit "frames" key, neu: direkte Liste)
            if isinstance(data, dict) and "frames" in data:
                return data["frames"]
            elif isinstance(data, list):
                return data
            return []
        
        elif path.suffix == '.parquet' and HAS_PARQUET:
            table = pq.read_table(path)
            df = table.to_pydict()
            frames = []
            num_rows = len(df.get("frame_index", []))
            for i in range(num_rows):
                frame = {}
                for key, values in df.items():
                    frame[key] = values[i]
                frames.append(frame)
            return frames
        
        return []

    def _get_timestamp(self, frame_data: Dict) -> float:
        """Extrahiert Timestamp aus Frame-Daten."""
        ts = frame_data.get("timestamp", 0)
        if isinstance(ts, list):
            return ts[0]
        return float(ts)

    def _get_arm_state(self, frame_data: Dict) -> Optional[Dict]:
        """Extrahiert Arm-State aus Frame-Daten (beide Formate)."""
        # Altes Format (JSON mit arm_state dict)
        if "arm_state" in frame_data:
            return frame_data["arm_state"]
        
        # LeRobot-Format (observation.state als Liste)
        if "observation.state" in frame_data:
            state = frame_data["observation.state"]
            gripper = frame_data.get("observation.gripper", [0.0])
            if isinstance(gripper, list):
                gripper_val = gripper[0]
            else:
                gripper_val = float(gripper)
            
            return {
                "base_deg": state[0],
                "shoulder_deg": state[1],
                "elbow_deg": state[2],
                "hand_deg": state[3],
                "gripper_open": gripper_val < 0.5
            }
        
        return None

    def _annotate_replay_frame(self, frame, frame_data: Dict, current_idx: int, total: int):
        """Annotiert Frame während Replay."""
        h, w = frame.shape[:2]
        
        # Replay-Indikator
        progress = current_idx / max(1, total - 1)
        cv2.putText(frame, f"REPLAY [{current_idx+1}/{total}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        
        # Progress-Bar
        bar_w = w - 20
        bar_x = 10
        bar_y = 45
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), (0, 200, 0), -1)
        
        # Arm-State anzeigen
        arm_state = self._get_arm_state(frame_data)
        if arm_state:
            state_str = (f"B:{arm_state['base_deg']:.0f} S:{arm_state['shoulder_deg']:.0f} "
                        f"E:{arm_state['elbow_deg']:.0f} H:{arm_state['hand_deg']:.0f} "
                        f"G:{'O' if arm_state.get('gripper_open', True) else 'C'}")
            cv2.putText(frame, state_str, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Action anzeigen
        action = frame_data.get("action_label", frame_data.get("action", ""))
        if action:
            cv2.putText(frame, f"Action: {action}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Hinweis
        cv2.putText(frame, "Q = Abbrechen", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

# ─── Teleop Controller ───────────────────────────────────────────────────────

class TeleopRecorder:
    """
    Fernsteuerung + Aufzeichnung für Behaviour Cloning.
    
    Smooth movement via:
    - Persistent key state tracking (not relying on OS key-repeat)
    - High-frequency command streaming in a separate thread
    - Small position increments at consistent rate
    """

    # ─── Steuerungs-Parameter (FIXED) ────────────────────────────────────────
    
    # Command streaming rate (separate from display loop)
    COMMAND_HZ = 50            # 50 commands/sec to servo
    COMMAND_INTERVAL = 1.0 / 50  # 20ms between commands
    
    # Speed levels: (base_step, shoulder_step, elbow_step, hand_step, servo_spd, servo_acc)
    SPEED_LEVELS = [
        # Level 0: Very Slow (precision)
        {"base": 0.15, "shoulder": 0.12, "elbow": 0.12, "hand": 0.2, "spd": 15, "acc": 30, "label": "VERY SLOW"},
        # Level 1: Slow
        {"base": 0.3, "shoulder": 0.25, "elbow": 0.25, "hand": 0.4, "spd": 30, "acc": 50, "label": "SLOW"},
        # Level 2: Medium (default)
        {"base": 0.6, "shoulder": 0.5, "elbow": 0.5, "hand": 0.7, "spd": 50, "acc": 100, "label": "MEDIUM"},
        # Level 3: Fast
        {"base": 1.0, "shoulder": 0.8, "elbow": 0.8, "hand": 1.2, "spd": 80, "acc": 150, "label": "FAST"},
        # Level 4: Very Fast
        {"base": 1.5, "shoulder": 1.2, "elbow": 1.2, "hand": 1.8, "spd": 100, "acc": 200, "label": "VERY FAST"},
    ]
    
    DEFAULT_SPEED_LEVEL = 2  # Medium
    
    # Limits
    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    # Timing für flüssige Steuerung
    KEY_HOLD_TIMEOUT = 0.35  # Sekunden: Taste gilt als "gehalten" für diese Dauer nach letztem Druck
    
    # Speed change keys
    SPEED_UP_KEY = ord('+')      # '+' or '=' key to increase speed
    SPEED_UP_KEY_ALT = ord('=')  # For keyboards without numpad
    SPEED_DOWN_KEY = ord('-')    # '-' key to decrease speed

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

        # ─── Speed control ───
        self._speed_level = self.DEFAULT_SPEED_LEVEL

        # ─── LED control ───
        self._led_brightness = 255  # Current LED brightness (0-255)
        self.LED_STEPS = [0, 51, 102, 153, 204, 255]  # 6 steps from off to full

        # ─── NEW: Persistent key state (press/release tracking) ───
        self._keys_down: set = set()           # Currently held keys
        self._key_last_seen: Dict[str, float] = {}  # Timestamp of last keypress event

        # ─── NEW: Command thread ───
        self._cmd_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0
        self._last_action = ""

        # ─── NEW: Frame image buffer for video recording ───
        self._frame_buffer: List[np.ndarray] = []
        self._record_video = True  # Kamera-Frames für Video aufzeichnen

        # ─── NEW: LeRobot Saver ───
        self._lerobot_saver: Optional[LeRobotSaver] = None

        # ─── NEW: Replay Engine ───
        self._replay_engine: Optional[ReplayEngine] = None

        # Window
        self._window_name = "RoArm Teleop"

    def _set_led_brightness(self, level_index: int):
        """Set LED brightness by step index (0-5)."""
        level_index = max(0, min(len(self.LED_STEPS) - 1, level_index))
        self._led_brightness = self.LED_STEPS[level_index]
        self._arm.set_led(self._led_brightness)
        if self._led_brightness == 0:
            print(f"  💡 LED: OFF")
        else:
            print(f"  💡 LED: {self._led_brightness}/255 (Step {level_index}/{len(self.LED_STEPS)-1})")

    def _count_existing_episodes(self) -> int:
        """Zählt bereits vorhandene Episoden im Output-Verzeichnis."""
        count = 0
        for f in self._output_dir.glob("episode_*.json"):
            count += 1
        # Auch LeRobot-Format zählen
        lerobot_data = self._output_dir / "data" / "chunk-000"
        if lerobot_data.exists():
            for f in lerobot_data.glob("episode_*.parquet"):
                count += 1
            for f in lerobot_data.glob("episode_*.json"):
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

    @property
    def _current_speed(self) -> dict:
        """Returns the current speed level configuration."""
        return self.SPEED_LEVELS[self._speed_level]

    
    def _change_speed(self, delta: int):
        """Change speed level by delta (+1 or -1). Clamps to valid range."""
        old_level = self._speed_level
        self._speed_level = max(0, min(len(self.SPEED_LEVELS) - 1, self._speed_level + delta))
        if self._speed_level != old_level:
            spd = self._current_speed
            print(f"  ⚡ Speed: {spd['label']} (Level {self._speed_level}/{len(self.SPEED_LEVELS)-1})")

    # ─── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialisiert Hardware."""
        print("=" * 60)
        print("  RoArm-M2-S Teleop Recorder")
        print("=" * 60)

        # Arm verbinden
        print("\n[1] Arm verbinden...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            print("  ✔ Arm verbunden")
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
        print(f"  ✔ Kamera {self._camera_index} ({w}x{h})")

        # YOLO
        if HAS_YOLO:
            print(f"\n[3] YOLO '{self._model_path}' laden...")
            try:
                self._model = YOLO(self._model_path)
                self._model.verbose = False
                ret, frame = self._camera.read()
                if ret:
                    self._model(frame, conf=self._confidence, verbose=False)
                print(f"  ✔ YOLO bereit")
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
        self._led_brightness = 255

        # ─── LeRobot Saver initialisieren ───
        self._lerobot_saver = LeRobotSaver(
            output_dir=self._output_dir / "lerobot_dataset",
            fps=30.0
        )
        print(f"  ✔ LeRobot Saver → {self._output_dir / 'lerobot_dataset'}")

        # ─── Replay Engine initialisieren ───
        self._replay_engine = ReplayEngine(self._arm, self._camera)
        print("  ✔ Replay Engine bereit")

        print("  ✔ Bereit")

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
        print("    P         Letzte Episode 1:1 abspielen (Replay)")
        print("    +/=       Geschwindigkeit erhöhen")
        print("    -         Geschwindigkeit verringern")
        print("    1-5       Geschwindigkeit direkt setzen (1=sehr langsam, 5=sehr schnell)")
        print("  LED-STEUERUNG (Shift + Zahl):")
        print("    !  (Shift+1)  LED AUS (0)")
        print('    "  (Shift+2)  LED  51/255 (20%)')
        print("    §  (Shift+3)  LED 102/255 (40%)")
        print("    $  (Shift+4)  LED 153/255 (60%)")
        print("    %  (Shift+5)  LED 204/255 (80%)")
        print("    &  (Shift+6)  LED 255/255 (100%)")
        print("    Q         Beenden")
        print("=" * 60)
        print(f"\n  Episoden bisher: {self._episode_count}")
        if self._target_class:
            print(f"  Ziel-Objekt: '{self._target_class}'")
        print()

        return True

    # ─── Main Loop ────────────────────────────────────────────────────────────

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

                action = ""
                for _ in range(50):
                    key = cv2.waitKey(1) & 0xFFFF
                    if key == -1 or key == 0xFFFF:
                        break
                    key_action = self._process_key(key)
                    if key_action:
                        action = key_action

                # Also: record the action from active keys if no key event this frame:
                if not action and self._recording:
                    active = self._get_active_keys()
                    if active:
                        # Use the first active movement as the action label
                        action = next(iter(active))

                # 4. Apply movement (at fixed rate, independent of display)
                self._apply_movement()

                # 5. Annotate
                self._annotate_frame(frame, detections, action, current_fps)

                # 6. Record (frame data + image for video)
                if self._recording:
                    self._record_frame(detections, action)
                    if self._record_video:
                        self._frame_buffer.append(frame.copy())

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

    # ─── Frame & Detection ────────────────────────────────────────────────────

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

    # ─── Tastatur ─────────────────────────────────────────────────────────────

    def _process_key(self, key: int) -> str:
        """
        Verarbeitet Tastendruck. Updates persistent key state.
        """
        if key == -1 or key == 0xFFFF:
            return ""

        now = time.time()
        key_low = key & 0xFF
        action = ""

        # ——— Pfeiltasten ———
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

        # ——— Ctrl+S → Save Recording (key code 19) ———
        elif key_low == 19:
            if self._recording:
                self._stop_recording(success=True)
            action = ""

        # ——— Ctrl+F → Fail Recording (key code 6) ———
        elif key_low == 6:
            if self._recording:
                self._stop_recording(success=False)
            action = ""

        # ——— W/A/S/D ———
        elif key_low == ord('w'):
            self._keys_down.add("elbow_up")
            self._key_last_seen["elbow_up"] = now
            action = "elbow_up"
        elif key_low == ord('s'):
            # Plain 's' is ALWAYS elbow_down now (no conflict with save)
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

        # ——— Gripper ———
        elif key_low == ord('o'):
            self._gripper_open()
            action = "gripper_open"
        elif key_low == ord('c'):
            self._gripper_close()
            action = "gripper_close"

        # ——— LED Control (Shift+1 through Shift+6) ———
        elif key_low == ord('!') or key == ord('!'):
            self._set_led_brightness(0)
            action = "led_0"
        elif key_low == ord('"') or key == ord('"'):
            self._set_led_brightness(1)
            action = "led_51"
        elif key_low == 167 or key == 167:
            self._set_led_brightness(2)
            action = "led_102"
        elif key_low == ord('$') or key == ord('$'):
            self._set_led_brightness(3)
            action = "led_153"
        elif key_low == ord('%') or key == ord('%'):
            self._set_led_brightness(4)
            action = "led_204"
        elif key_low == ord('&') or key == ord('&'):
            self._set_led_brightness(5)
            action = "led_255"

        # ——— Speed Control ———
        elif key_low == ord('+') or key_low == ord('='):
            self._change_speed(+1)
        elif key_low == ord('-') or key_low == ord('_'):
            self._change_speed(-1)
        elif key_low == ord('1'):
            self._speed_level = 0
            print(f"  ⚡ Speed: {self._current_speed['label']}")
        elif key_low == ord('2'):
            self._speed_level = 1
            print(f"  ⚡ Speed: {self._current_speed['label']}")
        elif key_low == ord('3'):
            self._speed_level = 2
            print(f"  ⚡ Speed: {self._current_speed['label']}")
        elif key_low == ord('4'):
            self._speed_level = 3
            print(f"  ⚡ Speed: {self._current_speed['label']}")
        elif key_low == ord('5'):
            self._speed_level = 4
            print(f"  ⚡ Speed: {self._current_speed['label']}")

        # ——— Recording ———
        elif key_low == ord('r'):
            self._start_recording()

        # ——— Plain 'f' is now free (no action, or you could assign it to something else) ———
        elif key_low == ord('f'):
            pass  # No longer stops recording; Ctrl+F does that now

        # ——— Replay ———
        elif key_low == ord('p'):
            self._replay_last_episode()

        # ——— Quit ———
        elif key_low == ord('q'):
            self._running = False

        if action:
            self._last_action = action
        return action

    def _apply_movement(self):
        """
        Apply movement at a FIXED RATE regardless of display loop timing.
        Uses dynamic speed levels for all joint movements.
        """
        now = time.time()
        dt = now - self._last_cmd_time

        if dt < self.COMMAND_INTERVAL:
            return

        active = self._get_active_keys()
        if not active:
            self._last_cmd_time = now
            return

        # Calculate steps proportional to actual elapsed time (velocity-based)
        time_factor = dt / self.COMMAND_INTERVAL  # Normally ~1.0
        time_factor = min(time_factor, 3.0)  # Cap to prevent jumps after stalls

        # Get current speed settings
        spd_cfg = self._current_speed

        moved = False
        state = self._arm_state

        if "base_left" in active:
            state.base_deg = min(self.BASE_MAX, state.base_deg + spd_cfg["base"] * time_factor)
            moved = True
        if "base_right" in active:
            state.base_deg = max(self.BASE_MIN, state.base_deg - spd_cfg["base"] * time_factor)
            moved = True
        if "shoulder_up" in active:
            state.shoulder_deg = min(self.SHOULDER_MAX, state.shoulder_deg + spd_cfg["shoulder"] * time_factor)
            moved = True
        if "shoulder_down" in active:
            state.shoulder_deg = max(self.SHOULDER_MIN, state.shoulder_deg - spd_cfg["shoulder"] * time_factor)
            moved = True
        if "elbow_up" in active:
            state.elbow_deg = max(self.ELBOW_MIN, state.elbow_deg - spd_cfg["elbow"] * time_factor)
            moved = True
        if "elbow_down" in active:
            state.elbow_deg = min(self.ELBOW_MAX, state.elbow_deg + spd_cfg["elbow"] * time_factor)
            moved = True
        if "hand_left" in active:
            state.hand_deg = max(self.HAND_MIN, state.hand_deg - spd_cfg["hand"] * time_factor)
            moved = True
        if "hand_right" in active:
            state.hand_deg = min(self.HAND_MAX, state.hand_deg + spd_cfg["hand"] * time_factor)
            moved = True

        if moved:
            self._send_arm_command()

        self._last_cmd_time = now

    def _send_arm_command(self):
        """
        Send current joint angles to the arm.
        Uses dynamic speed/acceleration from current speed level.
        """
        state = self._arm_state
        spd_cfg = self._current_speed
        cmd = {
            "T": 122,
            "b": round(state.base_deg, 1),
            "s": round(state.shoulder_deg, 1),
            "e": round(state.elbow_deg, 1),
            "h": round(state.hand_deg, 1),
            "spd": spd_cfg["spd"],
            "acc": spd_cfg["acc"]
        }
        self._arm._send_nowait(cmd)

    def _gripper_open(self):
        """Öffnet den Gripper und synchronisiert hand_deg."""
        self._arm_state.gripper_open = True
        GRIPPER_OPEN_DEG = 61.88
        self._arm_state.hand_deg = GRIPPER_OPEN_DEG
        self._arm.gripper_open()

    def _gripper_close(self):
        """Schließt den Gripper und synchronisiert hand_deg."""
        self._arm_state.gripper_open = False
        GRIPPER_CLOSED_DEG = 180.0
        self._arm_state.hand_deg = GRIPPER_CLOSED_DEG
        self._arm.gripper_close()

    # ─── Recording ────────────────────────────────────────────────────────────

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
        self._frame_buffer = []  # Reset frame buffer für Video
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

        duration = self._current_episode.end_time - self._current_episode.start_time
        num_frames = len(self._current_episode.frames)

        # ─── 1. Original JSON speichern (wie bisher) ───
        filename = self._output_dir / f"episode_{self._current_episode.episode_id:04d}.json"

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

        status = "✔ ERFOLG" if success else "✗ FEHLGESCHLAGEN"
        print(f"\n  ■ AUFNAHME GESPEICHERT: {filename}")
        print(f"    {status} | {num_frames} Frames | {duration:.1f}s")

        # ─── 2. LeRobot-Format speichern ───
        if self._lerobot_saver:
            try:
                lerobot_path = self._lerobot_saver.save_episode(
                    self._current_episode,
                    frames_images=self._frame_buffer if self._record_video else None
                )
                print(f"    📦 LeRobot: {lerobot_path}")
            except Exception as e:
                print(f"    [!] LeRobot-Speicherfehler: {e}")

        # ─── 3. Frame-Buffer leeren ───
        self._frame_buffer = []
        self._current_episode = None

    def _record_frame(self, detections: List[Dict], action: str):
        """Zeichnet einen Frame auf – inkl. Speed-Level, spd/acc und LED-Brightness."""
        if not self._current_episode:
            return

        state = self._arm_state
        now = time.time()
        spd_cfg = self._current_speed  # Aktuelle Geschwindigkeits-Konfiguration

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
            "speed_level": self._speed_level,
            "servo_spd": spd_cfg["spd"],       # <-- NEU: exakte Servo-Geschwindigkeit
            "servo_acc": spd_cfg["acc"],       # <-- NEU: exakte Servo-Beschleunigung
            "led_brightness": self._led_brightness,
            "detections": detections,
            "action": action,
            "rel_to_target": rel_to_target,
        }

        self._current_episode.frames.append(frame_data)

    # ─── Replay ───────────────────────────────────────────────────────────────

    def _replay_last_episode(self):
        """Spielt die letzte aufgezeichnete Episode 1:1 ab."""
        if self._recording:
            print("  [!] Kann nicht abspielen während Aufnahme läuft!")
            return

        if not self._replay_engine:
            print("  [!] Replay Engine nicht initialisiert!")
            return

        # Finde letzte Episode (JSON oder Parquet)
        episode_path = self._find_last_episode()
        if not episode_path:
            print("  [!] Keine Episode zum Abspielen gefunden!")
            return

        print(f"\n  ▶ Starte Replay: {episode_path.name}")
        self._replay_engine.replay_episode(episode_path, self._window_name)

    def _find_last_episode(self) -> Optional[Path]:
        """Findet die zuletzt gespeicherte Episode."""
        # Zuerst im Hauptverzeichnis (JSON)
        json_episodes = sorted(self._output_dir.glob("episode_*.json"))
        if json_episodes:
            return json_episodes[-1]

        # Dann im LeRobot-Verzeichnis
        lerobot_data = self._output_dir / "lerobot_dataset" / "data" / "chunk-000"
        if lerobot_data.exists():
            parquet_episodes = sorted(lerobot_data.glob("episode_*.parquet"))
            if parquet_episodes:
                return parquet_episodes[-1]
            json_episodes = sorted(lerobot_data.glob("episode_*.json"))
            if json_episodes:
                return json_episodes[-1]

        return None

    # ─── Annotation ───────────────────────────────────────────────────────────

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
        spd_label = self._current_speed["label"]
        led_pct = int(self._led_brightness / 255 * 100)
        status_line = (f"B:{state.base_deg:.0f} S:{state.shoulder_deg:.0f} "
                       f"E:{state.elbow_deg:.0f} H:{state.hand_deg:.0f} "
                       f"G:{'O' if state.gripper_open else 'C'} "
                       f"LED:{self._led_brightness}({led_pct}%) "
                       f"FPS:{fps:.0f} | SPD:{spd_label}")

        cv2.putText(frame, status_line, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

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
            cv2.putText(frame, f"Episodes: {self._episode_count} | R=Record P=Replay Q=Quit",
                        (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ─── Steuerungs-Overlay ───
        help_y = h_f - 40
        cv2.putText(frame, "Arrows=Base/Shoulder  W/S=Elbow  A/D=Hand  O/C=Grip  +/-=Speed  1-5=Level",
                    (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)
        cv2.putText(frame, "LED: Shift+1=OFF  Shift+2..6=Brightness Steps  |  P=Replay last episode",
                    (10, help_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

    # ─── Replay ───────────────────────────────────────────────────────────────

    def _replay_last_episode(self):
        """Spielt die letzte aufgezeichnete Episode 1:1 ab."""
        if self._recording:
            print("  [!] Kann nicht abspielen während Aufnahme läuft!")
            return

        # Finde letzte Episode
        episode_path = self._find_last_episode()
        if not episode_path:
            print("  [!] Keine Episode zum Abspielen gefunden!")
            return

        print(f"\n  ▶ Starte Replay: {episode_path.name}")
        self._do_replay(episode_path)

    def _find_last_episode(self) -> Optional[Path]:
        """Findet die zuletzt gespeicherte Episode."""
        # Zuerst im Hauptverzeichnis (JSON)
        json_episodes = sorted(self._output_dir.glob("episode_*.json"))
        if json_episodes:
            return json_episodes[-1]

        # Dann im LeRobot-Verzeichnis
        lerobot_data = self._output_dir / "lerobot_dataset" / "data" / "chunk-000"
        if lerobot_data.exists():
            parquet_episodes = sorted(lerobot_data.glob("episode_*.parquet"))
            if parquet_episodes:
                return parquet_episodes[-1]

        return None

    def _do_replay(self, episode_path: Path):
        """
        Spielt eine Episode 1:1 ab – exakt gleiche Gelenkwinkel, gleiches Timing,
        gleiche Servo-Geschwindigkeit und Beschleunigung wie bei der Aufnahme.
        """
        # Lade Episode
        with open(episode_path, 'r') as f:
            data = json.load(f)

        frames = data.get("frames", [])
        if not frames:
            print("  [!] Episode hat keine Frames!")
            return

        num_frames = len(frames)
        duration = data.get("duration_s", 0)
        print(f"    {num_frames} Frames | {duration:.1f}s Dauer")
        print(f"    Q = Abbrechen")

        # Tracking für Zustandsänderungen
        last_gripper_state = None
        last_led_brightness = None

        start_time = time.time()

        for i, frame_data in enumerate(frames):
            # Timing einhalten
            target_time = frame_data.get("timestamp", 0)
            elapsed = time.time() - start_time
            wait_time = target_time - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

            # Gelenkwinkel setzen
            arm = frame_data.get("arm_state", {})
            if arm:
                self._arm_state.base_deg = arm.get("base_deg", self._arm_state.base_deg)
                self._arm_state.shoulder_deg = arm.get("shoulder_deg", self._arm_state.shoulder_deg)
                self._arm_state.elbow_deg = arm.get("elbow_deg", self._arm_state.elbow_deg)
                self._arm_state.hand_deg = arm.get("hand_deg", self._arm_state.hand_deg)
                self._arm_state.gripper_open = arm.get("gripper_open", self._arm_state.gripper_open)

                # Aufgezeichnete Servo-Geschwindigkeit und Beschleunigung verwenden
                replay_spd = frame_data.get("servo_spd", 100)  # Fallback für alte Aufnahmen
                replay_acc = frame_data.get("servo_acc", 200)   # Fallback für alte Aufnahmen

                cmd = {
                    "T": 122,
                    "b": round(self._arm_state.base_deg, 1),
                    "s": round(self._arm_state.shoulder_deg, 1),
                    "e": round(self._arm_state.elbow_deg, 1),
                    "h": round(self._arm_state.hand_deg, 1),
                    "spd": replay_spd,   # <-- EXAKT wie bei Aufnahme
                    "acc": replay_acc    # <-- EXAKT wie bei Aufnahme
                }
                self._arm._send_nowait(cmd)

                # Gripper NUR bei Zustandsänderung senden
                gripper_open = arm.get("gripper_open", True)
                if gripper_open != last_gripper_state:
                    if gripper_open:
                        self._arm.gripper_open()
                    else:
                        self._arm.gripper_close()
                    last_gripper_state = gripper_open

            # LED-Brightness wiederherstellen (nur bei Änderung)
            led_brightness = frame_data.get("led_brightness", None)
            if led_brightness is not None and led_brightness != last_led_brightness:
                self._arm.set_led(led_brightness)
                self._led_brightness = led_brightness
                last_led_brightness = led_brightness

            # Speed-Level aus Aufnahme anzeigen (informativ)
            recorded_speed_level = frame_data.get("speed_level", None)

            # Live-Anzeige
            frame = self._get_frame()
            if frame is not None:
                h_f, w_f = frame.shape[:2]
                progress = (i + 1) / num_frames

                # Replay-Indikator
                cv2.putText(frame, f"REPLAY [{i+1}/{num_frames}]", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                # Progress-Bar
                bar_w = w_f - 20
                cv2.rectangle(frame, (10, 45), (10 + bar_w, 55), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 45), (10 + int(bar_w * progress), 55), (0, 200, 0), -1)

                # Arm-State
                gripper_char = 'O' if arm.get('gripper_open', True) else 'C'
                state_str = (f"B:{arm.get('base_deg', 0):.0f} S:{arm.get('shoulder_deg', 0):.0f} "
                            f"E:{arm.get('elbow_deg', 0):.0f} H:{arm.get('hand_deg', 0):.0f} "
                            f"G:{gripper_char}")
                cv2.putText(frame, state_str, (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Speed + LED Info (jetzt mit tatsächlichen spd/acc Werten)
                spd_label = ""
                if recorded_speed_level is not None and 0 <= recorded_speed_level < len(self.SPEED_LEVELS):
                    spd_label = self.SPEED_LEVELS[recorded_speed_level]["label"]
                replay_spd_display = frame_data.get("servo_spd", "?")
                replay_acc_display = frame_data.get("servo_acc", "?")
                led_info = f"LED:{led_brightness if led_brightness is not None else '?'}"
                cv2.putText(frame, f"Speed: {spd_label} (spd:{replay_spd_display} acc:{replay_acc_display}) | {led_info}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

                # Action
                action = frame_data.get("action", "")
                if action:
                    cv2.putText(frame, f"Action: {action}", (10, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Detections aus der Aufnahme anzeigen
                dets = frame_data.get("detections", [])
                for det in dets:
                    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                    label = f"{det['class']} {det['confidence']:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, f"[rec] {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                # Hinweis
                cv2.putText(frame, "Q = Abbrechen", (10, h_f - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                cv2.imshow(self._window_name, frame)

            # Abbruch mit Q
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("  [!] Replay abgebrochen")
                return

        actual_duration = time.time() - start_time
        print(f"  ■ REPLAY BEENDET ({actual_duration:.1f}s)")

    # ─── LeRobot Speicherung ─────────────────────────────────────────────────

    def _save_lerobot_format(self, episode: Episode, frame_images: List[np.ndarray] = None):
        """
        Speichert Episode zusätzlich im LeRobot-kompatiblen Format.
        
        Struktur:
          lerobot_dataset/
            meta/
              info.json
              episodes.jsonl
              stats.json
              tasks.jsonl
            data/
              chunk-000/
                episode_000000.parquet (oder .json als Fallback)
            videos/
              chunk-000/
                observation.images.top/
                  episode_000000.mp4
        """
        dataset_dir = self._output_dir / "lerobot_dataset"
        meta_dir = dataset_dir / "meta"
        data_dir = dataset_dir / "data" / "chunk-000"
        video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.top"

        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        ep_idx = episode.episode_id - 1  # 0-basiert

        # ─── Frames in LeRobot-Spaltenformat konvertieren ───
        records = []
        for i, frame_data in enumerate(episode.frames):
            arm = frame_data["arm_state"]

            # Action = nächster State (oder gleicher wenn letzter Frame)
            if i < len(episode.frames) - 1:
                next_arm = episode.frames[i + 1]["arm_state"]
                action_vec = [
                    next_arm["base_deg"],
                    next_arm["shoulder_deg"],
                    next_arm["elbow_deg"],
                    next_arm["hand_deg"],
                    0.0 if next_arm["gripper_open"] else 1.0
                ]
            else:
                action_vec = [
                    arm["base_deg"],
                    arm["shoulder_deg"],
                    arm["elbow_deg"],
                    arm["hand_deg"],
                    0.0 if arm["gripper_open"] else 1.0
                ]

            record = {
                "observation.state": [
                    arm["base_deg"],
                    arm["shoulder_deg"],
                    arm["elbow_deg"],
                    arm["hand_deg"]
                ],
                "observation.gripper": [0.0 if arm["gripper_open"] else 1.0],
                "action": action_vec,
                "timestamp": frame_data["timestamp"],
                "episode_index": ep_idx,
                "frame_index": i,
                "task_index": 0,
                "action_label": frame_data.get("action", ""),
            }
            records.append(record)

        # ─── Als JSON speichern (Parquet als optionales Upgrade) ───
        episode_file = data_dir / f"episode_{ep_idx:06d}.json"
        with open(episode_file, 'w') as f:
            json.dump(records, f, indent=2)

        # ─── Video speichern (wenn Frames vorhanden) ───
        if frame_images and len(frame_images) > 0:
            video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
            h, w = frame_images[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
            for img in frame_images:
                writer.write(img)
            writer.release()
            print(f"    📹 Video: {video_path} ({len(frame_images)} frames)")

        # ─── Episode-Index aktualisieren ───
        episodes_file = meta_dir / "episodes.jsonl"
        episode_entry = {
            "episode_index": ep_idx,
            "task_index": 0,
            "task": "Pick up target object with robot arm",
            "length": len(episode.frames),
            "target_class": episode.target_class,
            "success": episode.success,
            "duration_s": episode.end_time - episode.start_time
        }
        with open(episodes_file, 'a') as f:
            f.write(json.dumps(episode_entry) + "\n")

        # ─── Tasks-Datei ───
        tasks_file = meta_dir / "tasks.jsonl"
        if not tasks_file.exists():
            with open(tasks_file, 'w') as f:
                f.write(json.dumps({"task_index": 0, "task": "Pick up target object with robot arm"}) + "\n")

        # ─── Info.json aktualisieren ───
        info_file = meta_dir / "info.json"
        # Zähle alle Episoden
        total_episodes = ep_idx + 1
        total_frames = sum(1 for _ in open(episodes_file)) if episodes_file.exists() else 0

        info = {
            "codebase_version": "v2.1",
            "robot_type": "roarm_m2s",
            "total_episodes": total_episodes,
            "total_frames": len(records),
            "fps": 30.0,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [4],
                    "names": ["base_deg", "shoulder_deg", "elbow_deg", "hand_deg"]
                },
                "observation.gripper": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["gripper_position"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [5],
                    "names": ["base_deg", "shoulder_deg", "elbow_deg", "hand_deg", "gripper"]
                },
                "observation.images.top": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"]
                }
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.json",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "chunks_size": 1000
        }
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        # ─── Stats aktualisieren ───
        if records:
            states = np.array([r["observation.state"] for r in records])
            actions = np.array([r["action"] for r in records])
            stats = {
                "observation.state": {
                    "min": states.min(axis=0).tolist(),
                    "max": states.max(axis=0).tolist(),
                    "mean": states.mean(axis=0).tolist(),
                    "std": states.std(axis=0).tolist()
                },
                "action": {
                    "min": actions.min(axis=0).tolist(),
                    "max": actions.max(axis=0).tolist(),
                    "mean": actions.mean(axis=0).tolist(),
                    "std": actions.std(axis=0).tolist()
                }
            }
            stats_file = meta_dir / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

        print(f"    📦 LeRobot: {episode_file}")

    # ─── Shutdown ─────────────────────────────────────────────────────────────

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
        print("  ✔ Fertig")

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

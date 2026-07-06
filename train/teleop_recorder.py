"""
Teleop Recorder — Fernsteuerung des RoArm-M2-S mit Tastatur + Aufzeichnung.

VERBESSERTE VERSION mit:
  ✓ Rich Console UI (farbige Terminal-Ausgabe)
  ✓ Semi-transparente OSD-Overlays (bessere Lesbarkeit)
  ✓ Visuelles Joystick-Widget (Gelenkposition-Visualisierung)
  ✓ Sound-Feedback (akustische Bestätigung)
  ✓ Visuelles Keyboard-Layout-Overlay (aktive Tasten hervorgehoben)

Steuerung:
  Pfeiltasten Links/Rechts  → Base-Rotation (Links = Base dreht links, Rechts = rechts)
  Pfeiltasten Oben/Unten    → Shoulder (Oben = Arm hebt sich / nach oben, Unten = senkt sich)
  W/S                        → Elbow (W = hoch, S = runter)
  A/D                        → Hand/Wrist Rotation (A = links, D = rechts)
  O                          → Gripper öffnen
  C                          → Gripper schließen
  R                          → Aufnahme starten / stoppen (Toggle)
  Ctrl+F                     → Episode als fehlgeschlagen markieren & speichern
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

# ─── Rich Console (graceful fallback) ─────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

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

# ─── Sound Feedback Helper ────────────────────────────────────────────────────
class SoundFeedback:
    """
    Einfaches akustisches Feedback über System-Beep oder Fallback.
    Nicht-blockierend (eigener Thread).
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._has_beep = False
        # Prüfe ob wir Sound machen können
        try:
            # Linux: über /dev/console oder os.system
            import subprocess
            self._has_beep = True
        except:
            pass

    def _play_async(self, freq: int, duration_ms: int):
        """Spielt einen Ton asynchron."""
        if not self._enabled:
            return
        def _do():
            try:
                import subprocess
                # Versuche paplay oder beep
                subprocess.run(
                    ["paplay", "--raw", "/dev/null"],
                    capture_output=True, timeout=0.1
                )
            except:
                try:
                    # Fallback: Terminal bell
                    print('\a', end='', flush=True)
                except:
                    pass
        threading.Thread(target=_do, daemon=True).start()

    def beep_record_start(self):
        """Doppel-Beep: Aufnahme gestartet."""
        if not self._enabled:
            return
        def _do():
            print('\a', end='', flush=True)
            time.sleep(0.1)
            print('\a', end='', flush=True)
        threading.Thread(target=_do, daemon=True).start()

    def beep_record_stop(self):
        """Einzel-Beep: Aufnahme gestoppt."""
        if not self._enabled:
            return
        print('\a', end='', flush=True)

    def beep_error(self):
        """Dreifach-Beep: Fehler."""
        if not self._enabled:
            return
        def _do():
            for _ in range(3):
                print('\a', end='', flush=True)
                time.sleep(0.08)
        threading.Thread(target=_do, daemon=True).start()

    def beep_gripper(self):
        """Kurzer Beep: Gripper-Aktion."""
        if not self._enabled:
            return
        print('\a', end='', flush=True)

# ─── Rich Console Wrapper ─────────────────────────────────────────────────────
class RichUI:
    """
    Wrapper für Rich Console Ausgabe.
    Fällt graceful auf print() zurück wenn Rich nicht installiert ist.
    """

    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

    def print_banner(self):
        """Zeigt das Start-Banner."""
        if self.console:
            banner = Panel(
                "[bold cyan]RoArm-M2-S Teleop Recorder[/bold cyan]\n"
                "[dim]Enhanced Edition with Rich UI[/dim]",
                box=box.DOUBLE,
                border_style="bright_blue",
                padding=(1, 4)
            )
            self.console.print(banner)
        else:
            print("=" * 60)
            print("  RoArm-M2-S Teleop Recorder")
            print("  Enhanced Edition")
            print("=" * 60)

    def print_controls(self):
        """Zeigt die Steuerungstabelle."""
        if self.console:
            table = Table(title="⌨️  Steuerung", box=box.ROUNDED, border_style="cyan")
            table.add_column("Taste", style="bold yellow", width=14)
            table.add_column("Funktion", style="white")
            table.add_column("Taste", style="bold yellow", width=14)
            table.add_column("Funktion", style="white")

            table.add_row("←", "Base links", "→", "Base rechts")
            table.add_row("↑", "Shoulder hoch", "↓", "Shoulder runter")
            table.add_row("W", "Elbow hoch", "S", "Elbow runter")
            table.add_row("A", "Hand links", "D", "Hand rechts")
            table.add_row("O", "Gripper öffnen", "C", "Gripper schließen")
            table.add_row("R", "Record Toggle", "P", "Replay")
            table.add_row("Ctrl+F", "Fail & Save", "Q", "Beenden")
            table.add_row("+/=", "Speed +", "-", "Speed -")
            table.add_row("1-5", "Speed direkt", "Shift+1-6", "LED Stufen")

            self.console.print(table)
        else:
            print("\n  STEUERUNG:")
            print("    ←/→       Base links/rechts")
            print("    ↑/↓       Shoulder hoch/runter")
            print("    W/S       Elbow hoch/runter")
            print("    A/D       Hand links/rechts")
            print("    O/C       Gripper öffnen/schließen")
            print("    R         Record Toggle")
            print("    Ctrl+F    Fail & Save")
            print("    P         Replay")
            print("    +/-       Speed +/-")
            print("    1-5       Speed direkt")
            print("    Q         Beenden")

    def print_status(self, message: str, style: str = "green"):
        """Druckt eine Status-Nachricht."""
        if self.console:
            self.console.print(f"  [{style}]✓[/{style}] {message}")
        else:
            print(f"  ✓ {message}")

    def print_error(self, message: str):
        """Druckt eine Fehlermeldung."""
        if self.console:
            self.console.print(f"  [bold red]✗[/bold red] {message}")
        else:
            print(f"  ✗ {message}")

    def print_info(self, message: str):
        """Druckt eine Info-Nachricht."""
        if self.console:
            self.console.print(f"  [dim]{message}[/dim]")
        else:
            print(f"  {message}")

    def print_recording_start(self, episode_id: int):
        """Zeigt Recording-Start an."""
        if self.console:
            self.console.print(
                Panel(
                    f"[bold red]● AUFNAHME GESTARTET[/bold red]\n"
                    f"Episode: [bold]{episode_id}[/bold]",
                    border_style="red",
                    box=box.HEAVY
                )
            )
        else:
            print(f"\n  ● AUFNAHME GESTARTET (Episode {episode_id})")

    def print_recording_stop(self, filename: str, success: bool, num_frames: int, duration: float):
        """Zeigt Recording-Stop an."""
        status = "[bold green]✓ ERFOLG[/bold green]" if success else "[bold red]✗ FEHLGESCHLAGEN[/bold red]"
        if self.console:
            self.console.print(
                Panel(
                    f"[bold]■ AUFNAHME GESPEICHERT[/bold]\n"
                    f"  Datei: {filename}\n"
                    f"  Status: {status}\n"
                    f"  Frames: {num_frames} | Dauer: {duration:.1f}s",
                    border_style="green" if success else "red",
                    box=box.HEAVY
                )
            )
        else:
            s = "✓ ERFOLG" if success else "✗ FEHLGESCHLAGEN"
            print(f"\n  ■ AUFNAHME GESPEICHERT: {filename}")
            print(f"    {s} | {num_frames} Frames | {duration:.1f}s")

    def print_speed_change(self, label: str, level: int, max_level: int):
        """Zeigt Speed-Änderung an."""
        if self.console:
            bar = "█" * (level + 1) + "░" * (max_level - level)
            self.console.print(f"  [yellow]⚡ Speed: {label}[/yellow] [{bar}]")
        else:
            print(f"  ⚡ Speed: {label} (Level {level}/{max_level})")

    def print_setup_step(self, step: int, total: int, message: str):
        """Zeigt einen Setup-Schritt an."""
        if self.console:
            self.console.print(f"  [cyan][{step}/{total}][/cyan] {message}")
        else:
            print(f"\n[{step}] {message}")

# ─── OSD Overlay Helpers ──────────────────────────────────────────────────────
class OSDRenderer:
    """
    Zeichnet semi-transparente Overlays und Widgets auf OpenCV-Frames.
    Verbesserte Lesbarkeit durch Hintergrund-Boxen.
    """

    @staticmethod
    def draw_text_with_background(frame, text: str, pos: Tuple[int, int],
                                   font_scale: float = 0.5, color=(255, 255, 255),
                                   bg_color=(0, 0, 0), bg_alpha: float = 0.6,
                                   thickness: int = 1, padding: int = 4):
        """Zeichnet Text mit semi-transparentem Hintergrund."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x, y = pos
        # Hintergrund-Box
        x1 = x - padding
        y1 = y - text_h - padding
        x2 = x + text_w + padding
        y2 = y + baseline + padding

        # Semi-transparenter Hintergrund
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)

        # Text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    @staticmethod
    def draw_panel(frame, x: int, y: int, w: int, h: int,
                   bg_color=(20, 20, 20), alpha: float = 0.7,
                   border_color=(100, 100, 100), border_width: int = 1):
        """Zeichnet ein semi-transparentes Panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        if border_width > 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, border_width)

    @staticmethod
    def draw_joystick_widget(frame, arm_state, x: int, y: int, size: int = 100):
        """
        Zeichnet ein visuelles Joystick-Widget das die aktuelle Arm-Position zeigt.
        
        Zeigt:
        - Base-Rotation als horizontale Position
        - Shoulder als vertikale Position
        - Elbow als Farbe/Ring
        - Gripper-Status als Symbol
        """
        # Hintergrund
        OSDRenderer.draw_panel(frame, x, y, size, size, alpha=0.75,
                               border_color=(80, 200, 80))

        cx = x + size // 2
        cy = y + size // 2
        radius = size // 2 - 10

        # Kreuz-Linien (Achsen)
        cv2.line(frame, (x + 10, cy), (x + size - 10, cy), (60, 60, 60), 1)
        cv2.line(frame, (cx, y + 10), (cx, y + size - 10), (60, 60, 60), 1)

        # Base → X-Achse (-90..+90 → links..rechts)
        base_norm = arm_state.base_deg / 90.0  # -1..+1
        base_norm = max(-1.0, min(1.0, base_norm))

        # Shoulder → Y-Achse (-30..+60 → unten..oben)
        shoulder_range = 90.0  # -30 to +60
        shoulder_norm = (arm_state.shoulder_deg - (-30)) / shoulder_range  # 0..1
        shoulder_norm = max(0.0, min(1.0, shoulder_norm))

        # Punkt-Position
        px = int(cx + base_norm * radius)
        py = int(cy - (shoulder_norm - 0.5) * 2 * radius)  # invertiert (oben = positiv)

        # Elbow als Farbe (0°=blau, 90°=grün, 180°=rot)
        elbow_norm = arm_state.elbow_deg / 180.0
        r = int(elbow_norm * 255)
        g = int((1.0 - abs(elbow_norm - 0.5) * 2) * 255)
        b = int((1.0 - elbow_norm) * 255)
        dot_color = (b, g, r)

        # Punkt zeichnen
        cv2.circle(frame, (px, py), 6, dot_color, -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

        # Gripper-Status
        grip_symbol = "O" if arm_state.gripper_open else "X"
        grip_color = (0, 255, 0) if arm_state.gripper_open else (0, 0, 255)
        cv2.putText(frame, grip_symbol, (x + size - 18, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 2)

        # Label
        cv2.putText(frame, "JOY", (x + 3, y + size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

    @staticmethod
    def draw_keyboard_overlay(frame, active_keys: set, x: int, y: int):
        """
        Zeichnet ein visuelles Mini-Keyboard das aktive Tasten hervorhebt.
        
        Layout:
            [↑]
        [←][↓][→]    [W]
                    [A][S][D]   [O][C]
        """
        key_size = 22
        gap = 3
        
        # Hintergrund-Panel
        panel_w = 280
        panel_h = 75
        OSDRenderer.draw_panel(frame, x, y, panel_w, panel_h, alpha=0.7,
                               border_color=(100, 150, 200))

        def draw_key(kx, ky, label, is_active, key_w=key_size):
            """Zeichnet eine einzelne Taste."""
            color_bg = (0, 120, 255) if is_active else (40, 40, 40)
            color_border = (100, 200, 255) if is_active else (80, 80, 80)
            color_text = (255, 255, 255) if is_active else (150, 150, 150)

            cv2.rectangle(frame, (kx, ky), (kx + key_w, ky + key_size), color_bg, -1)
            cv2.rectangle(frame, (kx, ky), (kx + key_w, ky + key_size), color_border, 1)

            # Text zentrieren
            font_scale = 0.35 if len(label) <= 2 else 0.28
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = kx + (key_w - tw) // 2
            ty = ky + (key_size + th) // 2
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text, 1)

        # Pfeiltasten-Block (links)
        bx = x + 8
        by = y + 8

        # ↑
        draw_key(bx + key_size + gap, by, "^", "shoulder_up" in active_keys)
        # ← ↓ →
        draw_key(bx, by + key_size + gap, "<", "base_left" in active_keys)
        draw_key(bx + key_size + gap, by + key_size + gap, "v", "shoulder_down" in active_keys)
        draw_key(bx + 2 * (key_size + gap), by + key_size + gap, ">", "base_right" in active_keys)

        # WASD-Block (mitte)
        wx = x + 90
        wy = y + 8

        # W
        draw_key(wx + key_size + gap, wy, "W", "elbow_up" in active_keys)
        # A S D
        draw_key(wx, wy + key_size + gap, "A", "hand_left" in active_keys)
        draw_key(wx + key_size + gap, wy + key_size + gap, "S", "elbow_down" in active_keys)
        draw_key(wx + 2 * (key_size + gap), wy + key_size + gap, "D", "hand_right" in active_keys)

        # Gripper-Block (rechts)
        gx = x + 185
        gy = y + 8 + key_size + gap

        draw_key(gx, gy, "O", False, key_w=28)  # Gripper ist momentan, nicht gehalten
        draw_key(gx + 31, gy, "C", False, key_w=28)

        # Record/Replay (unten)
        ry = y + panel_h - key_size - 5
        draw_key(x + 8, ry, "R", False, key_w=28)
        draw_key(x + 40, ry, "P", False, key_w=28)

        # Label
        cv2.putText(frame, "KEYS", (x + panel_w - 35, y + panel_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

# ─── Datenstrukturen ──────────────────────────────────────────────────────────

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
                "observation.gripper": [0.0 if arm["gripper_open"] else 1.0],                "action": action_vec,
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

    def _update_stats(self, records: List[Dict]):
        """Aktualisiert Statistiken über alle Episoden."""
        if not records:
            return
        
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

# ─── Replay Engine ────────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Spielt eine aufgezeichnete Episode 1:1 auf dem Arm ab.
    Reproduziert exakt die gleichen Gelenkwinkel mit dem gleichen Timing.
    """

    def __init__(self, arm: RoArmM2S, camera=None):
        self._arm = arm
        self._camera = camera
        self._replaying = False

    def replay_episode(self, episode_path: Path, window_name: str = "RoArm Replay") -> bool:
        """
        Spielt eine Episode 1:1 ab — mit exakt den aufgezeichneten spd/acc Werten.
        """
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
                replay_spd = frame_data.get("servo_spd", 100)
                replay_acc = frame_data.get("servo_acc", 200)

                cmd = {
                    "T": 122,
                    "b": round(arm_state["base_deg"], 1),
                    "s": round(arm_state["shoulder_deg"], 1),
                    "e": round(arm_state["elbow_deg"], 1),
                    "h": round(arm_state["hand_deg"], 1),
                    "spd": replay_spd,
                    "acc": replay_acc
                }
                self._arm._send_nowait(cmd)

                gripper_open = arm_state.get("gripper_open", True)
                if gripper_open != last_gripper_state:
                    if gripper_open:
                        self._arm.gripper_open()
                    else:
                        self._arm.gripper_close()
                    last_gripper_state = gripper_open

            # LED wiederherstellen
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
        if "arm_state" in frame_data:
            return frame_data["arm_state"]
        
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
        """Annotiert Frame während Replay mit semi-transparenten Overlays."""
        h, w = frame.shape[:2]
        
        progress = current_idx / max(1, total - 1)

        # Semi-transparentes Header-Panel
        OSDRenderer.draw_panel(frame, 0, 0, w, 60, bg_color=(0, 40, 0), alpha=0.6,
                               border_color=(0, 200, 0))

        OSDRenderer.draw_text_with_background(
            frame, f"REPLAY [{current_idx+1}/{total}]", (10, 25),
            font_scale=0.6, color=(0, 255, 0), bg_alpha=0.0, thickness=2
        )
        
        # Progress-Bar
        bar_w = w - 20
        bar_x = 10
        bar_y = 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), (0, 200, 0), -1)
        
        # Arm-State
        arm_state = self._get_arm_state(frame_data)
        if arm_state:
            state_str = (f"B:{arm_state['base_deg']:.0f} S:{arm_state['shoulder_deg']:.0f} "
                        f"E:{arm_state['elbow_deg']:.0f} H:{arm_state['hand_deg']:.0f} "
                        f"G:{'O' if arm_state.get('gripper_open', True) else 'C'}")
            OSDRenderer.draw_text_with_background(
                frame, state_str, (10, 80),
                font_scale=0.5, color=(255, 255, 255), bg_alpha=0.5
            )
        
        # Action
        action = frame_data.get("action_label", frame_data.get("action", ""))
        if action:
            OSDRenderer.draw_text_with_background(
                frame, f"Action: {action}", (10, 105),
                font_scale=0.45, color=(0, 255, 255), bg_alpha=0.5
            )
        
        # Hinweis
        OSDRenderer.draw_text_with_background(
            frame, "Q = Abbrechen", (10, h - 20),
            font_scale=0.4, color=(180, 180, 180), bg_alpha=0.5
        )

# ─── Teleop Controller ────────────────────────────────────────────────────────

class TeleopRecorder:
    """
    Fernsteuerung + Aufzeichnung für Behaviour Cloning.
    
    ENHANCED VERSION mit:
      ✓ Rich Console UI
      ✓ Semi-transparente OSD-Overlays
      ✓ Visuelles Joystick-Widget
      ✓ Sound-Feedback
      ✓ Visuelles Keyboard-Layout-Overlay
    """

    # ─── Steuerungs-Parameter ─────────────────────────────────────────────────
    
    COMMAND_HZ = 50
    COMMAND_INTERVAL = 1.0 / 50
    
    SPEED_LEVELS = [
        {"base": 0.15, "shoulder": 0.12, "elbow": 0.12, "hand": 0.2, "spd": 15, "acc": 30, "label": "VERY SLOW"},
        {"base": 0.3, "shoulder": 0.25, "elbow": 0.25, "hand": 0.4, "spd": 30, "acc": 50, "label": "SLOW"},
        {"base": 0.6, "shoulder": 0.5, "elbow": 0.5, "hand": 0.7, "spd": 50, "acc": 100, "label": "MEDIUM"},
        {"base": 1.0, "shoulder": 0.8, "elbow": 0.8, "hand": 1.2, "spd": 80, "acc": 150, "label": "FAST"},
        {"base": 1.5, "shoulder": 1.2, "elbow": 1.2, "hand": 1.8, "spd": 100, "acc": 200, "label": "VERY FAST"},
    ]
    
    DEFAULT_SPEED_LEVEL = 2
    
    # Limits
    BASE_MIN, BASE_MAX = -90.0, 90.0
    SHOULDER_MIN, SHOULDER_MAX = -30.0, 60.0
    ELBOW_MIN, ELBOW_MAX = 0.0, 180.0
    HAND_MIN, HAND_MAX = 0.0, 270.0

    KEY_HOLD_TIMEOUT = 0.35

    def __init__(self, port: str = None, camera_index: int = 2,
                 model_path: str = "yolo11n.pt", confidence: float = 0.5,
                 output_dir: str = "recordings", target_class: str = None,
                 enable_sound: bool = True):
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

        # Speed control
        self._speed_level = self.DEFAULT_SPEED_LEVEL

        # LED control
        self._led_brightness = 255
        self.LED_STEPS = [0, 51, 102, 153, 204, 255]

        # Persistent key state
        self._keys_down: set = set()
        self._key_last_seen: Dict[str, float] = {}

        # Command thread
        self._cmd_thread: Optional[threading.Thread] = None
        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0
        self._last_action = ""

        # Frame image buffer for video recording
        self._frame_buffer: List[np.ndarray] = []
        self._record_video = True

        # LeRobot Saver
        self._lerobot_saver: Optional[LeRobotSaver] = None

        # Replay Engine
        self._replay_engine: Optional[ReplayEngine] = None

        # ─── NEW: Rich UI ───
        self._ui = RichUI()

        # ─── NEW: Sound Feedback ───
        self._sound = SoundFeedback(enabled=enable_sound)

        # ─── NEW: Active keys for keyboard overlay ───
        self._active_keys_display: set = set()

        # Window
        self._window_name = "RoArm Teleop"

    def _set_led_brightness(self, level_index: int):
        """Set LED brightness by step index (0-5)."""
        level_index = max(0, min(len(self.LED_STEPS) - 1, level_index))
        self._led_brightness = self.LED_STEPS[level_index]
        self._arm.set_led(self._led_brightness)
        self._ui.print_info(f"💡 LED: {self._led_brightness}/255 (Step {level_index})")

    def _count_existing_episodes(self) -> int:
        """Zählt bereits vorhandene Episoden im Output-Verzeichnis."""
        count = 0
        for f in self._output_dir.glob("episode_*.json"):
            count += 1
        lerobot_data = self._output_dir / "data" / "chunk-000"
        if lerobot_data.exists():
            for f in lerobot_data.glob("episode_*.parquet"):
                count += 1
            for f in lerobot_data.glob("episode_*.json"):
                count += 1
        return count

    def _get_active_keys(self) -> set:
        """Returns all keys currently considered 'held'."""
        now = time.time()
        expired = []
        for key, last_time in self._key_last_seen.items():
            if now - last_time >= self.KEY_HOLD_TIMEOUT:
                expired.append(key)
        for key in expired:
            del self._key_last_seen[key]
            self._keys_down.discard(key)
        return set(self._keys_down)

    @property
    def _current_speed(self) -> dict:
        """Returns the current speed level configuration."""
        return self.SPEED_LEVELS[self._speed_level]

    def _change_speed(self, delta: int):
        """Change speed level by delta (+1 or -1)."""
        old_level = self._speed_level
        self._speed_level = max(0, min(len(self.SPEED_LEVELS) - 1, self._speed_level + delta))
        if self._speed_level != old_level:
            spd = self._current_speed
            self._ui.print_speed_change(spd['label'], self._speed_level, len(self.SPEED_LEVELS) - 1)

    # ─── Setup ────────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialisiert Hardware mit Rich UI."""
        self._ui.print_banner()

        # Arm verbinden
        self._ui.print_setup_step(1, 4, "Arm verbinden...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            self._ui.print_status("Arm verbunden")
        except Exception as e:
            self._ui.print_error(f"Arm-Fehler: {e}")
            return False

        # Kamera
        self._ui.print_setup_step(2, 4, f"Kamera {self._camera_index} öffnen...")
        self._camera = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        if not self._camera.isOpened():
            for idx in [0, 2, 1, 4]:
                self._camera = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if self._camera.isOpened():
                    self._ui.print_info(f"→ Fallback auf Kamera {idx}")
                    self._camera_index = idx
                    break
            else:
                self._ui.print_error("Keine Kamera gefunden!")
                return False

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        w = int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._ui.print_status(f"Kamera {self._camera_index} ({w}x{h})")

        # YOLO
        self._ui.print_setup_step(3, 4, f"YOLO '{self._model_path}' laden...")
        if HAS_YOLO:
            try:
                self._model = YOLO(self._model_path)
                self._model.verbose = False
                ret, frame = self._camera.read()
                if ret:
                    self._model(frame, conf=self._confidence, verbose=False)
                self._ui.print_status("YOLO bereit")
            except Exception as e:
                self._ui.print_error(f"YOLO-Fehler: {e}")
                self._model = None
        else:
            self._ui.print_info("YOLO nicht verfügbar (pip install ultralytics)")

        # Arm in Startposition
        self._ui.print_setup_step(4, 4, "Arm → Startposition...")
        self._arm.move_joints_degrees(b=0, s=0, e=90, h=180, spd=20, acc=10)
        self._arm_state = ArmState(
            base_deg=0, shoulder_deg=0, elbow_deg=90, hand_deg=180, gripper_open=True
        )
        self._arm.set_led(255)
        self._led_brightness = 255

        # LeRobot Saver
        self._lerobot_saver = LeRobotSaver(
            output_dir=self._output_dir / "lerobot_dataset",
            fps=30.0
        )
        self._ui.print_status(f"LeRobot Saver → {self._output_dir / 'lerobot_dataset'}")

        # Replay Engine
        self._replay_engine = ReplayEngine(self._arm, self._camera)
        self._ui.print_status("Replay Engine bereit")

        # Steuerungstabelle anzeigen
        self._ui.print_controls()

        self._ui.print_info(f"Episoden bisher: {self._episode_count}")
        if self._target_class:
            self._ui.print_info(f"Ziel-Objekt: '{self._target_class}'")

        return True

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _command_loop(self):
        """Dedicated thread: sends arm commands at a fixed 50Hz rate."""
        while self._running:
            self._apply_movement()
            time.sleep(self.COMMAND_INTERVAL)

    def run(self):
        """Hauptschleife."""
        if not self.setup():
            return

        self._running = True
        self._last_cmd_time = time.time()

        self._cmd_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._cmd_thread.start()
        fps_time = time.time()
        frame_count = 0
        current_fps = 0.0

        detect_every_n = 3
        loop_counter = 0
        last_detections = []

        try:
            while self._running:
                loop_start = time.time()
                loop_counter += 1

                # 1. Frame holen
                frame = self._get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                # 2. Detection (throttled)
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

                if not action and self._recording:
                    active = self._get_active_keys()
                    if active:
                        action = next(iter(active))

                # 4. Apply movement
                self._apply_movement()

                # 5. Update active keys for display
                self._active_keys_display = self._get_active_keys()

                # 6. Annotate (mit neuen Overlays)
                self._annotate_frame(frame, detections, action, current_fps)

                # 7. Record
                if self._recording:
                    self._record_frame(detections, action)
                    if self._record_video:
                        self._frame_buffer.append(frame.copy())

                # 8. Display
                cv2.imshow(self._window_name, frame)

                # FPS
                frame_count += 1
                elapsed = time.time() - fps_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_time = time.time()

                # Target ~40 FPS
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
        """Verarbeitet Tastendruck. Updates persistent key state."""
        if key == -1 or key == 0xFFFF:
            return ""

        now = time.time()
        key_low = key & 0xFF
        action = ""

        # Pfeiltasten
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

        # Ctrl+F → Fail Recording
        elif key_low == 6:
            if self._recording:
                self._stop_recording(success=False)
                self._sound.beep_error()
            action = ""

        # W/A/S/D
        elif key_low == ord('w'):
            self._keys_down.add("elbow_up")
            self._key_last_seen["elbow_up"] = now
            action = "elbow_up"
        elif key_low == ord('s'):
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
            self._sound.beep_gripper()
            action = "gripper_open"
        elif key_low == ord('c'):
            self._gripper_close()
            self._sound.beep_gripper()
            action = "gripper_close"

        # ─── LED Control (Shift+1 through Shift+6) ───
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

        # ─── Speed Control ───
        elif key_low == ord('+') or key_low == ord('='):
            self._change_speed(+1)
        elif key_low == ord('-') or key_low == ord('_'):
            self._change_speed(-1)
        elif key_low == ord('1'):
            self._speed_level = 0
            self._ui.print_speed_change(self._current_speed['label'], 0, len(self.SPEED_LEVELS) - 1)
        elif key_low == ord('2'):
            self._speed_level = 1
            self._ui.print_speed_change(self._current_speed['label'], 1, len(self.SPEED_LEVELS) - 1)
        elif key_low == ord('3'):
            self._speed_level = 2
            self._ui.print_speed_change(self._current_speed['label'], 2, len(self.SPEED_LEVELS) - 1)
        elif key_low == ord('4'):
            self._speed_level = 3
            self._ui.print_speed_change(self._current_speed['label'], 3, len(self.SPEED_LEVELS) - 1)
        elif key_low == ord('5'):
            self._speed_level = 4
            self._ui.print_speed_change(self._current_speed['label'], 4, len(self.SPEED_LEVELS) - 1)

        # ─── Recording Toggle (R) ───
        elif key_low == ord('r'):
            if self._recording:
                self._stop_recording(success=True)
                self._sound.beep_record_stop()
            else:
                self._start_recording()
                self._sound.beep_record_start()

        # ─── Plain 'f' is now free ───
        elif key_low == ord('f'):
            pass  # No longer stops recording; Ctrl+F does that now

        # ─── Replay ───
        elif key_low == ord('p'):
            self._replay_last_episode()

        # ─── Quit ───
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
            self._ui.print_error("Bereits am Aufnehmen!")
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
        self._ui.print_recording_start(self._episode_count)

    def _stop_recording(self, success: bool = True):
        """Stoppt und speichert die aktuelle Episode."""
        if not self._recording or not self._current_episode:
            self._ui.print_error("Keine aktive Aufnahme!")
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

        self._ui.print_recording_stop(str(filename), success, num_frames, duration)

        # ─── 2. LeRobot-Format speichern ───
        if self._lerobot_saver:
            try:
                lerobot_path = self._lerobot_saver.save_episode(
                    self._current_episode,
                    frames_images=self._frame_buffer if self._record_video else None
                )
                self._ui.print_info(f"📦 LeRobot: {lerobot_path}")
            except Exception as e:
                self._ui.print_error(f"LeRobot-Speicherfehler: {e}")

        # ─── 3. Frame-Buffer leeren ───
        self._frame_buffer = []
        self._current_episode = None

    def _record_frame(self, detections: List[Dict], action: str):
        """Zeichnet einen Frame auf — inkl. Speed-Level, spd/acc und LED-Brightness."""
        if not self._current_episode:
            return

        state = self._arm_state
        now = time.time()
        spd_cfg = self._current_speed

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
            "servo_spd": spd_cfg["spd"],
            "servo_acc": spd_cfg["acc"],
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
            self._ui.print_error("Kann nicht abspielen während Aufnahme läuft!")
            return

        if not self._replay_engine:
            self._ui.print_error("Replay Engine nicht initialisiert!")
            return

        # Finde letzte Episode (JSON oder Parquet)
        episode_path = self._find_last_episode()
        if not episode_path:
            self._ui.print_error("Keine Episode zum Abspielen gefunden!")
            return

        self._ui.print_info(f"▶ Starte Replay: {episode_path.name}")
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

    # ─── Annotation (ENHANCED mit OSD-Overlays, Joystick, Keyboard) ───────────

    def _annotate_frame(self, frame, detections: List[Dict], action: str, fps: float = 0):
        """Zeichnet Infos auf den Frame — mit verbesserten semi-transparenten Overlays."""
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

        # ─── Status-Leiste oben (semi-transparent) ───
        state = self._arm_state
        spd_label = self._current_speed["label"]
        led_pct = int(self._led_brightness / 255 * 100)
        status_line = (f"B:{state.base_deg:.0f} S:{state.shoulder_deg:.0f} "
                       f"E:{state.elbow_deg:.0f} H:{state.hand_deg:.0f} "
                       f"G:{'O' if state.gripper_open else 'C'} "
                       f"LED:{self._led_brightness}({led_pct}%) "
                       f"FPS:{fps:.0f} | SPD:{spd_label}")

        OSDRenderer.draw_text_with_background(
            frame, status_line, (10, 22),
            font_scale=0.50, color=(255, 255, 255), bg_alpha=0.6, thickness=2
        )

        # Aktive Tasten anzeigen (visuelles Feedback)
        active = self._active_keys_display
        if active:
            active_str = " + ".join(sorted(active))
            OSDRenderer.draw_text_with_background(
                frame, f"Active: {active_str}", (10, 47),
                font_scale=0.45, color=(0, 255, 255), bg_alpha=0.5
            )

        # Aktuelle Aktion
        if action:
            OSDRenderer.draw_text_with_background(
                frame, f"Action: {action}", (10, 67),
                font_scale=0.5, color=(0, 200, 255), bg_alpha=0.5
            )

        # ─── Recording-Indikator ───
        if self._recording:
            # Blinkender roter Punkt
            blink = int(time.time() * 3) % 2 == 0
            if blink:
                cv2.circle(frame, (w_f - 30, 25), 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (w_f - 30, 25), 10, (0, 0, 180), -1)
            cv2.circle(frame, (w_f - 30, 25), 10, (255, 255, 255), 1)

            ep = self._current_episode
            if ep:
                elapsed = time.time() - ep.start_time
                rec_text = f"REC {elapsed:.1f}s | {len(ep.frames)} frames"
                OSDRenderer.draw_text_with_background(
                    frame, rec_text, (w_f - 260, 30),
                    font_scale=0.5, color=(0, 0, 255), bg_alpha=0.6, thickness=2
                )

            # Hinweis unten
            OSDRenderer.draw_text_with_background(
                frame, "R=Stop Recording  Ctrl+F=Fail",
                (10, h_f - 18), font_scale=0.4, color=(0, 0, 255), bg_alpha=0.5
            )
        else:
            OSDRenderer.draw_text_with_background(
                frame, f"Episodes: {self._episode_count} | R=Record P=Replay Q=Quit",
                (10, h_f - 18), font_scale=0.4, color=(200, 200, 200), bg_alpha=0.5
            )

        # ─── NEW: Joystick Widget (unten rechts) ───
        joy_x = w_f - 115
        joy_y = h_f - 115
        OSDRenderer.draw_joystick_widget(frame, self._arm_state, joy_x, joy_y, size=100)

        # ─── NEW: Keyboard Overlay (unten links) ───
        OSDRenderer.draw_keyboard_overlay(frame, self._active_keys_display, 10, h_f - 115)

        # ─── Steuerungs-Hilfe (ganz unten, dezent) ───
        OSDRenderer.draw_text_with_background(
            frame, "Arrows=Base/Shoulder  W/S=Elbow  A/D=Hand  O/C=Grip  +/-=Speed  1-5=Level",
            (10, h_f - 40), font_scale=0.33, color=(180, 180, 180), bg_alpha=0.4
        )

    # ─── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self):
        """Aufräumen."""
        if self._recording:
            self._stop_recording(success=False)

        self._ui.print_info("Shutdown...")
        if self._arm:
            self._arm.park()
            self._arm.set_led(0)
            time.sleep(1.0)
            self._arm.disconnect()
        if self._camera:
            self._camera.release()
        cv2.destroyAllWindows()
        self._ui.print_status("Fertig")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoArm-M2-S Teleop Recorder (Enhanced)")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index (default: 2)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO-Modell")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min. Confidence")
    parser.add_argument("--output", type=str, default="recordings", help="Output-Verzeichnis")
    parser.add_argument("--target", type=str, default=None, help="Ziel-Objekt (z.B. 'bottle')")
    parser.add_argument("--no-sound", action="store_true", help="Sound-Feedback deaktivieren")
    args = parser.parse_args()

    recorder = TeleopRecorder(
        port=args.port,
        camera_index=args.camera,
        model_path=args.model,
        confidence=args.confidence,
        output_dir=args.output,
        target_class=args.target,
        enable_sound=not args.no_sound,
    )
    recorder.run()

if __name__ == "__main__":
    main()

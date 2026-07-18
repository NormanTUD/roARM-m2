#!/usr/bin/env python3
"""dashboard.py - RoArm-M2-S Unified TUI Dashboard v2

Tabs:
- Tab 1: TEACH (Recording mit Live-Feedback)
- Tab 2: PLAY (Recordings abspielen)
- Tab 3: CALIBRATE (Kalibrierung starten/verwalten)
- Tab 4: SERVO (Einzelne Servos ansteuern/auslesen)
- Tab 5: LOGS (Live-Logs mit Regex-Suche)

Alle Aktionen per Keyboard Shortcuts.
Auto-Connect wenn USB-Port gefunden.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "numpy",
#     "scipy",
#     "textual>=0.79.0",
#     "matplotlib",
#     "rich",
#     "Pillow",
#     "pyyaml",
# ]
# ///

import os
os.environ["TEXTUAL_RUNNING"] = "1"

import sys
import re

from bootstrap import ensure_uv
ensure_uv()

import rich
from rich.console import Console
from rich.text import Text

ci_env: bool = os.getenv("CI", "false").lower() == "true"

terminal_width = 150

try:
    terminal_width = os.get_terminal_size().columns
except OSError:
    pass

console = Console(
    force_interactive=True,
    soft_wrap=True,
    color_system="256",
    force_terminal=not ci_env,
    width=max(200, terminal_width)
)

def spinner(text: str):
    return console.status(f"[bold green]{text}", speed=0.2, refresh_per_second=6)

with spinner("Importing base modules..."):
    import json
    import time
    import math
    import threading
    import asyncio
    import io
    import base64
    import subprocess
    import tempfile
    import math
    from pathlib import Path
    from datetime import datetime
    from typing import Optional

with spinner("Importing numpy..."):
    import numpy as np

with spinner("Importing textual..."):
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import (
        Container, Horizontal, Vertical, ScrollableContainer,
        VerticalScroll,
    )
    from textual.widgets import (
        Header, Footer, Static, Button, Label, Input,
        TabbedContent, TabPane, DataTable, ProgressBar,
        Switch, Select, ListView, ListItem, RichLog,
        Rule,
    )
    from textual.reactive import reactive
    from textual.timer import Timer
    from textual.message import Message
    from textual.binding import Binding
    from textual.css.query import NoMatches
    from textual.widgets import Static
    from textual.strip import Strip

with spinner("Importing robot..."):
    from robot import (
        RoArmConnection, find_arm_port, rad_to_deg, deg_to_rad,
        START_POSITION_DEG, POSITION_TOLERANCE, BAUDRATE,
    )

with spinner("Importing safety..."):
    from safety import SafeArm, SafetyLimits

with spinner("Importing Pillow..."):
    from PIL import Image

# ============================================================
# KINEMATIK-KONSTANTEN
# ============================================================

BASE_HEIGHT = 75.0       # Höhe Basis bis Shoulder-Gelenk
UPPER_ARM = 206.0        # Oberarm: Shoulder → Elbow
FOREARM = 206.0          # Unterarm: Elbow → Gripper-Ansatz
GRIPPER_LENGTH = 80.0    # Gripper-Länge (Teil des Unterarm-Segments)

ENDPOINT_SPEEDS = [(8, 4), (5, 2), (3, 1)]
ENDPOINT_SETTLE_WAIT = 0.8

# ============================================================
# ADAPTIVE TIMING CONSTANTS
# ============================================================

MIN_SPEED_FACTOR = 0.5
MAX_SPEED_FACTOR = 1.2
END_RAMP_PERCENT = 0.05
START_RAMP_PERCENT = 0.03

# ============================================================
# KONFIGURATION
# ============================================================

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

RECORD_HZ = 50
MOVE_THRESHOLD_DEG = 0.1
STREAM_HZ = 50
MIN_DELTA_DEG = 0.02

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import logging

def forward_kinematics(b_deg: float, s_deg: float, e_deg: float) -> dict:
    """
    Kinematik für RoArm-M2-S (ohne Wrist-Gelenk).
    
    Gelenke:
    - b: Base-Rotation um Z-Achse. b=0 → +X (nach vorne)
    - s: Shoulder. s=0 → Oberarm HORIZONTAL nach vorne.
         s>0 → nach oben, s<0 → nach unten.
    - e: Elbow. Innenwinkel zwischen Ober- und Unterarm.
         e=180° → gestreckt
         e=90° → Unterarm steht 90° zum Oberarm
    """
    b_rad = math.radians(b_deg)
    s_rad = math.radians(90.0 - s_deg)

    base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, BASE_HEIGHT])

    # Oberarm
    elbow_local_x = UPPER_ARM * math.cos(s_rad)
    elbow_local_z = BASE_HEIGHT + UPPER_ARM * math.sin(s_rad)

    # Unterarm: absoluter Winkel
    forearm_abs_angle = s_rad - math.radians(e_deg)

    # Gripper-Spitze (Unterarm + Gripper als ein Segment)
    total_forearm = FOREARM + GRIPPER_LENGTH
    gripper_local_x = elbow_local_x + total_forearm * math.cos(forearm_abs_angle)
    gripper_local_z = elbow_local_z + total_forearm * math.sin(forearm_abs_angle)

    # Base-Rotation um Z-Achse
    cos_b = math.cos(b_rad)
    sin_b = math.sin(b_rad)

    def rotate_base(x, z):
        return np.array([x * cos_b, x * sin_b, z])

    return {
        "base": base,
        "shoulder": np.array([0.0, 0.0, BASE_HEIGHT]),
        "elbow": rotate_base(elbow_local_x, elbow_local_z),
        "gripper": rotate_base(gripper_local_x, gripper_local_z),
    }



class TUILogHandler(logging.Handler):
    """Leitet robot.py Warnungen ins Teach-Log der TUI."""

    def __init__(self, app: "RoArmDashboard"):
        super().__init__()
        self.app = app

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.WARNING:
                styled = f"[yellow]{msg}[/]"
            else:
                styled = f"[dim]{msg}[/]"

            # Wenn wir schon im Main-Thread sind (z.B. periodic poll):
            try:
                self.app._log_teach(styled)
            except Exception:
                self.app.call_from_thread(self.app._log_teach, styled)
        except Exception:
            pass

class SmoothTrajectory:
    """Smooth time-continuous trajectory from discrete waypoints."""

    def __init__(self, waypoints: list, speed_factor: float = 1.0):
        self._waypoints = waypoints
        self._speed_factor = speed_factor
        self._splines = {}
        self._time_map = None
        self._t_new = None
        self._speed_profile = None
        self._total_duration = 0.0
        self._original_duration = 0.0
        self._build_splines()
        self._compute_adaptive_timing()

    def _build_splines(self):
        """Creates clamped cubic splines for each joint."""
        from scipy.interpolate import CubicSpline
        times = np.array([wp["t"] for wp in self._waypoints])
        if times[0] > 0.01:
            times = np.concatenate([[0.0, times[0] * 0.5], times])
        for joint in ["b", "s", "e", "h"]:
            values = np.array([wp[joint] for wp in self._waypoints])
            if len(times) > len(values):
                pad = np.array([values[0], values[0]])
                values = np.concatenate([pad, values])
            self._splines[joint] = CubicSpline(times, values, bc_type='clamped')
        self._original_duration = times[-1]

    def _compute_curvature(self, t_original: np.ndarray) -> np.ndarray:
        """Computes smoothed curvature magnitude along the trajectory."""
        curvature = np.zeros(len(t_original))
        for joint in ["b", "s", "e", "h"]:
            d2 = self._splines[joint](t_original, 2)
            curvature += d2 ** 2
        curvature = np.sqrt(curvature)
        kernel = np.ones(20) / 20
        return np.convolve(curvature, kernel, mode='same')

    def _curvature_to_speed_profile(self, curvature: np.ndarray) -> np.ndarray:
        """Converts curvature to a speed profile (inverse relationship)."""
        max_curv = np.percentile(curvature, 95) if curvature.max() > 0 else 1.0
        norm = np.clip(curvature / max(max_curv, 1e-6), 0, 1)
        return MAX_SPEED_FACTOR - norm * (MAX_SPEED_FACTOR - MIN_SPEED_FACTOR)

    def _apply_ramps(self, speed_profile: np.ndarray) -> np.ndarray:
        """Applies start and end ramps to the speed profile."""
        n = len(speed_profile)
        end_start = int(n * (1.0 - END_RAMP_PERCENT))
        for i in range(end_start, n):
            progress = (i - end_start) / (n - end_start)
            speed_profile[i] = min(speed_profile[i],
                MIN_SPEED_FACTOR + (1.0 - progress) * (speed_profile[i] - MIN_SPEED_FACTOR))
        start_end = int(n * START_RAMP_PERCENT)
        for i in range(start_end):
            progress = i / max(start_end, 1)
            speed_profile[i] = MIN_SPEED_FACTOR + progress * (speed_profile[i] - MIN_SPEED_FACTOR)
        return speed_profile

    def _compute_adaptive_timing(self):
        """Computes time reparameterization based on curvature."""
        from scipy.interpolate import CubicSpline
        n_samples = 500
        t_original = np.linspace(0, self._original_duration, n_samples)
        curvature = self._compute_curvature(t_original)
        speed_profile = self._curvature_to_speed_profile(curvature)
        speed_profile = self._apply_ramps(speed_profile)
        dt = t_original[1] - t_original[0]
        dt_new = dt / (speed_profile * self._speed_factor)
        t_new = np.cumsum(dt_new)
        t_new = np.insert(t_new, 0, 0.0)[:-1]
        self._total_duration = t_new[-1]
        self._t_new = t_new
        self._speed_profile = speed_profile
        self._time_map = CubicSpline(t_new, t_original, bc_type='natural')

    def get_duration(self) -> float:
        """Returns total duration of the smoothed trajectory."""
        return self._total_duration

    def sample(self, t_playback: float) -> dict:
        """Samples joint angles at the given playback time."""
        t_playback = np.clip(t_playback, 0, self._total_duration)
        t_orig = float(self._time_map(t_playback))
        t_orig = np.clip(t_orig, 0, self._original_duration)
        return {j: round(float(self._splines[j](t_orig)), 2)
                for j in ["b", "s", "e", "h"]}

    def get_speed_at(self, t_playback: float) -> float:
        """Returns the speed factor at the given playback time."""
        idx = np.searchsorted(self._t_new, t_playback)
        idx = min(idx, len(self._speed_profile) - 1)
        return self._speed_profile[idx]

# ============================================================
# SIMULATED ARM (when no real robot is connected)
# ============================================================

class SimulatedArm:
    """Simulates the RoArm when no physical robot is connected.
    
    Provides the same interface as RoArmConnection but moves
    joints virtually with realistic timing.
    """

    def __init__(self):
        self._position = {
            "b": 0.0,
            "s": 0.0,
            "e": 90.0,
            "h": 180.0,
        }
        self._target = None
        self._torque_on = True
        self._gripper_open = True
        self._move_speed = 30.0  # degrees per second base speed
        self._moving = False
        self._lock = threading.Lock()

    def read_position_deg(self) -> dict:
        """Returns current simulated position."""
        with self._lock:
            return self._position.copy()

    def read_position_averaged(self, n: int = 10, interval: float = 0.05) -> dict:
        """Simulates averaged reading (just returns current pos with tiny noise)."""
        pos = self.read_position_deg()
        # Add tiny noise to simulate real sensor
        for j in ["b", "s", "e", "h"]:
            pos[j] += np.random.normal(0, 0.02)
        return pos

    def move_to(self, b: float, s: float, e: float, h: float,
                spd: int = 20, acc: int = 10):
        """Simulates a move command — instantly updates target, 
        position interpolates over time."""
        with self._lock:
            self._target = {"b": b, "s": s, "e": e, "h": h}
            self._move_speed = spd * 1.5  # scale speed param to deg/s

    def move_to_fast(self, b: float, s: float, e: float, h: float,
                     spd: int = 50, acc: int = 30):
        """Fast move — for streaming playback simulation."""
        with self._lock:
            # In simulation, fast moves update position more directly
            self._target = {"b": b, "s": s, "e": e, "h": h}
            self._move_speed = spd * 2.0

    def step_simulation(self, dt: float):
        """Advances the simulation by dt seconds.
        
        Call this periodically (e.g., every 20ms) to animate movement.
        """
        with self._lock:
            if self._target is None:
                return
            if not self._torque_on:
                return

            all_arrived = True
            for j in ["b", "s", "e", "h"]:
                diff = self._target[j] - self._position[j]
                if abs(diff) < 0.01:
                    self._position[j] = self._target[j]
                else:
                    all_arrived = False
                    # Move towards target at _move_speed deg/s
                    max_step = self._move_speed * dt
                    step = max(-max_step, min(max_step, diff))
                    self._position[j] += step

            if all_arrived:
                self._target = None
                self._moving = False
            else:
                self._moving = True

    def wait_until_settled(self, tolerance_deg: float = 0.2,
                           stable_count: int = 6):
        """Simulates waiting for settle — just waits a bit."""
        # In simulation, we just sleep briefly
        time.sleep(0.3)

    def torque_on(self):
        with self._lock:
            self._torque_on = True

    def torque_off(self):
        with self._lock:
            self._torque_on = False
            self._target = None

    def gripper_open(self):
        self._gripper_open = True

    def gripper_close(self):
        self._gripper_open = False

    def close(self):
        """No-op for simulated arm."""
        pass

    @property
    def is_simulated(self) -> bool:
        return True

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return self._target is not None

class BrailleCanvas:
    """
    Zeichnet auf einem Braille-Raster.
    Jede Terminal-Zelle = 2x4 Braille-Dots → 2× horizontale, 4× vertikale Auflösung.
    """

    BRAILLE_BASE = 0x2800
    # Braille dot positions: (col, row) → bit
    DOT_MAP = {
        (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04,
        (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20,
        (0, 3): 0x40, (1, 3): 0x80,
    }

    def __init__(self, char_width: int, char_height: int):
        self.char_width = char_width
        self.char_height = char_height
        # Pixel-Auflösung
        self.px_width = char_width * 2
        self.px_height = char_height * 4
        # Buffer: char_height × char_width, jeder Eintrag ist ein Braille-Bitmask
        self._buf = [[0] * char_width for _ in range(char_height)]
        # Farb-Buffer: speichert die Farbe pro Zelle (letzte gesetzte gewinnt)
        self._color_buf = [[None] * char_width for _ in range(char_height)]

    def clear(self):
        for row in self._buf:
            for i in range(len(row)):
                row[i] = 0
        for row in self._color_buf:
            for i in range(len(row)):
                row[i] = None

    def set_pixel(self, px: int, py: int, color: str = None):
        """Setzt einen Pixel (in Braille-Koordinaten)."""
        if px < 0 or px >= self.px_width or py < 0 or py >= self.px_height:
            return
        char_col = px // 2
        char_row = py // 4
        sub_col = px % 2
        sub_row = py % 4
        bit = self.DOT_MAP.get((sub_col, sub_row), 0)
        self._buf[char_row][char_col] |= bit
        if color:
            self._color_buf[char_row][char_col] = color

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: str = None):
        """Bresenham-Linie in Braille-Pixeln."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_thick_line(self, x0: int, y0: int, x1: int, y1: int,
                        thickness: int = 2, color: str = None):
        """Dicke Linie (mehrere parallele Linien)."""
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            self.set_pixel(x0, y0, color)
            return

        # Normale berechnen
        nx = -dy / length
        ny = dx / length

        for t in range(-(thickness // 2), thickness // 2 + 1):
            ox = int(nx * t)
            oy = int(ny * t)
            self.draw_line(x0 + ox, y0 + oy, x1 + ox, y1 + oy, color)

    def draw_circle(self, cx: int, cy: int, r: int, color: str = None):
        """Midpoint-Circle in Braille-Pixeln."""
        x = r
        y = 0
        err = 1 - r

        while x >= y:
            for px, py in [(cx+x, cy+y), (cx-x, cy+y), (cx+x, cy-y), (cx-x, cy-y),
                           (cx+y, cy+x), (cx-y, cy+x), (cx+y, cy-x), (cx-y, cy-x)]:
                self.set_pixel(px, py, color)
            y += 1
            if err < 0:
                err += 2 * y + 1
            else:
                x -= 1
                err += 2 * (y - x) + 1

    def fill_circle(self, cx: int, cy: int, r: int, color: str = None):
        """Gefüllter Kreis."""
        for dy in range(-r, r + 1):
            dx = int(math.sqrt(r * r - dy * dy))
            for x in range(cx - dx, cx + dx + 1):
                self.set_pixel(x, cy + dy, color)

    def draw_ellipse_arc(self, cx: int, cy: int, rx: int, ry: int,
                         start_angle: float, end_angle: float,
                         steps: int = 60, color: str = None):
        """Zeichnet einen Ellipsen-Bogen."""
        for i in range(steps):
            t = start_angle + (end_angle - start_angle) * i / steps
            x = int(cx + rx * math.cos(t))
            y = int(cy + ry * math.sin(t))
            self.set_pixel(x, y, color)

    def render(self) -> list[Text]:
        """Gibt eine Liste von Rich Text-Zeilen zurück."""
        lines = []
        for row_idx in range(self.char_height):
            text = Text()
            for col_idx in range(self.char_width):
                bits = self._buf[row_idx][col_idx]
                char = chr(self.BRAILLE_BASE + bits) if bits else ' '
                color = self._color_buf[row_idx][col_idx]
                if color and bits:
                    text.append(char, style=color)
                else:
                    text.append(char)
            lines.append(text)
        return lines

# ============================================================
# RECORDING PARSER
# ============================================================

def parse_roarm_file(filepath: str) -> dict:
    """Parses a .roarm file including LED events."""
    waypoints = []
    events = []
    config = {"hz": 20, "threshold": 0.3, "gravity_comp": 1}
    start_pos = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#OFFSET"):
                continue
            if line.startswith("#CONFIG"):
                _parse_config_line(line, config)
            elif line.startswith("#START_POS"):
                start_pos = _parse_start_pos_line(line)
            elif line.startswith("#"):
                continue
            elif line.startswith("MOVE"):
                waypoints.append(_parse_move_line(line))
            elif line.startswith("GRIPPER"):
                events.append(_parse_gripper_line(line))
            elif line.startswith("LED"):
                events.append(_parse_led_line(line))

    if start_pos is None:
        start_pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
    return {
        "waypoints": waypoints,
        "events": events,
        "config": config,
        "start_pos": start_pos,
    }

def _parse_config_line(line: str, config: dict):
    """Parses a #CONFIG line into the config dict."""
    parts = line.split(" ", 1)
    if len(parts) == 2:
        key, val = parts[1].split("=", 1)
        config[key.strip()] = float(val.strip())

def _parse_start_pos_line(line: str) -> dict:
    """Parses a #START_POS line into a position dict."""
    vals = {}
    for p in line.split()[1:]:
        k, v = p.split("=")
        vals[k] = float(v)
    return vals

def _parse_move_line(line: str) -> dict:
    """Parses a MOVE line into a waypoint dict."""
    vals = {}
    for p in line.split()[1:]:
        k, v = p.split("=")
        vals[k] = float(v)
    return {
        "t": vals.get("t", 0.0), "b": vals.get("b", 0.0),
        "s": vals.get("s", 0.0), "e": vals.get("e", 90.0),
        "h": vals.get("h", 180.0),
    }

def _parse_gripper_line(line: str) -> dict:
    """Parses a GRIPPER line into an event dict."""
    parts = line.split()
    cmd = parts[1] if len(parts) > 1 else "OPEN"
    t = _extract_time_from_parts(parts)
    return {"t": t, "cmd": cmd}


def _parse_led_line(line: str) -> dict:
    """Parses a LED line into an event dict."""
    parts = line.split()
    cmd = "LED_ON" if (len(parts) > 1 and parts[1] == "ON") else "LED_OFF"
    t = _extract_time_from_parts(parts)
    return {"t": t, "cmd": cmd}

def _extract_time_from_parts(parts: list) -> float:
    """Extracts t=... value from a list of key=value parts."""
    for p in parts:
        if p.startswith("t="):
            return float(p.split("=")[1])
    return 0.0

# ============================================================
# SPARKLINE HISTORY TRACKER
# ============================================================

class JointHistory:
    """Hält die letzten N Werte pro Gelenk für Sparklines."""

    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.data = {"b": [], "s": [], "e": [], "h": []}

    def push(self, pos: dict):
        for j in ["b", "s", "e", "h"]:
            self.data[j].append(pos.get(j, 0.0))
            if len(self.data[j]) > self.max_len:
                self.data[j].pop(0)

    def get(self, joint: str) -> list:
        return self.data.get(joint, [])

    def clear(self):
        self.data = {"b": [], "s": [], "e": [], "h": []}

# ============================================================
# ANIMATED STATUS INDICATOR
# ============================================================

class ActivityIndicator:
    """Manages animated spinner states for the status bar."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    DOTS_FRAMES = [".", "..", "...", ".."]

    def __init__(self):
        self._active = False
        self._message = ""
        self._icon = ""
        self._frame_index = 0
        self._start_time = 0.0

    def start(self, message: str, icon: str = "⏳"):
        self._active = True
        self._message = message
        self._icon = icon
        self._frame_index = 0
        self._start_time = time.time()

    def stop(self):
        self._active = False
        self._message = ""
        self._icon = ""
        self._frame_index = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def elapsed(self) -> float:
        if not self._active:
            return 0.0
        return time.time() - self._start_time

    def next_frame(self) -> str:
        """Returns the next animated status string."""
        if not self._active:
            return ""
        spinner = self.SPINNER_FRAMES[self._frame_index % len(self.SPINNER_FRAMES)]
        dots = self.DOTS_FRAMES[self._frame_index % len(self.DOTS_FRAMES)]
        elapsed = self.elapsed
        self._frame_index += 1
        return f"{self._icon} {spinner} {self._message}{dots} [{elapsed:.1f}s]"

# ============================================================
# CSS STYLESHEET
# ============================================================

CSS = """
Screen {
    background: $surface;
}

#main-container {
    height: 1fr;
}

TabbedContent {
    height: 1fr;
}

TabPane {
    height: 1fr;
}

.tab-content {
    height: 1fr;
    padding: 1;
}

.tab-content > Horizontal {
    height: 1fr;
}

.tab-content > Horizontal > Vertical {
    width: 1fr;
    height: 1fr;
}

.arm-view {
    border: solid $primary;
    height: 32;
    min-height: 20;
    padding: 0 1;
    overflow: hidden;
}

.status-bar {
    dock: bottom;
    height: 3;
    background: $panel;
    padding: 0 2;
    layout: horizontal;
}

.status-bar Label {
    width: auto;
    margin: 0 1;
}

#status-activity {
    width: auto;
    min-width: 35;
    margin: 0 1;
    color: $warning;
}

.recording-timer {
    height: 1;
    margin: 0 1;
    color: $error;
}

.joint-display {
    height: 3;
    border: solid $secondary;
    padding: 0 1;
}

.control-buttons {
    height: 3;
    align: center middle;
}

.control-buttons Button {
    margin: 0 1;
}

.info-panel {
    border: solid $success;
    height: auto;
    max-height: 12;
    padding: 1;
}

#teach-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#play-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#calibrate-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

#servo-log {
    height: 1fr;
    min-height: 5;
    border: solid $primary;
}

.btn-record {
    background: $error;
    color: white;
}

.btn-play {
    background: $success;
    color: white;
}

.btn-stop {
    background: $warning;
    color: black;
}

DataTable {
    height: 1fr;
    min-height: 5;
}

RichLog {
    scrollbar-gutter: stable;
}

#teach-left {
    width: 2fr;
    height: 1fr;
}

#teach-right {
    width: 1fr;
    height: auto;
    max-height: 100%;
}

.servo-control-panel {
    height: auto;
    border: solid $accent;
    padding: 1;
    margin: 0 0 1 0;
}

.servo-slider-row {
    height: 3;
    layout: horizontal;
}

.servo-slider-row Label {
    width: 12;
}

.servo-slider-row Input {
    width: 12;
}

#log-search-input {
    dock: top;
    height: 3;
    margin: 0 0 1 0;
}

#log-viewer {
    height: 1fr;
    border: solid $primary;
}

.log-filter-bar {
    height: 3;
    layout: horizontal;
    dock: top;
}

.log-filter-bar Input {
    width: 1fr;
}

.log-filter-bar Button {
    width: auto;
    margin: 0 1;
}

.recording-active { border: heavy red; }
"""

# ============================================================
# CUSTOM WIDGETS
# ============================================================

class Arm3DWidget(Static):
    """
    Widget das den 3D-Arm als hochauflösende Braille-Grafik rendert.
    Nutzt 2×4 Braille-Dots pro Zeichen → deutlich schärfer als Half-Block.
    """

    b = reactive(0.0)
    s = reactive(0.0)
    e = reactive(90.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trail = []
        self._target = None
        self._cam_azimuth = 45.0
        self._cam_elevation = 25.0
        self._cam_distance = 600.0

    def on_resize(self, event) -> None:
        """Re-render bei Terminal-Resize für korrekte Proportionen."""
        self._refresh_display()

    def update_pose(self, b: float, s: float, e: float, target: dict = None):
        self.b = b
        self.s = s
        self.e = e
        self._target = target

        # Trail updaten
        positions = forward_kinematics(b, s, e)
        self._trail.append(positions["gripper"])
        if len(self._trail) > 60:
            self._trail.pop(0)

        self._refresh_display()

    def rotate(self, d_azimuth: float = 0, d_elevation: float = 0):
        self._cam_azimuth = (self._cam_azimuth + d_azimuth) % 360
        self._cam_elevation = max(-80, min(80, self._cam_elevation + d_elevation))
        self._refresh_display()

    def clear_trail(self):
        self._trail.clear()
        self._refresh_display()

    def _project_3d(self, x: float, y: float, z: float,
                    canvas_w: int, canvas_h: int) -> tuple[int, int]:
        """Projiziert 3D → 2D Braille-Pixel-Koordinaten."""
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)

        # Rotation um Z-Achse (Azimuth)
        x1 = x * math.cos(az) - y * math.sin(az)
        y1 = x * math.sin(az) + y * math.cos(az)
        z1 = z

        # Rotation um X-Achse (Elevation)
        y2 = y1 * math.cos(el) - z1 * math.sin(el)
        z2 = y1 * math.sin(el) + z1 * math.cos(el)
        x2 = x1

        # Perspektivische Projektion
        d = self._cam_distance
        scale = d / (d + y2 + 300)

        # Auf Canvas-Koordinaten mappen
        px = int(x2 * scale * 0.35 + canvas_w // 2)
        py = int(-z2 * scale * 0.35 + canvas_h * 0.72)

        return px, py

    def _refresh_display(self):
        # Widget-Größe ermitteln
        w = self.size.width if self.size.width > 0 else 70
        h = self.size.height if self.size.height > 0 else 24

        canvas = BrailleCanvas(w, h)
        # Braille-Pixel-Auflösung
        pw = canvas.px_width
        ph = canvas.px_height

        positions = forward_kinematics(self.b, self.s, self.e)

        # --- Boden-Gitter ---
        for g in range(-300, 301, 100):
            pts_x = []
            for i in range(-300, 301, 50):
                px, py = self._project_3d(g, i, 0, pw, ph)
                pts_x.append((px, py))
            for i in range(len(pts_x) - 1):
                canvas.draw_line(pts_x[i][0], pts_x[i][1],
                                 pts_x[i+1][0], pts_x[i+1][1], "bright_black")

            pts_y = []
            for i in range(-300, 301, 50):
                px, py = self._project_3d(i, g, 0, pw, ph)
                pts_y.append((px, py))
            for i in range(len(pts_y) - 1):
                canvas.draw_line(pts_y[i][0], pts_y[i][1],
                                 pts_y[i+1][0], pts_y[i+1][1], "bright_black")

        # --- Koordinatenachsen ---
        origin = self._project_3d(0, 0, 0, pw, ph)
        ax_x = self._project_3d(80, 0, 0, pw, ph)
        ax_y = self._project_3d(0, 80, 0, pw, ph)
        ax_z = self._project_3d(0, 0, 80, pw, ph)
        canvas.draw_line(origin[0], origin[1], ax_x[0], ax_x[1], "red")
        canvas.draw_line(origin[0], origin[1], ax_y[0], ax_y[1], "green")
        canvas.draw_line(origin[0], origin[1], ax_z[0], ax_z[1], "blue")

        # --- Arbeitsraum-Kreis ---
        from visualize import UPPER_ARM, FOREARM, GRIPPER_LENGTH
        reach = UPPER_ARM + FOREARM + GRIPPER_LENGTH
        prev = None
        for i in range(65):
            angle = 2 * math.pi * i / 64
            x = reach * math.cos(angle)
            y = reach * math.sin(angle)
            pt = self._project_3d(x, y, 0, pw, ph)
            if prev:
                canvas.draw_line(prev[0], prev[1], pt[0], pt[1], "cyan")
            prev = pt

        # --- Trail ---
        if self._trail and len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                p1 = self._project_3d(*self._trail[i-1], pw, ph)
                p2 = self._project_3d(*self._trail[i], pw, ph)
                # Fade: ältere Punkte dunkler
                alpha_color = "bright_red" if i > len(self._trail) * 0.7 else "red"
                canvas.draw_line(p1[0], p1[1], p2[0], p2[1], alpha_color)

        # --- Target ---
        if self._target:
            t_pos = forward_kinematics(self._target["b"], self._target["s"],
                                       self._target["e"])
            tp = self._project_3d(*t_pos["gripper"], pw, ph)
            canvas.draw_circle(tp[0], tp[1], 4, "bright_red")
            canvas.draw_line(tp[0]-3, tp[1]-3, tp[0]+3, tp[1]+3, "bright_red")
            canvas.draw_line(tp[0]-3, tp[1]+3, tp[0]+3, tp[1]-3, "bright_red")

        # --- Arm-Segmente ---
        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["gripper"]]
        projected = [self._project_3d(p[0], p[1], p[2], pw, ph) for p in pts]

        # Basis (dick, weiß/hellgrau)
        base_bottom = self._project_3d(0, 0, 0, pw, ph)
        canvas.draw_thick_line(base_bottom[0], base_bottom[1],
                               projected[0][0], projected[0][1],
                               thickness=4, color="white")

        # Oberarm (dick, blau)
        canvas.draw_thick_line(projected[1][0], projected[1][1],
                               projected[2][0], projected[2][1],
                               thickness=3, color="bright_blue")

        # Unterarm + Gripper (mittel, grün)
        canvas.draw_thick_line(projected[2][0], projected[2][1],
                               projected[3][0], projected[3][1],
                               thickness=2, color="bright_green")

        # --- Gelenk-Punkte ---
        joint_colors = ["white", "bright_red", "bright_blue", "bright_yellow"]
        joint_radii = [5, 4, 3, 3]
        for i, (px, py) in enumerate(projected):
            canvas.fill_circle(px, py, joint_radii[i], joint_colors[i])

        # --- Rendern ---
        lines = canvas.render()

        # Header-Info hinzufügen
        gp = positions["gripper"]
        header = Text()
        header.append(f" Az:{self._cam_azimuth:.0f}° El:{self._cam_elevation:.0f}°",
                      style="bright_black")
        header.append(f"  b={self.b:+.1f}° s={self.s:+.1f}° e={self.e:+.1f}°",
                      style="bold bright_white")
        header.append(f"  Grip:({gp[0]:.0f},{gp[1]:.0f},{gp[2]:.0f})",
                      style="cyan")

        all_lines = [header] + lines
        self.update(Text("\n").join(all_lines))

    def on_mount(self):
        self._refresh_display()

class JointSparklineWidget(Static):
    """Zeigt Sparkline + Wert für ein einzelnes Gelenk."""

    JOINT_COLORS = {
        "b": "bright_blue",
        "s": "bright_magenta",
        "e": "bright_yellow",
        "h": "bright_cyan",
    }
    JOINT_NAMES = {
        "b": "Base    ",
        "s": "Shoulder",
        "e": "Elbow   ",
        "h": "Hand    ",
    }

    def __init__(self, joint: str, **kwargs):
        super().__init__(**kwargs)
        self.joint = joint
        self._value = 0.0
        self._history = []
        self._max_history = 40

    def update_value(self, value: float):
        self._value = value
        self._history.append(value)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._refresh()

    def _refresh(self):
        color = self.JOINT_COLORS.get(self.joint, "white")
        name = self.JOINT_NAMES.get(self.joint, self.joint)

        sparkline = self._make_sparkline()

        # Safety-Farbe
        limits = {"b": 135, "s": 90, "e": 180, "h": 360}
        limit = limits.get(self.joint, 180)
        pct = abs(self._value) / limit
        if pct > 0.9:
            safety_color = "red"
        elif pct > 0.7:
            safety_color = "yellow"
        else:
            safety_color = "green"

        text = (
            f"[{color}]{name}[/] "
            f"[bold {safety_color}]{self._value:+7.2f}°[/] "
            f"[dim]{sparkline}[/]"
        )
        self.update(text)

    def _make_sparkline(self) -> str:
        if not self._history:
            return "▁" * 20

        bars = "▁▂▃▄▅▆▇█"
        values = self._history[-20:]

        if len(values) < 2:
            return "▄" * len(values)

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val < 0.01:
            return "▄" * len(values)

        result = ""
        for v in values:
            idx = int((v - min_val) / range_val * (len(bars) - 1))
            idx = max(0, min(len(bars) - 1, idx))
            result += bars[idx]

        return result

class TimelineWidget(Static):
    """Zeigt eine Timeline für Recordings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._duration = 0.0
        self._position = 0.0
        self._waypoints = []
        self._width = 60

    def set_recording(self, waypoints: list, gripper_cmds: list = None):
        self._waypoints = waypoints
        if waypoints:
            self._duration = waypoints[-1]["t"]
        self._refresh()

    def set_position(self, t: float):
        self._position = t
        self._refresh()

    def _refresh(self):
        if self._duration <= 0:
            self.update("[dim]Kein Recording geladen[/]")
            return

        bar_width = self._width
        cursor_pos = int((self._position / self._duration) * bar_width)
        cursor_pos = max(0, min(bar_width - 1, cursor_pos))

        bar = list("─" * bar_width)

        for wp in self._waypoints[::max(1, len(self._waypoints) // bar_width)]:
            idx = int((wp["t"] / self._duration) * bar_width)
            idx = max(0, min(bar_width - 1, idx))
            bar[idx] = "┃"

        bar[cursor_pos] = "▶"
        bar_str = "".join(bar)

        text = (
            f"[bold]Timeline[/] [{self._position:.2f}s / {self._duration:.2f}s]\n"
            f"[bright_blue]┃[/]{bar_str}[bright_blue]┃[/]\n"
            f"[dim]0s{'':>{bar_width - 8}}{self._duration:.1f}s[/]"
        )
        self.update(text)

# ============================================================
# MAIN APP
# ============================================================

class RoArmDashboard(App):
    """RoArm-M2-S Unified TUI Dashboard v2."""

    TITLE = "RoArm-M2-S Dashboard"
    SUB_TITLE = "Teach · Play · Calibrate · Servo · Logs"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_tab('teach')", "Teach", show=True),
        Binding("2", "switch_tab('play')", "Play", show=True),
        Binding("3", "switch_tab('calibrate')", "Calibrate", show=True),
        Binding("4", "switch_tab('servo')", "Servo", show=True),
        Binding("5", "switch_tab('logs')", "Logs", show=True),
        Binding("c", "connect", "Connect", show=True),
        Binding("t", "torque_release", "Torque Off", show=True),
        Binding("T", "torque_lock", "Torque On", show=True),
        Binding("space", "toggle_action", "Start/Stop", show=True),
        Binding("g", "gripper_toggle", "Gripper", show=True),
        Binding("h", "go_home", "Home", show=True),
        Binding("a", "rotate_left", "Rot←", show=False),
        Binding("d", "rotate_right", "Rot→", show=False),
        Binding("w", "rotate_up", "Rot↑", show=False),
        Binding("s", "rotate_down", "Rot↓", show=False),
        Binding("r", "read_position", "Read Pos", show=True),
        Binding("escape", "emergency_stop", "E-STOP", show=True, priority=True),
        Binding("l", "led_toggle", "LED", show=False),
    ]

    # --- Reactive State ---
    connected = reactive(False)
    recording = reactive(False)
    playing = reactive(False)
    torque_on_state = reactive(True)

    GRAVITY_COMP_SETTLE_MS = 30

    def __init__(self):
        super().__init__()
        self._arm: Optional[RoArmConnection] = None
        self._sim_arm: Optional[SimulatedArm] = None
        self._simulation_mode = False
        self._sim_timer: Optional[Timer] = None
        self._joint_history = JointHistory()
        self._current_pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}

        # Teach state
        self._teach_waypoints = []
        self._teach_start_time = 0.0
        self._teach_timer: Optional[Timer] = None
        self._gripper_open = True

        # Play state
        self._play_data = None
        self._play_start_time = 0.0

        # Logs state
        self._all_log_lines = []
        self._log_filter_pattern = ""

        # Activity indicator state
        self._activity = ActivityIndicator()
        self._activity_timer: Optional[Timer] = None

        # Recording elapsed timer
        self._recording_elapsed_timer: Optional[Timer] = None

        self._teach_waypoints = []
        self._teach_start_time = 0.0
        self._teach_timer: Optional[Timer] = None
        self._led_on = False
        self._gravity_comp_enabled = True
        self._speed_factor = 1.0
        self._loop_enabled = False
        self._loop_pause_s = 0.0

        # New state variables for enhanced features
        self._led_on = False
        self._gravity_comp_enabled = True
        self._speed_factor = 1.0
        self._last_play_commanded: Optional[dict] = None
        self._safe_arm: Optional[object] = None
        self._watchdog: Optional[object] = None
        self._current_monitor: Optional[object] = None


    def _init_safety_state(self):
        """Initializes safety-related state variables."""
        self._safe_arm: Optional['SafeArm'] = None
        self._watchdog: Optional['SafetyWatchdog'] = None
        self._current_monitor: Optional['CurrentMonitor'] = None
        self._rate_limiter = None


    def _setup_safety_layer(self, arm: RoArmConnection):
        """Wraps the raw arm connection in safety layers."""
        from safety import SafeArm, SafetyLimits, SafetyWatchdog, CurrentMonitor, RateLimiter
        limits = SafetyLimits(
            max_delta_per_cmd=20.0,
            max_continuous_move_s=90.0,
            max_plausible_error=5.0,
        )
        self._safe_arm = SafeArm(arm, limits=limits)
        self._watchdog = SafetyWatchdog(self._safe_arm)
        self._watchdog.start()
        self._current_monitor = CurrentMonitor(
            self._safe_arm, max_load_percent=85.0, max_stall_duration_s=3.0
        )
        self._rate_limiter = RateLimiter(max_hz=STREAM_HZ + 10)

    def _teardown_safety_layer(self):
        """Stops and cleans up safety layers."""
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog = None
        self._safe_arm = None
        self._current_monitor = None
        self._rate_limiter = None

    def _validate_trajectory(self, trajectory: 'SmoothTrajectory') -> tuple:
        """Validates trajectory for acceleration violations. Returns (ok, violations)."""
        from safety import TrajectoryValidator, SafetyLimits
        validator = TrajectoryValidator(SafetyLimits())
        return validator.validate_full_trajectory(trajectory)

    def _attempt_trajectory_repair(self, trajectory: 'SmoothTrajectory',
                                    violations: list) -> Optional['SmoothTrajectory']:
        """Attempts to repair a trajectory by local time-stretching."""
        from safety import TrajectorySmoother, SafetyLimits, TrajectoryValidator
        smoother = TrajectorySmoother(SafetyLimits())
        new_wps, n_fixed, added_time = smoother.smooth_trajectory(trajectory)
        if new_wps is None:
            return None
        repaired = SmoothTrajectory(new_wps, self._speed_factor)
        ok, _ = TrajectoryValidator(SafetyLimits()).validate_full_trajectory(repaired)
        if not ok:
            return None
        self._log_play(
            f"[yellow]⚠ Repariert: {n_fixed} Verletzungen, +{added_time:.2f}s[/]"
        )
        return repaired

    def _check_tracking_error(self, arm_raw, corrected: dict,
                               commands_sent: int) -> Optional[float]:
        """Periodically checks tracking error. Returns error or None."""
        FEEDBACK_INTERVAL = max(1, STREAM_HZ // 2)
        MAX_TRACKING_ERROR = 8.0
        if commands_sent % FEEDBACK_INTERVAL != 0 or commands_sent <= FEEDBACK_INTERVAL:
            return None
        try:
            actual = arm_raw.read_position_deg()
            if actual is None:
                return None
            err = max(abs(actual[j] - corrected[j]) for j in ["b", "s", "e", "h"])
            if err > MAX_TRACKING_ERROR:
                self._trigger_playback_estop(
                    f"Tracking Error {err:.1f}° > {MAX_TRACKING_ERROR}°")
            return err
        except Exception:
            return None

    def _check_stall_detection(self, actual_deg: dict) -> bool:
        """Checks for stalled servos. Returns True if stall detected."""
        if self._current_monitor is None:
            return False
        ok, reason = self._current_monitor.check(actual_deg)
        if not ok:
            self._trigger_playback_estop(reason)
            return True
        return False

    def _trigger_playback_estop(self, reason: str):
        """Triggers emergency stop during playback."""
        self.playing = False
        if self._arm:
            self._arm.torque_off()
        self.app.call_from_thread(
            self._log_play, f"[bold red]🚨 E-STOP: {reason}[/]"
        )

    def action_led_toggle(self) -> None:
        """Toggles the LED on/off and records the event."""
        arm = self._active_arm
        if arm is None:
            return
        self._led_on = not getattr(self, '_led_on', False)
        brightness = 255 if self._led_on else 0
        if hasattr(arm, 'send_cmd'):
            arm.send_cmd({"T": 114, "led": brightness})
        self._log_teach(f"[bold]💡 LED {'AN' if self._led_on else 'AUS'}[/]")
        if self.recording:
            elapsed = time.time() - self._teach_start_time
            cmd = "LED_ON" if self._led_on else "LED_OFF"
            self._teach_waypoints.append({"t": round(elapsed, 4), "cmd": cmd})


    def _teach_read_position(self, arm) -> Optional[dict]:
        """Reads position during teach, using gravity comp if enabled."""
        if self._gravity_comp_enabled and not self._is_sim:
            return self._read_with_gravity_comp(arm)
        return arm.read_position_deg()


    def _read_with_gravity_comp(self, arm) -> Optional[dict]:
        """Reads position with brief torque pulse to counter gravity drift."""
        arm.torque_on_fast()
        time.sleep(GRAVITY_COMP_SETTLE_MS / 1000.0)
        pos = arm.read_position_deg()
        arm.torque_off_fast()
        return pos


    def action_emergency_stop(self):
        """Immediate halt: stop all motion, torque off, cancel all workers."""
        self.playing = False
        self.recording = False
        if self._arm:
            self._arm.torque_off()  # or send stop command
        self._log_teach("[bold red]🚨 EMERGENCY STOP[/]")

    @property
    def _active_arm(self):
        """Returns the real arm if connected, otherwise the simulated arm."""
        if self._arm and self.connected:
            return self._arm
        if self._simulation_mode and self._sim_arm:
            return self._sim_arm
        return None

    @property
    def _is_sim(self) -> bool:
        """True if currently using simulated arm."""
        return self._simulation_mode and not self.connected

    @staticmethod
    def _apply_calibration_static(cal_model, target: dict) -> dict:
        """Wendet Kalibrierungskorrektur an (identisch zu play.py Logik)."""
        if cal_model is None or not cal_model.is_fitted:
            return target.copy()

        correction = cal_model.predict_correction(target)

        MAX_CORRECTION_DEG = 6.0
        for j in ["b", "s", "e"]:
            if abs(correction[j]) > MAX_CORRECTION_DEG:
                correction[j] = max(-MAX_CORRECTION_DEG,
                                   min(MAX_CORRECTION_DEG, correction[j]))

        return {
            "b": target["b"] - correction["b"],
            "s": target["s"] - correction["s"],
            "e": target["e"] - correction["e"],
            "h": target["h"],
        }

    def _show_calibration_info(self):
        """Zeigt Kalibrierungsinfos im Play-Log an."""
        cal_path = Path("calibration") / "roarm_calibration.cal"
        if not cal_path.exists():
            self._log_play("[yellow]⚠ Keine Kalibrierungsdatei gefunden[/]")
            self._log_play("[dim]  → Playback ohne Kalibrierung (wie raw Servo-Werte)[/]")
            return

        try:
            from calibrate import CalibrationModel
            model = CalibrationModel.load(str(cal_path))
            res = model.residuals

            self._log_play("[bold cyan]📐 Kalibrierungsmodell geladen:[/]")
            self._log_play(f"  Datei: {cal_path}")
            self._log_play(f"  Typ: Polynom 2. Ordnung (10 Koeffizienten/Gelenk)")
            self._log_play(
                f"  Residuen (Fit-Qualität):"
            )
            for j in ["b", "s", "e"]:
                r = res.get(j, 0)
                quality = "✅" if r < 0.3 else "⚠️" if r < 1.0 else "❌"
                self._log_play(f"    {j.upper()}: {r:.4f}° {quality}")

            # Diagnostik laden falls vorhanden
            diag_path = Path("calibration") / "roarm_diagnostics.json"
            if diag_path.exists():
                with open(diag_path, 'r') as f:
                    diag = json.load(f)

                if "pose_set" in diag:
                    self._log_play(f"  Pose-Set: {diag['pose_set']}")
                if "total_measurements" in diag:
                    self._log_play(f"  Messungen: {diag['total_measurements']}")
                if "repeats_per_pose" in diag:
                    self._log_play(f"  Wiederholungen/Pose: {diag['repeats_per_pose']}")
                if "avg_settle_time_s" in diag:
                    self._log_play(
                        f"  Ø Settle-Zeit: {diag['avg_settle_time_s']:.2f}s"
                    )
                if "avg_noise_std_deg" in diag:
                    noise = diag["avg_noise_std_deg"]
                    self._log_play(
                        f"  Ø Rauschen: b={noise.get('b',0):.4f}° "
                        f"s={noise.get('s',0):.4f}° e={noise.get('e',0):.4f}°"
                    )
                if "position_error_stats" in diag:
                    stats = diag["position_error_stats"]
                    self._log_play(f"  Positionsfehler (vor Kalibrierung):")
                    for j in ["b", "s", "e"]:
                        if j in stats:
                            s = stats[j]
                            self._log_play(
                                f"    {j.upper()}: mean={s['mean']:+.3f}° "
                                f"σ={s['std']:.3f}° max={s['max']:.3f}°"
                            )
                if "repeatability_deg" in diag:
                    rep = diag["repeatability_deg"]
                    self._log_play(
                        f"  Repeatability: Δb={rep.get('b',0):.3f}° "
                        f"Δs={rep.get('s',0):.3f}° Δe={rep.get('e',0):.3f}°"
                    )

            # Beispiel-Korrektur für Home-Position
            home = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
            corr = model.predict_correction(home)
            self._log_play(
                f"  [dim]Beispiel Home-Korrektur: "
                f"Δb={corr['b']:+.3f}° Δs={corr['s']:+.3f}° Δe={corr['e']:+.3f}°[/]"
            )

        except Exception as e:
            self._log_play(f"[red]Fehler beim Laden der Kalibrierung: {e}[/]")

    def _get_play_speed(self) -> float:
        """Reads the speed factor from the UI input."""
        try:
            inp = self.query_one("#play-speed-input", Input)
            return max(0.1, min(5.0, float(inp.value)))
        except (NoMatches, ValueError):
            return 1.0

    def _get_loop_pause(self) -> float:
        """Reads the loop pause duration from the UI input."""
        try:
            inp = self.query_one("#play-loop-pause-input", Input)
            return max(0.0, float(inp.value))
        except (NoMatches, ValueError):
            return 0.0


    def _is_loop_enabled(self) -> bool:
        """Checks if loop mode is enabled via the button state."""
        try:
            btn = self.query_one("#btn-play-loop", Button)
            return btn.variant == "success"
        except NoMatches:
            return False

    def _process_playback_event(self, arm, event: dict, elapsed: float):
        """Processes a single playback event (gripper or LED)."""
        cmd = event["cmd"]
        if cmd == "CLOSE":
            arm.gripper_close()
            self.app.call_from_thread(
                self._log_play, f"[bold]  ✊ Gripper ZU [{elapsed:.2f}s][/]")
        elif cmd == "OPEN":
            arm.gripper_open()
            self.app.call_from_thread(
                self._log_play, f"[bold]  ✋ Gripper AUF [{elapsed:.2f}s][/]")
        elif cmd == "LED_ON":
            if hasattr(arm, 'send_cmd'):
                arm.send_cmd({"T": 114, "led": 255})
            self.app.call_from_thread(
                self._log_play, f"[bold]  💡 LED AN [{elapsed:.2f}s][/]")
        elif cmd == "LED_OFF":
            if hasattr(arm, 'send_cmd'):
                arm.send_cmd({"T": 114, "led": 0})
            self.app.call_from_thread(
                self._log_play, f"[bold]  💡 LED AUS [{elapsed:.2f}s][/]")

    def _process_pending_events(self, arm, events: list, event_idx: int,
                                 elapsed: float) -> tuple:
        """Processes all events up to current elapsed time. Returns (new_idx, pause)."""
        total_pause = 0.0
        while event_idx < len(events):
            ev = events[event_idx]
            if ev["t"] <= elapsed:
                self._process_playback_event(arm, ev, elapsed)
                if ev["cmd"] in ("CLOSE", "OPEN"):
                    time.sleep(0.3)
                    total_pause += 0.3
                event_idx += 1
            else:
                break
        return event_idx, total_pause


    @on(Button.Pressed, "#btn-play-loop")
    def on_play_loop_toggle(self) -> None:
        """Toggles loop mode on/off."""
        try:
            btn = self.query_one("#btn-play-loop", Button)
            if btn.variant == "success":
                btn.variant = "default"
                btn.label = "🔁 Loop"
                self._log_play("[dim]Loop deaktiviert[/]")
            else:
                btn.variant = "success"
                btn.label = "🔁 Loop ✓"
                self._log_play("[green]Loop aktiviert[/]")
        except NoMatches:
            pass


    # ============================================================
    # COMPOSE (Layout)
    # ============================================================

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            with TabbedContent():
                # --- TAB 1: TEACH ---
                with TabPane("🎬 Teach [1]", id="teach"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical(id="teach-left"):
                                yield Arm3DWidget(
                                    id="teach-arm-view",
                                    classes="arm-view"
                                )
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "⏺ Record [Space]", id="btn-teach-record",
                                        classes="btn-record", variant="error"
                                    )
                                    yield Button(
                                        "⏹ Stop [Space]", id="btn-teach-stop",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "🏠 Home [h]", id="btn-teach-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "✊/✋ Gripper [g]", id="btn-gripper",
                                        variant="default"
                                    )
                                # Recording elapsed timer display
                                yield Label(
                                    "", id="teach-recording-timer",
                                    classes="recording-timer"
                                )

                            with Vertical(id="teach-right"):
                                yield JointSparklineWidget(
                                    "b", id="teach-joint-b",
                                    classes="joint-display"
                                )
                                yield JointSparklineWidget(
                                    "s", id="teach-joint-s",
                                    classes="joint-display"
                                )
                                yield JointSparklineWidget(
                                    "e", id="teach-joint-e",
                                    classes="joint-display"
                                )
                                yield JointSparklineWidget(
                                    "h", id="teach-joint-h",
                                    classes="joint-display"
                                )

                        yield RichLog(id="teach-log", highlight=True, markup=True)

                # --- TAB 2: PLAY ---
                with TabPane("▶️ Play [2]", id="play"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="play-arm-view",
                                    classes="arm-view"
                                )
                                yield TimelineWidget(id="play-timeline")
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "▶ Play [Space]", id="btn-play-start",
                                        classes="btn-play", variant="success"
                                    )
                                    yield Button(
                                        "⏹ Stop [Space]", id="btn-play-stop",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "🔁 Loop", id="btn-play-loop",
                                        variant="default"
                                    )

                                    yield Input(value="0", id="play-loop-pause-input", type="number")
                                    yield Label("Speed:", classes="joint-label")
                                    yield Input(value="1.0", id="play-speed-input", type="number")
                                    yield Label("Loop Pause (s):", classes="joint-label")

                            with Vertical():
                                yield Label("📁 Recordings:", classes="joint-label")
                                yield DataTable(id="recordings-table")

                        yield RichLog(id="play-log", highlight=True, markup=True)

                # --- TAB 3: CALIBRATE ---
                with TabPane("🎯 Calibrate [3]", id="calibrate"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="calibrate-arm-view",
                                    classes="arm-view"
                                )
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "▶ Start Calibration", id="btn-cal-start",
                                        classes="btn-play", variant="success"
                                    )
                                    yield Button(
                                        "⏹ Abort", id="btn-cal-abort",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "📂 Load Cal", id="btn-cal-load",
                                        variant="default"
                                    )

                            with Vertical():
                                yield Label("[bold]Calibration Settings[/]")
                                with Horizontal(classes="servo-slider-row"):
                                    yield Label("Pose Set:")
                                    yield Select(
                                        [(name, name) for name in
                                         ["minimal", "standard", "extended"]],
                                        value="standard",
                                        id="cal-pose-set"
                                    )
                                with Horizontal(classes="servo-slider-row"):
                                    yield Label("Repeats:")
                                    yield Input(
                                        value="3", id="cal-repeats",
                                        type="integer",
                                    )
                                with Horizontal(classes="servo-slider-row"):
                                    yield Label("Auto Accept:")
                                    yield Switch(id="cal-auto-accept", value=True)
                                yield Static(id="cal-status-panel",
                                             classes="info-panel")

                        yield RichLog(id="calibrate-log", highlight=True, markup=True)

                # --- TAB 4: SERVO CONTROL ---
                with TabPane("🔧 Servo [4]", id="servo"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="servo-arm-view",
                                    classes="arm-view"
                                )

                            with Vertical():
                                # Servo 1: Base
                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_blue]Servo 1: BASE[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [°]:")
                                        yield Input(
                                            value="0.0", id="servo-b-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-b-go",
                                                     variant="primary")
                                    yield Static(id="servo-b-readout")

                                # Servo 2: Shoulder
                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_magenta]Servo 2: SHOULDER[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [°]:")
                                        yield Input(
                                            value="0.0", id="servo-s-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-s-go",
                                                     variant="primary")
                                    yield Static(id="servo-s-readout")

                                # Servo 3: Elbow
                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_yellow]Servo 3: ELBOW[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [°]:")
                                        yield Input(
                                            value="90.0", id="servo-e-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-e-go",
                                                     variant="primary")
                                    yield Static(id="servo-e-readout")

                                # Servo 4: Hand/Gripper
                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_cyan]Servo 4: HAND[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [°]:")
                                        yield Input(
                                            value="180.0", id="servo-h-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-h-go",
                                                     variant="primary")
                                    yield Static(id="servo-h-readout")

                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "📑 Read All [r]", id="btn-servo-read",
                                        variant="default"
                                    )
                                    yield Button(
                                        "🏠 Home [h]", id="btn-servo-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "🔌 Torque Off [t]",
                                        id="btn-servo-torque-off",
                                        variant="warning"
                                    )

                        yield RichLog(id="servo-log", highlight=True, markup=True)

                # --- TAB 5: LOGS ---
                with TabPane("📋 Logs [5]", id="logs"):
                    with Vertical(classes="tab-content"):
                        with Horizontal(classes="log-filter-bar"):
                            yield Input(
                                placeholder="Filter (text or /regex/)...",
                                id="log-search-input"
                            )
                            yield Button("🔍 Filter", id="btn-log-filter",
                                         variant="primary")
                            yield Button("↻ Refresh", id="btn-log-refresh",
                                         variant="default")
                            yield Button("🗑 Clear", id="btn-log-clear",
                                         variant="error")
                        yield RichLog(id="log-viewer", highlight=True, markup=True)

        # Status-Bar
        with Horizontal(classes="status-bar"):
            yield Label("🔌 Disconnected", id="status-connection")
            yield Label("┊", id="status-sep1")
            yield Label("🔒 Torque ON", id="status-torque")
            yield Label("┊", id="status-sep2")
            yield Label("🛡️ Safety OK", id="status-safety")
            yield Label("┊", id="status-sep3")
            yield Label("⏱️ --", id="status-mode")
            yield Label("┊", id="status-sep4")
            yield Label("", id="status-activity")

        yield Footer()

    # ============================================================
    # ON MOUNT
    # ============================================================

    def on_mount(self) -> None:
        """Initialisierung nach dem Mounten."""
        # Robot-Logger in die TUI umleiten
        robot_logger = logging.getLogger("roarm.commands")
        handler = TUILogHandler(self)
        handler.setLevel(logging.WARNING)
        fmt = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(fmt)
        robot_logger.addHandler(handler)

        self._refresh_recordings_table()
        self._try_auto_connect()
        self._load_logs()
        self._apply_log_filter()

        # Kalibrierungs-Info im Play-Tab anzeigen
        self._show_calibration_info()

        # If no real robot connected, start simulation mode
        if not self.connected:
            self._enable_simulation_mode()

        # Periodischer Position-Poll (wenn verbunden oder simuliert)
        self.set_interval(0.5, self._periodic_position_poll)
        # Log-Refresh
        self.set_interval(2.0, self._periodic_log_refresh)

    def _enable_simulation_mode(self):
        """Activates simulation mode with a virtual arm."""
        self._sim_arm = SimulatedArm()
        self._simulation_mode = True

        # Start simulation step timer (50Hz for smooth animation)
        self._sim_timer = self.set_interval(0.02, self._sim_step)

        self._log_teach(
            "[bold cyan]🤖 SIMULATION MODE[/] — Kein Roboter verbunden"
        )
        self._log_teach(
            "[dim]  Alle Bewegungen werden virtuell simuliert.[/]"
        )
        self._log_teach(
            "[dim]  Drücke [c] um einen echten Roboter zu verbinden.[/]"
        )

        # Update status bar
        try:
            label = self.query_one("#status-connection", Label)
            label.update("🤖 SIMULATION")
        except NoMatches:
            pass

        try:
            mode_label = self.query_one("#status-mode", Label)
            mode_label.update("🤖 Sim Ready")
        except NoMatches:
            pass

        # Show initial position in views
        pos = self._sim_arm.read_position_deg()
        self._current_pos = pos
        self._update_joint_displays(pos)
        self._update_arm_views(pos)

    def _disable_simulation_mode(self):
        """Deactivates simulation mode."""
        self._simulation_mode = False
        if self._sim_timer:
            self._sim_timer.stop()
            self._sim_timer = None
        self._sim_arm = None

    def _sim_step(self):
        """Called every 20ms to advance the simulation."""
        if not self._simulation_mode or not self._sim_arm:
            return
        # Advance physics
        self._sim_arm.step_simulation(0.02)

        # If the sim arm is actively moving, update the display
        if self._sim_arm.is_moving:
            pos = self._sim_arm.read_position_deg()
            self._current_pos = pos
            self._update_joint_displays(pos)
            self._update_arm_views(pos)
            self._update_servo_readouts(pos)

    def _try_auto_connect(self):
        """Versucht automatisch den Arm zu finden und zu verbinden."""
        port = find_arm_port()
        if port:
            self._log_teach(f"[dim]🔍 Port gefunden: {port} — verbinde...[/]")
            self._do_connect(port)
        else:
            self._log_teach(
                "[yellow]⚠ Kein Arm-Port gefunden. "
                "Starte im Simulationsmodus. [c] zum Verbinden.[/]"
            )

    def _do_connect(self, port: str):
        """Connects to the arm and sets up safety layers."""
        try:
            self._arm = RoArmConnection(port)
            self.connected = True
            if self._simulation_mode:
                self._disable_simulation_mode()
            self._setup_safety_layer(self._arm)
            self._log_teach(f"[green]✅ Verbunden mit {port} (Safety aktiv)[/]")
            self._update_status_connection(port)
            pos = self._arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self._update_joint_displays(pos)
                self._update_arm_views(pos)
        except Exception as e:
            self._log_teach(f"[red]❌ Fehler: {e}[/]")

    # ============================================================
    # PERIODIC TASKS
    # ============================================================

    def _periodic_position_poll(self):
        """Pollt die Position wenn verbunden und nicht recording/playing."""
        if self.recording or self.playing:
            return

        arm = self._active_arm
        if arm is None:
            return

        # In simulation mode, the sim_step timer handles updates during movement
        # Only poll here for idle state display refresh
        if self._is_sim and self._sim_arm and self._sim_arm.is_moving:
            return  # sim_step handles this

        try:
            pos = arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self._update_joint_displays(pos)
                self._update_arm_views(pos)
                self._update_servo_readouts(pos)
        except Exception:
            pass

    # ============================================================
    # TABLE SETUP
    # ============================================================

    def _refresh_recordings_table(self):
        """Aktualisiert die Recordings-Liste."""
        try:
            table = self.query_one("#recordings-table", DataTable)
        except NoMatches:
            return

        table.clear(columns=True)
        table.add_columns("Datei", "Dauer", "WPs", "Datum")

        recordings = sorted(RECORDINGS_DIR.glob("*.roarm"), reverse=True)
        for rec in recordings[:20]:
            try:
                data = parse_roarm_file(str(rec))
                wps = data["waypoints"]
                duration = wps[-1]["t"] if wps else 0
                date = rec.stem.replace("recording_", "")
                table.add_row(rec.name, f"{duration:.1f}s", str(len(wps)), date)
            except Exception:
                table.add_row(rec.name, "?", "?", "?")

        if table.row_count > 0:
            table.move_cursor(row=0)

    # ============================================================
    # LOGGING HELPERS
    # ============================================================

    def _log_teach(self, msg: str):
        try:
            self.query_one("#teach-log", RichLog).write(msg)
        except NoMatches:
            pass

    def _log_play(self, msg: str):
        try:
            self.query_one("#play-log", RichLog).write(msg)
        except NoMatches:
            pass

    def _log_calibrate(self, msg: str):
        try:
            self.query_one("#calibrate-log", RichLog).write(msg)
        except NoMatches:
            pass

    def _log_servo(self, msg: str):
        try:
            self.query_one("#servo-log", RichLog).write(msg)
        except NoMatches:
            pass

    # ============================================================
    # STATUS UPDATES
    # ============================================================

    def _update_status_connection(self, port: str = None):
        try:
            label = self.query_one("#status-connection", Label)
            label.update(f"🔌 {port}" if port else "🔌 Disconnected")
        except NoMatches:
            pass

    def _update_status_torque(self, on: bool):
        try:
            label = self.query_one("#status-torque", Label)
            label.update("🔒 Torque ON" if on else "🔓 Torque OFF")
        except NoMatches:
            pass

    def _update_joint_displays(self, pos: dict):
        """Aktualisiert die Gelenk-Sparkline-Widgets."""
        self._joint_history.push(pos)
        for joint in ["b", "s", "e", "h"]:
            try:
                widget = self.query_one(
                    f"#teach-joint-{joint}", JointSparklineWidget
                )
                widget.update_value(pos[joint])
            except NoMatches:
                pass

    def _update_arm_views(self, pos: dict):
        """Aktualisiert alle 3D-Arm-Ansichten."""
        for view_id in ["teach-arm-view", "play-arm-view",
                        "calibrate-arm-view", "servo-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", Arm3DWidget)
                widget.update_pose(pos["b"], pos["s"], pos["e"])
            except NoMatches:
                pass

    def _update_servo_readouts(self, pos: dict):
        """Aktualisiert die Servo-Readout-Anzeigen."""
        for joint, name in [("b", "Base"), ("s", "Shoulder"),
                            ("e", "Elbow"), ("h", "Hand")]:
            try:
                widget = self.query_one(f"#servo-{joint}-readout", Static)
                widget.update(
                    f"  [dim]Current:[/] [bold]{pos[joint]:+7.2f}°[/]"
                )
            except NoMatches:
                pass

    # ============================================================
    # ACTIVITY INDICATOR METHODS
    # ============================================================

    def _start_activity(self, message: str, icon: str = "⏳"):
        """Starts the animated activity indicator in the status bar."""
        self._activity.start(message, icon)
        # Start the animation timer (updates every 100ms for smooth animation)
        if self._activity_timer is not None:
            self._activity_timer.stop()
        self._activity_timer = self.set_interval(0.1, self._tick_activity)

    def _stop_activity(self, final_message: str = ""):
        """Stops the animated activity indicator."""
        self._activity.stop()
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None
        try:
            label = self.query_one("#status-activity", Label)
            if final_message:
                label.update(final_message)
            else:
                label.update("")
        except NoMatches:
            pass

    def _tick_activity(self):
        """Called by timer to animate the activity indicator."""
        if not self._activity.is_active:
            return
        try:
            label = self.query_one("#status-activity", Label)
            label.update(self._activity.next_frame())
        except NoMatches:
            pass

    def _start_recording_timer(self):
        """Starts the recording elapsed time display."""
        self._update_recording_timer_display()
        if self._recording_elapsed_timer is not None:
            self._recording_elapsed_timer.stop()
        self._recording_elapsed_timer = self.set_interval(
            0.5, self._update_recording_timer_display
        )

    def _stop_recording_timer(self):
        """Stops the recording elapsed time display."""
        if self._recording_elapsed_timer is not None:
            self._recording_elapsed_timer.stop()
            self._recording_elapsed_timer = None
        try:
            label = self.query_one("#teach-recording-timer", Label)
            label.update("")
        except NoMatches:
            pass

    def _update_recording_timer_display(self):
        """Updates the recording timer label with elapsed time and waypoint count."""
        if not self.recording:
            return
        elapsed = time.time() - self._teach_start_time
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        wp_count = len(move_wps)

        # Pulsing dot animation
        dots = "●" if int(elapsed * 2) % 2 == 0 else "○"

        try:
            label = self.query_one("#teach-recording-timer", Label)
            label.update(
                f"[bold red]{dots} REC[/] "
                f"[white]{elapsed:.1f}s[/] "
                f"[dim]| {wp_count} WPs | {RECORD_HZ}Hz[/]"
            )
        except NoMatches:
            pass

    # ============================================================
    # ACTIONS (Keyboard Shortcuts)
    # ============================================================

    def action_switch_tab(self, tab_id: str) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = tab_id
        except NoMatches:
            pass

    def action_connect(self) -> None:
        if self.connected:
            self._disconnect()
        else:
            port = find_arm_port()
            if port:
                self._do_connect(port)
            else:
                self._log_teach("[red]❌ Kein Port gefunden![/]")

    def _disconnect(self):
        if self._arm:
            self._arm.close()
            self._arm = None
        self.connected = False
        self._log_teach("[yellow]🔌 Getrennt[/]")
        self._update_status_connection(None)

        # Re-enable simulation mode
        self._enable_simulation_mode()

    def action_torque_release(self) -> None:
        """Torque lösen (Taste t)."""
        arm = self._active_arm
        if arm is None:
            self._log_teach("[red]Nicht verbunden und keine Simulation![/]")
            return

        arm.torque_off()
        self.torque_on_state = False
        self._update_status_torque(False)

        if self._is_sim:
            self._log_teach("[yellow]🔓 Torque AUS (Sim) — Arm ist frei[/]")
        else:
            self._log_teach("[yellow]🔓 Torque AUS — Arm ist frei bewegbar[/]")
        self._log_servo("[yellow]🔓 Torque AUS[/]")

    def action_torque_lock(self) -> None:
        """Torque einschalten (Taste T/Shift+t)."""
        arm = self._active_arm
        if arm is None:
            self._log_teach("[red]Nicht verbunden und keine Simulation![/]")
            return

        arm.torque_on()
        self.torque_on_state = True
        self._update_status_torque(True)

        if self._is_sim:
            self._log_teach("[green]🔒 Torque AN (Sim) — Arm ist fixiert[/]")
        else:
            self._log_teach("[green]🔒 Torque AN — Arm ist fixiert[/]")
        self._log_servo("[green]🔒 Torque AN[/]")


    def action_toggle_action(self) -> None:
        """Start/Stop je nach aktivem Tab."""
        try:
            tabs = self.query_one(TabbedContent)
            active = tabs.active
        except NoMatches:
            return

        if active == "teach":
            if self.recording:
                self._stop_recording()
            else:
                self._start_recording()
        elif active == "play":
            if self.playing:
                self._stop_playback()
            else:
                self._start_playback()
        elif active == "calibrate":
            self._start_calibration()

    def action_led_toggle(self) -> None:
        """Toggles LED and records the event during teach."""
        arm = self._active_arm
        if arm is None:
            return
        self._led_on = not getattr(self, '_led_on', False)
        brightness = 255 if self._led_on else 0
        if hasattr(arm, 'send_cmd'):
            arm.send_cmd({"T": 114, "led": brightness})
        self._log_teach(f"[bold]💡 LED {'AN' if self._led_on else 'AUS'}[/]")
        if self.recording:
            elapsed = time.time() - self._teach_start_time
            cmd = "LED_ON" if self._led_on else "LED_OFF"
            self._teach_waypoints.append({"t": round(elapsed, 4), "cmd": cmd})


    def action_gripper_toggle(self) -> None:
        arm = self._active_arm
        if arm is None:
            return

        if self._gripper_open:
            arm.gripper_close()
            self._gripper_open = False
            self._log_teach("[bold]✊ Gripper ZU[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append(
                    {"t": round(elapsed, 4), "cmd": "GRIPPER_CLOSE"}
                )
        else:
            arm.gripper_open()
            self._gripper_open = True
            self._log_teach("[bold]✋ Gripper AUF[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append(
                    {"t": round(elapsed, 4), "cmd": "GRIPPER_OPEN"}
                )

    def action_go_home(self) -> None:
        self._go_home()

    def action_read_position(self) -> None:
        """Liest die aktuelle Position und zeigt sie an."""
        arm = self._active_arm
        if arm is None:
            self._log_servo("[red]Nicht verbunden und keine Simulation![/]")
            return

        pos = arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self._update_joint_displays(pos)
            self._update_arm_views(pos)
            self._update_servo_readouts(pos)
            sim_tag = " [dim](sim)[/]" if self._is_sim else ""
            self._log_servo(
                f"[green]📍 Position:{sim_tag}[/] b={pos['b']:+.2f}° "
                f"s={pos['s']:+.2f}° e={pos['e']:+.2f}° h={pos['h']:+.2f}°"
            )
        else:
            self._log_servo("[red]❌ Konnte Position nicht lesen[/]")

    def action_rotate_left(self) -> None:
        """3D-Ansicht nach links rotieren."""
        self._rotate_all_views(d_azimuth=-15)

    def action_rotate_right(self) -> None:
        """3D-Ansicht nach rechts rotieren."""
        self._rotate_all_views(d_azimuth=15)

    def action_rotate_up(self) -> None:
        """3D-Ansicht Elevation erhöhen."""
        self._rotate_all_views(d_elevation=10)

    def action_rotate_down(self) -> None:
        """3D-Ansicht Elevation verringern."""
        self._rotate_all_views(d_elevation=-10)

    def _rotate_all_views(self, d_azimuth: float = 0, d_elevation: float = 0):
        """Rotiert alle 3D-Views."""
        for view_id in ["teach-arm-view", "play-arm-view",
                        "calibrate-arm-view", "servo-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", Arm3DWidget)
                widget.rotate(d_azimuth, d_elevation)
            except NoMatches:
                pass

    # ============================================================
    # TEACH MODE
    # ============================================================

    @on(Button.Pressed, "#btn-teach-record")
    def on_teach_record(self) -> None:
        self._start_recording()

    @on(Button.Pressed, "#btn-teach-stop")
    def on_teach_stop(self) -> None:
        self._stop_recording()

    @on(Button.Pressed, "#btn-teach-home")
    def on_teach_home(self) -> None:
        self._go_home()

    @on(Button.Pressed, "#btn-gripper")
    def on_gripper_press(self) -> None:
        self.action_gripper_toggle()

    def _start_recording(self):
        arm = self._active_arm
        if arm is None:
            self._log_teach("[red]Nicht verbunden und keine Simulation![/]")
            return

        self.recording = True
        self._teach_waypoints = []
        self._teach_start_time = time.time()
        self._gripper_open = True

        # Torque aus (in sim mode, allows "virtual free movement")
        arm.torque_off()
        self.torque_on_state = False
        self._update_status_torque(False)

        # Buttons updaten
        try:
            self.query_one("#btn-teach-record", Button).disabled = True
            self.query_one("#btn-teach-stop", Button).disabled = False
        except NoMatches:
            pass

        # Trail löschen
        try:
            self.query_one("#teach-arm-view", Arm3DWidget).clear_trail()
        except NoMatches:
            pass

        if self._is_sim:
            self._log_teach("[bold red]⏺ AUFNAHME LÄUFT (Simulation)[/]")
            self._log_teach(
                "[dim]Simulierte Bewegung: Arm bewegt sich automatisch "
                "auf einer Demo-Trajektorie.[/]"
            )
            self._log_teach("[dim][Space]=Stop [g]=Gripper[/]")
            # Start simulated movement demo
            self._sim_recording_start_time = time.time()
        else:
            self._log_teach("[bold red]⏺ AUFNAHME LÄUFT[/]")
            self._log_teach("[dim]Bewege den Arm! [Space]=Stop [g]=Gripper [t]=Torque[/]")

        # Start activity indicator
        self._start_activity("Recording", "🔴")

        # Start recording elapsed timer
        self._start_recording_timer()

        # Timer starten
        self._teach_timer = self.set_interval(
            1.0 / RECORD_HZ, self._teach_poll_position
        )

        self.add_class("recording-active")

    def _teach_poll_position(self):
        """Polls position during teach, with optional gravity compensation."""
        if not self.recording:
            return
        arm = self._active_arm
        if arm is None:
            return
        if self._is_sim:
            pos = self._generate_sim_teach_position()
        else:
            pos = self._teach_read_position(arm)
            if pos is None:
                return
        self._current_pos = pos
        elapsed = time.time() - self._teach_start_time
        if self._should_record_waypoint(pos):
            self._teach_waypoints.append({
                "t": round(elapsed, 4),
                "b": pos["b"], "s": pos["s"],
                "e": pos["e"], "h": pos["h"],
            })
        self._update_joint_displays(pos)
        self._update_arm_views(pos)
        self._teach_log_periodic(pos, elapsed)

    def _generate_sim_teach_position(self) -> dict:
        """Generates simulated teach position (figure-8 demo)."""
        elapsed = time.time() - self._teach_start_time
        pos = {
            "b": 30.0 * math.sin(elapsed * 0.8),
            "s": 20.0 * math.sin(elapsed * 0.5) + 10.0,
            "e": 90.0 + 25.0 * math.sin(elapsed * 0.6),
            "h": 180.0 + 15.0 * math.sin(elapsed * 0.3),
        }
        with self._sim_arm._lock:
            self._sim_arm._position = pos.copy()
        return pos

    def _should_record_waypoint(self, pos: dict) -> bool:
        """Checks if position changed enough to record a new waypoint."""
        if not self._teach_waypoints:
            return True
        last_move = None
        for wp in reversed(self._teach_waypoints):
            if "cmd" not in wp:
                last_move = wp
                break
        if last_move is None:
            return True
        max_delta = max(abs(pos[j] - last_move[j]) for j in ["b", "s", "e", "h"])
        return max_delta >= MOVE_THRESHOLD_DEG

    def _teach_log_periodic(self, pos: dict, elapsed: float):
        """Logs teach status every 50 waypoints."""
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if len(move_wps) % 50 == 0 and len(move_wps) > 0:
            sim_tag = " (sim)" if self._is_sim else ""
            self._log_teach(
                f"[dim]  ◆ WP#{len(move_wps)}{sim_tag} [{elapsed:.1f}s] "
                f"b={pos['b']:+.1f}° s={pos['s']:+.1f}° "
                f"e={pos['e']:+.1f}° h={pos['h']:+.1f}°[/]"
            )

    def _stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self._teach_timer:
            self._teach_timer.stop()
            self._teach_timer = None

        arm = self._active_arm
        if arm:
            arm.torque_on()
            self.torque_on_state = True
            self._update_status_torque(True)

        try:
            self.query_one("#btn-teach-record", Button).disabled = False
            self.query_one("#btn-teach-stop", Button).disabled = True
        except NoMatches:
            pass

        # Stop activity indicator and recording timer
        self._stop_activity("✅ Recording saved")
        self._stop_recording_timer()

        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if not move_wps:
            self._log_teach("[yellow]Keine Wegpunkte aufgezeichnet![/]")
            self._stop_activity()
            return

        duration = move_wps[-1]["t"]
        sim_tag = " (Simulation)" if self._is_sim else ""
        self._log_teach(
            f"[green]⏹ Aufnahme gestoppt{sim_tag}: "
            f"{len(move_wps)} WPs, {duration:.1f}s[/]"
        )

        filepath = self._save_recording()
        if filepath:
            self._log_teach(f"[green]💾 Gespeichert: {filepath}[/]")
            self._refresh_recordings_table()

        # Clear the final message after 3 seconds
        self.set_timer(3.0, lambda: self._stop_activity())

        self.remove_class("recording-active")

    def _format_waypoint_line(self, wp: dict) -> str:
        """Formats a single waypoint or command as a .roarm file line."""
        if "cmd" not in wp:
            return (f"MOVE b={wp['b']:.2f} s={wp['s']:.2f} "
                    f"e={wp['e']:.2f} h={wp['h']:.2f} t={wp['t']:.4f}")
        cmd = wp["cmd"]
        if cmd == "GRIPPER_CLOSE":
            return f"GRIPPER CLOSE t={wp['t']:.4f}"
        elif cmd == "GRIPPER_OPEN":
            return f"GRIPPER OPEN t={wp['t']:.4f}"
        elif cmd == "LED_ON":
            return f"LED ON t={wp['t']:.4f}"
        elif cmd == "LED_OFF":
            return f"LED OFF t={wp['t']:.4f}"
        return ""

    def _build_recording_header(self, move_wps: list) -> list:
        """Builds the header lines for a .roarm recording file."""
        return [
            f"# RoArm-M2-S Recording (Dashboard v3)",
            f"# Datum: {datetime.now().isoformat()}",
            f"# Wegpunkte: {len(move_wps)}",
            f"# Dauer: {move_wps[-1]['t']:.2f}s",
            f"#",
            f"#CONFIG hz={RECORD_HZ}",
            f"#CONFIG threshold={MOVE_THRESHOLD_DEG}",
            f"#CONFIG gravity_comp={'1' if self._gravity_comp_enabled else '0'}",
            f"#START_POS b={START_POSITION_DEG['b']:.2f} "
            f"s={START_POSITION_DEG['s']:.2f} "
            f"e={START_POSITION_DEG['e']:.2f} "
            f"h={START_POSITION_DEG['h']:.2f}",
            "",
        ]


    def _save_recording(self) -> Optional[str]:
        """Saves recording with LED events and gravity comp config."""
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if not move_wps:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RECORDINGS_DIR / f"recording_{ts}.roarm"
        lines = self._build_recording_header(move_wps)
        for wp in self._teach_waypoints:
            line = self._format_waypoint_line(wp)
            if line:
                lines.append(line)
        with open(filename, 'w') as f:
            f.write("\n".join(lines) + "\n")
        return str(filename)

    @work(thread=True)
    def _go_home(self):
        arm = self._active_arm
        if arm is None:
            return

        # Start activity indicator
        self.app.call_from_thread(
            self._start_activity, "Homing", "🏠"
        )
        self.app.call_from_thread(
            self._log_teach, "[dim]🏠 Fahre zur Home-Position...[/]"
        )

        arm.torque_on()
        self.torque_on_state = True
        self.app.call_from_thread(self._update_status_torque, True)
        time.sleep(0.2)

        arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=25, acc=12
        )

        # Wait for movement to complete
        if self._is_sim:
            # In simulation, wait for the sim to reach target
            for _ in range(100):  # max 2 seconds
                time.sleep(0.02)
                self._sim_arm.step_simulation(0.02)
                pos = self._sim_arm.read_position_deg()
                self.app.call_from_thread(self._update_joint_displays, pos)
                self.app.call_from_thread(self._update_arm_views, pos)
                if not self._sim_arm.is_moving:
                    break
        else:
            time.sleep(2.0)

        pos = arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self.app.call_from_thread(self._update_joint_displays, pos)
            self.app.call_from_thread(self._update_arm_views, pos)

        # Stop activity indicator with success message
        self.app.call_from_thread(
            self._stop_activity, "✅ Home reached"
        )
        self.app.call_from_thread(
            self._log_teach, "[green]✅ Home-Position erreicht[/]"
        )

        # Clear the success message after 3 seconds
        self.app.call_from_thread(
            self.set_timer, 3.0, lambda: self._stop_activity()
        )


    # ============================================================
    # PLAY MODE
    # ============================================================

    @on(Button.Pressed, "#btn-play-start")
    def on_play_start(self) -> None:
        self._start_playback()

    @on(Button.Pressed, "#btn-play-stop")
    def on_play_stop(self) -> None:
        self._stop_playback()

    def _start_playback(self):
        arm = self._active_arm
        if arm is None:
            self._log_play("[red]Nicht verbunden und keine Simulation![/]")
            return

        try:
            table = self.query_one("#recordings-table", DataTable)

            if table.row_count == 0:
                self._log_play("[yellow]⚠ Keine Recordings vorhanden![/]")
                self._log_play(
                    "[dim]  → Nimm zuerst ein Recording im Teach-Tab auf "
                    "(Tab 1, Space)[/]"
                )
                return

            row_key = table.cursor_row
            if row_key is None:
                self._log_play(
                    "[yellow]⚠ Kein Recording ausgewählt! "
                    "Wähle eine Zeile in der Tabelle.[/]"
                )
                return
            row_data = table.get_row_at(row_key)
            filename = row_data[0]
        except NoMatches:
            self._log_play("[red]❌ Recordings-Tabelle nicht gefunden![/]")
            return
        except Exception as e:
            self._log_play(f"[red]❌ Konnte Recording nicht laden: {e}[/]")
            return

        filepath = RECORDINGS_DIR / filename
        if not filepath.exists():
            self._log_play(f"[red]Datei nicht gefunden: {filepath}[/]")
            return

        self._play_data = parse_roarm_file(str(filepath))
        wps = self._play_data["waypoints"]
        if not wps or len(wps) < 4:
            self._log_play("[red]Zu wenige Wegpunkte für Spline![/]")
            return

        # Timeline updaten
        try:
            timeline = self.query_one("#play-timeline", TimelineWidget)
            timeline.set_recording(wps, self._play_data.get("gripper_cmds"))
        except NoMatches:
            pass

        self.playing = True

        try:
            self.query_one("#btn-play-start", Button).disabled = True
            self.query_one("#btn-play-stop", Button).disabled = False
        except NoMatches:
            pass

        # Start activity indicator
        self._start_activity("Starting playback", "▶️")

        sim_tag = " (Simulation)" if self._is_sim else ""
        self._log_play(
            f"[green]▶ Playback{sim_tag}: {len(wps)} WPs, {wps[-1]['t']:.1f}s[/]"
        )

        self._run_playback(wps)

    def _load_calibration_model(self, is_sim: bool):
        """Loads calibration model if available."""
        if is_sim:
            return None
        try:
            from calibrate import CalibrationModel
            cal_path = Path("calibration") / "roarm_calibration.cal"
            if cal_path.exists():
                model = CalibrationModel.load(str(cal_path))
                self.app.call_from_thread(
                    self._log_play, f"[green]📂 Kalibrierung geladen: {cal_path}[/]"
                )
                return model
        except Exception as e:
            self.app.call_from_thread(
                self._log_play, f"[yellow]⚠ Kalibrierung nicht verfügbar: {e}[/]"
            )
        return None

    def _send_if_changed(self, arm, corrected: dict, last_pos: Optional[dict],
                         commands_sent: int, skipped: int,
                         rate_limiter, is_sim: bool, interval: float) -> tuple:
        """Sends command only if position changed enough. Returns (sent, cmds, skips)."""
        if last_pos:
            max_delta = max(abs(corrected[j] - last_pos[j]) for j in ["b", "s", "e", "h"])
            if max_delta < MIN_DELTA_DEG:
                return False, commands_sent, skipped + 1
        rate_limiter.acquire()
        arm.move_to_fast(
            corrected["b"], corrected["s"],
            corrected["e"], corrected["h"],
            spd=50, acc=30
        )
        if is_sim:
            self._sim_arm.step_simulation(interval)
        return True, commands_sent + 1, skipped

    def _update_playback_ui(self, arm, corrected: dict, elapsed: float, is_sim: bool):
        """Updates UI during playback (called from worker thread)."""
        if is_sim:
            pos = self._sim_arm.read_position_deg()
            self.app.call_from_thread(self._update_arm_views, pos)
            self.app.call_from_thread(self._update_joint_displays, pos)
        else:
            self.app.call_from_thread(self._update_arm_views, corrected)
        self.app.call_from_thread(self._update_play_timeline, elapsed)

    def _finalize_playback(self, arm, is_sim: bool, cal_model, duration: float):
        """Finalizes playback: verify endpoint, log summary, reset UI."""
        self.playing = False
        # Endpoint verification (real arm only)
        if not is_sim and self._last_play_commanded:
            final_target = self._last_play_commanded
            err = self._verify_endpoint(arm, final_target)
        # Stop activity
        self.app.call_from_thread(self._stop_activity, "✅ Playback complete")
        self.app.call_from_thread(self._playback_finished)
        self.app.call_from_thread(
            self.set_timer, 3.0, lambda: self._stop_activity()
        )

    def _arm_is_estopped(self) -> bool:
        """Checks if the safety layer has triggered an emergency stop."""
        if self._safe_arm and hasattr(self._safe_arm, 'is_emergency_stopped'):
            return self._safe_arm.is_emergency_stopped
        return False

    def _move_via_safe_up(self, arm, target_pose: dict):
        """Moves arm to target via safe-up position (collision avoidance)."""
        from calibrate import move_to_safe_up, move_from_safe_up_to_pose
        current = arm.read_position_deg()
        if current:
            move_to_safe_up(arm, current_pose=current)
        else:
            move_to_safe_up(arm, current_pose=None)
        move_from_safe_up_to_pose(arm, target_pose)


    def _run_loop(self, waypoints: list, arm, is_sim: bool):
        """Runs playback in loop mode with configurable pause."""
        pause_s = self._get_loop_pause()
        while self._is_loop_enabled() and not self._arm_is_estopped():
            if pause_s > 0:
                self.app.call_from_thread(
                    self._log_play, f"[dim]⏸ Loop-Pause: {pause_s:.1f}s[/]")
                time.sleep(pause_s)
            self.playing = True
            cal_model = self._load_calibration_model(is_sim)
            trajectory = SmoothTrajectory(waypoints, self._get_play_speed())
            duration = trajectory.get_duration()
            self._move_to_start_position(arm, waypoints[0], cal_model, is_sim)
            events = sorted(self._play_data.get("events", []), key=lambda x: x["t"])
            self._streaming_loop(arm, trajectory, duration, cal_model, events, is_sim)
            if self.playing:
                self._do_precision_endpoint(arm, trajectory, duration, cal_model, is_sim)
            self.playing = False


    def _do_precision_endpoint(self, arm, trajectory: 'SmoothTrajectory',
                               duration: float, cal_model, is_sim: bool):
        """Executes precision endpoint after streaming completes."""
        self.app.call_from_thread(self._start_activity, "Precision settle", "🎯")
        final_target = trajectory.sample(duration)
        final_corrected = self._apply_calibration_static(cal_model, final_target)
        if is_sim:
            self._precision_endpoint_sim(arm, final_corrected)
        else:
            self._precision_endpoint_real(arm, final_corrected)

    def _streaming_loop(self, arm, trajectory: 'SmoothTrajectory',
                        duration: float, cal_model, events: list, is_sim: bool):
        """Main streaming loop: samples trajectory and sends commands."""
        from safety import RateLimiter
        interval = 1.0 / STREAM_HZ
        rate_limiter = RateLimiter(max_hz=STREAM_HZ + 10)
        playback_start = time.time()
        last_pos = None
        commands_sent = 0
        skipped = 0
        event_idx = 0
        self._last_play_commanded = None

        while self.playing:
            loop_start = time.time()
            elapsed = loop_start - playback_start
            if elapsed >= duration:
                break
            # Events (gripper, LED)
            event_idx, pause = self._process_pending_events(
                arm, events, event_idx, elapsed)
            if pause > 0:
                playback_start += pause
                elapsed = time.time() - playback_start
                if elapsed >= duration:
                    break
            # Safety checks (real arm only)
            if not is_sim and self._current_monitor:
                target = trajectory.sample(elapsed)
                corrected = self._apply_calibration_static(cal_model, target)
                err = self._check_tracking_error(arm, corrected, commands_sent)
                if not self.playing:
                    break
            # Sample + send
            target = trajectory.sample(elapsed)
            corrected = self._apply_calibration_static(cal_model, target)
            sent, commands_sent, skipped = self._send_if_changed(
                arm, corrected, last_pos, commands_sent, skipped,
                rate_limiter, is_sim, interval
            )
            if sent:
                last_pos = corrected.copy()
                self._last_play_commanded = corrected.copy()
            # UI update (every 200ms)
            if commands_sent % max(1, STREAM_HZ // 5) == 0:
                self._update_playback_ui(arm, corrected, elapsed, is_sim)
            # Timing
            sleep_time = interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


    def _move_to_start_position(self, arm, first_wp: dict, cal_model, is_sim: bool):
        """Moves arm to the first waypoint before playback starts."""
        start_corrected = self._apply_calibration_static(cal_model, first_wp)
        arm.torque_on()
        time.sleep(0.2)
        arm.move_to(
            start_corrected["b"], start_corrected["s"],
            start_corrected["e"], start_corrected["h"],
            spd=20, acc=10
        )
        if is_sim:
            for _ in range(150):
                time.sleep(0.02)
                self._sim_arm.step_simulation(0.02)
                if not self._sim_arm.is_moving:
                    break
        else:
            time.sleep(2.0)


    @work(thread=True)
    def _run_playback(self, waypoints: list):
        """Full playback with SmoothTrajectory, safety, tracking, precision endpoint."""
        arm = self._active_arm
        if arm is None:
            self.app.call_from_thread(self._log_play, "[red]Kein Arm![/]")
            self.playing = False
            return
        is_sim = self._is_sim
        speed = self._get_play_speed()
        cal_model = self._load_calibration_model(is_sim)
        trajectory = SmoothTrajectory(waypoints, speed)
        # Pre-flight check
        if not is_sim:
            ok, violations = self._validate_trajectory(trajectory)
            if not ok:
                trajectory = self._attempt_trajectory_repair(trajectory, violations)
                if trajectory is None:
                    self.app.call_from_thread(
                        self._log_play, "[red]❌ Trajektorie unsicher![/]")
                    self.playing = False
                    return
        duration = trajectory.get_duration()
        self._move_to_start_position(arm, waypoints[0], cal_model, is_sim)
        self.app.call_from_thread(
            self._start_activity, f"Playing ({duration:.1f}s)", "▶️")
        events = sorted(self._play_data.get("events", []), key=lambda x: x["t"])
        self._streaming_loop(arm, trajectory, duration, cal_model, events, is_sim)
        if self.playing:
            self._do_precision_endpoint(arm, trajectory, duration, cal_model, is_sim)
        self._finalize_playback(arm, is_sim, cal_model, duration)
        # Loop handling
        if self._is_loop_enabled():
            self._run_loop(waypoints, arm, is_sim)

    def _update_play_timeline(self, elapsed: float):
        """Aktualisiert die Timeline-Position."""
        try:
            timeline = self.query_one("#play-timeline", TimelineWidget)
            timeline.set_position(elapsed)
        except NoMatches:
            pass

    def _playback_finished(self):
        """Wird nach dem Playback aufgerufen."""
        try:
            self.query_one("#btn-play-start", Button).disabled = False
            self.query_one("#btn-play-stop", Button).disabled = True
        except NoMatches:
            pass

    def _precision_endpoint_real(self, arm, final_corrected: dict):
        """Drives to endpoint with multiple passes for precision."""
        for spd, acc in ENDPOINT_SPEEDS:
            arm.move_to(
                final_corrected["b"], final_corrected["s"],
                final_corrected["e"], final_corrected["h"],
                spd=spd, acc=acc
            )
            time.sleep(ENDPOINT_SETTLE_WAIT)

    def _precision_endpoint_sim(self, arm, final_corrected: dict):
        """Precision endpoint for simulation mode."""
        arm.move_to(
            final_corrected["b"], final_corrected["s"],
            final_corrected["e"], final_corrected["h"],
            spd=5, acc=2
        )
        for _ in range(100):
            time.sleep(0.02)
            self._sim_arm.step_simulation(0.02)
            if not self._sim_arm.is_moving:
                break

    def _verify_endpoint(self, arm, final_target: dict) -> Optional[float]:
        """Reads final position and returns max error in degrees."""
        time.sleep(0.3)
        pos = arm.read_position_deg()
        if pos is None:
            return None
        err = max(abs(pos[j] - final_target[j]) for j in ["b", "s", "e", "h"])
        self.app.call_from_thread(
            self._log_play,
            f"[dim]  Endposition: Fehler={err:.3f}° "
            f"(b={pos['b']:.2f} s={pos['s']:.2f} e={pos['e']:.2f})[/]"
        )
        return err

    def _graceful_stop(self, arm_raw, last_commanded: dict):
        """Executes a graceful stop instead of hard torque-off."""
        from safety import GracefulStop
        if last_commanded and arm_raw:
            GracefulStop.execute(arm_raw, last_commanded)


    def _stop_playback(self):
        """Stops playback gracefully."""
        self.playing = False
        if self._arm and hasattr(self, '_last_play_commanded'):
            self._graceful_stop(self._arm, self._last_play_commanded)
        self._stop_activity("⏹ Stopped")
        self._log_play("[yellow]⏹ Playback gestoppt (graceful)[/]")
        try:
            self.query_one("#btn-play-start", Button).disabled = False
            self.query_one("#btn-play-stop", Button).disabled = True
        except NoMatches:
            pass
        self.set_timer(3.0, lambda: self._stop_activity())

    # ============================================================
    # CALIBRATE MODE
    # ============================================================

    @on(Button.Pressed, "#btn-cal-start")
    def on_cal_start(self) -> None:
        self._start_calibration()

    @on(Button.Pressed, "#btn-cal-abort")
    def on_cal_abort(self) -> None:
        self._abort_calibration()

    @on(Button.Pressed, "#btn-cal-load")
    def on_cal_load(self) -> None:
        self._load_calibration()

    def _start_calibration(self):
        """Startet die Kalibrierung."""
        arm = self._active_arm
        if arm is None:
            self._log_calibrate("[red]Nicht verbunden und keine Simulation![/]")
            return

        if self._is_sim:
            self._log_calibrate(
                "[yellow]⚠ Kalibrierung im Simulationsmodus nicht sinnvoll![/]"
            )
            self._log_calibrate(
                "[dim]  Verbinde einen echten Roboter für die Kalibrierung.[/]"
            )
            return

        if not self.connected or not self._arm:
            self._log_calibrate("[red]Nicht verbunden![/]")
            return

        try:
            pose_set_select = self.query_one("#cal-pose-set", Select)
            pose_set = pose_set_select.value or "standard"
        except NoMatches:
            pose_set = "standard"

        try:
            repeats_input = self.query_one("#cal-repeats", Input)
            repeats = int(repeats_input.value) if repeats_input.value else 3
        except (NoMatches, ValueError):
            repeats = 3

        try:
            auto_switch = self.query_one("#cal-auto-accept", Switch)
            auto_accept = auto_switch.value
        except NoMatches:
            auto_accept = True

        # Buttons updaten
        try:
            self.query_one("#btn-cal-start", Button).disabled = True
            self.query_one("#btn-cal-abort", Button).disabled = False
        except NoMatches:
            pass

        # Start activity indicator
        self._start_activity("Calibrating", "🎯")

        self._log_calibrate(
            f"[bold green]🎯 Kalibrierung gestartet[/]\n"
            f"  Pose-Set: {pose_set}\n"
            f"  Wiederholungen: {repeats}\n"
            f"  Auto-Accept: {'Ja' if auto_accept else 'Nein'}"
        )

        self._run_calibration_worker(pose_set, repeats, auto_accept)

    def _init_calibration_diagnostics(self, repeats: int, pose_set: str) -> dict:
        """Initializes the diagnostics dict for calibration."""
        return {
            "settle_times_s": [],
            "overshoot_deg": [],
            "noise_std_deg": [],
            "per_pose": [],
            "repeats_per_pose": repeats,
            "pose_set": pose_set,
            "total_measurements": 0,
            "repeatability_per_pose": [],
        }

    def _cal_log(self, msg: str):
        """Thread-safe calibration log."""
        self.app.call_from_thread(self._log_calibrate, msg)

    def _log_cal_pose_validation(self, valid: list, all_poses: list):
        """Logs pose validation results."""
        skipped = len(all_poses) - len(valid)
        if skipped > 0:
            self._cal_log(f"[yellow]⚠ {skipped} Posen übersprungen (außerhalb Grenzen)[/]")
        if len(valid) < 10:
            self._cal_log(f"[yellow]⚠ Nur {len(valid)} gültige Posen![/]")

    def _cal_safe_up_between(self):
        """Moves to safe-up between calibration poses."""
        from calibrate import move_to_safe_up
        current = self._arm.read_position_deg()
        if current:
            move_to_safe_up(self._arm, current_pose=current)
        else:
            move_to_safe_up(self._arm, current_pose=None)

    def _cal_show_diagnostics(self, diagnostics: dict, residuals: dict,
                              poses: list, repeats: int):
        """Shows calibration diagnostics tables in the log."""
        from calibrate import JOINTS
        # Residuals
        self._cal_log("\n[bold cyan]📊 Kalibrierungs-Ergebnis:[/]")
        for j in JOINTS:
            r = residuals.get(j, 0)
            q = "✅" if r < 0.3 else "⚠️" if r < 1.0 else "❌"
            self._cal_log(f"  {j.upper()}: RMS={r:.4f}° {q}")
        # Settle times
        if diagnostics["settle_times_s"]:
            arr = np.array(diagnostics["settle_times_s"])
            self._cal_log(
                f"  Settle: min={arr.min():.2f}s max={arr.max():.2f}s "
                f"avg={arr.mean():.2f}s"
            )
        # Overshoot
        if diagnostics["overshoot_deg"]:
            arr = np.array(diagnostics["overshoot_deg"])
            self._cal_log(f"  Overshoot: max={arr.max():.3f}° avg={arr.mean():.3f}°")
        # Repeatability
        if diagnostics["repeatability_per_pose"]:
            for j in JOINTS:
                vals = [r["repeat_std_deg"][j] for r in diagnostics["repeatability_per_pose"]]
                self._cal_log(f"  Repeatability {j.upper()}: σ={np.mean(vals):.4f}°")


    def _cal_save_results(self, model, diagnostics: dict, residuals: dict,
                          measurement_count: int, poses: list, repeats: int):
        """Saves calibration model and diagnostics JSON."""
        cal_path = Path("calibration") / "roarm_calibration.cal"
        cal_path.parent.mkdir(exist_ok=True)
        diagnostics["total_measurements"] = measurement_count
        model.save(str(cal_path), diagnostics=diagnostics)
        # Save diagnostics JSON
        diag_path = Path("calibration") / "roarm_diagnostics.json"
        with open(diag_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        self._cal_log(f"[green]✅ Kalibrierung gespeichert: {cal_path}[/]")
        self._cal_log(f"[green]✅ Diagnostik gespeichert: {diag_path}[/]")


    def _cal_fit_model(self, commanded: list, errors: list, repeats: int) -> tuple:
        """Fits the calibration model. Returns (model, residuals)."""
        from calibrate import CalibrationModel
        self._cal_log("\n[bold]📊 Fitte Kalibrierungsmodell...[/]")
        model = CalibrationModel()
        residuals = model.fit(commanded, errors)
        return model, residuals


    def _cal_repeatability_test(self, poses: list, diagnostics: dict):
        """Runs final repeatability test (home → ... → home)."""
        from calibrate import move_from_safe_up_to_pose, JOINTS
        self._cal_log("[dim]  🔄 Repeatability-Test (Home nochmal)...[/]")
        move_from_safe_up_to_pose(self._arm, poses[0])
        self._arm.move_to(poses[0]["b"], poses[0]["s"], poses[0]["e"], poses[0]["h"], spd=5, acc=3)
        self._arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
        repeat_pos = self._arm.read_position_averaged(n=10, interval=0.05)
        if repeat_pos and diagnostics["per_pose"]:
            first_measured = diagnostics["per_pose"][0].get("measured", {}) if diagnostics["per_pose"] else {}
            if first_measured:
                repeat_err = {j: abs(repeat_pos[j] - first_measured.get(j, repeat_pos[j]))
                              for j in JOINTS}
                diagnostics["repeatability_deg"] = repeat_err
                self._cal_log(
                    f"[dim]  🔄 Home→...→Home: Δb={repeat_err['b']:.3f}° "
                    f"Δs={repeat_err['s']:.3f}° Δe={repeat_err['e']:.3f}°[/]"
                )


    def _cal_aggregate_pose(self, pose: dict, measurements: list,
                            commanded: list, errors: list,
                            diagnostics: dict, repeats: int):
        """Aggregates measurements for a single pose into commanded/errors."""
        from calibrate import JOINTS
        if not measurements:
            return
        avg_error = {j: float(np.mean([m["error"][j] for m in measurements]))
                     for j in JOINTS}
        commanded.append(pose)
        errors.append(avg_error)
        # Repeatability
        if repeats > 1:
            repeat_std = {j: float(np.std([m["measured"][j] for m in measurements]))
                          for j in JOINTS}
            diagnostics["repeatability_per_pose"].append({
                "pose_index": len(commanded) - 1,
                "repeat_std_deg": repeat_std,
            })


    def _cal_compute_overshoot(self, settle_result: dict) -> float:
        """Computes max overshoot from settle readings."""
        from calibrate import JOINTS
        readings = settle_result.get("readings", [])
        final_pos = settle_result.get("pos")
        if not readings or not final_pos:
            return 0.0
        max_overshoot = 0.0
        for reading in readings:
            for j in JOINTS:
                if j in reading:
                    overshoot = abs(reading[j] - final_pos[j])
                    max_overshoot = max(max_overshoot, overshoot)
        return round(max_overshoot, 3)


    def _cal_measure_single_pose(self, pose: dict, pose_idx: int,
                                 rep: int, repeats: int, diagnostics: dict) -> Optional[dict]:
        """Measures a single pose. Returns error dict or None."""
        from calibrate import move_from_safe_up_to_pose, JOINTS
        move_start = time.time()
        move_from_safe_up_to_pose(self._arm, pose)
        # Precision settle
        self._arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
        result = self._arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
        settle_time = time.time() - move_start
        diagnostics["settle_times_s"].append(settle_time)
        # Overshoot
        overshoot = self._cal_compute_overshoot(result)
        diagnostics["overshoot_deg"].append(overshoot)
        # Read averaged position
        servo_avg = self._arm.read_position_averaged(n=10, interval=0.05)
        if not servo_avg:
            return None
        # Noise
        diagnostics["noise_std_deg"].append({
            j: servo_avg.get(f"{j}_std", 0) for j in JOINTS
        })
        error = {j: servo_avg[j] - pose[j] for j in JOINTS}
        # Update arm view
        self.app.call_from_thread(self._update_arm_views, {
            "b": servo_avg["b"], "s": servo_avg["s"],
            "e": servo_avg["e"], "h": servo_avg.get("h", 180.0)
        })
        self._cal_log(
            f"[dim]  ✓ Pose {pose_idx+1} Rep {rep+1}: "
            f"Δb={error['b']:+.2f}° Δs={error['s']:+.2f}° Δe={error['e']:+.2f}°[/]"
        )
        return {"error": error, "measured": {j: servo_avg[j] for j in JOINTS}}


    def _cal_update_progress(self, pose_idx: int, n_poses: int,
                             rep: int, repeats: int, count: int, total: int):
        """Updates calibration progress in UI."""
        pct = count / total * 100
        self.app.call_from_thread(
            self._update_cal_status,
            f"Pose {pose_idx+1}/{n_poses} · Rep {rep+1}/{repeats} · {pct:.0f}%"
        )
        self.app.call_from_thread(
            self._start_activity,
            f"Cal {pct:.0f}% P{pose_idx+1}/{n_poses}", "🎯"
        )


    @work(thread=True)
    def _run_calibration_worker(self, pose_set: str, repeats: int,
                                auto_accept: bool):
        """Full calibration with diagnostics, repeatability, overshoot."""
        from calibrate import (
            CalibrationModel, POSE_SETS, JOINTS,
            move_to_safe_up, move_from_safe_up_to_pose, validate_pose,
            run_manual_verification, integrate_manual_points,
        )
        poses = POSE_SETS.get(pose_set, POSE_SETS["standard"])
        valid_poses = [p for p in poses if validate_pose(p)]
        self._log_cal_pose_validation(valid_poses, poses)
        total = len(valid_poses) * repeats
        commanded, errors = [], []
        diagnostics = self._init_calibration_diagnostics(repeats, pose_set)
        self._arm.torque_on()
        time.sleep(0.2)
        self._cal_log("[dim]  Fahre zu Safe-UP...[/]")
        move_to_safe_up(self._arm, current_pose=None)
        measurement_count = 0
        for i, pose in enumerate(valid_poses):
            pose_measurements = []
            for rep in range(repeats):
                measurement_count += 1
                self._cal_update_progress(i, len(valid_poses), rep, repeats, measurement_count, total)
                if rep > 0 or i > 0:
                    self._cal_safe_up_between()
                result = self._cal_measure_single_pose(pose, i, rep, repeats, diagnostics)
                if result:
                    pose_measurements.append(result)
            self._cal_aggregate_pose(pose, pose_measurements, commanded, errors, diagnostics, repeats)
        # Repeatability test
        self._cal_repeatability_test(valid_poses, diagnostics)
        # Fit model
        model, residuals = self._cal_fit_model(commanded, errors, repeats)
        # Save
        self._cal_save_results(model, diagnostics, residuals, measurement_count, valid_poses, repeats)
        # Show diagnostics tables
        self._cal_show_diagnostics(diagnostics, residuals, valid_poses, repeats)
        # Cleanup
        self._cal_cleanup()

    def _cal_cleanup(self):
        """Cleanup after calibration: safe-up, reset buttons."""
        from calibrate import move_to_safe_up
        current = self._arm.read_position_deg()
        if current:
            move_to_safe_up(self._arm, current_pose=current)
        self.app.call_from_thread(self._cal_finished)
        self.app.call_from_thread(self._stop_activity, "✅ Calibration complete")
        self.app.call_from_thread(self.set_timer, 5.0, lambda: self._stop_activity())


    def _update_cal_status(self, text: str):
        """Aktualisiert das Calibration-Status-Panel."""
        try:
            panel = self.query_one("#cal-status-panel", Static)
            panel.update(f"[bold]{text}[/]")
        except NoMatches:
            pass

    def _cal_finished(self):
        try:
            self.query_one("#btn-cal-start", Button).disabled = False
            self.query_one("#btn-cal-abort", Button).disabled = True
        except NoMatches:
            pass

    def _abort_calibration(self):
        """Bricht die Kalibrierung ab."""
        self._stop_activity("⚠ Calibration aborted")
        self._log_calibrate("[yellow]⚠ Kalibrierung abgebrochen![/]")
        try:
            self.query_one("#btn-cal-start", Button).disabled = False
            self.query_one("#btn-cal-abort", Button).disabled = True
        except NoMatches:
            pass
        self.set_timer(3.0, lambda: self._stop_activity())

    def _load_calibration(self):
        """Lädt eine bestehende Kalibrierungsdatei."""
        cal_path = Path("calibration") / "roarm_calibration.cal"
        if not cal_path.exists():
            self._log_calibrate("[yellow]Keine Kalibrierungsdatei gefunden![/]")
            return

        try:
            from calibrate import CalibrationModel
            model = CalibrationModel.load(str(cal_path))
            self._log_calibrate(
                f"[green]✅ Kalibrierung geladen: {cal_path}[/]\n"
                f"  Residuen: b={model.residuals.get('b', 0):.4f}° "
                f"s={model.residuals.get('s', 0):.4f}° "
                f"e={model.residuals.get('e', 0):.4f}°"
            )
        except Exception as e:
            self._log_calibrate(f"[red]Fehler: {e}[/]")

    # ============================================================
    # SERVO CONTROL MODE
    # ============================================================

    @on(Button.Pressed, "#btn-servo-b-go")
    def on_servo_b_go(self) -> None:
        self._servo_go("b")

    @on(Button.Pressed, "#btn-servo-s-go")
    def on_servo_s_go(self) -> None:
        self._servo_go("s")

    @on(Button.Pressed, "#btn-servo-e-go")
    def on_servo_e_go(self) -> None:
        self._servo_go("e")

    @on(Button.Pressed, "#btn-servo-h-go")
    def on_servo_h_go(self) -> None:
        self._servo_go("h")

    @on(Button.Pressed, "#btn-servo-read")
    def on_servo_read(self) -> None:
        self.action_read_position()

    @on(Button.Pressed, "#btn-servo-home")
    def on_servo_home(self) -> None:
        self._go_home()

    @on(Button.Pressed, "#btn-servo-torque-off")
    def on_servo_torque_off(self) -> None:
        self.action_torque_release()

    def _servo_go(self, joint: str):
        """Fährt einen einzelnen Servo zur eingegebenen Position."""
        arm = self._active_arm
        if arm is None:
            self._log_servo("[red]Nicht verbunden und keine Simulation![/]")
            return

        try:
            input_widget = self.query_one(f"#servo-{joint}-input", Input)
            angle = float(input_widget.value)
        except (NoMatches, ValueError) as e:
            self._log_servo(f"[red]Ungültiger Wert: {e}[/]")
            return

        # Aktuelle Position lesen und nur das eine Gelenk ändern
        pos = self._current_pos.copy()
        pos[joint] = angle

        arm.torque_on()
        self.torque_on_state = True
        self._update_status_torque(True)
        time.sleep(0.1)

        joint_names = {"b": "Base", "s": "Shoulder", "e": "Elbow", "h": "Hand"}

        # Start activity indicator
        self._start_activity(f"Moving {joint_names[joint]}", "🎯")

        arm.move_to(
            pos["b"], pos["s"], pos["e"], pos["h"],
            spd=15, acc=8
        )

        sim_tag = " (sim)" if self._is_sim else ""
        self._log_servo(
            f"[green]→ {joint_names[joint]} → {angle:.2f}°{sim_tag}[/]"
        )

        # Nach kurzer Wartezeit Position lesen und activity stoppen
        self.set_timer(1.5, self._servo_read_after_move)

    def _servo_read_after_move(self):
        """Liest Position nach einem Servo-Move."""
        arm = self._active_arm
        if arm is None:
            self._stop_activity()
            return
        pos = arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self._update_joint_displays(pos)
            self._update_arm_views(pos)
            self._update_servo_readouts(pos)

        # Stop activity indicator
        self._stop_activity("✅ Move complete")
        self.set_timer(3.0, lambda: self._stop_activity())

    # ============================================================
    # LOGS TAB - KOMPLETT GEFIXT
    # ============================================================

    @on(TabbedContent.TabActivated)
    def on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Wird aufgerufen wenn ein Tab aktiviert wird."""
        if event.tab.id == "--content-tab-logs" or str(event.pane.id) == "logs":
            # Logs neu laden wenn der Tab sichtbar wird
            self._load_logs()
            self._apply_log_filter()

    def _load_logs(self) -> bool:
        """Lädt die neuesten Log-Dateien.

        Returns:
            True wenn neue Zeilen geladen wurden.
        """
        log_files = sorted(LOGS_DIR.glob("robot_commands_*.log"), reverse=True)

        if not log_files:
            # Auch andere Log-Patterns suchen
            log_files = sorted(LOGS_DIR.glob("*.log"), reverse=True)

        if not log_files:
            # Generiere eine synthetische "Willkommen"-Nachricht
            if not self._all_log_lines:
                self._all_log_lines = [
                    f"═══ Dashboard Log ═══\n",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.000 | INFO     | Dashboard gestartet\n",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.001 | INFO     | Log-Verzeichnis: {LOGS_DIR.absolute()}\n",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.002 | INFO     | Warte auf Arm-Verbindung...\n",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.003 | NOTE     | Verbinde den Arm um echte Logs zu sehen\n",
                ]
            return False

        new_lines = []
        # Die neuesten 3 Log-Dateien laden (für mehr Kontext)
        for log_file in log_files[:3]:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                # Dateiname als Header einfügen
                if lines:
                    new_lines.append(f"═══ {log_file.name} ═══\n")
                    new_lines.extend(lines)
            except (OSError, PermissionError) as e:
                new_lines.append(f"[FEHLER beim Lesen: {log_file.name}: {e}]\n")

        # Nur die letzten 1000 Zeilen behalten
        old_count = len(self._all_log_lines)
        self._all_log_lines = new_lines[-1000:]

        return len(self._all_log_lines) != old_count

    def _periodic_log_refresh(self):
        """Lädt neue Log-Zeilen und aktualisiert die Anzeige.
        
        Nur updaten wenn der Logs-Tab aktiv ist (Performance).
        """
        try:
            tabs = self.query_one(TabbedContent)
            if tabs.active != "logs":
                return
        except NoMatches:
            return

        had_lines = len(self._all_log_lines)
        self._load_logs()
        # Nur neu rendern wenn sich was geändert hat
        if len(self._all_log_lines) != had_lines:
            self._apply_log_filter(auto_scroll=True)

    @on(Button.Pressed, "#btn-log-filter")
    def on_log_filter(self) -> None:
        """Filter-Button gedrückt."""
        self._apply_log_filter()

    @on(Button.Pressed, "#btn-log-refresh")
    def on_log_refresh(self) -> None:
        """Refresh-Button: Logs neu laden und anzeigen."""
        self._load_logs()
        self._apply_log_filter()
        # Bestätigungsmeldung am Ende
        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.write(
                f"[bold green]─── ↻ Aktualisiert: "
                f"{len(self._all_log_lines)} Zeilen geladen "
                f"({datetime.now().strftime('%H:%M:%S')}) ───[/]"
            )
        except NoMatches:
            pass

    @on(Button.Pressed, "#btn-log-clear")
    def on_log_clear(self) -> None:
        """Clear-Button: Viewer leeren (nicht die Quelldaten)."""
        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.clear()
            viewer.write("[dim]Log-Anzeige geleert. [↻ Refresh] zum Neuladen.[/]")
        except NoMatches:
            pass

    @on(Input.Submitted, "#log-search-input")
    def on_log_search_submit(self, event: Input.Submitted) -> None:
        """Enter im Suchfeld: Filter anwenden."""
        self._apply_log_filter()

    @on(Input.Changed, "#log-search-input")
    def on_log_search_changed(self, event: Input.Changed) -> None:
        """Live-Filterung bei Eingabe (mit Debounce via Timer)."""
        if hasattr(self, '_log_search_timer') and self._log_search_timer:
            self._log_search_timer.stop()
        self._log_search_timer = self.set_timer(
            0.3, self._apply_log_filter
        )

    def _apply_log_filter(self, auto_scroll: bool = True):
        """Filtert und zeigt die Logs an."""
        # Filter-Pattern lesen
        try:
            search_input = self.query_one("#log-search-input", Input)
            pattern = search_input.value.strip()
        except NoMatches:
            pattern = ""

        # Viewer holen
        try:
            viewer = self.query_one("#log-viewer", RichLog)
        except NoMatches:
            return

        # Viewer leeren für neuen Inhalt
        viewer.clear()

        # Keine Daten? Hilfreiche Meldung
        if not self._all_log_lines:
            viewer.write("[bold yellow]⚠ Keine Log-Dateien gefunden.[/]")
            viewer.write("")
            viewer.write(f"[dim]Suchpfad: {LOGS_DIR.absolute()}[/]")
            viewer.write(f"[dim]Erwartetes Pattern: robot_commands_*.log[/]")
            viewer.write("")
            viewer.write(
                "[dim]Logs werden automatisch erstellt wenn der Arm "
                "verbunden wird.[/]"
            )
            if LOGS_DIR.exists():
                all_files = list(LOGS_DIR.iterdir())
                if all_files:
                    viewer.write(f"\n[dim]Dateien in {LOGS_DIR}/:[/]")
                    for f in sorted(all_files)[:20]:
                        try:
                            size = f.stat().st_size
                            viewer.write(f"[dim]  • {f.name} ({size} bytes)[/]")
                        except OSError:
                            viewer.write(f"[dim]  • {f.name}[/]")
                else:
                    viewer.write(f"\n[dim]Verzeichnis {LOGS_DIR}/ ist leer.[/]")
            return

        # --- Filtern ---
        filtered = self._filter_log_lines(pattern)

        # --- Header mit Statistik ---
        total = len(self._all_log_lines)
        shown = len(filtered)

        if pattern:
            viewer.write(
                f"[bold]🔍 Filter:[/] [cyan]'{pattern}'[/] → "
                f"[bold]{shown}[/]/{total} Zeilen"
            )
            viewer.write("─" * 60)
        else:
            viewer.write(
                f"[dim]📋 {total} Zeilen geladen | "
                f"Filter: Text, /regex/, !negation, term1|term2[/]"
            )
            viewer.write("─" * 60)

        # --- Zeilen anzeigen (max 500 für Performance) ---
        display_lines = filtered[-500:] if len(filtered) > 500 else filtered

        if len(filtered) > 500:
            viewer.write(
                f"[yellow]⚠ Zeige nur die letzten 500 von {len(filtered)} "
                f"Treffern[/]"
            )
            viewer.write("")

        for line in display_lines:
            styled = self._style_log_line(line.rstrip())
            viewer.write(styled)

        # --- Keine Treffer ---
        if not filtered and pattern:
            viewer.write("")
            viewer.write(f"[bold yellow]Keine Treffer für '{pattern}'[/]")
            viewer.write("")
            viewer.write("[dim]Tipps:[/]")
            viewer.write("[dim]  • Text-Suche ist case-insensitive[/]")
            viewer.write("[dim]  • /regex/ für reguläre Ausdrücke[/]")
            viewer.write("[dim]  • SEND, RECV, NOTE, TIMEOUT als Keywords[/]")
            viewer.write("[dim]  • !SEND_FAST um Fast-Sends auszublenden[/]")
            viewer.write("[dim]  • SEND|RECV für OR-Suche[/]")

    def _filter_log_lines(self, pattern: str) -> list:
        """Filtert Log-Zeilen nach Pattern."""
        if not pattern:
            return list(self._all_log_lines)

        negate = False
        if pattern.startswith("!"):
            negate = True
            pattern = pattern[1:]

        # Regex-Modus
        if pattern.startswith("/") and "/" in pattern[1:]:
            end_idx = pattern.rindex("/")
            if end_idx > 0:
                regex_str = pattern[1:end_idx]
                flags_str = pattern[end_idx+1:]

                flags = re.IGNORECASE
                if 's' in flags_str:
                    flags |= re.DOTALL
                if 'm' in flags_str:
                    flags |= re.MULTILINE
                if 'c' in flags_str:
                    flags &= ~re.IGNORECASE

                try:
                    regex = re.compile(regex_str, flags)
                except re.error as e:
                    return [f"[REGEX-FEHLER: {e}]\n"]

                if negate:
                    return [l for l in self._all_log_lines if not regex.search(l)]
                else:
                    return [l for l in self._all_log_lines if regex.search(l)]

        # OR-Suche
        if "|" in pattern:
            terms = [t.strip().lower() for t in pattern.split("|") if t.strip()]
            if negate:
                return [l for l in self._all_log_lines
                        if not any(t in l.lower() for t in terms)]
            else:
                return [l for l in self._all_log_lines
                        if any(t in l.lower() for t in terms)]

        # Einfache Textsuche
        pattern_lower = pattern.lower()
        if negate:
            return [l for l in self._all_log_lines if pattern_lower not in l.lower()]
        else:
            return [l for l in self._all_log_lines if pattern_lower in l.lower()]

    def _style_log_line(self, line: str) -> str:
        """Farbcodierte Log-Zeile."""
        if not line or "═══" in line:
            return f"[bold bright_white]{line}[/]"

        if "| ERROR" in line or "ERROR" in line.upper()[:50]:
            return f"[bold red]{line}[/]"
        elif "| WARNING" in line or "| TIMEOUT" in line:
            return f"[yellow]{line}[/]"
        elif "SEND_FAST" in line:
            return f"[dim bright_black]{line}[/]"
        elif "| SEND" in line:
            return f"[bright_blue]{line}[/]"
        elif "| RECV" in line:
            return f"[bright_green]{line}[/]"
        elif "NOTE" in line:
            return f"[bold bright_cyan]{line}[/]"
        elif "SESSION START" in line or "CONNECTED" in line:
            return f"[bold bright_magenta]{line}[/]"
        elif "DISCONNECTED" in line:
            return f"[bold yellow]{line}[/]"
        elif line.startswith("─") or line.startswith("═"):
            return f"[dim]{line}[/]"
        else:
            return line

    def _log_to_viewer(self, msg: str):
        """Schreibt eine einzelne Nachricht in den Log-Viewer."""
        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.write(msg)
        except NoMatches:
            pass

    def _initial_log_load(self):
        """Initiales Laden der Logs nach Mount-Delay."""
        self._load_logs()
        # Nur anzeigen wenn der Logs-Tab gerade aktiv ist
        try:
            tabs = self.query_one(TabbedContent)
            if tabs.active == "logs":
                self._apply_log_filter()
        except NoMatches:
            pass

    # ============================================================
    # WATCH: REACTIVE CHANGES
    # ============================================================

    def watch_connected(self, connected: bool) -> None:
        try:
            mode_label = self.query_one("#status-mode", Label)
            if connected:
                mode_label.update("⏱️ Ready")
            elif self._simulation_mode:
                mode_label.update("🤖 Sim Ready")
            else:
                mode_label.update("⏱️ --")
        except NoMatches:
            pass

    def watch_recording(self, recording: bool) -> None:
        try:
            mode_label = self.query_one("#status-mode", Label)
            if recording:
                mode_label.update("🔴 REC")
            elif self.playing:
                mode_label.update("▶️ PLAY")
            elif self.connected:
                mode_label.update("⏱️ Ready")
        except NoMatches:
            pass

    def watch_playing(self, playing: bool) -> None:
        try:
            mode_label = self.query_one("#status-mode", Label)
            if playing:
                mode_label.update("▶️ PLAY")
            elif self.recording:
                mode_label.update("🔴 REC")
            elif self.connected:
                mode_label.update("⏱️ Ready")
        except NoMatches:
            pass

# ============================================================
# MAIN
# ============================================================

def main():
    app = RoArmDashboard()
    app.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

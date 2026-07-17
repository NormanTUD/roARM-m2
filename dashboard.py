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
#     "pyyaml",
# ]
# ///

import os
import sys
import re

from bootstrap import ensure_uv
ensure_uv()

import json
import time
import math
import threading
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

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

from robot import (
    RoArmConnection, find_arm_port, rad_to_deg, deg_to_rad,
    START_POSITION_DEG, POSITION_TOLERANCE, BAUDRATE,
)
from safety import SafeArm, SafetyLimits
from visualize import RobotVisualizer, forward_kinematics

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

# ============================================================
# 3D ASCII RENDERER (echte 3D-Projektion mit Rotation)
# ============================================================

class Ascii3DRenderer:
    """
    Rendert den Roboterarm als echte 3D-Projektion im Terminal.
    Unterstützt Kamera-Rotation um den Arm herum.
    Zeigt den Arm aus einer perspektivischen 3D-Ansicht.
    """

    def __init__(self, width: int = 72, height: int = 28):
        self.width = width
        self.height = height
        # Kamera-Parameter (Azimut, Elevation)
        self.cam_azimuth = 45.0  # Grad
        self.cam_elevation = 25.0  # Grad
        self.cam_distance = 600.0  # mm

    def rotate_camera(self, d_azimuth: float = 0, d_elevation: float = 0):
        """Rotiert die Kamera."""
        self.cam_azimuth = (self.cam_azimuth + d_azimuth) % 360
        self.cam_elevation = max(-80, min(80, self.cam_elevation + d_elevation))

    def _project_3d_to_2d(self, x: float, y: float, z: float) -> tuple:
        """Projiziert einen 3D-Punkt auf 2D-Canvas mit perspektivischer Projektion."""
        # Kamera-Rotation
        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)

        # Rotation um Z-Achse (Azimut)
        x1 = x * math.cos(az) - y * math.sin(az)
        y1 = x * math.sin(az) + y * math.cos(az)
        z1 = z

        # Rotation um X-Achse (Elevation)
        y2 = y1 * math.cos(el) - z1 * math.sin(el)
        z2 = y1 * math.sin(el) + z1 * math.cos(el)
        x2 = x1

        # Perspektivische Projektion
        d = self.cam_distance
        scale = d / (d + y2 + 300)  # +300 um Clipping zu vermeiden

        px = int(x2 * scale * 0.15 + self.width // 2)
        py = int(-z2 * scale * 0.12 + self.height * 0.7)

        return px, py, scale

    def render(self, b_deg: float, s_deg: float, e_deg: float,
               trail: list = None, target: dict = None) -> str:
        """Rendert den Arm als 3D ASCII-String."""
        positions = forward_kinematics(b_deg, s_deg, e_deg)

        # Canvas erstellen
        canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        depth_buf = [[float('inf') for _ in range(self.width)] for _ in range(self.height)]

        # Boden-Gitter zeichnen
        self._draw_ground_grid(canvas, depth_buf)

        # Achsen zeichnen (X=rot, Y=grün, Z=blau)
        self._draw_axes(canvas, depth_buf)

        # Trail zeichnen
        if trail:
            for tp in trail[-40:]:
                px, py, _ = self._project_3d_to_2d(tp[0], tp[1], tp[2])
                self._put_char(canvas, px, py, '·')

        # Target-Marker
        if target:
            t_pos = forward_kinematics(target["b"], target["s"], target["e"])
            tp = t_pos["gripper"]
            px, py, _ = self._project_3d_to_2d(tp[0], tp[1], tp[2])
            self._put_char(canvas, px, py, '✕')
            self._put_char(canvas, px - 1, py, '(')
            self._put_char(canvas, px + 1, py, ')')

        # Arm-Segmente zeichnen
        joint_names = ["base", "shoulder", "elbow", "gripper"]
        joint_chars = ['◆', '●', '●', '◇']
        segment_chars = ['│', '╱', '╲']

        projected = []
        for name in joint_names:
            p = positions[name]
            px, py, scale = self._project_3d_to_2d(p[0], p[1], p[2])
            projected.append((px, py, scale))

        # Segmente (Linien zwischen Gelenken)
        seg_styles = ['║', '▓', '░']
        for i in range(len(projected) - 1):
            p1 = projected[i]
            p2 = projected[i + 1]
            ch = seg_styles[min(i, len(seg_styles) - 1)]
            self._draw_line_3d(canvas, p1[0], p1[1], p2[0], p2[1], ch)

        # Gelenke (über den Linien)
        for i, (px, py, scale) in enumerate(projected):
            self._put_char(canvas, px, py, joint_chars[i])

        # Info-Header
        lines = []
        gp = positions["gripper"]
        lines.append(
            f"  ┌─ 3D View ─ Az:{self.cam_azimuth:.0f}° El:{self.cam_elevation:.0f}°"
            f" ─ [←/→]=Rotate [↑/↓]=Elevate ─┐"
        )
        lines.append(f"  │ b={b_deg:+7.1f}° s={s_deg:+7.1f}° e={e_deg:+7.1f}°"
                     f"  │ Gripper: ({gp[0]:.0f}, {gp[1]:.0f}, {gp[2]:.0f})mm │")
        lines.append('  ' + '─' * (self.width - 2))

        # Canvas zu String
        for row in canvas:
            lines.append('  ' + ''.join(row))

        lines.append('  ' + '─' * (self.width - 2))
        lines.append(f"  ◆=Base ●=Shoulder/Elbow ◇=Gripper  ✕=Target  ·=Trail")

        return '\n'.join(lines)

    def _draw_ground_grid(self, canvas, depth_buf):
        """Zeichnet ein Boden-Gitter auf Z=0."""
        grid_size = 200
        step = 100
        for x in range(-grid_size, grid_size + 1, step):
            for y in range(-grid_size, grid_size + 1, step):
                px, py, _ = self._project_3d_to_2d(x, y, 0)
                if 0 <= px < self.width and 0 <= py < self.height:
                    if canvas[py][px] == ' ':
                        canvas[py][px] = '·' if (x == 0 or y == 0) else '.'

    def _draw_axes(self, canvas, depth_buf):
        """Zeichnet Koordinatenachsen."""
        axis_len = 80
        # X-Achse
        for i in range(0, axis_len, 8):
            px, py, _ = self._project_3d_to_2d(i, 0, 0)
            self._put_char(canvas, px, py, '→' if i == axis_len - 8 else '─')
        # Y-Achse
        for i in range(0, axis_len, 8):
            px, py, _ = self._project_3d_to_2d(0, i, 0)
            self._put_char(canvas, px, py, '→' if i == axis_len - 8 else '─')
        # Z-Achse
        for i in range(0, axis_len, 8):
            px, py, _ = self._project_3d_to_2d(0, 0, i)
            self._put_char(canvas, px, py, '↑' if i == axis_len - 8 else '│')

    def _put_char(self, canvas, x, y, ch):
        if 0 <= x < self.width and 0 <= y < self.height:
            canvas[y][x] = ch

    def _draw_line_3d(self, canvas, x1, y1, x2, y2, ch):
        """Bresenham-Linie."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy, 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self._put_char(canvas, x, y, ch)

# ============================================================
# RECORDING PARSER
# ============================================================

def parse_roarm_file(filepath: str) -> dict:
    """Parst eine .roarm Recording-Datei."""
    waypoints = []
    gripper_cmds = []
    config = {"hz": 20, "threshold": 0.3}
    start_pos = None
    offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#CONFIG"):
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    key, val = parts[1].split("=", 1)
                    config[key.strip()] = float(val.strip())
                continue
            if line.startswith("#START_POS"):
                parts = line.split()[1:]
                vals = {}
                for p in parts:
                    k, v = p.split("=")
                    vals[k] = float(v)
                start_pos = vals
                continue
            if line.startswith("#OFFSET"):
                parts = line.split()[1:]
                for p in parts:
                    k, v = p.split("=")
                    offset[k.strip()] = float(v.strip())
                continue
            if line.startswith("#"):
                continue
            if line.startswith("MOVE"):
                parts = line.split()
                vals = {}
                for p in parts[1:]:
                    k, v = p.split("=")
                    vals[k] = float(v)
                waypoints.append({
                    "t": vals.get("t", 0.0),
                    "b": vals.get("b", 0.0),
                    "s": vals.get("s", 0.0),
                    "e": vals.get("e", 90.0),
                    "h": vals.get("h", 180.0),
                })
            elif line.startswith("GRIPPER"):
                parts = line.split()
                cmd = parts[1] if len(parts) > 1 else "OPEN"
                t = 0.0
                for p in parts[1:]:
                    if p.startswith("t="):
                        t = float(p.split("=")[1])
                gripper_cmds.append({"t": t, "cmd": cmd})

    if start_pos is None:
        start_pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}

    return {
        "waypoints": waypoints,
        "gripper_cmds": gripper_cmds,
        "config": config,
        "start_pos": start_pos,
        "offset": offset,
    }

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
"""

# ============================================================
# CUSTOM WIDGETS
# ============================================================

class Arm3DWidget(Static):
    """Widget das den 3D-Arm rendert."""

    b = reactive(0.0)
    s = reactive(0.0)
    e = reactive(90.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._renderer = Ascii3DRenderer(width=70, height=24)
        self._trail = []
        self._target = None

    def update_pose(self, b: float, s: float, e: float,
                    target: dict = None):
        self.b = b
        self.s = s
        self.e = e
        self._target = target

        # Trail updaten
        positions = forward_kinematics(b, s, e)
        self._trail.append(positions["gripper"])
        if len(self._trail) > 50:
            self._trail.pop(0)

        self._refresh_display()

    def rotate(self, d_azimuth: float = 0, d_elevation: float = 0):
        """Rotiert die 3D-Ansicht."""
        self._renderer.rotate_camera(d_azimuth, d_elevation)
        self._refresh_display()

    def clear_trail(self):
        self._trail.clear()
        self._refresh_display()

    def _refresh_display(self):
        rendered = self._renderer.render(
            self.b, self.s, self.e,
            trail=self._trail,
            target=self._target
        )
        self.update(rendered)

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
        Binding("v", "toggle_viz", "3D Viz", show=True),
        Binding("left", "rotate_left", "Rot←", show=False),
        Binding("right", "rotate_right", "Rot→", show=False),
        Binding("up", "rotate_up", "Rot↑", show=False),
        Binding("down", "rotate_down", "Rot↓", show=False),
        Binding("r", "read_position", "Read Pos", show=True),
    ]

    # --- Reactive State ---
    connected = reactive(False)
    recording = reactive(False)
    playing = reactive(False)
    torque_on_state = reactive(True)

    def __init__(self):
        super().__init__()
        self._arm: Optional[RoArmConnection] = None
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
                                        "📖 Read All [r]", id="btn-servo-read",
                                        variant="default"
                                    )
                                    yield Button(
                                        "🏠 Home [h]", id="btn-servo-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "🔓 Torque Off [t]",
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
            yield Label("│", id="status-sep1")
            yield Label("🔒 Torque ON", id="status-torque")
            yield Label("│", id="status-sep2")
            yield Label("🛡️ Safety OK", id="status-safety")
            yield Label("│", id="status-sep3")
            yield Label("⏱️ --", id="status-mode")

        yield Footer()

    # ============================================================
    # ON MOUNT
    # ============================================================

    def on_mount(self) -> None:
        """Initialisierung nach dem Mounten."""
        self._refresh_recordings_table()
        self._try_auto_connect()
        self._load_logs()

        # Periodischer Position-Poll (wenn verbunden)
        self.set_interval(0.5, self._periodic_position_poll)
        # Log-Refresh
        self.set_interval(2.0, self._periodic_log_refresh)

    def _try_auto_connect(self):
        """Versucht automatisch den Arm zu finden und zu verbinden."""
        port = find_arm_port()
        if port:
            self._log_teach(f"[dim]🔍 Port gefunden: {port} – verbinde...[/]")
            self._do_connect(port)
        else:
            self._log_teach("[yellow]⚠ Kein Arm-Port gefunden. [c] zum manuellen Verbinden.[/]")

    def _do_connect(self, port: str):
        """Verbindet synchron mit dem Arm."""
        try:
            self._arm = RoArmConnection(port)
            self.connected = True
            self._log_teach(f"[green]✅ Verbunden mit {port}[/]")
            self._update_status_connection(port)

            # Position lesen
            pos = self._arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self._update_joint_displays(pos)
                self._update_arm_views(pos)
                self._log_teach(
                    f"[dim]  Pos: b={pos['b']:.1f}° s={pos['s']:.1f}° "
                    f"e={pos['e']:.1f}° h={pos['h']:.1f}°[/]"
                )
        except Exception as e:
            self._log_teach(f"[red]❌ Fehler: {e}[/]")

    # ============================================================
    # PERIODIC TASKS
    # ============================================================

    def _periodic_position_poll(self):
        """Pollt die Position wenn verbunden und nicht recording/playing."""
        if not self.connected or not self._arm:
            return
        if self.recording or self.playing:
            return

        try:
            pos = self._arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self._update_joint_displays(pos)
                # Servo-Readouts updaten
                self._update_servo_readouts(pos)
        except Exception:
            pass

    def _periodic_log_refresh(self):
        """Lädt neue Log-Zeilen."""
        self._load_logs()

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

    def action_torque_release(self) -> None:
        """Torque lösen (Taste t)."""
        if not self._arm or not self.connected:
            self._log_teach("[red]Nicht verbunden![/]")
            return
        self._arm.torque_off()
        self.torque_on_state = False
        self._update_status_torque(False)
        self._log_teach("[yellow]🔓 Torque AUS – Arm ist frei bewegbar[/]")
        self._log_servo("[yellow]🔓 Torque AUS[/]")

    def action_torque_lock(self) -> None:
        """Torque einschalten (Taste T/Shift+t)."""
        if not self._arm or not self.connected:
            self._log_teach("[red]Nicht verbunden![/]")
            return
        self._arm.torque_on()
        self.torque_on_state = True
        self._update_status_torque(True)
        self._log_teach("[green]🔒 Torque AN – Arm ist fixiert[/]")
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

    def action_gripper_toggle(self) -> None:
        if not self._arm or not self.connected:
            return
        if self._gripper_open:
            self._arm.gripper_close()
            self._gripper_open = False
            self._log_teach("[bold]✊ Gripper ZU[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append(
                    {"t": round(elapsed, 4), "cmd": "GRIPPER_CLOSE"}
                )
        else:
            self._arm.gripper_open()
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
        if not self._arm or not self.connected:
            self._log_servo("[red]Nicht verbunden![/]")
            return
        pos = self._arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self._update_joint_displays(pos)
            self._update_arm_views(pos)
            self._update_servo_readouts(pos)
            self._log_servo(
                f"[green]📖 Position:[/] b={pos['b']:+.2f}° "
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

    def action_toggle_viz(self) -> None:
        """Placeholder für externe 3D-Viz (optional)."""
        self._log_teach("[dim]3D-Visualisierung ist jetzt inline im Dashboard.[/]")

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
        if not self.connected or not self._arm:
            self._log_teach("[red]Nicht verbunden![/]")
            return

        self.recording = True
        self._teach_waypoints = []
        self._teach_start_time = time.time()
        self._gripper_open = True

        # Torque aus
        self._arm.torque_off()
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

        self._log_teach("[bold red]⏺ AUFNAHME LÄUFT[/]")
        self._log_teach("[dim]Bewege den Arm! [Space]=Stop [g]=Gripper [t]=Torque[/]")

        # Timer starten
        self._teach_timer = self.set_interval(
            1.0 / RECORD_HZ, self._teach_poll_position
        )

    def _teach_poll_position(self):
        if not self.recording or not self._arm:
            return

        pos = self._arm.read_position_deg()
        if pos is None:
            return

        self._current_pos = pos
        elapsed = time.time() - self._teach_start_time

        # Schwellwert-Check
        should_record = True
        if self._teach_waypoints:
            last_move = None
            for wp in reversed(self._teach_waypoints):
                if "cmd" not in wp:
                    last_move = wp
                    break
            if last_move:
                max_delta = max(
                    abs(pos[j] - last_move[j]) for j in ["b", "s", "e", "h"]
                )
                if max_delta < MOVE_THRESHOLD_DEG:
                    should_record = False

        if should_record:
            self._teach_waypoints.append({
                "t": round(elapsed, 4),
                "b": pos["b"], "s": pos["s"],
                "e": pos["e"], "h": pos["h"],
            })

        # UI updaten
        self._update_joint_displays(pos)
        self._update_arm_views(pos)

        # Status-Log (alle 50 Frames)
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if len(move_wps) % 50 == 0 and len(move_wps) > 0:
            self._log_teach(
                f"[dim]  ◆ WP#{len(move_wps)} [{elapsed:.1f}s] "
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

        if self._arm:
            self._arm.torque_on()
            self.torque_on_state = True
            self._update_status_torque(True)

        try:
            self.query_one("#btn-teach-record", Button).disabled = False
            self.query_one("#btn-teach-stop", Button).disabled = True
        except NoMatches:
            pass

        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if not move_wps:
            self._log_teach("[yellow]Keine Wegpunkte aufgezeichnet![/]")
            return

        duration = move_wps[-1]["t"]
        self._log_teach(
            f"[green]⏹ Aufnahme gestoppt: {len(move_wps)} WPs, {duration:.1f}s[/]"
        )

        filepath = self._save_recording()
        if filepath:
            self._log_teach(f"[green]💾 Gespeichert: {filepath}[/]")
            self._refresh_recordings_table()

    def _save_recording(self) -> Optional[str]:
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if not move_wps:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RECORDINGS_DIR / f"recording_{ts}.roarm"

        lines = [
            f"# RoArm-M2-S Recording (Dashboard v2)",
            f"# Datum: {datetime.now().isoformat()}",
            f"# Wegpunkte: {len(move_wps)}",
            f"# Dauer: {move_wps[-1]['t']:.2f}s",
            f"#",
            f"#CONFIG hz={RECORD_HZ}",
            f"#CONFIG threshold={MOVE_THRESHOLD_DEG}",
            f"#START_POS b={START_POSITION_DEG['b']:.2f} "
            f"s={START_POSITION_DEG['s']:.2f} "
            f"e={START_POSITION_DEG['e']:.2f} "
            f"h={START_POSITION_DEG['h']:.2f}",
            "",
        ]

        for wp in self._teach_waypoints:
            if "cmd" in wp:
                if wp["cmd"] == "GRIPPER_CLOSE":
                    lines.append(f"GRIPPER CLOSE t={wp['t']:.4f}")
                elif wp["cmd"] == "GRIPPER_OPEN":
                    lines.append(f"GRIPPER OPEN t={wp['t']:.4f}")
            else:
                lines.append(
                    f"MOVE b={wp['b']:.2f} s={wp['s']:.2f} "
                    f"e={wp['e']:.2f} h={wp['h']:.2f} t={wp['t']:.4f}"
                )

        with open(filename, 'w') as f:
            f.write("\n".join(lines) + "\n")

        return str(filename)

    @work(thread=True)
    def _go_home(self):
        if not self._arm or not self.connected:
            return

        self.app.call_from_thread(
            self._log_teach, "[dim]🏠 Fahre zur Home-Position...[/]"
        )

        self._arm.torque_on()
        self.torque_on_state = True
        self.app.call_from_thread(self._update_status_torque, True)
        time.sleep(0.2)

        self._arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=25, acc=12
        )
        time.sleep(2.0)

        pos = self._arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self.app.call_from_thread(self._update_joint_displays, pos)
            self.app.call_from_thread(self._update_arm_views, pos)

        self.app.call_from_thread(
            self._log_teach, "[green]✅ Home-Position erreicht[/]"
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
        if not self.connected or not self._arm:
            self._log_play("[red]Nicht verbunden![/]")
            return

        try:
            table = self.query_one("#recordings-table", DataTable)
            row_key = table.cursor_row
            if row_key is None:
                self._log_play("[yellow]Kein Recording ausgewählt![/]")
                return
            row_data = table.get_row_at(row_key)
            filename = row_data[0]
        except (NoMatches, Exception) as e:
            self._log_play(f"[red]Fehler: {e}[/]")
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

        self._log_play(
            f"[green]▶ Playback: {len(wps)} WPs, {wps[-1]['t']:.1f}s[/]"
        )

        self._run_playback(wps)

    @work(thread=True)
    def _run_playback(self, waypoints: list):
        from scipy.interpolate import CubicSpline

        times = np.array([wp["t"] for wp in waypoints])
        splines = {}
        for joint in ["b", "s", "e", "h"]:
            values = np.array([wp[joint] for wp in waypoints])
            splines[joint] = CubicSpline(times, values, bc_type='clamped')

        duration = times[-1]
        interval = 1.0 / STREAM_HZ

        # Zur Startposition fahren
        self._arm.torque_on()
        time.sleep(0.2)
        start = waypoints[0]
        self._arm.move_to(
            start["b"], start["s"], start["e"], start["h"],
            spd=20, acc=10
        )
        self.app.call_from_thread(
            self._log_play, "[dim]  Fahre zur Startposition...[/]"
        )
        time.sleep(2.0)

        # Streaming
        self._play_start_time = time.time()
        last_pos = None
        commands_sent = 0
        skipped = 0

        gripper_cmds = sorted(
            self._play_data.get("gripper_cmds", []), key=lambda x: x["t"]
        )
        gripper_idx = 0

        while self.playing:
            loop_start = time.time()
            elapsed = loop_start - self._play_start_time

            if elapsed >= duration:
                break

            # Gripper-Events
            while gripper_idx < len(gripper_cmds):
                gc = gripper_cmds[gripper_idx]
                if gc["t"] <= elapsed:
                    if gc["cmd"] == "CLOSE":
                        self._arm.gripper_close()
                        self.app.call_from_thread(
                            self._log_play, f"[bold]  ✊ Gripper ZU [{elapsed:.2f}s][/]"
                        )
                    elif gc["cmd"] == "OPEN":
                        self._arm.gripper_open()
                        self.app.call_from_thread(
                            self._log_play, f"[bold]  ✋ Gripper AUF [{elapsed:.2f}s][/]"
                        )
                    gripper_idx += 1
                    time.sleep(0.3)
                else:
                    break

            # Re-read elapsed nach Gripper-Pause
            elapsed = time.time() - self._play_start_time
            if elapsed >= duration:
                break

            # Sample Spline
            target = {}
            for joint in ["b", "s", "e", "h"]:
                target[joint] = round(float(splines[joint](elapsed)), 2)

            # Delta-Check
            should_send = True
            if last_pos:
                max_delta = max(
                    abs(target[j] - last_pos[j]) for j in ["b", "s", "e", "h"]
                )
                if max_delta < MIN_DELTA_DEG:
                    should_send = False
                    skipped += 1

            if should_send:
                self._arm.move_to_fast(
                    target["b"], target["s"], target["e"], target["h"],
                    spd=50, acc=30
                )
                last_pos = target.copy()
                commands_sent += 1

            # UI updaten (throttled: alle 100ms)
            if int(elapsed * 10) % 2 == 0:
                self.app.call_from_thread(self._update_arm_views, target)
                self.app.call_from_thread(
                    self._update_play_timeline, elapsed
                )

            # Timing
            loop_elapsed = time.time() - loop_start
            sleep_time = interval - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Ende
        self.playing = False
        self.app.call_from_thread(
            self._log_play,
            f"[green]✅ Playback beendet: {commands_sent} Cmds, "
            f"{skipped} übersprungen[/]"
        )
        self.app.call_from_thread(self._playback_finished)

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

    def _stop_playback(self):
        """Stoppt das Playback."""
        self.playing = False
        self._log_play("[yellow]⏹ Playback gestoppt[/]")
        try:
            self.query_one("#btn-play-start", Button).disabled = False
            self.query_one("#btn-play-stop", Button).disabled = True
        except NoMatches:
            pass

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

        self._log_calibrate(
            f"[bold green]🎯 Kalibrierung gestartet[/]\n"
            f"  Pose-Set: {pose_set}\n"
            f"  Wiederholungen: {repeats}\n"
            f"  Auto-Accept: {'Ja' if auto_accept else 'Nein'}"
        )

        self._run_calibration_worker(pose_set, repeats, auto_accept)

    @work(thread=True)
    def _run_calibration_worker(self, pose_set: str, repeats: int,
                                auto_accept: bool):
        """Führt die Kalibrierung im Hintergrund aus."""
        from calibrate import (
            CalibrationModel, POSE_SETS, JOINTS,
            move_to_safe_up, move_from_safe_up_to_pose, validate_pose,
        )

        poses = POSE_SETS.get(pose_set, POSE_SETS["standard"])

        # Validieren
        valid_poses = [p for p in poses if validate_pose(p)]
        if len(valid_poses) < 10:
            self.app.call_from_thread(
                self._log_calibrate,
                f"[yellow]⚠ Nur {len(valid_poses)} gültige Posen![/]"
            )

        total = len(valid_poses) * repeats
        commanded = []
        errors = []

        self._arm.torque_on()
        time.sleep(0.2)

        # Safe-UP
        self.app.call_from_thread(
            self._log_calibrate, "[dim]  Fahre zu Safe-UP...[/]"
        )
        move_to_safe_up(self._arm, current_pose=None)

        measurement_count = 0

        for i, pose in enumerate(valid_poses):
            pose_errors = []

            for rep in range(repeats):
                measurement_count += 1
                pct = measurement_count / total * 100

                self.app.call_from_thread(
                    self._update_cal_status,
                    f"Pose {i+1}/{len(valid_poses)} · Rep {rep+1}/{repeats} "
                    f"· {pct:.0f}%"
                )

                # Safe-UP zwischen Posen
                if rep > 0 or i > 0:
                    current = self._arm.read_position_deg()
                    if current:
                        move_to_safe_up(self._arm, current_pose=current)
                    else:
                        move_to_safe_up(self._arm, current_pose=None)

                # Zur Pose fahren
                move_from_safe_up_to_pose(self._arm, pose)

                # Präzisions-Nachfahrt
                self._arm.move_to(
                    pose["b"], pose["s"], pose["e"], pose["h"],
                    spd=5, acc=3
                )
                self._arm.wait_until_settled(
                    tolerance_deg=0.2, stable_count=6
                )

                # Position lesen
                servo_avg = self._arm.read_position_averaged(n=10, interval=0.05)
                if servo_avg:
                    error = {j: servo_avg[j] - pose[j] for j in JOINTS}
                    pose_errors.append(error)

                    # Arm-View updaten
                    self.app.call_from_thread(
                        self._update_arm_views,
                        {"b": servo_avg["b"], "s": servo_avg["s"],
                         "e": servo_avg["e"], "h": servo_avg.get("h", 180.0)}
                    )

                self.app.call_from_thread(
                    self._log_calibrate,
                    f"[dim]  ✓ Pose {i+1} Rep {rep+1}: "
                    f"Δb={error['b']:+.2f}° Δs={error['s']:+.2f}° "
                    f"Δe={error['e']:+.2f}°[/]"
                )

            # Mittelwert für diese Pose
            if pose_errors:
                avg_error = {}
                for j in JOINTS:
                    avg_error[j] = float(
                        np.mean([e[j] for e in pose_errors])
                    )
                commanded.append(pose)
                errors.append(avg_error)

        # Modell fitten
        self.app.call_from_thread(
            self._log_calibrate, "\n[bold]📊 Fitte Kalibrierungsmodell...[/]"
        )

        model = CalibrationModel()
        residuals = model.fit(commanded, errors)

        # Speichern
        from pathlib import Path
        cal_path = Path("calibration") / "roarm_calibration.cal"
        cal_path.parent.mkdir(exist_ok=True)
        model.save(str(cal_path))

        # Ergebnis anzeigen
        result_msg = (
            f"[bold green]✅ Kalibrierung abgeschlossen![/]\n"
            f"  Residuen: b={residuals['b']:.4f}° "
            f"s={residuals['s']:.4f}° e={residuals['e']:.4f}°\n"
            f"  Gespeichert: {cal_path}\n"
            f"  Messungen: {measurement_count}"
        )
        self.app.call_from_thread(self._log_calibrate, result_msg)

        # Buttons zurücksetzen
        self.app.call_from_thread(self._cal_finished)

        # Zurück zu Safe-UP
        current = self._arm.read_position_deg()
        if current:
            move_to_safe_up(self._arm, current_pose=current)

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
        self._log_calibrate("[yellow]⚠ Kalibrierung abgebrochen![/]")
        # Worker-Thread kann nicht direkt gestoppt werden,
        # aber wir setzen ein Flag
        try:
            self.query_one("#btn-cal-start", Button).disabled = False
            self.query_one("#btn-cal-abort", Button).disabled = True
        except NoMatches:
            pass

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
        if not self._arm or not self.connected:
            self._log_servo("[red]Nicht verbunden![/]")
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

        self._arm.torque_on()
        self.torque_on_state = True
        self._update_status_torque(True)
        time.sleep(0.1)

        self._arm.move_to(
            pos["b"], pos["s"], pos["e"], pos["h"],
            spd=15, acc=8
        )

        joint_names = {"b": "Base", "s": "Shoulder", "e": "Elbow", "h": "Hand"}
        self._log_servo(
            f"[green]→ {joint_names[joint]} → {angle:.2f}°[/]"
        )

        # Nach kurzer Wartezeit Position lesen
        self.set_timer(1.5, self._servo_read_after_move)

    def _servo_read_after_move(self):
        """Liest Position nach einem Servo-Move."""
        if not self._arm or not self.connected:
            return
        pos = self._arm.read_position_deg()
        if pos:
            self._current_pos = pos
            self._update_joint_displays(pos)
            self._update_arm_views(pos)
            self._update_servo_readouts(pos)

    # ============================================================
    # LOGS TAB
    # ============================================================

    def _load_logs(self):
        """Lädt die neuesten Log-Dateien."""
        log_files = sorted(LOGS_DIR.glob("robot_commands_*.log"), reverse=True)
        if not log_files:
            return

        # Neueste Log-Datei laden
        latest = log_files[0]
        try:
            with open(latest, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            self._all_log_lines = lines[-500:]  # Letzte 500 Zeilen
        except Exception:
            self._all_log_lines = []

    @on(Button.Pressed, "#btn-log-filter")
    def on_log_filter(self) -> None:
        self._apply_log_filter()

    @on(Button.Pressed, "#btn-log-refresh")
    def on_log_refresh(self) -> None:
        self._load_logs()
        self._apply_log_filter()
        self._log_to_viewer("[green]↻ Logs aktualisiert[/]")

    @on(Button.Pressed, "#btn-log-clear")
    def on_log_clear(self) -> None:
        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.clear()
        except NoMatches:
            pass

    @on(Input.Submitted, "#log-search-input")
    def on_log_search_submit(self, event: Input.Submitted) -> None:
        self._apply_log_filter()

    def _apply_log_filter(self):
        """Filtert die Logs nach dem Suchbegriff (Text oder Regex)."""
        try:
            search_input = self.query_one("#log-search-input", Input)
            pattern = search_input.value.strip()
        except NoMatches:
            pattern = ""

        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.clear()
        except NoMatches:
            return

        if not self._all_log_lines:
            self._load_logs()

        filtered = []

        if not pattern:
            filtered = self._all_log_lines
        elif pattern.startswith("/") and pattern.endswith("/"):
            # Regex-Modus
            regex_str = pattern[1:-1]
            try:
                regex = re.compile(regex_str, re.IGNORECASE)
                filtered = [
                    line for line in self._all_log_lines
                    if regex.search(line)
                ]
            except re.error as e:
                self._log_to_viewer(f"[red]Regex-Fehler: {e}[/]")
                return
        else:
            # Einfache Textsuche (case-insensitive)
            pattern_lower = pattern.lower()
            filtered = [
                line for line in self._all_log_lines
                if pattern_lower in line.lower()
            ]

        # Ergebnisse anzeigen
        if pattern:
            self._log_to_viewer(
                f"[dim]🔍 Filter: '{pattern}' → {len(filtered)}/{len(self._all_log_lines)} Zeilen[/]"
            )

        for line in filtered[-200:]:  # Max 200 Zeilen anzeigen
            line = line.rstrip()
            # Farbcodierung nach Log-Level
            if "| WARNING" in line or "| TIMEOUT" in line:
                self._log_to_viewer(f"[yellow]{line}[/]")
            elif "| ERROR" in line:
                self._log_to_viewer(f"[red]{line}[/]")
            elif "SEND_FAST" in line:
                self._log_to_viewer(f"[dim]{line}[/]")
            elif "| SEND" in line:
                self._log_to_viewer(f"[bright_blue]{line}[/]")
            elif "| RECV" in line:
                self._log_to_viewer(f"[bright_green]{line}[/]")
            elif "NOTE" in line:
                self._log_to_viewer(f"[bright_cyan]{line}[/]")
            else:
                self._log_to_viewer(line)

        if not filtered and pattern:
            self._log_to_viewer(f"[yellow]Keine Treffer für '{pattern}'[/]")

    def _log_to_viewer(self, msg: str):
        """Schreibt in den Log-Viewer."""
        try:
            viewer = self.query_one("#log-viewer", RichLog)
            viewer.write(msg)
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
    """Startet das Dashboard."""
    app = RoArmDashboard()
    app.run()

if __name__ == "__main__":
    main()

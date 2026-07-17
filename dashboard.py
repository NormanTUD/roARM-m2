#!/usr/bin/env python3
"""dashboard.py - RoArm-M2-S Unified TUI Dashboard

Alles in einem Textual-Interface:
- Tab 1: TEACH (Recording mit Live-Feedback)
- Tab 2: PLAY (Recordings abspielen mit Timeline)
- Tab 3: SEQUENCE (Macro-Builder: Recordings verketten)
- Tab 4: TIMELINE (Waypoint-Editor mit Scrubbing)

3D-Visualisierung: Läuft als separater Prozess (wie bisher),
wird aber vom Dashboard aus gestartet/gesteuert.

Self-bootstrapping via UV.
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

from bootstrap import ensure_uv
ensure_uv()

import json
import time
import math
import threading
import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
import select

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
    Sparkline, Rule,
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
SEQUENCES_DIR = Path("sequences")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

RECORD_HZ = 50
MOVE_THRESHOLD_DEG = 0.1
STREAM_HZ = 50


# ============================================================
# ASCII 3D RENDERER (für Textual-Integration)
# ============================================================

class Ascii3DView:
    """
    Rendert den Roboterarm als ASCII-Art im Terminal.
    Verwendet die Forward-Kinematik aus visualize.py.
    
    Zeigt 2 Ansichten nebeneinander:
    - Seitenansicht (XZ-Ebene)
    - Draufsicht (XY-Ebene)
    """

    def __init__(self, width: int = 60, height: int = 25):
        self.width = width
        self.height = height
        self.half_width = width // 2

    def render(self, b_deg: float, s_deg: float, e_deg: float,
               trail: list = None, target: dict = None) -> str:
        """Rendert den Arm als ASCII-String."""
        positions = forward_kinematics(b_deg, s_deg, e_deg)

        # Canvas für Seitenansicht (XZ) und Draufsicht (XY)
        side_canvas = self._create_canvas(self.half_width, self.height)
        top_canvas = self._create_canvas(self.half_width, self.height)

        # Skalierung: Arm-Reichweite → Canvas-Pixel
        reach = 500.0  # max Reichweite in mm
        scale_side = min(self.half_width - 4, self.height - 4) / (reach * 2)
        scale_top = min(self.half_width - 4, self.height - 4) / (reach * 2)

        # Seitenansicht (XZ-Ebene, Y ignoriert)
        pts_side = []
        for name in ["base", "shoulder", "elbow", "gripper"]:
            p = positions[name]
            # X → horizontal, Z → vertikal (invertiert für Terminal)
            px = int(p[0] * scale_side + self.half_width // 2)
            py = int((self.height - 2) - (p[2] + 50) * scale_side)
            pts_side.append((px, py))

        # Draufsicht (XY-Ebene)
        pts_top = []
        for name in ["base", "shoulder", "elbow", "gripper"]:
            p = positions[name]
            px = int(p[0] * scale_top + self.half_width // 2)
            py = int(p[1] * scale_top + self.height // 2)
            pts_top.append((px, py))

        # Linien zeichnen (Seitenansicht)
        self._draw_line(side_canvas, pts_side[0], pts_side[1], '│')  # Basis
        self._draw_line(side_canvas, pts_side[1], pts_side[2], '╱')  # Oberarm
        self._draw_line(side_canvas, pts_side[2], pts_side[3], '╲')  # Unterarm

        # Gelenke (Seitenansicht)
        symbols = ['◆', '●', '●', '◇']
        for i, (px, py) in enumerate(pts_side):
            self._put_char(side_canvas, px, py, symbols[i])

        # Linien zeichnen (Draufsicht)
        self._draw_line(top_canvas, pts_top[0], pts_top[1], '·')
        self._draw_line(top_canvas, pts_top[1], pts_top[2], '─')
        self._draw_line(top_canvas, pts_top[2], pts_top[3], '─')

        # Gelenke (Draufsicht)
        for i, (px, py) in enumerate(pts_top):
            self._put_char(top_canvas, px, py, symbols[i])

        # Trail in Draufsicht
        if trail:
            for tp in trail[-30:]:  # Letzte 30 Punkte
                px = int(tp[0] * scale_top + self.half_width // 2)
                py = int(tp[1] * scale_top + self.height // 2)
                self._put_char(top_canvas, px, py, '·')

        # Target-Marker
        if target:
            t_pos = forward_kinematics(target["b"], target["s"], target["e"])
            tp = t_pos["gripper"]
            # Seitenansicht
            tx = int(tp[0] * scale_side + self.half_width // 2)
            ty = int((self.height - 2) - (tp[2] + 50) * scale_side)
            self._put_char(side_canvas, tx, ty, '✕')
            # Draufsicht
            tx = int(tp[0] * scale_top + self.half_width // 2)
            ty = int(tp[1] * scale_top + self.height // 2)
            self._put_char(top_canvas, tx, ty, '✕')

        # Boden-Linie (Seitenansicht)
        ground_y = int((self.height - 2) - (-50 + 50) * scale_side)
        if 0 <= ground_y < self.height:
            for x in range(self.half_width):
                if side_canvas[ground_y][x] == ' ':
                    side_canvas[ground_y][x] = '─'

        # Zusammenfügen
        lines = []
        lines.append(f"{'  Seite (XZ)':^{self.half_width}}│{'  Oben (XY)':^{self.half_width}}")
        lines.append('─' * self.half_width + '┼' + '─' * self.half_width)
        for y in range(self.height):
            left = ''.join(side_canvas[y])
            right = ''.join(top_canvas[y])
            lines.append(f"{left}│{right}")

        # Info-Zeile
        gp = positions["gripper"]
        lines.append('─' * (self.half_width * 2 + 1))
        lines.append(
            f" b={b_deg:+6.1f}° s={s_deg:+6.1f}° e={e_deg:+6.1f}° "
            f"│ Gripper: ({gp[0]:.0f}, {gp[1]:.0f}, {gp[2]:.0f})mm"
        )

        return '\n'.join(lines)

    def _create_canvas(self, w, h):
        return [[' ' for _ in range(w)] for _ in range(h)]

    def _put_char(self, canvas, x, y, ch):
        h = len(canvas)
        w = len(canvas[0]) if canvas else 0
        if 0 <= x < w and 0 <= y < h:
            canvas[y][x] = ch

    def _draw_line(self, canvas, p1, p2, ch):
        """Bresenham-ähnliche Linie."""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy, 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self._put_char(canvas, x, y, ch)


# ============================================================
# SEQUENCE FORMAT (YAML)
# ============================================================

"""
Sequence-Format (sequences/*.yaml):

name: "Pick and Place"
description: "Greift Objekt und legt es ab"
steps:
  - type: play
    file: recordings/grab_approach.roarm
    speed: 1.0
  - type: wait
    duration: 0.5
  - type: gripper
    action: close
  - type: play
    file: recordings/place_object.roarm
    speed: 0.8
  - type: gripper
    action: open
  - type: goto
    step: 0
    repeat: 3
"""


def load_sequence(filepath: str) -> dict:
    """Lädt eine Sequence-YAML-Datei."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_sequence(filepath: str, sequence: dict):
    """Speichert eine Sequence-YAML-Datei."""
    with open(filepath, 'w') as f:
        yaml.dump(sequence, f, default_flow_style=False, allow_unicode=True)


# ============================================================
# RECORDING PARSER (aus play.py übernommen)
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

/* Die Horizontal-Container in den Tabs brauchen height */
.tab-content > Horizontal {
    height: 1fr;
}

/* Linke Spalte (Arm-View + Buttons) */
.tab-content > Horizontal > Vertical {
    width: 1fr;
    height: 1fr;
}

.arm-view {
    border: solid $primary;
    height: 28;
    min-height: 20;
    padding: 0 1;
    overflow: hidden;
}

.arm-view-label {
    text-align: center;
    color: $text-muted;
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

.joint-label {
    width: auto;
    color: $text;
}

.sparkline-container {
    height: 3;
    padding: 0 1;
}

.recording-list {
    height: 1fr;
    border: solid $accent;
}

.timeline-container {
    height: 5;
    border: solid $warning;
    padding: 1;
}

.sequence-editor {
    height: 1fr;
    width: 1fr;
}

.control-buttons {
    height: 3;
    align: center middle;
    dock: bottom;
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

.sequence-step {
    height: 3;
    padding: 0 1;
    border: solid $panel;
    margin: 0 0 1 0;
}

.timeline-bar {
    height: 1;
    background: $panel;
}

.timeline-cursor {
    color: $error;
}

.safety-ok {
    color: $success;
}

.safety-warn {
    color: $warning;
}

.safety-error {
    color: $error;
}

/* Fix: RichLog braucht Scrollbar-Platz */
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
"""

# ============================================================
# CUSTOM WIDGETS
# ============================================================

class ArmAsciiWidget(Static):
    """Widget das den ASCII-Arm rendert."""

    b = reactive(0.0)
    s = reactive(0.0)
    e = reactive(90.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._renderer = Ascii3DView(width=70, height=20)
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

    def update_value(self, value: float, safety_pct: float = 0.0):
        """
        value: aktueller Winkel
        safety_pct: 0.0-1.0, wie nah am Limit (für Farbcodierung)
        """
        self._value = value
        self._history.append(value)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._refresh()

    def _refresh(self):
        color = self.JOINT_COLORS.get(self.joint, "white")
        name = self.JOINT_NAMES.get(self.joint, self.joint)

        # Sparkline aus History
        sparkline = self._make_sparkline()

        # Safety-Farbe
        safety_color = "green"
        limits = {"b": 135, "s": 90, "e": 180, "h": 360}
        limit = limits.get(self.joint, 180)
        pct = abs(self._value) / limit
        if pct > 0.9:
            safety_color = "red"
        elif pct > 0.7:
            safety_color = "yellow"

        text = (
            f"[{color}]{name}[/] "
            f"[bold {safety_color}]{self._value:+7.2f}°[/] "
            f"[dim]{sparkline}[/]"
        )
        self.update(text)

    def _make_sparkline(self) -> str:
        """Erstellt eine Text-Sparkline."""
        if not self._history:
            return "▁" * 20

        bars = "▁▂▃▄▅▆▇█"
        values = self._history[-20:]  # Letzte 20

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
    """Zeigt eine interaktive Timeline für Recordings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._duration = 0.0
        self._position = 0.0
        self._waypoints = []
        self._gripper_events = []
        self._width = 60

    def set_recording(self, waypoints: list, gripper_cmds: list = None):
        self._waypoints = waypoints
        self._gripper_events = gripper_cmds or []
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

        # Timeline-Bar
        bar = list("─" * bar_width)

        # Waypoint-Dichte als Intensität
        for wp in self._waypoints[::max(1, len(self._waypoints) // bar_width)]:
            idx = int((wp["t"] / self._duration) * bar_width)
            idx = max(0, min(bar_width - 1, idx))
            bar[idx] = "━"

        # Gripper-Events
        for gc in self._gripper_events:
            idx = int((gc["t"] / self._duration) * bar_width)
            idx = max(0, min(bar_width - 1, idx))
            bar[idx] = "✦" if gc["cmd"] == "CLOSE" else "✧"

        # Cursor
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
    """RoArm-M2-S Unified TUI Dashboard."""

    TITLE = "RoArm-M2-S Dashboard"
    SUB_TITLE = "Teach · Play · Sequence · Timeline"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_tab('teach')", "Teach", show=True),
        Binding("2", "switch_tab('play')", "Play", show=True),
        Binding("3", "switch_tab('sequence')", "Sequence", show=True),
        Binding("4", "switch_tab('timeline')", "Timeline", show=True),
        Binding("v", "toggle_viz", "3D Viz", show=True),
        Binding("c", "connect", "Connect", show=True),
        Binding("space", "toggle_action", "Start/Stop", show=True),
        Binding("g", "gripper_toggle", "Gripper", show=True),
    ]

    # --- Reactive State ---
    connected = reactive(False)
    recording = reactive(False)
    playing = reactive(False)
    viz_active = reactive(False)

    def __init__(self):
        super().__init__()
        self._arm: Optional[RoArmConnection] = None
        self._safe_arm: Optional[SafeArm] = None
        self._viz: Optional[RobotVisualizer] = None
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
        self._play_timer: Optional[Timer] = None

        # Sequence state
        self._current_sequence = {"name": "Neue Sequence", "steps": []}

    # ============================================================
    # COMPOSE (Layout)
    # ============================================================

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            with TabbedContent():
                # --- TAB 1: TEACH ---
                with TabPane("🎬 Teach", id="teach"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical(id="teach-left"):
                                yield ArmAsciiWidget(
                                    id="teach-arm-view",
                                    classes="arm-view"
                                )
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "⏺ Record", id="btn-teach-record",
                                        classes="btn-record", variant="error"
                                    )
                                    yield Button(
                                        "⏹ Stop", id="btn-teach-stop",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "🏠 Home", id="btn-teach-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "✊/✋ Gripper", id="btn-gripper",
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
                with TabPane("▶️ Play", id="play"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield ArmAsciiWidget(
                                    id="play-arm-view",
                                    classes="arm-view"
                                )
                                yield TimelineWidget(id="play-timeline",
                                                     classes="timeline-container")
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "▶ Play", id="btn-play-start",
                                        classes="btn-play", variant="success"
                                    )
                                    yield Button(
                                        "⏹ Stop", id="btn-play-stop",
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

                # --- TAB 3: SEQUENCE ---
                with TabPane("📋 Sequence", id="sequence"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical(classes="sequence-editor"):
                                yield Label("[bold]Sequence Builder[/]")
                                yield Input(
                                    placeholder="Sequence Name...",
                                    id="seq-name-input"
                                )
                                yield DataTable(id="sequence-steps-table")
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "+ Play Step", id="btn-seq-add-play",
                                        variant="primary"
                                    )
                                    yield Button(
                                        "+ Wait", id="btn-seq-add-wait",
                                        variant="default"
                                    )
                                    yield Button(
                                        "+ Gripper", id="btn-seq-add-gripper",
                                        variant="default"
                                    )
                                    yield Button(
                                        "🗑 Remove", id="btn-seq-remove",
                                        variant="error"
                                    )

                            with Vertical():
                                yield Label("[bold]Gespeicherte Sequences:[/]")
                                yield DataTable(id="sequences-list-table")
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "▶ Run Sequence", id="btn-seq-run",
                                        classes="btn-play", variant="success"
                                    )
                                    yield Button(
                                        "💾 Save", id="btn-seq-save",
                                        variant="primary"
                                    )
                                    yield Button(
                                        "📂 Load", id="btn-seq-load",
                                        variant="default"
                                    )

                # --- TAB 4: TIMELINE ---
                with TabPane("⏱️ Timeline", id="timeline"):
                    with Vertical(classes="tab-content"):
                        yield ArmAsciiWidget(
                            id="timeline-arm-view",
                            classes="arm-view"
                        )
                        yield TimelineWidget(id="timeline-widget",
                                             classes="timeline-container")
                        with Horizontal(classes="control-buttons"):
                            yield Button(
                                "⏮", id="btn-tl-start", variant="default"
                            )
                            yield Button(
                                "◀", id="btn-tl-back", variant="default"
                            )
                            yield Button(
                                "▶/⏸", id="btn-tl-playpause",
                                variant="success"
                            )
                            yield Button(
                                "▶", id="btn-tl-forward", variant="default"
                            )
                            yield Button(
                                "⏭", id="btn-tl-end", variant="default"
                            )
                            yield Button(
                                "🗑 Delete WP", id="btn-tl-delete",
                                variant="error"
                            )
                            yield Button(
                                "⏸ Insert Pause", id="btn-tl-pause",
                                variant="warning"
                            )
                        yield DataTable(id="timeline-waypoints-table")

        # Status-Bar
        with Horizontal(classes="status-bar"):
            yield Label("🔌 Disconnected", id="status-connection")
            yield Label("│", id="status-sep1")
            yield Label("🌡️ --°C", id="status-temp")
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
        # Recordings-Tabelle befüllen
        self._refresh_recordings_table()
        self._refresh_sequences_table()
        self._setup_sequence_steps_table()
        self._setup_timeline_table()

        # Auto-Connect versuchen
        self.call_after_refresh(self._try_auto_connect)

    def _try_auto_connect(self):
        """Versucht automatisch den Arm zu finden."""
        port = find_arm_port()
        if port:
            self._log_teach(f"[dim]Port gefunden: {port}[/]")
            self._log_teach("[dim]Drücke [c] zum Verbinden[/]")
        else:
            self._log_teach("[yellow]Kein Arm-Port gefunden[/]")

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
        for rec in recordings[:20]:  # Max 20 anzeigen
            try:
                data = parse_roarm_file(str(rec))
                wps = data["waypoints"]
                duration = wps[-1]["t"] if wps else 0
                date = rec.stem.replace("recording_", "")
                table.add_row(
                    rec.name,
                    f"{duration:.1f}s",
                    str(len(wps)),
                    date,
                )
            except Exception:
                table.add_row(rec.name, "?", "?", "?")

    def _refresh_sequences_table(self):
        """Aktualisiert die Sequences-Liste."""
        try:
            table = self.query_one("#sequences-list-table", DataTable)
        except NoMatches:
            return

        table.clear(columns=True)
        table.add_columns("Name", "Steps", "Datei")

        sequences = sorted(SEQUENCES_DIR.glob("*.yaml"))
        for seq_file in sequences:
            try:
                seq = load_sequence(str(seq_file))
                name = seq.get("name", seq_file.stem)
                steps = len(seq.get("steps", []))
                table.add_row(name, str(steps), seq_file.name)
            except Exception:
                table.add_row(seq_file.stem, "?", seq_file.name)

    def _setup_sequence_steps_table(self):
        """Richtet die Sequence-Steps-Tabelle ein."""
        try:
            table = self.query_one("#sequence-steps-table", DataTable)
        except NoMatches:
            return

        table.clear(columns=True)
        table.add_columns("#", "Type", "Details", "Speed")

    def _setup_timeline_table(self):
        """Richtet die Timeline-Waypoints-Tabelle ein."""
        try:
            table = self.query_one("#timeline-waypoints-table", DataTable)
        except NoMatches:
            return

        table.clear(columns=True)
        table.add_columns("#", "t [s]", "Base", "Shoulder", "Elbow", "Hand", "Cmd")

    # ============================================================
    # LOGGING HELPERS
    # ============================================================

    def _log_teach(self, msg: str):
        try:
            log = self.query_one("#teach-log", RichLog)
            log.write(msg)
        except NoMatches:
            pass

    def _log_play(self, msg: str):
        try:
            log = self.query_one("#play-log", RichLog)
            log.write(msg)
        except NoMatches:
            pass

    # ============================================================
    # ACTIONS
    # ============================================================

    def action_switch_tab(self, tab_id: str) -> None:
        """Wechselt zum angegebenen Tab."""
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = tab_id
        except NoMatches:
            pass

    def action_toggle_viz(self) -> None:
        """Startet/Stoppt die externe 3D-Visualisierung."""
        if self._viz is not None and self._viz.is_running:
            self._viz.stop()
            self._viz = None
            self.viz_active = False
            self._log_teach("[dim]3D-Visualisierung gestoppt[/]")
        else:
            self._viz = RobotVisualizer(live=True, update_interval=0.05)
            self._viz.start()
            self.viz_active = True
            self._log_teach("[green]3D-Visualisierung gestartet[/]")
            # Aktuelle Position senden
            if self._arm:
                pos = self._arm.read_position_deg()
                if pos:
                    self._viz.update_pose(
                        pos["b"], pos["s"], pos["e"], pos["h"]
                    )

    def action_connect(self) -> None:
        """Verbindet mit dem Arm."""
        if self.connected:
            self._disconnect()
            return
        self._connect()

    @work(thread=True)
    def _connect(self) -> None:
        """Verbindung im Hintergrund-Thread."""
        port = find_arm_port()
        if port is None:
            self.app.call_from_thread(
                self._log_teach, "[red]❌ Kein Port gefunden![/]"
            )
            return

        try:
            self._arm = RoArmConnection(port)
            self.connected = True
            self.app.call_from_thread(
                self._log_teach,
                f"[green]✅ Verbunden mit {port}[/]"
            )
            self.app.call_from_thread(
                self._update_status_connection, port
            )

            # Position lesen
            pos = self._arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self.app.call_from_thread(self._update_joint_displays, pos)
                self.app.call_from_thread(self._update_arm_views, pos)

        except Exception as e:
            self.app.call_from_thread(
                self._log_teach, f"[red]❌ Fehler: {e}[/]"
            )

    def _disconnect(self):
        """Trennt die Verbindung."""
        if self._arm:
            self._arm.close()
            self._arm = None
        self.connected = False
        self._log_teach("[yellow]🔌 Getrennt[/]")
        self._update_status_connection(None)

    def _update_status_connection(self, port: str = None):
        """Aktualisiert die Status-Bar."""
        try:
            label = self.query_one("#status-connection", Label)
            if port:
                label.update(f"🔌 {port}")
            else:
                label.update("🔌 Disconnected")
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
        """Aktualisiert alle ASCII-Arm-Ansichten."""
        for view_id in ["teach-arm-view", "play-arm-view", "timeline-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", ArmAsciiWidget)
                widget.update_pose(pos["b"], pos["s"], pos["e"])
            except NoMatches:
                pass

        # Externe 3D-Viz
        if self._viz and self._viz.is_running:
            self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])

    # ============================================================
    # TEACH MODE
    # ============================================================

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
        """Gripper öffnen/schließen."""
        if not self._arm or not self.connected:
            return

        if self._gripper_open:
            self._arm.gripper_close()
            self._gripper_open = False
            self._log_teach("[bold]✊ Gripper ZU[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append({
                    "t": round(elapsed, 4), "cmd": "GRIPPER_CLOSE"
                })
        else:
            self._arm.gripper_open()
            self._gripper_open = True
            self._log_teach("[bold]✋ Gripper AUF[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append({
                    "t": round(elapsed, 4), "cmd": "GRIPPER_OPEN"
                })

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
        """Startet die Aufnahme."""
        if not self.connected or not self._arm:
            self._log_teach("[red]Nicht verbunden![/]")
            return

        self.recording = True
        self._teach_waypoints = []
        self._teach_start_time = time.time()
        self._gripper_open = True

        # Torque aus
        self._arm.torque_off()

        # Buttons updaten
        try:
            self.query_one("#btn-teach-record", Button).disabled = True
            self.query_one("#btn-teach-stop", Button).disabled = False
        except NoMatches:
            pass

        # Arm-View Trail löschen
        try:
            arm_view = self.query_one("#teach-arm-view", ArmAsciiWidget)
            arm_view.clear_trail()
        except NoMatches:
            pass

        # 3D-Viz Trail löschen
        if self._viz and self._viz.is_running:
            self._viz.clear_trail()

        self._log_teach("[bold red]⏺ AUFNAHME LÄUFT[/]")
        self._log_teach("[dim]Bewege den Arm! [Space]=Stop [g]=Gripper[/]")

        # Timer starten für Position-Polling
        self._teach_timer = self.set_interval(
            1.0 / RECORD_HZ, self._teach_poll_position
        )

    def _teach_poll_position(self):
        """Wird vom Timer aufgerufen - liest Position."""
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
                    abs(pos[j] - last_move[j])
                    for j in ["b", "s", "e", "h"]
                )
                if max_delta < MOVE_THRESHOLD_DEG:
                    should_record = False

        if should_record:
            self._teach_waypoints.append({
                "t": round(elapsed, 4),
                "b": pos["b"],
                "s": pos["s"],
                "e": pos["e"],
                "h": pos["h"],
            })

        # UI updaten
        self._update_joint_displays(pos)
        self._update_arm_views(pos)

        # Status im Log (alle 50 Frames)
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if len(move_wps) % 50 == 0 and len(move_wps) > 0:
            self._log_teach(
                f"[dim]  ◆ WP#{len(move_wps)} [{elapsed:.1f}s] "
                f"b={pos['b']:+.1f}° s={pos['s']:+.1f}° "
                f"e={pos['e']:+.1f}° h={pos['h']:+.1f}°[/]"
            )

    def _stop_recording(self):
        """Stoppt die Aufnahme und speichert."""
        if not self.recording:
            return

        self.recording = False

        # Timer stoppen
        if self._teach_timer:
            self._teach_timer.stop()
            self._teach_timer = None

        # Torque an
        if self._arm:
            self._arm.torque_on()

        # Buttons updaten
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
            f"[green]⏹ Aufnahme gestoppt: {len(move_wps)} WPs, "
            f"{duration:.1f}s[/]"
        )

        # Speichern
        filepath = self._save_recording()
        if filepath:
            self._log_teach(f"[green]💾 Gespeichert: {filepath}[/]")
            self._refresh_recordings_table()

    def _save_recording(self) -> Optional[str]:
        """Speichert die aktuelle Aufnahme."""
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        if not move_wps:
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RECORDINGS_DIR / f"recording_{ts}.roarm"

        lines = [
            f"# RoArm-M2-S Recording (Dashboard)",
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
        """Fährt zur Home-Position."""
        if not self._arm or not self.connected:
            return

        self.app.call_from_thread(
            self._log_teach, "[dim]🏠 Fahre zur Home-Position...[/]"
        )

        self._arm.torque_on()
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
        """Startet die Wiedergabe des ausgewählten Recordings."""
        if not self.connected or not self._arm:
            self._log_play("[red]Nicht verbunden![/]")
            return

        # Ausgewähltes Recording finden
        try:
            table = self.query_one("#recordings-table", DataTable)
            row_key = table.cursor_row
            if row_key is None:
                self._log_play("[yellow]Kein Recording ausgewählt![/]")
                return
            # Dateiname aus erster Spalte
            row_data = table.get_row_at(row_key)
            filename = row_data[0]
        except (NoMatches, Exception) as e:
            self._log_play(f"[red]Fehler: {e}[/]")
            return

        filepath = RECORDINGS_DIR / filename
        if not filepath.exists():
            self._log_play(f"[red]Datei nicht gefunden: {filepath}[/]")
            return

        self._log_play(f"[dim]Lade: {filename}[/]")

        # Parsen
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

        # Buttons
        try:
            self.query_one("#btn-play-start", Button).disabled = True
            self.query_one("#btn-play-stop", Button).disabled = False
        except NoMatches:
            pass

        self._log_play(
            f"[green]▶ Playback: {len(wps)} WPs, "
            f"{wps[-1]['t']:.1f}s[/]"
        )

        # Playback im Worker-Thread
        self._run_playback(wps)

    @work(thread=True)
    def _run_playback(self, waypoints: list):
        """Führt das Playback im Hintergrund aus."""
        from scipy.interpolate import CubicSpline

        # Spline erstellen
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
        time.sleep(2.0)

        # Streaming
        self._play_start_time = time.time()
        last_pos = None

        while self.playing:
            loop_start = time.time()
            elapsed = loop_start - self._play_start_time

            if elapsed >= duration:
                break

            # Sample
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

            if should_send:
                self._arm.move_to_fast(
                    target["b"], target["s"], target["e"], target["h"],
                    spd=50, acc=30
                )
                last_pos = target.copy()

            # UI updaten (nicht jeden Frame)
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
        self._log_play("[green]✅ Playback beendet[/]")
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
    # SEQUENCE MODE
    # ============================================================

    @on(Button.Pressed, "#btn-seq-add-play")
    def on_seq_add_play(self) -> None:
        """Fügt einen Play-Step zur Sequence hinzu."""
        # Ausgewähltes Recording nehmen
        try:
            table = self.query_one("#recordings-table", DataTable)
            row_key = table.cursor_row
            if row_key is not None:
                row_data = table.get_row_at(row_key)
                filename = row_data[0]
            else:
                filename = "recording.roarm"
        except (NoMatches, Exception):
            filename = "recording.roarm"

        step = {
            "type": "play",
            "file": str(RECORDINGS_DIR / filename),
            "speed": 1.0,
        }
        self._current_sequence["steps"].append(step)
        self._refresh_sequence_steps()

    @on(Button.Pressed, "#btn-seq-add-wait")
    def on_seq_add_wait(self) -> None:
        """Fügt einen Wait-Step hinzu."""
        step = {"type": "wait", "duration": 1.0}
        self._current_sequence["steps"].append(step)
        self._refresh_sequence_steps()

    @on(Button.Pressed, "#btn-seq-add-gripper")
    def on_seq_add_gripper(self) -> None:
        """Fügt einen Gripper-Step hinzu."""
        step = {"type": "gripper", "action": "close"}
        self._current_sequence["steps"].append(step)
        self._refresh_sequence_steps()

    @on(Button.Pressed, "#btn-seq-remove")
    def on_seq_remove(self) -> None:
        """Entfernt den ausgewählten Step."""
        try:
            table = self.query_one("#sequence-steps-table", DataTable)
            row_key = table.cursor_row
            if row_key is not None and row_key < len(self._current_sequence["steps"]):
                self._current_sequence["steps"].pop(row_key)
                self._refresh_sequence_steps()
        except (NoMatches, Exception):
            pass

    @on(Button.Pressed, "#btn-seq-save")
    def on_seq_save(self) -> None:
        """Speichert die aktuelle Sequence."""
        try:
            name_input = self.query_one("#seq-name-input", Input)
            name = name_input.value.strip() or "Neue Sequence"
        except NoMatches:
            name = "Neue Sequence"

        self._current_sequence["name"] = name
        filename = SEQUENCES_DIR / f"{name.lower().replace(' ', '_')}.yaml"
        save_sequence(str(filename), self._current_sequence)
        self._log_teach(f"[green]💾 Sequence gespeichert: {filename}[/]")
        self._refresh_sequences_table()

    @on(Button.Pressed, "#btn-seq-run")
    def on_seq_run(self) -> None:
        """Führt die aktuelle Sequence aus."""
        if not self.connected or not self._arm:
            self._log_teach("[red]Nicht verbunden![/]")
            return

        if not self._current_sequence["steps"]:
            self._log_teach("[yellow]Sequence ist leer![/]")
            return

        self._run_sequence()

    @work(thread=True)
    def _run_sequence(self):
        """Führt die Sequence im Hintergrund aus."""
        steps = self._current_sequence["steps"]
        self.app.call_from_thread(
            self._log_teach,
            f"[bold green]▶ Sequence '{self._current_sequence['name']}' "
            f"({len(steps)} Steps)[/]"
        )

        for step_idx, step in enumerate(steps):
            step_type = step.get("type", "unknown")

            self.app.call_from_thread(
                self._log_teach,
                f"  [dim]Step {step_idx + 1}/{len(steps)}: {step_type}[/]"
            )

            if step_type == "play":
                filepath = step.get("file", "")
                speed = step.get("speed", 1.0)

                if not Path(filepath).exists():
                    self.app.call_from_thread(
                        self._log_teach,
                        f"  [red]❌ Datei nicht gefunden: {filepath}[/]"
                    )
                    continue

                data = parse_roarm_file(filepath)
                wps = data["waypoints"]
                if not wps or len(wps) < 4:
                    self.app.call_from_thread(
                        self._log_teach,
                        f"  [yellow]⚠️ Zu wenige WPs in {filepath}[/]"
                    )
                    continue

                # Spline erstellen und abspielen
                from scipy.interpolate import CubicSpline

                times = np.array([wp["t"] for wp in wps])
                splines = {}
                for joint in ["b", "s", "e", "h"]:
                    values = np.array([wp[joint] for wp in wps])
                    splines[joint] = CubicSpline(
                        times, values, bc_type='clamped'
                    )

                duration = times[-1] / speed
                interval = 1.0 / STREAM_HZ

                # Zur Startposition
                start = wps[0]
                self._arm.torque_on()
                time.sleep(0.2)
                self._arm.move_to(
                    start["b"], start["s"], start["e"], start["h"],
                    spd=20, acc=10
                )
                time.sleep(2.0)

                # Streaming
                play_start = time.time()
                last_pos = None

                while True:
                    loop_start = time.time()
                    elapsed = (loop_start - play_start) * speed

                    if elapsed >= times[-1]:
                        break

                    target = {}
                    for joint in ["b", "s", "e", "h"]:
                        target[joint] = round(
                            float(splines[joint](elapsed)), 2
                        )

                    should_send = True
                    if last_pos:
                        max_delta = max(
                            abs(target[j] - last_pos[j])
                            for j in ["b", "s", "e", "h"]
                        )
                        if max_delta < 0.02:
                            should_send = False

                    if should_send:
                        self._arm.move_to_fast(
                            target["b"], target["s"],
                            target["e"], target["h"],
                            spd=50, acc=30
                        )
                        last_pos = target.copy()

                    # UI update (throttled)
                    if int(elapsed * 5) % 2 == 0:
                        self.app.call_from_thread(
                            self._update_arm_views, target
                        )

                    loop_elapsed = time.time() - loop_start
                    sleep_time = interval - loop_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self.app.call_from_thread(
                    self._log_teach,
                    f"  [green]  ✅ Play '{Path(filepath).name}' fertig[/]"
                )

            elif step_type == "wait":
                duration = step.get("duration", 1.0)
                self.app.call_from_thread(
                    self._log_teach,
                    f"  [dim]  ⏳ Warte {duration:.1f}s...[/]"
                )
                time.sleep(duration)

            elif step_type == "gripper":
                action = step.get("action", "open")
                if action == "close":
                    self._arm.gripper_close()
                    self.app.call_from_thread(
                        self._log_teach, "  [bold]  ✊ Gripper ZU[/]"
                    )
                else:
                    self._arm.gripper_open()
                    self.app.call_from_thread(
                        self._log_teach, "  [bold]  ✋ Gripper AUF[/]"
                    )
                time.sleep(0.5)

            elif step_type == "goto":
                goto_step = step.get("step", 0)
                repeat = step.get("repeat", 1)
                # Einfache Implementierung: Ignoriere goto für jetzt
                self.app.call_from_thread(
                    self._log_teach,
                    f"  [dim]  🔁 Goto step {goto_step} "
                    f"(repeat={repeat}) - TODO[/]"
                )

        self.app.call_from_thread(
            self._log_teach,
            f"[bold green]✅ Sequence '{self._current_sequence['name']}' "
            f"abgeschlossen![/]"
        )

    @on(Button.Pressed, "#btn-seq-load")
    def on_seq_load(self) -> None:
        """Lädt die ausgewählte Sequence."""
        try:
            table = self.query_one("#sequences-list-table", DataTable)
            row_key = table.cursor_row
            if row_key is None:
                return
            row_data = table.get_row_at(row_key)
            filename = row_data[2]  # Datei-Spalte
        except (NoMatches, Exception):
            return

        filepath = SEQUENCES_DIR / filename
        if not filepath.exists():
            return

        try:
            self._current_sequence = load_sequence(str(filepath))
            # Name-Input updaten
            try:
                name_input = self.query_one("#seq-name-input", Input)
                name_input.value = self._current_sequence.get("name", "")
            except NoMatches:
                pass
            self._refresh_sequence_steps()
            self._log_teach(
                f"[green]📂 Sequence geladen: {filename}[/]"
            )
        except Exception as e:
            self._log_teach(f"[red]Fehler beim Laden: {e}[/]")

    def _refresh_sequence_steps(self):
        """Aktualisiert die Sequence-Steps-Tabelle."""
        try:
            table = self.query_one("#sequence-steps-table", DataTable)
        except NoMatches:
            return

        table.clear()
        for i, step in enumerate(self._current_sequence.get("steps", [])):
            step_type = step.get("type", "?")
            details = ""
            speed = ""

            if step_type == "play":
                details = Path(step.get("file", "?")).name
                speed = f"{step.get('speed', 1.0):.1f}x"
            elif step_type == "wait":
                details = f"{step.get('duration', 0):.1f}s"
            elif step_type == "gripper":
                details = step.get("action", "?")
            elif step_type == "goto":
                details = (
                    f"→ step {step.get('step', 0)} "
                    f"(×{step.get('repeat', 1)})"
                )

            table.add_row(str(i + 1), step_type, details, speed)

    # ============================================================
    # TIMELINE MODE
    # ============================================================

    @on(Button.Pressed, "#btn-tl-start")
    def on_tl_start(self) -> None:
        """Springt zum Anfang der Timeline."""
        self._timeline_seek(0.0)

    @on(Button.Pressed, "#btn-tl-end")
    def on_tl_end(self) -> None:
        """Springt zum Ende der Timeline."""
        if self._play_data and self._play_data["waypoints"]:
            duration = self._play_data["waypoints"][-1]["t"]
            self._timeline_seek(duration)

    @on(Button.Pressed, "#btn-tl-back")
    def on_tl_back(self) -> None:
        """Springt 0.5s zurück."""
        try:
            timeline = self.query_one("#timeline-widget", TimelineWidget)
            new_t = max(0, timeline._position - 0.5)
            self._timeline_seek(new_t)
        except NoMatches:
            pass

    @on(Button.Pressed, "#btn-tl-forward")
    def on_tl_forward(self) -> None:
        """Springt 0.5s vorwärts."""
        try:
            timeline = self.query_one("#timeline-widget", TimelineWidget)
            if self._play_data and self._play_data["waypoints"]:
                duration = self._play_data["waypoints"][-1]["t"]
                new_t = min(duration, timeline._position + 0.5)
                self._timeline_seek(new_t)
        except NoMatches:
            pass

    @on(Button.Pressed, "#btn-tl-delete")
    def on_tl_delete(self) -> None:
        """Löscht den ausgewählten Waypoint."""
        if not self._play_data:
            return
        try:
            table = self.query_one("#timeline-waypoints-table", DataTable)
            row_key = table.cursor_row
            if row_key is not None and row_key < len(
                self._play_data["waypoints"]
            ):
                self._play_data["waypoints"].pop(row_key)
                self._refresh_timeline_data()
                self._log_teach(
                    f"[yellow]🗑 Waypoint #{row_key + 1} gelöscht[/]"
                )
        except (NoMatches, Exception):
            pass

    @on(Button.Pressed, "#btn-tl-pause")
    def on_tl_pause(self) -> None:
        """Fügt eine Pause (Zeitverschiebung) ein."""
        if not self._play_data or not self._play_data["waypoints"]:
            return

        try:
            table = self.query_one("#timeline-waypoints-table", DataTable)
            row_key = table.cursor_row
            if row_key is None:
                return
        except NoMatches:
            return

        # Alle Waypoints nach dem ausgewählten um 0.5s verschieben
        pause_duration = 0.5
        wps = self._play_data["waypoints"]
        if row_key < len(wps):
            insert_t = wps[row_key]["t"]
            for wp in wps[row_key:]:
                wp["t"] += pause_duration
            self._refresh_timeline_data()
            self._log_teach(
                f"[yellow]⏸ {pause_duration}s Pause eingefügt "
                f"bei t={insert_t:.2f}s[/]"
            )

    def _timeline_seek(self, t: float):
        """Springt zu einem Zeitpunkt in der Timeline."""
        if not self._play_data or not self._play_data["waypoints"]:
            return

        wps = self._play_data["waypoints"]

        # Timeline-Widget updaten
        try:
            timeline = self.query_one("#timeline-widget", TimelineWidget)
            timeline.set_position(t)
        except NoMatches:
            pass

        # Nächsten Waypoint finden und Arm-View updaten
        # Lineare Interpolation zwischen Waypoints
        pos = self._interpolate_at_time(wps, t)
        if pos:
            try:
                arm_view = self.query_one(
                    "#timeline-arm-view", ArmAsciiWidget
                )
                arm_view.update_pose(pos["b"], pos["s"], pos["e"])
            except NoMatches:
                pass

            # Wenn verbunden, auch den echten Arm bewegen
            if self.connected and self._arm:
                self._arm.move_to(
                    pos["b"], pos["s"], pos["e"], pos["h"],
                    spd=15, acc=8
                )

    def _interpolate_at_time(self, waypoints: list, t: float) -> dict:
        """Lineare Interpolation zwischen Waypoints."""
        if not waypoints:
            return None

        # Vor dem ersten Punkt
        if t <= waypoints[0]["t"]:
            return waypoints[0]

        # Nach dem letzten Punkt
        if t >= waypoints[-1]["t"]:
            return waypoints[-1]

        # Zwischen zwei Punkten interpolieren
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            if wp1["t"] <= t <= wp2["t"]:
                dt = wp2["t"] - wp1["t"]
                if dt < 0.0001:
                    return wp1
                alpha = (t - wp1["t"]) / dt
                return {
                    "t": t,
                    "b": wp1["b"] + alpha * (wp2["b"] - wp1["b"]),
                    "s": wp1["s"] + alpha * (wp2["s"] - wp1["s"]),
                    "e": wp1["e"] + alpha * (wp2["e"] - wp1["e"]),
                    "h": wp1["h"] + alpha * (wp2["h"] - wp1["h"]),
                }

        return waypoints[-1]

    def _refresh_timeline_data(self):
        """Aktualisiert Timeline-Widget und Waypoints-Tabelle."""
        if not self._play_data:
            return

        wps = self._play_data["waypoints"]

        # Timeline-Widget
        try:
            timeline = self.query_one("#timeline-widget", TimelineWidget)
            timeline.set_recording(wps, self._play_data.get("gripper_cmds"))
        except NoMatches:
            pass

        # Waypoints-Tabelle
        try:
            table = self.query_one("#timeline-waypoints-table", DataTable)
            table.clear()
            for i, wp in enumerate(wps[:100]):  # Max 100 anzeigen
                table.add_row(
                    str(i + 1),
                    f"{wp['t']:.3f}",
                    f"{wp['b']:.2f}",
                    f"{wp['s']:.2f}",
                    f"{wp['e']:.2f}",
                    f"{wp.get('h', 180.0):.2f}",
                    "",
                )
            # Gripper-Events auch anzeigen
            for gc in self._play_data.get("gripper_cmds", []):
                table.add_row(
                    "G",
                    f"{gc['t']:.3f}",
                    "-", "-", "-", "-",
                    gc["cmd"],
                )
        except NoMatches:
            pass

    # ============================================================
    # RECORDINGS TABLE SELECTION → TIMELINE LOAD
    # ============================================================

    @on(DataTable.RowSelected, "#recordings-table")
    def on_recording_selected(self, event: DataTable.RowSelected) -> None:
        """Wenn ein Recording ausgewählt wird, lade es in die Timeline."""
        try:
            table = self.query_one("#recordings-table", DataTable)
            row_data = table.get_row_at(event.cursor_row)
            filename = row_data[0]
        except (NoMatches, Exception):
            return

        filepath = RECORDINGS_DIR / filename
        if not filepath.exists():
            return

        try:
            self._play_data = parse_roarm_file(str(filepath))
            self._refresh_timeline_data()
        except Exception:
            pass

    # ============================================================
    # WATCH: REACTIVE CHANGES
    # ============================================================

    def watch_connected(self, connected: bool) -> None:
        """Reagiert auf Verbindungsänderungen."""
        try:
            mode_label = self.query_one("#status-mode", Label)
            if connected:
                mode_label.update("⏱️ Ready")
            else:
                mode_label.update("⏱️ --")
        except NoMatches:
            pass

    def watch_recording(self, recording: bool) -> None:
        """Reagiert auf Recording-Status."""
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
        """Reagiert auf Playback-Status."""
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
# KONSTANTE FÜR PLAYBACK (fehlte oben)
# ============================================================

MIN_DELTA_DEG = 0.02


# ============================================================
# MAIN
# ============================================================

def main():
    """Startet das Dashboard."""
    app = RoArmDashboard()
    app.run()


if __name__ == "__main__":
    main()

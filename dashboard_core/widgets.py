import math
from typing import Optional

import numpy as np
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

from .braille import BrailleCanvas
from .kinematics import (
    forward_kinematics, UPPER_ARM, FOREARM, GRIPPER_LENGTH,
)


class Arm3DWidget(Static):
    """Widget das den 3D-Arm als hochaufloesende Braille-Grafik rendert."""

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
        self._refresh_display()

    def update_pose(self, b: float, s: float, e: float, target: dict = None):
        self.b = b
        self.s = s
        self.e = e
        self._target = target

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
        az = math.radians(self._cam_azimuth)
        el = math.radians(self._cam_elevation)

        x1 = x * math.cos(az) - y * math.sin(az)
        y1 = x * math.sin(az) + y * math.cos(az)
        z1 = z

        y2 = y1 * math.cos(el) - z1 * math.sin(el)
        z2 = y1 * math.sin(el) + z1 * math.cos(el)
        x2 = x1

        d = self._cam_distance
        scale = d / (d + y2 + 300)

        px = int(x2 * scale * 0.35 + canvas_w // 2)
        py = int(-z2 * scale * 0.35 + canvas_h * 0.72)

        return px, py

    def _refresh_display(self):
        w = self.size.width if self.size.width > 0 else 70
        h = self.size.height if self.size.height > 0 else 24

        canvas = BrailleCanvas(w, h)
        pw = canvas.px_width
        ph = canvas.px_height

        positions = forward_kinematics(self.b, self.s, self.e)

        # Boden-Gitter
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

        # Koordinatenachsen
        origin = self._project_3d(0, 0, 0, pw, ph)
        ax_x = self._project_3d(80, 0, 0, pw, ph)
        ax_y = self._project_3d(0, 80, 0, pw, ph)
        ax_z = self._project_3d(0, 0, 80, pw, ph)
        canvas.draw_line(origin[0], origin[1], ax_x[0], ax_x[1], "red")
        canvas.draw_line(origin[0], origin[1], ax_y[0], ax_y[1], "green")
        canvas.draw_line(origin[0], origin[1], ax_z[0], ax_z[1], "blue")

        # Arbeitsraum-Kreis
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

        # Trail
        if self._trail and len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                p1 = self._project_3d(*self._trail[i-1], pw, ph)
                p2 = self._project_3d(*self._trail[i], pw, ph)

                dist = np.linalg.norm(self._trail[i] - self._trail[i-1])

                if dist < 2.0:
                    color = "cyan"
                elif dist < 5.0:
                    color = "green"
                elif dist < 10.0:
                    color = "yellow"
                else:
                    color = "bright_red"

                canvas.draw_line(p1[0], p1[1], p2[0], p2[1], color)

        # Target
        if self._target:
            t_pos = forward_kinematics(self._target["b"], self._target["s"],
                                       self._target["e"])
            tp = self._project_3d(*t_pos["gripper"], pw, ph)
            canvas.draw_circle(tp[0], tp[1], 4, "bright_red")
            canvas.draw_line(tp[0]-3, tp[1]-3, tp[0]+3, tp[1]+3, "bright_red")
            canvas.draw_line(tp[0]-3, tp[1]+3, tp[0]+3, tp[1]-3, "bright_red")

        # Arm-Segmente
        pts = [positions["base"], positions["shoulder"],
               positions["elbow"], positions["gripper"]]
        projected = [self._project_3d(p[0], p[1], p[2], pw, ph) for p in pts]

        base_bottom = self._project_3d(0, 0, 0, pw, ph)
        canvas.draw_thick_line(base_bottom[0], base_bottom[1],
                               projected[0][0], projected[0][1],
                               thickness=4, color="white")

        canvas.draw_thick_line(projected[1][0], projected[1][1],
                               projected[2][0], projected[2][1],
                               thickness=3, color="bright_blue")

        canvas.draw_thick_line(projected[2][0], projected[2][1],
                               projected[3][0], projected[3][1],
                               thickness=2, color="bright_green")

        joint_colors = ["white", "bright_red", "bright_blue", "bright_yellow"]
        joint_radii = [5, 4, 3, 3]
        for i, (px, py) in enumerate(projected):
            canvas.fill_circle(px, py, joint_radii[i], joint_colors[i])

        lines = canvas.render()

        gp = positions["gripper"]
        header = Text()
        header.append(f" Az:{self._cam_azimuth:.0f}\u00b0 El:{self._cam_elevation:.0f}\u00b0",
                      style="bright_black")
        header.append(f"  b={self.b:+.1f}\u00b0 s={self.s:+.1f}\u00b0 e={self.e:+.1f}\u00b0",
                      style="bold bright_white")
        header.append(f"  Grip:({gp[0]:.0f},{gp[1]:.0f},{gp[2]:.0f})",
                      style="cyan")

        if self.app._is_sim:
            header.append("  \u26a0 SIMULATED (kein realer Arm)", style="bold yellow")

        all_lines = [header] + lines
        self.update(Text("\n").join(all_lines))

    def on_mount(self):
        self._refresh_display()


class JointSparklineWidget(Static):
    """Zeigt Sparkline + Wert fuer ein einzelnes Gelenk."""

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
        color = "#00ff41"
        name = self.JOINT_NAMES.get(self.joint, self.joint)
        sparkline = self._make_sparkline()

        addr = f"0x{id(self) & 0xFFFF:04X}"

        raw_val = int((self._value + 180) / 360 * 4095)

        text = (
            f"[dim green]{addr}[/] "
            f"[bold #00ff41]{name}[/] "
            f"[bold white]{self._value:+7.2f}\u00b0[/] "
            f"[dim green]0x{raw_val:03X}[/] "
            f"[#00aa00]{sparkline}[/]"
        )
        self.update(text)

    def _make_sparkline(self) -> str:
        if not self._history:
            return "\u2581" * 20

        bars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        values = self._history[-20:]

        if len(values) < 2:
            return "\u2584" * len(values)

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val < 0.01:
            return "\u2584" * len(values)

        result = ""
        for v in values:
            idx = int((v - min_val) / range_val * (len(bars) - 1))
            idx = max(0, min(len(bars) - 1, idx))
            result += bars[idx]

        return result


class TimelineWidget(Static):
    """Zeigt eine Timeline fuer Recordings."""

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

        bar = list("\u2500" * bar_width)

        for wp in self._waypoints[::max(1, len(self._waypoints) // bar_width)]:
            idx = int((wp["t"] / self._duration) * bar_width)
            idx = max(0, min(bar_width - 1, idx))
            bar[idx] = "\u2503"

        bar[cursor_pos] = "\u25b6"
        bar_str = "".join(bar)

        text = (
            f"[bold]Timeline[/] [{self._position:.2f}s / {self._duration:.2f}s]\n"
            f"[bright_blue]\u2503[/]{bar_str}[bright_blue]\u2503[/]\n"
            f"[dim]0s{'':>{bar_width - 8}}{self._duration:.1f}s[/]"
        )
        self.update(text)


class RoarmFileViewer(Static):
    """Zeigt die .roarm-Datei live an mit Cursor auf der aktuellen Zeile."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lines: list[str] = []
        self._current_line_idx: int = -1
        self._context_lines: int = 10
        self._parsed_lines: list[dict] = []

    def set_finished(self):
        if self._parsed_lines:
            for i in range(len(self._parsed_lines) - 1, -1, -1):
                if self._parsed_lines[i]["type"] == "MOVE":
                    self._current_line_idx = i
                    break
            self._refresh_display()

    def reset(self):
        self._current_line_idx = -1
        self._refresh_display()

    def load_file(self, filepath: str):
        with open(filepath, 'r') as f:
            self._lines = f.readlines()
        self._current_line_idx = -1
        self._parsed_lines = []
        for i, line in enumerate(self._lines):
            stripped = line.strip()
            parsed = {"type": None, "t": None, "pos": None}
            if stripped.startswith("MOVE"):
                parsed["type"] = "MOVE"
                vals = {}
                for p in stripped.split()[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        try:
                            vals[k] = float(v)
                        except ValueError:
                            pass
                parsed["t"] = vals.get("t")
                parsed["pos"] = {
                    "b": vals.get("b", 0),
                    "s": vals.get("s", 0),
                    "e": vals.get("e", 0),
                    "h": vals.get("h", 180),
                }
            elif stripped.startswith("GRIPPER") or stripped.startswith("LED"):
                parsed["type"] = "EVENT"
                parsed["t"] = self._extract_time(stripped)
            self._parsed_lines.append(parsed)
        self._refresh_display()

    def set_current_time(self, elapsed: float):
        best_idx = -1
        for i, parsed in enumerate(self._parsed_lines):
            if parsed["t"] is not None and parsed["t"] <= elapsed:
                best_idx = i
            elif parsed["t"] is not None and parsed["t"] > elapsed:
                break

        if best_idx != self._current_line_idx:
            self._current_line_idx = best_idx
            self._refresh_display()

    def set_current_line_by_position(self, pos: dict):
        best_idx = -1
        best_dist = float('inf')

        for i, parsed in enumerate(self._parsed_lines):
            if parsed["type"] != "MOVE" or parsed["pos"] is None:
                continue
            dist = sum(
                (pos.get(j, 0) - parsed["pos"].get(j, 0)) ** 2
                for j in ["b", "s", "e", "h"]
            )
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx != self._current_line_idx and best_idx >= 0:
            self._current_line_idx = best_idx
            self._refresh_display()

    def _extract_time(self, line: str) -> Optional[float]:
        for part in line.split():
            if part.startswith("t="):
                try:
                    return float(part.split("=")[1])
                except ValueError:
                    return None
        return None

    def _refresh_display(self):
        if not self._lines:
            self.update("[dim]Keine Datei geladen[/]")
            return

        idx = self._current_line_idx
        total = len(self._lines)

        context = self._context_lines
        if idx >= 0:
            start = max(0, idx - context)
            end = min(total, idx + context + 1)
        else:
            start = 0
            end = min(total, context * 2)

        output_parts = []

        if idx >= 0:
            progress_pct = (idx / max(total - 1, 1)) * 100
            progress_bar_w = 30
            filled = int(progress_pct / 100 * progress_bar_w)
            bar = "\u2588" * filled + "\u2591" * (progress_bar_w - filled)
            output_parts.append(
                f"[bold cyan]\U0001f4c4 .roarm[/] "
                f"[dim]Zeile[/] [bold white]{idx + 1}[/][dim]/{total}[/] "
                f"[dim]{bar}[/] [bold]{progress_pct:.0f}%[/]\n"
            )
        else:
            output_parts.append(
                f"[bold cyan]\U0001f4c4 .roarm[/] [dim]Warte auf Start...[/]\n"
            )

        output_parts.append("\u2500" * 54 + "\n")

        for i in range(start, end):
            line_content = self._lines[i].rstrip()
            line_num = f"{i + 1:4d}"

            if i == idx:
                output_parts.append(
                    f"[bold white on dark_green] \u25b6 {line_num} \u2502 {line_content} [/]\n"
                )
            elif idx >= 0 and i == idx + 1:
                output_parts.append(
                    f"[yellow]   {line_num} \u2502 \u27a4 {line_content}[/]\n"
                )
            elif self._lines[i].strip().startswith("#"):
                output_parts.append(
                    f"[dim italic]   {line_num} \u2502 {line_content}[/]\n"
                )
            elif idx >= 0 and i < idx:
                output_parts.append(
                    f"[dim green]   {line_num} \u2502 \u2713 {line_content}[/]\n"
                )
            else:
                output_parts.append(
                    f"[dim]   {line_num} \u2502   {line_content}[/]\n"
                )

        output_parts.append("\u2500" * 54)

        self.update("".join(output_parts))

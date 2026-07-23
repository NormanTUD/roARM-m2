"""RoArm Dashboard - main application class."""

import os
import sys
import json
import time
import logging
from typing import Optional

import psutil
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import (
    Container, Horizontal, Vertical, ScrollableContainer,
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
from safety import SafeArm, SafetyLimits, SafetyWatchdog, CurrentMonitor, RateLimiter

from .kinematics import (
    STREAM_HZ, RECORDINGS_DIR, LOGS_DIR,
)
from .sim_arm import SimulatedArm
from .utils import TUILogHandler, JointHistory, ActivityIndicator
from .widgets import (
    Arm3DWidget, JointSparklineWidget, TimelineWidget, RoarmFileViewer,
)
from .css import CSS
from .recording import parse_roarm_file

from . import mode_teach
from . import mode_play
from . import mode_calibrate
from . import mode_servo


class RoArmDashboard(App):
    """RoArm-M2-S Unified TUI Dashboard v2."""

    TITLE = "RoArm-M2-S Dashboard"
    SUB_TITLE = ""
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
        Binding("r", "read_position", "Read Pos", show=True),
        Binding("escape", "emergency_stop", "E-STOP", show=True, priority=True),
        Binding("l", "led_toggle", "LED", show=False),
        Binding("j", "toggle_joint_control", "Joy Mode", show=True),
    ]

    connected = reactive(False)
    recording = reactive(False)
    playing = reactive(False)
    torque_on_state = reactive(True)

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

        self._led_on = False
        import argparse
        _parser = argparse.ArgumentParser(description="Robot Control Script")
        _parser.add_argument(
            '--enable-gravity-comp',
            action='store_true',
            help="Enable gravity compensation (default: False)"
        )
        _args = _parser.parse_args()

        self._gravity_comp_enabled = _args.enable_gravity_comp
        self._speed_factor = 1.0
        self._loop_enabled = False
        self._loop_pause_s = 0.0

        self._rate_limiter = None

        self._last_play_commanded: Optional[dict] = None
        self._safe_arm: Optional[object] = None
        self._watchdog: Optional[object] = None
        self._current_monitor: Optional[object] = None

        # Joint control mode (keyboard driving)
        self._joint_control_mode = False
        self._key_timestamps: dict = {}
        self._joint_ctrl_timer: Optional[Timer] = None
        self._joint_speed = 1.5
        self._joint_target = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        self._joint_virtual_pos = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        self._key_release_timeout = 0.12
        self._last_joint_move_time: float = 0.0

    def _reset_file_viewer(self):
        try:
            file_viewer = self.query_one("#roarm-file-viewer", RoarmFileViewer)
            file_viewer.reset()
        except NoMatches:
            pass

    def _setup_safety_layer(self, arm: RoArmConnection):
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
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog = None
        self._safe_arm = None
        self._current_monitor = None
        self._rate_limiter = None

    def action_emergency_stop(self):
        self.playing = False
        self.recording = False
        self._joint_control_mode = False
        self._key_timestamps.clear()
        if self._joint_ctrl_timer is not None:
            self._joint_ctrl_timer.stop()
            self._joint_ctrl_timer = None
        if self._teach_timer:
            self._teach_timer.stop()
            self._teach_timer = None
        if self._sim_arm:
            self._sim_arm.torque_off()
        self._stop_activity("\U0001f6a8 E-STOP")
        self._stop_recording_timer()
        self._log_teach("[bold red]\U0001f6a8 EMERGENCY STOP[/]")
        try:
            self.query_one("#btn-teach-record", Button).disabled = False
            self.query_one("#btn-teach-stop", Button).disabled = True
            self.query_one("#btn-play-start", Button).disabled = False
            self.query_one("#btn-play-stop", Button).disabled = True
        except NoMatches:
            pass

    @property
    def _active_arm(self):
        if self._arm and self.connected:
            return self._arm
        if self._simulation_mode and self._sim_arm:
            return self._sim_arm
        return None

    @property
    def _is_sim(self) -> bool:
        return self._simulation_mode and not self.connected

    # ============================================================
    # COMPOSE (Layout)
    # ============================================================

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="main-container"):
            with TabbedContent():
                # --- TAB 1: TEACH ---
                with TabPane("\U0001f3ac Teach [1]", id="teach"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical(id="teach-left"):
                                yield Arm3DWidget(
                                    id="teach-arm-view",
                                    classes="arm-view"
                                )
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "\u23fa Record [Space]", id="btn-teach-record",
                                        classes="btn-record", variant="error"
                                    )
                                    yield Button(
                                        "\u23f9 Stop [Space]", id="btn-teach-stop",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "\U0001f3e0 Home [h]", id="btn-teach-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "\u270a/\u270b Gripper [g]", id="btn-gripper",
                                        variant="default"
                                    )
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
                with TabPane("\u25b6\ufe0f Play [2]", id="play"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="play-arm-view",
                                    classes="arm-view"
                                )
                                yield TimelineWidget(id="play-timeline")
                                with Horizontal(classes="control-buttons"):
                                    yield Button("\u25b6 Play [Space]", id="btn-play-start",
                                                 classes="btn-play", variant="success")
                                    yield Button("\u23f9 Stop [Space]", id="btn-play-stop",
                                                 classes="btn-stop", variant="warning", disabled=True)
                                    yield Button("\U0001f501 Loop", id="btn-play-loop", variant="default")
                                    yield Label("Speed:", classes="joint-label")
                                    yield Input(value="1.0", id="play-speed-input", type="number")
                                    yield Label("Loop Pause (s):", classes="joint-label")
                                    yield Input(value="0", id="play-loop-pause-input", type="number")

                            with Vertical():
                                yield Label("\U0001f4c1 Recordings:", classes="joint-label")
                                yield DataTable(id="recordings-table")

                        with Horizontal():
                            yield RichLog(id="play-log", highlight=True, markup=True)
                            yield RoarmFileViewer(id="roarm-file-viewer")

                # --- TAB 3: CALIBRATE ---
                with TabPane("\U0001f3af Calibrate [3]", id="calibrate"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="calibrate-arm-view",
                                    classes="arm-view"
                                )
                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "\u25b6 Start Calibration", id="btn-cal-start",
                                        classes="btn-play", variant="success"
                                    )
                                    yield Button(
                                        "\u23f9 Abort", id="btn-cal-abort",
                                        classes="btn-stop", variant="warning",
                                        disabled=True
                                    )
                                    yield Button(
                                        "\U0001f4c2 Load Cal", id="btn-cal-load",
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
                with TabPane("\U0001f527 Servo [4]", id="servo"):
                    with Vertical(classes="tab-content"):
                        with Horizontal():
                            with Vertical():
                                yield Arm3DWidget(
                                    id="servo-arm-view",
                                    classes="arm-view"
                                )

                            with Vertical():
                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_blue]Servo 1: BASE[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [\u00b0]:")
                                        yield Input(
                                            value="0.0", id="servo-b-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-b-go",
                                                     variant="primary")
                                    yield Static(id="servo-b-readout")

                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_magenta]Servo 2: SHOULDER[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [\u00b0]:")
                                        yield Input(
                                            value="0.0", id="servo-s-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-s-go",
                                                     variant="primary")
                                    yield Static(id="servo-s-readout")

                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_yellow]Servo 3: ELBOW[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [\u00b0]:")
                                        yield Input(
                                            value="90.0", id="servo-e-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-e-go",
                                                     variant="primary")
                                    yield Static(id="servo-e-readout")

                                with Vertical(classes="servo-control-panel"):
                                    yield Label("[bright_cyan]Servo 4: HAND[/]")
                                    with Horizontal(classes="servo-slider-row"):
                                        yield Label("Angle [\u00b0]:")
                                        yield Input(
                                            value="180.0", id="servo-h-input",
                                            type="number",
                                        )
                                        yield Button("Go", id="btn-servo-h-go",
                                                     variant="primary")
                                    yield Static(id="servo-h-readout")

                                with Horizontal(classes="control-buttons"):
                                    yield Button(
                                        "\U0001f4d1 Read All [r]", id="btn-servo-read",
                                        variant="default"
                                    )
                                    yield Button(
                                        "\U0001f3e0 Home [h]", id="btn-servo-home",
                                        variant="default"
                                    )
                                    yield Button(
                                        "\U0001f50c Torque Off [t]",
                                        id="btn-servo-torque-off",
                                        variant="warning"
                                    )

                        yield RichLog(id="servo-log", highlight=True, markup=True)

        # Status-Bar
        with Horizontal(classes="status-bar"):
            yield Label("\U0001f50c Disconnected", id="status-connection")
            yield Label("\u2502", id="status-sep1")
            yield Label("\U0001f512 Torque ON", id="status-torque")
            yield Label("\u2502", id="status-sep2")
            yield Label("\U0001f6e1\ufe0f Safety OK", id="status-safety")
            yield Label("\u2502", id="status-sep3")
            yield Label("\u23f1\ufe0f --", id="status-mode")
            yield Label("\u2502", id="status-sep4")
            yield Label("", id="status-activity")

        yield Footer()

    # ============================================================
    # ON MOUNT
    # ============================================================

    def on_mount(self) -> None:
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

        mode_play._show_calibration_info(self)

        if not self.connected:
            self._enable_simulation_mode()

        self.set_interval(0.5, self._periodic_position_poll)
        self._header_timer = self.set_interval(1.0, self._update_header)

    def _measure_serial_latency(self) -> Optional[float]:
        if not self._arm or self._is_sim:
            return None
        try:
            start = time.perf_counter()
            pos = self._arm.read_position_deg()
            if pos is None:
                return None
            latency_ms = (time.perf_counter() - start) * 1000
            return latency_ms
        except Exception:
            return None

    def _update_header(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        latency_str = ""
        if hasattr(self, '_last_latency_check'):
            if time.time() - self._last_latency_check > 5.0:
                lat = self._measure_serial_latency()
                if lat is not None:
                    self._cached_latency = lat
                self._last_latency_check = time.time()
        else:
            self._last_latency_check = time.time()
            self._cached_latency = None

        if hasattr(self, '_cached_latency') and self._cached_latency:
            lat = self._cached_latency
            if lat < 10:
                latency_str = f"LAT:{lat:.0f}ms\u2713"
            elif lat < 30:
                latency_str = f"LAT:{lat:.0f}ms"
            else:
                latency_str = f"LAT:{lat:.0f}ms\u26a0"

        uptime = time.time() - getattr(self, '_start_time', time.time())

        new_sub_title = f"PID:{os.getpid()} | CPU:{cpu:.0f}% MEM:{mem:.0f}% | UP:{uptime:.0f}s | HZ:{STREAM_HZ}"

        if latency_str:
            new_sub_title += f"| {latency_str}"

        self.sub_title = (new_sub_title)

    def _enable_simulation_mode(self):
        self._sim_arm = SimulatedArm()
        self._simulation_mode = True

        self._sim_timer = self.set_interval(0.01, self._sim_step)

        self._log_teach(
            "[bold cyan]\U0001f916 SIMULATION MODE[/] \u2014 Kein Roboter verbunden"
        )
        self._log_teach(
            "[dim]  Alle Bewegungen werden virtuell simuliert.[/]"
        )
        self._log_teach(
            "[dim]  Druecke [c] um einen echten Roboter zu verbinden.[/]"
        )

        try:
            label = self.query_one("#status-connection", Label)
            label.update("\U0001f916 SIMULATION")
        except NoMatches:
            pass

        try:
            mode_label = self.query_one("#status-mode", Label)
            mode_label.update("\U0001f916 Sim Ready")
        except NoMatches:
            pass

        pos = self._sim_arm.read_position_deg()
        self._current_pos = pos
        self._update_joint_displays(pos)
        self._update_arm_views(pos)

        try:
            safety_label = self.query_one("#status-safety", Label)
            safety_label.update("\u26a0 Kein realer Arm verbunden")
        except NoMatches:
            pass

    def _disable_simulation_mode(self):
        self._simulation_mode = False
        if self._sim_timer:
            self._sim_timer.stop()
            self._sim_timer = None
        self._sim_arm = None

    def _sim_step(self):
        if not self._simulation_mode or not self._sim_arm:
            return
        self._sim_arm.step_simulation(0.01)

        pos = self._sim_arm.read_position_deg()
        self._current_pos = pos
        self._update_joint_displays(pos)
        self._update_arm_views(pos)
        self._update_servo_readouts(pos)

    def _try_auto_connect(self):
        port = find_arm_port()
        if port:
            self._log_teach(f"[dim]\U0001f50d Port gefunden: {port} \u2014 verbinde...[/]")
            self._do_connect(port)
        else:
            self._log_teach(
                "[yellow]\u26a0 Kein Arm-Port gefunden. "
                "Starte im Simulationsmodus. [c] zum Verbinden.[/]"
            )

    def _do_connect(self, port: str):
        try:
            self._arm = RoArmConnection(port)
            self.connected = True
            if self._simulation_mode:
                self._disable_simulation_mode()
            self._setup_safety_layer(self._arm)
            self._log_teach(f"[green]\u2705 Verbunden mit {port} (Safety aktiv)[/]")
            self._update_status_connection(port)
            pos = self._arm.read_position_deg()
            if pos:
                self._current_pos = pos
                self._update_joint_displays(pos)
                self._update_arm_views(pos)
        except Exception as e:
            self._log_teach(f"[red]\u274c Fehler: {e}[/]")

    # ============================================================
    # PERIODIC TASKS
    # ============================================================

    def _periodic_position_poll(self):
        if self.recording or self.playing:
            return

        if self._is_sim and self._sim_arm:
            return

        arm = self._active_arm
        if arm is None:
            return

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
            label.update(f"\U0001f50c {port}@{BAUDRATE}" if port else "\U0001f50c Disconnected")
        except NoMatches:
            pass

    def _update_status_torque(self, on: bool):
        try:
            label = self.query_one("#status-torque", Label)
            label.update("\U0001f512 Torque ON" if on else "\U0001f513 Torque OFF")
        except NoMatches:
            pass

    def _update_joint_displays(self, pos: dict):
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
        for view_id in ["teach-arm-view", "play-arm-view",
                        "calibrate-arm-view", "servo-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", Arm3DWidget)
                widget.update_pose(pos["b"], pos["s"], pos["e"])
            except NoMatches:
                pass

    def _update_servo_readouts(self, pos: dict):
        for joint, name in [("b", "Base"), ("s", "Shoulder"),
                            ("e", "Elbow"), ("h", "Hand")]:
            try:
                widget = self.query_one(f"#servo-{joint}-readout", Static)
                widget.update(
                    f"  [dim]Current:[/] [bold]{pos[joint]:+7.2f}\u00b0[/]"
                )
            except NoMatches:
                pass

    # ============================================================
    # ACTIVITY INDICATOR METHODS
    # ============================================================

    def _start_activity(self, message: str, icon: str = "\u23f3"):
        self._activity.start(message, icon)
        if self._activity_timer is not None:
            self._activity_timer.stop()
        self._activity_timer = self.set_interval(0.1, self._tick_activity)

    def _stop_activity(self, final_message: str = ""):
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
        if not self._activity.is_active:
            return
        try:
            label = self.query_one("#status-activity", Label)
            label.update(self._activity.next_frame())
        except NoMatches:
            pass

    def _start_recording_timer(self):
        self._update_recording_timer_display()
        if self._recording_elapsed_timer is not None:
            self._recording_elapsed_timer.stop()
        self._recording_elapsed_timer = self.set_interval(
            0.5, self._update_recording_timer_display
        )

    def _stop_recording_timer(self):
        if self._recording_elapsed_timer is not None:
            self._recording_elapsed_timer.stop()
            self._recording_elapsed_timer = None
        try:
            label = self.query_one("#teach-recording-timer", Label)
            label.update("")
        except NoMatches:
            pass

    def _update_recording_timer_display(self):
        if not self.recording:
            return
        from .kinematics import RECORD_HZ
        elapsed = time.time() - self._teach_start_time
        move_wps = [wp for wp in self._teach_waypoints if "cmd" not in wp]
        wp_count = len(move_wps)

        dots = "\u25cf" if int(elapsed * 2) % 2 == 0 else "\u25cb"

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
                self._log_teach("[red]\u274c Kein Port gefunden![/]")

    def _disconnect(self):
        self._teardown_safety_layer()
        if self._arm:
            self._arm.close()
            self._arm = None
        self.connected = False
        self._log_teach("[yellow]\U0001f50c Getrennt[/]")
        self._update_status_connection(None)
        self._enable_simulation_mode()

    def action_torque_release(self) -> None:
        arm = self._active_arm
        if arm is None:
            self._log_teach("[red]Nicht verbunden und keine Simulation![/]")
            return

        arm.torque_off()
        self.torque_on_state = False
        self._update_status_torque(False)

        if self._is_sim:
            self._log_teach("[yellow]\U0001f513 Torque AUS (Sim) \u2014 Arm ist frei[/]")
        else:
            self._log_teach("[yellow]\U0001f513 Torque AUS \u2014 Arm ist frei bewegbar[/]")
        self._log_servo("[yellow]\U0001f513 Torque AUS[/]")

    def action_torque_lock(self) -> None:
        arm = self._active_arm
        if arm is None:
            self._log_teach("[red]Nicht verbunden und keine Simulation![/]")
            return

        arm.torque_on()
        self.torque_on_state = True
        self._update_status_torque(True)

        if self._is_sim:
            self._log_teach("[green]\U0001f512 Torque AN (Sim) \u2014 Arm ist fixiert[/]")
        else:
            self._log_teach("[green]\U0001f512 Torque AN \u2014 Arm ist fixiert[/]")
        self._log_servo("[green]\U0001f512 Torque AN[/]")

    def action_toggle_action(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            active = tabs.active
        except NoMatches:
            return

        if active == "teach":
            if self.recording:
                mode_teach.stop_recording(self)
            else:
                mode_teach.start_recording(self)
        elif active == "play":
            if self.playing:
                mode_play.stop_playback(self)
            else:
                mode_play.start_playback(self)
        elif active == "calibrate":
            mode_calibrate.start_calibration(self)

    def action_led_toggle(self) -> None:
        arm = self._active_arm
        if arm is None:
            return
        self._led_on = not getattr(self, '_led_on', False)
        brightness = 255 if self._led_on else 0
        if hasattr(arm, 'send_cmd'):
            arm.send_cmd({"T": 114, "led": brightness})
        self._log_teach(f"[bold]\U0001f4a1 LED {'AN' if self._led_on else 'AUS'}[/]")
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
            self._log_teach("[bold]\u270a Gripper ZU[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append(
                    {"t": round(elapsed, 4), "cmd": "GRIPPER_CLOSE"}
                )
        else:
            arm.gripper_open()
            self._gripper_open = True
            self._log_teach("[bold]\u270b Gripper AUF[/]")
            if self.recording:
                elapsed = time.time() - self._teach_start_time
                self._teach_waypoints.append(
                    {"t": round(elapsed, 4), "cmd": "GRIPPER_OPEN"}
                )

    def action_go_home(self) -> None:
        self.run_worker(lambda: mode_teach.go_home(self), thread=True)

    def action_read_position(self) -> None:
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
                f"[green]\U0001f4cd Position:{sim_tag}[/] b={pos['b']:+.2f}\u00b0 "
                f"s={pos['s']:+.2f}\u00b0 e={pos['e']:+.2f}\u00b0 h={pos['h']:+.2f}\u00b0"
            )
        else:
            self._log_servo("[red]\u274c Konnte Position nicht lesen[/]")

    def action_rotate_left(self) -> None:
        self._rotate_all_views(d_azimuth=-15)

    def action_rotate_right(self) -> None:
        self._rotate_all_views(d_azimuth=15)

    def action_rotate_up(self) -> None:
        self._rotate_all_views(d_elevation=10)

    def action_rotate_down(self) -> None:
        self._rotate_all_views(d_elevation=-10)

    def _rotate_all_views(self, d_azimuth: float = 0, d_elevation: float = 0):
        for view_id in ["teach-arm-view", "play-arm-view",
                        "calibrate-arm-view", "servo-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", Arm3DWidget)
                widget.rotate(d_azimuth, d_elevation)
            except NoMatches:
                pass

    # ============================================================
    # JOINT CONTROL MODE (Keyboard Driving)
    # ============================================================

    _JOINT_KEYS = {"w", "a", "s", "d", "x", "c", "r", "f"}

    def on_key(self, event) -> None:
        key = event.key
        focused = self.focused
        is_input_focused = isinstance(focused, Input)

        if self._joint_control_mode and key in self._JOINT_KEYS and not is_input_focused:
            event.stop()
            self._key_timestamps[key] = time.time()
            if self._joint_ctrl_timer is None:
                self._joint_ctrl_timer = self.set_interval(
                    0.02, self._joint_ctrl_step
                )
        elif not self._joint_control_mode and key in ("w", "a", "s", "d") and not is_input_focused:
            if key == "w":
                self._rotate_all_views(d_elevation=10)
            elif key == "s":
                self._rotate_all_views(d_elevation=-10)
            elif key == "a":
                self._rotate_all_views(d_azimuth=-15)
            elif key == "d":
                self._rotate_all_views(d_azimuth=15)

    def action_toggle_joint_control(self) -> None:
        self._joint_control_mode = not self._joint_control_mode
        self._key_timestamps.clear()
        if self._joint_ctrl_timer is not None:
            self._joint_ctrl_timer.stop()
            self._joint_ctrl_timer = None

        if self._joint_control_mode:
            self._joint_target = self._current_pos.copy()
            self._joint_virtual_pos = self._current_pos.copy()
            self.add_class("joint-control-active")
            self._log_teach(
                "[bold cyan]\U0001f3ae JOINT CONTROL MODE[/] "
                "[dim]\u2014 WASD/Base+Shoulder XC/Elbow RF/Hand[/]"
            )
            self._log_teach(
                "[dim]  [j] exits | Hold keys = move | [g] gripper still works[/]"
            )
        else:
            self.remove_class("joint-control-active")
            self._log_teach("[dim]\U0001f3ae Joint control mode OFF[/]")

        self._set_arm_views_joint_control(self._joint_control_mode)
        self._update_joint_control_status()

    def _joint_ctrl_step(self) -> None:
        if not self._joint_control_mode:
            return
        arm = self._active_arm
        if arm is None:
            return

        now = time.time()
        active = {
            k for k, t in self._key_timestamps.items()
            if now - t < self._key_release_timeout
        }

        if not active:
            if self._joint_virtual_vel != {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}:
                self._joint_virtual_vel = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
                if self._is_sim and self._sim_arm:
                    self._sim_arm.move_to(
                        self._joint_target["b"], self._joint_target["s"],
                        self._joint_target["e"], self._joint_target["h"],
                        spd=5, acc=10,
                    )
                elif self._arm:
                    self._arm.move_to_fast(
                        self._joint_target["b"], self._joint_target["s"],
                        self._joint_target["e"], self._joint_target["h"],
                        spd=5, acc=10,
                    )
            return

        vel = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}

        if "w" in active:
            vel["s"] -= self._joint_speed
        if "s" in active:
            vel["s"] += self._joint_speed
        if "a" in active:
            vel["b"] -= self._joint_speed
        if "d" in active:
            vel["b"] += self._joint_speed
        if "x" in active:
            vel["e"] -= self._joint_speed
        if "c" in active:
            vel["e"] += self._joint_speed
        if "r" in active:
            vel["h"] -= self._joint_speed
        if "f" in active:
            vel["h"] += self._joint_speed

        self._joint_virtual_vel = vel

        dt = 0.02
        for j in ["b", "s", "e", "h"]:
            self._joint_virtual_pos[j] += vel[j] * dt

        self._joint_virtual_pos["e"] = max(0.0, min(180.0, self._joint_virtual_pos["e"]))
        self._joint_virtual_pos["h"] = max(0.0, min(360.0, self._joint_virtual_pos["h"]))
        for j in ["b", "s"]:
            self._joint_virtual_pos[j] = max(-180.0, min(180.0, self._joint_virtual_pos[j]))

        _LOOKAHEAD_S = 0.3
        for j in ["b", "s", "e", "h"]:
            self._joint_target[j] = self._joint_virtual_pos[j] + vel[j] * _LOOKAHEAD_S

        self._joint_target["e"] = max(0.0, min(180.0, self._joint_target["e"]))
        self._joint_target["h"] = max(0.0, min(360.0, self._joint_target["h"]))
        for j in ["b", "s"]:
            self._joint_target[j] = max(-180.0, min(180.0, self._joint_target[j]))

        if self._is_sim and self._sim_arm:
            self._sim_arm.move_to(
                self._joint_target["b"], self._joint_target["s"],
                self._joint_target["e"], self._joint_target["h"],
                spd=0, acc=0,
            )
        elif self._arm:
            self._arm.move_to_fast(
                self._joint_target["b"], self._joint_target["s"],
                self._joint_target["e"], self._joint_target["h"],
                spd=0, acc=0,
            )

        self._last_joint_move_time = now
        pos = self._joint_virtual_pos.copy()
        self._current_pos = pos
        self._update_joint_displays(pos)
        self._update_arm_views(pos)
        self._update_servo_readouts(pos)

    def _update_joint_control_status(self) -> None:
        try:
            mode_label = self.query_one("#status-mode", Label)
            if self._joint_control_mode:
                mode_label.update("\U0001f3ae JOY CTRL")
            elif self.recording:
                mode_label.update("\U0001f534 REC")
            elif self.playing:
                mode_label.update("\u25b6\ufe0f PLAY")
            elif self.connected:
                mode_label.update("\u23f1\ufe0f Ready")
            elif self._simulation_mode:
                mode_label.update("\U0001f916 Sim Ready")
            else:
                mode_label.update("\u23f1\ufe0f --")
        except NoMatches:
            pass

    def _set_arm_views_joint_control(self, active: bool) -> None:
        for view_id in ["teach-arm-view", "play-arm-view",
                        "calibrate-arm-view", "servo-arm-view"]:
            try:
                widget = self.query_one(f"#{view_id}", Arm3DWidget)
                widget.set_joint_control_mode(active)
            except NoMatches:
                pass

    # ============================================================
    # TEACH MODE (wired to mode_teach)
    # ============================================================

    @on(Button.Pressed, "#btn-teach-record")
    def on_teach_record(self) -> None:
        mode_teach.start_recording(self)

    @on(Button.Pressed, "#btn-teach-stop")
    def on_teach_stop(self) -> None:
        mode_teach.stop_recording(self)

    @on(Button.Pressed, "#btn-teach-home")
    def on_teach_home(self) -> None:
        self.action_go_home()

    @on(Button.Pressed, "#btn-gripper")
    def on_gripper_press(self) -> None:
        self.action_gripper_toggle()

    # ============================================================
    # PLAY MODE (wired to mode_play)
    # ============================================================

    @on(Button.Pressed, "#btn-play-start")
    def on_play_start(self) -> None:
        mode_play.start_playback(self)

    @on(Button.Pressed, "#btn-play-stop")
    def on_play_stop(self) -> None:
        mode_play.stop_playback(self)

    @on(Button.Pressed, "#btn-play-loop")
    def on_play_loop_toggle(self) -> None:
        try:
            btn = self.query_one("#btn-play-loop", Button)
            if btn.variant == "success":
                btn.variant = "default"
                btn.label = "\U0001f501 Loop"
                self._log_play("[dim]Loop deaktiviert[/]")
            else:
                btn.variant = "success"
                btn.label = "\U0001f501 Loop \u2713"
                self._log_play("[green]Loop aktiviert[/]")
        except NoMatches:
            pass

    # ============================================================
    # CALIBRATE MODE (wired to mode_calibrate)
    # ============================================================

    @on(Button.Pressed, "#btn-cal-start")
    def on_cal_start(self) -> None:
        mode_calibrate.start_calibration(self)

    @on(Button.Pressed, "#btn-cal-abort")
    def on_cal_abort(self) -> None:
        mode_calibrate.abort_calibration(self)

    @on(Button.Pressed, "#btn-cal-load")
    def on_cal_load(self) -> None:
        mode_calibrate.load_calibration(self)

    def _update_cal_status(self, text: str):
        mode_calibrate._update_cal_status(self, text)

    # ============================================================
    # SERVO MODE (wired to mode_servo)
    # ============================================================

    @on(Button.Pressed, "#btn-servo-b-go")
    def on_servo_b_go(self) -> None:
        mode_servo.servo_go(self, "b")

    @on(Button.Pressed, "#btn-servo-s-go")
    def on_servo_s_go(self) -> None:
        mode_servo.servo_go(self, "s")

    @on(Button.Pressed, "#btn-servo-e-go")
    def on_servo_e_go(self) -> None:
        mode_servo.servo_go(self, "e")

    @on(Button.Pressed, "#btn-servo-h-go")
    def on_servo_h_go(self) -> None:
        mode_servo.servo_go(self, "h")

    @on(Button.Pressed, "#btn-servo-read")
    def on_servo_read(self) -> None:
        self.action_read_position()

    @on(Button.Pressed, "#btn-servo-home")
    def on_servo_home(self) -> None:
        self.action_go_home()

    @on(Button.Pressed, "#btn-servo-torque-off")
    def on_servo_torque_off(self) -> None:
        self.action_torque_release()

    # ============================================================
    # WATCH: REACTIVE CHANGES
    # ============================================================

    def watch_connected(self, connected: bool) -> None:
        self._update_joint_control_status()

    def watch_recording(self, recording: bool) -> None:
        self._update_joint_control_status()

    def watch_playing(self, playing: bool) -> None:
        self._update_joint_control_status()

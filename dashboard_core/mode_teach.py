"""Teach mode methods for RoArmDashboard."""

import time
import math

from textual import on
from textual.widgets import Button, Label
from textual.css.query import NoMatches

from robot import START_POSITION_DEG

from .kinematics import (
    RECORD_HZ, MOVE_THRESHOLD_DEG, GRAVITY_COMP_SETTLE_MS,
)
from .widgets import Arm3DWidget


def start_recording(d):
    arm = d._active_arm
    if arm is None:
        d._log_teach("[red]Nicht verbunden und keine Simulation![/]")
        return

    d.recording = True
    d._teach_waypoints = []
    d._teach_start_time = time.time()
    d._gripper_open = True

    arm.torque_off()
    d.torque_on_state = False
    d._update_status_torque(False)

    try:
        d.query_one("#btn-teach-record", Button).disabled = True
        d.query_one("#btn-teach-stop", Button).disabled = False
    except NoMatches:
        pass

    try:
        d.query_one("#teach-arm-view", Arm3DWidget).clear_trail()
    except NoMatches:
        pass

    if d._is_sim:
        d._log_teach("[bold red]\u23fa AUFNAHME LAEUFT (Simulation)[/]")
        d._log_teach(
            "[dim]Simulierte Bewegung: Arm bewegt sich automatisch "
            "auf einer Demo-Trajektorie.[/]"
        )
        d._log_teach("[dim][Space]=Stop [g]=Gripper[/]")
        d._sim_recording_start_time = time.time()
    else:
        d._log_teach("[bold red]\u23fa AUFNAHME LAEUFT[/]")
        d._log_teach("[dim]Bewege den Arm! <Space>=Stop <g>=Gripper <t>=Torque[/]")

    d._start_activity("Recording", "\U0001f534")
    d._start_recording_timer()

    d._teach_timer = d.set_interval(
        1.0 / RECORD_HZ, lambda: teach_poll_position(d)
    )

    d.add_class("recording-active")


def teach_poll_position(d):
    if not d.recording:
        return
    arm = d._active_arm
    if arm is None:
        return
    if d._is_sim:
        pos = _generate_sim_teach_position(d)
    else:
        pos = _teach_read_position(d, arm)
        if pos is None:
            return
    d._current_pos = pos
    elapsed = time.time() - d._teach_start_time
    if _should_record_waypoint(d, pos):
        d._teach_waypoints.append({
            "t": round(elapsed, 4),
            "b": pos["b"], "s": pos["s"],
            "e": pos["e"], "h": pos["h"],
        })
    d._update_joint_displays(pos)
    d._update_arm_views(pos)
    _teach_log_periodic(d, pos, elapsed)


def _generate_sim_teach_position(d) -> dict:
    elapsed = time.time() - d._teach_start_time
    pos = {
        "b": 30.0 * math.sin(elapsed * 0.8),
        "s": 20.0 * math.sin(elapsed * 0.5) + 10.0,
        "e": 90.0 + 25.0 * math.sin(elapsed * 0.6),
        "h": 180.0 + 15.0 * math.sin(elapsed * 0.3),
    }
    with d._sim_arm._lock:
        d._sim_arm._position = pos.copy()
    return pos


def _teach_read_position(d, arm) -> dict:
    if d._gravity_comp_enabled and not d._is_sim:
        elapsed = time.time() - d._teach_start_time
        if int(elapsed * 0.5) != getattr(d, '_last_grav_pulse', -1):
            d._last_grav_pulse = int(elapsed * 0.5)
            return _read_with_gravity_comp(arm)
    if hasattr(arm, 'read_position_deg_single'):
        return arm.read_position_deg_single()
    return arm.read_position_deg()


def _read_with_gravity_comp(arm) -> dict:
    arm.torque_on_fast()
    time.sleep(GRAVITY_COMP_SETTLE_MS / 1000.0)
    pos = arm.read_position_deg()
    arm.torque_off_fast()
    return pos


def _should_record_waypoint(d, pos: dict) -> bool:
    if not d._teach_waypoints:
        return True
    last_move = None
    for wp in reversed(d._teach_waypoints):
        if "cmd" not in wp:
            last_move = wp
            break
    if last_move is None:
        return True
    max_delta = max(abs(pos[j] - last_move[j]) for j in ["b", "s", "e", "h"])
    return max_delta >= MOVE_THRESHOLD_DEG


def _teach_log_periodic(d, pos: dict, elapsed: float):
    move_wps = [wp for wp in d._teach_waypoints if "cmd" not in wp]
    if len(move_wps) % 50 == 0 and len(move_wps) > 0:
        sim_tag = " (sim)" if d._is_sim else ""
        d._log_teach(
            f"[dim]  \u25c6 WP#{len(move_wps)}{sim_tag} [{elapsed:.1f}s] "
            f"b={pos['b']:+.1f}\u00b0 s={pos['s']:+.1f}\u00b0 "
            f"e={pos['e']:+.1f}\u00b0 h={pos['h']:+.1f}\u00b0[/]"
        )


def stop_recording(d):
    if not d.recording:
        return

    d.recording = False

    if d._teach_timer:
        d._teach_timer.stop()
        d._teach_timer = None

    arm = d._active_arm
    if arm:
        arm.torque_on()
        d.torque_on_state = True
        d._update_status_torque(True)

    try:
        d.query_one("#btn-teach-record", Button).disabled = False
        d.query_one("#btn-teach-stop", Button).disabled = True
    except NoMatches:
        pass

    d._stop_activity("\u2705 Recording saved")
    d._stop_recording_timer()

    move_wps = [wp for wp in d._teach_waypoints if "cmd" not in wp]
    if not move_wps:
        d._log_teach("[yellow]Keine Wegpunkte aufgezeichnet![/]")
        d._stop_activity()
        return

    duration = move_wps[-1]["t"]
    sim_tag = " (Simulation)" if d._is_sim else ""
    d._log_teach(
        f"[green]\u23f9 Aufnahme gestoppt{sim_tag}: "
        f"{len(move_wps)} WPs, {duration:.1f}s[/]"
    )

    filepath = _save_recording(d)
    if filepath:
        d._log_teach(f"[green]\U0001f4be Gespeichert: {filepath}[/]")
        d._refresh_recordings_table()

    d.set_timer(3.0, lambda: d._stop_activity())

    d.remove_class("recording-active")


def _format_waypoint_line(wp: dict) -> str:
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


def _build_recording_header(d, move_wps: list) -> list:
    from datetime import datetime
    from robot import START_POSITION_DEG as _SPD
    return [
        f"# RoArm-M2-S Recording (Dashboard v3)",
        f"# Datum: {datetime.now().isoformat()}",
        f"# Wegpunkte: {len(move_wps)}",
        f"# Dauer: {move_wps[-1]['t']:.2f}s",
        f"#",
        f"#CONFIG hz={RECORD_HZ}",
        f"#CONFIG threshold={MOVE_THRESHOLD_DEG}",
        f"#CONFIG gravity_comp={'1' if d._gravity_comp_enabled else '0'}",
        f"#START_POS b={_SPD['b']:.2f} "
        f"s={_SPD['s']:.2f} "
        f"e={_SPD['e']:.2f} "
        f"h={_SPD['h']:.2f}",
        "",
    ]


def _save_recording(d) -> str:
    from pathlib import Path
    from datetime import datetime
    from .kinematics import RECORDINGS_DIR

    move_wps = [wp for wp in d._teach_waypoints if "cmd" not in wp]
    if not move_wps:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RECORDINGS_DIR / f"recording_{ts}.roarm"
    lines = _build_recording_header(d, move_wps)
    for wp in d._teach_waypoints:
        line = _format_waypoint_line(wp)
        if line:
            lines.append(line)
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return str(filename)


def go_home(d):
    from textual import work

    arm = d._active_arm
    if arm is None:
        return
    d.call_from_thread(d._start_activity, "Homing", "\U0001f3e0")
    d.call_from_thread(d._log_teach, "[dim]\U0001f3e0 Fahre zur Home-Position...[/]")
    arm.torque_on()
    d.torque_on_state = True
    d.call_from_thread(d._update_status_torque, True)
    time.sleep(0.2)
    arm.move_to(
        START_POSITION_DEG["b"], START_POSITION_DEG["s"],
        START_POSITION_DEG["e"], START_POSITION_DEG["h"],
        spd=25, acc=12
    )
    if d._is_sim:
        for _ in range(100):
            time.sleep(0.02)
            d._sim_arm.step_simulation(0.02)
            pos = d._sim_arm.read_position_deg()
            d.call_from_thread(d._update_joint_displays, pos)
            d.call_from_thread(d._update_arm_views, pos)
            if not d._sim_arm.is_moving:
                break
    else:
        time.sleep(2.0)
    pos = arm.read_position_deg()
    if pos:
        d._current_pos = pos
        d.call_from_thread(d._update_joint_displays, pos)
        d.call_from_thread(d._update_arm_views, pos)
    d.call_from_thread(d._stop_activity, "\u2705 Home reached")
    d.call_from_thread(d._log_teach, "[green]\u2705 Home-Position erreicht[/]")
    d.call_from_thread(d.set_timer, 3.0, lambda: d._stop_activity())

"""Waypoint capture mode.

A second, complementary way of recording movements. In contrast to the
continuous time-sampled recording in mode_teach, here only the endpoints
are saved:

  1. The user starts a capture session (torque is released).
  2. The user moves the arm by hand to a desired pose.
  3. The user presses a key (Space) to capture that pose. Torque is
     pulsed on briefly so the servos can report their position
     accurately, then released again so the arm is free to move.
  4. Repeat.
  5. The user stops the session. The captured waypoints are saved as a
     .roarm file with mode=waypoint. On playback, the internal motion
     between waypoints is generated automatically with a synchronized
     trapezoidal velocity profile (see trapezoid_trajectory.py).
"""

import time
from datetime import datetime
from pathlib import Path

from textual.widgets import Button, Static
from textual.css.query import NoMatches

from .kinematics import (
    RECORDINGS_DIR,
    WAYPOINT_TORQUE_PULSE_S,
    WAYPOINT_CAPTURE_SETTLE_S,
    WAYPOINT_V_MAX_DEG_S,
    WAYPOINT_A_MAX_DEG_S2,
    WAYPOINT_DEFAULT_SPEED,
)


# ---------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------

def start_session(d):
    """Begin a waypoint capture session: torque off, ready to capture."""
    if d.waypoint_capturing:
        return
    arm = d._active_arm
    if arm is None:
        d._log_waypoint("[red]Nicht verbunden und keine Simulation![/]")
        return

    d.waypoint_capturing = True
    d._waypoints = []
    d._waypoint_start_pos = None
    d._gripper_open = True

    try:
        arm.torque_off()
    except Exception:
        pass
    d.torque_on_state = False
    d._update_status_torque(False)

    try:
        d.query_one("#btn-waypoint-start", Button).disabled = True
        d.query_one("#btn-waypoint-stop", Button).disabled = False
        d.query_one("#btn-waypoint-capture", Button).disabled = False
    except NoMatches:
        pass

    try:
        d.query_one("#waypoint-arm-view").clear_trail()
    except NoMatches:
        pass

    d._log_waypoint("[bold cyan]\U0001f3af WAYPOINT CAPTURE[/]")
    d._log_waypoint(
        "[dim]  Bewege den Arm frei. Druecke [Space] zum Aufnehmen "
        "eines Wegpunkts.[/]"
    )
    d._log_waypoint(
        "[dim]  [Enter] = Save & Stop | [Discard] = Verwerfen | "
        "[g] Gripper | [Esc] Notfall-Stop[/]"
    )
    d._start_activity("Waypoint capture", "\U0001f3af")
    update_waypoint_status_label(d)
    d.add_class("waypoint-active")


def stop_session(d):
    """End the session and save the captured waypoints."""
    if not d.waypoint_capturing:
        return
    d.waypoint_capturing = False

    arm = d._active_arm
    if arm:
        try:
            arm.torque_on()
        except Exception:
            pass
    d.torque_on_state = True
    d._update_status_torque(True)

    try:
        d.query_one("#btn-waypoint-start", Button).disabled = False
        d.query_one("#btn-waypoint-stop", Button).disabled = True
        d.query_one("#btn-waypoint-capture", Button).disabled = True
    except NoMatches:
        pass

    if not d._waypoints:
        d._log_waypoint("[yellow]Keine Wegpunkte aufgezeichnet \u2014 verworfen.[/]")
        d._stop_activity("\u23f9 Cancelled")
        d.remove_class("waypoint-active")
        d.set_timer(2.0, lambda: d._stop_activity())
        return

    filepath = _save_recording(d)
    d._stop_activity("\u2705 Saved")
    d._log_waypoint(
        f"[green]\u2705 {len(d._waypoints)} Wegpunkte gespeichert:[/] {filepath}"
    )
    d._refresh_recordings_table()
    d.remove_class("waypoint-active")
    d.set_timer(3.0, lambda: d._stop_activity())


def cancel_session(d):
    """Discard captured waypoints without saving."""
    if not d.waypoint_capturing:
        return
    n = len(d._waypoints)
    d.waypoint_capturing = False
    d._waypoints = []
    update_waypoint_status_label(d)

    arm = d._active_arm
    if arm:
        try:
            arm.torque_on()
        except Exception:
            pass
    d.torque_on_state = True
    d._update_status_torque(True)

    try:
        d.query_one("#btn-waypoint-start", Button).disabled = False
        d.query_one("#btn-waypoint-stop", Button).disabled = True
        d.query_one("#btn-waypoint-capture", Button).disabled = True
    except NoMatches:
        pass

    d._stop_activity("\u2716 Cancelled")
    d._log_waypoint(f"[yellow]\u2716 Verworfen ({n} Wegpunkte)[/]")
    d.remove_class("waypoint-active")
    d.set_timer(2.0, lambda: d._stop_activity())


# ---------------------------------------------------------------
# Capture: torque pulse -> read -> torque off
# ---------------------------------------------------------------

def capture_waypoint(d):
    """Public entry: dispatch the capture to a worker thread.

    We never want to block the UI thread while the servo torque is on
    and the position is being read.
    """
    if not d.waypoint_capturing:
        return
    if d._waypoint_capture_busy:
        return
    d._waypoint_capture_busy = True

    try:
        d.query_one("#btn-waypoint-capture", Button).disabled = True
    except NoMatches:
        pass

    d.run_worker(
        lambda: _capture_worker(d),
        thread=True,
        exclusive=True,
        group="waypoint-capture",
    )


def _capture_worker(d):
    """Torque pulse -> settle -> read -> torque off, all on a worker."""
    try:
        arm = d._active_arm
        if arm is None:
            d.call_from_thread(
                d._log_waypoint, "[red]Kein Arm verfuegbar![/]")
            return

        d.call_from_thread(d._start_activity, "Measuring", "\U0001f4cf")

        is_sim = d._is_sim
        pos = None

        if is_sim:
            time.sleep(0.05)
            pos = arm.read_position_deg()
        else:
            try:
                arm.torque_on_fast()
            except Exception:
                arm.torque_on()
            time.sleep(WAYPOINT_CAPTURE_SETTLE_S)

            try:
                pos = arm.read_position_deg_single()
                if pos is None:
                    pos = arm.read_position_deg()
            except Exception as e:
                d.call_from_thread(
                    d._log_waypoint, f"[red]Lese-Fehler: {e}[/]")

            try:
                arm.torque_off_fast()
            except Exception:
                arm.torque_off()
            time.sleep(WAYPOINT_TORQUE_PULSE_S - WAYPOINT_CAPTURE_SETTLE_S)

        if pos is None:
            d.call_from_thread(
                d._log_waypoint,
                "[red]\u274c Konnte Position nicht lesen \2014 "
                "Wegpunkt verworfen[/]"
            )
            d.call_from_thread(d._stop_activity, "")
            return

        if not d.waypoint_capturing:
            return

        idx = len(d._waypoints)
        if idx == 0 and d._waypoint_start_pos is None:
            d._waypoint_start_pos = {
                "b": pos["b"], "s": pos["s"],
                "e": pos["e"], "h": pos["h"],
            }

        d._waypoints.append({
            "b": round(pos["b"], 2),
            "s": round(pos["s"], 2),
            "e": round(pos["e"], 2),
            "h": round(pos["h"], 2),
        })

        d.call_from_thread(_on_waypoint_captured, d, idx, pos)

    finally:
        d._waypoint_capture_busy = False
        if d.waypoint_capturing:
            d.call_from_thread(_reenable_capture_button, d)
        d.call_from_thread(d._stop_activity, "\U0001f3af Ready")


def _reenable_capture_button(d):
    try:
        d.query_one("#btn-waypoint-capture", Button).disabled = False
    except NoMatches:
        pass


def _on_waypoint_captured(d, idx: int, pos: dict):
    n = len(d._waypoints)
    sim_tag = " [dim](sim)[/]" if d._is_sim else ""
    d._log_waypoint(
        f"[bold green]\u25c6 WP#{idx + 1}{sim_tag}[/] "
        f"b={pos['b']:+7.2f}\u00b0 s={pos['s']:+7.2f}\u00b0 "
        f"e={pos['e']:+7.2f}\u00b0 h={pos['h']:+7.2f}\u00b0 "
        f"[dim]({n} total)[/]"
    )
    d._current_pos = pos
    d._update_joint_displays(pos)
    d._update_arm_views(pos)

    try:
        view = d.query_one("#waypoint-arm-view")
        view.update_pose(pos["b"], pos["s"], pos["e"])
    except NoMatches:
        pass


# ---------------------------------------------------------------
# File output
# ---------------------------------------------------------------

def _save_recording(d) -> str:
    if not d._waypoints:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RECORDINGS_DIR / f"waypoint_{ts}.roarm"

    start = d._waypoint_start_pos or d._waypoints[0]
    lines = [
        "# RoArm-M2-S Waypoint Recording (Dashboard v3)",
        f"# Datum: {datetime.now().isoformat()}",
        f"# Modus: waypoint (endpoints only)",
        f"# Wegpunkte: {len(d._waypoints)}",
        "#",
        "#CONFIG mode=waypoint",
        "#CONFIG threshold=0.0",
        "#CONFIG hz=0",
        f"#CONFIG v_max={WAYPOINT_V_MAX_DEG_S}",
        f"#CONFIG a_max={WAYPOINT_A_MAX_DEG_S2}",
        f"#CONFIG speed={WAYPOINT_DEFAULT_SPEED}",
        "#CONFIG gravity_comp=0",
        f"#START_POS b={start['b']:.2f} "
        f"s={start['s']:.2f} "
        f"e={start['e']:.2f} "
        f"h={start['h']:.2f}",
        "",
    ]
    for i, wp in enumerate(d._waypoints):
        lines.append(
            f"MOVE b={wp['b']:.2f} s={wp['s']:.2f} "
            f"e={wp['e']:.2f} h={wp['h']:.2f} t={i:.0f}"
        )
    lines.append("")

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    return str(filename)


# ---------------------------------------------------------------
# Status display helpers
# ---------------------------------------------------------------

def update_waypoint_status_label(d):
    """Refresh the waypoint status indicator. Safe to call from any timer."""
    try:
        label = d.query_one("#waypoint-status", Static)
    except NoMatches:
        return
    if not d.waypoint_capturing:
        label.update("")
        return
    n = len(d._waypoints)
    if d._waypoint_capture_busy:
        msg = "[bold yellow]\u23fa Measuring\u2026[/]"
    elif n == 0:
        msg = (
            "[bold cyan]\u25c6 Ready[/] "
            "[dim]| Space = Capture | Enter = Save[/]"
        )
    else:
        msg = (
            f"[bold cyan]\u25c6 {n} WP[/] "
            f"[dim]| Space = Capture | Enter = Save[/]"
        )
    label.update(msg)

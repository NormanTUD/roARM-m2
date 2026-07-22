"""Servo mode methods for RoArmDashboard."""

import time

from textual import on
from textual.widgets import Button, Input, Static
from textual.css.query import NoMatches


def servo_go(d, joint: str):
    arm = d._active_arm
    if arm is None:
        d._log_servo("[red]Nicht verbunden und keine Simulation![/]")
        return

    try:
        input_widget = d.query_one(f"#servo-{joint}-input", Input)
        angle = float(input_widget.value)
    except (NoMatches, ValueError) as e:
        d._log_servo(f"[red]Ungueltiger Wert: {e}[/]")
        return

    pos = d._current_pos.copy()
    pos[joint] = angle

    d._current_pos = pos
    d._update_arm_views(pos)
    d._update_joint_displays(pos)
    d._update_servo_readouts(pos)

    if d._is_sim:
        d._log_servo(
            "[bold yellow]\u26a0 SIMULATION \u2013 Kein realer Arm angeschlossen![/]"
        )

    arm.torque_on()
    d.torque_on_state = True
    d._update_status_torque(True)
    time.sleep(0.1)

    arm.move_to(pos["b"], pos["s"], pos["e"], pos["h"], spd=20, acc=10)

    d._start_activity(f"Moving {joint.upper()}", "\U0001f3af")

    sim_tag = " (sim)" if d._is_sim else ""
    d._log_servo(
        f"[green]\U0001f3af {joint.upper()} \u2192 {angle:.2f}\u00b0{sim_tag}[/]"
    )

    if d._is_sim:
        d.set_timer(0.5, lambda: _servo_read_after_move(d))
    else:
        d.set_timer(1.5, lambda: _servo_read_after_move(d))


def _servo_read_after_move(d):
    arm = d._active_arm
    if arm is None:
        d._stop_activity()
        return
    pos = arm.read_position_deg()
    if pos:
        d._current_pos = pos
        d._update_joint_displays(pos)
        d._update_arm_views(pos)
        d._update_servo_readouts(pos)

    d._stop_activity("\u2705 Move complete")
    d.set_timer(3.0, lambda: d._stop_activity())

"""Play mode methods for RoArmDashboard."""

import json
import time
from pathlib import Path

from textual import on, work
from textual.widgets import Button, DataTable, Input, Label, TabbedContent
from textual.css.query import NoMatches

from .kinematics import (
    STREAM_HZ, STREAM_SPD, STREAM_ACC, LOOKAHEAD_S,
    STREAM_FLUSH_INTERVAL, STREAM_MIN_SEND_INTERVAL_S,
    STREAM_UI_UPDATE_INTERVAL_S, STREAM_EVENT_PAUSE_S,
    STREAM_PREBUFFER_COMMANDS, MAX_TRACKING_ERROR,
    ENDPOINT_SPEEDS, ENDPOINT_SETTLE_WAIT, TRACKING_CHECK_ENABLED,
    RECORDINGS_DIR,
)
from .trajectory import SmoothTrajectory
from .recording import parse_roarm_file
from .widgets import TimelineWidget, RoarmFileViewer

MIN_DELTA_DEG = 0.15

def start_playback(d):
    arm = d._active_arm
    if arm is None:
        d._log_play("[red]Nicht verbunden und keine Simulation![/]")
        return

    try:
        table = d.query_one("#recordings-table", DataTable)

        if table.row_count == 0:
            d._log_play("[yellow]\u26a0 Keine Recordings vorhanden![/]")
            d._log_play(
                "[dim]  \u2192 Nimm zuerst ein Recording im Teach-Tab auf "
                "(Tab 1, Space)[/]"
            )
            return

        row_key = table.cursor_row
        if row_key is None:
            d._log_play(
                "[yellow]\u26a0 Kein Recording ausgewaehlt! "
                "Waehle eine Zeile in der Tabelle.[/]"
            )
            return
        row_data = table.get_row_at(row_key)
        filename = row_data[0]
    except NoMatches:
        d._log_play("[red]\u274c Recordings-Tabelle nicht gefunden![/]")
        return
    except Exception as e:
        d._log_play(f"[red]\u274c Konnte Recording nicht laden: {e}[/]")
        return

    filepath = RECORDINGS_DIR / filename
    if not filepath.exists():
        d._log_play(f"[red]Datei nicht gefunden: {filepath}[/]")
        return

    d._play_data = parse_roarm_file(str(filepath))

    try:
        viewer = d.query_one("#roarm-file-viewer", RoarmFileViewer)
        viewer.load_file(str(filepath))
    except NoMatches:
        pass

    wps = d._play_data["waypoints"]
    if not wps or len(wps) < 4:
        d._log_play("[red]Zu wenige Wegpunkte fuer Spline![/]")
        return

    try:
        timeline = d.query_one("#play-timeline", TimelineWidget)
        timeline.set_recording(wps, d._play_data.get("gripper_cmds"))
    except NoMatches:
        pass

    d.playing = True

    try:
        d.query_one("#btn-play-start", Button).disabled = True
        d.query_one("#btn-play-stop", Button).disabled = False
    except NoMatches:
        pass

    d._start_activity("Starting playback", "\u25b6\ufe0f")

    sim_tag = " (Simulation)" if d._is_sim else ""
    d._log_play(
        f"[green]\u25b6 Playback{sim_tag}: {len(wps)} WPs, {wps[-1]['t']:.1f}s[/]"
    )

    _run_playback(d, wps)


def _load_calibration_model(d, is_sim: bool):
    if is_sim:
        return None
    try:
        from calibrate import CalibrationModel
        cal_path = Path("calibration") / "roarm_calibration.cal"
        if cal_path.exists():
            model = CalibrationModel.load(str(cal_path))
            d.call_from_thread(
                d._log_play, f"[green]\U0001f4c2 Kalibrierung geladen: {cal_path}[/]"
            )
            return model
    except Exception as e:
        d.call_from_thread(
            d._log_play, f"[yellow]\u26a0 Kalibrierung nicht verfuegbar: {e}[/]"
        )
    return None


def _apply_calibration_static(cal_model, target: dict) -> dict:
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


def _show_calibration_info(d):
    cal_path = Path("calibration") / "roarm_calibration.cal"
    if not cal_path.exists():
        d._log_play("[yellow]\u26a0 Keine Kalibrierungsdatei gefunden[/]")
        d._log_play("[dim]  \u2192 Playback ohne Kalibrierung (wie raw Servo-Werte)[/]")
        return

    try:
        from calibrate import CalibrationModel
        model = CalibrationModel.load(str(cal_path))
        res = model.residuals

        d._log_play("[bold cyan]\U0001f4d0 Kalibrierungsmodell geladen:[/]")
        d._log_play(f"  Datei: {cal_path}")
        d._log_play(f"  Typ: Polynom 2. Ordnung (10 Koeffizienten/Gelenk)")
        d._log_play(
            f"  Residuen (Fit-Qualitaet):"
        )
        for j in ["b", "s", "e"]:
            r = res.get(j, 0)
            quality = "\u2705" if r < 0.3 else "\u26a0\ufe0f" if r < 1.0 else "\u274c"
            d._log_play(f"    {j.upper()}: {r:.4f}\u00b0 {quality}")

        diag_path = Path("calibration") / "roarm_diagnostics.json"
        if diag_path.exists():
            import json as _json
            with open(diag_path, 'r') as f:
                diag = _json.load(f)

            if "pose_set" in diag:
                d._log_play(f"  Pose-Set: {diag['pose_set']}")
            if "total_measurements" in diag:
                d._log_play(f"  Messungen: {diag['total_measurements']}")
            if "repeats_per_pose" in diag:
                d._log_play(f"  Wiederholungen/Pose: {diag['repeats_per_pose']}")
            if "avg_settle_time_s" in diag:
                d._log_play(
                    f"  \u00d8 Settle-Zeit: {diag['avg_settle_time_s']:.2f}s"
                )
            if "avg_noise_std_deg" in diag:
                noise = diag["avg_noise_std_deg"]
                d._log_play(
                    f"  \u00d8 Rauschen: b={noise.get('b',0):.4f}\u00b0 "
                    f"s={noise.get('s',0):.4f}\u00b0 e={noise.get('e',0):.4f}\u00b0"
                )
            if "position_error_stats" in diag:
                stats = diag["position_error_stats"]
                d._log_play(f"  Positionsfehler (vor Kalibrierung):")
                for j in ["b", "s", "e"]:
                    if j in stats:
                        s = stats[j]
                        d._log_play(
                            f"    {j.upper()}: mean={s['mean']:+.3f}\u00b0 "
                            f"\u03c3={s['std']:.3f}\u00b0 max={s['max']:.3f}\u00b0"
                        )
            if "repeatability_deg" in diag:
                rep = diag["repeatability_deg"]
                d._log_play(
                    f"  Repeatability: \u0394b={rep.get('b',0):.3f}\u00b0 "
                    f"\u0394s={rep.get('s',0):.3f}\u00b0 \u0394e={rep.get('e',0):.3f}\u00b0"
                )

        home = {"b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0}
        corr = model.predict_correction(home)
        d._log_play(
            f"  [dim]Beispiel Home-Korrektur: "
            f"\u0394b={corr['b']:+.3f}\u00b0 \u0394s={corr['s']:+.3f}\u00b0 "
            f"\u0394e={corr['e']:+.3f}\u00b0[/]"
        )

    except Exception as e:
        d._log_play(f"[red]Fehler beim Laden der Kalibrierung: {e}[/]")


def _get_play_speed(d) -> float:
    try:
        inp = d.query_one("#play-speed-input", Input)
        return max(0.1, min(5.0, float(inp.value)))
    except (NoMatches, ValueError):
        return 1.0


def _get_loop_pause(d) -> float:
    try:
        inp = d.query_one("#play-loop-pause-input", Input)
        return max(0.0, float(inp.value))
    except (NoMatches, ValueError):
        return 0.0


def _is_loop_enabled(d) -> bool:
    try:
        btn = d.query_one("#btn-play-loop", Button)
        return btn.variant == "success"
    except NoMatches:
        return False


def _process_playback_event(d, arm, event: dict, elapsed: float):
    cmd = event["cmd"]
    if cmd == "CLOSE":
        arm.gripper_close()
        d.call_from_thread(
            d._log_play, f"[bold]  \u270a Gripper ZU [{elapsed:.2f}s][/]")
    elif cmd == "OPEN":
        arm.gripper_open()
        d.call_from_thread(
            d._log_play, f"[bold]  \u270b Gripper AUF [{elapsed:.2f}s][/]")
    elif cmd == "LED_ON":
        if hasattr(arm, 'send_cmd'):
            arm.send_cmd({"T": 114, "led": 255})
        d.call_from_thread(
            d._log_play, f"[bold]  \U0001f4a1 LED AN [{elapsed:.2f}s][/]")
    elif cmd == "LED_OFF":
        if hasattr(arm, 'send_cmd'):
            arm.send_cmd({"T": 114, "led": 0})
        d.call_from_thread(
            d._log_play, f"[bold]  \U0001f4a1 LED AUS [{elapsed:.2f}s][/]")


def _process_pending_events(d, arm, events: list, event_idx: int,
                             elapsed: float) -> tuple:
    total_pause = 0.0
    while event_idx < len(events):
        ev = events[event_idx]
        if ev["t"] <= elapsed:
            _process_playback_event(d, arm, ev, elapsed)
            if ev["cmd"] in ("CLOSE", "OPEN"):
                time.sleep(STREAM_EVENT_PAUSE_S)
                total_pause += STREAM_EVENT_PAUSE_S
            event_idx += 1
        else:
            break
    return event_idx, total_pause


def _arm_is_estopped(d) -> bool:
    if d._safe_arm and hasattr(d._safe_arm, 'is_emergency_stopped'):
        return d._safe_arm.is_emergency_stopped
    return False


def _run_loop(d, waypoints: list, arm, is_sim: bool):
    pause_s = _get_loop_pause(d)
    while _is_loop_enabled(d) and not _arm_is_estopped(d):
        if pause_s > 0:
            d.call_from_thread(
                d._log_play, f"[dim]\u23f8 Loop-Pause: {pause_s:.1f}s[/]")
            time.sleep(pause_s)

        try:
            d.call_from_thread(d._reset_file_viewer)
        except Exception:
            pass

        d.playing = True
        cal_model = _load_calibration_model(d, is_sim)
        trajectory = SmoothTrajectory(waypoints, _get_play_speed(d))
        duration = trajectory.get_duration()
        _move_to_start_position(d, arm, waypoints[0], cal_model, is_sim)
        events = sorted(d._play_data.get("events", []), key=lambda x: x["t"])
        _streaming_loop(d, arm, trajectory, duration, cal_model, events, is_sim)
        if d.playing:
            _do_precision_endpoint(d, arm, trajectory, duration, cal_model, is_sim)
        d.playing = False


def _do_precision_endpoint(d, arm, trajectory: 'SmoothTrajectory',
                           duration: float, cal_model, is_sim: bool):
    d.call_from_thread(d._start_activity, "Precision settle", "\U0001f3af")
    final_target = trajectory.sample(duration)
    final_corrected = _apply_calibration_static(cal_model, final_target)
    if is_sim:
        _precision_endpoint_sim(d, arm, final_corrected)
    else:
        _precision_endpoint_real(arm, final_corrected)


def _precision_endpoint_real(arm, final_corrected: dict):
    for spd, acc in ENDPOINT_SPEEDS:
        arm.move_to(
            final_corrected["b"], final_corrected["s"],
            final_corrected["e"], final_corrected["h"],
            spd=spd, acc=acc
        )
        time.sleep(ENDPOINT_SETTLE_WAIT)


def _precision_endpoint_sim(d, arm, final_corrected: dict):
    arm.move_to(
        final_corrected["b"], final_corrected["s"],
        final_corrected["e"], final_corrected["h"],
        spd=5, acc=2
    )
    for _ in range(100):
        time.sleep(0.02)
        d._sim_arm.step_simulation(0.02)
        if not d._sim_arm.is_moving:
            break


def _safe_read_error(pos: dict, target: dict):
    MAX_PLAUSIBLE = 30.0
    err = max(
        abs(pos[j] - target[j])
        for j in ["b", "s", "e", "h"])
    if err > MAX_PLAUSIBLE:
        return None
    return err


def _verify_endpoint(d, arm, final_target):
    time.sleep(0.3)
    if hasattr(arm, 'flush_and_read'):
        pos = arm.flush_and_read()
    else:
        pos = arm.read_position_deg()
    if pos is None:
        return None
    err = max(
        abs(pos[j] - final_target[j])
        for j in ["b", "s", "e", "h"])
    d.call_from_thread(
        d._log_play,
        f"[dim]  Endposition: Fehler={err:.3f}\u00b0 "
        f"(b={pos['b']:.2f} s={pos['s']:.2f} "
        f"e={pos['e']:.2f})[/]")
    return err


def _graceful_stop(d, arm_raw, last_commanded: dict):
    if not last_commanded or not arm_raw:
        return
    from safety import GracefulStop
    raw = getattr(arm_raw, '_arm_raw', arm_raw)
    GracefulStop.execute(raw, last_commanded)


def stop_playback(d):
    d.playing = False
    if d._arm and not d._is_sim and hasattr(d, '_last_play_commanded'):
        _graceful_stop(d, d._arm, d._last_play_commanded)
    elif d._is_sim and d._sim_arm:
        d._sim_arm.torque_off()
    d._stop_activity("\u23f9 Stopped")
    d._log_play("[yellow]\u23f9 Playback gestoppt (graceful)[/]")

    try:
        file_viewer = d.query_one("#roarm-file-viewer", RoarmFileViewer)
        file_viewer.reset()
    except NoMatches:
        pass

    try:
        d.query_one("#btn-play-start", Button).disabled = False
        d.query_one("#btn-play-stop", Button).disabled = True
    except NoMatches:
        pass
    d.set_timer(3.0, lambda: d._stop_activity())


def _lightweight_ui_update(d, pos: dict, elapsed: float):
    try:
        timeline = d.query_one("#play-timeline", TimelineWidget)
        timeline.set_position(elapsed)
    except NoMatches:
        pass

    d._joint_history.push(pos)
    from .widgets import JointSparklineWidget
    for joint in ["b", "s", "e", "h"]:
        try:
            widget = d.query_one(f"#teach-joint-{joint}", JointSparklineWidget)
            widget.update_value(pos[joint])
        except NoMatches:
            pass

    try:
        tabs = d.query_one(TabbedContent)
        if tabs.active == "play":
            from .widgets import Arm3DWidget
            widget = d.query_one("#play-arm-view", Arm3DWidget)
            widget.update_pose(pos["b"], pos["s"], pos["e"])
    except NoMatches:
        pass

    try:
        file_viewer = d.query_one("#roarm-file-viewer", RoarmFileViewer)
        file_viewer.set_current_line_by_position(pos)
    except NoMatches:
        pass


def _move_to_start_position(d, arm, first_wp: dict, cal_model, is_sim: bool):
    start_corrected = _apply_calibration_static(cal_model, first_wp)
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
            d._sim_arm.step_simulation(0.02)
            if not d._sim_arm.is_moving:
                break
    else:
        MAX_WAIT = 8.0
        TOLERANCE = 2.0
        start_wait = time.time()
        while time.time() - start_wait < MAX_WAIT:
            time.sleep(0.3)
            pos = arm.read_position_deg()
            if pos is None:
                continue
            err = max(abs(pos[j] - start_corrected[j])
                      for j in ["b", "s", "e", "h"])
            if err < TOLERANCE:
                d.call_from_thread(
                    d._log_play,
                    f"[green]  \u2713 Startposition erreicht (err={err:.1f}\u00b0)[/]"
                )
                break
        else:
            pos = arm.read_position_deg()
            if pos:
                err = max(abs(pos[j] - start_corrected[j])
                          for j in ["b", "s", "e", "h"])
                d.call_from_thread(
                    d._log_play,
                    f"[yellow]\u26a0 Start-Timeout: err={err:.1f}\u00b0 "
                    f"(Ziel: b={start_corrected['b']:.1f} "
                    f"s={start_corrected['s']:.1f} "
                    f"e={start_corrected['e']:.1f})[/]"
                )
                if err > MAX_TRACKING_ERROR:
                    d.call_from_thread(
                        d._log_play,
                        "[red]\u2717 Startposition nicht erreichbar! Abbruch.[/]"
                    )
                    d.playing = False
                    return
        time.sleep(0.5)


def _update_play_timeline(d, elapsed: float):
    try:
        timeline = d.query_one("#play-timeline", TimelineWidget)
        timeline.set_position(elapsed)
    except NoMatches:
        pass


def _playback_finished(d):
    try:
        d.query_one("#btn-play-start", Button).disabled = False
        d.query_one("#btn-play-stop", Button).disabled = True
    except NoMatches:
        pass


def _streaming_loop(d, arm, trajectory, duration,
                    cal_model, events, is_sim):
    interval = 1.0 / STREAM_HZ

    if not is_sim and hasattr(arm, 'enter_streaming_mode'):
        arm.enter_streaming_mode()
    elif not is_sim:
        arm._ser.reset_input_buffer()
        arm._ser.reset_output_buffer()

    if not is_sim and d._safe_arm:
        d._safe_arm.start_streaming()

    RAMP_UP_COMMANDS = 10

    playback_start = time.time()
    last_pos = None
    commands_sent = 0
    skipped = 0
    event_idx = 0
    d._last_play_commanded = None
    last_ui_update = 0.0
    last_log_time = 0.0
    last_send_time = 0.0

    if not is_sim:
        for pre_i in range(STREAM_PREBUFFER_COMMANDS):
            pre_t = interval * pre_i
            if pre_t >= duration:
                break
            pre_sample_t = min(pre_t + 0.05, duration)
            pre_target = trajectory.sample(pre_sample_t)
            pre_corrected = _apply_calibration_static(cal_model, pre_target)
            cmd = {
                "T": 122,
                "b": round(pre_corrected["b"], 2),
                "s": round(pre_corrected["s"], 2),
                "e": round(pre_corrected["e"], 2),
                "h": round(pre_corrected["h"], 2),
                "spd": STREAM_SPD,
                "acc": STREAM_ACC,
            }
            msg = json.dumps(cmd, separators=(',', ':'))
            arm._ser.write(msg.encode() + b'\n')
            last_pos = pre_corrected.copy()
            d._last_play_commanded = pre_corrected.copy()
            commands_sent += 1
        arm._ser.flush()
        time.sleep(interval * STREAM_PREBUFFER_COMMANDS)

    try:
        while d.playing:
            loop_start = time.time()
            elapsed = loop_start - playback_start

            if elapsed >= duration:
                break

            # --- Event processing ---
            if event_idx < len(events) and events[event_idx]["t"] <= elapsed:
                while event_idx < len(events) and events[event_idx]["t"] <= elapsed:
                    ev = events[event_idx]
                    cmd_str = ev.get("cmd", "")
                    if cmd_str == "CLOSE":
                        if hasattr(arm, 'send_cmd_fast'):
                            arm.send_cmd_fast({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})
                        else:
                            arm.gripper_close()
                    elif cmd_str == "OPEN":
                        if hasattr(arm, 'send_cmd_fast'):
                            arm.send_cmd_fast({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})
                        else:
                            arm.gripper_open()
                    elif cmd_str == "LED_ON":
                        if hasattr(arm, 'send_cmd_fast'):
                            arm.send_cmd_fast({"T": 114, "led": 255})
                    elif cmd_str == "LED_OFF":
                        if hasattr(arm, 'send_cmd_fast'):
                            arm.send_cmd_fast({"T": 114, "led": 0})
                    event_idx += 1
                time.sleep(STREAM_EVENT_PAUSE_S)
                playback_start += STREAM_EVENT_PAUSE_S
                elapsed = time.time() - playback_start
                if elapsed >= duration:
                    break

            # --- SINGLE, UNIFIED lookahead + sampling computation ---
            # Step 1: Base lookahead (with ramp-up for first commands)
            if commands_sent < RAMP_UP_COMMANDS:
                ramp_progress = commands_sent / RAMP_UP_COMMANDS
                current_lookahead = 0.05 + ramp_progress * (LOOKAHEAD_S - 0.05)
            else:
                current_lookahead = LOOKAHEAD_S

            # Step 2: Speed-adaptive lookahead adjustment.
            # In slow (high-curvature) regions, look further ahead so each
            # command represents a meaningful position delta that the servo
            # can execute smoothly without restarting its motion profile.
            speed_at_t = trajectory.get_speed_at(elapsed)
            if speed_at_t < 0.7:
                current_lookahead = current_lookahead + (1.0 - speed_at_t) * 0.2

            # Step 3: Sample the trajectory at the computed lookahead time
            sample_time = min(elapsed + current_lookahead, duration)
            target = trajectory.sample(sample_time)
            corrected = _apply_calibration_static(cal_model, target)

            # --- Send decision: only gate on minimum serial timing ---
            should_send = True
            time_since_last_send = loop_start - last_send_time
            if time_since_last_send < 0.005:  # 5ms = serial write time
                should_send = False

            if should_send:
                if is_sim:
                    arm.move_to_fast(
                        corrected["b"], corrected["s"],
                        corrected["e"], corrected["h"],
                        spd=80, acc=50
                    )
                    d._sim_arm.step_simulation(interval)
                else:
                    cmd = {
                        "T": 122,
                        "b": round(corrected["b"], 2),
                        "s": round(corrected["s"], 2),
                        "e": round(corrected["e"], 2),
                        "h": round(corrected["h"], 2),
                        "spd": STREAM_SPD,
                        "acc": STREAM_ACC,
                    }
                    msg = json.dumps(cmd, separators=(',', ':'))
                    arm._ser.write(msg.encode() + b'\n')

                last_pos = corrected.copy()
                d._last_play_commanded = corrected.copy()
                commands_sent += 1
                last_send_time = time.time()

                if commands_sent % STREAM_FLUSH_INTERVAL == 0:
                    if not is_sim:
                        arm._ser.flush()

            now = time.time()
            if now - last_ui_update > STREAM_UI_UPDATE_INTERVAL_S:
                last_ui_update = now
                ui_pos = corrected.copy() if not is_sim else d._sim_arm.read_position_deg()
                ui_elapsed = elapsed
                try:
                    d.call_from_thread(_lightweight_ui_update, d, ui_pos, ui_elapsed)
                except Exception:
                    pass

            if now - last_log_time > 5.0:
                last_log_time = now
                pct = (elapsed / duration) * 100
                d.call_from_thread(
                    d._log_play,
                    f"[dim]  \u25b6 {pct:.0f}% | t={elapsed:.1f}s | "
                    f"cmds={commands_sent} skip={skipped} "
                    f"rate={commands_sent/max(elapsed,0.1):.0f}Hz[/]"
                )

            target_time = loop_start + interval
            while time.time() < target_time:
                pass

    finally:
        if not is_sim:
            arm._ser.flush()
            time.sleep(0.02)

            if hasattr(arm, 'exit_streaming_mode'):
                arm.exit_streaming_mode()
            else:
                arm._ser.timeout = 0.1

        if not is_sim and d._safe_arm:
            d._safe_arm.end_streaming()

        total_time = time.time() - playback_start
        actual_hz = commands_sent / max(total_time, 0.01)
        d.call_from_thread(
            d._log_play,
            f"[green]  \u2713 Stream done: {commands_sent} cmds, "
            f"{skipped} skipped, {total_time:.2f}s actual, "
            f"{actual_hz:.0f}Hz effective[/]"
        )

def _finalize_playback(d, arm, is_sim, cal_model, duration):
    d.playing = False
    if not is_sim and d._last_play_commanded:
        final_target = d._last_play_commanded
        _verify_endpoint(d, arm, final_target)

    try:
        file_viewer = d.query_one("#roarm-file-viewer", RoarmFileViewer)
        file_viewer.set_finished()
        d.call_from_thread(
            d.set_timer, 2.0,
            lambda: d._reset_file_viewer()
        )
    except NoMatches:
        pass

    d.call_from_thread(
        d._stop_activity, "\u2705 Playback complete")
    d.call_from_thread(lambda: _playback_finished(d))
    d.call_from_thread(
        d.set_timer, 3.0, lambda: d._stop_activity())


@work(thread=True)
def _run_playback(d, waypoints: list):
    arm = d._active_arm
    if arm is None:
        d.call_from_thread(d._log_play, "[red]Kein Arm![/]")
        d.playing = False
        d.call_from_thread(lambda: _playback_finished(d))
        return
    is_sim = d._is_sim
    speed = _get_play_speed(d)
    cal_model = _load_calibration_model(d, is_sim)

    SERVO_MAX_SPEED = 50.0

    max_speed_in_recording = 0.0
    worst_line_idx = 0
    worst_joint = ""
    for i in range(1, len(waypoints)):
        dt = waypoints[i]["t"] - waypoints[i-1]["t"]
        if dt > 0.001:
            for j in ["b", "s", "e", "h"]:
                spd_deg_s = abs(waypoints[i][j] - waypoints[i-1][j]) / dt
                if spd_deg_s > max_speed_in_recording:
                    max_speed_in_recording = spd_deg_s
                    worst_line_idx = i
                    worst_joint = j

    d.call_from_thread(
        d._log_play,
        f"[dim]  \U0001f4ca Max Speed im Recording: {max_speed_in_recording:.1f}\u00b0/s "
        f"(Zeile {worst_line_idx+1}, Joint {worst_joint.upper()}, "
        f"t={waypoints[worst_line_idx]['t']:.2f}s)[/]"
    )

    if max_speed_in_recording > SERVO_MAX_SPEED:
        auto_speed = SERVO_MAX_SPEED / max_speed_in_recording
        if auto_speed < speed:
            old_speed = speed
            speed = auto_speed
            d.call_from_thread(
                d._log_play,
                f"[yellow]\u26a0 Speed auto-reduziert: {old_speed:.2f}x \u2192 {speed:.2f}x "
                f"(Recording: {max_speed_in_recording:.0f}\u00b0/s > "
                f"Servo-Max: {SERVO_MAX_SPEED}\u00b0/s)[/]"
            )

    trajectory = SmoothTrajectory(waypoints, speed)

    if not is_sim:
        from safety import TrajectoryValidator, SafetyLimits
        validator = TrajectoryValidator(SafetyLimits())
        ok, violations = validator.validate_full_trajectory(trajectory)
        if not ok:
            from safety import TrajectorySmoother
            smoother = TrajectorySmoother(SafetyLimits())
            new_wps, n_fixed, added_time = smoother.smooth_trajectory(trajectory)
            if new_wps is None:
                d.call_from_thread(
                    d._log_play, "[red]Trajektorie unsicher![/]")
                d.playing = False
                d.call_from_thread(lambda: _playback_finished(d))
                return
            trajectory = SmoothTrajectory(new_wps, speed)
            ok, _ = TrajectoryValidator(SafetyLimits()).validate_full_trajectory(trajectory)
            if not ok:
                d.call_from_thread(
                    d._log_play, "[red]Trajektorie unsicher![/]")
                d.playing = False
                d.call_from_thread(lambda: _playback_finished(d))
                return
            d.call_from_thread(
                d._log_play,
                f"[yellow]\u26a0 Repariert: {n_fixed} Verletzungen, +{added_time:.2f}s[/]"
            )

    try:
        duration = trajectory.get_duration()
        d.call_from_thread(
            d._log_play,
            f"[green]\u25b6 Playback: {len(waypoints)} WPs, "
            f"{duration:.1f}s (speed={speed:.2f}x)[/]"
        )

        _move_to_start_position(d, arm, waypoints[0], cal_model, is_sim)
        if not d.playing:
            d.call_from_thread(lambda: _playback_finished(d))
            return

        d.call_from_thread(
            d._start_activity,
            f"Playing ({duration:.1f}s)", "\u25b6\ufe0f")
        events = sorted(
            d._play_data.get("events", []),
            key=lambda x: x["t"])

        _streaming_loop(
            d, arm, trajectory, duration,
            cal_model, events, is_sim)
        if d.playing:
            _do_precision_endpoint(
                d, arm, trajectory, duration, cal_model, is_sim)
        _finalize_playback(d, arm, is_sim, cal_model, duration)
        if _is_loop_enabled(d):
            _run_loop(d, waypoints, arm, is_sim)
    except Exception as e:
        d.playing = False
        d.call_from_thread(
            d._log_play,
            f"[bold red]Playback-Fehler: {e}[/]")
        d.call_from_thread(lambda: _playback_finished(d))

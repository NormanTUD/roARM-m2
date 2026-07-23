"""Calibrate mode methods for RoArmDashboard."""

import json
import time
from pathlib import Path

import numpy as np
from textual import on, work
from textual.widgets import Button, Input, Select, Switch, Static
from textual.css.query import NoMatches


def start_calibration(d):
    arm = d._active_arm
    if arm is None:
        d._log_calibrate("[red]Nicht verbunden und keine Simulation![/]")
        return

    if d._is_sim:
        d._log_calibrate(
            "[yellow]\u26a0 Kalibrierung im Simulationsmodus nicht sinnvoll![/]"
        )
        d._log_calibrate(
            "[dim]  Verbinde einen echten Roboter fuer die Kalibrierung.[/]"
        )
        return

    if not d.connected or not d._arm:
        d._log_calibrate("[red]Nicht verbunden![/]")
        return

    try:
        pose_set_select = d.query_one("#cal-pose-set", Select)
        pose_set = pose_set_select.value or "standard"
    except NoMatches:
        pose_set = "standard"

    try:
        repeats_input = d.query_one("#cal-repeats", Input)
        repeats = int(repeats_input.value) if repeats_input.value else 3
    except (NoMatches, ValueError):
        repeats = 3

    try:
        auto_switch = d.query_one("#cal-auto-accept", Switch)
        auto_accept = auto_switch.value
    except NoMatches:
        auto_accept = True

    try:
        d.query_one("#btn-cal-start", Button).disabled = True
        d.query_one("#btn-cal-abort", Button).disabled = False
    except NoMatches:
        pass

    d._start_activity("Calibrating", "\U0001f3af")

    d._log_calibrate(
        f"[bold green]\U0001f3af Kalibrierung gestartet[/]\n"
        f"  Pose-Set: {pose_set}\n"
        f"  Wiederholungen: {repeats}\n"
        f"  Auto-Accept: {'Ja' if auto_accept else 'Nein'}"
    )

    _run_calibration_worker(d, pose_set, repeats, auto_accept)


def _init_calibration_diagnostics(repeats: int, pose_set: str) -> dict:
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


def _cal_log(d, msg: str):
    d.call_from_thread(d._log_calibrate, msg)


def _log_cal_pose_validation(d, valid: list, all_poses: list):
    skipped = len(all_poses) - len(valid)
    if skipped > 0:
        _cal_log(d, f"[yellow]\u26a0 {skipped} Posen uebersprungen (ausserhalb Grenzen)[/]")
    if len(valid) < 10:
        _cal_log(d, f"[yellow]\u26a0 Nur {len(valid)} gueltige Posen![/]")


def _cal_safe_up_between(d):
    from calibrate import move_to_safe_up
    current = d._arm.read_position_deg()
    if current:
        move_to_safe_up(d._arm, current_pose=current)
    else:
        move_to_safe_up(d._arm, current_pose=None)


def _cal_show_diagnostics(d, diagnostics: dict, residuals: dict,
                           poses: list, repeats: int):
    from .kinematics import JOINTS
    _cal_log(d, "\n[bold cyan]\U0001f4ca Kalibrierungs-Ergebnis:[/]")
    for j in JOINTS:
        r = residuals.get(j, 0)
        q = "\u2705" if r < 0.3 else "\u26a0\ufe0f" if r < 1.0 else "\u274c"
        _cal_log(d, f"  {j.upper()}: RMS={r:.4f}\u00b0 {q}")
    if diagnostics["settle_times_s"]:
        arr = np.array(diagnostics["settle_times_s"])
        _cal_log(d,
            f"  Settle: min={arr.min():.2f}s max={arr.max():.2f}s "
            f"avg={arr.mean():.2f}s"
        )
    if diagnostics["overshoot_deg"]:
        arr = np.array(diagnostics["overshoot_deg"])
        _cal_log(d, f"  Overshoot: max={arr.max():.3f}\u00b0 avg={arr.mean():.3f}\u00b0")
    if diagnostics["repeatability_per_pose"]:
        for j in JOINTS:
            vals = [r["repeat_std_deg"][j] for r in diagnostics["repeatability_per_pose"]]
            _cal_log(d, f"  Repeatability {j.upper()}: \u03c3={np.mean(vals):.4f}\u00b0")


def _cal_save_results(d, model, diagnostics: dict, residuals: dict,
                       measurement_count: int, poses: list, repeats: int):
    cal_path = Path("calibration") / "roarm_calibration.cal"
    cal_path.parent.mkdir(exist_ok=True)
    diagnostics["total_measurements"] = measurement_count
    model.save(str(cal_path), diagnostics=diagnostics)
    diag_path = Path("calibration") / "roarm_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    _cal_log(d, f"[green]\u2705 Kalibrierung gespeichert: {cal_path}[/]")
    _cal_log(d, f"[green]\u2705 Diagnostik gespeichert: {diag_path}[/]")


def _cal_fit_model(d, commanded: list, errors: list, repeats: int) -> tuple:
    from calibrate import CalibrationModel
    _cal_log(d, "\n[bold]\U0001f4ca Fitte Kalibrierungsmodell...[/]")
    model = CalibrationModel()
    residuals = model.fit(commanded, errors)
    return model, residuals


def _cal_repeatability_test(d, poses: list, diagnostics: dict):
    from calibrate import move_from_safe_up_to_pose, JOINTS
    _cal_log(d, "[dim]  \U0001f504 Repeatability-Test (Home nochmal)...[/]")
    move_from_safe_up_to_pose(d._arm, poses[0])
    d._arm.move_to(poses[0]["b"], poses[0]["s"], poses[0]["e"], poses[0]["h"], spd=5, acc=3)
    d._arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
    repeat_pos = d._arm.read_position_averaged(n=10, interval=0.05)
    if repeat_pos and diagnostics["per_pose"]:
        first_measured = diagnostics["per_pose"][0].get("measured", {}) if diagnostics["per_pose"] else {}
        if first_measured:
            repeat_err = {j: abs(repeat_pos[j] - first_measured.get(j, repeat_pos[j]))
                          for j in JOINTS}
            diagnostics["repeatability_deg"] = repeat_err
            _cal_log(d,
                f"[dim]  \U0001f504 Home\u2192...\u2192Home: \u0394b={repeat_err['b']:.3f}\u00b0 "
                f"\u0394s={repeat_err['s']:.3f}\u00b0 \u0394e={repeat_err['e']:.3f}\u00b0[/]"
            )


def _cal_run_manual_verification(d, arm, commanded: list,
                                  errors: list, diagnostics: dict) -> tuple:
    from calibrate import integrate_manual_points
    manual_path = Path("calibration") / "manual_points.json"
    if not manual_path.exists():
        return commanded, errors
    try:
        with open(manual_path, 'r') as f:
            manual_points = json.load(f)
        if manual_points:
            old_count = len(commanded)
            commanded, errors = integrate_manual_points(
                commanded, errors, manual_points, weight=2.0
            )
            _cal_log(d,
                f"[green]  \u2713 {len(commanded) - old_count} manuelle Punkte "
                f"integriert (\u00d72 Gewichtung)[/]"
            )
            diagnostics["manual_verification"] = {
                "n_points": len(manual_points),
                "weight": 2.0,
            }
    except Exception as e:
        _cal_log(d, f"[yellow]  \u26a0 Manuelle Punkte nicht ladbar: {e}[/]")
    return commanded, errors


def _cal_aggregate_pose(d, pose: dict, measurements: list,
                        commanded: list, errors: list,
                        diagnostics: dict, repeats: int):
    from .kinematics import JOINTS
    if not measurements:
        return
    avg_error = {j: float(np.mean([m["error"][j] for m in measurements]))
                 for j in JOINTS}
    commanded.append(pose)
    errors.append(avg_error)
    if repeats > 1:
        repeat_std = {j: float(np.std([m["measured"][j] for m in measurements]))
                      for j in JOINTS}
        diagnostics["repeatability_per_pose"].append({
            "pose_index": len(commanded) - 1,
            "repeat_std_deg": repeat_std,
        })


def _cal_compute_overshoot(d, settle_result: dict) -> float:
    from .kinematics import JOINTS
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


def _cal_measure_single_pose(d, pose: dict, pose_idx: int,
                              rep: int, repeats: int, diagnostics: dict):
    from calibrate import move_from_safe_up_to_pose
    from .kinematics import JOINTS
    move_start = time.time()
    move_from_safe_up_to_pose(d._arm, pose)
    d._arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
    result = d._arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
    settle_time = time.time() - move_start
    diagnostics["settle_times_s"].append(settle_time)
    overshoot = _cal_compute_overshoot(d, result)
    diagnostics["overshoot_deg"].append(overshoot)
    servo_avg = d._arm.read_position_averaged(n=10, interval=0.05)
    if not servo_avg:
        return None
    diagnostics["noise_std_deg"].append({
        j: servo_avg.get(f"{j}_std", 0) for j in JOINTS
    })
    error = {j: servo_avg[j] - pose[j] for j in JOINTS}
    measured = {j: servo_avg[j] for j in JOINTS}
    diagnostics["per_pose"].append({
        "pose_index": pose_idx,
        "repeat": rep,
        "commanded": {j: pose[j] for j in JOINTS},
        "measured": measured,
        "error": error,
        "settle_time_s": settle_time,
        "overshoot_deg": overshoot,
    })
    d.call_from_thread(d._update_arm_views, {
        "b": servo_avg["b"], "s": servo_avg["s"],
        "e": servo_avg["e"], "h": servo_avg.get("h", 180.0)
    })
    _cal_log(d,
        f"[dim]  \u2713 Pose {pose_idx+1} Rep {rep+1}: "
        f"\u0394b={error['b']:+.2f}\u00b0 \u0394s={error['s']:+.2f}\u00b0 "
        f"\u0394e={error['e']:+.2f}\u00b0[/]"
    )
    return {"error": error, "measured": measured}


def _cal_update_progress(d, pose_idx: int, n_poses: int,
                          rep: int, repeats: int, count: int, total: int):
    pct = count / total * 100
    d.call_from_thread(
        d._update_cal_status,
        f"Pose {pose_idx+1}/{n_poses} \u00b7 Rep {rep+1}/{repeats} \u00b7 {pct:.0f}%"
    )
    d.call_from_thread(
        d._start_activity,
        f"Cal {pct:.0f}% P{pose_idx+1}/{n_poses}", "\U0001f3af"
    )


@work(thread=True)
def _run_calibration_worker(d, pose_set: str, repeats: int,
                             auto_accept: bool):
    from calibrate import (
        POSE_SETS, validate_pose, move_to_safe_up,
    )

    poses = POSE_SETS.get(pose_set, POSE_SETS["standard"])
    valid_poses = [p for p in poses if validate_pose(p)]
    _log_cal_pose_validation(d, valid_poses, poses)
    total = len(valid_poses) * repeats
    commanded, errors = [], []
    diagnostics = _init_calibration_diagnostics(repeats, pose_set)
    d._arm.torque_on()
    time.sleep(0.2)
    _cal_log(d, "[dim]  Fahre zu Safe-UP...[/]")
    move_to_safe_up(d._arm, current_pose=None)
    measurement_count = 0
    for i, pose in enumerate(valid_poses):
        pose_measurements = []
        for rep in range(repeats):
            measurement_count += 1
            _cal_update_progress(d, i, len(valid_poses), rep, repeats, measurement_count, total)
            if rep > 0 or i > 0:
                _cal_safe_up_between(d)
            result = _cal_measure_single_pose(d, pose, i, rep, repeats, diagnostics)
            if result:
                pose_measurements.append(result)
        _cal_aggregate_pose(d, pose, pose_measurements, commanded, errors, diagnostics, repeats)
    _cal_repeatability_test(d, valid_poses, diagnostics)

    commanded, errors = _cal_run_manual_verification(
        d, d._arm, commanded, errors, diagnostics
    )

    model, residuals = _cal_fit_model(d, commanded, errors, repeats)
    _cal_save_results(d, model, diagnostics, residuals, measurement_count, valid_poses, repeats)
    _cal_show_diagnostics(d, diagnostics, residuals, valid_poses, repeats)
    _cal_cleanup(d)


def _cal_cleanup(d):
    from calibrate import move_to_safe_up
    current = d._arm.read_position_deg()
    if current:
        move_to_safe_up(d._arm, current_pose=current)
    d.call_from_thread(_cal_finished, d)
    d.call_from_thread(d._stop_activity, "\u2705 Calibration complete")
    d.call_from_thread(d.set_timer, 5.0, lambda: d._stop_activity())


def _update_cal_status(d, text: str):
    try:
        panel = d.query_one("#cal-status-panel", Static)
        panel.update(f"[bold]{text}[/]")
    except NoMatches:
        pass


def _cal_finished(d):
    try:
        d.query_one("#btn-cal-start", Button).disabled = False
        d.query_one("#btn-cal-abort", Button).disabled = True
    except NoMatches:
        pass


def abort_calibration(d):
    d._stop_activity("\u26a0 Calibration aborted")
    d._log_calibrate("[yellow]\u26a0 Kalibrierung abgebrochen![/]")
    try:
        d.query_one("#btn-cal-start", Button).disabled = False
        d.query_one("#btn-cal-abort", Button).disabled = True
    except NoMatches:
        pass
    d.set_timer(3.0, lambda: d._stop_activity())


def load_calibration(d):
    from calibrate import CalibrationModel
    cal_path = Path("calibration") / "roarm_calibration.cal"
    if not cal_path.exists():
        d._log_calibrate("[yellow]Keine Kalibrierungsdatei gefunden![/]")
        return

    try:
        model = CalibrationModel.load(str(cal_path))
        d._log_calibrate(
            f"[green]\u2705 Kalibrierung geladen: {cal_path}[/]\n"
            f"  Residuen: b={model.residuals.get('b', 0):.4f}\u00b0 "
            f"s={model.residuals.get('s', 0):.4f}\u00b0 "
            f"e={model.residuals.get('e', 0):.4f}\u00b0"
        )
    except Exception as e:
        d._log_calibrate(f"[red]Fehler: {e}[/]")

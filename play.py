#!/usr/bin/env python3
"""smooth_player.py - RoArm-M2-S Flüssige Wiedergabe durch Cubic Spline + High-Frequency Streaming
Statt diskrete Punkte abzufahren, wird eine glatte Kurve interpoliert
und mit hoher Frequenz an den Arm gestreamt, sodass der Arm nie "anhält".
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "numpy",
#     "scipy",
#     "rich",
#     "matplotlib",
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
import serial
import serial.tools.list_ports
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

from robot import (
    RoArmConnection, find_arm_port, rad_to_deg,
    POSITION_TOLERANCE,
)

from safety import (
    SafeArm, SafetyLimits, SafetyWatchdog,
    CurrentMonitor, ThermalEstimator, RateLimiter,
    TrajectoryValidator, GracefulStop
)

from ui import (
    console, print_banner, print_section, print_position,
    print_connection_status, print_trajectory_info, print_preflight_check,
    print_safety_violation, print_emergency_stop, playback_summary,
    PlaybackDisplay, print_success, print_warning, print_error
)

# Kalibrierungsmodell (optional)
try:
    from calibrate import CalibrationModel, JOINTS
except (ImportError, ModuleNotFoundError):
    CalibrationModel = None

# ============================================================
# KONFIGURATION
# ============================================================

BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1

# Streaming-Parameter
STREAM_HZ = 40             # Befehle pro Sekunde (30-50 Hz ideal)
STREAM_SPD = 30        # statt 50
STREAM_ACC = 15        # statt 30
MIN_DELTA_DEG = 0.05       # Minimale Änderung zum Senden

# Adaptive Timing Parameter
MIN_SPEED_FACTOR = 0.3     # Nie langsamer als 30% der Normalgeschwindigkeit
MAX_SPEED_FACTOR = 1.3 # statt 2.5
END_RAMP_PERCENT = 0.10    # Letzte 10% abbremsen
START_RAMP_PERCENT = 0.05  # Erste 5% sanft anfahren

# Endpoint Precision
ENDPOINT_SETTLE_PASSES = 3
ENDPOINT_FINAL_SPD = 3
ENDPOINT_FINAL_ACC = 1
ENDPOINT_SETTLE_WAIT = 0.8


# ============================================================
# TRAJEKTORIEN-GLÄTTUNG
# ============================================================

class SmoothTrajectory:
    """
    Erzeugt eine glatte, zeitkontinuierliche Trajektorie aus diskreten Wegpunkten.
    
    Prinzip:
    1. Cubic Spline über alle Wegpunkte → C2-stetige Kurve (Position, Geschwindigkeit, 
       Beschleunigung sind stetig)
    2. Geschwindigkeitsprofil: Schnell auf geraden Strecken, langsam bei Richtungswechsel
    3. High-Frequency Streaming: Alle 25ms ein neuer Befehl → Servo bremst nie ab
    """
    
    def __init__(self, waypoints: list, speed_factor: float = 1.0):
        self._waypoints = waypoints
        self._speed_factor = speed_factor
        self._splines = {}
        self._time_map = None
        self._total_duration = 0.0
        self._t_new = None
        self._speed_profile_debug = None
        self._original_duration = 0.0
        
        self._build_splines()
        self._compute_adaptive_timing()
    
    def _build_splines(self):
        """Erstellt Cubic Splines für jedes Gelenk."""
        times = np.array([wp["t"] for wp in self._waypoints])
        
        # FIX: Wenn erstes Sample nicht bei t=0 ist, 
        # füge einen Punkt bei t=0 ein (= gleiche Position wie erster Punkt)
        if times[0] > 0.01:
            # Stillstand am Anfang: Position bei t=0 = Position bei erstem Wegpunkt
            t_pad = np.array([0.0, times[0] * 0.5])  # Zwei Punkte für stabilen Start
            times = np.concatenate([t_pad, times])
            
        for joint in ["b", "s", "e", "h"]:
            values = np.array([wp[joint] for wp in self._waypoints])
            
            # FIX: Padding-Werte = erster Wegpunkt (Stillstand)
            if len(times) > len(values):
                first_val = values[0]
                pad = np.array([first_val, first_val])
                values = np.concatenate([pad, values])
            
            # bc_type='clamped' statt 'natural' → Geschwindigkeit = 0 an den Enden
            self._splines[joint] = CubicSpline(times, values, bc_type='clamped')
        
        self._original_duration = times[-1]
      
    def _compute_adaptive_timing(self):
        """
        Berechnet eine Zeitumparametrisierung:
        - Hohe Krümmung (Richtungswechsel) → Zeit dehnen (langsamer fahren)
        - Niedrige Krümmung (gerade Strecke) → Zeit stauchen (schneller fahren)
        """
        n_samples = 500
        t_original = np.linspace(0, self._original_duration, n_samples)
        
        # Berechne Krümmung (2. Ableitung) an jedem Punkt
        curvature = np.zeros(n_samples)
        for joint in ["b", "s", "e", "h"]:
            d2 = self._splines[joint](t_original, 2)  # 2. Ableitung
            curvature += d2 ** 2
        curvature = np.sqrt(curvature)
        
        # Glätte die Krümmung
        kernel_size = 20
        curvature_smooth = np.convolve(curvature, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Normalisiere
        max_curv = np.percentile(curvature_smooth, 95) if curvature_smooth.max() > 0 else 1.0
        norm_curv = np.clip(curvature_smooth / max(max_curv, 1e-6), 0, 1)
        
        # Speed = invers zur Krümmung
        speed_profile = MAX_SPEED_FACTOR - norm_curv * (MAX_SPEED_FACTOR - MIN_SPEED_FACTOR)
        
        # Sanftes Abbremsen am Ende
        end_ramp_start = int(n_samples * (1.0 - END_RAMP_PERCENT))
        for i in range(end_ramp_start, n_samples):
            progress = (i - end_ramp_start) / (n_samples - end_ramp_start)
            speed_profile[i] = min(speed_profile[i], 
                                    MIN_SPEED_FACTOR + (1.0 - progress) * (speed_profile[i] - MIN_SPEED_FACTOR))
        
        # Sanftes Anfahren am Anfang
        start_ramp_end = int(n_samples * START_RAMP_PERCENT)
        for i in range(start_ramp_end):
            progress = i / max(start_ramp_end, 1)
            speed_profile[i] = MIN_SPEED_FACTOR + progress * (speed_profile[i] - MIN_SPEED_FACTOR)
        
        # Zeitumparametrisierung: dt_new = dt_original / speed
        dt = t_original[1] - t_original[0]
        dt_new = dt / (speed_profile * self._speed_factor)
        
        # Kumulative neue Zeit
        t_new = np.cumsum(dt_new)
        t_new = np.insert(t_new, 0, 0.0)[:-1]
        
        self._total_duration = t_new[-1]
        self._t_new = t_new
        self._speed_profile_debug = speed_profile
        
        # Inverse Mapping: gegeben neue Zeit, finde originale Zeit
        self._time_map = CubicSpline(t_new, t_original, bc_type='natural')
    
    def get_duration(self) -> float:
        """Gesamtdauer der geglätteten Trajektorie."""
        return self._total_duration
    
    def sample(self, t_playback: float) -> dict:
        """
        Gibt die Gelenkwinkel zum Zeitpunkt t_playback zurück.
        t_playback ist in der "neuen" (adaptiven) Zeitskala.
        """
        t_playback = np.clip(t_playback, 0, self._total_duration)
        
        # Mappe zurück auf originale Zeit
        t_orig = float(self._time_map(t_playback))
        t_orig = np.clip(t_orig, 0, self._original_duration)
        
        # Sample die Splines
        result = {}
        for joint in ["b", "s", "e", "h"]:
            result[joint] = round(float(self._splines[joint](t_orig)), 2)
        
        return result
    
    def get_speed_at(self, t_playback: float) -> float:
        """Gibt den aktuellen Speed-Faktor zurück (für Visualisierung)."""
        idx = np.searchsorted(self._t_new, t_playback)
        idx = min(idx, len(self._speed_profile_debug) - 1)
        return self._speed_profile_debug[idx]


# ============================================================
# DATEI PARSEN
# ============================================================

def parse_roarm_file(filepath: str) -> dict:
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


def apply_offset_to_waypoints(waypoints: list, offset: dict, blend_points: int = 5) -> list:
    """Wendet den Offset auf die letzten N Wegpunkte an (linearer Blend)."""
    has_offset = any(abs(v) > 0.001 for v in offset.values())
    if not has_offset:
        return waypoints

    total = len(waypoints)
    blend_start = max(0, total - blend_points)
    result = []

    for i, wp in enumerate(waypoints):
        if i < blend_start:
            result.append(wp.copy())
        else:
            progress = (i - blend_start) / max(1, total - 1 - blend_start)
            result.append({
                "t": wp["t"],
                "b": round(wp["b"] + offset["b"] * progress, 2),
                "s": round(wp["s"] + offset["s"] * progress, 2),
                "e": round(wp["e"] + offset["e"] * progress, 2),
                "h": round(wp["h"] + offset["h"] * progress, 2),
            })

    return result


# ============================================================
# SMOOTH PLAYER
# ============================================================

class SmoothPlayer:
    """
    Spielt .roarm Dateien flüssig ab mit Cubic Spline + High-Frequency Streaming.
    
    Prinzip: Statt "fahre zu Punkt X, warte, fahre zu Punkt Y"
    wird alle 25ms ein neuer Zielpunkt gesendet. Der Servo ist
    IMMER in Bewegung und bremst nie vollständig ab.
    """

    def __init__(self, filepath: str, port: str = None, speed: float = 1.0,
                 loop: bool = False, verify: bool = True, manual_offset: dict = None,
                 stream_hz: int = STREAM_HZ, visualize: bool = False):
        self._filepath = filepath
        self._port = port
        self._speed = speed
        self._loop = loop
        self._verify = verify
        self._manual_offset = manual_offset
        self._stream_hz = stream_hz
        self._arm: SafeArm = None
        self._arm_raw: RoArmConnection = None
        self._data = None
        self._trajectory: SmoothTrajectory = None
        self._watchdog: SafetyWatchdog = None
        self._current_monitor: CurrentMonitor = None
        self._thermal: ThermalEstimator = None
        self._rate_limiter: RateLimiter = None
        self._visualize = visualize

        # Kalibrierungsmodell laden
        self._cal_model = None
        if CalibrationModel is not None:
            cal_path = Path("calibration/roarm_calibration.cal")
            if cal_path.exists():
                try:
                    self._cal_model = CalibrationModel.load(str(cal_path))
                    print(f"📂 Kalibrierungsmodell geladen")
                except Exception as e:
                    print(f"⚠️  Kalibrierung fehlgeschlagen: {e}")
            else:
                print(f"📂 Keine Kalibrierungsdatei gefunden (optional)")
        else:
            print(f"📂 numpy/calibrate nicht verfügbar, keine Kalibrierung")


    def _apply_calibration(self, target: dict) -> dict:
        """Wendet Kalibrierungskorrektur an - MIT CLAMP."""
        if self._cal_model and self._cal_model.is_fitted:
            correction = self._cal_model.predict_correction(target)
            
            # CLAMP: Korrektur darf nie mehr als ±3° betragen!
            MAX_CORRECTION_DEG = 3.0
            for j in ["b", "s", "e"]:
                if abs(correction[j]) > MAX_CORRECTION_DEG:
                    print(f"  ⚠️  Kalibrierung {j}={correction[j]:+.2f}° "
                          f"geclampt auf ±{MAX_CORRECTION_DEG}°")
                    correction[j] = max(-MAX_CORRECTION_DEG, 
                                       min(MAX_CORRECTION_DEG, correction[j]))
            
            return {
                "b": target["b"] - correction["b"],
                "s": target["s"] - correction["s"],
                "e": target["e"] - correction["e"],
                "h": target["h"],
            }
        return target

    def connect(self) -> bool:
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            return False
        try:
            self._arm_raw = RoArmConnection(port)

            if self._visualize:
                from visualize import VisualizingArm
                self._viz_arm = VisualizingArm(self._arm_raw, show_target=True, trail=True)
                print("   🖥️  3D-Visualisierung aktiv")
            else:
                self._viz_arm = None

            print_connection_status(port, success=True, safety_features=[
                "SafeArm (Positions-/Speed-Validierung)",
                "Watchdog (Überhitzungs-Timer)",
                "CurrentMonitor (Stall-Erkennung)",
                "ThermalEstimator (Temperatur-Schätzung)",
                f"RateLimiter (max {self._stream_hz + 10} Hz)",
            ])

            # Safety-Wrapper mit Limits
            limits = SafetyLimits(
                max_delta_per_cmd=20.0,
                max_continuous_move_s=90.0,
                max_plausible_error=5.0,
            )
            self._arm = SafeArm(self._arm_raw, limits=limits)

            # Watchdog starten
            self._watchdog = SafetyWatchdog(self._arm)
            self._watchdog.start()

            # Current Monitor (Stall-Erkennung)
            self._current_monitor = CurrentMonitor(
                self._arm,
                max_load_percent=85.0,
                max_stall_duration_s=3.0
            )

            # Thermal Estimator
            self._thermal = ThermalEstimator(
                ambient_temp_c=25.0,
                thermal_time_constant_s=120.0,
                max_safe_temp_c=55.0,
                warning_temp_c=45.0
            )

            # Rate Limiter (max Stream-Hz + Sicherheitsmarge)
            self._rate_limiter = RateLimiter(max_hz=self._stream_hz + 10)

            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False


    def load(self) -> bool:
        path = Path(self._filepath)
        if not path.exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return False

        self._data = parse_roarm_file(str(path))

        wps = self._data["waypoints"]
        if not wps:
            print("   ❌ Keine Wegpunkte in der Datei!")
            return False

        if len(wps) < 4:
            print(f"   ❌ Mindestens 4 Wegpunkte nötig für Spline (habe {len(wps)})")
            return False

        # Offset anwenden
        offset = self._manual_offset if self._manual_offset else self._data["offset"]
        has_offset = any(abs(v) > 0.001 for v in offset.values())

        if has_offset:
            wps = apply_offset_to_waypoints(wps, offset)

        # Trajektorie erstellen
        self._trajectory = SmoothTrajectory(wps, self._speed)
        duration = self._trajectory.get_duration()
        
        # Geschwindigkeitsprofil-Statistik
        n_test = 100
        speeds = [self._trajectory.get_speed_at(t) for t in np.linspace(0, duration, n_test)]

        # Gelenkbereiche anzeigen
        for j in ["b", "s", "e", "h"]:
            vals = [wp[j] for wp in wps]

        print_trajectory_info(
            n_waypoints=len(wps),
            duration_original=wps[-1]['t'],
            duration_smooth=duration,
            stream_hz=self._stream_hz,
            speed_stats={"min": min(speeds), "max": max(speeds), "avg": float(np.mean(speeds))},
            joint_ranges={
                j: {"min": min(wp[j] for wp in wps), "max": max(wp[j] for wp in wps)}
                for j in ["b", "s", "e", "h"]
            },
            has_offset=has_offset,
            offset=offset,
        )

        # PRE-FLIGHT CHECK
        validator = TrajectoryValidator(SafetyLimits())
        is_safe, violations = validator.validate_full_trajectory(self._trajectory)

        print_preflight_check(is_safe, violations)

        if not is_safe:
            print(f"   🛑 TRAJEKTORIE UNSICHER! {len(violations)} Verletzungen:")
            for v in violations[:10]:
                print(f"      ⚠️  {v}")
            print(f"\n   Abbruch. Trajektorie wird NICHT abgespielt.")
            return False
        else:
            print(f"   ✅ Trajektorie sicher (alle Punkte innerhalb Grenzen)")

        return True

    def go_to_start(self) -> bool:
        """Fährt zur Startposition - mit Safety-Checks."""
        start = self._data["start_pos"]
        print(f"\n🏁 Fahre zur Startposition...")
        print(f"   Ziel: b={start['b']:.2f}° s={start['s']:.2f}° "
              f"e={start['e']:.2f}° h={start['h']:.2f}°")

        # Emergency Stop prüfen
        if self._arm.is_emergency_stopped:
            print(f"   🚨 Emergency Stop aktiv - kann nicht fahren!")
            return False

        self._arm.torque_on()
        time.sleep(0.2)

        # Erster Move: Validator braucht eine initiale Position
        # Setze last_commanded manuell damit der Sprung-Check nicht auslöst
        self._arm.validator._last_commanded = {
            "b": start["b"], "s": start["s"],
            "e": start["e"], "h": start["h"], "t": time.time()
        }

        self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=25, acc=10)
        time.sleep(2.0)

        self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=10, acc=5)
        time.sleep(1.5)

        if self._verify:
            pos = self._arm.flush_and_read()
            if pos:
                err = self._arm.safe_read_error(pos, start)
                if err is None:
                    print(f"   ⚠️ Unplausibler Read bei Startposition, fahre trotzdem weiter")
                elif err > POSITION_TOLERANCE:
                    print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                          f"e={pos['e']:.2f}° h={pos['h']:.2f}°")
                    print(f"   Max Fehler: {err:.2f}° - korrigiere...")
                    self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=5, acc=3)
                    time.sleep(2.0)
                else:
                    print(f"   ✅ Startposition OK (Fehler: {err:.2f}°)")

        # Zum ersten Punkt der Trajektorie
        first = self._trajectory.sample(0.0)
        corrected = self._apply_calibration(first)
        self._arm.move_to(corrected["b"], corrected["s"], corrected["e"], corrected["h"], spd=15, acc=8)
        time.sleep(1.0)

        return True


    def play_once(self):
        """Spielt die Trajektorie einmal flüssig ab mit allen Safety-Checks."""
        traj = self._trajectory
        duration = traj.get_duration()
        interval = 1.0 / self._stream_hz

        gripper_cmds = sorted(self._data["gripper_cmds"], key=lambda x: x["t"])
        gripper_idx = 0

        print(f"\n▶ STREAMING PLAYBACK ({self._stream_hz} Hz, {duration:.2f}s)")
        print(f"   {'─' * 55}")

        last_pos = None
        commands_sent = 0
        skipped = 0

        # === SAFETY: Streaming-Start registrieren + Buffer flush ===
        self._arm.start_streaming()
        self._current_monitor.reset()

        # Feedback-Check Konfiguration
        FEEDBACK_CHECK_INTERVAL = max(1, self._stream_hz // 2)  # Alle 0.5s
        MAX_TRACKING_ERROR_DEG = 8.0

        playback_start = time.time()

        try:
            display = PlaybackDisplay(total_duration=duration, stream_hz=self._stream_hz)
            display.start()

            tracking_err = None

            while True:
                elapsed = time.time() - playback_start
                if elapsed >= duration:
                    break

                # === SAFETY: Emergency Stop prüfen ===
                if self._arm.is_emergency_stopped:
                    print(f"\n   🚨 EMERGENCY STOP aktiv - Abbruch!")
                    break

                # === SAFETY: Thermal Check ===
                temp, temp_status = self._thermal.get_status()
                if temp_status == "CRITICAL":
                    print(f"\n   🌡️ ÜBERHITZUNGSSCHUTZ: ~{temp:.0f}°C (geschätzt)")
                    self._arm.trigger_emergency_stop(
                        f"Thermischer Schutz: geschätzte Temperatur {temp:.0f}°C")
                    break
                elif temp_status == "HOT" and commands_sent % (self._stream_hz * 5) == 0:
                    # Alle 5s warnen wenn HOT
                    print(f"\n   🌡️ Warnung: ~{temp:.0f}°C (geschätzt)")

                # Gripper-Befehle (basierend auf Original-Zeitskala, approximiert)
                approx_original_time = elapsed * self._speed
                while gripper_idx < len(gripper_cmds):
                    gc = gripper_cmds[gripper_idx]
                    if gc["t"] <= approx_original_time:
                        if gc["cmd"] == "CLOSE":
                            self._arm.gripper_close()
                            print(f"\n   ✊ GRIPPER ZU [{elapsed:.2f}s]")
                        else:
                            self._arm.gripper_open()
                            print(f"\n   ✋ GRIPPER AUF [{elapsed:.2f}s]")
                        time.sleep(0.3)
                        playback_start += 0.3
                        gripper_idx += 1
                    else:
                        break

                # Nächsten Punkt auf der glatten Kurve samplen
                target = traj.sample(elapsed)

                # Kalibrierung anwenden
                corrected = self._apply_calibration(target)

                # Nur senden wenn sich genug geändert hat
                should_send = True
                delta_deg = 0.0
                if last_pos:
                    delta_deg = max(abs(corrected[j] - last_pos[j]) for j in ["b", "s", "e", "h"])
                    if delta_deg < MIN_DELTA_DEG:
                        should_send = False
                        skipped += 1

                if should_send:
                    # === SAFETY: Rate Limiter ===
                    self._rate_limiter.acquire()

                    # === SAFETY: Sicherer Send (wird validiert) ===
                    sent = self._arm.move_to_fast(
                        corrected["b"], corrected["s"], corrected["e"], corrected["h"],
                        spd=STREAM_SPD, acc=STREAM_ACC
                    )

                    if not sent:
                        print(f"\n   🛑 Befehl blockiert durch Safety-Layer - stoppe Playback!")
                        self._arm.trigger_emergency_stop("Befehl durch Safety blockiert")
                        break

                    commands_sent += 1
                    last_pos = corrected.copy()

                    # === SAFETY: Thermal Update ===
                    self._thermal.update(is_moving=True, delta_deg=delta_deg)

                    # === SAFETY: Feedback-Loop (periodisch Position prüfen) ===
                    if commands_sent % FEEDBACK_CHECK_INTERVAL == 0 and commands_sent > FEEDBACK_CHECK_INTERVAL:
                        # Position lesen ohne den Stream zu sehr zu stören
                        try:
                            actual = self._arm_raw.read_position_raw()
                            if actual and "b" in actual:
                                actual_deg = {
                                    "b": round(rad_to_deg(actual["b"]), 2),
                                    "s": round(rad_to_deg(actual["s"]), 2),
                                    "e": round(rad_to_deg(actual["e"]), 2),
                                    "h": round(rad_to_deg(actual.get("t", actual.get("h", 0))), 2),
                                }

                                # Plausibilitätscheck
                                ok, _ = self._arm.validator.validate_read_position(actual_deg)
                                if ok:
                                    tracking_err = max(
                                        abs(actual_deg[j] - corrected[j]) for j in ["b", "s", "e", "h"]
                                    )

                                    if tracking_err > MAX_TRACKING_ERROR_DEG:
                                        print(f"\n   🛑 TRACKING ERROR: {tracking_err:.1f}° "
                                              f"(max erlaubt: {MAX_TRACKING_ERROR_DEG}°)")
                                        print(f"      Soll: b={corrected['b']:.1f} s={corrected['s']:.1f} "
                                              f"e={corrected['e']:.1f}")
                                        print(f"      Ist:  b={actual_deg['b']:.1f} s={actual_deg['s']:.1f} "
                                              f"e={actual_deg['e']:.1f}")
                                        self._arm.trigger_emergency_stop(
                                            f"Tracking Error {tracking_err:.1f}° > {MAX_TRACKING_ERROR_DEG}°")
                                        break

                                    # === SAFETY: Stall-Erkennung ===
                                    ok_stall, stall_reason = self._current_monitor.check(actual_deg)
                                    if not ok_stall:
                                        print(f"\n   🛑 {stall_reason}")
                                        self._arm.trigger_emergency_stop(stall_reason)
                                        break
                        except Exception:
                            pass  # Read-Fehler während Streaming sind OK, nicht kritisch

                    if self._viz_arm:
                        self._viz_arm.visualizer.update_pose(
                            corrected["b"], corrected["s"],
                            corrected["e"], corrected["h"]
                        )

                    # Live-Ausgabe
                    speed_now = traj.get_speed_at(elapsed)
                    bar_len = int((speed_now / MAX_SPEED_FACTOR) * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    progress_pct = (elapsed / duration) * 100

                else:
                    # Auch bei Nicht-Senden: Thermal updaten (Haltestrom)
                    self._thermal.update(is_moving=False, delta_deg=0.0)
                    # FIX: last_pos trotzdem aktualisieren damit delta_deg beim
                    # nächsten Send korrekt berechnet wird
                    last_pos = corrected.copy()


                display.update(
                    elapsed=elapsed,
                    target=target,
                    speed_factor=traj.get_speed_at(elapsed),
                    commands_sent=commands_sent,
                    skipped=skipped,
                    thermal_temp=temp,
                    thermal_status=temp_status,
                    tracking_error=tracking_err if 'tracking_err' in dir() else None,
                )

                # Timing einhalten
                next_time = playback_start + (commands_sent + skipped + 1) * interval
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            display.stop()

        except KeyboardInterrupt:
            print(f"\n\n   ⏹ Manuell abgebrochen!")
            self._arm.trigger_emergency_stop("Manueller Abbruch (Ctrl+C)")

        # === SAFETY: Streaming-Ende + Buffer Flush ===
        self._arm.end_streaming()

        # === Wenn Emergency Stop: Hier aufhören ===
        if self._arm.is_emergency_stopped:
            print(f"\n   ⚠️  Playback wurde durch Safety-System gestoppt.")
            print(f"      Arm ist schlaff (Torque off). Prüfe den Arm!")
            return

        # ============================================================
        # PRECISION ENDPOINT (nur wenn kein Emergency Stop)
        # ============================================================
        final = traj.sample(duration)
        corrected_final = self._apply_calibration(final)

        print(f"\n\n   🎯 Precision-Endpunkt ({ENDPOINT_SETTLE_PASSES} Passes):")

        speeds_sequence = [
            (8, 4),
            (5, 2),
            (ENDPOINT_FINAL_SPD, ENDPOINT_FINAL_ACC),
        ]

        for pass_num in range(ENDPOINT_SETTLE_PASSES):
            spd, acc = speeds_sequence[min(pass_num, len(speeds_sequence) - 1)]

            # === SAFETY: Auch Precision-Moves durch SafeArm ===
            sent = self._arm.move_to(
                corrected_final["b"], corrected_final["s"],
                corrected_final["e"], corrected_final["h"],
                spd=spd, acc=acc
            )

            if not sent:
                print(f"      ⚠️ Precision-Move blockiert, überspringe")
                break

            time.sleep(ENDPOINT_SETTLE_WAIT)

            # === SAFETY: Sicherer Read mit flush ===
            pos = self._arm.flush_and_read()

            if pos is None:
                print(f"      Pass {pass_num + 1}: spd={spd} acc={acc} → (kein Read)")
                continue

            # === SAFETY: Fehler plausibel? ===
            err = self._arm.safe_read_error(pos, final)

            if err is None:
                print(f"      ⚠️ Unplausibler Read, überspringe Precision-Endpoint")
                break

            status = "✅" if err < 0.3 else "⚠️"
            print(f"      Pass {pass_num + 1}: spd={spd} acc={acc} → "
                  f"Fehler={err:.3f}° {status}")

            if err < 0.15:
                print(f"      → Präzision erreicht, fertig")
                break

            # Statistik
            actual_duration = time.time() - playback_start

        playback_summary(
            duration_actual=actual_duration,
            duration_planned=duration,
            commands_sent=commands_sent,
            skipped=skipped,
            final_error=err if err is not None else None,
            thermal_temp=temp,
            rate_limiter_violations=self._rate_limiter.violations,
        )

        # Rate Limiter Stats
        if self._rate_limiter.violations > 0:
            print(f"   ⚠️  Rate-Limiter Eingriffe: {self._rate_limiter.violations}")

        # Thermal Status
        temp, temp_status = self._thermal.get_status()
        temp_icon = {"OK": "✅", "WARM": "🌡️", "HOT": "⚠️", "CRITICAL": "🚨"}
        print(f"   Temperatur (geschätzt): ~{temp:.0f}°C {temp_icon.get(temp_status, '')}")

        # Endposition verifizieren
        if self._verify:
            time.sleep(0.3)
            pos = self._arm.flush_and_read()
            if pos:
                err = self._arm.safe_read_error(pos, final)
                if err is not None:
                    print(f"   Endposition Fehler: {err:.2f}°")
                    print(f"   Soll: b={final['b']:.2f}° s={final['s']:.2f}° "
                          f"e={final['e']:.2f}° h={final['h']:.2f}°")
                    print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                          f"e={pos['e']:.2f}° h={pos['h']:.2f}°")
                else:
                    print(f"   ⚠️  Endposition-Read unplausibel, übersprungen")

    def run(self):
        print_banner("play", "Cubic Spline + High-Frequency Streaming\n🛡️ MIT SAFETY-LAYER")

        if not self.load():
            return

        if not self.connect():
            return

        if not self.go_to_start():
            self._cleanup()
            return

        print(f"\n{'─' * 60}")
        print(f"  Bereit! Drücke ENTER um die Wiedergabe zu starten.")
        if self._loop:
            print(f"  (Loop-Modus: Ctrl+C zum Stoppen)")
        print(f"{'─' * 60}")
        input()

        # Abspielen
        try:
            if self._loop:
                loop_count = 0
                while True:
                    # Emergency Stop prüfen vor jedem Loop
                    if self._arm.is_emergency_stopped:
                        print(f"\n   🚨 Emergency Stop aktiv - Loop beendet")
                        break

                    loop_count += 1
                    print(f"\n{'═' * 40} Loop #{loop_count} {'═' * 40}")

                    # Thermal Check zwischen Loops
                    temp, temp_status = self._thermal.get_status()
                    if temp_status in ("HOT", "CRITICAL"):
                        pause = self._thermal.get_recommended_pause_s()
                        print(f"\n   🌡️ Abkühlpause: {pause:.0f}s (geschätzt ~{temp:.0f}°C)")
                        self._arm.torque_off()
                        time.sleep(pause)
                        self._thermal.update_idle()
                        self._arm.torque_on()
                        time.sleep(0.5)

                    self.go_to_start()
                    time.sleep(0.5)
                    self.play_once()

                    # Nach jedem Loop: Emergency prüfen
                    if self._arm.is_emergency_stopped:
                        break

                    time.sleep(1.0)
            else:
                self.play_once()
        except KeyboardInterrupt:
            print("\n\n   ⏹ Abgebrochen!")
            if not self._arm.is_emergency_stopped:
                # Sanfter Stopp bei manuellem Abbruch
                GracefulStop.execute(self._arm_raw, self._arm.validator._last_commanded)

        self._cleanup()


    def _cleanup(self):
        """Aufräumen am Ende."""
        # Watchdog stoppen
        if self._watchdog:
            self._watchdog.stop()

        if self._arm and not self._arm.is_emergency_stopped:
            print("\n🔒 Torque bleibt AN")
            self._arm.torque_on()
            time.sleep(0.3)
        elif self._arm and self._arm.is_emergency_stopped:
            print("\n⚠️  Emergency Stop war aktiv - Arm ist schlaff")
            print("   Prüfe den Arm bevor du ihn wieder benutzt!")

        if self._arm_raw:
            self._arm_raw.close()

        # Finale Statistik
        if self._rate_limiter and self._rate_limiter.violations > 0:
            print(f"   Rate-Limiter Eingriffe gesamt: {self._rate_limiter.violations}")

        if self._viz_arm:
            self._viz_arm.visualizer.stop()

        print("✅ Fertig!\n")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Smooth Play Mode (Cubic Spline + Streaming)")
    p.add_argument("file", type=str, help=".roarm Datei zum Abspielen")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Geschwindigkeit (0.5=halb, 2.0=doppelt)")
    p.add_argument("--loop", action="store_true",
                   help="Endlos wiederholen")
    p.add_argument("--no-verify", action="store_true",
                   help="Positionsverifikation überspringen")
    p.add_argument("--hz", type=int, default=STREAM_HZ,
                   help=f"Stream-Frequenz in Hz (default: {STREAM_HZ})")
    p.add_argument("--offset", type=str, default=None,
                   help="Manueller Offset: 'b=0.5,s=-0.3,e=0.1,h=0.0'")
    p.add_argument("-V", "--visualize", action="store_true",
                   help="3D-Visualisierung des Arms während der Wiedergabe")
    args = p.parse_args()

    # Manuellen Offset parsen
    manual_offset = None
    if args.offset:
        manual_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        for part in args.offset.split(","):
            k, v = part.strip().split("=")
            manual_offset[k.strip()] = float(v.strip())

    player = SmoothPlayer(
        filepath=args.file,
        port=args.port,
        speed=args.speed,
        loop=args.loop,
        verify=not args.no_verify,
        manual_offset=manual_offset,
        stream_hz=args.hz,
        visualize=args.visualize,
    )
    player.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

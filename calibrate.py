#!/usr/bin/env python3
"""calibrate.py - RoArm-M2-S Kinematisches Kalibrierungsmodell

Workflow:
1. Roboter fährt N vordefinierte Posen an (mehrfach, parametrisierbar)
2. Zwischen jeder Pose fährt er in eine sichere UP-Position (Kollisionsvermeidung)
3. Wartet bis Arm stillsteht (aktives Polling)
4. Misst Servo-Feedback (oder User misst manuell)
5. Modell wird gefittet: Soll → Korrektur
6. Kalibrierungsdatei + Diagnostik wird gespeichert

Flags:
  --auto         Akzeptiert Servo-Werte automatisch (kein User-Input)
  --no-identify  Überspringt Gelenk-Identifikation
  --port PORT    Serieller Port (sonst auto-detect)
  --repeats N    Wie oft jede Pose angefahren wird (default: 3)
  --pose-set     Welches Posen-Set: 'standard', 'extended', 'minimal'
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "numpy",
#     "rich",
# ]
# ///

import os
import sys

from bootstrap import ensure_uv
ensure_uv()

import numpy as np
from pathlib import Path
import json
import time
import math
import serial
import serial.tools.list_ports
import threading
from rich.panel import Panel
from rich.table import Table
from rich import box

from robot import (
    RoArmConnection, find_arm_port, rad_to_deg,
    BAUDRATE,
)

from ui import (
    console, print_banner, print_section, print_step,
    calibration_progress, calibration_pose_table, calibration_summary,
    print_connection_status, print_success, print_warning,
    joint_table, print_position, print_info
)

# ============================================================
# KONFIGURATION
# ============================================================

JOINTS = ["b", "s", "e"]

# --- SAFE UP POSITION ---
# Diese Position wird zwischen JEDER Pose angefahren.
# Muss so gewählt sein, dass der Arm über allen Hindernissen auf dem Schreibtisch ist.
# s=0 und e=90 = Arm zeigt gerade nach oben/vorne, weit weg vom Tisch.
SAFE_UP_POSITION = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

# Geschwindigkeit für Safe-Up Bewegung
SAFE_UP_SPD = 25
SAFE_UP_ACC = 12

# --- MEHRFACH-MESSUNGEN ---
# Jede Pose wird REPEATS_PER_POSE mal angefahren (immer über UP-Position).
# Mehr Wiederholungen = bessere Kalibrierung (Repeatability wird gemittelt).
REPEATS_PER_POSE = 3

# --- POSEN-SETS ---
# Erweitert: Mehr Posen in allen Richtungen, aber NUR sichere Bereiche
# (kein ganz nach unten wo Boden/Widerstand ist)

# Grenzen des sicheren Arbeitsraums (parametrisiert)
SAFE_LIMITS = {
    "b_min": -90.0,   # Base links max
    "b_max": 90.0,    # Base rechts max
    "s_min": -25.0,   # weiter runter erlauben
    "s_max": 45.0,    # Shoulder hoch
    "e_min": 20.0,    # Ellbogen noch gestreckter
    "e_max": 150.0,   # Ellbogen weit
}

# Minimales Set (schnell, 8 Posen)
CALIBRATION_POSES_MINIMAL = [
    {"b": 0.0,   "s": 0.0,   "e": 90.0,  "h": 180.0},   # Home
    {"b": -45.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Links
    {"b": 45.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Rechts
    {"b": 0.0,   "s": 30.0,  "e": 90.0,  "h": 180.0},   # Schulter hoch
    {"b": 0.0,   "s": -10.0, "e": 90.0,  "h": 180.0},   # Schulter leicht runter
    {"b": 0.0,   "s": 0.0,   "e": 50.0,  "h": 180.0},   # Ellbogen eng
    {"b": 0.0,   "s": 0.0,   "e": 130.0, "h": 180.0},   # Ellbogen weit
    {"b": -30.0, "s": 20.0,  "e": 60.0,  "h": 180.0},   # Kombi
]

# Standard-Set (12 Posen, wie bisher)
CALIBRATION_POSES_STANDARD = [
    {"b": 0.0,   "s": 0.0,   "e": 90.0,  "h": 180.0},   # Home
    {"b": -45.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Links
    {"b": 45.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Rechts
    {"b": 0.0,   "s": 30.0,  "e": 90.0,  "h": 180.0},   # Schulter hoch
    {"b": 0.0,   "s": -10.0, "e": 90.0,  "h": 180.0},   # Schulter runter (sicher)
    {"b": 0.0,   "s": 0.0,   "e": 50.0,  "h": 180.0},   # Ellbogen eng
    {"b": 0.0,   "s": 0.0,   "e": 135.0, "h": 180.0},   # Ellbogen weit
    {"b": -30.0, "s": 20.0,  "e": 60.0,  "h": 180.0},   # Kombi 1
    {"b": 30.0,  "s": 20.0,  "e": 60.0,  "h": 180.0},   # Kombi 2
    {"b": -30.0, "s": -5.0,  "e": 120.0, "h": 180.0},   # Kombi 3
    {"b": 30.0,  "s": -5.0,  "e": 120.0, "h": 180.0},   # Kombi 4
    {"b": 0.0,   "s": 15.0,  "e": 70.0,  "h": 180.0},   # Zentrum
]

# Erweitertes Set (24 Posen, bessere Abdeckung des Arbeitsraums)
CALIBRATION_POSES_EXTENDED = [
    # === Boden-Berührung (links, mitte, rechts) ===
    {"b": -90.0, "s": -15.0, "e": 30.0,  "h": 180.0},   # Ganz links, Boden
    {"b": 0.0,   "s": -15.0, "e": 30.0,  "h": 180.0},   # Exakt Mitte, Boden
    {"b": 90.0,  "s": -15.0, "e": 30.0,  "h": 180.0},   # Ganz rechts, Boden

    # === Einzelachsen-Sweeps ===
    # Base sweep (Schulter/Ellbogen neutral)
    {"b": -90.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base ganz links
    {"b": -60.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base links
    {"b": -30.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base leicht links
    {"b": 0.0,   "s": 0.0,   "e": 90.0,  "h": 180.0},   # Home
    {"b": 30.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base leicht rechts
    {"b": 60.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base rechts
    {"b": 90.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Base ganz rechts

    # Shoulder sweep (Base/Ellbogen neutral)
    {"b": 0.0,   "s": -10.0, "e": 90.0,  "h": 180.0},   # Schulter leicht runter
    {"b": 0.0,   "s": 15.0,  "e": 90.0,  "h": 180.0},   # Schulter mittel
    {"b": 0.0,   "s": 30.0,  "e": 90.0,  "h": 180.0},   # Schulter hoch
    {"b": 0.0,   "s": 45.0,  "e": 90.0,  "h": 180.0},   # Schulter ganz hoch

    # Elbow sweep (Base/Schulter neutral)
    {"b": 0.0,   "s": 0.0,   "e": 40.0,  "h": 180.0},   # Ellbogen sehr eng
    {"b": 0.0,   "s": 0.0,   "e": 65.0,  "h": 180.0},   # Ellbogen eng
    {"b": 0.0,   "s": 0.0,   "e": 115.0, "h": 180.0},   # Ellbogen weit
    {"b": 0.0,   "s": 0.0,   "e": 140.0, "h": 180.0},   # Ellbogen sehr weit

    # === Kombinationen (Multi-Achsen) ===
    {"b": -45.0, "s": 25.0,  "e": 60.0,  "h": 180.0},   # Links-Hoch-Eng
    {"b": 45.0,  "s": 25.0,  "e": 60.0,  "h": 180.0},   # Rechts-Hoch-Eng
    {"b": -45.0, "s": 25.0,  "e": 120.0, "h": 180.0},   # Links-Hoch-Weit
    {"b": 45.0,  "s": 25.0,  "e": 120.0, "h": 180.0},   # Rechts-Hoch-Weit
    {"b": -45.0, "s": -5.0,  "e": 70.0,  "h": 180.0},   # Links-Runter-Eng
    {"b": 45.0,  "s": -5.0,  "e": 70.0,  "h": 180.0},   # Rechts-Runter-Eng
    {"b": -60.0, "s": 15.0,  "e": 80.0,  "h": 180.0},   # Weit-Links-Mittel
    {"b": 60.0,  "s": 15.0,  "e": 80.0,  "h": 180.0},   # Weit-Rechts-Mittel
    {"b": 0.0,   "s": 35.0,  "e": 50.0,  "h": 180.0},   # Zentrum-Hoch-Eng
]

# Mapping für CLI
POSE_SETS = {
    "minimal": CALIBRATION_POSES_MINIMAL,
    "standard": CALIBRATION_POSES_STANDARD,
    "extended": CALIBRATION_POSES_EXTENDED,
}


# ============================================================
# FEHLERMODELL: Polynom 2. Ordnung (nur b, s, e)
# ============================================================

class CalibrationModel:
    """
    Modelliert den Fehler jedes Gelenks als Polynom 2. Ordnung
    der Soll-Winkel der 3 echten Gelenke (b, s, e).

    Für Gelenk j:
      error_j = c0 + c1*b + c2*s + c3*e
                + c4*b² + c5*s² + c6*e²
                + c7*b*s + c8*b*e + c9*s*e

    = 10 Koeffizienten pro Gelenk → mindestens 10 Messpunkte nötig
    """

    def __init__(self):
        self.coefficients = {"b": None, "s": None, "e": None}
        self.is_fitted = False
        self.residuals = {}

    def _build_features(self, poses: list) -> np.ndarray:
        X = []
        for pose in poses:
            b, s, e = pose["b"], pose["s"], pose["e"]
            bn, sn, en = b / 90.0, s / 45.0, e / 90.0
            X.append([
                1.0,
                bn, sn, en,
                bn**2, sn**2, en**2,
                bn*sn, bn*en, sn*en,
            ])
        return np.array(X)

    def fit(self, commanded_poses: list, measured_errors: list):
        X = self._build_features(commanded_poses)
        for joint in JOINTS:
            y = np.array([err[joint] for err in measured_errors])
            lambda_reg = 0.01
            XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
            Xty = X.T @ y
            self.coefficients[joint] = np.linalg.solve(XtX, Xty)
            y_pred = X @ self.coefficients[joint]
            self.residuals[joint] = np.sqrt(np.mean((y - y_pred)**2))
        self.is_fitted = True
        return self.residuals

    def predict_correction(self, pose: dict) -> dict:
        if not self.is_fitted:
            return {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        X = self._build_features([pose])
        correction = {"h": 0.0}
        for joint in JOINTS:
            if self.coefficients[joint] is not None:
                correction[joint] = float(X[0] @ self.coefficients[joint])
            else:
                correction[joint] = 0.0
        return correction

    def save(self, filepath: str, diagnostics: dict = None):
        data = {
            "type": "polynomial_calibration_v2",
            "note": "h is gripper (EOAT), not calibrated",
            "joints_calibrated": JOINTS,
            "joints": {},
            "residuals": self.residuals,
        }
        if diagnostics:
            data["diagnostics"] = diagnostics
        for joint in JOINTS:
            if self.coefficients[joint] is not None:
                data["joints"][joint] = self.coefficients[joint].tolist()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Kalibrierung gespeichert: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CalibrationModel":
        with open(filepath, 'r') as f:
            data = json.load(f)
        model = cls()
        for joint in JOINTS:
            if joint in data.get("joints", {}):
                model.coefficients[joint] = np.array(data["joints"][joint])
        model.residuals = data.get("residuals", {})
        model.is_fitted = True
        return model

# ============================================================
# SAFE-UP BEWEGUNG (Kollisionsvermeidung)
# ============================================================

def move_to_safe_up(arm: RoArmConnection, current_pose: dict = None):
    """
    Fährt den Arm in die sichere UP-Position.
    
    Strategie:
    1. Zuerst Ellbogen auf 90° (Arm nach oben klappen)
    2. Dann Schulter auf 0° (Arm gerade)
    3. Dann Base auf 0° (Drehung zur Mitte)
    
    So wird vermieden, dass der Arm beim Drehen irgendwo anstößt.
    """
    up = SAFE_UP_POSITION

    if current_pose:
        # Schritt 1: Ellbogen sicher hochklappen (wichtigste Achse zuerst!)
        arm.move_to(
            current_pose["b"], current_pose["s"], up["e"], up["h"],
            spd=SAFE_UP_SPD, acc=SAFE_UP_ACC
        )
        arm.wait_until_settled(tolerance_deg=1.5, stable_count=3, timeout=8.0)

        # Schritt 2: Schulter in sichere Position
        arm.move_to(
            current_pose["b"], up["s"], up["e"], up["h"],
            spd=SAFE_UP_SPD, acc=SAFE_UP_ACC
        )
        arm.wait_until_settled(tolerance_deg=1.5, stable_count=3, timeout=8.0)

        # Schritt 3: Base zur Mitte
        arm.move_to(
            up["b"], up["s"], up["e"], up["h"],
            spd=SAFE_UP_SPD, acc=SAFE_UP_ACC
        )
        arm.wait_until_settled(tolerance_deg=1.0, stable_count=3, timeout=8.0)
    else:
        # Direkt zur UP-Position (wenn keine aktuelle Pose bekannt)
        arm.move_to(up["b"], up["s"], up["e"], up["h"], spd=SAFE_UP_SPD, acc=SAFE_UP_ACC)
        arm.wait_until_settled(tolerance_deg=1.0, stable_count=4, timeout=10.0)


def move_from_safe_up_to_pose(arm: RoArmConnection, target_pose: dict):
    """
    Fährt von der UP-Position sicher zur Zielpose.
    
    Strategie (umgekehrt):
    1. Base drehen (Arm ist oben, kann frei drehen)
    2. Schulter einstellen
    3. Ellbogen zum Ziel (letztes, da am nächsten an Hindernissen)
    """
    up = SAFE_UP_POSITION

    # Schritt 1: Base zum Ziel drehen (Arm ist oben = sicher)
    arm.move_to(
        target_pose["b"], up["s"], up["e"], up["h"],
        spd=SAFE_UP_SPD, acc=SAFE_UP_ACC
    )
    arm.wait_until_settled(tolerance_deg=1.5, stable_count=3, timeout=8.0)

    # Schritt 2: Schulter einstellen
    arm.move_to(
        target_pose["b"], target_pose["s"], up["e"], up["h"],
        spd=SAFE_UP_SPD, acc=SAFE_UP_ACC
    )
    arm.wait_until_settled(tolerance_deg=1.5, stable_count=3, timeout=8.0)

    # Schritt 3: Ellbogen zum Ziel (langsamer, da potenziell nah an Hindernissen)
    arm.move_to(
        target_pose["b"], target_pose["s"], target_pose["e"], target_pose["h"],
        spd=15, acc=8
    )
    arm.wait_until_settled(tolerance_deg=1.0, stable_count=3, timeout=8.0)

# ============================================================
# POSE-VALIDIERUNG
# ============================================================

def validate_pose(pose: dict) -> bool:
    """
    Prüft ob eine Pose innerhalb der sicheren Grenzen liegt.
    Gibt True zurück wenn sicher, False wenn außerhalb.
    """
    if pose["b"] < SAFE_LIMITS["b_min"] or pose["b"] > SAFE_LIMITS["b_max"]:
        return False
    if pose["s"] < SAFE_LIMITS["s_min"] or pose["s"] > SAFE_LIMITS["s_max"]:
        return False
    if pose["e"] < SAFE_LIMITS["e_min"] or pose["e"] > SAFE_LIMITS["e_max"]:
        return False
    return True


# ============================================================
# KALIBRIERUNGS-WORKFLOW
# ============================================================

def run_calibration(arm, poses=None, auto_accept: bool = False,
                    repeats: int = REPEATS_PER_POSE,
                    pose_set_name: str = "standard"):

    if poses is None:
        poses = POSE_SETS.get(pose_set_name, CALIBRATION_POSES_STANDARD)

    # Posen validieren
    valid_poses = []
    for i, pose in enumerate(poses):
        if validate_pose(pose):
            valid_poses.append(pose)
        else:
            print_warning(f"Pose {i+1} übersprungen (außerhalb sicherer Grenzen): "
                          f"b={pose['b']:.1f}° s={pose['s']:.1f}° e={pose['e']:.1f}°")
    poses = valid_poses

    if len(poses) < 10:
        print_warning(f"Nur {len(poses)} gültige Posen! Mindestens 10 empfohlen für guten Fit.")

    commanded = []
    errors = []
    diagnostics = {
        "settle_times_s": [],
        "overshoot_deg": [],
        "noise_std_deg": [],
        "per_pose": [],
        "repeats_per_pose": repeats,
        "pose_set": pose_set_name,
        "total_measurements": 0,
        "repeatability_per_pose": [],
    }

    total_measurements = len(poses) * repeats

    # ═══════════════════════════════════════════════════════════
    # BANNER (schon mit Rich)
    # ═══════════════════════════════════════════════════════════
    print_banner("calibrate", f"{len(poses)} Posen × {repeats} Wiederholungen = {total_measurements} Messungen")

    if not auto_accept:
        console.print(Panel(
            "[dim]Ablauf pro Messung:[/]\n"
            "  1. Arm fährt zur SAFE-UP Position (über Hindernisse)\n"
            "  2. Arm fährt zur Soll-Position (sicher von oben)\n"
            "  3. Wartet bis Arm stillsteht\n"
            "  4. Misst Servo-Feedback\n"
            "  5. Zurück zu SAFE-UP\n\n"
            f"  Jede Pose wird [bold]{repeats}×[/] angefahren für bessere Statistik.\n"
            "  Tipp: [bold cyan][w][/] = Gelenk wackeln lassen",
            title="Ablauf",
            border_style="dim",
            box=box.ROUNDED,
        ))

    if not auto_accept:
        input("\n  [ENTER] um zu starten...")
    else:
        print_info("Starte automatische Kalibrierung...")
        time.sleep(1.0)

    joints_identified = False
    total_start = time.time()
    measurement_count = 0

    # === Zuerst zur Safe-UP Position fahren ===
    print_info("Fahre zur Safe-UP Position...")
    arm.torque_on()
    time.sleep(0.2)
    move_to_safe_up(arm, current_pose=None)
    print_success("Safe-UP erreicht")

    # ═══════════════════════════════════════════════════════════
    # HAUPTSCHLEIFE MIT RICH PROGRESS
    # ═══════════════════════════════════════════════════════════
    progress = calibration_progress(total_measurements)

    with progress:
        task = progress.add_task(
            f"Kalibrierung ({pose_set_name})",
            total=total_measurements
        )

        for i, pose in enumerate(poses):
            pose_measurements = []

            for rep in range(repeats):
                measurement_count += 1

                # Progress-Bar updaten
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Pose {i+1}/{len(poses)} • Rep {rep+1}/{repeats}"
                    )
                )

                # --- SCHRITT 1: Safe-UP ---
                if rep > 0 or i > 0:
                    current = arm.read_position_deg()
                    if current:
                        move_to_safe_up(arm, current_pose=current)
                    else:
                        move_to_safe_up(arm, current_pose=None)

                # --- SCHRITT 2: Von Safe-UP zur Zielpose ---
                move_start = time.time()
                move_from_safe_up_to_pose(arm, pose)

                # --- SCHRITT 3: Präzisions-Nachfahrt ---
                arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
                result_precise = arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
                total_settle = time.time() - move_start
                timed_out = result_precise.get("timeout", False)

                diagnostics["settle_times_s"].append(total_settle)

                # Overshoot berechnen
                if result_precise["readings"] and result_precise["pos"]:
                    final = result_precise["pos"]
                    max_overshoot = 0.0
                    for reading in result_precise["readings"]:
                        for j in JOINTS:
                            overshoot = abs(reading[j] - final[j])
                            max_overshoot = max(max_overshoot, overshoot)
                    diagnostics["overshoot_deg"].append(round(max_overshoot, 3))
                else:
                    diagnostics["overshoot_deg"].append(0.0)

                # --- SCHRITT 4: Position auslesen (gemittelt) ---
                servo_avg = arm.read_position_averaged(n=10, interval=0.05)
                if servo_avg:
                    diagnostics["noise_std_deg"].append({
                        "b": servo_avg["b_std"],
                        "s": servo_avg["s_std"],
                        "e": servo_avg["e_std"],
                    })
                else:
                    servo_avg = {"b": pose["b"], "s": pose["s"], "e": pose["e"], "h": pose["h"],
                                 "b_std": 0, "s_std": 0, "e_std": 0, "n_samples": 0}
                    diagnostics["noise_std_deg"].append({"b": 0, "s": 0, "e": 0})

                # --- SCHRITT 5: Messung (Auto oder Manuell) ---
                if auto_accept:
                    measured = {j: servo_avg[j] for j in JOINTS}
                else:
                    # Bei manueller Eingabe: Progress pausieren
                    progress.stop()

                    # Zeige aktuelle Pose-Tabelle
                    console.print(calibration_pose_table(
                        pose_index=i,
                        total_poses=len(poses),
                        repeat=rep,
                        total_repeats=repeats,
                        commanded=pose,
                        measured={j: servo_avg[j] for j in JOINTS},
                    ))

                    console.print("\n  Miss jetzt die TATSÄCHLICHEN Winkel (nur b, s, e).", style="dim")
                    console.print("  (Leer = Servo-Wert | [w] = Gelenk wackeln)", style="dim")

                    measured = {}
                    for joint in JOINTS:
                        joint_names = {
                            "b": "BASE (Drehung links/rechts)",
                            "s": "SHOULDER (Schulter hoch/runter)",
                            "e": "ELBOW (Ellbogen auf/zu)",
                        }
                        default = servo_avg[joint]

                        while True:
                            val = input(f"    {joint} [{joint_names[joint]}] "
                                       f"(Soll={pose[joint]:.1f}°, Servo={default:.3f}°): ").strip()
                            if val == "":
                                measured[joint] = default
                                break
                            else:
                                try:
                                    measured[joint] = float(val)
                                    break
                                except ValueError:
                                    print_error("Ungültige Eingabe, nochmal...")

                    progress.start()  # Progress wieder starten

                # Fehler berechnen
                error = {j: measured[j] - pose[j] for j in JOINTS}
                pose_measurements.append({
                    "measured": measured.copy(),
                    "error": error.copy(),
                    "settle_time_s": total_settle,
                    "timed_out": timed_out,
                })

                # Pose-Diagnostik speichern
                diagnostics["per_pose"].append({
                    "pose_index": i,
                    "repeat": rep,
                    "commanded": {j: pose[j] for j in JOINTS},
                    "measured": measured,
                    "error": error,
                    "settle_time_s": total_settle,
                    "overshoot_deg": diagnostics["overshoot_deg"][-1],
                    "noise_std": diagnostics["noise_std_deg"][-1],
                    "timed_out": timed_out,
                })

            # === Nach allen Wiederholungen: Mittelwert für diese Pose ===
            avg_error = {}
            for j in JOINTS:
                errors_j = [m["error"][j] for m in pose_measurements]
                avg_error[j] = float(np.mean(errors_j))

            # Repeatability für diese Pose
            if repeats > 1:
                repeat_std = {}
                for j in JOINTS:
                    measured_vals = [m["measured"][j] for m in pose_measurements]
                    repeat_std[j] = float(np.std(measured_vals))
                diagnostics["repeatability_per_pose"].append({
                    "pose_index": i,
                    "commanded": {j: pose[j] for j in JOINTS},
                    "repeat_std_deg": repeat_std,
                    "n_repeats": repeats,
                })

            commanded.append(pose)
            errors.append(avg_error)

    # Progress ist jetzt fertig (with-Block verlassen)

    diagnostics["total_measurements"] = measurement_count

    # === Zurück zur Safe-UP Position am Ende ===
    print_info("Fahre zurück zu Safe-UP...")
    current = arm.read_position_deg()
    if current:
        move_to_safe_up(arm, current_pose=current)
    else:
        move_to_safe_up(arm, current_pose=None)

    # ═══════════════════════════════════════════════════════════
    # ZUSAMMENFASSUNG MIT RICH TABLES
    # ═══════════════════════════════════════════════════════════

    total_time = time.time() - total_start
    print_section("ERGEBNIS")

    # --- Diagnostik-Tabelle ---
    settle_arr = np.array(diagnostics["settle_times_s"])
    overshoot_arr = np.array(diagnostics["overshoot_deg"])
    noise_b = np.array([n["b"] for n in diagnostics["noise_std_deg"]])
    noise_s = np.array([n["s"] for n in diagnostics["noise_std_deg"]])
    noise_e = np.array([n["e"] for n in diagnostics["noise_std_deg"]])

    diag_table = Table(
        title="📊 Diagnostik",
        box=box.ROUNDED,
        border_style="cyan",
    )
    diag_table.add_column("Metrik", style="bold", width=30)
    diag_table.add_column("Min", justify="right", width=10)
    diag_table.add_column("Max", justify="right", width=10)
    diag_table.add_column("Mittel", justify="right", width=10)
    diag_table.add_column("σ", justify="right", width=10)

    diag_table.add_row(
        "Settle-Zeit [s]",
        f"{settle_arr.min():.2f}",
        f"{settle_arr.max():.2f}",
        f"{settle_arr.mean():.2f}",
        f"{settle_arr.std():.2f}",
    )
    diag_table.add_row(
        "Overshoot [°]",
        f"{overshoot_arr.min():.3f}",
        f"{overshoot_arr.max():.3f}",
        f"{overshoot_arr.mean():.3f}",
        f"{overshoot_arr.std():.3f}",
    )
    diag_table.add_row(
        "Rauschen b [°]",
        f"{noise_b.min():.4f}",
        f"{noise_b.max():.4f}",
        f"{noise_b.mean():.4f}",
        f"{noise_b.std():.4f}",
    )
    diag_table.add_row(
        "Rauschen s [°]",
        f"{noise_s.min():.4f}",
        f"{noise_s.max():.4f}",
        f"{noise_s.mean():.4f}",
        f"{noise_s.std():.4f}",
    )
    diag_table.add_row(
        "Rauschen e [°]",
        f"{noise_e.min():.4f}",
        f"{noise_e.max():.4f}",
        f"{noise_e.mean():.4f}",
        f"{noise_e.std():.4f}",
    )
    console.print(diag_table)

    # --- Repeatability-Tabelle (wenn > 1 Wiederholung) ---
    if diagnostics["repeatability_per_pose"]:
        all_rep_b = [r["repeat_std_deg"]["b"] for r in diagnostics["repeatability_per_pose"]]
        all_rep_s = [r["repeat_std_deg"]["s"] for r in diagnostics["repeatability_per_pose"]]
        all_rep_e = [r["repeat_std_deg"]["e"] for r in diagnostics["repeatability_per_pose"]]

        rep_table = Table(
            title=f"🔄 Repeatability (σ über {repeats} Wiederholungen)",
            box=box.ROUNDED,
            border_style="magenta",
        )
        rep_table.add_column("Gelenk", style="bold", width=10)
        rep_table.add_column("Mittel σ [°]", justify="right", width=14)
        rep_table.add_column("Max σ [°]", justify="right", width=12)
        rep_table.add_column("Qualität", justify="center", width=10)

        for j, vals in [("b", all_rep_b), ("s", all_rep_s), ("e", all_rep_e)]:
            mean_val = np.mean(vals)
            max_val = max(vals)
            quality = "✅" if mean_val < 0.1 else "⚠️" if mean_val < 0.3 else "❌"
            style = "green" if mean_val < 0.1 else "yellow" if mean_val < 0.3 else "red"
            rep_table.add_row(
                f"[joint.{j}]{j.upper()}[/]",
                f"[{style}]{mean_val:.4f}[/]",
                f"{max_val:.4f}",
                quality,
            )
        console.print(rep_table)

    # --- Positionsfehler-Tabelle (vor Kalibrierung) ---
    err_b = np.array([e["b"] for e in errors])
    err_s = np.array([e["s"] for e in errors])
    err_e = np.array([e["e"] for e in errors])

    err_table = Table(
        title="🎯 Positionsfehler (Soll vs. Ist, vor Kalibrierung)",
        box=box.ROUNDED,
        border_style="yellow",
    )
    err_table.add_column("Gelenk", style="bold", width=10)
    err_table.add_column("Mean [°]", justify="right", width=12)
    err_table.add_column("σ [°]", justify="right", width=10)
    err_table.add_column("Max |err| [°]", justify="right", width=14)

    for j, arr in [("b", err_b), ("s", err_s), ("e", err_e)]:
        err_table.add_row(
            f"[joint.{j}]{j.upper()}[/]",
            f"{arr.mean():+.3f}",
            f"{arr.std():.3f}",
            f"{np.abs(arr).max():.3f}",
        )
    console.print(err_table)

    # --- Repeatability-Test: Home nochmal anfahren ---
    print_info("Repeatability-Test (fahre Home nochmal an über Safe-UP)...")
    move_from_safe_up_to_pose(arm, poses[0])
    arm.move_to(poses[0]["b"], poses[0]["s"], poses[0]["e"], poses[0]["h"], spd=5, acc=3)
    arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
    repeat_pos = arm.read_position_averaged(n=10, interval=0.05)

    if repeat_pos:
        first_home = diagnostics["per_pose"][0]["measured"]
        repeat_err = {j: abs(repeat_pos[j] - first_home[j]) for j in JOINTS}
        console.print(f"  🔄 Home→...→Home: "
                      f"Δb={repeat_err['b']:.3f}° Δs={repeat_err['s']:.3f}° Δe={repeat_err['e']:.3f}°",
                      style="dim")
        diagnostics["repeatability_deg"] = repeat_err
    else:
        diagnostics["repeatability_deg"] = {"b": 0, "s": 0, "e": 0}

    # Diagnostik-Zusammenfassung in dict
    diagnostics["total_time_s"] = total_time
    diagnostics["avg_settle_time_s"] = float(settle_arr.mean())
    diagnostics["max_settle_time_s"] = float(settle_arr.max())
    diagnostics["avg_overshoot_deg"] = float(overshoot_arr.mean())
    diagnostics["avg_noise_std_deg"] = {
        "b": float(noise_b.mean()),
        "s": float(noise_s.mean()),
        "e": float(noise_e.mean()),
    }
    diagnostics["position_error_stats"] = {
        "b": {"mean": float(err_b.mean()), "std": float(err_b.std()), "max": float(np.abs(err_b).max())},
        "s": {"mean": float(err_s.mean()), "std": float(err_s.std()), "max": float(np.abs(err_s).max())},
        "e": {"mean": float(err_e.mean()), "std": float(err_e.std()), "max": float(np.abs(err_e).max())},
    }

    # ═══════════════════════════════════════════════════════════
    # MODELL FITTEN + ERGEBNIS MIT calibration_summary()
    # ═══════════════════════════════════════════════════════════

    console.print(f"\n")
    console.print(Panel(
        "[bold]Möchtest du zusätzlich manuelle Verifikationspunkte erfassen?[/]\n\n"
        "Das verbessert die Kalibrierung mit Real-World Ground Truth.\n"
        "Du ziehst den Arm an Positionen und sagst wo er wirklich ist.\n\n"
        "[dim]Empfohlen: 3-5 Punkte an kritischen Arbeitspositionen.[/]",
        title="Manuelle Verifikation?",
        border_style="magenta",
    ))
    
    if n_manual_override > 0:
        n_manual = n_manual_override
    else:
        # Interaktiv fragen (wie bisher)
        manual_input = input("  Anzahl Punkte (0 = überspringen, empfohlen: 5): ").strip()
        n_manual = 0
        try:
            n_manual = int(manual_input) if manual_input else 0
        except ValueError:
            n_manual = 0
    
    if n_manual > 0:
        manual_points = run_manual_verification(arm, n_points=n_manual)
        
        if manual_points:
            # Manuelle Punkte in die Kalibrierungsdaten integrieren
            old_count = len(commanded)
            commanded, errors = integrate_manual_points(
                commanded, errors, manual_points, weight=2.0
            )
            new_count = len(commanded)
            
            print_success(f"{new_count - old_count} manuelle Datenpunkte hinzugefügt "
                         f"(gewichtet ×2)")
            
            # In Diagnostik speichern
            diagnostics["manual_verification"] = {
                "n_points": len(manual_points),
                "points": manual_points,
                "weight": 2.0,
            }

    print_section("MODELL FITTEN")
    print_info(f"Fitte Polynom 2. Ordnung ({len(commanded)} Datenpunkte, "
               f"je gemittelt über {repeats} Messungen)...")

    model = CalibrationModel()
    residuals = model.fit(commanded, errors)

    # ✅ HIER die Rich-Zusammenfassung aus ui.py verwenden:
    calibration_summary(
        residuals=residuals,
        total_time=total_time,
        n_poses=len(poses),
        n_repeats=repeats,
    )

    # Speichern
    cal_path = Path("calibration") / "roarm_calibration.cal"
    cal_path.parent.mkdir(exist_ok=True)
    model.save(str(cal_path), diagnostics=diagnostics)

    diag_path = Path("calibration") / "roarm_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print_success(f"Diagnostik gespeichert: {diag_path}")

    # Zurück zu Safe-UP am Ende
    print_info("Fahre zurück zu Safe-UP...")
    current = arm.read_position_deg()
    if current:
        move_to_safe_up(arm, current_pose=current)
    else:
        move_to_safe_up(arm, current_pose=None)

    return model

# ============================================================
# MANUELLE VERIFIKATIONSPUNKTE (NEUER WORKFLOW)
# ============================================================

def run_manual_verification(arm: RoArmConnection, n_points: int = 5) -> list:
    """
    Neuer Workflow:
    1. Torque AUS → User bewegt Arm an gewünschte Position
    2. Enter → Position wird aufgezeichnet
    3. Arm fährt zurück zu Safe-UP / Startposition
    4. Arm fährt die aufgezeichnete Position nochmal an (Torque AN)
    5. "Drück Enter" → Torque wird AUS
    6. User korrigiert den Arm dahin wo er WIRKLICH sein sollte
    7. Enter → korrigierte Position wird gelesen
    8. Differenz = Kalibrierungsfehler (ohne dass User Winkel wissen muss)
    """
    manual_points = []

    print_section("MANUELLE VERIFIKATION (Replay + Korrektur)")
    console.print(Panel(
        f"[bold]Du wirst den Arm [cyan]{n_points}×[/cyan] manuell positionieren.[/]\n\n"
        "Ablauf pro Punkt:\n"
        "  1. Torque AUS → Ziehe den Arm an eine Position\n"
        "  2. [ENTER] → Position wird gespeichert\n"
        "  3. Arm fährt zurück zur Safe-UP Position\n"
        "  4. Arm fährt deine Position nochmal an (Torque AN)\n"
        "  5. [ENTER] → Torque wird AUS\n"
        "  6. Korrigiere den Arm dahin wo er WIRKLICH sein sollte\n"
        "  7. [ENTER] → Differenz wird gemessen\n",
        title="Manuelle Verifikation (Replay-Methode)",
        border_style="magenta",
        box=box.ROUNDED,
    ))

    for i in range(n_points):
        console.print(f"\n  [bold magenta]──── Punkt {i+1}/{n_points} ────[/]")

        # === NEU: Zuerst zur Safe-UP/Default-Position fahren ===
        console.print("  [dim]Fahre zur Ausgangsposition...[/]")
        arm.torque_on()
        time.sleep(0.2)
        current = arm.read_position_deg()
        if current:
            move_to_safe_up(arm, current_pose=current)
        else:
            move_to_safe_up(arm, current_pose=None)
        console.print("  [green]✓ Ausgangsposition erreicht[/]")

        # ═══════════════════════════════════════════════════════
        # SCHRITT 1: User positioniert den Arm frei
        # ═══════════════════════════════════════════════════════
        console.print("  [dim]Drücke [ENTER] um Torque zu deaktivieren...[/]")
        input()

        arm.torque_off()
        time.sleep(0.3)

        console.print("  [bold green]✋ Torque AUS[/] – Bewege den Arm an die gewünschte Position.")
        console.print("  [dim]Drücke [ENTER] wenn der Arm an der Zielposition ist...[/]")
        input()

        # ═══════════════════════════════════════════════════════
        # SCHRITT 2: Position aufzeichnen (= gewollte Position)
        # ═══════════════════════════════════════════════════════
        # Kurz Torque an für stabilen Read
        arm.torque_on()
        time.sleep(0.05)

        desired_pos = arm.read_position_averaged(n=10, interval=0.05)

        if not desired_pos:
            print_warning("Konnte Position nicht lesen, überspringe...")
            arm.torque_off_fast(exclude_gripper=True)
            continue

        console.print(f"\n  [bold]📍 Gewünschte Position aufgezeichnet:[/]")
        console.print(f"    b = [cyan]{desired_pos['b']:+8.3f}°[/]")
        console.print(f"    s = [cyan]{desired_pos['s']:+8.3f}°[/]")
        console.print(f"    e = [cyan]{desired_pos['e']:+8.3f}°[/]")

        # ═══════════════════════════════════════════════════════
        # SCHRITT 3: Zurück zur Safe-UP Position
        # ═══════════════════════════════════════════════════════
        console.print(f"\n  [dim]Fahre zurück zu Safe-UP...[/]")
        current = arm.read_position_deg()
        if current:
            move_to_safe_up(arm, current_pose=current)
        else:
            move_to_safe_up(arm, current_pose=None)

        time.sleep(0.5)

        # ═══════════════════════════════════════════════════════
        # SCHRITT 4: Arm fährt die Position nochmal an
        # ═══════════════════════════════════════════════════════
        console.print(f"  [dim]Fahre die aufgezeichnete Position nochmal an...[/]")

        target_pose = {
            "b": desired_pos["b"],
            "s": desired_pos["s"],
            "e": desired_pos["e"],
            "h": desired_pos.get("h", 180.0),
        }

        move_from_safe_up_to_pose(arm, target_pose)

        # Präzisions-Nachfahrt
        arm.move_to(target_pose["b"], target_pose["s"], target_pose["e"],
                    target_pose["h"], spd=5, acc=3)
        arm.wait_until_settled(tolerance_deg=0.3, stable_count=5, timeout=10.0)

        # Messen wo der Arm gelandet ist (= Replay-Position)
        replay_pos = arm.read_position_averaged(n=10, interval=0.05)
        if not replay_pos:
            print_warning("Konnte Replay-Position nicht lesen, überspringe...")
            continue

        console.print(f"\n  [bold]🎯 Arm ist angekommen bei:[/]")
        console.print(f"    b = [cyan]{replay_pos['b']:+8.3f}°[/]")
        console.print(f"    s = [cyan]{replay_pos['s']:+8.3f}°[/]")
        console.print(f"    e = [cyan]{replay_pos['e']:+8.3f}°[/]")

        # ═══════════════════════════════════════════════════════
        # SCHRITT 5: User korrigiert den Arm
        # ═══════════════════════════════════════════════════════
        console.print(f"\n  [bold yellow]👉 Drücke [ENTER] → Torque wird AUS[/]")
        console.print(f"  [dim]Dann korrigiere den Arm dahin wo er WIRKLICH sein sollte.[/]")
        input()

        arm.torque_off_fast(exclude_gripper=True)
        time.sleep(0.3)

        console.print("  [bold green]✋ Torque AUS[/] – Korrigiere den Arm jetzt!")
        console.print("  [dim]Drücke [ENTER] wenn der Arm an der RICHTIGEN Position ist...[/]")
        input()

        # ═══════════════════════════════════════════════════════
        # SCHRITT 6: Korrigierte Position lesen
        # ═══════════════════════════════════════════════════════
        arm.torque_on()
        time.sleep(0.05)

        corrected_pos = arm.read_position_averaged(n=10, interval=0.05)

        if not corrected_pos:
            print_warning("Konnte korrigierte Position nicht lesen, überspringe...")
            arm.torque_off_fast(exclude_gripper=True)
            continue

        # ═══════════════════════════════════════════════════════
        # SCHRITT 7: Fehler berechnen und anzeigen
        # ═══════════════════════════════════════════════════════
        # Fehler = wo der Arm war (replay) vs. wo er sein SOLLTE (korrigiert)
        # error = replay - corrected (= wie weit der Arm daneben lag)
        error = {j: replay_pos[j] - corrected_pos[j] for j in JOINTS}

        console.print(f"\n  [bold]📊 Ergebnis:[/]")
        console.print(f"    Replay-Position:    b={replay_pos['b']:+.3f}°  "
                      f"s={replay_pos['s']:+.3f}°  e={replay_pos['e']:+.3f}°")
        console.print(f"    Korrigierte Pos:    b={corrected_pos['b']:+.3f}°  "
                      f"s={corrected_pos['s']:+.3f}°  e={corrected_pos['e']:+.3f}°")

        for j in JOINTS:
            color = "green" if abs(error[j]) < 0.5 else "yellow" if abs(error[j]) < 2.0 else "red"
            console.print(f"    Δ{j} = [{color}]{error[j]:+.3f}°[/]  "
                          f"(Arm lag {abs(error[j]):.3f}° daneben)")

        manual_points.append({
            "desired_position": {j: desired_pos[j] for j in JOINTS},
            "replay_position": {j: replay_pos[j] for j in JOINTS},
            "corrected_position": {j: corrected_pos[j] for j in JOINTS},
            "error": error,
            "replay_std": {j: replay_pos.get(f"{j}_std", 0) for j in JOINTS},
            "corrected_std": {j: corrected_pos.get(f"{j}_std", 0) for j in JOINTS},
        })

        # Torque aus für nächsten Punkt (oder an lassen am Ende)
        if i < n_points - 1:
            arm.torque_off_fast(exclude_gripper=True)
            time.sleep(0.2)

    # Am Ende Torque an
    arm.torque_on()
    time.sleep(0.3)

    console.print(f"\n  [bold green]✅ {len(manual_points)} manuelle Punkte erfasst![/]")

    return manual_points


def integrate_manual_points(commanded: list, errors: list,
                            manual_points: list, weight: float = 2.0):
    """
    Integriert manuelle Verifikationspunkte in die Kalibrierungsdaten.

    Bei der neuen Methode:
    - "commanded" = die gewünschte Position (wo der User den Arm hinbewegt hat)
    - "error" = replay_position - corrected_position
      (= wie weit der Arm daneben lag als er die Position nochmal anfuhr)
    """
    new_commanded = list(commanded)
    new_errors = list(errors)

    for point in manual_points:
        # Die "commanded" Position ist die gewünschte Position
        pose = {j: point["desired_position"][j] for j in JOINTS}
        pose["h"] = 180.0

        # Der Fehler ist: wo der Arm gelandet ist - wo er sein sollte
        error = point["error"]

        # Mehrfach einfügen für höhere Gewichtung
        n_copies = max(1, int(weight))
        for _ in range(n_copies):
            new_commanded.append(pose)
            new_errors.append(error)

    return new_commanded, new_errors

# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Kalibrierung")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--auto", action="store_true",
                   help="Servo-Werte automatisch akzeptieren (kein User-Input)")
    p.add_argument("--no-identify", action="store_true",
                   help="Gelenk-Identifikation überspringen")
    p.add_argument("--repeats", type=int, default=REPEATS_PER_POSE,
                   help=f"Wiederholungen pro Pose (default: {REPEATS_PER_POSE})")
    p.add_argument("--pose-set", type=str, default="standard",
                   choices=list(POSE_SETS.keys()),
                   help="Posen-Set: minimal (8), standard (12), extended (24)")
    p.add_argument("--manual-points", type=int, default=0,
                   help="Anzahl manueller Verifikationspunkte (default: 0 = interaktiv fragen)")
    p.add_argument("--manual-only", action="store_true",
                   help="Nur manuelle Verifikation, automatische Posen überspringen")
    args = p.parse_args()

    port = args.port or find_arm_port()
    if port is None:
        print("❌ Kein serieller Port gefunden!")
        sys.exit(1)

    print(f"🔌 Verbinde mit {port}...")
    try:
        arm = RoArmConnection(port)
        print(f"   ✔ Verbunden")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        sys.exit(1)

    try:
        if args.manual_only:
            # ═══════════════════════════════════════════════════════
            # NUR MANUELLE VERIFIKATION (kein automatischer Posen-Durchlauf)
            # ═══════════════════════════════════════════════════════
            print_banner("calibrate", "Nur manuelle Verifikation (Replay + Korrektur)")

            n_manual = args.manual_points if args.manual_points > 0 else 5

            arm.torque_on()
            time.sleep(0.2)

            # Zur Safe-UP fahren
            print_info("Fahre zur Safe-UP Position...")
            move_to_safe_up(arm, current_pose=None)
            print_success("Safe-UP erreicht")

            manual_points = run_manual_verification(arm, n_points=n_manual)

            if manual_points:
                # Modell nur aus manuellen Punkten fitten
                commanded = []
                errors = []
                commanded, errors = integrate_manual_points(
                    commanded, errors, manual_points, weight=1.0
                )

                if len(commanded) >= 3:
                    print_section("MODELL FITTEN")
                    print_info(f"Fitte Polynom aus {len(commanded)} manuellen Datenpunkten...")

                    model = CalibrationModel()
                    residuals = model.fit(commanded, errors)

                    calibration_summary(
                        residuals=residuals,
                        total_time=0,
                        n_poses=0,
                        n_repeats=0,
                    )

                    cal_path = Path("calibration") / "roarm_calibration.cal"
                    cal_path.parent.mkdir(exist_ok=True)
                    model.save(str(cal_path), diagnostics={
                        "mode": "manual_only",
                        "n_manual_points": len(manual_points),
                        "manual_points": manual_points,
                    })
                else:
                    print_warning(
                        f"Nur {len(commanded)} Datenpunkte – mindestens 10 nötig für Polynom-Fit.\n"
                        f"   Brauche mindestens 3 manuelle Punkte (mit weight=2.0) oder 10 (mit weight=1.0)."
                    )

            # Zurück zu Safe-UP
            print_info("Fahre zurück zu Safe-UP...")
            current = arm.read_position_deg()
            if current:
                move_to_safe_up(arm, current_pose=current)
            else:
                move_to_safe_up(arm, current_pose=None)

        else:
            # ═══════════════════════════════════════════════════════
            # NORMALER WORKFLOW (automatisch + optional manuell)
            # ═══════════════════════════════════════════════════════
            model = run_calibration(
                arm,
                auto_accept=args.auto,
                repeats=args.repeats,
                pose_set_name=args.pose_set,
                n_manual_override=args.manual_points,  # NEU: Übergabe
            )

        print(f"\n✅ Kalibrierung abgeschlossen!")
        print(f"   Datei: calibration/roarm_calibration.cal")
        print(f"   Wird automatisch von play.py geladen.")
    except KeyboardInterrupt:
        print("\n\n   ⏹ Abgebrochen!")
    finally:
        arm.torque_on()
        time.sleep(0.3)
        arm.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("You exited with CTRL-C")

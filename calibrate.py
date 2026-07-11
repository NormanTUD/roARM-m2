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
# ]
# ///

import os
import sys

def _ensure_uv():
    if os.environ.get("_UV_SAFE_ENV") == "1":
        return
    os.environ["_UV_SAFE_ENV"] = "1"
    from datetime import datetime, timedelta, timezone
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
    except FileNotFoundError:
        print("uv nicht installiert.")
        sys.exit(1)

_ensure_uv()

import numpy as np
from pathlib import Path
import json
import time
import math
import serial
import serial.tools.list_ports
import threading

# ============================================================
# KONFIGURATION
# ============================================================

BAUDRATE = 115200
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
    "s_min": -15.0,   # Shoulder runter (NICHT weiter, da Tisch!)
    "s_max": 45.0,    # Shoulder hoch
    "e_min": 30.0,    # Ellbogen eng (NICHT weiter, Kollision!)
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
# ARM-VERBINDUNG
# ============================================================

def find_arm_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(x in desc for x in ["usb", "ch340", "cp210", "ftdi"]):
            return p.device
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    if ports:
        return ports[0].device
    return None


def rad_to_deg(rad: float) -> float:
    return rad * (180.0 / math.pi)


class RoArmConnection:
    def __init__(self, port: str, baudrate: int = BAUDRATE):
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=0.1)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def send_cmd(self, cmd: dict) -> str:
        with self._lock:
            self._ser.reset_input_buffer()
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.01)
            response = ""
            deadline = time.time() + 0.2
            while time.time() < deadline:
                if self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        response = line
                        if '"T":1051' in line or '"b"' in line:
                            return line
                else:
                    time.sleep(0.005)
            return response

    def read_position_raw(self) -> dict:
        resp = self.send_cmd({"T": 105})
        if not resp:
            return None
        try:
            start = resp.find('{')
            end = resp.rfind('}')
            if start >= 0 and end > start:
                data = json.loads(resp[start:end+1])
                if "b" in data:
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def read_position_deg(self) -> dict:
        raw = self.read_position_raw()
        if raw is None:
            return None
        return {
            "b": round(rad_to_deg(raw["b"]), 2),
            "s": round(rad_to_deg(raw["s"]), 2),
            "e": round(rad_to_deg(raw["e"]), 2),
            "h": round(rad_to_deg(raw.get("t", raw.get("h", 0))), 2),
        }

    def move_to(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                spd: int = 20, acc: int = 10):
        cmd = {
            "T": 122,
            "b": round(b_deg, 2),
            "s": round(s_deg, 2),
            "e": round(e_deg, 2),
            "h": round(h_deg, 2),
            "spd": spd,
            "acc": acc,
        }
        self.send_cmd(cmd)

    def wait_until_settled(self, tolerance_deg: float = 0.3, stable_count: int = 5,
                           poll_interval: float = 0.15, timeout: float = 15.0) -> dict:
        """
        Wartet bis der Arm stillsteht.
        Gibt dict zurück: {"pos": final_pos, "settle_time_s": float, "readings": list}
        """
        stable = 0
        last_pos = None
        start = time.time()
        all_readings = []

        while time.time() - start < timeout:
            pos = self.read_position_deg()
            if pos is None:
                time.sleep(poll_interval)
                continue

            all_readings.append({"t": time.time() - start, **pos})

            if last_pos is not None:
                max_delta = max(
                    abs(pos["b"] - last_pos["b"]),
                    abs(pos["s"] - last_pos["s"]),
                    abs(pos["e"] - last_pos["e"]),
                )
                if max_delta < tolerance_deg:
                    stable += 1
                    if stable >= stable_count:
                        settle_time = time.time() - start
                        return {
                            "pos": pos,
                            "settle_time_s": settle_time,
                            "readings": all_readings,
                        }
                else:
                    stable = 0

            last_pos = pos
            time.sleep(poll_interval)

        # Timeout
        settle_time = time.time() - start
        return {
            "pos": last_pos,
            "settle_time_s": settle_time,
            "readings": all_readings,
            "timeout": True,
        }

    def read_position_averaged(self, n: int = 8, interval: float = 0.04) -> dict:
        """Liest n Positionen und gibt den Mittelwert zurück."""
        readings = []
        for _ in range(n):
            pos = self.read_position_deg()
            if pos:
                readings.append(pos)
            time.sleep(interval)

        if not readings:
            return None

        avg = {}
        for joint in ["b", "s", "e", "h"]:
            values = [r[joint] for r in readings]
            avg[joint] = round(np.mean(values), 3)
            avg[f"{joint}_std"] = round(np.std(values), 4)
        avg["n_samples"] = len(readings)
        return avg

    def torque_off(self):
        self.send_cmd({"T": 210, "cmd": 0})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 0})
            time.sleep(0.02)

    def torque_on(self):
        self.send_cmd({"T": 210, "cmd": 1})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 1})
            time.sleep(0.02)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


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
# GELENK-IDENTIFIKATION
# ============================================================

def identify_joint(arm, joint: str, current_pose: dict):
    WIGGLE_AMOUNT = 15.0
    WIGGLE_SPD = 20
    WIGGLE_ACC = 10

    joint_names = {
        "b": "BASE (Drehung links/rechts)",
        "s": "SHOULDER (Schulter hoch/runter)",
        "e": "ELBOW (Ellbogen auf/zu)",
        "h": "HAND/GRIPPER (Greifer auf/zu)",
    }

    print(f"\n  👁️  Zeige Gelenk: {joint} = {joint_names[joint]}")

    if joint == "h":
        print(f"      Gripper öffnet und schließt...")
        arm.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})
        time.sleep(1.0)
        arm.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})
        time.sleep(1.0)
        arm.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})
        time.sleep(0.8)
        arm.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})
        time.sleep(0.5)
    else:
        print(f"      Bewege jetzt ±{WIGGLE_AMOUNT}° hin und her...")
        pose_plus = current_pose.copy()
        pose_minus = current_pose.copy()
        pose_plus[joint] = current_pose[joint] + WIGGLE_AMOUNT
        pose_minus[joint] = current_pose[joint] - WIGGLE_AMOUNT

        arm.move_to(pose_plus["b"], pose_plus["s"], pose_plus["e"], pose_plus["h"],
                    spd=WIGGLE_SPD, acc=WIGGLE_ACC)
        arm.wait_until_settled()

        arm.move_to(pose_minus["b"], pose_minus["s"], pose_minus["e"], pose_minus["h"],
                    spd=WIGGLE_SPD, acc=WIGGLE_ACC)
        arm.wait_until_settled()

        arm.move_to(pose_plus["b"], pose_plus["s"], pose_plus["e"], pose_plus["h"],
                    spd=WIGGLE_SPD, acc=WIGGLE_ACC)
        arm.wait_until_settled()

        arm.move_to(current_pose["b"], current_pose["s"], current_pose["e"], current_pose["h"],
                    spd=WIGGLE_SPD, acc=WIGGLE_ACC)
        arm.wait_until_settled()

    print(f"      ✅ Das war Gelenk '{joint}'")


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
                    skip_identify: bool = False, repeats: int = REPEATS_PER_POSE,
                    pose_set_name: str = "standard"):
    """
    Kalibrierungs-Workflow mit:
    - Mehrfach-Messungen pro Pose (parametrisierbar)
    - Safe-UP zwischen jeder Pose (Kollisionsvermeidung)
    - Erweitertes Posen-Set
    
    auto_accept: True = Servo-Werte automatisch akzeptieren (kein User-Input)
    skip_identify: True = Gelenk-Identifikation überspringen
    repeats: Wie oft jede Pose angefahren wird
    pose_set_name: 'minimal', 'standard', 'extended'
    """
    if poses is None:
        poses = POSE_SETS.get(pose_set_name, CALIBRATION_POSES_STANDARD)

    # Posen validieren
    valid_poses = []
    for i, pose in enumerate(poses):
        if validate_pose(pose):
            valid_poses.append(pose)
        else:
            print(f"  ⚠️  Pose {i+1} übersprungen (außerhalb sicherer Grenzen): "
                  f"b={pose['b']:.1f}° s={pose['s']:.1f}° e={pose['e']:.1f}°")
    poses = valid_poses

    if len(poses) < 10:
        print(f"  ⚠️  Nur {len(poses)} gültige Posen! Mindestens 10 empfohlen für guten Fit.")

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

    print(f"\n{'='*60}")
    print(f"  KALIBRIERUNG - {len(poses)} Posen × {repeats} Wiederholungen = {total_measurements} Messungen")
    print(f"  Kalibriert: b (Base), s (Shoulder), e (Elbow)")
    print(f"  Modus: {'AUTO (Servo-Werte)' if auto_accept else 'MANUELL (User misst)'}")
    print(f"  Posen-Set: {pose_set_name}")
    print(f"  Safe-UP Position: b={SAFE_UP_POSITION['b']:.0f}° s={SAFE_UP_POSITION['s']:.0f}° "
          f"e={SAFE_UP_POSITION['e']:.0f}°")
    print(f"{'='*60}")

    if not auto_accept:
        print(f"\n  Ablauf pro Messung:")
        print(f"  1. Arm fährt zur SAFE-UP Position (über Hindernisse)")
        print(f"  2. Arm fährt zur Soll-Position (sicher von oben)")
        print(f"  3. Wartet bis Arm stillsteht")
        print(f"  4. Misst Servo-Feedback")
        print(f"  5. Zurück zu SAFE-UP")
        print(f"\n  Jede Pose wird {repeats}× angefahren für bessere Statistik.")
        print(f"\n  Tipp: [w] = Gelenk wackeln lassen")

    if not skip_identify and not auto_accept:
        print(f"\n  Soll ich bei der ersten Pose alle Gelenke einzeln bewegen")
        print(f"  damit du siehst welches welches ist? (j/n)")
        show_joints = input("  > ").strip().lower() != 'n'
    else:
        show_joints = False

    if not auto_accept:
        input("\n  [ENTER] um zu starten...")
    else:
        print(f"\n  Starte automatische Kalibrierung...\n")
        time.sleep(1.0)

    joints_identified = False
    total_start = time.time()
    measurement_count = 0

    # === Zuerst zur Safe-UP Position fahren ===
    print(f"\n  🏠 Fahre zur Safe-UP Position...")
    arm.torque_on()
    time.sleep(0.2)
    move_to_safe_up(arm, current_pose=None)
    print(f"  ✅ Safe-UP erreicht")

    # === Hauptschleife: Jede Pose × Wiederholungen ===
    for i, pose in enumerate(poses):
        pose_measurements = []  # Alle Messungen für diese Pose

        for rep in range(repeats):
            measurement_count += 1

            print(f"\n{'─'*60}")
            print(f"  Pose {i+1}/{len(poses)} | Wiederholung {rep+1}/{repeats} "
                  f"| Messung {measurement_count}/{total_measurements}")
            print(f"  Soll: b={pose['b']:.1f}° s={pose['s']:.1f}° e={pose['e']:.1f}°")

            # --- SCHRITT 1: Safe-UP (falls nicht schon dort) ---
            if rep > 0 or i > 0:
                print(f"  ⬆️  Fahre zu Safe-UP...", end="", flush=True)
                # Aktuelle Position lesen für sichere Rückfahrt
                current = arm.read_position_deg()
                if current:
                    move_to_safe_up(arm, current_pose=current)
                else:
                    move_to_safe_up(arm, current_pose=None)
                print(f" ✅")

            # --- SCHRITT 2: Von Safe-UP zur Zielpose ---
            print(f"  ⬇️  Fahre zur Pose...", end="", flush=True)
            move_start = time.time()
            move_from_safe_up_to_pose(arm, pose)
            print(f" angekommen", flush=True)

            # --- SCHRITT 3: Präzisions-Nachfahrt ---
            arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
            print(f"  ⏳ Präzisions-Nachfahrt...", end="", flush=True)
            result_precise = arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
            total_settle = time.time() - move_start
            timed_out = result_precise.get("timeout", False)
            print(f" {'⚠️ TIMEOUT' if timed_out else '✅ still'} ({total_settle:.1f}s)", flush=True)

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

            # Gelenk-Identifikation (nur bei allererster Messung)
            if show_joints and not joints_identified and i == 0 and rep == 0:
                print(f"\n  🔍 GELENK-IDENTIFIKATION:")
                input(f"     [ENTER] um zu starten...")
                for joint in ["b", "s", "e", "h"]:
                    identify_joint(arm, joint, pose)
                    time.sleep(0.3)
                joints_identified = True
                print(f"\n  ✅ Alle Gelenke identifiziert!")

                # Nochmal zur Pose fahren
                arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=10, acc=5)
                arm.wait_until_settled()
                arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
                arm.wait_until_settled()

            # --- SCHRITT 4: Position auslesen (gemittelt) ---
            servo_avg = arm.read_position_averaged(n=10, interval=0.05)
            if servo_avg:
                print(f"  📊 Servo (Mittel ×{servo_avg['n_samples']}): "
                      f"b={servo_avg['b']:.3f}° s={servo_avg['s']:.3f}° e={servo_avg['e']:.3f}°")
                print(f"     Rauschen (σ): "
                      f"b={servo_avg['b_std']:.4f}° s={servo_avg['s_std']:.4f}° e={servo_avg['e_std']:.4f}°")
                diagnostics["noise_std_deg"].append({
                    "b": servo_avg["b_std"],
                    "s": servo_avg["s_std"],
                    "e": servo_avg["e_std"],
                })
            else:
                servo_avg = {"b": pose["b"], "s": pose["s"], "e": pose["e"], "h": pose["h"],
                             "b_std": 0, "s_std": 0, "e_std": 0}
                print(f"  ⚠️ Konnte Position nicht lesen, verwende Soll-Werte")
                diagnostics["noise_std_deg"].append({"b": 0, "s": 0, "e": 0})

            # --- SCHRITT 5: Messung (Auto oder Manuell) ---
            if auto_accept:
                measured = {j: servo_avg[j] for j in JOINTS}
            else:
                print(f"\n  Miss jetzt die TATSÄCHLICHEN Winkel (nur b, s, e).")
                print(f"  (Leer = Servo-Wert | [w] = Gelenk wackeln)")

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
                        if val.lower() == 'w':
                            identify_joint(arm, joint, pose)
                            arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
                            arm.wait_until_settled()
                            servo_avg = arm.read_position_averaged(n=10, interval=0.05)
                            if servo_avg:
                                default = servo_avg[joint]
                            continue
                        elif val == "":
                            measured[joint] = default
                            break
                        else:
                            try:
                                measured[joint] = float(val)
                                break
                            except ValueError:
                                print(f"      ❌ Ungültige Eingabe, nochmal...")

            # Fehler berechnen
            error = {j: measured[j] - pose[j] for j in JOINTS}
            pose_measurements.append({
                "measured": measured.copy(),
                "error": error.copy(),
                "settle_time_s": total_settle,
                "timed_out": timed_out,
            })

            # Pose-Diagnostik speichern
            pose_diag = {
                "pose_index": i,
                "repeat": rep,
                "commanded": {j: pose[j] for j in JOINTS},
                "measured": measured,
                "error": error,
                "settle_time_s": total_settle,
                "overshoot_deg": diagnostics["overshoot_deg"][-1],
                "noise_std": diagnostics["noise_std_deg"][-1],
                "timed_out": timed_out,
            }
            diagnostics["per_pose"].append(pose_diag)

            print(f"  → Fehler: Δb={error['b']:+.3f}° Δs={error['s']:+.3f}° Δe={error['e']:+.3f}°")

        # === Nach allen Wiederholungen: Mittelwert für diese Pose ===
        avg_error = {}
        for j in JOINTS:
            errors_j = [m["error"][j] for m in pose_measurements]
            avg_error[j] = float(np.mean(errors_j))

        # Repeatability für diese Pose (Standardabweichung über Wiederholungen)
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
            print(f"\n  📈 Pose {i+1} Repeatability (σ über {repeats} Wiederholungen):")
            print(f"     b={repeat_std['b']:.4f}° s={repeat_std['s']:.4f}° e={repeat_std['e']:.4f}°")
            print(f"     Mittlerer Fehler: Δb={avg_error['b']:+.3f}° "
                  f"Δs={avg_error['s']:+.3f}° Δe={avg_error['e']:+.3f}°")

        # Gemittelten Fehler für das Modell verwenden
        commanded.append(pose)
        errors.append(avg_error)

    diagnostics["total_measurements"] = measurement_count

    # === Zurück zur Safe-UP Position am Ende ===
    print(f"\n  🏠 Fahre zurück zu Safe-UP...")
    current = arm.read_position_deg()
    if current:
        move_to_safe_up(arm, current_pose=current)
    else:
        move_to_safe_up(arm, current_pose=None)

    # ============================================================
    # ZUSAMMENFASSUNG & MODELL FITTEN
    # ============================================================

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ERGEBNIS")
    print(f"{'='*60}")

    print(f"\n  ⏱️  Gesamtzeit: {total_time:.1f}s ({total_time/total_measurements:.1f}s pro Messung)")
    print(f"      {len(poses)} Posen × {repeats} Wiederholungen = {total_measurements} Messungen")

    # Settle-Time Statistik
    settle_arr = np.array(diagnostics["settle_times_s"])
    print(f"\n  🔍 Settle-Zeiten:")
    print(f"     Min: {settle_arr.min():.2f}s  Max: {settle_arr.max():.2f}s  "
          f"Mittel: {settle_arr.mean():.2f}s  σ: {settle_arr.std():.2f}s")

    # Overshoot Statistik
    overshoot_arr = np.array(diagnostics["overshoot_deg"])
    print(f"\n  📈 Overshoot (max Abweichung während Bewegung):")
    print(f"     Min: {overshoot_arr.min():.3f}°  Max: {overshoot_arr.max():.3f}°  "
          f"Mittel: {overshoot_arr.mean():.3f}°")

    # Rauschen Statistik
    noise_b = np.array([n["b"] for n in diagnostics["noise_std_deg"]])
    noise_s = np.array([n["s"] for n in diagnostics["noise_std_deg"]])
    noise_e = np.array([n["e"] for n in diagnostics["noise_std_deg"]])
    print(f"\n  📉 Servo-Rauschen (σ im Stillstand):")
    print(f"     b: {noise_b.mean():.4f}°  s: {noise_s.mean():.4f}°  e: {noise_e.mean():.4f}°")

    # Repeatability Statistik (über alle Posen)
    if diagnostics["repeatability_per_pose"]:
        all_rep_b = [r["repeat_std_deg"]["b"] for r in diagnostics["repeatability_per_pose"]]
        all_rep_s = [r["repeat_std_deg"]["s"] for r in diagnostics["repeatability_per_pose"]]
        all_rep_e = [r["repeat_std_deg"]["e"] for r in diagnostics["repeatability_per_pose"]]
        print(f"\n  🔄 Repeatability (σ über {repeats} Wiederholungen, gemittelt über alle Posen):")
        print(f"     b: {np.mean(all_rep_b):.4f}°  s: {np.mean(all_rep_s):.4f}°  e: {np.mean(all_rep_e):.4f}°")
        print(f"     Max: b={max(all_rep_b):.4f}° s={max(all_rep_s):.4f}° e={max(all_rep_e):.4f}°")

    # Fehler-Statistik (vor Kalibrierung) - basierend auf gemittelten Fehlern
    err_b = np.array([e["b"] for e in errors])
    err_s = np.array([e["s"] for e in errors])
    err_e = np.array([e["e"] for e in errors])
    print(f"\n  🎯 Positionsfehler (Soll vs. Ist, gemittelt über Wiederholungen):")
    print(f"     b: mean={err_b.mean():+.3f}° σ={err_b.std():.3f}° "
          f"max={np.abs(err_b).max():.3f}°")
    print(f"     s: mean={err_s.mean():+.3f}° σ={err_s.std():.3f}° "
          f"max={np.abs(err_s).max():.3f}°")
    print(f"     e: mean={err_e.mean():+.3f}° σ={err_e.std():.3f}° "
          f"max={np.abs(err_e).max():.3f}°")

    # Repeatability-Test: Home nochmal anfahren (über Safe-UP!)
    print(f"\n  🔄 Repeatability-Test (fahre Home nochmal an über Safe-UP)...")
    move_from_safe_up_to_pose(arm, poses[0])
    arm.move_to(poses[0]["b"], poses[0]["s"], poses[0]["e"], poses[0]["h"], spd=5, acc=3)
    arm.wait_until_settled(tolerance_deg=0.2, stable_count=6)
    repeat_pos = arm.read_position_averaged(n=10, interval=0.05)

    if repeat_pos:
        first_home = diagnostics["per_pose"][0]["measured"]
        repeat_err = {j: abs(repeat_pos[j] - first_home[j]) for j in JOINTS}
        print(f"     Repeatability (Home→...→Home): "
              f"Δb={repeat_err['b']:.3f}° Δs={repeat_err['s']:.3f}° Δe={repeat_err['e']:.3f}°")
        diagnostics["repeatability_deg"] = repeat_err
    else:
        diagnostics["repeatability_deg"] = {"b": 0, "s": 0, "e": 0}

    # Diagnostik-Zusammenfassung
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

    # Modell fitten
    print(f"\n{'─'*60}")
    print(f"  Fitte Kalibrierungsmodell ({len(commanded)} Datenpunkte, je gemittelt über {repeats} Messungen)...")

    model = CalibrationModel()
    residuals = model.fit(commanded, errors)

    print(f"\n  Modell-Güte (RMS-Residuen nach Fit):")
    for joint, rms in residuals.items():
        print(f"    {joint}: {rms:.4f}°")

    total_rms = np.sqrt(np.mean([r**2 for r in residuals.values()]))
    print(f"    Gesamt: {total_rms:.4f}°")

    if total_rms < 0.5:
        print(f"  ✅ Sehr gute Kalibrierung!")
    elif total_rms < 1.0:
        print(f"  ⚠️ Akzeptable Kalibrierung")
    else:
        print(f"  ❌ Schlechte Kalibrierung - evtl. Messfehler?")

    # Speichern (mit Diagnostik)
    cal_path = Path("calibration") / "roarm_calibration.cal"
    cal_path.parent.mkdir(exist_ok=True)
    model.save(str(cal_path), diagnostics=diagnostics)

    # Auch rohe Diagnostik separat speichern
    diag_path = Path("calibration") / "roarm_diagnostics.json"
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  📊 Diagnostik gespeichert: {diag_path}")

    # Zurück zu Safe-UP am Ende
    print(f"\n  🏠 Fahre zurück zu Safe-UP...")
    current = arm.read_position_deg()
    if current:
        move_to_safe_up(arm, current_pose=current)
    else:
        move_to_safe_up(arm, current_pose=None)

    return model


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
    args = p.parse_args()

    port = args.port or find_arm_port()
    if port is None:
        print("❌ Kein serieller Port gefunden!")
        sys.exit(1)

    print(f"📌 Verbinde mit {port}...")
    try:
        arm = RoArmConnection(port)
        print(f"   ✅ Verbunden")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        sys.exit(1)

    try:
        model = run_calibration(
            arm,
            auto_accept=args.auto,
            skip_identify=args.no_identify,
            repeats=args.repeats,
            pose_set_name=args.pose_set,
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
    main()

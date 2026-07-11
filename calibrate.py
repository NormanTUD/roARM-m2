#!/usr/bin/env python3
"""calibrate.py - RoArm-M2-S Kinematisches Kalibrierungsmodell

Workflow:
1. Roboter fährt N vordefinierte Posen an
2. User misst die TATSÄCHLICHE Position (Winkelmesser, Lineal, Kamera)
3. Modell wird gefittet: Soll → Korrektur
4. Kalibrierungsdatei wird gespeichert (.cal)

Beim Playback: Soll_korrigiert = Soll + Modell(Soll)

HINWEIS: "h" ist der Gripper (EOAT), kein echtes Rotationsgelenk.
         Kalibriert werden nur b (Base), s (Shoulder), e (Elbow).
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

# Die 3 echten Gelenke (h = Gripper, wird nicht kalibriert)
JOINTS = ["b", "s", "e"]

# Kalibrierposen - nur b, s, e variieren. h bleibt bei 180 (Gripper geschlossen)
CALIBRATION_POSES = [
    {"b": 0.0,   "s": 0.0,   "e": 90.0,  "h": 180.0},   # Home
    {"b": -45.0, "s": 0.0,   "e": 90.0,  "h": 180.0},   # Links
    {"b": 45.0,  "s": 0.0,   "e": 90.0,  "h": 180.0},   # Rechts
    {"b": 0.0,   "s": 30.0,  "e": 90.0,  "h": 180.0},   # Schulter hoch
    {"b": 0.0,   "s": -20.0, "e": 90.0,  "h": 180.0},   # Schulter runter
    {"b": 0.0,   "s": 0.0,   "e": 45.0,  "h": 180.0},   # Ellbogen eng
    {"b": 0.0,   "s": 0.0,   "e": 135.0, "h": 180.0},   # Ellbogen weit
    {"b": -30.0, "s": 20.0,  "e": 60.0,  "h": 180.0},   # Kombi 1
    {"b": 30.0,  "s": 20.0,  "e": 60.0,  "h": 180.0},   # Kombi 2
    {"b": -30.0, "s": -10.0, "e": 120.0, "h": 180.0},   # Kombi 3
    {"b": 30.0,  "s": -10.0, "e": 120.0, "h": 180.0},   # Kombi 4
    {"b": 0.0,   "s": 15.0,  "e": 70.0,  "h": 180.0},   # Zentrum
]


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
        self.coefficients = {
            "b": None,
            "s": None,
            "e": None,
        }
        self.is_fitted = False
        self.residuals = {}

    def _build_features(self, poses: list) -> np.ndarray:
        """Baut die Feature-Matrix für Polynom 2. Ordnung (3 Variablen)."""
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
        """
        Fittet das Modell.

        commanded_poses: Liste von {"b":..., "s":..., "e":..., "h":...}
        measured_errors: Liste von {"b": error_b, "s": error_s, "e": error_e}
                        wobei error = gemessen - befohlen
        """
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
        """Gibt die Korrektur für eine Soll-Pose zurück."""
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

    def save(self, filepath: str):
        """Speichert das Modell als JSON."""
        data = {
            "type": "polynomial_calibration_v2",
            "note": "h is gripper (EOAT), not calibrated",
            "joints_calibrated": JOINTS,
            "joints": {},
            "residuals": self.residuals,
        }
        for joint in JOINTS:
            if self.coefficients[joint] is not None:
                data["joints"][joint] = self.coefficients[joint].tolist()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Kalibrierung gespeichert: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CalibrationModel":
        """Lädt ein gespeichertes Modell."""
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
                           poll_interval: float = 0.15, timeout: float = 12.0):
        """
        Wartet bis der Arm stillsteht.
        Liest wiederholt die Position und prüft ob sich nichts mehr ändert.

        tolerance_deg: Max. Änderung zwischen zwei Lesungen (pro Gelenk) um als "still" zu gelten
        stable_count: Wie viele aufeinanderfolgende stabile Lesungen nötig sind
        poll_interval: Sekunden zwischen Lesungen
        timeout: Max. Wartezeit bevor aufgegeben wird
        """
        stable = 0
        last_pos = None
        start = time.time()

        while time.time() - start < timeout:
            pos = self.read_position_deg()
            if pos is None:
                time.sleep(poll_interval)
                continue

            if last_pos is not None:
                max_delta = max(
                    abs(pos["b"] - last_pos["b"]),
                    abs(pos["s"] - last_pos["s"]),
                    abs(pos["e"] - last_pos["e"]),
                )
                if max_delta < tolerance_deg:
                    stable += 1
                    if stable >= stable_count:
                        return pos  # Arm steht still
                else:
                    stable = 0

            last_pos = pos
            time.sleep(poll_interval)

        # Timeout - gib letzte bekannte Position zurück
        return last_pos

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
# GELENK-IDENTIFIKATION
# ============================================================

def identify_joint(arm, joint: str, current_pose: dict):
    """
    Bewegt ein einzelnes Gelenk hin und her damit der User sieht welches es ist.
    Für "h" (Gripper) wird T:106 verwendet (auf/zu).
    """
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
# KALIBRIERUNGS-WORKFLOW
# ============================================================

def run_calibration(arm, poses=CALIBRATION_POSES):
    """
    Interaktiver Kalibrierungs-Workflow.
    Kalibriert nur b, s, e (die 3 echten Gelenke).
    h (Gripper) wird übersprungen.
    """
    commanded = []
    errors = []

    print(f"\n{'='*60}")
    print(f"  KALIBRIERUNG - {len(poses)} Posen")
    print(f"  Kalibriert: b (Base), s (Shoulder), e (Elbow)")
    print(f"  NICHT kalibriert: h (Gripper/EOAT)")
    print(f"{'='*60}")

    print(f"\n  Ablauf pro Pose:")
    print(f"  1. Arm fährt zur Soll-Position (wartet bis er stillsteht)")
    print(f"  2. Du misst den tatsächlichen Winkel (b, s, e)")
    print(f"  3. Eingabe (oder ENTER = Servo-Wert übernehmen)")

    print(f"\n  Messmethoden:")
    print(f"  • Digitaler Winkelmesser an jedem Gelenk")
    print(f"  • Oder: Endpunkt-Position mit Lineal messen")

    # Gelenk-Identifikation?
    print(f"\n  Soll ich bei der ersten Pose alle Gelenke einzeln bewegen")
    print(f"  damit du siehst welches welches ist? (j/n)")
    show_joints = input("  > ").strip().lower() != 'n'

    input("\n  [ENTER] um zu starten...")

    joints_identified = False

    for i, pose in enumerate(poses):
        print(f"\n{'─'*60}")
        print(f"  Pose {i+1}/{len(poses)}")
        print(f"  Soll: b={pose['b']:.1f}° s={pose['s']:.1f}° e={pose['e']:.1f}°")

        # Arm hinfahren
        arm.torque_on()
        time.sleep(0.2)

        # Schnell in die Nähe
        arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=15, acc=8)
        print(f"  ⏳ Fahre zur Position...", end="", flush=True)
        arm.wait_until_settled()
        print(f" angekommen.", flush=True)

        # Langsam und präzise nachfahren
        arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
        print(f"  ⏳ Präzisions-Nachfahrt...", end="", flush=True)
        arm.wait_until_settled()
        print(f" fertig.", flush=True)

        # Gelenk-Identifikation (nur bei erster Pose)
        if show_joints and not joints_identified:
            print(f"\n  🔍 GELENK-IDENTIFIKATION:")
            print(f"     Ich bewege jetzt jedes Gelenk einzeln.")
            input(f"     [ENTER] um zu starten...")

            for joint in ["b", "s", "e", "h"]:
                identify_joint(arm, joint, pose)
                time.sleep(0.3)

            joints_identified = True
            print(f"\n  ✅ Alle Gelenke identifiziert!")
            print(f"     h (Gripper) wird NICHT kalibriert.")

            # Nochmal zur Pose fahren
            arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=10, acc=5)
            arm.wait_until_settled()
            arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
            arm.wait_until_settled()

        # === JETZT erst Position auslesen (Arm steht garantiert still) ===
        # Mehrfach lesen und Mittelwert bilden für noch bessere Genauigkeit
        readings = []
        for _ in range(5):
            pos = arm.read_position_deg()
            if pos:
                readings.append(pos)
            time.sleep(0.05)

        if readings:
            servo_pos = {
                "b": round(np.mean([r["b"] for r in readings]), 2),
                "s": round(np.mean([r["s"] for r in readings]), 2),
                "e": round(np.mean([r["e"] for r in readings]), 2),
                "h": round(np.mean([r["h"] for r in readings]), 2),
            }
            print(f"  ✅ Arm steht still. Servo meldet (Mittel aus {len(readings)} Lesungen):")
            print(f"     b={servo_pos['b']:.2f}° s={servo_pos['s']:.2f}° "
                  f"e={servo_pos['e']:.2f}°")
        else:
            servo_pos = {"b": pose["b"], "s": pose["s"], "e": pose["e"], "h": pose["h"]}
            print(f"  ⚠️ Konnte Servo-Position nicht lesen, verwende Soll-Werte")

        # User-Messung (nur b, s, e)
        print(f"\n  Miss jetzt die TATSÄCHLICHEN Winkel (nur b, s, e).")
        print(f"  (Leer = Servo-Wert | [w] = Gelenk wackeln)")

        measured = {}
        for joint in JOINTS:
            joint_names = {
                "b": "BASE (Drehung links/rechts)",
                "s": "SHOULDER (Schulter hoch/runter)",
                "e": "ELBOW (Ellbogen auf/zu)",
            }
            default = servo_pos[joint]

            while True:
                val = input(f"    {joint} [{joint_names[joint]}] "
                           f"(Soll={pose[joint]:.1f}°, Servo={default:.2f}°): ").strip()

                if val.lower() == 'w':
                    identify_joint(arm, joint, pose)
                    arm.move_to(pose["b"], pose["s"], pose["e"], pose["h"], spd=5, acc=3)
                    arm.wait_until_settled()
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

        # Fehler berechnen: gemessen - befohlen
        error = {
            "b": measured["b"] - pose["b"],
            "s": measured["s"] - pose["s"],
            "e": measured["e"] - pose["e"],
        }

        commanded.append(pose)
        errors.append(error)

        print(f"  → Fehler: Δb={error['b']:+.3f}° Δs={error['s']:+.3f}° "
              f"Δe={error['e']:+.3f}°")

    # Modell fitten
    print(f"\n{'='*60}")
    print(f"  Fitte Kalibrierungsmodell...")

    model = CalibrationModel()
    residuals = model.fit(commanded, errors)

    print(f"\n  Modell-Güte (RMS-Residuen):")
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

    # Speichern
    cal_path = Path("calibration") / "roarm_calibration.cal"
    cal_path.parent.mkdir(exist_ok=True)
    model.save(str(cal_path))

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Kalibrierung")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    args = p.parse_args()

    port = args.port or find_arm_port()
    if port is None:
        print("❌ Kein serieller Port gefunden!")
        sys.exit(1)

    print(f"🔌 Verbinde mit {port}...")
    try:
        arm = RoArmConnection(port)
        print(f"   ✅ Verbunden")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        sys.exit(1)

    try:
        model = run_calibration(arm)
        print(f"\n✅ Kalibrierung abgeschlossen!")
        print(f"   Datei: calibration/roarm_calibration.cal")
        print(f"   Wird automatisch von play_teached.py geladen.")
    except KeyboardInterrupt:
        print("\n\n   ⏹ Abgebrochen!")
    finally:
        arm.torque_on()
        time.sleep(0.3)
        arm.close()


if __name__ == "__main__":
    main()

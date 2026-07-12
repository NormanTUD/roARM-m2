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
        print("uv nicht installiert. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

import json
import time
import math
import threading
import serial
import serial.tools.list_ports
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

from safety import SafeArm, SafetyLimits, SafetyWatchdog

# Kalibrierungsmodell (optional)
try:
    from calibrate import CalibrationModel, JOINTS
except (ImportError, ModuleNotFoundError):
    CalibrationModel = None

# ============================================================
# KONFIGURATION
# ============================================================

START_POSITION_DEG = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

POSITION_TOLERANCE = 1.0
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
# HILFSFUNKTIONEN
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


def clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ============================================================
# ARM-KOMMUNIKATION
# ============================================================

class RoArmConnection:
    def __init__(self, port: str, baudrate: int = BAUDRATE):
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=SERIAL_TIMEOUT)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def send_cmd(self, cmd: dict) -> str:
        with self._lock:
            time.sleep(0.5)
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

    def send_cmd_fast(self, cmd: dict):
        """Sendet Befehl ohne auf Antwort zu warten (für Streaming)."""
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()

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

    def move_to_fast(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                     spd: int = 50, acc: int = 30):
        """Schneller Move ohne auf Antwort zu warten (für Streaming)."""
        cmd = {
            "T": 122,
            "b": round(b_deg, 2),
            "s": round(s_deg, 2),
            "e": round(e_deg, 2),
            "h": round(h_deg, 2),
            "spd": spd,
            "acc": acc,
        }
        self.send_cmd_fast(cmd)

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

    def gripper_open(self):
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


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
        
        for joint in ["b", "s", "e", "h"]:
            values = np.array([wp[joint] for wp in self._waypoints])
            # bc_type='natural' → Beschleunigung = 0 an den Enden (sanfter Start/Stopp)
            self._splines[joint] = CubicSpline(times, values, bc_type='natural')
        
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
                 stream_hz: int = STREAM_HZ):
        self._filepath = filepath
        self._port = port
        self._speed = speed
        self._loop = loop
        self._verify = verify
        self._manual_offset = manual_offset
        self._stream_hz = stream_hz
        self._arm: RoArmConnection = None
        self._data = None
        self._trajectory: SmoothTrajectory = None

        # Kalibrierungsmodell laden
        self._cal_model = None
        if CalibrationModel is not None:
            cal_path = Path("calibration/roarm_calibration.cal")
            if cal_path.exists():
                try:
                    self._cal_model = CalibrationModel.load(str(cal_path))
                    print(f"📐 Kalibrierungsmodell geladen")
                except Exception as e:
                    print(f"⚠️  Kalibrierung fehlgeschlagen: {e}")
            else:
                print(f"📐 Keine Kalibrierungsdatei gefunden (optional)")
        else:
            print(f"📐 numpy/calibrate nicht verfügbar, keine Kalibrierung")

    def _apply_calibration(self, target: dict) -> dict:
        """Wendet Kalibrierungskorrektur an."""
        if self._cal_model and self._cal_model.is_fitted:
            correction = self._cal_model.predict_correction(target)
            return {
                "b": target["b"] - correction["b"],
                "s": target["s"] - correction["s"],
                "e": target["e"] - correction["e"],
                "h": target["h"],  # h nicht korrigieren
            }
        return target

    def connect(self) -> bool:
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            return False
        print(f"🔌 Verbinde mit {port}...")
        try:
            self._arm_raw = RoArmConnection(port)

            print(f"   ✅ Verbunden")

            limits = SafetyLimits(
                    max_delta_per_cmd=20.0,        # Für Streaming etwas großzügiger
                    max_continuous_move_s=90.0,    # Max 90s am Stück
                    max_plausible_error=5.0,       # Fehler > 5° = Müll
                    )
            self._arm = SafeArm(self._arm_raw, limits=limits)

            # Watchdog starten
            self._watchdog = SafetyWatchdog(self._arm)
            self._watchdog.start()

            print(f"   ✅ Safety Watchguard started")

            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False

    def load(self) -> bool:
        path = Path(self._filepath)
        if not path.exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return False

        print(f"📁 Lade: {path.name}")
        self._data = parse_roarm_file(str(path))

        wps = self._data["waypoints"]
        if not wps:
            print("   ❌ Keine Wegpunkte in der Datei!")
            return False

        if len(wps) < 4:
            print(f"   ❌ Mindestens 4 Wegpunkte nötig für Spline (habe {len(wps)})")
            return False

        print(f"   ✅ {len(wps)} Wegpunkte geladen")
        print(f"   Original-Dauer: {wps[-1]['t']:.2f}s")
        print(f"   Gripper-Befehle: {len(self._data['gripper_cmds'])}")

        # Offset anwenden
        offset = self._manual_offset if self._manual_offset else self._data["offset"]
        has_offset = any(abs(v) > 0.001 for v in offset.values())

        if has_offset:
            print(f"\n   📐 Offset-Korrektur aktiv:")
            print(f"      Δb={offset['b']:+.3f}° Δs={offset['s']:+.3f}° "
                  f"Δe={offset['e']:+.3f}° Δh={offset['h']:+.3f}°")
            wps = apply_offset_to_waypoints(wps, offset)

        # Trajektorie erstellen
        print(f"\n🧮 Erstelle glatte Trajektorie (Cubic Spline)...")
        self._trajectory = SmoothTrajectory(wps, self._speed)
        duration = self._trajectory.get_duration()
        
        print(f"   Geglättete Dauer: {duration:.2f}s")
        print(f"   Stream-Rate: {self._stream_hz} Hz")
        print(f"   Erwartete Befehle: ~{int(duration * self._stream_hz)}")
        
        # Geschwindigkeitsprofil-Statistik
        n_test = 100
        speeds = [self._trajectory.get_speed_at(t) for t in np.linspace(0, duration, n_test)]
        print(f"   Speed-Profil: min={min(speeds):.2f}x max={max(speeds):.2f}x avg={np.mean(speeds):.2f}x")

        # Gelenkbereiche anzeigen
        for j in ["b", "s", "e", "h"]:
            vals = [wp[j] for wp in wps]
            print(f"   {j}: {min(vals):.2f}° → {max(vals):.2f}° (Δ{max(vals)-min(vals):.2f}°)")

        return True

    def go_to_start(self) -> bool:
        start = self._data["start_pos"]
        print(f"\n🏁 Fahre zur Startposition...")
        print(f"   Ziel: b={start['b']:.2f}° s={start['s']:.2f}° "
              f"e={start['e']:.2f}° h={start['h']:.2f}°")

        self._arm.torque_on()
        time.sleep(0.2)

        self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=25, acc=10)
        time.sleep(2.0)

        self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=10, acc=5)
        time.sleep(1.5)

        if self._verify:
            pos = self._arm.read_position_deg()
            if pos:
                err = max(abs(pos[j] - final[j]) for j in ["b", "s", "e", "h"])
                if err > 180.0:  # OFFENSICHTLICH MÜLL
                    print("⚠️ Ungültiger Read, überspringe Precision-Endpoint")
                    return
            if pos:
                max_err = max(abs(pos[j] - start[j]) for j in ["b", "s", "e", "h"])
                print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                      f"e={pos['e']:.2f}° h={pos['h']:.2f}°")
                print(f"   Max Fehler: {max_err:.2f}°")
                if max_err > POSITION_TOLERANCE:
                    self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=5, acc=3)
                    time.sleep(2.0)
                else:
                    print(f"   ✅ OK")

        # Zum ersten Punkt der Trajektorie
        first = self._trajectory.sample(0.0)
        corrected = self._apply_calibration(first)
        self._arm.move_to(corrected["b"], corrected["s"], corrected["e"], corrected["h"], spd=15, acc=8)
        time.sleep(1.0)

        return True

    def play_once(self):
        """Spielt die Trajektorie einmal flüssig ab."""
        traj = self._trajectory
        duration = traj.get_duration()
        interval = 1.0 / self._stream_hz
        
        gripper_cmds = sorted(self._data["gripper_cmds"], key=lambda x: x["t"])
        gripper_idx = 0
        # Gripper-Zeiten auf neue Zeitskala umrechnen (approximiert)
        # Wir verwenden die originale Zeit und prüfen gegen elapsed * speed_factor
        
        print(f"\n▶ STREAMING PLAYBACK ({self._stream_hz} Hz, {duration:.2f}s)")
        print(f"   {'─' * 55}")

        last_pos = None
        commands_sent = 0
        skipped = 0
        
        playback_start = time.time()

        while True:
            elapsed = time.time() - playback_start
            if elapsed >= duration:
                break

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
                    # Kompensiere die Pause
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
            if last_pos:
                max_delta = max(abs(corrected[j] - last_pos[j]) for j in ["b", "s", "e", "h"])
                if max_delta < MIN_DELTA_DEG:
                    should_send = False
                    skipped += 1

            if should_send:
                self._arm.move_to_fast(
                    corrected["b"], corrected["s"], corrected["e"], corrected["h"],
                    spd=STREAM_SPD, acc=STREAM_ACC
                )
                commands_sent += 1
                last_pos = corrected.copy()

                # Live-Ausgabe
                speed_now = traj.get_speed_at(elapsed)
                bar_len = int((speed_now / MAX_SPEED_FACTOR) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                progress_pct = (elapsed / duration) * 100

                clear_line()
                sys.stdout.write(
                    f"   ▶ [{progress_pct:5.1f}%] "
                    f"[{elapsed:6.2f}s/{duration:.2f}s] "
                    f"b={target['b']:7.2f}° s={target['s']:7.2f}° "
                    f"e={target['e']:7.2f}° "
                    f"|{bar}| v={speed_now:.1f}x"
                )
                sys.stdout.flush()

            # Timing einhalten
            next_time = playback_start + (commands_sent + 1) * interval
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ============================================================
        # PRECISION ENDPOINT: Mehrfaches Nachsetzen am Ende
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
            self._arm.move_to(
                corrected_final["b"], corrected_final["s"],
                corrected_final["e"], corrected_final["h"],
                spd=spd, acc=acc
            )
            time.sleep(ENDPOINT_SETTLE_WAIT)

            pos = self._arm.read_position_deg()
            if pos:
                err = max(abs(pos[j] - final[j]) for j in ["b", "s", "e", "h"])
                if err > 180.0:  # OFFENSICHTLICH MÜLL
                    print("⚠️ Ungültiger Read, überspringe Precision-Endpoint")
                    return
            if pos:
                err = max(abs(pos[j] - final[j]) for j in ["b", "s", "e", "h"])
                status = "✅" if err < 0.3 else "⚠️"
                print(f"      Pass {pass_num + 1}: spd={spd} acc={acc} → "
                      f"Fehler={err:.3f}° {status}")
                if err < 0.15:
                    print(f"      → Präzision erreicht, fertig")
                    break
            else:
                print(f"      Pass {pass_num + 1}: spd={spd} acc={acc} → (kein Read)")

        # Statistik
        actual_duration = time.time() - playback_start
        print(f"\n   ⏱ Fertig!")
        print(f"   Dauer: {actual_duration:.2f}s (Soll: {duration:.2f}s)")
        print(f"   Befehle gesendet: {commands_sent}")
        print(f"   Übersprungen (< {MIN_DELTA_DEG}° Δ): {skipped}")

        # Endposition verifizieren
        if self._verify:
            time.sleep(0.3)
            pos = self._arm.read_position_deg()
            if pos:
                err = max(abs(pos[j] - final[j]) for j in ["b", "s", "e", "h"])
                if err > 180.0:  # OFFENSICHTLICH MÜLL
                    print("⚠️ Ungültiger Read, überspringe Precision-Endpoint")
                    return
            if pos:
                max_err = max(abs(pos[j] - final[j]) for j in ["b", "s", "e", "h"])
                print(f"   Endposition Fehler: {max_err:.2f}°")
                print(f"   Soll: b={final['b']:.2f}° s={final['s']:.2f}° "
                      f"e={final['e']:.2f}° h={final['h']:.2f}°")
                print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                      f"e={pos['e']:.2f}° h={pos['h']:.2f}°")

    def run(self):
        print("=" * 60)
        print("  RoArm-M2-S SMOOTH PLAY MODE (Cubic Spline + Streaming)")
        print("  Flüssige Bewegung durch High-Frequency Command Streaming")
        print("=" * 60)

        if not self.load():
            return

        if not self.connect():
            return

        if not self.go_to_start():
            self._arm.close()
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
                    loop_count += 1
                    print(f"\n{'═' * 40} Loop #{loop_count} {'═' * 40}")
                    self.go_to_start()
                    time.sleep(0.5)
                    self.play_once()
                    time.sleep(1.0)
            else:
                self.play_once()
        except KeyboardInterrupt:
            print("\n\n   ⏹ Abgebrochen!")

        # Aufräumen
        print("\n🔒 Torque bleibt AN")
        self._arm.torque_on()
        time.sleep(0.3)
        self._arm.close()
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
    )
    player.run()


if __name__ == "__main__":
    main()

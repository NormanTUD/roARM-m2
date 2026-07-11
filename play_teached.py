#!/usr/bin/env python3
"""play_teached.py - RoArm-M2-S Playback mit Lookahead-Geschwindigkeitssteuerung
Erkennt Richtungsänderungen voraus und bremst rechtzeitig ab.
Schnell bei gleichbleibender Richtung, langsam bei Richtungswechsel/Stopp.
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

import json
import time
import math
import threading
import serial
import serial.tools.list_ports
from pathlib import Path

# Kalibrierungsmodell (optional)
try:
    import numpy as np
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

# Lookahead-Parameter
LOOKAHEAD_POINTS = 5        # Wie viele Punkte vorausschauen
SPD_MAX = 80                # Maximale Geschwindigkeit (Firmware-Einheit)
SPD_MIN = 8                 # Minimale Geschwindigkeit
ACC_MAX = 40                # Maximale Beschleunigung
ACC_MIN = 5                 # Minimale Beschleunigung
DIRECTION_THRESHOLD = 0.3   # Ab wann gilt eine Achse als "bewegt" (Grad)
BRAKE_COSINE_THRESHOLD = 0.5  # Unter diesem Cosinus-Wert wird gebremst (0=90°, -1=180°)


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
# LOOKAHEAD GESCHWINDIGKEITSBERECHNUNG
# ============================================================

def compute_direction_vector(wp_from: dict, wp_to: dict) -> list:
    """Berechnet den Richtungsvektor zwischen zwei Wegpunkten (4D: b,s,e,h)."""
    return [
        wp_to["b"] - wp_from["b"],
        wp_to["s"] - wp_from["s"],
        wp_to["e"] - wp_from["e"],
        wp_to["h"] - wp_from["h"],
    ]


def vector_magnitude(v: list) -> float:
    """Betrag eines Vektors."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1: list, v2: list) -> float:
    """
    Kosinus-Ähnlichkeit zwischen zwei Vektoren.
    +1 = gleiche Richtung, 0 = rechtwinklig, -1 = entgegengesetzt.
    """
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    if mag1 < 0.01 or mag2 < 0.01:
        return 0.0  # Einer der Vektoren ist ~null → "Stopp" → bremsen
    dot = sum(a * b for a, b in zip(v1, v2))
    return dot / (mag1 * mag2)


def compute_speed_for_waypoint(waypoints: list, index: int, speed_factor: float) -> tuple:
    """
    Berechnet optimale (spd, acc) für einen Wegpunkt basierend auf:
    1. Wie schnell muss ich sein um rechtzeitig beim nächsten Punkt zu sein?
    2. Kommt bald eine Richtungsänderung? → Abbremsen
    3. Geht es in die gleiche Richtung weiter? → Schnell bleiben
    
    Returns: (spd, acc)
    """
    total = len(waypoints)
    wp = waypoints[index]
    
    # --- Basis-Geschwindigkeit aus Zeitintervall ---
    if index < total - 1:
        dt = (waypoints[index + 1]["t"] - wp["t"]) / speed_factor
        vec_current = compute_direction_vector(wp, waypoints[index + 1])
        distance = vector_magnitude(vec_current)
    else:
        # Letzter Punkt → langsam ankommen
        return (SPD_MIN, ACC_MIN)
    
    if dt <= 0 or distance < 0.05:
        # Kein Zeitunterschied oder keine Bewegung
        return (SPD_MIN, ACC_MIN)
    
    # Benötigte Grad/Sekunde
    deg_per_sec = distance / dt
    
    # Basis-Speed aus benötigter Geschwindigkeit (Firmware-Kalibrierung)
    # Der Faktor hängt von der Firmware ab - hier ~1:1 Mapping
    base_spd = max(SPD_MIN, min(SPD_MAX, int(deg_per_sec * 0.8)))
    
    # --- Lookahead: Richtungsänderung erkennen ---
    min_cosine = 1.0  # Schlechteste Richtungsübereinstimmung im Lookahead
    approaching_stop = False
    
    for look in range(1, LOOKAHEAD_POINTS + 1):
        future_idx = index + look
        if future_idx >= total - 1:
            # Ende der Aufnahme naht → abbremsen
            remaining_points = total - 1 - index
            if remaining_points <= 3:
                approaching_stop = True
            break
        
        # Richtungsvektor des zukünftigen Segments
        vec_future = compute_direction_vector(
            waypoints[future_idx], waypoints[future_idx + 1]
        )
        
        # Vergleiche aktuelle Richtung mit zukünftiger
        cos_sim = cosine_similarity(vec_current, vec_future)
        
        # Je weiter weg der Lookahead, desto weniger Gewicht
        # (nahe Richtungsänderungen sind wichtiger)
        weight = 1.0 - (look - 1) * 0.15  # 1.0, 0.85, 0.7, 0.55, 0.4
        weighted_cos = cos_sim * weight + (1.0 - weight)  # Bias Richtung "kein Bremsen" für ferne Punkte
        
        min_cosine = min(min_cosine, weighted_cos)
        
        # Prüfe ob zukünftiges Segment ein "Stopp" ist (sehr kleine Bewegung)
        future_mag = vector_magnitude(vec_future)
        future_dt = (waypoints[future_idx + 1]["t"] - waypoints[future_idx]["t"]) / speed_factor
        if future_dt > 0 and future_mag / future_dt < 2.0:  # < 2°/s = quasi Stillstand
            if look <= 3:
                approaching_stop = True
    
    # --- Speed-Modifikation basierend auf Lookahead ---
    
    if approaching_stop:
        # Stopp kommt → stark abbremsen
        # Je näher am Stopp, desto langsamer
        remaining = total - 1 - index
        brake_factor = max(0.2, min(1.0, remaining / 5.0))
        final_spd = max(SPD_MIN, int(base_spd * brake_factor * 0.6))
        final_acc = max(ACC_MIN, int(final_spd * 0.4))
    
    elif min_cosine < BRAKE_COSINE_THRESHOLD:
        # Richtungsänderung kommt → proportional abbremsen
        # min_cosine: 0.5 → leicht bremsen, 0.0 → mittel, -1.0 → stark
        # Mapping: cosine [−1, 0.5] → brake_factor [0.25, 1.0]
        brake_factor = max(0.25, (min_cosine + 1.0) / (BRAKE_COSINE_THRESHOLD + 1.0))
        final_spd = max(SPD_MIN, int(base_spd * brake_factor))
        final_acc = max(ACC_MIN, int(final_spd * 0.5))
    
    else:
        # Gleiche Richtung → volle Geschwindigkeit
        final_spd = base_spd
        final_acc = max(ACC_MIN, min(ACC_MAX, int(final_spd * 0.6)))
    
    return (final_spd, final_acc)


def precompute_speeds(waypoints: list, speed_factor: float) -> list:
    """
    Berechnet für ALLE Wegpunkte die optimale Geschwindigkeit vor.
    Zweiter Pass: Rückwärts-Glättung damit Bremsungen nicht zu spät kommen.
    
    Returns: Liste von (spd, acc) Tupeln.
    """
    total = len(waypoints)
    speeds = []
    
    # --- Pass 1: Vorwärts - Lookahead-basierte Geschwindigkeit ---
    for i in range(total):
        spd, acc = compute_speed_for_waypoint(waypoints, i, speed_factor)
        speeds.append([spd, acc])
    
    # --- Pass 2: Rückwärts - Sicherstellen dass wir rechtzeitig bremsen können ---
    # Wenn Punkt i+1 langsam ist, darf Punkt i nicht zu schnell sein
    # (sonst kann der Arm nicht rechtzeitig abbremsen)
    MAX_SPEED_JUMP = 15  # Maximaler Speed-Unterschied zwischen aufeinanderfolgenden Punkten
    
    for i in range(total - 2, -1, -1):
        next_spd = speeds[i + 1][0]
        current_spd = speeds[i][0]
        
        # Wenn der nächste Punkt viel langsamer ist, müssen wir jetzt schon bremsen
        if current_spd > next_spd + MAX_SPEED_JUMP:
            speeds[i][0] = next_spd + MAX_SPEED_JUMP
            speeds[i][1] = max(ACC_MIN, int(speeds[i][0] * 0.5))
    
    # --- Pass 3: Vorwärts - Sanftes Beschleunigen ---
    # Nicht von 0 auf 100 springen
    for i in range(1, total):
        prev_spd = speeds[i - 1][0]
        current_spd = speeds[i][0]
        
        if current_spd > prev_spd + MAX_SPEED_JUMP:
            speeds[i][0] = prev_spd + MAX_SPEED_JUMP
            speeds[i][1] = max(ACC_MIN, int(speeds[i][0] * 0.5))
    
    return [(s[0], s[1]) for s in speeds]


# ============================================================
# ARM-VERBINDUNG
# ============================================================

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

# Konstante oben bei den anderen Konstanten hinzufügen:
OFFSET_BLEND_POINTS = 5
ENDPOINT_SETTLE_PASSES = 3
ENDPOINT_FINAL_SPD = 3
ENDPOINT_FINAL_ACC = 1
ENDPOINT_SETTLE_WAIT = 0.8


def apply_offset_to_waypoints(waypoints: list, offset: dict, blend_points: int = OFFSET_BLEND_POINTS) -> list:
    """
    Wendet den Offset auf die letzten N Wegpunkte an (linearer Blend).
    So wird der Endpunkt korrigiert ohne den restlichen Pfad zu verzerren.
    """
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
# PLAYER
# ============================================================

class RoArmPlayer:
    """
    Spielt .roarm Dateien ab mit Lookahead-Geschwindigkeitssteuerung.
    - Gleiche Richtung → schnell
    - Richtungsänderung voraus → frühzeitig abbremsen
    - Keine Glättung, keine Interpolation, kein Verschlucken
    """

    def __init__(self, filepath: str, port: str = None, speed: float = 1.0,
                 loop: bool = False, verify: bool = True, manual_offset: dict = None):
        self._filepath = filepath
        self._port = port
        self._speed = speed
        self._loop = loop
        self._verify = verify
        self._manual_offset = manual_offset
        self._arm: RoArmConnection = None
        self._data = None
        self._precomputed_speeds = None
        self._corrected_waypoints = None

        self._cal_model = None
        if CalibrationModel is not None:
            cal_path = Path("calibration/roarm_calibration.cal")
            if cal_path.exists():
                try:
                    self._cal_model = CalibrationModel.load(str(cal_path))
                    print(f"📐 Kalibrierungsmodell geladen")
                except Exception as e:
                    print(f"⚠️ Kalibrierung fehlgeschlagen: {e}")
            else:
                print(f"📐 Keine Kalibrierungsdatei gefunden (optional)")
        else:
            print(f"📐 numpy/calibrate nicht verfügbar, keine Kalibrierung")

    def _move_corrected(self, b: float, s: float, e: float, h: float,
                        spd: int = 20, acc: int = 10):
        """Sendet korrigierten Befehl unter Berücksichtigung der Kalibrierung."""
        if self._cal_model and self._cal_model.is_fitted:
            correction = self._cal_model.predict_correction(
                {"b": b, "s": s, "e": e, "h": h}
            )
            b = b - correction["b"]
            s = s - correction["s"]
            e = e - correction["e"]
            # h nicht korrigieren (Gripper)
        self._arm.move_to(b, s, e, h, spd=spd, acc=acc)

    def connect(self) -> bool:
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            return False
        print(f"🔌 Verbinde mit {port}...")
        try:
            self._arm = RoArmConnection(port)
            print(f"   ✅ Verbunden")
            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False

    def load(self) -> bool:
        path = Path(self._filepath)
        if not path.exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return False

        print(f"📄 Lade: {path.name}")
        self._data = parse_roarm_file(str(path))

        wps = self._data["waypoints"]
        if not wps:
            print("   ❌ Keine Wegpunkte in der Datei!")
            return False

        print(f"   ✅ {len(wps)} Wegpunkte geladen")
        print(f"   Dauer: {wps[-1]['t']:.2f}s (bei {self._speed}x → {wps[-1]['t']/self._speed:.2f}s)")
        print(f"   Gripper-Befehle: {len(self._data['gripper_cmds'])}")

        # Offset bestimmen und anwenden
        offset = self._manual_offset if self._manual_offset else self._data["offset"]
        has_offset = any(abs(v) > 0.001 for v in offset.values())

        if has_offset:
            print(f"\n   📐 Offset-Korrektur aktiv:")
            print(f"      Δb={offset['b']:+.3f}° Δs={offset['s']:+.3f}° "
                  f"Δe={offset['e']:+.3f}° Δh={offset['h']:+.3f}°")
            print(f"      Blend über letzte {OFFSET_BLEND_POINTS} Punkte")
            self._corrected_waypoints = apply_offset_to_waypoints(wps, offset)
        else:
            self._corrected_waypoints = wps

        # Geschwindigkeiten vorberechnen (auf korrigierten Waypoints)
        print(f"\n🧮 Berechne Lookahead-Geschwindigkeiten...")
        self._precomputed_speeds = precompute_speeds(self._corrected_waypoints, self._speed)

        # Statistik anzeigen
        spds = [s[0] for s in self._precomputed_speeds]
        print(f"   Speed-Bereich: {min(spds)} - {max(spds)} (Ø {sum(spds)/len(spds):.1f})")

        # Zeige Brems-Zonen
        brake_zones = 0
        in_brake = False
        for spd, _ in self._precomputed_speeds:
            if spd < SPD_MAX * 0.4 and not in_brake:
                brake_zones += 1
                in_brake = True
            elif spd >= SPD_MAX * 0.4:
                in_brake = False
        print(f"   Erkannte Brems-Zonen: {brake_zones}")

        for j in ["b", "s", "e", "h"]:
            vals = [wp[j] for wp in self._corrected_waypoints]
            print(f"   {j}: {min(vals):.2f}° → {max(vals):.2f}° (Δ{max(vals)-min(vals):.2f}°)")

        return True

    def go_to_start(self) -> bool:
        start = self._data["start_pos"]
        print(f"\n📍 Fahre zur Startposition...")
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
                max_err = max(abs(pos[j] - start[j]) for j in ["b", "s", "e", "h"])
                print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                      f"e={pos['e']:.2f}° h={pos['h']:.2f}°")
                print(f"   Max Fehler: {max_err:.2f}°")
                if max_err > POSITION_TOLERANCE:
                    self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=5, acc=3)
                    time.sleep(2.0)
                else:
                    print(f"   ✅ OK")

        # Zum ersten Wegpunkt (korrigiert)
        first_wp = self._corrected_waypoints[0]
        self._arm.move_to(first_wp["b"], first_wp["s"], first_wp["e"], first_wp["h"], spd=15, acc=8)
        time.sleep(1.0)

        return True

    def play_once(self):
        """Spielt die Aufnahme einmal ab mit Offset-Korrektur und Precision-Endpunkt."""
        wps = self._corrected_waypoints
        speeds = self._precomputed_speeds
        gripper_cmds = sorted(self._data["gripper_cmds"], key=lambda x: x["t"])
        gripper_idx = 0

        total_wps = len(wps)
        print(f"\n▶ WIEDERGABE ({total_wps} Wegpunkte, Speed: {self._speed}x, Lookahead: {LOOKAHEAD_POINTS})")
        print(f"   {'─' * 55}")

        last_sent = {"b": wps[0]["b"], "s": wps[0]["s"], "e": wps[0]["e"], "h": wps[0]["h"]}
        playback_start = time.time()
        commands_sent = 0
        skipped = 0

        for i, wp in enumerate(wps):
            target_time = playback_start + (wp["t"] / self._speed)

            # Gripper-Befehle
            while gripper_idx < len(gripper_cmds):
                gc = gripper_cmds[gripper_idx]
                if gc["t"] <= wp["t"]:
                    if gc["cmd"] == "CLOSE":
                        self._arm.gripper_close()
                        print(f"\n   ✊ GRIPPER ZU [{gc['t']:.2f}s]")
                    else:
                        self._arm.gripper_open()
                        print(f"\n   ✋ GRIPPER AUF [{gc['t']:.2f}s]")
                    time.sleep(0.3)
                    playback_start += 0.3
                    gripper_idx += 1
                else:
                    break

            # Warten bis Zeitpunkt
            now = time.time()
            wait_time = target_time - now
            if wait_time > 0:
                time.sleep(wait_time)
            elif wait_time < -0.1:
                if i < total_wps - 1:
                    skipped += 1
                    continue

            # Delta prüfen
            max_delta = max(
                abs(wp["b"] - last_sent["b"]),
                abs(wp["s"] - last_sent["s"]),
                abs(wp["e"] - last_sent["e"]),
                abs(wp["h"] - last_sent["h"]),
            )

            if max_delta < 0.05 and i < total_wps - 1:
                skipped += 1
                continue

            # Vorberechnete Geschwindigkeit verwenden
            spd, acc = speeds[i]

            # Befehl senden
            self._move_corrected(wp["b"], wp["s"], wp["e"], wp["h"], spd=spd, acc=acc)
            commands_sent += 1
            last_sent = {"b": wp["b"], "s": wp["s"], "e": wp["e"], "h": wp["h"]}

            # Live-Ausgabe mit Speed-Info
            elapsed = time.time() - playback_start
            bar_len = int((spd / SPD_MAX) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            clear_line()
            sys.stdout.write(
                f"   ▶ {i+1:4d}/{total_wps} "
                f"[{elapsed:6.2f}s] "
                f"b={wp['b']:7.2f}° s={wp['s']:7.2f}° "
                f"e={wp['e']:7.2f}° h={wp['h']:7.2f}° "
                f"|{bar}| spd={spd:2d} acc={acc:2d}"
            )
            sys.stdout.flush()

        # ============================================================
        # PRECISION ENDPOINT (Feature 2): Mehrfaches Nachsetzen
        # ============================================================
        last_wp = wps[-1]
        print(f"\n\n   🎯 Precision-Endpunkt ({ENDPOINT_SETTLE_PASSES} Passes):")

        speeds_sequence = [
            (8, 4),    # Pass 1: mittel
            (5, 2),    # Pass 2: langsam
            (ENDPOINT_FINAL_SPD, ENDPOINT_FINAL_ACC),  # Pass 3: sehr langsam
        ]

        for pass_num in range(ENDPOINT_SETTLE_PASSES):
            spd, acc = speeds_sequence[min(pass_num, len(speeds_sequence) - 1)]

            self._arm.move_to(last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"],
                              spd=spd, acc=acc)
            time.sleep(ENDPOINT_SETTLE_WAIT)

            # Position prüfen
            pos = self._arm.read_position_deg()
            if pos:
                err = max(abs(pos[j] - last_wp[j]) for j in ["b", "s", "e", "h"])
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
        print(f"   Dauer: {actual_duration:.2f}s (Soll: {wps[-1]['t']/self._speed:.2f}s)")
        print(f"   Befehle gesendet: {commands_sent}/{total_wps}")
        print(f"   Übersprungen: {skipped}")

        # Endposition verifizieren
        if self._verify:
            time.sleep(0.3)
            pos = self._arm.read_position_deg()
            if pos:
                max_err = max(abs(pos[j] - last_wp[j]) for j in ["b", "s", "e", "h"])
                print(f"   Endposition Fehler: {max_err:.2f}°")
                print(f"   Soll: b={last_wp['b']:.2f}° s={last_wp['s']:.2f}° "
                      f"e={last_wp['e']:.2f}° h={last_wp['h']:.2f}°")
                print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° "
                      f"e={pos['e']:.2f}° h={pos['h']:.2f}°")

    def run(self):
        print("=" * 60)
        print("  RoArm-M2-S PLAY MODE (Lookahead)")
        print("  Schnell bei gleicher Richtung, bremst vor Richtungswechsel")
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
    global LOOKAHEAD_POINTS, SPD_MAX, SPD_MIN
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Play Mode - Precision Edition")
    p.add_argument("file", type=str, help=".roarm Datei zum Abspielen")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Geschwindigkeit (0.5=halb, 2.0=doppelt)")
    p.add_argument("--loop", action="store_true",
                   help="Endlos wiederholen")
    p.add_argument("--no-verify", action="store_true",
                   help="Positionsverifikation überspringen")
    p.add_argument("--lookahead", type=int, default=LOOKAHEAD_POINTS,
                   help=f"Lookahead-Punkte (default: {LOOKAHEAD_POINTS})")
    p.add_argument("--spd-max", type=int, default=SPD_MAX,
                   help=f"Max Speed (default: {SPD_MAX})")
    p.add_argument("--spd-min", type=int, default=SPD_MIN,
                   help=f"Min Speed (default: {SPD_MIN})")
    p.add_argument("--offset", type=str, default=None,
                   help="Manueller Offset: 'b=0.5,s=-0.3,e=0.1,h=0.0'")
    args = p.parse_args()

    # Globale Parameter überschreiben
    LOOKAHEAD_POINTS = args.lookahead
    SPD_MAX = args.spd_max
    SPD_MIN = args.spd_min

    # Manuellen Offset parsen
    manual_offset = None
    if args.offset:
        manual_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        for part in args.offset.split(","):
            k, v = part.strip().split("=")
            manual_offset[k.strip()] = float(v.strip())

    player = RoArmPlayer(
        filepath=args.file,
        port=args.port,
        speed=args.speed,
        loop=args.loop,
        verify=not args.no_verify,
        manual_offset=manual_offset,
    )
    player.run()


if __name__ == "__main__":
    main()

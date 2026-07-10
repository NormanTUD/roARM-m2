#!/usr/bin/env python3
"""play.py - RoArm-M2-S Playback (Clean Edition)
Spielt .roarm Dateien ab. Keine Interpolation, keine Backlash-Kompensation.
Fährt exakt die aufgezeichneten Punkte ab mit korrektem Timing.
Zeigt jeden Schritt live in der Konsole.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
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
# ARM-VERBINDUNG (identisch zum teach.py)
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
        """Sendet einen JSON-Befehl und liest die Antwort."""
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
        """Liest die aktuelle Position EINMAL. Gibt dict mit Radians zurück."""
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
        """Liest Position und konvertiert zu Grad. Kein Filter, kein Averaging."""
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
        """Bewegt den Arm zu einer Position in Grad."""
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
        """Schaltet alle Servos frei."""
        self.send_cmd({"T": 210, "cmd": 0})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 0})
            time.sleep(0.02)

    def torque_on(self):
        """Schaltet alle Servos fest."""
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
    """
    Parst eine .roarm Datei. Gibt zurück:
    {
        "waypoints": [{"t": ..., "b": ..., "s": ..., "e": ..., "h": ...}, ...],
        "gripper_cmds": [{"t": ..., "cmd": "OPEN"/"CLOSE"}, ...],
        "config": {"hz": ..., "threshold": ...},
        "start_pos": {"b": ..., "s": ..., "e": ..., "h": ...},
    }
    """
    waypoints = []
    gripper_cmds = []
    config = {"hz": 20, "threshold": 0.3}
    start_pos = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Kommentare und Config
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

            if line.startswith("#"):
                continue

            # MOVE Befehle
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

            # GRIPPER Befehle
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
    }


# ============================================================
# PLAYER
# ============================================================

class RoArmPlayer:
    """
    Spielt .roarm Dateien ab.
    KEIN Kalman, KEIN Backlash, KEINE Interpolation.
    Fährt exakt die aufgezeichneten Punkte mit korrektem Timing ab.
    """

    def __init__(self, filepath: str, port: str = None, speed: float = 1.0,
                 loop: bool = False, verify: bool = True):
        self._filepath = filepath
        self._port = port
        self._speed = speed
        self._loop = loop
        self._verify = verify
        self._arm: RoArmConnection = None
        self._data = None

    def connect(self) -> bool:
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            return False
        print(f"🔌 Verbinde mit {port}...")
        try:
            self._arm = RoArmConnection(port)
            print(f"   ✓ Verbunden")
            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False

    def load(self) -> bool:
        """Lädt und validiert die .roarm Datei."""
        path = Path(self._filepath)
        if not path.exists():
            print(f"❌ Datei nicht gefunden: {path}")
            return False

        print(f"📂 Lade: {path.name}")
        self._data = parse_roarm_file(str(path))

        wps = self._data["waypoints"]
        if not wps:
            print("   ❌ Keine Wegpunkte in der Datei!")
            return False

        print(f"   ✓ {len(wps)} Wegpunkte geladen")
        print(f"   Dauer: {wps[-1]['t']:.2f}s (bei {self._speed}x → {wps[-1]['t']/self._speed:.2f}s)")
        print(f"   Gripper-Befehle: {len(self._data['gripper_cmds'])}")
        print(f"   Start: b={self._data['start_pos']['b']:.2f}° "
              f"s={self._data['start_pos']['s']:.2f}° "
              f"e={self._data['start_pos']['e']:.2f}° "
              f"h={self._data['start_pos']['h']:.2f}°")

        # Zeige Bewegungsbereich
        for j in ["b", "s", "e", "h"]:
            vals = [wp[j] for wp in wps]
            print(f"   {j}: {min(vals):.2f}° → {max(vals):.2f}° (Δ{max(vals)-min(vals):.2f}°)")

        return True

    def go_to_start(self) -> bool:
        """Fährt zur Startposition der Aufnahme."""
        start = self._data["start_pos"]
        print(f"\n📍 Fahre zur Startposition...")
        print(f"   Ziel: b={start['b']:.2f}° s={start['s']:.2f}° "
              f"e={start['e']:.2f}° h={start['h']:.2f}°")

        self._arm.torque_on()
        time.sleep(0.2)

        # Erste Bewegung
        self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=25, acc=10)
        time.sleep(2.0)

        # Zweite, langsame Bewegung
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
                    print(f"   ⚠ Abweichung > {POSITION_TOLERANCE}°, nochmal...")
                    self._arm.move_to(start["b"], start["s"], start["e"], start["h"], spd=5, acc=3)
                    time.sleep(2.0)
                    pos = self._arm.read_position_deg()
                    if pos:
                        max_err = max(abs(pos[j] - start[j]) for j in ["b", "s", "e", "h"])
                        print(f"   Neuer Fehler: {max_err:.2f}°")
                else:
                    print(f"   ✓ OK")
            else:
                print(f"   ⚠ Kann Position nicht lesen")

        # Zum ersten Wegpunkt fahren (falls anders als Start)
        first_wp = self._data["waypoints"][0]
        self._arm.move_to(first_wp["b"], first_wp["s"], first_wp["e"], first_wp["h"], spd=15, acc=8)
        time.sleep(1.0)

        return True

    def play_once(self):
        """Spielt die Aufnahme einmal ab."""
        wps = self._data["waypoints"]
        gripper_cmds = sorted(self._data["gripper_cmds"], key=lambda x: x["t"])
        gripper_idx = 0

        total_wps = len(wps)
        print(f"\n▶ WIEDERGABE ({total_wps} Wegpunkte, Speed: {self._speed}x)")
        print(f"   {'─' * 55}")

        # Letzter gesendeter Befehl (für Delta-Berechnung)
        last_sent = {"b": wps[0]["b"], "s": wps[0]["s"], "e": wps[0]["e"], "h": wps[0]["h"]}

        playback_start = time.time()
        commands_sent = 0
        skipped = 0

        for i, wp in enumerate(wps):
            # Ziel-Zeitpunkt berechnen (mit Speed-Faktor)
            target_time = playback_start + (wp["t"] / self._speed)

            # Gripper-Befehle die vor diesem Zeitpunkt liegen ausführen
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
                    # Zeit-Kompensation für Gripper-Pause
                    playback_start += 0.3
                    gripper_idx += 1
                else:
                    break

            # Warten bis der richtige Zeitpunkt ist
            now = time.time()
            wait_time = target_time - now
            if wait_time > 0:
                time.sleep(wait_time)
            elif wait_time < -0.1:
                # Wir sind zu spät - Wegpunkt überspringen wenn nicht der letzte
                if i < total_wps - 1:
                    skipped += 1
                    continue

            # Berechne Delta zum letzten gesendeten Befehl
            delta_b = abs(wp["b"] - last_sent["b"])
            delta_s = abs(wp["s"] - last_sent["s"])
            delta_e = abs(wp["e"] - last_sent["e"])
            delta_h = abs(wp["h"] - last_sent["h"])
            max_delta = max(delta_b, delta_s, delta_e, delta_h)

            # NUR senden wenn sich tatsächlich was bewegt hat (> 0.05°)
            # Das verhindert unnötige Befehle die den Bus überlasten
            if max_delta < 0.05 and i < total_wps - 1:
                skipped += 1
                continue

            # Befehl senden - DIREKT, ohne Backlash, ohne Kompensation
            # spd=0, acc=0 = so schnell wie möglich (Firmware-Default)
            self._arm.move_to(wp["b"], wp["s"], wp["e"], wp["h"], spd=0, acc=0)
            commands_sent += 1
            last_sent = {"b": wp["b"], "s": wp["s"], "e": wp["e"], "h": wp["h"]}

            # Live-Ausgabe
            elapsed = time.time() - playback_start
            delta_str = ""
            if max_delta >= 0.1:
                parts = []
                if delta_b >= 0.1: parts.append(f"b:{wp['b']-last_sent['b']:+.2f}")
                if delta_s >= 0.1: parts.append(f"s:{wp['s']-last_sent['s']:+.2f}")
                if delta_e >= 0.1: parts.append(f"e:{wp['e']-last_sent['e']:+.2f}")
                if delta_h >= 0.1: parts.append(f"h:{wp['h']-last_sent['h']:+.2f}")
                delta_str = " | " + " ".join(parts)

            clear_line()
            sys.stdout.write(
                f"   ▶ {i+1:4d}/{total_wps} "
                f"[{elapsed:6.2f}s] "
                f"b={wp['b']:7.2f}° s={wp['s']:7.2f}° "
                f"e={wp['e']:7.2f}° h={wp['h']:7.2f}°"
                f"{delta_str}"
            )
            sys.stdout.flush()

        # Letzten Punkt nochmal senden für Präzision
        last_wp = wps[-1]
        time.sleep(0.1)
        self._arm.move_to(last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"], spd=10, acc=5)
        time.sleep(0.5)

        actual_duration = time.time() - playback_start
        print(f"\n\n   ⏹ Fertig!")
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
        """Hauptprogramm."""
        print("=" * 60)
        print("  RoArm-M2-S PLAY MODE")
        print("  Spielt aufgezeichnete Bewegungen exakt ab")
        print("  Keine Interpolation, keine Kompensation")
        print("=" * 60)

        # Laden
        if not self.load():
            return

        # Verbinden
        if not self.connect():
            return

        # Zur Startposition
        if not self.go_to_start():
            self._arm.close()
            return

        # Warten
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
        print("✓ Fertig!\n")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Play Mode - Aufnahmen abspielen")
    p.add_argument("file", type=str, help=".roarm Datei zum Abspielen")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Geschwindigkeit (0.5=halb, 2.0=doppelt)")
    p.add_argument("--loop", action="store_true",
                   help="Endlos wiederholen")
    p.add_argument("--no-verify", action="store_true",
                   help="Positionsverifikation überspringen")
    args = p.parse_args()

    player = RoArmPlayer(
        filepath=args.file,
        port=args.port,
        speed=args.speed,
        loop=args.loop,
        verify=not args.no_verify,
    )
    player.run()


if __name__ == "__main__":
    main()

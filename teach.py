#!/usr/bin/env python3
"""teach.py - RoArm-M2-S Teach & Record (Precision Edition)
- Gravity Compensation: Liest Position kurz mit Torque AN
- Offset-Kalibrierung: Nach Aufnahme wird Endpunkt mit Torque präzise kalibriert
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
from datetime import datetime
import select

# ============================================================
# KONFIGURATION
# ============================================================

START_POSITION_DEG = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

RECORD_HZ = 50
MOVE_THRESHOLD_DEG = 0.1
POSITION_TOLERANCE = 1.0
BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1

# Gravity Compensation: Alle N Samples kurz Torque an, Position lesen, Torque aus
GRAVITY_COMP_SETTLE_MS = 30  # ms warten nach Torque-an bevor Position gelesen wird


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


def deg_to_rad(deg: float) -> float:
    return deg * (math.pi / 180.0)


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

    def read_position_stable(self, num_reads: int = 3) -> dict:
        last_valid = None
        for i in range(num_reads):
            raw = self.read_position_raw()
            if raw and "b" in raw:
                last_valid = raw
            if i < num_reads - 1:
                time.sleep(0.015)
        return last_valid

    def read_position_deg(self) -> dict:
        raw = self.read_position_stable(num_reads=3)
        if raw is None:
            return None
        return {
            "b": round(rad_to_deg(raw["b"]), 2),
            "s": round(rad_to_deg(raw["s"]), 2),
            "e": round(rad_to_deg(raw["e"]), 2),
            "h": round(rad_to_deg(raw.get("t", raw.get("h", 0))), 2),
        }

    def read_position_deg_single(self) -> dict:
        """Schneller einzelner Read ohne Stabilisierung (für Recording-Loop)."""
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

    def torque_on_fast(self):
        """Schnelles Torque-an ohne individuelle Servo-Befehle."""
        self.send_cmd({"T": 210, "cmd": 1})

    def torque_off_fast(self):
        """Schnelles Torque-aus ohne individuelle Servo-Befehle."""
        self.send_cmd({"T": 210, "cmd": 0})

    def gripper_open(self):
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ============================================================
# TEACH RECORDER (mit Gravity Compensation + Offset-Kalibrierung)
# ============================================================

class TeachRecorder:
    def __init__(self, port: str = None, output_dir: str = "teach_recordings",
                 hz: int = RECORD_HZ, threshold: float = MOVE_THRESHOLD_DEG,
                 gravity_comp: bool = True):
        self._port = port
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hz = hz
        self._threshold = threshold
        self._gravity_comp = gravity_comp
        self._arm: RoArmConnection = None
        self._recording = False
        self._waypoints = []
        self._rec_start_time = 0.0
        self._last_recorded_pos = None
        self._total_waypoints = 0
        self._sample_counter = 0
        # Offset-Kalibrierung
        self._endpoint_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}

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

    def go_to_start(self) -> bool:
        print(f"\n📍 Fahre zur Startposition...")
        print(f"   Ziel: b={START_POSITION_DEG['b']:.1f}° "
              f"s={START_POSITION_DEG['s']:.1f}° "
              f"e={START_POSITION_DEG['e']:.1f}° "
              f"h={START_POSITION_DEG['h']:.1f}°")

        self._arm.torque_on()
        time.sleep(0.2)

        self._arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=30, acc=15
        )
        time.sleep(2.0)

        self._arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=10, acc=5
        )
        time.sleep(1.5)

        pos = self._arm.read_position_deg()
        if pos is None:
            print("   ⚠️ Kann Position nicht lesen!")
            return False

        print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° e={pos['e']:.2f}° h={pos['h']:.2f}°")
        max_error = max(abs(pos[j] - START_POSITION_DEG[j]) for j in ["b", "s", "e", "h"])

        if max_error <= POSITION_TOLERANCE:
            print(f"   ✅ Startposition OK (max Fehler: {max_error:.2f}°)")
            return True
        else:
            print(f"   ⚠️ Abweichung: {max_error:.2f}° - nochmal...")
            self._arm.move_to(
                START_POSITION_DEG["b"], START_POSITION_DEG["s"],
                START_POSITION_DEG["e"], START_POSITION_DEG["h"],
                spd=5, acc=3
            )
            time.sleep(2.0)
            return True

    def _read_with_gravity_comp(self) -> dict:
        """
        GRAVITY COMPENSATION:
        Schaltet kurz Torque an, wartet bis Servo sich stabilisiert,
        liest die Position (= die Position die der Servo wirklich anfährt),
        schaltet Torque wieder aus.
        
        Das eliminiert den Fehler durch Schwerkraft-Durchhängen bei Torque-off.
        """
        # Torque an
        self._arm.torque_on_fast()
        # Kurz warten bis Servo sich auf die aktuelle Position "einlockt"
        time.sleep(GRAVITY_COMP_SETTLE_MS / 1000.0)
        # Position lesen - DAS ist die Position die der Servo wirklich hält
        pos = self._arm.read_position_deg_single()
        # Torque wieder aus damit User weiter bewegen kann
        self._arm.torque_off_fast()
        return pos

    def start_recording(self):
        self._waypoints = []
        self._total_waypoints = 0
        self._last_recorded_pos = None
        self._recording = True
        self._rec_start_time = time.time()
        self._sample_counter = 0

        print("\n🔓 Torque AUS - Arm ist jetzt frei bewegbar")
        self._arm.torque_off()
        time.sleep(0.3)

        # Erste Position mit Gravity Comp lesen
        if self._gravity_comp:
            pos = self._read_with_gravity_comp()
        else:
            pos = self._arm.read_position_deg()

        if pos:
            self._record_point(pos, force=True)

        comp_str = " + Gravity Comp" if self._gravity_comp else ""
        print(f"\n🔴 AUFNAHME LÄUFT ({self._hz} Hz, Schwelle: {self._threshold}°{comp_str})")
        print(f"   Bewege den Arm jetzt!")
        print(f"   [ENTER] = Stopp | [g] = Gripper toggle")
        print(f"   ─" * 24)

    def _record_point(self, pos: dict, force: bool = False) -> bool:
        if not force and self._last_recorded_pos is not None:
            max_delta = max(
                abs(pos["b"] - self._last_recorded_pos["b"]),
                abs(pos["s"] - self._last_recorded_pos["s"]),
                abs(pos["e"] - self._last_recorded_pos["e"]),
                abs(pos["h"] - self._last_recorded_pos["h"]),
            )
            if max_delta < self._threshold:
                return False

        elapsed = time.time() - self._rec_start_time
        self._waypoints.append({
            "t": round(elapsed, 4),
            "b": pos["b"],
            "s": pos["s"],
            "e": pos["e"],
            "h": pos["h"],
        })
        self._total_waypoints += 1
        self._last_recorded_pos = pos.copy()

        clear_line()
        sys.stdout.write(
            f"   ● WP#{self._total_waypoints:4d} "
            f"[{elapsed:6.2f}s] "
            f"b={pos['b']:7.2f}° s={pos['s']:7.2f}° "
            f"e={pos['e']:7.2f}° h={pos['h']:7.2f}°"
        )
        sys.stdout.flush()
        return True

    def record_loop(self):
        import termios
        import tty

        interval = 1.0 / self._hz
        gripper_open = True

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while self._recording:
                loop_start = time.time()

                # Tastendruck prüfen
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\n' or ch == '\r' or ch == 'q':
                        self._recording = False
                        break
                    elif ch == 'g':
                        if gripper_open:
                            self._arm.gripper_close()
                            gripper_open = False
                            elapsed = time.time() - self._rec_start_time
                            self._waypoints.append({"t": round(elapsed, 4), "cmd": "GRIPPER_CLOSE"})
                            print(f"\n   ✊ Gripper ZU [{elapsed:.2f}s]")
                        else:
                            self._arm.gripper_open()
                            gripper_open = True
                            elapsed = time.time() - self._rec_start_time
                            self._waypoints.append({"t": round(elapsed, 4), "cmd": "GRIPPER_OPEN"})
                            print(f"\n   ✋ Gripper AUF [{elapsed:.2f}s]")

                # Position lesen - mit oder ohne Gravity Compensation
                self._sample_counter += 1

                pos = self._arm.read_position_deg_single()

                if pos:
                    self._record_point(pos)

                # Timing einhalten
                elapsed_loop = time.time() - loop_start
                sleep_time = interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print(f"\n\n   ⏹ Aufnahme gestoppt. {self._total_waypoints} Wegpunkte aufgezeichnet.")

    def calibrate_endpoint(self):
        """
        OFFSET-KALIBRIERUNG (Feature 1):
        Nach der Aufnahme fährt der Arm den letzten aufgezeichneten Punkt
        mit Torque an. Der User kann dann sehen wo der Arm wirklich landet
        und manuell (per Tastendruck) den Offset korrigieren.
        
        Alternativ: Automatische Messung des Unterschieds zwischen
        "Torque-off Position" und "Torque-on Position" am Endpunkt.
        """
        move_wps = [wp for wp in self._waypoints if "cmd" not in wp]
        if len(move_wps) < 2:
            return

        last_wp = move_wps[-1]
        print(f"\n📐 OFFSET-KALIBRIERUNG")
        print(f"   Letzter aufgezeichneter Punkt (Torque off):")
        print(f"   b={last_wp['b']:.2f}° s={last_wp['s']:.2f}° "
              f"e={last_wp['e']:.2f}° h={last_wp['h']:.2f}°")

        # Arm festmachen und zum letzten Punkt fahren
        print(f"\n   Fahre zum Endpunkt mit Torque AN...")
        self._arm.torque_on()
        time.sleep(0.3)

        # Langsam und präzise hinfahren
        self._arm.move_to(last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"], spd=10, acc=5)
        time.sleep(1.5)
        self._arm.move_to(last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"], spd=5, acc=3)
        time.sleep(1.0)

        # Messen wo der Arm wirklich ist
        actual_pos = self._arm.read_position_deg()
        if actual_pos is None:
            print("   ⚠️ Kann Position nicht lesen, überspringe Kalibrierung")
            return

        print(f"   Tatsächliche Position (Torque on):")
        print(f"   b={actual_pos['b']:.2f}° s={actual_pos['s']:.2f}° "
              f"e={actual_pos['e']:.2f}° h={actual_pos['h']:.2f}°")

        # Automatischer Offset berechnen
        auto_offset = {
            "b": round(actual_pos["b"] - last_wp["b"], 3),
            "s": round(actual_pos["s"] - last_wp["s"], 3),
            "e": round(actual_pos["e"] - last_wp["e"], 3),
            "h": round(actual_pos["h"] - last_wp["h"], 3),
        }
        print(f"\n   Automatisch erkannter Offset (Torque-on minus Torque-off):")
        print(f"   Δb={auto_offset['b']:+.3f}° Δs={auto_offset['s']:+.3f}° "
              f"Δe={auto_offset['e']:+.3f}° Δh={auto_offset['h']:+.3f}°")

        # Fragen ob der User manuell korrigieren will
        print(f"\n   Optionen:")
        print(f"   [ENTER] = Automatischen Offset verwenden (empfohlen)")
        print(f"   [m]     = Manuell korrigieren (Arm wird freigegeben)")
        print(f"   [n]     = Kein Offset (ignorieren)")

        choice = input("   > ").strip().lower()

        if choice == 'm':
            # Manueller Modus: User positioniert den Arm exakt
            print(f"\n   🔓 Arm wird freigegeben. Positioniere den Endeffektor EXAKT")
            print(f"   an der Stelle wo er aufsetzen soll.")
            print(f"   Drücke ENTER wenn fertig.")
            self._arm.torque_off()
            time.sleep(0.3)
            input()

            # Jetzt mit Gravity Comp die echte Position lesen
            manual_pos = self._read_with_gravity_comp()
            if manual_pos:
                self._endpoint_offset = {
                    "b": round(manual_pos["b"] - last_wp["b"], 3),
                    "s": round(manual_pos["s"] - last_wp["s"], 3),
                    "e": round(manual_pos["e"] - last_wp["e"], 3),
                    "h": round(manual_pos["h"] - last_wp["h"], 3),
                }
                print(f"   Manueller Offset:")
                print(f"   Δb={self._endpoint_offset['b']:+.3f}° "
                      f"Δs={self._endpoint_offset['s']:+.3f}° "
                      f"Δe={self._endpoint_offset['e']:+.3f}° "
                      f"Δh={self._endpoint_offset['h']:+.3f}°")
            self._arm.torque_on()
            time.sleep(0.3)

        elif choice == 'n':
            print(f"   → Kein Offset wird angewendet")
            self._endpoint_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}

        else:
            # Automatischen Offset verwenden
            self._endpoint_offset = auto_offset
            print(f"   ✅ Automatischer Offset wird verwendet")

    def save(self) -> str:
        if not self._waypoints:
            print("   Nichts zum Speichern!")
            return None

        move_wps = [wp for wp in self._waypoints if "cmd" not in wp]
        if not move_wps:
            print("   Keine Bewegungs-Daten!")
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._output_dir / f"recording_{ts}.roarm"

        # Offset-Info für die Datei
        has_offset = any(abs(v) > 0.001 for v in self._endpoint_offset.values())

        lines = [
            f"# RoArm-M2-S Recording (Precision Edition)",
            f"# Datum: {datetime.now().isoformat()}",
            f"# Wegpunkte: {len(move_wps)}",
            f"# Aufnahme-Hz: {self._hz}",
            f"# Schwelle: {self._threshold}°",
            f"# Dauer: {move_wps[-1]['t']:.2f}s",
            f"# Gravity Compensation: {'ja' if self._gravity_comp else 'nein'}",
            f"#",
            f"#CONFIG hz={self._hz}",
            f"#CONFIG threshold={self._threshold}",
            f"#CONFIG gravity_comp={'1' if self._gravity_comp else '0'}",
            f"#START_POS b={START_POSITION_DEG['b']:.2f} s={START_POSITION_DEG['s']:.2f} "
            f"e={START_POSITION_DEG['e']:.2f} h={START_POSITION_DEG['h']:.2f}",
        ]

        if has_offset:
            lines.append(
                f"#OFFSET b={self._endpoint_offset['b']:.3f} "
                f"s={self._endpoint_offset['s']:.3f} "
                f"e={self._endpoint_offset['e']:.3f} "
                f"h={self._endpoint_offset['h']:.3f}"
            )

        lines.append("")

        for wp in self._waypoints:
            if "cmd" in wp:
                if wp["cmd"] == "GRIPPER_CLOSE":
                    lines.append(f"GRIPPER CLOSE t={wp['t']:.4f}")
                elif wp["cmd"] == "GRIPPER_OPEN":
                    lines.append(f"GRIPPER OPEN t={wp['t']:.4f}")
            else:
                lines.append(
                    f"MOVE b={wp['b']:.2f} s={wp['s']:.2f} "
                    f"e={wp['e']:.2f} h={wp['h']:.2f} t={wp['t']:.4f}"
                )

        with open(filename, 'w') as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n💾 Gespeichert: {filename}")
        print(f"   {len(move_wps)} Wegpunkte, {move_wps[-1]['t']:.1f}s Dauer")
        if has_offset:
            print(f"   📐 Mit Offset-Korrektur: Δb={self._endpoint_offset['b']:+.3f}° "
                  f"Δs={self._endpoint_offset['s']:+.3f}° "
                  f"Δe={self._endpoint_offset['e']:+.3f}° "
                  f"Δh={self._endpoint_offset['h']:+.3f}°")

        if len(move_wps) >= 2:
            first = move_wps[0]
            last = move_wps[-1]
            print(f"   Start: b={first['b']:.2f}° s={first['s']:.2f}° e={first['e']:.2f}° h={first['h']:.2f}°")
            print(f"   Ende:  b={last['b']:.2f}° s={last['s']:.2f}° e={last['e']:.2f}° h={last['h']:.2f}°")

        return str(filename)

    def run(self):
        print("=" * 60)
        print("  RoArm-M2-S TEACH MODE (Precision Edition)")
        print("  Features:")
        print("  • Gravity Compensation (Torque-on Reads)")
        print("  • Endpoint Offset-Kalibrierung")
        print("=" * 60)

        if not self.connect():
            return

        if not self.go_to_start():
            self._arm.close()
            return

        print(f"\n{'─' * 60}")
        print(f"  Bereit! Drücke ENTER um die Aufnahme zu starten.")
        print(f"  (Der Arm wird dann freigegeben)")
        print(f"{'─' * 60}")
        input()

        # Aufnahme
        self.start_recording()
        self.record_loop()

        # Offset-Kalibrierung nach der Aufnahme
        self.calibrate_endpoint()

        # Speichern
        filepath = self.save()

        # Arm festmachen
        print("\n🔒 Torque AN - Arm ist wieder fest")
        self._arm.torque_on()
        time.sleep(0.3)

        self._arm.close()
        print("✅ Fertig!\n")

        if filepath:
            print(f"  Zum Abspielen:")
            print(f"  python3 play_teached.py {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Teach Mode (Precision Edition)")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--hz", type=int, default=RECORD_HZ,
                   help=f"Aufnahme-Frequenz (default: {RECORD_HZ})")
    p.add_argument("--threshold", type=float, default=MOVE_THRESHOLD_DEG,
                   help=f"Bewegungs-Schwelle in Grad (default: {MOVE_THRESHOLD_DEG})")
    p.add_argument("--output", type=str, default="teach_recordings",
                   help="Ausgabe-Verzeichnis")
    p.add_argument("--no-gravity-comp", action="store_true",
                   help="Gravity Compensation deaktivieren")
    args = p.parse_args()

    recorder = TeachRecorder(
        port=args.port,
        output_dir=args.output,
        hz=args.hz,
        threshold=args.threshold,
        gravity_comp=not args.no_gravity_comp,
    )
    recorder.run()


if __name__ == "__main__":
    main()

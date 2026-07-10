#!/usr/bin/env python3
"""teach.py - RoArm-M2-S Teach & Record (Clean Edition, keine Kamera)
Zeichnet Gelenkpositionen auf die du dem Arm physisch gibst.
Zeigt jeden Schritt live in der Konsole an.
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

# Startposition in GRAD (wird beim Start angefahren)
START_POSITION_DEG = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

# Aufnahme-Einstellungen
RECORD_HZ = 50            # statt 20
MOVE_THRESHOLD_DEG = 0.1  # statt 0.3
POSITION_TOLERANCE = 1.0  # Toleranz für Startposition in Grad

# Serielle Verbindung
BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_arm_port() -> str:
    """Findet den seriellen Port des Arms."""
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
    """Radians zu Grad, KEINE Rundung hier."""
    return rad * (180.0 / math.pi)


def deg_to_rad(deg: float) -> float:
    """Grad zu Radians."""
    return deg * (math.pi / 180.0)


def clear_line():
    """Löscht die aktuelle Konsolenzeile."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ============================================================
# ARM-KOMMUNIKATION (minimal, direkt, ohne Overengineering)
# ============================================================

class RoArmConnection:
    """Direkte serielle Verbindung zum RoArm-M2-S. Kein Kalman, kein Median."""

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
                        # Wenn wir Feedback bekommen, sofort zurück
                        if '"T":1051' in line or '"b"' in line:
                            return line
                else:
                    time.sleep(0.005)
            return response

    def read_position_raw(self) -> dict:
        """
        Liest die aktuelle Position EINMAL. Gibt Radians zurück wie die Firmware.
        Kein Filter, kein Averaging. Die rohen Werte.
        """
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
        """
        Liest N mal und nimmt den LETZTEN stabilen Wert.
        Kein Kalman, kein Median - einfach den letzten konsistenten Wert.
        Grund: Die Firmware gibt manchmal beim ersten Read nach einer Pause
        einen veralteten Wert zurück.
        """
        last_valid = None
        for i in range(num_reads):
            raw = self.read_position_raw()
            if raw and "b" in raw:
                last_valid = raw
            if i < num_reads - 1:
                time.sleep(0.015)  # 15ms zwischen Reads
        return last_valid

    def read_position_deg(self) -> dict:
        """Liest Position und konvertiert zu Grad. Rundet auf 2 Dezimalstellen."""
        raw = self.read_position_stable(num_reads=3)
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
        """Schaltet alle Servos frei (Arm kann bewegt werden)."""
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
# TEACH RECORDER
# ============================================================

class TeachRecorder:
    def __init__(self, port: str = None, output_dir: str = "teach_recordings",
                 hz: int = RECORD_HZ, threshold: float = MOVE_THRESHOLD_DEG):
        self._port = port
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hz = hz
        self._threshold = threshold
        self._arm: RoArmConnection = None
        self._recording = False
        self._waypoints = []  # Liste von (time, b, s, e, h)
        self._rec_start_time = 0.0
        self._last_recorded_pos = None
        self._total_waypoints = 0

    def connect(self) -> bool:
        """Verbindet mit dem Arm."""
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            print("   Prüfe ob der Arm per USB verbunden ist.")
            return False

        print(f"🔌 Verbinde mit {port}...")
        try:
            self._arm = RoArmConnection(port)
            print(f"   ✓ Verbunden")
            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False

    def go_to_start(self) -> bool:
        """Fährt zur Startposition und verifiziert."""
        print(f"\n📍 Fahre zur Startposition...")
        print(f"   Ziel: b={START_POSITION_DEG['b']:.1f}° "
              f"s={START_POSITION_DEG['s']:.1f}° "
              f"e={START_POSITION_DEG['e']:.1f}° "
              f"h={START_POSITION_DEG['h']:.1f}°")

        # Torque an
        self._arm.torque_on()
        time.sleep(0.2)

        # Erste Bewegung: schnell in die Nähe
        self._arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=30, acc=15
        )
        time.sleep(2.0)

        # Zweite Bewegung: langsam und präzise
        self._arm.move_to(
            START_POSITION_DEG["b"], START_POSITION_DEG["s"],
            START_POSITION_DEG["e"], START_POSITION_DEG["h"],
            spd=10, acc=5
        )
        time.sleep(1.5)

        # Verifizieren
        pos = self._arm.read_position_deg()
        if pos is None:
            print("   ⚠ Kann Position nicht lesen!")
            return False

        print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° e={pos['e']:.2f}° h={pos['h']:.2f}°")

        max_error = 0.0
        for joint in ["b", "s", "e", "h"]:
            err = abs(pos[joint] - START_POSITION_DEG[joint])
            max_error = max(max_error, err)

        if max_error <= POSITION_TOLERANCE:
            print(f"   ✓ Startposition OK (max Fehler: {max_error:.2f}°)")
            return True
        else:
            print(f"   ⚠ Abweichung: {max_error:.2f}° (Toleranz: {POSITION_TOLERANCE}°)")
            # Nochmal versuchen
            self._arm.move_to(
                START_POSITION_DEG["b"], START_POSITION_DEG["s"],
                START_POSITION_DEG["e"], START_POSITION_DEG["h"],
                spd=5, acc=3
            )
            time.sleep(2.0)
            pos = self._arm.read_position_deg()
            if pos:
                max_error = max(abs(pos[j] - START_POSITION_DEG[j]) for j in ["b", "s", "e", "h"])
                print(f"   Zweiter Versuch: max Fehler = {max_error:.2f}°")
                if max_error <= POSITION_TOLERANCE:
                    print(f"   ✓ OK")
                    return True
            print(f"   ⚠ Startposition nicht exakt, fahre trotzdem fort")
            return True  # Trotzdem weitermachen

    def start_recording(self):
        """Startet die Aufnahme."""
        self._waypoints = []
        self._total_waypoints = 0
        self._last_recorded_pos = None
        self._recording = True
        self._rec_start_time = time.time()

        # Torque aus damit man den Arm bewegen kann
        print("\n🔓 Torque AUS - Arm ist jetzt frei bewegbar")
        self._arm.torque_off()
        time.sleep(0.3)

        # Erste Position sofort aufzeichnen
        pos = self._arm.read_position_deg()
        if pos:
            self._record_point(pos, force=True)

        print(f"\n🔴 AUFNAHME LÄUFT ({self._hz} Hz, Schwelle: {self._threshold}°)")
        print(f"   Bewege den Arm jetzt!")
        print(f"   [ENTER] = Stopp | [g] = Gripper toggle")
        print(f"   ─────────────────────────────────────────────────")

    def _record_point(self, pos: dict, force: bool = False) -> bool:
        """
        Zeichnet einen Punkt auf, wenn sich genug bewegt hat.
        Gibt True zurück wenn aufgezeichnet wurde.
        """
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

        # Live-Ausgabe
        delta_str = ""
        if self._total_waypoints > 1 and len(self._waypoints) >= 2:
            prev = self._waypoints[-2]
            deltas = []
            for j in ["b", "s", "e", "h"]:
                d = pos[j] - prev[j]
                if abs(d) >= 0.1:
                    deltas.append(f"{j}:{d:+.2f}")
            delta_str = " | " + " ".join(deltas) if deltas else ""

        clear_line()
        sys.stdout.write(
            f"   ● WP#{self._total_waypoints:4d} "
            f"[{elapsed:6.2f}s] "
            f"b={pos['b']:7.2f}° s={pos['s']:7.2f}° "
            f"e={pos['e']:7.2f}° h={pos['h']:7.2f}°"
            f"{delta_str}"
        )
        sys.stdout.flush()
        return True

    def record_loop(self):
        """Hauptschleife der Aufnahme. Blockiert bis ENTER gedrückt wird."""
        import termios
        import tty

        interval = 1.0 / self._hz
        gripper_open = True

        # Terminal in raw mode für nicht-blockierende Eingabe
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())

            while self._recording:
                loop_start = time.time()

                # Nicht-blockierend auf Tastendruck prüfen
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
                            self._waypoints.append({
                                "t": round(elapsed, 4),
                                "cmd": "GRIPPER_CLOSE"
                            })
                            print(f"\n   ✊ Gripper ZU [{elapsed:.2f}s]")
                        else:
                            self._arm.gripper_open()
                            gripper_open = True
                            elapsed = time.time() - self._rec_start_time
                            self._waypoints.append({
                                "t": round(elapsed, 4),
                                "cmd": "GRIPPER_OPEN"
                            })
                            print(f"\n   ✋ Gripper AUF [{elapsed:.2f}s]")

                # Position lesen und aufzeichnen
                pos = self._arm.read_position_deg()
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

    def save(self) -> str:
        """Speichert die Aufnahme als .roarm Datei."""
        if not self._waypoints:
            print("   Nichts zum Speichern!")
            return None

        # Nur MOVE-Waypoints zählen
        move_wps = [wp for wp in self._waypoints if "cmd" not in wp]
        if not move_wps:
            print("   Keine Bewegungs-Daten!")
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._output_dir / f"recording_{ts}.roarm"

        lines = [
            f"# RoArm-M2-S Recording",
            f"# Datum: {datetime.now().isoformat()}",
            f"# Wegpunkte: {len(move_wps)}",
            f"# Aufnahme-Hz: {self._hz}",
            f"# Schwelle: {self._threshold}°",
            f"# Dauer: {move_wps[-1]['t']:.2f}s",
            f"#",
            f"#CONFIG hz={self._hz}",
            f"#CONFIG threshold={self._threshold}",
            f"#START_POS b={START_POSITION_DEG['b']:.2f} s={START_POSITION_DEG['s']:.2f} "
            f"e={START_POSITION_DEG['e']:.2f} h={START_POSITION_DEG['h']:.2f}",
            f"",
        ]

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

        # Zusammenfassung der Bewegung
        if len(move_wps) >= 2:
            first = move_wps[0]
            last = move_wps[-1]
            print(f"   Start: b={first['b']:.2f}° s={first['s']:.2f}° e={first['e']:.2f}° h={first['h']:.2f}°")
            print(f"   Ende:  b={last['b']:.2f}° s={last['s']:.2f}° e={last['e']:.2f}° h={last['h']:.2f}°")

            # Min/Max pro Gelenk
            for j in ["b", "s", "e", "h"]:
                vals = [wp[j] for wp in move_wps]
                print(f"   {j}: min={min(vals):.2f}° max={max(vals):.2f}° range={max(vals)-min(vals):.2f}°")

        return str(filename)

    def run(self):
        """Hauptprogramm."""
        print("=" * 60)
        print("  RoArm-M2-S TEACH MODE")
        print("  Aufzeichnung durch physisches Bewegen des Arms")
        print("  Keine Kamera, reine Konsolen-Ausgabe")
        print("=" * 60)

        # Verbinden
        if not self.connect():
            return

        # Zur Startposition fahren
        if not self.go_to_start():
            self._arm.close()
            return

        # Warten auf Benutzer
        print(f"\n{'─' * 60}")
        print(f"  Bereit! Drücke ENTER um die Aufnahme zu starten.")
        print(f"  (Der Arm wird dann freigegeben)")
        print(f"{'─' * 60}")
        input()

        # Aufnahme starten
        self.start_recording()
        self.record_loop()

        # Speichern
        filepath = self.save()

        # Arm wieder festmachen
        print("\n🔒 Torque AN - Arm ist wieder fest")
        self._arm.torque_on()
        time.sleep(0.3)

        # Verbindung schließen
        self._arm.close()
        print("✓ Fertig!\n")

        if filepath:
            print(f"  Zum Abspielen:")
            print(f"  python3 play.py {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Teach Mode - Bewegungen aufzeichnen")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--hz", type=int, default=RECORD_HZ,
                   help=f"Aufnahme-Frequenz in Hz (default: {RECORD_HZ})")
    p.add_argument("--threshold", type=float, default=MOVE_THRESHOLD_DEG,
                   help=f"Bewegungs-Schwelle in Grad (default: {MOVE_THRESHOLD_DEG})")
    p.add_argument("--output", type=str, default="teach_recordings",
                   help="Ausgabe-Verzeichnis")
    args = p.parse_args()

    recorder = TeachRecorder(
        port=args.port,
        output_dir=args.output,
        hz=args.hz,
        threshold=args.threshold,
    )
    recorder.run()


if __name__ == "__main__":
    main()

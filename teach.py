#!/usr/bin/env python3
"""teach.py - RoArm-M2-S Teach & Record (Precision Edition + Live-Visualisierung)
- Gravity Compensation: Liest Position kurz mit Torque AN
- Offset-Kalibrierung: Nach Aufnahme wird Endpunkt mit Torque präzise kalibriert
- 3D-Visualisierung: Zeigt live die Arm-Position während der Aufnahme
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "rich",
#     "matplotlib",
#     "numpy",
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
from datetime import datetime
import select

from robot import (
    RoArmConnection, find_arm_port, rad_to_deg, deg_to_rad,
    clear_line, START_POSITION_DEG, POSITION_TOLERANCE, BAUDRATE,
)
from ui import (
    console, print_banner, print_connection_status, print_success,
    print_warning, print_info, print_position_inline, TeachDisplay,
)
from visualize import RobotVisualizer

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
# TEACH RECORDER (mit Gravity Compensation + Offset-Kalibrierung + Visualisierung)
# ============================================================

class TeachRecorder:
    def __init__(self, port: str = None, output_dir: str = "recordings",
                 hz: int = RECORD_HZ, threshold: float = MOVE_THRESHOLD_DEG,
                 gravity_comp: bool = True, visualize: bool = True):
        self._port = port
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hz = hz
        self._threshold = threshold
        self._gravity_comp = gravity_comp
        self._visualize = visualize
        self._arm: RoArmConnection = None
        self._recording = False
        self._waypoints = []
        self._rec_start_time = 0.0
        self._last_recorded_pos = None
        self._total_waypoints = 0
        self._sample_counter = 0
        # Offset-Kalibrierung
        self._endpoint_offset = {"b": 0.0, "s": 0.0, "e": 0.0, "h": 0.0}
        # 3D-Visualisierung
        self._viz: RobotVisualizer = None

    def connect(self) -> bool:
        port = self._port or find_arm_port()
        if port is None:
            print("❌ FEHLER: Kein serieller Port gefunden!")
            return False
        print(f"🔌 Verbinde mit {port}...")
        try:
            self._arm = RoArmConnection(port)
            print(f"   ✔ Verbunden")
            
            # Visualisierung starten
            if self._visualize:
                print(f"   🖥️  Starte 3D-Visualisierung...")
                self._viz = RobotVisualizer(live=True, update_interval=0.05)
                self._viz.start()
                time.sleep(0.5)
                # Initiale Position anzeigen
                self._viz.update_pose(
                    START_POSITION_DEG["b"], START_POSITION_DEG["s"],
                    START_POSITION_DEG["e"], START_POSITION_DEG["h"]
                )
                print(f"   ✔ Visualisierung aktiv")
            
            return True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")
            return False

    def go_to_start(self) -> bool:
        print(f"\n🏠 Fahre zur Startposition...")
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
        
        # Visualisierung updaten während der Arm zur Startposition fährt
        if self._viz:
            self._viz.update_pose(
                START_POSITION_DEG["b"], START_POSITION_DEG["s"],
                START_POSITION_DEG["e"], START_POSITION_DEG["h"],
                target=START_POSITION_DEG
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

        # Visualisierung mit tatsächlicher Position updaten
        if self._viz and pos:
            self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])

        print(f"   Ist:  b={pos['b']:.2f}° s={pos['s']:.2f}° e={pos['e']:.2f}° h={pos['h']:.2f}°")
        max_error = max(abs(pos[j] - START_POSITION_DEG[j]) for j in ["b", "s", "e", "h"])

        if max_error <= POSITION_TOLERANCE:
            print(f"   ✔ Startposition OK (max Fehler: {max_error:.2f}°)")
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
        """
        self._arm.torque_on_fast()
        time.sleep(GRAVITY_COMP_SETTLE_MS / 1000.0)
        pos = self._arm.read_position_deg_single()
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

        # Trail löschen für neue Aufnahme
        if self._viz:
            self._viz.clear_trail()

        # Erste Position mit Gravity Comp lesen
        if self._gravity_comp:
            pos = self._read_with_gravity_comp()
        else:
            pos = self._arm.read_position_deg()

        if pos:
            self._record_point(pos, force=True)
            # Visualisierung updaten
            if self._viz:
                self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])

        comp_str = " + Gravity Comp" if self._gravity_comp else ""
        viz_str = " + 3D-Viz" if self._viz else ""
        print(f"\n🔴 AUFNAHME LÄUFT ({self._hz} Hz, Schwelle: {self._threshold}°{comp_str}{viz_str})")
        print(f"   Bewege den Arm jetzt!")
        print(f"   [ENTER] = Stopp | [g] = Gripper toggle")
        print(f"   {'─' * 24}")

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

            teach_display = TeachDisplay(hz=self._hz, threshold=self._threshold, gravity_comp=self._gravity_comp)
            teach_display.start()

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

                # Position lesen
                self._sample_counter += 1
                pos = self._arm.read_position_deg_single()

                if pos:
                    self._record_point(pos)
                    
                    # === VISUALISIERUNG UPDATEN ===
                    if self._viz:
                        self._viz.update_pose(pos["b"], pos["s"], pos["e"], pos["h"])

                # Timing einhalten
                elapsed_loop = time.time() - loop_start
                sleep_time = interval - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

                teach_display.update(
                    elapsed=time.time() - self._rec_start_time,
                    waypoint_count=self._total_waypoints,
                    current_pos=pos,
                    last_recorded=self._last_recorded_pos,
                )

            teach_display.stop()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print(f"\n\n   ⏹ Aufnahme gestoppt. {self._total_waypoints} Wegpunkte aufgezeichnet.")

    def calibrate_endpoint(self):
        """
        OFFSET-KALIBRIERUNG:
        Nach der Aufnahme fährt der Arm den letzten aufgezeichneten Punkt
        mit Torque an. Der User kann dann sehen wo der Arm wirklich landet.
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
        
        # Visualisierung: Target anzeigen
        if self._viz:
            self._viz.update_pose(
                last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"],
                target={"b": last_wp["b"], "s": last_wp["s"], "e": last_wp["e"], "h": last_wp["h"]}
            )
        
        time.sleep(1.5)
        self._arm.move_to(last_wp["b"], last_wp["s"], last_wp["e"], last_wp["h"], spd=5, acc=3)
        time.sleep(1.0)

        # Messen wo der Arm wirklich ist
        actual_pos = self._arm.read_position_deg()
        if actual_pos is None:
            print("   ⚠️ Kann Position nicht lesen, überspringe Kalibrierung")
            return

        # Visualisierung mit tatsächlicher Position
        if self._viz and actual_pos:
            self._viz.update_pose(
                actual_pos["b"], actual_pos["s"], actual_pos["e"], actual_pos["h"],
                target={"b": last_wp["b"], "s": last_wp["s"], "e": last_wp["e"], "h": last_wp["h"]}
            )

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

        print(f"\n   Optionen:")
        print(f"   [ENTER] = Automatischen Offset verwenden (empfohlen)")
        print(f"   [m]     = Manuell korrigieren (Arm wird freigegeben)")
        print(f"   [n]     = Kein Offset (ignorieren)")

        choice = input("   > ").strip().lower()

        if choice == 'm':
            print(f"\n   🔓 Arm wird freigegeben. Positioniere den Endeffektor EXAKT")
            print(f"   an der Stelle wo er aufsetzen soll.")
            print(f"   Drücke ENTER wenn fertig.")
            self._arm.torque_off()
            time.sleep(0.3)
            input()

            manual_pos = self._read_with_gravity_comp()
            if manual_pos:
                self._endpoint_offset = {
                    "b": round(manual_pos["b"] - last_wp["b"], 3),
                    "s": round(manual_pos["s"] - last_wp["s"], 3),
                    "e": round(manual_pos["e"] - last_wp["e"], 3),
                    "h": round(manual_pos["h"] - last_wp["h"], 3),
                }
                # Visualisierung updaten
                if self._viz:
                    self._viz.update_pose(manual_pos["b"], manual_pos["s"], manual_pos["e"], manual_pos["h"])
                    
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
            self._endpoint_offset = auto_offset
            print(f"   ✔ Automatischer Offset wird verwendet")

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

        has_offset = any(abs(v) > 0.001 for v in self._endpoint_offset.values())

        lines = [
            f"# RoArm-M2-S Recording (Precision Edition)",
            f"# Datum: {datetime.now().isoformat()}",
            f"# Wegpunkte: {len(move_wps)}",
            f"# Aufnahme-Hz: {self._hz}",
            f"# Schwelle: {self._threshold}°",
            f"# Dauer: {move_wps[-1]['t']:.2f}s",
            f"# Gravity Compensation: {'ja' if self._gravity_comp else 'nein'}",
            f"# Visualisierung: {'ja' if self._visualize else 'nein'}",
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
        print_banner("teach", "Gravity Compensation + Endpoint Offset-Kalibrierung + 3D-Visualisierung")

        if not self.connect():
            return

        if not self.go_to_start():
            self._arm.close()
            if self._viz:
                self._viz.stop()
            return

        print(f"\n{'─' * 60}")
        print(f"  Bereit! Drücke ENTER um die Aufnahme zu starten.")
        print(f"  (Der Arm wird dann freigegeben)")
        if self._viz:
            print(f"  🖥️  3D-Visualisierung läuft im separaten Fenster")
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

        # Visualisierung stoppen
        if self._viz:
            print("   🖥️  Visualisierung wird geschlossen...")
            self._viz.stop()

        self._arm.close()
        print("✔ Fertig!\n")

        if filepath:
            print(f"  Zum Abspielen:")
            print(f"  python3 play.py {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="RoArm-M2-S Teach Mode (Precision + Visualisierung)")
    p.add_argument("--port", type=str, default=None, help="Serieller Port (auto-detect)")
    p.add_argument("--hz", type=int, default=RECORD_HZ,
                   help=f"Aufnahme-Frequenz (default: {RECORD_HZ})")
    p.add_argument("--threshold", type=float, default=MOVE_THRESHOLD_DEG,
                   help=f"Bewegungs-Schwelle in Grad (default: {MOVE_THRESHOLD_DEG})")
    p.add_argument("--output", type=str, default="recordings",
                   help="Ausgabe-Verzeichnis")
    p.add_argument("--no-gravity-comp", action="store_true",
                   help="Gravity Compensation deaktivieren")
    p.add_argument("--no-viz", action="store_true",
                   help="3D-Visualisierung deaktivieren")
    args = p.parse_args()

    recorder = TeachRecorder(
        port=args.port,
        output_dir=args.output,
        hz=args.hz,
        threshold=args.threshold,
        gravity_comp=not args.no_gravity_comp,
        visualize=not args.no_viz,
    )
    recorder.run()


if __name__ == "__main__":
    main()


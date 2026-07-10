#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserial",
#     "opencv-python",
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
        print("uv is not installed. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

_ensure_uv()

"""
RoArm-M2-S Teach & Record
==========================
Führe den Arm per Hand, drücke Tasten für Greifer/LED, und alles wird
in einer .roarm-Skriptdatei + Kamerabildern aufgezeichnet.

Steuerung (im OpenCV-Fenster):
  SPACE  → Wegpunkt aufzeichnen (aktuelle Position)
  O      → Greifer öffnen (wird aufgezeichnet)
  C      → Greifer schließen (wird aufgezeichnet)
  L      → LED an (255)
  K      → LED aus (0)
  W      → Wartezeit einfügen (1s, konfigurierbar)
  R      → Aufnahme starten/stoppen
  P      → Aufgezeichnetes Skript abspielen
  Q      → Beenden

Die .roarm-Datei ist eine einfache zeilenbasierte Sprache:
  #CONFIG speed_scale=1.0
  #CONFIG spd=50
  #CONFIG acc=100
  MOVE b=0.0 s=0.0 e=90.0 h=180.0
  GRIPPER OPEN
  GRIPPER CLOSE
  LED 255
  LED 0
  WAIT 1.0
  FRAME frame_000001.jpg
"""

import json
import time
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2

# Import our arm library
from roarm_m2s import RoArmM2S


class TeachRecorder:
    """
    Teach-Modus: Torque aus, Arm per Hand führen, Wegpunkte aufzeichnen.
    Erzeugt .roarm Skriptdatei + Kamerabilder.
    """

    POLL_HZ = 10  # Wie oft Position abgefragt wird bei kontinuierlicher Aufnahme

    def __init__(self, port: str = None, camera_index: int = 2,
                 output_dir: str = "teach_recordings",
                 continuous: bool = True, wait_seconds: float = 1.0):
        self._port = port
        self._camera_index = camera_index
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._continuous = continuous  # Kontinuierlich aufzeichnen vs. nur bei SPACE
        self._wait_seconds = wait_seconds

        self._arm: RoArmM2S = None
        self._camera = None

        self._recording = False
        self._commands: list = []  # Liste von Skript-Zeilen
        self._frame_count = 0
        self._session_dir: Path = None
        self._images_dir: Path = None
        self._script_path: Path = None

        # Config defaults (werden in .roarm geschrieben)
        self._speed_scale = 1.0
        self._spd = 50
        self._acc = 100

        self._running = False
        self._window_name = "RoArm Teach & Record"

    def setup(self) -> bool:
        """Hardware initialisieren."""
        print("=" * 60)
        print("  RoArm-M2-S Teach & Record")
        print("=" * 60)

        # Arm verbinden
        print("\n  [1/3] Arm verbinden...")
        try:
            self._arm = RoArmM2S(port=self._port, enable_vision=False)
            print(f"    ✓ Verbunden: {self._arm.port}")
        except Exception as e:
            print(f"    ✗ Fehler: {e}")
            return False

        # Kamera
        print(f"  [2/3] Kamera {self._camera_index}...")
        self._camera = cv2.VideoCapture(self._camera_index)
        if not self._camera.isOpened():
            for idx in [0, 2, 1, 4]:
                self._camera = cv2.VideoCapture(idx)
                if self._camera.isOpened():
                    self._camera_index = idx
                    break
        if self._camera.isOpened():
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"    ✓ Kamera {self._camera_index}")
        else:
            print("    ⚠ Keine Kamera (läuft ohne Bilder)")
            self._camera = None

        # Torque ausschalten → Arm per Hand führbar
        print("  [3/3] Torque Lock AUS (Arm frei bewegbar)...")
        self._arm.set_torque(enable=False)
        time.sleep(0.5)
        print("    ✓ Arm ist jetzt frei bewegbar!")

        print("\n  Steuerung:")
        print("    SPACE → Wegpunkt aufzeichnen")
        print("    O/C   → Greifer öffnen/schließen")
        print("    L/K   → LED an/aus")
        print("    W     → Wartezeit einfügen")
        print("    R     → Aufnahme Start/Stop")
        print("    P     → Skript abspielen")
        print("    Q     → Beenden")
        if self._continuous:
            print(f"\n    [Kontinuierlich-Modus: {self.POLL_HZ} Hz Aufzeichnung]")
        print()
        return True

    def _create_session(self):
        """Erstellt neues Aufnahme-Verzeichnis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._output_dir / f"session_{timestamp}"
        self._images_dir = self._session_dir / "frames"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._script_path = self._session_dir / f"program_{timestamp}.roarm"
        self._commands = []
        self._frame_count = 0
        print(f"\n  📁 Session: {self._session_dir}")

    def _save_frame(self, frame) -> str:
        """Speichert Kamerabild, gibt Dateinamen zurück."""
        if frame is None:
            return ""
        self._frame_count += 1
        filename = f"frame_{self._frame_count:06d}.jpg"
        filepath = self._images_dir / filename
        cv2.imwrite(str(filepath), frame)
        return filename

    def _get_arm_position(self) -> dict:
        """Holt aktuelle Gelenkwinkel vom Arm (Feedback)."""
        status = self._arm.get_status()
        if status:
            import math
            return {
                "b": round(math.degrees(status.base_rad), 2),
                "s": round(math.degrees(status.shoulder_rad), 2),
                "e": round(math.degrees(status.elbow_rad), 2),
                "h": round(math.degrees(status.eoat_rad), 2),
            }
        return None

    def _record_waypoint(self, frame=None):
        """Zeichnet aktuellen Wegpunkt auf."""
        pos = self._get_arm_position()
        if pos is None:
            print("    ⚠ Konnte Position nicht lesen!")
            return

        # MOVE Befehl
        cmd = f"MOVE b={pos['b']} s={pos['s']} e={pos['e']} h={pos['h']}"
        self._commands.append(cmd)

        # Kamerabild
        if frame is not None:
            filename = self._save_frame(frame)
            if filename:
                self._commands.append(f"FRAME {filename}")

        print(f"    📍 Waypoint: B={pos['b']:.1f} S={pos['s']:.1f} E={pos['e']:.1f} H={pos['h']:.1f}")

    def _save_script(self):
        """Speichert die .roarm Skriptdatei."""
        if not self._commands:
            print("    ⚠ Nichts aufgezeichnet!")
            return

        lines = [
            f"# RoArm-M2-S Teach Recording",
            f"# Erstellt: {datetime.now().isoformat()}",
            f"# Frames: {self._frame_count}",
            f"#CONFIG speed_scale={self._speed_scale}",
            f"#CONFIG spd={self._spd}",
            f"#CONFIG acc={self._acc}",
            "",
        ]
        lines.extend(self._commands)

        with open(self._script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

        print(f"\n  💾 Gespeichert: {self._script_path}")
        print(f"     {len(self._commands)} Befehle, {self._frame_count} Frames")

    def _play_script(self, script_path: Path = None):
        """Spielt ein .roarm Skript ab."""
        path = script_path or self._script_path
        if not path or not path.exists():
            print("    ⚠ Kein Skript zum Abspielen!")
            return

        print(f"\n  ▶ Abspielen: {path.name}")

        # Torque wieder an
        self._arm.set_torque(enable=True)
        time.sleep(0.3)

        # Config defaults
        spd = self._spd
        acc = self._acc
        speed_scale = self._speed_scale

        with open(path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") and not line.startswith("#CONFIG"):
                continue

            if line.startswith("#CONFIG"):
                # Parse config
                _, kv = line.split(" ", 1)
                key, val = kv.split("=")
                if key == "speed_scale":
                    speed_scale = float(val)
                elif key == "spd":
                    spd = int(val)
                elif key == "acc":
                    acc = int(val)
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "MOVE":
                # Parse: MOVE b=X s=X e=X h=X
                vals = {}
                for p in parts[1:]:
                    k, v = p.split("=")
                    vals[k] = float(v)

                actual_spd = int(spd * speed_scale)
                actual_acc = int(acc * speed_scale)
                self._arm.move_joints_degrees(
                    b=vals.get("b", 0),
                    s=vals.get("s", 0),
                    e=vals.get("e", 90),
                    h=vals.get("h", 180),
                    spd=actual_spd,
                    acc=actual_acc
                )
                # Warte proportional zur Geschwindigkeit
                time.sleep(0.3 / speed_scale)

            elif cmd == "GRIPPER":
                action = parts[1] if len(parts) > 1 else "OPEN"
                if action == "OPEN":
                    self._arm.gripper_open()
                else:
                    self._arm.gripper_close()
                time.sleep(0.5)

            elif cmd == "LED":
                brightness = int(parts[1]) if len(parts) > 1 else 0
                self._arm.set_led(brightness)

            elif cmd == "WAIT":
                wait_time = float(parts[1]) if len(parts) > 1 else 1.0
                wait_time /= speed_scale
                time.sleep(wait_time)

            elif cmd == "FRAME":
                pass  # Frames sind nur für KI-Training, nicht für Playback

            # Live-Anzeige während Playback
            if self._camera:
                self._camera.grab()
                ret, frame = self._camera.retrieve()
                if ret:
                    cv2.putText(frame, f"REPLAY: {line[:50]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow(self._window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        print("  ■ Abspielen beendet")

        # Torque wieder aus für weiteres Teaching
        self._arm.set_torque(enable=False)
        time.sleep(0.3)

    def run(self):
        """Hauptschleife."""
        if not self.setup():
            return

        self._running = True
        last_record_time = 0
        record_interval = 1.0 / self.POLL_HZ

        try:
            while self._running:
                # Frame holen
                frame = None
                if self._camera:
                    self._camera.grab()
                    ret, frame = self._camera.retrieve()
                    if not ret:
                        frame = None

                # Kontinuierliche Aufnahme
                now = time.time()
                if self._recording and self._continuous:
                    if now - last_record_time >= record_interval:
                        self._record_waypoint(frame)
                        last_record_time = now

                # Anzeige
                if frame is not None:
                    # Status-Overlay
                    status_color = (0, 0, 255) if self._recording else (200, 200, 200)
                    status_text = "● REC" if self._recording else "○ IDLE"
                    cv2.putText(frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                    if self._recording:
                        cv2.putText(frame, f"Cmds: {len(self._commands)}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.putText(frame, "R=Rec O/C=Grip L/K=LED W=Wait P=Play Q=Quit",
                                (10, frame.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                    cv2.imshow(self._window_name, frame)

                # Tasteneingabe
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    self._running = False

                elif key == ord('r'):
                    if not self._recording:
                        self._create_session()
                        self._recording = True
                        last_record_time = time.time()
                        print("\n  ● AUFNAHME GESTARTET")
                    else:
                        self._recording = False
                        self._save_script()
                        print("  ■ AUFNAHME GESTOPPT")

                elif key == ord(' '):
                    # Manueller Wegpunkt (auch ohne kontinuierliche Aufnahme)
                    if self._recording:
                        self._record_waypoint(frame)

                elif key == ord('o'):
                    if self._recording:
                        self._commands.append("GRIPPER OPEN")
                        print("    🤏 Greifer ÖFFNEN")
                    self._arm.gripper_open()

                elif key == ord('c'):
                    if self._recording:
                        self._commands.append("GRIPPER CLOSE")
                        print("    🤏 Greifer SCHLIESSEN")
                    self._arm.gripper_close()

                elif key == ord('l'):
                    if self._recording:
                        self._commands.append("LED 255")
                        print("    💡 LED AN")
                    self._arm.set_led(255)

                elif key == ord('k'):
                    if self._recording:
                        self._commands.append("LED 0")
                        print("    💡 LED AUS")
                    self._arm.set_led(0)

                elif key == ord('w'):
                    if self._recording:
                        self._commands.append(f"WAIT {self._wait_seconds}")
                        print(f"    ⏱ WAIT {self._wait_seconds}s")

                elif key == ord('p'):
                    if not self._recording:
                        # Finde letztes Skript
                        scripts = sorted(self._output_dir.rglob("*.roarm"))
                        if scripts:
                            self._play_script(scripts[-1])
                        else:
                            print("    ⚠ Kein Skript gefunden!")

        except KeyboardInterrupt:
            print("\n  [Abgebrochen]")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Aufräumen."""
        if self._recording:
            self._recording = False
            self._save_script()

        if self._arm:
            self._arm.set_torque(enable=True)
            time.sleep(0.3)
            self._arm.park()
            time.sleep(1.0)
            self._arm.set_led(0)
            self._arm.disconnect()

        if self._camera:
            self._camera.release()

        cv2.destroyAllWindows()
        print("\n  ✓ Beendet")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoArm-M2-S Teach & Record")
    parser.add_argument("--port", type=str, default=None, help="Serieller Port")
    parser.add_argument("--camera", type=int, default=2, help="Kamera-Index")
    parser.add_argument("--output", type=str, default="teach_recordings", help="Output-Verzeichnis")
    parser.add_argument("--manual", action="store_true",
                        help="Nur bei SPACE aufzeichnen (nicht kontinuierlich)")
    parser.add_argument("--hz", type=int, default=10, help="Aufnahme-Frequenz (Hz)")
    parser.add_argument("--wait", type=float, default=1.0, help="Standard-Wartezeit (s)")
    args = parser.parse_args()

    recorder = TeachRecorder(
        port=args.port,
        camera_index=args.camera,
        output_dir=args.output,
        continuous=not args.manual,
        wait_seconds=args.wait,
    )
    recorder.POLL_HZ = args.hz
    recorder.run()


if __name__ == "__main__":
    main()

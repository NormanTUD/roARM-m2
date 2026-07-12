#!/usr/bin/env python3
"""robot.py - RoArm-M2-S Basis-Verbindungsklasse

Zentrale Klasse für die Kommunikation mit dem RoArm-M2-S.
Alle anderen Module (teach, play, calibrate) importieren von hier.

Enthält:
- RoArmConnection: Serielle Kommunikation + alle Arm-Befehle
- Port-Erkennung
- Einheiten-Konvertierung
- Gemeinsame Konstanten
"""

import json
import time
import math
import threading
import serial
import serial.tools.list_ports
from typing import Optional


# ============================================================
# GEMEINSAME KONSTANTEN
# ============================================================

BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1

START_POSITION_DEG = {
    "b": 0.0,
    "s": 0.0,
    "e": 90.0,
    "h": 180.0,
}

POSITION_TOLERANCE = 1.0


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_arm_port() -> Optional[str]:
    """Auto-Detect des seriellen Ports für den RoArm-M2-S.
    
    Sucht nach USB-Serial-Adaptern (CH340, CP210x, FTDI) oder
    typischen Linux-Gerätenamen (ttyUSB, ttyACM).
    
    Returns:
        Port-String oder None wenn nichts gefunden.
    """
    ports = list(serial.tools.list_ports.comports())
    
    # Priorität 1: Bekannte USB-Serial-Chips
    for p in ports:
        desc = (p.description or "").lower()
        if any(x in desc for x in ["usb", "ch340", "cp210", "ftdi"]):
            return p.device
    
    # Priorität 2: Linux-typische Namen
    for p in ports:
        if "ttyUSB" in p.device or "ttyACM" in p.device:
            return p.device
    
    # Fallback: Erster verfügbarer Port
    if ports:
        return ports[0].device
    
    return None


def rad_to_deg(rad: float) -> float:
    """Radiant zu Grad."""
    return rad * (180.0 / math.pi)


def deg_to_rad(deg: float) -> float:
    """Grad zu Radiant."""
    return deg * (math.pi / 180.0)


def clear_line():
    """Löscht die aktuelle Terminal-Zeile."""
    import sys
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


# ============================================================
# ARM-KOMMUNIKATION (Basis-Klasse)
# ============================================================

class RoArmConnection:
    """Serielle Verbindung zum RoArm-M2-S.
    
    Kapselt alle Low-Level-Kommunikation:
    - JSON-Befehle senden/empfangen
    - Position lesen (raw/deg/stable/averaged)
    - Bewegungsbefehle (normal/fast)
    - Torque on/off
    - Gripper open/close
    - Wait-until-settled (aktives Polling)
    
    Usage:
        arm = RoArmConnection("/dev/ttyUSB0")
        arm.move_to(0, 0, 90, 180, spd=20, acc=10)
        pos = arm.read_position_deg()
        arm.close()
    """

    def __init__(self, port: str, baudrate: int = BAUDRATE, 
                 timeout: float = SERIAL_TIMEOUT,
                 init_delay: float = 0.3):
        """
        Args:
            port: Serieller Port (z.B. "/dev/ttyUSB0")
            baudrate: Baudrate (default: 115200)
            timeout: Serial read timeout in Sekunden
            init_delay: Wartezeit nach Verbindungsaufbau
        """
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._ser.setRTS(False)
        self._ser.setDTR(False)
        self._lock = threading.Lock()
        time.sleep(init_delay)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    # ----------------------------------------------------------
    # LOW-LEVEL KOMMUNIKATION
    # ----------------------------------------------------------

    def send_cmd(self, cmd: dict, timeout: float = 0.2) -> str:
        """Sendet einen JSON-Befehl und wartet auf Antwort.
        
        Args:
            cmd: Dictionary das als JSON gesendet wird
            timeout: Max. Wartezeit auf Antwort in Sekunden
            
        Returns:
            Antwort-String oder "" wenn keine Antwort
        """
        with self._lock:
            self._ser.reset_input_buffer()
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()
            time.sleep(0.01)

            response = ""
            deadline = time.time() + timeout
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
        """Sendet Befehl OHNE auf Antwort zu warten (für Streaming).
        
        Achtung: Kein Feedback ob Befehl angekommen ist!
        Nur für High-Frequency-Streaming verwenden.
        """
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()

    # ----------------------------------------------------------
    # POSITION LESEN
    # ----------------------------------------------------------

    def read_position_raw(self) -> Optional[dict]:
        """Liest die aktuelle Position als Radiant-Dictionary.
        
        Returns:
            Dict mit Keys "b", "s", "e", "t"/"h" in Radiant,
            oder None bei Fehler.
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

    def read_position_deg(self) -> Optional[dict]:
        """Liest die aktuelle Position in Grad.
        
        Returns:
            Dict {"b": float, "s": float, "e": float, "h": float} in Grad,
            oder None bei Fehler.
        """
        raw = self.read_position_raw()
        if raw is None:
            return None
        return {
            "b": round(rad_to_deg(raw["b"]), 2),
            "s": round(rad_to_deg(raw["s"]), 2),
            "e": round(rad_to_deg(raw["e"]), 2),
            "h": round(rad_to_deg(raw.get("t", raw.get("h", 0))), 2),
        }

    def read_position_deg_single(self) -> Optional[dict]:
        """Schneller einzelner Read ohne Stabilisierung.
        
        Für Recording-Loops wo Geschwindigkeit wichtiger ist als
        absolute Genauigkeit.
        """
        return self.read_position_deg()

    def read_position_stable(self, num_reads: int = 3, 
                              interval: float = 0.015) -> Optional[dict]:
        """Liest Position mehrfach und gibt den letzten gültigen Wert zurück.
        
        Filtert einzelne Ausreißer durch mehrfaches Lesen.
        
        Args:
            num_reads: Anzahl der Leseversuche
            interval: Pause zwischen Reads in Sekunden
        """
        last_valid = None
        for i in range(num_reads):
            raw = self.read_position_raw()
            if raw and "b" in raw:
                last_valid = raw
            if i < num_reads - 1:
                time.sleep(interval)
        
        if last_valid is None:
            return None
        return {
            "b": round(rad_to_deg(last_valid["b"]), 2),
            "s": round(rad_to_deg(last_valid["s"]), 2),
            "e": round(rad_to_deg(last_valid["e"]), 2),
            "h": round(rad_to_deg(last_valid.get("t", last_valid.get("h", 0))), 2),
        }

    def read_position_averaged(self, n: int = 8, 
                                interval: float = 0.04) -> Optional[dict]:
        """Liest n Positionen und gibt den Mittelwert zurück.
        
        Für Kalibrierung: Reduziert Rauschen durch Mittelung.
        
        Args:
            n: Anzahl Messungen
            interval: Pause zwischen Messungen
            
        Returns:
            Dict mit gemittelten Werten + Standardabweichung pro Gelenk,
            oder None wenn keine Reads möglich.
        """
        readings = []
        for _ in range(n):
            pos = self.read_position_deg()
            if pos:
                readings.append(pos)
            time.sleep(interval)

        if not readings:
            return None

        # Mittelwert + Standardabweichung berechnen
        avg = {}
        for joint in ["b", "s", "e", "h"]:
            values = [r[joint] for r in readings]
            avg[joint] = round(sum(values) / len(values), 3)
            # Standardabweichung (ohne numpy-Abhängigkeit)
            mean = avg[joint]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            avg[f"{joint}_std"] = round(math.sqrt(variance), 4)
        avg["n_samples"] = len(readings)
        return avg

    # ----------------------------------------------------------
    # BEWEGUNGSBEFEHLE
    # ----------------------------------------------------------

    def move_to(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                spd: int = 20, acc: int = 10):
        """Fährt zur angegebenen Position (mit Antwort-Warten).
        
        Args:
            b_deg: Base-Winkel in Grad
            s_deg: Shoulder-Winkel in Grad
            e_deg: Elbow-Winkel in Grad
            h_deg: Hand/Gripper-Winkel in Grad
            spd: Geschwindigkeit (1-50)
            acc: Beschleunigung (1-30)
        """
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
        """Schneller Move ohne auf Antwort zu warten (für Streaming).
        
        Identisch zu move_to() aber verwendet send_cmd_fast().
        Nur für High-Frequency-Streaming verwenden!
        """
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

    # ----------------------------------------------------------
    # TORQUE CONTROL
    # ----------------------------------------------------------

    def torque_on(self):
        """Schaltet Torque für alle Servos ein (Arm hält Position)."""
        self.send_cmd({"T": 210, "cmd": 1})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 1})
            time.sleep(0.02)

    def torque_off(self):
        """Schaltet Torque für alle Servos aus (Arm wird schlaff)."""
        self.send_cmd({"T": 210, "cmd": 0})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 0})
            time.sleep(0.02)

    def torque_on_fast(self):
        """Schnelles Torque-an ohne individuelle Servo-Befehle.
        
        Für Recording-Loop wo Geschwindigkeit wichtig ist.
        Weniger zuverlässig als torque_on() aber schneller.
        """
        self.send_cmd({"T": 210, "cmd": 1})

    def torque_off_fast(self):
        """Schnelles Torque-aus ohne individuelle Servo-Befehle."""
        self.send_cmd({"T": 210, "cmd": 0})

    # ----------------------------------------------------------
    # GRIPPER
    # ----------------------------------------------------------

    def gripper_open(self):
        """Öffnet den Gripper."""
        self.send_cmd({"T": 106, "cmd": 1.08, "spd": 50, "acc": 20})

    def gripper_close(self):
        """Schließt den Gripper."""
        self.send_cmd({"T": 106, "cmd": 3.14, "spd": 50, "acc": 20})

    # ----------------------------------------------------------
    # WARTEN / SETTLING
    # ----------------------------------------------------------

    def wait_until_settled(self, tolerance_deg: float = 0.3, 
                           stable_count: int = 5,
                           poll_interval: float = 0.15, 
                           timeout: float = 15.0) -> dict:
        """Wartet bis der Arm stillsteht.
        
        Pollt die Position und prüft ob sich der Arm weniger als
        tolerance_deg bewegt hat für stable_count aufeinanderfolgende Reads.
        
        Args:
            tolerance_deg: Max. Änderung die als "still" gilt
            stable_count: Wie viele stabile Reads hintereinander nötig
            poll_interval: Pause zwischen Reads
            timeout: Maximale Wartezeit
            
        Returns:
            Dict mit:
            - "pos": Letzte gelesene Position
            - "settle_time_s": Wie lange es gedauert hat
            - "readings": Alle gelesenen Positionen
            - "timeout": True wenn Timeout erreicht (optional)
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

        # Timeout erreicht
        settle_time = time.time() - start
        return {
            "pos": last_pos,
            "settle_time_s": settle_time,
            "readings": all_readings,
            "timeout": True,
        }

    # ----------------------------------------------------------
    # VERBINDUNG
    # ----------------------------------------------------------

    def close(self):
        """Schließt die serielle Verbindung."""
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def serial(self) -> serial.Serial:
        """Zugriff auf das Serial-Objekt (für Safety-Layer Buffer-Flush)."""
        return self._ser

# ============================================================
# OPTIONAL: 3D-VISUALISIERUNG
# ============================================================

def create_visualized_arm(port: str = None, **kwargs) -> "VisualizingArm":
    """
    Erstellt eine RoArmConnection MIT 3D-Visualisierung.

    Jede Bewegung wird automatisch im 3D-Plot angezeigt.

    Usage:
        from robot import create_visualized_arm
        arm = create_visualized_arm()
        arm.move_to(30, 10, 60, 180)  # Wird live im 3D-Plot gezeigt!
        arm.close()
    """
    from visualize import VisualizingArm

    port = port or find_arm_port()
    if port is None:
        raise RuntimeError("Kein serieller Port gefunden!")

    raw_arm = RoArmConnection(port, **kwargs)
    return VisualizingArm(raw_arm)

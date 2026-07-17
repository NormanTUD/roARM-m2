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

import os
import sys
import json
import time
import math
import threading
import serial
import serial.tools.list_ports
from typing import Optional

# Am Anfang von robot.py hinzufügen (nach den bestehenden Imports):

import logging
from datetime import datetime

# ============================================================
# ZENTRALES COMMAND-LOGGING
# ============================================================

def setup_command_logger(log_dir: str = "logs") -> logging.Logger:
    """Richtet den zentralen Command-Logger ein.

    Loggt ALLES was an den Robot gesendet wird in eine
    menschenlesbare Datei mit Timestamp.

    Log-Format:
        2026-07-12 07:27:03.142 | SEND | {"T":122,"b":10.0,"s":5.0,"e":85.0,"h":180.0,"spd":20,"acc":10}
        2026-07-12 07:27:03.158 | RECV | {"T":1051,"b":0.17,"s":0.08,"e":1.57,"t":3.14}
        2026-07-12 07:27:03.200 | SEND_FAST | {"T":122,"b":10.5,...}
        2026-07-12 07:27:05.000 | NOTE | Torque ON (all servos)
    """
    from pathlib import Path
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"robot_commands_{timestamp}.log"

    logger = logging.getLogger("roarm.commands")
    logger.setLevel(logging.DEBUG)

    # Keine doppelten Handler
    if logger.handlers:
        return logger

    # Datei-Handler: Alles loggen
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Menschenlesbares Format mit Millisekunden
    fmt = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Optional: Auch auf Console (nur Warnungen+)
    if not os.environ.get("TEXTUAL_RUNNING"):
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.info(f"=== SESSION START === Port wird gleich verbunden")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Log-Datei: {log_file}")

    print(f"  📝 Command-Log: {log_file}")

    return logger


# Globaler Logger (wird beim ersten Import initialisiert)
_cmd_logger = setup_command_logger()

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
# MENSCHENLESBARE COMMAND-ABSTRAKTION
# ============================================================

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional


class CommandType(IntEnum):
    """Alle RoArm-M2-S Befehls-IDs mit lesbaren Namen."""
    READ_POSITION = 105
    GRIPPER_CONTROL = 106
    MOVE_JOINTS = 122
    TORQUE_ALL = 210
    TORQUE_SINGLE = 212


class TorqueState(IntEnum):
    """Torque ein/aus."""
    OFF = 0
    ON = 1


class GripperState:
    """Gripper-Positionen als benannte Konstanten."""
    OPEN = 1.08
    CLOSED = 3.14


@dataclass
class JointAngles:
    """Gelenkwinkel in Grad – menschenlesbar."""
    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 90.0
    hand: float = 180.0

    def to_cmd(self, speed: int = 20, acceleration: int = 10) -> dict:
        """Konvertiert zu RoArm JSON-Befehl."""
        return {
            "T": CommandType.MOVE_JOINTS,
            "b": round(self.base, 2),
            "s": round(self.shoulder, 2),
            "e": round(self.elbow, 2),
            "h": round(self.hand, 2),
            "spd": speed,
            "acc": acceleration,
        }

    @classmethod
    def from_raw(cls, data: dict) -> "JointAngles":
        """Erstellt JointAngles aus Raw-Response."""
        return cls(
            base=round(rad_to_deg(data["b"]), 2),
            shoulder=round(rad_to_deg(data["s"]), 2),
            elbow=round(rad_to_deg(data["e"]), 2),
            hand=round(rad_to_deg(data.get("t", data.get("h", 0))), 2),
        )

    def __str__(self):
        return (f"Base={self.base:.1f}° | Shoulder={self.shoulder:.1f}° | "
                f"Elbow={self.elbow:.1f}° | Hand={self.hand:.1f}°")


class RoArmHumanAPI:
    """Menschenlesbare High-Level API für den RoArm-M2-S.
    
    Statt:
        arm.send_cmd({"T": 210, "cmd": 0})
        arm.send_cmd({"T": 122, "b": 10.0, "s": 5.0, "e": 85.0, "h": 180.0, "spd": 20, "acc": 10})
    
    Jetzt:
        arm.torque(TorqueState.OFF)
        arm.move(JointAngles(base=10, shoulder=5, elbow=85, hand=180))
    
    Oder noch einfacher:
        arm.gripper_open()
        arm.move_to(base=10, shoulder=5, elbow=85)
        arm.torque_off()
    """

    def __init__(self, connection):
        self._conn = connection
        self._log = logging.getLogger("roarm.commands")

    # --- Torque ---

    def torque(self, state: TorqueState):
        """Setzt Torque für alle Servos.
        
        Args:
            state: TorqueState.ON oder TorqueState.OFF
        """
        action = "ON" if state == TorqueState.ON else "OFF"
        self._log.info(f"NOTE     | >>> TORQUE {action} (all servos)")
        self._conn.send_cmd({"T": CommandType.TORQUE_ALL, "cmd": int(state)})
        time.sleep(0.03)
        for servo_id in range(1, 5):
            self._conn.send_cmd({"T": CommandType.TORQUE_SINGLE, "id": servo_id, "cmd": int(state)})
            time.sleep(0.02)

    def torque_on(self):
        """Schaltet Torque für alle Servos ein.
        
        Sendet zuerst den globalen Torque-Befehl, dann einzeln
        an jeden Servo (1-4) für zuverlässiges Aktivieren.
        """
        self._log.info("NOTE     | >>> TORQUE ON (all servos)")
        self.send_cmd({"T": CommandType.TORQUE_ALL, "cmd": TorqueState.ON})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": CommandType.TORQUE_SINGLE, "id": sid, "cmd": TorqueState.ON})
            time.sleep(0.02)

    def torque_off(self):
        """Schaltet Torque für alle Servos aus.
        
        Sendet zuerst den globalen Torque-Befehl, dann einzeln
        an jeden Servo (1-4) für zuverlässiges Deaktivieren.
        Arm ist danach frei bewegbar.
        """
        self._log.info("NOTE     | >>> TORQUE OFF (all servos)")
        self.send_cmd({"T": CommandType.TORQUE_ALL, "cmd": TorqueState.OFF})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": CommandType.TORQUE_SINGLE, "id": sid, "cmd": TorqueState.OFF})
            time.sleep(0.02)

    # --- Gripper ---

    def gripper(self, position: float, speed: int = 50, acceleration: int = 20):
        """Gripper auf beliebige Position fahren.
        
        Args:
            position: GripperState.OPEN (1.08) oder GripperState.CLOSED (3.14)
                      oder beliebiger Wert dazwischen
            speed: Geschwindigkeit (1-50)
            acceleration: Beschleunigung (1-30)
        """
        state_name = "OPEN" if position <= 2.0 else "CLOSE"
        self._log.info(f"NOTE     | >>> GRIPPER {state_name} (pos={position})")
        self._conn.send_cmd({
            "T": CommandType.GRIPPER_CONTROL,
            "cmd": position,
            "spd": speed,
            "acc": acceleration,
        })

    def gripper_open(self):
        """Gripper öffnen."""
        self.gripper(GripperState.OPEN)

    def gripper_close(self):
        """Gripper schließen."""
        self.gripper(GripperState.CLOSED)

    # --- Bewegung ---

    def move(self, angles: JointAngles, speed: int = 20, acceleration: int = 10):
        """Fährt zur angegebenen Gelenkposition.
        
        Args:
            angles: JointAngles-Objekt mit base/shoulder/elbow/hand
            speed: Geschwindigkeit (1-50)
            acceleration: Beschleunigung (1-30)
        """
        self._conn.send_cmd(angles.to_cmd(speed, acceleration))

    def move_to(self, base: float = 0, shoulder: float = 0, 
                elbow: float = 90, hand: float = 180,
                speed: int = 20, acceleration: int = 10):
        """Fährt zur Position mit benannten Parametern.
        
        Beispiel:
            arm.move_to(base=30, shoulder=10, elbow=60, hand=180)
            arm.move_to(base=45)  # Nur Base drehen, Rest Default
        """
        angles = JointAngles(base=base, shoulder=shoulder, 
                            elbow=elbow, hand=hand)
        self.move(angles, speed, acceleration)

    def move_fast(self, base: float = 0, shoulder: float = 0,
                  elbow: float = 90, hand: float = 180,
                  speed: int = 50, acceleration: int = 30):
        """Schnelle Bewegung ohne Antwort-Warten (für Streaming)."""
        angles = JointAngles(base=base, shoulder=shoulder,
                            elbow=elbow, hand=hand)
        self._conn.send_cmd_fast(angles.to_cmd(speed, acceleration))

    # --- Position lesen ---

    def read_position(self) -> Optional[JointAngles]:
        """Liest aktuelle Gelenkwinkel als JointAngles-Objekt."""
        raw = self._conn.read_position_raw()
        if raw is None:
            return None
        return JointAngles.from_raw(raw)

    def where_am_i(self) -> str:
        """Gibt aktuelle Position als lesbaren String zurück."""
        pos = self.read_position()
        if pos is None:
            return "⚠️  Position nicht lesbar!"
        return f"📍 {pos}"

    # --- Warten ---

    def wait_until_stopped(self, tolerance_deg: float = 0.3,
                           timeout: float = 15.0) -> dict:
        """Wartet bis der Arm stillsteht."""
        return self._conn.wait_until_settled(
            tolerance_deg=tolerance_deg, timeout=timeout
        )

    # --- Kontext-Manager ---

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

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
        
        # Logger referenzieren
        self._log = logging.getLogger("roarm.commands")
        self._log.info(f"=== CONNECTED === port={port} baud={baudrate}")

    # ----------------------------------------------------------
    # LOW-LEVEL KOMMUNIKATION
    # ----------------------------------------------------------

    def send_cmd(self, cmd: dict, timeout: float = 0.2) -> str:
        """Sendet einen JSON-Befehl und wartet auf Antwort."""
        with self._lock:
            self._ser.reset_input_buffer()
            msg = json.dumps(cmd, separators=(',', ':'))
            
            # >>> LOGGING: Was gesendet wird
            self._log.info(f"SEND     | {msg}")
            
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
                        
                        # >>> LOGGING: Was empfangen wird
                        self._log.info(f"RECV     | {line}")
                        
                        if '"T":1051' in line or '"b"' in line:
                            return line
                else:
                    time.sleep(0.005)
            
            if not response:
                self._log.warning(f"TIMEOUT  | Keine Antwort auf: {msg}")
            
            return response

    def send_cmd_fast(self, cmd: dict):
        """Sendet Befehl OHNE auf Antwort zu warten (für Streaming)."""
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':'))
            
            # >>> LOGGING: Fast-Send
            self._log.debug(f"SEND_FAST| {msg}")
            
            self._ser.write(msg.encode() + b'\n')
            self._ser.flush()

    # ----------------------------------------------------------
    # POSITION LESEN
    # ----------------------------------------------------------

    def read_position_raw(self) -> Optional[dict]:
        """Liest die aktuelle Position als Radiant-Dictionary.
        
        Sendet den READ_POSITION-Befehl und parst die JSON-Antwort.
        
        Returns:
            Dict mit Keys "b", "s", "e", "t"/"h" in Radiant,
            oder None bei Fehler/Timeout.
        """
        resp = self.send_cmd({"T": CommandType.READ_POSITION})
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
        
        Sendet einen MOVE_JOINTS-Befehl mit den angegebenen Gelenkwinkeln.
        Wartet auf Bestätigung vom Controller.
        
        Args:
            b_deg: Base-Winkel in Grad
            s_deg: Shoulder-Winkel in Grad
            e_deg: Elbow-Winkel in Grad
            h_deg: Hand/Gripper-Winkel in Grad
            spd: Geschwindigkeit (1-50)
            acc: Beschleunigung (1-30)
        """
        cmd = {
            "T": CommandType.MOVE_JOINTS,
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
        Nur für High-Frequency-Streaming verwenden wo Latenz
        wichtiger ist als Bestätigung!
        """
        cmd = {
            "T": CommandType.MOVE_JOINTS,
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
        """Schaltet Torque für alle Servos ein."""
        self._log.info("NOTE     | >>> TORQUE ON (all servos)")
        self.send_cmd({"T": 210, "cmd": 1})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 1})
            time.sleep(0.02)

    def torque_off(self):
        """Schaltet Torque für alle Servos aus."""
        self._log.info("NOTE     | >>> TORQUE OFF (all servos)")
        self.send_cmd({"T": 210, "cmd": 0})
        time.sleep(0.03)
        for sid in range(1, 5):
            self.send_cmd({"T": 212, "id": sid, "cmd": 0})
            time.sleep(0.02)

    def torque_on_fast(self, exclude_gripper=False):
        if exclude_gripper:
            # Nur Servos 1-3
            for sid in range(1, 4):
                self.send_cmd({"T": 212, "id": sid, "cmd": 1})
        else:
            self.send_cmd({"T": 210, "cmd": 1})

    def torque_off_fast(self, exclude_gripper=False):
        if exclude_gripper:
            for sid in range(1, 4):
                self.send_cmd({"T": 212, "id": sid, "cmd": 0})
        else:
            self.send_cmd({"T": 210, "cmd": 0})

    # ----------------------------------------------------------
    # GRIPPER
    # ----------------------------------------------------------

    def gripper_open(self):
        """Öffnet den Gripper vollständig.

        Schaltet zuerst Torque für den Gripper-Servo (ID 4) explizit ein,
        dann fährt den Gripper-Servo auf die OPEN-Position (1.08 rad).
        """
        self._log.info("NOTE     | >>> GRIPPER OPEN")
        # Gripper-Servo (ID 4) Torque explizit einschalten
        self.send_cmd({"T": 212, "id": 4, "cmd": 1})
        time.sleep(0.02)
        self.send_cmd({
            "T": CommandType.GRIPPER_CONTROL,
            "cmd": GripperState.OPEN,
            "spd": 0,
            "acc": 0,
        })

    def gripper_close(self):
        """Schließt den Gripper vollständig.

        Schaltet zuerst Torque für den Gripper-Servo (ID 4) explizit ein,
        dann fährt den Gripper-Servo auf die CLOSED-Position (3.14 rad).
        """
        self._log.info("NOTE     | >>> GRIPPER CLOSE")
        # Gripper-Servo (ID 4) Torque explizit einschalten
        self.send_cmd({"T": 212, "id": 4, "cmd": 1})
        time.sleep(0.02)
        self.send_cmd({
            "T": CommandType.GRIPPER_CONTROL,
            "cmd": GripperState.CLOSED,
            "spd": 0,
            "acc": 0,
        })

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
        self._log.info("=== DISCONNECTED ===")
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

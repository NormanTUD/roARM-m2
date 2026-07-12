#!/usr/bin/env python3
"""sim_robot.py - Simulierter RoArm-M2-S für automatisierte Tests

Ersetzt RoArmConnection durch eine Software-Simulation:
- Simuliert Servo-Positionen mit realistischem Verhalten
- Konfigurierbare Latenz, Rauschen, Drift
- Fehlersimulation (Timeout, Müll-Reads, Stall)
- Logging aller Befehle für Assertions in Tests
"""

import time
import math
import json
import threading
import random
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """Konfiguration der Simulation."""
    
    # Bewegungssimulation
    max_velocity_deg_s: float = 180.0      # Max Geschwindigkeit pro Gelenk
    acceleration_deg_s2: float = 300.0     # Beschleunigung
    position_noise_deg: float = 0.05       # Gaußsches Rauschen auf Position-Reads
    settle_noise_deg: float = 0.02         # Rauschen nach Settling
    
    # Timing
    read_latency_s: float = 0.01           # Latenz bei Position-Read
    command_latency_s: float = 0.005       # Latenz bei Befehl-Senden
    
    # Fehler-Simulation (Wahrscheinlichkeiten pro Aufruf)
    read_failure_probability: float = 0.0   # Wahrscheinlichkeit für None-Read
    garbage_read_probability: float = 0.0   # Wahrscheinlichkeit für Müll-Werte
    stall_probability: float = 0.0          # Wahrscheinlichkeit für Stall
    
    # Drift/Offset (simuliert mechanische Ungenauigkeiten)
    offset_b: float = 0.0
    offset_s: float = 0.0
    offset_e: float = 0.0
    offset_h: float = 0.0
    
    # Gravity-Effekt (Torque-off Durchhängen)
    gravity_droop_s_deg: float = -2.0      # Shoulder hängt durch bei Torque-off
    gravity_droop_e_deg: float = -1.5      # Elbow hängt durch bei Torque-off


@dataclass
class CommandLog:
    """Ein geloggter Befehl."""
    timestamp: float
    command_type: str       # "move", "move_fast", "torque_on", "torque_off", etc.
    params: dict = field(default_factory=dict)
    

class SimulatedArm:
    """
    Simulierter RoArm-M2-S.
    
    Drop-in Replacement für RoArmConnection mit identischem Interface.
    Simuliert:
    - Servo-Positionen mit Trägheit und Rauschen
    - Torque on/off Verhalten
    - Gripper-Zustände
    - Settling-Dynamik
    - Konfigurierbare Fehler
    
    Usage:
        arm = SimulatedArm()  # Statt RoArmConnection(port)
        arm.move_to(0, 0, 90, 180, spd=20, acc=10)
        pos = arm.read_position_deg()
    """
    
    def __init__(self, config: SimConfig = None, initial_position: dict = None):
        self.config = config or SimConfig()
        self.port = "/dev/simulated"
        
        # Aktuelle Position (simuliert)
        self._position = initial_position or {
            "b": 0.0, "s": 0.0, "e": 90.0, "h": 180.0
        }
        
        # Zielposition (wohin sich der Arm bewegt)
        self._target = self._position.copy()
        
        # Zustand
        self._torque_on = True
        self._gripper_open = True
        self._is_moving = False
        self._move_start_time = 0.0
        self._move_duration = 0.0
        self._move_start_pos = self._position.copy()
        
        # Logging
        self._command_log: list[CommandLog] = []
        self._lock = threading.Lock()
        
        # Simulations-Thread
        self._running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()
        
        # Simuliertes Serial-Objekt (für Kompatibilität)
        self._ser = _FakeSerial()
    
    # ----------------------------------------------------------
    # SIMULATION ENGINE
    # ----------------------------------------------------------
    
    def _simulation_loop(self):
        """Hintergrund-Thread der die Arm-Bewegung simuliert."""
        dt = 0.005  # 200 Hz interne Simulation
        while self._running:
            time.sleep(dt)
            with self._lock:
                self._update_position(dt)
    
    def _update_position(self, dt: float):
        """Aktualisiert die simulierte Position basierend auf Ziel."""
        if not self._torque_on:
            # Bei Torque-off: Gravity-Effekt
            self._position["s"] += self.config.gravity_droop_s_deg * dt * 0.1
            self._position["e"] += self.config.gravity_droop_e_deg * dt * 0.1
            return
        
        if not self._is_moving:
            return
        
        # Einfache lineare Interpolation mit Geschwindigkeitslimit
        all_arrived = True
        for joint in ["b", "s", "e", "h"]:
            diff = self._target[joint] - self._position[joint]
            if abs(diff) < 0.01:
                self._position[joint] = self._target[joint]
                continue
            
            all_arrived = False
            # Geschwindigkeit basierend auf spd-Parameter (vereinfacht)
            max_step = self.config.max_velocity_deg_s * dt
            step = max(-max_step, min(max_step, diff))
            self._position[joint] += step
        
        if all_arrived:
            self._is_moving = False
    
    def _add_noise(self, pos: dict) -> dict:
        """Fügt realistisches Rauschen zur Position hinzu."""
        noise = self.config.position_noise_deg
        return {
            joint: round(pos[joint] + random.gauss(0, noise), 2)
            for joint in ["b", "s", "e", "h"]
        }
    
    def _apply_offset(self, pos: dict) -> dict:
        """Wendet simulierten mechanischen Offset an."""
        return {
            "b": pos["b"] + self.config.offset_b,
            "s": pos["s"] + self.config.offset_s,
            "e": pos["e"] + self.config.offset_e,
            "h": pos["h"] + self.config.offset_h,
        }
    
    def _log(self, cmd_type: str, params: dict = None):
        """Loggt einen Befehl."""
        self._command_log.append(CommandLog(
            timestamp=time.time(),
            command_type=cmd_type,
            params=params or {},
        ))
    
    # ----------------------------------------------------------
    # PUBLIC API (identisch zu RoArmConnection)
    # ----------------------------------------------------------
    
    def send_cmd(self, cmd: dict, timeout: float = 0.2) -> str:
        """Simuliert send_cmd."""
        time.sleep(self.config.command_latency_s)
        self._log("send_cmd", cmd)
        
        # T=105 = Position lesen
        if cmd.get("T") == 105:
            pos = self._get_position_rad()
            if pos:
                return json.dumps(pos)
            return ""
        
        # T=122 = Move
        if cmd.get("T") == 122:
            self._execute_move(cmd)
            return '{"T":122,"ok":1}'
        
        # T=210 = Torque
        if cmd.get("T") == 210:
            self._torque_on = cmd.get("cmd", 1) == 1
            self._log("torque_on" if self._torque_on else "torque_off")
            return '{"T":210,"ok":1}'
        
        # T=212 = Individual servo torque
        if cmd.get("T") == 212:
            return '{"T":212,"ok":1}'
        
        # T=106 = Gripper
        if cmd.get("T") == 106:
            self._gripper_open = cmd.get("cmd", 1.08) < 2.0
            self._log("gripper_open" if self._gripper_open else "gripper_close")
            return '{"T":106,"ok":1}'
        
        return '{"ok":1}'
    
    def send_cmd_fast(self, cmd: dict):
        """Simuliert send_cmd_fast (kein Warten auf Antwort)."""
        time.sleep(self.config.command_latency_s * 0.5)
        self._log("send_cmd_fast", cmd)
        
        if cmd.get("T") == 122:
            self._execute_move(cmd)
        elif cmd.get("T") == 210:
            self._torque_on = cmd.get("cmd", 1) == 1
    
    def _execute_move(self, cmd: dict):
        """Führt einen Move-Befehl aus."""
        with self._lock:
            self._target = {
                "b": cmd.get("b", self._target["b"]),
                "s": cmd.get("s", self._target["s"]),
                "e": cmd.get("e", self._target["e"]),
                "h": cmd.get("h", self._target["h"]),
            }
            self._is_moving = True
            self._move_start_time = time.time()
            self._move_start_pos = self._position.copy()
    
    def _get_position_rad(self) -> Optional[dict]:
        """Gibt aktuelle Position in Radiant zurück (wie echter Arm)."""
        with self._lock:
            pos = self._apply_offset(self._position.copy())
        
        return {
            "b": pos["b"] * math.pi / 180.0,
            "s": pos["s"] * math.pi / 180.0,
            "e": pos["e"] * math.pi / 180.0,
            "t": pos["h"] * math.pi / 180.0,
        }
    
    # ----------------------------------------------------------
    # POSITION LESEN (identisch zu RoArmConnection)
    # ----------------------------------------------------------
    
    def read_position_raw(self) -> Optional[dict]:
        """Simuliert read_position_raw."""
        time.sleep(self.config.read_latency_s)
        
        # Fehler-Simulation
        if random.random() < self.config.read_failure_probability:
            return None
        
        if random.random() < self.config.garbage_read_probability:
            # Müll-Read (wie der 30120°-Bug)
            return {"b": random.uniform(-500, 500), "s": 999.9, "e": -999.9, "t": 0.0}
        
        return self._get_position_rad()
    
    def read_position_deg(self) -> Optional[dict]:
        """Simuliert read_position_deg."""
        raw = self.read_position_raw()
        if raw is None:
            return None
        
        pos = {
            "b": round(raw["b"] * 180.0 / math.pi, 2),
            "s": round(raw["s"] * 180.0 / math.pi, 2),
            "e": round(raw["e"] * 180.0 / math.pi, 2),
            "h": round(raw.get("t", raw.get("h", 0)) * 180.0 / math.pi, 2),
        }
        
        # Rauschen hinzufügen
        return self._add_noise(pos)
    
    def read_position_deg_single(self) -> Optional[dict]:
        """Schneller Read ohne Stabilisierung."""
        return self.read_position_deg()
    
    def read_position_stable(self, num_reads: int = 3,
                              interval: float = 0.015) -> Optional[dict]:
        """Simuliert read_position_stable."""
        last_valid = None
        for i in range(num_reads):
            pos = self.read_position_deg()
            if pos:
                last_valid = pos
            if i < num_reads - 1:
                time.sleep(interval)
        return last_valid
    
    def read_position_averaged(self, n: int = 8,
                                interval: float = 0.04) -> Optional[dict]:
        """Simuliert read_position_averaged."""
        readings = []
        for _ in range(n):
            pos = self.read_position_deg()
            if pos:
                readings.append(pos)
            time.sleep(interval * 0.1)  # Verkürzt für Tests
        
        if not readings:
            return None
        
        avg = {}
        for joint in ["b", "s", "e", "h"]:
            values = [r[joint] for r in readings]
            avg[joint] = round(sum(values) / len(values), 3)
            mean = avg[joint]
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            avg[f"{joint}_std"] = round(math.sqrt(variance), 4)
        avg["n_samples"] = len(readings)
        return avg
    
    # ----------------------------------------------------------
    # BEWEGUNGSBEFEHLE (identisch zu RoArmConnection)
    # ----------------------------------------------------------
    
    def move_to(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                spd: int = 20, acc: int = 10):
        """Simuliert move_to."""
        cmd = {"T": 122, "b": round(b_deg, 2), "s": round(s_deg, 2),
               "e": round(e_deg, 2), "h": round(h_deg, 2), "spd": spd, "acc": acc}
        self._log("move_to", cmd)
        self._execute_move(cmd)
    
    def move_to_fast(self, b_deg: float, s_deg: float, e_deg: float, h_deg: float,
                     spd: int = 50, acc: int = 30):
        """Simuliert move_to_fast."""
        cmd = {"T": 122, "b": round(b_deg, 2), "s": round(s_deg, 2),
               "e": round(e_deg, 2), "h": round(h_deg, 2), "spd": spd, "acc": acc}
        self._log("move_to_fast", cmd)
        self._execute_move(cmd)
    
    # ----------------------------------------------------------
    # TORQUE CONTROL
    # ----------------------------------------------------------
    
    def torque_on(self):
        """Simuliert torque_on."""
        self._torque_on = True
        self._log("torque_on")
    
    def torque_off(self):
        """Simuliert torque_off."""
        self._torque_on = False
        self._log("torque_off")
    
    def torque_on_fast(self):
        """Simuliert torque_on_fast."""
        self._torque_on = True
        self._log("torque_on_fast")
    
    def torque_off_fast(self):
        """Simuliert torque_off_fast."""
        self._torque_on = False
        self._log("torque_off_fast")
    
    # ----------------------------------------------------------
    # GRIPPER
    # ----------------------------------------------------------
    
    def gripper_open(self):
        """Simuliert gripper_open."""
        self._gripper_open = True
        self._log("gripper_open")
    
    def gripper_close(self):
        """Simuliert gripper_close."""
        self._gripper_open = False
        self._log("gripper_close")
    
    # ----------------------------------------------------------
    # WARTEN / SETTLING
    # ----------------------------------------------------------
    
    def wait_until_settled(self, tolerance_deg: float = 0.3,
                           stable_count: int = 5,
                           poll_interval: float = 0.15,
                           timeout: float = 15.0) -> dict:
        """Simuliert wait_until_settled (beschleunigt für Tests)."""
        start = time.time()
        all_readings = []
        
        # Warte bis Simulation die Zielposition erreicht hat
        # (verkürzt für Tests)
        max_wait = min(timeout, 2.0)  # Max 2s in Simulation
        stable = 0
        last_pos = None
        
        while time.time() - start < max_wait:
            pos = self.read_position_deg()
            if pos is None:
                time.sleep(poll_interval * 0.1)
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
                        return {
                            "pos": pos,
                            "settle_time_s": time.time() - start,
                            "readings": all_readings,
                        }
                else:
                    stable = 0
            
            last_pos = pos
            time.sleep(poll_interval * 0.1)  # Beschleunigt
        
        # Timeout - Position direkt auf Ziel setzen (für Tests)
        with self._lock:
            self._position = self._target.copy()
            self._is_moving = False
        
        pos = self.read_position_deg()
        return {
            "pos": pos,
            "settle_time_s": time.time() - start,
            "readings": all_readings,
            "timeout": True,
        }
    
    # ----------------------------------------------------------
    # VERBINDUNG
    # ----------------------------------------------------------
    
    def close(self):
        """Stoppt die Simulation."""
        self._running = False
        if self._sim_thread.is_alive():
            self._sim_thread.join(timeout=1.0)
        self._log("close")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def serial(self):
        """Fake Serial-Objekt für Kompatibilität."""
        return self._ser
    
    # ----------------------------------------------------------
    # TEST-HILFSFUNKTIONEN (nicht in RoArmConnection)
    # ----------------------------------------------------------
    
    def get_true_position(self) -> dict:
        """Gibt die echte Position OHNE Rauschen/Offset zurück (für Tests)."""
        with self._lock:
            return self._position.copy()
    
    def set_position(self, pos: dict):
        """Setzt die Position direkt (für Test-Setup)."""
        with self._lock:
            self._position = pos.copy()
            self._target = pos.copy()
            self._is_moving = False
    
    def get_command_log(self) -> list[CommandLog]:
        """Gibt alle geloggten Befehle zurück."""
        return self._command_log.copy()
    
    def clear_log(self):
        """Löscht das Command-Log."""
        self._command_log.clear()
    
    def get_commands_of_type(self, cmd_type: str) -> list[CommandLog]:
        """Filtert Log nach Befehlstyp."""
        return [c for c in self._command_log if c.command_type == cmd_type]
    
    @property
    def is_torque_on(self) -> bool:
        return self._torque_on
    
    @property
    def is_gripper_open(self) -> bool:
        return self._gripper_open
    
    @property
    def is_moving(self) -> bool:
        return self._is_moving
    
    def wait_for_arrival(self, timeout: float = 5.0):
        """Wartet bis der simulierte Arm das Ziel erreicht hat."""
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if not self._is_moving:
                    return True
            time.sleep(0.01)
        # Force-arrive für Tests
        with self._lock:
            self._position = self._target.copy()
            self._is_moving = False
        return False
    
    def inject_fault(self, fault_type: str, duration_s: float = 1.0):
        """Injiziert einen Fehler für Tests.
        
        fault_types:
        - "read_failure": Reads geben None zurück
        - "garbage": Reads geben Müll zurück
        - "stall": Arm bewegt sich nicht mehr
        """
        original_config = SimConfig(
            read_failure_probability=self.config.read_failure_probability,
            garbage_read_probability=self.config.garbage_read_probability,
        )
        
        if fault_type == "read_failure":
            self.config.read_failure_probability = 1.0
        elif fault_type == "garbage":
            self.config.garbage_read_probability = 1.0
        elif fault_type == "stall":
            self.config.max_velocity_deg_s = 0.0
        
        def _restore():
            time.sleep(duration_s)
            self.config.read_failure_probability = original_config.read_failure_probability
            self.config.garbage_read_probability = original_config.garbage_read_probability
            self.config.max_velocity_deg_s = SimConfig().max_velocity_deg_s
        
        threading.Thread(target=_restore, daemon=True).start()


class _FakeSerial:
    """Fake Serial-Objekt für Kompatibilität mit Safety-Layer."""
    
    def __init__(self):
        self.is_open = True
        self.in_waiting = 0
    
    def reset_input_buffer(self):
        pass
    
    def reset_output_buffer(self):
        pass
    
    def close(self):
        self.is_open = False
    
    def write(self, data):
        pass
    
    def flush(self):
        pass
    
    def readline(self):
        return b""
    
    def setRTS(self, val):
        pass
    
    def setDTR(self, val):
        pass

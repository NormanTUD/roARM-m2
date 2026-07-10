"""
Hardware-Abstraction für den RoArm-M2-S.
Einzige Datei die mit dem seriellen Port spricht.
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ArmState:
    """Aktueller Zustand aller Gelenke."""
    base_deg: float = 0.0
    shoulder_deg: float = 0.0
    elbow_deg: float = 90.0
    hand_deg: float = 180.0
    gripper_open: bool = True
    led_brightness: int = 255

    def to_list(self) -> list:
        """6-dim Vektor: [base, shoulder, elbow, hand, gripper, led]"""
        return [
            self.base_deg, self.shoulder_deg,
            self.elbow_deg, self.hand_deg,
            0.0 if self.gripper_open else 1.0,
            self.led_brightness / 255.0,
        ]

    @classmethod
    def from_list(cls, vec: list) -> "ArmState":
        return cls(
            base_deg=vec[0], shoulder_deg=vec[1],
            elbow_deg=vec[2], hand_deg=vec[3],
            gripper_open=vec[4] < 0.5,
            led_brightness=int(vec[5] * 255),
        )

    def copy(self) -> "ArmState":
        return ArmState(
            base_deg=self.base_deg, shoulder_deg=self.shoulder_deg,
            elbow_deg=self.elbow_deg, hand_deg=self.hand_deg,
            gripper_open=self.gripper_open, led_brightness=self.led_brightness,
        )


# Joint limits
LIMITS = {
    "base": (-90.0, 90.0),
    "shoulder": (-30.0, 60.0),
    "elbow": (0.0, 180.0),
    "hand": (0.0, 270.0),
}

# Speed presets
SPEED_PRESETS = {
    "very_slow": {"spd": 15, "acc": 30, "step": 0.15},
    "slow": {"spd": 30, "acc": 50, "step": 0.3},
    "medium": {"spd": 50, "acc": 100, "step": 0.6},
    "fast": {"spd": 80, "acc": 150, "step": 1.0},
    "very_fast": {"spd": 100, "acc": 200, "step": 1.5},
}


class RoArmHardware:
    """Low-level Kommunikation mit dem Arm über Serial."""

    BAUDRATE = 115200
    ESP32_VID_PIDS = [
        (0x1A86, 0x7523), (0x1A86, 0x55D4), (0x10C4, 0xEA60),
        (0x303A, 0x1001), (0x0403, 0x6001), (0x0403, 0x6015),
    ]

    def __init__(self, port: Optional[str] = None):
        self._port = port or self._find_port()
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._connect()

    @classmethod
    def _find_port(cls) -> str:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.vid and p.pid:
                for vid, pid in cls.ESP32_VID_PIDS:
                    if p.vid == vid and p.pid == pid:
                        return p.device
        for p in ports:
            if any(n in p.device.lower() for n in ['ttyusb', 'ttyacm']):
                return p.device
        raise ConnectionError("RoArm-M2-S nicht gefunden.")

    def _connect(self):
        self._ser = serial.Serial(self._port, self.BAUDRATE, timeout=1.0)
        self._ser.setRTS(True)
        self._ser.setDTR(True)
        time.sleep(2)
        self._ser.reset_input_buffer()
        print(f"[Hardware] ✓ Verbunden: {self._port}")

    def send(self, command: dict, wait: float = 0.5) -> list:
        """Sendet JSON-Befehl, wartet, gibt Antworten zurück."""
        with self._lock:
            msg = json.dumps(command, separators=(',', ':')) + '\n'
            self._ser.write(msg.encode())
            self._ser.flush()
            time.sleep(wait)
            lines = []
            while self._ser.in_waiting:
                line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    lines.append(line)
            return lines

    def send_nowait(self, command: dict):
        """
        Sendet ohne zu warten (für Echtzeit-Steuerung).
        Non-blocking lock — skips if busy to avoid stalling control loop.
        Drains input buffer to prevent overflow.
        """
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            msg = json.dumps(command, separators=(',', ':')) + '\n'
            self._ser.write(msg.encode())
            self._ser.flush()
            while self._ser.in_waiting:
                self._ser.readline()
        finally:
            self._lock.release()

    def move_joints(self, state: ArmState, spd: int = 50, acc: int = 100):
        """Bewegt alle Gelenke gleichzeitig (non-blocking)."""
        self.send_nowait({
            "T": 122,
            "b": round(state.base_deg, 1),
            "s": round(state.shoulder_deg, 1),
            "e": round(state.elbow_deg, 1),
            "h": round(state.hand_deg, 1),
            "spd": spd, "acc": acc,
        })

    def gripper_open(self):
        """Non-blocking gripper open."""
        self.send_nowait({"T": 106, "cmd": 1.08, "spd": 0, "acc": 0})

    def gripper_close(self):
        """Non-blocking gripper close."""
        self.send_nowait({"T": 106, "cmd": 3.14, "spd": 0, "acc": 0})

    def set_led(self, brightness: int):
        """Non-blocking LED set."""
        self.send_nowait({"T": 114, "led": brightness})

    def park(self):
        """Blocking park (used at shutdown)."""
        self.send({"T": 122, "b": 0, "s": 0, "e": 90, "h": 180, "spd": 15, "acc": 10}, 2.0)

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()


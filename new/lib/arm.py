"""
lib/arm.py — Clean hardware abstraction for RoArm-M2-S.

Single responsibility: send commands to the arm, read state.
No vision, no recording, no policy logic.
"""

import json
import time
import threading
from typing import Optional, List, Tuple
from dataclasses import dataclass

import serial
import serial.tools.list_ports


@dataclass
class ArmState:
    """Current arm joint state."""
    base_deg: float = 0.0
    shoulder_deg: float = 0.0
    elbow_deg: float = 90.0
    hand_deg: float = 180.0
    gripper_open: bool = True
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    voltage: float = 0.0


class RoArmController:
    """
    Low-level arm controller. Thin wrapper over serial protocol.
    
    Each method does ONE thing:
    - move_joints(): set absolute joint angles
    - move_joints_relative(): offset from current
    - move_cartesian(): set XYZ position
    - move_cartesian_relative(): offset XYZ
    - gripper_open() / gripper_close()
    - set_led()
    - get_state()
    - home() / park()
    """

    BAUDRATE = 115200
    ESP32_VID_PIDS = [
        (0x1A86, 0x7523), (0x1A86, 0x55D4), (0x10C4, 0xEA60),
        (0x303A, 0x1001), (0x0403, 0x6001), (0x0403, 0x6015),
    ]

    # Speed presets
    SPEED_MAP = {
        'very_slow': (15, 30),
        'slow': (30, 50),
        'medium': (50, 100),
        'fast': (80, 150),
        'very_fast': (100, 200),
    }

    def __init__(self, port: str = None):
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._state = ArmState()
        self._speed = 'medium'
        self._led = 0

        self._connect(port)

    def _connect(self, port: str = None):
        """Find and connect to the arm."""
        if not port:
            port = self._find_port()
        if not port:
            raise ConnectionError("RoArm-M2-S not found")

        self._ser = serial.Serial(port, self.BAUDRATE, timeout=1.0)
        self._ser.setRTS(True)
        self._ser.setDTR(True)
        time.sleep(2)
        self._ser.reset_input_buffer()
        print(f"[Arm] ✓ Connected: {port}")

    def _find_port(self) -> Optional[str]:
        """Auto-detect arm serial port."""
        for p in serial.tools.list_ports.comports():
            if p.vid and p.pid:
                for vid, pid in self.ESP32_VID_PIDS:
                    if p.vid == vid and p.pid == pid:
                        return p.device
        return None

    def _send(self, cmd: dict, wait: float = 0.5) -> List[str]:
        """Send JSON command, return response lines."""
        if not self._ser or not self._ser.is_open:
            return []
        with self._lock:
            msg = json.dumps(cmd, separators=(',', ':')) + '\n'
            self._ser.write(msg.encode())
            self._ser.flush()
            time.sleep(wait)
            lines = []
            while self._ser.in_waiting:
                line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    lines.append(line)
            return lines

    def _send_nowait(self, cmd: dict):
        """Send command without waiting (for streaming control)."""
        if not self._ser or not self._ser.is_open:
            return
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            msg = json.dumps(cmd, separators=(',', ':')) + '\n'
            self._ser.write(msg.encode())
            self._ser.flush()
            while self._ser.in_waiting:
                self._ser.readline()
        finally:
            self._lock.release()

    # ─── Movement ─────────────────────────────────────────────────────

    def move_joints(self, base: float = None, shoulder: float = None,
                    elbow: float = None, hand: float = None,
                    speed: str = None):
        """
        Move to absolute joint angles (degrees).
        Only specified joints are moved; others stay at current position.
        """
        spd, acc = self.SPEED_MAP.get(speed or self._speed, (50, 100))

        b = base if base is not None else self._state.base_deg
        s = shoulder if shoulder is not None else self._state.shoulder_deg
        e = elbow if elbow is not None else self._state.elbow_deg
        h = hand if hand is not None else self._state.hand_deg

        cmd = {"T": 122, "b": round(b, 1), "s": round(s, 1),
               "e": round(e, 1), "h": round(h, 1), "spd": spd, "acc": acc}
        self._send_nowait(cmd)

        self._state.base_deg = b
        self._state.shoulder_deg = s
        self._state.elbow_deg = e
        self._state.hand_deg = h

    def move_joints_relative(self, base: float = 0, shoulder: float = 0,
                             elbow: float = 0, hand: float = 0):
            if dsl_line:
                lines.append(f"  {dsl_line}")
            prev_state = step

        return '\n'.join(lines)

    def _step_to_dsl(self, step: RecordedStep, prev_step: Optional[RecordedStep]) -> Optional[str]:
        """
        Convert a single recorded step to a DSL command string.
        
        Compares with previous step to only emit meaningful changes.
        Adds wait commands based on time deltas.
        """
        parts = []

        # Time delta → wait command
        if prev_step:
            dt = step.timestamp - prev_step.timestamp
            if dt > 0.1:
                parts.append(f"wait {dt:.1f}")

        # Check what changed
        if prev_step:
            prev_arm = prev_step.arm_state
            curr_arm = step.arm_state

            # Joint changes
            changed_joints = {}
            for joint in ['base_deg', 'shoulder_deg', 'elbow_deg', 'hand_deg']:
                prev_val = prev_arm.get(joint, 0)
                curr_val = curr_arm.get(joint, 0)
                if abs(curr_val - prev_val) > 0.5:  # Threshold for meaningful change
                    # Map to DSL joint names
                    name_map = {
                        'base_deg': 'base',
                        'shoulder_deg': 'shoulder',
                        'elbow_deg': 'elbow',
                        'hand_deg': 'hand',
                    }
                    changed_joints[name_map[joint]] = round(curr_val, 1)

            if changed_joints:
                joint_str = ' '.join(f"{k}={v}" for k, v in changed_joints.items())
                parts.append(f"move {joint_str}")

            # Gripper change
            prev_grip = prev_arm.get('gripper_open', True)
            curr_grip = curr_arm.get('gripper_open', True)
            if curr_grip != prev_grip:
                if curr_grip:
                    parts.append("gripper open")
                else:
                    parts.append("gripper close")

            # LED change
            if step.led_brightness != prev_step.led_brightness:
                parts.append(f"led {step.led_brightness}")

        else:
            # First step: emit full state
            arm = step.arm_state
            parts.append(
                f"move base={round(arm.get('base_deg', 0), 1)} "
                f"shoulder={round(arm.get('shoulder_deg', 0), 1)} "
                f"elbow={round(arm.get('elbow_deg', 90), 1)} "
                f"hand={round(arm.get('hand_deg', 180), 1)}"
            )

        # Detections → detect command (if new objects appeared)
        if step.detections and (not prev_step or not prev_step.detections):
            parts.append("detect")

        if not parts:
            return None

        return '\n  '.join(parts)

    def _export_frames_to_dir(self, frames_dir: Path):
        """Export all buffered frames as JPGs for YOLO annotation."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        import cv2

        for i, frame in enumerate(self._frame_buffer):
            filepath = frames_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(filepath), frame)

        print(f"[Recorder] ✓ Exported {len(self._frame_buffer)} frames → {frames_dir}")

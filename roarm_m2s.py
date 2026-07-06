"""
RoArm-M2-S Full Abstraction Library
Provides a high-level Pythonic interface for the Waveshare RoArm-M2-S robotic arm.
Supports automatic port detection, UART and HTTP communication, all JSON commands.
"""

import serial
import serial.tools.list_ports
import json
import time
import sys
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union


@dataclass
class ArmStatus:
    """Parsed feedback from CMD_SERVO_RAD_FEEDBACK (T:105)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    base_rad: float = 0.0
    shoulder_rad: float = 0.0
    elbow_rad: float = 0.0
    eoat_rad: float = 0.0
    torque_base: int = 0
    torque_shoulder: int = 0
    torque_elbow: int = 0
    torque_hand: int = 0
    torque_switch_base: bool = True
    torque_switch_shoulder: bool = True
    torque_switch_elbow: bool = True
    torque_switch_hand: bool = True
    voltage: float = 0.0  # in Volts


class RoArmM2S:
    """
    High-level abstraction for the Waveshare RoArm-M2-S robotic arm.
    
    Supports:
    - Automatic serial port detection
    - Joint angle control (degrees and radians)
    - Cartesian (inverse kinematics) control
    - Gripper/EoAT control
    - Torque control
    - Dynamic force adaptation
    - Continuous movement
    - PID configuration
    - WiFi configuration
    - FLASH file system operations
    - Mission/step recording and playback
    - ESP-NOW control
    - LED control
    - Context manager (with statement)
    
    Usage:
        with RoArmM2S() as arm:
            arm.move_to_init()
            arm.move_joints_degrees(b=30, s=0, e=90, h=180)
            arm.gripper_open()
    """

    BAUDRATE = 115200
    # Known USB-serial chip identifiers for ESP32
    ESP32_VID_PIDS = [
        (0x1A86, 0x7523),  # CH340
        (0x1A86, 0x55D4),  # CH9102
        (0x10C4, 0xEA60),  # CP2102
        (0x303A, 0x1001),  # ESP32-S2/S3 native USB
        (0x0403, 0x6001),  # FTDI
        (0x0403, 0x6015),  # FTDI FT231X
    ]

    def __init__(self, port: Optional[str] = None, baudrate: int = BAUDRATE,
                 auto_connect: bool = True, timeout: float = 1.0):
        """
        Initialize the RoArm-M2-S controller.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3'). Auto-detected if None.
            baudrate: Communication baud rate (default 115200).
            auto_connect: Automatically connect on initialization.
            timeout: Serial read timeout in seconds.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self._connected = False
        self._lock = threading.Lock()

        if auto_connect:
            self.connect()

    # ─── Connection Management ────────────────────────────────────────────

    @classmethod
    def find_port(cls) -> Optional[str]:
        """
        Automatically detect the serial port of the RoArm-M2-S.
        
        Returns:
            The port name string, or None if not found.
        """
        ports = serial.tools.list_ports.comports()
        candidates = []

        for p in ports:
            # Match known ESP32 USB-serial chips
            if p.vid is not None and p.pid is not None:
                for vid, pid in cls.ESP32_VID_PIDS:
                    if p.vid == vid and p.pid == pid:
                        candidates.append(p.device)
                        break

        if not candidates:
            # Fallback: look for common names
            for p in ports:
                dev = p.device.lower()
                if any(name in dev for name in ['ttyusb', 'ttyacm', 'ch340', 'cp210', 'wchusbserial']):
                    candidates.append(p.device)

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Try to identify by description
            for p in ports:
                if p.device in candidates:
                    desc = (p.description or '').lower()
                    if 'esp32' in desc or 'cp210' in desc or 'ch340' in desc or 'ch910' in desc:
                        return p.device
            # Return first candidate
            return candidates[0]
        return None

    @classmethod
    def list_ports(cls) -> List[Dict[str, Any]]:
        """List all available serial ports with details."""
        ports = serial.tools.list_ports.comports()
        result = []
        for p in ports:
            result.append({
                'device': p.device,
                'description': p.description,
                'hwid': p.hwid,
                'vid': hex(p.vid) if p.vid else None,
                'pid': hex(p.pid) if p.pid else None,
                'manufacturer': p.manufacturer,
            })
        return result

    def connect(self, port: Optional[str] = None) -> None:
        """
        Connect to the robotic arm.
        
        Args:
            port: Override port. If None, uses self.port or auto-detects.
        """
        if port:
            self.port = port

        if not self.port:
            self.port = self.find_port()
            if not self.port:
                raise ConnectionError(
                    "Could not auto-detect RoArm-M2-S. Available ports:\n"
                    + "\n".join(f"  {p['device']}: {p['description']}" for p in self.list_ports())
                    + "\nPlease specify the port manually."
                )

        print(f"[RoArm-M2-S] Connecting on {self.port} @ {self.baudrate} baud...")
        self.ser = serial.Serial(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout
        )
        # Set control lines for ESP32 boot mode
        self.ser.setRTS(True)
        self.ser.setDTR(True)
        time.sleep(2)  # Wait for ESP32 boot

        # Flush any boot messages
        self._flush_input()
        self._connected = True
        print(f"[RoArm-M2-S] Connected successfully on {self.port}")

    def disconnect(self) -> None:
        """Disconnect from the robotic arm."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self._connected = False
            print("[RoArm-M2-S] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self.ser is not None and self.ser.is_open

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ─── Low-Level Communication ──────────────────────────────────────────

    def _flush_input(self) -> None:
        """Flush the serial input buffer."""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()

    def _send(self, command: dict, wait_time: float = 0.5) -> List[str]:
        """
        Send a JSON command and return response lines.
        
        Args:
            command: Dictionary to serialize as JSON.
            wait_time: Time to wait for response.
            
        Returns:
            List of response strings.
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to RoArm-M2-S!")

        with self._lock:
            json_cmd = json.dumps(command) + '\n'
            self.ser.write(json_cmd.encode('utf-8'))
            self.ser.flush()
            time.sleep(wait_time)

            responses = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
            return responses

    def send_raw(self, json_string: str, wait_time: float = 0.5) -> List[str]:
        """Send a raw JSON string command."""
        if not self.is_connected:
            raise ConnectionError("Not connected to RoArm-M2-S!")

        with self._lock:
            self.ser.write((json_string.strip() + '\n').encode('utf-8'))
            self.ser.flush()
            time.sleep(wait_time)

            responses = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    responses.append(line)
            return responses

    def _parse_feedback(self, responses: List[str]) -> Optional[Dict[str, Any]]:
        """Try to parse a JSON response from the arm."""
        for line in responses:
            try:
                data = json.loads(line)
                return data
            except json.JSONDecodeError:
                continue
        return None

    # ─── Movement: Reset ──────────────────────────────────────────────────

    def move_to_init(self, wait_time: float = 3.0) -> List[str]:
        """
        Move all joints to the initial/home position (CMD 100).
        This command blocks until complete.
        """
        return self._send({"T": 100}, wait_time=wait_time)

    # ─── Movement: Joint Angle Control (Degrees) ──────────────────────────

    def move_joints_degrees(self, b: float = 0, s: float = 0, e: float = 90,
                            h: float = 180, spd: float = 10, acc: float = 10) -> List[str]:
        """
        Move all joints using angles in degrees (CMD 122).
        
        Args:
            b: Base joint angle (-180° to 180°). Positive = left.
            s: Shoulder joint angle (-90° to 90°). Positive = forward.
            e: Elbow joint angle (0° to 180°, default 90°). Increase = downward.
            h: EoAT/Hand angle (45° to 180° for clamp, 45° to 315° for wrist).
            spd: Speed in °/s. 0 = maximum speed.
            acc: Acceleration in °/s². 0 = maximum acceleration.
        """
        cmd = {"T": 122, "b": b, "s": s, "e": e, "h": h, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    def move_single_joint_degrees(self, joint: int, angle: float,
                                   spd: float = 10, acc: float = 10) -> List[str]:
        """
        Move a single joint using angle in degrees (CMD 121).
        
        Args:
            joint: 1=Base, 2=Shoulder, 3=Elbow, 4=EoAT
            angle: Target angle in degrees.
            spd: Speed in °/s.
            acc: Acceleration in °/s².
        """
        cmd = {"T": 121, "joint": joint, "angle": angle, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    # ─── Movement: Joint Angle Control (Radians) ──────────────────────────

    def move_joints_radians(self, base: float = 0, shoulder: float = 0,
                            elbow: float = 1.57, hand: float = 3.14,
                            spd: int = 0, acc: int = 10) -> List[str]:
        """
        Move all joints using radians (CMD 102).
        
        Args:
            base: Base angle in rad (-3.14 to 3.14).
            shoulder: Shoulder angle in rad (-1.57 to 1.57).
            elbow: Elbow angle in rad (0 to 3.14, default 1.57).
            hand: EoAT angle in rad (1.08 to 3.14 for clamp).
            spd: Speed in steps/s (4096 steps = 1 revolution). 0 = max.
            acc: Acceleration (0-254, unit: 100 steps/s²). 0 = max.
        """
        cmd = {"T": 102, "base": base, "shoulder": shoulder,
               "elbow": elbow, "hand": hand, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    def move_single_joint_radians(self, joint: int, rad: float,
                                   spd: int = 0, acc: int = 10) -> List[str]:
        """
        Move a single joint using radians (CMD 101).
        
        Args:
            joint: 1=Base, 2=Shoulder, 3=Elbow, 4=EoAT
            rad: Target angle in radians.
            spd: Speed in steps/s. 0 = max.
            acc: Acceleration. 0 = max.
        """
        cmd = {"T": 101, "joint": joint, "rad": rad, "spd": spd, "acc": acc}
        return self._send(cmd, wait_time=2.0)

    # ─── Movement: Cartesian / Inverse Kinematics ─────────────────────────

    def move_cartesian(self, x: float, y: float, z: float,
                       t: float = 3.14, spd: float = 0.25) -> List[str]:
        """
        Move EoAT to XYZ coordinates using inverse kinematics (CMD 104).
        This command may block.
        
        Args:
            x: X position in mm (positive = forward).
            y: Y position in mm (positive = left).
            z: Z position in mm (positive = up).
            t: EoAT angle in radians.
            spd: Movement speed (higher = faster).
        """
        cmd = {"T": 104, "x": x, "y": y, "z": z, "t": t, "spd": spd}
        return self._send(cmd, wait_time=3.0)

    def move_cartesian_direct(self, x: float, y: float, z: float,
                              t: float = 3.14) -> List[str]:
        """
        Move EoAT directly to XYZ (CMD 1041). Non-blocking, no interpolation.
        Suitable for continuous streaming of small position changes.
        """
        cmd = {"T": 1041, "x": x, "y": y, "z": z, "t": t}
        return self._send(cmd, wait_time=0.1)

    def move_single_axis(self, axis: int, pos: float, spd: float = 0.25) -> List[str]:
        """
        Move a single axis to a position (CMD 103). Blocks.
        
        Args:
            axis: 1=X, 2=Y, 3=Z, 4=T (EoAT angle in rad).
            pos: Target position in mm (or rad for axis 4).
            spd: Speed.
        """
        cmd = {"T": 103, "axis": axis, "pos": pos, "spd": spd}
        return self._send(cmd, wait_time=3.0)

    # ─── Movement: Continuous Control ─────────────────────────────────────

    def continuous_move(self, mode: int = 0, axis: int = 1,
                        cmd: int = 0, spd: float = 5) -> List[str]:
        """
        Continuous movement control (CMD 123).
        
        Args:
            mode: 0 = angle control, 1 = coordinate control.
            axis: In angle mode: 1=Base, 2=Shoulder, 3=Elbow, 4=Hand.
                  In coord mode: 1=X, 2=Y, 3=Z, 4=Hand.
            cmd: 0=STOP, 1=INCREASE, 2=DECREASE.
            spd: Speed coefficient (0-20 recommended).
        """
        return self._send({"T": 123, "m": mode, "axis": axis, "cmd": cmd, "spd": spd}, wait_time=0.2)

    def continuous_stop(self, mode: int = 0, axis: int = 1) -> List[str]:
        """Stop continuous movement on the given axis."""
        return self.continuous_move(mode=mode, axis=axis, cmd=0, spd=0)

    # ─── EoAT / Gripper Control ───────────────────────────────────────────

    def gripper_set(self, rad: float = 3.14, spd: int = 0, acc: int = 0) -> List[str]:
        """
        Set gripper/EoAT position in radians (CMD 106).
        
        Args:
            rad: Target angle. ~1.08 = fully open, ~3.14 = fully closed (clamp).
            spd: Speed. 0 = max.
            acc: Acceleration. 0 = max.
        """
        return self._send({"T": 106, "cmd": rad, "spd": spd, "acc": acc}, wait_time=1.0)

    def gripper_open(self, amount: float = 1.08) -> List[str]:
        """Open the gripper (default fully open)."""
        return self.gripper_set(rad=amount)

    def gripper_close(self, amount: float = 3.14) -> List[str]:
        """Close the gripper (default fully closed)."""
        return self.gripper_set(rad=amount)

    def set_gripper_torque(self, torque: int = 200) -> List[str]:
        """
        Set maximum gripper torque (CMD 107).
        
        Args:
            torque: 200 = 20% max torque, 1000 = 100% max torque.
        """
        return self._send({"T": 107, "tor": torque}, wait_time=0.5)

    def set_eoat_type(self, mode: int = 0) -> List[str]:
        """
        Set EoAT type (CMD 1).
        
        Args:
            mode: 0 = clamp (default), 1 = wrist.
        """
        return self._send({"T": 1, "cmd": mode}, wait_time=0.5)

    # ─── Feedback / Status ────────────────────────────────────────────────

    def get_status_raw(self) -> List[str]:
        """Get raw feedback from the arm (CMD 105)."""
        return self._send({"T": 105}, wait_time=0.5)

    def get_status(self) -> Optional[ArmStatus]:
        """
        Get parsed arm status including position, angles, torque, and voltage.
        
        Returns:
            ArmStatus dataclass or None if parsing fails.
        """
        responses = self.get_status_raw()
        data = self._parse_feedback(responses)
        if data and data.get("T") == 1051:
            return ArmStatus(
                x=data.get("x", 0),
                y=data.get("y", 0),
                z=data.get("z", 0),
                base_rad=data.get("b", 0),
                shoulder_rad=data.get("s", 0),
                elbow_rad=data.get("e", 0),
                eoat_rad=data.get("t", 0),
                torque_base=data.get("torB", 0),
                torque_shoulder=data.get("torS", 0),
                torque_elbow=data.get("torE", 0),
                torque_hand=data.get("torH", 0),
                torque_switch_base=bool(data.get("torswitchB", 1)),
                torque_switch_shoulder=bool(data.get("torswitchS", 1)),
                torque_switch_elbow=bool(data.get("torswitchE", 1)),
                torque_switch_hand=bool(data.get("torswitchH", 1)),
                voltage=data.get("v", 0) / 100.0,
            )
        return None

    # ─── Torque Control ───────────────────────────────────────────────────

    def set_torque(self, enable: bool = True) -> List[str]:
        """
        Enable or disable torque lock (CMD 210).
        When disabled, joints can be moved manually.
        """
        return self._send({"T": 210, "cmd": 1 if enable else 0}, wait_time=0.5)

    def torque_on(self) -> List[str]:
        """Enable torque lock on all joints."""
        return self.set_torque(True)

    def torque_off(self) -> List[str]:
        """Disable torque lock (allow manual movement)."""
        return self.set_torque(False)

    # ─── Dynamic Adaptation ───────────────────────────────────────────────

    def set_dynamic_adaptation(self, enable: bool = True,
                                b: int = 60, s: int = 110,
                                e: int = 50, h: int = 50) -> List[str]:
        """
        Enable/disable dynamic force self-adaptation (CMD 112).
        When enabled, the arm returns to position after external force.
        
        Args:
            enable: True to enable, False to disable.
            b, s, e, h: Max torque limits per joint.
        """
        if enable:
            return self._send({"T": 112, "mode": 1, "b": b, "s": s, "e": e, "h": h}, wait_time=0.5)
        else:
            return self._send({"T": 112, "mode": 0, "b": 1000, "s": 1000, "e": 1000, "h": 1000}, wait_time=0.5)

    # ─── PID Control ──────────────────────────────────────────────────────

    def set_joint_pid(self, joint: int, p: int = 16, i: int = 0) -> List[str]:
        """
        Set PID values for a joint (CMD 108).
        
        Args:
            joint: 1=Base, 2=Shoulder, 3=Elbow, 4=EoAT.
            p: Proportional coefficient (default 16).
            i: Integral coefficient (default 0, set in multiples of 8).
        """
        return self._send({"T": 108, "joint": joint, "p": p, "i": i}, wait_time=0.5)

    def reset_pid(self) -> List[str]:
        """Reset all joint PID values to defaults (CMD 109)."""
        return self._send({"T": 109}, wait_time=0.5)

    # ─── LED Control ──────────────────────────────────────────────────────

    def set_led(self, brightness: int = 0) -> List[str]:
        """
        Set LED brightness (CMD 114).
        
        Args:
            brightness: 0 (off) to 255 (max).
        """
        return self._send({"T": 114, "led": brightness}, wait_time=0.3)

    # ─── Delay Command ────────────────────────────────────────────────────

    def delay(self, ms: int = 1000) -> List[str]:
        """Send a delay command to the arm (CMD 111)."""
        return self._send({"T": 111, "cmd": ms}, wait_time=ms / 1000.0 + 0.5)

    # ─── WiFi Configuration ───────────────────────────────────────────────

    def wifi_get_info(self) -> List[str]:
        """Get WiFi configuration info (CMD 405)."""
        return self._send({"T": 405}, wait_time=1.0)

    def wifi_set_mode_on_boot(self, mode: int = 3) -> List[str]:
        """Set WiFi mode on boot (CMD 401). 0=off, 1=AP, 2=STA, 3=AP+STA."""
        return self._send({"T": 401, "cmd": mode}, wait_time=0.5)

    def wifi_set_sta(self, ssid: str, password: str) -> List[str]:
        """Connect to existing WiFi (CMD 403)."""
        return self._send({"T": 403, "ssid": ssid, "password": password}, wait_time=5.0)

    def wifi_set_ap(self, ssid: str = "RoArm-M2", password: str = "12345678") -> List[str]:
        """Create WiFi hotspot (CMD 402)."""
        return self._send({"T": 402, "ssid": ssid, "password": password}, wait_time=2.0)

    def wifi_stop(self) -> List[str]:
        """Turn off WiFi (CMD 408)."""
        return self._send({"T": 408}, wait_time=0.5)

    # ─── FLASH File System ────────────────────────────────────────────────

    def flash_scan_files(self) -> List[str]:
        """Scan all files in FLASH (CMD 200)."""
        return self._send({"T": 200}, wait_time=1.0)

    def flash_create_file(self, name: str, content: str) -> List[str]:
        """Create a new file in FLASH (CMD 201)."""
        return self._send({"T": 201, "name": name, "content": content}, wait_time=1.0)

    def flash_read_file(self, name: str) -> List[str]:
        """Read file content from FLASH (CMD 202)."""
        return self._send({"T": 202, "name": name}, wait_time=1.0)

    def flash_delete_file(self, name: str) -> List[str]:
        """Delete a file from FLASH (CMD 203)."""
        return self._send({"T": 203, "name": name}, wait_time=1.0)

    def flash_append_line(self, name: str, content: str) -> List[str]:
        """Append a line to a file (CMD 204)."""
        return self._send({"T": 204, "name": name, "content": content}, wait_time=1.0)

    # ─── Mission / Step Recording ─────────────────────────────────────────

    def mission_create(self, name: str, intro: str = "") -> List[str]:
        """Create a new mission file (CMD 220)."""
        return self._send({"T": 220, "name": name, "intro": intro}, wait_time=1.0)

    def mission_read(self, name: str) -> List[str]:
        """Read mission file content (CMD 221)."""
        return self._send({"T": 221, "name": name}, wait_time=1.0)

    def mission_append_step(self, name: str, step_json: str) -> List[str]:
        """Append a JSON command step to mission (CMD 222)."""
        return self._send({"T": 222, "name": name, "step": step_json}, wait_time=1.0)

    def mission_append_current_pos(self, name: str, spd: float = 0.25) -> List[str]:
        """Append current position as a step (CMD 223)."""
        return self._send({"T": 223, "name": name, "spd": spd}, wait_time=1.0)

    def mission_append_delay(self, name: str, delay_ms: int = 1000) -> List[str]:
        """Append a delay step to mission (CMD 224)."""
        return self._send({"T": 224, "name": name, "delay": delay_ms}, wait_time=0.5)

    def mission_play(self, name: str, times: int = 1) -> List[str]:
        """
        Play a mission file (CMD 242).
        
        Args:
            name: Mission name.
            times: Number of loops. -1 = infinite.
        """
        return self._send({"T": 242, "name": name, "times": times}, wait_time=2.0)

    def mission_delete_step(self, name: str, step_num: int) -> List[str]:
        """Delete a step from mission (CMD 231)."""
        return self._send({"T": 231, "name": name, "stepNum": step_num}, wait_time=1.0)

    # ─── ESP-NOW ──────────────────────────────────────────────────────────

    def espnow_get_mac(self) -> List[str]:
        """Get this device's MAC address (CMD 302)."""
        return self._send({"T": 302}, wait_time=0.5)

    def espnow_set_mode(self, mode: int = 3) -> List[str]:
        """Set ESP-NOW operation mode (CMD 301). 0=off, 1=multicast, 2=unicast/broadcast, 3=follower."""
        return self._send({"T": 301, "mode": mode}, wait_time=0.5)

    def espnow_add_peer(self, mac: str) -> List[str]:
        """Add a MAC address to peer list (CMD 303)."""
        return self._send({"T": 303, "mac": mac}, wait_time=0.5)

    def espnow_remove_peer(self, mac: str) -> List[str]:
        """Remove a MAC address from peer list (CMD 304)."""
        return self._send({"T": 304, "mac": mac}, wait_time=0.5)

    # ─── Convenience / High-Level Methods ─────────────────────────────────

    def home(self) -> List[str]:
        """Alias for move_to_init()."""
        return self.move_to_init()

    def park(self) -> List[str]:
        """Move to a safe parking position."""
        return self.move_joints_degrees(b=0, s=0, e=90, h=180, spd=15, acc=10)

    def wave(self, cycles: int = 3, speed: float = 20) -> None:
        """Perform a simple wave gesture."""
        for _ in range(cycles):
            self.move_joints_degrees(b=0, s=30, e=45, h=180, spd=speed)
            time.sleep(0.8)
            self.move_joints_degrees(b=0, s=-10, e=120, h=180, spd=speed)
            time.sleep(0.8)
        self.move_to_init()

    def nod(self, cycles: int = 2, speed: float = 15) -> None:
        """Perform a nodding gesture."""
        for _ in range(cycles):
            self.move_joints_degrees(b=0, s=20, e=70, h=180, spd=speed)
            time.sleep(0.6)
            self.move_joints_degrees(b=0, s=-10, e=110, h=180, spd=speed)
            time.sleep(0.6)
        self.park()

    def pick_and_place(self, pick_x: float, pick_y: float, pick_z: float,
                       place_x: float, place_y: float, place_z: float,
                       hover_offset: float = 50, spd: float = 0.25) -> None:
        """
        Simple pick-and-place operation using Cartesian coordinates.
        
        Args:
            pick_x/y/z: Pick position in mm.
            place_x/y/z: Place position in mm.
            hover_offset: Height offset for approach/retract in mm.
            spd: Movement speed.
        """
        # Open gripper
        self.gripper_open()
        time.sleep(0.5)

        # Move above pick position
        self.move_cartesian(pick_x, pick_y, pick_z + hover_offset, spd=spd)
        time.sleep(0.5)

        # Descend to pick
        self.move_cartesian(pick_x, pick_y, pick_z, spd=spd)
        time.sleep(0.5)

        # Close gripper
        self.gripper_close()
        time.sleep(0.5)

        # Lift
        self.move_cartesian(pick_x, pick_y, pick_z + hover_offset, spd=spd)
        time.sleep(0.5)

        # Move above place position
        self.move_cartesian(place_x, place_y, place_z + hover_offset, spd=spd)
        time.sleep(0.5)

        # Descend to place
        self.move_cartesian(place_x, place_y, place_z, spd=spd)
        time.sleep(0.5)

        # Open gripper
        self.gripper_open()
        time.sleep(0.5)

        # Retract
        self.move_cartesian(place_x, place_y, place_z + hover_offset, spd=spd)

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        return f"<RoArmM2S port={self.port} status={status}>"

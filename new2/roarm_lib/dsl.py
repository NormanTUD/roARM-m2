"""
RoArm DSL — Domain Specific Language für Robotersteuerung.

Menschenlesbare .roarm-Dateien die:
- Von Hand geschrieben werden können
- Vom Recorder automatisch erzeugt werden
- Step-by-Step interpretiert werden können
- Funktionen mit Parametern und Defaults unterstützen
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

from .hardware import RoArmHardware, ArmState, SPEED_PRESETS, LIMITS
from .vision import VisionSystem, Detection


@dataclass
class DSLFunction:
    """Eine definierte Funktion in der DSL."""
    name: str
    params: Dict[str, Any]  # name → default_value
    body: List[str]         # Zeilen des Funktionskörpers
    source_line: int = 0    # Wo definiert (für Debugging)


@dataclass
class DSLCommand:
    """Ein geparster Befehl."""
    type: str           # move, wait, gripper, led, call, when, print, ...
    args: Dict[str, Any]
    raw_line: str = ""
    line_number: int = 0


class DSLParser:
    """Parst .roarm-Dateien in Befehle und Funktionen."""

    def __init__(self):
        self.functions: Dict[str, DSLFunction] = {}
        self.defaults: Dict[str, Any] = {"speed": "medium", "acceleration": 100}
        self.commands: List[DSLCommand] = []

    def parse_file(self, path: Path) -> "DSLParser":
        """Parst eine .roarm-Datei."""
        with open(path, 'r') as f:
            lines = f.readlines()
        return self.parse_lines(lines)

    def parse_string(self, text: str) -> "DSLParser":
        """Parst einen DSL-String."""
        return self.parse_lines(text.strip().split('\n'))

    def parse_lines(self, lines: List[str]) -> "DSLParser":
        """Hauptparser."""
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()
            i += 1

            # Leerzeilen und Kommentare
            if not stripped or stripped.startswith('#'):
                continue

            # Defaults-Block
            if stripped == "defaults:":
                while i < len(lines):
                    inner = lines[i].rstrip()
                    if not inner.startswith('  ') and not inner.startswith('\t'):
                        break
                    key_val = inner.strip().split(':', 1)
                    if len(key_val) == 2:
                        self.defaults[key_val[0].strip()] = self._parse_value(key_val[1].strip())
                    i += 1
                continue

            # Funktionsdefinition
            if stripped.startswith("function "):
                func, i = self._parse_function(stripped, lines, i)
                self.functions[func.name] = func
                continue

            # Normaler Befehl
            cmd = self._parse_command(stripped, i - 1)
            if cmd:
                self.commands.append(cmd)

        return self

    def _parse_function(self, header: str, lines: List[str], i: int) -> tuple:
        """Parst eine Funktionsdefinition."""
        # function name(param1=default1, param2=default2):
        match = re.match(r'function\s+(\w+)\s*\(([^)]*)\)\s*:', header)
        if not match:
            match = re.match(r'function\s+(\w+)\s*:', header)
            name = match.group(1) if match else "unknown"
            params = {}
        else:
            name = match.group(1)
            params = self._parse_params(match.group(2))

        body = []
        while i < len(lines):
            inner = lines[i].rstrip()
            if not inner.startswith('  ') and not inner.startswith('\t'):
                if inner.strip() and not inner.strip().startswith('#'):
                    break
            body.append(inner.strip())
            i += 1

        # Leere Zeilen am Ende entfernen
        while body and not body[-1]:
            body.pop()

        return DSLFunction(name=name, params=params, body=body, source_line=i), i

    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parst Parameter-Liste: 'speed=medium, height=90'"""
        params = {}
        if not params_str.strip():
            return params
        for part in params_str.split(','):
            part = part.strip()
            if '=' in part:
                key, val = part.split('=', 1)
                params[key.strip()] = self._parse_value(val.strip())
            else:
                params[part.strip()] = None
        return params

    def _parse_command(self, line: str, line_num: int) -> Optional[DSLCommand]:
        """Parst eine einzelne Befehlszeile."""
        parts = line.split()
        if not parts:
            return None

        cmd_type = parts[0].lower()

        if cmd_type == "move":
            args = self._parse_key_value_args(parts[1:])
            return DSLCommand("move", args, line, line_num)

        elif cmd_type == "wait":
            duration = float(parts[1]) if len(parts) > 1 else 0.5
            return DSLCommand("wait", {"duration": duration}, line, line_num)

        elif cmd_type == "gripper":
            action = parts[1] if len(parts) > 1 else "open"
            return DSLCommand("gripper", {"action": action}, line, line_num)

        elif cmd_type == "led":
            brightness = int(parts[1]) if len(parts) > 1 else 255
            return DSLCommand("led", {"brightness": brightness}, line, line_num)

        elif cmd_type == "call":
            func_name = parts[1] if len(parts) > 1 else ""
            args = self._parse_key_value_args(parts[2:])
            return DSLCommand("call", {"function": func_name, **args}, line, line_num)

        elif cmd_type == "when":
            # when see "class_name":
            match = re.match(r'when\s+see\s+"([^"]+)"', line)
            if match:
                return DSLCommand("when_see", {"class": match.group(1)}, line, line_num)
            match = re.match(r'when\s+see\s+\$(\w+)', line)
            if match:
                return DSLCommand("when_see", {"class_var": match.group(1)}, line, line_num)

        elif cmd_type == "move_toward":
            target = parts[1] if len(parts) > 1 else ""
            return DSLCommand("move_toward", {"target": target.strip('"')}, line, line_num)

        elif cmd_type == "print":
            msg = line[len("print"):].strip().strip('"')
            return DSLCommand("print", {"message": msg}, line, line_num)

        elif cmd_type == "otherwise:":
            return DSLCommand("otherwise", {}, line, line_num)

        return DSLCommand("unknown", {"raw": line}, line, line_num)

    def _parse_key_value_args(self, parts: List[str]) -> Dict[str, Any]:
        """Parst key=value Paare aus einer Argumentliste."""
        args = {}
        for part in parts:
            if '=' in part:
                key, val = part.split('=', 1)
                args[key.strip()] = self._parse_value(val.strip())
        return args

    def _parse_value(self, val: str) -> Any:
        """Parst einen Wert (Zahl, String, Variable)."""
        if val.startswith('$'):
            return ("var", val[1:])
        if val.startswith('"') and val.endswith('"'):
            return val[1:-1]
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val


class DSLInterpreter:
    """
    Step-by-Step Interpreter für .roarm-Dateien.
    Führt Befehle einzeln aus, kann pausiert und debuggt werden.
    """

    def __init__(self, hardware: "RoArmHardware", vision: "VisionSystem" = None):
        self._hw = hardware
        self._vision = vision
        self._parser: Optional[DSLParser] = None
        self._state = ArmState()
        self._variables: Dict[str, Any] = {}
        self._running = False
        self._paused = False
        self._step_mode = False
        self._current_line = 0
        self._call_stack: List[Tuple[str, int, Dict]] = []  # (func_name, line, locals)

        # Callbacks für UI
        self.on_step: Optional[Callable[[DSLCommand, ArmState], None]] = None
        self.on_detection: Optional[Callable[[str, list], None]] = None
        self.on_print: Optional[Callable[[str], None]] = None

    def load_file(self, path: Path):
        """Lädt und parst eine .roarm-Datei."""
        self._parser = DSLParser()
        self._parser.parse_file(path)
        self._current_line = 0
        print(f"[DSL] ✓ Geladen: {path.name} "
              f"({len(self._parser.commands)} Befehle, "
              f"{len(self._parser.functions)} Funktionen)")

    def load_string(self, text: str):
        """Lädt DSL aus einem String."""
        self._parser = DSLParser()
        self._parser.parse_string(text)
        self._current_line = 0

    @property
    def functions(self) -> Dict[str, DSLFunction]:
        return self._parser.functions if self._parser else {}

    @property
    def is_running(self) -> bool:
        return self._running

    def run(self, step_mode: bool = False):
        """Führt das gesamte Skript aus."""
        if not self._parser:
            raise RuntimeError("Kein Skript geladen!")

        self._running = True
        self._paused = False
        self._step_mode = step_mode
        self._current_line = 0

        self._execute_commands(self._parser.commands)
        self._running = False

    def step(self):
        """Führt genau einen Befehl aus (für Step-by-Step Debugging)."""
        self._paused = False

    def pause(self):
        """Pausiert die Ausführung."""
        self._paused = True

    def stop(self):
        """Stoppt die Ausführung."""
        self._running = False

    def _execute_commands(self, commands: List[DSLCommand], local_vars: Dict = None):
        """Führt eine Liste von Befehlen aus."""
        vars_ctx = local_vars or {}

        for i, cmd in enumerate(commands):
            if not self._running:
                break

            # Step-Mode: warte auf step()
            if self._step_mode:
                self._paused = True
                while self._paused and self._running:
                    import time
                    time.sleep(0.05)

            # Callback
            if self.on_step:
                self.on_step(cmd, self._state)

            self._execute_one(cmd, vars_ctx)

    def _execute_one(self, cmd: DSLCommand, local_vars: Dict):
        """Führt einen einzelnen Befehl aus."""
        import time as _time

        if cmd.type == "move":
            self._exec_move(cmd, local_vars)

        elif cmd.type == "wait":
            duration = self._resolve_value(cmd.args.get("duration", 0.5), local_vars)
            _time.sleep(float(duration))

        elif cmd.type == "gripper":
            action = cmd.args.get("action", "open")
            if action == "open":
                self._hw.gripper_open()
                self._state.gripper_open = True
            else:
                self._hw.gripper_close()
                self._state.gripper_open = False

        elif cmd.type == "led":
            brightness = int(self._resolve_value(cmd.args.get("brightness", 255), local_vars))
            self._hw.set_led(brightness)
            self._state.led_brightness = brightness

        elif cmd.type == "call":
            self._exec_call(cmd, local_vars)

        elif cmd.type == "when_see":
            self._exec_when_see(cmd, local_vars)

        elif cmd.type == "move_toward":
            self._exec_move_toward(cmd, local_vars)

        elif cmd.type == "print":
            msg = self._resolve_value(cmd.args.get("message", ""), local_vars)
            if self.on_print:
                self.on_print(str(msg))
            else:
                print(f"  [DSL] {msg}")

    def _exec_move(self, cmd: DSLCommand, local_vars: Dict):
        """Führt einen move-Befehl aus."""
        args = cmd.args

        if "base" in args:
            self._state.base_deg = float(self._resolve_value(args["base"], local_vars))
        if "shoulder" in args:
            self._state.shoulder_deg = float(self._resolve_value(args["shoulder"], local_vars))
        if "elbow" in args:
            self._state.elbow_deg = float(self._resolve_value(args["elbow"], local_vars))
        if "hand" in args:
            self._state.hand_deg = float(self._resolve_value(args["hand"], local_vars))

        # Speed aus args oder defaults
        speed_name = self._resolve_value(
            args.get("speed", self._parser.defaults.get("speed", "medium")),
            local_vars
        )
        from .hardware import SPEED_PRESETS
        spd_cfg = SPEED_PRESETS.get(str(speed_name), SPEED_PRESETS["medium"])

        self._hw.move_joints(self._state, spd=spd_cfg["spd"], acc=spd_cfg["acc"])

    def _exec_call(self, cmd: DSLCommand, local_vars: Dict):
        """Ruft eine Funktion auf."""
        func_name = cmd.args.get("function", "")
        if func_name not in self._parser.functions:
            print(f"  [DSL] ⚠ Funktion '{func_name}' nicht gefunden!")
            return

        func = self._parser.functions[func_name]

        # Parameter auflösen
        func_locals = dict(func.params)  # Defaults
        for key, val in cmd.args.items():
            if key == "function":
                continue
            if key in func.params:
                func_locals[key] = self._resolve_value(val, local_vars)

        # Call-Stack
        self._call_stack.append((func_name, 0, func_locals))

        # Body parsen und ausführen
        body_parser = DSLParser()
        body_commands = []
        for line in func.body:
            parsed = body_parser._parse_command(line.strip(), 0)
            if parsed:
                body_commands.append(parsed)

        self._execute_commands(body_commands, func_locals)
        self._call_stack.pop()

    def _exec_when_see(self, cmd: DSLCommand, local_vars: Dict):
        """Führt when see aus — wartet auf YOLO-Detection."""
        if not self._vision or not self._vision.has_model:
            print("  [DSL] ⚠ Kein YOLO-Modell geladen für 'when see'!")
            return

        target_class = cmd.args.get("class", "")
        if not target_class:
            var_name = cmd.args.get("class_var", "")
            target_class = str(self._resolve_value(("var", var_name), local_vars))

        # Versuche Detection (mit Timeout)
        import time as _time
        timeout = 5.0
        start = _time.time()

        while _time.time() - start < timeout and self._running:
            detections = self._vision.detect(target_classes=[target_class])
            if detections:
                if self.on_detection:
                    self.on_detection(target_class, detections)
                return  # Erfolgreich — nächster Befehl wird ausgeführt
            _time.sleep(0.1)

        print(f"  [DSL] ⚠ '{target_class}' nicht erkannt (Timeout)")

    def _exec_move_toward(self, cmd: DSLCommand, local_vars: Dict):
        """Bewegt den Arm Richtung erkanntes Objekt."""
        if not self._vision or not self._vision.has_model:
            return

        target = self._resolve_value(cmd.args.get("target", ""), local_vars)
        detections = self._vision.detect(target_classes=[str(target)])

        if not detections:
            print(f"  [DSL] ⚠ '{target}' nicht sichtbar für move_toward")
            return

        det = detections[0]
        # Einfache Proportionalsteuerung: Offset von Bildmitte → Base-Korrektur
        offset_x = det.bbox.x_center - 0.5  # -0.5..+0.5
        offset_y = det.bbox.y_center - 0.5

        # Base korrigieren (horizontal)
        self._state.base_deg -= offset_x * 20.0  # 20° pro halbe Bildbreite
        # Shoulder korrigieren (vertikal)
        self._state.shoulder_deg += offset_y * 10.0

        # Clamp
        self._state.base_deg = max(-90, min(90, self._state.base_deg))
        self._state.shoulder_deg = max(-30, min(60, self._state.shoulder_deg))

        self._hw.move_joints(self._state, spd=30, acc=50)

    def _resolve_value(self, val, local_vars: Dict):
        """Löst einen Wert auf (Variable, Literal, etc.)."""
        if isinstance(val, tuple) and len(val) == 2 and val[0] == "var":
            var_name = val[1]
            if var_name in local_vars:
                return local_vars[var_name]
            if var_name in self._variables:
                return self._variables[var_name]
            if self._parser and var_name in self._parser.defaults:
                return self._parser.defaults[var_name]
            return var_name  # Unresolved → als String
        return val


class DSLRecorder:
    """
    Zeichnet Bewegungen auf und erzeugt .roarm-Dateien.
    
    Features:
    - Aufzeichnung als Sequenz von move/wait/gripper Befehlen
    - Funktion-Definition per Knopfdruck (Start/Stop)
    - Sane Defaults aus der Aufzeichnung
    - JPG-Export für YOLO-Annotation
    """

    def __init__(self, output_dir: Path, vision: "VisionSystem" = None):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._vision = vision

        self._recording = False
        self._function_recording = False
        self._current_function_name: Optional[str] = None
        self._frames: List[Dict] = []
        self._function_frames: List[Dict] = []
        self._functions: Dict[str, List[Dict]] = {}

        # Image output
        self._image_dir = output_dir / "images"
        self._image_dir.mkdir(exist_ok=True)
        self._image_count = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_recording_function(self) -> bool:
        return self._function_recording

    @property
    def current_function_name(self) -> Optional[str]:
        return self._current_function_name

    def start_recording(self):
        """Startet die Aufzeichnung."""
        self._recording = True
        self._frames = []
        print("[DSL-Rec] ● Aufzeichnung gestartet")

    def stop_recording(self) -> Path:
        """Stoppt und speichert als .roarm-Datei."""
        self._recording = False
        path = self._save_as_dsl(self._frames, "recording")
        print(f"[DSL-Rec] ■ Gespeichert: {path}")
        return path

    def start_function(self, name: str):
        """Startet Funktions-Aufzeichnung (per Knopfdruck)."""
        self._function_recording = True
        self._current_function_name = name
        self._function_frames = []
        print(f"[DSL-Rec] ● Funktion '{name}' aufzeichnen...")

    def stop_function(self):
        """Stoppt Funktions-Aufzeichnung."""
        if not self._function_recording:
            return

        name = self._current_function_name
        self._functions[name] = self._function_frames.copy()
        self._function_recording = False
        self._current_function_name = None
        print(f"[DSL-Rec] ■ Funktion '{name}' gespeichert ({len(self._function_frames)} Schritte)")

    def record_frame(self, state: "ArmState", action: str = "", detections: list = None):
        """Zeichnet einen Frame auf."""
        import time

        frame = {
            "timestamp": time.time(),
            "state": state.copy(),
            "action": action,
            "detections": detections or [],
        }

        if self._recording:
            self._frames.append(frame)

        if self._function_recording:
            self._function_frames.append(frame)

    def save_image(self, frame_img) -> Optional[Path]:
        """Speichert aktuellen Kamera-Frame als JPG für YOLO-Annotation."""
        if self._vision and frame_img is not None:
            self._image_count += 1
            path = self._vision.save_frame_as_jpg(
                frame_img, self._image_dir, prefix="train"
            )
            return path
        return None

    def export_all_images(self, every_n: int = 5) -> int:
        """Exportiert jeden n-ten Frame als JPG."""
        count = 0
        # Wird vom Recorder aufgerufen der die Frames hat
        return count

    def _save_as_dsl(self, frames: List[Dict], name: str) -> Path:
        """Konvertiert aufgezeichnete Frames in eine .roarm-Datei."""
        import time as _time

        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        filename = self._output_dir / f"{name}_{timestamp}.roarm"

        lines = []
        lines.append(f"# Aufgezeichnet am {_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# {len(frames)} Schritte")
        lines.append("")
        lines.append("defaults:")
        lines.append("  speed: medium")
        lines.append("  acceleration: 100")
        lines.append("")

        # Funktionen einfügen
        for func_name, func_frames in self._functions.items():
            lines.append(self._frames_to_function(func_name, func_frames))
            lines.append("")

        # Hauptsequenz
        lines.append("# Hauptsequenz")
        prev_time = None
        prev_state = None

        for frame in frames:
            state = frame["state"]
            action = frame["action"]
            ts = frame["timestamp"]

            # Wait einfügen — IMMER die tatsächliche Zeit zwischen Frames
            if prev_time:
                wait_time = round(ts - prev_time, 3)
                if wait_time > 0.005:  # Nur wenn > 5ms (filter noise)
                    lines.append(f"wait {wait_time}")

            # Nur move wenn sich was geändert hat
            if prev_state is None or self._state_changed(prev_state, state):
                move_parts = []
                move_parts.append(f"base={state.base_deg:.0f}")
                move_parts.append(f"shoulder={state.shoulder_deg:.0f}")
                move_parts.append(f"elbow={state.elbow_deg:.0f}")
                move_parts.append(f"hand={state.hand_deg:.0f}")
                lines.append(f"move {' '.join(move_parts)}")

            # Gripper
            if action == "gripper_open":
                lines.append("gripper open")
            elif action == "gripper_close":
                lines.append("gripper close")

            prev_time = ts
            prev_state = state

        with open(filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        return filename

    def _frames_to_function(self, name: str, frames: List[Dict]) -> str:
        """Konvertiert Frames in eine DSL-Funktion."""
        if not frames:
            return f"function {name}():\n  # leer"

        lines = [f"function {name}(speed=medium):"]

        prev_state = None
        prev_time = None

        for frame in frames:
            state = frame["state"]
            ts = frame["timestamp"]

            # Wait einfügen — tatsächliche Zeit
            if prev_time:
                wait_time = round(ts - prev_time, 3)
                if wait_time > 0.005:
                    lines.append(f"  wait {wait_time}")

            if prev_state is None or self._state_changed(prev_state, state):
                move_parts = []
                move_parts.append(f"base={state.base_deg:.0f}")
                move_parts.append(f"shoulder={state.shoulder_deg:.0f}")
                move_parts.append(f"elbow={state.elbow_deg:.0f}")
                move_parts.append(f"hand={state.hand_deg:.0f}")
                move_parts.append("speed=$speed")
                lines.append(f"  move {' '.join(move_parts)}")

            if frame["action"] == "gripper_open":
                lines.append("  gripper open")
            elif frame["action"] == "gripper_close":
                lines.append("  gripper close")

            prev_state = state
            prev_time = ts

        return '\n'.join(lines)

    def _state_changed(self, a: "ArmState", b: "ArmState", threshold: float = 1.0) -> bool:
        """Prüft ob sich der State signifikant geändert hat."""
        return (
            abs(a.base_deg - b.base_deg) > threshold or
            abs(a.shoulder_deg - b.shoulder_deg) > threshold or
            abs(a.elbow_deg - b.elbow_deg) > threshold or
            abs(a.hand_deg - b.hand_deg) > threshold or
            a.gripper_open != b.gripper_open
        )

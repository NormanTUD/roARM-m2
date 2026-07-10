"""
RoArm DSL — Parser, AST, and Step-by-Step Interpreter
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum, auto
from pathlib import Path
import re


# ═══ AST Nodes ═══════════════════════════════════════════════════════════

class NodeType(Enum):
    MOVE = auto()
    MOVE_REL = auto()
    MOVE_CARTESIAN = auto()
    GRIPPER = auto()
    WAIT = auto()
    LED = auto()
    DETECT = auto()
    CENTER_ON = auto()
    SET = auto()
    HOME = auto()
    PARK = auto()
    PRINT = auto()
    IF = auto()
    REPEAT = auto()
    FUNCTION_DEF = auto()
    FUNCTION_CALL = auto()
    DEFAULTS = auto()
    MAIN = auto()


@dataclass
class ASTNode:
    type: NodeType
    params: Dict[str, Any] = field(default_factory=dict)
    children: List['ASTNode'] = field(default_factory=list)
    else_children: List['ASTNode'] = field(default_factory=list)
    line_number: int = 0
    source_line: str = ""


@dataclass
class Function:
    name: str
    params: Dict[str, Any]  # param_name → default_value
    body: List[ASTNode] = field(default_factory=list)


@dataclass
class Program:
    defaults: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Function] = field(default_factory=dict)
    main: List[ASTNode] = field(default_factory=list)
    source_path: Optional[str] = None


# ═══ Parser ══════════════════════════════════════════════════════════════

class DSLParser:
    """Parses .roarm files into an AST (Program)."""

    def parse_file(self, path: str) -> Program:
        with open(path, 'r') as f:
            lines = f.readlines()
        return self.parse_lines(lines, source_path=path)

    def parse_string(self, text: str) -> Program:
        return self.parse_lines(text.splitlines(keepends=True))

    def parse_lines(self, lines: List[str], source_path: str = None) -> Program:
        program = Program(source_path=source_path)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith('#'):
                continue

            if line == 'defaults:':
                i = self._parse_defaults(lines, i, program)
            elif line.startswith('function '):
                i = self._parse_function(lines, i, line, program)
            elif line == 'main:':
                program.main, i = self._parse_block(lines, i)
            else:
                # Top-level statement (implicit main)
                node = self._parse_statement(line, i - 1)
                if node:
                    program.main.append(node)

        return program

    def _parse_defaults(self, lines, i, program) -> int:
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            if not line.startswith(' ') and not line.startswith('\t'):
                if ':' not in line or line.endswith(':'):
                    break
            # Parse key: value
            match = re.match(r'\s*(\w+):\s*(.+)', lines[i])
            if match:
                key, value = match.group(1), match.group(2).strip()
                program.defaults[key] = self._parse_value(value)
            i += 1
        return i

    def _parse_function(self, lines, i, header_line, program) -> int:
        # function name(param1=default1, param2=default2):
        match = re.match(r'function\s+(\w+)\s*\(([^)]*)\)\s*:', header_line)
        if not match:
            raise SyntaxError(f"Invalid function definition: {header_line}")

        name = match.group(1)
        params_str = match.group(2)
        params = self._parse_params(params_str)

        # Skip docstring
        if i < len(lines) and '"""' in lines[i]:
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1

        body, i = self._parse_block(lines, i)
        program.functions[name] = Function(name=name, params=params, body=body)
        return i

    def _parse_block(self, lines, i, indent_level=1) -> tuple:
        """Parse an indented block until 'end' or dedent."""
        nodes = []
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith('#'):
                i += 1
                continue

            if line == 'end':
                i += 1
                break

            # Check for sub-blocks
            if line.startswith('if ') or line.startswith('repeat '):
                node, i = self._parse_compound(lines, i, line)
                nodes.append(node)
            else:
                node = self._parse_statement(line, i)
                if node:
                    nodes.append(node)
                i += 1

        return nodes, i

    def _parse_compound(self, lines, i, line) -> tuple:
        """Parse if/repeat with their blocks."""
        if line.startswith('if '):
            condition = line[3:].rstrip(':')
            i += 1
            body, i = self._parse_block(lines, i)

            else_body = []
            if i < len(lines) and lines[i].strip().startswith('else'):
                i += 1
                else_body, i = self._parse_block(lines, i)

            node = ASTNode(
                type=NodeType.IF,
                params={'condition': condition},
                children=body,
                else_children=else_body,
                line_number=i,
                source_line=line
            )
            return node, i

        elif line.startswith('repeat '):
            params = self._parse_repeat_header(line)
            i += 1
            body, i = self._parse_block(lines, i)
            node = ASTNode(
                type=NodeType.REPEAT,
                params=params,
                children=body,
                line_number=i,
                source_line=line
            )
            return node, i

        return None, i + 1

    def _parse_statement(self, line: str, line_num: int) -> Optional[ASTNode]:
        """Parse a single statement line."""
        parts = line.split()
        if not parts:
            return None

        cmd = parts[0]
        rest = line[len(cmd):].strip()

        type_map = {
            'move': NodeType.MOVE,
            'move_rel': NodeType.MOVE_REL,
            'gripper': NodeType.GRIPPER,
            'wait': NodeType.WAIT,
            'led': NodeType.LED,
            'detect': NodeType.DETECT,
            'center_on': NodeType.CENTER_ON,
            'set': NodeType.SET,
            'home': NodeType.HOME,
            'park': NodeType.PARK,
            'print': NodeType.PRINT,
        }

        if cmd in type_map:
            params = self._parse_inline_params(rest, cmd)
            return ASTNode(
                type=type_map[cmd],
                params=params,
                line_number=line_num,
                source_line=line
            )

        # Check if it's a cartesian move
        if cmd == 'move' and any(k in rest for k in ['x=', 'y=', 'z=']):
            params = self._parse_inline_params(rest, cmd)
            return ASTNode(
                type=NodeType.MOVE_CARTESIAN,
                params=params,
                line_number=line_num,
                source_line=line
            )

        # Function call
        match = re.match(r'(\w+)\s*(.*)', line)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)
            params = self._parse_inline_params(args_str, func_name)
            params['_function_name'] = func_name
            return ASTNode(
                type=NodeType.FUNCTION_CALL,
                params=params,
                line_number=line_num,
                source_line=line
            )

        return None

    def _parse_inline_params(self, text: str, cmd: str) -> Dict[str, Any]:
        """Parse key=value pairs from a line."""
        params = {}

        # Handle positional args (e.g., 'wait 0.5', 'led 128')
        if cmd in ('wait', 'led', 'print'):
            params['value'] = self._parse_value(text)
            return params

        if cmd == 'gripper':
            parts = text.split()
            params['action'] = parts[0] if parts else 'open'
            for part in parts[1:]:
                if '=' in part:
                    k, v = part.split('=', 1)
                    params[k] = self._parse_value(v)
            return params

        # Key=value pairs
        for match in re.finditer(r'(\w+)=({[^}]+}|"[^"]*"|\S+)', text):
            key, value = match.group(1), match.group(2)
            params[key] = self._parse_value(value)

        return params

    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """Parse function parameter definitions."""
        params = {}
        if not params_str.strip():
            return params
        for part in params_str.split(','):
            part = part.strip()
            if '=' in part:
                name, default = part.split('=', 1)
                params[name.strip()] = self._parse_value(default.strip())
            else:
                params[part.strip()] = None
        return params

    def _parse_repeat_header(self, line: str) -> Dict[str, Any]:
        """Parse repeat header: 'repeat 5:' or 'repeat from=-90 to=90 step=30 as angle:'"""
        params = {}
        rest = line[7:].rstrip(':').strip()

        if rest.isdigit():
            params['count'] = int(rest)
        else:
            for match in re.finditer(r'(\w+)=([^\s]+)', rest):
                params[match.group(1)] = self._parse_value(match.group(2))
            as_match = re.search(r'as\s+(\w+)', rest)
            if as_match:
                params['variable'] = as_match.group(1)

        return params

    def _parse_value(self, text: str) -> Any:
        """Parse a value string into Python type."""
        text = text.strip().strip('"').strip("'")
        if text in ('true', 'True'):
            return True
        if text in ('false', 'False'):
            return False
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
        # Check for expression with braces {expr}
        if text.startswith('{') and text.endswith('}'):
            return ('expr', text[1:-1])
        return text


# ═══ Step-by-Step Interpreter ════════════════════════════════════════════

class InterpreterState(Enum):
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STEPPING = auto()
    FINISHED = auto()
    ERROR = auto()


@dataclass
class ExecutionContext:
    """Runtime state for the interpreter."""
    variables: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    last_detections: List[Dict] = field(default_factory=list)
    step_count: int = 0
    current_node: Optional[ASTNode] = None


class DSLInterpreter:
    """
    Step-by-step interpreter for .roarm programs.
    
    Supports:
    - Single-step execution (for debugging)
    - Continuous run
    - Pause/Resume
    - Variable inspection
    - Breakpoints
    """

    def __init__(self, arm, vision, defaults: Dict = None):
        self._arm = arm          # lib.arm.RoArmController
        self._vision = vision    # lib.vision.VisionPipeline
        self._state = InterpreterState.READY
        self._context = ExecutionContext()
        self._program: Optional[Program] = None
        self._execution_queue: List[ASTNode] = []
        self._breakpoints: set = set()  # line numbers

        # Callbacks for UI
        self.on_step: Optional[Callable] = None      # Called before each step
        self.on_detect: Optional[Callable] = None    # Called when detect runs
        self.on_error: Optional[Callable] = None

        # Apply defaults
        if defaults:
            self._context.variables.update(defaults)

    def load(self, program: Program):
        """Load a parsed program."""
        self._program = program
        self._context.variables.update(program.defaults)
        self._execution_queue = list(program.main)
        self._state = InterpreterState.READY

    def step(self) -> bool:
        """Execute one step. Returns False when finished."""
        if not self._execution_queue:
            self._state = InterpreterState.FINISHED
            return False

        node = self._execution_queue.pop(0)
        self._context.current_node = node
        self._context.step_count += 1

        # Callback
        if self.on_step:
            self.on_step(node, self._context)

        # Check breakpoint
        if node.line_number in self._breakpoints:
            self._state = InterpreterState.PAUSED
            return True

        try:
            self._execute_node(node)
        except Exception as e:
            self._state = InterpreterState.ERROR
            if self.on_error:
                self.on_error(e, node)
            raise

        return True

    def run(self):
        """Run until finished or paused."""
        self._state = InterpreterState.RUNNING
        while self._state == InterpreterState.RUNNING:
            if not self.step():
                break

    def pause(self):
        self._state = InterpreterState.PAUSED

    def resume(self):
        self._state = InterpreterState.RUNNING
        self.run()

    def _execute_node(self, node: ASTNode):
        """Execute a single AST node."""
        handlers = {
            NodeType.MOVE: self._exec_move,
            NodeType.MOVE_REL: self._exec_move_rel,
            NodeType.MOVE_CARTESIAN: self._exec_move_cartesian,
            NodeType.GRIPPER: self._exec_gripper,
            NodeType.WAIT: self._exec_wait,
            NodeType.LED: self._exec_led,
            NodeType.DETECT: self._exec_detect,
            NodeType.CENTER_ON: self._exec_center_on,
            NodeType.SET: self._exec_set,
            NodeType.HOME: self._exec_home,
            NodeType.PARK: self._exec_park,
            NodeType.PRINT: self._exec_print,
            NodeType.IF: self._exec_if,
            NodeType.REPEAT: self._exec_repeat,
            NodeType.FUNCTION_CALL: self._exec_function_call,
        }

        handler = handlers.get(node.type)
        if handler:
            handler(node)

    def _resolve_param(self, value, context=None):
        """Resolve parameter value (handle expressions and variables)."""
        if isinstance(value, tuple) and value[0] == 'expr':
            # Simple expression evaluation with variables
            expr = value[1]
            for var_name, var_val in self._context.variables.items():
                expr = expr.replace(var_name, str(var_val))
            return eval(expr)  # Safe in controlled DSL context
        return value

    def _exec_move(self, node: ASTNode):
        p = {k: self._resolve_param(v) for k, v in node.params.items()}
        self._arm.move_joints(
            base=p.get('base'),
            shoulder=p.get('shoulder'),
            elbow=p.get('elbow'),
            hand=p.get('hand'),
            speed=p.get('speed', self._context.variables.get('speed', 'medium'))
        )

    def _exec_move_rel(self, node: ASTNode):
        p = {k: self._resolve_param(v) for k, v in node.params.items()}
        self._arm.move_joints_relative(
            base=p.get('base', 0),
            shoulder=p.get('shoulder', 0),
            elbow=p.get('elbow', 0),
            hand=p.get('hand', 0),
        )

    def _exec_move_cartesian(self, node: ASTNode):
        p = {k: self._resolve_param(v) for k, v in node.params.items()}
        self._arm.move_cartesian(
            x=p.get('x'), y=p.get('y'), z=p.get('z')
        )

    def _exec_gripper(self, node: ASTNode):
        action = node.params.get('action', 'open')
        force = node.params.get('force', 300)
        if action == 'open':
            self._arm.gripper_open()
        else:
            self._arm.gripper_close(force=force)

    def _exec_wait(self, node: ASTNode):
        import time
        duration = self._resolve_param(node.params.get('value', 0.5))
        time.sleep(float(duration))

    def _exec_led(self, node: ASTNode):
        value = node.params.get('value', 0)
        if value == 'off':
            value = 0
        self._arm.set_led(int(self._resolve_param(value)))

    def _exec_detect(self, node: ASTNode):
        """Run YOLO detection, store results."""
        target = node.params.get('target')
        detections = self._vision.detect(target_classes=[target] if target else None)
        self._context.last_detections = detections
        if self.on_detect:
            self.on_detect(detections)

    def _exec_center_on(self, node: ASTNode):
        target = node.params.get('target')
        threshold = node.params.get('threshold', 20)
        max_iter = node.params.get('max_iterations', 8)
        self._vision.center_arm_on_target(
            self._arm, target, threshold_px=threshold, max_iter=max_iter
        )

    def _exec_set(self, node: ASTNode):
        for key, value in node.params.items():
            self._context.variables[key] = self._resolve_param(value)
            if key == 'speed':
                self._arm.set_speed_level(value)

    def _exec_home(self, node: ASTNode):
        self._arm.home()

    def _exec_park(self, node: ASTNode):
        self._arm.park()

    def _exec_print(self, node: ASTNode):
        value = node.params.get('value', '')
        print(f"  [DSL] {value}")

    def _exec_if(self, node: ASTNode):
        condition = node.params.get('condition', '')
        result = self._evaluate_condition(condition)
        if result:
            # Prepend if-body to execution queue
            self._execution_queue = node.children + self._execution_queue
        elif node.else_children:
            self._execution_queue = node.else_children + self._execution_queue

    def _exec_repeat(self, node: ASTNode):
        params = node.params
        if 'count' in params:
            count = int(self._resolve_param(params['count']))
            expanded = node.children * count
            self._execution_queue = expanded + self._execution_queue
        elif 'from' in params and 'to' in params:
            start = int(self._resolve_param(params['from']))
            end = int(self._resolve_param(params['to']))
            step = int(self._resolve_param(params.get('step', 1)))
            var_name = params.get('variable', '_i')

            expanded = []
            for val in range(start, end + 1, step):
                # Set variable before each iteration
                set_node = ASTNode(
                    type=NodeType.SET,
                    params={var_name: val}
                )
                expanded.append(set_node)
                expanded.extend(node.children)

            self._execution_queue = expanded + self._execution_queue

    def _exec_function_call(self, node: ASTNode):
        func_name = node.params.get('_function_name')
        if not self._program or func_name not in self._program.functions:
            raise RuntimeError(f"Unknown function: {func_name}")

        func = self._program.functions[func_name]

        # Bind parameters (with defaults)
        for param_name, default_val in func.params.items():
            actual_val = node.params.get(param_name, default_val)
            self._context.variables[param_name] = self._resolve_param(actual_val)

        # Push function body to front of execution queue
        self._context.call_stack.append(func_name)
        self._execution_queue = list(func.body) + self._execution_queue

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition string."""
        # detected("class_name")
        match = re.match(r'detected\("([^"]+)"\)', condition)
        if match:
            target = match.group(1)
            return any(d['class'] == target for d in self._context.last_detections)

        # has_target
        if condition == 'has_target':
            return len(self._context.last_detections) > 0

        return False

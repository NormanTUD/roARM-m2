import time
import logging


class TUILogHandler(logging.Handler):
    """Leitet robot.py Warnungen ins Teach-Log der TUI."""

    def __init__(self, app):
        super().__init__()
        self.app = app

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.WARNING:
                styled = f"[yellow]{msg}[/]"
            else:
                styled = f"[dim]{msg}[/]"

            try:
                self.app._log_teach(styled)
            except Exception:
                self.app.call_from_thread(self.app._log_teach, styled)
        except Exception:
            pass


class JointHistory:
    """Haelt die letzten N Werte pro Gelenk fuer Sparklines."""

    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.data = {"b": [], "s": [], "e": [], "h": []}

    def push(self, pos: dict):
        for j in ["b", "s", "e", "h"]:
            self.data[j].append(pos.get(j, 0.0))
            if len(self.data[j]) > self.max_len:
                self.data[j].pop(0)

    def get(self, joint: str) -> list:
        return self.data.get(joint, [])

    def clear(self):
        self.data = {"b": [], "s": [], "e": [], "h": []}


class ActivityIndicator:
    """Manages animated spinner states for the status bar."""

    SPINNER_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2823", "\u280f"]
    DOTS_FRAMES = [".", "..", "...", ".."]

    def __init__(self):
        self._active = False
        self._message = ""
        self._icon = ""
        self._frame_index = 0
        self._start_time = 0.0

    def start(self, message: str, icon: str = "\u23f3"):
        self._active = True
        self._message = message
        self._icon = icon
        self._frame_index = 0
        self._start_time = time.time()

    def stop(self):
        self._active = False
        self._message = ""
        self._icon = ""
        self._frame_index = 0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def elapsed(self) -> float:
        if not self._active:
            return 0.0
        return time.time() - self._start_time

    def next_frame(self) -> str:
        if not self._active:
            return ""
        spinner = self.SPINNER_FRAMES[self._frame_index % len(self.SPINNER_FRAMES)]
        dots = self.DOTS_FRAMES[self._frame_index % len(self.DOTS_FRAMES)]
        elapsed = self.elapsed
        self._frame_index += 1
        dots_padded = f"{dots:<3}"
        return f"{self._icon} {spinner} {self._message}{dots_padded} [{elapsed:.1f}s]"

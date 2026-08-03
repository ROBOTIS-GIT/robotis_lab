"""Runtime reset shortcut handling for Isaac Sim and terminal sessions."""

from __future__ import annotations

import atexit
import select
import sys
import termios
import threading
import tty


class ResetRequestHandler:
    _RESET_KEYS = {"r", "R"}

    def __init__(self, enable_gui: bool, enable_stdin: bool = True):
        self.gui_enabled = False
        self.stdin_enabled = False
        self._reset_requested = False
        self._lock = threading.Lock()
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._carb = None
        self._stdin_thread = None
        self._stdin_stop_event = threading.Event()
        self._stdin_fd = None
        self._stdin_attrs = None
        self._closed = False

        if enable_gui:
            self._try_enable_gui_keyboard()
        if enable_stdin:
            self._try_enable_stdin()

        if self.gui_enabled or self.stdin_enabled:
            sources = []
            if self.gui_enabled:
                sources.append("Isaac Sim window")
            if self.stdin_enabled:
                sources.append("terminal")
            print(f"[INFO] Reset shortcut enabled: press R in {' or '.join(sources)}.")

    def _try_enable_gui_keyboard(self):
        try:
            import carb
            import omni.appwindow

            appwindow = omni.appwindow.get_default_app_window()
            if appwindow is None:
                return
            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = appwindow.get_keyboard()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            self.gui_enabled = True
        except Exception as exc:
            print(f"[WARN] GUI reset shortcut unavailable: {exc}")

    def _try_enable_stdin(self):
        if not sys.stdin.isatty():
            return

        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._stdin_thread = threading.Thread(target=self._read_stdin_loop, daemon=True)
            self._stdin_thread.start()
            self.stdin_enabled = True
            atexit.register(self.close)
        except Exception as exc:
            self._restore_stdin()
            print(f"[WARN] Terminal reset shortcut unavailable: {exc}")

    def _read_stdin_loop(self):
        while not self._stdin_stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char in self._RESET_KEYS:
                self._request_reset()

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_input = getattr(event, "input", "")
        input_name = getattr(event_input, "name", event_input)
        if not isinstance(input_name, str):
            input_name = str(input_name)
        if (
            event.type == self._carb.input.KeyboardEventType.KEY_PRESS
            and input_name in self._RESET_KEYS
        ):
            self._request_reset()
        return True

    def _request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset_request(self) -> bool:
        with self._lock:
            reset_requested = self._reset_requested
            self._reset_requested = False
        return reset_requested

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stdin_stop_event.set()
        self._restore_stdin()
        if self._stdin_thread is not None:
            self._stdin_thread.join(timeout=0.5)
        if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _restore_stdin(self):
        if self._stdin_fd is not None and self._stdin_attrs is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSANOW, self._stdin_attrs)
            self._stdin_attrs = None

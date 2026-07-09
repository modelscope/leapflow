"""Application-based TUI controller — the hybrid architecture core.

Combines prompt_toolkit's Application (persistent layout, fixed input,
modal-ready widget system) with Rich rendering (Markdown, panels, syntax
highlighting) for the scrollback output region.

Architecture
~~~~~~~~~~~~
All logic runs on a single asyncio event loop — no threads::

    Application ← handle_enter (key binding)
         ↑              ↓
         │     pending_input (asyncio.Queue)
    invalidate()        ↓
         ←── _process_loop (asyncio.Task)
                        ↓
                on_input(text) ← business logic in interactive.py
                        ↓
           Console.print → patch_stdout → output above layout
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    Sequence,
    Union,
)

from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea

from leapflow.cli.tui_app.command import TuiCommand
from leapflow.cli.tui_app.input import build_completer
from leapflow.cli.tui_app.status import StatusBar
from leapflow.cli.tui_app.theme import Theme

if TYPE_CHECKING:
    from leapflow.cli.tui_app.console import LeapConsole

_HISTORY_FILENAME = "history"
_REFRESH_INTERVAL_S = 0.5

InputHandler = Callable[[str], Union[Awaitable[None], None]]


class LeapApp:
    """Hybrid TUI: prompt_toolkit Application + Rich output.

    Owns the Application, HSplit layout (spinner → status bar → input),
    and the async worker task that drains input and dispatches to the
    caller-supplied ``on_input`` callback.

    All output flows through ``rich.Console`` via ``patch_stdout``,
    appearing above the fixed layout at the terminal bottom.
    """

    def __init__(
        self,
        *,
        console: "LeapConsole",
        theme: Theme,
        status: StatusBar,
        commands: Sequence[tuple[str, str]] = (),
        data_dir: Optional[Path] = None,
        on_input: Optional[InputHandler] = None,
    ) -> None:
        self._console = console
        self._theme = theme
        self._status = status
        self._on_input = on_input

        self._should_exit = False
        self._agent_running = False
        self._prompt_mode = "idle"
        self._spinner_text = ""
        self._tool_start_time: float = 0.0
        self._next_command_id = 1
        self._active_command: Optional[TuiCommand] = None
        self._pending_input: asyncio.Queue[TuiCommand] = asyncio.Queue()

        data_dir = data_dir or Path(
            os.environ.get("LEAPFLOW_DATA_DIR", "~/.leapflow")
        ).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = data_dir / _HISTORY_FILENAME

        self._input_area = self._build_input_area(commands)
        self._app = self._build_application()

    # ── Public state properties ──────────────────────────────────────

    @property
    def agent_running(self) -> bool:
        return self._agent_running

    @agent_running.setter
    def agent_running(self, value: bool) -> None:
        self._agent_running = value
        self._invalidate()

    @property
    def prompt_mode(self) -> str:
        return self._prompt_mode

    @prompt_mode.setter
    def prompt_mode(self, value: str) -> None:
        self._prompt_mode = value
        self._invalidate()

    @property
    def spinner_text(self) -> str:
        return self._spinner_text

    @spinner_text.setter
    def spinner_text(self, value: str) -> None:
        self._spinner_text = value
        self._tool_start_time = time.monotonic() if value else 0.0
        self._invalidate()

    def submit_text(self, text: str) -> TuiCommand:
        """Submit user text into the serial TUI command queue."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("Cannot submit an empty TUI command")
        command = TuiCommand.create(command_id=self._next_command_id, text=normalized)
        self._next_command_id += 1
        self._pending_input.put_nowait(command)
        self._sync_task_counts()
        if self._active_command is not None or self._pending_input.qsize() > 1:
            self._console.command_card(command)
        self._invalidate()
        return command

    # ── Lifecycle ────────────────────────────────────────────────────

    async def run(self) -> int:
        """Start the TUI — blocks until user exits. Returns exit code."""
        worker = asyncio.create_task(self._process_loop())
        try:
            with patch_stdout(raw=True):
                await self._app.run_async()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            self._should_exit = True
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
        return 0

    def exit(self) -> None:
        """Request graceful TUI shutdown."""
        self._should_exit = True
        if self._app.is_running:
            self._app.exit()

    # ── Async worker ─────────────────────────────────────────────────

    async def _process_loop(self) -> None:
        """Drain pending_input and dispatch to on_input handler."""
        while not self._should_exit:
            try:
                command = await asyncio.wait_for(
                    self._pending_input.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            if not command.text.strip():
                self._sync_task_counts()
                continue

            self._active_command = command.mark_running()
            self._agent_running = True
            self._sync_task_counts()
            self._console.command_card(self._active_command)
            try:
                if self._on_input is not None:
                    result = self._on_input(command.text)
                    if asyncio.iscoroutine(result):
                        await result
                if self._active_command is not None:
                    finished = self._active_command.mark_done()
                    self._console.command_card(finished)
            except Exception as exc:
                if self._active_command is not None:
                    failed = self._active_command.mark_failed(f"{type(exc).__name__}: {exc}")
                    self._console.command_card(failed)
                self._console.error(f"{exc}")
                self._spinner_text = ""
                self._tool_start_time = 0.0
            finally:
                self._active_command = None
                self._agent_running = False
                self._sync_task_counts()
                self._invalidate()

    # ── Layout construction ──────────────────────────────────────────

    def _build_input_area(
        self, commands: Sequence[tuple[str, str]]
    ) -> TextArea:
        completer = build_completer(commands)
        ref = self

        def get_prompt():
            return ref._prompt_fragments()

        area = TextArea(
            height=Dimension(min=1, max=4, preferred=1),
            prompt=get_prompt,
            style="class:input-area",
            multiline=True,
            wrap_lines=True,
            dont_extend_height=True,
            history=FileHistory(str(self._history_path)),
            completer=completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
        )
        area.buffer.tempfile_suffix = ".md"
        return area

    def _build_application(self) -> Application[Any]:
        spinner = Window(
            content=FormattedTextControl(self._spinner_fragments),
            height=self._spinner_height,
            wrap_lines=True,
        )
        status_bar = Window(
            content=FormattedTextControl(self._status),
            height=1,
            wrap_lines=False,
            style="class:status-bar",
        )
        layout = Layout(HSplit([spinner, status_bar, self._input_area]))

        cursor_kwargs: dict[str, Any] = {}
        try:
            from prompt_toolkit.cursor_shapes import (
                CursorShape,
                SimpleCursorShapeConfig,
            )
            cursor_kwargs["cursor"] = SimpleCursorShapeConfig(CursorShape.BLOCK)
        except ImportError:
            pass

        return Application(
            layout=layout,
            key_bindings=self._build_keybindings(),
            style=self._build_style(),
            full_screen=False,
            mouse_support=False,
            refresh_interval=_REFRESH_INTERVAL_S,
            erase_when_done=True,
            **cursor_kwargs,
        )

    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()
        ref = self

        @kb.add(Keys.Enter)
        def _(event):
            text = event.app.current_buffer.text.strip()
            if not text:
                return
            ref.submit_text(text)
            event.app.current_buffer.reset(append_to_history=True)

        @kb.add(Keys.Escape, Keys.Enter)
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.ControlD)
        def _(event):
            ref._should_exit = True
            event.app.exit()

        @kb.add(Keys.ControlC)
        def _(event):
            if event.current_buffer.text:
                event.current_buffer.reset()
                return
            if ref._agent_running:
                ref._console.system(
                    "Task is running. Keep typing to queue the next instruction; "
                    "use /exit after it finishes to leave."
                )
                return
            ref._should_exit = True
            event.app.exit()

        return kb

    def _build_style(self) -> PTStyle:
        t = self._theme
        input_style = f"bg:{t.input_bg} {t.input_text} bold"
        disabled_style = f"bg:{t.input_bg} {t.input_disabled_text}"
        return PTStyle.from_dict({
            "input-area": input_style,
            "input-area.disabled": disabled_style,
            "prompt": t.prompt_char,
            "prompt.working": t.accent_dim,
            "prompt.recording": t.recording,
            "prompt.paused": t.prompt_paused,
            "prompt.executing": t.executing,
            "status-bar": f"bg:{t.toolbar_bg} {t.toolbar_fg}",
            "status-bar.strong": f"bg:{t.toolbar_bg} bold {t.accent}",
            "status-bar.dim": f"bg:{t.toolbar_bg} {t.text_muted}",
            "status-bar.good": f"bg:{t.toolbar_bg} {t.success}",
            "status-bar.warn": f"bg:{t.toolbar_bg} {t.warning}",
            "status-bar.bad": f"bg:{t.toolbar_bg} {t.error}",
            "hint": t.text_dim,
            "auto-suggest": t.auto_suggest,
            "placeholder": t.input_placeholder,
            "selection": f"bg:{t.input_selection_bg} {t.input_selection_fg}",
        })

    # ── Fragment providers ───────────────────────────────────────────

    def _prompt_fragments(self) -> list[tuple[str, str]]:
        if self._agent_running:
            return [("class:prompt.working", "⚕ ")]
        _mode_prompts = {
            "learning": [
                ("class:prompt.recording", "● rec "),
                ("class:prompt", "❯ "),
            ],
            "paused": [
                ("class:prompt.paused", "⏸ "),
                ("class:prompt", "❯ "),
            ],
            "executing": [("class:prompt.executing", "▶ ")],
        }
        return _mode_prompts.get(self._prompt_mode, [("class:prompt", "❯ ")])

    def _spinner_fragments(self) -> list[tuple[str, str]]:
        if not self._spinner_text:
            return []
        elapsed = ""
        if self._tool_start_time > 0:
            dt = time.monotonic() - self._tool_start_time
            if dt >= 60:
                m, s = int(dt // 60), int(dt % 60)
                elapsed = f"  ({m:02d}m{s:02d}s)"
            else:
                elapsed = f"  ({dt:.1f}s)"
        return [("class:hint", f"  {self._spinner_text}{elapsed}")]

    def _spinner_height(self) -> int:
        return 1 if self._spinner_text else 0

    # ── Internal ─────────────────────────────────────────────────────

    def _sync_task_counts(self) -> None:
        self._status.update_task_counts(
            running=1 if self._active_command is not None else 0,
            queued=self._pending_input.qsize(),
        )

    def _invalidate(self) -> None:
        if self._app.is_running:
            self._app.invalidate()

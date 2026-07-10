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
from leapflow.cli.tui_app.paste import (
    PASTE_FRAGMENT_WINDOW_S,
    PasteHeuristics,
    PasteStore,
)
from leapflow.cli.tui_app.status import StatusBar
from leapflow.cli.tui_app.theme import Theme

if TYPE_CHECKING:
    from leapflow.cli.tui_app.console import LeapConsole

_HISTORY_FILENAME = "history"
_REFRESH_INTERVAL_S = 0.5
_PASTE_COMPACTOR_ATTR = "_leapflow_paste_compactor_installed"

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
        self._paste_heuristics = PasteHeuristics()
        self._paste_store = PasteStore(self._paste_heuristics)
        self._paste_blocks = self._paste_store.blocks
        self._paste_fragment_text = ""
        self._paste_fragment_start_cursor = 0
        self._paste_fragment_last_at = 0.0
        self._active_fragment_marker: Optional[str] = None
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
        normalized = self._resolve_paste_blocks(text).strip()
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

    def _should_compact_paste(self, text: str) -> bool:
        """Return True when rendering pasted text directly would hurt TUI responsiveness."""
        return self._paste_heuristics.should_compact_block(text)

    def _paste_marker(self, text: str) -> str:
        """Create a safe visible marker while keeping full text in the side channel."""
        return self._paste_store.create_marker(text)

    def _resolve_paste_blocks(self, text: str) -> str:
        """Replace compact paste markers with the original full pasted content."""
        return self._paste_store.resolve(text)

    def _clear_paste_state(self) -> None:
        """Clear both side-channel paste blocks and in-flight fragment state."""
        self._paste_store.clear()
        self._reset_paste_fragment_window()

    def _reset_paste_fragment_window(self) -> None:
        """Forget pending fragmented paste detection state."""
        self._paste_fragment_text = ""
        self._paste_fragment_start_cursor = 0
        self._paste_fragment_last_at = 0.0
        self._active_fragment_marker = None

    def _fragment_continues(self, now: float) -> bool:
        """Return True when an insert arrives inside the paste-fragment window."""
        if self._paste_fragment_last_at <= 0:
            return False
        return now - self._paste_fragment_last_at <= PASTE_FRAGMENT_WINDOW_S

    def _replace_fragment_with_marker(self, buffer: Any, marker: str) -> bool:
        """Replace already-rendered fragmented paste text with a safe marker."""
        start = self._paste_fragment_start_cursor
        fragment = self._paste_fragment_text
        end = start + len(fragment)
        try:
            current = buffer.text
            if current[start:end] != fragment:
                return False
            buffer.text = f"{current[:start]}{marker}{current[end:]}"
            buffer.cursor_position = start + len(marker)
            return True
        except (AttributeError, TypeError, ValueError):
            return False

    def _insert_paste_text(self, buffer: Any, text: str) -> None:
        """Insert pasted text, compacting large blocks to keep terminal rendering smooth."""
        if not text:
            return
        if self._should_compact_paste(text):
            buffer.insert_text(self._paste_marker(text))
            self._reset_paste_fragment_window()
            self._invalidate()
            return
        buffer.insert_text(text)

    def _install_paste_compactor(self, buffer: Any) -> None:
        """Compact bulk and fragmented Buffer inserts before they hurt rendering."""
        if getattr(buffer, _PASTE_COMPACTOR_ATTR, False):
            return
        original_insert_text = buffer.insert_text
        ref = self

        def insert_text(text: str, *args: Any, **kwargs: Any) -> None:
            if not isinstance(text, str) or not text:
                original_insert_text(text, *args, **kwargs)
                return

            now = time.monotonic()
            if ref._active_fragment_marker and ref._fragment_continues(now):
                ref._paste_store.append_to_marker(ref._active_fragment_marker, text)
                ref._paste_fragment_last_at = now
                ref._invalidate()
                return
            if not ref._fragment_continues(now):
                ref._reset_paste_fragment_window()

            if ref._should_compact_paste(text):
                original_insert_text(ref._paste_marker(text), *args, **kwargs)
                ref._paste_fragment_last_at = now
                ref._invalidate()
                return

            if not ref._paste_fragment_text:
                ref._paste_fragment_start_cursor = int(getattr(buffer, "cursor_position", 0))
            original_insert_text(text, *args, **kwargs)
            ref._paste_fragment_text += text
            ref._paste_fragment_last_at = now

            if ref._paste_heuristics.should_compact_fragment_window(ref._paste_fragment_text):
                marker = ref._paste_marker(ref._paste_fragment_text)
                if ref._replace_fragment_with_marker(buffer, marker):
                    ref._paste_fragment_text = ""
                    ref._active_fragment_marker = marker
                    ref._paste_fragment_last_at = now
                    ref._invalidate()

        buffer.insert_text = insert_text
        setattr(buffer, _PASTE_COMPACTOR_ATTR, True)

    def _accept_auto_suggestion(self, buffer: Any) -> bool:
        """Accept the visible auto-suggestion when one is available."""
        suggestion = getattr(buffer, "suggestion", None)
        suggestion_text = getattr(suggestion, "text", "") if suggestion else ""
        if not suggestion_text:
            return False
        buffer.insert_text(suggestion_text)
        return True

    def _accept_or_start_completion(self, buffer: Any) -> None:
        """Accept the selected completion or open/cycle completion candidates."""
        complete_state = getattr(buffer, "complete_state", None)
        current_completion = getattr(complete_state, "current_completion", None)
        if current_completion is not None:
            buffer.apply_completion(current_completion)
            return
        if self._accept_auto_suggestion(buffer):
            return
        buffer.complete_next()

    def _move_right_or_accept_suggestion(self, buffer: Any) -> None:
        """Keep normal Right-arrow movement while accepting visible suggestions at EOL."""
        if getattr(buffer, "cursor_position", 0) >= len(getattr(buffer, "text", "")):
            if self._accept_auto_suggestion(buffer):
                return
        buffer.cursor_right()

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
        self._install_paste_compactor(area.buffer)
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
            has_compacted_paste = ref._paste_store.has_blocks
            ref.submit_text(text)
            event.app.current_buffer.reset(append_to_history=not has_compacted_paste)
            ref._reset_paste_fragment_window()

        @kb.add(Keys.Escape, Keys.Enter)
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.Tab)
        def _(event):
            ref._accept_or_start_completion(event.current_buffer)

        @kb.add(Keys.Right)
        def _(event):
            ref._move_right_or_accept_suggestion(event.current_buffer)

        @kb.add(Keys.BracketedPaste)
        def _(event):
            ref._insert_paste_text(event.current_buffer, event.data)

        @kb.add(Keys.ControlD)
        def _(event):
            ref._should_exit = True
            event.app.exit()

        @kb.add(Keys.ControlC)
        def _(event):
            if event.current_buffer.text:
                event.current_buffer.reset()
                ref._clear_paste_state()
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

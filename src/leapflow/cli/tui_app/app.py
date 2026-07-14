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
import shutil
import time
from collections import deque
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
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text.utils import fragment_list_len
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer, Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import TextArea

from leapflow.cli.tui_app.approval_modal import ApprovalModal, request_is_expired
from leapflow.cli.tui_app.command import TuiCommand, TuiCommandStatus
from leapflow.cli.tui_app.input import build_completer
from leapflow.cli.tui_app.paste import (
    PASTE_FRAGMENT_WINDOW_S,
    PasteHeuristics,
    PasteStore,
)
from leapflow.cli.tui_app.status import StatusBar
from leapflow.cli.tui_app.theme import Theme
from leapflow.security.approval import ApprovalDecision, ApprovalRequest

if TYPE_CHECKING:
    from leapflow.cli.tui_app.console import LeapConsole

_HISTORY_FILENAME = "history"
_REFRESH_INTERVAL_S = 0.5
_PASTE_COMPACTOR_ATTR = "_leapflow_paste_compactor_installed"
_PLACEHOLDER_CURSOR_SPACER = "  "

InputHandler = Callable[[str], Union[Awaitable[None], None]]
ControlHandler = Callable[[str], bool]


def _format_inline_duration(seconds: float) -> str:
    """Format short elapsed text for inline TUI guidance."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m{seconds - minutes * 60:.0f}s"


class _DynamicPlaceholderProcessor(Processor):
    """Render the input prompt and contextual placeholder text."""

    def __init__(
        self,
        provider: Callable[[], str],
        prompt_provider: Callable[[], list[tuple[str, str]]],
    ) -> None:
        self._provider = provider
        self._prompt_provider = prompt_provider

    def apply_transformation(self, transformation_input: Any) -> Transformation:
        document = getattr(transformation_input, "document", None)
        is_first_line = getattr(transformation_input, "lineno", 0) == 0
        has_text = bool(getattr(document, "text", "")) if document is not None else False
        if not is_first_line:
            return Transformation(transformation_input.fragments)
        prompt_fragments = self._prompt_provider()
        if has_text:
            prefix = prompt_fragments
            fragments = [*prefix, *transformation_input.fragments]
            shift_position = fragment_list_len(prefix)
            return Transformation(
                fragments,
                source_to_display=lambda index: index + shift_position,
                display_to_source=lambda index: index - shift_position,
            )
        placeholder = self._provider()
        placeholder_fragments = [
            ("class:placeholder", _PLACEHOLDER_CURSOR_SPACER),
            ("class:placeholder", placeholder),
        ] if placeholder else []
        fragments = [*prompt_fragments, *placeholder_fragments, *transformation_input.fragments]
        shift_position = fragment_list_len(prompt_fragments)
        return Transformation(
            fragments,
            source_to_display=lambda index: index + shift_position,
            display_to_source=lambda _index: 0,
        )


class _CommandQueue:
    """Small observable async queue for TUI command scheduling."""

    def __init__(self) -> None:
        self._items: deque[TuiCommand] = deque()
        self._ready = asyncio.Event()

    def put_nowait(self, command: TuiCommand) -> None:
        self._items.append(command)
        self._ready.set()

    async def get(self) -> TuiCommand:
        while not self._items:
            self._ready.clear()
            await self._ready.wait()
        command = self._items.popleft()
        if not self._items:
            self._ready.clear()
        return command

    def get_nowait(self) -> TuiCommand:
        if not self._items:
            raise asyncio.QueueEmpty
        command = self._items.popleft()
        if not self._items:
            self._ready.clear()
        return command

    def qsize(self) -> int:
        return len(self._items)

    def snapshot(self) -> list[TuiCommand]:
        return list(self._items)


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
        on_control: Optional[ControlHandler] = None,
    ) -> None:
        self._console = console
        self._theme = theme
        self._status = status
        self._on_input = on_input
        self._on_control = on_control

        self._should_exit = False
        self._agent_running = False
        self._prompt_mode = "idle"
        self._spinner_text = ""
        self._tool_start_time: float = 0.0
        self._next_command_id = 1
        self._queue_paused = False
        self._active_dispatch_task: Optional[asyncio.Task[Any]] = None
        self._active_terminal_status: Optional[TuiCommandStatus] = None
        self._active_terminal_reason = ""
        self._last_control_c_at = 0.0
        self._paste_heuristics = PasteHeuristics()
        self._paste_store = PasteStore(self._paste_heuristics)
        self._paste_blocks = self._paste_store.blocks
        self._paste_fragment_text = ""
        self._paste_fragment_start_cursor = 0
        self._paste_fragment_last_at = 0.0
        self._active_fragment_marker: Optional[str] = None
        self._active_command: Optional[TuiCommand] = None
        self._pending_input = _CommandQueue()
        self._approval_modal: Optional[ApprovalModal] = None

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

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Show a native TUI approval modal and return the selected decision."""
        if request_is_expired(request):
            return ApprovalDecision.DENY
        if self._approval_modal is not None:
            return ApprovalDecision.DENY
        from leapflow.cli.approval_view import remaining_seconds

        modal = ApprovalModal.create(request)
        self._approval_modal = modal
        self._input_area.buffer.reset()
        self._clear_paste_state()
        self._invalidate()
        try:
            timeout = remaining_seconds(request)
            return await asyncio.wait_for(asyncio.shield(modal.future), timeout=timeout)
        except asyncio.TimeoutError:
            modal.deny()
            return ApprovalDecision.DENY
        finally:
            if self._approval_modal is modal:
                self._approval_modal = None
            self._invalidate()

    def submit_text(self, text: str) -> TuiCommand:
        """Submit user text into the serial TUI command queue."""
        normalized = self._resolve_paste_blocks(text).strip()
        if not normalized:
            raise ValueError("Cannot submit an empty TUI command")
        if self._dispatch_control_text(normalized):
            return TuiCommand.create(command_id=0, text=normalized).mark_done()
        command = TuiCommand.create(command_id=self._next_command_id, text=normalized)
        self._next_command_id += 1
        self._pending_input.put_nowait(command)
        self._sync_task_counts()
        if self._active_command is not None or self._pending_input.qsize() > 1 or self._queue_paused:
            self._console.command_card(command)
        self._invalidate()
        return command

    @property
    def queue_paused(self) -> bool:
        """Return whether queued commands are currently held."""
        return self._queue_paused

    @property
    def active_command(self) -> Optional[TuiCommand]:
        """Return the currently running command, if any."""
        return self._active_command

    def complete_active_command_in_response(self) -> Optional[TuiCommand]:
        """Mark the active command done when its status is rendered with the response label."""
        if self._active_command is None:
            return None
        completed = self._active_command.mark_done()
        self._active_command = completed
        self._active_terminal_status = TuiCommandStatus.DONE
        self._active_terminal_reason = ""
        self._sync_task_counts()
        self._invalidate()
        return completed

    def block_active_command_in_response(self, reason: str) -> Optional[TuiCommand]:
        """Mark the active command blocked when recovery guidance is rendered inline."""
        if self._active_command is None:
            return None
        blocked = self._active_command.mark_blocked(reason)
        self._active_command = blocked
        self._active_terminal_status = TuiCommandStatus.BLOCKED
        self._active_terminal_reason = reason
        self._sync_task_counts()
        self._invalidate()
        return blocked

    def queued_commands(self) -> list[TuiCommand]:
        """Return a snapshot of pending commands in queue order."""
        return self._pending_input.snapshot()

    def pause_queue(self) -> bool:
        """Pause starting future queued commands; current work continues."""
        if self._queue_paused:
            return False
        self._queue_paused = True
        self._sync_task_counts()
        self._invalidate()
        return True

    def resume_queue(self) -> bool:
        """Resume starting queued commands."""
        if not self._queue_paused:
            return False
        self._queue_paused = False
        self._sync_task_counts()
        self._invalidate()
        return True

    def clear_queued_commands(self, reason: str = "cleared by user") -> list[TuiCommand]:
        """Drop every pending command and render them as skipped."""
        dropped: list[TuiCommand] = []
        while True:
            try:
                command = self._pending_input.get_nowait()
            except asyncio.QueueEmpty:
                break
            skipped = command.mark_skipped(reason)
            dropped.append(skipped)
            self._console.command_card(skipped)
        self._sync_task_counts()
        self._invalidate()
        return dropped

    def drop_queued_command(self, command_id: int, reason: str = "dropped by user") -> Optional[TuiCommand]:
        """Drop one queued command by id while preserving queue order."""
        kept: list[TuiCommand] = []
        dropped: Optional[TuiCommand] = None
        while True:
            try:
                command = self._pending_input.get_nowait()
            except asyncio.QueueEmpty:
                break
            if command.id == command_id and dropped is None:
                dropped = command.mark_skipped(reason)
                self._console.command_card(dropped)
            else:
                kept.append(command)
        for command in kept:
            self._pending_input.put_nowait(command)
        self._sync_task_counts()
        self._invalidate()
        return dropped

    def request_cancel_active(self, reason: str = "cancelled by user") -> Optional[TuiCommand]:
        """Mark current work as cancelled and cancel its dispatch task."""
        return self._request_finish_active(TuiCommandStatus.CANCELLED, reason)

    def request_skip_active(self, reason: str = "skipped by user") -> Optional[TuiCommand]:
        """Mark current work as skipped and cancel its dispatch task."""
        return self._request_finish_active(TuiCommandStatus.SKIPPED, reason)

    def _request_finish_active(self, status: TuiCommandStatus, reason: str) -> Optional[TuiCommand]:
        if self._active_command is None:
            return None
        self._active_terminal_status = status
        self._active_terminal_reason = reason
        terminal = self._terminal_command(self._active_command, status, reason)
        self._active_command = terminal
        self._console.command_card(terminal)
        task = self._active_dispatch_task
        if task is not None and not task.done():
            task.cancel()
        self._spinner_text = ""
        self._tool_start_time = 0.0
        self._sync_task_counts()
        self._invalidate()
        return terminal

    def _terminal_command(
        self,
        command: TuiCommand,
        status: TuiCommandStatus,
        reason: str,
    ) -> TuiCommand:
        if status == TuiCommandStatus.CANCELLED:
            return command.mark_cancelled(reason)
        if status == TuiCommandStatus.SKIPPED:
            return command.mark_skipped(reason)
        if status == TuiCommandStatus.BLOCKED:
            return command.mark_blocked(reason)
        return command.mark_failed(reason)

    def _dispatch_control_text(self, text: str) -> bool:
        handler = self._on_control
        if handler is None:
            return False
        try:
            handled = handler(text)
        except Exception as exc:
            self._console.error(f"Task control failed: {exc}")
            return True
        if handled:
            self._invalidate()
        return handled

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

    def _completion_is_open(self, buffer: Any) -> bool:
        """Return True when prompt_toolkit is showing completion candidates."""
        return getattr(buffer, "complete_state", None) is not None

    def _close_completion(self, buffer: Any) -> bool:
        """Close the active completion menu when present."""
        if not self._completion_is_open(buffer):
            return False
        buffer.cancel_completion()
        self._invalidate()
        return True

    def _cursor_at_first_input_line(self, buffer: Any) -> bool:
        """Return True when Up should leave line navigation and enter history."""
        try:
            return buffer.document.cursor_position_row <= 0
        except (AttributeError, TypeError):
            return True

    def _cursor_at_last_input_line(self, buffer: Any) -> bool:
        """Return True when Down should leave line navigation and enter history."""
        try:
            document = buffer.document
            return document.cursor_position_row >= document.line_count - 1
        except (AttributeError, TypeError):
            return True

    def _move_history_backward(self, buffer: Any) -> bool:
        """Move to an older history entry while preserving the current draft."""
        before_index = getattr(buffer, "working_index", None)
        try:
            buffer.history_backward()
        except AttributeError:
            return False
        moved = getattr(buffer, "working_index", None) != before_index
        if moved:
            self._invalidate()
        return moved

    def _move_history_forward(self, buffer: Any) -> bool:
        """Move to a newer history entry, restoring the draft at the newest slot."""
        before_index = getattr(buffer, "working_index", None)
        try:
            buffer.history_forward()
        except AttributeError:
            return False
        moved = getattr(buffer, "working_index", None) != before_index
        if moved:
            self._invalidate()
        return moved

    def _completion_previous_or_cursor_up(self, buffer: Any) -> None:
        """Navigate completion candidates, then history, then multiline cursor movement."""
        if self._completion_is_open(buffer):
            buffer.complete_previous()
            return
        if self._cursor_at_first_input_line(buffer) and self._move_history_backward(buffer):
            return
        buffer.cursor_up()

    def _completion_next_or_cursor_down(self, buffer: Any) -> None:
        """Navigate completion candidates, then history, then multiline cursor movement."""
        if self._completion_is_open(buffer):
            buffer.complete_next()
            return
        if self._cursor_at_last_input_line(buffer) and self._move_history_forward(buffer):
            return
        buffer.cursor_down()

    def _accept_or_start_completion(self, buffer: Any) -> None:
        """Accept the selected completion or open completion candidates."""
        complete_state = getattr(buffer, "complete_state", None)
        current_completion = getattr(complete_state, "current_completion", None)
        if complete_state is not None and current_completion is None:
            buffer.complete_next()
            complete_state = getattr(buffer, "complete_state", None)
            current_completion = getattr(complete_state, "current_completion", None)
        if current_completion is not None:
            buffer.apply_completion(current_completion)
            return
        if self._accept_auto_suggestion(buffer):
            return
        buffer.start_completion(select_first=True)

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
            if self._queue_paused:
                await asyncio.sleep(0.1)
                continue
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
            self._active_terminal_status = None
            self._active_terminal_reason = ""
            self._agent_running = True
            self._sync_task_counts()
            self._console.command_card(self._active_command)
            try:
                if self._on_input is not None:
                    result = self._on_input(command.text)
                    if asyncio.iscoroutine(result):
                        self._active_dispatch_task = asyncio.create_task(result)
                        await self._active_dispatch_task
                if self._active_command is not None and self._active_terminal_status is None:
                    finished = self._active_command.mark_done()
                    self._console.command_card(finished)
            except asyncio.CancelledError:
                if self._active_command is not None and self._active_terminal_status is not None:
                    terminal = self._terminal_command(
                        self._active_command,
                        self._active_terminal_status,
                        self._active_terminal_reason or self._active_terminal_status.value,
                    )
                    if terminal.status != self._active_command.status:
                        self._console.command_card(terminal)
                elif self._active_command is not None:
                    cancelled = self._active_command.mark_cancelled("cancelled by user")
                    self._console.command_card(cancelled)
            except Exception as exc:
                if self._active_command is not None:
                    failed = self._active_command.mark_failed(f"{type(exc).__name__}: {exc}")
                    self._console.command_card(failed)
                self._console.error(f"{exc}")
                self._spinner_text = ""
                self._tool_start_time = 0.0
            finally:
                self._active_command = None
                self._active_dispatch_task = None
                self._active_terminal_status = None
                self._active_terminal_reason = ""
                self._agent_running = False
                self._sync_task_counts()
                self._invalidate()

    # ── Layout construction ──────────────────────────────────────────

    def _build_input_area(
        self, commands: Sequence[tuple[str, str]]
    ) -> TextArea:
        completer = build_completer(commands)
        ref = self

        area = TextArea(
            height=Dimension(min=1, max=4, preferred=1),
            prompt="",
            style="class:input-area",
            multiline=True,
            wrap_lines=True,
            dont_extend_height=True,
            history=FileHistory(str(self._history_path)),
            completer=completer,
            complete_while_typing=True,
            auto_suggest=AutoSuggestFromHistory(),
            input_processors=[_DynamicPlaceholderProcessor(
                ref._placeholder_text,
                ref._prompt_fragments,
            )],
        )
        area.buffer.tempfile_suffix = ".md"
        self._install_paste_compactor(area.buffer)
        return area

    def _build_application(self) -> Application[Any]:
        no_approval = Condition(lambda: self._approval_modal is None)
        has_approval = Condition(lambda: self._approval_modal is not None)

        spinner = Window(
            content=FormattedTextControl(self._spinner_fragments),
            height=self._spinner_height,
            wrap_lines=True,
        )
        status_gap = Window(
            content=FormattedTextControl(lambda: []),
            height=1,
            style="class:status-gap",
        )
        status_bar = Window(
            content=FormattedTextControl(self._status),
            height=1,
            wrap_lines=False,
            style="class:status-bar",
        )
        approval_panel = ConditionalContainer(
            Window(
                content=FormattedTextControl(self._approval_fragments),
                height=self._approval_height,
                wrap_lines=False,
                dont_extend_height=True,
                style="class:approval.modal",
            ),
            filter=has_approval,
        )
        root = HSplit([
            ConditionalContainer(spinner, filter=no_approval),
            approval_panel,
            ConditionalContainer(status_gap, filter=no_approval),
            status_bar,
            ConditionalContainer(self._input_area, filter=no_approval),
        ])
        layout = Layout(
            FloatContainer(
                content=root,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(
                            max_height=12,
                            scroll_offset=1,
                            display_arrows=True,
                        ),
                    ),
                ],
            )
        )

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
        approval_filter = Condition(lambda: ref._approval_modal is not None)

        def choose_approval_text(text: str) -> None:
            modal = ref._approval_modal
            if modal is None:
                return
            if modal.choose_text(text):
                ref._invalidate()

        for key in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "y", "o", "s", "a", "n", "d", "v"):
            @kb.add(key, filter=approval_filter)
            def _(event, key=key):
                choose_approval_text(key)

        @kb.add(Keys.Enter)
        def _(event):
            if ref._approval_modal is not None:
                ref._approval_modal.choose_selected()
                ref._invalidate()
                return
            buffer = event.app.current_buffer
            complete_state = getattr(buffer, "complete_state", None)
            current_completion = getattr(complete_state, "current_completion", None)
            if complete_state is not None and current_completion is None:
                buffer.complete_next()
                complete_state = getattr(buffer, "complete_state", None)
                current_completion = getattr(complete_state, "current_completion", None)
            if current_completion is not None:
                before = buffer.text
                buffer.apply_completion(current_completion)
                if buffer.text != before:
                    return
            text = buffer.text.strip()
            if not text:
                return
            has_compacted_paste = ref._paste_store.has_blocks
            ref.submit_text(text)
            buffer.reset(append_to_history=not has_compacted_paste)
            ref._reset_paste_fragment_window()

        @kb.add(Keys.Escape)
        def _(event):
            if ref._approval_modal is not None:
                ref._approval_modal.deny()
                ref._invalidate()
                return
            if ref._close_completion(event.current_buffer):
                return
            event.current_buffer.reset()
            ref._clear_paste_state()

        @kb.add(Keys.Escape, Keys.Enter)
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.Tab)
        def _(event):
            ref._accept_or_start_completion(event.current_buffer)

        @kb.add(Keys.Up)
        def _(event):
            if ref._approval_modal is not None:
                ref._approval_modal.move(-1)
                ref._invalidate()
                return
            ref._completion_previous_or_cursor_up(event.current_buffer)

        @kb.add(Keys.Down)
        def _(event):
            if ref._approval_modal is not None:
                ref._approval_modal.move(1)
                ref._invalidate()
                return
            ref._completion_next_or_cursor_down(event.current_buffer)

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
            if ref._approval_modal is not None:
                ref._approval_modal.deny()
                ref._invalidate()
                return
            if event.current_buffer.text:
                event.current_buffer.reset()
                ref._clear_paste_state()
                return
            if ref._agent_running:
                now = time.monotonic()
                if now - ref._last_control_c_at <= 2.0 and ref._dispatch_control_text("/cancel"):
                    ref._last_control_c_at = 0.0
                    return
                ref._last_control_c_at = now
                ref._console.system(
                    "Task is running. Type /cancel to stop, /skip to continue the queue, "
                    "or press Ctrl+C again to cancel."
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
            "status-gap": "",
            "status-bar": f"bg:{t.toolbar_bg} {t.statusbar_fg}",
            "status-bar.strong": f"bg:{t.toolbar_bg} bold {t.statusbar_accent}",
            "status-bar.dim": f"bg:{t.toolbar_bg} {t.statusbar_dim}",
            "status-bar.good": f"bg:{t.toolbar_bg} {t.statusbar_good}",
            "status-bar.warn": f"bg:{t.toolbar_bg} {t.warning}",
            "status-bar.bad": f"bg:{t.toolbar_bg} {t.error}",
            "hint": t.text_dim,
            "auto-suggest": t.auto_suggest,
            "placeholder": f"{t.input_placeholder} nobold",
            "selection": f"bg:{t.input_selection_bg} {t.input_selection_fg}",
            "completion-menu": f"bg:{t.toolbar_bg} {t.input_text}",
            "completion-menu.completion": f"bg:{t.toolbar_bg} {t.input_text}",
            "completion-menu.completion.current": f"bg:{t.input_selection_bg} bold {t.input_selection_fg}",
            "completion-menu.meta.completion": f"bg:{t.toolbar_bg} {t.text_muted}",
            "completion-menu.meta.completion.current": f"bg:{t.input_selection_bg} {t.input_selection_fg}",
            "approval.modal": f"bg:{t.input_bg} {t.text}",
            "approval.border": t.warning,
            "approval.title": f"bold {t.warning}",
            "approval.summary": f"bold {t.text}",
            "approval.label": t.text_dim,
            "approval.detail": t.warning,
            "approval.dim": t.text_muted,
            "approval.option": t.text,
            "approval.selected": f"bg:{t.input_selection_bg} bold {t.input_selection_fg}",
        })

    # ── Fragment providers ───────────────────────────────────────────

    def _prompt_fragments(self) -> list[tuple[str, str]]:
        if self._queue_paused:
            return [
                ("class:prompt.paused", "⏸ "),
                ("class:prompt", "❯ "),
            ]
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

    def _placeholder_text(self) -> str:
        """Return contextual input guidance for an empty buffer."""
        if self._prompt_mode == "learning":
            return "/teach stop · /teach discard · /teach pause · /teach resume · /teach skip · /annotate <text>"
        if self._queue_paused:
            return "Queue paused · /resume continue · /drop <id> remove · /queue view"
        if self._active_command is not None:
            command = self._active_command
            elapsed = _format_inline_duration(command.elapsed_s)
            return (
                f"Running {command.label} {elapsed} · type to queue next · "
                "/cancel stop · /skip next · /queue view"
            )
        if self._pending_input.qsize() > 0:
            return f"{self._pending_input.qsize()} queued · /pause hold · /queue view · /drop <id> remove"
        return "Ask LeapFlow… /help commands · /status runtime · /queue tasks"

    def _approval_max_lines(self) -> int:
        """Terminal-aware max height for the inline approval panel."""
        terminal_lines = shutil.get_terminal_size((80, 24)).lines
        return max(8, terminal_lines - 2)

    def _approval_fragments(self) -> list[tuple[str, str]]:
        modal = self._approval_modal
        if modal is None:
            return []
        return modal.fragments(max_lines=self._approval_max_lines())

    def _approval_height(self) -> int:
        modal = self._approval_modal
        if modal is None:
            return 0
        return modal.line_count(max_lines=self._approval_max_lines())

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
        text = f"  {self._spinner_text}{elapsed}"
        if self._active_command is not None and self._active_command.elapsed_s >= 30:
            text += " · /cancel stop · /skip next · /queue view"
        return [("class:hint", text)]

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

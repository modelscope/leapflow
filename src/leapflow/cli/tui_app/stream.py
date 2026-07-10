"""Streaming LLM output renderer — Application-compatible.

Accumulates streaming token deltas, tracks tool call state, and
renders the final response via Rich Console.  No ``rich.Live`` —
all output flows through ``patch_stdout`` to appear above the
Application layout.

During streaming, the caller updates ``LeapApp.spinner_text`` for
visual feedback (e.g. tool name + elapsed timer).  On finish, the
accumulated response is rendered as Rich Markdown.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.text import Text

_FINAL_RESPONSE_INDENT_SPACES = 4

if TYPE_CHECKING:
    from leapflow.cli.tui_app.console import LeapConsole


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as a compact human-readable string."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m{secs:.0f}s"


class StreamRenderer:
    """Accumulates streaming output and renders on finish.

    Usage::

        renderer = StreamRenderer(console)
        renderer.start()
        renderer.feed("Hello ")
        renderer.feed("**world**!")
        spinner = renderer.tool_started("shell")   # returns "⚡ shell"
        renderer.tool_finished("shell")             # prints ✓ line
        renderer.finish()                           # prints markdown + label
    """

    def __init__(self, console: "LeapConsole") -> None:
        self._console = console
        self._buffer: str = ""
        self._thinking_buffer: str = ""
        self._start_time: float = 0.0
        self._tool_start_time: float = 0.0
        self._active_tool: str = ""
        self._tool_history: list[tuple[str, float]] = []

    @property
    def text(self) -> str:
        """Full accumulated response text."""
        return self._buffer

    @property
    def has_output(self) -> bool:
        return bool(self._buffer.strip() or self._thinking_buffer.strip())

    @property
    def elapsed(self) -> float:
        """Seconds since streaming started."""
        return time.monotonic() - self._start_time if self._start_time else 0.0

    @property
    def tool_count(self) -> int:
        return len(self._tool_history)

    def start(self) -> None:
        """Begin a new streaming session."""
        self._buffer = ""
        self._thinking_buffer = ""
        self._active_tool = ""
        self._tool_history = []
        self._start_time = time.monotonic()
        self._tool_start_time = 0.0

    def feed(self, chunk: str) -> None:
        """Append a text chunk to the response buffer."""
        self._buffer += chunk

    def feed_thinking(self, chunk: str) -> None:
        """Append a thinking/reasoning chunk."""
        self._thinking_buffer += chunk

    def tool_started(self, name: str) -> str:
        """Mark a tool call as started. Returns spinner text for LeapApp."""
        self._active_tool = name
        self._tool_start_time = time.monotonic()
        return f"⚡ {name}"

    def tool_finished(self, name: str = "", output: str = "") -> None:
        """Mark a tool call as finished; print completion line immediately."""
        tool_name = name or self._active_tool
        if tool_name and self._tool_start_time > 0:
            duration = time.monotonic() - self._tool_start_time
            self._tool_history.append((tool_name, duration))
            line = Text()
            line.append("  ✓ ", style="leap.success")
            line.append(tool_name, style="bold")
            line.append(f"  {_format_elapsed(duration)}", style="dim")
            self._console.print(line)
        self._active_tool = ""
        self._tool_start_time = 0.0

    def finish(self) -> None:
        """Render all accumulated content to the console."""
        if self._thinking_buffer.strip():
            self._console.thinking(self._thinking_buffer)

        if self._buffer.strip():
            self._console.markdown(self._buffer, indent=_FINAL_RESPONSE_INDENT_SPACES)

        self._console.response_label(self.elapsed, tool_count=self.tool_count)
        self._console.newline()

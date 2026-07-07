"""Live streaming renderer for LLM output.

Displays streaming token deltas with real-time markdown rendering,
tool call annotations with elapsed timers, and thinking indicators.
Uses ``rich.Live`` for flicker-free incremental updates.

On finish, the transient Live display is replaced with a permanent
Rich Markdown render, so the response persists in scrollback.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from leapflow.cli.tui_app.theme import Theme


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
    """Incrementally renders streaming LLM output with rich formatting.

    Features:
    - Live markdown rendering during streaming
    - Thinking/reasoning display in a dimmed panel
    - Tool call indicators with elapsed timers
    - Persistent tool completion lines in scrollback
    - Response label with total elapsed time on finish

    Usage::

        renderer = StreamRenderer(console, theme)
        renderer.start()
        renderer.feed("Hello ")
        renderer.feed("**world**!")
        renderer.tool_started("shell", "git status")
        renderer.tool_finished("shell", "on branch main\\n...")
        renderer.finish()
    """

    def __init__(self, console: Console, theme: Theme) -> None:
        self._console = console
        self._theme = theme
        self._buffer: str = ""
        self._thinking_buffer: str = ""
        self._live: Optional[Live] = None
        self._start_time: float = 0.0
        self._tool_start_time: float = 0.0
        self._active_tool: str = ""
        self._finished = False
        self._tool_history: List[Tuple[str, float]] = []

    @property
    def text(self) -> str:
        """The full accumulated text so far."""
        return self._buffer

    @property
    def elapsed(self) -> float:
        """Seconds since streaming started."""
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def start(self) -> None:
        """Begin a streaming session with a live-updating display."""
        self._buffer = ""
        self._thinking_buffer = ""
        self._active_tool = ""
        self._finished = False
        self._tool_history = []
        self._start_time = time.monotonic()
        self._tool_start_time = 0.0
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def feed(self, chunk: str) -> None:
        """Append a text chunk to the streaming buffer."""
        if self._finished:
            return
        self._buffer += chunk
        self._refresh()

    def feed_thinking(self, chunk: str) -> None:
        """Append a thinking/reasoning chunk."""
        if self._finished:
            return
        self._thinking_buffer += chunk
        self._refresh()

    def tool_started(self, name: str, args_summary: str = "") -> None:
        """Mark a tool call as starting."""
        self._active_tool = name
        self._tool_start_time = time.monotonic()
        self._refresh()

    def tool_finished(self, name: str = "", output: str = "") -> None:
        """Mark a tool call as finished and record in history."""
        tool_name = name or self._active_tool
        if tool_name and self._tool_start_time > 0:
            duration = time.monotonic() - self._tool_start_time
            self._tool_history.append((tool_name, duration))
        self._active_tool = ""
        self._tool_start_time = 0.0
        self._refresh()

    def finish(self) -> None:
        """End the streaming session and render final output."""
        self._finished = True
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        """Update the live display."""
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        """Build the Rich renderable for the current state."""
        parts = []

        if self._thinking_buffer:
            thinking = Text(self._thinking_buffer.strip(), style="dim")
            parts.append(Panel(
                thinking,
                title="💭 thinking",
                title_align="left",
                border_style=self._theme.border,
                padding=(0, 1),
            ))

        for tool_name, duration in self._tool_history:
            line = Text()
            line.append("  ✓ ", style=self._theme.success)
            line.append(tool_name, style="bold")
            line.append(f"  {_format_elapsed(duration)}", style="dim")
            parts.append(line)

        if self._buffer:
            code_theme = "monokai" if self._theme.name == "dark" else "default"
            md = Markdown(self._buffer, code_theme=code_theme)
            parts.append(md)
        elif not self._thinking_buffer and not self._active_tool:
            parts.append(Spinner("dots", text="  Thinking…", style=self._theme.accent_dim))

        if self._active_tool:
            tool_text = Text()
            tool_text.append("  ⚡ ", style=self._theme.accent)
            tool_text.append(self._active_tool, style="bold")
            if self._tool_start_time > 0:
                elapsed = time.monotonic() - self._tool_start_time
                tool_text.append(f"  ({_format_elapsed(elapsed)})", style="dim")
            else:
                tool_text.append(" …", style="dim")
            parts.append(tool_text)

        if len(parts) == 1:
            return parts[0]
        return Group(*parts)

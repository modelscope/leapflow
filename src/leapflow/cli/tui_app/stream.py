"""Live streaming renderer for LLM output.

Displays streaming token deltas with real-time markdown rendering,
tool call annotations, and thinking indicators.  Uses ``rich.Live``
for flicker-free incremental updates.
"""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from leapflow.cli.tui_app.theme import Theme


class StreamRenderer:
    """Incrementally renders streaming LLM output with rich formatting.

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
        self._tool_label: str = ""
        self._start_time: float = 0.0
        self._finished = False

    @property
    def text(self) -> str:
        """The full accumulated text so far."""
        return self._buffer

    def start(self) -> None:
        """Begin a streaming session with a live-updating display."""
        self._buffer = ""
        self._thinking_buffer = ""
        self._tool_label = ""
        self._finished = False
        self._start_time = time.monotonic()
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
        self._tool_label = f"⚡ {name}" + (f" {args_summary}" if args_summary else "")
        self._refresh()

    def tool_finished(self, name: str = "", output: str = "") -> None:
        """Mark a tool call as finished."""
        self._tool_label = ""
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

        if self._buffer:
            code_theme = "monokai" if self._theme.name == "dark" else "default"
            md = Markdown(self._buffer, code_theme=code_theme)
            parts.append(md)
        elif not self._thinking_buffer:
            parts.append(Spinner("dots", text="  Thinking...", style=self._theme.accent_dim))

        if self._tool_label:
            tool_text = Text()
            tool_text.append("  ")
            tool_text.append(self._tool_label, style=self._theme.accent)
            tool_text.append(" ")
            tool_text.append("…", style="dim")
            parts.append(tool_text)

        if len(parts) == 1:
            return parts[0]
        return Group(*parts)

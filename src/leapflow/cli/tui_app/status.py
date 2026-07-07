"""Status bar for the prompt_toolkit bottom toolbar.

Hermes-style layout:
``⚡ model │ 38.3K/1M │ [░░░░░░░░░░] 4% │ 2m │ ⏲ 6s │ ✓ 2m``

Segments: model, context tokens, progress bar, session duration,
last-turn elapsed, session up-time checkmark.
"""

from __future__ import annotations

import time
from typing import Optional

from prompt_toolkit.formatted_text import FormattedText


def _compact_tokens(n: int) -> str:
    """Format token count as compact string: 12.4K, 1.2M, etc."""
    if n < 0:
        return "?"
    if n < 1_000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000:
        return f"{n // 1000}K"
    return f"{n / 1_000_000:.1f}M"


def _format_duration(seconds: float) -> str:
    """Format seconds as compact duration: 42s, 2m, 1h23m."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins}m" if mins else f"{hours}h"


def _progress_bar(pct: int, width: int = 10) -> str:
    """Build a Unicode progress bar: [████░░░░░░]."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


class StatusBar:
    """Builds bottom-toolbar content for the prompt.

    Updated by the interactive loop before each prompt cycle.
    """

    def __init__(self) -> None:
        self.mode: str = "idle"
        self.skill_count: int = 0
        self.platform_online: bool = False
        self.model_name: str = ""
        self.session_turns: int = 0
        self.context_used: int = 0
        self.context_max: int = 0
        self.last_turn_elapsed: float = 0.0
        self._turn_start: float = 0.0
        self._session_start: float = time.monotonic()

    def mark_turn_start(self) -> None:
        """Record the start of a new turn for elapsed tracking."""
        self._turn_start = time.monotonic()

    def mark_turn_end(self) -> None:
        """Record the end of a turn."""
        if self._turn_start > 0:
            self.last_turn_elapsed = time.monotonic() - self._turn_start
            self._turn_start = 0.0

    def __call__(self) -> FormattedText:
        """Called by prompt_toolkit to render the bottom toolbar."""
        parts: list[tuple[str, str]] = []

        # Mode indicator (compact)
        mode_map = {
            "idle": ("class:bottom-toolbar.text", " ⚡ "),
            "learning": ("#FF6B6B bold", " ● "),
            "paused": ("#FFD700 bold", " ⏸ "),
            "executing": ("#87D687 bold", " ▶ "),
        }
        style, icon = mode_map.get(self.mode, mode_map["idle"])
        parts.append((style, icon))

        # Model name
        if self.model_name:
            display_model = self.model_name
            if len(display_model) > 20:
                display_model = display_model[:18] + "…"
            parts.append(("class:bottom-toolbar.text", f"{display_model} "))
            parts.append(("class:bottom-toolbar.text", "│ "))

        # Context tokens + progress bar
        if self.context_max > 0:
            used = _compact_tokens(self.context_used)
            total = _compact_tokens(self.context_max)
            pct = int(self.context_used * 100 / self.context_max) if self.context_max else 0

            parts.append(("class:bottom-toolbar.text", f"{used}/{total} "))
            parts.append(("class:bottom-toolbar.text", "│ "))

            # Color-coded progress bar
            if pct >= 90:
                bar_style = "#FF6B6B"
            elif pct >= 75:
                bar_style = "#FFD700"
            else:
                bar_style = "class:bottom-toolbar.text"
            bar = _progress_bar(pct)
            parts.append((bar_style, f"[{bar}] "))
            parts.append(("class:bottom-toolbar.text", f"{pct}% "))
            parts.append(("class:bottom-toolbar.text", "│ "))

        # Session duration
        session_elapsed = time.monotonic() - self._session_start
        parts.append(("class:bottom-toolbar.text", f"{_format_duration(session_elapsed)} "))

        # Last turn elapsed
        if self.last_turn_elapsed > 0:
            parts.append(("class:bottom-toolbar.text", "│ "))
            if self.last_turn_elapsed < 1.0:
                elapsed_str = f"{self.last_turn_elapsed * 1000:.0f}ms"
            else:
                elapsed_str = f"{self.last_turn_elapsed:.1f}s"
            parts.append(("class:bottom-toolbar.text", f"⏲ {elapsed_str} "))

        # Session uptime checkmark
        parts.append(("class:bottom-toolbar.text", "│ "))
        parts.append(("#87D687", f"✓ {_format_duration(session_elapsed)} "))

        return FormattedText(parts)

    def update(
        self,
        *,
        mode: Optional[str] = None,
        skill_count: Optional[int] = None,
        platform_online: Optional[bool] = None,
        model_name: Optional[str] = None,
        session_turns: Optional[int] = None,
        context_used: Optional[int] = None,
        context_max: Optional[int] = None,
    ) -> None:
        """Selectively update status fields."""
        if mode is not None:
            self.mode = mode
        if skill_count is not None:
            self.skill_count = skill_count
        if platform_online is not None:
            self.platform_online = platform_online
        if model_name is not None:
            self.model_name = model_name
        if session_turns is not None:
            self.session_turns = session_turns
        if context_used is not None:
            self.context_used = context_used
        if context_max is not None:
            self.context_max = context_max

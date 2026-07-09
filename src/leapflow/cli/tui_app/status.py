"""Status bar for the Application layout.

Hermes-style single-line status rendered via ``FormattedTextControl``::

    ⚡ qwen3-plus │ 38.3K/1M │ [████░░░░░░] 38% │ 12m │ ⏲ 2.1s │ ✓ 12m

All style strings use ``class:status-bar*`` Application style classes
defined in ``app.py``.  The ``StatusBar`` is callable — pass it directly
as the ``text`` argument to ``FormattedTextControl``.
"""

from __future__ import annotations

import shutil
import time
from typing import Optional

from leapflow.cli.tui_app.theme import ResolvedTheme, Theme, detect_theme


def _compact_tokens(n: int) -> str:
    """Format token count with adaptive precision.

    < 1K   → ``0.1K``, ``0.9K``  (1 decimal)
    1–10K  → ``1.2K``, ``9.8K``  (1 decimal)
    10–999K → ``12K``, ``256K``   (integer K)
    ≥ 1M   → ``1.2M``            (1 decimal)
    """
    if n < 0:
        return "?"
    if n == 0:
        return "0"
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000:
        return f"{n // 1000}K"
    return f"{n / 1_000_000:.1f}M"


def _format_duration(seconds: float) -> str:
    """Format seconds as compact duration: ``42s``, ``2m``, ``1h23m``."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins}m" if mins else f"{hours}h"


def _progress_bar(pct: float, width: int = 10) -> str:
    """Unicode progress bar: ``████░░░░░░``."""
    filled = int(pct / 100 * width)
    if pct > 0 and filled == 0:
        filled = 1
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def _format_percent(used: int, total: int) -> str:
    """Format utilization percentage with useful precision at low values."""
    if total <= 0:
        return "0%"
    pct = max(0.0, min(used * 100 / total, 100.0))
    if 0 < pct < 10:
        return f"{pct:.1f}%"
    return f"{int(pct)}%"


def _truncate(text: str, limit: int) -> str:
    """Return text truncated to a single-cell-safe display budget."""
    if limit <= 1:
        return "…"
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StatusBar:
    """Produces status-bar fragments for ``FormattedTextControl``.

    Updated by the interactive loop; called by the Application on
    every redraw cycle to get fresh fragments.
    """

    def __init__(self, theme: Optional[Theme | ResolvedTheme] = None) -> None:
        self._theme = theme or detect_theme()
        self.mode: str = "idle"
        self.skill_count: int = 0
        self.platform_online: bool = False
        self.model_name: str = ""
        self.session_turns: int = 0
        self.context_used: int = 0
        self.context_max: int = 0
        self.running_tasks: int = 0
        self.queued_tasks: int = 0
        self.last_turn_elapsed: float = 0.0
        self._turn_start: float = 0.0
        self._session_start: float = time.monotonic()

    def mark_turn_start(self) -> None:
        """Record start of a new agent turn."""
        self._turn_start = time.monotonic()

    def mark_turn_end(self) -> None:
        """Record end of the agent turn."""
        if self._turn_start > 0:
            self.last_turn_elapsed = time.monotonic() - self._turn_start
            self._turn_start = 0.0

    def __call__(self) -> list[tuple[str, str]]:
        """Called by FormattedTextControl on every Application redraw."""
        parts: list[tuple[str, str]] = []
        width = shutil.get_terminal_size().columns
        compact = width < 72
        narrow = width < 52

        _modes = {
            "idle":      ("class:status-bar",     " ⚡ "),
            "learning":  ("class:status-bar.bad",  " ● "),
            "paused":    ("class:status-bar.warn", " ⏸ "),
            "executing": ("class:status-bar.good", " ▶ "),
            "daemon":    ("class:status-bar.good", " ◆ "),
        }
        style, icon = _modes.get(self.mode, _modes["idle"])
        parts.append((style, icon))

        if self.running_tasks or self.queued_tasks:
            if narrow:
                task_text = f"r{self.running_tasks} q{self.queued_tasks} "
            else:
                task_text = f"running:{self.running_tasks} queued:{self.queued_tasks} "
            parts.append(("class:status-bar.strong", task_text))
            parts.append(("class:status-bar.dim", "│ "))

        if self.model_name and not narrow:
            model_limit = 12 if compact else 20
            display = _truncate(self.model_name, model_limit)
            parts.append(("class:status-bar.strong", f"{display} "))
            parts.append(("class:status-bar.dim", "│ "))

        if self.context_max > 0:
            used = _compact_tokens(self.context_used)
            total = _compact_tokens(self.context_max)
            pct = min(self.context_used * 100 / self.context_max, 100.0)
            pct_text = _format_percent(self.context_used, self.context_max)
            if narrow:
                parts.append(("class:status-bar", f"{pct_text} "))
            else:
                parts.append(("class:status-bar", f"{used}/{total} "))
                parts.append(("class:status-bar.dim", "│ "))

                bar_cls = "class:status-bar"
                if pct >= 90:
                    bar_cls = "class:status-bar.bad"
                elif pct >= 75:
                    bar_cls = "class:status-bar.warn"
                if compact:
                    parts.append(("class:status-bar", f"{pct_text} "))
                else:
                    bar = _progress_bar(pct)
                    parts.append((bar_cls, f"[{bar}] "))
                    parts.append(("class:status-bar", f"{pct_text} "))
            parts.append(("class:status-bar.dim", "│ "))

        elapsed = time.monotonic() - self._session_start
        parts.append(("class:status-bar", f"{_format_duration(elapsed)} "))

        if self.last_turn_elapsed > 0 and not narrow:
            parts.append(("class:status-bar.dim", "│ "))
            if self.last_turn_elapsed < 1.0:
                e_str = f"{self.last_turn_elapsed * 1000:.0f}ms"
            else:
                e_str = f"{self.last_turn_elapsed:.1f}s"
            parts.append(("class:status-bar", f"⏲ {e_str} "))

        if not compact:
            parts.append(("class:status-bar.dim", "│ "))
            parts.append(("class:status-bar.good", f"✓ {_format_duration(elapsed)} "))

        return parts

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
        running_tasks: Optional[int] = None,
        queued_tasks: Optional[int] = None,
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
        if running_tasks is not None:
            self.running_tasks = running_tasks
        if queued_tasks is not None:
            self.queued_tasks = queued_tasks

    def update_task_counts(self, *, running: int, queued: int) -> None:
        """Update task counters shown in the status bar."""
        self.running_tasks = max(0, running)
        self.queued_tasks = max(0, queued)
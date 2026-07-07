"""Status bar for the prompt_toolkit bottom toolbar.

Provides a context-aware status line showing mode, skill count,
platform connection, model info, context pressure, and turn timing.
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


class StatusBar:
    """Builds bottom-toolbar content for the prompt.

    Updated by the interactive loop before each prompt cycle.
    Displays: mode | skills | platform | model | context usage | turn elapsed
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

        mode_styles = {
            "idle": ("class:bottom-toolbar.text", " ⏵ idle "),
            "learning": ("#FF6B6B bold", " ● recording "),
            "paused": ("#FFD700 bold", " ⏸ paused "),
            "executing": ("#87D687 bold", " ▶ running "),
        }
        style, label = mode_styles.get(self.mode, mode_styles["idle"])
        parts.append((style, label))
        parts.append(("class:bottom-toolbar.text", "│"))

        parts.append(("class:bottom-toolbar.text", f" skills:{self.skill_count} "))
        parts.append(("class:bottom-toolbar.text", "│"))

        conn_style = "#87D687" if self.platform_online else "class:bottom-toolbar.text"
        conn_label = " ● " if self.platform_online else " ○ "
        parts.append((conn_style, conn_label))
        parts.append(("class:bottom-toolbar.text", "│"))

        if self.model_name:
            display_model = self.model_name
            if len(display_model) > 20:
                display_model = display_model[:18] + "…"
            parts.append(("class:bottom-toolbar.text", f" {display_model} "))
            parts.append(("class:bottom-toolbar.text", "│"))

        if self.context_max > 0:
            used = _compact_tokens(self.context_used)
            total = _compact_tokens(self.context_max)
            pct = int(self.context_used * 100 / self.context_max) if self.context_max else 0

            if pct >= 90:
                ctx_style = "#FF6B6B bold"
            elif pct >= 75:
                ctx_style = "#FFD700"
            else:
                ctx_style = "class:bottom-toolbar.text"

            parts.append((ctx_style, f" ctx:{used}/{total} ({pct}%) "))
            parts.append(("class:bottom-toolbar.text", "│"))

        if self.session_turns > 0:
            parts.append(("class:bottom-toolbar.text", f" t:{self.session_turns}"))
            if self.last_turn_elapsed > 0:
                if self.last_turn_elapsed < 1.0:
                    elapsed_str = f"{self.last_turn_elapsed * 1000:.0f}ms"
                else:
                    elapsed_str = f"{self.last_turn_elapsed:.1f}s"
                parts.append(("class:bottom-toolbar.text", f" ⏱{elapsed_str}"))
            parts.append(("class:bottom-toolbar.text", " "))

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

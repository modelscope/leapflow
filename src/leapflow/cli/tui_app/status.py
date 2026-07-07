"""Status bar for the prompt_toolkit bottom toolbar.

Provides a context-aware status line showing mode, skill count,
platform connection, model info, and elapsed time.
"""

from __future__ import annotations

from typing import Optional

from prompt_toolkit.formatted_text import FormattedText


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

    def __call__(self) -> FormattedText:
        """Called by prompt_toolkit to render the bottom toolbar."""
        parts: list[tuple[str, str]] = []

        mode_styles = {
            "idle": ("class:bottom-toolbar.text", " ⏵ idle "),
            "learning": ("ansired bold", " ● recording "),
            "paused": ("ansiyellow bold", " ⏸ paused "),
            "executing": ("ansigreen bold", " ▶ running "),
        }
        style, label = mode_styles.get(self.mode, mode_styles["idle"])
        parts.append((style, label))

        parts.append(("class:bottom-toolbar.text", "│"))

        parts.append(("class:bottom-toolbar.text", f" skills: {self.skill_count} "))
        parts.append(("class:bottom-toolbar.text", "│"))

        conn_style = "ansigreen" if self.platform_online else "class:bottom-toolbar.text"
        conn_label = " ● " if self.platform_online else " ○ "
        parts.append((conn_style, conn_label))
        parts.append(("class:bottom-toolbar.text", "│"))

        if self.model_name:
            display_model = self.model_name
            if len(display_model) > 20:
                display_model = display_model[:18] + "…"
            parts.append(("class:bottom-toolbar.text", f" {display_model} "))

        if self.session_turns > 0:
            parts.append(("class:bottom-toolbar.text", f"│ turns: {self.session_turns} "))

        return FormattedText(parts)

    def update(
        self,
        *,
        mode: Optional[str] = None,
        skill_count: Optional[int] = None,
        platform_online: Optional[bool] = None,
        model_name: Optional[str] = None,
        session_turns: Optional[int] = None,
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

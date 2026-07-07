"""Adaptive theming with automatic light/dark detection.

Detects terminal background color via environment heuristics and provides
a coherent color palette for the entire TUI.  All colors are centralized
here — components never hardcode ANSI sequences.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Immutable color palette consumed by all TUI components."""

    # Identity
    name: str

    # Primary accent (brand color)
    accent: str
    accent_dim: str

    # Semantic colors
    success: str
    warning: str
    error: str
    info: str

    # Text hierarchy
    text: str
    text_dim: str
    text_muted: str

    # Structural
    border: str
    border_dim: str
    panel_title: str

    # Status indicators
    recording: str
    executing: str

    # Code
    code_bg: str

    # Prompt
    prompt_char: str


_DARK = Theme(
    name="dark",
    accent="cyan",
    accent_dim="dim cyan",
    success="green",
    warning="yellow",
    error="red",
    info="blue",
    text="white",
    text_dim="dim white",
    text_muted="bright_black",
    border="bright_black",
    border_dim="dim",
    panel_title="bold cyan",
    recording="bold red",
    executing="bold green",
    code_bg="grey11",
    prompt_char="bold cyan",
)

_LIGHT = Theme(
    name="light",
    accent="dark_cyan",
    accent_dim="dim dark_cyan",
    success="dark_green",
    warning="dark_orange3",
    error="red3",
    info="blue3",
    text="black",
    text_dim="dim",
    text_muted="grey50",
    border="grey70",
    border_dim="dim grey50",
    panel_title="bold dark_cyan",
    recording="bold red3",
    executing="bold dark_green",
    code_bg="grey93",
    prompt_char="bold dark_cyan",
)


def detect_light_mode() -> bool:
    """Heuristic detection of light terminal background.

    Checks (in order):
    1. ``LEAPFLOW_TUI_THEME`` env var (``light`` or ``dark``)
    2. ``COLORFGBG`` (set by many terminals: ``fg;bg`` where bg>=8 is dark)
    3. macOS Terminal.app defaults to light
    4. Default: dark
    """
    explicit = os.environ.get("LEAPFLOW_TUI_THEME", "").lower()
    if explicit == "light":
        return True
    if explicit == "dark":
        return False

    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            bg = int(colorfgbg.rsplit(";", 1)[1])
            return bg >= 8 or bg == 7
        except (ValueError, IndexError):
            pass

    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program == "Apple_Terminal":
        return True

    return False


def detect_theme() -> Theme:
    """Return the appropriate theme based on terminal environment."""
    if not sys.stdout.isatty():
        return _DARK
    return _LIGHT if detect_light_mode() else _DARK

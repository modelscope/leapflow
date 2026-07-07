"""Adaptive theming with automatic light/dark detection.

Detects terminal background color via environment heuristics and provides
a coherent color palette for the entire TUI.  All colors are centralized
here — components never hardcode ANSI sequences.

The dark theme uses a warm gold/amber palette inspired by premium
terminal tools.  The light theme uses dark counterparts for readability.
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
    accent="#FFBF00",
    accent_dim="#B8860B",
    success="#87D687",
    warning="#FFD700",
    error="#FF6B6B",
    info="#87CEEB",
    text="#FFF8DC",
    text_dim="#B8860B",
    text_muted="#8B8682",
    border="#CD7F32",
    border_dim="#8B6914",
    panel_title="bold #FFD700",
    recording="bold #FF6B6B",
    executing="bold #87D687",
    code_bg="grey11",
    prompt_char="bold #FFD700",
)

_LIGHT = Theme(
    name="light",
    accent="#996600",
    accent_dim="#B8860B",
    success="dark_green",
    warning="#996600",
    error="red3",
    info="blue3",
    text="black",
    text_dim="#8B6914",
    text_muted="grey50",
    border="#CD7F32",
    border_dim="grey50",
    panel_title="bold #996600",
    recording="bold red3",
    executing="bold dark_green",
    code_bg="grey93",
    prompt_char="bold #996600",
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

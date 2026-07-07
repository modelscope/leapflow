"""TUI rendering primitives for the interactive REPL.

Pre-renders an input frame (input line, status bar) then positions the cursor
on the input line so the user sees the status bar while typing.

Pure ANSI escapes — no curses or third-party dependencies.
"""

from __future__ import annotations

import sys

from leapflow.cli.banner import BRIGHT_CYAN, DIM, RESET

# ANSI cursor movement
_CUR_UP = "\033[A"
_CUR_DOWN = "\033[B"
_ERASE_LINE = "\033[2K"
_COL0 = "\r"

# Readline invisible-char markers (so readline calculates prompt width correctly)
_RL_START = "\001"
_RL_END = "\002"


def _is_tty() -> bool:
    return sys.stdout.isatty()


def prompt_char(mode: str) -> str:
    if mode == "learning":
        return "\033[1;31m● rec\033[0m"
    elif mode == "paused":
        return "\033[1;33m⏸ paused\033[0m"
    elif mode == "executing":
        return "\033[1;32m▶\033[0m"
    return f"{BRIGHT_CYAN}❯{RESET}"


def _rl_prompt(mode: str) -> str:
    """Build a readline-safe prompt string with invisible ANSI markers."""
    if mode == "learning":
        return f" {_RL_START}\033[1;31m{_RL_END}● rec{_RL_START}\033[0m{_RL_END} "
    elif mode == "paused":
        return f" {_RL_START}\033[1;33m{_RL_END}⏸ paused{_RL_START}\033[0m{_RL_END} "
    elif mode == "executing":
        return f" {_RL_START}\033[1;32m{_RL_END}▶{_RL_START}\033[0m{_RL_END} "
    return f" {_RL_START}{BRIGHT_CYAN}{_RL_END}❯{_RL_START}{RESET}{_RL_END} "


def status_bar(mode: str, skill_count: int, bridge_online: bool) -> str:
    mode_indicators = {
        "idle": "⏵ idle",
        "learning": "● recording",
        "paused": "⏸ paused",
        "executing": "▶ running",
    }
    mode_str = mode_indicators.get(mode, "⏵ idle")
    conn_str = "online" if bridge_online else "offline"
    return f"{DIM}  {mode_str} │ skills: {skill_count} │ platform: {conn_str}{RESET}"


def render_input_frame(mode: str, skill_count: int, bridge_online: bool) -> str:
    """Pre-render the input frame and return the readline prompt string.

    Renders 2 lines:
                                  (input line — cursor ends up here)
        ⏵ idle │ skills: N │ ... (status bar)

    Then moves cursor back up to the input line so the user types there,
    seeing the status bar below their cursor.

    Returns a readline-safe prompt string for use with input().
    """
    if not _is_tty():
        prefixes = {"learning": "learn> ", "executing": "exec> "}
        return prefixes.get(mode, "leap> ")

    status = status_bar(mode, skill_count, bridge_online)

    # Print: blank input line, status bar
    sys.stdout.write("\n")
    sys.stdout.write(f"{status}")
    # Move cursor up 1 line (from status → onto input line)
    sys.stdout.write(f"\033[1A{_COL0}")
    sys.stdout.flush()

    return _rl_prompt(mode)


def finish_input_frame() -> None:
    """After input() returns, erase the pre-rendered status bar and continue.

    When the user presses Enter, the cursor advances to the line containing
    the status bar. We erase it so output starts directly from this line
    with no extra blank lines.
    """
    if not _is_tty():
        return
    sys.stdout.write(f"{_ERASE_LINE}{_COL0}")
    sys.stdout.flush()

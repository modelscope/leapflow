"""LeapFlow welcome banner.

Animated TUI splash shown by `leap` when invoked without a subcommand.
Pure ANSI escape codes — no third-party dependencies.

Animation budget: ~1.5s on a TTY; static fallback when stdout is not a TTY.
"""

from __future__ import annotations

import sys
import time

from leapflow.version import __version__

# ── ANSI escape codes ─────────────────────────────────────────────────────
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[1;36m"      # bold cyan — LEAP accent color
BRIGHT_ORANGE = "\033[1;38;5;208m"  # bold orange — tagline initials
DIM_WHITE = "\033[2;37m"        # dim white — secondary text
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"
# \r returns to col 0; we use it to overwrite a line for "fade-in" effect.

VERSION = __version__
INDENT = "    "
W = 50  # inner width of the framed boxes


# ── Low-level helpers ─────────────────────────────────────────────────────

def _tty() -> bool:
    return sys.stdout.isatty()


def _w(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _pad(visible_len: int) -> tuple[int, int]:
    """Return (left, right) spaces to center content of given visible width."""
    left = (W - visible_len) // 2
    return left, W - visible_len - left


def _box_line(content: str = "", visible_len: int = 0) -> str:
    """Build a centered box row: '│ ...content... │'."""
    left, right = _pad(visible_len)
    return (
        f"{INDENT}{DIM}│{RESET}"
        f"{' ' * left}{content}{' ' * right}"
        f"{DIM}│{RESET}"
    )


def _empty_box_line() -> str:
    return f"{INDENT}{DIM}│{RESET}{' ' * W}{DIM}│{RESET}"


def _border(left: str, right: str) -> str:
    return f"{INDENT}{DIM}{left}{'─' * W}{right}{RESET}"


# ── Content builders ──────────────────────────────────────────────────────

def _leap_logo_line(progress: int) -> str:
    """Render 'L . E . A . P', with the first `progress` letters lit bright cyan."""
    letters = ("L", "E", "A", "P")
    parts: list[str] = []
    for i, ch in enumerate(letters):
        color = BRIGHT_CYAN if i < progress else DIM
        parts.append(f"{color}{ch}{RESET}")
        if i < 3:
            parts.append(f"{DIM} . {RESET}")
    return _box_line("".join(parts), visible_len=13)


def _tagline_line(lit: bool) -> str:
    """'Learning and Evolving from Actual Practice' with L/E/A/P in orange when lit."""
    if not lit:
        return _box_line(
            f"{DIM}Learning and Evolving from Actual Practice{RESET}",
            visible_len=42,
        )
    content = (
        f"{BRIGHT_ORANGE}L{RESET}{DIM_WHITE}earning and{RESET} "
        f"{BRIGHT_ORANGE}E{RESET}{DIM_WHITE}volving from{RESET} "
        f"{BRIGHT_ORANGE}A{RESET}{DIM_WHITE}ctual{RESET} "
        f"{BRIGHT_ORANGE}P{RESET}{DIM_WHITE}ractice{RESET}"
    )
    return _box_line(content, visible_len=42)


def _version_box_line() -> str:
    text = f"Agent v{VERSION}"
    return _box_line(f"{DIM}{text}{RESET}", visible_len=len(text))


# ── Animation phases ──────────────────────────────────────────────────────

def _typewriter(text: str, delay: float = 0.012) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)


def _animate_logo_box() -> None:
    """Phase 1: progressive fade-in of the LEAP logo box."""
    _w(_border("╭", "╮") + "\n")
    time.sleep(0.06)
    _w(_empty_box_line() + "\n")
    time.sleep(0.06)

    # Light up L · E · A · P one letter at a time, overwriting the same line.
    for progress in range(5):  # 0 (all dim) → 4 (all lit)
        _w("\r" + _leap_logo_line(progress))
        time.sleep(0.07)
    _w("\n")

    _w(_empty_box_line() + "\n")
    time.sleep(0.05)

    # Tagline: dim flash, then re-paint with LEAP letters highlighted.
    _w(_tagline_line(lit=False))
    time.sleep(0.12)
    _w("\r" + _tagline_line(lit=True) + "\n")

    _w(_empty_box_line() + "\n")
    time.sleep(0.04)
    _w(_version_box_line() + "\n")
    time.sleep(0.04)
    _w(_empty_box_line() + "\n")
    time.sleep(0.04)
    _w(_border("╰", "╯") + "\n")


def _animate_subtagline() -> None:
    """Phase 2: typewriter sub-tagline below the box."""
    _w("\n")
    _w(f"{INDENT}{DIM}")
    _typewriter("Agents that learn by watching you work,")
    _w(f"{RESET}\n{INDENT}{DIM}")
    _typewriter("then do it for you.")
    _w(f"{RESET}\n")


def _print_quickstart() -> None:
    """Phase 3: static Quick Start command card."""
    title = " Quick Start "
    bar = "─" * (W - 1 - len(title))  # 1 char for the leading "─" before title
    _w("\n")
    _w(
        f"{INDENT}{DIM}┌─{RESET}{BOLD}{title}{RESET}"
        f"{DIM}{bar}┐{RESET}\n"
    )

    rows = (
        ("leap",            "Interactive REPL mode"),
        ("leap \"...\"",    "Single-turn chat"),
        ("leap learn",      "Learn a new skill"),
        ("leap run",        "Execute a learned skill"),
    )
    cmd_w = 18  # padded command column width
    for cmd, desc in rows:
        cmd_padded = cmd.ljust(cmd_w)
        visible = 2 + cmd_w + 2 + len(desc)  # "  " + cmd + "  " + desc
        right = W - visible
        _w(
            f"{INDENT}{DIM}│{RESET}"
            f"  {CYAN}{cmd_padded}{RESET}  {DIM_WHITE}{desc}{RESET}"
            f"{' ' * right}"
            f"{DIM}│{RESET}\n"
        )
    _w(f"{INDENT}{DIM}└{'─' * W}┘{RESET}\n")
    _w(f"\n{INDENT}{DIM}Run `leap --help` for full command reference.{RESET}\n\n")


# ── Static fallback (non-TTY: pipes, redirects, CI logs) ──────────────────

def _print_static() -> None:
    for line in (
        "",
        _border("╭", "╮"),
        _empty_box_line(),
        _leap_logo_line(progress=4),
        _empty_box_line(),
        _tagline_line(lit=True),
        _empty_box_line(),
        _version_box_line(),
        _empty_box_line(),
        _border("╰", "╯"),
        "",
        f"{INDENT}{DIM}Agents that learn by watching you work,{RESET}",
        f"{INDENT}{DIM}then do it for you.{RESET}",
    ):
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    _print_quickstart()


# ── Public entry ──────────────────────────────────────────────────────────

def display_welcome() -> None:
    """Render the LEAP welcome banner.

    Animated on TTYs; degraded to a static render when stdout is piped/redirected.
    Ctrl+C during animation exits cleanly without a traceback.
    """
    if not _tty():
        try:
            _print_static()
        except (BrokenPipeError, KeyboardInterrupt):
            pass
        return

    try:
        _w(CURSOR_HIDE)
        _w("\n")
        _animate_logo_box()
        _animate_subtagline()
        _print_quickstart()
    except KeyboardInterrupt:
        # Graceful interrupt: leave a clean newline, suppress traceback.
        _w("\n")
    finally:
        _w(CURSOR_SHOW)

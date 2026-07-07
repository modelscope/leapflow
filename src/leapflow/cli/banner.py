"""LeapFlow welcome banner.

Two display modes:
1. **Rich panel** (interactive REPL): two-column layout with tools, skills,
   session info — high information density.
2. **Animated ASCII** (``leap`` without subcommand): progressive fade-in of
   the LEAP logo with typewriter tagline.

The Rich banner is used by ``cmd_interactive``; the animated version by the
bare ``leap`` entrypoint.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from leapflow.version import __version__

# ── ANSI escape codes (animated banner only) ─────────────────────────
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
BRIGHT_CYAN = "\033[1;36m"
BRIGHT_ORANGE = "\033[1;38;5;208m"
DIM_WHITE = "\033[2;37m"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"

VERSION = __version__
INDENT = "    "
W = 50

# ── Rich banner (interactive mode) ───────────────────────────────────

_TOOL_CATEGORIES: Dict[str, str] = {
    "file_list": "file",
    "file_read": "file",
    "file_write": "file",
    "shell_run": "shell",
    "text_search": "text",
    "text_replace": "text",
    "time_get": "system",
    "env_info": "system",
    "skills_list": "skill",
    "skill_view": "skill",
    "memory_search": "memory",
    "memory_add": "memory",
    "delegate_task": "agent",
    "hub_push": "hub",
    "hub_pull": "hub",
    "hub_search": "hub",
}


def _categorize_tools(
    tool_defs: Sequence[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Group tool names by category for display."""
    groups: Dict[str, List[str]] = {}
    for td in tool_defs:
        name = td.get("function", {}).get("name", "")
        if not name:
            continue
        cat = _TOOL_CATEGORIES.get(name, "other")
        groups.setdefault(cat, []).append(name)
    return dict(sorted(groups.items()))


def _categorize_skills(
    skills: Sequence[Any],
) -> Dict[str, List[str]]:
    """Group skills by category (from metadata or 'general')."""
    groups: Dict[str, List[str]] = {}
    for s in skills:
        cat = getattr(s, "category", None) or "general"
        groups.setdefault(cat, []).append(s.name)
    return dict(sorted(groups.items()))


def display_rich_banner(
    *,
    model: str = "",
    cwd: str = "",
    session_id: str = "",
    platform_online: bool = False,
    tool_defs: Optional[Sequence[Dict[str, Any]]] = None,
    skills: Optional[Sequence[Any]] = None,
    context_length: int = 0,
    mcp_tools: int = 0,
) -> None:
    """Print the information-dense Rich banner panel."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        display_welcome()
        return

    console = Console(highlight=False, soft_wrap=True)
    term_width = shutil.get_terminal_size().columns

    # ── Colors ──
    accent = "cyan"
    dim = "bright_black"
    text_color = ""

    # ── Left column: logo + metadata ──
    left_lines: list[str] = []
    left_lines.append("")
    left_lines.append(f"[bold {accent}]L . E . A . P[/]")
    left_lines.append(f"[dim]Learning and Evolving from Actual Practice[/]")
    left_lines.append("")
    if model:
        model_short = model.split("/")[-1] if "/" in model else model
        ctx_str = f"  ({context_length // 1000}K ctx)" if context_length else ""
        left_lines.append(f"[{accent}]{model_short}[/][dim]{ctx_str}[/]")
    if cwd:
        short_cwd = cwd.replace(os.path.expanduser("~"), "~")
        left_lines.append(f"[{dim}]{short_cwd}[/]")
    if session_id:
        left_lines.append(f"[{dim}]session: {session_id[:12]}[/]")
    left_lines.append("")

    left_content = "\n".join(left_lines)

    # ── Right column: tools + skills ──
    right_lines: list[str] = []

    # Tools
    if tool_defs:
        tool_groups = _categorize_tools(tool_defs)
        right_lines.append(f"[bold {accent}]Available Tools[/]")
        for cat, names in tool_groups.items():
            names_str = ", ".join(sorted(names))
            if len(names_str) > 50:
                names_str = names_str[:47] + "…"
            right_lines.append(f"[{dim}]{cat}:[/] {names_str}")

        if mcp_tools > 0:
            right_lines.append(f"[{dim}]mcp:[/] {mcp_tools} platform tools")

    # Skills
    if skills:
        right_lines.append("")
        right_lines.append(f"[bold {accent}]Available Skills[/]")
        skill_groups = _categorize_skills(skills)
        for cat, names in skill_groups.items():
            names_str = ", ".join(sorted(names))
            if len(names_str) > 50:
                names_str = names_str[:47] + "…"
            right_lines.append(f"[{dim}]{cat}:[/] {names_str}")

    if not right_lines:
        right_lines.append(f"[{dim}]No tools or skills loaded[/]")

    # Summary line
    tool_count = len(tool_defs) if tool_defs else 0
    skill_count = len(skills) if skills else 0
    summary_parts = []
    if tool_count:
        summary_parts.append(f"{tool_count} tools")
    if skill_count:
        summary_parts.append(f"{skill_count} skills")
    if mcp_tools:
        summary_parts.append(f"{mcp_tools} mcp")
    summary_parts.append("/help for commands")
    right_lines.append("")
    right_lines.append(f"[{dim}]{' · '.join(summary_parts)}[/]")

    right_content = "\n".join(right_lines)

    # ── Assembly ──
    layout = Table.grid(padding=(0, 3))
    layout.add_column("left", justify="center", min_width=min(40, term_width // 3))
    layout.add_column("right", justify="left")
    layout.add_row(left_content, right_content)

    # Compact mode for narrow terminals
    if term_width < 80:
        panel = Panel(
            Text.from_markup(f"[bold {accent}]L.E.A.P[/]  [dim]v{VERSION}[/]"),
            border_style=dim,
            padding=(0, 1),
        )
    else:
        version_label = f"LeapFlow v{VERSION}"
        conn_indicator = "[green]●[/]" if platform_online else f"[{dim}]○[/]"
        title = f"[bold {accent}]{version_label}[/]  {conn_indicator}"
        panel = Panel(
            layout,
            title=title,
            border_style=dim,
            padding=(0, 1),
        )

    console.print()
    console.print(panel)


# ── Animated ASCII banner (bare `leap` command) ──────────────────────

def _tty() -> bool:
    return sys.stdout.isatty()


def _w(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _pad(visible_len: int) -> tuple[int, int]:
    left = (W - visible_len) // 2
    return left, W - visible_len - left


def _box_line(content: str = "", visible_len: int = 0) -> str:
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


def _leap_logo_line(progress: int) -> str:
    letters = ("L", "E", "A", "P")
    parts: list[str] = []
    for i, ch in enumerate(letters):
        color = BRIGHT_CYAN if i < progress else DIM
        parts.append(f"{color}{ch}{RESET}")
        if i < 3:
            parts.append(f"{DIM} . {RESET}")
    return _box_line("".join(parts), visible_len=13)


def _tagline_line(lit: bool) -> str:
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


def _typewriter(text: str, delay: float = 0.012) -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)


def _animate_logo_box() -> None:
    _w(_border("╭", "╮") + "\n")
    time.sleep(0.06)
    _w(_empty_box_line() + "\n")
    time.sleep(0.06)
    for progress in range(5):
        _w("\r" + _leap_logo_line(progress))
        time.sleep(0.07)
    _w("\n")
    _w(_empty_box_line() + "\n")
    time.sleep(0.05)
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
    _w("\n")
    _w(f"{INDENT}{DIM}")
    _typewriter("Agents that learn by watching you work,")
    _w(f"{RESET}\n{INDENT}{DIM}")
    _typewriter("then do it for you.")
    _w(f"{RESET}\n")


def _print_quickstart() -> None:
    title = " Quick Start "
    bar = "─" * (W - 1 - len(title))
    _w("\n")
    _w(
        f"{INDENT}{DIM}┌─{RESET}{BOLD}{title}{RESET}"
        f"{DIM}{bar}┐{RESET}\n"
    )
    rows = (
        ("leap", "Interactive REPL mode"),
        ("leap \"...\"", "Single-turn chat"),
        ("leap teach", "Teach a new skill"),
        ("leap run", "Execute a learned skill"),
    )
    cmd_w = 18
    for cmd, desc in rows:
        cmd_padded = cmd.ljust(cmd_w)
        visible = 2 + cmd_w + 2 + len(desc)
        right = W - visible
        _w(
            f"{INDENT}{DIM}│{RESET}"
            f"  {CYAN}{cmd_padded}{RESET}  {DIM_WHITE}{desc}{RESET}"
            f"{' ' * right}"
            f"{DIM}│{RESET}\n"
        )
    _w(f"{INDENT}{DIM}└{'─' * W}┘{RESET}\n")
    _w(f"\n{INDENT}{DIM}Run `leap --help` for full command reference.{RESET}\n\n")


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


def display_welcome() -> None:
    """Render the animated LEAP welcome banner (bare ``leap`` entrypoint).

    For the Rich interactive banner, use ``display_rich_banner()``.
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
        _w("\n")
    finally:
        _w(CURSOR_SHOW)

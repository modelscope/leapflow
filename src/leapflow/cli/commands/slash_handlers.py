"""Slash-command handler implementations.

Each handler follows the signature ``(ctx, console, args) -> None``.
All display logic uses ``LeapConsole`` for consistent theming.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.cli.tui_app.console import LeapConsole


def handle_status(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display session status: model, context, platform, session info."""
    from rich.panel import Panel
    from rich.text import Text

    info = Text()

    model = getattr(ctx.settings, "model", "unknown")
    info.append("Model:     ", style="dim")
    info.append(f"{model}\n", style="bold")

    engine = ctx.engine
    if engine is not None:
        cap = getattr(engine, "model_capabilities", None)
        ctx_len = getattr(cap, "context_length", 0) if cap else 0
        ctx_used = getattr(engine, "context_token_count", 0)
        turn_count = getattr(engine, "turn_count", 0)

        if ctx_len:
            pct = int(ctx_used * 100 / ctx_len) if ctx_len else 0
            info.append("Context:   ", style="dim")
            pct_style = "bold red" if pct >= 90 else ("yellow" if pct >= 75 else "")
            info.append(f"{ctx_used:,} / {ctx_len:,} ({pct}%)\n", style=pct_style)

        info.append("Turns:     ", style="dim")
        info.append(f"{turn_count}\n")

    platform_status = "connected" if (hasattr(ctx.rpc, "connected") and ctx.rpc.connected) else "mock"
    info.append("Platform:  ", style="dim")
    p_style = "green" if platform_status == "connected" else "dim"
    info.append(f"{platform_status}\n", style=p_style)

    cwd = os.getcwd().replace(os.path.expanduser("~"), "~")
    info.append("CWD:       ", style="dim")
    info.append(f"{cwd}\n")

    session_id = getattr(ctx.session, "session_id", "")
    if session_id:
        info.append("Session:   ", style="dim")
        info.append(f"{session_id}\n")

    from leapflow.engine.session import SessionMode
    mode = "idle"
    if ctx.session:
        if ctx.session.mode == SessionMode.LEARNING:
            mode = "learning"
        elif ctx.session.mode == SessionMode.EXECUTING:
            mode = "executing"
    info.append("Mode:      ", style="dim")
    info.append(f"{mode}\n")

    console.print(Panel(
        info,
        title="[bold cyan]LeapFlow Status[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))


def handle_tools(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display available tools grouped by category."""
    from rich.table import Table
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS
    from leapflow.cli.banner import _categorize_tools

    tool_groups = _categorize_tools(TOOL_DEFINITIONS)

    table = Table(
        title="Available Tools",
        show_header=True,
        header_style="bold",
        border_style="bright_black",
        title_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Tools")

    for cat, names in tool_groups.items():
        table.add_row(cat, ", ".join(sorted(names)))

    mcp_count = 0
    if hasattr(ctx.rpc, "connected") and ctx.rpc.connected:
        mcp_count = len(getattr(ctx, "platform_tools", []))
    if mcp_count > 0:
        table.add_row("mcp", f"{mcp_count} platform tools")

    console.print(table)
    console.system(f"{sum(len(v) for v in tool_groups.values())} tools available")


def handle_usage(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display token usage for the current session."""
    from leapflow.cli.tui_app.status import _compact_tokens

    engine = ctx.engine
    if engine is None:
        console.warning("No active engine — send a message first.")
        return

    tracker = getattr(engine, "_turn_usage", None)
    if tracker is None:
        console.warning("Usage tracking not available.")
        return

    console.print()

    total = getattr(tracker, "total_tokens", 0)
    input_t = getattr(tracker, "total_input_tokens", 0)
    output_t = getattr(tracker, "total_output_tokens", 0)
    turn_count = getattr(engine, "turn_count", 0)

    cap = getattr(engine, "model_capabilities", None)
    ctx_len = getattr(cap, "context_length", 0) if cap else 0
    ctx_used = getattr(engine, "context_token_count", 0)

    lines = [
        f"  Model:           {getattr(ctx.settings, 'model', 'unknown')}",
        f"  Input tokens:    {_compact_tokens(input_t):>8}  ({input_t:,})",
        f"  Output tokens:   {_compact_tokens(output_t):>8}  ({output_t:,})",
        f"  Total tokens:    {_compact_tokens(total):>8}  ({total:,})",
        f"  Turns:           {turn_count}",
    ]

    if ctx_len > 0:
        pct = int(ctx_used * 100 / ctx_len) if ctx_len else 0
        lines.append(
            f"  Context:         {_compact_tokens(ctx_used)}/{_compact_tokens(ctx_len)} ({pct}%)"
        )

    for line in lines:
        console.system(line)
    console.print()


def handle_model(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Show or switch the active model."""
    model_arg = args.strip()
    if not model_arg:
        model = getattr(ctx.settings, "model", "unknown")
        console.system(f"Current model: {model}")
        cap = None
        engine = ctx.engine
        if engine is not None:
            cap = getattr(engine, "model_capabilities", None)
        if cap:
            ctx_len = getattr(cap, "context_length", 0)
            if ctx_len:
                console.system(f"Context length: {ctx_len:,}")
        return

    if hasattr(ctx.settings, "model"):
        old = ctx.settings.model
        ctx.settings.model = model_arg
        console.success(f"Model switched: {old} → {model_arg}")
        console.system("Note: takes effect on next turn.")
    else:
        console.warning("Model switching not supported in current config.")


def handle_clear(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

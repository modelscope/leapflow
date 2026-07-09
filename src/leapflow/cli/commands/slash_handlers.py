"""Slash-command handler implementations.

Each handler follows the signature ``(ctx, console, args) -> None``.
All display logic uses ``LeapConsole`` for consistent theming.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.cli.tui_app.console import LeapConsole


def handle_status(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display session status: model, context, platform, session info."""
    from rich.panel import Panel
    from rich.text import Text

    info = Text()

    info.append("Model:     ", style="dim")
    info.append(f"{ctx.settings.llm_model}\n", style="bold")

    engine = ctx.engine
    if engine is not None:
        cap_registry = getattr(engine, "model_capabilities", None)
        ctx_len = 0
        if cap_registry is not None:
            caps = cap_registry.resolve(ctx.settings.llm_model)
            ctx_len = caps.context_length
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

    gw = getattr(ctx, "gateway_server", None)
    if gw is not None:
        statuses = gw.platform_status()
        connected = [s for s in statuses if s.connected]
        info.append("Gateway:   ", style="dim")
        if connected:
            names = []
            for s in connected:
                m = gw.manifests.get(s.platform_id)
                names.append(m.display_name if m else s.platform_id)
            info.append(f"{', '.join(names)}\n", style="green")
        else:
            info.append("no connections\n", style="dim")

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

    tracker = getattr(engine, "usage_tracker", None)
    if tracker is None:
        console.warning("Usage tracking not available.")
        return

    console.print()

    summary = tracker.summary()
    turn_count = getattr(engine, "turn_count", 0)

    cap_registry = getattr(engine, "model_capabilities", None)
    ctx_len = 0
    if cap_registry is not None:
        caps = cap_registry.resolve(ctx.settings.llm_model)
        ctx_len = caps.context_length
    ctx_used = getattr(engine, "context_token_count", 0)

    lines = [
        f"  Model:           {ctx.settings.llm_model}",
        f"  Input tokens:    {_compact_tokens(summary.prompt_tokens):>8}  ({summary.prompt_tokens:,})",
        f"  Output tokens:   {_compact_tokens(summary.completion_tokens):>8}  ({summary.completion_tokens:,})",
        f"  Total tokens:    {_compact_tokens(summary.total_tokens):>8}  ({summary.total_tokens:,})",
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
        console.system(f"Current model: {ctx.settings.llm_model}")
        engine = ctx.engine
        if engine is not None:
            cap_registry = getattr(engine, "model_capabilities", None)
            if cap_registry is not None:
                caps = cap_registry.resolve(ctx.settings.llm_model)
                console.system(f"Context length: {caps.context_length:,}")
        return

    console.warning(
        "Model switching requires restarting with LEAPFLOW_LLM_MODEL env var."
    )
    console.system(f"  Current: {ctx.settings.llm_model}")
    console.system(f"  Example: LEAPFLOW_LLM_MODEL={model_arg} leap")


def handle_gateway(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display gateway status: connected platforms, available integrations."""
    from rich.panel import Panel
    from rich.text import Text

    gw = getattr(ctx, "gateway_server", None)
    if gw is None:
        console.warning("Gateway not initialised.")
        return

    statuses = gw.platform_status()
    if not statuses:
        console.system("No platform manifests discovered.")
        return

    info = Text()
    connected = [s for s in statuses if s.connected]
    configured = [s for s in statuses if not s.connected and s.error == "configured but not connected"]
    available = [s for s in statuses if not s.connected and not s.error]

    import time

    if connected:
        info.append("Connected\n", style="bold green")
        for s in connected:
            m = gw.manifests.get(s.platform_id)
            name = m.display_name if m else s.platform_id
            uptime = ""
            if s.connected_since > 0:
                secs = int(time.time() - s.connected_since)
                if secs < 60:
                    uptime = f" ({secs}s)"
                elif secs < 3600:
                    uptime = f" ({secs // 60}m)"
                else:
                    uptime = f" ({secs // 3600}h {(secs % 3600) // 60}m)"
            info.append(f"  ● {name}{uptime}\n", style="green")

    if configured:
        info.append("Configured (not connected)\n", style="bold yellow")
        for s in configured:
            m = gw.manifests.get(s.platform_id)
            name = m.display_name if m else s.platform_id
            info.append(f"  ○ {name}\n", style="yellow")

    if available:
        info.append("Available\n", style="bold dim")
        names = [gw.manifests[s.platform_id].display_name for s in available if s.platform_id in gw.manifests]
        info.append(f"  {', '.join(names)}\n", style="dim")

    info.append("\n", style="dim")
    info.append('Say "connect to <platform>" to set up a new integration.', style="dim italic")

    console.print(Panel(
        info,
        title="[bold cyan]Gateway[/]",
        border_style="bright_black",
        padding=(0, 2),
    ))


def handle_clear(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

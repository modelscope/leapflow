"""Slash-command handler implementations.

Each handler follows the signature ``(ctx, console, args) -> None``.
All display logic uses ``LeapConsole`` for consistent theming.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.cli.tui_app.console import LeapConsole


def build_tools_payload(ctx: "Context") -> dict[str, Any]:
    """Build a serializable tools summary for local or daemon rendering."""
    from leapflow.cli.banner import _categorize_tools
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

    tool_groups = _categorize_tools(TOOL_DEFINITIONS)
    groups = {category: sorted(names) for category, names in tool_groups.items()}
    mcp_count = 0
    if hasattr(ctx.rpc, "connected") and ctx.rpc.connected:
        mcp_count = len(getattr(ctx, "platform_tools", []))
    return {
        "ok": True,
        "groups": groups,
        "total": sum(len(names) for names in groups.values()),
        "mcp_count": mcp_count,
    }


def render_tools_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a serializable tools summary."""
    from rich.table import Table

    if not payload.get("ok", True):
        console.warning(str(payload.get("error") or "Tools are not available."))
        return

    groups = dict(payload.get("groups") or {})
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

    for category, names in groups.items():
        table.add_row(str(category), ", ".join(sorted(str(name) for name in names)))

    mcp_count = int(payload.get("mcp_count") or 0)
    if mcp_count > 0:
        table.add_row("mcp", f"{mcp_count} platform tools")

    console.print(table)
    console.system(f"{int(payload.get('total') or 0)} tools available")


def build_usage_payload(ctx: "Context") -> dict[str, Any]:
    """Build a serializable token usage summary."""
    engine = ctx.engine
    if engine is None:
        return {"ok": False, "error": "No active engine — send a message first."}

    tracker = getattr(engine, "usage_tracker", None)
    if tracker is None:
        return {"ok": False, "error": "Usage tracking not available."}

    summary = tracker.summary()
    cap_registry = getattr(engine, "model_capabilities", None)
    context_length = 0
    if cap_registry is not None:
        caps = cap_registry.resolve(ctx.settings.llm_model)
        context_length = int(caps.context_length)

    return {
        "ok": True,
        "model": ctx.settings.llm_model,
        "prompt_tokens": int(summary.prompt_tokens),
        "completion_tokens": int(summary.completion_tokens),
        "total_tokens": int(summary.total_tokens),
        "turn_count": int(getattr(engine, "turn_count", 0)),
        "context_used": int(getattr(engine, "context_token_count", 0)),
        "context_length": context_length,
    }


def render_usage_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a serializable token usage summary."""
    from leapflow.cli.tui_app.status import _compact_tokens

    if not payload.get("ok", True):
        console.warning(str(payload.get("error") or "Usage tracking not available."))
        return

    console.print()
    prompt_tokens = int(payload.get("prompt_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or 0)
    turn_count = int(payload.get("turn_count") or 0)
    context_used = int(payload.get("context_used") or 0)
    context_length = int(payload.get("context_length") or 0)
    lines = [
        f"  Model:           {payload.get('model') or ''}",
        f"  Input tokens:    {_compact_tokens(prompt_tokens):>8}  ({prompt_tokens:,})",
        f"  Output tokens:   {_compact_tokens(completion_tokens):>8}  ({completion_tokens:,})",
        f"  Total tokens:    {_compact_tokens(total_tokens):>8}  ({total_tokens:,})",
        f"  Turns:           {turn_count}",
    ]
    if context_length > 0:
        pct = int(context_used * 100 / context_length)
        lines.append(
            f"  Context:         {_compact_tokens(context_used)}/{_compact_tokens(context_length)} ({pct}%)"
        )
    for line in lines:
        console.system(line)
    console.print()


def build_model_payload(ctx: "Context", args: str = "") -> dict[str, Any]:
    """Build a serializable model summary."""
    model_arg = args.strip()
    engine = ctx.engine
    context_length = 0
    if engine is not None:
        cap_registry = getattr(engine, "model_capabilities", None)
        if cap_registry is not None:
            caps = cap_registry.resolve(ctx.settings.llm_model)
            context_length = int(caps.context_length)
    return {
        "ok": True,
        "model": ctx.settings.llm_model,
        "context_length": context_length,
        "requested_model": model_arg,
        "switch_supported": False,
        "env_var": "LEAPFLOW_LLM_MODEL",
    }


def render_model_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a serializable model summary."""
    if not payload.get("ok", True):
        console.warning(str(payload.get("error") or "Model information is not available."))
        return
    requested = str(payload.get("requested_model") or "")
    model = str(payload.get("model") or "")
    if not requested:
        console.system(f"Current model: {model}")
        context_length = int(payload.get("context_length") or 0)
        if context_length > 0:
            console.system(f"Context length: {context_length:,}")
        return

    console.warning("Model switching requires restarting with LEAPFLOW_LLM_MODEL env var.")
    console.system(f"  Current: {model}")
    console.system(f"  Example: LEAPFLOW_LLM_MODEL={requested} leap")


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

    config_path = ctx.settings.data_dir / ".env"
    info.append("Config:    ", style="dim")
    info.append(f"{str(config_path).replace(os.path.expanduser('~'), '~')}\n")

    project_env = os.path.join(os.getcwd(), ".env")
    info.append("Override:  ", style="dim")
    info.append(f"{project_env.replace(os.path.expanduser('~'), '~')}\n")

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
    render_tools_payload(console, build_tools_payload(ctx))


def handle_usage(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Display token usage for the current session."""
    render_usage_payload(console, build_usage_payload(ctx))


def handle_model(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Show or switch the active model."""
    render_model_payload(console, build_model_payload(ctx, args))


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

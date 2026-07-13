"""Slash-command handler implementations.

Each handler follows the signature ``(ctx, console, args) -> None``.
All display logic uses ``LeapConsole`` for consistent theming.
"""

from __future__ import annotations

import os
import shlex
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


def _app_usage() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "Usage: /app [platform] | /app status [platform] | /app connect <platform> [--option value] | /app disconnect <platform> | /app remove <platform> | /app events [status|start|stop] <platform> | /app actions <platform>",
        "next_actions": ("/app", "/app <platform>", "/app status <platform>"),
    }


def _parse_app_options(tokens: list[str]) -> tuple[dict[str, str], str]:
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--") or token == "--":
            return {}, f"Unexpected argument: {token}"
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            if not key:
                return {}, f"Invalid option: {token}"
            options[key] = value
            index += 1
            continue
        if index + 1 >= len(tokens):
            return {}, f"Missing value for option: {token}"
        key = key_value
        if not key:
            return {}, f"Invalid option: {token}"
        options[key] = tokens[index + 1]
        index += 2
    return options, ""


def _parse_app_params(args: str) -> dict[str, Any]:
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return {"ok": False, "error": f"Invalid /app arguments: {exc}"}

    if not tokens or tokens[0].lower() == "list":
        if len(tokens) > 1:
            return _app_usage()
        return {"ok": True, "params": {"action": "list"}, "view": "list"}

    head = tokens[0].lower()
    if head == "status":
        if len(tokens) > 2:
            return _app_usage()
        params: dict[str, Any] = {"action": "status"}
        if len(tokens) == 2:
            params["platform"] = tokens[1].lower()
        return {"ok": True, "params": params, "view": "status"}

    if head == "connect":
        if len(tokens) < 2:
            return _app_usage()
        options, error = _parse_app_options(tokens[2:])
        if error:
            return {
                "ok": False,
                "error": error,
                "next_actions": ("/app connect <platform> --<option> <value>",),
            }
        params = {"action": "connect", "platform": tokens[1].lower()}
        if options:
            params["options"] = options
        return {"ok": True, "params": params, "view": "connect"}

    if head in {"disconnect", "remove"}:
        if len(tokens) != 2:
            return _app_usage()
        return {
            "ok": True,
            "params": {"action": head, "platform": tokens[1].lower()},
            "view": head,
        }

    if head == "events":
        if len(tokens) == 2:
            return {
                "ok": True,
                "params": {"action": "events_status", "platform": tokens[1].lower()},
                "view": "events",
            }
        if len(tokens) == 3 and tokens[1].lower() in {"status", "start", "stop"}:
            event_action = tokens[1].lower()
            action = "events_status" if event_action == "status" else f"events_{event_action}"
            return {
                "ok": True,
                "params": {"action": action, "platform": tokens[2].lower()},
                "view": "events",
            }
        return _app_usage()

    if head == "actions":
        if len(tokens) != 2:
            return _app_usage()
        return {"ok": True, "platform": tokens[1].lower(), "view": "actions"}

    if len(tokens) == 1:
        return {
            "ok": True,
            "params": {"action": "guide", "platform": tokens[0].lower()},
            "view": "guide",
        }

    return _app_usage()


def _resolve_app_platform(gw: Any, platform: str) -> tuple[str, str]:
    if not platform:
        return "", ""
    manifests = getattr(gw, "manifests", {}) or {}
    if platform in manifests:
        return platform, ""
    folded = platform.casefold()
    matches = [pid for pid in manifests if str(pid).casefold() == folded]
    if len(matches) == 1:
        return matches[0], ""
    return "", f"Unknown platform: {platform}"


def _available_app_ids(gw: Any) -> list[str]:
    manifests = getattr(gw, "manifests", {}) or {}
    return sorted(str(platform_id) for platform_id in manifests)


def _coerce_app_option(field: Any, value: str) -> tuple[Any, str]:
    choices = tuple(getattr(field, "choices", ()) or ())
    if choices and value not in choices:
        return None, f"Invalid value for --{field.key}: {value}. Choices: {', '.join(map(str, choices))}"

    field_type = str(getattr(field, "field_type", "string") or "string").lower()
    if field_type in {"string", "choice"}:
        return value, ""
    if field_type in {"integer", "int"}:
        try:
            return int(value), ""
        except ValueError:
            return None, f"Invalid integer for --{field.key}: {value}"
    if field_type in {"number", "float"}:
        try:
            return float(value), ""
        except ValueError:
            return None, f"Invalid number for --{field.key}: {value}"
    if field_type in {"boolean", "bool"}:
        lowered = value.casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True, ""
        if lowered in {"0", "false", "no", "off"}:
            return False, ""
        return None, f"Invalid boolean for --{field.key}: {value}"
    return value, ""


def _validate_app_options(manifest: Any, options: dict[str, str]) -> tuple[dict[str, Any], str]:
    if not options:
        return {}, ""
    fields = {str(field.key): field for field in getattr(manifest, "options", ())}
    unknown = sorted(key for key in options if key not in fields)
    if unknown:
        available = ", ".join(sorted(fields)) or "no options"
        return {}, f"Unknown option(s): {', '.join(unknown)}. Available options: {available}"

    coerced: dict[str, Any] = {}
    for key, value in options.items():
        converted, error = _coerce_app_option(fields[key], value)
        if error:
            return {}, error
        coerced[key] = converted
    return coerced, ""


async def build_app_payload(ctx: "Context", args: str = "") -> dict[str, Any]:
    """Build a serializable App Connector slash-command payload."""
    gw = getattr(ctx, "gateway_server", None)
    if gw is None:
        return {
            "ok": False,
            "error": "Gateway is not initialised in this session.",
            "next_actions": ("Start an in-process LeapFlow session", "Use /help to inspect runtime support"),
        }

    parsed = _parse_app_params(args)
    if not parsed.get("ok"):
        parsed.setdefault("available", _available_app_ids(gw))
        return parsed

    from leapflow.tools.gateway_tool import platform_connect_handler, set_gateway_server

    set_gateway_server(gw)
    if parsed.get("view") == "actions":
        platform, error = _resolve_app_platform(gw, str(parsed.get("platform") or ""))
        if error:
            return {"ok": False, "error": error, "available": _available_app_ids(gw)}
        manifest = gw.manifests.get(platform)
        if manifest is None:
            return {"ok": False, "error": f"Unknown platform: {platform}", "available": _available_app_ids(gw)}
        actions = dict(getattr(manifest, "actions", {}) or {})
        return {
            "ok": True,
            "view": "actions",
            "platform": platform,
            "name": getattr(manifest, "display_name", platform),
            "actions": actions,
            "domains": list(actions.get("initial_domains") or ()),
        }

    params = dict(parsed.get("params") or {})
    platform = str(params.get("platform") or "")
    if platform:
        resolved, error = _resolve_app_platform(gw, platform)
        if error:
            return {"ok": False, "error": error, "available": _available_app_ids(gw)}
        params["platform"] = resolved
        manifest = gw.manifests.get(resolved)
        if manifest is not None and params.get("action") == "connect":
            options, option_error = _validate_app_options(
                manifest,
                dict(params.get("options") or {}),
            )
            if option_error:
                return {
                    "ok": False,
                    "error": option_error,
                    "available": _available_app_ids(gw),
                    "next_actions": (f"/app {resolved}", f"/app connect {resolved}"),
                }
            if options:
                params["options"] = options

    result = await platform_connect_handler(params)
    view = str(parsed.get("view") or "")
    if params.get("action") == "connect" and result.get("setup_guide"):
        view = "guide"
    return {
        "ok": bool(result.get("ok")),
        "view": view,
        "params": params,
        "result": result,
    }


def _append_result_metadata(lines: list[str], result: dict[str, Any]) -> None:
    diagnostics = result.get("diagnostics") or result.get("metadata")
    if isinstance(diagnostics, dict) and diagnostics:
        for key in ("backend_kind", "profile", "identity", "auth_status", "configured", "current_mode"):
            if diagnostics.get(key) is not None:
                lines.append(f"  {key}: {diagnostics[key]}")
    recovery_hint = result.get("recovery_hint")
    if recovery_hint:
        lines.append(f"  recovery: {recovery_hint}")
    next_steps = result.get("next_steps")
    if isinstance(next_steps, list) and next_steps:
        lines.append("  next steps:")
        lines.extend(f"    - {step}" for step in next_steps)


def render_app_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a serializable App Connector slash-command payload."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if not payload.get("ok"):
        result = dict(payload.get("result") or {})
        message = (
            payload.get("error")
            or result.get("error")
            or result.get("detail")
            or "App command failed."
        )
        console.warning(str(message))
        lines: list[str] = []
        _append_result_metadata(lines, result)
        for line in lines:
            console.system(line)
        available = payload.get("available")
        if isinstance(available, list) and available:
            console.system("Available: " + ", ".join(str(item) for item in available))
        for action in payload.get("next_actions") or ():
            console.system(f"  → {action}")
        return

    result = dict(payload.get("result") or {})
    view = str(payload.get("view") or "")
    if view == "list":
        platforms = list(result.get("platforms") or [])
        table = Table(title="App Connector", show_header=True, header_style="bold", border_style="bright_black")
        table.add_column("App", style="cyan")
        table.add_column("ID", style="bold")
        table.add_column("State")
        table.add_column("Category")
        table.add_column("Next")
        for entry in platforms:
            platform_id = str(entry.get("id") or "")
            state = str(entry.get("state") or "available")
            style = "green" if state == "connected" else ("yellow" if state == "configured" else "dim")
            table.add_row(
                str(entry.get("name") or platform_id),
                platform_id,
                f"[{style}]{state}[/]",
                str(entry.get("category") or ""),
                f"/app {platform_id}",
            )
        console.print(table)
        console.system("Next: /app <platform> · /app status <platform> · /app connect <platform>")
        return

    if view == "guide":
        info = Text()
        info.append(str(result.get("setup_guide") or "No setup guide available.") + "\n")
        steps = result.get("setup_steps") or []
        if steps:
            info.append("\nSteps\n", style="bold")
            for index, step in enumerate(steps, start=1):
                info.append(f"  {index}. {step}\n")
        preflight = result.get("preflight_checks") or []
        if preflight:
            info.append("\nPreflight\n", style="bold")
            for check in preflight:
                command = check.get("command") or ""
                label = check.get("label") or check.get("key") or "check"
                info.append(f"  - {label}: {command}\n")
        fields = result.get("required_fields") or []
        if fields:
            info.append("\nRequired fields\n", style="bold")
            for field in fields:
                required = "required" if field.get("required") else "optional"
                info.append(f"  - {field.get('key')}: {field.get('label')} ({required})\n")
        console_url = result.get("console_url")
        if console_url:
            info.append(f"\nConsole: {console_url}\n", style="dim")
        platform = str(result.get("platform") or payload.get("params", {}).get("platform") or "App")
        console.print(Panel(info, title=f"[bold cyan]{platform} setup[/]", border_style="bright_black", padding=(0, 2)))
        platform_id = str(payload.get("params", {}).get("platform") or "<platform>")
        if fields:
            console.system(
                "Next: provide required credentials through the guided chat flow, "
                f"then run /app connect {platform_id}"
            )
        else:
            console.system(f"Next: /app connect {platform_id}")
        return

    if view == "status":
        platforms = result.get("platforms")
        if isinstance(platforms, list):
            table = Table(title="App Status", show_header=True, header_style="bold", border_style="bright_black")
            table.add_column("ID", style="bold")
            table.add_column("Connected")
            table.add_column("Detail")
            for entry in platforms:
                detail = str(entry.get("error") or entry.get("uptime") or "available")
                table.add_row(str(entry.get("id") or ""), "yes" if entry.get("connected") else "no", detail)
            console.print(table)
            return
        lines = [f"platform: {result.get('platform')}", f"connected: {bool(result.get('connected'))}"]
        if result.get("error"):
            lines.append(f"detail: {result['error']}")
        if result.get("uptime"):
            lines.append(f"uptime: {result['uptime']}")
        _append_result_metadata(lines, result)
        console.print(Panel("\n".join(lines), title="[bold cyan]App Status[/]", border_style="bright_black", padding=(0, 2)))
        return

    if view == "actions":
        domains = payload.get("domains") or []
        actions = dict(payload.get("actions") or {})
        lines = [f"platform: {payload.get('name')}"]
        if domains:
            lines.append("domains: " + ", ".join(str(item) for item in domains))
        if actions.get("pack"):
            lines.append(f"action pack: {actions['pack']}")
        console.print(Panel("\n".join(lines), title="[bold cyan]App Actions[/]", border_style="bright_black", padding=(0, 2)))
        return

    if result.get("ok"):
        platform = result.get("platform") or payload.get("params", {}).get("platform") or "App"
        status = result.get("status") or result.get("detail") or "ok"
        lines = [f"{platform}: {status}"]
        _append_result_metadata(lines, result)
        console.success("App command completed")
        console.system("\n".join(lines))
        return

    message = result.get("error") or result.get("detail") or "App command failed."
    console.warning(str(message))
    lines: list[str] = []
    _append_result_metadata(lines, result)
    for line in lines:
        console.system(line)


async def handle_app(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Manage external App Connector integrations from slash commands."""
    render_app_payload(console, await build_app_payload(ctx, args))



def handle_clear(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")

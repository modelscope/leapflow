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
    """Build a serializable model summary or update the active model."""
    model_arg = args.strip()
    if model_arg:
        payload = build_config_payload(ctx, f"llm set --model {shlex.quote(model_arg)}")
        payload["view"] = "model"
        payload["requested_model"] = model_arg
        return payload

    engine = ctx.engine
    context_length = 0
    if engine is not None:
        cap_registry = getattr(engine, "model_capabilities", None)
        if cap_registry is not None:
            caps = cap_registry.resolve(ctx.settings.llm_model)
            context_length = int(caps.context_length)
    return {
        "ok": True,
        "view": "model",
        "model": ctx.settings.llm_model,
        "context_length": context_length,
        "requested_model": "",
    }


def render_model_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a serializable model summary."""
    if not payload.get("ok", True):
        console.warning(str(payload.get("error") or payload.get("message") or "Model information is not available."))
        return
    requested = str(payload.get("requested_model") or "")
    model = str(payload.get("model") or "")
    if requested:
        console.success(f"Model updated: {model}")
        if payload.get("reloaded"):
            console.system("Configuration reloaded for this session.")
        return

    console.system(f"Current model: {model}")
    context_length = int(payload.get("context_length") or 0)
    if context_length > 0:
        console.system(f"Context length: {context_length:,}")


def build_config_payload(ctx: "Context", args: str = "") -> dict[str, Any]:
    """Execute a config command and return a serializable payload."""
    from leapflow.config_service import ConfigService

    service = ConfigService(ctx.settings)
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        return {"ok": False, "message": f"Invalid /config syntax: {exc}"}
    if not tokens:
        tokens = ["show"]
    action = tokens[0]
    try:
        if action == "show":
            if len(tokens) > 1:
                return {
                    "ok": True,
                    "view": "config",
                    "mode": "show_detail",
                    "field": _config_field_to_dict(service.describe(tokens[1])),
                }
            return _config_snapshot_payload(service)
        if action == "keys":
            return {"ok": True, "view": "config", "mode": "keys", "sources": list(service.writable_keys())}
        if action == "list":
            category = tokens[1] if len(tokens) > 1 and not tokens[1].startswith("--") else None
            return {"ok": True, "view": "config", "mode": "list", "fields": [_config_field_to_dict(item) for item in service.list_fields(category)]}
        if action == "sources":
            return {"ok": True, "view": "config", "mode": "sources", "sources": list(service.sources())}
        if action == "get" and len(tokens) == 2:
            value = service.get(tokens[1])
            return {"ok": True, "view": "config", "mode": "get", "values": [_config_value_to_dict(value)]}
        if action == "set" and len(tokens) >= 3:
            scope = _option_value(tokens[3:], "--scope", "profile")
            result = service.set(tokens[1], tokens[2], scope=scope)  # type: ignore[arg-type]
            return _config_mutation_payload(ctx, service, result)
        if action == "unset" and len(tokens) >= 2:
            scope = _option_value(tokens[2:], "--scope", "profile")
            result = service.unset(tokens[1], scope=scope)  # type: ignore[arg-type]
            return _config_mutation_payload(ctx, service, result)
        if action == "llm":
            return _build_llm_config_payload(ctx, service, tokens[1:])
        if action == "secret":
            return _build_secret_config_payload(ctx, service, tokens[1:])
    except (KeyError, ValueError, RuntimeError) as exc:
        return {"ok": False, "message": f"Config error: {exc}"}
    return {"ok": False, "message": "Usage: /config [show|list|keys|sources|get|set|unset|llm|secret] ..."}


def render_config_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render config command output."""
    if not payload.get("ok", True):
        console.warning(str(payload.get("message") or "Config command failed."))
        return
    mode = str(payload.get("mode") or "show")
    if mode in {"sources", "keys"}:
        label = "Writable config keys" if mode == "keys" else "Config sources"
        sources = [str(source) for source in payload.get("sources") or []]
        if not sources:
            console.system(f"No {label.lower()} found.")
            return
        console.system(f"{label}:")
        for source in sources:
            console.system(f"  {source}")
        return
    if mode == "list":
        fields = [dict(item) for item in payload.get("fields") or []]
        if not fields:
            console.system("No writable config fields found.")
            return
        from rich.table import Table

        table = Table(title="Writable config fields", show_lines=False)
        table.add_column("Key", no_wrap=True)
        table.add_column("Value", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Scope", no_wrap=True)
        table.add_column("Reload", no_wrap=True)
        table.add_column("Description")
        for item in fields:
            table.add_row(
                str(item.get("key") or ""),
                str(item.get("value") or ""),
                str(item.get("type") or ""),
                ",".join(str(scope) for scope in item.get("scopes") or []),
                str(item.get("hot_reload") or ""),
                str(item.get("description") or ""),
            )
        console.print(table)
        console.system("Use /config show <key>, /config get <key>, /config set <key> <value>, or /config keys for compact output.")
        return
    if mode == "show_detail":
        field = dict(payload.get("field") or {})
        if not field:
            console.warning("No config field details found.")
            return
        from rich.panel import Panel
        from rich.text import Text

        details = Text()
        details.append(f"value: {field.get('value') or ''}\n")
        details.append(f"type: {field.get('type') or ''}\n")
        details.append(f"category: {field.get('category') or ''}\n")
        details.append(f"scope: {','.join(str(scope) for scope in field.get('scopes') or [])}\n")
        details.append(f"reload: {field.get('hot_reload') or ''}\n")
        details.append(f"secret: {'true' if field.get('secret') else 'false'}\n")
        value_hint = str(field.get("value_hint") or "")
        if value_hint:
            details.append(f"value hint: {value_hint}\n")
        details.append(f"description: {field.get('description') or ''}")
        examples = [str(example) for example in field.get("examples") or []]
        if examples:
            details.append(f"\nexample: {examples[0]}")
        console.print(Panel(details, title=str(field.get("key") or "Config field"), border_style="cyan"))
        return
    if mode == "mutation":
        console.success(str(payload.get("message") or "Config updated."))
        for key in payload.get("changed_keys") or []:
            console.system(f"  {key}")
        if payload.get("reloaded"):
            console.system("Configuration reloaded for this session.")
        return
    values = payload.get("values") or []
    if values:
        for item in values:
            console.system(f"{item.get('key')}={item.get('value')}")
    warnings = payload.get("warnings") or []
    for warning in warnings:
        console.warning(str(warning))


def _config_snapshot_payload(service: Any) -> dict[str, Any]:
    snapshot = service.snapshot()
    return {
        "ok": True,
        "view": "config",
        "mode": "show",
        "values": [_config_value_to_dict(value) for value in snapshot.values],
        "sources": list(snapshot.sources),
        "warnings": list(snapshot.warnings),
    }


def _config_value_to_dict(value: Any) -> dict[str, Any]:
    return {"key": value.key, "value": value.value, "source": value.source, "secret": value.secret}


def _config_field_to_dict(value: Any) -> dict[str, Any]:
    return {
        "key": value.key,
        "value": value.value,
        "type": value.value_type,
        "category": value.category,
        "scopes": list(value.scopes),
        "hot_reload": value.hot_reload,
        "secret": value.secret,
        "description": value.description,
        "value_hint": value.value_hint,
        "examples": list(value.examples),
    }


def _build_llm_config_payload(ctx: "Context", service: Any, tokens: list[str]) -> dict[str, Any]:
    action = tokens[0] if tokens else "show"
    if action == "show":
        payload = _config_snapshot_payload(service)
        payload["values"] = [item for item in payload["values"] if str(item.get("key") or "").startswith("llm.")]
        return payload
    if action != "set":
        return {"ok": False, "message": "Usage: /config llm [show|set --model NAME --base-url URL --api-key KEY]"}
    options = tokens[1:]
    if "--ask-api-key" in options:
        return {"ok": False, "message": "Use `leap config llm key` in a terminal for secure API key prompts."}
    result = service.configure_llm(
        api_key=_option_value(options, "--api-key"),
        base_url=_option_value(options, "--base-url"),
        model=_option_value(options, "--model"),
        context_length=_option_int(options, "--context-length"),
        max_retries=_option_int(options, "--max-retries"),
        scope=_option_value(options, "--scope", "profile"),
    )
    return _config_mutation_payload(ctx, service, result)


def _build_secret_config_payload(ctx: "Context", service: Any, tokens: list[str]) -> dict[str, Any]:
    action = tokens[0] if tokens else "list"
    if action == "list":
        return {"ok": True, "view": "config", "mode": "sources", "sources": list(service.list_secrets())}
    if action == "set" and len(tokens) >= 3:
        scope = _option_value(tokens[3:], "--scope", "profile")
        result = service.set_secret(tokens[1], tokens[2], scope=scope)  # type: ignore[arg-type]
        return _config_mutation_payload(ctx, service, result)
    if action == "get" and len(tokens) >= 2:
        scope = _option_value(tokens[2:], "--scope", "profile")
        reveal = "--reveal" in tokens[2:]
        value = service.get_secret(tokens[1], scope=scope, reveal=reveal)  # type: ignore[arg-type]
        return {"ok": True, "view": "config", "mode": "get", "values": [{"key": tokens[1], "value": value, "secret": not reveal}]}
    if action == "delete" and len(tokens) >= 2:
        scope = _option_value(tokens[2:], "--scope", "profile")
        result = service.delete_secret(tokens[1], scope=scope)  # type: ignore[arg-type]
        return _config_mutation_payload(ctx, service, result)
    return {"ok": False, "message": "Usage: /config secret [list|set|get|delete] ..."}


def _config_mutation_payload(ctx: "Context", service: Any, result: Any) -> dict[str, Any]:
    reloaded = False
    if result.changed_keys:
        reloaded = bool(ctx.reload_runtime_config_if_changed(force=True))
    return {
        "ok": bool(result.ok),
        "view": "config",
        "mode": "mutation",
        "message": result.message,
        "changed_keys": list(result.changed_keys),
        "reloaded": reloaded,
        "model": ctx.settings.llm_model,
    }


def _option_value(tokens: list[str], option: str, default: str | None = None) -> str | None:
    if option not in tokens:
        return default
    index = tokens.index(option)
    if index + 1 >= len(tokens):
        raise ValueError(f"Missing value for {option}")
    return tokens[index + 1]


def _option_int(tokens: list[str], option: str) -> int | None:
    value = _option_value(tokens, option)
    return int(value) if value is not None else None


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

    info.append("Profile:   ", style="dim")
    info.append(f"{ctx.settings.profile}\n")

    info.append("Config:    ", style="dim")
    info.append(f"{str(ctx.settings.profile_layout.config_dir).replace(os.path.expanduser('~'), '~')}\n")

    user_config = str(ctx.settings.layout.user_config_path).replace(os.path.expanduser("~"), "~")
    info.append("User cfg:  ", style="dim")
    info.append(f"{user_config}\n")

    workspace_config = str(ctx.settings.workspace_root / ".leapflow" / "config.yaml")
    info.append("Workspace: ", style="dim")
    info.append(f"{workspace_config.replace(os.path.expanduser('~'), '~')}\n")

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


def handle_config(ctx: "Context", console: "LeapConsole", args: str) -> None:
    """View or update runtime configuration."""
    render_config_payload(console, build_config_payload(ctx, args))


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

    from leapflow.tools.gateway_tool import platform_action_capability_summary, platform_connect_handler, set_gateway_server

    set_gateway_server(gw)
    if parsed.get("view") == "actions":
        platform, error = _resolve_app_platform(gw, str(parsed.get("platform") or ""))
        if error:
            return {"ok": False, "error": error, "available": _available_app_ids(gw)}
        manifest = gw.manifests.get(platform)
        if manifest is None:
            return {"ok": False, "error": f"Unknown platform: {platform}", "available": _available_app_ids(gw)}
        actions = dict(getattr(manifest, "actions", {}) or {})
        action_summaries = platform_action_capability_summary(platform)
        return {
            "ok": True,
            "view": "actions",
            "platform": platform,
            "name": getattr(manifest, "display_name", platform),
            "action_pack": str(actions.get("pack") or ""),
            "actions": [item["name"] for item in action_summaries if item.get("name")],
            "action_details": action_summaries,
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
        action_names = list(payload.get("actions") or [])
        action_details = list(payload.get("action_details") or [])
        lines = [f"platform: {payload.get('name')}"]
        if payload.get("action_pack"):
            lines.append(f"action pack: {payload['action_pack']}")
        if action_names:
            lines.append("registered actions:")
            details_by_name = {str(item.get("name")): item for item in action_details}
            for name in action_names:
                detail = details_by_name.get(name, {})
                description = str(detail.get("description") or "").strip()
                suffix = f" — {description}" if description else ""
                lines.append(f"  - {name}{suffix}")
        else:
            lines.append("registered actions: none")
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


# ══════════════════════════════════════════════════════════════════════
# Unified command_execute: dispatches any engine-routed slash command
# ══════════════════════════════════════════════════════════════════════


async def command_execute(ctx: "Context", name: str, args: str = "") -> dict[str, Any]:
    """Execute a slash command and return a serializable result payload.

    This is the unified entry point for daemon-mode command execution.
    Returns a dict with at minimum ``ok`` and ``message`` keys, plus
    optional structured data for rich TUI rendering.
    """
    if name == "status":
        return build_status_payload(ctx)
    if name == "tools":
        return build_tools_payload(ctx)
    if name == "usage":
        return build_usage_payload(ctx)
    if name == "model":
        return build_model_payload(ctx, args)
    if name == "config":
        return build_config_payload(ctx, args)
    if name == "gateway":
        return build_gateway_payload(ctx)
    if name == "host":
        return await _execute_host(ctx, args)
    if _is_app_command_name(name):
        app_args = name[len("app"):].strip()
        if app_args:
            app_args = app_args + (" " + args if args else "")
        else:
            app_args = args
        return await build_app_payload(ctx, app_args)
    if name.startswith("teach") or name == "annotate":
        return await _execute_teach(ctx, name, args)
    if name.startswith("skills"):
        return _execute_skills(ctx, name, args)
    if name.startswith("hub"):
        return await _execute_hub(ctx, name, args)
    if name == "run":
        return {"ok": True, "stream": True, "prompt": args}
    if name == "arm":
        return await _execute_scheduler_arm(ctx, args)
    if name == "tasks":
        return _execute_scheduler_tasks(ctx)
    return {"ok": False, "message": f"Unknown command: /{name}"}


def _is_app_command_name(name: str) -> bool:
    return name == "app" or name.startswith("app ")


def build_status_payload(ctx: "Context") -> dict[str, Any]:
    """Build a serializable status summary."""
    engine = ctx.engine
    context_length = 0
    context_used = 0
    turn_count = 0
    if engine is not None:
        cap_registry = getattr(engine, "model_capabilities", None)
        if cap_registry is not None:
            caps = cap_registry.resolve(ctx.settings.llm_model)
            context_length = int(caps.context_length)
        context_used = int(getattr(engine, "context_token_count", 0))
        turn_count = int(getattr(engine, "turn_count", 0))

    platform_status = "connected" if (hasattr(ctx.rpc, "connected") and ctx.rpc.connected) else "mock"
    cwd = os.getcwd().replace(os.path.expanduser("~"), "~")

    from leapflow.engine.session import SessionMode
    mode = "idle"
    if ctx.session:
        if ctx.session.mode == SessionMode.LEARNING:
            mode = "learning"
        elif ctx.session.mode == SessionMode.EXECUTING:
            mode = "executing"

    gateway_connected: list[str] = []
    gw = getattr(ctx, "gateway_server", None)
    if gw is not None:
        statuses = gw.platform_status()
        for s in statuses:
            if s.connected:
                m = gw.manifests.get(s.platform_id)
                gateway_connected.append(m.display_name if m else s.platform_id)

    return {
        "ok": True,
        "view": "status",
        "model": ctx.settings.llm_model,
        "context_length": context_length,
        "context_used": context_used,
        "turn_count": turn_count,
        "platform": platform_status,
        "cwd": cwd,
        "config_path": str(ctx.settings.profile_layout.config_dir).replace(os.path.expanduser("~"), "~"),
        "user_config_path": str(ctx.settings.layout.user_config_path).replace(os.path.expanduser("~"), "~"),
        "workspace_config_path": str(ctx.settings.workspace_root / ".leapflow" / "config.yaml").replace(os.path.expanduser("~"), "~"),
        "session_id": getattr(ctx.session, "session_id", "") if ctx.session else "",
        "mode": mode,
        "gateway_connected": gateway_connected,
    }


def build_gateway_payload(ctx: "Context") -> dict[str, Any]:
    """Build a serializable gateway status payload."""
    import time as _time

    gw = getattr(ctx, "gateway_server", None)
    if gw is None:
        return {"ok": False, "message": "Gateway not initialised."}

    statuses = gw.platform_status()
    if not statuses:
        return {"ok": True, "view": "gateway", "connected": [], "configured": [], "available": []}

    connected = []
    configured = []
    available = []
    for s in statuses:
        m = gw.manifests.get(s.platform_id)
        name = m.display_name if m else s.platform_id
        if s.connected:
            uptime = ""
            if s.connected_since > 0:
                secs = int(_time.time() - s.connected_since)
                if secs < 60:
                    uptime = f"{secs}s"
                elif secs < 3600:
                    uptime = f"{secs // 60}m"
                else:
                    uptime = f"{secs // 3600}h {(secs % 3600) // 60}m"
            connected.append({"name": name, "id": s.platform_id, "uptime": uptime})
        elif s.error:
            configured.append({"name": name, "id": s.platform_id})
        else:
            available.append({"name": name, "id": s.platform_id})
    return {"ok": True, "view": "gateway", "connected": connected, "configured": configured, "available": available}


async def _execute_host(ctx: "Context", args: str) -> dict[str, Any]:
    """Execute /host command."""
    action = args.strip().lower() if args.strip() else "status"
    if action not in {"status", "start", "stop", "restart"}:
        return {"ok": False, "message": "Usage: /host [status|start|stop|restart]"}
    if action == "status":
        result = await ctx.host_backend_status()
    elif action == "start":
        result = await ctx.host_backend_start()
    elif action == "stop":
        result = await ctx.host_backend_stop()
    else:
        result = await ctx.host_backend_restart()
    return {"ok": True, "view": "host", "action": action, "result": result}


async def _distill_background(session) -> None:
    """Run distillation in background without blocking the RPC response."""
    import logging
    _log = logging.getLogger(__name__)
    try:
        final = await session.await_learning()
        if final and final.candidates:
            _log.info(
                "background_distill: %d candidates, activated=%s",
                len(final.candidates),
                final.activated_skill_names or [],
            )
        else:
            _log.info("background_distill: no candidates produced")
    except Exception:
        _log.warning("background_distill failed", exc_info=True)


async def _execute_teach(ctx: "Context", name: str, args: str) -> dict[str, Any]:
    """Execute teach commands.

    Returns ``session_mode`` in the payload so the TUI client can track
    whether it should route subsequent inputs as annotations.
    """
    from leapflow.engine.session import SessionMode

    full_cmd = name + (" " + args if args else "")
    if full_cmd in ("teach start", "teach") or full_cmd.startswith("teach start "):
        if ctx.session is None:
            return {"ok": False, "message": "No active session.", "session_mode": "idle"}
        if ctx.session.mode == SessionMode.LEARNING:
            return {"ok": False, "message": "Already in teaching mode. Say '/teach stop' to end.", "session_mode": "learning"}
        goal = args if name == "teach start" else ""
        try:
            session = await ctx.session.enter_learning(goal=goal)
            msg = f"Teaching started — session {session.session_id}"
            if goal:
                msg += f"\nGoal: {goal}"
            msg += "\nCommands: /teach stop │ /teach discard │ /teach pause │ /teach resume │ /teach skip [n] │ /annotate <text>"
            return {"ok": True, "message": msg, "session_mode": "learning"}
        except Exception as e:
            return {"ok": False, "message": str(e)}

    if name == "teach stop":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode.", "session_mode": "idle"}
        try:
            result = await ctx.session.exit_learning()
            msg = f"Recording stopped — {result.step_count} steps, {result.duration:.1f}s"
            if result.step_count == 0:
                msg += "\nNo steps recorded — nothing to distill."
            elif not getattr(ctx.settings, "has_llm_credentials", False):
                msg += "\nNo LLM credentials — distillation skipped."
            elif ctx.session.has_pending_distillation:
                import asyncio
                asyncio.create_task(_distill_background(ctx.session))
                msg += "\nDistillation started in background."
            else:
                report = getattr(result, "learnability_report", None)
                reason = getattr(report, "reason", "assessment decided to skip") if report else "auto-learn disabled"
                msg += f"\nDistillation skipped: {reason}"
            return {
                "ok": True,
                "message": msg,
                "step_count": result.step_count,
                "duration": result.duration,
                "session_mode": "idle",
            }
        except Exception as e:
            return {"ok": False, "message": str(e), "session_mode": "idle"}

    if name == "teach pause":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode."}
        ctx.session.pause_learning()
        return {"ok": True, "message": "Teaching paused.", "session_mode": "paused"}

    if name == "teach resume":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode."}
        ctx.session.resume_learning()
        return {"ok": True, "message": "Teaching resumed.", "session_mode": "learning"}

    if name == "teach discard":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode."}
        ctx.session.discard_learning()
        return {"ok": True, "message": "Recording discarded.", "session_mode": "idle"}

    if name == "teach skip":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode."}
        n = 1
        if args.strip().isdigit():
            n = int(args.strip())
        ctx.session.skip_steps(n)
        return {"ok": True, "message": f"Marked last {n} step(s) as noise.", "session_mode": "learning"}

    if name == "annotate":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            return {"ok": False, "message": "Not in teaching mode."}
        if not args.strip():
            return {"ok": False, "message": "Usage: /annotate <text>"}
        ctx.session.annotate(args.strip())
        return {"ok": True, "message": f"Annotation recorded: {args.strip()}", "session_mode": "learning"}

    if name == "teach status":
        if not ctx.session:
            return {"ok": True, "message": "No active session.", "session_mode": "idle"}
        mode = ctx.session.mode.value
        result_payload: dict[str, Any] = {"ok": True, "session_mode": mode}
        if mode == "evolving" or ctx.session.is_distilling:
            result_payload["message"] = "Distillation in progress…"
            result_payload["distilling"] = True
        elif mode == "learning":
            step_count = ctx.session.recording_step_count
            result_payload["message"] = f"Recording — {step_count} steps captured."
            result_payload["step_count"] = step_count
        else:
            last = ctx.session.last_result
            if last:
                candidates = getattr(last, "candidates", None) or []
                new_skills = getattr(last, "new_skills", None) or []
                result_payload["message"] = (
                    f"Idle. Last result: {last.step_count} steps, "
                    f"{len(candidates)} candidates, "
                    f"{len(new_skills)} activated."
                )
            else:
                result_payload["message"] = "Idle. No previous distillation results."
        return result_payload

    return {"ok": False, "message": f"Unknown teach command: /{full_cmd}"}


def _execute_skills(ctx: "Context", name: str, args: str) -> dict[str, Any]:
    """Execute skills commands."""
    full_cmd = name + (" " + args if args else "")
    if full_cmd in ("skills", "skills list"):
        skills = ctx.registry.list_all() if ctx.registry else []
        if not skills:
            return {"ok": True, "view": "skills_list", "skills": []}
        entries = []
        for s in skills:
            m = s.metadata
            entries.append({
                "name": s.name,
                "version": m.version,
                "confidence": m.confidence,
                "description": s.description[:80],
            })
        return {"ok": True, "view": "skills_list", "skills": entries}

    if name == "skills show":
        skill_name = args.strip()
        if not skill_name:
            return {"ok": False, "message": "Usage: /skills show <name>"}
        skill = ctx.registry.get(skill_name) if ctx.registry else None
        if skill is None:
            return {"ok": False, "message": f"Skill '{skill_name}' not found."}
        m = skill.metadata
        return {
            "ok": True,
            "view": "skills_show",
            "name": skill.name,
            "description": skill.description,
            "version": m.version,
            "confidence": m.confidence,
            "triggers": list(skill.triggers) if skill.triggers else [],
        }

    if name == "skills disable":
        skill_name = args.strip()
        if not skill_name:
            return {"ok": False, "message": "Usage: /skills disable <name>"}
        found = False
        if ctx.skill_lib and ctx.skill_lib.deactivate_parameterized(skill_name):
            found = True
        if ctx.registry and ctx.registry.unregister(skill_name):
            found = True
        if found:
            return {"ok": True, "message": f"Skill '{skill_name}' disabled."}
        return {"ok": False, "message": f"Skill '{skill_name}' not found."}

    if name == "skills delete":
        skill_name = args.strip()
        if not skill_name:
            return {"ok": False, "message": "Usage: /skills delete <name>"}
        found = False
        if ctx.skill_lib:
            stored = ctx.skill_lib.load_skill_by_title(skill_name)
            if stored:
                stored.status = "deleted"
                ctx.skill_lib.update_skill(stored)
                found = True
        if ctx.registry and ctx.registry.unregister(skill_name):
            found = True
        if found:
            return {"ok": True, "message": f"Skill '{skill_name}' deleted."}
        return {"ok": False, "message": f"Skill '{skill_name}' not found."}

    return {"ok": False, "message": f"Unknown skills command: /{full_cmd}"}


async def _execute_hub(ctx: "Context", name: str, args: str) -> dict[str, Any]:
    """Execute hub commands."""
    try:
        from leapflow.cli.commands.hub import cmd_hub_payload
        return await cmd_hub_payload(ctx, name, args)
    except ImportError:
        pass
    # Fallback: basic hub dispatch
    sub = name[len("hub"):].strip() if name.startswith("hub") else ""
    if not sub:
        sub = args.split()[0] if args.split() else ""
        args = " ".join(args.split()[1:])
    command_parts = ["/hub"]
    if sub:
        command_parts.append(sub)
    if args:
        command_parts.append(args)
    command = " ".join(command_parts)
    return {"ok": False, "message": f"Hub command '{command}' is not yet implemented in this runtime."}


def _execute_scheduler_tasks(ctx: "Context") -> dict[str, Any]:
    """Execute /tasks command."""
    scheduler = getattr(ctx, "scheduler", None)
    if scheduler is None:
        return {"ok": True, "view": "tasks", "tasks": [], "message": "No scheduler active."}
    tasks = scheduler.list_tasks() if hasattr(scheduler, "list_tasks") else []
    entries = [
        {"name": t.name, "schedule": t.schedule, "next_run": str(getattr(t, "next_run", ""))}
        for t in tasks
    ]
    return {"ok": True, "view": "tasks", "tasks": entries}


async def _execute_scheduler_arm(ctx: "Context", args: str) -> dict[str, Any]:
    """Execute /arm command."""
    scheduler = getattr(ctx, "scheduler", None)
    if scheduler is None:
        return {"ok": False, "message": "Scheduler not active in this session."}
    tokens = args.strip().split(None, 1)
    if len(tokens) < 2:
        return {"ok": False, "message": "Usage: /arm <skill> <cron>"}
    skill_name, cron_expr = tokens
    try:
        task_id = await scheduler.arm(skill_name, cron_expr)
        return {"ok": True, "message": f"Armed: {skill_name} → {cron_expr} (id={task_id})"}
    except Exception as e:
        return {"ok": False, "message": str(e)}


def render_command_payload(console: "LeapConsole", payload: dict[str, Any]) -> None:
    """Render a generic command_execute result payload in the TUI."""
    if not payload.get("ok"):
        console.warning(str(payload.get("message") or payload.get("error") or "Command failed."))
        return

    view = str(payload.get("view") or "")

    if view == "status":
        _render_status_view(console, payload)
        return
    if view == "model":
        render_model_payload(console, payload)
        return
    if view == "config":
        render_config_payload(console, payload)
        return
    if view == "gateway":
        _render_gateway_view(console, payload)
        return
    if view == "host":
        _render_host_view(console, payload)
        return
    if view == "skills_list":
        _render_skills_list_view(console, payload)
        return
    if view == "skills_show":
        _render_skills_show_view(console, payload)
        return
    if view == "tasks":
        _render_tasks_view(console, payload)
        return

    msg = payload.get("message")
    if msg:
        console.success(str(msg))


def _render_status_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.text import Text

    info = Text()
    info.append("Model:     ", style="dim")
    info.append(f"{payload.get('model')}\n", style="bold")

    ctx_len = int(payload.get("context_length") or 0)
    ctx_used = int(payload.get("context_used") or 0)
    turn_count = int(payload.get("turn_count") or 0)
    if ctx_len:
        pct = int(ctx_used * 100 / ctx_len) if ctx_len else 0
        info.append("Context:   ", style="dim")
        pct_style = "bold red" if pct >= 90 else ("yellow" if pct >= 75 else "")
        info.append(f"{ctx_used:,} / {ctx_len:,} ({pct}%)\n", style=pct_style)
    info.append("Turns:     ", style="dim")
    info.append(f"{turn_count}\n")
    info.append("Platform:  ", style="dim")
    p_status = str(payload.get("platform") or "")
    p_style = "green" if p_status == "connected" else "dim"
    info.append(f"{p_status}\n", style=p_style)
    info.append("CWD:       ", style="dim")
    info.append(f"{payload.get('cwd')}\n")
    info.append("Config:    ", style="dim")
    info.append(f"{payload.get('config_path')}\n")
    session_id = payload.get("session_id")
    if session_id:
        info.append("Session:   ", style="dim")
        info.append(f"{session_id}\n")
    info.append("Mode:      ", style="dim")
    info.append(f"{payload.get('mode')}\n")
    gw = payload.get("gateway_connected") or []
    if gw:
        info.append("Gateway:   ", style="dim")
        info.append(f"{', '.join(gw)}\n", style="green")

    console.print(Panel(info, title="[bold cyan]LeapFlow Status[/]", border_style="bright_black", padding=(0, 2)))


def _render_gateway_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.text import Text

    info = Text()
    connected = payload.get("connected") or []
    configured = payload.get("configured") or []
    available = payload.get("available") or []

    if connected:
        info.append("Connected\n", style="bold green")
        for entry in connected:
            uptime = f" ({entry['uptime']})" if entry.get("uptime") else ""
            info.append(f"  ● {entry['name']}{uptime}\n", style="green")
    if configured:
        info.append("Configured (not connected)\n", style="bold yellow")
        for entry in configured:
            info.append(f"  ○ {entry['name']}\n", style="yellow")
    if available:
        info.append("Available\n", style="bold dim")
        names = [entry["name"] for entry in available]
        info.append(f"  {', '.join(names)}\n", style="dim")
    info.append("\n", style="dim")
    info.append('Say "connect to <platform>" to set up a new integration.', style="dim italic")
    console.print(Panel(info, title="[bold cyan]Gateway[/]", border_style="bright_black", padding=(0, 2)))


def _render_host_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    result = payload.get("result") or {}
    action = payload.get("action") or "status"
    if action != "status":
        console.success(f"Host {action} completed.")
    lines = []
    for key in ("status", "backend", "pid", "session_id"):
        if key in result:
            lines.append(f"  {key}: {result[key]}")
    if lines:
        console.system("\n".join(lines))


def _render_skills_list_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    from rich.table import Table

    skills = payload.get("skills") or []
    if not skills:
        console.system("No skills registered.")
        return
    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Name", style="cyan", max_width=30)
    table.add_column("Version", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Description", max_width=40)
    for s in skills:
        table.add_row(
            str(s.get("name") or ""),
            f"v{s.get('version', 0)}",
            f"{float(s.get('confidence') or 0):.0%}",
            str(s.get("description") or "")[:40],
        )
    console.print(table)


def _render_skills_show_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    from rich.panel import Panel
    from rich.text import Text

    info = Text()
    info.append(f"Name:        {payload.get('name')}\n")
    info.append(f"Description: {payload.get('description')}\n")
    info.append(f"Version:     v{payload.get('version', 0)}\n")
    info.append(f"Confidence:  {float(payload.get('confidence') or 0):.0%}\n")
    triggers = payload.get("triggers") or []
    if triggers:
        info.append(f"Triggers:    {', '.join(triggers)}")
    console.print(Panel(info, title=str(payload.get("name") or "Skill"), border_style="cyan"))


def _render_tasks_view(console: "LeapConsole", payload: dict[str, Any]) -> None:
    tasks = payload.get("tasks") or []
    msg = payload.get("message")
    if msg:
        console.system(str(msg))
        return
    if not tasks:
        console.system("No scheduled tasks.")
        return
    from rich.table import Table
    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Schedule")
    table.add_column("Next Run")
    for t in tasks:
        table.add_row(str(t.get("name") or ""), str(t.get("schedule") or ""), str(t.get("next_run") or ""))
    console.print(table)

"""Gateway tools for the agent — configuration AND messaging.

Two tools:
- ``gateway_connect``: conversational platform configuration
- ``gateway_send``: proactive outbound messaging to connected platforms

SECURITY: Tool results NEVER contain credential values.
Credentials flow: user message → LLM parse → tool handler → vault.
Tool returns only status information back to LLM.
"""
from __future__ import annotations

import hashlib
import json as _json
import logging
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

_gateway_server_ref: Any = None
_approval_gate: Any = None
_pending_app_onboarding: Dict[str, Any] | None = None

# Task-scoped side-effect dedup: tracks completed (platform, action, payload)
# fingerprints within the current user turn. Reset at each engine.run() via
# reset_platform_action_scope(). Prevents cross-turn duplicate execution of
# send/write/execute actions regardless of LLM behavior.
_task_completed_actions: Dict[str, Dict[str, Any]] = {}

_SIDE_EFFECT_KINDS = frozenset({"send", "write", "execute"})


def reset_platform_action_scope() -> None:
    """Reset the task-scoped action dedup state. Called at each turn boundary."""
    _task_completed_actions.clear()


def _action_fingerprint(platform: str, action: str, payload: Mapping[str, Any]) -> str:
    """Stable fingerprint for dedup: hash of (platform, action, sorted payload)."""
    raw = f"{platform}:{action}:{_json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

# Single source of truth for the platform_connect management namespace. Both
# the platform_connect tool schema enum and the platform_action namespace
# guard below are generated from this tuple so the two never drift apart.
PLATFORM_CONNECT_ACTIONS: tuple[str, ...] = (
    "list",
    "guide",
    "preflight",
    "connect",
    "disconnect",
    "remove",
    "status",
    "events_start",
    "events_stop",
    "events_status",
)
_PLATFORM_CONNECT_ACTIONS = frozenset(PLATFORM_CONNECT_ACTIONS)


def set_gateway_server(server: Any) -> None:
    """Install ``GatewayServer`` reference for tool dispatch (late-bound)."""
    global _gateway_server_ref, _pending_app_onboarding
    if server is not _gateway_server_ref:
        _pending_app_onboarding = None
    _gateway_server_ref = server


def set_gateway_approval_gate(gate: Any) -> None:
    """Install approval gate for outbound messaging (late-bound)."""
    global _approval_gate
    _approval_gate = gate


def build_app_connector_prompt_section() -> str:
    """Build a compact App Connector capability index for the current prompt.

    This function does not classify user text. It only exposes the currently
    supported platform manifests so the LLM can use its normal language
    understanding in the same turn and choose the right connector tool.
    """
    if _gateway_server_ref is None:
        return ""
    manifests = getattr(_gateway_server_ref, "manifests", {}) or {}
    if not manifests:
        return ""

    lines = [
        "\n## App Connector Capability Index",
        "LeapFlow can onboard and manage external apps through `platform_connect` and execute exact registered business actions through `platform_action`.",
        "For requests about connecting, setting up, configuring, enabling, or managing a supported app, use `platform_connect` first instead of generating SDK/Webhook sample code.",
        f"`platform_connect.action` is the management namespace: {', '.join(PLATFORM_CONNECT_ACTIONS)}.",
        "`platform_action.action` is only for exact registered platform business actions listed below, such as `im.send_message`; never use management actions like `list` or `guide` there.",
        "All business fields MUST go inside `payload`; top-level keys are only `platform`, `action`, and `payload`.",
        "Do not invent platform IDs or platform action names. If the needed action is not listed, ask for discovery/clarification instead of guessing.",
        "Use `platform_connect` with `action='guide'` and the matching `platform` to start onboarding; use `action='list'` when the app is unclear.",
        "When a pending onboarding state is present, continue from that state with `platform_connect` instead of asking the user to restate the app.",
        "Supported apps:",
    ]
    for platform_id, manifest in sorted(manifests.items()):
        backend = dict(getattr(manifest, "backend", {}) or {})
        backend_kind = str(backend.get("kind") or "adapter")
        action_summaries = _platform_action_summaries(platform_id)
        if action_summaries:
            lines.append(
                f"- `{platform_id}`: {getattr(manifest, 'display_name', platform_id)} "
                f"(category={getattr(manifest, 'category', '')}; backend={backend_kind})"
            )
            for item in action_summaries:
                payload_sig = _format_payload_signature(item)
                lines.append(f"  - `{item['name']}` {payload_sig}")
        else:
            lines.append(
                f"- `{platform_id}`: {getattr(manifest, 'display_name', platform_id)} "
                f"(category={getattr(manifest, 'category', '')}; backend={backend_kind}; platform_action actions=none registered)"
            )
    if _pending_app_onboarding:
        state = dict(_pending_app_onboarding)
        next_actions = state.get("next_actions") or []
        next_text = "; next=" + " | ".join(str(item) for item in next_actions[:3]) if next_actions else ""
        lines.extend([
            "",
            "Pending App Onboarding State:",
            (
                f"- platform=`{state.get('platform_id', '')}`; stage={state.get('stage', 'unknown')}; "
                f"backend={state.get('backend_kind', 'unknown')}; recoverable={state.get('recoverable', True)}{next_text}"
            ),
            "Use this state to continue app onboarding; do not ask the user to repeat known platform details.",
        ])
    return "\n".join(lines) + "\n"


def _format_payload_signature(summary: Dict[str, Any]) -> str:
    """Format a compact payload signature for the capability index.

    Example output: ``payload={chat_id*, text*} [send/high]``
    """
    required_set = frozenset(summary.get("required") or ())
    fields = summary.get("fields") or []
    parts: list[str] = []
    for f in fields:
        parts.append(f"{f}*" if f in required_set else f)
    payload_str = f"payload={{{', '.join(parts)}}}" if parts else "payload={}"
    effect = summary.get("effect") or ""
    risk = summary.get("risk_level") or ""
    meta = "/".join(filter(None, [effect, risk]))
    return f"{payload_str} [{meta}]" if meta else payload_str


def _manifest_action_specs(platform: str) -> Mapping[str, Any]:
    """Return action-pack specs declared by a platform manifest."""
    manifest = _gateway_server_ref.manifests.get(platform) if _gateway_server_ref is not None else None
    if manifest is None:
        return {}
    actions = dict(getattr(manifest, "actions", {}) or {})
    pack = str(actions.get("pack") or "")
    if not pack:
        return {}
    try:
        from leapflow.gateway.connectors.action_registry import ActionRegistry

        return ActionRegistry.from_module(pack).all()
    except (ImportError, AttributeError, ValueError):
        logger.debug("platform.action_pack_lookup_failed platform=%s", platform, exc_info=True)
        return {}


def _compact_action_summary(spec: Any) -> Dict[str, Any]:
    """Return a compact, LLM-readable platform action contract entry."""
    schema = getattr(spec, "schema", {}) or {}
    required = schema.get("required") or () if isinstance(schema, Mapping) else ()
    properties = schema.get("properties") or {} if isinstance(schema, Mapping) else {}
    return {
        "name": str(getattr(spec, "name", "") or ""),
        "description": str(getattr(spec, "description", "") or ""),
        "effect": str(getattr(spec, "effect", "") or ""),
        "risk_level": str(getattr(spec, "risk_level", "") or ""),
        "required": [str(item) for item in required],
        "fields": [str(key) for key in properties.keys()] if isinstance(properties, Mapping) else [],
    }


def _platform_action_summaries(platform: str) -> List[Dict[str, Any]]:
    """Return registered platform action summaries sorted by action name."""
    specs = _manifest_action_specs(platform)
    return [
        _compact_action_summary(spec)
        for _name, spec in sorted(specs.items())
    ]


def platform_action_capability_summary(platform: str) -> List[Dict[str, Any]]:
    """Public accessor for registered platform action summaries.

    Used by slash commands and other callers that need to display the exact
    business action contract without reaching into module-private state.
    """
    return _platform_action_summaries(platform)


# ═══════════════════════════════════════════════════════════════
# Tool handler
# ═══════════════════════════════════════════════════════════════

async def gateway_connect_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle ``gateway_connect`` tool calls from the agent.

    Actions:
        list       — show available and connected platforms
        guide      — get setup instructions for a platform
        connect    — connect with provided credentials
        disconnect — disconnect a platform
        status     — check connection health
    """
    if _gateway_server_ref is None:
        return {"ok": False, "error": "Gateway not initialised"}

    action = params.get("action", "list")
    platform = params.get("platform", "")

    dispatch = {
        "list": _action_list,
        "guide": _action_guide,
        "connect": _action_connect,
        "disconnect": _action_disconnect,
        "remove": _action_remove,
        "status": _action_status,
    }
    handler = dispatch.get(action)
    if handler is None:
        return {"ok": False, "error": f"Unknown action: {action}"}

    return await handler(platform, params)


async def platform_connect_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle platform connection management across backend kinds."""
    if _gateway_server_ref is None:
        return {"ok": False, "error": "Gateway not initialised"}
    action = params.get("action", "list")
    platform = params.get("platform", "")
    if action == "connect":
        if not platform:
            return {"ok": False, "error": "platform is required"}
        return await _action_connect(platform, params)
    if action == "preflight":
        if not platform:
            return {"ok": False, "error": "platform is required"}
        return await _action_preflight(platform, params)
    if action == "events_start":
        if not platform:
            return {"ok": False, "error": "platform is required"}
        return await _gateway_server_ref.start_platform_events(
            platform,
            checkpoint=str(params.get("checkpoint") or ""),
        )
    if action == "events_stop":
        if not platform:
            return {"ok": False, "error": "platform is required"}
        return await _gateway_server_ref.stop_platform_events(platform)
    if action == "events_status":
        if not platform:
            return {"ok": False, "error": "platform is required"}
        return await _gateway_server_ref.platform_event_status(platform)
    if action in {"list", "guide", "disconnect", "remove", "status"}:
        return await gateway_connect_handler(params)
    return {"ok": False, "error": f"Unknown action: {action}"}


# ── Action implementations ───────────────────────────────────

async def _action_list(
    _platform: str, _params: Dict[str, Any],
) -> Dict[str, Any]:
    """List available platforms and their connection status."""
    import time

    statuses = _gateway_server_ref.platform_status()
    platforms = []
    for s in statuses:
        manifest = _gateway_server_ref.manifests.get(s.platform_id)
        state = "connected" if s.connected else (
            "configured" if s.error == "configured but not connected" else "available"
        )
        entry: Dict[str, Any] = {
            "id": s.platform_id,
            "name": manifest.display_name if manifest else s.platform_id,
            "state": state,
            "category": manifest.category if manifest else "",
        }
        if s.connected and s.connected_since > 0:
            uptime_s = int(time.time() - s.connected_since)
            entry["uptime"] = _format_uptime(uptime_s)
        if getattr(s, "metadata", None):
            entry["diagnostics"] = dict(s.metadata)
        platforms.append(entry)
    return {"ok": True, "platforms": platforms}


def _format_uptime(seconds: int) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h {minutes}m"


async def _action_guide(
    platform: str, _params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return setup guide for a platform (no credentials exposed)."""
    manifest = _gateway_server_ref.manifests.get(platform)
    if manifest is None:
        available = list(_gateway_server_ref.manifests.keys())
        return {
            "ok": False,
            "error": f"Unknown platform: {platform}.  Available: {available}",
        }

    fields = [
        {
            "key": c.key,
            "label": c.label,
            "required": c.required,
            "help": c.help_zh or c.help_en,
        }
        for c in manifest.credentials
    ]
    options = [
        {
            "key": o.key,
            "label": o.label,
            "default": o.default,
            "choices": list(o.choices),
            "help": o.help_zh or o.help_en,
        }
        for o in manifest.options
        if not o.advanced
    ]

    result: Dict[str, Any] = {
        "ok": True,
        "platform": manifest.display_name,
        "setup_guide": (
            manifest.setup_guide.summary_zh or manifest.setup_guide.summary_en
        ),
        "required_fields": fields,
        "console_url": manifest.setup_guide.console_url,
    }

    if manifest.setup_guide.steps_zh:
        result["setup_steps"] = list(manifest.setup_guide.steps_zh)
    elif manifest.setup_guide.steps_en:
        result["setup_steps"] = list(manifest.setup_guide.steps_en)

    if options:
        result["optional_settings"] = options

    preflight_checks = _manifest_preflight_checks(manifest)
    if preflight_checks:
        result["preflight_checks"] = preflight_checks
        preflight = await _run_manifest_preflight(platform, manifest, _params)
        result["preflight_result"] = preflight
        result["onboarding_state"] = _remember_app_onboarding(platform, manifest, preflight)
        result["recovery_hint"] = preflight.get("recovery_hint") or "Complete the preflight checks first; LeapFlow will then reuse the authorized backend profile."
        if preflight.get("next_steps"):
            result["next_steps"] = preflight["next_steps"]

    result["setup_form"] = {
        "fields": [
            {
                "key": c.key,
                "label": c.label,
                "type": "password" if c.secret else "text",
                "required": c.required,
            }
            for c in manifest.credentials
        ],
        "console_url": manifest.setup_guide.console_url,
        "backend": dict(manifest.backend),
        "actions": dict(manifest.actions),
    }

    required = [c for c in manifest.credentials if c.required]
    if required:
        labels = " / ".join(c.label for c in required)
        result["prompt_hint"] = (
            f"Present the setup steps above, then ask the user to provide "
            f"{labels} in a single reply.  Do NOT repeat credential values "
            f"in your response."
        )
    else:
        result["prompt_hint"] = (
            "Present the setup steps and current preflight result. If all safe checks are ready, "
            "call platform_connect with action='connect'. If a preflight step requires installation, "
            "authorization, or credentials, ask only for that missing user action and avoid repeating known details."
        )

    return result


async def _action_connect(
    platform: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Connect a platform with provided credentials.

    SECURITY: credentials flow into ``connect_platform()`` which encrypts
    and persists them.  Return value contains **only** status.
    """
    manifest = _gateway_server_ref.manifests.get(platform)
    credentials = params.get("credentials", {})
    options = params.get("options", {})

    if not platform:
        return {"ok": False, "error": "Platform ID is required"}
    if manifest is None:
        return {"ok": False, "error": f"Unknown platform: {platform}"}
    has_required_credentials = any(c.required for c in manifest.credentials)
    if has_required_credentials and not credentials:
        return await _action_guide(platform, params)

    result = await _gateway_server_ref.connect_platform(
        platform, credentials, options,
    )
    return _decorate_app_connection_result(platform, manifest, result)


async def _action_preflight(
    platform: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run safe manifest-driven preflight checks for a platform."""
    manifest = _gateway_server_ref.manifests.get(platform)
    if not platform:
        return {"ok": False, "error": "Platform ID is required"}
    if manifest is None:
        return {"ok": False, "error": f"Unknown platform: {platform}"}
    preflight = await _run_manifest_preflight(platform, manifest, params)
    state = _remember_app_onboarding(platform, manifest, preflight)
    return {
        "ok": bool(preflight.get("ready")),
        "platform": platform,
        "stage": state.get("stage"),
        "preflight_result": preflight,
        "onboarding_state": state,
        "recovery_hint": preflight.get("recovery_hint", ""),
        "next_steps": list(preflight.get("next_steps") or ()),
    }


def _decorate_app_connection_result(platform: str, manifest: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    response = dict(result)
    if response.get("ok"):
        response["onboarding_state"] = _remember_app_onboarding(
            platform,
            manifest,
            {
                "ready": True,
                "stage": "connected",
                "backend_kind": _manifest_backend_kind(manifest),
                "recoverable": False,
                "next_steps": [],
            },
        )
        return response

    diagnostics = dict(response.get("diagnostics") or response.get("metadata") or {})
    stage = _stage_from_backend_metadata(diagnostics, default="connect_failed")
    preflight = {
        "ready": False,
        "stage": stage,
        "backend_kind": _manifest_backend_kind(manifest),
        "recoverable": bool(diagnostics.get("recoverable", True)),
        "recovery_hint": response.get("recovery_hint") or diagnostics.get("recovery_hint") or response.get("error", ""),
        "next_steps": response.get("next_steps") or diagnostics.get("next_steps") or [],
        "metadata": diagnostics,
    }
    response["onboarding_state"] = _remember_app_onboarding(platform, manifest, preflight)
    response.setdefault("recovery_hint", preflight["recovery_hint"])
    if preflight["next_steps"]:
        response.setdefault("next_steps", preflight["next_steps"])
    return response


def _manifest_backend_kind(manifest: Any) -> str:
    backend = dict(getattr(manifest, "backend", {}) or {})
    return str(backend.get("kind") or "adapter")


def _manifest_option_default(manifest: Any, key: str, fallback: Any = "") -> Any:
    for option in getattr(manifest, "options", ()) or ():
        if str(getattr(option, "key", "")) == key:
            value = getattr(option, "default", fallback)
            return fallback if value is None else value
    return fallback


def _stage_from_backend_metadata(metadata: Dict[str, Any], *, default: str) -> str:
    detail = str(metadata.get("detail") or metadata.get("error") or metadata.get("last_error") or "").lower()
    auth_status = str(metadata.get("auth_status") or "").lower()
    failure_code = str(metadata.get("failure_code") or metadata.get("error_code") or "").lower()
    if failure_code == "cli_contract_mismatch":
        return "cli_contract_mismatch"
    if metadata.get("binary_path") or auth_status == "authorized":
        if auth_status == "authorized":
            return "auth_ready"
        if auth_status in {"not_ready", "unknown"}:
            return "auth_missing"
        return default
    if "binary not found" in detail or "not found" in detail or auth_status == "unknown":
        return "cli_missing"
    if "unauthor" in detail or "login" in detail or "token" in detail:
        return "auth_missing"
    return default


async def _run_manifest_preflight(platform: str, manifest: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    backend_kind = _manifest_backend_kind(manifest)
    if backend_kind != "cli":
        return {
            "ready": True,
            "stage": "ready",
            "backend_kind": backend_kind,
            "recoverable": False,
            "checks": [],
            "next_steps": [f"platform_connect action=connect platform={platform}"],
        }

    from leapflow.gateway.backends.cli_backend import CliBackend

    backend = dict(getattr(manifest, "backend", {}) or {})
    options = dict(params.get("options") or {})
    binary = str(options.get("binary") or backend.get("binary") or _manifest_option_default(manifest, "binary", "lark-cli"))
    profile = str(options.get("profile") or _manifest_option_default(manifest, "profile", ""))
    identity = str(options.get("identity") or _manifest_option_default(manifest, "identity", ""))
    status = await CliBackend(binary=binary, profile=profile, identity=identity, timeout_s=6.0).status()
    metadata = dict(status.metadata)
    stage = "auth_ready" if status.ok else _stage_from_backend_metadata(metadata, default="auth_missing")
    checks = _preflight_checks_with_status(_manifest_preflight_checks(manifest), stage)
    next_steps = list(metadata.get("next_steps") or [])
    if status.ok:
        next_steps = [f"platform_connect action=connect platform={platform}"]
    return {
        "ready": bool(status.ok),
        "stage": stage,
        "backend_kind": backend_kind,
        "recoverable": bool(metadata.get("recoverable", not status.ok)),
        "checks": checks,
        "detail": status.detail,
        "metadata": metadata,
        "recovery_hint": metadata.get("recovery_hint", "") if not status.ok else "CLI backend is authorized; continue with platform_connect action='connect'.",
        "next_steps": next_steps,
    }


def _preflight_checks_with_status(checks: list[Dict[str, Any]], stage: str) -> list[Dict[str, Any]]:
    decorated: list[Dict[str, Any]] = []
    for check in checks:
        item = dict(check)
        key = str(item.get("key") or "")
        if stage == "auth_ready":
            item["status"] = "passed"
        elif stage == "cli_missing":
            item["status"] = "failed" if key == "cli_installed" else "blocked"
        elif stage == "auth_missing":
            item["status"] = "passed" if key == "cli_installed" else ("failed" if key == "cli_status" else "blocked")
        elif stage == "cli_contract_mismatch":
            item["status"] = "passed" if key == "cli_installed" else ("failed" if key == "cli_status" else "blocked")
        else:
            item["status"] = "pending"
        decorated.append(item)
    return decorated


def _remember_app_onboarding(platform: str, manifest: Any, preflight: Dict[str, Any]) -> Dict[str, Any]:
    global _pending_app_onboarding
    state = {
        "platform_id": platform,
        "platform": getattr(manifest, "display_name", platform),
        "stage": preflight.get("stage") or "unknown",
        "backend_kind": preflight.get("backend_kind") or _manifest_backend_kind(manifest),
        "ready": bool(preflight.get("ready")),
        "recoverable": bool(preflight.get("recoverable", True)),
        "last_error": preflight.get("detail") or preflight.get("recovery_hint") or "",
        "next_actions": list(preflight.get("next_steps") or ()),
    }
    _pending_app_onboarding = state if state["stage"] != "connected" else None
    return state


async def _action_disconnect(
    platform: str, _params: Dict[str, Any],
) -> Dict[str, Any]:
    """Disconnect a platform (keeps saved credentials for reconnect)."""
    if not platform:
        return {"ok": False, "error": "Platform ID is required"}
    return await _gateway_server_ref.disconnect_platform(platform)


async def _action_remove(
    platform: str, _params: Dict[str, Any],
) -> Dict[str, Any]:
    """Disconnect AND delete saved credentials for a platform.

    Unlike ``disconnect``, this fully removes the platform configuration
    from ``gateway.yaml`` and the ``auto_connect`` list.
    """
    if not platform:
        return {"ok": False, "error": "Platform ID is required"}

    await _gateway_server_ref.disconnect_platform(platform)
    _gateway_server_ref.remove_platform_config(platform)
    return {"ok": True, "status": "removed"}


async def _action_status(
    platform: str, _params: Dict[str, Any],
) -> Dict[str, Any]:
    """Check status of a specific platform or all platforms."""
    import time

    statuses = _gateway_server_ref.platform_status()

    def _status_entry(s: Any) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "id": s.platform_id,
            "connected": s.connected,
        }
        if s.error:
            entry["error"] = s.error
        if s.connected and s.connected_since > 0:
            entry["uptime"] = _format_uptime(int(time.time() - s.connected_since))
        if getattr(s, "metadata", None):
            entry["diagnostics"] = dict(s.metadata)
            if s.metadata.get("recovery_hint"):
                entry["recovery_hint"] = s.metadata["recovery_hint"]
        return entry

    if platform:
        for s in statuses:
            if s.platform_id == platform:
                result = _status_entry(s)
                result["ok"] = True
                result["platform"] = result.pop("id")
                return result
        return {"ok": False, "error": f"Platform not found: {platform}"}
    return {"ok": True, "platforms": [_status_entry(s) for s in statuses]}


def _manifest_preflight_checks(manifest: Any) -> list[Dict[str, Any]]:
    backend = dict(getattr(manifest, "backend", {}) or {})
    if backend.get("kind") != "cli":
        return []
    binary = str(backend.get("binary") or "lark-cli")
    return [
        {
            "key": "cli_installed",
            "label": "Install CLI",
            "kind": "check",
            "auto_run": True,
            "requires_approval": False,
            "command": f"{binary} --version",
            "help": f"Ensure '{binary}' is installed and available on PATH.",
            "failure_code": "cli_missing",
        },
        {
            "key": "cli_authorized",
            "label": "Authorize profile",
            "kind": "interactive_auth",
            "auto_run": False,
            "requires_approval": True,
            "command": f"{binary} auth login --json",
            "help": "Authorize the selected CLI identity once in the official CLI.",
            "failure_code": "auth_missing",
        },
        {
            "key": "cli_status",
            "label": "Verify status",
            "kind": "check",
            "auto_run": True,
            "requires_approval": False,
            "command": f"{binary} auth status --json",
            "help": "Confirm the selected CLI profile is ready before connecting.",
            "failure_code": "auth_missing",
        },
    ]


def _manifest_action_spec(platform: str, action: str) -> Any:
    return _manifest_action_specs(platform).get(action)


def _platform_action_unavailable_response(platform: str, action: str) -> Dict[str, Any]:
    manifest = _gateway_server_ref.manifests.get(platform) if _gateway_server_ref is not None else None
    available_platforms = sorted((_gateway_server_ref.manifests or {}).keys()) if _gateway_server_ref is not None else []
    if manifest is None:
        return {
            "ok": False,
            "failure_code": "unknown_platform",
            "error": f"Unknown platform: {platform}",
            "available_platforms": available_platforms,
            "recovery_hint": "Use platform_connect action=list to inspect supported platform IDs before calling platform_action.",
            "retryable": True,
        }
    action_summaries = _platform_action_summaries(platform)
    action_names = [item["name"] for item in action_summaries if item.get("name")]
    if action in _PLATFORM_CONNECT_ACTIONS:
        return {
            "ok": False,
            "failure_code": "wrong_action_namespace",
            "error": f"'{action}' is a platform_connect management action, not a platform_action business action.",
            "platform": platform,
            "requested_action": action,
            "tool_namespace": "platform_action",
            "correct_tool": "platform_connect",
            "available_management_actions": sorted(_PLATFORM_CONNECT_ACTIONS),
            "available_actions": action_summaries,
            "available_action_names": action_names,
            "recovery_hint": f"Call platform_connect with platform='{platform}' and action='{action}', or choose one exact business action from available_action_names for platform_action.",
            "retryable": True,
        }
    if _manifest_action_spec(platform, action) is None:
        return {
            "ok": False,
            "failure_code": "unknown_platform_action",
            "error": f"Unknown platform action: {platform}.{action}",
            "platform": platform,
            "requested_action": action,
            "available_actions": action_summaries,
            "available_action_names": action_names,
            "recovery_hint": "Use exactly one registered platform_action action name from available_action_names; do not infer or invent action names from domains.",
            "retryable": True,
        }
    recovery_hint = "Connect the platform backend before running registered platform actions."
    next_steps = [
        f"platform_connect action=guide platform={platform}",
        f"platform_connect action=preflight platform={platform}",
        f"platform_connect action=connect platform={platform}",
    ]
    response: Dict[str, Any] = {
        "ok": False,
        "failure_code": "platform_not_connected",
        "error": f"Platform '{platform}' is not connected; registered action '{action}' cannot run yet.",
        "platform": platform,
        "requested_action": action,
        "available_actions": action_summaries,
        "available_action_names": action_names,
        "recovery_hint": recovery_hint,
        "next_steps": next_steps,
        "retryable": True,
    }
    if _pending_app_onboarding and _pending_app_onboarding.get("platform_id") == platform:
        response["onboarding_state"] = dict(_pending_app_onboarding)
    return response


def _structured_validation_error(
    spec: Any,
    validation: Any,
    raw_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a machine-readable validation error response with recovery metadata."""
    schema = getattr(spec, "schema", {}) or {}
    required = schema.get("required") or ()
    properties = schema.get("properties") or {}
    response: Dict[str, Any] = {
        "ok": False,
        "failure_code": validation.failure_code or "validation_failed",
        "error": validation.error,
        "action": getattr(spec, "name", ""),
        "retryable": True,
        "recovery_hint": validation.recovery_hint,
    }
    if validation.missing_fields:
        response["missing_fields"] = list(validation.missing_fields)
        response["field_paths"] = [f"payload.{f}" for f in validation.missing_fields]
    if validation.type_errors:
        response["type_errors"] = list(validation.type_errors)
    response["expected_schema"] = {
        "required": [str(f) for f in required],
        "fields": {
            str(k): str((v.get("description") or v.get("type") or ""))
            for k, v in properties.items()
            if isinstance(v, Mapping)
        },
    }
    return response


# ═══════════════════════════════════════════════════════════════
# gateway_send handler — proactive outbound messaging
# ═══════════════════════════════════════════════════════════════

async def gateway_send_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle ``gateway_send`` tool calls — send messages to connected platforms.

    Enables the agent to proactively message any connected platform
    conversation (e.g. post to a Feishu group, reply in a Telegram chat).

    First use per platform requires user approval (session-scoped).
    """
    if _gateway_server_ref is None:
        return {"ok": False, "error": "Gateway not initialised"}

    platform = params.get("platform", "")
    chat_id = params.get("chat_id", "")
    text = params.get("text", "")

    if not platform:
        return {"ok": False, "error": "platform is required"}
    if not chat_id:
        return {"ok": False, "error": "chat_id is required"}
    if not text:
        return {"ok": False, "error": "text is required"}

    spec = _gateway_server_ref.get_platform_action_spec(platform, "im.send_message")
    if spec is None:
        unavailable = _platform_action_unavailable_response(platform, "im.send_message")
        if unavailable.get("failure_code") != "unknown_platform_action":
            return unavailable
    else:
        delegated = await platform_action_handler({
            "platform": platform,
            "action": "im.send_message",
            "payload": {
                "chat_id": chat_id,
                "text": text,
                "thread_id": params.get("thread_id", ""),
            },
            "source_tool": "gateway_send",
        })
        if delegated.get("ok"):
            if delegated.get("resource_id") and "message_id" not in delegated:
                delegated["message_id"] = delegated["resource_id"]
            delegated.setdefault("source_tool", "gateway_send")
        return delegated

    if _approval_gate is not None:
        try:
            from leapflow.security.actions import ActionDescriptor
            from leapflow.security.approval import ApprovalDecision, ApprovalRequest

            action = ActionDescriptor.gateway_send(platform, chat_id, text, metadata={
                "thread_id": params.get("thread_id", ""),
            })
            if hasattr(_approval_gate, "evaluate"):
                result = await _approval_gate.evaluate(action)
                if not getattr(result, "approved", False):
                    error = str(getattr(result, "denial_message", "") or "Outbound message denied by approval gate")
                    return {"ok": False, "error": error}
            else:
                preview = text[:80] + ("…" if len(text) > 80 else "")
                decision = await _approval_gate.request_approval(ApprovalRequest(
                    category=action.kind,
                    detail=f"Send to {platform}/{chat_id}: {preview}",
                    risk_hint=0.5,
                    metadata={"platform": platform, "chat_id": chat_id},
                    action=action,
                ))
                if decision not in {
                    ApprovalDecision.ALLOW,
                    ApprovalDecision.ALLOW_ONCE,
                    ApprovalDecision.ALLOW_SESSION,
                    ApprovalDecision.ALLOW_ALWAYS,
                }:
                    return {"ok": False, "error": "Outbound message denied by approval gate"}
        except Exception:
            logger.debug("gateway_send approval check failed", exc_info=True)
            return {"ok": False, "error": "Outbound message approval check failed"}

    return await _gateway_server_ref.send_message(
        platform,
        chat_id,
        text,
        thread_id=params.get("thread_id", ""),
    )


async def platform_action_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered platform action through the gateway boundary."""
    if _gateway_server_ref is None:
        return {"ok": False, "error": "Gateway not initialised"}
    platform = str(params.get("platform") or "")
    action_name = str(params.get("action") or "")
    if not platform:
        return {"ok": False, "error": "platform is required"}
    if not action_name:
        return {"ok": False, "error": "action is required"}

    spec = _gateway_server_ref.get_platform_action_spec(platform, action_name)
    if spec is None:
        return _platform_action_unavailable_response(platform, action_name)

    from leapflow.gateway.connectors.action_registry import normalize_payload, validate_payload

    payload = normalize_payload(spec, params)
    validation = validate_payload(spec, payload)
    if not validation.ok:
        return _structured_validation_error(spec, validation, params)

    # Task-scoped side-effect dedup: block re-execution of identical
    # send/write/execute actions within the same user turn.
    fp = _action_fingerprint(platform, action_name, payload)
    if spec.effect in _SIDE_EFFECT_KINDS and fp in _task_completed_actions:
        logger.info(
            "side_effect_dedup: blocked duplicate %s.%s fp=%s",
            platform, action_name, fp,
        )
        original = _task_completed_actions[fp]
        return {
            "ok": True,
            "already_executed": True,
            "execution_note": (
                f"This exact action ({platform}.{action_name}) was already executed "
                "successfully in this task. Do not re-invoke. Report the original "
                "result to the user."
            ),
            "original_result": original,
            "retryable": False,
        }

    preview = await _gateway_server_ref.preview_platform_action(platform, action_name, payload)
    if not preview.get("ok"):
        return {"ok": False, "error": str(preview.get("error") or "Platform action preview failed")}

    if _approval_gate is not None:
        try:
            from leapflow.security.actions import ActionDescriptor
            from leapflow.security.approval import ApprovalDecision, ApprovalRequest

            approval_action = ActionDescriptor.platform_action(
                platform,
                action_name,
                payload,
                backend_kind=spec.backend_kind,
                metadata={
                    "effect": spec.effect,
                    "risk_level": spec.risk_level,
                    "output_policy": spec.output_policy,
                    "preview": preview.get("summary", ""),
                },
            )
            if hasattr(_approval_gate, "evaluate"):
                result = await _approval_gate.evaluate(approval_action)
                if not getattr(result, "approved", False):
                    error = str(getattr(result, "denial_message", "") or "Platform action denied by approval gate")
                    return {"ok": False, "error": error}
            else:
                decision = await _approval_gate.request_approval(ApprovalRequest(
                    category=approval_action.kind,
                    detail=str(preview.get("summary") or approval_action.summary),
                    risk_hint=0.6,
                    metadata={
                        "platform": platform,
                        "action": action_name,
                        "backend_kind": spec.backend_kind,
                        "risk_level": spec.risk_level,
                    },
                    action=approval_action,
                ))
                if decision not in {
                    ApprovalDecision.ALLOW,
                    ApprovalDecision.ALLOW_ONCE,
                    ApprovalDecision.ALLOW_SESSION,
                    ApprovalDecision.ALLOW_ALWAYS,
                }:
                    return {"ok": False, "error": "Platform action denied by approval gate"}
        except Exception:
            logger.debug("platform_action approval check failed", exc_info=True)
            return {"ok": False, "error": "Platform action approval check failed"}

    result = await _gateway_server_ref.execute_platform_action(platform, action_name, payload)

    # Register successful side-effect actions for dedup.
    if result.get("ok") and spec.effect in _SIDE_EFFECT_KINDS:
        _task_completed_actions[fp] = {
            k: v for k, v in result.items()
            if k in ("ok", "resource_id", "data", "action", "platform")
        }

    return result


# ═══════════════════════════════════════════════════════════════
# Tool registration (OpenAI function calling schema)
# ═══════════════════════════════════════════════════════════════

GATEWAY_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "platform_action",
            "description": (
                "Execute an exact registered business action on an external platform through "
                "LeapFlow's App Connector layer. Actions must be copied from the App Connector "
                "Capability Index and are addressed as domain.operation, "
                "e.g. im.send_message or docs.create_markdown. All business fields (chat_id, text, query, etc.) "
                "MUST be placed inside `payload`, never at the top level. "
                "Example: {\"platform\":\"feishu\",\"action\":\"im.send_message\",\"payload\":{\"chat_id\":\"oc_xxx\",\"text\":\"hello\"}}. "
                "Do not invent action names, do not use management actions such as list/guide/connect/status here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "description": "Platform ID, e.g. feishu"},
                    "action": {"type": "string", "description": "Exact registered business action from the Capability Index, e.g. im.send_message"},
                    "payload": {"type": "object", "description": "Action payload — all business fields go here (e.g. chat_id, text, query). See Capability Index for required/optional fields per action."},
                    "backend_kind": {"type": "string", "description": "Optional backend hint: cli/rest/mcp"},
                },
                "required": ["platform", "action", "payload"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "platform_connect",
            "description": (
                "List, guide, connect, disconnect, remove, or check status for external "
                "platforms using the App Connector management namespace. Supports REST and CLI "
                "backends. Use this for management actions such as list/guide/preflight/connect/status; "
                "use platform_action only for exact registered business actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(PLATFORM_CONNECT_ACTIONS)},
                    "platform": {"type": "string", "description": "Platform ID"},
                    "credentials": {"type": "object", "description": "Optional credentials for REST-style backends"},
                    "options": {"type": "object", "description": "Backend options such as profile, identity, or binary"},
                    "checkpoint": {"type": "string", "description": "Optional event source resume checkpoint"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gateway_send",
            "description": (
                "Send a message to a connected external platform "
                "(Feishu group, Telegram chat, DingTalk conversation, etc.).  "
                "Requires the platform to be connected via gateway_connect first.  "
                "Use gateway_connect with action='list' to see connected platforms "
                "and available chat IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Platform ID (feishu, telegram, dingtalk, etc.)",
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Target chat/group/channel ID",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text to send",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread/topic ID for threaded replies (optional)",
                    },
                },
                "required": ["platform", "chat_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gateway_connect",
            "description": (
                "Connect, configure, or manage external platform integrations "
                "(Feishu, DingTalk, Telegram, Slack, Discord, etc.).  "
                "Conversational flow: 1) call 'guide' to get setup steps + "
                "required fields, 2) present the steps to the user and ask "
                "for ALL required credentials in a single message, 3) call "
                "'connect' with the credentials.  Goal: complete in 1–2 user "
                "turns.  NEVER include credential values in your text response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list", "guide", "connect", "disconnect",
                            "remove", "status",
                        ],
                        "description": (
                            "Action to perform.  'disconnect' pauses the "
                            "connection (credentials kept for reconnect); "
                            "'remove' deletes saved credentials entirely."
                        ),
                    },
                    "platform": {
                        "type": "string",
                        "description": (
                            "Platform ID (feishu, dingtalk, telegram, etc.)"
                        ),
                    },
                    "credentials": {
                        "type": "object",
                        "description": (
                            "Platform credentials (keys vary by platform)"
                        ),
                    },
                    "options": {
                        "type": "object",
                        "description": (
                            "Optional platform configuration overrides"
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
]

GATEWAY_BRIDGE_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "gp_platform_connect",
        "description": "Connect or manage external platform integrations via App Connector.",
        "parameters": {
            "action": "string (required) — list/guide/connect/disconnect/remove/status/events_start/events_stop/events_status",
            "platform": "string (optional) — platform ID",
            "credentials": "object (optional) — backend credentials",
            "options": "object (optional) — backend options",
            "checkpoint": "string (optional) — event source resume checkpoint",
        },
        "handler": platform_connect_handler,
    },
    {
        "name": "gp_platform_action",
        "description": "Execute a registered platform action via App Connector.",
        "parameters": {
            "platform": "string (required) — platform ID",
            "action": "string (required) — domain.operation",
            "payload": "object (required) — action payload",
        },
        "handler": platform_action_handler,
        "mutates_state": True,
    },
    {
        "name": "gp_gateway_connect",
        "description": "Connect or manage external platform integrations.",
        "parameters": {
            "action": "string (required) — list/guide/connect/disconnect/remove/status",
            "platform": "string (optional) — platform ID",
            "credentials": "object (optional) — platform credentials",
            "options": "object (optional) — configuration overrides",
        },
        "handler": gateway_connect_handler,
    },
    {
        "name": "gp_gateway_send",
        "description": "Send a message to a connected external platform.",
        "parameters": {
            "platform": "string (required) — platform ID",
            "chat_id": "string (required) — target chat/group ID",
            "text": "string (required) — message text",
            "thread_id": "string (optional) — thread ID for replies",
        },
        "handler": gateway_send_handler,
    },
]

GATEWAY_TOOL_HANDLERS: Dict[str, Any] = {
    "platform_connect": platform_connect_handler,
    "gp_platform_connect": platform_connect_handler,
    "platform_action": platform_action_handler,
    "gp_platform_action": platform_action_handler,
    "gateway_connect": gateway_connect_handler,
    "gp_gateway_connect": gateway_connect_handler,
    "gateway_send": gateway_send_handler,
    "gp_gateway_send": gateway_send_handler,
}

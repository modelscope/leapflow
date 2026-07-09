"""Gateway tools for the agent — configuration AND messaging.

Two tools:
- ``gateway_connect``: conversational platform configuration
- ``gateway_send``: proactive outbound messaging to connected platforms

SECURITY: Tool results NEVER contain credential values.
Credentials flow: user message → LLM parse → tool handler → vault.
Tool returns only status information back to LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_gateway_server_ref: Any = None
_approval_gate: Any = None


def set_gateway_server(server: Any) -> None:
    """Install ``GatewayServer`` reference for tool dispatch (late-bound)."""
    global _gateway_server_ref
    _gateway_server_ref = server


def set_gateway_approval_gate(gate: Any) -> None:
    """Install approval gate for outbound messaging (late-bound)."""
    global _approval_gate
    _approval_gate = gate


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
    }

    required = [c for c in manifest.credentials if c.required]
    labels = " / ".join(c.label for c in required)
    result["prompt_hint"] = (
        f"Present the setup steps above, then ask the user to provide "
        f"{labels} in a single reply.  Do NOT repeat credential values "
        f"in your response."
    )

    return result


async def _action_connect(
    platform: str, params: Dict[str, Any],
) -> Dict[str, Any]:
    """Connect a platform with provided credentials.

    SECURITY: credentials flow into ``connect_platform()`` which encrypts
    and persists them.  Return value contains **only** status.
    """
    credentials = params.get("credentials", {})
    options = params.get("options", {})

    if not platform:
        return {"ok": False, "error": "Platform ID is required"}
    if not credentials:
        return await _action_guide(platform, params)

    return await _gateway_server_ref.connect_platform(
        platform, credentials, options,
    )


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

    if _approval_gate is not None:
        try:
            from leapflow.security.approval import ApprovalDecision, ApprovalRequest

            preview = text[:80] + ("…" if len(text) > 80 else "")
            decision = await _approval_gate.request_approval(ApprovalRequest(
                category=f"gateway_send:{platform}",
                detail=f"Send to {platform}/{chat_id}: {preview}",
                risk_hint=0.5,
                metadata={"platform": platform, "chat_id": chat_id},
            ))
            if decision == ApprovalDecision.DENY:
                return {"ok": False, "error": "Outbound message denied by approval gate"}
        except Exception:
            logger.debug("gateway_send approval check failed", exc_info=True)

    return await _gateway_server_ref.send_message(
        platform,
        chat_id,
        text,
        thread_id=params.get("thread_id", ""),
    )


# ═══════════════════════════════════════════════════════════════
# Tool registration (OpenAI function calling schema)
# ═══════════════════════════════════════════════════════════════

GATEWAY_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
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
    "gateway_connect": gateway_connect_handler,
    "gp_gateway_connect": gateway_connect_handler,
    "gateway_send": gateway_send_handler,
    "gp_gateway_send": gateway_send_handler,
}

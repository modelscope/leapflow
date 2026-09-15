# Copyright (c) Alibaba, Inc. and its affiliates.
"""Gateway tools plugin — exposes platform connectivity and messaging as a ToolPlugin."""

from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata


class GatewayToolsPlugin:
    """Agent-facing platform integration tools (connect, action, send)."""

    def __init__(self) -> None:
        self._gateway_server: Any = None

    @property
    def plugin_id(self) -> str:
        return "gateway"

    @property
    def category(self) -> str:
        return "gateway"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.gateway_tool import (
            PLATFORM_CONNECT_ACTIONS,
            gateway_connect_handler,
            gateway_send_handler,
            platform_action_handler,
            platform_connect_handler,
        )

        return [
            ToolMetadata(
                name="platform_action",
                description=(
                    "Execute an exact registered business action on an external platform through "
                    "LeapFlow's App Connector layer. Actions must be copied from the App Connector "
                    "Capability Index and are addressed as domain.operation, "
                    "e.g. im.send_message or docs.create_markdown. All business fields (chat_id, text, query, etc.) "
                    "MUST be placed inside `payload`, never at the top level. "
                    'Example: {"platform":"feishu","action":"im.send_message","payload":{"chat_id":"oc_xxx","text":"hello"}}. '
                    "Do not invent action names, do not use management actions such as list/guide/connect/status here."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string", "description": "Platform ID, e.g. feishu"},
                        "action": {
                            "type": "string",
                            "description": "Exact registered business action from the Capability Index, e.g. im.send_message",
                        },
                        "payload": {
                            "type": "object",
                            "description": "Action payload — all business fields go here (e.g. chat_id, text, query). See Capability Index for required/optional fields per action.",
                        },
                        "backend_kind": {
                            "type": "string",
                            "description": "Optional backend hint: cli/rest/mcp",
                        },
                    },
                    "required": ["platform", "action", "payload"],
                },
                handler=platform_action_handler,
                x_leapflow={
                    "category": "gateway",
                    "risk_level": "high",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                mutates_state=True,
                provides_capabilities=("platform.action",),
                requires_capabilities=("platform.connect",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="platform_connect",
                description=(
                    "List, guide, connect, disconnect, remove, or check status for external "
                    "platforms using the App Connector management namespace. Supports REST and CLI "
                    "backends. Use this for management actions such as list/guide/preflight/connect/status; "
                    "use platform_action only for exact registered business actions."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(PLATFORM_CONNECT_ACTIONS)},
                        "platform": {"type": "string", "description": "Platform ID"},
                        "credentials": {
                            "type": "object",
                            "description": "Optional credentials for REST-style backends",
                        },
                        "options": {
                            "type": "object",
                            "description": "Backend options such as profile, identity, or binary",
                        },
                        "checkpoint": {
                            "type": "string",
                            "description": "Optional event source resume checkpoint",
                        },
                    },
                    "required": ["action"],
                },
                handler=platform_connect_handler,
                x_leapflow={
                    "category": "gateway",
                    "risk_level": "medium",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                provides_capabilities=("platform.connect",),
            ),
            ToolMetadata(
                name="gateway_send",
                description=(
                    "Send a message to a connected external platform "
                    "(Feishu group, Telegram chat, DingTalk conversation, etc.).  "
                    "Requires the platform to be connected via gateway_connect first.  "
                    "Use gateway_connect with action='list' to see connected platforms "
                    "and available chat IDs."
                ),
                parameters_schema={
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
                        "text": {"type": "string", "description": "Message text to send"},
                        "thread_id": {
                            "type": "string",
                            "description": "Thread/topic ID for threaded replies (optional)",
                        },
                    },
                    "required": ["platform", "chat_id", "text"],
                },
                handler=gateway_send_handler,
                x_leapflow={
                    "category": "gateway",
                    "risk_level": "high",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                mutates_state=True,
                provides_capabilities=("platform.send_message",),
                requires_capabilities=("platform.configure",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="gateway_connect",
                description=(
                    "Connect, configure, or manage external platform integrations "
                    "(Feishu, DingTalk, Telegram, Slack, Discord, etc.).  "
                    "Conversational flow: 1) call 'guide' to get setup steps + "
                    "required fields, 2) present the steps to the user and ask "
                    "for ALL required credentials in a single message, 3) call "
                    "'connect' with the credentials.  Goal: complete in 1\u20132 user "
                    "turns.  NEVER include credential values in your text response."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "guide", "connect", "disconnect", "remove", "status"],
                            "description": (
                                "Action to perform.  'disconnect' pauses the "
                                "connection (credentials kept for reconnect); "
                                "'remove' deletes saved credentials entirely."
                            ),
                        },
                        "platform": {
                            "type": "string",
                            "description": "Platform ID (feishu, dingtalk, telegram, etc.)",
                        },
                        "credentials": {
                            "type": "object",
                            "description": "Platform credentials (keys vary by platform)",
                        },
                        "options": {
                            "type": "object",
                            "description": "Optional platform configuration overrides",
                        },
                    },
                    "required": ["action"],
                },
                handler=gateway_connect_handler,
                x_leapflow={
                    "category": "gateway",
                    "risk_level": "medium",
                    "schema_cost": "high",
                    "requires_approval": True,
                },
                provides_capabilities=("platform.configure",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["gateway_server"]

    def bind_runtime(self, **deps: Any) -> None:
        if "gateway_server" in deps:
            self._gateway_server = deps["gateway_server"]
            # Propagate to the module-level ref used by handler functions
            from leapflow.tools.gateway_tool import set_gateway_server

            set_gateway_server(deps["gateway_server"])


# Module-level instance for plugin discovery
plugin = GatewayToolsPlugin()

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Configuration tools plugin — list, get, and set LeapFlow settings."""

from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata


class ConfigToolsPlugin:
    """Agent-facing LeapFlow configuration tools (key-based, never path-based)."""

    def __init__(self) -> None:
        self._approval_gate: Any = None

    @property
    def plugin_id(self) -> str:
        return "config_tools"

    @property
    def category(self) -> str:
        return "config"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.config_model_service import config_model_service_handler
        from leapflow.tools.config_tools import (
            config_get_handler,
            config_list_handler,
            config_set_handler,
        )

        return [
            ToolMetadata(
                name="config_list",
                description=(
                    "List LeapFlow's own writable settings (LLM model and endpoint, daemon, memory, "
                    "perception, gateway, …) with current values. Use this to discover the exact "
                    "key before changing anything. Call without a category first; narrow using only "
                    "returned categories or key prefixes, not guessed feature names. If the catalog "
                    "has no field for the requested feature, report it as unsupported. This is the "
                    "only correct way to inspect LeapFlow configuration \u2014 never read config files "
                    "from disk."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category filter, e.g. 'LLM Provider' or 'Runtime'",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max fields to return (default 60)",
                        },
                    },
                },
                handler=config_list_handler,
                x_leapflow={
                    "category": "config",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("config.list",),
            ),
            ToolMetadata(
                name="config_get",
                description=(
                    "Read one LeapFlow setting by key (e.g. 'llm.model', 'llm.base_url', "
                    "'daemon.log_level'), returning its current value, type, scopes, and whether a "
                    "change needs a daemon restart. Discover unfamiliar keys with config_list rather "
                    "than guessing them. There is no 'llm.provider' key: provider behavior "
                    "is inferred from the OpenAI-compatible 'llm.base_url'. Never read LeapFlow config "
                    "files from disk — use this."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Dot-separated config key, e.g. 'llm.model'",
                        },
                    },
                    "required": ["key"],
                },
                handler=config_get_handler,
                x_leapflow={
                    "category": "config",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("config.get",),
            ),
            ToolMetadata(
                name="config_set",
                description=(
                    "Change one LeapFlow setting by key, e.g. switch an OpenAI-compatible LLM with "
                    "key='llm.model' and key='llm.base_url'. There is no 'llm.provider' key: provider "
                    "behavior is inferred from the endpoint. Values are validated and coerced; credentials "
                    "are stored in the vault automatically. Call config_list or config_get first if unsure "
                    "of the key. The result states whether a `leap daemon restart` is required; that "
                    "restart is for the user to run after the turn, never from inside it. Never edit "
                    "LeapFlow config files directly."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Dot-separated config key, e.g. 'llm.model'",
                        },
                        "value": {"description": "New value; coerced to the field's declared type"},
                        "scope": {
                            "type": "string",
                            "enum": ["profile", "workspace"],
                            "description": "Where to persist (default: profile)",
                        },
                    },
                    "required": ["key", "value"],
                },
                handler=config_set_handler,
                x_leapflow={
                    "category": "config",
                    "risk_level": "medium",
                    "schema_cost": "low",
                    "requires_approval": True,
                    "mutates_state": True,
                    "idempotency_scope": "turn",
                },
                mutates_state=True,
                provides_capabilities=("config.set",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="config_model_service",
                description=(
                    "Connect or manage an advisory specialist alongside the unchanged primary chat model. "
                    "Available with an empty catalog. show reads profile declarations; set saves one bounded "
                    "declaration after session-routed approval; unset removes only that ID after approval. "
                    "enable/disable changes only its centralized admission flag after approval, not the serving process. "
                    "check is local-only unless the user explicitly supplies a text-only schema_version=1 "
                    "case with artifact_scope=synthetic or recorded, input and expected JSON schema. "
                    "API checks require endpoint consent. Never supply paths, image data or URLs as evidence; "
                    "use model_query with owned artifact IDs. No serving-process management or physical qualification."
                ),
                parameters_schema={
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "operation": {"type": "string", "enum": ["show", "set", "unset", "check", "enable", "disable"]},
                        "service_id": {"type": "string", "maxLength": 64},
                        "declaration": {
                            "type": "object", "description": "Single v1 specialist declaration; vault refs only.",
                            "properties": {
                                "schema_version": {"type": "integer", "enum": [1]},
                                "provider": {"type": "string"}, "base_url": {"type": "string"},
                                "model": {"type": "string"}, "task": {"type": "string"},
                                "codec": {"type": "string", "enum": ["json.v1"]},
                                "output": {"type": "string", "enum": ["json_text", "json_schema", "tool_call"]},
                                "auth": {"type": "object", "description": "mode=none or mode=secret with vault ref"},
                                "processing_location": {"type": "string", "enum": ["local", "remote", "unknown"]},
                                "allow_remote": {"type": "boolean"},
                                "enabled": {"type": "boolean", "description": "Admission control; defaults to true. Does not manage serving processes."},
                                "limits": {"type": "object"}, "identity": {"type": "object"},
                                "model_info": {"type": "object"}, "deployment": {"type": "object"},
                                "request_options": {"type": "object"}, "prompt_template": {"type": "string"},
                            },
                            "required": ["schema_version", "provider", "base_url", "model", "task", "auth"],
                            "additionalProperties": False,
                        },
                        "case": {"type": "object", "description": "Explicit inline text-only fixture; no artifacts."},
                    },
                    "required": ["operation"],
                },
                handler=config_model_service_handler,
                x_leapflow={"category": "config", "risk_level": "medium", "schema_cost": "medium",
                            "requires_approval": True, "mutates_state": True, "idempotency_scope": "turn"},
                mutates_state=True,
                execution_policy="external_side_effect",
                provides_capabilities=("config.model_service",),
                requires_platform_capabilities=("file.ops",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return ["config_approval_gate"]

    def bind_runtime(self, **deps: Any) -> None:
        if "config_approval_gate" in deps:
            self._approval_gate = deps["config_approval_gate"]
            # Propagate to the module-level gate used by config_set_handler
            from leapflow.tools.config_tools import set_config_approval_gate

            set_config_approval_gate(deps["config_approval_gate"])


# Module-level instance for plugin discovery
plugin = ConfigToolsPlugin()

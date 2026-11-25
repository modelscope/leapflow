# Copyright (c) Alibaba, Inc. and its affiliates.
"""Optional specialist tool publication through the ordinary plugin lifecycle."""
from __future__ import annotations

from typing import Any

from leapflow.plugins.protocol import ToolMetadata
from leapflow.tools.model_tools import ModelTools


class ModelServicesPlugin:
    def __init__(self) -> None:
        self._behavior = ModelTools()
        self._scope: Any = None

    @property
    def plugin_id(self) -> str:
        return "model_collaboration"

    @property
    def category(self) -> str:
        return "model"

    @property
    def dependencies(self) -> list[str]:
        return ["model_services", "settings", "model_approval_gate", "effect_scope"]

    def bind_runtime(self, **deps: Any) -> None:
        if "model_services" in deps:
            self._behavior.client = deps["model_services"]
        if "settings" in deps:
            self._behavior.settings = deps["settings"]
        if "model_approval_gate" in deps:
            self._behavior.gate = deps["model_approval_gate"]
        scope = deps.get("effect_scope")
        if scope is not None and scope is not self._scope:
            self._scope = scope
            scope.async_effect(self._behavior.aclose)

    @property
    def tools(self) -> list[ToolMetadata]:
        client = self._behavior.client
        if client is None or not client.describe()["services"]:
            return []
        metadata = {"category": "model", "risk_level": "read_only", "schema_cost": "low",
                    "requires_approval": False, "execution_policy": "read_only"}
        return [
            ToolMetadata(
                name="model_describe",
                description="Describe configured specialist services and selected input/result schemas. Cached; no network probe.",
                parameters_schema={"type": "object", "properties": {"service_id": {"type": "string"}}},
                handler=self._behavior.describe, x_leapflow={**metadata, "summary": "describe specialist services"},
                provides_capabilities=("model.describe",),
            ),
            ToolMetadata(
                name="model_query",
                description=("Read-only typed specialist query. Use model_describe for the selected input schema. "
                             "Artifacts map labels to caller-owned session cache IDs, never paths/URLs/base64. "
                             "Call directly; runtime requests any necessary service consent. Do not ask for duplicate consent in dialogue. "
                             "Decisions are advisory, predictions hypothetical, evaluations teacher assessments. "
                             "Physical actions require hw_skill_* / policy.run; generated calls never execute."),
                parameters_schema={"type": "object", "properties": {
                    "service_id": {"type": "string"}, "input": {"type": "object"},
                    "artifacts": {"type": "object", "additionalProperties": {"type": "string"}},
                }, "required": ["service_id", "input"], "additionalProperties": False},
                handler=self._behavior.query, x_leapflow={**metadata, "summary": "query an advisory specialist"},
                provides_capabilities=("model.query",),
            ),
        ]


plugin = ModelServicesPlugin()

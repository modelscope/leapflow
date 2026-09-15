# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive plugin closed-loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.plugins.adaptive_loop import (
    AdaptiveLoopMutation,
    AdaptiveLoopRequest,
    AdaptivePluginLoop,
)
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore


async def _handler(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True, "value": kwargs}


@dataclass
class _Plugin:
    plugin_id: str
    tools: list[ToolMetadata]
    category: str = "test"
    dependencies: list[str] | None = None

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []

    def bind_runtime(self, **deps: Any) -> None:
        return None


class _InstallingActor:
    def __init__(self, registry: ToolPluginRegistry, plugin: _Plugin) -> None:
        self.registry = registry
        self.plugin = plugin
        self.calls: list[tuple[str, str]] = []

    async def install(
        self,
        *,
        plugin_id: str,
        code: str,
        proposal_id: str = "",
        version_label: str = "",
    ) -> Mapping[str, Any]:
        self.calls.append(("install", plugin_id))
        self.registry.register(self.plugin)
        self.registry.publish_plugin_tools(self.plugin)
        return {"ok": True, "action": "install", "plugin_id": plugin_id}

    async def disable(self, *, plugin_id: str) -> Mapping[str, Any]:
        self.calls.append(("disable", plugin_id))
        self.registry.unregister_plugin(plugin_id)
        return {"ok": True, "action": "disable", "plugin_id": plugin_id}

    async def remove(self, *, plugin_id: str, delete_source: bool = True) -> Mapping[str, Any]:
        self.calls.append(("remove", plugin_id))
        self.registry.unregister_plugin(plugin_id)
        return {"ok": True, "action": "remove", "plugin_id": plugin_id}


def _env() -> EnvironmentFingerprint:
    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset({Capability.FILE_OPS}))
    )


def _req(capability: str) -> CapabilityRequirement:
    return CapabilityRequirement.create(
        capability,
        "explicit_request",
        requirement_id=f"req-{capability}",
    )


def _tool(name: str, *, provides: tuple[str, ...]) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        description=f"{name} tool",
        parameters_schema={"type": "object", "properties": {}},
        handler=_handler,
        provides_capabilities=provides,
    )


@pytest.mark.asyncio
async def test_adaptive_loop_installs_then_re_resolves(tmp_path) -> None:
    registry = ToolPluginRegistry()
    registry.assemble()
    plugin = _Plugin("json_loop", [_tool("json_pretty_loop", provides=("json.pretty",))])
    actor = _InstallingActor(registry, plugin)
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")
    loop = AdaptivePluginLoop(registry=registry, plan_store=store, lifecycle_actor=actor)

    request = AdaptiveLoopRequest(
        environment=_env(),
        requirements=(_req("json.pretty"),),
        source="unit_closed_loop",
        loop_id="loop-install",
        mutation=AdaptiveLoopMutation(
            action="install",
            plugin_id="json_loop",
            code="# fixture code",
        ),
    )

    result = await loop.run(request)

    assert result.ok is True
    assert result.before.plan.executable is True
    assert result.before.resolutions[0].selected is None
    assert result.after is not None
    assert result.after.resolutions[0].selected is not None
    assert result.after.resolutions[0].selected.candidate.tool_name == "json_pretty_loop"
    assert result.selected_delta["added"] == {"json.pretty": "json_pretty_loop"}
    assert result.registry_version_after > result.registry_version_before
    records = store.list_records(limit=5)
    assert [record["phase"] for record in records] == ["after_install", "before"]
    assert records[0]["mutation"]["action"] == "install"


@pytest.mark.asyncio
async def test_adaptive_loop_remove_changes_selection(tmp_path) -> None:
    registry = ToolPluginRegistry()
    plugin = _Plugin("json_loop", [_tool("json_pretty_loop", provides=("json.pretty",))])
    registry.register(plugin)
    registry.assemble()
    actor = _InstallingActor(registry, plugin)
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")
    loop = AdaptivePluginLoop(registry=registry, plan_store=store, lifecycle_actor=actor)

    request = AdaptiveLoopRequest(
        environment=_env(),
        requirements=(_req("json.pretty"),),
        source="unit_closed_loop",
        loop_id="loop-remove",
        mutation=AdaptiveLoopMutation(action="remove", plugin_id="json_loop"),
    )

    result = await loop.run(request)

    assert result.ok is True
    assert result.before.resolutions[0].selected is not None
    assert result.after is not None
    assert result.after.resolutions[0].selected is None
    assert result.selected_delta["removed"] == {"json.pretty": "json_pretty_loop"}

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the Sub-Agent Monitor dashboard panel.

Covers:
- SubagentManager.get_active_state() snapshot structure
- _MONITOR_EVENTS includes the three subagent event types
- subagents.yaml template loads and validates against COMPONENT_CATALOG
- DashboardDataProvider.subagent_state() is part of the Protocol
- DashboardViewBuilder can build the subagents template
- DaemonClient.subagent_state RPC is registered in METHOD_REGISTRY
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest
import yaml

from leapflow.engine.subagent import (
    SubagentConfig,
    SubagentManager,
    SubagentResult,
)


# ── get_active_state structure ────────────────────────────────────────────


class StubExecutor:
    """Executor that returns immediately with a controlled result."""

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        return SubagentResult(
            session_id="sub_test123",
            goal=config.goal,
            summary="Done.",
            status="completed",
            elapsed_s=1.5,
            tool_calls=3,
        )


class FailingExecutor:
    """Executor that always fails."""

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        return SubagentResult(
            session_id="sub_fail456",
            goal=config.goal,
            summary="Boom",
            status="failed",
            elapsed_s=0.5,
            tool_calls=0,
            error="test_error",
        )


def test_get_active_state_empty():
    """A fresh manager returns the expected empty structure."""
    mgr = SubagentManager(max_depth=2, max_concurrent=3)
    state = mgr.get_active_state()

    assert isinstance(state, dict)
    assert "active" in state and "recent" in state and "stats" in state and "config" in state
    assert state["active"] == []
    assert state["recent"] == []
    assert state["stats"]["total_delegated"] == 0
    assert state["stats"]["completed"] == 0
    assert state["stats"]["failed"] == 0
    assert state["stats"]["avg_duration"] == 0.0
    assert state["stats"]["success_rate"] == 0.0
    assert state["config"]["max_depth"] == 2
    assert state["config"]["max_concurrent"] == 3


@pytest.mark.asyncio
async def test_get_active_state_after_completion():
    """After a successful delegation, stats and recent are updated."""
    mgr = SubagentManager(executor=StubExecutor(), max_depth=2, max_concurrent=3)
    config = SubagentConfig(goal="test task", depth=0)
    result = await mgr.delegate(config)

    assert result.status == "completed"
    state = mgr.get_active_state()
    assert state["stats"]["total_delegated"] == 1
    assert state["stats"]["completed"] == 1
    assert state["stats"]["failed"] == 0
    assert state["stats"]["success_rate"] == 1.0
    assert state["stats"]["avg_duration"] > 0
    assert len(state["recent"]) == 1
    assert state["recent"][0]["status"] == "completed"
    assert state["recent"][0]["goal"] == "test task"
    assert state["active"] == []  # completed, no longer active


@pytest.mark.asyncio
async def test_get_active_state_after_failure():
    """After a failed delegation, stats reflect the failure."""
    mgr = SubagentManager(executor=FailingExecutor(), max_depth=2, max_concurrent=3)
    config = SubagentConfig(goal="fail task", depth=0)
    result = await mgr.delegate(config)

    assert result.status == "failed"
    state = mgr.get_active_state()
    assert state["stats"]["total_delegated"] == 1
    assert state["stats"]["completed"] == 0
    assert state["stats"]["failed"] == 1
    assert state["stats"]["success_rate"] == 0.0
    assert len(state["recent"]) == 1
    assert state["recent"][0]["status"] == "failed"
    assert state["recent"][0].get("error") == "test_error"


@pytest.mark.asyncio
async def test_get_active_state_mixed():
    """Multiple delegations update stats correctly."""
    mgr = SubagentManager(executor=StubExecutor(), max_depth=2, max_concurrent=3)
    await mgr.delegate(SubagentConfig(goal="task 1", depth=0))
    await mgr.delegate(SubagentConfig(goal="task 2", depth=0))

    state = mgr.get_active_state()
    assert state["stats"]["total_delegated"] == 2
    assert state["stats"]["completed"] == 2
    assert len(state["recent"]) == 2
    # Recent is newest-first
    assert state["recent"][0]["goal"] == "task 2"
    assert state["recent"][1]["goal"] == "task 1"


# ── _MONITOR_EVENTS ─────────────────────────────────────────────────────


def test_monitor_events_include_subagent():
    """_MONITOR_EVENTS includes all three subagent event types."""
    from leapflow.dashboard.server import _MONITOR_EVENTS

    assert "subagent.started" in _MONITOR_EVENTS
    assert "subagent.completed" in _MONITOR_EVENTS
    assert "subagent.failed" in _MONITOR_EVENTS


# ── Template YAML ────────────────────────────────────────────────────────


def test_subagents_template_loads():
    """subagents.yaml loads as valid YAML with expected top-level keys."""
    template_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "leapflow" / "dashboard" / "templates" / "subagents.yaml"
    )
    with open(template_path) as f:
        raw = yaml.safe_load(f)

    assert isinstance(raw, dict)
    assert raw["template"] == "subagents"
    assert raw["version"] == 2
    assert "layout" in raw
    assert isinstance(raw["layout"], list)
    assert len(raw["layout"]) > 0


def test_subagents_template_validates():
    """subagents.yaml validates against COMPONENT_CATALOG (no unknown types)."""
    from leapflow.dashboard.templates import TemplateLibrary

    lib = TemplateLibrary()
    raw = lib.load("subagents")
    assert raw is not None, "subagents template not found in TemplateLibrary"
    error = lib.validate(raw)
    assert error is None, f"Template validation failed: {error}"


def test_subagents_template_component_types():
    """Every component type used in subagents.yaml is in COMPONENT_CATALOG."""
    from leapflow.dashboard.viewspec import COMPONENT_TYPES

    template_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "leapflow" / "dashboard" / "templates" / "subagents.yaml"
    )
    with open(template_path) as f:
        raw = yaml.safe_load(f)

    used_types: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if "type" in node:
                used_types.add(node["type"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(raw.get("layout", []))
    unknown = used_types - COMPONENT_TYPES
    assert not unknown, f"Unknown component types in subagents.yaml: {unknown}"


# ── DashboardDataProvider Protocol ───────────────────────────────────────


def test_protocol_includes_subagent_state():
    """DashboardDataProvider Protocol has subagent_state method."""
    from leapflow.dashboard.service import DashboardDataProvider

    assert hasattr(DashboardDataProvider, "subagent_state")
    # Verify it is callable from the protocol definition
    import inspect
    members = dict(inspect.getmembers(DashboardDataProvider))
    assert "subagent_state" in members


# ── DashboardViewBuilder.build for subagents ─────────────────────────────


class StubSubagentProvider:
    """Minimal provider implementing only what _build_subagents needs."""

    async def watches(self) -> list[dict[str, Any]]:
        return []

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return []

    async def signal_metrics(self) -> dict[str, Any]:
        return {}

    async def evolution_projection(self, *, session_id: str) -> dict[str, Any]:
        return {}

    async def evolution_projection_aggregate(self) -> dict[str, Any]:
        return {}

    async def hardware_inventory(self) -> dict[str, Any]:
        return {}

    async def hardware_device(self, device_id: str) -> dict[str, Any]:
        return {}

    async def subagent_state(self) -> dict[str, Any]:
        return {
            "active": [
                {
                    "subagent_id": "sub_abc",
                    "goal": "Do something",
                    "depth": 0,
                    "parent_session_id": "main_123",
                    "start_time": 1000000.0,
                    "elapsed_s": 5.0,
                },
            ],
            "recent": [
                {
                    "subagent_id": "sub_xyz",
                    "goal": "Previous task",
                    "depth": 0,
                    "parent_session_id": "main_123",
                    "status": "completed",
                    "duration_s": 2.5,
                    "tool_calls": 4,
                    "timestamp": 1000010.0,
                },
            ],
            "stats": {
                "total_delegated": 2,
                "completed": 1,
                "failed": 0,
                "avg_duration": 2.5,
                "success_rate": 1.0,
            },
            "config": {
                "max_depth": 2,
                "max_concurrent": 3,
                "summary_max_chars": 4000,
            },
        }


@pytest.mark.asyncio
async def test_build_subagents_view():
    """DashboardViewBuilder produces a valid ViewSpec for the subagents template."""
    from leapflow.dashboard.intent import DashboardIntent
    from leapflow.dashboard.service import DashboardViewBuilder
    from leapflow.dashboard.viewspec import validate_viewspec

    builder = DashboardViewBuilder()
    intent = DashboardIntent.from_params({"template": "subagents"})
    provider = StubSubagentProvider()
    spec = await builder.build(intent, provider)

    assert isinstance(spec, dict)
    assert spec.get("title")
    errors = validate_viewspec(spec)
    assert not errors, f"ViewSpec validation errors: {errors}"
    # Check root has content (not empty)
    assert len(spec.get("root", [])) > 0


@pytest.mark.asyncio
async def test_build_subagents_empty_state():
    """When subagent_state returns empty, the view shows an empty state."""
    from leapflow.dashboard.intent import DashboardIntent
    from leapflow.dashboard.service import DashboardViewBuilder
    from leapflow.dashboard.viewspec import validate_viewspec

    class EmptyProvider(StubSubagentProvider):
        async def subagent_state(self) -> dict[str, Any]:
            return {}

    builder = DashboardViewBuilder()
    intent = DashboardIntent.from_params({"template": "subagents"})
    provider = EmptyProvider()
    spec = await builder.build(intent, provider)

    assert isinstance(spec, dict)
    errors = validate_viewspec(spec)
    assert not errors, f"ViewSpec validation errors: {errors}"


# ── Daemon RPC registration ──────────────────────────────────────────────


def test_subagent_state_rpc_registered():
    """subagent.state is registered in the daemon METHOD_REGISTRY."""
    from leapflow.daemon.protocol import METHOD_REGISTRY

    assert "subagent.state" in METHOD_REGISTRY
    assert METHOD_REGISTRY["subagent.state"] == "subagent_state"


# ── Template discoverability ─────────────────────────────────────────────


def test_subagents_in_template_library():
    """TemplateLibrary discovers subagents.yaml as a builtin template."""
    from leapflow.dashboard.templates import TemplateLibrary

    lib = TemplateLibrary()
    assert "subagents" in lib.names()
    assert "subagents" in lib.visible_names()

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the SelfAwarenessPlugin (pull-based, session-scoped redesign)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from leapflow.plugins.tool_plugins.self_awareness import SelfAwarenessPlugin


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run(coro: Any) -> Any:
    """Drive an async handler to completion in a sync test."""
    return asyncio.run(coro)


def _status(**overrides: Any) -> dict[str, Any]:
    """A status snapshot shaped like ``RuntimeLeapService.status()``."""
    base: dict[str, Any] = {
        "build": {
            "version": "1.2.3",
            "commit": "deadbeef",
            "dirty_digest": "",
            "pid": 42,
            "started_at": 0.0,
        },
        "model": "gpt-4o",
        "llm_context_length": 1_000_000,
        "pid": 1234,
        "uptime_s": 780.0,
        "context_budget_snapshot": {
            "total_tokens": 112_000,
            "context_length": 1_000_000,
            "context_posture": "converging",
            "disclosure_level": "EXPANDED",
        },
        "context_used": 112_000,
        "context_posture": "converging",
        "session_turn_count": 13,
        "cache_hit_rate": 62.3,
        "evolution_performance": {"action_recorder": {"p50": 12.5}},
        "pending_approvals": 0,
        "active_clients": 2,
        "connected_clients": 1,
        "host_backend": {"backend": "cua-driver", "started": True},
        "environment_sources": {"active": ["lark_event_source"], "dropped": 0},
    }
    base.update(overrides)
    return base


def _make_plugin(status: dict[str, Any] | None = None, provider: Any = None) -> SelfAwarenessPlugin:
    """Create a plugin, optionally binding a status provider."""
    p = SelfAwarenessPlugin()
    if provider is not None:
        p.bind_runtime(runtime_status_provider=provider)
    elif status is not None:
        p.bind_runtime(runtime_status_provider=lambda _sid: status)
    return p


def _stub_registry(
    tool_count: int = 5,
    plugin_ids: tuple[str, ...] = ("file_ops", "shell_terminal"),
    version: int = 1,
) -> MagicMock:
    reg = MagicMock()
    reg.version = version
    metas = [
        SimpleNamespace(
            name=f"tool_{i}",
            x_leapflow={"category": "system" if i % 2 == 0 else "code"},
            mutates_state=(i % 2 == 1),
        )
        for i in range(tool_count)
    ]
    reg.all_metadata = metas
    reg.plugins = {pid: MagicMock() for pid in plugin_ids}
    reg.conflicts = []
    reg.tool_handlers = {f"tool_{i}": (lambda: None) for i in range(tool_count)}
    return reg


# ── Facet: identity ──────────────────────────────────────────────────────────


class TestFacetIdentity:
    def test_identity_from_status(self) -> None:
        p = _make_plugin(status=_status(model="gpt-4o"))
        result = _run(p._handle_self_describe(facet="identity"))
        assert result["available"] is True
        assert result["version"] == "1.2.3"
        assert result["commit"] == "deadbeef"
        assert result["model"] == "gpt-4o"
        assert "uptime" in result

    def test_identity_without_status_uses_captured_build(self) -> None:
        # No provider: identity still resolves from the captured build info and
        # live settings (pull), never going dark.
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="identity"))
        assert result["available"] is True
        assert isinstance(result["version"], str) and result["version"]
        assert "model" in result


# ── Facet: capabilities (live global registry) ───────────────────────────────


class TestFacetCapabilities:
    def test_capabilities_from_live_registry(self) -> None:
        # capabilities pulls from the process-global registry, so it works with
        # no injection at all — the failure mode that left it dark before.
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="capabilities"))
        assert result["available"] is True
        assert result["tool_count"] >= 1
        assert result["plugin_count"] >= 1
        assert "self_management" in result["plugin_ids"]
        assert result["supports_plugins"] is True
        assert "tools_by_category" in result
        assert "mutating_tool_count" in result
        # Skills are part of the single self-cognition surface.
        assert "skills" in result and isinstance(result["skills"], dict)

    def test_capabilities_includes_skill_inventory(self, tmp_path) -> None:
        """The capabilities facet reports skills from the discovery subsystem, so
        self_describe alone answers 'what skills do you have' — no separate call."""
        import leapflow.skills.discovery as disc
        from leapflow.skills.index import SkillIndex

        saved_index, saved_reg = disc._skill_index, disc._skill_registry
        try:
            disc._skill_index = SkillIndex(tmp_path)
            disc._skill_registry = None
            result = _run(_make_plugin()._handle_self_describe(facet="capabilities"))
            skills = result["skills"]
            assert skills["available"] is True
            assert skills["count"] >= 12
            assert "web_research" in skills["names"]
        finally:
            disc._skill_index, disc._skill_registry = saved_index, saved_reg


# ── Facet: runtime ───────────────────────────────────────────────────────────


class TestFacetRuntime:
    def test_runtime_from_status(self) -> None:
        p = _make_plugin(status=_status())
        result = _run(p._handle_self_describe(facet="runtime"))
        assert result["available"] is True
        assert result["context_used"] == 112_000
        assert result["session_turn_count"] == 13
        assert result["cache_hit_rate"] == "62.3%"
        assert result["posture"] == "converging"
        assert result["disclosure_level"] == "EXPANDED"

    def test_runtime_without_status_degrades(self) -> None:
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="runtime"))
        assert result["available"] is False
        assert "daemon" in result["reason"]


# ── Facet: evolution ─────────────────────────────────────────────────────────


class TestFacetEvolution:
    def test_evolution_from_status(self) -> None:
        p = _make_plugin(status=_status(pending_approvals=2))
        result = _run(p._handle_self_describe(facet="evolution"))
        assert result["available"] is True
        assert result["pending_approvals"] == 2
        assert "performance" in result

    def test_evolution_without_status_degrades(self) -> None:
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="evolution"))
        assert result["available"] is False


# ── Facet: platform ──────────────────────────────────────────────────────────


class TestFacetPlatform:
    def test_platform_from_status(self) -> None:
        p = _make_plugin(status=_status())
        result = _run(p._handle_self_describe(facet="platform"))
        assert result["available"] is True
        assert result["active_clients"] == 2
        assert result["host_backend"]["backend"] == "cua-driver"
        assert result["environment_sources"]["active"] == ["lark_event_source"]

    def test_platform_without_status_degrades(self) -> None:
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="platform"))
        assert result["available"] is False


# ── Facet: all ───────────────────────────────────────────────────────────────


class TestFacetAll:
    def test_all_merges_sub_reports(self) -> None:
        p = _make_plugin(status=_status())
        result = _run(p._handle_self_describe(facet="all"))
        assert set(result) == {"identity", "capabilities", "runtime", "evolution", "platform"}
        assert result["identity"]["available"] is True
        assert result["capabilities"]["available"] is True
        assert result["runtime"]["available"] is True


class TestInvalidFacet:
    def test_unknown_facet_returns_error(self) -> None:
        p = _make_plugin()
        result = _run(p._handle_self_describe(facet="nonexistent"))
        assert "error" in result


# ── Async provider + session scoping ─────────────────────────────────────────


class TestStatusProvider:
    def test_async_provider_is_awaited(self) -> None:
        async def provider(_sid: str) -> dict[str, Any]:
            return _status(model="async-model")

        p = _make_plugin(provider=provider)
        result = _run(p._handle_self_describe(facet="identity"))
        assert result["model"] == "async-model"

    def test_provider_receives_active_session_id(self) -> None:
        """The provider must be keyed by the active turn's session id so the
        daemon resolves the caller's own session, never a shared one."""
        from leapflow.tools.execution_context import (
            ToolExecutionContext,
            reset_tool_context,
            set_tool_context,
        )

        seen: list[str] = []

        def provider(session_id: str) -> dict[str, Any]:
            seen.append(session_id)
            return _status()

        p = _make_plugin(provider=provider)
        token = set_tool_context(
            ToolExecutionContext.from_strings(workspace_root=".", session_id="sess-123")
        )
        try:
            _run(p._handle_self_describe(facet="runtime"))
        finally:
            reset_tool_context(token)
        assert seen == ["sess-123"]

    def test_provider_failure_degrades(self) -> None:
        def provider(_sid: str) -> dict[str, Any]:
            raise RuntimeError("status boom")

        p = _make_plugin(provider=provider)
        result = _run(p._handle_self_describe(facet="platform"))
        assert result["available"] is False


# ── Registry version gate ────────────────────────────────────────────────────


class TestRegistryVersionGate:
    def test_cache_hit_on_same_version(self) -> None:
        reg = _stub_registry(tool_count=3, version=5)
        p = SelfAwarenessPlugin()
        p._registry_or_none = lambda: reg  # type: ignore[method-assign]

        r1 = _run(p._handle_self_describe(facet="capabilities"))
        assert r1["tool_count"] == 3

        # Mutate registry data but keep the same version → stale cache returned.
        reg.all_metadata = [SimpleNamespace(name="x", x_leapflow={"category": "new"}, mutates_state=False)]
        reg.plugins = {"only": MagicMock()}

        r2 = _run(p._handle_self_describe(facet="capabilities"))
        assert r2["tool_count"] == 3
        assert r2["plugin_count"] == 2

    def test_cache_invalidated_on_version_bump(self) -> None:
        reg = _stub_registry(tool_count=3, version=5)
        p = SelfAwarenessPlugin()
        p._registry_or_none = lambda: reg  # type: ignore[method-assign]

        r1 = _run(p._handle_self_describe(facet="capabilities"))
        assert r1["tool_count"] == 3

        reg.version = 6
        reg.all_metadata = [SimpleNamespace(name="x", x_leapflow={"category": "new"}, mutates_state=False)]
        reg.plugins = {"only": MagicMock()}
        reg.conflicts = []

        r2 = _run(p._handle_self_describe(facet="capabilities"))
        assert r2["tool_count"] == 1
        assert r2["plugin_count"] == 1


# ── runtime_snapshot ─────────────────────────────────────────────────────────


class TestRuntimeSnapshot:
    def test_returns_expected_keys(self) -> None:
        p = _make_plugin(status=_status())
        result = _run(p._handle_runtime_snapshot())
        assert set(result.keys()) == {
            "model", "context", "posture", "disclosure",
            "turn", "cache_hit_rate", "uptime", "tools_available",
            "pending_approvals",
        }

    def test_context_format(self) -> None:
        p = _make_plugin(status=_status())
        result = _run(p._handle_runtime_snapshot())
        assert "112K" in result["context"]
        assert "1M" in result["context"]
        assert "11%" in result["context"]

    def test_snapshot_without_provider(self) -> None:
        p = _make_plugin()
        result = _run(p._handle_runtime_snapshot())
        assert result["context"] == "unknown"
        # tools_available pulls from the live global registry, not a bound dep.
        assert result["tools_available"] >= 1
        assert result["turn"] == 0


# ── PCD metadata ─────────────────────────────────────────────────────────────


class TestPCDMetadata:
    def test_self_describe_schema(self) -> None:
        describe_tool = next(t for t in SelfAwarenessPlugin().tools if t.name == "self_describe")
        assert describe_tool.x_leapflow["category"] == "system"
        assert describe_tool.x_leapflow["risk_level"] == "read_only"

    def test_runtime_snapshot_schema(self) -> None:
        snapshot_tool = next(t for t in SelfAwarenessPlugin().tools if t.name == "runtime_snapshot")
        assert snapshot_tool.x_leapflow["category"] == "system"
        assert snapshot_tool.x_leapflow["risk_level"] == "read_only"

    def test_openai_schema_includes_x_leapflow(self) -> None:
        for tool in SelfAwarenessPlugin().tools:
            x = tool.to_openai_schema()["function"].get("x_leapflow", {})
            assert x.get("category") == "system"
            assert x.get("risk_level") == "read_only"


# ── Plugin protocol compliance ───────────────────────────────────────────────


class TestPluginProtocol:
    def test_plugin_id(self) -> None:
        assert SelfAwarenessPlugin().plugin_id == "self_awareness"

    def test_category(self) -> None:
        assert SelfAwarenessPlugin().category == "system"

    def test_dependencies(self) -> None:
        assert "runtime_status_provider" in SelfAwarenessPlugin().dependencies

    def test_bind_runtime_status_provider(self) -> None:
        p = SelfAwarenessPlugin()
        fn = lambda _sid: _status()  # noqa: E731
        p.bind_runtime(runtime_status_provider=fn)
        assert p._status_provider is fn

    def test_module_level_plugin_exists(self) -> None:
        from leapflow.plugins.tool_plugins.self_awareness import plugin

        assert plugin.plugin_id == "self_awareness"

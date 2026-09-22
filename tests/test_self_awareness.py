# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the SelfAwarenessPlugin."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from leapflow.plugins.tool_plugins.self_awareness import SelfAwarenessPlugin, _DAEMON_CACHE_TTL_S


# ── Stub factories ───────────────────────────────────────────────────────────


def _stub_build_info(
    version: str = "0.3.0",
    commit: str = "abc123",
    dirty_digest: str | None = None,
    started_at: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        version=version,
        commit=commit,
        dirty_digest=dirty_digest,
        pid=42,
        started_at=started_at or time.time(),
    )


def _stub_registry(
    tool_count: int = 5,
    plugin_ids: tuple[str, ...] = ("file_ops", "shell_terminal"),
    version: int = 1,
) -> MagicMock:
    reg = MagicMock()
    reg.version = version
    # all_metadata returns a list of stubs with x_leapflow
    metas = []
    for i in range(tool_count):
        meta = SimpleNamespace(
            name=f"tool_{i}",
            x_leapflow={"category": "system" if i % 2 == 0 else "code"},
        )
        metas.append(meta)
    reg.all_metadata = metas
    reg.plugins = {pid: MagicMock() for pid in plugin_ids}
    reg.conflicts = []
    reg.tool_handlers = {f"tool_{i}": lambda: None for i in range(tool_count)}
    return reg


def _stub_engine(
    context_snapshot: dict[str, Any] | None = None,
    turn_count: int = 5,
    cache_hit_rate: float | None = 62.3,
    model: str = "deepseek-chat",
    context_length: int = 1_000_000,
) -> SimpleNamespace:
    snapshot = context_snapshot or {
        "total_tokens": 112_000,
        "context_length": context_length,
        "context_posture": "converging",
        "disclosure_level": "EXPANDED",
    }
    summary_ns = SimpleNamespace(cache_hit_rate=cache_hit_rate)
    tracker = SimpleNamespace(summary=lambda: summary_ns)
    settings = SimpleNamespace(llm_model=model, llm_context_length=context_length)
    return SimpleNamespace(
        context_budget_snapshot=lambda: snapshot,
        _session_turn_count=turn_count,
        _usage_tracker=tracker,
        _settings=settings,
    )


def _stub_daemon_status(
    model: str = "deepseek-chat",
    uptime_s: float = 780.0,
    pending_approvals: int = 0,
    context_used: int = 112_000,
    llm_context_length: int = 1_000_000,
) -> dict[str, Any]:
    return {
        "pid": 1234,
        "model": model,
        "uptime_s": uptime_s,
        "llm_context_length": llm_context_length,
        "context_used": context_used,
        "pending_approvals": pending_approvals,
        "active_clients": 2,
        "connected_clients": 1,
        "host_backend": {"backend": "cua-driver", "started": True},
        "environment_sources": {"active": ["lark_event_source"], "dropped": 0},
        "evolution_performance": {"action_recorder": {"p50": 12.5}},
        "context_posture": "converging",
    }


def _make_plugin(
    daemon_status: dict[str, Any] | None = None,
    registry: Any = None,
    engine: Any = None,
    build_info: Any = None,
) -> SelfAwarenessPlugin:
    """Create a plugin with stub dependencies injected."""
    p = SelfAwarenessPlugin()
    if build_info is not None:
        p.bind_runtime(build_info=build_info)
    if registry is not None:
        p.bind_runtime(registry=registry)
    if engine is not None:
        p.bind_runtime(engine=engine)
    if daemon_status is not None:
        p.inject_daemon_cache(daemon_status)
    return p


# ── Facet tests ──────────────────────────────────────────────────────────────


class TestFacetIdentity:
    def test_identity_with_all_deps(self) -> None:
        p = _make_plugin(
            build_info=_stub_build_info(version="1.2.3", commit="deadbeef"),
            daemon_status=_stub_daemon_status(model="gpt-4o"),
        )
        result = p._handle_self_describe(facet="identity")
        assert result["available"] is True
        assert result["version"] == "1.2.3"
        assert result["commit"] == "deadbeef"
        assert result["model"] == "gpt-4o"
        assert "uptime" in result

    def test_identity_without_build_info(self) -> None:
        p = _make_plugin(daemon_status=_stub_daemon_status())
        result = p._handle_self_describe(facet="identity")
        assert result["available"] is True
        assert result["version"] == "unknown"

    def test_identity_without_daemon(self) -> None:
        engine = _stub_engine(model="claude-3")
        p = _make_plugin(build_info=_stub_build_info(), engine=engine)
        result = p._handle_self_describe(facet="identity")
        assert result["available"] is True
        assert result["model"] == "claude-3"


class TestFacetCapabilities:
    def test_capabilities_with_registry(self) -> None:
        reg = _stub_registry(tool_count=6, plugin_ids=("a", "b", "c"))
        p = _make_plugin(registry=reg)
        result = p._handle_self_describe(facet="capabilities")
        assert result["available"] is True
        assert result["tool_count"] == 6
        assert result["plugin_count"] == 3
        assert "tools_by_category" in result

    def test_capabilities_without_registry(self) -> None:
        p = _make_plugin()
        result = p._handle_self_describe(facet="capabilities")
        assert result["available"] is False
        assert "registry" in result["reason"]


class TestFacetRuntime:
    def test_runtime_with_engine(self) -> None:
        engine = _stub_engine(turn_count=13, cache_hit_rate=62.3)
        p = _make_plugin(engine=engine)
        result = p._handle_self_describe(facet="runtime")
        assert result["available"] is True
        assert result["context_used"] == 112_000
        assert result["session_turn_count"] == 13
        assert result["cache_hit_rate"] == "62.3%"
        assert result["posture"] == "converging"

    def test_runtime_without_engine(self) -> None:
        p = _make_plugin()
        result = p._handle_self_describe(facet="runtime")
        assert result["available"] is False
        assert "engine" in result["reason"]


class TestFacetEvolution:
    def test_evolution_with_daemon(self) -> None:
        p = _make_plugin(daemon_status=_stub_daemon_status(pending_approvals=2))
        result = p._handle_self_describe(facet="evolution")
        assert result["available"] is True
        assert result["pending_approvals"] == 2
        assert "performance" in result

    def test_evolution_without_daemon(self) -> None:
        p = _make_plugin()
        result = p._handle_self_describe(facet="evolution")
        assert result["available"] is False


class TestFacetPlatform:
    def test_platform_with_daemon(self) -> None:
        p = _make_plugin(daemon_status=_stub_daemon_status())
        result = p._handle_self_describe(facet="platform")
        assert result["available"] is True
        assert result["active_clients"] == 2
        assert result["host_backend"]["backend"] == "cua-driver"
        assert result["environment_sources"]["active"] == ["lark_event_source"]

    def test_platform_without_daemon(self) -> None:
        p = _make_plugin()
        result = p._handle_self_describe(facet="platform")
        assert result["available"] is False


class TestFacetAll:
    def test_all_merges_sub_reports(self) -> None:
        p = _make_plugin(
            build_info=_stub_build_info(),
            registry=_stub_registry(),
            engine=_stub_engine(),
            daemon_status=_stub_daemon_status(),
        )
        result = p._handle_self_describe(facet="all")
        assert "identity" in result
        assert "capabilities" in result
        assert "runtime" in result
        assert "evolution" in result
        assert "platform" in result
        assert result["identity"]["available"] is True
        assert result["capabilities"]["available"] is True
        assert result["runtime"]["available"] is True


class TestInvalidFacet:
    def test_unknown_facet_returns_error(self) -> None:
        p = _make_plugin()
        result = p._handle_self_describe(facet="nonexistent")
        assert "error" in result


# ── Registry version gate ────────────────────────────────────────────────────


class TestRegistryVersionGate:
    def test_cache_hit_on_same_version(self) -> None:
        reg = _stub_registry(tool_count=3, version=5)
        p = _make_plugin(registry=reg)

        # First call populates cache
        r1 = p._handle_self_describe(facet="capabilities")
        assert r1["tool_count"] == 3

        # Mutate registry data but keep same version
        reg.all_metadata = [SimpleNamespace(name="x", x_leapflow={"category": "new"})]
        reg.plugins = {"only": MagicMock()}

        # Second call returns cached (stale) data
        r2 = p._handle_self_describe(facet="capabilities")
        assert r2["tool_count"] == 3  # Still cached
        assert r2["plugin_count"] == 2  # Still cached

    def test_cache_invalidated_on_version_bump(self) -> None:
        reg = _stub_registry(tool_count=3, version=5)
        p = _make_plugin(registry=reg)

        r1 = p._handle_self_describe(facet="capabilities")
        assert r1["tool_count"] == 3

        # Bump version AND change data
        reg.version = 6
        reg.all_metadata = [SimpleNamespace(name="x", x_leapflow={"category": "new"})]
        reg.plugins = {"only": MagicMock()}
        reg.conflicts = []

        r2 = p._handle_self_describe(facet="capabilities")
        assert r2["tool_count"] == 1  # Rebuilt
        assert r2["plugin_count"] == 1


# ── Daemon TTL cache ─────────────────────────────────────────────────────────


class TestDaemonTTLCache:
    def test_within_ttl_returns_cached(self) -> None:
        p = _make_plugin(daemon_status=_stub_daemon_status(model="model-v1"))
        # Reading daemon cache should return the injected data within TTL
        cache = p._get_daemon_cache()
        assert cache.get("model") == "model-v1"

    def test_expired_ttl_attempts_refresh(self) -> None:
        p = _make_plugin(daemon_status=_stub_daemon_status(model="old-model"))
        # Force cache timestamp to be expired
        p._daemon_cache_ts = time.monotonic() - _DAEMON_CACHE_TTL_S - 10

        # Without a real daemon client, refresh returns empty (no async loop)
        # The cache is cleared because no daemon_client is bound
        cache = p._get_daemon_cache()
        assert isinstance(cache, dict)

    def test_inject_resets_timestamp(self) -> None:
        p = _make_plugin()
        p.inject_daemon_cache(_stub_daemon_status(model="fresh"))
        cache = p._get_daemon_cache()
        assert cache.get("model") == "fresh"


# ── runtime_snapshot ─────────────────────────────────────────────────────────


class TestRuntimeSnapshot:
    def test_returns_expected_keys(self) -> None:
        p = _make_plugin(
            daemon_status=_stub_daemon_status(),
            engine=_stub_engine(),
            registry=_stub_registry(),
        )
        result = p._handle_runtime_snapshot()
        expected_keys = {
            "model", "context", "posture", "disclosure",
            "turn", "cache_hit_rate", "uptime", "tools_available",
            "pending_approvals",
        }
        assert set(result.keys()) == expected_keys

    def test_context_format(self) -> None:
        p = _make_plugin(
            daemon_status=_stub_daemon_status(),
            engine=_stub_engine(),
        )
        result = p._handle_runtime_snapshot()
        assert "112K" in result["context"]
        assert "1M" in result["context"]
        assert "11%" in result["context"]

    def test_snapshot_without_deps(self) -> None:
        p = _make_plugin()
        result = p._handle_runtime_snapshot()
        assert result["model"] == ""
        assert result["context"] == "unknown"
        assert result["tools_available"] == 0


# ── PCD metadata ─────────────────────────────────────────────────────────────


class TestPCDMetadata:
    def test_self_describe_schema(self) -> None:
        p = SelfAwarenessPlugin()
        tools = p.tools
        describe_tool = next(t for t in tools if t.name == "self_describe")
        assert describe_tool.x_leapflow["category"] == "system"
        assert describe_tool.x_leapflow["risk_level"] == "read_only"

    def test_runtime_snapshot_schema(self) -> None:
        p = SelfAwarenessPlugin()
        tools = p.tools
        snapshot_tool = next(t for t in tools if t.name == "runtime_snapshot")
        assert snapshot_tool.x_leapflow["category"] == "system"
        assert snapshot_tool.x_leapflow["risk_level"] == "read_only"

    def test_openai_schema_includes_x_leapflow(self) -> None:
        p = SelfAwarenessPlugin()
        tools = p.tools
        for tool in tools:
            schema = tool.to_openai_schema()
            x = schema["function"].get("x_leapflow", {})
            assert x.get("category") == "system"
            assert x.get("risk_level") == "read_only"


# ── Plugin protocol compliance ───────────────────────────────────────────────


class TestPluginProtocol:
    def test_plugin_id(self) -> None:
        p = SelfAwarenessPlugin()
        assert p.plugin_id == "self_awareness"

    def test_category(self) -> None:
        p = SelfAwarenessPlugin()
        assert p.category == "system"

    def test_dependencies(self) -> None:
        p = SelfAwarenessPlugin()
        assert "daemon_client" in p.dependencies

    def test_bind_runtime_engine_weakref(self) -> None:
        """Engine should be stored as a weak reference."""
        p = SelfAwarenessPlugin()
        engine = _stub_engine()
        p.bind_runtime(engine=engine)
        resolved = p._resolve_engine()
        assert resolved is engine

    def test_bind_runtime_none_engine(self) -> None:
        p = SelfAwarenessPlugin()
        p.bind_runtime(engine=None)
        assert p._resolve_engine() is None

    def test_module_level_plugin_exists(self) -> None:
        from leapflow.plugins.tool_plugins.self_awareness import plugin

        assert plugin.plugin_id == "self_awareness"

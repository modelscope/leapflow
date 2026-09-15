# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Cordis P1: dependency-driven fiber activation and bind ordering.

Two behaviours are covered:

1. ``ScopedToolRegistry`` promotes a plugin fiber from LOADING to ACTIVE only
   once every declared dependency is satisfiable (provider fiber ACTIVE or dep
   present in ``last_bound_deps``). Plugins with no declared dependencies keep
   the pre-P1 fast path (PENDING -> ACTIVE, no LOADING). Unsatisfiable/circular
   dependencies fall back to force-activation instead of deadlocking.

2. ``ToolPluginRegistry.bind_runtime`` distributes runtime deps in provider ->
   consumer (topological) order derived from declared inter-plugin dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from leapflow.domain.plugin_fiber import FiberState
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.scoped_registry import ScopedToolRegistry


# ════════════════════════════════════════════════════════════════
# Fakes
# ════════════════════════════════════════════════════════════════


def _noop_handler(**kwargs: Any) -> str:
    return "ok"


def _make_tool(name: str) -> ToolMetadata:
    return ToolMetadata(
        name=name,
        description=f"Test tool: {name}",
        parameters_schema={"type": "object", "properties": {}},
        handler=_noop_handler,
    )


@dataclass
class FakePlugin:
    """Minimal ToolPlugin whose plugin_id doubles as the service it provides."""

    _plugin_id: str
    _deps: list[str] = field(default_factory=list)
    _tools: list[ToolMetadata] = field(default_factory=list)
    _category: str = "test"

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def category(self) -> str:
        return self._category

    @property
    def tools(self) -> list[ToolMetadata]:
        return self._tools

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    def bind_runtime(self, **deps: Any) -> None:
        pass


@dataclass
class RecordingPlugin:
    """Plugin that appends its plugin_id to a shared list on each bind_runtime."""

    _plugin_id: str
    _calls: list[str]
    _deps: list[str] = field(default_factory=list)

    @property
    def plugin_id(self) -> str:
        return self._plugin_id

    @property
    def category(self) -> str:
        return "test"

    @property
    def tools(self) -> list[ToolMetadata]:
        return []

    @property
    def dependencies(self) -> list[str]:
        return self._deps

    def bind_runtime(self, **deps: Any) -> None:
        self._calls.append(self._plugin_id)


# ════════════════════════════════════════════════════════════════
# P1 Item 1 — dependency-driven fiber activation
# ════════════════════════════════════════════════════════════════


def test_dependency_driven_activation_basic() -> None:
    """Provider registered before consumer: both end ACTIVE after adoption."""
    reg = ToolPluginRegistry()
    provider = FakePlugin("service_x", _tools=[_make_tool("x_tool")])
    consumer = FakePlugin("consumer", _deps=["service_x"], _tools=[_make_tool("c_tool")])
    reg.register(provider)
    reg.register(consumer)

    scoped = ScopedToolRegistry(reg)
    scoped.adopt_existing_plugins()

    assert scoped.get_fiber("service_x").state == FiberState.ACTIVE
    assert scoped.get_fiber("consumer").state == FiberState.ACTIVE


def test_dependency_driven_activation_reverse_order() -> None:
    """Consumer registered first stays LOADING until its provider activates."""
    reg = ToolPluginRegistry()
    scoped = ScopedToolRegistry(reg)

    # Consumer arrives first and enters LOADING because its provider is absent.
    consumer = FakePlugin("consumer", _deps=["service_x"])
    fiber_c = scoped.create_fiber("consumer")
    fiber_c.begin_loading()
    scoped.scoped_register(consumer, fiber_c)
    assert fiber_c.state == FiberState.LOADING

    # Provider arrives later. Once it is ACTIVE, the consumer auto-activates.
    provider = FakePlugin("service_x")
    fiber_p = scoped.create_fiber("service_x")
    fiber_p.activate()  # no-dep provider activates immediately
    scoped.scoped_register(provider, fiber_p)

    assert fiber_p.state == FiberState.ACTIVE
    assert fiber_c.state == FiberState.ACTIVE


def test_no_deps_activates_immediately() -> None:
    """A plugin with an empty dependencies list never enters LOADING."""
    reg = ToolPluginRegistry()
    plugin = FakePlugin("standalone", _tools=[_make_tool("s_tool")])
    reg.register(plugin)

    scoped = ScopedToolRegistry(reg)
    scoped.adopt_existing_plugins()

    fiber = scoped.get_fiber("standalone")
    assert fiber.state == FiberState.ACTIVE


def test_circular_deps_fallback(caplog: Any) -> None:
    """Mutually dependent plugins are force-activated (no deadlock), with a warning."""
    reg = ToolPluginRegistry()
    plugin_a = FakePlugin("plugin_a", _deps=["plugin_b"])
    plugin_b = FakePlugin("plugin_b", _deps=["plugin_a"])
    reg.register(plugin_a)
    reg.register(plugin_b)

    scoped = ScopedToolRegistry(reg)
    with caplog.at_level(logging.WARNING, logger="leapflow.plugins.scoped_registry"):
        scoped.adopt_existing_plugins()

    assert scoped.get_fiber("plugin_a").state == FiberState.ACTIVE
    assert scoped.get_fiber("plugin_b").state == FiberState.ACTIVE
    assert any("possible cycle" in rec.message for rec in caplog.records)


def test_external_deps_do_not_warn(caplog: Any) -> None:
    """Late-bound runtime deps (no providing plugin) force-activate quietly."""
    reg = ToolPluginRegistry()
    # 'file_read_gate' is a runtime dep injected later via bind_runtime, not a plugin.
    plugin = FakePlugin("io_plugin", _deps=["file_read_gate"])
    reg.register(plugin)

    scoped = ScopedToolRegistry(reg)
    with caplog.at_level(logging.WARNING, logger="leapflow.plugins.scoped_registry"):
        scoped.adopt_existing_plugins()

    assert scoped.get_fiber("io_plugin").state == FiberState.ACTIVE
    assert not any("possible cycle" in rec.message for rec in caplog.records)


def test_dep_satisfied_by_last_bound_deps() -> None:
    """A dependency already present in last_bound_deps counts as satisfied."""
    reg = ToolPluginRegistry()
    plugin = FakePlugin("needs_mgr", _deps=["memory_manager"])
    reg.register(plugin)
    # Inject the runtime dep before adoption so it is immediately satisfiable.
    reg.bind_runtime(memory_manager=object())

    scoped = ScopedToolRegistry(reg)
    scoped.adopt_existing_plugins()

    assert scoped.get_fiber("needs_mgr").state == FiberState.ACTIVE


# ════════════════════════════════════════════════════════════════
# P1 Item 2 — provider-consumer (topological) bind ordering
# ════════════════════════════════════════════════════════════════


def test_topological_bind_order() -> None:
    """A -> B -> C dependency chain binds C first, then B, then A."""
    reg = ToolPluginRegistry()
    calls: list[str] = []
    # Each plugin also declares the shared runtime dep so bind_runtime visits it.
    plugin_a = RecordingPlugin("A", calls, _deps=["B", "shared"])
    plugin_b = RecordingPlugin("B", calls, _deps=["C", "shared"])
    plugin_c = RecordingPlugin("C", calls, _deps=["shared"])
    # Register in dependent-first (worst) order to prove ordering is not insertion.
    reg.register(plugin_a)
    reg.register(plugin_b)
    reg.register(plugin_c)

    reg.bind_runtime(shared=object())

    assert calls == ["C", "B", "A"]


def test_topological_sort_no_deps() -> None:
    """With no inter-plugin deps, registration order is preserved."""
    reg = ToolPluginRegistry()
    calls: list[str] = []
    reg.register(RecordingPlugin("first", calls, _deps=["shared"]))
    reg.register(RecordingPlugin("second", calls, _deps=["shared"]))
    reg.register(RecordingPlugin("third", calls, _deps=["shared"]))

    reg.bind_runtime(shared=object())

    assert calls == ["first", "second", "third"]


def test_topological_order_helper_direct() -> None:
    """_topological_plugin_order returns providers before consumers."""
    reg = ToolPluginRegistry()
    reg.register(FakePlugin("A", _deps=["B"]))
    reg.register(FakePlugin("B", _deps=["C"]))
    reg.register(FakePlugin("C"))

    order = reg._topological_plugin_order()
    assert order.index("C") < order.index("B") < order.index("A")


def test_topological_order_cycle_fallback(caplog: Any) -> None:
    """A cycle falls back to registration order without raising."""
    reg = ToolPluginRegistry()
    reg.register(FakePlugin("A", _deps=["B"]))
    reg.register(FakePlugin("B", _deps=["A"]))

    with caplog.at_level(logging.WARNING, logger="leapflow.plugins.registry"):
        order = reg._topological_plugin_order()

    assert order == ["A", "B"]
    assert any("Circular inter-plugin dependency" in rec.message for rec in caplog.records)


# ════════════════════════════════════════════════════════════════
# Full boot simulation with dependency ordering
# ════════════════════════════════════════════════════════════════


def test_adopt_existing_plugins_uses_dependency_order() -> None:
    """A full chain adopted in shuffled order still resolves every fiber to ACTIVE."""
    reg = ToolPluginRegistry()
    # Register in an order where consumers precede their providers.
    reg.register(FakePlugin("top", _deps=["mid"], _tools=[_make_tool("top_tool")]))
    reg.register(FakePlugin("mid", _deps=["base"], _tools=[_make_tool("mid_tool")]))
    reg.register(FakePlugin("base", _tools=[_make_tool("base_tool")]))
    reg.register(FakePlugin("independent", _tools=[_make_tool("ind_tool")]))

    scoped = ScopedToolRegistry(reg)
    scoped.adopt_existing_plugins()

    for pid in ("top", "mid", "base", "independent"):
        assert scoped.get_fiber(pid).state == FiberState.ACTIVE, pid

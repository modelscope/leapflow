# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for tool-name conflict arbitration in ToolPluginRegistry.

Tool names are a single global namespace consumed by the provider: two plugins
cannot both expose the same name. The registry keeps the incumbent (first
indexed) and rejects the challenger, recording the rejected claim instead of
silently overwriting the live handler or emitting a duplicate schema. These are
pure single-module invariants, so they belong in the mock layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from leapflow.learning.plugin_stats import PluginUsageTracker
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.registry import CapabilityConflict, ToolPluginRegistry


async def _handler_a(**kwargs: Any) -> dict[str, Any]:
    return {"who": "A"}


async def _handler_b(**kwargs: Any) -> dict[str, Any]:
    return {"who": "B"}


@dataclass
class _FakePlugin:
    """Minimal ToolPlugin satisfying the runtime-checkable Protocol."""

    _plugin_id: str
    _tools: list[ToolMetadata] = field(default_factory=list)
    _category: str = "test"
    _dependencies: list[str] = field(default_factory=list)

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
        return self._dependencies

    def bind_runtime(self, **deps: Any) -> None:
        return None


def _tool(name: str, handler: Any, description: str = "") -> ToolMetadata:
    return ToolMetadata(
        name=name,
        description=description or f"tool {name}",
        parameters_schema={"type": "object", "properties": {}},
        handler=handler,
    )


def _plugin(plugin_id: str, name: str, handler: Any, description: str = "") -> _FakePlugin:
    return _FakePlugin(_plugin_id=plugin_id, _tools=[_tool(name, handler, description)])


def _schema_names(reg: ToolPluginRegistry) -> list[str]:
    return [d["function"]["name"] for d in reg.tool_definitions]


def test_no_conflict_when_names_unique() -> None:
    """Distinct tool names assemble cleanly with no conflict records."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "tool_a", _handler_a))
    reg.register(_plugin("plugin_b", "tool_b", _handler_b))
    reg.assemble()

    assert sorted(_schema_names(reg)) == ["tool_a", "tool_b"]
    assert reg.conflicts == []


def test_capability_catalog_lazily_assembles_registry() -> None:
    """Cold registry introspection must not report an empty tool catalog."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "tool_a", _handler_a))

    catalog = reg.capability_catalog()

    assert [entry["function"]["name"] for entry in catalog] == ["tool_a"]


def test_duplicate_name_keeps_incumbent_and_rejects_challenger() -> None:
    """First plugin to claim a name wins; the later one is rejected, not merged."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "dup", _handler_a, "from a"))
    reg.register(_plugin("plugin_b", "dup", _handler_b, "from b"))
    reg.assemble()

    # Exactly one schema is emitted for the shared name -- no duplicate reaches
    # the provider, and metadata is not double-counted.
    assert _schema_names(reg).count("dup") == 1
    assert sum(1 for m in reg.all_metadata if m.name == "dup") == 1

    # The incumbent's handler stays live; the challenger never overwrites it.
    assert reg.tool_handlers["dup"] is _handler_a

    conflicts = reg.conflicts
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert isinstance(conflict, CapabilityConflict)
    assert conflict.tool_name == "dup"
    assert conflict.kept_plugin == "plugin_a"
    assert conflict.rejected_plugin == "plugin_b"


def test_removing_rejected_challenger_preserves_incumbent_tool() -> None:
    """Disposing the loser must not tear down the winner's live handler.

    The challenger's plugin lists ``dup`` in its tools, but it does not own the
    live name; unregistering it by name would otherwise remove the incumbent.
    """
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "dup", _handler_a))
    reg.register(_plugin("plugin_b", "dup", _handler_b))
    reg.assemble()

    assert reg.unregister_plugin("plugin_b") is True

    assert "dup" in reg.tool_handlers
    assert reg.tool_handlers["dup"] is _handler_a
    # The conflict record referencing the removed plugin is purged.
    assert reg.conflicts == []


def test_removing_incumbent_removes_the_tool() -> None:
    """Disposing the owner removes the name from the live catalog."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "dup", _handler_a))
    reg.register(_plugin("plugin_b", "dup", _handler_b))
    reg.assemble()

    assert reg.unregister_plugin("plugin_a") is True

    assert "dup" not in reg.tool_handlers
    assert "dup" not in _schema_names(reg)
    assert reg.conflicts == []


def test_conflict_is_non_fatal_other_tools_survive() -> None:
    """One colliding tool must not break assembly for the rest of a plugin."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "shared", _handler_a))
    challenger = _FakePlugin(
        _plugin_id="plugin_b",
        _tools=[_tool("shared", _handler_b), _tool("unique_b", _handler_b)],
    )
    reg.register(challenger)
    reg.assemble()

    assert reg.tool_handlers["shared"] is _handler_a
    assert "unique_b" in reg.tool_handlers  # sibling tool still published
    assert [c.tool_name for c in reg.conflicts] == ["shared"]


def test_late_tool_cannot_shadow_a_plugin_tool() -> None:
    """register_late_tool obeys the same first-wins arbitration."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "dup", _handler_a))
    reg.assemble()

    late_def = {
        "type": "function",
        "function": {"name": "dup", "description": "late", "parameters": {}},
    }
    reg.register_late_tool(late_def, _handler_b, "dup")

    assert reg.tool_handlers["dup"] is _handler_a
    assert _schema_names(reg).count("dup") == 1
    assert [(c.kept_plugin, c.rejected_plugin) for c in reg.conflicts] == [
        ("plugin_a", "late_tool")
    ]


def test_late_tool_registers_when_name_is_free() -> None:
    """A late tool with a fresh name is published and owns its name."""
    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "tool_a", _handler_a))
    reg.assemble()

    late_def = {
        "type": "function",
        "function": {"name": "session_search", "description": "late", "parameters": {}},
    }
    reg.register_late_tool(late_def, _handler_b, "session_search")

    assert "session_search" in reg.tool_handlers
    assert reg.conflicts == []


def test_usage_tracker_uses_live_tool_owner_after_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage/trust must accrue to the first-wins owner, not the rejected plugin."""
    import leapflow.plugins as plugins_module

    reg = ToolPluginRegistry()
    reg.register(_plugin("plugin_a", "dup", _handler_a))
    reg.register(_plugin("plugin_b", "dup", _handler_b))
    reg.assemble()
    monkeypatch.setattr(plugins_module, "_registry", reg)

    ledger = PluginTrustLedger(candidate_at=1)
    tracker = PluginUsageTracker()
    tracker.set_trust_ledger(ledger)

    tracker.record("dup", ok=True, duration_ms=1.0)

    assert ledger.level("plugin_a") is PluginTrustLevel.CANDIDATE
    assert ledger.level("plugin_b") is PluginTrustLevel.DRAFT
    assert tracker.stats_for_plugin("plugin_a") is not None
    assert tracker.stats_for_plugin("plugin_b") is None

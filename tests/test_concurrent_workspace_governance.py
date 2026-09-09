"""F7 / F8: concurrent workspaces, and the fiber lifecycle of the changed contracts.

**F7 (MANDATORY).** This work introduced process-global governance state, so AGENTS.md
requires two sessions in two workspaces asserting that neither sees the other's
identity, usage or turn state.

The design question it forced, answered here rather than deferred: *should workspace
A's failures be able to quarantine a plugin serving workspace B?* **Yes** — and it is
not a compromise. Plugins are process-global; a plugin that keeps failing is broken as
*code*, not "broken for workspace A". Trust already works exactly this way
(`PluginTrustLedger` is process-level), so making the quarantine streak per-workspace
would have made quarantine disagree with the trust it is supposed to escalate.

What was missing was not isolation but **attribution**: a cross-workspace quarantine
has to be auditable rather than mysterious. So the contributing workspaces travel with
the decision and appear in its trace, and the tests below pin both halves — the shared
streak *and* the absence of any session/identity leak.

**F8.** The changed Protocols (`evolution_contracts`, `AdaptiveEvolutionPolicy`) alter
what the agent can load, so AGENTS.md requires exercising register → publish → reload →
dispose against a real registry rather than a fake.
"""

from __future__ import annotations

import asyncio

from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.domain.evolution_trace import EvolutionTrace
from leapflow.evolution.observations import (
    CoevolutionObservations,
    install_observations,
)
from leapflow.evolution.sweep import CoevolutionSweep
from leapflow.learning.outcome_governance_feed import QuarantineCandidateTracker
from leapflow.telemetry import evolution_tap

WS_A = "/work/alpha"
WS_B = "/work/beta"


class _Sink:
    def __init__(self) -> None:
        self.traces: list[EvolutionTrace] = []

    def record(self, trace: EvolutionTrace) -> None:
        self.traces.append(trace)

    def of(self, kind: str):
        return [t for t in self.traces if t.kind == kind]


class _Governor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def record_outcome(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"action": "quarantine", "trust_level": "DRAFT"})()


def _requirement():
    return EvolutionIntent.create(
        "chat.reply", "gap", expected_effect="the reply appears"
    ).to_requirement()


def _drive_engine_outcome(item, workspace):
    from leapflow.engine.engine import AgentEngine

    AgentEngine._record_coevolution_outcome(item, workspace)


def _registry_for(tool_name, plugin_id):
    class _Reg:
        tool_owners = {tool_name: plugin_id}

    return _Reg()


# ── F7: two workspaces, one process-global plugin ─────────────────────────────


def test_both_workspaces_are_attributed_to_one_shared_streak(monkeypatch):
    """The intended semantics, made explicit: shared streak, named contributors."""
    buf = CoevolutionObservations()
    install_observations(buf)
    monkeypatch.setattr(
        "leapflow.plugins.get_registry", lambda: _registry_for("gen_tool", "gen_p")
    )
    try:
        buf.record_acquisition("gen_p")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_p")

        _drive_engine_outcome({"name": "gen_tool", "result": {"ok": False}}, WS_A)
        _drive_engine_outcome({"name": "gen_tool", "result": {"ok": False}}, WS_B)

        assert buf.contributing_workspaces("gen_p") == (WS_A, WS_B)
    finally:
        install_observations(None)


def test_a_cross_workspace_quarantine_is_flagged_in_its_trace(monkeypatch):
    """The audit must be able to see that another workspace drove the decision."""
    buf = CoevolutionObservations()
    install_observations(buf)
    sink = _Sink()
    evolution_tap.install_sink(sink)
    monkeypatch.setattr(
        "leapflow.plugins.get_registry", lambda: _registry_for("gen_tool", "gen_p")
    )
    try:
        buf.record_acquisition("gen_p")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_p")
        _drive_engine_outcome({"name": "gen_tool", "result": {"ok": False}}, WS_A)
        _drive_engine_outcome({"name": "gen_tool", "result": {"ok": False}}, WS_B)

        tracker = QuarantineCandidateTracker(quarantine_after=1)
        tracker.record("gen_p", "gen_tool", ok=False)
        asyncio.run(CoevolutionSweep(governor=_Governor(), tracker=tracker).run())

        detail = sink.of("quarantine_drain")[0].detail
        assert detail["cross_workspace"] is True
        assert detail["contributing_workspaces"] == [WS_A, WS_B]
    finally:
        evolution_tap.install_sink(None)
        install_observations(None)


def test_single_workspace_quarantine_is_not_flagged_as_cross(monkeypatch):
    buf = CoevolutionObservations()
    install_observations(buf)
    sink = _Sink()
    evolution_tap.install_sink(sink)
    monkeypatch.setattr(
        "leapflow.plugins.get_registry", lambda: _registry_for("gen_tool", "gen_p")
    )
    try:
        buf.record_acquisition("gen_p")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_p")
        _drive_engine_outcome({"name": "gen_tool", "result": {"ok": False}}, WS_A)

        tracker = QuarantineCandidateTracker(quarantine_after=1)
        tracker.record("gen_p", "gen_tool", ok=False)
        asyncio.run(CoevolutionSweep(governor=_Governor(), tracker=tracker).run())

        detail = sink.of("quarantine_drain")[0].detail
        assert detail["cross_workspace"] is False
        assert detail["contributing_workspaces"] == [WS_A]
    finally:
        evolution_tap.install_sink(None)
        install_observations(None)


def test_no_session_or_client_identity_leaks_through_governance(monkeypatch):
    """The actual isolation contract: usage and turn state must not cross.

    Attribution carries a *workspace path*, which is what makes a shared decision
    auditable. It must not carry a session id, client id, turn id, or conversation
    content -- those are per-client identity that one workspace may never learn about
    another.
    """
    buf = CoevolutionObservations()
    install_observations(buf)
    sink = _Sink()
    evolution_tap.install_sink(sink)
    monkeypatch.setattr(
        "leapflow.plugins.get_registry", lambda: _registry_for("gen_tool", "gen_p")
    )
    try:
        buf.record_acquisition("gen_p")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen_p")
        for ws in (WS_A, WS_B):
            _drive_engine_outcome(
                {
                    "name": "gen_tool",
                    "arguments": {"secret": "workspace-private argument"},
                    "result": {"ok": False, "error": "boom"},
                },
                ws,
            )

        tracker = QuarantineCandidateTracker(quarantine_after=1)
        tracker.record("gen_p", "gen_tool", ok=False)
        asyncio.run(CoevolutionSweep(governor=_Governor(), tracker=tracker).run())

        blob = repr([t.to_dict() for t in sink.traces]) + repr(buf.stats())
        for forbidden in ("session_id", "client_id", "turn_id", "workspace-private"):
            assert forbidden not in blob
    finally:
        evolution_tap.install_sink(None)
        install_observations(None)


def test_verdicts_do_not_depend_on_which_workspace_called(monkeypatch):
    """Same plugin id means the same code; attribution must not change a verdict."""
    monkeypatch.setattr(
        "leapflow.plugins.get_registry", lambda: _registry_for("gen_tool", "gen_p")
    )
    verdicts = []
    for ws in (WS_A, WS_B, ""):
        buf = CoevolutionObservations()
        install_observations(buf)
        try:
            buf.record_acquisition("gen_p")
            buf.record_resolution(requirement=_requirement(), selected_plugin="gen_p")
            _drive_engine_outcome(
                {"name": "gen_tool", "result": {"ok": True, "effect": "the reply appears"}}, ws
            )
            outcome = asyncio.run(
                CoevolutionSweep().run(verifications=buf.drain_verifications())
            )
            verdicts.append((outcome.verified, outcome.refuted, outcome.unverifiable))
        finally:
            install_observations(None)
    assert len(set(verdicts)) == 1, verdicts


def test_attribution_is_bounded_by_plugin_not_by_session():
    """Long-lived multi-workspace processes must not grow attribution without bound."""
    buf = CoevolutionObservations()
    install_observations(buf)
    try:
        for i in range(500):
            buf.record_tool_outcome("p", "t", ok=True, workspace=f"/ws/{i % 3}")
        assert len(buf.contributing_workspaces("p")) == 3
    finally:
        install_observations(None)


# ── F8: register → publish → reload → dispose on a real registry ──────────────


def test_changed_contracts_survive_a_real_fiber_lifecycle(tmp_path):
    """Exercise the real registry, not a fake, per AGENTS.md.

    The Protocols this programme changed (`evolution_contracts`, the policy's
    dependency surface) affect what the agent can load, so the check that matters is
    that a plugin carrying them can complete the whole fiber lifecycle and leave
    nothing registered behind.
    """
    from leapflow.plugins.registry import ToolPluginRegistry
    from leapflow.plugins.scoped_registry import ScopedToolRegistry

    module = tmp_path / "eff_plugin.py"
    module.write_text(
        "from typing import Any\n"
        "from leapflow.plugins.protocol import ToolMetadata\n"
        "class P:\n"
        "    @property\n"
        "    def plugin_id(self): return 'eff_p'\n"
        "    @property\n"
        "    def category(self): return 'custom'\n"
        "    @property\n"
        "    def dependencies(self): return []\n"
        "    def bind_runtime(self, **deps: Any) -> None: pass\n"
        "    @property\n"
        "    def tools(self):\n"
        "        return [ToolMetadata(name='eff_tool', description='d',\n"
        "            parameters_schema={'type':'object','properties':{}},\n"
        "            handler=self._h,\n"
        "            x_leapflow={'category':'custom','risk_level':'read_only'})]\n"
        "    async def _h(self, **kw: Any) -> dict:\n"
        "        return {'ok': True, 'effect': 'the reply appears'}\n"
        "plugin = P()\n",
        encoding="utf-8",
    )

    registry = ToolPluginRegistry()
    scoped = ScopedToolRegistry(registry)
    spec = _load(module)
    # Register *through* the scoped registry so the plugin gets a real fiber and a real
    # effect scope -- the thing whose lifecycle is under test.
    fiber = scoped.create_fiber("eff_p")
    scoped.scoped_register(spec, fiber)
    registry.assemble()

    assert "eff_tool" in registry.tool_handlers
    assert registry.tool_owners.get("eff_tool") == "eff_p"
    version_after_publish = registry.version

    # A handler that reports its effect is what makes C-1 verification possible, so
    # assert the contract survives the round trip rather than just that reload ran.
    result = asyncio.run(registry.tool_handlers["eff_tool"]())
    assert result["effect"] == "the reply appears"

    reloaded = scoped.reload("eff_p")
    assert reloaded.plugin_id == "eff_p"
    registry.assemble()
    assert "eff_tool" in registry.tool_handlers
    assert registry.version >= version_after_publish

    # The effect contract must survive re-import, or C-1 silently stops working after
    # the first hot reload.
    result = asyncio.run(registry.tool_handlers["eff_tool"]())
    assert result["effect"] == "the reply appears"

    disposed = scoped.dispose_plugin("eff_p", prune_metadata=True)
    assert disposed.plugin_id == "eff_p"
    registry.assemble()
    assert "eff_tool" not in registry.tool_handlers
    assert "eff_p" not in registry.tool_owners.values()


def _load(path):
    """Import a plugin module by path, recording the source for file-backed reload."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = module.plugin
    setattr(plugin, "__leapflow_plugin_path__", str(path))
    return plugin

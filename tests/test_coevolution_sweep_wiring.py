"""Phase A: the co-evolution capabilities are wired, and the wiring is driven.

The prior round shipped `CapabilityEffectVerifier`, `QuarantineCandidateTracker`
and `UnselectableArtifactReaper` with **zero production callers** — which the
evolution dashboard reported as three `NO_EVIDENCE` rows rather than treating the
modules' presence as proof. This suite closes that.

Per AGENTS.md, a test whose purpose is wiring must construct the real object and
drive the production path, so the context tests below call
`_run_coevolution_sweep` on a real `CoevolutionSweep` and assert the governed
effects — not the collaborators.
"""

from __future__ import annotations

import asyncio

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_trace import EvolutionStage, EvolutionTrace
from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.evolution.sweep import CoevolutionSweep, SweepOutcome
from leapflow.learning.outcome_governance_feed import QuarantineCandidateTracker
from leapflow.telemetry import evolution_tap


class _CapturingSink:
    """Minimal EvolutionTraceSink that records what the sweep emitted."""

    def __init__(self) -> None:
        self.traces: list[EvolutionTrace] = []

    def record(self, trace: EvolutionTrace) -> None:
        self.traces.append(trace)

    def kinds(self) -> set[str]:
        return {t.kind for t in self.traces}

    def of(self, kind: str) -> list[EvolutionTrace]:
        return [t for t in self.traces if t.kind == kind]


class _Governor:
    def __init__(self, action: str = "probation_execute") -> None:
        self.calls: list[dict] = []
        self._action = action

    async def record_outcome(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"action": self._action, "trust_level": "DRAFT"})()


def _requirement(expected: str = "the reply appears in the thread"):
    return EvolutionIntent.create(
        "chat.reply", "send path no-ops", expected_effect=expected
    ).to_requirement()


def _sink():
    sink = _CapturingSink()
    evolution_tap.install_sink(sink)
    return sink


def _teardown():
    evolution_tap.install_sink(None)


# ── every branch emits, including the no-ops ──────────────────────────────────


def test_empty_sweep_still_records_its_no_op_branches():
    """A quiet sweep must be distinguishable from a sweep that never ran."""
    sink = _sink()
    try:
        outcome = asyncio.run(CoevolutionSweep().run())
        assert outcome == SweepOutcome()
        # All three segments reported, each flagged as a no-op.
        assert sink.kinds() == {"effect_verification", "quarantine_drain", "reclamation"}
        assert all(t.detail.get("no_op") for t in sink.traces)
    finally:
        _teardown()


def test_verified_effect_is_recorded_and_feeds_trust():
    sink = _sink()
    governor = _Governor()
    try:
        outcome = asyncio.run(CoevolutionSweep(governor=governor).run(
            verifications=[(
                _requirement(),
                {"ok": True, "observed_effect": "the reply appears in the thread"},
                "gen1",
            )],
        ))
        assert outcome.verified == 1 and outcome.refuted == 0
        assert governor.calls[0]["ok"] is True
        trace = sink.of("effect_verification")[0]
        assert trace.stage is EvolutionStage.LEARN
        assert trace.detail["verified"] is True
        assert trace.correlation["plugin_id"] == "gen1"
    finally:
        _teardown()


def test_refuted_effect_feeds_a_failure_even_though_the_call_succeeded():
    """The point of WM-6: ok-but-no-effect must not be recorded as success."""
    sink = _sink()
    governor = _Governor()
    try:
        outcome = asyncio.run(CoevolutionSweep(governor=governor).run(
            verifications=[(
                _requirement(), {"ok": True, "observed_effect": "nothing happened"}, "gen1",
            )],
        ))
        assert outcome.refuted == 1
        assert governor.calls[0]["ok"] is False
        assert governor.calls[0]["failure_class"] == "expected_effect_absent"
        assert sink.of("effect_verification")[0].detail["verified"] is False
    finally:
        _teardown()


def test_unverifiable_verdict_never_touches_trust():
    """A missing declaration must not quarantine a healthy plugin."""
    sink = _sink()
    governor = _Governor()
    try:
        bare = CapabilityRequirement.create("chat.reply", "unknown_tool")
        outcome = asyncio.run(CoevolutionSweep(governor=governor).run(
            verifications=[(bare, {"ok": True, "observed_effect": "sent"}, "gen1")],
        ))
        assert outcome.unverifiable == 1
        assert governor.calls == []          # nothing recorded
        assert sink.of("effect_verification")[0].detail["verified"] is None
    finally:
        _teardown()


def test_quarantine_candidate_is_drained_and_recorded():
    sink = _sink()
    governor = _Governor(action="quarantine")
    tracker = QuarantineCandidateTracker(quarantine_after=2)
    try:
        tracker.record("bad", "bad_tool", ok=False)
        tracker.record("bad", "bad_tool", ok=False)   # crosses the threshold
        assert tracker.pending() == 1

        outcome = asyncio.run(CoevolutionSweep(governor=governor, tracker=tracker).run())
        assert len(outcome.quarantined) == 1
        assert tracker.pending() == 0                 # cleared by the drain
        trace = sink.of("quarantine_drain")[0]
        assert trace.stage is EvolutionStage.ACT
        assert trace.detail["action"] == "quarantine"
    finally:
        _teardown()


def test_reclamation_candidate_is_recorded_as_residue():
    sink = _sink()
    try:
        resolutions = [
            {"selected_plugin": "incumbent", "exclusions": {"gen_overrisk": ["risk_cost"]}}
            for _ in range(3)
        ]
        outcome = asyncio.run(CoevolutionSweep().run(
            acquired_plugin_ids=["gen_overrisk"], resolutions=resolutions,
        ))
        assert [c.plugin_id for c in outcome.reclamation] == ["gen_overrisk"]
        trace = sink.of("reclamation")[0]
        assert trace.correlation["plugin_id"] == "gen_overrisk"
        assert "never selected" in trace.summary
    finally:
        _teardown()


def test_verification_failure_can_feed_the_same_sweep_ordering():
    """Verification runs before the drain, so a refuted verdict is governed first."""
    governor = _Governor()
    tracker = QuarantineCandidateTracker(quarantine_after=1)
    _sink()
    try:
        tracker.record("other", "other_tool", ok=False)
        asyncio.run(CoevolutionSweep(governor=governor, tracker=tracker).run(
            verifications=[(_requirement(), {"ok": False}, "gen1")],
        ))
        assert [c["plugin_id"] for c in governor.calls] == ["gen1", "other"]
    finally:
        _teardown()


def test_governance_failure_does_not_break_the_sweep():
    class _Broken:
        async def record_outcome(self, **kwargs):
            raise OSError("store down")

    _sink()
    try:
        outcome = asyncio.run(CoevolutionSweep(governor=_Broken()).run(
            verifications=[(_requirement(), {"ok": False}, "gen1")],
        ))
        assert outcome.refuted == 1          # the verdict still stands
    finally:
        _teardown()


def test_sweep_is_inert_when_tracing_is_disabled():
    """Observability must never affect the observed."""
    evolution_tap.install_sink(None)
    outcome = asyncio.run(CoevolutionSweep().run())
    assert outcome == SweepOutcome()


# ── the production wiring, driven (AGENTS.md: do not fabricate the wiring) ─────


class _Ctx:
    """Bind the real production methods onto a minimal host.

    Deliberately not `object.__new__` on the real context plus private-attribute
    assignment: that pattern cannot detect a wrong attribute *name*. These bind the
    actual unbound functions from `Context`, so the assertions run the same
    code a session-end runs.
    """

    def __init__(self, **attrs) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)

    async def run_sweep(self):
        from leapflow.cli.context import Context

        return await Context._run_coevolution_sweep(self)


def test_production_sweep_hook_builds_and_runs_a_real_sweep():
    """Drives `_run_coevolution_sweep` itself, not a hand-made CoevolutionSweep.

    Inputs arrive through the process observation buffer, which is where the engine,
    the install path and the tool-outcome sink deposit them in production.
    """
    from leapflow.evolution.observations import CoevolutionObservations, install_observations

    sink = _sink()
    governor = _Governor(action="quarantine")
    tracker = QuarantineCandidateTracker(quarantine_after=1)
    tracker.record("bad", "bad_tool", ok=False)

    buf = CoevolutionObservations()
    install_observations(buf)
    requirement = _requirement()
    buf.record_acquisition("gen1")
    buf.record_acquisition("gen_overrisk")
    buf.record_resolution(requirement=requirement, selected_plugin="gen1")
    buf.record_tool_outcome("gen1", "gen1_tool", ok=True, observed_effect="nothing happened")
    for _ in range(3):
        buf.record_resolution(
            requirement=requirement,
            selected_plugin="x",
            exclusions={"gen_overrisk": ["risk_cost"]},
        )
    try:
        ctx = _Ctx(lifecycle_governor=governor, _quarantine_tracker=tracker)
        outcome = asyncio.run(ctx.run_sweep())

        assert outcome is not None
        assert outcome.refuted == 1                                  # WM-6 wired
        assert len(outcome.quarantined) == 1                         # A-4 wired
        assert [c.plugin_id for c in outcome.reclamation] == ["gen_overrisk"]  # LF-10 wired
        # All three dashboard segments now have observed output.
        assert sink.kinds() == {"effect_verification", "quarantine_drain", "reclamation"}
        assert outcome.to_dict()["quarantined"] == 1
        # Verifications were drained, so a second sweep cannot double-govern them.
        assert buf.drain_verifications() == ()
    finally:
        install_observations(None)
        _teardown()


def test_production_hook_uses_the_shared_process_tracker():
    from leapflow.evolution.observations import CoevolutionObservations, install_observations

    _sink()
    install_observations(CoevolutionObservations())
    try:
        ctx = _Ctx(lifecycle_governor=_Governor())
        outcome = asyncio.run(ctx.run_sweep())
        assert outcome is not None
        assert outcome.to_dict()["quarantined"] == 0
    finally:
        install_observations(None)
        _teardown()


def test_production_hook_survives_a_broken_collaborator():
    """A failure to govern must not fail the session that produced the trajectory."""
    _sink()
    try:
        ctx = _Ctx(lifecycle_governor=object())
        outcome = asyncio.run(ctx.run_sweep())
        assert outcome is None or isinstance(outcome, SweepOutcome)
    finally:
        _teardown()


# ── P5 enforcement, wired into the gap gate ───────────────────────────────────


def _loop(tmp_path):
    from leapflow.plugins.adaptive_loop import AdaptivePluginLoop
    from leapflow.plugins.registry import ToolPluginRegistry
    from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore

    return AdaptivePluginLoop(
        registry=ToolPluginRegistry(),
        plan_store=JsonCapabilityPlanStore(tmp_path / "plans.json"),
    )


def _env():
    from leapflow.domain.environment_fingerprint import EnvironmentFingerprint

    return EnvironmentFingerprint(
        platform_id="linux_gnome", os_version="x", platform_capabilities=(), workspace_root="/w"
    )


def test_gap_gate_is_unrestricted_by_default(tmp_path):
    """Shipped behaviour: any origin may drive acquisition."""
    loop = _loop(tmp_path)
    shipped = CapabilityRequirement.create("list_dir", "unknown_tool")
    unmet = loop.unmet_requirements([shipped], _env())
    assert [r.origin for r in unmet] == ["unknown_tool"]


def test_gap_gate_excludes_unauthorised_origins(tmp_path):
    """The goal, enforceable: only world-model requirements reach the gap set."""
    loop = _loop(tmp_path)
    wm = EvolutionIntent.create("chat.reply", "gap").to_requirement()
    shipped = CapabilityRequirement.create("list_dir", "unknown_tool")

    unmet = loop.unmet_requirements([wm, shipped], _env(), authorising_origins=("world_model",))
    assert [r.origin for r in unmet] == ["world_model"]

    # And with nothing authorised, the gate is empty rather than permissive.
    assert loop.unmet_requirements([shipped], _env(), authorising_origins=("world_model",)) == ()

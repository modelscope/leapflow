# Copyright (c) Alibaba, Inc. and its affiliates.
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
import time
from pathlib import Path

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
        # All four segments reported, each flagged as a no-op.
        assert sink.kinds() == {"effect_verification", "quarantine_drain", "proposal_expiry", "reclamation"}
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

    def _resolve_lifecycle_governor(self):
        """Bound because the production hook resolves the governor through it.

        The hook used to read ``self.lifecycle_governor`` directly -- an attribute nothing
        in production ever assigned, so every session swept with ``governor=None``. It now
        goes through a resolver that honours an injected governor first and builds one from
        the profile layout otherwise, and a host driving the real hook has to expose the
        same surface or it would exercise the broad ``except`` instead of the code.
        """
        from leapflow.cli.context import Context

        return Context._resolve_lifecycle_governor(self)

    def _active_proposal_ids(self):
        """Bound because the production hook maps plugin_id -> proposal_id through it.

        The sweep now feeds ``LifecycleGovernor.record_outcome`` a proposal id keyed off
        the live queue, so the real hook calls this. With no profile layout on the double
        it degrades to an empty map -- the honest 'no queued proposals' state -- and the
        assertions still run the real ``_run_coevolution_sweep`` body.
        """
        from leapflow.cli.context import Context

        return Context._active_proposal_ids(self)


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
        # All four dashboard segments now have observed output.
        assert sink.kinds() == {"effect_verification", "quarantine_drain", "proposal_expiry", "reclamation"}
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


def test_active_proposal_ids_targets_the_newest_record_for_a_plugin(tmp_path):
    """One plugin, several active records: governance must target the newest.

    ``active()`` is newest-first, so a naive overwrite would leave the map pointing at
    the *oldest* record and ``record_outcome`` would update the wrong lifecycle entry.
    A plugin can legitimately hold several active records (different capability,
    environment, or source), so this is a reachable case, not a corner one.
    """
    from types import SimpleNamespace

    from leapflow.cli.context import Context
    from leapflow.domain.capability_requirement import CapabilityRequirement
    from leapflow.layout import ProfileLayout
    from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
    from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore

    layout = ProfileLayout(root=tmp_path / "profile", profile_id="p")
    layout.root.mkdir(parents=True, exist_ok=True)
    queue = EvolutionCapabilityProposalStore(
        DuckDBEvolutionEventStore(tmp_path / "events.duckdb"), profile_id="p"
    )
    queue.enqueue(
        requirements=(
            CapabilityRequirement.create("chat.reply", "world_model", requirement_id="req-a"),
        ),
        metadata={"plugin_id": "shared_plugin"},
    )
    newer = queue.enqueue(
        requirements=(
            CapabilityRequirement.create("chat.send", "world_model", requirement_id="req-b"),
        ),
        metadata={"plugin_id": "shared_plugin"},
    )
    # Bump the second record so it is unambiguously the most recently touched.
    queue.update(newer.proposal_id, status="GENERATED")

    ctx = _Ctx(
        settings=SimpleNamespace(profile_layout=layout),
        _capability_proposal_queue=queue,
    )
    mapping = Context._active_proposal_ids(ctx)
    assert mapping["shared_plugin"] == newer.proposal_id


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


# ── proposal expiry sweep tests ───────────────────────────────────────────────


def _proposal_queue(tmp_path: Path, *, ttl_hours: int = 72):
    from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
    from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore

    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    return EvolutionCapabilityProposalStore(events, profile_id="sweep-p", proposal_ttl_hours=ttl_hours)


def _orchestrator(queue):
    from leapflow.evolution.artifact_store import ContentAddressedArtifactStore
    from leapflow.plugins.adaptive_policy import AdaptiveEvolutionPolicy
    from leapflow.plugins.proposal_orchestrator import ProposalOrchestrator
    import tempfile

    return ProposalOrchestrator(
        queue=queue,
        artifact_store=ContentAddressedArtifactStore(Path(tempfile.mkdtemp()) / "artifacts"),
        approval_gate=None,
        policy=AdaptiveEvolutionPolicy(autonomy_level="generate_only"),
    )


def test_sweep_expires_stale_proposals(tmp_path: Path):
    """A proposal with expires_at in the past is swept to EXPIRED."""
    queue = _proposal_queue(tmp_path / "expire", ttl_hours=0)
    # Create with an explicit past expires_at via low-level update
    queue.enqueue(
        requirements=(CapabilityRequirement.create("chat.reply", "world_model", requirement_id="req-exp"),),
    )
    # Manually set expires_at in the past by re-creating with occurred_at far back
    # Since ttl_hours=0 means expires_at=None, we need a different approach.
    # Use a queue with ttl_hours=1, but create with occurred_at far in the past.
    queue2 = _proposal_queue(tmp_path / "expire2", ttl_hours=1)
    queue2.enqueue(
        requirements=(CapabilityRequirement.create("chat.stale", "world_model", requirement_id="req-stale"),),
    )
    # The item was created "now" with expires_at = now + 3600. Force expiry by
    # creating a proposal with occurred_at far in the past.
    queue3 = _proposal_queue(tmp_path / "expire3", ttl_hours=1)
    past_item, ev = queue3.prepare_enqueue(
        requirements=(CapabilityRequirement.create("chat.old", "world_model", requirement_id="req-old"),),
        occurred_at=1.0,  # epoch second 1 = way in the past
    )
    assert ev is not None
    queue3._event_store.append(ev)
    assert past_item.expires_at is not None
    assert past_item.expires_at < time.time()  # Definitely expired

    orch = _orchestrator(queue3)
    try:
        outcome = asyncio.run(
            CoevolutionSweep(
                orchestrator=orch, proposal_store=queue3,
            ).run()
        )
        assert outcome.expired == 1
        refreshed = queue3.get(past_item.proposal_id)
        assert refreshed is not None
        assert refreshed.status == "EXPIRED"
        assert refreshed.metadata["terminal_reason"] == "ttl_exceeded"
    finally:
        _teardown()


def test_sweep_supersedes_outdated_proposal(tmp_path: Path):
    """Two proposals with same requirements: the older is SUPERSEDED."""
    queue = _proposal_queue(tmp_path / "supersede", ttl_hours=0)  # no TTL expiry
    older, ev_old = queue.prepare_enqueue(
        requirements=(CapabilityRequirement.create("chat.reply", "world_model", requirement_id="req-a"),),
        occurred_at=100.0,
    )
    assert ev_old is not None
    queue._event_store.append(ev_old)

    newer, ev_new = queue.prepare_enqueue(
        requirements=(CapabilityRequirement.create("chat.reply", "world_model", requirement_id="req-a"),),
        environment={"fingerprint_id": "different"},  # different env => different proposal_id
        occurred_at=200.0,
    )
    assert ev_new is not None
    queue._event_store.append(ev_new)
    assert older.proposal_id != newer.proposal_id

    orch = _orchestrator(queue)
    try:
        outcome = asyncio.run(
            CoevolutionSweep(
                orchestrator=orch, proposal_store=queue,
            ).run()
        )
        assert outcome.superseded == 1
        old_item = queue.get(older.proposal_id)
        assert old_item is not None
        assert old_item.status == "SUPERSEDED"
        new_item = queue.get(newer.proposal_id)
        assert new_item is not None
        assert new_item.status == "PENDING"  # newer remains active
    finally:
        _teardown()


def test_proposal_without_ttl_not_expired(tmp_path: Path):
    """A proposal with expires_at=None is not touched by TTL sweep."""
    queue = _proposal_queue(tmp_path / "no_ttl", ttl_hours=0)  # expires_at=None
    item = queue.enqueue(
        requirements=(CapabilityRequirement.create("chat.reply", "world_model", requirement_id="req-no-ttl"),),
    )
    assert item.expires_at is None

    orch = _orchestrator(queue)
    try:
        outcome = asyncio.run(
            CoevolutionSweep(
                orchestrator=orch, proposal_store=queue,
            ).run()
        )
        assert outcome.expired == 0
        assert outcome.superseded == 0
        refreshed = queue.get(item.proposal_id)
        assert refreshed is not None
        assert refreshed.status == "PENDING"  # unchanged
    finally:
        _teardown()


def test_sweep_proposal_expiry_cold_path():
    """Proposal expiry lives only in CoevolutionSweep.run(), never per-turn."""
    import ast
    import inspect
    import textwrap
    from leapflow.evolution.sweep import CoevolutionSweep

    source = textwrap.dedent(inspect.getsource(CoevolutionSweep.run))
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and "expir" in getattr(node.func, "attr", "").lower()
    ]
    assert "_sweep_proposal_expiry" in calls, (
        "proposal_expiry must be called inside CoevolutionSweep.run()"
    )

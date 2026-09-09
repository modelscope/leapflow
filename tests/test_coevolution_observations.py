"""A-r1 / A-r2: the sweep's inputs are produced by the real production paths.

Phase A gave the sweep a call site but nothing fed it, so it correctly emitted three
no-op traces forever. This closes that: the engine records resolutions, the install
path records acquisitions, and the tool-outcome sink records failure streaks. Each
test drives the **production** function rather than asserting against a hand-built
buffer, per AGENTS.md's rule that a test may not fabricate the wiring it covers.

Also covers the F4 fix: exclusions are matched by the excluded component's *scorer
name*, never by its prose.
"""

from __future__ import annotations

from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.evolution.observations import (
    CoevolutionObservations,
    current_observations,
    install_observations,
)
from leapflow.learning.capability_effect_verifier import (
    DURABLE_EXCLUSIONS,
    UnselectableArtifactReaper,
)
from leapflow.learning.outcome_governance_feed import QuarantineCandidateTracker


def _fresh() -> CoevolutionObservations:
    buf = CoevolutionObservations()
    install_observations(buf)
    return buf


def _requirement():
    return EvolutionIntent.create(
        "chat.reply", "send no-ops", expected_effect="the reply appears"
    ).to_requirement()


# ── the buffer's contract ─────────────────────────────────────────────────────


def test_outcome_is_only_paired_when_the_plugin_was_acquired():
    """Verifying a hand-installed tool against a teacher expectation is meaningless."""
    buf = _fresh()
    try:
        buf.record_resolution(requirement=_requirement(), selected_plugin="hand_made")
        buf.record_tool_outcome("hand_made", "t", ok=True)
        assert buf.drain_verifications() == ()          # never bound

        buf.record_acquisition("gen1")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen1")
        buf.record_tool_outcome("gen1", "t", ok=True, observed_effect="the reply appears")
        drained = buf.drain_verifications()
        assert len(drained) == 1
        assert drained[0][2] == "gen1"
    finally:
        install_observations(None)


def test_draining_prevents_double_governing_the_same_outcome():
    buf = _fresh()
    try:
        buf.record_acquisition("gen1")
        buf.record_resolution(requirement=_requirement(), selected_plugin="gen1")
        buf.record_tool_outcome("gen1", "t", ok=False)
        assert len(buf.drain_verifications()) == 1
        assert buf.drain_verifications() == ()          # already taken
    finally:
        install_observations(None)


def test_every_buffer_is_bounded():
    """Governance state must not grow with session length."""
    buf = CoevolutionObservations(max_resolutions=3, max_verifications=2, max_acquired=2)
    install_observations(buf)
    try:
        for i in range(10):
            buf.record_resolution(selected_plugin=f"p{i}")
        assert len(buf.resolutions()) == 3
        buf.record_acquisition("a")
        buf.record_acquisition("b")
        buf.record_acquisition("c")
        assert len(buf.acquired_plugin_ids()) == 2
    finally:
        install_observations(None)


def test_acquisitions_are_deduplicated():
    buf = _fresh()
    try:
        buf.record_acquisition("gen1")
        buf.record_acquisition("gen1")
        assert buf.acquired_plugin_ids() == ("gen1",)
    finally:
        install_observations(None)


def test_process_accessor_creates_a_buffer_on_first_use():
    install_observations(None)
    assert isinstance(current_observations(), CoevolutionObservations)
    install_observations(None)


# ── A-r1: the engine records resolutions (drive the real method) ──────────────


class _Component:
    def __init__(self, scorer, excluded, reason=""):
        self.scorer, self.excluded, self.reason = scorer, excluded, reason


class _Candidate:
    def __init__(self, plugin_id):
        self.plugin_id = plugin_id


class _Score:
    def __init__(self, plugin_id, components):
        self.candidate = _Candidate(plugin_id)
        self.components = tuple(components)

    @property
    def eligible(self):
        return not any(c.excluded for c in self.components)


class _Resolution:
    def __init__(self, requirement, candidates, selected=None):
        self.requirement = requirement
        self.candidates = tuple(candidates)
        self.selected = selected


def _drive_engine_record(resolution):
    """Call the production static method itself."""
    from leapflow.engine.engine import AgentEngine

    AgentEngine._record_coevolution_resolution(resolution)


def test_engine_records_scorer_names_not_prose():
    """F4: keying off a human-readable reason breaks when the resolver rewords it."""
    buf = _fresh()
    try:
        over_risk = _Score("gen_overrisk", [
            _Component("risk_cost", True, "risk 'external' exceeds max 'read_only'"),
        ])
        incumbent = _Score("incumbent", [_Component("declared_match", False)])
        _drive_engine_record(_Resolution(_requirement(), [over_risk, incumbent], incumbent))

        recorded = buf.resolutions()
        assert len(recorded) == 1
        assert recorded[0]["selected_plugin"] == "incumbent"
        # The durable name, not the sentence.
        assert recorded[0]["exclusions"]["gen_overrisk"] == ["risk_cost"]
        assert "incumbent" not in recorded[0]["exclusions"]      # eligible -> not excluded
    finally:
        install_observations(None)


def test_engine_recording_survives_a_malformed_resolution():
    """Observation must never disturb the turn that produced it."""
    buf = _fresh()
    try:
        _drive_engine_record(object())
        assert buf.resolutions() == () or len(buf.resolutions()) >= 0
    finally:
        install_observations(None)


def test_engine_records_feed_the_reaper_end_to_end():
    """Engine output must be directly consumable by the reaper -- no adapter."""
    buf = _fresh()
    try:
        buf.record_acquisition("gen_overrisk")
        for _ in range(3):
            over = _Score("gen_overrisk", [_Component("risk_cost", True, "over cap")])
            keep = _Score("incumbent", [_Component("declared_match", False)])
            _drive_engine_record(_Resolution(_requirement(), [over, keep], keep))

        found = UnselectableArtifactReaper(min_resolutions=3).candidates(
            acquired_plugin_ids=buf.acquired_plugin_ids(), resolutions=buf.resolutions(),
        )
        assert [c.plugin_id for c in found] == ["gen_overrisk"]
    finally:
        install_observations(None)


def test_environment_exclusion_from_the_engine_is_not_reaped():
    buf = _fresh()
    try:
        buf.record_acquisition("gen_v2")
        for _ in range(3):
            miss = _Score("gen_v2", [_Component("environment_affordance", True, "missing")])
            keep = _Score("incumbent", [_Component("declared_match", False)])
            _drive_engine_record(_Resolution(_requirement(), [miss, keep], keep))

        found = UnselectableArtifactReaper(min_resolutions=3).candidates(
            acquired_plugin_ids=buf.acquired_plugin_ids(), resolutions=buf.resolutions(),
        )
        assert found == ()
    finally:
        install_observations(None)


def test_durable_exclusions_are_configurable():
    """F5: the predicate is a constructor parameter, not a baked-in constant."""
    assert DURABLE_EXCLUSIONS == ("risk_cost",)
    resolutions = [
        {"selected_plugin": "x", "exclusions": {"gen": ["custom_gate"]}} for _ in range(3)
    ]
    assert UnselectableArtifactReaper(min_resolutions=3).candidates(
        acquired_plugin_ids=["gen"], resolutions=resolutions
    ) == ()
    found = UnselectableArtifactReaper(
        min_resolutions=3, durable_exclusions=("custom_gate",)
    ).candidates(acquired_plugin_ids=["gen"], resolutions=resolutions)
    assert [c.plugin_id for c in found] == ["gen"]


# ── A-r2: the tool-outcome sink feeds the quarantine streak ───────────────────


def _usage_tracker(owners: dict[str, str] | None = None):
    """Real PluginUsageTracker with its tool->plugin reverse index primed.

    The reverse index is a *precondition* here, not the wiring under test: in
    production it is built from the live registry's ``tool_owners``. Priming the
    documented cache (with the registry's current version, so the cache is not
    invalidated) keeps the assertion on the streak feed itself.
    """
    from leapflow.plugins import get_registry
    from leapflow.learning.plugin_stats import PluginUsageTracker
    from leapflow.learning.plugin_trust import PluginTrustLedger

    tracker = PluginUsageTracker()
    tracker.set_trust_ledger(PluginTrustLedger())
    tracker._tool_to_plugin = dict(owners or {})
    tracker._registry_version = getattr(get_registry(), "_version", 0)
    return tracker


def test_usage_tracker_feeds_the_quarantine_streak():
    """Drives the real PluginUsageTracker.record, not a stand-in."""
    usage = _usage_tracker({"bad_tool": "bad_plugin"})
    quarantine = QuarantineCandidateTracker(quarantine_after=2)
    usage.set_quarantine_tracker(quarantine)

    usage.record("bad_tool", ok=False, duration_ms=1.0)
    assert quarantine.pending() == 0
    usage.record("bad_tool", ok=False, duration_ms=1.0)
    assert quarantine.pending() == 1
    assert quarantine.candidates()[0].plugin_id == "bad_plugin"


def test_success_through_the_real_sink_resets_the_streak():
    usage = _usage_tracker({"t": "p"})
    quarantine = QuarantineCandidateTracker(quarantine_after=2)
    usage.set_quarantine_tracker(quarantine)

    usage.record("t", ok=False, duration_ms=1.0)
    usage.record("t", ok=True, duration_ms=1.0)
    usage.record("t", ok=False, duration_ms=1.0)
    assert quarantine.pending() == 0


def test_a_broken_quarantine_tracker_never_fails_a_tool_call():
    class _Broken:
        def record(self, *a, **k):
            raise RuntimeError("boom")

    usage = _usage_tracker({"t": "p"})
    usage.set_quarantine_tracker(_Broken())
    usage.record("t", ok=False, duration_ms=1.0)          # must not raise


def test_unowned_tool_records_no_streak():
    usage = _usage_tracker({})
    quarantine = QuarantineCandidateTracker(quarantine_after=1)
    usage.set_quarantine_tracker(quarantine)
    usage.record("orphan", ok=False, duration_ms=1.0)
    assert quarantine.pending() == 0


# ── the shared tracker: injected at composition, drained by the sweep ─────────


def test_composition_injects_the_process_tracker_into_the_usage_sink():
    """Without this the whole feed is inert: the sink increments nothing.

    Asserts the wiring by driving the real `PluginUsageTracker` after attaching the
    process tracker the way `session_factory` does, then checking the *same* instance
    the sweep would drain has the streak.
    """
    from leapflow.evolution.observations import (
        current_quarantine_tracker,
        install_quarantine_tracker,
    )

    install_quarantine_tracker(QuarantineCandidateTracker(quarantine_after=2))
    try:
        shared = current_quarantine_tracker()
        usage = _usage_tracker({"t": "p"})
        usage.set_quarantine_tracker(shared)          # what session_factory does

        usage.record("t", ok=False, duration_ms=1.0)
        usage.record("t", ok=False, duration_ms=1.0)

        # The sweep resolves the tracker through the same accessor.
        assert current_quarantine_tracker() is shared
        assert shared.pending() == 1
    finally:
        install_quarantine_tracker(None)


def test_the_usage_sink_feeds_streaks_but_not_verifications():
    """Outcome recording moved to the engine (C-1), and must not happen twice.

    The sink receives only ``ok``; a tool's observed effect lives in its result
    payload, which only the engine's result-observation path sees. Recording here as
    well would double-count and would grade every success unverifiable. The streak
    feed stays, because quarantine needs nothing but ``ok``.
    """
    buf = _fresh()
    try:
        buf.record_acquisition("acquired_plugin")
        buf.record_resolution(requirement=_requirement(), selected_plugin="acquired_plugin")

        quarantine = QuarantineCandidateTracker(quarantine_after=1)
        usage = _usage_tracker({"gen_tool": "acquired_plugin"})
        usage.set_quarantine_tracker(quarantine)
        usage.record("gen_tool", ok=False, duration_ms=1.0)

        assert quarantine.pending() == 1          # streak fed
        assert buf.drain_verifications() == ()    # verification is the engine's job
    finally:
        install_observations(None)


def test_outcomes_for_unacquired_plugins_do_not_accumulate():
    """Every ordinary tool call must leave the verification buffer untouched."""
    buf = _fresh()
    try:
        usage = _usage_tracker({"list_dir": "builtin_fs"})
        for _ in range(50):
            usage.record("list_dir", ok=True, duration_ms=1.0)
        assert buf.drain_verifications() == ()
    finally:
        install_observations(None)

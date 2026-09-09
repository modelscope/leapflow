"""WM-6 / LF-10 / A-4 / P5: verify, then govern.

* **WM-6** -- an acquired capability is verified by its *observed effect*, not by
  conformance (`PluginValidator`) or by *declared* fitness (re-resolution). v0.5 and
  v0.7 both recorded that gap.
* **LF-10** -- artifacts that keep failing verification are reclaimed by the
  existing governor; the residual case (never selected at all) is found
  conservatively by `UnselectableArtifactReaper`.
* **A-4** -- quarantine finally has a feed, split so the hot path stays trivial and
  governance runs on a cold path.
* **P5** -- an enforcement mode where only listed requirement origins may drive an
  acquisition; the executable form of "the world model is the first driver".
"""

from __future__ import annotations

import asyncio

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.evolution_intent import EvolutionIntent
from leapflow.learning.capability_effect_verifier import (
    EFFECT_ABSENT,
    EFFECT_UNREPORTED,
    EXECUTION_FAILED,
    NO_OUTCOME,
    UNVERIFIABLE,
    VERIFIED,
    CapabilityEffectVerifier,
    UnselectableArtifactReaper,
)
from leapflow.learning.outcome_governance_feed import (
    QuarantineCandidateTracker,
    drain_quarantine_candidates,
    filter_authorised,
    origin_may_authorise,
)


def _requirement(expected_effect: str = "the reply appears in the thread"):
    intent = EvolutionIntent.create(
        "chat.reply", "the send path silently no-ops", expected_effect=expected_effect
    )
    return intent.to_requirement()


# ── WM-6: verification by observed effect ─────────────────────────────────────


def test_matching_effect_verifies():
    verdict = CapabilityEffectVerifier().verify(
        _requirement(),
        {"ok": True, "observed_effect": "the reply appears in the thread"},
        plugin_id="gen1",
    )
    assert verdict.verified is True
    assert verdict.reason == VERIFIED
    assert verdict.should_record_outcome is True


def test_absent_effect_fails_even_when_the_call_succeeded():
    """The core of WM-6: 'the tool returned ok' is not 'the capability worked'."""
    verdict = CapabilityEffectVerifier().verify(
        _requirement(),
        {"ok": True, "observed_effect": "nothing happened"},
        plugin_id="gen1",
    )
    assert verdict.verified is False
    assert verdict.reason == EFFECT_ABSENT
    assert verdict.should_record_outcome is True


def test_execution_failure_is_a_decided_negative():
    verdict = CapabilityEffectVerifier().verify(
        _requirement(), {"ok": False, "observed_effect": "exception"}, plugin_id="gen1"
    )
    assert verdict.verified is False
    assert verdict.reason == EXECUTION_FAILED


def test_no_outcome_is_unverifiable_not_failed():
    verdict = CapabilityEffectVerifier().verify(_requirement(), None, plugin_id="gen1")
    assert verdict.verified is None
    assert verdict.reason == NO_OUTCOME
    assert verdict.should_record_outcome is False


def test_missing_declaration_is_unverifiable_not_failed():
    """A metadata omission must not quarantine a healthy plugin."""
    requirement = CapabilityRequirement.create("chat.reply", "unknown_tool")
    verdict = CapabilityEffectVerifier().verify(
        requirement, {"ok": True, "observed_effect": "sent"}, plugin_id="gen1"
    )
    assert verdict.verified is None
    assert verdict.reason == UNVERIFIABLE
    assert verdict.should_record_outcome is False


def test_a_silent_tool_is_unverifiable_not_refuted():
    """Absence of evidence is not evidence of absence.

    A handler written before the effect convention succeeds and says nothing. Refuting
    that would demote it and quarantine it after three calls, punishing a plugin for a
    reporting omission rather than for failing.
    """
    verdict = CapabilityEffectVerifier().verify(
        _requirement(), {"ok": True, "observed_effect": ""}, plugin_id="gen1"
    )
    assert verdict.verified is None
    assert verdict.reason == EFFECT_UNREPORTED
    assert verdict.should_record_outcome is False


def test_stopwords_alone_do_not_verify():
    """'the in of' overlapping must not be read as the effect occurring."""
    verdict = CapabilityEffectVerifier().verify(
        _requirement("the reply appears in the thread"),
        {"ok": True, "observed_effect": "the of in and it"},
        plugin_id="gen1",
    )
    assert verdict.verified is False


def test_verdict_feeds_the_governor_and_quarantines_a_useless_artifact(tmp_path):
    """WM-6 + LF-10: repeated verification failure reclaims the artifact."""
    from leapflow.learning.plugin_trust import PluginTrustLedger
    from leapflow.plugins.lifecycle_governor import LifecycleGovernor
    from leapflow.storage.capability_proposal_queue import JsonCapabilityProposalQueue
    from leapflow.storage.plugin_outcome_store import JsonPluginOutcomeStore

    disabled: list[str] = []

    class _Actor:
        async def disable(self, *, plugin_id):
            disabled.append(plugin_id)
            return {"ok": True}

    queue = JsonCapabilityProposalQueue(tmp_path / "q.json")
    item = queue.enqueue(requirements=[_requirement()], source="test")
    governor = LifecycleGovernor(
        proposal_queue=queue,
        outcome_store=JsonPluginOutcomeStore(tmp_path / "o.json"),
        lifecycle_actor=_Actor(),
        trust_ledger=PluginTrustLedger(),
        quarantine_after=3,
    )
    verifier = CapabilityEffectVerifier()
    requirement = _requirement()

    actions = []
    for _ in range(3):
        verdict = verifier.verify(
            requirement, {"ok": True, "observed_effect": "nothing happened"},
            plugin_id="gen_useless",
        )
        assert verdict.should_record_outcome
        result = asyncio.run(governor.record_outcome(
            proposal_id=item.proposal_id, plugin_id=verdict.plugin_id,
            tool_name="gen_useless", ok=bool(verdict.verified),
        ))
        actions.append(result.action)

    assert actions[-1] == "quarantine"
    assert disabled == ["gen_useless"]      # the useless artifact was reclaimed


# ── LF-10: the residual case the governor cannot reach ────────────────────────


def _resolution(selected: str = "", exclusions: dict | None = None):
    return {"selected_plugin": selected, "exclusions": exclusions or {}}


def test_never_selected_risk_excluded_artifact_is_a_candidate():
    reaper = UnselectableArtifactReaper(min_resolutions=3)
    resolutions = [
        _resolution("incumbent", {"gen_overrisk": ["risk_cost"]})
        for _ in range(3)
    ]
    found = reaper.candidates(acquired_plugin_ids=["gen_overrisk"], resolutions=resolutions)
    assert [c.plugin_id for c in found] == ["gen_overrisk"]
    assert found[0].resolutions_seen == 3


def test_environment_excluded_artifact_is_not_reaped():
    """An environment can change and make it viable again; a risk cap will not."""
    reaper = UnselectableArtifactReaper(min_resolutions=3)
    resolutions = [
        _resolution("incumbent", {"gen_v2": ["environment_affordance"]})
        for _ in range(3)
    ]
    assert reaper.candidates(acquired_plugin_ids=["gen_v2"], resolutions=resolutions) == ()


def test_a_single_selection_spares_the_artifact():
    reaper = UnselectableArtifactReaper(min_resolutions=3)
    resolutions = [
        _resolution("incumbent", {"gen_x": ["risk_cost"]}),
        _resolution("gen_x", {}),
        _resolution("incumbent", {"gen_x": ["risk_cost"]}),
    ]
    assert reaper.candidates(acquired_plugin_ids=["gen_x"], resolutions=resolutions) == ()


def test_too_few_resolutions_condemns_nobody():
    reaper = UnselectableArtifactReaper(min_resolutions=3)
    resolutions = [_resolution("incumbent", {"gen_x": ["risk_cost"]})]
    assert reaper.candidates(acquired_plugin_ids=["gen_x"], resolutions=resolutions) == ()


def test_hand_installed_plugins_are_never_reaped():
    reaper = UnselectableArtifactReaper(min_resolutions=1)
    resolutions = [_resolution("incumbent", {"hand_made": ["risk_cost"]})] * 3
    assert reaper.candidates(acquired_plugin_ids=[], resolutions=resolutions) == ()


# ── A-4: quarantine feed, hot path trivial / governance deferred ──────────────


def test_streak_only_marks_at_the_threshold():
    tracker = QuarantineCandidateTracker(quarantine_after=3)
    assert tracker.record("p", "t", ok=False) is False
    assert tracker.record("p", "t", ok=False) is False
    assert tracker.record("p", "t", ok=False) is True
    assert tracker.pending() == 1
    assert tracker.candidates()[0].failure_streak == 3


def test_success_resets_the_streak_so_intermittent_failures_survive():
    tracker = QuarantineCandidateTracker(quarantine_after=3)
    tracker.record("p", "t", ok=False)
    tracker.record("p", "t", ok=False)
    tracker.record("p", "t", ok=True)          # recovered
    assert tracker.record("p", "t", ok=False) is False
    assert tracker.pending() == 0


def test_tracker_does_no_io_and_ignores_unknown_plugins():
    tracker = QuarantineCandidateTracker(quarantine_after=1)
    assert tracker.record("", "t", ok=False) is False   # unresolvable tool -> no-op
    assert tracker.pending() == 0


def test_drain_governs_each_candidate_then_clears_it():
    calls: list[str] = []

    class _Governor:
        async def record_outcome(self, **kwargs):
            calls.append(kwargs["plugin_id"])
            return type("R", (), {"action": "quarantine", "trust_level": "DRAFT"})()

    tracker = QuarantineCandidateTracker(quarantine_after=1)
    tracker.record("a", "ta", ok=False)
    tracker.record("b", "tb", ok=False)

    handled = asyncio.run(drain_quarantine_candidates(tracker, _Governor()))
    assert sorted(h["plugin_id"] for h in handled) == ["a", "b"]
    assert sorted(calls) == ["a", "b"]
    assert tracker.pending() == 0

    # A second drain is a no-op, not a double punishment.
    assert asyncio.run(drain_quarantine_candidates(tracker, _Governor())) == ()


def test_one_failing_candidate_does_not_stop_the_others():
    class _Governor:
        async def record_outcome(self, **kwargs):
            if kwargs["plugin_id"] == "bad":
                raise OSError("store down")
            return type("R", (), {"action": "quarantine", "trust_level": "DRAFT"})()

    tracker = QuarantineCandidateTracker(quarantine_after=1)
    tracker.record("bad", "t1", ok=False)
    tracker.record("good", "t2", ok=False)
    handled = asyncio.run(drain_quarantine_candidates(tracker, _Governor()))
    assert [h["plugin_id"] for h in handled] == ["good"]
    assert tracker.pending() == 0              # both cleared regardless


# ── P5: only authorised origins may drive acquisition ────────────────────────


def test_unrestricted_by_default():
    for origin in ("unknown_tool", "world_model", "environment_probe", "task_contract"):
        assert origin_may_authorise(origin, None) is True
        assert origin_may_authorise(origin, ()) is True


def test_restricting_to_world_model_is_the_goal_in_executable_form():
    allowed = ("world_model",)
    assert origin_may_authorise("world_model", allowed) is True
    assert origin_may_authorise("unknown_tool", allowed) is False
    assert origin_may_authorise("environment_probe", allowed) is False


def test_filter_keeps_only_authorised_requirements():
    wm = EvolutionIntent.create("chat.reply", "gap").to_requirement()
    shipped = CapabilityRequirement.create("list_dir", "unknown_tool")
    assert len(filter_authorised([wm, shipped], None)) == 2
    kept = filter_authorised([wm, shipped], ("world_model",))
    assert [r.origin for r in kept] == ["world_model"]


def test_config_defaults_keep_both_new_switches_off():
    import dataclasses

    from leapflow.config import Settings

    fields = {f.name: f for f in dataclasses.fields(Settings)}
    assert fields["evolution_authorising_origins"].default == ()
    assert fields["accepted_evidence_kinds"].default == ()

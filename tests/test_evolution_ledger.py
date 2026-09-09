"""EvolutionLedger: rebuilding causal episodes from records that already exist.

The point of this stage is that no probe is needed, so the tests are mostly about
the two ends the decision record never joined: the environment evidence that
preceded it, and whether the gap it was meant to close actually closed.

The negative cases carry the most weight:

* a retired observation must be reported as ``declared_fitness``, never as
  verified -- the engine retires on re-resolution, and the recorded v0.7 defect was
  a wrongly selected tool retiring the evidence for its own gap;
* a recurrence must surface as ``reopened``, which is the outcome the system could
  not express at all before the observation lifecycle was closed;
* the two proposal vocabularies must not be conflated;
* one malformed record must not blank the timeline.
"""

from __future__ import annotations

from leapflow.domain.evolution_trace import (
    ABORTED,
    COMMITTED,
    DECLARED_FITNESS,
    NOT_APPLICABLE,
    OPEN,
    REOPENED,
    RESOLVED,
    STILL_OPEN,
    EvolutionStage,
)
from leapflow.evolution import EvolutionLedger


class _Plans:
    """Mirrors ``JsonCapabilityPlanStore.list_records``: non-mappings filtered, newest first."""

    def __init__(self, records: list) -> None:
        self._records = records

    def list_records(self, *, limit: int = 20):
        usable = [dict(r) for r in self._records if isinstance(r, dict)]
        ordered = sorted(usable, key=lambda r: float(r.get("created_at") or 0.0), reverse=True)
        return ordered if limit <= 0 else ordered[:limit]


class _Observations:
    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def list_observations(self, *, limit: int = 50):
        return list(self._records) if limit <= 0 else list(self._records)[:limit]


class _Level:
    def __init__(self, name: str) -> None:
        self.name = name


class _Trust:
    def __init__(self, levels: dict[str, str]) -> None:
        self._levels = levels

    def level(self, plugin_id: str) -> _Level:
        return _Level(self._levels.get(plugin_id, "DRAFT"))


def _record(**over) -> dict:
    base = {
        "record_id": "r1",
        "created_at": 1000.0,
        "source": "runtime",
        "requirements": [
            {
                "requirement_id": "req-unknown-tool-list_dir",
                "capability": "list_dir",
                "origin": "unknown_tool",
                "evidence": "Runtime attempted unknown tool 'list_dir'.",
                "metadata": {},
            }
        ],
        "resolutions": [],
        "plan": {"executable": True},
        "observation_ids": ["obs-1"],
    }
    base.update(over)
    return base


def _observation(**over) -> dict:
    base = {
        "observation_id": "obs-1",
        "first_seen_at": 900.0,
        "last_seen_at": 950.0,
        "occurrence_count": 3,
        "result": {"error_type": "unknown_tool", "original_tool_name": "list_dir"},
    }
    base.update(over)
    return base


def _ledger(records, observations=None, trust=None, **kw) -> EvolutionLedger:
    return EvolutionLedger(
        plan_store=_Plans(records),
        observation_store=_Observations(observations or []),
        trust_ledger=trust,
        **kw,
    )


# ── the two ends the decision record never joined ────────────────────────────


def test_episode_joins_environment_cause_to_framework_change():
    """One record plus its observations is a complete five-stage story."""
    record = _record(
        mutation={"action": "install", "plugin_id": "list_dir_plugin"},
        registry_version_before=7,
        registry_version_after=8,
        policy_decision={"action": "install", "autonomy_level": "trusted", "reason": "ok"},
    )
    episodes = _ledger([record], [_observation()], _Trust({"list_dir_plugin": "DRAFT"})).recent_episodes()

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_id == "ep-r1"
    assert episode.driver == "unknown_tool"
    assert episode.capability == "list_dir"
    assert episode.mutation_action == "install"
    assert episode.plugin_id == "list_dir_plugin"
    assert episode.registry_before == 7 and episode.registry_after == 8
    assert episode.framework_changed is True
    assert episode.status == COMMITTED
    assert episode.policy_action == "install"
    assert episode.trust_now == "DRAFT"

    stages = episode.stages_present
    assert EvolutionStage.OBSERVE in stages
    assert EvolutionStage.ORIENT in stages
    assert EvolutionStage.DECIDE in stages
    assert EvolutionStage.ACT in stages


def test_world_model_hypothesis_travels_into_the_episode():
    """The teacher's own words reach the timeline without any probe.

    They ride on the requirement metadata, which is the only place they become
    durable; the driver's own counts are never persisted.
    """
    record = _record(
        requirements=[
            {
                "requirement_id": "req-wm-wmi-abc",
                "capability": "ui.chat.send",
                "origin": "world_model",
                "evidence": "the agent had no way to send a chat message",
                "metadata": {
                    "intent_id": "wmi-abc",
                    "confidence": 0.8,
                    "expected_effect": "message appears in the thread",
                },
            }
        ],
        observation_ids=["obs-wm"],
    )
    observation = _observation(
        observation_id="obs-wm",
        result={"error_type": "world_model_intent", "capability": "ui.chat.send"},
    )
    episode = _ledger([record], [observation]).recent_episodes()[0]

    assert episode.driver == "world_model"
    assert episode.intent_id == "wmi-abc"
    assert episode.confidence == 0.8
    assert "chat message" in episode.hypothesis


def test_confidence_is_clamped_and_never_raises():
    record = _record(
        requirements=[
            {
                "capability": "x",
                "origin": "world_model",
                "metadata": {"intent_id": "wmi-1", "confidence": "not-a-number"},
            }
        ]
    )
    assert _ledger([record]).recent_episodes()[0].confidence == 0.0


# ── gap closure: the consequence ─────────────────────────────────────────────


def test_retired_observation_is_declared_fitness_never_verified():
    """The engine retires on re-resolution, which does not prove the tool works.

    Reporting this as verified would reproduce the recorded v0.7 defect in the
    reader's head: a wrongly selected tool retired the evidence for its own gap.
    """
    record = _record(
        mutation={"action": "install", "plugin_id": "p"},
        registry_version_before=1,
        registry_version_after=2,
    )
    observation = _observation(status="resolved", status_reason="list_dir resolved")
    episode = _ledger([record], [observation]).recent_episodes()[0]

    assert episode.gap_closure == RESOLVED
    assert episode.verification_tier == DECLARED_FITNESS
    # Effect verification is not wired, so there is no verdict to report.
    assert episode.effect_verdict == ""
    assert "declared fitness" in episode.outcome


def test_recurrence_surfaces_as_a_regression():
    """The outcome the system could not express before the lifecycle was closed."""
    record = _record(
        mutation={"action": "install", "plugin_id": "p"},
        registry_version_before=1,
        registry_version_after=2,
    )
    observation = _observation(
        status="open", status_reason="reopened after recurrence at 1200.0"
    )
    episode = _ledger([record], [observation]).recent_episodes()[0]

    assert episode.gap_closure == REOPENED
    assert episode.outcome == "regressed"
    learn = [t for t in episode.traces if t.stage is EvolutionStage.LEARN]
    assert any(t.kind == "observation_reopened" for t in learn)


def test_framework_changed_but_gap_still_open_is_distinguished():
    """Installed something and the gap did not close: not a success, not a failure."""
    record = _record(
        mutation={"action": "install", "plugin_id": "p"},
        registry_version_before=1,
        registry_version_after=2,
    )
    episode = _ledger([record], [_observation()]).recent_episodes()[0]

    assert episode.gap_closure == STILL_OPEN
    assert episode.verification_tier == ""
    assert "still open" in episode.outcome


def test_absent_status_field_counts_as_open():
    """A freshly written observation carries no ``status`` at all.

    The store's own ``unresolved()`` treats that absence as open, so the ledger
    must too -- otherwise a brand-new gap would read as closed.
    """
    observation = _observation()
    assert "status" not in observation
    record = _record(mutation={"action": "install"}, registry_version_before=1, registry_version_after=2)
    assert _ledger([record], [observation]).recent_episodes()[0].gap_closure == STILL_OPEN


def test_no_linked_observation_is_not_applicable_not_a_failure():
    episode = _ledger([_record(observation_ids=[])]).recent_episodes()[0]
    assert episode.gap_closure == NOT_APPLICABLE


# ── episode status ───────────────────────────────────────────────────────────


def test_deciding_not_to_change_still_closes_the_episode():
    """Why the framework did *not* evolve is part of transparency, not an open loop."""
    record = _record(policy_decision={"action": "observe_only", "reason": "risk too high"})
    episode = _ledger([record]).recent_episodes()[0]

    assert episode.status == COMMITTED
    assert episode.outcome == "no action (observe_only)"


def test_attempted_mutation_that_did_not_move_the_registry_stays_open():
    """An attempt with no effect must not read as a clean conclusion."""
    record = _record(
        mutation={"action": "install", "plugin_id": "p"},
        registry_version_before=5,
        registry_version_after=5,
    )
    episode = _ledger([record]).recent_episodes()[0]

    assert episode.framework_changed is False
    assert episode.status == OPEN
    assert "registry unchanged" in episode.outcome


def test_stale_unclosed_episode_is_aborted():
    record = _record(created_at=100.0)
    episodes = _ledger([record], episode_ttl_s=60.0).recent_episodes(now=1000.0)
    assert episodes[0].status == ABORTED


def test_ttl_is_not_applied_without_a_clock():
    """``now=0`` means "no clock supplied", not "the epoch"."""
    record = _record(created_at=100.0)
    assert _ledger([record], episode_ttl_s=60.0).recent_episodes()[0].status == OPEN


# ── vocabularies and decision transparency ───────────────────────────────────


def test_the_two_proposal_vocabularies_are_not_conflated():
    """A decision record's ``proposal.status`` is the acquisition lifecycle.

    The review vocabulary lives in a different store and answers a different
    question, so borrowing this value for it would merge two things the code
    upstream explicitly warns must stay apart.
    """
    record = _record(proposal={"proposal_id": "cp-1", "status": "PROBATION"})
    episode = _ledger([record]).recent_episodes()[0]

    assert episode.lifecycle_status == "PROBATION"
    assert episode.review_status == ""


def test_losing_candidates_are_preserved_in_the_decide_trace():
    """Why a candidate lost is half of decision transparency."""
    record = _record(
        resolutions=[
            {
                "requirement": {"capability": "list_dir"},
                "selected": {"candidate": {"tool_name": "file_list"}},
                "candidates": [
                    {
                        "candidate": {"tool_name": "other"},
                        "eligible": False,
                        "exclusion_reasons": ["risk 'high' exceeds max 'read_only'"],
                    }
                ],
            }
        ],
        policy_decision={"action": "install", "reason": "chosen"},
    )
    episode = _ledger([record]).recent_episodes()[0]
    decide = episode.trace_of(EvolutionStage.DECIDE)

    assert decide is not None
    assert decide.detail["resolutions"][0]["candidates"][0]["exclusion_reasons"]


def test_trust_at_decision_is_read_from_the_record_and_now_from_the_ledger():
    """Two different facts, so two different fields.

    There is no stored history to reconstruct a before/after pair from; presenting
    a live reading as an "after" would imply a comparison never made.
    """
    record = _record(
        mutation={"action": "install", "plugin_id": "p"},
        proposal={"status": "INSTALLED", "trust_state": {"trust_level": "DRAFT"}},
    )
    episode = _ledger([record], trust=_Trust({"p": "CANDIDATE"})).recent_episodes()[0]

    assert episode.trust_at_decision == "DRAFT"
    assert episode.trust_now == "CANDIDATE"


# ── robustness ───────────────────────────────────────────────────────────────


def test_one_malformed_record_does_not_blank_the_timeline():
    """A record the builder cannot read must cost only itself.

    ``requirements`` as a string gets past the store (it filters non-mappings at
    the top level only) and breaks the per-record assembly, which is exactly the
    case the inner guard exists for.
    """
    good = _record(record_id="good", created_at=2000.0)
    broken = _record(record_id="broken", created_at=2001.0, requirements="not-a-list-of-mappings")
    episodes = _ledger([good, broken]).recent_episodes()

    ids = [episode.episode_id for episode in episodes]
    assert "ep-good" in ids


def test_unreadable_plan_store_returns_empty_rather_than_raising():
    class _Broken:
        def list_records(self, **_kw):
            raise OSError("disk gone")

    ledger = EvolutionLedger(plan_store=_Broken())
    assert ledger.recent_episodes() == ()


def test_unreadable_observation_store_degrades_to_no_cause():
    class _Broken:
        def list_observations(self, **_kw):
            raise OSError("disk gone")

    ledger = EvolutionLedger(plan_store=_Plans([_record()]), observation_store=_Broken())
    episode = ledger.recent_episodes()[0]
    # The decision is still there; only the cause is missing.
    assert episode.capability == "list_dir"
    assert episode.gap_closure == NOT_APPLICABLE


def test_absent_observation_store_is_tolerated():
    ledger = EvolutionLedger(plan_store=_Plans([_record()]))
    assert ledger.recent_episodes()[0].capability == "list_dir"


def test_trust_read_failure_does_not_break_the_episode():
    class _Broken:
        def level(self, _plugin_id):
            raise RuntimeError("ledger closed")

    record = _record(mutation={"action": "install", "plugin_id": "p"})
    episode = _ledger([record], trust=_Broken()).recent_episodes()[0]
    assert episode.trust_now == ""


def test_records_are_newest_first_and_bounded():
    records = [_record(record_id=f"r{i}", created_at=float(i)) for i in range(10)]
    episodes = _ledger(records).recent_episodes(limit=3)
    assert [e.episode_id for e in episodes] == ["ep-r9", "ep-r8", "ep-r7"]


def test_traces_are_ordered_by_time():
    record = _record(
        created_at=1000.0,
        mutation={"action": "install", "plugin_id": "p"},
        policy_decision={"action": "install"},
    )
    episode = _ledger([record], [_observation(first_seen_at=500.0)]).recent_episodes()[0]
    timestamps = [trace.ts for trace in episode.traces]
    assert timestamps == sorted(timestamps)
    # The cause precedes the decision.
    assert episode.traces[0].stage is EvolutionStage.OBSERVE


# ── against the real stores ──────────────────────────────────────────


def test_full_lifecycle_against_the_real_stores(tmp_path):
    """Drive the three interesting endings through the real store implementations.

    The fakes above encode this module's assumptions about field names and status
    defaults; only the real stores can confirm them. This walks one capability from
    acquired-but-unhelpful, to closed, to *recurred* -- the last being the outcome
    the system could not express at all before the observation lifecycle was closed.
    """
    from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
    from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore

    observations = JsonCapabilityObservationStore(tmp_path / "obs.json")
    plans = JsonCapabilityPlanStore(tmp_path / "plans.json")
    evidence = {
        "error_type": "world_model_intent",
        "capability": "ui.chat.send",
        "evidence": "the agent had no way to send a chat message",
    }

    stored = observations.add_observation(result=evidence, source="world_model")
    # A freshly written observation carries no status at all; the ledger must read
    # that absence as open, exactly as the store's own unresolved() does.
    assert "status" not in stored

    plans.add_record(
        requirements=[
            {
                "requirement_id": "req-wm-wmi-1",
                "capability": "ui.chat.send",
                "origin": "world_model",
                "evidence": evidence["evidence"],
                "metadata": {"intent_id": "wmi-1", "confidence": 0.9},
            }
        ],
        mutation={"action": "install", "plugin_id": "chat_plugin"},
        registry_version_before=10,
        registry_version_after=11,
        policy_decision={"action": "install", "autonomy_level": "trusted"},
        proposal={"status": "INSTALLED", "trust_state": {"trust_level": "DRAFT"}},
        observation_ids=[stored["observation_id"]],
    )

    def episode():
        return EvolutionLedger(
            plan_store=plans, observation_store=observations
        ).recent_episodes()[0]

    # 1. Acquired, and the gap it was meant to close is still open.
    first = episode()
    assert first.driver == "world_model"
    assert first.intent_id == "wmi-1"
    assert first.confidence == 0.9
    assert first.framework_changed is True
    assert first.lifecycle_status == "INSTALLED"
    assert first.review_status == ""
    assert first.gap_closure == STILL_OPEN
    assert first.verification_tier == ""

    # 2. Retired -- but only ever on declared fitness.
    observations.mark_status(stored["observation_id"], "resolved", reason="resolved")
    closed = episode()
    assert closed.gap_closure == RESOLVED
    assert closed.verification_tier == DECLARED_FITNESS

    # 3. The same evidence recurs: the store reopens the record, and the episode
    #    must report a regression rather than keeping its clean conclusion.
    observations.add_observation(result=evidence, source="world_model")
    regressed = episode()
    assert regressed.gap_closure == REOPENED
    assert regressed.outcome == "regressed"
    assert any(
        trace.kind == "observation_reopened"
        for trace in regressed.traces
        if trace.stage is EvolutionStage.LEARN
    )

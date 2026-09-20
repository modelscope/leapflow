# Copyright (c) Alibaba, Inc. and its affiliates.
"""Channel C2: the teacher's rebind recommendation reaches the selection layer.

Until now the most frequent verdict a real model produced had nowhere to land. Measured
across three S9 runs, ``rebind`` was the action it reached for most readily -- and its
``target`` travelled only as a line of prose in the student's context. The resolver, which
is what actually picks a provider, never heard about it.

The whole design of this channel is "preference, not gate", and every test here exists to
hold one half of that: the recommendation must change the outcome when the recommended
provider is admissible, and must change nothing at all when it is not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.plugins.capability_resolver import (
    _DEFAULT_SCORERS,
    CapabilityCandidate,
    CapabilityResolver,
    DistilledPreferenceScorer,
    EnvironmentAffordanceScorer,
    ResolverContext,
)
from leapflow.storage.distilled_knowledge_store import EvolutionDistilledKnowledgeStore
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore


def _seed(events: DuckDBEvolutionEventStore, verdict: AdaptationVerdict, seq: int = 0) -> None:
    events.append(
        EvolutionEvent.create(
            EvolutionEventType.TEACHER_VERDICT_RECORDED,
            context=EvolutionContext(profile_id="p", decision_id=verdict.verdict_id),
            payload=verdict.to_dict(),
            producer="test",
            dedup_key=f"teacher.verdict_recorded:{verdict.verdict_id}:{seq}",
        )
    )


def _knowledge(tmp_path, *verdicts: AdaptationVerdict):
    events = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    for index, verdict in enumerate(verdicts):
        _seed(events, verdict, index)
    store = EvolutionDistilledKnowledgeStore(events, profile_id="p")
    store.refresh()
    return events, store


def _env(*affordances: str) -> EnvironmentFingerprint:
    return EnvironmentFingerprint.from_platform_manifest(
        PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset({Capability.FILE_OPS}))
    )


def _requirement() -> CapabilityRequirement:
    return CapabilityRequirement.create(
        "chat.reply", "world_model", max_risk_level="read_only"
    )


def _candidate(plugin_id: str, tool: str, *affordances: str) -> CapabilityCandidate:
    return CapabilityCandidate(
        plugin_id=plugin_id,
        tool_name=tool,
        provides_capabilities=("chat.reply",),
        requires_environment_affordances=tuple(affordances),
        risk_level="read_only",
    )


def _resolve(candidates, *, preferences=(), affordance_scorer=False):
    scorers: tuple[Any, ...] = _DEFAULT_SCORERS
    if affordance_scorer:
        scorers = (*scorers, EnvironmentAffordanceScorer())
    scorers = (*scorers, DistilledPreferenceScorer())
    context = ResolverContext(environment=_env(), distilled_preferences=tuple(preferences))
    return CapabilityResolver(scorers).resolve_all([_requirement()], candidates, context)[0]


# ── only rebind becomes a preference ──────────────────────────────────────────


def test_only_rebind_verdicts_become_selection_preferences(tmp_path):
    """An escalate target names what a *person* must do; absorb has no target at all.

    Admitting either would turn an instruction to a human into a machine's selection
    preference, which is the one direction this channel must never go.
    """
    events, store = _knowledge(
        tmp_path,
        AdaptationVerdict.create(
            "rebind", "chat.reply", "the app is now v2", target="chat_reply_v2_native"
        ),
        AdaptationVerdict.create("absorb", "chat.react", "the control moved"),
        AdaptationVerdict.create(
            "escalate", "drive.upload", "refused", target="grant the drive.file scope"
        ),
    )

    assert store.rebind_preferences() == (("chat.reply", "chat_reply_v2_native"),)
    events.close()


def test_a_rebind_without_a_target_is_not_a_preference(tmp_path):
    events, store = _knowledge(
        tmp_path,
        AdaptationVerdict.create("rebind", "chat.reply", "something moved"),
    )
    assert store.rebind_preferences() == ()
    events.close()


def test_a_preference_retires_with_the_knowledge_behind_it(tmp_path):
    """Nothing to unlearn: the entry stops being read when it stops being true."""
    events, store = _knowledge(
        tmp_path,
        AdaptationVerdict.create("rebind", "chat.reply", "v2 now", target="chat_reply_v2"),
    )
    assert store.rebind_preferences()

    store.retract("chat.reply", reason="observed working again")
    assert store.rebind_preferences() == ()
    events.close()


def test_a_newer_verdict_supersedes_the_preference(tmp_path):
    import time

    events, store = _knowledge(
        tmp_path,
        AdaptationVerdict.create("rebind", "chat.reply", "v2 now", target="chat_reply_v2"),
        AdaptationVerdict.create(
            "rebind", "chat.reply", "v3 now", target="chat_reply_v3",
            created_at=time.time() + 10,
        ),
    )
    assert store.rebind_preferences() == (("chat.reply", "chat_reply_v3"),)
    events.close()


# ── preference, not gate ──────────────────────────────────────────────────────


def test_the_recommended_provider_wins_when_it_is_admissible():
    resolution = _resolve(
        [_candidate("v1", "chat_reply_v1"), _candidate("v2", "chat_reply_v2_native")],
        preferences=(("chat.reply", "chat_reply_v2_native"),),
    )
    assert resolution.selected.candidate.tool_name == "chat_reply_v2_native"


def test_a_recommendation_cannot_make_an_inadmissible_candidate_win():
    """The half that makes this safe.

    Hindsight is evidence about the world; a declaration is a fact about the code. When
    they disagree the code wins -- so a candidate whose affordances are absent stays
    excluded no matter what was recommended.
    """
    resolution = _resolve(
        [
            _candidate("v1", "chat_reply_v1"),
            _candidate("v2", "chat_reply_v2_native", "app.chat.v9"),
        ],
        preferences=(("chat.reply", "chat_reply_v2_native"),),
        affordance_scorer=True,
    )
    assert resolution.selected.candidate.plugin_id == "v1"


def test_no_recommendation_leaves_selection_exactly_as_it_was():
    """The scorer contributes zero when there is nothing to say."""
    with_scorer = _resolve(
        [_candidate("v1", "chat_reply_v1"), _candidate("v2", "chat_reply_v2_native")]
    )
    baseline = CapabilityResolver(_DEFAULT_SCORERS).resolve_all(
        [_requirement()],
        [_candidate("v1", "chat_reply_v1"), _candidate("v2", "chat_reply_v2_native")],
        ResolverContext(environment=_env()),
    )[0]
    assert (
        with_scorer.selected.candidate.plugin_id == baseline.selected.candidate.plugin_id
    )


def test_a_recommendation_for_another_capability_is_ignored():
    resolution = _resolve(
        [_candidate("v1", "chat_reply_v1"), _candidate("v2", "chat_reply_v2_native")],
        preferences=(("mail.send", "chat_reply_v2_native"),),
    )
    assert resolution.selected.candidate.plugin_id == "v1"


def test_the_preference_weight_stays_below_the_structural_weights():
    """A recommendation is evidence; it must not outvote a declaration."""
    from leapflow.plugins.capability_resolver import ResolverWeights

    weights = ResolverWeights()
    assert weights.distilled_preference < weights.declared_match
    assert weights.distilled_preference < weights.environment_fit


# ── the reader, which is where a snapshot would go stale ──────────────────────


def test_the_engine_reads_preferences_per_resolution_not_once():
    """A captured snapshot would keep preferring a provider the knowledge dropped.

    Retraction and supersession happen between sessions, so the value has to be read when
    it is used.
    """
    from leapflow.engine.engine import AgentEngine

    events = DuckDBEvolutionEventStore(Path(tempfile.mkdtemp()) / "events.duckdb")
    store = EvolutionDistilledKnowledgeStore(events, profile_id="p")
    store.refresh()
    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = store
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(distilled_knowledge_limit=12)

    assert engine._rebind_preferences() == ()
    _seed(
        events,
        AdaptationVerdict.create("rebind", "chat.reply", "v2 now", target="chat_reply_v2"),
    )
    store.refresh()
    assert engine._rebind_preferences() == (("chat.reply", "chat_reply_v2"),)
    store.retract("chat.reply")
    assert engine._rebind_preferences() == ()
    events.close()


def test_a_failing_store_costs_a_preference_not_a_resolution():
    from leapflow.engine.engine import AgentEngine

    class _Broken:
        def rebind_preferences(self):
            raise OSError("disk gone")

        def live(self):
            return ()

    engine = AgentEngine.__new__(AgentEngine)
    engine._knowledge_store = _Broken()
    engine._knowledge_store_unavailable = False
    engine._environment_fingerprint_id = ""
    engine._settings = SimpleNamespace(distilled_knowledge_limit=12)

    assert engine._rebind_preferences() == ()

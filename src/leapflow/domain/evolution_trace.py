"""Causal types for framework self-evolution: one atomic fact, and one episode.

Two concepts sit beside :mod:`leapflow.domain.evolution_intent`, and the pairing is
deliberate:

* an ``EvolutionIntent`` is a *hypothesis* -- what should evolve and why, authored
  by the world model before anything happens;
* an :class:`EvolutionTrace` is a *fact* -- what actually happened, recorded after
  it happened;
* an :class:`EvolutionEpisode` stitches traces into one causal story, so
  "the environment changed, therefore the framework changed" becomes a thing a
  person can read.

Named ``Trace`` rather than ``Signal`` because ``Signal`` already means something
else here: ``InteractionSignal`` and ``SignalSource`` are the perception layer's
vocabulary, and reusing the word for a governance fact would suggest these flow
through the same pipeline. They do not.

The stage vocabulary is the OODA loop applied to the framework itself, with the
two ends the adaptive loop never had: what preceded the decision (the environment
or the teacher) and what came of it (trust, and whether the gap actually closed).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

# ── Episode lifecycle ────────────────────────────────────────────────────
#: The episode reached a conclusion: the framework changed, or a decision
#: explicitly declined to change it.
COMMITTED = "committed"
#: Still in flight, or waiting for a later stage to arrive.
OPEN = "open"
#: No further trace arrived before the time-to-live elapsed.
ABORTED = "aborted"

# ── Gap closure ──────────────────────────────────────────────────────────
#: The observation that motivated this episode was retired.
RESOLVED = "resolved"
#: A retired observation recurred. The highest-value outcome to surface: the
#: evolution looked successful and the problem came back.
REOPENED = "reopened"
#: The framework changed but the motivating observation is still open.
STILL_OPEN = "still_open"
#: Nothing to close -- no observation was linked to this episode.
NOT_APPLICABLE = "not_applicable"

# ── Verification tier ────────────────────────────────────────────────────
#: The artifact parses, imports and satisfies the Protocol. Says nothing about
#: whether it works.
CONFORMANCE = "conformance"
#: A candidate *declared* it provides the capability and its declared affordances
#: are present. Still says nothing about whether it works -- a structurally
#: perfect adapter aimed at the wrong thing passes this.
DECLARED_FITNESS = "declared_fitness"
#: The declared ``expected_effect`` was compared against an observed outcome.
#: The only tier that can show a capability actually delivered.
OBSERVED_EFFECT = "observed_effect"


class EvolutionStage(str, Enum):
    """Which question a trace answers."""

    OBSERVE = "observe"   # what changed in the environment, or what did the teacher conclude
    ORIENT = "orient"     # which capability is therefore missing
    DECIDE = "decide"     # should it change, how, and why this candidate
    ACT = "act"           # what the framework actually did
    LEARN = "learn"       # how it went, and did the gap really close


@dataclass(frozen=True)
class EvolutionTrace:
    """One atomic fact in an evolution episode.

    ``correlation`` carries the natural keys that stitch traces into one episode;
    every one of them already exists upstream (``intent_id``, ``requirement_id``,
    ``record_id``, ``observation_id``, ``lifecycle_proposal_id``, ``plugin_id``,
    ``registry_version``), which is why no pervasive new identifier had to be
    threaded through the core to make this work.

    ``detail`` is a domain-private escape hatch, exactly like ``Finding.payload``:
    core code never inspects it, only the ledger and the view do. That is what
    lets a later stage add fields without touching anything upstream.
    """

    stage: EvolutionStage
    kind: str
    ts: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation: Mapping[str, str] = field(default_factory=dict)
    summary: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stage": self.stage.value,
            "kind": self.kind,
            "ts": self.ts,
            "correlation": dict(self.correlation),
            "summary": self.summary,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class EvolutionEpisode:
    """One causal chain: an environment change, and the framework change it drove.

    The derived fields below are computed once when the episode is assembled so a
    view can bind them directly. They are not decoration: recomputing them in a
    template or a frontend is how two surfaces end up disagreeing about the same
    episode.

    Fields a stage cannot yet establish stay at their empty default rather than
    being guessed. ``effect_verdict`` is the clearest case: verification by
    observed effect exists in the tree but nothing calls it, so an episode today
    can honestly report ``verification_tier == DECLARED_FITNESS`` and must not
    imply more.
    """

    episode_id: str
    opened_at: float
    status: str = OPEN
    traces: tuple[EvolutionTrace, ...] = ()
    closed_at: float = 0.0

    # ── What set this in motion ──
    driver: str = ""                    # world_model | unknown_tool | environment_probe | manual
    capability: str = ""
    intent_id: str = ""
    hypothesis: str = ""                # the teacher's own words, when it authored this
    confidence: float = 0.0             # model self-report; prioritisation only, never permission

    # ── What the framework did ──
    mutation_action: str = ""           # install | reload | disable | remove | rollback | none
    plugin_id: str = ""
    registry_before: int = -1
    registry_after: int = -1

    # ── Governance state (two vocabularies, never merged) ──
    lifecycle_status: str = ""          # PENDING … VERIFIED / QUARANTINED (acquisition journey)
    review_status: str = ""             # draft | review | approved | rejected (human review)
    policy_action: str = ""
    autonomy_level: str = ""

    # ── How it turned out ──
    gap_closure: str = NOT_APPLICABLE
    verification_tier: str = ""
    effect_verdict: str = ""
    trust_at_decision: str = ""
    trust_now: str = ""
    outcome: str = ""

    @property
    def framework_changed(self) -> bool:
        """Whether the registry version actually moved.

        The hard test for "the framework really changed", as opposed to a decision
        that merely intended to change it.
        """
        return self.registry_after > self.registry_before >= 0

    @property
    def stages_present(self) -> frozenset[EvolutionStage]:
        return frozenset(trace.stage for trace in self.traces)

    def trace_of(self, stage: EvolutionStage) -> EvolutionTrace | None:
        return next((trace for trace in self.traces if trace.stage is stage), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "status": self.status,
            "framework_changed": self.framework_changed,
            "stages": sorted(stage.value for stage in self.stages_present),
            "driver": self.driver,
            "capability": self.capability,
            "intent_id": self.intent_id,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "mutation_action": self.mutation_action,
            "plugin_id": self.plugin_id,
            "registry_before": self.registry_before,
            "registry_after": self.registry_after,
            "lifecycle_status": self.lifecycle_status,
            "review_status": self.review_status,
            "policy_action": self.policy_action,
            "autonomy_level": self.autonomy_level,
            "gap_closure": self.gap_closure,
            "verification_tier": self.verification_tier,
            "effect_verdict": self.effect_verdict,
            "trust_at_decision": self.trust_at_decision,
            "trust_now": self.trust_now,
            "outcome": self.outcome,
            "traces": [trace.to_dict() for trace in self.traces],
        }


@runtime_checkable
class EvolutionTraceSink(Protocol):
    """Consumes evolution traces. Absent by default, so emitting is a no-op.

    Declared here, in the domain layer, so a probe at a mutation point depends on
    this contract and never on the ledger, the monitor subsystem, or an event bus.
    The implementation is injected by the daemon; without it the framework behaves
    exactly as it did before this module existed.
    """

    def record(self, trace: EvolutionTrace) -> None:
        ...


__all__ = [
    "ABORTED",
    "COMMITTED",
    "CONFORMANCE",
    "DECLARED_FITNESS",
    "NOT_APPLICABLE",
    "OBSERVED_EFFECT",
    "OPEN",
    "REOPENED",
    "RESOLVED",
    "STILL_OPEN",
    "EvolutionEpisode",
    "EvolutionStage",
    "EvolutionTrace",
    "EvolutionTraceSink",
]

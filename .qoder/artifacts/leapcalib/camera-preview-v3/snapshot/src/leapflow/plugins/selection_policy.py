# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pluggable selection policy: which admissible candidate to use, and why.

The resolver scores candidates; a policy chooses among them. Splitting the two is
what makes a learning strategy expressible at all. A ``CapabilityScorer`` sees one
candidate at a time and returns a number, which is enough for a weighted sum and
insufficient for anything else: an upper-confidence bound needs the total pull count
across *all* candidates, Thompson sampling needs to draw for all of them and compare,
and both need somewhere to put a posterior and a way to be told what happened.

Three properties of this seam are load-bearing:

* **The policy only ever sees admissible candidates.** Hard constraints -- a risk
  ceiling, a missing platform affordance, a frozen plugin -- are applied by scorers
  marking a component ``excluded``, and excluded candidates are filtered out before
  the policy is consulted. Exploration must not be able to reach a tool the risk
  policy refused; safety is not a term to be traded off.
* **The policy explains itself.** ``SelectionOutcome.reason`` is rendered on the
  evolution board beside the choice. "Sampled at random" is not an explanation; a
  posterior and an exploration bonus are. Without this a learning policy makes the
  decision history unauditable, which costs more than the regret it saves.
* **Reward may abstain.** ``RewardSignal.value is None`` means "no information",
  not "failure". The effect channel is three-valued and its abstain class is large:
  a successful call whose handler reported no observable effect is the normal state
  for every tool written before the convention existed. A policy that folded
  abstention into failure would drive every arm's posterior toward zero and, through
  trust, quarantine healthy plugins for a reporting omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.domain.capability_requirement import CapabilityRequirement
    from leapflow.plugins.capability_resolver import CandidateScore, ResolverContext


class RewardSignal(NamedTuple):
    """What was learned from one selection, or that nothing was.

    ``value`` is ``None`` for an abstention and must leave a posterior untouched.
    ``source`` names the channel so a policy can weight an execution result
    differently from a verified effect -- they answer different questions ("did the
    call work" versus "did the capability deliver").
    """

    value: float | None
    source: str = ""
    confidence: float = 1.0

    @property
    def informative(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class SelectionOutcome:
    """One policy's choice, with the explanation the board renders."""

    selected: CandidateScore
    policy_id: str
    reason: str = ""
    #: Whether this choice departed from the highest-scoring candidate. The single
    #: most important thing to surface about a learning policy: an operator seeing a
    #: lower-scored tool selected must be able to tell deliberate exploration from a
    #: scoring bug.
    explored: bool = False
    arbitration_used: bool = False


@dataclass(frozen=True)
class PolicyDeps:
    """Host-side services a policy may read. All optional, absence degrades.

    Injected at construction rather than reached for globally so a policy is
    testable in isolation, and so the set of things a policy is allowed to touch is
    visible in one place.
    """

    trust_ledger: Any = None
    usage_tracker: Any = None
    #: Optional tie-break hook, typically LLM-backed. A live object rather than a
    #: configuration value, which is why it travels here and not in ``params``.
    arbiter: Any = None
    #: Durable home for whatever state a policy accumulates. ``None`` for the shipped
    #: set, which is stateless. A policy that kept state in memory only would restart
    #: cold on every daemon restart and never converge -- the trap trust already
    #: learned (it flushes on level transitions plus ``atexit``) -- so the slot exists
    #: for the case rather than being invented when it arrives.
    stats_store: Any = None


@runtime_checkable
class SelectionPolicy(Protocol):
    """Chooses among admissible candidates and learns from the result."""

    policy_id: str

    def select(
        self,
        requirement: CapabilityRequirement,
        eligible: Sequence[CandidateScore],
        context: ResolverContext,
    ) -> SelectionOutcome:
        """Pick one candidate. ``eligible`` is never empty and never excluded."""
        ...

    def observe(
        self,
        requirement: CapabilityRequirement,
        chosen_tool: str,
        reward: RewardSignal,
    ) -> None:
        """Record what came of a selection. Must ignore an abstaining reward."""
        ...


@runtime_checkable
class SelectionPolicyPlugin(Protocol):
    """Declares a policy and builds it from configuration.

    Deliberately shaped like ``LLMProviderPlugin`` rather than ``ToolPlugin``: a
    selection policy is invoked by the framework inside a turn and needs host-side
    services, so it must not be sandboxed, and it earns no Progressive Trust because
    it executes no tools -- a trust level for it would be a meaningless number that
    the plugin roster would nonetheless render.
    """

    @property
    def policy_id(self) -> str:
        """Stable identifier used in config, e.g. ``greedy``, ``thompson``."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for the config catalog and diagnostics."""
        ...

    def create(self, params: Mapping[str, Any], deps: PolicyDeps) -> SelectionPolicy:
        """Build a configured policy. ``params`` is the open per-policy dict.

        Open rather than a typed schema on purpose: a fixed schema would mean
        editing core settings to add a strategy, which is the opposite of the
        extensibility this seam exists for.
        """
        ...


@dataclass(frozen=True)
class PolicyDescriptor:
    """What the config catalog and the board show about an available policy."""

    policy_id: str
    display_name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "display_name": self.display_name,
            "params": dict(self.params),
        }


__all__ = [
    "PolicyDeps",
    "PolicyDescriptor",
    "RewardSignal",
    "SelectionOutcome",
    "SelectionPolicy",
    "SelectionPolicyPlugin",
]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deterministic adaptive plugin capability resolution.

The resolver answers: given structured requirements and the current environment,
which live plugin tools are best suited, and why were other candidates rejected
or ranked lower? It is intentionally metadata-driven: no natural-language keyword
matching and no hidden intent classification occur here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from leapflow.domain.capability_requirement import CapabilityRequirement
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.learning.plugin_stats import PluginUsageTracker
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins._builtin_policies import GreedyPolicy
from leapflow.plugins.protocol import ToolMetadata
from leapflow.plugins.selection_policy import SelectionPolicy

_RISK_RANK = {
    "read_only": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "mutating": 4,
    "external": 5,
}
_MAX_RISK_RANK = max(_RISK_RANK.values())


def _as_tuple(values: Sequence[str] | str | None = None) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(str(v) for v in values if str(v))


@dataclass(frozen=True)
class CapabilityCandidate:
    """One live tool candidate owned by a plugin."""

    plugin_id: str
    tool_name: str
    description: str = ""
    provides_capabilities: tuple[str, ...] = field(default_factory=tuple)
    requires_capabilities: tuple[str, ...] = field(default_factory=tuple)
    requires_platform_capabilities: tuple[str, ...] = field(default_factory=tuple)
    requires_environment_affordances: tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "read_only"
    requires_approval: bool = False
    mutates_state: bool = False
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_tool(cls, plugin_id: str, tool: ToolMetadata) -> "CapabilityCandidate":
        """Create a candidate from ToolMetadata."""
        raw = dict(tool.x_leapflow or {})
        return cls(
            plugin_id=str(plugin_id),
            tool_name=tool.name,
            description=tool.description,
            provides_capabilities=_as_tuple(
                tool.provides_capabilities
                or tuple(raw.get("provides_capabilities") or ())
            ),
            requires_capabilities=_as_tuple(
                tool.requires_capabilities
                or tuple(raw.get("requires_capabilities") or ())
            ),
            requires_platform_capabilities=_as_tuple(
                tool.requires_platform_capabilities
                or tuple(raw.get("requires_platform_capabilities") or ())
            ),
            requires_environment_affordances=_as_tuple(
                getattr(tool, "requires_environment_affordances", ())
                or tuple(raw.get("requires_environment_affordances") or ())
            ),
            risk_level=str(raw.get("risk_level") or "read_only"),
            requires_approval=bool(raw.get("requires_approval", False)),
            mutates_state=bool(tool.mutates_state or raw.get("mutates_state", False)),
            metadata=tuple(sorted((str(k), str(v)) for k, v in raw.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "tool_name": self.tool_name,
            "description": self.description,
            "provides_capabilities": list(self.provides_capabilities),
            "requires_capabilities": list(self.requires_capabilities),
            "requires_platform_capabilities": list(self.requires_platform_capabilities),
            "requires_environment_affordances": list(self.requires_environment_affordances),
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "mutates_state": self.mutates_state,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResolverWeights:
    """Configurable weights for deterministic candidate scoring."""

    declared_match: float = 1.0
    environment_fit: float = 1.0
    risk_cost: float = 1.0
    trust: float = 1.0
    reliability: float = 1.0
    #: Weight for the teacher's rebind recommendation. Deliberately below the structural
    #: weights: a hindsight recommendation is evidence, and it must not outvote a
    #: declaration that a candidate cannot run here.
    distilled_preference: float = 0.5


@dataclass(frozen=True)
class ResolverContext:
    """Read-only evidence available to scorers."""

    environment: EnvironmentFingerprint
    trust_ledger: PluginTrustLedger | None = None
    usage_tracker: PluginUsageTracker | None = None
    weights: ResolverWeights = field(default_factory=ResolverWeights)
    #: ``capability -> preferred plugin or tool name``, from the teacher's ``rebind``
    #: verdicts. Read-only evidence like everything else here: the resolver still decides.
    distilled_preferences: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScoreComponent:
    """One scorer's contribution and explanation."""

    scorer: str
    score: float
    weight: float
    reason: str
    excluded: bool = False

    @property
    def weighted_score(self) -> float:
        return 0.0 if self.excluded else self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "scorer": self.scorer,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "reason": self.reason,
            "excluded": self.excluded,
        }


@dataclass(frozen=True)
class CandidateScore:
    """Scored candidate with all explanation fragments retained."""

    candidate: CapabilityCandidate
    components: tuple[ScoreComponent, ...]

    @property
    def eligible(self) -> bool:
        return not any(c.excluded for c in self.components)

    @property
    def total_score(self) -> float:
        return round(sum(c.weighted_score for c in self.components), 6)

    @property
    def exclusion_reasons(self) -> tuple[str, ...]:
        return tuple(c.reason for c in self.components if c.excluded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "eligible": self.eligible,
            "total_score": self.total_score,
            "components": [c.to_dict() for c in self.components],
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True)
class CapabilityResolution:
    """Transparent decision for one requirement."""

    requirement: CapabilityRequirement
    candidates: tuple[CandidateScore, ...]
    selected: CandidateScore | None = None
    arbitration_used: bool = False
    #: Which policy chose, and whether it departed from the argmax. Recorded so a
    #: decision stays auditable once selection is pluggable: an operator seeing a
    #: lower-scored tool selected must be able to tell exploration from a bug.
    policy_id: str = ""
    explored: bool = False
    reason: str = ""

    @property
    def unmet(self) -> bool:
        return self.selected is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement.to_dict(),
            "selected": self.selected.to_dict() if self.selected else None,
            "unmet": self.unmet,
            "arbitration_used": self.arbitration_used,
            "policy_id": self.policy_id,
            "explored": self.explored,
            "reason": self.reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@runtime_checkable
class CapabilityScorer(Protocol):
    """Protocol for pluggable deterministic scoring dimensions."""

    name: str

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        ...


@runtime_checkable
class CapabilityArbiter(Protocol):
    """Optional tie-break hook, typically LLM-backed outside deterministic tests."""

    def choose(
        self,
        requirement: CapabilityRequirement,
        tied: Sequence[CandidateScore],
        context: ResolverContext,
    ) -> str | None:
        """Return the selected tool_name among tied candidates, or None."""
        ...


class DeclaredMatchScorer:
    name = "declared_match"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        if requirement.capability in candidate.provides_capabilities:
            return ScoreComponent(
                self.name,
                1.0,
                context.weights.declared_match,
                f"candidate declares capability {requirement.capability!r}",
            )
        return ScoreComponent(
            self.name,
            0.0,
            context.weights.declared_match,
            f"candidate does not declare capability {requirement.capability!r}",
            excluded=True,
        )


class EnvironmentFitScorer:
    name = "environment_fit"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        required = tuple(
            dict.fromkeys(
                requirement.required_platform_capabilities
                + candidate.requires_platform_capabilities
            )
        )
        missing = tuple(c for c in required if not context.environment.supports_capability(c))
        if missing:
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.environment_fit,
                "missing platform capabilities: " + ", ".join(missing),
                excluded=True,
            )
        return ScoreComponent(
            self.name,
            1.0,
            context.weights.environment_fit,
            "all required platform capabilities are present",
        )


class EnvironmentAffordanceScorer:
    """Exclude a candidate whose declared app-level affordances the task
    environment does not offer.

    Mirrors ``EnvironmentFitScorer`` but reads the candidate's
    ``requires_environment_affordances`` (task/app-level) rather than its host
    ``requires_platform_capabilities``. The two are separate declarations so a
    tool that needs ``ui.chat.send.v2`` is excluded when the app presents v1,
    without conflating that with a host capability. Not in ``_DEFAULT_SCORERS``:
    it is injected explicitly (``CapabilityResolver(scorers=...)``) by callers
    that resolve against a task environment, so default resolution is unchanged.
    """

    name = "environment_affordance"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        required = candidate.requires_environment_affordances
        missing = tuple(
            affordance
            for affordance in required
            if not context.environment.supports_capability(affordance)
        )
        if missing:
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.environment_fit,
                "missing environment affordances: " + ", ".join(missing),
                excluded=True,
            )
        return ScoreComponent(
            self.name,
            1.0,
            context.weights.environment_fit,
            "all required environment affordances are present",
        )


class RiskCostScorer:
    name = "risk_cost"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        rank = _RISK_RANK.get(candidate.risk_level, _MAX_RISK_RANK)
        max_rank = _RISK_RANK.get(requirement.max_risk_level, _MAX_RISK_RANK)
        if rank > max_rank:
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.risk_cost,
                f"risk {candidate.risk_level!r} exceeds max {requirement.max_risk_level!r}",
                excluded=True,
            )
        if candidate.requires_approval and not requirement.allows_autonomous_approval:
            reason = "requires approval; approval mode remains review_required"
        else:
            reason = f"risk {candidate.risk_level!r} is within requirement limit"
        return ScoreComponent(
            self.name,
            1.0 - (rank / _MAX_RISK_RANK),
            context.weights.risk_cost,
            reason,
        )


class TrustScorer:
    name = "trust"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        if context.trust_ledger is None:
            return ScoreComponent(self.name, 0.0, context.weights.trust, "trust ledger unavailable")
        level = context.trust_ledger.level(candidate.plugin_id)
        return ScoreComponent(
            self.name,
            float(level) / float(PluginTrustLevel.PRODUCTION),
            context.weights.trust,
            f"plugin trust level is {level.name}",
        )


class FrozenExclusionScorer:
    """Exclude a candidate whose plugin trust has been permanently frozen.

    Defense in depth at the selection layer. ``TrustScorer`` only *scores* trust,
    so a plugin frozen by an internal defect stays selectable for as long as it
    remains registered -- "frozen implies never re-selected" holds today only
    because ``LifecycleGovernor`` also quarantines (and thus unregisters) on the
    same event. Any path that freezes trust without unregistering would leave the
    plugin eligible; this scorer closes that independently of governance.

    Not in ``_DEFAULT_SCORERS``: it is injected explicitly
    (``CapabilityResolver(scorers=...)``), so default resolution is unchanged.
    """

    name = "frozen_exclusion"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        ledger = context.trust_ledger
        is_frozen = getattr(ledger, "is_frozen", None) if ledger is not None else None
        if callable(is_frozen) and is_frozen(candidate.plugin_id):
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.trust,
                f"plugin {candidate.plugin_id!r} is frozen by an internal defect",
                excluded=True,
            )
        return ScoreComponent(
            self.name,
            1.0,
            context.weights.trust,
            "plugin is not frozen",
        )


class ReliabilityScorer:
    name = "reliability"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        if context.usage_tracker is None:
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.reliability,
                "usage tracker unavailable",
            )
        stats = context.usage_tracker.stats_for_plugin(candidate.plugin_id)
        if stats is None:
            return ScoreComponent(
                self.name,
                0.0,
                context.weights.reliability,
                "no usage samples for plugin",
            )
        return ScoreComponent(
            self.name,
            max(0.0, 1.0 - stats.error_rate),
            context.weights.reliability,
            f"error_rate={stats.error_rate:.4f}, p95_ms={stats.p95_duration_ms:.2f}",
        )


class DistilledPreferenceScorer:
    """Prefer the provider the teacher named in a ``rebind`` verdict.

    This is channel C2, and until now the teacher's most frequent recommendation had
    nowhere to land: a ``rebind`` naming ``chat_reply_v2_native`` reached the student as a
    line of prose and the selection layer never heard about it. Measured on a real model,
    ``rebind`` was the answer it reached for most readily -- so leaving it inert wasted the
    verdict the loop produces most.

    A preference, never a gate, and weighted below the structural scorers on purpose:

    * It **adds** to a candidate's score rather than excluding its rivals, so a
      recommendation cannot make an inadmissible candidate win -- affordance and frozen
      exclusions still apply and still exclude.
    * It cannot outvote a declaration. Hindsight is evidence about the world; a
      declaration is a fact about the code, and when they disagree the code wins.
    * It expires with the knowledge that produced it. The entry is retracted when the
      capability recovers and superseded by the next verdict, so a stale preference stops
      being read rather than having to be unlearned.
    """

    name = "distilled_preference"

    def score(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> ScoreComponent:
        preferred = dict(context.distilled_preferences).get(requirement.capability, "")
        if not preferred:
            return ScoreComponent(
                self.name, 0.0, context.weights.distilled_preference, "no recommendation"
            )
        matched = preferred in (candidate.plugin_id, candidate.tool_name)
        return ScoreComponent(
            self.name,
            1.0 if matched else 0.0,
            context.weights.distilled_preference,
            f"teacher recommended {preferred}"
            + ("" if matched else f"; this candidate is {candidate.plugin_id}"),
        )


_DEFAULT_SCORERS: tuple[CapabilityScorer, ...] = (
    DeclaredMatchScorer(),
    EnvironmentFitScorer(),
    RiskCostScorer(),
    TrustScorer(),
    ReliabilityScorer(),
)


class CapabilityResolver:
    """Resolve structured requirements to the best live plugin tool candidates."""

    def __init__(
        self,
        scorers: Sequence[CapabilityScorer] = _DEFAULT_SCORERS,
        policy: SelectionPolicy | None = None,
    ) -> None:
        self._scorers = tuple(scorers)
        # Greedy by default, which reproduces the selection this resolver made before
        # the seam existed. Built directly rather than through the registry so a
        # resolver constructed in a test needs no process-wide state.
        self._policy: SelectionPolicy = policy or GreedyPolicy()

    def resolve_all(
        self,
        requirements: Sequence[CapabilityRequirement],
        candidates: Sequence[CapabilityCandidate],
        context: ResolverContext,
    ) -> tuple[CapabilityResolution, ...]:
        """Resolve multiple requirements independently."""
        return tuple(self.resolve_one(r, candidates, context) for r in requirements)

    def resolve_one(
        self,
        requirement: CapabilityRequirement,
        candidates: Sequence[CapabilityCandidate],
        context: ResolverContext,
    ) -> CapabilityResolution:
        """Score every candidate, then let the policy choose among the admissible ones.

        The split is the seam: scoring is per-candidate and stateless, selection sees
        the whole admissible set and may carry state. Only ``eligible`` reaches the
        policy -- a candidate excluded by a risk ceiling, a missing affordance or a
        frozen plugin is filtered out first, so exploration can never reach a tool the
        safety layer refused.
        """
        scored = tuple(self._score_candidate(requirement, c, context) for c in candidates)
        eligible = tuple(c for c in scored if c.eligible)
        if not eligible:
            return CapabilityResolution(
                requirement=requirement,
                candidates=scored,
                selected=None,
                reason="no eligible candidate declared the required capability and environment fit",
            )
        outcome = self._policy.select(requirement, eligible, context)
        return CapabilityResolution(
            requirement=requirement,
            candidates=tuple(sorted(scored, key=self._sort_key)),
            selected=outcome.selected,
            arbitration_used=outcome.arbitration_used,
            policy_id=outcome.policy_id,
            explored=outcome.explored,
            reason=(
                f"selected {outcome.selected.candidate.tool_name!r} by "
                f"{outcome.policy_id}: {outcome.reason}"
            ),
        )

    def _score_candidate(
        self,
        requirement: CapabilityRequirement,
        candidate: CapabilityCandidate,
        context: ResolverContext,
    ) -> CandidateScore:
        return CandidateScore(
            candidate=candidate,
            components=tuple(s.score(requirement, candidate, context) for s in self._scorers),
        )

    @staticmethod
    def _sort_key(score: CandidateScore) -> tuple[bool, float, str, str]:
        return (not score.eligible, -score.total_score, score.candidate.plugin_id, score.candidate.tool_name)


def candidates_from_registry(registry: Any) -> tuple[CapabilityCandidate, ...]:
    """Build candidates from the registry's live, conflict-resolved catalog."""
    owners = getattr(registry, "tool_owners", {})
    result: list[CapabilityCandidate] = []
    for plugin_id, plugin in registry.plugins.items():
        for tool in plugin.tools:
            if owners and owners.get(tool.name) != plugin_id:
                continue
            if tool.name not in registry.tool_handlers:
                continue
            result.append(CapabilityCandidate.from_tool(plugin_id, tool))
    return tuple(result)

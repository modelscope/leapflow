# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in selection policies. One module registering many, like the LLM providers.

``GreedyPolicy`` is the shipped default and reproduces the resolver's previous
selection exactly: highest weighted score, ties broken by a stable sort on
``(plugin_id, tool_name)``, with an optional arbiter consulted only among ties. That
equivalence is the point -- introducing the seam must change no behaviour, so the
first policy is measured against what it replaced rather than described as similar.

The arbiter moved here from the resolver deliberately. A tie-break is a property of
*greedy* scoring: under Thompson sampling two candidates never tie, because each draw
is continuous. Leaving it in the resolver would have made every future policy inherit
a hook that means nothing to it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from leapflow.plugins.selection_policy import (
    PolicyDeps,
    RewardSignal,
    SelectionOutcome,
    SelectionPolicy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.domain.capability_requirement import CapabilityRequirement
    from leapflow.plugins.capability_resolver import CandidateScore, ResolverContext
    from leapflow.plugins.selection_policy_registry import SelectionPolicyRegistry

logger = logging.getLogger(__name__)


class GreedyPolicy:
    """Always the highest-scoring admissible candidate. No exploration, no state.

    The honest baseline, and also the reason a learning policy is worth building: a
    candidate with no usage samples scores zero on reliability and, at ``DRAFT``, zero
    on trust -- so against an incumbent with any history it loses deterministically.
    Trust is earned by being selected, and selection requires trust, which closes a
    loop that no amount of tuning the weights opens. This policy cannot escape that;
    naming it here is what makes the next one a decision rather than a preference.
    """

    policy_id = "greedy"

    def __init__(self, arbiter: Any = None) -> None:
        self._arbiter = arbiter

    def select(
        self,
        requirement: CapabilityRequirement,
        eligible: Sequence[CandidateScore],
        context: ResolverContext,
    ) -> SelectionOutcome:
        top_score = max(c.total_score for c in eligible)
        tied = tuple(c for c in eligible if c.total_score == top_score)
        selected = _stable_first(tied)
        arbitration_used = False
        if len(tied) > 1 and self._arbiter is not None:
            try:
                chosen = self._arbiter.choose(requirement, tied, context)
            except Exception:  # noqa: BLE001 - an advisory hook must not fail selection
                logger.debug("greedy: arbiter failed", exc_info=True)
                chosen = None
            picked = next((c for c in tied if c.candidate.tool_name == chosen), None)
            if picked is not None:
                selected = picked
                arbitration_used = True
        return SelectionOutcome(
            selected=selected,
            policy_id=self.policy_id,
            reason=(
                f"highest score {top_score:.3f}"
                + (f" among {len(tied)} tied" if len(tied) > 1 else "")
                + (" (arbitrated)" if arbitration_used else "")
            ),
            # Greedy is the argmax by construction, so it never explores. Stated
            # rather than left to default: the board reads this field to tell
            # deliberate exploration from a scoring bug.
            explored=False,
            arbitration_used=arbitration_used,
        )

    def observe(
        self,
        requirement: CapabilityRequirement,
        chosen_tool: str,
        reward: RewardSignal,
    ) -> None:
        """Greedy learns nothing. Accepting the call keeps the seam uniform.

        A no-op rather than an omission: the feedback edge is wired for every policy,
        so adding a learning one is a new file and a config value rather than a change
        to the call sites that report outcomes.
        """
        return None


def _stable_first(scores: Sequence[CandidateScore]) -> CandidateScore:
    """Deterministic tie-break, identical to the resolver's previous rule."""
    return sorted(scores, key=lambda s: (s.candidate.plugin_id, s.candidate.tool_name))[0]


class GreedyPolicyPlugin:
    """Declares :class:`GreedyPolicy` to the registry."""

    @property
    def policy_id(self) -> str:
        return GreedyPolicy.policy_id

    @property
    def display_name(self) -> str:
        return "Greedy (highest score)"

    def create(self, params: Mapping[str, Any], deps: PolicyDeps) -> SelectionPolicy:
        # ``arbiter`` arrives through deps rather than params: it is a live object,
        # not a configuration value, and config carries no object references.
        return GreedyPolicy(arbiter=deps.arbiter)


def register_builtin_policies(registry: SelectionPolicyRegistry) -> None:
    """Register every built-in policy. One call site, so the set is auditable."""
    registry.register(GreedyPolicyPlugin())


__all__ = [
    "GreedyPolicy",
    "GreedyPolicyPlugin",
    "register_builtin_policies",
]

"""Policy decisions for adaptive plugin evolution.

The policy is intentionally metadata-driven. It never inspects natural-language
user intent; callers supply structured requirements, risk, trust, and status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from leapflow.learning.plugin_trust import PluginTrustLevel
from leapflow.plugins.evolution_contracts import EvolutionProposalView

AutonomyLevel = Literal[
    "observe_only",
    "proposal_only",
    "generate_only",
    "approve_to_install",
    "trusted_autonomous",
    "production_autonomous",
]
PolicyAction = Literal[
    "observe_only",
    "propose",
    "generate",
    "request_approval",
    "install",
    "probation_execute",
    "disable",
    "rollback",
    "quarantine",
    "none",
]

_AUTONOMY_RANK = {
    "observe_only": 0,
    "proposal_only": 1,
    "generate_only": 2,
    "approve_to_install": 3,
    "trusted_autonomous": 4,
    "production_autonomous": 5,
}
_RISK_RANK = {
    "none": 0,
    "read_only": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "mutating": 4,
    "external": 5,
}


@dataclass(frozen=True)
class AdaptivePolicyDecision:
    """One deterministic next-action decision for a proposal."""

    action: PolicyAction
    reason: str
    autonomy_level: AutonomyLevel
    requires_approval: bool = False
    allowed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "autonomy_level": self.autonomy_level,
            "requires_approval": self.requires_approval,
            "allowed": self.allowed,
            "metadata": dict(self.metadata),
        }


class AdaptiveEvolutionPolicy:
    """Decide next evolution actions from structured governance state."""

    def __init__(self, *, autonomy_level: AutonomyLevel = "observe_only") -> None:
        if autonomy_level not in _AUTONOMY_RANK:
            raise ValueError(f"unknown autonomy level: {autonomy_level}")
        self._autonomy_level = autonomy_level

    @property
    def autonomy_level(self) -> AutonomyLevel:
        return self._autonomy_level

    def decide(
        self,
        proposal: EvolutionProposalView,
        *,
        trust_level: PluginTrustLevel | str | int = PluginTrustLevel.DRAFT,
        usage: Mapping[str, Any] | None = None,
        sandbox_validated: bool = False,
        rollback_available: bool = False,
    ) -> AdaptivePolicyDecision:
        """Return the next safe action for a proposal queue item."""
        risk_level = _risk_level(proposal)
        risk_rank = _RISK_RANK.get(risk_level, _RISK_RANK["external"])
        rank = _AUTONOMY_RANK[self._autonomy_level]
        trust = _coerce_trust(trust_level)
        usage_payload = dict(usage or {})
        failure_streak = int(usage_payload.get("consecutive_failures") or 0)
        hard_failure = bool(usage_payload.get("hard_failure", False))

        if hard_failure:
            return AdaptivePolicyDecision(
                "quarantine",
                "hard failure freezes plugin trust and requires quarantine",
                self._autonomy_level,
                requires_approval=False,
                metadata={"risk_level": risk_level},
            )
        if failure_streak >= 3:
            return AdaptivePolicyDecision(
                "rollback" if rollback_available else "disable",
                "failure streak exceeded lifecycle threshold",
                self._autonomy_level,
                requires_approval=not rollback_available,
                metadata={"failure_streak": failure_streak, "risk_level": risk_level},
            )

        status = proposal.status
        if rank <= _AUTONOMY_RANK["observe_only"]:
            return AdaptivePolicyDecision(
                "observe_only",
                "autonomy level permits observation only",
                self._autonomy_level,
                metadata={"proposal_status": status, "risk_level": risk_level},
            )
        if status == "PENDING":
            if rank == _AUTONOMY_RANK["proposal_only"]:
                return AdaptivePolicyDecision(
                    "propose",
                    "proposal is queued for human review",
                    self._autonomy_level,
                    metadata={"risk_level": risk_level},
                )
            return AdaptivePolicyDecision(
                "generate",
                "policy permits generating a validated artifact before install",
                self._autonomy_level,
                metadata={"risk_level": risk_level},
            )
        if status == "GENERATED":
            if risk_rank > _RISK_RANK["read_only"] or rank < _AUTONOMY_RANK["trusted_autonomous"]:
                return AdaptivePolicyDecision(
                    "request_approval",
                    "generated artifact requires approval before installation",
                    self._autonomy_level,
                    requires_approval=True,
                    metadata={"risk_level": risk_level},
                )
            if not sandbox_validated:
                return AdaptivePolicyDecision(
                    "request_approval",
                    "sandbox validation evidence is missing",
                    self._autonomy_level,
                    requires_approval=True,
                    allowed=False,
                    metadata={"risk_level": risk_level},
                )
            return AdaptivePolicyDecision(
                "install",
                "trusted autonomous policy permits read-only sandbox-validated install",
                self._autonomy_level,
                requires_approval=False,
                metadata={"risk_level": risk_level, "trust_level": trust.name},
            )
        if status in {"APPROVED", "INSTALLED"}:
            return AdaptivePolicyDecision(
                "probation_execute",
                "installed proposal should gather probation usage evidence",
                self._autonomy_level,
                metadata={"risk_level": risk_level, "trust_level": trust.name},
            )
        if status == "PROBATION":
            if trust >= PluginTrustLevel.VERIFIED:
                return AdaptivePolicyDecision(
                    "none",
                    "probation complete; proposal can be marked verified",
                    self._autonomy_level,
                    metadata={"trust_level": trust.name},
                )
            return AdaptivePolicyDecision(
                "probation_execute",
                "more successful usage is required before verification",
                self._autonomy_level,
                metadata={"trust_level": trust.name},
            )
        return AdaptivePolicyDecision(
            "none",
            "terminal proposal state has no automatic next action",
            self._autonomy_level,
            metadata={"proposal_status": status, "risk_level": risk_level},
        )


def _risk_level(proposal: EvolutionProposalView) -> str:
    risk = dict(proposal.risk or {})
    value = str(risk.get("risk_level") or risk.get("max_risk_level") or "read_only")
    for requirement in proposal.requirements:
        if isinstance(requirement, Mapping):
            value = str(requirement.get("max_risk_level") or value)
    return value


def _coerce_trust(value: PluginTrustLevel | str | int) -> PluginTrustLevel:
    if isinstance(value, PluginTrustLevel):
        return value
    if isinstance(value, int):
        try:
            return PluginTrustLevel(value)
        except ValueError:
            return PluginTrustLevel.DRAFT
    try:
        return PluginTrustLevel[str(value)]
    except KeyError:
        return PluginTrustLevel.DRAFT


__all__ = ["AdaptiveEvolutionPolicy", "AdaptivePolicyDecision", "AutonomyLevel", "PolicyAction"]

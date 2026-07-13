"""Approval policy evaluation built on structured risk assessments."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from leapflow.security.actions import ActionDescriptor
from leapflow.security.risk import RiskAssessment, RiskLevel


class PolicyVerdict(str, Enum):
    """Policy result before human interaction."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    """Decision produced by ApprovalPolicyEngine."""

    verdict: PolicyVerdict
    reason: str = ""
    allow_permanent: bool = True


@runtime_checkable
class ApprovalPolicyRule(Protocol):
    """Optional extension rule for approval policy."""

    def check(self, action: ActionDescriptor, risk: RiskAssessment) -> PolicyDecision | None: ...


class ApprovalPolicyEngine:
    """Small policy engine: hardline deny, meaningful risk ask, safe allow."""

    def __init__(self, rules: list[ApprovalPolicyRule] | None = None) -> None:
        self._rules = list(rules or [])

    def evaluate(self, action: ActionDescriptor, risk: RiskAssessment) -> PolicyDecision:
        for rule in self._rules:
            decision = rule.check(action, risk)
            if decision is not None:
                return decision

        if risk.hardline or risk.level == RiskLevel.CRITICAL:
            return PolicyDecision(
                verdict=PolicyVerdict.DENY,
                reason="; ".join(risk.reasons) or "hardline_block",
                allow_permanent=False,
            )
        if risk.level in {RiskLevel.HIGH, RiskLevel.MEDIUM} or risk.score >= 0.35:
            return PolicyDecision(
                verdict=PolicyVerdict.ASK,
                reason="; ".join(risk.reasons) or "approval_required",
                allow_permanent=risk.allow_permanent,
            )
        return PolicyDecision(verdict=PolicyVerdict.ALLOW, reason="low_risk")

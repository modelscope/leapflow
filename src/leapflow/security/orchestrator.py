"""Approval orchestration: policy, grants, prompting, and audit."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from leapflow.security.actions import ActionDescriptor
from leapflow.security.grants import (
    ApprovalAuditLog,
    ApprovalGrant,
    ApprovalGrantStore,
    ApprovalScope,
    InMemoryApprovalGrantStore,
    grant_key,
)
from leapflow.security.policy import ApprovalPolicyEngine, PolicyVerdict
from leapflow.security.risk import DefaultRiskClassifier, RiskAssessment, RiskClassifier, RiskLevel


@dataclass(frozen=True)
class ApprovalResult:
    """Final approval decision consumed by tool handlers."""

    approved: bool
    decision: str
    action: ActionDescriptor
    risk: RiskAssessment
    scope: str = ApprovalScope.ONCE.value
    reason: str = ""
    user_consent: bool = False

    @property
    def denial_message(self) -> str:
        if self.approved:
            return ""
        if self.risk.hardline or self.risk.level == RiskLevel.CRITICAL:
            return (
                "BLOCKED: This action is prohibited by LeapFlow's hardline safety policy. "
                "Run it manually outside the agent if you genuinely need it."
            )
        return (
            "BLOCKED: User denied this action. The user has not consented to this outcome. "
            "Do not retry, rephrase, or attempt the same outcome through another tool. "
            "Ask the user for a revised instruction."
        )


class ApprovalOrchestrator:
    """Coordinates risk assessment, grant lookup, human approval, and audit."""

    def __init__(
        self,
        gate: Any,
        *,
        risk_classifier: RiskClassifier | None = None,
        policy: ApprovalPolicyEngine | None = None,
        grants: ApprovalGrantStore | None = None,
        audit: ApprovalAuditLog | None = None,
    ) -> None:
        self._gate = gate
        self._risk = risk_classifier or DefaultRiskClassifier()
        self._policy = policy or ApprovalPolicyEngine()
        self._grants = grants or InMemoryApprovalGrantStore()
        self._audit = audit or ApprovalAuditLog()

    @property
    def audit(self) -> ApprovalAuditLog:
        return self._audit

    @property
    def grants(self) -> ApprovalGrantStore:
        return self._grants

    async def evaluate(self, action: ActionDescriptor) -> ApprovalResult:
        """Return an approval result, prompting only when policy requires it."""
        from leapflow.security.approval import ApprovalDecision, ApprovalRequest

        risk = self._risk.assess(action)
        policy = self._policy.evaluate(action, risk)
        if policy.verdict == PolicyVerdict.ALLOW:
            return self._approved(action, risk, actor="policy", reason=policy.reason)
        if policy.verdict == PolicyVerdict.DENY:
            return self._denied(action, risk, actor="policy", reason=policy.reason)

        existing = self._existing_grant(action)
        if existing is not None:
            if existing.decision.startswith("deny"):
                return self._denied(action, risk, actor="grant", reason=existing.reason)
            return self._approved(action, risk, actor="grant", scope=existing.scope, reason=existing.reason)

        request = ApprovalRequest(
            category=action.kind,
            detail=action.detail,
            risk_hint=risk.score,
            metadata={
                "action": action.to_dict(),
                "risk": risk.to_dict(),
                "allow_permanent": policy.allow_permanent,
            },
            action=action,
            risk=risk,
            choices=self._choices(policy.allow_permanent),
            default_choice="deny" if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL} else "allow_once",
            expires_at=time.time() + 120.0,
            display={
                "title": self._title(risk),
                "summary": action.summary,
                "reason": risk.explanation,
            },
        )
        decision = await self._gate.request_approval(request)
        if decision in {
            ApprovalDecision.ALLOW,
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.ALLOW_SESSION,
            ApprovalDecision.ALLOW_ALWAYS,
        }:
            scope = self._scope_from_decision(decision)
            if scope in {ApprovalScope.SESSION.value, ApprovalScope.PROFILE.value}:
                self._grants.put(ApprovalGrant(
                    key=grant_key(action, ApprovalScope(scope)),
                    scope=scope,
                    decision="allow",
                    action_kind=action.kind,
                    effect=action.effect,
                    resource=action.resource,
                    reason="user_approved",
                ))
            return self._approved(action, risk, actor="user", scope=scope, reason=decision.value)

        if decision == ApprovalDecision.DENY_ALWAYS:
            self._grants.put(ApprovalGrant(
                key=grant_key(action, ApprovalScope.SESSION),
                scope=ApprovalScope.SESSION.value,
                decision="deny",
                action_kind=action.kind,
                effect=action.effect,
                resource=action.resource,
                reason="user_denied",
            ))
            return self._denied(
                action,
                risk,
                actor="user",
                reason=decision.value,
                scope=ApprovalScope.SESSION.value,
            )
        return self._denied(action, risk, actor="user", reason=decision.value)

    async def check(self, command: str) -> bool:
        """Legacy shell approval adapter used by existing tool gates."""
        result = await self.evaluate(ActionDescriptor.shell(command))
        return result.approved

    def _existing_grant(self, action: ActionDescriptor) -> ApprovalGrant | None:
        for scope in (ApprovalScope.TURN, ApprovalScope.SESSION, ApprovalScope.PROFILE):
            existing = self._grants.get(grant_key(action, scope))
            if existing is not None:
                return existing
        return None

    def _approved(
        self,
        action: ActionDescriptor,
        risk: RiskAssessment,
        *,
        actor: str,
        scope: str = ApprovalScope.ONCE.value,
        reason: str = "",
    ) -> ApprovalResult:
        self._audit.record(
            action=action,
            decision="allow",
            risk_level=risk.level.value,
            risk_reasons=risk.reasons,
            scope=scope,
            actor=actor,
            reason=reason,
        )
        return ApprovalResult(
            approved=True,
            decision="allow",
            action=action,
            risk=risk,
            scope=scope,
            reason=reason,
            user_consent=actor == "user",
        )

    def _denied(
        self,
        action: ActionDescriptor,
        risk: RiskAssessment,
        *,
        actor: str,
        reason: str = "",
        scope: str = ApprovalScope.ONCE.value,
    ) -> ApprovalResult:
        self._audit.record(
            action=action,
            decision="deny",
            risk_level=risk.level.value,
            risk_reasons=risk.reasons,
            scope=scope,
            actor=actor,
            reason=reason,
        )
        return ApprovalResult(
            approved=False,
            decision="deny",
            action=action,
            risk=risk,
            scope=scope,
            reason=reason,
            user_consent=False,
        )

    @staticmethod
    def _choices(allow_permanent: bool) -> tuple[str, ...]:
        base = ["allow_once", "allow_session"]
        if allow_permanent:
            base.append("allow_always")
        base.extend(["deny", "deny_always", "show_details"])
        return tuple(base)

    @staticmethod
    def _scope_from_decision(decision: Any) -> str:
        value = getattr(decision, "value", str(decision))
        if value in {"allow_session", "session"}:
            return ApprovalScope.SESSION.value
        if value in {"allow_always", "always"}:
            return ApprovalScope.PROFILE.value
        return ApprovalScope.ONCE.value

    @staticmethod
    def _title(risk: RiskAssessment) -> str:
        if risk.level == RiskLevel.CRITICAL:
            return "Critical Action Blocked"
        if risk.level == RiskLevel.HIGH:
            return "High Risk Action"
        return "Action Approval"

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Approval orchestration: policy, grants, prompting, and audit."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from leapflow.security.actions import ActionDescriptor
from leapflow.security.grants import (
    ApprovalAuditLog,
    ApprovalGrant,
    ApprovalGrantStore,
    ApprovalScope,
    InMemoryApprovalGrantStore,
    RevocableApprovalGrantStore,
    grant_key,
)
from leapflow.security.guardian import DenialBreaker, GuardianConfig, GuardianDecisionAdapter
from leapflow.security.policy import ApprovalPolicyEngine, PolicyVerdict
from leapflow.security.risk import DefaultRiskClassifier, RiskAssessment, RiskClassifier, RiskLevel

logger = logging.getLogger(__name__)


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
    """Coordinates risk assessment, grant lookup, human approval, and audit.

    Optional Guardian integration: when a ``GuardianDecisionAdapter`` is
    injected, the orchestrator consults the LLM for ASK-tier requests before
    falling through to the human prompt.  The Guardian can auto-approve
    (low LLM score), auto-deny (high LLM score), or defer to the human
    ("review").  A ``DenialBreaker`` prevents infinite loops.
    """

    def __init__(
        self,
        gate: Any,
        *,
        risk_classifier: RiskClassifier | None = None,
        policy: ApprovalPolicyEngine | None = None,
        grants: ApprovalGrantStore | None = None,
        audit: ApprovalAuditLog | None = None,
        guardian: GuardianDecisionAdapter | None = None,
        guardian_config: GuardianConfig | None = None,
    ) -> None:
        self._gate = gate
        self._risk = risk_classifier or DefaultRiskClassifier()
        self._policy = policy or ApprovalPolicyEngine()
        self._grants = grants or InMemoryApprovalGrantStore()
        self._audit = audit or ApprovalAuditLog()
        self._persistence_tasks: set[asyncio.Task[ApprovalResult]] = set()
        self._guardian = guardian
        self._guardian_config = guardian_config or (guardian.config if guardian else GuardianConfig())
        self._denial_breaker = DenialBreaker(
            max_consecutive_denials=self._guardian_config.max_consecutive_denials,
        )

    @property
    def audit(self) -> ApprovalAuditLog:
        return self._audit

    @property
    def grants(self) -> ApprovalGrantStore:
        return self._grants

    @property
    def denial_breaker(self) -> DenialBreaker:
        return self._denial_breaker

    def reset_turn(self) -> None:
        """Reset per-turn state (call between turns)."""
        self._denial_breaker.reset()

    async def evaluate(self, action: ActionDescriptor) -> ApprovalResult:
        """Return an approval result, prompting only when policy requires it."""
        from leapflow.security.approval import ApprovalDecision, ApprovalRequest

        risk = self._risk.assess(action)
        policy = self._policy.evaluate(action, risk)
        if policy.verdict == PolicyVerdict.ALLOW:
            self._denial_breaker.record_approval()
            return await asyncio.to_thread(self._approved, action, risk, actor="policy", reason=policy.reason)
        if policy.verdict == PolicyVerdict.DENY:
            self._denial_breaker.record_denial()
            return await asyncio.to_thread(self._denied, action, risk, actor="policy", reason=policy.reason)

        # DenialBreaker: fast-deny if too many consecutive denials
        if self._denial_breaker.is_tripped():
            return await asyncio.to_thread(self._denied,
                action, risk, actor="denial_breaker",
                reason="consecutive denial limit reached",
            )

        existing = await asyncio.to_thread(self._existing_grant, action)
        if existing is not None:
            if existing.decision.startswith("deny"):
                self._denial_breaker.record_denial()
                return await asyncio.to_thread(self._denied, action, risk, actor="grant", reason=existing.reason)
            self._denial_breaker.record_approval()
            return await asyncio.to_thread(self._approved, action, risk, actor="grant", scope=existing.scope, reason=existing.reason)

        # Model-service consent is an explicit human choice, not a Guardian exemption.
        model_consent = action.metadata.get("platform") == "model_services" and action.metadata.get("approval_binding")
        revision = (await asyncio.to_thread(self._grants.revision, model_consent)
                    if model_consent and isinstance(self._grants, RevocableApprovalGrantStore) else None)
        guardian_result = None if model_consent else await self._try_guardian(action, risk)
        if guardian_result is not None:
            return guardian_result

        request = ApprovalRequest(
            category=action.kind,
            detail=action.detail,
            risk_hint=risk.score,
            metadata={
                "action": action.to_dict(),
                "risk": risk.to_dict(),
                "allow_permanent": policy.allow_permanent,
                "grant_owner": "orchestrator",
            },
            action=action,
            risk=risk,
            choices=self._choices(policy.allow_permanent),
            default_choice="deny" if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL} else "allow_once",
            display={
                "title": self._title(risk),
                "summary": action.summary,
                "reason": risk.explanation,
            },
        )
        decision = await self._gate.request_approval(request)
        if decision == ApprovalDecision.ALLOW_ALWAYS and not policy.allow_permanent:
            decision = ApprovalDecision.DENY
        if decision in {
            ApprovalDecision.ALLOW,
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.ALLOW_SESSION,
            ApprovalDecision.ALLOW_ALWAYS,
        }:
            self._denial_breaker.record_approval()
            scope = self._scope_from_decision(decision)
            if scope in {ApprovalScope.SESSION.value, ApprovalScope.PROFILE.value}:
                return await self._remember(action, risk, decision.value, scope, revision)
            return await asyncio.to_thread(self._approved, action, risk, actor="user", scope=scope, reason=decision.value)

        if decision == ApprovalDecision.DENY_ALWAYS:
            self._denial_breaker.record_denial()
            return await self._remember(action, risk, decision.value, ApprovalScope.SESSION.value, revision)
        self._denial_breaker.record_denial()
        return await asyncio.to_thread(self._denied, action, risk, actor="user", reason=decision.value)

    async def _remember(
        self, action: ActionDescriptor, risk: RiskAssessment, decision: str,
        scope: str, revision: tuple[str, ...] | None,
    ) -> ApprovalResult:
        """Keep a consent commit and its audit alive together if the caller cancels."""
        worker = asyncio.create_task(asyncio.to_thread(self._persist_decision, action, risk, decision, scope, revision))
        self._persistence_tasks.add(worker)
        worker.add_done_callback(self._persistence_finished)
        return await asyncio.shield(worker)

    def _persistence_finished(self, worker: asyncio.Task[ApprovalResult]) -> None:
        self._persistence_tasks.discard(worker)
        if not worker.cancelled() and (error := worker.exception()) is not None:
            logger.error("Approval persistence failed", exc_info=(type(error), error, error.__traceback__))

    def _persist_decision(
        self, action: ActionDescriptor, risk: RiskAssessment, decision: str,
        scope: str, revision: tuple[str, ...] | None,
    ) -> ApprovalResult:
        allowed = decision.startswith("allow")
        grant = ApprovalGrant(
            key=grant_key(action, scope), scope=scope, decision="allow" if allowed else "deny",
            action_kind=action.kind, effect=action.effect, resource=action.resource,
            reason="user_approved" if allowed else "user_denied",
            binding=dict(action.metadata.get("approval_binding") or {}),
        )
        if revision is not None and isinstance(self._grants, RevocableApprovalGrantStore):
            if not self._grants.put_if_current(grant, revision):
                return self._denied(action, risk, actor="user", reason="consent_revoked_during_approval", scope=scope)
        else:
            self._grants.put(grant)
        finish = self._approved if allowed else self._denied
        return finish(action, risk, actor="user", scope=scope, reason=decision)

    async def check(self, command: str) -> bool:
        """Legacy shell approval adapter used by existing tool gates."""
        result = await self.evaluate(ActionDescriptor.shell(command))
        return result.approved

    def _existing_grant(self, action: ActionDescriptor) -> ApprovalGrant | None:
        for scope in (ApprovalScope.TURN, ApprovalScope.SESSION, ApprovalScope.PROFILE):
            existing = self._grants.get(grant_key(action, scope))
            binding = action.metadata.get("approval_binding")
            if existing is not None and (binding is None or existing.binding == binding):
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
        # Session bypass is explicit consent for every action that can reach an
        # approval prompt. ``allow_permanent`` governs only persistence beyond
        # the current session, never the meaning of “Allow ALL for this session”.
        base = ["allow_once", "allow_session", "allow_all_session"]
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

    async def _try_guardian(
        self,
        action: ActionDescriptor,
        risk: RiskAssessment,
    ) -> ApprovalResult | None:
        """Consult the Guardian LLM if configured and applicable.

        Returns an ``ApprovalResult`` when the Guardian auto-resolves the
        request (approve or deny).  Returns ``None`` when the request should
        proceed to the human approval prompt.
        """
        mode = self._guardian_config.mode
        if mode == "static_only" or self._guardian is None:
            return None

        try:
            verdict = await self._guardian.evaluate(
                tool_name=action.kind,
                detail=action.detail,
                risk_hint=risk.score,
                session_id=action.session_id,
                metadata={"action_id": action.action_id, "effect": action.effect},
            )
        except Exception as exc:
            logger.warning("guardian evaluation failed: %s", exc)
            if mode == "hybrid":
                return None  # degrade to human prompt
            # llm_assisted: failure is an error but we still degrade safely
            return None

        if verdict.recommendation == "approve":
            self._denial_breaker.record_approval()
            return await asyncio.to_thread(self._approved,
                action, risk, actor="guardian",
                reason=f"guardian auto-approve (score={verdict.risk_score:.2f})",
            )
        if verdict.recommendation == "deny":
            self._denial_breaker.record_denial()
            return await asyncio.to_thread(self._denied,
                action, risk, actor="guardian",
                reason=f"guardian auto-deny (score={verdict.risk_score:.2f})",
            )
        # "review" — fall through to human prompt
        return None

    @staticmethod
    def _title(risk: RiskAssessment) -> str:
        if risk.level == RiskLevel.CRITICAL:
            return "Critical Action Blocked"
        if risk.level == RiskLevel.HIGH:
            return "High Risk Action"
        return "Action Approval"

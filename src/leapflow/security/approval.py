"""Unified approval framework for actions requiring human confirmation.

This module is the compatibility-facing API for LeapFlow approvals.  It keeps
legacy tool gates working while carrying the richer Action Approval model used
by the policy/orchestrator layer.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from leapflow.security.actions import ActionDescriptor
from leapflow.security.risk import RiskAssessment

logger = logging.getLogger(__name__)


class ApprovalDecision(Enum):
    """Result of an approval request."""

    ALLOW = "allow"
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"
    DENY_ALWAYS = "deny_always"
    CANCEL_WORKFLOW = "cancel_workflow"


@dataclass(frozen=True)
class ApprovalRequest:
    """Structured request for human approval.

    ``category`` and ``detail`` preserve the historical API.  ``action`` and
    ``risk`` carry the generalized Action Approval payload used by new callers.
    """

    category: str
    detail: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    risk_hint: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    action: ActionDescriptor | None = None
    risk: RiskAssessment | None = None
    choices: tuple[str, ...] = ("allow_once", "allow_session", "deny")
    default_choice: str = "deny"
    expires_at: float | None = None
    display: dict[str, Any] = field(default_factory=dict)

    @property
    def grant_key(self) -> str:
        """Return a fine-grained session grant key for this request."""
        if self.action is not None:
            return f"{self.category}:{self.action.signature()}"
        return self.category

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "detail": self.detail,
            "request_id": self.request_id,
            "risk_hint": self.risk_hint,
            "metadata": dict(self.metadata),
            "action": self.action.to_dict() if self.action else None,
            "risk": self.risk.to_dict() if self.risk else None,
            "choices": list(self.choices),
            "default_choice": self.default_choice,
            "expires_at": self.expires_at,
            "display": dict(self.display),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequest":
        raw_action = data.get("action")
        raw_risk = data.get("risk")
        return cls(
            category=str(data.get("category") or "external_action"),
            detail=str(data.get("detail") or ""),
            request_id=str(data.get("request_id") or uuid.uuid4().hex),
            risk_hint=float(data.get("risk_hint") or 0.5),
            metadata=dict(data.get("metadata") or {}),
            action=ActionDescriptor.from_dict(raw_action) if isinstance(raw_action, dict) else None,
            risk=RiskAssessment.from_dict(raw_risk) if isinstance(raw_risk, dict) else None,
            choices=tuple(str(item) for item in data.get("choices") or ("allow_once", "allow_session", "deny")),
            default_choice=str(data.get("default_choice") or "deny"),
            expires_at=data.get("expires_at"),
            display=dict(data.get("display") or {}),
        )


@runtime_checkable
class ApprovalGate(Protocol):
    """Protocol for all human-in-the-loop approval decisions."""

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision: ...


class SessionAwareGate:
    """Wrap a base gate with fine-grained session memory and audit history."""

    def __init__(self, delegate: ApprovalGate) -> None:
        self._delegate = delegate
        self._approved_categories: set[str] = set()
        self._decision_log: list[dict[str, Any]] = []

    async def check(self, command: str) -> bool:
        """Legacy ``CommandApprovalGate`` compatibility for shell tools."""
        action = ActionDescriptor.shell(command)
        decision = await self.request_approval(ApprovalRequest(
            category=action.kind,
            detail=command,
            risk_hint=0.7,
            action=action,
        ))
        return decision in {
            ApprovalDecision.ALLOW,
            ApprovalDecision.ALLOW_ONCE,
            ApprovalDecision.ALLOW_SESSION,
            ApprovalDecision.ALLOW_ALWAYS,
        }

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision:
        grant_key = request.grant_key
        if grant_key in self._approved_categories:
            self._log_decision(request, ApprovalDecision.ALLOW, auto=True)
            return ApprovalDecision.ALLOW

        decision = await self._delegate.request_approval(request)
        if decision == ApprovalDecision.ALLOW_SESSION:
            self._approved_categories.add(grant_key)
            self._log_decision(request, ApprovalDecision.ALLOW_SESSION, session=True)
            return ApprovalDecision.ALLOW_SESSION
        if decision == ApprovalDecision.ALLOW_ALWAYS:
            self._approved_categories.add(grant_key)
            self._log_decision(request, ApprovalDecision.ALLOW_ALWAYS, session=True)
            return ApprovalDecision.ALLOW_ALWAYS

        self._log_decision(request, decision)
        return decision

    def reset(self) -> None:
        """Clear all session approvals."""
        self._approved_categories.clear()

    @property
    def approved_categories(self) -> frozenset[str]:
        return frozenset(self._approved_categories)

    @property
    def decision_log(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._decision_log)

    def _log_decision(
        self,
        request: ApprovalRequest,
        decision: ApprovalDecision,
        *,
        auto: bool = False,
        session: bool = False,
    ) -> None:
        entry = {
            "ts": time.time(),
            "request_id": request.request_id,
            "category": request.category,
            "grant_key": request.grant_key,
            "decision": decision.value,
            "detail": request.detail[:200],
            "risk": request.risk.to_dict() if request.risk else None,
        }
        if auto:
            entry["reason"] = "session_approved"
        elif session:
            entry["reason"] = "user_allow_session"
        self._decision_log.append(entry)

        level = logging.DEBUG if auto else logging.INFO
        logger.log(
            level,
            "approval.%s category=%s key=%s detail=%s%s",
            decision.value,
            request.category,
            request.grant_key,
            request.detail[:80],
            " (session-approved)" if auto else "",
        )


class DenyAllGate:
    """Gate that denies all requests in non-interactive or headless contexts."""

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision:
        logger.info(
            "approval.deny category=%s detail=%s (non-interactive)",
            request.category,
            request.detail[:80],
        )
        return ApprovalDecision.DENY

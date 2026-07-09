"""Unified approval framework for actions requiring human confirmation.

Replaces fragmented per-tool gates with a single ``ApprovalGate`` Protocol
and a session-aware wrapper that remembers per-category decisions.

Design principles:
- **Minimal interruption**: "always for this session" eliminates repeat prompts
- **Fail-closed**: non-interactive environments deny by default
- **Audit trail**: every decision is logged at INFO level
- **Protocol-based**: swappable implementations (TUI, Web UI, headless CI)

Categories
~~~~~~~~~~
``shell_dangerous``
    Shell commands matching dangerous regex patterns (sudo, rm -r, etc.)

``file_write``
    File writes to non-trivial paths (> 500 chars, non-safe extensions)

``gateway_send``
    Proactive outbound messaging to external platforms (per platform)

``external_action``
    Reserved for future extensibility (webhooks, API calls, etc.)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)


class ApprovalDecision(Enum):
    """Result of an approval request."""

    ALLOW = "allow"
    DENY = "deny"
    ALLOW_SESSION = "allow_session"


@dataclass(frozen=True)
class ApprovalRequest:
    """Structured request for human approval.

    ``category`` groups related actions; session memory uses this key.
    ``detail`` is the human-readable description shown in the prompt.
    ``risk_hint`` is advisory (0.0–1.0); the gate decides policy.
    """

    category: str
    detail: str
    risk_hint: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ApprovalGate(Protocol):
    """Protocol for all human-in-the-loop approval decisions.

    Implementations must handle the prompt display and input collection.
    Return ``ALLOW_SESSION`` to suppress future prompts for the same
    ``category`` during this session.
    """

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision: ...


class SessionAwareGate:
    """Wraps a base ``ApprovalGate`` with per-category session memory.

    Once a category is approved with ``ALLOW_SESSION``, all subsequent
    requests for that category are auto-approved without prompting.
    Individual ``ALLOW`` decisions are not remembered.

    Also implements ``check(command) -> bool`` for backward compatibility
    with ``CommandApprovalGate`` in ``shell_tools.py``.
    """

    def __init__(self, delegate: ApprovalGate) -> None:
        self._delegate = delegate
        self._approved_categories: Set[str] = set()
        self._decision_log: list[Dict[str, Any]] = []

    async def check(self, command: str) -> bool:
        """``CommandApprovalGate`` compatibility for shell_tools.py."""
        decision = await self.request_approval(ApprovalRequest(
            category="shell_dangerous",
            detail=command,
            risk_hint=0.7,
        ))
        return decision in (ApprovalDecision.ALLOW, ApprovalDecision.ALLOW_SESSION)

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision:
        if request.category in self._approved_categories:
            self._log_decision(request, ApprovalDecision.ALLOW, auto=True)
            return ApprovalDecision.ALLOW

        decision = await self._delegate.request_approval(request)

        if decision == ApprovalDecision.ALLOW_SESSION:
            self._approved_categories.add(request.category)
            self._log_decision(request, ApprovalDecision.ALLOW, session=True)
            return ApprovalDecision.ALLOW

        self._log_decision(request, decision)
        return decision

    def reset(self) -> None:
        """Clear all session approvals (e.g. on session restart)."""
        self._approved_categories.clear()

    @property
    def approved_categories(self) -> frozenset[str]:
        return frozenset(self._approved_categories)

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
            "category": request.category,
            "decision": decision.value,
            "detail": request.detail[:200],
        }
        if auto:
            entry["reason"] = "session_approved"
        elif session:
            entry["reason"] = "user_allow_session"
        self._decision_log.append(entry)

        level = logging.DEBUG if auto else logging.INFO
        logger.log(
            level,
            "approval.%s category=%s detail=%s%s",
            decision.value,
            request.category,
            request.detail[:80],
            " (session-approved)" if auto else "",
        )


class DenyAllGate:
    """Gate that denies all requests (non-interactive / CI environments)."""

    async def request_approval(
        self, request: ApprovalRequest,
    ) -> ApprovalDecision:
        logger.info(
            "approval.deny category=%s detail=%s (non-interactive)",
            request.category,
            request.detail[:80],
        )
        return ApprovalDecision.DENY

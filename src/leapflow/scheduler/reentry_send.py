"""S2 outbound SO1+SO4: outbound contracts, target resolution, and the pure
governance decision for autonomous re-entry sends.

This is the *decision kernel* for governed proactive delivery: it decides, for a
resolved target, whether an autonomous send may go out automatically (Progressive
Trust), must be queued for human approval, or is denied — under hard rate /
idempotency / global-budget guards. It performs no I/O and sends nothing; the
integration layer (SO3) consumes ``SendDecision`` to actually send or enqueue.

Default-off: the governor returns ``BLOCKED("disabled")`` unless explicitly
enabled, so behavior is unchanged until the feature is turned on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from leapflow.security.send_trust import SendTrustLedger


@dataclass(frozen=True)
class SendTarget:
    """Resolved outbound destination for a re-entry result."""

    platform: str
    chat: str

    def scope_key(self) -> str:
        """Rate-limit / identity key (platform + chat)."""
        return f"{self.platform}:{self.chat}"

    def grant_key(self, action: str = "reply") -> str:
        """Trust grant key (platform + chat + action)."""
        return f"{self.platform}:{self.chat}:{action}"


@dataclass(frozen=True)
class ReentrySendSpec:
    """A proposed autonomous outbound send from a completed re-entry."""

    target: Optional[SendTarget]
    text: str
    origin_trigger_id: str
    kind: str = "reply"

    def idempotency_key(self) -> str:
        """Stable key so a re-fired trigger cannot double-send the same content."""
        digest = hashlib.sha256(f"{self.origin_trigger_id}:{self.text}".encode("utf-8"))
        return digest.hexdigest()[:16]


def resolve_reentry_send_target(trigger: Any) -> Optional[SendTarget]:
    """Resolve the outbound target from a re-entry trigger (pure).

    First phase: only event-triggered re-entries carry an explicit
    ``event_match`` with platform + chat, which is the originating chat to reply
    to. Time-triggered re-entries have no resolvable target here and return
    ``None`` (no send), keeping the default conservative.
    """
    event_match = getattr(trigger, "event_match", None) or {}
    if not isinstance(event_match, dict):
        return None
    platform = str(event_match.get("platform") or "").strip()
    chat = str(event_match.get("chat") or "").strip()
    if platform and chat:
        return SendTarget(platform=platform, chat=chat)
    return None


class SendRateLimiter:
    """Per-scope sliding-window rate limiter (``per_hour`` sends max)."""

    def __init__(self, *, per_hour: int) -> None:
        self._per_hour = int(per_hour)
        self._events: Dict[str, List[float]] = {}

    def allow(self, scope_key: str, *, now: float) -> bool:
        if self._per_hour <= 0:
            return True  # unlimited
        window = [t for t in self._events.get(scope_key, []) if now - t < 3600.0]
        if len(window) >= self._per_hour:
            self._events[scope_key] = window
            return False
        window.append(now)
        self._events[scope_key] = window
        return True


class SendAction(Enum):
    """Governance verdict for a proposed autonomous send."""

    AUTO_ALLOW = "auto_allow"          # trust sufficient -> send now
    NEEDS_APPROVAL = "needs_approval"  # queue for asynchronous human approval
    DENY = "deny"                      # no approver and no trust -> do not send
    BLOCKED = "blocked"                # guard tripped (disabled/target/rate/budget/dup)


@dataclass(frozen=True)
class SendDecision:
    action: SendAction
    reason: str


class SendGovernor:
    """Pure decision flow for governed autonomous re-entry sends (SO1+SO4).

    Combines the feature gate, target resolution, idempotency, global budget,
    rate limit, and the Progressive Trust ledger into a single verdict. Holds no
    transport; ``record_sent`` is called by the integration layer after a send
    actually commits so duplicates and the global budget are tracked exactly.
    """

    def __init__(
        self,
        *,
        trust: SendTrustLedger,
        rate: SendRateLimiter,
        enabled: bool,
        global_budget: int = 50,
    ) -> None:
        self._trust = trust
        self._rate = rate
        self._enabled = bool(enabled)
        self._global_budget = int(global_budget)
        self._sent_total = 0
        self._seen_keys: set[str] = set()

    def decide(
        self,
        spec: ReentrySendSpec,
        *,
        destructive: bool,
        has_approver: bool,
        now: float,
    ) -> SendDecision:
        if not self._enabled:
            return SendDecision(SendAction.BLOCKED, "disabled")
        if spec.target is None:
            return SendDecision(SendAction.BLOCKED, "no_target")
        if spec.idempotency_key() in self._seen_keys:
            return SendDecision(SendAction.BLOCKED, "duplicate")
        if self._global_budget > 0 and self._sent_total >= self._global_budget:
            return SendDecision(SendAction.BLOCKED, "global_budget_exhausted")
        if not self._rate.allow(spec.target.scope_key(), now=now):
            return SendDecision(SendAction.BLOCKED, "rate_limited")
        if self._trust.auto_approve_ok(spec.target.grant_key(spec.kind), destructive=destructive):
            return SendDecision(SendAction.AUTO_ALLOW, "trust_verified")
        if has_approver:
            return SendDecision(SendAction.NEEDS_APPROVAL, "queue_for_human")
        return SendDecision(SendAction.DENY, "no_approver_no_trust")

    def record_sent(self, spec: ReentrySendSpec) -> None:
        """Account for a committed send (idempotency + global budget)."""
        self._sent_total += 1
        self._seen_keys.add(spec.idempotency_key())

    def record_human_allow(self, grant_key: str) -> None:
        """A human approved a send in this scope: accrue Progressive Trust."""
        self._trust.record_allow(grant_key)

    def record_human_deny(self, grant_key: str) -> None:
        """A human denied a send in this scope: freeze trust back to DRAFT."""
        self._trust.record_deny(grant_key)

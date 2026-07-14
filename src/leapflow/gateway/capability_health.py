"""Capability health tracking for platform actions.

Maintains a session-scoped ledger that records authorization failures for
platform capabilities and exposes a feasibility check so platform_action_handler
can block hopeless retries before they reach the approval gate.

Design principles:
- Platform-neutral: no Feishu/CLI-specific logic here.
- Capability granularity: tracks by (platform, capability) so a read failure
  does not block send, and vice versa.
- Recoverability classes:
    retryable:          transient — may succeed on retry (rate-limit, timeout)
    user_action:        user must do something lightweight (re-auth, retry later)
    admin_required:     an external administrator must change authorization
    non_recoverable:    hard failure (e.g. feature disabled, unsupported)
- Ledger is session-scoped; cleared between agent turns via clear() if needed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from leapflow.gateway.connectors.protocol import ActionFailure, ActionSpec

logger = logging.getLogger(__name__)

# Recoverability ranks — higher rank means further from automatic recovery.
_RECOVERABILITY_RANK: Dict[str, int] = {
    "retryable": 0,
    "user_action": 1,
    "admin_required": 2,
    "non_recoverable": 3,
}

# Failure classes that should block approval because no amount of user consent
# can fix the underlying platform authorization gap.
_BLOCKS_APPROVAL_CLASSES = frozenset({"authorization", "scope_denied"})

# Failure classes that are transient and should NOT block approval.
_TRANSIENT_CLASSES = frozenset({"timeout", "rate_limit", "transient"})


@dataclass
class CapabilityStatus:
    """Recorded health status for a (platform, capability) pair."""

    platform: str
    capability: str
    failure: ActionFailure
    recorded_at: float = field(default_factory=time.monotonic)
    # TTL in seconds; 0 = session-scoped (never expires within a session).
    ttl_s: float = 0.0

    @property
    def is_expired(self) -> bool:
        if self.ttl_s <= 0:
            return False
        return (time.monotonic() - self.recorded_at) > self.ttl_s

    @property
    def should_block_approval(self) -> bool:
        """Return True when this failure class cannot be resolved by user consent."""
        f = self.failure
        if f.failure_class in _TRANSIENT_CLASSES:
            return False
        return f.blocks_approval or f.failure_class in _BLOCKS_APPROVAL_CLASSES


class CapabilityHealthLedger:
    """Session-scoped ledger tracking per-capability authorization health.

    Thread-safety: single-threaded async usage expected; no locking.
    """

    def __init__(self) -> None:
        # Key: (platform, capability)
        self._records: Dict[Tuple[str, str], CapabilityStatus] = {}

    def record_failure(
        self,
        platform: str,
        capability: str,
        failure: ActionFailure,
        *,
        ttl_s: float = 0.0,
    ) -> None:
        """Record a capability failure.

        If a worse (higher-rank) failure already exists for this capability,
        the existing entry is replaced only when the new failure has equal or
        higher rank.
        """
        key = (platform, capability)
        existing = self._records.get(key)
        new_rank = _RECOVERABILITY_RANK.get(failure.recoverability, 0)
        if existing is not None and not existing.is_expired:
            old_rank = _RECOVERABILITY_RANK.get(existing.failure.recoverability, 0)
            if old_rank >= new_rank:
                return  # Keep the existing worse record.
        self._records[key] = CapabilityStatus(
            platform=platform,
            capability=capability,
            failure=failure,
            ttl_s=ttl_s,
        )
        logger.debug(
            "capability_ledger.recorded platform=%s capability=%s class=%s code=%s recoverability=%s",
            platform, capability, failure.failure_class, failure.failure_code, failure.recoverability,
        )

    def get(self, platform: str, capability: str) -> CapabilityStatus | None:
        """Return the recorded status for a capability, or None if healthy/expired."""
        key = (platform, capability)
        status = self._records.get(key)
        if status is None or status.is_expired:
            if status is not None:
                del self._records[key]
            return None
        return status

    def check_feasibility(
        self,
        platform: str,
        spec: ActionSpec,
    ) -> dict[str, Any]:
        """Return feasibility dict for a platform action.

        Returns ``{"ok": True}`` when the action may proceed to approval.
        Returns a structured failure dict when the ledger has a blocking record
        so platform_action_handler can short-circuit without touching approval.
        """
        capability = spec.capability or spec.name
        status = self.get(platform, capability)
        if status is None or not status.should_block_approval:
            return {"ok": True}
        failure = status.failure
        response: dict[str, Any] = {
            "ok": False,
            "failure_code": failure.failure_code,
            "failure_class": failure.failure_class,
            "error": failure.message or failure.recovery_hint or "Platform capability is not authorized.",
            "recoverability": failure.recoverability,
            "retryable": failure.recoverability not in ("admin_required", "non_recoverable"),
            "capability": capability,
            "platform": platform,
            "action": spec.name,
            "skip_approval": True,  # Signal to caller: no approval needed or useful.
        }
        failure_dict = failure.as_dict()
        for key in ("recovery_hint", "next_steps", "missing_scopes", "required_scopes",
                    "requested_scopes", "granted_scopes", "identity", "console_url"):
            if failure_dict.get(key):
                response[key] = failure_dict[key]
        return response

    def clear(self, platform: str | None = None) -> None:
        """Clear ledger entries for a platform, or all entries if platform is None."""
        if platform is None:
            self._records.clear()
        else:
            keys = [k for k in self._records if k[0] == platform]
            for key in keys:
                del self._records[key]

    def summary(self) -> List[dict[str, Any]]:
        """Return a compact summary of all non-expired records."""
        result = []
        for (platform, capability), status in list(self._records.items()):
            if status.is_expired:
                continue
            result.append({
                "platform": platform,
                "capability": capability,
                "failure_class": status.failure.failure_class,
                "failure_code": status.failure.failure_code,
                "recoverability": status.failure.recoverability,
                "blocks_approval": status.should_block_approval,
            })
        return result

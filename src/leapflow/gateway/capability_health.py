"""Capability health tracking for platform actions.

Maintains a session-scoped ledger that records authorization failures for
platform capabilities and exposes a feasibility check so platform_action_handler
can block hopeless retries before they reach the approval gate.

Design principles:
- Platform-neutral: no Feishu/CLI-specific logic here.
- Capability granularity: tracks by (platform, capability) so a read failure
  does not block send, and vice versa.
- Platform degradation: when an authorization failure is recorded, the entire
  platform is marked as degraded for side-effect actions. This prevents
  hallucinated resource IDs from reaching the approval gate when fundamental
  permissions are missing.
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
from typing import Any, Dict, List, Set, Tuple

from leapflow.gateway.connectors.protocol import ActionFailure, ActionSpec

logger = logging.getLogger(__name__)

_RECOVERABILITY_RANK: Dict[str, int] = {
    "retryable": 0,
    "user_action": 1,
    "admin_required": 2,
    "non_recoverable": 3,
}

_BLOCKS_APPROVAL_CLASSES = frozenset({"authorization", "scope_denied"})
_TRANSIENT_CLASSES = frozenset({"timeout", "rate_limit", "transient"})

_SIDE_EFFECT_KINDS = frozenset({"send", "write", "execute"})


@dataclass
class CapabilityStatus:
    """Recorded health status for a (platform, capability) pair."""

    platform: str
    capability: str
    failure: ActionFailure
    recorded_at: float = field(default_factory=time.monotonic)
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
        self._records: Dict[Tuple[str, str], CapabilityStatus] = {}
        # Platform-level degradation: platforms with hard auth failures that
        # should block side-effect actions even for capabilities not yet tested.
        self._degraded_platforms: Dict[str, str] = {}  # platform → reason

    @property
    def degraded_platforms(self) -> Dict[str, str]:
        """Return a copy of the degradation map for diagnostics."""
        return dict(self._degraded_platforms)

    def record_failure(
        self,
        platform: str,
        capability: str,
        failure: ActionFailure,
        *,
        ttl_s: float = 0.0,
    ) -> None:
        """Record a capability failure.

        When the failure is a hard authorization error (admin_required or
        non_recoverable), also marks the platform as degraded — blocking
        subsequent side-effect actions across ALL capabilities.
        """
        key = (platform, capability)
        existing = self._records.get(key)
        new_rank = _RECOVERABILITY_RANK.get(failure.recoverability, 0)
        if existing is not None and not existing.is_expired:
            old_rank = _RECOVERABILITY_RANK.get(existing.failure.recoverability, 0)
            if old_rank >= new_rank:
                return
        self._records[key] = CapabilityStatus(
            platform=platform,
            capability=capability,
            failure=failure,
            ttl_s=ttl_s,
        )
        if (
            failure.failure_class in _BLOCKS_APPROVAL_CLASSES
            and failure.recoverability in ("admin_required", "non_recoverable")
        ):
            reason = (
                f"Authorization failure on {capability}: "
                f"{failure.message or failure.failure_code}"
            )
            self._degraded_platforms[platform] = reason
            logger.info(
                "platform_degraded platform=%s capability=%s code=%s",
                platform, capability, failure.failure_code,
            )
        logger.debug(
            "capability_ledger.recorded platform=%s capability=%s class=%s code=%s recoverability=%s",
            platform, capability, failure.failure_class, failure.failure_code, failure.recoverability,
        )

    def is_platform_degraded(self, platform: str) -> bool:
        """Return True if the platform has an unresolved authorization failure."""
        return platform in self._degraded_platforms

    def platform_degradation_reason(self, platform: str) -> str:
        """Return the degradation reason, or empty string if not degraded."""
        return self._degraded_platforms.get(platform, "")

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

        Checks (in order):
        1. Direct capability failure (same as before)
        2. Platform-level degradation for side-effect actions
        """
        capability = spec.capability or spec.name

        # Direct capability failure check.
        status = self.get(platform, capability)
        if status is not None and status.should_block_approval:
            return self._failure_response(platform, spec, status.failure)

        # Platform degradation blocks side-effect actions even when the
        # specific capability has not been tested yet.
        if spec.effect in _SIDE_EFFECT_KINDS and self.is_platform_degraded(platform):
            reason = self._degraded_platforms[platform]
            return {
                "ok": False,
                "failure_code": "platform_degraded",
                "failure_class": "authorization",
                "error": (
                    f"Platform '{platform}' has authorization failures that block "
                    f"side-effect actions. {reason}"
                ),
                "recoverability": "admin_required",
                "retryable": False,
                "capability": capability,
                "platform": platform,
                "action": spec.name,
                "skip_approval": True,
                "platform_degraded": True,
                "degradation_reason": reason,
                "llm_instruction": (
                    f"STOP: Platform '{platform}' is missing required permissions. "
                    "Do NOT fabricate or guess resource IDs. Report this failure to the user "
                    "and recommend they fix the authorization in the developer console."
                ),
            }

        return {"ok": True}

    def clear(self, platform: str | None = None) -> None:
        """Clear ledger entries (and degradation) for a platform or all."""
        if platform is None:
            self._records.clear()
            self._degraded_platforms.clear()
        else:
            keys = [k for k in self._records if k[0] == platform]
            for key in keys:
                del self._records[key]
            self._degraded_platforms.pop(platform, None)

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
                "platform_degraded": self.is_platform_degraded(platform),
            })
        return result

    def _failure_response(
        self,
        platform: str,
        spec: ActionSpec,
        failure: ActionFailure,
    ) -> dict[str, Any]:
        """Build a structured failure dict for a blocked capability."""
        capability = spec.capability or spec.name
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
            "skip_approval": True,
            "llm_instruction": (
                f"STOP: The capability '{capability}' is not authorized on platform '{platform}'. "
                "Do NOT fabricate resource IDs or attempt dependent actions. "
                "Report this failure to the user."
            ),
        }
        failure_dict = failure.as_dict()
        for key in ("recovery_hint", "next_steps", "missing_scopes", "required_scopes",
                    "requested_scopes", "granted_scopes", "identity", "console_url"):
            if failure_dict.get(key):
                response[key] = failure_dict[key]
        return response

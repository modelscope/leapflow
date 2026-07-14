"""Capability health tracking for platform actions.

Maintains a session-scoped ledger that records authorization failures for
platform capabilities and exposes a feasibility check so platform_action_handler
can block unsafe side-effect retries before they reach the approval gate.

Design principles:
- Platform-neutral: no Feishu/CLI-specific logic here.
- Capability granularity: tracks by (platform, capability) so successful
  revalidation of one capability clears only that capability's stale failure.
- Dynamic recovery: read actions are allowed to revalidate stale authorization
  failures, and successful actions actively clear the matching failure record.
- Platform degradation: unresolved authorization failures block side-effect
  actions across the platform until the relevant capability is revalidated or
  the record expires. This prevents hallucinated resource IDs from reaching the
  approval gate when fundamental permissions are still missing.
- Recoverability classes:
    retryable:          transient — may succeed on retry (rate-limit, timeout)
    user_action:        user must do something lightweight (re-auth, retry later)
    admin_required:     an external administrator must change authorization
    non_recoverable:    hard failure (e.g. feature disabled, unsupported)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from leapflow.gateway.connectors.protocol import ActionFailure, ActionSpec

logger = logging.getLogger(__name__)

DEFAULT_AUTHORIZATION_FAILURE_TTL_S = 300.0

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

    def __init__(self, *, authorization_failure_ttl_s: float = DEFAULT_AUTHORIZATION_FAILURE_TTL_S) -> None:
        self._records: Dict[Tuple[str, str], CapabilityStatus] = {}
        self._authorization_failure_ttl_s = max(0.0, authorization_failure_ttl_s)
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
        effective_ttl_s = ttl_s
        if effective_ttl_s <= 0 and self._is_degrading_failure(failure):
            effective_ttl_s = self._authorization_failure_ttl_s
        self._records[key] = CapabilityStatus(
            platform=platform,
            capability=capability,
            failure=failure,
            ttl_s=effective_ttl_s,
        )
        self._refresh_platform_degradation(platform)
        logger.debug(
            "capability_ledger.recorded platform=%s capability=%s class=%s code=%s recoverability=%s",
            platform, capability, failure.failure_class, failure.failure_code, failure.recoverability,
        )

    def record_success(self, platform: str, capability: str) -> bool:
        """Clear stale health records after a capability succeeds.

        Returns True when a previous failure was removed. Platform degradation is
        recomputed from the remaining unresolved failures.
        """
        removed = self.clear_capability(platform, capability)
        if removed:
            logger.info(
                "capability_ledger.revalidated platform=%s capability=%s",
                platform, capability,
            )
        return removed

    def clear_capability(self, platform: str, capability: str) -> bool:
        """Clear one capability failure and refresh platform degradation."""
        key = (platform, capability)
        removed = self._records.pop(key, None) is not None
        if removed:
            self._refresh_platform_degradation(platform)
        return removed

    def is_platform_degraded(self, platform: str) -> bool:
        """Return True if the platform has an unresolved authorization failure."""
        self._purge_expired(platform)
        return platform in self._degraded_platforms

    def platform_degradation_reason(self, platform: str) -> str:
        """Return the degradation reason, or empty string if not degraded."""
        self._purge_expired(platform)
        return self._degraded_platforms.get(platform, "")

    def get(self, platform: str, capability: str) -> CapabilityStatus | None:
        """Return the recorded status for a capability, or None if healthy/expired."""
        key = (platform, capability)
        status = self._records.get(key)
        if status is None or status.is_expired:
            if status is not None:
                del self._records[key]
                self._refresh_platform_degradation(platform)
            return None
        return status

    def check_feasibility(
        self,
        platform: str,
        spec: ActionSpec,
    ) -> dict[str, Any]:
        """Return feasibility dict for a platform action.

        Checks (in order):
        1. Direct capability failure for side-effect actions
        2. Platform-level degradation for side-effect actions

        Read actions intentionally pass through after a prior failure so they can
        revalidate external authorization changes, such as a freshly granted
        Feishu scope or refreshed CLI token.
        """
        capability = spec.capability or spec.name

        status = self.get(platform, capability)
        if status is not None and status.should_block_approval:
            if spec.effect not in _SIDE_EFFECT_KINDS:
                return {
                    "ok": True,
                    "permission_revalidation": True,
                    "previous_failure_code": status.failure.failure_code,
                    "previous_failure_class": status.failure.failure_class,
                    "capability": capability,
                    "platform": platform,
                    "action": spec.name,
                }
            return self._failure_response(platform, spec, status.failure)

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
        self._purge_expired()
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
                    "requested_scopes", "granted_scopes", "identity", "console_url",
                    "scope_relation", "scope_source"):
            if failure_dict.get(key):
                response[key] = failure_dict[key]
        return response

    @staticmethod
    def _is_degrading_failure(failure: ActionFailure) -> bool:
        return (
            failure.failure_class in _BLOCKS_APPROVAL_CLASSES
            and failure.recoverability in ("admin_required", "non_recoverable")
        )

    def _purge_expired(self, platform: str | None = None) -> None:
        expired_platforms: set[str] = set()
        for key, status in list(self._records.items()):
            if platform is not None and key[0] != platform:
                continue
            if status.is_expired:
                del self._records[key]
                expired_platforms.add(key[0])
        for platform_id in expired_platforms:
            self._refresh_platform_degradation(platform_id)

    def _refresh_platform_degradation(self, platform: str) -> None:
        for (record_platform, capability), status in list(self._records.items()):
            if record_platform != platform:
                continue
            if status.is_expired:
                del self._records[(record_platform, capability)]
                continue
            if self._is_degrading_failure(status.failure):
                reason = (
                    f"Authorization failure on {capability}: "
                    f"{status.failure.message or status.failure.failure_code}"
                )
                self._degraded_platforms[platform] = reason
                logger.info(
                    "platform_degraded platform=%s capability=%s code=%s",
                    platform, capability, status.failure.failure_code,
                )
                return
        self._degraded_platforms.pop(platform, None)

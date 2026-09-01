"""Per-(device, channel) trust gate for hardware write approval.

Progressive trust for the physical domain: a channel that consistently produces
correct outcomes earns the right to skip per-command approval, but only when
the channel is declared reversible.  Irreversible channels (dispense, emit)
*always* require approval regardless of trust level — the cost of being wrong
once is unrecoverable.

The trust gradient is:

    UNTRUSTED  → every write requires approval (default)
    CANDIDATE  → has shown correct outcomes, still requires approval
    VERIFIED   → reversible channels may skip per-command approval
    PRODUCTION → same as VERIFIED for now; reserved for future autonomy

Trust accrues from ``record_success`` (write + correct outcome observation)
and decays from ``record_failure`` (write + outcome deviation or transport
failure).  A hard failure (e.g. internal defect) freezes the channel to
UNTRUSTED permanently.

Integration:
    - Wraps or decorates the existing ``ApprovalOrchestrator`` — it does not
      replace it.  When trust is high enough and the channel is reversible,
      the gate returns an auto-approved result instead of prompting.
    - Plugs into ``PluginTrustLedger`` for the plugin-level trust ledger,
      mapping ``(device_id, channel_id)`` to a synthetic plugin-id key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

_CANDIDATE_AT = 3
"""Consecutive successes to reach CANDIDATE."""

_VERIFIED_AT = 8
"""Consecutive successes to reach VERIFIED (approval exemption for reversible)."""

_PRODUCTION_AT = 20
"""Consecutive successes to reach PRODUCTION."""

_DEMOTE_AFTER = 2
"""Consecutive failures to demote one level."""


class HardwareTrustLevel(IntEnum):
    """Trust gradient for a (device, channel) pair."""

    UNTRUSTED = 0
    CANDIDATE = 1
    VERIFIED = 2
    PRODUCTION = 3


@dataclass(frozen=True)
class TrustRecord:
    """Snapshot of trust state for one (device, channel) pair."""

    device_id: str
    channel_id: str
    level: HardwareTrustLevel
    consecutive_ok: int
    consecutive_fail: int
    frozen: bool


class HardwareTrustGate:
    """Per-(device, channel) trust ledger and approval short-circuit.

    When trust is VERIFIED or above *and* the channel is declared reversible,
    the gate reports ``may_skip_approval=True`` so the caller can bypass the
    human prompt.  All other cases require normal approval flow.

    ``allow_permanent`` is True only for reversible channels at VERIFIED+,
    matching the platform rule that ``allow_permanent=True`` is reserved for
    actions whose effect can be undone.
    """

    def __init__(
        self,
        *,
        candidate_at: int = _CANDIDATE_AT,
        verified_at: int = _VERIFIED_AT,
        production_at: int = _PRODUCTION_AT,
        demote_after: int = _DEMOTE_AFTER,
        plugin_trust_ledger: Any = None,
    ) -> None:
        self._candidate_at = max(1, candidate_at)
        self._verified_at = max(1, verified_at)
        self._production_at = max(1, production_at)
        self._demote_after = max(1, demote_after)
        self._plugin_trust = plugin_trust_ledger
        self._consecutive_ok: Dict[Tuple[str, str], int] = {}
        self._consecutive_fail: Dict[Tuple[str, str], int] = {}
        self._levels: Dict[Tuple[str, str], HardwareTrustLevel] = {}
        self._frozen: set[Tuple[str, str]] = set()

    # ── Query ──

    def level(self, device_id: str, channel_id: str) -> HardwareTrustLevel:
        """Current trust level for one channel."""
        key = (device_id, channel_id)
        if key in self._frozen:
            return HardwareTrustLevel.UNTRUSTED
        return self._levels.get(key, HardwareTrustLevel.UNTRUSTED)

    def may_skip_approval(
        self,
        device_id: str,
        channel_id: str,
        *,
        reversible: bool,
    ) -> bool:
        """Return whether this channel has earned approval exemption.

        Only reversible channels at VERIFIED or above qualify.  Irreversible
        channels *always* require approval — the cost of being wrong once
        cannot be recovered.
        """
        if not reversible:
            return False
        return self.level(device_id, channel_id) >= HardwareTrustLevel.VERIFIED

    def allow_permanent(
        self,
        device_id: str,
        channel_id: str,
        *,
        reversible: bool,
    ) -> bool:
        """Whether the orchestrator may offer a persistent (profile-level) grant.

        True only for reversible channels at VERIFIED+, matching the platform
        rule that ``allow_permanent=True`` is reserved for undoable effects.
        """
        return self.may_skip_approval(
            device_id, channel_id, reversible=reversible
        )

    # ── Mutation ──

    def record_success(self, device_id: str, channel_id: str) -> None:
        """A write was executed and the outcome matched the command."""
        key = (device_id, channel_id)
        if key in self._frozen:
            return
        self._consecutive_ok[key] = self._consecutive_ok.get(key, 0) + 1
        self._consecutive_fail[key] = 0
        self._maybe_promote(key)
        self._sync_plugin_trust(key, success=True)

    def record_failure(
        self,
        device_id: str,
        channel_id: str,
        *,
        hard: bool = False,
    ) -> None:
        """A write failed or the outcome deviated from the command.

        ``hard=True`` freezes the channel to UNTRUSTED permanently (until
        manual reset).
        """
        key = (device_id, channel_id)
        if hard:
            self._frozen.add(key)
            self._levels[key] = HardwareTrustLevel.UNTRUSTED
            self._consecutive_ok[key] = 0
            self._consecutive_fail[key] = 0
            self._sync_plugin_trust(key, success=False, hard=True)
            return
        if key in self._frozen:
            return
        self._consecutive_fail[key] = self._consecutive_fail.get(key, 0) + 1
        self._consecutive_ok[key] = 0
        if self._consecutive_fail[key] >= self._demote_after:
            self._demote(key)
        self._sync_plugin_trust(key, success=False)

    # ── Introspection ──

    def trust_record(
        self, device_id: str, channel_id: str,
    ) -> TrustRecord:
        """Return a snapshot of the trust state for one channel."""
        key = (device_id, channel_id)
        return TrustRecord(
            device_id=device_id,
            channel_id=channel_id,
            level=self.level(device_id, channel_id),
            consecutive_ok=self._consecutive_ok.get(key, 0),
            consecutive_fail=self._consecutive_fail.get(key, 0),
            frozen=key in self._frozen,
        )

    def all_records(self) -> list[TrustRecord]:
        """Return trust state for every tracked channel."""
        keys = set(self._levels.keys()) | set(self._frozen)
        return [
            self.trust_record(d, c) for d, c in sorted(keys)
        ]

    # ── Internal ──

    def _maybe_promote(self, key: Tuple[str, str]) -> None:
        streak = self._consecutive_ok.get(key, 0)
        current = self._levels.get(key, HardwareTrustLevel.UNTRUSTED)
        if current < HardwareTrustLevel.PRODUCTION and streak >= self._production_at:
            self._levels[key] = HardwareTrustLevel.PRODUCTION
        elif current < HardwareTrustLevel.VERIFIED and streak >= self._verified_at:
            self._levels[key] = HardwareTrustLevel.VERIFIED
        elif current < HardwareTrustLevel.CANDIDATE and streak >= self._candidate_at:
            self._levels[key] = HardwareTrustLevel.CANDIDATE

    def _demote(self, key: Tuple[str, str]) -> None:
        current = self._levels.get(key, HardwareTrustLevel.UNTRUSTED)
        if current > HardwareTrustLevel.UNTRUSTED:
            self._levels[key] = HardwareTrustLevel(current - 1)
        self._consecutive_fail[key] = 0

    def _sync_plugin_trust(
        self,
        key: Tuple[str, str],
        *,
        success: bool,
        hard: bool = False,
    ) -> None:
        """Forward trust events to the plugin-level trust ledger if available."""
        if self._plugin_trust is None:
            return
        plugin_id = f"hw:{key[0]}:{key[1]}"
        try:
            if success:
                self._plugin_trust.record_success(plugin_id)
            else:
                self._plugin_trust.record_failure(plugin_id, hard=hard)
        except Exception:
            logger.debug(
                "Failed to sync trust to plugin ledger for %s", plugin_id,
                exc_info=True,
            )

    @staticmethod
    def _key_for(device_id: str, channel_id: str) -> Tuple[str, str]:
        return (device_id, channel_id)


__all__ = [
    "HardwareTrustGate",
    "HardwareTrustLevel",
    "TrustRecord",
]

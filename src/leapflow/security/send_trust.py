"""S2 outbound SO2: send-scope Progressive Trust ledger.

Autonomous re-entry has no synchronous human approver, so an outbound send is
only auto-approved once a specific ``(platform, chat, action)`` scope has earned
trust through *repeated human approvals* (mirroring the skill trust gradient
DRAFT -> CANDIDATE -> VERIFIED -> PRODUCTION). A single human DENY freezes the
scope back to DRAFT (conservative). Destructive targets (cross-chat, broadcast,
first-time) are never auto-approved regardless of trust.

Pure and hermetic; ``to_state``/``load_state`` allow later durable persistence
without changing the decision logic.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict


class SendTrustLevel(IntEnum):
    """Trust gradient for an outbound send scope (higher = more autonomy)."""

    DRAFT = 0
    CANDIDATE = 1
    VERIFIED = 2
    PRODUCTION = 3


_PRODUCTION_AT = 8


class SendTrustLedger:
    """Per-scope trust earned by human approvals of outbound sends.

    Trust rises only via ``record_allow`` (a human approved a send in that
    scope) and is frozen to DRAFT by ``record_deny``. ``auto_approve_ok`` gates
    autonomous sends: only VERIFIED+ and non-destructive targets may bypass
    human approval.
    """

    def __init__(self, *, verified_at: int = 3) -> None:
        self._verified_at = max(1, int(verified_at))
        self._allows: Dict[str, int] = {}
        self._frozen: set[str] = set()

    def level(self, grant_key: str) -> SendTrustLevel:
        if grant_key in self._frozen:
            return SendTrustLevel.DRAFT
        count = self._allows.get(grant_key, 0)
        if count >= _PRODUCTION_AT:
            return SendTrustLevel.PRODUCTION
        if count >= self._verified_at:
            return SendTrustLevel.VERIFIED
        if count >= 1:
            return SendTrustLevel.CANDIDATE
        return SendTrustLevel.DRAFT

    def record_allow(self, grant_key: str) -> None:
        """A human approved a send in this scope: unfreeze and accrue trust."""
        self._frozen.discard(grant_key)
        self._allows[grant_key] = self._allows.get(grant_key, 0) + 1

    def record_deny(self, grant_key: str) -> None:
        """A human denied a send in this scope: freeze it back to DRAFT."""
        self._frozen.add(grant_key)

    def auto_approve_ok(self, grant_key: str, *, destructive: bool) -> bool:
        """Whether an autonomous send may bypass human approval.

        Never for destructive targets (cross-chat / broadcast / first-time);
        otherwise only when the scope has reached VERIFIED trust.
        """
        if destructive:
            return False
        return self.level(grant_key) >= SendTrustLevel.VERIFIED

    # ── Durable state (for later persistence; logic-neutral) ──

    def to_state(self) -> Dict[str, Any]:
        return {
            "verified_at": self._verified_at,
            "allows": dict(self._allows),
            "frozen": sorted(self._frozen),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._verified_at = max(1, int(state.get("verified_at", self._verified_at)))
        self._allows = {str(k): int(v) for k, v in (state.get("allows") or {}).items()}
        self._frozen = {str(k) for k in (state.get("frozen") or [])}

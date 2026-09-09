"""Progressive trust ledger for plugins.

Trust is earned through consistent successful execution (not human approval).
Unlike SendTrustLedger (earned by human actions), PluginTrustLedger promotes
plugins that demonstrate reliability over time.

Levels:
    DRAFT      — New/unknown plugin, no track record
    CANDIDATE  — Consecutive successes >= candidate_at (default 5)
    VERIFIED   — Consecutive successes >= verified_at (default 20)
    PRODUCTION — Consecutive successes >= production_at (default 50)

Demotion: consecutive failures >= demote_after → downgrade one level.
A single hard failure (internal_defect) → freeze to DRAFT.

Pure and hermetic; ``to_state``/``load_state`` allow later durable persistence
without changing the decision logic.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict


class PluginTrustLevel(IntEnum):
    """Trust gradient for a plugin (higher = more proven reliability)."""

    DRAFT = 0
    CANDIDATE = 1
    VERIFIED = 2
    PRODUCTION = 3


class PluginTrustLedger:
    """Per-plugin trust earned by consecutive successful executions.

    Trust rises via ``record_success`` and falls via ``record_failure``.
    A hard failure freezes the plugin to DRAFT permanently (until manual reset).
    """

    def __init__(
        self,
        *,
        candidate_at: int = 5,
        verified_at: int = 20,
        production_at: int = 50,
        demote_after: int = 3,
    ) -> None:
        self._candidate_at = max(1, int(candidate_at))
        self._verified_at = max(1, int(verified_at))
        self._production_at = max(1, int(production_at))
        self._demote_after = max(1, int(demote_after))
        self._consecutive_ok: Dict[str, int] = {}
        self._consecutive_fail: Dict[str, int] = {}
        self._levels: Dict[str, PluginTrustLevel] = {}
        self._frozen: set[str] = set()

    def level(self, plugin_id: str) -> PluginTrustLevel:
        """Current trust level for the given plugin."""
        if plugin_id in self._frozen:
            return PluginTrustLevel.DRAFT
        return self._levels.get(plugin_id, PluginTrustLevel.DRAFT)

    def is_frozen(self, plugin_id: str) -> bool:
        """Whether the plugin is permanently frozen by an internal defect.

        A frozen plugin reports ``DRAFT``, but ``DRAFT`` alone cannot distinguish
        "new and unproven" from "permanently disqualified" -- so consumers that
        must exclude rather than merely down-rank need this predicate. Selection
        is the case in point: the resolver's trust dimension only *scores*, so
        without an explicit frozen check a frozen-but-registered plugin stays
        eligible whenever governance has not also unregistered it.
        """
        return plugin_id in self._frozen

    def record_success(self, plugin_id: str) -> None:
        """Record a successful execution — accrue trust, may promote."""
        if plugin_id in self._frozen:
            return
        self._consecutive_ok[plugin_id] = self._consecutive_ok.get(plugin_id, 0) + 1
        self._consecutive_fail[plugin_id] = 0
        self._maybe_promote(plugin_id)

    def record_failure(self, plugin_id: str, *, hard: bool = False) -> None:
        """Record a failed execution — may demote.

        If hard=True (internal defect), freeze immediately to DRAFT.
        """
        if hard:
            self._frozen.add(plugin_id)
            self._levels[plugin_id] = PluginTrustLevel.DRAFT
            self._consecutive_ok[plugin_id] = 0
            self._consecutive_fail[plugin_id] = 0
            return
        if plugin_id in self._frozen:
            return
        self._consecutive_fail[plugin_id] = self._consecutive_fail.get(plugin_id, 0) + 1
        self._consecutive_ok[plugin_id] = 0
        if self._consecutive_fail[plugin_id] >= self._demote_after:
            self._demote(plugin_id)

    # ── Internal promotion / demotion ──

    def _maybe_promote(self, plugin_id: str) -> None:
        streak = self._consecutive_ok.get(plugin_id, 0)
        current = self._levels.get(plugin_id, PluginTrustLevel.DRAFT)
        if current < PluginTrustLevel.PRODUCTION and streak >= self._production_at:
            self._levels[plugin_id] = PluginTrustLevel.PRODUCTION
        elif current < PluginTrustLevel.VERIFIED and streak >= self._verified_at:
            self._levels[plugin_id] = PluginTrustLevel.VERIFIED
        elif current < PluginTrustLevel.CANDIDATE and streak >= self._candidate_at:
            self._levels[plugin_id] = PluginTrustLevel.CANDIDATE

    def _demote(self, plugin_id: str) -> None:
        current = self._levels.get(plugin_id, PluginTrustLevel.DRAFT)
        if current > PluginTrustLevel.DRAFT:
            self._levels[plugin_id] = PluginTrustLevel(current - 1)
        # Reset consecutive fail counter after demotion
        self._consecutive_fail[plugin_id] = 0

    # ── Durable state (for later persistence; logic-neutral) ──

    def to_state(self) -> Dict[str, Any]:
        """Serialize ledger state for persistence."""
        return {
            "candidate_at": self._candidate_at,
            "verified_at": self._verified_at,
            "production_at": self._production_at,
            "demote_after": self._demote_after,
            "consecutive_ok": dict(self._consecutive_ok),
            "consecutive_fail": dict(self._consecutive_fail),
            "levels": {k: v.value for k, v in self._levels.items()},
            "frozen": sorted(self._frozen),
        }

    @classmethod
    def load_state(cls, state: Dict[str, Any]) -> "PluginTrustLedger":
        """Restore ledger from serialized state."""
        if not state:
            return cls()
        ledger = cls(
            candidate_at=int(state.get("candidate_at", 5)),
            verified_at=int(state.get("verified_at", 20)),
            production_at=int(state.get("production_at", 50)),
            demote_after=int(state.get("demote_after", 3)),
        )
        ledger._consecutive_ok = {
            str(k): int(v) for k, v in (state.get("consecutive_ok") or {}).items()
        }
        ledger._consecutive_fail = {
            str(k): int(v) for k, v in (state.get("consecutive_fail") or {}).items()
        }
        ledger._levels = {
            str(k): PluginTrustLevel(int(v))
            for k, v in (state.get("levels") or {}).items()
        }
        ledger._frozen = {str(k) for k in (state.get("frozen") or [])}
        return ledger

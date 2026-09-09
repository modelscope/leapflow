"""Lifecycle governance for adaptive plugin proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.evolution_contracts import EvolutionLifecycleStore, OutcomeStore


@dataclass(frozen=True)
class LifecycleGovernanceResult:
    """Result of applying governance to one plugin outcome."""

    action: str
    plugin_id: str
    proposal_id: str = ""
    trust_level: str = "DRAFT"
    failure_streak: int = 0
    lifecycle_result: Mapping[str, Any] = field(default_factory=dict)
    outcome: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.lifecycle_result.get("ok", True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "plugin_id": self.plugin_id,
            "proposal_id": self.proposal_id,
            "trust_level": self.trust_level,
            "failure_streak": self.failure_streak,
            "lifecycle_result": dict(self.lifecycle_result),
            "outcome": dict(self.outcome),
        }


class LifecycleGovernor:
    """Update proposal lifecycle state from trust and execution outcomes.

    The stores are Protocol-typed rather than concrete so this machinery can be
    driven by whichever backing the live acquisition chain uses, not only by the
    default capability-proposal queue.
    """

    def __init__(
        self,
        *,
        proposal_queue: EvolutionLifecycleStore,
        outcome_store: OutcomeStore,
        lifecycle_actor: Any = None,
        trust_ledger: PluginTrustLedger | None = None,
        quarantine_after: int = 3,
        verified_at: PluginTrustLevel = PluginTrustLevel.VERIFIED,
    ) -> None:
        self._proposal_queue = proposal_queue
        self._outcome_store = outcome_store
        self._lifecycle_actor = lifecycle_actor
        self._trust_ledger = trust_ledger or PluginTrustLedger()
        self._quarantine_after = max(1, int(quarantine_after))
        self._verified_at = verified_at

    async def record_outcome(
        self,
        *,
        proposal_id: str,
        plugin_id: str,
        tool_name: str,
        ok: bool,
        requirement_id: str = "",
        plan_id: str = "",
        duration_ms: float = 0.0,
        failure_class: str = "",
        side_effect_state: str = "none",
        hard_failure: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> LifecycleGovernanceResult:
        """Record one outcome and apply lifecycle governance."""
        outcome = self._outcome_store.add_outcome(
            plugin_id=plugin_id,
            tool_name=tool_name,
            ok=ok,
            requirement_id=requirement_id,
            plan_id=plan_id,
            duration_ms=duration_ms,
            failure_class=failure_class,
            side_effect_state=side_effect_state,
            metadata=metadata,
        )
        if ok:
            self._trust_ledger.record_success(plugin_id)
        else:
            self._trust_ledger.record_failure(plugin_id, hard=hard_failure)

        trust = self._trust_ledger.level(plugin_id)
        failure_streak = self._outcome_store.failure_streak(plugin_id)
        lifecycle_result: Mapping[str, Any] = {"ok": True}
        action = "probation_execute"

        if hard_failure or failure_streak >= self._quarantine_after:
            action = "quarantine"
            if self._lifecycle_actor is not None:
                lifecycle_result = await self._lifecycle_actor.disable(plugin_id=plugin_id)
            self._proposal_queue.update(
                proposal_id,
                status="QUARANTINED" if lifecycle_result.get("ok", True) else "FAILED",
                trust_state={"level": trust.name, "failure_streak": failure_streak},
                test_results=[outcome],
                install_result=lifecycle_result,
            )
        elif trust >= self._verified_at:
            action = "verify"
            self._proposal_queue.update(
                proposal_id,
                status="VERIFIED",
                trust_state={"level": trust.name, "failure_streak": failure_streak},
                test_results=[outcome],
            )
        else:
            self._proposal_queue.update(
                proposal_id,
                status="PROBATION",
                trust_state={"level": trust.name, "failure_streak": failure_streak},
                test_results=[outcome],
            )
        return LifecycleGovernanceResult(
            action=action,
            plugin_id=plugin_id,
            proposal_id=proposal_id,
            trust_level=trust.name,
            failure_streak=failure_streak,
            lifecycle_result=lifecycle_result,
            outcome=outcome,
        )


__all__ = ["LifecycleGovernanceResult", "LifecycleGovernor"]

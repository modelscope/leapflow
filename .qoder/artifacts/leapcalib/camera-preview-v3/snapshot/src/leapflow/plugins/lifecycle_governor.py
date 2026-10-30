# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lifecycle governance for adaptive plugin proposals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.plugins.evolution_contracts import EvolutionLifecycleStore, OutcomeStore

logger = logging.getLogger(__name__)


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
        degradation_sink: Any = None,
    ) -> None:
        self._proposal_queue = proposal_queue
        self._outcome_store = outcome_store
        self._lifecycle_actor = lifecycle_actor
        self._trust_ledger = trust_ledger or PluginTrustLedger()
        self._quarantine_after = max(1, int(quarantine_after))
        self._verified_at = verified_at
        # Receives the health of a plugin that remains *in service*, on every outcome:
        # a non-zero failure streak is the state between healthy and quarantined, which
        # had no expression before, and a zero streak retires it again. Injected rather
        # than reached for, because turning this into capability-scoped evidence needs
        # the registry's declarations and this class deliberately has no registry.
        # Absence degrades to today's behaviour.
        self._degradation_sink = degradation_sink

    def _report_health(
        self,
        plugin_id: str,
        failure_streak: int,
        trust: PluginTrustLevel,
        failure_class: str = "",
    ) -> None:
        """Hand the still-serving plugin's health to the sink, if one is installed.

        Called for both outcomes. ``failure_streak`` carries the whole state
        declaratively: non-zero means "failing while still in service" -- the state
        between healthy and quarantined that had no expression before -- and zero means
        the provider has recovered, so whatever degradation was recorded for it can be
        retired. One signal, both directions, so the sink never has to infer recovery
        from an absence of reports.

        ``failure_class`` travels with it because the streak alone cannot say what kind
        of failure it was, and the answer differs entirely: a timeout is the retry
        layer's business, a missing scope is the operator's, and only a semantic
        mismatch is evidence about the implementation. Flattening them all into "failed
        twice" asks a hindsight evaluator to adjudicate a transient.

        The streak travels as *evidence strength*, never as a gate: no threshold here
        decides that a rival should be built. That judgement needs to tell a badly
        written implementation from a changed environment -- both produce consecutive
        failures and they want opposite actions -- which a counter cannot do and a
        hindsight evaluator can.

        Never raises: governance bookkeeping must not fail the sweep that drives it.
        """
        if self._degradation_sink is None:
            return
        try:
            self._degradation_sink(
                plugin_id=str(plugin_id or ""),
                failure_streak=int(failure_streak),
                trust_level=trust.name,
                failure_class=str(failure_class or ""),
            )
        except Exception:  # noqa: BLE001 - reporting is advisory
            logger.debug("lifecycle_governor: health not reported", exc_info=True)

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
        internal_defect = hard_failure or str(failure_class) == "internal_defect"
        if ok:
            self._trust_ledger.record_success(plugin_id)
        else:
            self._trust_ledger.record_failure(plugin_id, hard=internal_defect)

        trust = self._trust_ledger.level(plugin_id)
        failure_streak = self._outcome_store.failure_streak(plugin_id)
        trust_state = {
            "level": trust.name,
            "failure_streak": failure_streak,
            "frozen": self._trust_ledger.is_frozen(plugin_id),
        }
        lifecycle_result: Mapping[str, Any] = {"ok": True}
        action = "probation_execute"

        if internal_defect or failure_streak >= self._quarantine_after:
            action = "quarantine"
            if self._lifecycle_actor is not None:
                lifecycle_result = await self._lifecycle_actor.disable(plugin_id=plugin_id)
            self._transition(
                proposal_id,
                "QUARANTINED" if lifecycle_result.get("ok", True) else "FAILED",
                trust_state=trust_state,
                test_results=[outcome],
                install_result=lifecycle_result,
                metadata={
                    "terminal_reason": "internal_defect"
                    if internal_defect
                    else "failure_streak_exceeded"
                },
            )
        elif trust >= self._verified_at:
            action = "verify"
            self._transition(
                proposal_id,
                "VERIFIED",
                trust_state=trust_state,
                test_results=[outcome],
            )
        else:
            self._transition(
                proposal_id,
                "PROBATION",
                trust_state=trust_state,
                test_results=[outcome],
            )
        if action != "quarantine":
            # Reported on success as well as failure, because a health signal that only
            # ever fires one way has no way back. On success the streak is 0 by
            # construction, and that is the retirement signal: without it a degradation
            # record stays open forever, ``unresolved()`` grows monotonically, and the
            # teacher keeps being told a capability is failing long after it recovered
            # -- which would drive it to propose rivals for a healthy provider.
            #
            # Quarantine is excluded either way: a disabled plugin is a gap, not a
            # degradation, and it is no longer serving anything to recover.
            self._report_health(plugin_id, failure_streak, trust, failure_class)
        return LifecycleGovernanceResult(
            action=action,
            plugin_id=plugin_id,
            proposal_id=proposal_id,
            trust_level=trust.name,
            failure_streak=failure_streak,
            lifecycle_result=lifecycle_result,
            outcome=outcome,
        )

    def _transition(self, proposal_id: str, status: str, **changes: Any) -> None:
        """Use the lifecycle state machine when the backing store exposes it."""
        if not proposal_id:
            return
        transition = getattr(self._proposal_queue, "transition", None)
        if callable(transition):
            current = self._proposal_queue.get(proposal_id)
            if current is None:
                return
            if current.status == status:
                self._proposal_queue.update(proposal_id, **changes)
            else:
                transition(proposal_id, status, **changes)
            return
        self._proposal_queue.update(proposal_id, status=status, **changes)


__all__ = ["LifecycleGovernanceResult", "LifecycleGovernor"]

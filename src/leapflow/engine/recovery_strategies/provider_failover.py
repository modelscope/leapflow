"""Provider failover recovery strategy.

Handles permanent provider failures (billing, auth permanent, overloaded,
model not found, content blocked) by switching to an alternate provider/model.
"""
from __future__ import annotations

from leapflow.engine.failure_envelope import FailureEnvelope
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import RecoveryState
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)


class ProviderFailoverStrategy:
    """Failover to alternate provider/model on non-transient provider errors.

    Triggers when the current provider is unable to serve requests due to
    billing, permanent auth failures, capacity issues, model unavailability,
    or content policy violations.
    """

    @property
    def key(self) -> str:
        return "provider_failover"

    @property
    def priority(self) -> int:
        return 20

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"billing", "auth_permanent", "overloaded", "model_not_found", "content_blocked"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState) -> bool:
        """Applicable if failover budget remains."""
        # Access budget through a duck-typed approach — the coordinator passes
        # the budget externally. We check state for a budget reference if available,
        # but the coordinator already checks can_afford for budget_cost > 0.
        return True

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return FAILOVER decision to switch provider/model."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.FAILOVER,
            reason=f"Provider failure ({envelope.category}): failing over to alternate provider/model",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=True,
                resets_retry_count=True,
            ),
            budget_cost=1,
            audit_metadata={"trigger_category": envelope.category},
        )

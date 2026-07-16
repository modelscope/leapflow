"""Thinking mode disable recovery strategy.

Handles format errors by disabling the LLM's thinking/reasoning mode
which can conflict with certain output format expectations.
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


class ThinkingDisableStrategy:
    """Disable thinking mode to resolve format conflicts.

    Some LLM providers produce format errors when thinking/reasoning mode
    is active alongside structured output requirements. This strategy
    disables thinking mode as a transform-and-retry action.
    """

    @property
    def key(self) -> str:
        return "thinking_disable"

    @property
    def priority(self) -> int:
        return 30

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"format_error"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Always applicable -- the one-shot guard handles dedup at coordinator level."""
        return True

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return TRANSFORM_AND_RETRY to disable thinking mode."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Disable thinking mode to resolve format conflict",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=False,
                resets_retry_count=False,
            ),
            budget_cost=0,
        )

"""Tool schema expansion recovery strategy.

Handles unknown tool errors by expanding the tool schema to include
additional tools that the LLM attempted to call but were not in scope.
"""
from __future__ import annotations

from leapflow.engine.failure_envelope import FailureEnvelope, Recoverability
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import RecoveryState
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)


class ToolSchemaExpandStrategy:
    """Expand tool schema when an unknown tool call is auto-recoverable.

    When the LLM produces a tool call for a tool not in the current schema
    but marked as retryable (indicating the tool exists but wasn't disclosed),
    this strategy expands the schema and retries.
    """

    @property
    def key(self) -> str:
        return "tool_schema_expand"

    @property
    def priority(self) -> int:
        return 40

    @property
    def repeatable(self) -> bool:
        return False

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"tool"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"tool_unknown"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable when the tool error is marked as auto-recoverable."""
        return envelope.recoverability == Recoverability.AUTO_RECOVER

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return TRANSFORM_AND_RETRY to expand the tool schema."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Expand tool schema for unknown tool recovery",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=False,
                resets_retry_count=False,
            ),
            budget_cost=0,
            audit_metadata={"tool_name": envelope.context.tool_name},
        )

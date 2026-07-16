"""Native-to-text fallback recovery strategy.

Handles format errors by falling back from native tool calling mode to
text-based tool calling when the native mode produces parsing conflicts.
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


class NativeToTextFallbackStrategy:
    """Fall back from native tool calling to text-based tool calling.

    When native (function calling) mode produces format errors due to
    provider limitations or model incompatibilities, this strategy switches
    to text-mode tool calling where tool calls are parsed from LLM text output.
    """

    @property
    def key(self) -> str:
        return "native_to_text"

    @property
    def priority(self) -> int:
        return 35

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"format_error"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable when native tool mode is indicated by failure context.

        Checks for indicators that native tool calling mode is active:
        - Message mentions 'tool_call', 'function_call', or 'native'
        - The failure indicates a tool calling parse/format issue
        """
        msg = envelope.message.lower()
        return (
            "tool_call" in msg
            or "function_call" in msg
            or "native" in msg
            or "tool_use" in msg
            or "tools" in msg
        )

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return TRANSFORM_AND_RETRY to switch from native to text tool mode."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Fall back from native tool calling to text mode",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=False,
                resets_retry_count=False,
            ),
            budget_cost=0,
        )

"""Context compression recovery strategy.

Handles context overflow errors by progressively compressing the conversation
context through three phases: history summarization, multimodal-to-text
conversion, and disclosure shrinkage.
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

_PHASES = (
    "history_summarize",
    "multimodal_to_text",
    "disclosure_shrink",
)


class ContextCompressStrategy:
    """Progressive context compression for context overflow errors.

    Three compression phases are applied in order:
    1. history_summarize — Summarize older conversation history
    2. multimodal_to_text — Convert multimodal content to text descriptions
    3. disclosure_shrink — Reduce progressive disclosure payload size
    """

    @property
    def key(self) -> str:
        return "context_compress"

    @property
    def priority(self) -> int:
        return 10

    @property
    def repeatable(self) -> bool:
        return True

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"context_overflow", "payload_too_large"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable if there are remaining compression phases."""
        return state.compress_phase_index < len(_PHASES)

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return TRANSFORM_AND_RETRY with the current compression phase."""
        phase_idx = state.compress_phase_index
        phase_name = _PHASES[phase_idx]

        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason=f"Context overflow: applying compression phase '{phase_name}' "
                   f"({phase_idx + 1}/{len(_PHASES)})",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=False,
                resets_retry_count=False,
            ),
            budget_cost=0,
            audit_metadata={"phase": phase_name, "phase_index": phase_idx},
        )

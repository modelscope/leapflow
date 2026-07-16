"""Multimodal strip recovery strategy.

Handles image-too-large errors by stripping multimodal content and converting
to text descriptions before retry.
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


class MultimodalStripStrategy:
    """Strip multimodal content when images are too large for the provider.

    Converts images and other multimodal content to text descriptions,
    allowing the request to proceed within provider size limits.
    """

    @property
    def key(self) -> str:
        return "multimodal_strip"

    @property
    def priority(self) -> int:
        return 15

    @property
    def repeatable(self) -> bool:
        return False

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset({"image_too_large"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable when the failure message indicates an image size issue."""
        msg = envelope.message.lower()
        return "image" in msg or "large" in msg or "size" in msg or envelope.category == "image_too_large"

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return TRANSFORM_AND_RETRY to strip multimodal content."""
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Image too large: stripping multimodal content to text descriptions",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=False,
                resets_retry_count=False,
            ),
            budget_cost=0,
        )

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Compression timeout recovery strategy with stepped cooldown.

When context compression operations time out repeatedly, this strategy
applies escalating cooldown periods (60s → 300s → 900s) inspired by
Hermes's stepped cooldown pattern.  After all compression paths are
exhausted it inserts a deterministic summary placeholder so the turn
can continue without LLM-generated compression.
"""
from __future__ import annotations

import time

from leapflow.engine.recovery.failure_envelope import FailureEnvelope
from leapflow.engine.recovery.recovery_budget import RecoveryBudget
from leapflow.engine.recovery.recovery_coordinator import RecoveryState
from leapflow.engine.recovery.recovery_decision import (
    BackoffConfig,
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)

# Stepped cooldown: consecutive timeout count → cooldown seconds.
# Counts beyond the last entry clamp to the final tier.
_COOLDOWN_TIERS: tuple[tuple[int, float], ...] = (
    (1, 60.0),
    (2, 300.0),
    (3, 900.0),
)

# Categories that indicate a compression timeout.
_APPLICABLE_CATEGORIES: frozenset[str] = frozenset({
    "compression_timeout",
    "context_compression_timeout",
})

# Placeholder injected when deterministic degradation is triggered.
DETERMINISTIC_SUMMARY_PLACEHOLDER = (
    "[DETERMINISTIC SUMMARY — compression unavailable]"
)

# Maximum consecutive timeouts before we give up retrying and degrade.
_MAX_RETRIES_BEFORE_DEGRADATION = 3


def _cooldown_for_count(count: int) -> float:
    """Return the cooldown duration (seconds) for *count* consecutive timeouts."""
    for threshold, cooldown in _COOLDOWN_TIERS:
        if count <= threshold:
            return cooldown
    # Clamp to the highest tier.
    return _COOLDOWN_TIERS[-1][1]


class CompressionTimeoutStrategy:
    """Stepped cooldown for compression timeout failures.

    Cooldown ladder (consecutive timeout count → wait):
        1  → 60 s
        2  → 300 s
        3+ → 900 s  (then deterministic degradation)

    When the consecutive count reaches ``_MAX_RETRIES_BEFORE_DEGRADATION``
    *and* the budget cannot afford another attempt, the strategy emits a
    ``SKIP_AND_CONTINUE`` with the deterministic summary placeholder so the
    agent loop can proceed without compressed context.
    """

    def __init__(self) -> None:
        self._consecutive_timeouts: int = 0
        self._last_cooldown_end: float = 0.0

    # -- Protocol properties --------------------------------------------------

    @property
    def key(self) -> str:
        return "compression_timeout"

    @property
    def priority(self) -> int:
        # Between context_compress (10) and multimodal_strip (15).
        # Acts as a timeout-aware fallback when compression stalls.
        return 12

    @property
    def repeatable(self) -> bool:
        return True

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm", "system"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return _APPLICABLE_CATEGORIES

    # -- Cooldown bookkeeping (called externally or by tests) -----------------

    @property
    def consecutive_timeouts(self) -> int:
        """Current consecutive timeout count."""
        return self._consecutive_timeouts

    def record_success(self) -> None:
        """Reset the consecutive timeout counter after a successful compression."""
        self._consecutive_timeouts = 0
        self._last_cooldown_end = 0.0

    def _advance_timeout(self) -> float:
        """Increment the counter and return the new cooldown duration."""
        self._consecutive_timeouts += 1
        return _cooldown_for_count(self._consecutive_timeouts)

    # -- Protocol methods -----------------------------------------------------

    def can_apply(
        self,
        envelope: FailureEnvelope,
        state: RecoveryState,
        budget: RecoveryBudget | None = None,
    ) -> bool:
        """Applicable when the failure is a compression timeout.

        If a cooldown period is still in effect we defer to the caller (the
        coordinator will see ``False`` and move on to the next strategy).
        """
        if budget is not None and not budget.can_afford(1, envelope.category):
            # Budget exhausted — trigger deterministic degradation path.
            return self._consecutive_timeouts >= _MAX_RETRIES_BEFORE_DEGRADATION

        # Honour active cooldown — do not re-enter compression while cooling.
        now = time.monotonic()
        if self._last_cooldown_end > 0 and now < self._last_cooldown_end:
            return False

        return True

    def decide(
        self,
        envelope: FailureEnvelope,
        state: RecoveryState,
    ) -> RecoveryDecision:
        """Produce a stepped-cooldown retry or deterministic degradation."""
        cooldown = self._advance_timeout()

        # --- Deterministic degradation path ---
        if self._consecutive_timeouts >= _MAX_RETRIES_BEFORE_DEGRADATION:
            return RecoveryDecision.create(
                envelope=envelope,
                action=RecoveryAction.SKIP_AND_CONTINUE,
                reason=(
                    f"Compression timed out {self._consecutive_timeouts} consecutive "
                    f"times — degrading to deterministic summary"
                ),
                strategy_key=self.key,
                retry_semantics=RetrySemantics(
                    consumes_retry_budget=False,
                    resets_retry_count=False,
                ),
                budget_cost=0,
                audit_metadata={
                    "consecutive_timeouts": self._consecutive_timeouts,
                    "degradation": True,
                    "placeholder": DETERMINISTIC_SUMMARY_PLACEHOLDER,
                },
                transform_description=DETERMINISTIC_SUMMARY_PLACEHOLDER,
            )

        # --- Normal cooldown-and-retry path ---
        self._last_cooldown_end = time.monotonic() + cooldown

        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            reason=(
                f"Compression timeout #{self._consecutive_timeouts}: "
                f"cooldown {cooldown:.0f}s before retry"
            ),
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=True,
                resets_retry_count=False,
                backoff_config=BackoffConfig(
                    base_delay=cooldown,
                    max_delay=cooldown,
                    jitter_ratio=0.0,
                    algorithm="fixed",
                ),
            ),
            budget_cost=1,
            audit_metadata={
                "consecutive_timeouts": self._consecutive_timeouts,
                "cooldown_seconds": cooldown,
                "degradation": False,
            },
        )

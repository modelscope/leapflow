"""Jittered retry recovery strategy.

The lowest-priority catch-all retry strategy for transient failures.
Applies decorrelated jitter backoff with category-specific delay parameters.
"""
from __future__ import annotations

from leapflow.engine.failure_envelope import FailureEnvelope, Recoverability
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import RecoveryState
from leapflow.engine.recovery_decision import (
    BackoffConfig,
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)

# Category-specific backoff configurations
_BACKOFF_CONFIGS: dict[str, BackoffConfig] = {
    "rate_limited": BackoffConfig(
        base_delay=5.0,
        max_delay=120.0,
        jitter_ratio=0.5,
        algorithm="decorrelated_jitter",
    ),
    "transient": BackoffConfig(
        base_delay=1.0,
        max_delay=60.0,
        jitter_ratio=0.5,
        algorithm="decorrelated_jitter",
    ),
    "overloaded": BackoffConfig(
        base_delay=1.0,
        max_delay=60.0,
        jitter_ratio=0.5,
        algorithm="decorrelated_jitter",
    ),
    "tool_timeout": BackoffConfig(
        base_delay=2.0,
        max_delay=30.0,
        jitter_ratio=0.5,
        algorithm="decorrelated_jitter",
    ),
}

_DEFAULT_BACKOFF = BackoffConfig(
    base_delay=1.0,
    max_delay=60.0,
    jitter_ratio=0.5,
    algorithm="decorrelated_jitter",
)


class JitteredRetryStrategy:
    """Catch-all retry with decorrelated jitter backoff for transient failures.

    This is the lowest-priority strategy that handles any AUTO_RETRY-recoverable
    failure with exponential backoff and jitter. It consumes retry budget and
    uses category-specific delay parameters.
    """

    @property
    def key(self) -> str:
        return "jittered_retry"

    @property
    def priority(self) -> int:
        return 100

    @property
    def repeatable(self) -> bool:
        return True

    @property
    def applicable_sources(self) -> frozenset[str]:
        return frozenset({"llm", "tool", "system"})

    @property
    def applicable_categories(self) -> frozenset[str]:
        return frozenset()  # Empty = wildcard, matches all categories

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Applicable for AUTO_RETRY failures when budget allows."""
        return envelope.recoverability == Recoverability.AUTO_RETRY

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Return RETRY_WITH_BACKOFF with category-appropriate backoff config."""
        backoff = _BACKOFF_CONFIGS.get(envelope.category, _DEFAULT_BACKOFF)

        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            reason=f"Transient failure ({envelope.category}): retrying with jittered backoff",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=True,
                resets_retry_count=False,
                backoff_config=backoff,
            ),
            budget_cost=1,
            audit_metadata={
                "category": envelope.category,
                "base_delay": backoff.base_delay,
                "max_delay": backoff.max_delay,
            },
        )

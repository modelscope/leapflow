"""Iteration budget management for bounded agent loops."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class BudgetStatus(Enum):
    OK = "ok"
    WARNING = "warning"
    SOFT_LIMIT = "soft_limit"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for iteration budget control."""

    max_iterations: int = 20
    soft_limit: int = 14
    warning_threshold: int = 10
    refundable_tools: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"shell"})
    )


class IterationBudget:
    """Tracks iteration consumption with three-tier alerting."""

    def __init__(self, config: BudgetConfig):
        self._config = config
        self._consumed = 0
        self._refunded = 0

    def consume(self) -> BudgetStatus:
        """Consume one iteration. Returns current budget status."""
        self._consumed += 1
        used = self._consumed - self._refunded
        if used >= self._config.max_iterations:
            return BudgetStatus.EXHAUSTED
        if used >= self._config.soft_limit:
            return BudgetStatus.SOFT_LIMIT
        if used >= self._config.warning_threshold:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def refund(self, reason: str = "") -> None:
        """Refund one iteration (e.g., long-running tool calls)."""
        self._refunded += 1

    @property
    def remaining(self) -> int:
        return self._config.max_iterations - (self._consumed - self._refunded)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def used(self) -> int:
        return self._consumed - self._refunded

    @classmethod
    def for_react(cls, config: BudgetConfig) -> "IterationBudget":
        """Factory for the main ReAct loop."""
        return cls(config)

    @classmethod
    def for_tool_execution(cls, max_calls: int = 30, soft: int = 24) -> "IterationBudget":
        """Factory for tool execution within a single skill step."""
        return cls(
            BudgetConfig(
                max_iterations=max_calls,
                soft_limit=soft,
                warning_threshold=max(1, soft - 4),
            )
        )

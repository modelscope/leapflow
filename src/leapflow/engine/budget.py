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
    """Configuration for iteration budget control.

    Supports two modes, selected purely by configuration:

    * Fixed (default): ``iter_ceiling <= max_iterations``. The cap is
      ``max_iterations`` and the soft/warning tiers are the absolute
      ``soft_limit`` / ``warning_threshold``. This is what bounded, one-shot
      budgets (e.g. per-skill tool execution) want, and it is byte-identical to
      the historical behavior.
    * Elastic: ``iter_ceiling > max_iterations``. ``max_iterations`` becomes the
      baseline floor and the effective cap can be raised at runtime up to
      ``iter_ceiling`` via :meth:`IterationBudget.retarget`, proportionally to an
      observed difficulty signal. Soft/warning tiers then scale with the current
      effective cap (``soft_ratio`` / ``warning_ratio``) so "approaching the
      limit" tracks the widened horizon rather than a stale constant.
    """

    max_iterations: int = 20
    soft_limit: int = 14
    warning_threshold: int = 10
    iter_ceiling: int = 0
    scale_k: float = 1.0
    max_refunds: int = 0
    soft_ratio: float = 0.8
    warning_ratio: float = 0.5
    refundable_tools: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"shell"})
    )

    @property
    def ceiling(self) -> int:
        """Absolute upper bound on iterations (elastic ceiling or fixed cap)."""
        return self.iter_ceiling if self.iter_ceiling > self.max_iterations else self.max_iterations

    @property
    def elastic(self) -> bool:
        """Whether this budget can widen its cap at runtime."""
        return self.iter_ceiling > self.max_iterations


class IterationBudget:
    """Tracks iteration consumption with three-tier alerting.

    The effective cap starts at ``config.max_iterations`` (the baseline) and, for
    elastic configs, may be raised monotonically toward ``config.ceiling`` via
    :meth:`retarget` as difficulty rises. It is never lowered below what has
    already been consumed (a physical constraint) nor below the baseline.
    """

    def __init__(self, config: BudgetConfig):
        self._config = config
        self._consumed = 0
        self._refunded = 0
        self._effective_max = max(1, config.max_iterations)

    def consume(self) -> BudgetStatus:
        """Consume one iteration. Returns current budget status."""
        self._consumed += 1
        used = self._consumed - self._refunded
        if used >= self._effective_max:
            return BudgetStatus.EXHAUSTED
        if self._config.elastic:
            soft = max(1, round(self._config.soft_ratio * self._effective_max))
            warning = max(1, round(self._config.warning_ratio * self._effective_max))
        else:
            soft = self._config.soft_limit
            warning = self._config.warning_threshold
        if used >= soft:
            return BudgetStatus.SOFT_LIMIT
        if used >= warning:
            return BudgetStatus.WARNING
        return BudgetStatus.OK

    def refund(self, reason: str = "") -> None:
        """Refund one iteration (e.g., long-running tool calls).

        Bounded by ``config.max_refunds`` when set (>0) so a slow, refundable
        tool cannot manufacture an unbounded loop; ``0`` means unbounded (legacy).
        """
        if self._config.max_refunds and self._refunded >= self._config.max_refunds:
            return
        self._refunded += 1

    def elastic_max(self, difficulty: float) -> int:
        """Target cap for a given difficulty in [0, 1] (clamped to [base, ceiling])."""
        base = self._config.max_iterations
        span = max(0, self._config.ceiling - base)
        clamped = max(0.0, min(1.0, difficulty))
        target = base + round(self._config.scale_k * clamped * span)
        return max(base, min(target, self._config.ceiling))

    def retarget(self, new_max: int) -> None:
        """Raise the effective cap toward a new target (monotonic, bounded).

        Never lowers below what is already consumed or below the baseline, and
        never exceeds the configured ceiling.
        """
        floor = max(self._config.max_iterations, self.used)
        candidate = min(int(new_max), self._config.ceiling)
        self._effective_max = max(self._effective_max, candidate, floor)

    @property
    def remaining(self) -> int:
        return self._effective_max - (self._consumed - self._refunded)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def used(self) -> int:
        return self._consumed - self._refunded

    @property
    def effective_max(self) -> int:
        """Current effective iteration cap (baseline, possibly widened)."""
        return self._effective_max

    @classmethod
    def for_react(cls, config: BudgetConfig) -> "IterationBudget":
        """Factory for the main ReAct loop."""
        return cls(config)

    @classmethod
    def for_tool_execution(cls, max_calls: int = 30, soft: int = 24) -> "IterationBudget":
        """Factory for tool execution within a single skill step (fixed budget)."""
        return cls(
            BudgetConfig(
                max_iterations=max_calls,
                soft_limit=soft,
                warning_threshold=max(1, soft - 4),
            )
        )

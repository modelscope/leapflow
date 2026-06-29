"""Token-bucket based budget controller for world model learning operations.

Manages compute budgets for prediction, comparison, and replay calls
to prevent unbounded LLM token consumption during curiosity-driven learning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class _TokenPool:
    """Single refillable token pool with session-aware recalibration."""

    max_tokens: float
    tokens: float = field(init=False)
    _created_at: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.max_tokens

    def consume(self) -> bool:
        """Consume one token. Returns True if successful."""
        if self.tokens < 1.0:
            return False
        self.tokens = max(0.0, self.tokens - 1.0)
        return True

    @property
    def available(self) -> bool:
        return self.tokens >= 1.0

    def recalibrate(self, remaining_fraction: float) -> None:
        """Adjust tokens proportionally to remaining session time."""
        if remaining_fraction <= 0:
            return
        expected = self.max_tokens * remaining_fraction
        if self.tokens > expected * 1.5:
            self.tokens = expected * 1.2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


class LearningBudgetController:
    """Multi-pool budget controller for world model learning operations.

    Each pool independently tracks a token budget that depletes as learning
    operations (prediction, comparison, replay) are performed during a session.
    Supports end-of-session rebalancing based on learning outcomes.
    """

    def __init__(
        self,
        prediction_budget: int = 50,
        comparison_budget: int = 20,
        replay_budget: int = 3,
        grading_budget: int = 5,
        distillation_budget: int = 2,
        *,
        budget_bounds: Optional[Dict[str, tuple]] = None,
        discovery_baseline: int = 2,
        regression_baseline: int = 1,
    ) -> None:
        self._pools: Dict[str, _TokenPool] = {
            "prediction": _TokenPool(max_tokens=float(prediction_budget)),
            "comparison": _TokenPool(max_tokens=float(comparison_budget)),
            "replay": _TokenPool(max_tokens=float(replay_budget)),
            "grading": _TokenPool(max_tokens=float(grading_budget)),
            "distillation": _TokenPool(max_tokens=float(distillation_budget)),
        }
        self._bounds: Dict[str, tuple] = budget_bounds or {
            "prediction": (10.0, 200.0),
            "comparison": (5.0, 80.0),
            "replay": (1.0, 20.0),
            "grading": (1.0, 20.0),
            "distillation": (1.0, 10.0),
        }
        self._discovery_baseline = max(1, discovery_baseline)
        self._regression_baseline = max(1, regression_baseline)

    def has_tokens(self, pool_name: str) -> bool:
        pool = self._pools.get(pool_name)
        return pool is not None and pool.available

    def spend(self, pool_name: str) -> bool:
        """Spend one token from the named pool. Returns False if exhausted."""
        pool = self._pools.get(pool_name)
        if pool is None:
            return False
        return pool.consume()

    def report_session_progress(self, fraction_remaining: float) -> None:
        """Recalibrate all pools based on estimated session progress."""
        for pool in self._pools.values():
            pool.recalibrate(fraction_remaining)

    def adjust_for_accuracy(self, recent_delta_mean: float) -> None:
        """Dynamically shift budget allocation based on prediction accuracy."""
        pred = self._pools.get("prediction")
        comp = self._pools.get("comparison")
        replay = self._pools.get("replay")
        if not (pred and comp and replay):
            return

        if recent_delta_mean < 0.1:
            pred.max_tokens *= 0.8
            replay.max_tokens = min(replay.max_tokens * 1.2, 10.0)
        elif recent_delta_mean > 0.5:
            pred.max_tokens = min(pred.max_tokens * 1.3, 200.0)
            comp.max_tokens = min(comp.max_tokens * 1.2, 80.0)

    @property
    def status(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {"tokens": p.tokens, "max": p.max_tokens}
            for name, p in self._pools.items()
        }

    def rebalance_from_session_outcome(
        self,
        skills_discovered: int = 0,
        regressions_detected: int = 0,
        avg_prediction_delta: float = 0.0,
    ) -> None:
        """Adaptively adjust pool capacities based on session learning outcomes.

        Called at end-of-session. Shifts budget toward areas that showed
        higher value or urgency during the completed session.
        """
        discovery_ratio = min(skills_discovered / self._discovery_baseline, 2.0)
        regression_ratio = min(regressions_detected / self._regression_baseline, 2.0)
        efficiency = 1.0 - min(avg_prediction_delta, 1.0)

        adjustments: Dict[str, float] = {
            "prediction": 0.8 + discovery_ratio * 0.4 - efficiency * 0.2,
            "comparison": 0.8 + regression_ratio * 0.2,
            "replay": 0.8 + regression_ratio * 0.4,
            "grading": 0.8 + discovery_ratio * 0.2,
            "distillation": 0.8 + discovery_ratio * 0.3,
        }

        for pool_name, factor in adjustments.items():
            pool = self._pools.get(pool_name)
            if pool is None:
                continue
            lo, hi = self._bounds.get(pool_name, (1.0, 200.0))
            pool.max_tokens = _clamp(pool.max_tokens * factor, lo, hi)

        logger.info(
            "budget.rebalanced discovered=%d regressions=%d delta=%.2f",
            skills_discovered, regressions_detected, avg_prediction_delta,
        )

    def reset(self) -> None:
        """Reset all pools to their max capacity."""
        for pool in self._pools.values():
            pool.tokens = pool.max_tokens

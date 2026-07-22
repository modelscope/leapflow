"""Recovery budget — global constraint system for recovery attempts within a turn.

The budget prevents infinite recovery loops by enforcing hard caps on retries,
transforms, failovers, credential rotations, and wall-clock time within a
single agent loop turn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecoveryBudget:
    """Global constraint system for recovery attempts within a turn.

    Provides per-turn and per-category limits plus a wall-clock deadline.
    Strategies consult the budget before deciding whether to attempt recovery.
    """

    max_retries_per_turn: int = 8
    max_retry_per_category: int = 3
    max_transform_attempts: int = 2
    max_failovers: int = 2
    max_credential_rotations: int = 3
    turn_deadline_s: float = 300.0
    total_recovery_actions: int = 12

    # Internal accounting — not meant for external configuration
    _consumed: int = field(default=0, init=False, repr=False)
    _category_consumed: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _transforms_used: int = field(default=0, init=False, repr=False)
    _failovers_used: int = field(default=0, init=False, repr=False)
    _rotations_used: int = field(default=0, init=False, repr=False)
    _deadline_start: float = field(default=0.0, init=False, repr=False)

    def start_deadline(self) -> None:
        """Called at turn start to anchor the wall-clock deadline."""
        self._deadline_start = time.monotonic()

    def new_turn(self) -> None:
        """Reset all per-turn accounting for a new turn.

        Configuration limits (max_retries_per_turn, etc.) remain unchanged;
        only the consumption counters and deadline are reset.
        """
        self._consumed = 0
        self._category_consumed.clear()
        self._transforms_used = 0
        self._failovers_used = 0
        self._rotations_used = 0
        self.start_deadline()

    def can_afford(self, cost: int, category: str = "") -> bool:
        """Check whether spending `cost` actions is within budget.

        Checks global budget, per-category budget, and deadline.
        """
        if self.is_deadline_exceeded():
            return False
        if self._consumed + cost > self.total_recovery_actions:
            return False
        if category:
            cat_used = self._category_consumed.get(category, 0)
            if cat_used + cost > self.max_retry_per_category:
                return False
        return True

    def consume(self, cost: int, category: str = "") -> None:
        """Record that `cost` recovery actions were consumed.

        Raises ValueError if the budget is exceeded — callers should check
        can_afford first.
        """
        if not self.can_afford(cost, category):
            raise ValueError(
                f"Recovery budget exceeded: consumed={self._consumed}, "
                f"cost={cost}, category={category!r}, "
                f"category_used={self._category_consumed.get(category, 0)}"
            )
        self._consumed += cost
        if category:
            self._category_consumed[category] = (
                self._category_consumed.get(category, 0) + cost
            )

    def consume_transform(self) -> None:
        """Record a transform attempt (context compression, payload reduction)."""
        self._transforms_used += 1

    def consume_failover(self) -> None:
        """Record a failover attempt (model switch, endpoint rotation)."""
        self._failovers_used += 1

    def consume_rotation(self) -> None:
        """Record a credential rotation attempt."""
        self._rotations_used += 1

    def can_transform(self) -> bool:
        """Whether transform budget remains."""
        return self._transforms_used < self.max_transform_attempts

    def can_failover(self) -> bool:
        """Whether failover budget remains."""
        return self._failovers_used < self.max_failovers

    def can_rotate(self) -> bool:
        """Whether credential rotation budget remains."""
        return self._rotations_used < self.max_credential_rotations

    def is_deadline_exceeded(self) -> bool:
        """Whether the wall-clock deadline for this turn has been exceeded.

        A non-positive ``turn_deadline_s`` means *no* wall-clock deadline
        (unlimited): recovery is then bounded only by the action-count budget, so
        a long-running task is never denied recovery for a late transient error
        merely because wall-clock time has elapsed.
        """
        if self._deadline_start == 0.0 or self.turn_deadline_s <= 0:
            return False
        elapsed = time.monotonic() - self._deadline_start
        return elapsed > self.turn_deadline_s

    def remaining(self) -> int:
        """Remaining global recovery action budget."""
        return max(0, self.total_recovery_actions - self._consumed)

    def category_remaining(self, category: str) -> int:
        """Remaining budget for a specific error category."""
        used = self._category_consumed.get(category, 0)
        return max(0, self.max_retry_per_category - used)

    def summary(self) -> dict[str, Any]:
        """Return an audit-friendly summary of budget consumption."""
        return {
            "consumed": self._consumed,
            "remaining": self.remaining(),
            "total_budget": self.total_recovery_actions,
            "category_consumed": dict(self._category_consumed),
            "transforms_used": self._transforms_used,
            "failovers_used": self._failovers_used,
            "rotations_used": self._rotations_used,
            "deadline_exceeded": self.is_deadline_exceeded(),
            "elapsed_s": (
                round(time.monotonic() - self._deadline_start, 2)
                if self._deadline_start > 0
                else 0.0
            ),
        }

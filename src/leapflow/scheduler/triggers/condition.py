"""Condition-based trigger: fires when a declarative condition is met.

Supports simple comparison expressions like:
    "file_count > 50"
    "disk_usage > 90"
    "queue_length >= 100"

Does NOT use eval() — implements a safe expression parser.
"""

from __future__ import annotations

import operator
import re
import time
from typing import Any, Callable, ClassVar, Dict, Optional


# Supported comparison operators
_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# Pattern: metric_name <op> value (with optional % suffix on value)
_CONDITION_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*"  # metric name
    r"([><=!]+)\s*"  # operator
    r"(\d+(?:\.\d+)?)\s*%?\s*$"  # numeric value (optional % ignored)
)


class ConditionTrigger:
    """Fires when a declarative condition expression evaluates to True.

    The condition is evaluated against a context dict that maps metric
    names to their current numeric values. The context is updated
    externally via :meth:`update_context`.

    Example expressions:
        "file_count > 50"
        "disk_usage > 90"
        "queue_length >= 100"

    Security: Does NOT use eval(). Implements a simple comparison parser.
    """

    trigger_type: ClassVar[str] = "condition"

    def __init__(
        self,
        expression: str,
        *,
        check_interval: float = 60.0,
        next_due_at: float = 0.0,
    ) -> None:
        if not expression:
            raise ValueError("expression must not be empty")
        self._expression = expression.strip()
        self._check_interval = check_interval
        self._next_due_at = next_due_at if next_due_at > 0 else time.time()
        self._context: Dict[str, float] = {}

        # Parse and validate expression at construction time
        self._parsed = self._parse_expression(self._expression)
        if self._parsed is None:
            raise ValueError(
                f"Cannot parse condition expression: {self._expression!r}. "
                f"Expected format: 'metric_name <op> value' where op is one of "
                f"{list(_OPERATORS.keys())}"
            )

    # ------------------------------------------------------------------
    # Trigger Protocol
    # ------------------------------------------------------------------

    @property
    def trigger_type(self) -> str:  # type: ignore[override]
        return "condition"

    def is_due(self, now: float) -> bool:
        """Evaluate the condition against current context.

        Only checks when now >= next_due_at (respects check_interval).
        Returns True if the condition is satisfied.
        """
        if now < self._next_due_at:
            return False
        return self._evaluate()

    def advance(self, now: float) -> None:
        """Schedule the next check at now + check_interval."""
        self._next_due_at = now + self._check_interval

    @property
    def next_due_at(self) -> float:
        return self._next_due_at

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "trigger_type": "condition",
            "expression": self._expression,
            "check_interval": self._check_interval,
            "next_due_at": self._next_due_at,
        }

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def update_context(self, context: Dict[str, float]) -> None:
        """Update the metric context used for condition evaluation.

        Args:
            context: Dict mapping metric names to numeric values.
        """
        self._context.update(context)

    def set_metric(self, name: str, value: float) -> None:
        """Set a single metric value in the context.

        Args:
            name: Metric name (must match expression's left-hand side).
            value: Current numeric value.
        """
        self._context[name] = value

    # ------------------------------------------------------------------
    # Expression parsing (safe, no eval)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_expression(
        expression: str,
    ) -> Optional[tuple[str, Callable[[Any, Any], bool], float]]:
        """Parse a condition expression into (metric, operator_fn, threshold).

        Returns None if the expression cannot be parsed.
        """
        match = _CONDITION_RE.match(expression)
        if not match:
            return None

        metric_name = match.group(1)
        op_str = match.group(2)
        threshold = float(match.group(3))

        op_fn = _OPERATORS.get(op_str)
        if op_fn is None:
            return None

        return (metric_name, op_fn, threshold)

    def _evaluate(self) -> bool:
        """Evaluate the parsed condition against current context."""
        if self._parsed is None:
            return False

        metric_name, op_fn, threshold = self._parsed
        current_value = self._context.get(metric_name)
        if current_value is None:
            # Metric not in context — condition cannot be satisfied
            return False

        return op_fn(float(current_value), threshold)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def deserialize(cls, data: dict) -> "ConditionTrigger":
        """Reconstruct from serialized dict."""
        return cls(
            expression=data["expression"],
            check_interval=data.get("check_interval", 60.0),
            next_due_at=data.get("next_due_at", 0.0),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def expression(self) -> str:
        """The condition expression string."""
        return self._expression

    @property
    def check_interval(self) -> float:
        """Seconds between condition evaluations."""
        return self._check_interval

    @property
    def context(self) -> Dict[str, float]:
        """Current metric context (read-only copy)."""
        return dict(self._context)

    def __repr__(self) -> str:
        return (
            f"ConditionTrigger(expression={self._expression!r}, "
            f"check_interval={self._check_interval})"
        )

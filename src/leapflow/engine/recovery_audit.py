"""Recovery audit system — structured logging of all recovery decisions.

Every RecoveryCoordinator.evaluate() call produces an audit entry written
to a JSONL file for post-hoc analysis, debugging, and learning.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryAuditEntry:
    """Immutable audit record for one recovery decision."""

    timestamp: float
    session_id: str
    turn_id: int

    # Input (from FailureEnvelope)
    envelope_id: str
    failure_source: str
    failure_category: str
    failure_code: str
    recoverability: str

    # Decision (from RecoveryDecision)
    decision_id: str
    strategy_key: str
    action: str
    reason: str
    budget_cost: int = 0

    # Budget snapshot
    budget_consumed: int = 0
    budget_remaining: int = 0

    # Outcome (filled async after execution)
    outcome: str = ""           # "success" | "failure" | "timeout" | ""
    outcome_reason: str = ""
    elapsed_ms: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {k: v for k, v in asdict(self).items() if v != "" and v != 0}


class RecoveryAuditSink(Protocol):
    """Protocol for audit sinks. Allows different backends."""

    def record(self, entry: RecoveryAuditEntry) -> None:
        """Write an audit entry."""
        ...

    def update_outcome(self, decision_id: str, outcome: str,
                       reason: str = "", elapsed_ms: float = 0.0) -> None:
        """Update the outcome of a previously recorded decision."""
        ...


class JsonlAuditSink:
    """JSONL file-based audit sink.

    Writes one JSON line per entry to a .jsonl file.
    Thread-safe via append mode.
    """

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else None
        self._entries: list[RecoveryAuditEntry] = []  # In-memory buffer for testing

    def record(self, entry: RecoveryAuditEntry) -> None:
        """Write entry to JSONL file and in-memory buffer."""
        self._entries.append(entry)
        if self._path:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_json_dict(), ensure_ascii=False))
                    f.write("\n")
            except OSError as exc:
                logger.warning("Failed to write audit entry: %s", exc)

    def update_outcome(self, decision_id: str, outcome: str,
                       reason: str = "", elapsed_ms: float = 0.0) -> None:
        """Update outcome by writing an update record."""
        update = {
            "type": "outcome_update",
            "decision_id": decision_id,
            "outcome": outcome,
            "outcome_reason": reason,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
        }
        if self._path:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(update, ensure_ascii=False))
                    f.write("\n")
            except OSError as exc:
                logger.warning("Failed to write audit outcome: %s", exc)

    @property
    def entries(self) -> list[RecoveryAuditEntry]:
        """In-memory entries (for testing)."""
        return list(self._entries)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics from in-memory entries."""
        if not self._entries:
            return {"total": 0}
        total = len(self._entries)
        by_strategy: dict[str, int] = {}
        by_action: dict[str, int] = {}
        for e in self._entries:
            by_strategy[e.strategy_key] = by_strategy.get(e.strategy_key, 0) + 1
            by_action[e.action] = by_action.get(e.action, 0) + 1
        return {
            "total": total,
            "by_strategy": by_strategy,
            "by_action": by_action,
        }


def create_audit_entry(
    envelope: "FailureEnvelope",
    decision: "RecoveryDecision",
    budget: "RecoveryBudget",
    session_id: str = "",
    turn_id: int = 0,
) -> RecoveryAuditEntry:
    """Factory function to create an audit entry from coordinator outputs.

    Handles attribute access safely for enum values and budget internals.
    """
    from leapflow.engine.failure_envelope import FailureEnvelope  # noqa: F811
    from leapflow.engine.recovery_decision import RecoveryDecision  # noqa: F811
    from leapflow.engine.recovery_budget import RecoveryBudget  # noqa: F811

    return RecoveryAuditEntry(
        timestamp=time.time(),
        session_id=session_id,
        turn_id=turn_id,
        envelope_id=envelope.envelope_id,
        failure_source=envelope.source.value if hasattr(envelope.source, 'value') else str(envelope.source),
        failure_category=envelope.category,
        failure_code=envelope.failure_code,
        recoverability=envelope.recoverability.value if hasattr(envelope.recoverability, 'value') else str(envelope.recoverability),
        decision_id=decision.decision_id,
        strategy_key=decision.strategy_key,
        action=decision.action.value if hasattr(decision.action, 'value') else str(decision.action),
        reason=decision.reason,
        budget_cost=decision.budget_cost,
        budget_consumed=budget._consumed if hasattr(budget, '_consumed') else 0,
        budget_remaining=budget.remaining() if hasattr(budget, 'remaining') else 0,
    )

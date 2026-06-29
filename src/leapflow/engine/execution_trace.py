"""Execution state machine and trace recording for the agent loop."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionMode(Enum):
    """States of the agent execution loop state machine."""
    PREPARING = "preparing"
    REASONING = "reasoning"
    ROUTING = "routing"
    ACTING = "acting"
    OBSERVING = "observing"
    RECOVERING = "recovering"
    COMPLETE = "complete"


@dataclass
class TraceEntry:
    """A single step in the execution trace."""
    state: ExecutionMode
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class ExecutionTrace:
    """Records the execution trajectory for learning signal emission."""
    entries: List[TraceEntry] = field(default_factory=list)

    def record(
        self,
        state: ExecutionMode,
        *,
        action: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Append a trace entry (sync, <10us)."""
        self.entries.append(TraceEntry(
            state=state,
            action=action,
            observation=observation,
            error=error,
            timestamp=time.time(),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        ))

    @property
    def has_learning_signal(self) -> bool:
        """Whether this trace contains enough signal for the evolution ring."""
        acting_count = sum(1 for e in self.entries if e.state == ExecutionMode.ACTING)
        return acting_count >= 2

    @property
    def step_count(self) -> int:
        return len(self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.tokens_used for e in self.entries)

    @property
    def success(self) -> bool:
        """Whether the loop completed successfully (vs error/exhaustion)."""
        if not self.entries:
            return False
        last = self.entries[-1]
        return last.state == ExecutionMode.COMPLETE and last.error is None

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging."""
        return {
            "steps": self.step_count,
            "tokens": self.total_tokens,
            "success": self.success,
            "actions": sum(1 for e in self.entries if e.state == ExecutionMode.ACTING),
            "errors": sum(1 for e in self.entries if e.error),
        }

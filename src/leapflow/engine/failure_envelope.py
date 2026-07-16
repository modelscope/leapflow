"""Structured failure representation for the recovery subsystem.

FailureEnvelope wraps every failure encountered in the agent loop with
enough metadata to drive automated recovery decisions: source, category,
recoverability, side-effect state, and context.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureSource(Enum):
    """Origin subsystem of a failure."""

    LLM = "llm"
    TOOL = "tool"
    SYSTEM = "system"
    SECURITY = "security"


class Recoverability(Enum):
    """How a failure can potentially be recovered."""

    AUTO_RETRY = "auto_retry"
    AUTO_RECOVER = "auto_recover"
    USER_FIXABLE = "user_fixable"
    ADMIN_REQUIRED = "admin_required"
    NON_RECOVERABLE = "non_recoverable"


class SideEffectState(Enum):
    """Observed side-effect state at the point of failure."""

    NONE = "none"
    COMMITTED = "committed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureContext:
    """Execution context at the moment of failure.

    Uses tuple of pairs for arguments to remain hashable in a frozen dataclass.
    """

    tool_name: str = ""
    arguments: tuple[tuple[str, Any], ...] = ()
    execution_id: str = ""
    trace_id: str = ""
    turn_id: int = 0
    session_id: str = ""
    provider: str = ""
    model: str = ""
    attempt_number: int = 0
    elapsed_ms: float = 0.0

    @classmethod
    def from_dict_args(
        cls,
        *,
        tool_name: str = "",
        arguments: dict[str, Any] | None = None,
        execution_id: str = "",
        trace_id: str = "",
        turn_id: int = 0,
        session_id: str = "",
        provider: str = "",
        model: str = "",
        attempt_number: int = 0,
        elapsed_ms: float = 0.0,
    ) -> FailureContext:
        """Factory that converts a dict of arguments into frozen-compatible tuple form."""
        args_tuple = tuple(sorted((arguments or {}).items()))
        return cls(
            tool_name=tool_name,
            arguments=args_tuple,
            execution_id=execution_id,
            trace_id=trace_id,
            turn_id=turn_id,
            session_id=session_id,
            provider=provider,
            model=model,
            attempt_number=attempt_number,
            elapsed_ms=elapsed_ms,
        )

    @property
    def arguments_dict(self) -> dict[str, Any]:
        """Reconstruct arguments as a dict for downstream consumers."""
        return dict(self.arguments)


@dataclass(frozen=True)
class RecoveryHint:
    """Provider- or system-generated hint for how to resolve a failure."""

    hint_text: str = ""
    suggested_command: str = ""
    documentation_url: str = ""


@dataclass(frozen=True)
class FailureEnvelope:
    """Immutable, self-describing failure record for recovery coordination.

    Every failure in the agent loop is wrapped in this envelope before being
    passed to the RecoveryCoordinator for decision-making.
    """

    envelope_id: str
    source: FailureSource
    category: str
    failure_class: str
    failure_code: str
    message: str
    recoverability: Recoverability
    side_effect_state: SideEffectState
    context: FailureContext
    timestamp: float
    provider_hint: RecoveryHint | None = None

    @classmethod
    def create(
        cls,
        *,
        source: FailureSource,
        category: str,
        failure_class: str,
        failure_code: str,
        message: str,
        recoverability: Recoverability,
        side_effect_state: SideEffectState = SideEffectState.NONE,
        context: FailureContext | None = None,
        provider_hint: RecoveryHint | None = None,
    ) -> FailureEnvelope:
        """Convenience factory that auto-generates envelope_id and timestamp."""
        return cls(
            envelope_id=uuid.uuid4().hex,
            source=source,
            category=category,
            failure_class=failure_class,
            failure_code=failure_code,
            message=message,
            recoverability=recoverability,
            side_effect_state=side_effect_state,
            context=context or FailureContext(),
            timestamp=time.time(),
            provider_hint=provider_hint,
        )

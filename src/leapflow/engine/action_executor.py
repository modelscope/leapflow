# Copyright (c) Alibaba, Inc. and its affiliates.
"""Single execution boundary for durable, no-LLM action evidence."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from leapflow.domain.evolution_event import (
    ActionEvidenceUnavailable,
    EvolutionContext,
    EvolutionEvent,
)
from leapflow.engine.tool_execution import ExecutionPolicy

logger = logging.getLogger(__name__)

ActionOperation = Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class ActionInvocation:
    """Complete execution identity passed to the shared action boundary."""

    action_type: str
    action_name: str
    arguments: Mapping[str, Any]
    execution_id: str
    execution_policy: ExecutionPolicy
    context: EvolutionContext
    goal: str = ""

    @property
    def requires_durable_start(self) -> bool:
        """Every mutation must be evidenced before its side effect can begin."""
        return self.execution_policy != "read_only"


@runtime_checkable
class ActionRecorderPort(Protocol):
    """Recorder contract kept independent from a concrete event transport."""

    async def started(
        self,
        *,
        context: EvolutionContext,
        action_type: str,
        action_name: str,
        arguments: Mapping[str, Any] | None,
        execution_policy: str,
        critical: bool,
        goal: str = "",
        occurred_at: float | None = None,
    ) -> EvolutionEvent: ...

    async def completed(
        self,
        *,
        context: EvolutionContext,
        started_event: EvolutionEvent,
        action_type: str,
        action_name: str,
        result: Any,
        duration_ms: float,
        occurred_at: float | None = None,
        critical: bool = False,
    ) -> EvolutionEvent: ...

    async def failed_exception(
        self,
        *,
        context: EvolutionContext,
        started_event: EvolutionEvent,
        action_type: str,
        action_name: str,
        error: BaseException,
        duration_ms: float,
        critical: bool = False,
    ) -> EvolutionEvent: ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Execute one action through the system's single evidence boundary."""

    async def execute(self, invocation: ActionInvocation, operation: ActionOperation) -> Any: ...


class RecordedActionExecutor:
    """Record action lifecycle facts without adding an LLM call to the hot path."""

    def __init__(self, recorder: ActionRecorderPort | None) -> None:
        self._recorder = recorder

    async def execute(self, invocation: ActionInvocation, operation: ActionOperation) -> Any:
        recorder = self._recorder
        if recorder is None:
            return await operation()

        started_at = time.perf_counter()
        try:
            started = await recorder.started(
                context=invocation.context,
                action_type=invocation.action_type,
                action_name=invocation.action_name,
                arguments=invocation.arguments,
                execution_policy=invocation.execution_policy,
                critical=invocation.requires_durable_start,
                goal=invocation.goal,
            )
        except Exception as exc:
            if invocation.requires_durable_start:
                raise ActionEvidenceUnavailable(
                    "mutating action refused because its audit start could not be persisted"
                ) from exc
            logger.warning("action evidence start unavailable", exc_info=True)
            return await operation()

        try:
            result = await operation()
        except Exception as exc:
            try:
                await recorder.failed_exception(
                    context=invocation.context,
                    started_event=started,
                    action_type=invocation.action_type,
                    action_name=invocation.action_name,
                    error=exc,
                    duration_ms=(time.perf_counter() - started_at) * 1000.0,
                    critical=invocation.requires_durable_start,
                )
            except Exception:
                logger.error("action failure evidence could not be persisted", exc_info=True)
            raise

        try:
            await recorder.completed(
                context=invocation.context,
                started_event=started,
                action_type=invocation.action_type,
                action_name=invocation.action_name,
                result=result,
                duration_ms=(time.perf_counter() - started_at) * 1000.0,
                critical=invocation.requires_durable_start,
            )
        except Exception:
            logger.error("action completion evidence could not be persisted", exc_info=True)
            if invocation.requires_durable_start and isinstance(result, dict):
                result = {
                    **result,
                    "audit_incomplete": True,
                    "side_effect_uncertain": True,
                }
        return result


__all__ = [
    "ActionExecutor",
    "ActionInvocation",
    "ActionOperation",
    "ActionRecorderPort",
    "RecordedActionExecutor",
]

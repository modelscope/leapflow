# Copyright (c) Alibaba, Inc. and its affiliates.
"""O(1), no-LLM evidence recorder for every agent action.

This recorder replaces ``PredictionLoop`` as the universal execution evidence
boundary. It does not predict, compare semantically, capture screenshots, or call a
model. Rich evaluation belongs to the session-finalization cold path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import (
    ActionEvidenceUnavailable,
    EvolutionContext,
    EvolutionEvent,
    content_hash,
)
from leapflow.engine.tools.tool_execution import exit_code_from
from leapflow.performance import LatencySummary, RollingLatency
from leapflow.security.redact import redact_sensitive_text

_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|secret|token|password|credential|authorization|cookie)", re.I
)
_MAX_DEPTH = 4
_MAX_ITEMS = 32
_MAX_TEXT = 1000


@dataclass(frozen=True)
class ActionRecorderMetrics:
    started_latency: LatencySummary
    completed_latency: LatencySummary


class ActionRecorder:
    """Emit immutable action lifecycle facts through an evolution outbox."""

    def __init__(self, outbox: Any, *, producer_version: str = "") -> None:
        self._outbox = outbox
        self._producer_version = str(producer_version)
        self._started_latency = RollingLatency()
        self._completed_latency = RollingLatency()

    @property
    def metrics(self) -> ActionRecorderMetrics:
        return ActionRecorderMetrics(
            started_latency=self._started_latency.snapshot(),
            completed_latency=self._completed_latency.snapshot(),
        )

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
    ) -> EvolutionEvent:
        """Record action start; critical side effects cross a durable barrier."""
        started_at = perf_counter()
        try:
            sanitized = sanitize_evidence(arguments or {})
            event = EvolutionEvent.create(
                EvolutionEventType.ACTION_STARTED,
                context=context,
                payload={
                    "action_type": str(action_type),
                    "action_name": str(action_name),
                    "execution_policy": str(execution_policy),
                    "argument_names": sorted(str(key) for key in (arguments or {})),
                    "arguments": sanitized,
                    "arguments_hash": content_hash(sanitized),
                    "goal": sanitize_evidence(goal),
                    "critical": bool(critical),
                },
                producer="agent.action_recorder",
                producer_version=self._producer_version,
                privacy_class="session",
                occurred_at=occurred_at,
                dedup_key=f"action.started:{context.action_id}",
            )
            await self._outbox.publish(event, critical=critical)
            return event
        finally:
            self._started_latency.observe((perf_counter() - started_at) * 1000.0)

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
    ) -> EvolutionEvent:
        """Record a successful or failed completion using the normalized result."""
        started_at = perf_counter()
        try:
            ok = bool(result.get("ok", True)) if isinstance(result, Mapping) else True
            event_type = (
                EvolutionEventType.ACTION_COMPLETED if ok else EvolutionEventType.ACTION_FAILED
            )
            sanitized_result = sanitize_evidence(result)
            payload: dict[str, Any] = {
                "action_type": str(action_type),
                "action_name": str(action_name),
                "ok": ok,
                "duration_ms": max(0.0, float(duration_ms)),
                "result": sanitized_result,
                "result_hash": content_hash(sanitized_result),
            }
            exit_code = exit_code_from(result)
            if exit_code is not None:
                payload["exit_code"] = exit_code
            if isinstance(result, Mapping):
                for key in (
                    "failure_code",
                    "retryable",
                    "execution_status",
                    "execution_policy",
                    "side_effect_uncertain",
                    "counts_as_failure",
                    "already_executed",
                    "duplicate_suppressed",
                ):
                    if key in result:
                        payload[key] = sanitize_evidence(result[key])
            event = EvolutionEvent.create(
                event_type,
                context=context.with_ids(causation_id=started_event.event_id),
                payload=payload,
                producer="agent.action_recorder",
                producer_version=self._producer_version,
                privacy_class="session",
                occurred_at=occurred_at,
                dedup_key=f"{event_type}:{context.action_id}",
            )
            await self._outbox.publish(event, critical=critical)
            return event
        finally:
            self._completed_latency.observe((perf_counter() - started_at) * 1000.0)

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
    ) -> EvolutionEvent:
        """Record an exception before the execution path re-raises it."""
        result = {
            "ok": False,
            "failure_code": type(error).__name__,
            "error": redact_sensitive_text(str(error), force=True),
            "retryable": True,
        }
        return await self.completed(
            context=context,
            started_event=started_event,
            action_type=action_type,
            action_name=action_name,
            result=result,
            duration_ms=duration_ms,
            critical=critical,
        )


def sanitize_evidence(value: Any, *, _depth: int = 0) -> Any:
    """Return a bounded, JSON-safe, secret-redacted evidence projection."""
    if _depth >= _MAX_DEPTH:
        return {"truncated": True, "type": type(value).__name__}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = redact_sensitive_text(value, force=True)
        return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:_MAX_ITEMS]:
            name = str(key)
            result[name] = (
                "[REDACTED]"
                if _SECRET_KEY.search(name)
                else sanitize_evidence(item, _depth=_depth + 1)
            )
        if len(items) > _MAX_ITEMS:
            result["items_omitted"] = len(items) - _MAX_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [sanitize_evidence(item, _depth=_depth + 1) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            result.append({"items_omitted": len(items) - _MAX_ITEMS})
        return result
    return sanitize_evidence(str(value), _depth=_depth + 1)


__all__ = [
    "ActionEvidenceUnavailable",
    "ActionRecorder",
    "ActionRecorderMetrics",
    "sanitize_evidence",
]

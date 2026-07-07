"""Per-turn usage tracking and cost estimation.

Accumulates token usage, latency, and tool call metrics across a single
agent turn. Emitted as structured audit events for observability.

Design:
- Immutable summary via frozen dataclass
- Mutable tracker reset per turn
- Provider-aware (tracks which provider served each call)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnUsageSummary:
    """Immutable snapshot of a completed turn's resource usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    compression_applied: bool = False
    provider_name: str = ""
    model: str = ""


@dataclass
class _ToolCallRecord:
    name: str
    ok: bool
    duration_ms: float


class TurnUsageTracker:
    """Mutable per-turn usage accumulator.

    Usage:
        tracker = TurnUsageTracker()
        tracker.record_api_call(resp.usage, provider="primary")
        tracker.record_tool_call("shell_run", True, 150.0)
        summary = tracker.summary()
        tracker.reset()
    """

    def __init__(self) -> None:
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: int = 0
        self._api_calls: int = 0
        self._tool_records: List[_ToolCallRecord] = []
        self._compression_applied: bool = False
        self._provider_name: str = ""
        self._model: str = ""

    def record_api_call(
        self,
        usage: Dict[str, int],
        *,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Accumulate usage from an LLM API response."""
        self._api_calls += 1
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)
        self._total_tokens += usage.get("total_tokens", 0)
        self._total_latency_ms += usage.get("latency_ms", 0)
        if provider:
            self._provider_name = provider
        if model:
            self._model = model

    def record_tool_call(
        self, name: str, success: bool, duration_ms: float
    ) -> None:
        """Record a single tool execution."""
        self._tool_records.append(_ToolCallRecord(name, success, duration_ms))

    def mark_compression(self) -> None:
        self._compression_applied = True

    def summary(self) -> TurnUsageSummary:
        """Build immutable summary of accumulated usage."""
        return TurnUsageSummary(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=self._total_tokens,
            latency_ms=self._total_latency_ms,
            api_calls=self._api_calls,
            tool_calls=len(self._tool_records),
            tool_successes=sum(1 for r in self._tool_records if r.ok),
            tool_failures=sum(1 for r in self._tool_records if not r.ok),
            compression_applied=self._compression_applied,
            provider_name=self._provider_name,
            model=self._model,
        )

    def reset(self) -> None:
        """Reset for next turn."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        self._api_calls = 0
        self._tool_records.clear()
        self._compression_applied = False

    def format_log_line(self) -> str:
        """One-line summary for structured logging."""
        s = self.summary()
        return (
            f"tokens={s.total_tokens} "
            f"(prompt={s.prompt_tokens} completion={s.completion_tokens}) "
            f"api_calls={s.api_calls} tools={s.tool_calls} "
            f"(ok={s.tool_successes} fail={s.tool_failures}) "
            f"latency={s.latency_ms}ms provider={s.provider_name}"
        )

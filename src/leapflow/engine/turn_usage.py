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
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnUsageSummary:
    """Immutable snapshot of a completed turn's resource usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    api_calls: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    compression_applied: bool = False
    provider_name: str = ""
    model: str = ""

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from the provider prefix cache."""
        return round(self.cached_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def effective_prompt_tokens(self, cached_price_ratio: float = 0.1) -> float:
        """Prompt tokens weighted by cache pricing (cached reads are cheaper).

        ``cached_price_ratio`` is the price of a cached-read token relative to an
        uncached (miss) token; providers typically bill cached reads at ~0.1x.
        Used by cost accounting so a well-cached long task costs less per turn.
        """
        miss = max(0, self.prompt_tokens - self.cached_tokens)
        return round(miss + self.cached_tokens * max(0.0, cached_price_ratio), 2)


def cost_ceiling_exceeded(
    *,
    effective_prompt_tokens: float,
    context_length: int,
    context_multiple: float,
) -> bool:
    """Whether cumulative effective prompt cost has crossed the turn ceiling.

    The ceiling is ``context_length * context_multiple`` effective prompt tokens
    accumulated across the turn. ``context_multiple <= 0`` disables it (the
    elastic iteration cap remains the hard bound). Intended as a *soft* safety:
    callers nudge finalization rather than hard-stopping, so no work is lost.
    """
    if context_multiple <= 0 or context_length <= 0:
        return False
    return effective_prompt_tokens >= context_length * context_multiple


def build_adaptive_learning_signal(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compact adaptive-depth orient snapshot for S3 calibration (observe-only).

    Derives, from the turn's last context snapshot, the predicted difficulty /
    posture / commitment so offline analysis (S3-L2) can relate them to the
    recorded outcome and effort and calibrate the difficulty weights and
    posture/commitment thresholds. Purely derived; never changes behavior.
    """
    snap = snapshot or {}
    signal: Dict[str, Any] = {
        "final_difficulty": round(float(snap.get("difficulty", 0.0) or 0.0), 4),
        "final_posture": str(snap.get("context_posture", "") or ""),
        "prefix_committed": bool(snap.get("prefix_committed", False)),
    }
    open_questions = snap.get("open_questions", None)
    if open_questions is not None:
        signal["open_questions"] = open_questions
    effective = snap.get("cumulative_effective_tokens", None)
    if effective:
        signal["cumulative_effective_tokens"] = effective
    return signal


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
        self._cached_tokens: int = 0
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
        self._cached_tokens += usage.get("cached_tokens", 0)
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
            cached_tokens=self._cached_tokens,
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
        self._cached_tokens = 0
        self._total_latency_ms = 0
        self._api_calls = 0
        self._tool_records.clear()
        self._compression_applied = False

    def to_learning_signal(self) -> Dict[str, Any]:
        """Structured signal for evolution episode context.

        Returns a lightweight dict describing runtime difficulty: retries,
        failovers, compressions, tool failure rates, and latency. The
        evolution pipeline can use these to identify "hard" action patterns
        and allocate attention/replay accordingly.
        """
        s = self.summary()
        return {
            "api_retries": max(0, s.api_calls - 1),
            "compression_applied": s.compression_applied,
            "tool_failure_rate": round(s.tool_failures / max(s.tool_calls, 1), 3),
            "total_latency_ms": s.latency_ms,
            "total_tokens": s.total_tokens,
            "cache_hit_rate": s.cache_hit_rate,
        }

    def format_log_line(self) -> str:
        """One-line summary for structured logging."""
        s = self.summary()
        return (
            f"tokens={s.total_tokens} "
            f"(prompt={s.prompt_tokens} completion={s.completion_tokens}) "
            f"cache_hit={s.cache_hit_rate:.0%} "
            f"api_calls={s.api_calls} tools={s.tool_calls} "
            f"(ok={s.tool_successes} fail={s.tool_failures}) "
            f"latency={s.latency_ms}ms provider={s.provider_name}"
        )

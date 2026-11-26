# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-turn and session-level usage tracking and cost estimation.

Accumulates token usage, latency, and tool call metrics across a single
agent turn. Emitted as structured audit events for observability.

Design:
- Immutable summary via frozen dataclass
- Mutable tracker reset per turn, session-level accumulators survive reset
- Provider-aware (tracks which provider served each call)
- Dual-caliber cache hit rate: per-turn average and token-weighted cumulative
  (aligned with DeepSeek ecosystem reporting)
"""
from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from leapflow.engine.tools.tool_execution import (
    TIER_FAILED,
    TIER_REFUSED,
    TIER_REPAIRED,
    TIER_SUCCEEDED,
)

from leapflow.domain.events import RecordAdmission, RecordScope
from leapflow.llm.services.telemetry import InferenceAttempt, admitted_record_value, freeze_usage, usage_totals

logger = logging.getLogger(__name__)

# Default number of initial turns excluded from steady-state metrics.
# Cold-start turns have low cache hit rates because the provider's prefix
# cache has not been populated yet.  Configurable via TurnUsageTracker.
DEFAULT_STEADY_STATE_SKIP_TURNS: int = 3


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
    # Graded apart from ``tool_failures`` (see ``tool_execution.result_tier``): a
    # refusal changed nothing and a repair was a reshaped call, not a failure.
    tool_refusals: int = 0
    tool_repairs: int = 0
    compression_applied: bool = False
    provider_name: str = ""
    model: str = ""

    @property
    def cache_hit_rate(self) -> float:
        """Per-turn cache hit rate: cached_tokens / prompt_tokens.

        This is the per-turn caliber — the ratio for a single turn's API
        calls.  For session-level token-weighted cumulative rates (aligned
        with the DeepSeek ecosystem), use ``SessionCacheStats``.
        """
        return round(self.cached_tokens / self.prompt_tokens, 4) if self.prompt_tokens else 0.0

    def effective_prompt_tokens(self, cached_price_ratio: float = 0.1) -> float:
        """Prompt tokens weighted by cache pricing (cached reads are cheaper).

        ``cached_price_ratio`` is the price of a cached-read token relative to an
        uncached (miss) token; providers typically bill cached reads at ~0.1x.
        Used by cost accounting so a well-cached long task costs less per turn.
        """
        miss = max(0, self.prompt_tokens - self.cached_tokens)
        return round(miss + self.cached_tokens * max(0.0, cached_price_ratio), 2)


@dataclass(frozen=True)
class SessionCacheStats:
    """Session-level cache hit rate statistics with dual-caliber support.

    Provides both the **token-weighted cumulative** rate (``Σcached / Σprompt``,
    comparable to DeepSeek ecosystem reporting) and a **steady-state** rate
    that excludes the first *N* cold-start turns where the provider's prefix
    cache has not yet been populated.

    Instances are obtained via ``TurnUsageTracker.session_cache_stats()``.
    """

    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    steady_prompt_tokens: int = 0
    steady_cached_tokens: int = 0
    completed_turns: int = 0
    steady_state_skip_turns: int = DEFAULT_STEADY_STATE_SKIP_TURNS
    per_turn_rates: Tuple[float, ...] = ()

    @property
    def token_weighted_hit_rate(self) -> float:
        """Token-weighted cumulative cache hit rate (DeepSeek-comparable).

        ``Σcached_tokens / Σprompt_tokens`` across all turns in the session.
        """
        if self.total_prompt_tokens <= 0:
            return 0.0
        return round(self.total_cached_tokens / self.total_prompt_tokens, 4)

    @property
    def steady_state_hit_rate(self) -> float:
        """Token-weighted cache hit rate excluding the first *N* cold-start turns."""
        if self.steady_prompt_tokens <= 0:
            return 0.0
        return round(self.steady_cached_tokens / self.steady_prompt_tokens, 4)

    @property
    def per_turn_average_hit_rate(self) -> float:
        """Arithmetic mean of per-turn cache hit rates.

        This is the legacy caliber (``mean(cached_i / prompt_i)``).  It
        under-weights high-token turns and over-weights early low-token turns.
        """
        if not self.per_turn_rates:
            return 0.0
        return round(sum(self.per_turn_rates) / len(self.per_turn_rates), 4)


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
    tier: str = TIER_SUCCEEDED


class TurnUsageTracker:
    """Mutable per-turn usage accumulator with session-level cache stats.

    Per-turn counters are reset each turn via ``reset()``.  Session-level
    cumulative counters survive resets and power the dual-caliber cache
    hit rate reporting (per-turn average vs token-weighted cumulative).

    Usage:
        tracker = TurnUsageTracker()
        tracker.record_api_call(resp.usage, provider="primary")
        tracker.record_tool_call("shell_run", True, 150.0)
        summary = tracker.summary()
        stats = tracker.session_cache_stats()  # dual-caliber snapshot
        tracker.reset()
    """

    def __init__(
        self,
        *,
        steady_state_skip_turns: int = DEFAULT_STEADY_STATE_SKIP_TURNS,
        max_feedback_records: int = 32,
    ) -> None:
        # ── Per-turn (reset each turn) ──
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
        self._plugin_stats_sink: Optional[Any] = None
        self._inference_records: dict[str, InferenceAttempt] = {}
        # Only admitted, redacted receipts enter retrievable history. Full records
        # live in the owning transcript; this deque is an explicitly bounded view.
        self._collaboration_history: deque[RecordAdmission] = deque(maxlen=max(0, max_feedback_records))
        self._plan_history: deque[RecordAdmission] = deque(maxlen=max(0, max_feedback_records))

        # ── Session-level (survive reset) ──
        self._steady_state_skip_turns: int = max(0, steady_state_skip_turns)
        self._turn_index: int = 0  # current turn number (0-based)
        self._session_prompt_tokens: int = 0
        self._session_cached_tokens: int = 0
        self._steady_prompt_tokens: int = 0
        self._steady_cached_tokens: int = 0
        self._per_turn_rates: List[float] = []

    def record_api_call(
        self,
        usage: Dict[str, int],
        *,
        provider: str = "",
        model: str = "",
        call_id: str = "",
    ) -> None:
        """Accumulate usage from an LLM API response.

        Handles provider-specific usage semantics:
        - **OpenAI/DeepSeek**: ``prompt_tokens`` is the full prompt token count
          (including cached reads).  ``cached_tokens`` ≤ ``prompt_tokens``.
        - **Anthropic**: ``prompt_tokens`` = ``input_tokens`` which *excludes*
          cache reads/writes.  The Anthropic provider preserves the original
          ``cache_read_input_tokens`` and ``cache_creation_input_tokens`` keys.
          When detected, the effective prompt denominator is recomputed as
          ``input_tokens + cache_read + cache_creation`` so that
          ``cached / effective_prompt ≤ 1.0``.

        Detection is structural (key existence), not provider-name matching.
        """
        call_id = call_id or uuid.uuid4().hex
        if any(record.call_id == call_id for record in self._inference_records.values()):
            return
        self.record_inference(InferenceAttempt(
            call_id=call_id, attempt_id=call_id, source="primary", provider=provider, model=model,
            status="returned", elapsed_ms=float(usage.get("latency_ms", 0) or 0),
            usage_json=freeze_usage(usage), accounting_level="call_boundary",
        ))

    def _accumulate_primary(self, usage: Dict[str, Any], *, provider: str, model: str) -> None:
        """Primary cache counters remain separate from specialist compute usage."""
        self._api_calls += 1
        prompt = usage.get("prompt_tokens", 0)
        cached = usage.get("cached_tokens", 0)

        # Anthropic semantic adaptation: input_tokens excludes cache
        # reads/writes, so prompt_tokens alone understates the true prompt
        # consumption.  Recompute when Anthropic-specific keys are present.
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_create = usage.get("cache_creation_input_tokens", 0) or 0
        if cache_read or cache_create:
            prompt = prompt + cache_read + cache_create

        self._prompt_tokens += prompt
        self._completion_tokens += usage.get("completion_tokens", 0)
        self._total_tokens += usage.get("total_tokens", 0)
        self._cached_tokens += cached
        self._total_latency_ms += usage.get("latency_ms", 0)
        if provider:
            self._provider_name = provider
        if model:
            self._model = model

        # Session-level accumulation (O(1), cold-path safe)
        self._session_prompt_tokens += prompt
        self._session_cached_tokens += cached
        if self._turn_index >= self._steady_state_skip_turns:
            self._steady_prompt_tokens += prompt
            self._steady_cached_tokens += cached

    def record_inference(self, attempt: InferenceAttempt) -> None:
        """Count every unique attempt, keeping specialist usage out of primary counters."""
        if attempt.attempt_id in self._inference_records:
            return
        self._inference_records[attempt.attempt_id] = attempt
        if attempt.source == "primary":
            usage = json.loads(attempt.usage_json)
            usage.setdefault("latency_ms", int(attempt.elapsed_ms))
            self._accumulate_primary(usage, provider=attempt.provider, model=attempt.model)

    def inference_records(self) -> tuple[InferenceAttempt, ...]:
        return tuple(self._inference_records.values())

    def combined_usage(self) -> dict[str, Any]:
        return usage_totals(list(self._inference_records.values()))

    def record_collaboration(self, receipt: RecordAdmission) -> None:
        """Retain private signal receipts, never model-authored success claims."""
        if not isinstance(receipt, RecordAdmission):
            return
        histories = {"model.consultation": self._collaboration_history, "model.plan_link": self._plan_history}
        history = histories.get(receipt.event_type)
        if (history is not None
                and admitted_record_value(receipt, receipt.scope, event_type=receipt.event_type) is not None
                and receipt not in history):
            history.append(receipt)

    def consultation_reference(self, scope: RecordScope, task_id: str, call_id: str) -> dict[str, Any] | None:
        """Resolve an explicit reference only within the owning admitted task."""
        if not task_id or not scope.session_id or not scope.workspace_id or not scope.profile_id:
            return None
        for receipt in reversed(self._collaboration_history):
            value = admitted_record_value(receipt, scope, event_type="model.consultation", identity=call_id)
            if value is not None and value.get("task_id") == task_id:
                return value
        return None

    def restore_collaboration(self, metadata: dict[str, Any], scope: RecordScope) -> None:
        """Restore only previously admitted receipts from the owning transcript."""
        rows = metadata.get("collaboration_signal_receipts", [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if (not isinstance(row, dict) or row.get("admitted") is not True
                    or row.get("lineage_eligible") is not True or row.get("scope") != asdict(scope)
                    or row.get("event_type") not in {"model.consultation", "model.plan_link"}
                    or not isinstance(row.get("record_id"), str) or not row["record_id"]):
                continue
            receipt = RecordAdmission(True, str(row.get("reason") or ""), scope,
                row["event_type"], row.get("payload_json", "{}"))
            if admitted_record_value(receipt, scope, event_type=row["event_type"],
                                     identity=row.get("record_id", "")) is not None:
                self.record_collaboration(receipt)

    def collaboration_feedback(self, scope: RecordScope, *, service_id: str = "") -> dict[str, Any]:
        """Bounded caller-only validity/latency history, retrieved outside the prefix."""
        selected = []
        for receipt in self._collaboration_history:
            if receipt.scope != scope:
                continue
            value = admitted_record_value(receipt, scope, event_type="model.consultation")
            if value is None or service_id and value.get("service_id") != service_id:
                continue
            selected.append({key: value.get(key) for key in (
                "call_id", "service_id", "status", "failure_code", "elapsed_ms", "adoption")})
        call_ids = {item["call_id"] for item in selected}
        plans = []
        for receipt in self._plan_history:
            value = admitted_record_value(receipt, scope, event_type="model.plan_link")
            if value is None:
                continue
            references = value.get("references", [])
            if not isinstance(references, list):
                continue
            if service_id and not any(isinstance(ref, dict) and ref.get("call_id") in call_ids for ref in references):
                continue
            plans.append(value)
        return {"window": "bounded_session", "records": selected, "plan_links": plans,
                "eligible": sum(item.get("status") == "eligible" for item in selected),
                "noneligible": sum(item.get("status") != "eligible" for item in selected),
                "measured_task_outcome": "unverified", "physical_authority": False}

    def set_plugin_stats_sink(self, sink: Any) -> None:
        """Install a cross-turn stats accumulator. Receives all record_tool_call data."""
        self._plugin_stats_sink = sink

    def record_tool_call(
        self,
        name: str,
        success: bool,
        duration_ms: float,
        *,
        tier: str = "",
        failure_code: str = "",
    ) -> None:
        """Record a single tool execution and the result tier it was graded.

        Without a *tier* the call is graded by *success* alone.
        """
        tier = tier or (TIER_SUCCEEDED if success else TIER_FAILED)
        self._tool_records.append(_ToolCallRecord(name, success, duration_ms, tier))
        sink = self._plugin_stats_sink
        if sink is None:
            return
        # Telemetry must never fail a turn: a malformed or misbehaving stats sink
        # (e.g. one injected by a plugin, or leaked across tests) is contained and
        # logged rather than propagated into the agent loop.
        try:
            sink.record(name, success, duration_ms, tier=tier, failure_code=failure_code)
        except Exception:  # noqa: BLE001 - stats recording is telemetry, never a gate
            logger.debug("plugin stats sink.record failed", exc_info=True)

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
            tool_failures=sum(1 for r in self._tool_records if r.tier == TIER_FAILED),
            tool_refusals=sum(1 for r in self._tool_records if r.tier == TIER_REFUSED),
            tool_repairs=sum(1 for r in self._tool_records if r.tier == TIER_REPAIRED),
            compression_applied=self._compression_applied,
            provider_name=self._provider_name,
            model=self._model,
        )

    def reset(self) -> None:
        """Reset per-turn counters for next turn.

        Session-level accumulators are preserved.  The per-turn cache hit
        rate for the finishing turn is recorded into the session history
        before counters are cleared.
        """
        # Commit the finishing turn's per-turn rate before clearing
        if self._prompt_tokens > 0:
            self._per_turn_rates.append(
                round(self._cached_tokens / self._prompt_tokens, 4)
            )
        elif self._api_calls > 0:
            # API call(s) with zero prompt tokens — record 0.0
            self._per_turn_rates.append(0.0)
        # else: no API calls this turn — skip (avoid polluting rates)

        self._turn_index += 1

        # ── Per-turn reset ──
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._cached_tokens = 0
        self._total_latency_ms = 0
        self._api_calls = 0
        self._tool_records.clear()
        self._inference_records.clear()
        self._compression_applied = False

    def session_cache_stats(self) -> SessionCacheStats:
        """Snapshot of session-level cache hit rate statistics.

        Includes the **current** (in-progress) turn's data in the totals.
        Call after ``record_api_call()`` for up-to-date figures.
        """
        # Include the current (not-yet-reset) turn's rate in the sequence
        current_rates = list(self._per_turn_rates)
        if self._prompt_tokens > 0:
            current_rates.append(
                round(self._cached_tokens / self._prompt_tokens, 4)
            )
        elif self._api_calls > 0:
            current_rates.append(0.0)

        return SessionCacheStats(
            total_prompt_tokens=self._session_prompt_tokens,
            total_cached_tokens=self._session_cached_tokens,
            steady_prompt_tokens=self._steady_prompt_tokens,
            steady_cached_tokens=self._steady_cached_tokens,
            completed_turns=len(current_rates),
            steady_state_skip_turns=self._steady_state_skip_turns,
            per_turn_rates=tuple(current_rates),
        )

    def to_learning_signal(self) -> Dict[str, Any]:
        """Structured signal for evolution episode context.

        Returns a lightweight dict describing runtime difficulty: retries,
        failovers, compressions, tool failure rates, and latency. The
        evolution pipeline can use these to identify "hard" action patterns
        and allocate attention/replay accordingly.
        """
        s = self.summary()
        stats = self.session_cache_stats()
        return {
            "api_retries": max(0, s.api_calls - 1),
            "compression_applied": s.compression_applied,
            "tool_failure_rate": round(s.tool_failures / max(s.tool_calls, 1), 3),
            "tool_refusal_rate": round(s.tool_refusals / max(s.tool_calls, 1), 3),
            "tool_repair_rate": round(s.tool_repairs / max(s.tool_calls, 1), 3),
            "total_latency_ms": s.latency_ms,
            "total_tokens": s.total_tokens,
            "cache_hit_rate": s.cache_hit_rate,
            "cache_hit_rate_token_weighted": stats.token_weighted_hit_rate,
            "cache_hit_rate_steady_state": stats.steady_state_hit_rate,
            "inference_usage": self.combined_usage(),
        }

    def format_log_line(self) -> str:
        """One-line summary for structured logging (dual-caliber cache metrics)."""
        s = self.summary()
        stats = self.session_cache_stats()
        return (
            f"tokens={s.total_tokens} "
            f"(prompt={s.prompt_tokens} completion={s.completion_tokens}) "
            f"cache_hit={s.cache_hit_rate:.0%} "
            f"[session: tw={stats.token_weighted_hit_rate:.0%} "
            f"steady={stats.steady_state_hit_rate:.0%}] "
            f"api_calls={s.api_calls} tools={s.tool_calls} "
            f"(ok={s.tool_successes} fail={s.tool_failures} "
            f"refused={s.tool_refusals} repaired={s.tool_repairs}) "
            f"latency={s.latency_ms}ms provider={s.provider_name}"
        )

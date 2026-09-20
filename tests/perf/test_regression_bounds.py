# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deterministic performance regression bounds for key hot/cold-path operations.

These tests assert upper bounds on cheap, deterministic operations so that a
gross algorithmic regression (e.g. O(1) -> O(n²)) is caught by CI without
needing non-deterministic benchmark infrastructure.

Rules:
- No network, no LLM, no disk I/O in the timed section.
- Bounds are generous (10x–100x headroom) so CI jitter does not flake.
- Each test runs a representative workload and asserts wall-clock < threshold.
"""
from __future__ import annotations

import time

from leapflow.engine.cost_calculator import compute_cost
from leapflow.engine.turn_usage import (
    TurnUsageSummary,
    TurnUsageTracker,
    cost_ceiling_exceeded,
)
from leapflow.performance import RollingLatency, aggregate_latency_snapshots


# ═══════════════════════════════════════════════════════════════
#  RollingLatency.observe() — must stay O(1)
# ═══════════════════════════════════════════════════════════════


class TestRollingLatencyBounds:
    """RollingLatency.observe() is O(1); 10k observations must finish fast."""

    def test_observe_10k_under_50ms(self) -> None:
        rl = RollingLatency(capacity=2048)
        start = time.perf_counter()
        for i in range(10_000):
            rl.observe(float(i))
        elapsed_ms = (time.perf_counter() - start) * 1000
        # O(1) per append into a bounded deque — 10k should be well under 50ms
        assert elapsed_ms < 50, f"10k observe() took {elapsed_ms:.1f}ms (limit 50ms)"

    def test_snapshot_after_fill(self) -> None:
        """Snapshot (cold-path sort) on a full 2048-sample buffer."""
        rl = RollingLatency(capacity=2048)
        for i in range(2048):
            rl.observe(float(i % 100))
        start = time.perf_counter()
        snap = rl.snapshot()
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 2048-element sort is cheap; 10ms is generous
        assert elapsed_ms < 10, f"snapshot() took {elapsed_ms:.1f}ms (limit 10ms)"
        assert snap.count == 2048
        assert snap.p50_ms >= 0


# ═══════════════════════════════════════════════════════════════
#  TurnUsageTracker accumulation — must stay O(1) per call
# ═══════════════════════════════════════════════════════════════


class TestTurnUsageTrackerBounds:
    """Tracker accumulation and summary are O(1)."""

    def test_record_api_call_1k_under_20ms(self) -> None:
        tracker = TurnUsageTracker(steady_state_skip_turns=3)
        usage = {
            "prompt_tokens": 5000,
            "completion_tokens": 800,
            "total_tokens": 5800,
            "cached_tokens": 3000,
        }
        start = time.perf_counter()
        for _ in range(1_000):
            tracker.record_api_call(usage, provider="test", model="test-model")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"1k record_api_call() took {elapsed_ms:.1f}ms (limit 20ms)"

    def test_summary_is_instant(self) -> None:
        tracker = TurnUsageTracker()
        tracker.record_api_call({"prompt_tokens": 1000, "completion_tokens": 200})
        start = time.perf_counter()
        for _ in range(1_000):
            tracker.summary()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, f"1k summary() took {elapsed_ms:.1f}ms (limit 10ms)"

    def test_session_cache_stats_bounded(self) -> None:
        """Session cache stats after 100 turns stays fast."""
        tracker = TurnUsageTracker(steady_state_skip_turns=3)
        for _ in range(100):
            tracker.record_api_call({"prompt_tokens": 5000, "cached_tokens": 3000})
            tracker.reset()

        start = time.perf_counter()
        stats = tracker.session_cache_stats()
        elapsed_ms = (time.perf_counter() - start) * 1000
        # stats involves copying per_turn_rates (100 floats) — trivially fast
        assert elapsed_ms < 5, f"session_cache_stats() took {elapsed_ms:.1f}ms (limit 5ms)"
        assert stats.completed_turns == 100

    def test_effective_prompt_tokens_under_1ms(self) -> None:
        summary = TurnUsageSummary(prompt_tokens=100_000, cached_tokens=80_000)
        start = time.perf_counter()
        for _ in range(10_000):
            summary.effective_prompt_tokens(cached_price_ratio=0.1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, f"10k effective_prompt_tokens() took {elapsed_ms:.1f}ms (limit 10ms)"


# ═══════════════════════════════════════════════════════════════
#  cost_ceiling_exceeded — pure arithmetic, must be instant
# ═══════════════════════════════════════════════════════════════


class TestCostCeilingBounds:
    def test_ceiling_check_10k_under_5ms(self) -> None:
        start = time.perf_counter()
        for i in range(10_000):
            cost_ceiling_exceeded(
                effective_prompt_tokens=float(i * 100),
                context_length=128_000,
                context_multiple=2.0,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5, f"10k ceiling checks took {elapsed_ms:.1f}ms (limit 5ms)"


# ═══════════════════════════════════════════════════════════════
#  compute_cost — cold-path, but should still be < 1ms per call
# ═══════════════════════════════════════════════════════════════


PRICING_CONFIG = {
    "deepseek-chat": {
        "input_per_mtok": 0.27,
        "output_per_mtok": 1.10,
        "cached_input_ratio": 0.1,
    },
    "gpt-4o": {
        "input_per_mtok": 2.50,
        "output_per_mtok": 10.00,
        "cached_input_ratio": 0.5,
    },
}


class TestComputeCostBounds:
    def test_compute_cost_1k_under_20ms(self) -> None:
        start = time.perf_counter()
        for _ in range(1_000):
            result = compute_cost(
                prompt_tokens=500_000,
                completion_tokens=100_000,
                cached_tokens=200_000,
                model="deepseek-chat",
                pricing_config=PRICING_CONFIG,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"1k compute_cost() took {elapsed_ms:.1f}ms (limit 20ms)"
        assert result.known

    def test_compute_cost_missing_model_still_fast(self) -> None:
        """Missing pricing should short-circuit quickly."""
        start = time.perf_counter()
        for _ in range(1_000):
            result = compute_cost(
                prompt_tokens=500_000,
                completion_tokens=100_000,
                cached_tokens=200_000,
                model="unknown-model",
                pricing_config=PRICING_CONFIG,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 20, f"1k compute_cost(missing) took {elapsed_ms:.1f}ms (limit 20ms)"
        assert not result.known


# ═══════════════════════════════════════════════════════════════
#  Latency aggregation — pure read, must be fast
# ═══════════════════════════════════════════════════════════════


class TestAggregationBounds:
    def test_aggregate_20_snapshots_under_5ms(self) -> None:
        """Aggregating 20 named snapshots should be trivial."""
        samplers = {}
        for i in range(20):
            rl = RollingLatency(capacity=100)
            for j in range(100):
                rl.observe(float(j + i))
            samplers[f"component_{i}"] = rl.snapshot()

        start = time.perf_counter()
        result = aggregate_latency_snapshots(samplers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5, f"aggregate 20 snapshots took {elapsed_ms:.1f}ms (limit 5ms)"
        assert len(result) == 20

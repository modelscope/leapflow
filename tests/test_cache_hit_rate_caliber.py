# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for dual-caliber cache hit rate measurement (token-weighted vs per-turn avg).

Validates that TurnUsageTracker produces correct session-level statistics
aligned with the DeepSeek ecosystem's token-weighted cumulative caliber,
while preserving the legacy per-turn average caliber and supporting
cold-start / steady-state separation.

Reference data from temp/dev_hermes/design/34_deepseek_cache_hit_rate_analysis_and_plan.md:
  Report 33 (10 turns):
    per-turn avg  = 70.2%
    token-weighted = 76.7%  (Σcached=9088, Σprompt=11850)
    R6-10 steady   = 82.7%

See also: AGENTS.md § Session Engine is the Only Reporting Source.
"""
from __future__ import annotations

from leapflow.engine.turn_usage import (
    DEFAULT_STEADY_STATE_SKIP_TURNS,
    SessionCacheStats,
    TurnUsageTracker,
    TurnUsageSummary,
)

# ── Report 33 reference data (OpenAI path, 10 turns) ──────────────────────
REPORT_33_TURNS = [
    {"prompt_tokens": 317, "cached_tokens": 128, "completion_tokens": 30, "total_tokens": 347},
    {"prompt_tokens": 489, "cached_tokens": 256, "completion_tokens": 40, "total_tokens": 529},
    {"prompt_tokens": 688, "cached_tokens": 384, "completion_tokens": 50, "total_tokens": 738},
    {"prompt_tokens": 875, "cached_tokens": 640, "completion_tokens": 60, "total_tokens": 935},
    {"prompt_tokens": 1119, "cached_tokens": 768, "completion_tokens": 70, "total_tokens": 1189},
    {"prompt_tokens": 1280, "cached_tokens": 1024, "completion_tokens": 80, "total_tokens": 1360},
    {"prompt_tokens": 1447, "cached_tokens": 1152, "completion_tokens": 90, "total_tokens": 1537},
    {"prompt_tokens": 1627, "cached_tokens": 1408, "completion_tokens": 100, "total_tokens": 1727},
    {"prompt_tokens": 1881, "cached_tokens": 1536, "completion_tokens": 110, "total_tokens": 1991},
    {"prompt_tokens": 2127, "cached_tokens": 1792, "completion_tokens": 120, "total_tokens": 2247},
]


def _simulate_session(
    turns: list[dict[str, int]],
    *,
    steady_state_skip_turns: int = DEFAULT_STEADY_STATE_SKIP_TURNS,
) -> TurnUsageTracker:
    """Feed *turns* through a tracker, calling reset() between turns."""
    tracker = TurnUsageTracker(steady_state_skip_turns=steady_state_skip_turns)
    for i, usage in enumerate(turns):
        tracker.record_api_call(usage, provider="test", model="test-model")
        if i < len(turns) - 1:
            tracker.reset()
    return tracker


# ═══════════════════════════════════════════════════════════════════════════
#  Core: token-weighted ≠ per-turn average
# ═══════════════════════════════════════════════════════════════════════════

class TestDualCaliberDivergence:
    """Token-weighted cumulative rate diverges from per-turn average."""

    def test_report_33_token_weighted_vs_per_turn_avg(self) -> None:
        """Reproduce the Report 33 caliber gap: 70.2% (per-turn) vs 76.7% (tw)."""
        tracker = _simulate_session(REPORT_33_TURNS)
        stats = tracker.session_cache_stats()

        # Token-weighted: Σcached=9088, Σprompt=11850 → 76.69%
        assert stats.total_cached_tokens == 9088
        assert stats.total_prompt_tokens == 11850
        tw = stats.token_weighted_hit_rate
        assert 0.766 <= tw <= 0.768, f"expected ~0.767, got {tw}"

        # Per-turn average: mean of per-turn rates
        avg = stats.per_turn_average_hit_rate
        assert 0.700 <= avg <= 0.704, f"expected ~0.702, got {avg}"

        # The two calibers MUST differ
        assert tw != avg
        # Token-weighted is higher (large late turns dominate)
        assert tw > avg

    def test_uniform_turns_calibers_converge(self) -> None:
        """When all turns have identical ratios, both calibers agree."""
        turns = [
            {"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 10, "total_tokens": 110},
            {"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 10, "total_tokens": 110},
            {"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 10, "total_tokens": 110},
        ]
        tracker = _simulate_session(turns, steady_state_skip_turns=0)
        stats = tracker.session_cache_stats()
        assert stats.token_weighted_hit_rate == stats.per_turn_average_hit_rate == 0.8


# ═══════════════════════════════════════════════════════════════════════════
#  Cold-start / steady-state separation
# ═══════════════════════════════════════════════════════════════════════════

class TestSteadyStateSeparation:
    """Steady-state rate correctly excludes cold-start turns."""

    def test_default_skip_3_turns(self) -> None:
        """Default steady_state_skip_turns=3: turns 0-2 excluded."""
        tracker = _simulate_session(REPORT_33_TURNS)
        stats = tracker.session_cache_stats()

        # Steady-state = turns 3-9 (7 turns), token-weighted
        expected_steady_prompt = sum(t["prompt_tokens"] for t in REPORT_33_TURNS[3:])
        expected_steady_cached = sum(t["cached_tokens"] for t in REPORT_33_TURNS[3:])
        assert stats.steady_prompt_tokens == expected_steady_prompt
        assert stats.steady_cached_tokens == expected_steady_cached

        steady = stats.steady_state_hit_rate
        # R4-10 token-weighted ≈ 80.3% (from analysis doc)
        assert 0.80 <= steady <= 0.81, f"expected ~0.803, got {steady}"
        # Steady > overall (cold-start drags overall down)
        assert steady > stats.token_weighted_hit_rate

    def test_custom_skip_5_turns(self) -> None:
        """Skip first 5 turns: R6-R10 steady-state ≈ 82.7%."""
        tracker = _simulate_session(REPORT_33_TURNS, steady_state_skip_turns=5)
        stats = tracker.session_cache_stats()

        expected_steady_prompt = sum(t["prompt_tokens"] for t in REPORT_33_TURNS[5:])
        expected_steady_cached = sum(t["cached_tokens"] for t in REPORT_33_TURNS[5:])
        assert stats.steady_prompt_tokens == expected_steady_prompt
        assert stats.steady_cached_tokens == expected_steady_cached

        steady = stats.steady_state_hit_rate
        assert 0.826 <= steady <= 0.828, f"expected ~0.827, got {steady}"

    def test_skip_all_turns_returns_zero(self) -> None:
        """If skip_turns >= total turns, steady-state rate is 0.0."""
        tracker = _simulate_session(REPORT_33_TURNS, steady_state_skip_turns=100)
        stats = tracker.session_cache_stats()
        assert stats.steady_state_hit_rate == 0.0
        assert stats.steady_prompt_tokens == 0
        assert stats.steady_cached_tokens == 0

    def test_skip_zero_equals_overall(self) -> None:
        """skip_turns=0 means no cold-start exclusion, steady == overall."""
        tracker = _simulate_session(REPORT_33_TURNS, steady_state_skip_turns=0)
        stats = tracker.session_cache_stats()
        assert stats.steady_state_hit_rate == stats.token_weighted_hit_rate
        assert stats.steady_prompt_tokens == stats.total_prompt_tokens


# ═══════════════════════════════════════════════════════════════════════════
#  Edge cases and safety
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Boundary conditions: zero tokens, single turn, no API calls."""

    def test_zero_prompt_tokens_no_division_error(self) -> None:
        """Σprompt=0 must not raise ZeroDivisionError."""
        tracker = TurnUsageTracker()
        stats = tracker.session_cache_stats()
        assert stats.token_weighted_hit_rate == 0.0
        assert stats.steady_state_hit_rate == 0.0
        assert stats.per_turn_average_hit_rate == 0.0
        assert stats.completed_turns == 0

    def test_single_cold_start_turn(self) -> None:
        """Single turn with cached=0 (pure cold start)."""
        tracker = TurnUsageTracker()
        tracker.record_api_call(
            {"prompt_tokens": 500, "cached_tokens": 0, "completion_tokens": 50, "total_tokens": 550}
        )
        stats = tracker.session_cache_stats()
        assert stats.token_weighted_hit_rate == 0.0
        assert stats.per_turn_average_hit_rate == 0.0
        assert stats.completed_turns == 1
        # Turn 0 is cold-start, so no steady-state data
        assert stats.steady_state_hit_rate == 0.0

    def test_api_call_with_zero_prompt_records_zero_rate(self) -> None:
        """API call where prompt_tokens=0: rate recorded as 0.0, no crash."""
        tracker = TurnUsageTracker()
        tracker.record_api_call({"prompt_tokens": 0, "cached_tokens": 0, "total_tokens": 10})
        stats = tracker.session_cache_stats()
        assert stats.token_weighted_hit_rate == 0.0
        assert stats.per_turn_average_hit_rate == 0.0

    def test_session_cache_stats_is_frozen(self) -> None:
        """SessionCacheStats is immutable (frozen dataclass)."""
        stats = SessionCacheStats(total_prompt_tokens=100, total_cached_tokens=80)
        try:
            stats.total_prompt_tokens = 200  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass  # expected

    def test_turn_usage_summary_cache_hit_rate_unchanged(self) -> None:
        """Existing TurnUsageSummary.cache_hit_rate property is backward-compatible."""
        summary = TurnUsageSummary(prompt_tokens=1000, cached_tokens=800)
        assert summary.cache_hit_rate == 0.8
        # Zero prompt → 0.0
        assert TurnUsageSummary().cache_hit_rate == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  format_log_line dual-caliber output
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatLogLine:
    """format_log_line() must include both per-turn and session-level calibers."""

    def test_log_line_contains_dual_caliber(self) -> None:
        """Log line has per-turn cache_hit and session tw/steady markers."""
        tracker = _simulate_session(REPORT_33_TURNS[:5])
        line = tracker.format_log_line()
        assert "cache_hit=" in line
        assert "[session: tw=" in line
        assert "steady=" in line

    def test_log_line_first_turn(self) -> None:
        """First turn log line shows 0% steady (all turns are cold-start)."""
        tracker = TurnUsageTracker()
        tracker.record_api_call(
            {"prompt_tokens": 317, "cached_tokens": 128, "completion_tokens": 30, "total_tokens": 347}
        )
        line = tracker.format_log_line()
        assert "cache_hit=40%" in line
        assert "tw=40%" in line
        # Steady is 0% because turn 0 < DEFAULT_STEADY_STATE_SKIP_TURNS
        assert "steady=0%" in line


# ═══════════════════════════════════════════════════════════════════════════
#  to_learning_signal dual-caliber fields
# ═══════════════════════════════════════════════════════════════════════════

class TestLearningSignal:
    """to_learning_signal() exposes both calibers for evolution pipeline."""

    def test_signal_has_dual_caliber_keys(self) -> None:
        tracker = _simulate_session(REPORT_33_TURNS[:5])
        signal = tracker.to_learning_signal()
        # Legacy per-turn
        assert "cache_hit_rate" in signal
        # New token-weighted
        assert "cache_hit_rate_token_weighted" in signal
        assert "cache_hit_rate_steady_state" in signal

    def test_signal_values_match_stats(self) -> None:
        tracker = _simulate_session(REPORT_33_TURNS)
        signal = tracker.to_learning_signal()
        stats = tracker.session_cache_stats()
        assert signal["cache_hit_rate_token_weighted"] == stats.token_weighted_hit_rate
        assert signal["cache_hit_rate_steady_state"] == stats.steady_state_hit_rate
        # Legacy key = current turn's per-turn rate
        summary = tracker.summary()
        assert signal["cache_hit_rate"] == summary.cache_hit_rate


# ═══════════════════════════════════════════════════════════════════════════
#  Session-level accumulator isolation (no cross-turn pollution)
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionAccumulatorIsolation:
    """Each tracker instance is independent (session isolation)."""

    def test_two_trackers_independent(self) -> None:
        """Two trackers accumulate independently (concurrent TUI sessions)."""
        t1 = TurnUsageTracker()
        t2 = TurnUsageTracker()

        t1.record_api_call({"prompt_tokens": 1000, "cached_tokens": 800, "total_tokens": 1100})
        t2.record_api_call({"prompt_tokens": 500, "cached_tokens": 100, "total_tokens": 600})

        s1 = t1.session_cache_stats()
        s2 = t2.session_cache_stats()

        assert s1.total_prompt_tokens == 1000
        assert s2.total_prompt_tokens == 500
        assert s1.token_weighted_hit_rate == 0.8
        assert s2.token_weighted_hit_rate == 0.2

    def test_reset_preserves_session_accumulators(self) -> None:
        """reset() clears per-turn but preserves session-level state."""
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        tracker.record_api_call({"prompt_tokens": 1000, "cached_tokens": 800, "total_tokens": 1100})
        tracker.reset()
        tracker.record_api_call({"prompt_tokens": 2000, "cached_tokens": 1600, "total_tokens": 2200})

        stats = tracker.session_cache_stats()
        assert stats.total_prompt_tokens == 3000
        assert stats.total_cached_tokens == 2400
        assert stats.token_weighted_hit_rate == 0.8  # 2400/3000
        assert stats.completed_turns == 2

        # Per-turn summary is only for current turn
        summary = tracker.summary()
        assert summary.prompt_tokens == 2000


# ═══════════════════════════════════════════════════════════════════════════
#  SessionCacheStats standalone properties
# ═══════════════════════════════════════════════════════════════════════════

class TestSessionCacheStatsProperties:
    """Direct property tests on SessionCacheStats."""

    def test_per_turn_average_empty(self) -> None:
        stats = SessionCacheStats()
        assert stats.per_turn_average_hit_rate == 0.0

    def test_per_turn_average_calculation(self) -> None:
        stats = SessionCacheStats(per_turn_rates=(0.4, 0.6, 0.8))
        assert stats.per_turn_average_hit_rate == 0.6  # (0.4+0.6+0.8)/3

    def test_default_steady_state_skip_turns_constant(self) -> None:
        assert DEFAULT_STEADY_STATE_SKIP_TURNS == 3


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-API-call per turn
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiApiCallPerTurn:
    """Turns with multiple API calls (e.g. retry, tool-call continuation)."""

    def test_multiple_api_calls_accumulate_correctly(self) -> None:
        tracker = TurnUsageTracker(steady_state_skip_turns=0)
        # Two API calls in one turn
        tracker.record_api_call({"prompt_tokens": 500, "cached_tokens": 400, "total_tokens": 600})
        tracker.record_api_call({"prompt_tokens": 600, "cached_tokens": 500, "total_tokens": 700})

        stats = tracker.session_cache_stats()
        assert stats.total_prompt_tokens == 1100
        assert stats.total_cached_tokens == 900
        tw = stats.token_weighted_hit_rate
        assert abs(tw - 900 / 1100) < 0.001

        # Per-turn rate = combined (900/1100)
        summary = tracker.summary()
        assert summary.prompt_tokens == 1100
        assert summary.cached_tokens == 900

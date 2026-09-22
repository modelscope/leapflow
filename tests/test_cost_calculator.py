# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the config-driven cost calculator and pricing config catalog.

Covers:
1. Cost computation with configured pricing (including cached-token ratio).
2. Missing pricing => cost unknown (None), no crash.
3. Exact match, prefix match, and regex match resolution.
4. Config catalog exposes the usage.pricing key.
5. Latency aggregation helper.
6. Usage payload integration with cost and latency.
"""
from __future__ import annotations

from leapflow.engine.cost_calculator import (
    ModelPricing,
    compute_cost,
    format_cost,
    resolve_pricing,
)
from leapflow.performance import LatencySummary, RollingLatency, aggregate_latency_snapshots


# ═══════════════════════════════════════════════════════════════
#  ModelPricing validation
# ═══════════════════════════════════════════════════════════════


class TestModelPricing:
    def test_valid_pricing(self) -> None:
        p = ModelPricing(input_per_mtok=2.5, output_per_mtok=10.0, cached_input_ratio=0.5)
        assert p.validate()

    def test_negative_input_invalid(self) -> None:
        p = ModelPricing(input_per_mtok=-1.0, output_per_mtok=10.0)
        assert not p.validate()

    def test_ratio_above_one_invalid(self) -> None:
        p = ModelPricing(input_per_mtok=2.5, output_per_mtok=10.0, cached_input_ratio=1.5)
        assert not p.validate()


# ═══════════════════════════════════════════════════════════════
#  Pricing resolution
# ═══════════════════════════════════════════════════════════════


SAMPLE_PRICING = {
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
    "qwen": {
        "input_per_mtok": 0.50,
        "output_per_mtok": 2.00,
        "cached_input_ratio": 0.1,
    },
}


class TestResolvePricing:
    def test_exact_match(self) -> None:
        p = resolve_pricing("deepseek-chat", SAMPLE_PRICING)
        assert p is not None
        assert p.input_per_mtok == 0.27
        assert p.output_per_mtok == 1.10
        assert p.cached_input_ratio == 0.1

    def test_exact_match_case_insensitive(self) -> None:
        p = resolve_pricing("GPT-4o", SAMPLE_PRICING)
        assert p is not None
        assert p.input_per_mtok == 2.50

    def test_prefix_match(self) -> None:
        p = resolve_pricing("qwen3.7-plus", SAMPLE_PRICING)
        assert p is not None
        assert p.input_per_mtok == 0.50

    def test_no_match_returns_none(self) -> None:
        p = resolve_pricing("claude-3-opus", SAMPLE_PRICING)
        assert p is None

    def test_empty_model_returns_none(self) -> None:
        p = resolve_pricing("", SAMPLE_PRICING)
        assert p is None

    def test_empty_config_returns_none(self) -> None:
        p = resolve_pricing("gpt-4o", {})
        assert p is None

    def test_longest_prefix_wins(self) -> None:
        """When multiple prefixes match, the longest one wins."""
        config = {
            "gpt": {"input_per_mtok": 1.0, "output_per_mtok": 3.0},
            "gpt-4": {"input_per_mtok": 2.0, "output_per_mtok": 8.0},
        }
        p = resolve_pricing("gpt-4o-2024", config)
        assert p is not None
        assert p.input_per_mtok == 2.0

    def test_malformed_entry_returns_none(self) -> None:
        config = {"test-model": "not-a-dict"}
        p = resolve_pricing("test-model", config)
        assert p is None


# ═══════════════════════════════════════════════════════════════
#  Cost computation
# ═══════════════════════════════════════════════════════════════


class TestComputeCost:
    def test_basic_cost_computation(self) -> None:
        """Cost = (miss * input_rate + cached * input_rate * ratio + output * output_rate) / 1M."""
        result = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            cached_tokens=200_000,
            model="deepseek-chat",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.known
        # miss = 800k, cached = 200k
        # input_cost = 800k/1M * 0.27 = 0.216
        # cached_cost = 200k/1M * 0.27 * 0.1 = 0.0054
        # output_cost = 500k/1M * 1.10 = 0.55
        # total = 0.216 + 0.0054 + 0.55 = 0.7714
        assert result.dollar_cost is not None
        assert abs(result.dollar_cost - 0.7714) < 0.0001

    def test_all_cached_tokens(self) -> None:
        """All prompt tokens cached: only cached rate + output rate applied."""
        result = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=1_000_000,
            model="gpt-4o",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.known
        # miss = 0, cached = 1M, output = 0
        # cached_cost = 1M/1M * 2.50 * 0.5 = 1.25
        assert result.dollar_cost is not None
        assert abs(result.dollar_cost - 1.25) < 0.0001

    def test_zero_tokens(self) -> None:
        result = compute_cost(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model="gpt-4o",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.known
        assert result.dollar_cost == 0.0

    def test_missing_pricing_returns_unknown(self) -> None:
        """No pricing for model => cost unknown, no crash."""
        result = compute_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=100,
            model="claude-3-opus",
            pricing_config=SAMPLE_PRICING,
        )
        assert not result.known
        assert result.dollar_cost is None
        assert result.model == "claude-3-opus"

    def test_empty_pricing_config(self) -> None:
        result = compute_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            cached_tokens=0,
            model="gpt-4o",
            pricing_config={},
        )
        assert not result.known
        assert result.dollar_cost is None

    def test_prefix_match_in_compute(self) -> None:
        """Prefix matching works through compute_cost."""
        result = compute_cost(
            prompt_tokens=100_000,
            completion_tokens=50_000,
            cached_tokens=0,
            model="qwen3.7-plus",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.known
        # miss = 100k, cached = 0
        # input = 100k/1M * 0.50 = 0.05
        # output = 50k/1M * 2.00 = 0.10
        assert result.dollar_cost is not None
        assert abs(result.dollar_cost - 0.15) < 0.0001

    def test_exact_match_source(self) -> None:
        result = compute_cost(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=0,
            model="deepseek-chat",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.pricing_source == "exact"

    def test_prefix_match_source(self) -> None:
        result = compute_cost(
            prompt_tokens=1000,
            completion_tokens=100,
            cached_tokens=0,
            model="qwen3.7-plus",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.pricing_source == "prefix"


# ═══════════════════════════════════════════════════════════════
#  Format cost
# ═══════════════════════════════════════════════════════════════


class TestFormatCost:
    def test_none_shows_unknown(self) -> None:
        assert format_cost(None) == "unknown"

    def test_small_cost_four_decimals(self) -> None:
        assert format_cost(0.0042) == "$0.0042"

    def test_large_cost_two_decimals(self) -> None:
        assert format_cost(1.50) == "$1.50"

    def test_zero_cost(self) -> None:
        assert format_cost(0.0) == "$0.0000"


# ═══════════════════════════════════════════════════════════════
#  Config catalog exposes the pricing key
# ═══════════════════════════════════════════════════════════════


class TestConfigCatalogPricing:
    def test_pricing_key_in_field_specs(self) -> None:
        from leapflow.config_service import _FIELD_SPECS

        assert "usage.pricing" in _FIELD_SPECS, (
            "usage.pricing must be registered in the config catalog"
        )
        spec = _FIELD_SPECS["usage.pricing"]
        assert spec.category == "Usage"
        assert "pricing" in spec.description.lower()
        assert spec.hot_reload in ("yes", "partial")


# ═══════════════════════════════════════════════════════════════
#  Latency aggregation
# ═══════════════════════════════════════════════════════════════


class TestLatencyAggregation:
    def test_empty_snapshots(self) -> None:
        result = aggregate_latency_snapshots({})
        assert result == {}

    def test_zero_count_filtered(self) -> None:
        """Snapshots with count=0 are filtered out."""
        result = aggregate_latency_snapshots({"empty": LatencySummary()})
        assert result == {}

    def test_non_empty_included(self) -> None:
        rl = RollingLatency(capacity=10)
        rl.observe(10.0)
        rl.observe(20.0)
        snap = rl.snapshot()
        result = aggregate_latency_snapshots({"test": snap})
        assert "test" in result
        data = result["test"]
        assert data["count"] == 2
        assert data["p50_ms"] > 0

    def test_multiple_labels(self) -> None:
        rl1 = RollingLatency(capacity=10)
        rl1.observe(5.0)
        rl2 = RollingLatency(capacity=10)
        rl2.observe(50.0)
        result = aggregate_latency_snapshots({
            "fast": rl1.snapshot(),
            "slow": rl2.snapshot(),
        })
        assert len(result) == 2
        assert result["fast"]["p50_ms"] == 5.0
        assert result["slow"]["p50_ms"] == 50.0


# ═══════════════════════════════════════════════════════════════
#  Usage payload rendering helper (unit-test the aggregation logic)
# ═══════════════════════════════════════════════════════════════


class TestUsagePayloadCostIntegration:
    """Test that cost data flows through the usage payload structure."""

    def test_cost_in_payload_when_pricing_configured(self) -> None:
        """Verify the payload structure includes cost fields."""
        from leapflow.engine.cost_calculator import compute_cost, format_cost

        result = compute_cost(
            prompt_tokens=500_000,
            completion_tokens=100_000,
            cached_tokens=50_000,
            model="deepseek-chat",
            pricing_config=SAMPLE_PRICING,
        )
        # Simulate payload construction
        payload = {
            "dollar_cost": result.dollar_cost,
            "dollar_cost_formatted": format_cost(result.dollar_cost),
            "pricing_source": result.pricing_source,
        }
        assert payload["dollar_cost"] is not None
        assert "$" in payload["dollar_cost_formatted"]
        assert payload["pricing_source"] == "exact"

    def test_no_cost_when_pricing_missing(self) -> None:
        """Verify graceful degradation in payload."""
        from leapflow.engine.cost_calculator import compute_cost

        result = compute_cost(
            prompt_tokens=500_000,
            completion_tokens=100_000,
            cached_tokens=50_000,
            model="unknown-model",
            pricing_config=SAMPLE_PRICING,
        )
        assert result.dollar_cost is None
        assert not result.known

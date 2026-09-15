# Copyright (c) Alibaba, Inc. and its affiliates.
"""Comprehensive tests for the Learning Plugin Evolution integration.

Covers:
- PluginTrustLedger: promotion, demotion, hard failure freeze, state roundtrip
- PluginUsageTracker: sample accumulation, bounded memory, stats aggregation, trust forwarding
- PluginAdvisor: recommendation engine with various error rate / trust level combinations
- Integration: wiring advisor into self_management plugin_status
"""

from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import patch, MagicMock

from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel
from leapflow.learning.plugin_stats import PluginUsageTracker, PluginStats
from leapflow.learning.plugin_advisor import (
    PluginAdvisor,
    PluginRecommendation,
    get_default_advisor,
    set_default_advisor,
)


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def ledger():
    """Fresh PluginTrustLedger with test-friendly thresholds."""
    return PluginTrustLedger(candidate_at=5, verified_at=20, production_at=50, demote_after=3)


@pytest.fixture
def tracker():
    """Fresh PluginUsageTracker with bounded samples."""
    return PluginUsageTracker(max_samples_per_tool=100)


@pytest.fixture
def advisor(ledger, tracker):
    """PluginAdvisor wired with test ledger and tracker."""
    tracker.set_trust_ledger(ledger)
    return PluginAdvisor(ledger, tracker)


# ════════════════════════════════════════════════════════════════
# TestPluginTrustLedger
# ════════════════════════════════════════════════════════════════


class TestPluginTrustLedger:
    """Tests for PluginTrustLedger progressive trust and demotion mechanics."""

    def test_initial_level_is_draft(self, ledger: PluginTrustLedger) -> None:
        """A never-seen plugin starts at DRAFT."""
        assert ledger.level("new_plugin") == PluginTrustLevel.DRAFT

    def test_promotion_to_candidate(self, ledger: PluginTrustLedger) -> None:
        """5 consecutive successes → CANDIDATE."""
        for _ in range(5):
            ledger.record_success("p1")
        assert ledger.level("p1") == PluginTrustLevel.CANDIDATE

    def test_promotion_to_verified(self, ledger: PluginTrustLedger) -> None:
        """20 consecutive successes → VERIFIED."""
        for _ in range(20):
            ledger.record_success("p2")
        assert ledger.level("p2") == PluginTrustLevel.VERIFIED

    def test_promotion_to_production(self, ledger: PluginTrustLedger) -> None:
        """50 consecutive successes → PRODUCTION."""
        for _ in range(50):
            ledger.record_success("p3")
        assert ledger.level("p3") == PluginTrustLevel.PRODUCTION

    def test_failure_resets_consecutive_ok(self, ledger: PluginTrustLedger) -> None:
        """success×4 → fail → success×4 → still DRAFT (needs 5 consecutive)."""
        for _ in range(4):
            ledger.record_success("p4")
        ledger.record_failure("p4")
        for _ in range(4):
            ledger.record_success("p4")
        assert ledger.level("p4") == PluginTrustLevel.DRAFT

    def test_sustained_failure_demotes(self, ledger: PluginTrustLedger) -> None:
        """Promote to CANDIDATE → 3 consecutive failures → back to DRAFT."""
        # Promote to CANDIDATE first
        for _ in range(5):
            ledger.record_success("p5")
        assert ledger.level("p5") == PluginTrustLevel.CANDIDATE
        # 3 consecutive failures → demotion
        for _ in range(3):
            ledger.record_failure("p5")
        assert ledger.level("p5") == PluginTrustLevel.DRAFT

    def test_hard_failure_freezes(self, ledger: PluginTrustLedger) -> None:
        """Promote to VERIFIED → hard failure → DRAFT, frozen; no further promotion."""
        for _ in range(20):
            ledger.record_success("p6")
        assert ledger.level("p6") == PluginTrustLevel.VERIFIED
        # Hard failure freezes to DRAFT
        ledger.record_failure("p6", hard=True)
        assert ledger.level("p6") == PluginTrustLevel.DRAFT
        # Further successes don't promote (frozen)
        for _ in range(100):
            ledger.record_success("p6")
        assert ledger.level("p6") == PluginTrustLevel.DRAFT

    def test_to_state_and_load_state(self, ledger: PluginTrustLedger) -> None:
        """Roundtrip preserves levels, consecutive counts, and frozen set."""
        # Create diverse state
        for _ in range(5):
            ledger.record_success("promoted")
        assert ledger.level("promoted") == PluginTrustLevel.CANDIDATE

        for _ in range(20):
            ledger.record_success("verified_then_frozen")
        ledger.record_failure("verified_then_frozen", hard=True)

        # Serialize and restore
        state = ledger.to_state()
        restored = PluginTrustLedger.load_state(state)

        assert restored.level("promoted") == PluginTrustLevel.CANDIDATE
        assert restored.level("verified_then_frozen") == PluginTrustLevel.DRAFT
        # Frozen should persist
        for _ in range(100):
            restored.record_success("verified_then_frozen")
        assert restored.level("verified_then_frozen") == PluginTrustLevel.DRAFT

        # Config thresholds preserved
        assert state["candidate_at"] == 5
        assert state["verified_at"] == 20
        assert state["production_at"] == 50
        assert state["demote_after"] == 3


# ════════════════════════════════════════════════════════════════
# TestPluginUsageTracker
# ════════════════════════════════════════════════════════════════


class TestPluginUsageTracker:
    """Tests for PluginUsageTracker sample recording, bounding, and aggregation."""

    def test_record_accumulates_samples(self, tracker: PluginUsageTracker) -> None:
        """Recording 5 calls accumulates exactly 5 samples for the tool."""
        for i in range(5):
            tracker.record("my_tool", ok=True, duration_ms=10.0 + i)
        # Access internal deque directly
        assert len(tracker._samples["my_tool"]) == 5

    def test_bounded_memory(self, tracker: PluginUsageTracker) -> None:
        """Recording more than maxlen samples doesn't exceed the deque bound."""
        # tracker fixture has max_samples_per_tool=100
        for i in range(150):
            tracker.record("overflow_tool", ok=True, duration_ms=float(i))
        assert len(tracker._samples["overflow_tool"]) == 100
        # Oldest samples are evicted (deque discards from left)
        oldest = tracker._samples["overflow_tool"][0]
        assert oldest.duration_ms == 50.0  # first 50 evicted

    def test_stats_aggregation(self, tracker: PluginUsageTracker) -> None:
        """Record a mix of success/fail with known durations and verify PluginStats."""
        # Patch the reverse index so stats_for_plugin can find our tools
        fake_index = {"tool_a": "my_plugin", "tool_b": "my_plugin"}
        tracker._get_reverse_index = lambda: fake_index

        # Record 8 successes (10ms each) and 2 failures (50ms each)
        for _ in range(8):
            tracker.record("tool_a", ok=True, duration_ms=10.0)
        for _ in range(2):
            tracker.record("tool_b", ok=False, duration_ms=50.0)

        stats = tracker.stats_for_plugin("my_plugin")
        assert stats is not None
        assert stats.total_calls == 10
        assert stats.successes == 8
        assert stats.failures == 2
        assert stats.error_rate == pytest.approx(0.2, abs=0.001)
        # avg_duration: (8*10 + 2*50) / 10 = 180/10 = 18.0
        assert stats.avg_duration_ms == pytest.approx(18.0, abs=0.1)

    def test_trust_forwarding(self, tracker: PluginUsageTracker, ledger: PluginTrustLedger) -> None:
        """Setting a trust ledger causes record() to forward success/failure to it."""
        tracker.set_trust_ledger(ledger)
        # Patch the reverse index to map tool→plugin
        tracker._get_reverse_index = lambda: {"fwd_tool": "forwarded_plugin"}

        # Record 5 successes → should promote to CANDIDATE
        for _ in range(5):
            tracker.record("fwd_tool", ok=True, duration_ms=5.0)
        assert ledger.level("forwarded_plugin") == PluginTrustLevel.CANDIDATE

        # Record 3 failures → should demote back to DRAFT
        for _ in range(3):
            tracker.record("fwd_tool", ok=False, duration_ms=5.0)
        assert ledger.level("forwarded_plugin") == PluginTrustLevel.DRAFT


# ════════════════════════════════════════════════════════════════
# TestPluginAdvisor
# ════════════════════════════════════════════════════════════════


class TestPluginAdvisor:
    """Tests for the PluginAdvisor recommendation engine."""

    def test_insufficient_data_returns_none(
        self, advisor: PluginAdvisor, tracker: PluginUsageTracker
    ) -> None:
        """Fewer than 3 calls → no recommendation."""
        tracker._get_reverse_index = lambda: {"adv_tool": "adv_plugin"}

        tracker.record("adv_tool", ok=True, duration_ms=10.0)
        tracker.record("adv_tool", ok=True, duration_ms=10.0)
        rec = advisor.recommend("adv_plugin")
        assert rec is None

    def test_high_error_investigate(
        self, advisor: PluginAdvisor, tracker: PluginUsageTracker, ledger: PluginTrustLedger
    ) -> None:
        """>20% error rate → action='investigate'."""
        tracker._get_reverse_index = lambda: {"err_tool": "err_plugin"}

        # 4 success + 2 fail = 6 calls, error_rate = 2/6 ≈ 33% (>20%)
        for _ in range(4):
            tracker.record("err_tool", ok=True, duration_ms=10.0)
        for _ in range(2):
            tracker.record("err_tool", ok=False, duration_ms=10.0)

        rec = advisor.recommend("err_plugin")
        assert rec is not None
        assert rec.action == "investigate"

    def test_sustained_failure_demote(
        self, advisor: PluginAdvisor, tracker: PluginUsageTracker, ledger: PluginTrustLedger
    ) -> None:
        """>30% error rate + VERIFIED trust → action='demote'."""
        tracker._get_reverse_index = lambda: {"dem_tool": "dem_plugin"}

        # Record samples first (3 success + 3 fail = 50% error rate)
        for _ in range(3):
            tracker.record("dem_tool", ok=True, duration_ms=10.0)
        for _ in range(3):
            tracker.record("dem_tool", ok=False, duration_ms=10.0)

        # Set VERIFIED trust AFTER recording to avoid trust forwarding demotion
        ledger._levels["dem_plugin"] = PluginTrustLevel.VERIFIED

        rec = advisor.recommend("dem_plugin")
        assert rec is not None
        assert rec.action == "demote"
        assert "VERIFIED" in rec.trust_level

    def test_success_low_trust_promote(
        self, advisor: PluginAdvisor, tracker: PluginUsageTracker, ledger: PluginTrustLedger
    ) -> None:
        """<5% error rate + DRAFT with enough calls → action='promote'."""
        tracker._get_reverse_index = lambda: {"promo_tool": "promo_plugin"}

        # 10 successes, 0 failures → error_rate 0%
        for _ in range(10):
            tracker.record("promo_tool", ok=True, duration_ms=10.0)

        rec = advisor.recommend("promo_plugin")
        assert rec is not None
        assert rec.action == "promote"
        assert rec.confidence > 0.9

    def test_stable_no_recommendation(
        self, advisor: PluginAdvisor, tracker: PluginUsageTracker, ledger: PluginTrustLedger
    ) -> None:
        """~10% error rate at PRODUCTION → None (stable, no recommendation)."""
        # Plugin already at PRODUCTION — no promotion possible
        ledger._levels["stable_plugin"] = PluginTrustLevel.PRODUCTION
        tracker._get_reverse_index = lambda: {"stable_tool": "stable_plugin"}

        # 9 success + 1 fail = 10 calls, error_rate=10% (between 5% and 20%)
        for _ in range(9):
            tracker.record("stable_tool", ok=True, duration_ms=10.0)
        tracker.record("stable_tool", ok=False, duration_ms=10.0)

        rec = advisor.recommend("stable_plugin")
        assert rec is None


# ════════════════════════════════════════════════════════════════
# TestIntegration
# ════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration test: advisor wired into self_management plugin_status."""

    @pytest.fixture
    def self_mgmt_plugin(self):
        """Get the self_management plugin from the global registry."""
        from leapflow.plugins import get_registry

        reg = get_registry()
        reg.assemble()
        plugin = reg.get_plugin("self_management")
        yield plugin

    @pytest.mark.asyncio
    async def test_plugin_status_includes_trust_and_recommendation(
        self, self_mgmt_plugin: Any
    ) -> None:
        """Wire advisor, call plugin_status, verify trust_level and recommendation fields."""
        # Create fresh learning stack
        ledger = PluginTrustLedger(candidate_at=5, verified_at=20, production_at=50, demote_after=3)
        tracker = PluginUsageTracker(max_samples_per_tool=100)
        tracker.set_trust_ledger(ledger)
        adv = PluginAdvisor(ledger, tracker)

        # Install global advisor
        set_default_advisor(adv)
        try:
            # First call — no usage data yet, trust_level should appear as DRAFT
            result = await self_mgmt_plugin._plugin_status_handler(plugin_id="text_utils")
            assert result["ok"] is True
            assert result["trust_level"] == "DRAFT"
            # No recommendation with insufficient data
            assert "recommendation" not in result

            # Now record failures to trigger a recommendation
            # Patch the reverse index so tracker can map tool→plugin
            tracker._get_reverse_index = lambda: {"text_search": "text_utils", "text_replace": "text_utils"}

            # Record enough failures to trigger "investigate" (>20% error rate)
            for _ in range(4):
                tracker.record("text_search", ok=True, duration_ms=10.0)
            for _ in range(3):
                tracker.record("text_search", ok=False, duration_ms=50.0)

            # Second call — recommendation should now appear
            result2 = await self_mgmt_plugin._plugin_status_handler(plugin_id="text_utils")
            assert result2["ok"] is True
            assert result2["trust_level"] == "DRAFT"
            assert "recommendation" in result2
            assert result2["recommendation"]["action"] in ("investigate", "demote")
            assert "confidence" in result2["recommendation"]
        finally:
            # Cleanup: remove global advisor
            set_default_advisor(None)

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for CompressionTimeoutStrategy — stepped cooldown and deterministic degradation."""
from __future__ import annotations

import time
from unittest.mock import patch

from leapflow.engine.recovery.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.recovery.recovery_budget import RecoveryBudget
from leapflow.engine.recovery.recovery_coordinator import RecoveryState
from leapflow.engine.recovery.recovery_decision import RecoveryAction
from leapflow.engine.recovery.strategies.compression_timeout import (
    DETERMINISTIC_SUMMARY_PLACEHOLDER,
    CompressionTimeoutStrategy,
    _cooldown_for_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_envelope(category: str = "compression_timeout") -> FailureEnvelope:
    """Build a minimal FailureEnvelope for compression timeout tests."""
    return FailureEnvelope.create(
        source=FailureSource.SYSTEM,
        category=category,
        failure_class="TimeoutError",
        failure_code="COMPRESSION_TIMEOUT",
        message="Compression stage timed out after 30s",
        recoverability=Recoverability.AUTO_RETRY,
        side_effect_state=SideEffectState.NONE,
        context=FailureContext(),
    )


def _make_state(**overrides) -> RecoveryState:
    """Build a RecoveryState with optional overrides."""
    state = RecoveryState()
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ---------------------------------------------------------------------------
# _cooldown_for_count unit tests
# ---------------------------------------------------------------------------

class TestCooldownForCount:
    """Verify the stepped cooldown mapping."""

    def test_first_timeout_60s(self):
        assert _cooldown_for_count(1) == 60.0

    def test_second_timeout_300s(self):
        assert _cooldown_for_count(2) == 300.0

    def test_third_timeout_900s(self):
        assert _cooldown_for_count(3) == 900.0

    def test_beyond_third_clamps_to_900s(self):
        assert _cooldown_for_count(4) == 900.0
        assert _cooldown_for_count(10) == 900.0


# ---------------------------------------------------------------------------
# Protocol property tests
# ---------------------------------------------------------------------------

class TestProtocolProperties:
    """Verify the strategy satisfies the RecoveryStrategy protocol shape."""

    def test_key(self):
        s = CompressionTimeoutStrategy()
        assert s.key == "compression_timeout"

    def test_priority_between_compress_and_retry(self):
        s = CompressionTimeoutStrategy()
        assert 10 < s.priority < 100, "Should sit between context_compress (10) and jittered_retry (100)"

    def test_repeatable(self):
        s = CompressionTimeoutStrategy()
        assert s.repeatable is True

    def test_applicable_sources(self):
        s = CompressionTimeoutStrategy()
        assert "llm" in s.applicable_sources
        assert "system" in s.applicable_sources

    def test_applicable_categories(self):
        s = CompressionTimeoutStrategy()
        assert "compression_timeout" in s.applicable_categories
        assert "context_compression_timeout" in s.applicable_categories


# ---------------------------------------------------------------------------
# Stepped cooldown correctness
# ---------------------------------------------------------------------------

class TestSteppedCooldown:
    """The core graduated cooldown ladder."""

    def test_first_timeout_yields_60s_retry(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()
        decision = s.decide(env, state)

        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.strategy_key == "compression_timeout"
        assert decision.retry_semantics.backoff_config is not None
        assert decision.retry_semantics.backoff_config.base_delay == 60.0
        assert decision.budget_cost == 1
        meta = decision.audit_metadata_dict
        assert meta["consecutive_timeouts"] == 1
        assert meta["cooldown_seconds"] == 60.0
        assert meta["degradation"] is False

    def test_second_timeout_yields_300s_retry(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()
        s.decide(env, state)  # 1st
        decision = s.decide(env, state)  # 2nd

        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.retry_semantics.backoff_config.base_delay == 300.0
        assert decision.audit_metadata_dict["consecutive_timeouts"] == 2

    def test_third_timeout_triggers_degradation(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()

        s.decide(env, state)  # 1st
        s.decide(env, state)  # 2nd
        decision = s.decide(env, state)  # 3rd — degradation

        assert decision.action == RecoveryAction.SKIP_AND_CONTINUE
        assert decision.budget_cost == 0
        assert decision.retry_semantics.consumes_retry_budget is False
        meta = decision.audit_metadata_dict
        assert meta["degradation"] is True
        assert meta["placeholder"] == DETERMINISTIC_SUMMARY_PLACEHOLDER
        assert decision.transform_description == DETERMINISTIC_SUMMARY_PLACEHOLDER


# ---------------------------------------------------------------------------
# Consecutive timeout counter behaviour
# ---------------------------------------------------------------------------

class TestConsecutiveCounter:
    """Counter increments and resets correctly."""

    def test_counter_increments(self):
        s = CompressionTimeoutStrategy()
        assert s.consecutive_timeouts == 0
        s.decide(_make_envelope(), _make_state())
        assert s.consecutive_timeouts == 1
        s.decide(_make_envelope(), _make_state())
        assert s.consecutive_timeouts == 2

    def test_record_success_resets_counter(self):
        s = CompressionTimeoutStrategy()
        s.decide(_make_envelope(), _make_state())
        s.decide(_make_envelope(), _make_state())
        assert s.consecutive_timeouts == 2

        s.record_success()
        assert s.consecutive_timeouts == 0

    def test_counter_survives_across_envelopes(self):
        """Different envelope instances still share the strategy counter."""
        s = CompressionTimeoutStrategy()
        s.decide(_make_envelope(), _make_state())
        s.decide(_make_envelope(category="context_compression_timeout"), _make_state())
        assert s.consecutive_timeouts == 2

    def test_reset_then_new_sequence(self):
        """After reset, the cooldown ladder restarts from tier 1."""
        s = CompressionTimeoutStrategy()
        s.decide(_make_envelope(), _make_state())
        s.decide(_make_envelope(), _make_state())
        s.record_success()

        decision = s.decide(_make_envelope(), _make_state())
        assert decision.retry_semantics.backoff_config.base_delay == 60.0
        assert s.consecutive_timeouts == 1


# ---------------------------------------------------------------------------
# Deterministic degradation
# ---------------------------------------------------------------------------

class TestDeterministicDegradation:
    """All compression paths exhausted → skip with placeholder."""

    def test_degradation_after_max_retries(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()

        for _ in range(2):
            d = s.decide(env, state)
            assert d.action == RecoveryAction.RETRY_WITH_BACKOFF

        d = s.decide(env, state)
        assert d.action == RecoveryAction.SKIP_AND_CONTINUE
        assert DETERMINISTIC_SUMMARY_PLACEHOLDER in d.transform_description

    def test_further_decides_stay_degraded(self):
        """Once at the degradation tier, subsequent calls stay degraded."""
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()

        for _ in range(3):
            s.decide(env, state)

        # 4th call — still degraded
        d = s.decide(env, state)
        assert d.action == RecoveryAction.SKIP_AND_CONTINUE
        assert d.audit_metadata_dict["consecutive_timeouts"] == 4


# ---------------------------------------------------------------------------
# Budget-exhausted behaviour
# ---------------------------------------------------------------------------

class TestBudgetExhausted:
    """When the budget is spent, strategy changes applicability."""

    def test_can_apply_false_when_budget_exhausted_and_count_low(self):
        """Budget gone + few timeouts → not applicable (let other strategies handle)."""
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()
        budget = RecoveryBudget(total_recovery_actions=0)

        assert s.can_apply(env, state, budget) is False

    def test_can_apply_true_when_budget_exhausted_but_degradation_ready(self):
        """Budget gone + enough timeouts → applicable for degradation (costs 0)."""
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()
        budget = RecoveryBudget(total_recovery_actions=0)

        # Pump consecutive counter past threshold
        for _ in range(3):
            s.decide(env, state)

        assert s.can_apply(env, state, budget) is True


# ---------------------------------------------------------------------------
# Cooldown enforcement in can_apply
# ---------------------------------------------------------------------------

class TestCooldownEnforcement:
    """Active cooldown period blocks re-entry."""

    def test_can_apply_false_during_cooldown(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()

        s.decide(env, state)  # sets _last_cooldown_end ~60s from now
        assert s.can_apply(env, state) is False  # still in cooldown

    def test_can_apply_true_after_cooldown_expires(self):
        s = CompressionTimeoutStrategy()
        env = _make_envelope()
        state = _make_state()

        s.decide(env, state)

        # Fast-forward past cooldown
        with patch("leapflow.engine.recovery.strategies.compression_timeout.time") as mock_time:
            mock_time.monotonic.return_value = time.monotonic() + 120
            assert s.can_apply(env, state) is True


# ---------------------------------------------------------------------------
# Registration in default_strategies
# ---------------------------------------------------------------------------

class TestRegistration:
    """Strategy appears in the default strategy list."""

    def test_in_default_strategies(self):
        from leapflow.engine.recovery.strategies import default_strategies
        strategies = default_strategies()
        keys = [s.key for s in strategies]
        assert "compression_timeout" in keys

    def test_ordered_by_priority(self):
        from leapflow.engine.recovery.strategies import default_strategies
        strategies = default_strategies()
        # Find neighbors: should be after context_compress (10) and before multimodal_strip
        keys = [s.key for s in strategies]
        idx_compress = keys.index("context_compress")
        idx_ct = keys.index("compression_timeout")
        assert idx_ct == idx_compress + 1, (
            "compression_timeout should be right after context_compress in the list"
        )

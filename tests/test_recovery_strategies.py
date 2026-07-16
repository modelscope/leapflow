"""Tests for built-in recovery strategies.

Covers each strategy:
- can_apply returns True for matching envelope and False for non-matching
- decide returns correct RecoveryAction and RetrySemantics
- Priority ordering is correct
- Strategies respect budget limits
- JitteredRetry uses different BackoffConfig based on category
"""
from __future__ import annotations

import pytest

from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.recovery_coordinator import RecoveryState, RecoveryStrategy
from leapflow.engine.recovery_decision import (
    BackoffConfig,
    RecoveryAction,
    RetrySemantics,
)
from leapflow.engine.recovery_strategies import (
    ContextCompressStrategy,
    CredentialRotateStrategy,
    JitteredRetryStrategy,
    MultimodalStripStrategy,
    NativeToTextFallbackStrategy,
    ProviderFailoverStrategy,
    ThinkingDisableStrategy,
    ToolSchemaExpandStrategy,
    default_strategies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    *,
    source: FailureSource = FailureSource.LLM,
    category: str = "transient",
    message: str = "test error",
    recoverability: Recoverability = Recoverability.AUTO_RETRY,
    tool_name: str = "",
) -> FailureEnvelope:
    return FailureEnvelope.create(
        source=source,
        category=category,
        failure_class="test",
        failure_code="test_code",
        message=message,
        recoverability=recoverability,
        context=FailureContext.from_dict_args(tool_name=tool_name),
    )


def _fresh_state() -> RecoveryState:
    return RecoveryState()


# ===========================================================================
# Protocol Compliance Tests
# ===========================================================================


class TestProtocolCompliance:
    def test_all_strategies_implement_protocol(self) -> None:
        strategies = default_strategies()
        for s in strategies:
            assert isinstance(s, RecoveryStrategy), f"{s.__class__.__name__} does not implement Protocol"

    def test_default_strategies_priority_ordering(self) -> None:
        strategies = default_strategies()
        priorities = [s.priority for s in strategies]
        assert priorities == sorted(priorities), "Strategies should be in priority order"

    def test_all_strategies_have_unique_keys(self) -> None:
        strategies = default_strategies()
        keys = [s.key for s in strategies]
        assert len(keys) == len(set(keys)), "Strategy keys must be unique"

    def test_strategy_count(self) -> None:
        strategies = default_strategies()
        assert len(strategies) == 8


# ===========================================================================
# ContextCompressStrategy Tests
# ===========================================================================


class TestContextCompressStrategy:
    def test_priority(self) -> None:
        s = ContextCompressStrategy()
        assert s.priority == 10

    def test_applicable_sources(self) -> None:
        s = ContextCompressStrategy()
        assert s.applicable_sources == frozenset({"llm"})

    def test_applicable_categories(self) -> None:
        s = ContextCompressStrategy()
        assert s.applicable_categories == frozenset({"context_overflow", "payload_too_large"})

    def test_can_apply_fresh_state(self) -> None:
        s = ContextCompressStrategy()
        env = _make_envelope(category="context_overflow")
        state = _fresh_state()
        assert s.can_apply(env, state) is True

    def test_can_apply_exhausted_phases(self) -> None:
        s = ContextCompressStrategy()
        env = _make_envelope(category="context_overflow")
        state = _fresh_state()
        state.compress_phase_index = 3
        assert s.can_apply(env, state) is False

    def test_decide_phase_progression(self) -> None:
        s = ContextCompressStrategy()
        env = _make_envelope(category="context_overflow")
        state = _fresh_state()

        d1 = s.decide(env, state)
        assert d1.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert "history_summarize" in d1.reason
        assert state.compress_phase_index == 1

        d2 = s.decide(env, state)
        assert "multimodal_to_text" in d2.reason
        assert state.compress_phase_index == 2

        d3 = s.decide(env, state)
        assert "disclosure_shrink" in d3.reason
        assert state.compress_phase_index == 3

    def test_decide_does_not_consume_budget(self) -> None:
        s = ContextCompressStrategy()
        env = _make_envelope(category="context_overflow")
        state = _fresh_state()
        decision = s.decide(env, state)
        assert decision.retry_semantics.consumes_retry_budget is False
        assert decision.budget_cost == 0


# ===========================================================================
# MultimodalStripStrategy Tests
# ===========================================================================


class TestMultimodalStripStrategy:
    def test_priority(self) -> None:
        assert MultimodalStripStrategy().priority == 15

    def test_can_apply_with_image_message(self) -> None:
        s = MultimodalStripStrategy()
        env = _make_envelope(category="image_too_large", message="Image file too large to encode")
        assert s.can_apply(env, _fresh_state()) is True

    def test_can_apply_no_image_keyword(self) -> None:
        s = MultimodalStripStrategy()
        env = _make_envelope(category="image_too_large", message="random error text")
        # Still applies because category itself is image_too_large
        assert s.can_apply(env, _fresh_state()) is True

    def test_decide_action(self) -> None:
        s = MultimodalStripStrategy()
        env = _make_envelope(category="image_too_large", message="Image too large")
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert decision.retry_semantics.consumes_retry_budget is False


# ===========================================================================
# ProviderFailoverStrategy Tests
# ===========================================================================


class TestProviderFailoverStrategy:
    def test_priority(self) -> None:
        assert ProviderFailoverStrategy().priority == 20

    def test_applicable_categories(self) -> None:
        s = ProviderFailoverStrategy()
        expected = frozenset({"billing", "auth_permanent", "overloaded", "model_not_found", "content_blocked"})
        assert s.applicable_categories == expected

    def test_can_apply(self) -> None:
        s = ProviderFailoverStrategy()
        env = _make_envelope(category="billing")
        assert s.can_apply(env, _fresh_state()) is True

    def test_decide_action(self) -> None:
        s = ProviderFailoverStrategy()
        env = _make_envelope(category="billing")
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.FAILOVER
        assert decision.retry_semantics.consumes_retry_budget is True
        assert decision.retry_semantics.resets_retry_count is True
        assert decision.budget_cost == 1


# ===========================================================================
# CredentialRotateStrategy Tests
# ===========================================================================


class TestCredentialRotateStrategy:
    def test_priority(self) -> None:
        assert CredentialRotateStrategy().priority == 25

    def test_applicable_categories(self) -> None:
        s = CredentialRotateStrategy()
        assert s.applicable_categories == frozenset({"auth_error", "rate_limited", "billing"})

    def test_decide_action(self) -> None:
        s = CredentialRotateStrategy()
        env = _make_envelope(category="auth_error")
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.FAILOVER
        assert decision.retry_semantics.consumes_retry_budget is True
        assert decision.retry_semantics.resets_retry_count is True


# ===========================================================================
# ThinkingDisableStrategy Tests
# ===========================================================================


class TestThinkingDisableStrategy:
    def test_priority(self) -> None:
        assert ThinkingDisableStrategy().priority == 30

    def test_applicable_categories(self) -> None:
        assert ThinkingDisableStrategy().applicable_categories == frozenset({"format_error"})

    def test_can_apply_always_true(self) -> None:
        s = ThinkingDisableStrategy()
        env = _make_envelope(category="format_error")
        assert s.can_apply(env, _fresh_state()) is True

    def test_decide_action(self) -> None:
        s = ThinkingDisableStrategy()
        env = _make_envelope(category="format_error")
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert "thinking mode" in decision.reason.lower()
        assert decision.retry_semantics.consumes_retry_budget is False


# ===========================================================================
# NativeToTextFallbackStrategy Tests
# ===========================================================================


class TestNativeToTextFallbackStrategy:
    def test_priority(self) -> None:
        assert NativeToTextFallbackStrategy().priority == 35

    def test_can_apply_with_tool_call_message(self) -> None:
        s = NativeToTextFallbackStrategy()
        env = _make_envelope(category="format_error", message="Failed to parse tool_call response")
        assert s.can_apply(env, _fresh_state()) is True

    def test_can_apply_with_native_message(self) -> None:
        s = NativeToTextFallbackStrategy()
        env = _make_envelope(category="format_error", message="native function calling error")
        assert s.can_apply(env, _fresh_state()) is True

    def test_can_apply_without_keywords(self) -> None:
        s = NativeToTextFallbackStrategy()
        env = _make_envelope(category="format_error", message="generic format error")
        assert s.can_apply(env, _fresh_state()) is False

    def test_decide_action(self) -> None:
        s = NativeToTextFallbackStrategy()
        env = _make_envelope(category="format_error", message="tool_call parse failed")
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert "text mode" in decision.reason.lower()
        assert decision.retry_semantics.consumes_retry_budget is False


# ===========================================================================
# ToolSchemaExpandStrategy Tests
# ===========================================================================


class TestToolSchemaExpandStrategy:
    def test_priority(self) -> None:
        assert ToolSchemaExpandStrategy().priority == 40

    def test_applicable_sources(self) -> None:
        assert ToolSchemaExpandStrategy().applicable_sources == frozenset({"tool"})

    def test_applicable_categories(self) -> None:
        assert ToolSchemaExpandStrategy().applicable_categories == frozenset({"tool_unknown"})

    def test_can_apply_auto_recover(self) -> None:
        s = ToolSchemaExpandStrategy()
        env = _make_envelope(
            source=FailureSource.TOOL,
            category="tool_unknown",
            recoverability=Recoverability.AUTO_RECOVER,
            tool_name="web_search",
        )
        assert s.can_apply(env, _fresh_state()) is True

    def test_can_apply_not_auto_recover(self) -> None:
        s = ToolSchemaExpandStrategy()
        env = _make_envelope(
            source=FailureSource.TOOL,
            category="tool_unknown",
            recoverability=Recoverability.NON_RECOVERABLE,
        )
        assert s.can_apply(env, _fresh_state()) is False

    def test_decide_action(self) -> None:
        s = ToolSchemaExpandStrategy()
        env = _make_envelope(
            source=FailureSource.TOOL,
            category="tool_unknown",
            recoverability=Recoverability.AUTO_RECOVER,
            tool_name="advanced_search",
        )
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert "schema" in decision.reason.lower()
        assert decision.budget_cost == 0


# ===========================================================================
# JitteredRetryStrategy Tests
# ===========================================================================


class TestJitteredRetryStrategy:
    def test_priority(self) -> None:
        assert JitteredRetryStrategy().priority == 100

    def test_applicable_sources(self) -> None:
        assert JitteredRetryStrategy().applicable_sources == frozenset({"llm", "tool", "system"})

    def test_applicable_categories(self) -> None:
        expected = frozenset({"transient", "rate_limited", "overloaded", "tool_timeout"})
        assert JitteredRetryStrategy().applicable_categories == expected

    def test_can_apply_auto_retry(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="transient", recoverability=Recoverability.AUTO_RETRY)
        assert s.can_apply(env, _fresh_state()) is True

    def test_can_apply_not_auto_retry(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="transient", recoverability=Recoverability.USER_FIXABLE)
        assert s.can_apply(env, _fresh_state()) is False

    def test_decide_rate_limited_backoff(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="rate_limited", recoverability=Recoverability.AUTO_RETRY)
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.retry_semantics.backoff_config is not None
        assert decision.retry_semantics.backoff_config.base_delay == 5.0
        assert decision.retry_semantics.backoff_config.max_delay == 120.0
        assert decision.retry_semantics.consumes_retry_budget is True
        assert decision.budget_cost == 1

    def test_decide_transient_backoff(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="transient", recoverability=Recoverability.AUTO_RETRY)
        decision = s.decide(env, _fresh_state())
        assert decision.retry_semantics.backoff_config is not None
        assert decision.retry_semantics.backoff_config.base_delay == 1.0
        assert decision.retry_semantics.backoff_config.max_delay == 60.0

    def test_decide_overloaded_backoff(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="overloaded", recoverability=Recoverability.AUTO_RETRY)
        decision = s.decide(env, _fresh_state())
        assert decision.retry_semantics.backoff_config is not None
        assert decision.retry_semantics.backoff_config.base_delay == 1.0
        assert decision.retry_semantics.backoff_config.max_delay == 60.0

    def test_decide_tool_timeout_backoff(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(
            source=FailureSource.TOOL,
            category="tool_timeout",
            recoverability=Recoverability.AUTO_RETRY,
        )
        decision = s.decide(env, _fresh_state())
        assert decision.retry_semantics.backoff_config is not None
        assert decision.retry_semantics.backoff_config.base_delay == 2.0
        assert decision.retry_semantics.backoff_config.max_delay == 30.0

    def test_decide_does_not_reset_retry_count(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(category="transient", recoverability=Recoverability.AUTO_RETRY)
        decision = s.decide(env, _fresh_state())
        assert decision.retry_semantics.resets_retry_count is False


# ===========================================================================
# Integration: Strategy Priority Ordering
# ===========================================================================


class TestStrategyPriorityOrdering:
    def test_priorities_are_in_expected_order(self) -> None:
        strategies = default_strategies()
        expected_keys = [
            "context_compress",
            "multimodal_strip",
            "provider_failover",
            "credential_rotate",
            "thinking_disable",
            "native_to_text",
            "tool_schema_expand",
            "jittered_retry",
        ]
        actual_keys = [s.key for s in strategies]
        assert actual_keys == expected_keys

    def test_priorities_are_strictly_increasing(self) -> None:
        strategies = default_strategies()
        for i in range(len(strategies) - 1):
            assert strategies[i].priority < strategies[i + 1].priority, (
                f"{strategies[i].key} (priority={strategies[i].priority}) should be "
                f"lower than {strategies[i+1].key} (priority={strategies[i+1].priority})"
            )

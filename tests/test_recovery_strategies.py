# Copyright (c) Alibaba, Inc. and its affiliates.
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

from leapflow.engine.recovery.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
)
from leapflow.engine.recovery.recovery_coordinator import RecoveryState, RecoveryStrategy
from leapflow.engine.recovery.recovery_decision import (
    RecoveryAction,
)
from leapflow.engine.recovery.strategies import (
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


# ===========================================================================
# ContextCompressStrategy Tests
# ===========================================================================


class TestContextCompressStrategy:
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
        assert d1.audit_metadata_dict["phase_index"] == 0
        # decide() no longer mutates state; coordinator handles phase advancement
        assert state.compress_phase_index == 0

        # Simulate coordinator committing phase advancement
        state.compress_phase_index = 1
        d2 = s.decide(env, state)
        assert "multimodal_to_text" in d2.reason
        assert d2.audit_metadata_dict["phase_index"] == 1

        state.compress_phase_index = 2
        d3 = s.decide(env, state)
        assert "disclosure_shrink" in d3.reason
        assert d3.audit_metadata_dict["phase_index"] == 2

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

    def test_decide_system_timeout(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(
            source=FailureSource.SYSTEM,
            category="system_timeout",
            recoverability=Recoverability.AUTO_RETRY,
        )
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.retry_semantics.backoff_config is not None

    def test_decide_system_network(self) -> None:
        s = JitteredRetryStrategy()
        env = _make_envelope(
            source=FailureSource.SYSTEM,
            category="system_network",
            recoverability=Recoverability.AUTO_RETRY,
        )
        decision = s.decide(env, _fresh_state())
        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF


# ===========================================================================
# Integration: routing contract through the coordinator
# ===========================================================================


class TestStrategyRoutingContract:
    """What the registry must guarantee, asserted as behavior.

    Per-strategy ``priority``/``applicable_*`` assertions were removed: copying
    implementation constants into the test freezes them without proving
    anything, and any real regression shows up here instead — as the wrong
    strategy winning for a given failure.
    """

    @pytest.mark.parametrize(
        ("source", "category", "message", "recoverability", "expected_key"),
        [
            # Context pressure is compressed before anything else is tried.
            (FailureSource.LLM, "context_overflow", "too many tokens",
             Recoverability.AUTO_RETRY, "context_compress"),
            (FailureSource.LLM, "payload_too_large", "payload too large",
             Recoverability.AUTO_RETRY, "context_compress"),
            # Oversized images are stripped rather than compressed away.
            (FailureSource.LLM, "image_too_large", "Image too large",
             Recoverability.AUTO_RETRY, "multimodal_strip"),
            # Permanent provider-side conditions fail over instead of retrying.
            (FailureSource.LLM, "billing", "quota exhausted",
             Recoverability.AUTO_RETRY, "provider_failover"),
            (FailureSource.LLM, "model_not_found", "no such model",
             Recoverability.AUTO_RETRY, "provider_failover"),
            # Credential problems rotate before giving up.
            (FailureSource.LLM, "auth_error", "invalid api key",
             Recoverability.AUTO_RETRY, "credential_rotate"),
            # Malformed output: drop thinking mode first…
            (FailureSource.LLM, "format_error", "unparseable output",
             Recoverability.AUTO_RETRY, "thinking_disable"),
            # …unknown tools get their schema disclosed rather than retried blind.
            (FailureSource.TOOL, "tool_unknown", "no such tool",
             Recoverability.AUTO_RECOVER, "tool_schema_expand"),
            # Anything transient falls through to backoff retry.
            (FailureSource.LLM, "transient", "connection reset",
             Recoverability.AUTO_RETRY, "jittered_retry"),
            (FailureSource.SYSTEM, "system_network", "network down",
             Recoverability.AUTO_RETRY, "jittered_retry"),
            (FailureSource.TOOL, "tool_timeout", "timed out",
             Recoverability.AUTO_RETRY, "jittered_retry"),
        ],
    )
    def test_failure_routes_to_expected_strategy(
        self, source, category, message, recoverability, expected_key,
    ) -> None:
        from leapflow.engine.recovery.recovery_budget import RecoveryBudget
        from leapflow.engine.recovery.recovery_coordinator import RecoveryCoordinator

        coord = RecoveryCoordinator(
            strategies=default_strategies(),
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coord.budget.start_deadline()
        envelope = _make_envelope(
            source=source, category=category, message=message,
            recoverability=recoverability, tool_name="some_tool",
        )

        decision = coord.evaluate(envelope)

        assert decision.strategy_key == expected_key
        assert decision.reason, "every decision must carry an explainable reason"

    def test_priorities_are_strictly_increasing(self) -> None:
        """Ties would make routing order depend on registration order."""
        strategies = default_strategies()
        for i in range(len(strategies) - 1):
            assert strategies[i].priority < strategies[i + 1].priority, (
                f"{strategies[i].key} (priority={strategies[i].priority}) should be "
                f"lower than {strategies[i+1].key} (priority={strategies[i+1].priority})"
            )

    def test_only_idempotent_strategies_are_repeatable(self) -> None:
        """Repeatable strategies must be safe to re-apply within one turn.

        Compression advances through phases, compression_timeout escalates
        through cooldown tiers, and jittered retry backs off — all three
        converge. Every other strategy mutates provider/credential/mode
        state and must fire at most once per turn.
        """
        strategies = default_strategies()
        repeatable = {s.key for s in strategies if s.repeatable}
        assert repeatable == {"context_compress", "compression_timeout", "jittered_retry"}

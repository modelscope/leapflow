"""Comprehensive tests for the P0 recovery coordinator subsystem.

Covers:
- FailureEnvelope construction and immutability
- RecoveryDecision construction and properties
- RecoveryBudget: can_afford, consume, deadline, per-category limits
- OneShotGuard: is_available, mark_used, idempotency
- RecoveryCoordinator: evaluate with mock strategies, priority ordering,
  one-shot enforcement, terminal decision, budget exhaustion
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    RecoveryHint,
    SideEffectState,
)
from leapflow.engine.oneshot_guard import OneShotGuard
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import (
    RecoveryCoordinator,
    RecoveryState,
    RecoveryStrategy,
)
from leapflow.engine.recovery_decision import (
    BackoffConfig,
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)


# ---------------------------------------------------------------------------
# Test helpers: mock strategies implementing the Protocol
# ---------------------------------------------------------------------------


@dataclass
class MockRetryStrategy:
    """A mock strategy that always suggests retry with backoff."""

    key: str = "mock_retry"
    priority: int = 10
    repeatable: bool = True
    applicable_sources: frozenset[str] = frozenset({"llm"})
    applicable_categories: frozenset[str] = frozenset({"rate_limited", "transient"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        return state.consecutive_failures < 5

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            reason="Transient error, retrying",
            strategy_key=self.key,
            retry_semantics=RetrySemantics(
                consumes_retry_budget=True,
                backoff_config=BackoffConfig(base_delay=1.0),
            ),
            budget_cost=1,
        )


@dataclass
class MockCompressStrategy:
    """A mock strategy that suggests transform-and-retry for context overflow."""

    key: str = "mock_compress"
    priority: int = 5
    repeatable: bool = True
    applicable_sources: frozenset[str] = frozenset()
    applicable_categories: frozenset[str] = frozenset({"context_overflow"})

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        return True

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Context overflow, compressing",
            strategy_key=self.key,
            budget_cost=1,
        )


@dataclass
class MockFailoverStrategy:
    """A mock strategy that suggests failover."""

    key: str = "mock_failover"
    priority: int = 20
    repeatable: bool = False
    applicable_sources: frozenset[str] = frozenset({"llm"})
    applicable_categories: frozenset[str] = frozenset()

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        return state.consecutive_failures >= 3

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.FAILOVER,
            reason="Too many consecutive failures, switching model",
            strategy_key=self.key,
            budget_cost=2,
        )


@dataclass
class MockCatchAllStrategy:
    """Low priority strategy that matches anything."""

    key: str = "mock_catchall"
    priority: int = 100
    repeatable: bool = True
    applicable_sources: frozenset[str] = frozenset()
    applicable_categories: frozenset[str] = frozenset()

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        return True

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.SKIP_AND_CONTINUE,
            reason="Catch-all: skip and continue",
            strategy_key=self.key,
            budget_cost=0,
        )


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    *,
    source: FailureSource = FailureSource.LLM,
    category: str = "rate_limited",
    recoverability: Recoverability = Recoverability.AUTO_RETRY,
    side_effect_state: SideEffectState = SideEffectState.NONE,
) -> FailureEnvelope:
    return FailureEnvelope.create(
        source=source,
        category=category,
        failure_class="transient",
        failure_code="rate_limit_exceeded",
        message="Rate limit exceeded",
        recoverability=recoverability,
        side_effect_state=side_effect_state,
    )


# ===========================================================================
# FailureEnvelope Tests
# ===========================================================================


class TestFailureEnvelope:
    def test_create_generates_id_and_timestamp(self) -> None:
        env = _make_envelope()
        assert len(env.envelope_id) == 32
        assert env.timestamp > 0
        assert env.source == FailureSource.LLM
        assert env.category == "rate_limited"

    def test_frozen_immutability(self) -> None:
        env = _make_envelope()
        with pytest.raises(AttributeError):
            env.message = "changed"  # type: ignore[misc]

    def test_failure_context_from_dict_args(self) -> None:
        ctx = FailureContext.from_dict_args(
            tool_name="shell_run",
            arguments={"cmd": "ls", "timeout": 30},
            execution_id="exec-1",
            turn_id=5,
        )
        assert ctx.tool_name == "shell_run"
        assert ctx.arguments_dict == {"cmd": "ls", "timeout": 30}
        assert ctx.turn_id == 5

    def test_failure_context_empty(self) -> None:
        ctx = FailureContext()
        assert ctx.arguments == ()
        assert ctx.arguments_dict == {}

    def test_recovery_hint(self) -> None:
        hint = RecoveryHint(
            hint_text="Retry after 60s",
            suggested_command="leap config llm key",
        )
        assert hint.hint_text == "Retry after 60s"
        assert hint.documentation_url == ""

    def test_envelope_with_context_and_hint(self) -> None:
        ctx = FailureContext.from_dict_args(tool_name="gp_web_search")
        hint = RecoveryHint(hint_text="Check API key")
        env = FailureEnvelope.create(
            source=FailureSource.TOOL,
            category="auth_error",
            failure_class="authorization",
            failure_code="missing_scope",
            message="Missing read scope",
            recoverability=Recoverability.USER_FIXABLE,
            context=ctx,
            provider_hint=hint,
        )
        assert env.context.tool_name == "gp_web_search"
        assert env.provider_hint is not None
        assert env.provider_hint.hint_text == "Check API key"


# ===========================================================================
# RecoveryDecision Tests
# ===========================================================================


class TestRecoveryDecision:
    def test_create_generates_id(self) -> None:
        env = _make_envelope()
        dec = RecoveryDecision.create(
            envelope=env,
            action=RecoveryAction.RETRY_WITH_BACKOFF,
            reason="Transient error",
            strategy_key="test_strategy",
            budget_cost=1,
        )
        assert len(dec.decision_id) == 32
        assert dec.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert dec.strategy_key == "test_strategy"

    def test_is_terminal(self) -> None:
        env = _make_envelope()
        dec = RecoveryDecision.create(
            envelope=env,
            action=RecoveryAction.HALT_CLEAN,
            reason="Budget exhausted",
            strategy_key="test",
        )
        assert dec.is_terminal is True
        assert dec.is_retry is False

    def test_is_retry(self) -> None:
        env = _make_envelope()
        dec = RecoveryDecision.create(
            envelope=env,
            action=RecoveryAction.TRANSFORM_AND_RETRY,
            reason="Compressing",
            strategy_key="compress",
        )
        assert dec.is_retry is True
        assert dec.is_terminal is False

    def test_audit_metadata_roundtrip(self) -> None:
        env = _make_envelope()
        dec = RecoveryDecision.create(
            envelope=env,
            action=RecoveryAction.SKIP_AND_CONTINUE,
            reason="Skip",
            strategy_key="skip",
            audit_metadata={"attempt": 3, "source": "test"},
        )
        assert dec.audit_metadata_dict == {"attempt": 3, "source": "test"}

    def test_backoff_config_frozen(self) -> None:
        cfg = BackoffConfig(base_delay=2.0, max_delay=120.0)
        with pytest.raises(AttributeError):
            cfg.base_delay = 5.0  # type: ignore[misc]

    def test_retry_semantics_defaults(self) -> None:
        sem = RetrySemantics()
        assert sem.consumes_retry_budget is True
        assert sem.resets_retry_count is False
        assert sem.backoff_config is None


# ===========================================================================
# RecoveryBudget Tests
# ===========================================================================


class TestRecoveryBudget:
    def test_initial_state(self) -> None:
        budget = RecoveryBudget()
        assert budget.remaining() == 12
        assert budget.can_afford(1) is True
        assert budget.is_deadline_exceeded() is False

    def test_consume_reduces_remaining(self) -> None:
        budget = RecoveryBudget(total_recovery_actions=5)
        budget.start_deadline()
        budget.consume(2)
        assert budget.remaining() == 3
        budget.consume(3)
        assert budget.remaining() == 0
        assert budget.can_afford(1) is False

    def test_consume_raises_on_overbudget(self) -> None:
        budget = RecoveryBudget(total_recovery_actions=3)
        budget.start_deadline()
        budget.consume(3)
        with pytest.raises(ValueError, match="Recovery budget exceeded"):
            budget.consume(1)

    def test_per_category_limits(self) -> None:
        budget = RecoveryBudget(max_retry_per_category=2, total_recovery_actions=10)
        budget.start_deadline()
        assert budget.category_remaining("rate_limited") == 2
        budget.consume(1, "rate_limited")
        assert budget.category_remaining("rate_limited") == 1
        budget.consume(1, "rate_limited")
        assert budget.category_remaining("rate_limited") == 0
        assert budget.can_afford(1, "rate_limited") is False
        # Other category is still available
        assert budget.can_afford(1, "transient") is True

    def test_deadline_exceeded(self) -> None:
        budget = RecoveryBudget(turn_deadline_s=0.01)
        budget.start_deadline()
        time.sleep(0.02)
        assert budget.is_deadline_exceeded() is True
        assert budget.can_afford(1) is False

    def test_deadline_not_started(self) -> None:
        budget = RecoveryBudget(turn_deadline_s=0.001)
        # Deadline not started — should never be exceeded
        assert budget.is_deadline_exceeded() is False

    def test_zero_deadline_is_unlimited(self) -> None:
        # A non-positive deadline means unlimited wall-clock time (P0'): a long
        # task is never denied recovery for a late error; the action-count budget
        # remains the bound.
        budget = RecoveryBudget(turn_deadline_s=0.0)
        budget.start_deadline()
        time.sleep(0.02)
        assert budget.is_deadline_exceeded() is False
        assert budget.can_afford(1) is True

    def test_transform_failover_rotation_tracking(self) -> None:
        budget = RecoveryBudget(
            max_transform_attempts=1,
            max_failovers=1,
            max_credential_rotations=1,
        )
        assert budget.can_transform() is True
        budget.consume_transform()
        assert budget.can_transform() is False

        assert budget.can_failover() is True
        budget.consume_failover()
        assert budget.can_failover() is False

        assert budget.can_rotate() is True
        budget.consume_rotation()
        assert budget.can_rotate() is False

    def test_summary(self) -> None:
        budget = RecoveryBudget(total_recovery_actions=10)
        budget.start_deadline()
        budget.consume(2, "rate_limited")
        budget.consume_transform()
        summary = budget.summary()
        assert summary["consumed"] == 2
        assert summary["remaining"] == 8
        assert summary["category_consumed"] == {"rate_limited": 2}
        assert summary["transforms_used"] == 1


# ===========================================================================
# OneShotGuard Tests
# ===========================================================================


class TestOneShotGuard:
    def test_fresh_guard_all_available(self) -> None:
        guard = OneShotGuard()
        assert guard.is_available("compress") is True
        assert guard.is_available("failover") is True
        assert len(guard) == 0

    def test_mark_used_makes_unavailable(self) -> None:
        guard = OneShotGuard()
        guard.mark_used("compress")
        assert guard.is_available("compress") is False
        assert guard.is_available("failover") is True
        assert len(guard) == 1

    def test_mark_used_is_idempotent(self) -> None:
        guard = OneShotGuard()
        guard.mark_used("compress")
        ts1 = guard.usage_history()["compress"]
        time.sleep(0.01)
        guard.mark_used("compress")
        ts2 = guard.usage_history()["compress"]
        # Timestamp should not change on re-marking
        assert ts1 == ts2

    def test_used_strategies_ordered_by_time(self) -> None:
        guard = OneShotGuard()
        guard.mark_used("a")
        time.sleep(0.01)
        guard.mark_used("b")
        strategies = guard.used_strategies()
        assert strategies == ["a", "b"]

    def test_contains_operator(self) -> None:
        guard = OneShotGuard()
        assert "x" not in guard
        guard.mark_used("x")
        assert "x" in guard

    def test_reset_clears_all(self) -> None:
        guard = OneShotGuard()
        guard.mark_used("a")
        guard.mark_used("b")
        guard.reset()
        assert guard.is_available("a") is True
        assert len(guard) == 0


# ===========================================================================
# RecoveryCoordinator Tests
# ===========================================================================


class TestRecoveryCoordinator:
    def test_evaluate_matches_strategy_by_source_and_category(self) -> None:
        strategies: list[RecoveryStrategy] = [MockRetryStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope(source=FailureSource.LLM, category="rate_limited")
        decision = coord.evaluate(env)

        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.strategy_key == "mock_retry"

    def test_evaluate_priority_ordering(self) -> None:
        """Higher priority (lower number) strategies are evaluated first."""
        # MockCompressStrategy has priority 5, MockRetryStrategy has 10
        strategies: list[RecoveryStrategy] = [
            MockRetryStrategy(),
            MockCompressStrategy(),
        ]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        # context_overflow matches compress (priority 5) before retry
        env = _make_envelope(category="context_overflow")
        decision = coord.evaluate(env)
        assert decision.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert decision.strategy_key == "mock_compress"

    def test_evaluate_skips_non_matching_source(self) -> None:
        """Strategy with source filter skips non-matching envelopes."""
        strategies: list[RecoveryStrategy] = [
            MockRetryStrategy(),  # only matches LLM source
            MockCatchAllStrategy(),
        ]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope(source=FailureSource.TOOL, category="rate_limited")
        decision = coord.evaluate(env)
        # Falls through to catch-all
        assert decision.strategy_key == "mock_catchall"

    def test_evaluate_one_shot_enforcement(self) -> None:
        """Failover marks one-shot, subsequent calls skip it."""
        strategies: list[RecoveryStrategy] = [
            MockFailoverStrategy(),
            MockCatchAllStrategy(),
        ]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()
        # Set consecutive failures high enough for failover
        coord.state.consecutive_failures = 5

        env = _make_envelope(source=FailureSource.LLM, category="transient")
        decision1 = coord.evaluate(env)
        assert decision1.action == RecoveryAction.FAILOVER

        # Second call should skip failover (one-shot used) and fall to catch-all
        coord.state.consecutive_failures = 5
        decision2 = coord.evaluate(env)
        assert decision2.strategy_key == "mock_catchall"

    def test_evaluate_terminal_on_no_match(self) -> None:
        """When no strategy matches, a terminal HALT_CLEAN decision is returned."""
        coord = RecoveryCoordinator(strategies=[])
        coord.budget.start_deadline()

        env = _make_envelope()
        decision = coord.evaluate(env)
        assert decision.action == RecoveryAction.HALT_CLEAN
        assert "No applicable" in decision.reason

    def test_evaluate_terminal_on_budget_exhaustion(self) -> None:
        """When budget is exhausted, terminal decision regardless of strategies."""
        budget = RecoveryBudget(total_recovery_actions=0)
        budget.start_deadline()
        strategies: list[RecoveryStrategy] = [MockRetryStrategy()]
        coord = RecoveryCoordinator(strategies=strategies, budget=budget)

        env = _make_envelope()
        decision = coord.evaluate(env)
        assert decision.action == RecoveryAction.HALT_CLEAN
        assert "exhausted" in decision.reason.lower()

    def test_evaluate_terminal_on_deadline_exceeded(self) -> None:
        """When deadline is exceeded, terminal decision immediately."""
        budget = RecoveryBudget(turn_deadline_s=0.01)
        budget.start_deadline()
        time.sleep(0.02)
        strategies: list[RecoveryStrategy] = [MockRetryStrategy()]
        coord = RecoveryCoordinator(strategies=strategies, budget=budget)

        env = _make_envelope()
        decision = coord.evaluate(env)
        assert decision.action == RecoveryAction.HALT_CLEAN
        assert "deadline" in decision.reason.lower()

    def test_evaluate_non_recoverable_short_circuits(self) -> None:
        """Non-recoverable envelopes skip all strategies."""
        strategies: list[RecoveryStrategy] = [MockCatchAllStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope(recoverability=Recoverability.NON_RECOVERABLE)
        decision = coord.evaluate(env)
        assert decision.action == RecoveryAction.HALT_CLEAN
        assert "non-recoverable" in decision.reason.lower()

    def test_evaluate_budget_cost_check(self) -> None:
        """Strategy with cost > remaining budget is skipped."""
        budget = RecoveryBudget(total_recovery_actions=1)
        budget.start_deadline()
        # MockFailoverStrategy costs 2
        strategies: list[RecoveryStrategy] = [
            MockFailoverStrategy(),
            MockCatchAllStrategy(),  # costs 0
        ]
        coord = RecoveryCoordinator(strategies=strategies, budget=budget)
        coord.state.consecutive_failures = 5

        env = _make_envelope(source=FailureSource.LLM)
        decision = coord.evaluate(env)
        # Failover costs 2 but budget only has 1, should skip to catch-all
        assert decision.strategy_key == "mock_catchall"

    def test_on_strategy_outcome_success_resets_counters(self) -> None:
        coord = RecoveryCoordinator()
        coord.state.consecutive_failures = 3
        coord.state.consecutive_api_errors = 2
        coord.on_strategy_outcome("dec-1", success=True)
        assert coord.state.consecutive_failures == 0
        assert coord.state.consecutive_api_errors == 0

    def test_on_strategy_outcome_failure_increments(self) -> None:
        coord = RecoveryCoordinator()
        coord.state.consecutive_failures = 2
        coord.on_strategy_outcome("dec-1", success=False)
        assert coord.state.consecutive_failures == 3

    def test_record_success(self) -> None:
        coord = RecoveryCoordinator()
        coord.state.consecutive_failures = 5
        coord.state.consecutive_api_errors = 3
        coord.state.last_error_category = "rate_limited"
        coord.record_success()
        assert coord.state.consecutive_failures == 0
        assert coord.state.consecutive_api_errors == 0
        assert coord.state.last_error_category == ""

    def test_audit_log_populated(self) -> None:
        strategies: list[RecoveryStrategy] = [MockCatchAllStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope()
        coord.evaluate(env)
        assert len(coord.audit_log) == 1
        entry = coord.audit_log[0]
        assert entry["event"] == "recovery_decision"
        assert entry["strategy_key"] == "mock_catchall"
        assert "budget_remaining" in entry

    def test_protocol_compliance(self) -> None:
        """Mock strategies satisfy the RecoveryStrategy protocol."""
        assert isinstance(MockRetryStrategy(), RecoveryStrategy)
        assert isinstance(MockCompressStrategy(), RecoveryStrategy)
        assert isinstance(MockFailoverStrategy(), RecoveryStrategy)
        assert isinstance(MockCatchAllStrategy(), RecoveryStrategy)

    def test_multiple_evaluations_consume_budget(self) -> None:
        """Multiple evaluations consume budget incrementally."""
        budget = RecoveryBudget(total_recovery_actions=3)
        budget.start_deadline()
        strategies: list[RecoveryStrategy] = [MockRetryStrategy()]
        coord = RecoveryCoordinator(strategies=strategies, budget=budget)

        env = _make_envelope()
        d1 = coord.evaluate(env)
        assert d1.action == RecoveryAction.RETRY_WITH_BACKOFF
        d2 = coord.evaluate(env)
        assert d2.action == RecoveryAction.RETRY_WITH_BACKOFF
        d3 = coord.evaluate(env)
        assert d3.action == RecoveryAction.RETRY_WITH_BACKOFF
        # Budget now exhausted
        d4 = coord.evaluate(env)
        assert d4.action == RecoveryAction.HALT_CLEAN

    def test_new_turn_resets_guard_and_state(self) -> None:
        """new_turn() resets one-shot guard and per-turn state."""
        strategies: list[RecoveryStrategy] = [MockFailoverStrategy(), MockCatchAllStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()
        coord.state.consecutive_failures = 5

        env = _make_envelope(source=FailureSource.LLM, category="transient")
        d1 = coord.evaluate(env)
        assert d1.action == RecoveryAction.FAILOVER

        # Failover is one-shot, should fall to catch-all
        coord.state.consecutive_failures = 5
        d2 = coord.evaluate(env)
        assert d2.strategy_key == "mock_catchall"

        # After new_turn, failover should be available again
        coord.new_turn(turn_id=2)
        coord.state.consecutive_failures = 5
        d3 = coord.evaluate(env)
        assert d3.action == RecoveryAction.FAILOVER
        assert coord.state.current_turn_id == 2

    def test_non_repeatable_transform_is_one_shot(self) -> None:
        """Non-repeatable TRANSFORM_AND_RETRY strategies fire only once."""

        @dataclass
        class MockTransformStrategy:
            key: str = "mock_transform"
            priority: int = 5
            repeatable: bool = False
            applicable_sources: frozenset[str] = frozenset()
            applicable_categories: frozenset[str] = frozenset({"format_error"})

            def can_apply(self, envelope, state, budget=None):
                return True

            def decide(self, envelope, state):
                return RecoveryDecision.create(
                    envelope=envelope,
                    action=RecoveryAction.TRANSFORM_AND_RETRY,
                    reason="Transform once",
                    strategy_key=self.key,
                    budget_cost=0,
                )

        strategies: list[RecoveryStrategy] = [MockTransformStrategy(), MockCatchAllStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope(category="format_error")
        d1 = coord.evaluate(env)
        assert d1.action == RecoveryAction.TRANSFORM_AND_RETRY
        assert d1.strategy_key == "mock_transform"

        # Second call should skip transform (one-shot marked) and fall to catch-all
        d2 = coord.evaluate(env)
        assert d2.strategy_key == "mock_catchall"

    def test_repeatable_strategy_fires_multiple_times(self) -> None:
        """Repeatable strategies can fire multiple times per turn."""
        strategies: list[RecoveryStrategy] = [MockRetryStrategy()]
        coord = RecoveryCoordinator(strategies=strategies)
        coord.budget.start_deadline()

        env = _make_envelope(source=FailureSource.LLM, category="rate_limited")
        d1 = coord.evaluate(env)
        assert d1.strategy_key == "mock_retry"
        d2 = coord.evaluate(env)
        assert d2.strategy_key == "mock_retry"
        d3 = coord.evaluate(env)
        assert d3.strategy_key == "mock_retry"

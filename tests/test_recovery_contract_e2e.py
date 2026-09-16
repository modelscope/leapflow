# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end guards for the recovery contracts in AGENTS.md.

The existing recovery tests are unit-level: they exercise one budget method or
one strategy's ``decide()`` in isolation. These tests instead drive the whole
documented pipeline — ``FailureEnvelope`` → ``RecoveryDecision`` →
``StrategyOutcome`` — through a real ``RecoveryCoordinator`` with the real
default strategy registry, which is where a contract actually holds or breaks.

Covered contracts:
- Single Recovery Decision Point (one coordinator, explainable decisions)
- Budget-Constrained Recovery (per-category, global, deadline → clean halt)
- Side-Effect Gating (committed/partial/unknown must block automatic retry)
"""

from __future__ import annotations

import time

import pytest

from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.recovery_decision import RecoveryAction
from leapflow.engine.recovery_strategies import default_strategies

# Actions that re-run work and can therefore duplicate an already-applied effect.
_AUTOMATIC_RETRY_ACTIONS = frozenset({
    RecoveryAction.RETRY_WITH_BACKOFF,
    RecoveryAction.TRANSFORM_AND_RETRY,
    RecoveryAction.FAILOVER,
})


def _coordinator(**budget_kwargs) -> RecoveryCoordinator:
    """Build a coordinator over the real default strategies."""
    budget = RecoveryBudget(**budget_kwargs) if budget_kwargs else RecoveryBudget()
    coord = RecoveryCoordinator(strategies=default_strategies(), budget=budget)
    coord.budget.start_deadline()
    return coord


def _envelope(
    *,
    source: FailureSource = FailureSource.LLM,
    category: str = "transient",
    message: str = "connection reset by peer",
    recoverability: Recoverability = Recoverability.AUTO_RETRY,
    side_effect_state: SideEffectState = SideEffectState.NONE,
    tool_name: str = "",
) -> FailureEnvelope:
    return FailureEnvelope.create(
        source=source,
        category=category,
        failure_class="transient",
        failure_code="reset",
        message=message,
        recoverability=recoverability,
        side_effect_state=side_effect_state,
        context=FailureContext.from_dict_args(tool_name=tool_name),
    )


# ── Single Recovery Decision Point ───────────────────────────────────────


def test_every_decision_is_explainable_and_attributed() -> None:
    """A decision without a reason or owning strategy cannot be audited."""
    coord = _coordinator(total_recovery_actions=16)

    for source, category in (
        (FailureSource.LLM, "context_overflow"),
        (FailureSource.LLM, "transient"),
        (FailureSource.TOOL, "tool_timeout"),
        (FailureSource.SYSTEM, "system_network"),
    ):
        decision = coord.evaluate(_envelope(source=source, category=category))

        assert decision.reason, f"{source}/{category} produced an unexplained decision"
        assert decision.strategy_key, f"{source}/{category} produced an unattributed decision"
        assert decision.decision_id, "decisions must be identifiable for outcome feedback"


def test_decisions_are_audited_with_budget_accounting() -> None:
    """Every evaluation appends one audit entry carrying its cost accounting."""
    coord = _coordinator(total_recovery_actions=8)

    coord.evaluate(_envelope())
    coord.evaluate(_envelope(category="context_overflow"))

    assert len(coord.audit_log) == 2
    for entry in coord.audit_log:
        assert entry["event"] == "recovery_decision"
        assert entry["strategy_key"]
        assert entry["reason"]
        # Cost accounting must be present so budget drift is diagnosable.
        assert "budget_cost" in entry and "budget_remaining" in entry


def test_outcome_feedback_closes_the_loop() -> None:
    """StrategyOutcome feedback is what lets the next decision differ.

    Without it the coordinator would evaluate every failure against a stale
    state and keep picking the same strategy.
    """
    coord = _coordinator()
    decision = coord.evaluate(_envelope())

    coord.on_strategy_outcome(decision.decision_id, success=False)
    assert coord.state.consecutive_failures == 1

    coord.on_strategy_outcome(decision.decision_id, success=True)
    assert coord.state.consecutive_failures == 0, "success must reset the failure streak"


def test_non_recoverable_failure_never_reaches_a_strategy() -> None:
    """Non-recoverable failures short-circuit to a terminal decision."""
    coord = _coordinator()

    decision = coord.evaluate(_envelope(recoverability=Recoverability.NON_RECOVERABLE))

    assert decision.is_terminal
    assert decision.action == RecoveryAction.HALT_CLEAN
    assert "non-recoverable" in decision.reason.lower()


# ── Budget-Constrained Recovery ──────────────────────────────────────────


def test_per_category_exhaustion_halts_deterministically() -> None:
    """A single failure category cannot consume unbounded recovery attempts."""
    coord = _coordinator(total_recovery_actions=10, max_retry_per_category=2)

    first = coord.evaluate(_envelope())
    second = coord.evaluate(_envelope())
    third = coord.evaluate(_envelope())

    assert first.action in _AUTOMATIC_RETRY_ACTIONS
    assert second.action in _AUTOMATIC_RETRY_ACTIONS
    assert third.action == RecoveryAction.HALT_CLEAN, "category budget must stop retrying"
    # A different category is still serviceable: the limit is per-category.
    assert coord.evaluate(_envelope(category="tool_timeout", source=FailureSource.TOOL)).action \
        in _AUTOMATIC_RETRY_ACTIONS


def test_global_budget_exhaustion_halts_and_stays_halted() -> None:
    """Exhaustion is terminal and idempotent — never a flapping decision."""
    coord = _coordinator(total_recovery_actions=2, max_retry_per_category=99)

    coord.evaluate(_envelope())
    coord.evaluate(_envelope())
    halted = [coord.evaluate(_envelope()) for _ in range(3)]

    assert all(d.action == RecoveryAction.HALT_CLEAN for d in halted)
    assert all("exhausted" in d.reason.lower() for d in halted)
    # Terminal decisions must not keep charging the budget.
    assert coord.budget.remaining() == 0


def test_deadline_exceeded_halts_before_any_strategy_runs() -> None:
    """A turn that ran out of wall-clock time stops instead of retrying."""
    budget = RecoveryBudget(turn_deadline_s=0.01)
    budget.start_deadline()
    coord = RecoveryCoordinator(strategies=default_strategies(), budget=budget)
    time.sleep(0.02)

    decision = coord.evaluate(_envelope())

    assert decision.action == RecoveryAction.HALT_CLEAN
    assert "deadline" in decision.reason.lower()


def test_exhausted_budget_reports_what_was_attempted() -> None:
    """The halt reason must name the attempts, or the user learns nothing.

    Uses a budget-consuming category on purpose: transform strategies such as
    ``context_compress`` cost 0, so they would never exhaust the budget.
    """
    coord = _coordinator(total_recovery_actions=1)

    coord.evaluate(_envelope(category="transient"))
    halt = coord.evaluate(_envelope(category="transient"))

    assert halt.action == RecoveryAction.HALT_CLEAN
    assert "exhausted" in halt.reason.lower()
    assert "jittered_retry" in halt.reason or "attempted" in halt.reason.lower()


def test_new_turn_restores_one_shot_strategies() -> None:
    """One-shot strategies are per-turn, not per-process."""
    coord = _coordinator(total_recovery_actions=16)

    first = coord.evaluate(_envelope(category="billing"))
    assert first.strategy_key == "provider_failover"
    # Same turn: the one-shot failover is spent, so something else must answer.
    second = coord.evaluate(_envelope(category="billing"))
    assert second.strategy_key != "provider_failover"

    coord.new_turn(turn_id=2)
    assert coord.evaluate(_envelope(category="billing")).strategy_key == "provider_failover"


# ── Side-Effect Gating ───────────────────────────────────────────────────


def test_read_only_failure_is_freely_retryable() -> None:
    """The baseline: with no side effect, automatic retry is correct."""
    coord = _coordinator(total_recovery_actions=16)

    decision = coord.evaluate(_envelope(side_effect_state=SideEffectState.NONE))

    assert decision.action in _AUTOMATIC_RETRY_ACTIONS


@pytest.mark.parametrize(
    "side_effect_state",
    [SideEffectState.COMMITTED, SideEffectState.PARTIAL, SideEffectState.UNKNOWN],
)
def test_applied_side_effects_block_automatic_retry(side_effect_state) -> None:
    """Contract: mutated state permits only user-mediated/checkpoint resumption.

    Retrying after an effect landed can send a message twice, write a file
    twice, or re-charge an external API. ``UNKNOWN`` is included because it is
    what ``external_side_effect`` maps to — exempting it would leave outbound
    sends ungated.
    """
    coord = _coordinator(total_recovery_actions=16)

    decision = coord.evaluate(
        _envelope(
            source=FailureSource.TOOL,
            category="tool_timeout",
            side_effect_state=side_effect_state,
            tool_name="gateway_send",
        )
    )

    assert decision.action not in _AUTOMATIC_RETRY_ACTIONS, (
        f"{side_effect_state.name} side effects must not be retried automatically; "
        f"got {decision.action.name} from {decision.strategy_key}"
    )
    assert decision.action == RecoveryAction.HALT_WITH_CHECKPOINT
    # A gated halt must tell the user what to check, not just stop.
    assert decision.interaction is not None
    assert decision.interaction.resumption_key
    assert decision.interaction.suggested_actions
    assert "gateway_send" in decision.reason


def test_side_effect_gate_does_not_consume_recovery_budget() -> None:
    """Being blocked is not an attempt; it must not eat the turn's budget."""
    coord = _coordinator(total_recovery_actions=4)
    before = coord.budget.remaining()

    decision = coord.evaluate(
        _envelope(side_effect_state=SideEffectState.COMMITTED, tool_name="file_write")
    )

    assert decision.action == RecoveryAction.HALT_WITH_CHECKPOINT
    assert coord.budget.remaining() == before


def test_side_effect_gate_is_audited() -> None:
    """A withheld retry must be visible in the audit trail."""
    coord = _coordinator(total_recovery_actions=8)

    coord.evaluate(_envelope(side_effect_state=SideEffectState.PARTIAL, tool_name="scm_sync"))

    assert coord.audit_log[-1]["strategy_key"] == "<side_effect_gated>"


def test_side_effect_state_survives_the_envelope_roundtrip() -> None:
    """Whatever the gate ends up doing, the signal must reach the decision.

    Guards the input half of the gap above: the classifier's judgement has to be
    observable on the envelope the coordinator receives.
    """
    envelope = _envelope(side_effect_state=SideEffectState.COMMITTED)

    assert envelope.side_effect_state is SideEffectState.COMMITTED

    coord = _coordinator()
    decision = coord.evaluate(envelope)
    assert decision.envelope.side_effect_state is SideEffectState.COMMITTED


def test_classifier_maps_external_side_effect_to_a_gated_state() -> None:
    """An outbound external call must never be classified as effect-free."""
    from leapflow.engine.unified_classifier import UnifiedErrorClassifier

    mapped = UnifiedErrorClassifier._side_effect_state_from_policy("external_side_effect")
    idempotent = UnifiedErrorClassifier._side_effect_state_from_policy("mutating_idempotent")
    read_only = UnifiedErrorClassifier._side_effect_state_from_policy("read_only")

    assert read_only is SideEffectState.NONE
    assert mapped is not SideEffectState.NONE, "external side effects must be gated"
    assert idempotent is not SideEffectState.NONE, "mutations must be gated"

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Recovery coordinator — unified recovery decision entry point for the agent loop.

Replaces the scattered if/elif chains in _handle_api_error() and the inline
break/continue decisions in _unified_tool_loop() with a strategy-based,
budget-aware, auditable coordinator.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from leapflow.engine.failure_envelope import FailureEnvelope, Recoverability, SideEffectState
from leapflow.engine.interaction_request import (
    InteractionRequest,
    InteractionType,
    Severity,
    SuggestedAction,
    TimeoutBehavior,
)
from leapflow.engine.oneshot_guard import OneShotGuard
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)

logger = logging.getLogger(__name__)

# Actions that re-run work already attempted. Safe only when the failed attempt
# left no trace; after a mutation they can duplicate the effect.
_REPLAYING_ACTIONS = frozenset({
    RecoveryAction.RETRY_WITH_BACKOFF,
    RecoveryAction.TRANSFORM_AND_RETRY,
    RecoveryAction.FAILOVER,
})


@dataclass
class RecoveryState:
    """Shared mutable state visible to all strategies during a turn.

    Strategies may read and update this state to coordinate across multiple
    failures within the same turn (e.g. tracking consecutive failures to
    escalate from retry to halt).
    """

    consecutive_failures: int = 0
    consecutive_api_errors: int = 0
    last_error_category: str = ""
    compress_phase_index: int = 0
    last_failure_timestamp: float = 0.0
    total_decisions: int = 0
    current_turn_id: int = 0


@runtime_checkable
class RecoveryStrategy(Protocol):
    """Protocol for pluggable recovery strategies.

    Strategies are registered with the coordinator and evaluated in priority
    order. Each strategy encapsulates domain logic for a specific failure
    pattern (e.g. "retry on rate limit", "compress on context overflow").
    """

    @property
    def key(self) -> str:
        """Unique identifier for this strategy (used by OneShotGuard)."""
        ...

    @property
    def priority(self) -> int:
        """Lower number = higher priority. Evaluated first."""
        ...

    @property
    def repeatable(self) -> bool:
        """Whether this strategy can fire multiple times per turn.

        Repeatable strategies manage their own deduplication (e.g. phase index)
        and are NOT guarded by the OneShotGuard. Non-repeatable strategies
        fire at most once per turn and are automatically one-shot guarded.
        """
        ...

    @property
    def applicable_sources(self) -> frozenset[str]:
        """Set of FailureSource values this strategy handles. Empty = all."""
        ...

    @property
    def applicable_categories(self) -> frozenset[str]:
        """Set of error category strings this strategy handles. Empty = all."""
        ...

    def can_apply(self, envelope: FailureEnvelope, state: RecoveryState,
                  budget: RecoveryBudget | None = None) -> bool:
        """Whether this strategy can handle the given failure envelope."""
        ...

    def decide(self, envelope: FailureEnvelope, state: RecoveryState) -> RecoveryDecision:
        """Produce a recovery decision for the given failure."""
        ...


class RecoveryCoordinator:
    """Unified recovery decision entry point for the agent loop.

    Orchestrates strategy evaluation, one-shot enforcement, budget tracking,
    and audit logging. The agent loop calls evaluate() on each failure and
    receives a deterministic RecoveryDecision.
    """

    def __init__(
        self,
        strategies: list[RecoveryStrategy] | None = None,
        budget: RecoveryBudget | None = None,
    ) -> None:
        self._strategies: list[RecoveryStrategy] = sorted(
            strategies or [], key=lambda s: s.priority
        )
        self._budget = budget or RecoveryBudget()
        self._guard = OneShotGuard()
        self._state = RecoveryState()
        self._audit: list[dict[str, Any]] = []

    def evaluate(self, envelope: FailureEnvelope) -> RecoveryDecision:
        """Main entry point: evaluate a failure and return a recovery decision.

        Pipeline:
        1. Check budget deadline — if exceeded, emit terminal decision.
        2. Iterate strategies by priority.
        3. For each: check source/category match → one-shot guard → can_apply → decide.
        4. If none applies: return terminal decision.
        """
        # Deadline check
        if self._budget.is_deadline_exceeded():
            reason = (
                f"Turn deadline exceeded ({self._budget.turn_deadline_s}s). "
                f"Strategies attempted: {', '.join(self._guard.used_strategies()) or 'none'}."
            )
            decision = self._make_terminal(envelope, reason=reason)
            self._record_audit(decision, strategy_key="<deadline>")
            return decision

        # Global budget exhaustion
        if self._budget.remaining() == 0:
            reason = (
                f"Recovery budget exhausted ({self._budget._consumed}/{self._budget.total_recovery_actions}). "
                f"Strategies attempted: {', '.join(self._guard.used_strategies()) or 'none'}. "
                f"Last error: {envelope.category}/{envelope.failure_code}."
            )
            decision = self._make_terminal(envelope, reason=reason)
            self._record_audit(decision, strategy_key="<budget_exhausted>")
            return decision

        # Non-recoverable failures short-circuit
        if envelope.recoverability == Recoverability.NON_RECOVERABLE:
            reason = (
                f"Failure is non-recoverable: {envelope.message[:100]}. "
                f"Category: {envelope.category}, code: {envelope.failure_code}."
            )
            decision = self._make_terminal(envelope, reason=reason)
            self._record_audit(decision, strategy_key="<non_recoverable>")
            return decision

        # Strategy evaluation
        blocked_by_side_effect: list[str] = []
        for strategy in self._strategies:
            if not self._matches_source(strategy, envelope):
                continue
            if not self._matches_category(strategy, envelope):
                continue
            # One-shot guard: only check for non-repeatable strategies
            if not strategy.repeatable and not self._guard.is_available(strategy.key):
                continue
            if not strategy.can_apply(envelope, self._state, self._budget):
                continue

            # Budget pre-check for the strategy's expected cost
            decision = strategy.decide(envelope, self._state)

            # Side-effect gating: once the attempt may have mutated state, a
            # replaying action can duplicate it. Reject the candidate (rolling
            # back whatever decide() staged) and keep looking for a strategy
            # that does not replay; if none exists, halt with a checkpoint below.
            if self._replay_blocked_by_side_effect(envelope, decision):
                self._rollback_state_changes(decision, strategy)
                blocked_by_side_effect.append(strategy.key)
                continue

            if decision.budget_cost > 0:
                if not self._budget.can_afford(decision.budget_cost, envelope.category):
                    # Rollback any state mutations from decide()
                    self._rollback_state_changes(decision, strategy)
                    continue

            # Commit the decision
            if decision.budget_cost > 0:
                self._budget.consume(decision.budget_cost, envelope.category)

            # Commit state changes AFTER decision is accepted
            self._commit_state_changes(decision, strategy)

            # Track type-specific budget counters
            self._consume_type_budget(decision)

            # Mark one-shot guard for non-repeatable strategies
            if not strategy.repeatable:
                self._guard.mark_used(strategy.key)

            self._update_state(envelope, decision)
            self._record_audit(decision, strategy_key=strategy.key)
            return decision

        if blocked_by_side_effect:
            decision = self._side_effect_halt(envelope, blocked_by_side_effect)
            self._record_audit(decision, strategy_key="<side_effect_gated>")
            return decision

        # No strategy matched
        decision = self.terminal_decision(envelope)
        self._record_audit(decision, strategy_key="<none_matched>")
        return decision

    @staticmethod
    def _replay_blocked_by_side_effect(
        envelope: FailureEnvelope, decision: RecoveryDecision,
    ) -> bool:
        """Return whether the side-effect state forbids this replaying action.

        Every state other than ``NONE`` blocks: ``COMMITTED`` and ``PARTIAL``
        mean state changed, and ``UNKNOWN`` means it may have. ``UNKNOWN`` is
        deliberately included rather than treated as safe — it is both the
        classifier's fallback and the state it assigns to ``external_side_effect``
        (an outbound send, an external API call), so exempting it would leave the
        highest-risk case ungated.
        """
        if envelope.side_effect_state is SideEffectState.NONE:
            return False
        return decision.action in _REPLAYING_ACTIONS

    def _side_effect_halt(
        self, envelope: FailureEnvelope, blocked_strategies: list[str],
    ) -> RecoveryDecision:
        """Halt with a checkpoint because replaying could duplicate an effect.

        Resumption is user-mediated on purpose: only the user (or a read-back of
        the target's state) can settle whether the effect landed.
        """
        tool_name = envelope.context.tool_name or "the operation"
        state_label = envelope.side_effect_state.value
        reason = (
            f"Automatic retry is blocked: {tool_name} failed with side-effect state "
            f"'{state_label}', so replaying it could duplicate the effect. "
            f"Strategies withheld: {', '.join(blocked_strategies)}."
        )
        interaction = InteractionRequest.create(
            interaction_type=InteractionType.RETRY_CHOICE,
            severity=Severity.WARNING,
            title=f"{tool_name} may already have taken effect",
            description=(
                f"It failed with '{envelope.failure_code or envelope.category}', but its "
                f"side-effect state is '{state_label}', so retrying automatically could "
                "apply it twice. Check whether it took effect, then decide how to proceed."
            ),
            suggested_actions=(
                SuggestedAction(
                    label="Verify the target state, then retry if it did not apply",
                    is_default=True,
                ),
                SuggestedAction(label="Abandon this step and continue"),
            ),
            context={
                "tool_name": tool_name,
                "side_effect_state": state_label,
                "failure_code": envelope.failure_code,
                "category": envelope.category,
            },
            resumption_key=envelope.envelope_id,
            timeout_behavior=TimeoutBehavior.PERSIST,
        )
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.HALT_WITH_CHECKPOINT,
            reason=reason,
            strategy_key="<side_effect_gated>",
            retry_semantics=RetrySemantics(consumes_retry_budget=False),
            budget_cost=0,
            interaction=interaction,
        )

    def on_strategy_outcome(self, decision_id: str, success: bool) -> None:
        """Record the outcome of executing a recovery decision.

        Updates consecutive failure counters and appends to audit log.
        """
        if success:
            self._state.consecutive_failures = 0
            self._state.consecutive_api_errors = 0
        else:
            self._state.consecutive_failures += 1

        self._audit.append({
            "event": "strategy_outcome",
            "decision_id": decision_id,
            "success": success,
            "timestamp": time.time(),
            "consecutive_failures": self._state.consecutive_failures,
        })

    def record_success(self) -> None:
        """Record a successful operation (resets failure counts)."""
        self._state.consecutive_failures = 0
        self._state.consecutive_api_errors = 0
        self._state.last_error_category = ""

    def terminal_decision(self, envelope: FailureEnvelope) -> RecoveryDecision:
        """Generate a terminal (halt) decision when no strategy applies."""
        return self._make_terminal(
            envelope,
            reason="No applicable recovery strategy found",
        )

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Return the decision audit trail for this turn."""
        return list(self._audit)

    @property
    def state(self) -> RecoveryState:
        """Access shared recovery state."""
        return self._state

    @property
    def budget(self) -> RecoveryBudget:
        """Access the recovery budget."""
        return self._budget

    @property
    def guard(self) -> OneShotGuard:
        """Access the one-shot guard (primarily for testing/introspection)."""
        return self._guard

    def new_turn(self, turn_id: int = 0) -> None:
        """Reset per-turn state for a new turn. Called at turn start."""
        self._guard.new_turn()
        self._budget.new_turn()
        self._state.compress_phase_index = 0
        self._state.consecutive_failures = 0
        self._state.consecutive_api_errors = 0
        self._state.current_turn_id = turn_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_source(strategy: RecoveryStrategy, envelope: FailureEnvelope) -> bool:
        """Check if the strategy accepts the envelope's failure source."""
        sources = strategy.applicable_sources
        if not sources:
            return True
        return envelope.source.value in sources

    @staticmethod
    def _matches_category(strategy: RecoveryStrategy, envelope: FailureEnvelope) -> bool:
        """Check if the strategy accepts the envelope's error category."""
        categories = strategy.applicable_categories
        if not categories:
            return True
        return envelope.category in categories

    def _update_state(self, envelope: FailureEnvelope, decision: RecoveryDecision) -> None:
        """Update shared state after a decision is committed."""
        self._state.last_error_category = envelope.category
        self._state.last_failure_timestamp = envelope.timestamp
        self._state.total_decisions += 1
        if envelope.source.value == "llm":
            self._state.consecutive_api_errors += 1

    def _commit_state_changes(self, decision: RecoveryDecision, strategy: RecoveryStrategy) -> None:
        """Commit strategy-specific state changes after a decision is accepted."""
        # ContextCompressStrategy phase advancement
        if strategy.key == "context_compress":
            phase_index = decision.audit_metadata_dict.get("phase_index")
            if phase_index is not None:
                self._state.compress_phase_index = phase_index + 1

    def _rollback_state_changes(self, decision: RecoveryDecision, strategy: RecoveryStrategy) -> None:
        """Rollback state mutations from decide() when a decision is rejected."""
        # ContextCompressStrategy phase rollback
        if strategy.key == "context_compress":
            phase_index = decision.audit_metadata_dict.get("phase_index")
            if phase_index is not None:
                self._state.compress_phase_index = phase_index

    def _make_terminal(self, envelope: FailureEnvelope, *, reason: str) -> RecoveryDecision:
        """Create a HALT_CLEAN terminal decision carrying what the user should do.

        A terminal decision is the last thing a stopped turn can say, so it always
        carries an ``InteractionRequest``: the raw ``reason`` is written for the
        audit log and reads as internal jargon ("No applicable recovery strategy
        found") when shown alone. The request names the failure and the next step.
        """
        return RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.HALT_CLEAN,
            reason=reason,
            strategy_key="<terminal>",
            retry_semantics=RetrySemantics(consumes_retry_budget=False),
            budget_cost=0,
            interaction=self._terminal_interaction(envelope, reason),
        )

    def _terminal_interaction(self, envelope: FailureEnvelope, reason: str) -> InteractionRequest:
        """Build the user-facing request that accompanies a terminal halt."""
        hint = getattr(getattr(envelope, "provider_hint", None), "hint_text", "") or ""
        failure = envelope.failure_code or envelope.category or "unknown failure"
        attempted = ", ".join(self._guard.used_strategies())
        description = envelope.message or reason
        if hint:
            description = f"{description} {hint}"
        if attempted:
            description = f"{description} Recovery already tried: {attempted}."
        actions = [
            SuggestedAction(label="Rephrase or narrow the request and try again", is_default=True),
            SuggestedAction(label="Check the daemon log for the full error", command="leap daemon logs"),
        ]
        return InteractionRequest.create(
            interaction_type=InteractionType.RETRY_CHOICE,
            severity=Severity.ERROR,
            title=f"This turn stopped: {failure}",
            description=description,
            suggested_actions=tuple(actions),
            context={
                "category": envelope.category,
                "failure_code": envelope.failure_code,
                "source": getattr(envelope.source, "value", str(envelope.source)),
                "reason": reason,
            },
            resumption_key=envelope.envelope_id,
            timeout_behavior=TimeoutBehavior.PERSIST,
        )

    def _consume_type_budget(self, decision: RecoveryDecision) -> None:
        """Consume type-specific budget counters based on action type."""
        if decision.action == RecoveryAction.TRANSFORM_AND_RETRY:
            self._budget.consume_transform()
        elif decision.action == RecoveryAction.FAILOVER:
            if "credential" in decision.strategy_key:
                self._budget.consume_rotation()
            else:
                self._budget.consume_failover()

    def _record_audit(self, decision: RecoveryDecision, *, strategy_key: str) -> None:
        """Append a decision record to the audit log."""
        self._audit.append({
            "event": "recovery_decision",
            "decision_id": decision.decision_id,
            "envelope_id": decision.envelope.envelope_id,
            "action": decision.action.value,
            "strategy_key": strategy_key,
            "reason": decision.reason,
            "budget_cost": decision.budget_cost,
            "budget_remaining": self._budget.remaining(),
            "timestamp": time.time(),
        })

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Contracts for how a defect inside LeapFlow is reported, not laundered.

Written after an outage where one mistyped attribute name made the agent unusable
for every turn. The failure itself was trivial; what made it unusable, and then
undiagnosable, was the path it took:

  AttributeError("... has no attribute '_context_window_controller'")
    -> raised inside the LLM call's try block (bookkeeping shared the block)
    -> ErrorClassifier matched "context" in the message -> context_overflow
    -> recoverability auto_recover -> three rounds of context compression,
       provider failover and credential rotation on a context that was fine
    -> fourth round: no strategy left -> "No applicable recovery strategy found"
    -> no traceback logged on that branch, audit sink in memory only

Each layer is pinned below so the chain cannot re-form.
"""

from __future__ import annotations

import json

from leapflow.engine.recovery.error_classifier import ErrorClassifier
from leapflow.engine.recovery.failure_envelope import FailureSource, Recoverability
from leapflow.engine.recovery.recovery_audit import JsonlAuditSink, create_audit_entry
from leapflow.engine.recovery.recovery_budget import RecoveryBudget
from leapflow.engine.recovery.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.recovery.recovery_decision import RecoveryAction
from leapflow.engine.recovery.strategies import default_strategies
from leapflow.engine.recovery.unified_classifier import (
    INTERNAL_DEFECT_CATEGORY,
    UnifiedErrorClassifier,
)

# The exact exception from the outage. Its message contains "context", which is
# what the text-matching provider classifier keyed on.
_OUTAGE_EXC = AttributeError(
    "'AgentEngine' object has no attribute '_context_window_controller'"
)


def _classifier() -> UnifiedErrorClassifier:
    return UnifiedErrorClassifier(ErrorClassifier())


def _budget() -> RecoveryBudget:
    budget = RecoveryBudget(
        turn_deadline_s=0, total_recovery_actions=20, max_retry_per_category=10,
    )
    budget.start_deadline()
    return budget


# ── Classification: a bug is a bug, whatever its message says ────────────

def test_internal_defect_is_not_classified_as_a_provider_condition() -> None:
    """The outage's exception must never be read as a context overflow again."""
    envelope = _classifier().classify_llm_error(_OUTAGE_EXC, provider="dashscope", model="m")

    assert envelope.category == INTERNAL_DEFECT_CATEGORY
    assert envelope.category != "context_overflow"
    assert envelope.source is FailureSource.SYSTEM
    assert envelope.recoverability is Recoverability.NON_RECOVERABLE
    assert "AttributeError" in envelope.message


def test_defect_types_are_matched_by_type_not_message() -> None:
    """Type-based dispatch: the message text must not influence the verdict."""
    classifier = _classifier()
    for exc in (
        AttributeError("rate limit exceeded"),      # message looks transient
        TypeError("context length exceeded"),       # message looks like overflow
        KeyError("timeout"),                        # message looks transient
        NameError("invalid api key"),               # message looks like auth
        IndexError("list index out of range"),
        AssertionError("should not happen"),
    ):
        envelope = classifier.classify_llm_error(exc)
        assert envelope.category == INTERNAL_DEFECT_CATEGORY, exc
        assert envelope.recoverability is Recoverability.NON_RECOVERABLE, exc


def test_real_provider_conditions_still_use_the_provider_taxonomy() -> None:
    """The narrowing must not swallow genuine provider errors."""
    classifier = _classifier()

    transient = classifier.classify_llm_error(ConnectionError("connection reset by peer"))
    assert transient.source is FailureSource.LLM
    assert transient.category != INTERNAL_DEFECT_CATEGORY
    assert transient.recoverability is not Recoverability.NON_RECOVERABLE

    overflow = classifier.classify_llm_error(
        RuntimeError("This model's maximum context length is 8192 tokens")
    )
    assert overflow.source is FailureSource.LLM
    assert overflow.category == "context_overflow"


# ── Recovery: no compression/failover/rotation for a local bug ───────────

def test_defect_halts_immediately_without_burning_strategies() -> None:
    """Previously this consumed four rounds before giving up.

    Compression, provider failover, and credential rotation cannot repair a
    programming error; attempting them cost ~2 minutes per turn and reported the
    failure as something it was not.
    """
    coordinator = RecoveryCoordinator(strategies=default_strategies(), budget=_budget())
    coordinator.new_turn(turn_id=0)

    decision = coordinator.evaluate(_classifier().classify_llm_error(_OUTAGE_EXC))

    assert decision.action is RecoveryAction.HALT_CLEAN
    assert decision.strategy_key == "<terminal>"
    assert "context_compress" not in coordinator.guard.used_strategies()
    assert coordinator.budget.remaining() == coordinator.budget.total_recovery_actions


# ── The halt must tell the user something ────────────────────────────────

def test_terminal_decision_carries_an_actionable_interaction() -> None:
    """A stopped turn must not surface internal jargon as its whole answer."""
    from leapflow.engine._message_helpers import _terminal_failure_text

    coordinator = RecoveryCoordinator(strategies=default_strategies(), budget=_budget())
    coordinator.new_turn(turn_id=0)
    decision = coordinator.evaluate(_classifier().classify_llm_error(_OUTAGE_EXC))

    assert decision.interaction is not None
    rendered = _terminal_failure_text(decision)
    assert "No applicable recovery strategy found" not in rendered
    assert "AttributeError" in rendered
    assert decision.interaction.suggested_actions


def test_no_strategy_terminal_also_explains_itself() -> None:
    """The exact message from the incident must never be the whole answer."""
    from leapflow.engine._message_helpers import _terminal_failure_text

    coordinator = RecoveryCoordinator(strategies=[], budget=_budget())
    coordinator.new_turn(turn_id=0)
    envelope = _classifier().classify_llm_error(ConnectionError("connection reset by peer"))

    decision = coordinator.terminal_decision(envelope)

    assert decision.reason == "No applicable recovery strategy found"
    rendered = _terminal_failure_text(decision)
    assert rendered != decision.reason
    assert "stopped" in rendered.lower()


# ── Diagnosability: the record must outlive the turn ─────────────────────

def test_recovery_audit_is_written_to_disk(tmp_path) -> None:
    """An in-memory sink loses the only record of why a turn stopped."""
    path = tmp_path / "audit" / "runtime.jsonl"
    sink = JsonlAuditSink(path)
    budget = _budget()
    coordinator = RecoveryCoordinator(strategies=default_strategies(), budget=budget)
    coordinator.new_turn(turn_id=0)
    envelope = _classifier().classify_llm_error(_OUTAGE_EXC)
    decision = coordinator.evaluate(envelope)

    sink.record(create_audit_entry(envelope, decision, budget, session_id="s", turn_id=0))

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert lines, "recovery decisions must be persisted"
    assert lines[0]["failure_category"] == INTERNAL_DEFECT_CATEGORY
    assert lines[0]["action"] == RecoveryAction.HALT_CLEAN.value


def test_engine_points_the_audit_sink_at_the_profile_layout(tmp_path) -> None:
    """The sink must be constructed with a layout-owned path, not left in memory."""
    from conftest import StubLLM, make_settings
    from leapflow.engine.engine import AgentEngine
    from leapflow.engine import build_default_registry
    from leapflow.engine.intent_classifier import Intent
    from leapflow.memory import (
        EpisodicMemoryProvider,
        SemanticMemoryProvider,
        WorkingMemoryProvider,
    )
    from leapflow.platform.mock import MockBridge

    class _Classifier:
        async def classify(self, user_text: str) -> Intent:
            return Intent(label="complex", reason="test")

    settings = make_settings(str(tmp_path))
    store = SemanticMemoryProvider(source=settings.duckdb_path)
    try:
        rpc = MockBridge()
        llm = StubLLM(["ok"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        registry = build_default_registry(rpc, llm, wm, store)
        engine = AgentEngine(
            settings, rpc, llm, wm, store, EpisodicMemoryProvider(), registry, _Classifier(),
        )

        expected = getattr(getattr(settings, "profile_layout", None), "audit_log_path", None)
        if expected is not None:
            assert engine._audit_sink._path == expected
    finally:
        store.close()

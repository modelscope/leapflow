# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for P2 4.2-D Provider Context Handoff.

Covers:
- _active_context_length() reflects the FailoverChain's active provider
  window after a failover (and the primary's before).
- _post_failover_recompress() triggers force-compress when the new
  provider's context window is smaller than the current estimated payload.
- No recompression when the new window is large enough.
- Audit evidence is recorded for the handoff recompression.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from leapflow.engine.engine import AgentEngine
from leapflow.engine.recovery_coordinator import RecoveryCoordinator
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)
from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
)
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.llm.model_capabilities import ModelCapabilityRegistry


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _make_envelope(
    *,
    category: str = "billing",
) -> FailureEnvelope:
    return FailureEnvelope.create(
        source=FailureSource.LLM,
        category=category,
        failure_class="test",
        failure_code="test_code",
        message="provider failure",
        recoverability=Recoverability.AUTO_RETRY,
        context=FailureContext.from_dict_args(),
    )


def _make_failover_decision(envelope: FailureEnvelope | None = None) -> RecoveryDecision:
    if envelope is None:
        envelope = _make_envelope()
    return RecoveryDecision.create(
        envelope=envelope,
        action=RecoveryAction.FAILOVER,
        reason="Provider failure: failing over",
        strategy_key="provider_failover",
        retry_semantics=RetrySemantics(
            consumes_retry_budget=True,
            resets_retry_count=True,
        ),
        budget_cost=1,
        audit_metadata={"trigger_category": "billing"},
    )


class _FakeChain:
    """Minimal mock of a FailoverChain exposing context_length and model."""

    def __init__(self, context_length: int, model: str = "primary-model") -> None:
        self.context_length = context_length
        self.model = model

    def _failover(self, reason: str) -> bool:
        return True


def _stub_engine(
    *,
    llm_context_length: int = 128_000,
    chain_context_length: int | None = None,
    chain_model: str | None = None,
    llm_model: str = "test-model",
    registry: ModelCapabilityRegistry | None = None,
) -> AgentEngine:
    """Build a partial AgentEngine with just enough attributes for budget tests."""
    engine = object.__new__(AgentEngine)
    engine._settings = SimpleNamespace(
        llm_model=llm_model,
        llm_context_length=llm_context_length,
    )
    engine._model_capabilities = registry if registry is not None else ModelCapabilityRegistry()
    if chain_context_length is not None:
        engine._llm = _FakeChain(
            chain_context_length,
            model=chain_model or llm_model,
        )
    else:
        engine._llm = SimpleNamespace()  # no context_length attribute
    return engine


# ═══════════════════════════════════════════════════════════════
# _active_context_length — chain-aware
# ═══════════════════════════════════════════════════════════════


class TestActiveContextLengthChainAware:
    """_active_context_length() must use the FailoverChain's live window."""

    def test_primary_provider_uses_configured_budget(self) -> None:
        """On primary, chain.context_length == configured; result should match."""
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=128_000,
        )
        assert AgentEngine._active_context_length(engine) == 128_000

    def test_failover_to_smaller_window_caps_budget(self) -> None:
        """After failover, a smaller chain.context_length must reduce the budget."""
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=32_000,
        )
        assert AgentEngine._active_context_length(engine) == 32_000

    def test_failover_to_larger_window_keeps_configured_budget(self) -> None:
        """A fallback with a larger window doesn't raise above configured budget."""
        engine = _stub_engine(
            llm_context_length=64_000,
            chain_context_length=200_000,
        )
        assert AgentEngine._active_context_length(engine) == 64_000

    def test_no_chain_context_length_falls_back_to_settings(self) -> None:
        """When LLM backend doesn't expose context_length, settings drive budget."""
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=None,  # no chain attribute
        )
        assert AgentEngine._active_context_length(engine) == 128_000

    def test_chain_model_used_for_capability_lookup(self) -> None:
        """After failover, capability lookup should use the active chain model."""
        registry = ModelCapabilityRegistry()
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=64_000,
            chain_model="fallback-model",
            llm_model="primary-model",
            registry=registry,
        )
        # The chain model is used for lookup; "fallback-model" won't be in the
        # registry, so non-authoritative → budget is min(128k, 64k) = 64k.
        result = AgentEngine._active_context_length(engine)
        assert result == 64_000


# ═══════════════════════════════════════════════════════════════
# _post_failover_recompress
# ═══════════════════════════════════════════════════════════════


def _engine_for_recompress(
    *,
    new_window: int = 32_000,
    estimated_tokens: int = 60_000,
) -> AgentEngine:
    """Build a partial engine wired for _post_failover_recompress testing."""
    engine = object.__new__(AgentEngine)
    engine._settings = SimpleNamespace(
        llm_model="test-model",
        llm_context_length=128_000,
    )
    engine._model_capabilities = None  # skip registry
    engine._llm = _FakeChain(new_window)

    # Stub the compressor
    compressor = MagicMock()
    compressor.force_compress.return_value = [{"role": "system", "content": "compressed"}]
    engine._compressor = compressor

    # Stub the estimator
    estimator = MagicMock()
    estimator.estimate_messages.return_value = estimated_tokens
    context_controller = SimpleNamespace(estimator=estimator)
    engine._context_controller = context_controller

    # Stub usage tracker and audit sink
    engine._usage_tracker = MagicMock()
    engine._audit_sink = MagicMock()

    return engine


class TestPostFailoverRecompress:
    """_post_failover_recompress() must compress when payload exceeds new window."""

    def test_recompress_triggered_when_payload_exceeds_new_window(self) -> None:
        """Payload larger than new window → force_compress called."""
        engine = _engine_for_recompress(new_window=32_000, estimated_tokens=60_000)
        messages = [
            {"role": "system", "content": "x" * 200_000},
            {"role": "user", "content": "hello"},
        ]

        coordinator = RecoveryCoordinator(
            strategies=[],
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coordinator.budget.start_deadline()
        decision = _make_failover_decision()

        result = AgentEngine._post_failover_recompress(
            engine, messages, coordinator, decision,
        )

        assert result is True
        engine._compressor.force_compress.assert_called_once_with(messages)
        engine._usage_tracker.mark_compression.assert_called_once()
        engine._audit_sink.update_outcome.assert_called_once()
        # Verify audit records the handoff reason
        call_args = engine._audit_sink.update_outcome.call_args
        assert call_args[0][1] == "success"
        assert "post-failover recompression" in call_args[1]["reason"]

    def test_no_recompress_when_payload_fits_new_window(self) -> None:
        """Payload within the new window → no compression needed."""
        engine = _engine_for_recompress(new_window=128_000, estimated_tokens=60_000)
        messages = [{"role": "user", "content": "hello"}]

        coordinator = RecoveryCoordinator(
            strategies=[],
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coordinator.budget.start_deadline()
        decision = _make_failover_decision()

        result = AgentEngine._post_failover_recompress(
            engine, messages, coordinator, decision,
        )

        assert result is False
        engine._compressor.force_compress.assert_not_called()
        engine._usage_tracker.mark_compression.assert_not_called()
        engine._audit_sink.update_outcome.assert_not_called()

    def test_no_recompress_when_payload_equals_window(self) -> None:
        """Payload exactly at the window → no compression (equal is acceptable)."""
        engine = _engine_for_recompress(new_window=60_000, estimated_tokens=60_000)
        messages = [{"role": "user", "content": "hello"}]

        coordinator = RecoveryCoordinator(
            strategies=[],
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coordinator.budget.start_deadline()
        decision = _make_failover_decision()

        result = AgentEngine._post_failover_recompress(
            engine, messages, coordinator, decision,
        )

        assert result is False
        engine._compressor.force_compress.assert_not_called()

    def test_recompress_records_coordinator_outcome(self) -> None:
        """Coordinator audit log must record the recompression outcome."""
        engine = _engine_for_recompress(new_window=16_000, estimated_tokens=50_000)
        messages = [{"role": "user", "content": "hello"}]

        coordinator = RecoveryCoordinator(
            strategies=[],
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coordinator.budget.start_deadline()
        decision = _make_failover_decision()

        AgentEngine._post_failover_recompress(
            engine, messages, coordinator, decision,
        )

        # Coordinator on_strategy_outcome should have been called
        log = coordinator.audit_log
        assert any(
            entry.get("event") == "strategy_outcome"
            and entry.get("decision_id") == decision.decision_id
            and entry.get("success") is True
            for entry in log
        ), f"Expected strategy_outcome in audit log, got: {log}"

    def test_messages_replaced_in_place_after_recompress(self) -> None:
        """Messages list must be mutated in-place with compressed content."""
        engine = _engine_for_recompress(new_window=16_000, estimated_tokens=50_000)
        compressed_result = [{"role": "system", "content": "compressed"}]
        engine._compressor.force_compress.return_value = compressed_result

        messages = [
            {"role": "system", "content": "very long context..."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "a " * 20_000},
        ]

        coordinator = RecoveryCoordinator(
            strategies=[],
            budget=RecoveryBudget(total_recovery_actions=32),
        )
        coordinator.budget.start_deadline()
        decision = _make_failover_decision()

        AgentEngine._post_failover_recompress(
            engine, messages, coordinator, decision,
        )

        # Messages should be replaced in-place
        assert messages == compressed_result


# ═══════════════════════════════════════════════════════════════
# Integration: failover + context budget coherence
# ═══════════════════════════════════════════════════════════════


class TestFailoverContextBudgetCoherence:
    """After failover, _active_context_length returns the new provider's window."""

    def test_budget_changes_after_simulated_failover(self) -> None:
        """Simulating a failover by swapping the chain validates budget tracking."""
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=128_000,
        )
        assert AgentEngine._active_context_length(engine) == 128_000

        # Simulate failover to smaller provider
        engine._llm = _FakeChain(32_000, model="fallback-model")
        assert AgentEngine._active_context_length(engine) == 32_000

    def test_budget_restores_after_primary_recovery(self) -> None:
        """When primary is restored, budget goes back to the original window."""
        engine = _stub_engine(
            llm_context_length=128_000,
            chain_context_length=128_000,
        )
        assert AgentEngine._active_context_length(engine) == 128_000

        # Failover to smaller
        engine._llm = _FakeChain(32_000, model="fallback-model")
        assert AgentEngine._active_context_length(engine) == 32_000

        # Restore primary
        engine._llm = _FakeChain(128_000, model="primary-model")
        assert AgentEngine._active_context_length(engine) == 128_000

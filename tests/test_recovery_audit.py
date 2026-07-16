"""Tests for recovery_audit module — structured audit logging for recovery decisions."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from leapflow.engine.failure_envelope import (
    FailureContext,
    FailureEnvelope,
    FailureSource,
    Recoverability,
    SideEffectState,
)
from leapflow.engine.recovery_audit import (
    JsonlAuditSink,
    RecoveryAuditEntry,
    create_audit_entry,
)
from leapflow.engine.recovery_budget import RecoveryBudget
from leapflow.engine.recovery_decision import (
    RecoveryAction,
    RecoveryDecision,
    RetrySemantics,
)


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_envelope() -> FailureEnvelope:
    """Create a sample FailureEnvelope for testing."""
    return FailureEnvelope.create(
        source=FailureSource.LLM,
        category="rate_limited",
        failure_class="rate_limited",
        failure_code="llm_rate_limited",
        message="Rate limit exceeded",
        recoverability=Recoverability.AUTO_RETRY,
        side_effect_state=SideEffectState.NONE,
        context=FailureContext.from_dict_args(provider="openai", model="gpt-4"),
    )


@pytest.fixture
def sample_decision(sample_envelope: FailureEnvelope) -> RecoveryDecision:
    """Create a sample RecoveryDecision for testing."""
    return RecoveryDecision.create(
        envelope=sample_envelope,
        action=RecoveryAction.RETRY_WITH_BACKOFF,
        reason="Rate limited, retrying with backoff",
        strategy_key="jittered_retry",
        budget_cost=1,
    )


@pytest.fixture
def sample_budget() -> RecoveryBudget:
    """Create a sample RecoveryBudget for testing."""
    budget = RecoveryBudget()
    budget.start_deadline()
    budget.consume(2, "rate_limited")
    return budget


# ─── RecoveryAuditEntry Tests ────────────────────────────────────────

class TestRecoveryAuditEntry:
    """Tests for RecoveryAuditEntry dataclass."""

    def test_creation(self) -> None:
        """RecoveryAuditEntry can be created with required fields."""
        entry = RecoveryAuditEntry(
            timestamp=time.time(),
            session_id="sess-1",
            turn_id=3,
            envelope_id="env-abc",
            failure_source="llm",
            failure_category="rate_limited",
            failure_code="llm_rate_limited",
            recoverability="auto_retry",
            decision_id="dec-xyz",
            strategy_key="jittered_retry",
            action="retry_with_backoff",
            reason="Rate limited",
        )
        assert entry.session_id == "sess-1"
        assert entry.turn_id == 3
        assert entry.strategy_key == "jittered_retry"

    def test_to_json_dict_excludes_empty_optional_fields(self) -> None:
        """to_json_dict() excludes empty outcome fields and zero elapsed_ms."""
        entry = RecoveryAuditEntry(
            timestamp=1234567890.0,
            session_id="sess-1",
            turn_id=3,
            envelope_id="env-abc",
            failure_source="llm",
            failure_category="rate_limited",
            failure_code="llm_rate_limited",
            recoverability="auto_retry",
            decision_id="dec-xyz",
            strategy_key="jittered_retry",
            action="retry_with_backoff",
            reason="Rate limited",
            budget_cost=0,
            budget_consumed=0,
            budget_remaining=0,
            outcome="",
            outcome_reason="",
            elapsed_ms=0.0,
        )
        d = entry.to_json_dict()
        # budget_cost=0 and turn_id kept (they are regular fields with valid zero values)
        assert d["budget_cost"] == 0
        assert d["budget_consumed"] == 0
        assert d["budget_remaining"] == 0
        assert d["turn_id"] == 3
        # Optional outcome fields removed when at default
        assert "outcome" not in d
        assert "outcome_reason" not in d
        assert "elapsed_ms" not in d
        assert d["session_id"] == "sess-1"
        assert d["timestamp"] == 1234567890.0

    def test_to_json_dict_includes_nonzero(self) -> None:
        """to_json_dict() includes non-zero / non-empty values."""
        entry = RecoveryAuditEntry(
            timestamp=1234567890.0,
            session_id="sess-1",
            turn_id=0,
            envelope_id="env-abc",
            failure_source="llm",
            failure_category="rate_limited",
            failure_code="llm_rate_limited",
            recoverability="auto_retry",
            decision_id="dec-xyz",
            strategy_key="jittered_retry",
            action="retry_with_backoff",
            reason="Rate limited",
            budget_cost=2,
            budget_consumed=5,
            budget_remaining=7,
            outcome="success",
        )
        d = entry.to_json_dict()
        assert d["budget_cost"] == 2
        assert d["budget_consumed"] == 5
        assert d["budget_remaining"] == 7
        assert d["outcome"] == "success"

    def test_frozen_immutability(self) -> None:
        """RecoveryAuditEntry is immutable (frozen dataclass)."""
        entry = RecoveryAuditEntry(
            timestamp=time.time(),
            session_id="s",
            turn_id=1,
            envelope_id="e",
            failure_source="llm",
            failure_category="c",
            failure_code="f",
            recoverability="auto_retry",
            decision_id="d",
            strategy_key="k",
            action="a",
            reason="r",
        )
        with pytest.raises(Exception):
            entry.session_id = "modified"  # type: ignore[misc]


# ─── JsonlAuditSink Tests ────────────────────────────────────────────

class TestJsonlAuditSink:
    """Tests for JsonlAuditSink."""

    def _make_entry(self, strategy: str = "jittered_retry",
                    action: str = "retry_with_backoff") -> RecoveryAuditEntry:
        """Helper to create test entries."""
        return RecoveryAuditEntry(
            timestamp=time.time(),
            session_id="sess-1",
            turn_id=1,
            envelope_id="env-1",
            failure_source="llm",
            failure_category="rate_limited",
            failure_code="llm_rate_limited",
            recoverability="auto_retry",
            decision_id="dec-1",
            strategy_key=strategy,
            action=action,
            reason="test reason",
            budget_cost=1,
        )

    def test_record_to_memory(self) -> None:
        """record() stores entries in in-memory buffer."""
        sink = JsonlAuditSink()
        entry = self._make_entry()
        sink.record(entry)
        assert len(sink.entries) == 1
        assert sink.entries[0] is entry

    def test_record_to_file(self, tmp_path: Path) -> None:
        """record() writes JSONL entries to disk."""
        audit_file = tmp_path / "audit.jsonl"
        sink = JsonlAuditSink(path=audit_file)
        entry = self._make_entry()
        sink.record(entry)

        assert audit_file.exists()
        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["strategy_key"] == "jittered_retry"
        assert data["session_id"] == "sess-1"

    def test_record_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple records produce multiple JSONL lines."""
        audit_file = tmp_path / "audit.jsonl"
        sink = JsonlAuditSink(path=audit_file)
        sink.record(self._make_entry(strategy="s1"))
        sink.record(self._make_entry(strategy="s2"))
        sink.record(self._make_entry(strategy="s3"))

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 3
        assert len(sink.entries) == 3

    def test_update_outcome(self, tmp_path: Path) -> None:
        """update_outcome() writes an outcome update record."""
        audit_file = tmp_path / "audit.jsonl"
        sink = JsonlAuditSink(path=audit_file)
        sink.record(self._make_entry())
        sink.update_outcome("dec-1", "success", reason="Retry succeeded", elapsed_ms=150.5)

        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 2
        update = json.loads(lines[1])
        assert update["type"] == "outcome_update"
        assert update["decision_id"] == "dec-1"
        assert update["outcome"] == "success"
        assert update["elapsed_ms"] == 150.5

    def test_summary_empty(self) -> None:
        """summary() returns total=0 for empty sink."""
        sink = JsonlAuditSink()
        assert sink.summary() == {"total": 0}

    def test_summary_statistics(self) -> None:
        """summary() generates correct statistics."""
        sink = JsonlAuditSink()
        sink.record(self._make_entry(strategy="jittered_retry", action="retry_with_backoff"))
        sink.record(self._make_entry(strategy="jittered_retry", action="retry_with_backoff"))
        sink.record(self._make_entry(strategy="context_compress", action="transform_and_retry"))
        sink.record(self._make_entry(strategy="provider_failover", action="failover"))

        s = sink.summary()
        assert s["total"] == 4
        assert s["by_strategy"]["jittered_retry"] == 2
        assert s["by_strategy"]["context_compress"] == 1
        assert s["by_strategy"]["provider_failover"] == 1
        assert s["by_action"]["retry_with_backoff"] == 2
        assert s["by_action"]["transform_and_retry"] == 1
        assert s["by_action"]["failover"] == 1

    def test_file_write_failure_graceful(self, tmp_path: Path) -> None:
        """File write failure doesn't crash; entry still stored in memory."""
        # Point to a path that cannot be written (directory as file)
        bad_path = tmp_path / "readonly_dir" / "nested" / "audit.jsonl"
        # Create the parent as a file to block directory creation
        bad_path_parent = tmp_path / "readonly_dir"
        bad_path_parent.mkdir()
        (bad_path_parent / "nested").write_text("blocker")  # file blocks mkdir

        sink = JsonlAuditSink(path=bad_path)
        entry = self._make_entry()
        # Should not raise
        sink.record(entry)
        # Entry still in memory buffer
        assert len(sink.entries) == 1

    def test_no_path_memory_only(self) -> None:
        """Without a path, sink operates in memory-only mode."""
        sink = JsonlAuditSink(path=None)
        sink.record(self._make_entry())
        sink.update_outcome("dec-1", "failure")
        assert len(sink.entries) == 1


# ─── create_audit_entry Factory Tests ─────────────────────────────────

class TestCreateAuditEntry:
    """Tests for the create_audit_entry factory function."""

    def test_basic_creation(
        self, sample_envelope: FailureEnvelope, sample_decision: RecoveryDecision,
        sample_budget: RecoveryBudget,
    ) -> None:
        """Factory creates an audit entry from real coordinator types."""
        entry = create_audit_entry(
            sample_envelope, sample_decision, sample_budget,
            session_id="test-session", turn_id=5,
        )
        assert entry.session_id == "test-session"
        assert entry.turn_id == 5
        assert entry.envelope_id == sample_envelope.envelope_id
        assert entry.failure_source == "llm"
        assert entry.failure_category == "rate_limited"
        assert entry.failure_code == "llm_rate_limited"
        assert entry.recoverability == "auto_retry"
        assert entry.decision_id == sample_decision.decision_id
        assert entry.strategy_key == "jittered_retry"
        assert entry.action == "retry_with_backoff"
        assert entry.reason == "Rate limited, retrying with backoff"
        assert entry.budget_cost == 1
        assert entry.budget_consumed == 2  # We consumed 2 in the fixture
        assert entry.budget_remaining == 10  # 12 total - 2 consumed

    def test_serialization_round_trip(
        self, sample_envelope: FailureEnvelope, sample_decision: RecoveryDecision,
        sample_budget: RecoveryBudget,
    ) -> None:
        """Audit entry serializes to valid JSON."""
        entry = create_audit_entry(
            sample_envelope, sample_decision, sample_budget,
            session_id="s1", turn_id=2,
        )
        json_str = json.dumps(entry.to_json_dict(), ensure_ascii=False)
        data = json.loads(json_str)
        assert data["failure_source"] == "llm"
        assert data["strategy_key"] == "jittered_retry"

    def test_defaults_for_session_and_turn(
        self, sample_envelope: FailureEnvelope, sample_decision: RecoveryDecision,
        sample_budget: RecoveryBudget,
    ) -> None:
        """Factory uses default empty session_id and turn_id=0."""
        entry = create_audit_entry(sample_envelope, sample_decision, sample_budget)
        assert entry.session_id == ""
        assert entry.turn_id == 0

    def test_tool_failure_envelope(self) -> None:
        """Factory works with tool-source envelopes."""
        envelope = FailureEnvelope.create(
            source=FailureSource.TOOL,
            category="tool_permission",
            failure_class="authorization",
            failure_code="access_denied",
            message="Permission denied",
            recoverability=Recoverability.NON_RECOVERABLE,
        )
        decision = RecoveryDecision.create(
            envelope=envelope,
            action=RecoveryAction.HALT_CLEAN,
            reason="Non-recoverable permission failure",
            strategy_key="<terminal>",
        )
        budget = RecoveryBudget()
        entry = create_audit_entry(envelope, decision, budget)
        assert entry.failure_source == "tool"
        assert entry.action == "halt_clean"
        assert entry.recoverability == "non_recoverable"

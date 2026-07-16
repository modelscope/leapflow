"""Tests for the recovery checkpoint system (cross-turn state persistence)."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from leapflow.engine.recovery_checkpoint import (
    CheckpointResumer,
    CheckpointState,
    CheckpointStore,
    DefaultDriftVerifier,
    InMemoryCheckpointStore,
    Precondition,
    RecoveryCheckpoint,
    ResumeResult,
    StepRecord,
)


# ---------------------------------------------------------------------------
# RecoveryCheckpoint dataclass tests
# ---------------------------------------------------------------------------


class TestRecoveryCheckpoint:
    """Tests for RecoveryCheckpoint creation and computed properties."""

    def test_creation_defaults(self) -> None:
        cp = RecoveryCheckpoint()
        assert cp.checkpoint_id  # UUID auto-generated
        assert cp.state == CheckpointState.PENDING
        assert cp.version == 1
        assert cp.ttl_seconds == 3600.0
        assert cp.resume_attempts == 0
        assert cp.max_resume_attempts == 3
        assert cp.consumed_at is None

    def test_is_expired_false_when_fresh(self) -> None:
        cp = RecoveryCheckpoint(created_at=time.time(), ttl_seconds=3600.0)
        assert cp.is_expired is False

    def test_is_expired_true_when_past_ttl(self) -> None:
        cp = RecoveryCheckpoint(created_at=time.time() - 7200, ttl_seconds=3600.0)
        assert cp.is_expired is True

    def test_is_consumable_when_pending_and_fresh(self) -> None:
        cp = RecoveryCheckpoint(created_at=time.time(), ttl_seconds=3600.0)
        assert cp.is_consumable is True

    def test_is_consumable_false_when_expired(self) -> None:
        cp = RecoveryCheckpoint(created_at=time.time() - 7200, ttl_seconds=3600.0)
        assert cp.is_consumable is False

    def test_is_consumable_false_when_consumed(self) -> None:
        cp = RecoveryCheckpoint(state=CheckpointState.CONSUMED)
        assert cp.is_consumable is False

    def test_is_consumable_false_when_max_attempts_reached(self) -> None:
        cp = RecoveryCheckpoint(resume_attempts=3, max_resume_attempts=3)
        assert cp.is_consumable is False

    def test_expires_at(self) -> None:
        now = time.time()
        cp = RecoveryCheckpoint(created_at=now, ttl_seconds=1800.0)
        assert cp.expires_at == pytest.approx(now + 1800.0, abs=0.01)

    def test_step_record_frozen(self) -> None:
        step = StepRecord(step_id="s1", tool_name="bash", action="run")
        assert step.step_id == "s1"
        with pytest.raises(Exception):
            step.step_id = "s2"  # type: ignore[misc]

    def test_precondition_frozen(self) -> None:
        pre = Precondition(key="session_id", expected_value="abc", critical=True)
        assert pre.key == "session_id"
        with pytest.raises(Exception):
            pre.key = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InMemoryCheckpointStore tests
# ---------------------------------------------------------------------------


class TestInMemoryCheckpointStore:
    """Tests for the in-memory checkpoint store."""

    def setup_method(self) -> None:
        self.store = InMemoryCheckpointStore()

    def test_save_and_load_roundtrip(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-1", session_id="sess-1")
        self.store.save(cp)
        loaded = self.store.load("cp-1")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp-1"
        assert loaded.session_id == "sess-1"

    def test_load_returns_none_for_nonexistent(self) -> None:
        assert self.store.load("nonexistent") is None

    def test_load_returns_none_for_expired(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-expired",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        self.store.save(cp)
        assert self.store.load("cp-expired") is None

    def test_load_marks_expired_state(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-exp",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        self.store.save(cp)
        result = self.store.load("cp-exp")
        assert result is None
        # Expired checkpoint is removed from store
        assert self.store.load("cp-exp") is None

    def test_load_and_consume_cas_success(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-cas")
        self.store.save(cp)
        result = self.store.load_and_consume("cp-cas")
        assert result is not None
        assert result.state == CheckpointState.CONSUMED
        assert result.consumed_at is not None
        assert result.version == 2  # Bumped from 1

    def test_load_and_consume_second_call_returns_none(self) -> None:
        """CAS semantics: second consume attempt must fail."""
        cp = RecoveryCheckpoint(checkpoint_id="cp-cas2")
        self.store.save(cp)
        first = self.store.load_and_consume("cp-cas2")
        assert first is not None
        second = self.store.load_and_consume("cp-cas2")
        assert second is None

    def test_load_and_consume_returns_none_for_expired(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-cas-exp",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        self.store.save(cp)
        assert self.store.load_and_consume("cp-cas-exp") is None
        assert cp.state == CheckpointState.EXPIRED

    def test_load_and_consume_respects_max_resume_attempts(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-attempts",
            resume_attempts=3,
            max_resume_attempts=3,
        )
        self.store.save(cp)
        assert self.store.load_and_consume("cp-attempts") is None

    def test_load_and_consume_nonexistent(self) -> None:
        assert self.store.load_and_consume("nope") is None

    def test_list_pending_returns_pending_only(self) -> None:
        cp1 = RecoveryCheckpoint(checkpoint_id="cp-p1", session_id="s1")
        cp2 = RecoveryCheckpoint(
            checkpoint_id="cp-p2", session_id="s1",
            state=CheckpointState.CONSUMED,
        )
        cp3 = RecoveryCheckpoint(checkpoint_id="cp-p3", session_id="s1")
        self.store.save(cp1)
        self.store.save(cp2)
        self.store.save(cp3)
        pending = self.store.list_pending()
        ids = [c.checkpoint_id for c in pending]
        assert "cp-p1" in ids
        assert "cp-p3" in ids
        assert "cp-p2" not in ids

    def test_list_pending_filters_by_session(self) -> None:
        cp1 = RecoveryCheckpoint(checkpoint_id="cp-s1", session_id="alpha")
        cp2 = RecoveryCheckpoint(checkpoint_id="cp-s2", session_id="beta")
        self.store.save(cp1)
        self.store.save(cp2)
        result = self.store.list_pending(session_id="alpha")
        assert len(result) == 1
        assert result[0].session_id == "alpha"

    def test_list_pending_excludes_expired(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-lp-exp",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        self.store.save(cp)
        assert self.store.list_pending() == []
        assert cp.state == CheckpointState.EXPIRED

    def test_cancel_success(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-cancel")
        self.store.save(cp)
        assert self.store.cancel("cp-cancel") is True
        assert cp.state == CheckpointState.CANCELLED
        assert cp.version == 2

    def test_cancel_fails_if_not_pending(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-cancel2", state=CheckpointState.CONSUMED,
        )
        self.store.save(cp)
        assert self.store.cancel("cp-cancel2") is False

    def test_cancel_nonexistent(self) -> None:
        assert self.store.cancel("nope") is False

    def test_cleanup_expired_removes_old_entries(self) -> None:
        cp1 = RecoveryCheckpoint(
            checkpoint_id="cp-old",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        cp2 = RecoveryCheckpoint(checkpoint_id="cp-fresh")
        self.store.save(cp1)
        self.store.save(cp2)
        removed = self.store.cleanup_expired()
        assert removed == 1
        assert self.store.load("cp-old") is None
        assert self.store.load("cp-fresh") is not None

    def test_cleanup_expired_returns_zero_when_none_expired(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-active")
        self.store.save(cp)
        assert self.store.cleanup_expired() == 0

    def test_save_overwrites_existing(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-ow", session_id="v1")
        self.store.save(cp)
        cp2 = RecoveryCheckpoint(checkpoint_id="cp-ow", session_id="v2")
        self.store.save(cp2)
        loaded = self.store.load("cp-ow")
        assert loaded is not None
        assert loaded.session_id == "v2"


# ---------------------------------------------------------------------------
# CheckpointResumer tests
# ---------------------------------------------------------------------------


class TestCheckpointResumer:
    """Tests for the checkpoint resume orchestration."""

    def setup_method(self) -> None:
        self.store = InMemoryCheckpointStore()
        self.resumer = CheckpointResumer(store=self.store)

    def test_resume_success_path(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-resume",
            session_id="s1",
            messages_snapshot=[{"role": "user", "content": "hello"}],
            failure_envelope_data={"message": "rate limit exceeded"},
            context_data={"plan_step": 3},
        )
        self.store.save(cp)
        result = self.resumer.resume("cp-resume")
        assert result.success is True
        assert result.context_data == {"plan_step": 3}
        # Messages include original + recovery injection
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "hello"
        assert "[Recovery]" in result.messages[1]["content"]
        assert "rate limit exceeded" in result.messages[1]["content"]

    def test_resume_fails_on_expired_checkpoint(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-exp-resume",
            created_at=time.time() - 7200,
            ttl_seconds=3600.0,
        )
        self.store.save(cp)
        result = self.resumer.resume("cp-exp-resume")
        assert result.success is False
        assert "not available" in result.reason

    def test_resume_fails_on_consumed_checkpoint(self) -> None:
        cp = RecoveryCheckpoint(checkpoint_id="cp-consumed-r")
        self.store.save(cp)
        # First consume succeeds
        first = self.resumer.resume("cp-consumed-r")
        assert first.success is True
        # Second fails (CAS semantics)
        second = self.resumer.resume("cp-consumed-r")
        assert second.success is False

    def test_resume_fails_on_critical_precondition_drift(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-drift",
            preconditions=(
                Precondition(
                    key="session_id",
                    expected_value="sess-original",
                    critical=True,
                ),
            ),
        )
        self.store.save(cp)
        result = self.resumer.resume(
            "cp-drift",
            current_context={"session_id": "sess-different"},
        )
        assert result.success is False
        assert "Critical precondition drift" in result.reason
        assert "session_id" in result.reason

    def test_resume_succeeds_with_noncritical_drift(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-warn",
            messages_snapshot=[{"role": "assistant", "content": "working..."}],
            failure_envelope_data={"message": "timeout"},
            preconditions=(
                Precondition(
                    key="workspace_root",
                    expected_value="/old/path",
                    critical=False,
                    description="Workspace may have changed",
                ),
            ),
        )
        self.store.save(cp)
        result = self.resumer.resume(
            "cp-warn",
            current_context={"workspace_root": "/new/path"},
        )
        assert result.success is True
        assert len(result.drift_warnings) == 1
        assert "workspace_root" in result.drift_warnings[0]

    def test_resume_injects_recovery_message(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-msg",
            messages_snapshot=[
                {"role": "user", "content": "deploy the service"},
                {"role": "assistant", "content": "I will deploy now"},
            ],
            failure_envelope_data={"message": "connection refused"},
        )
        self.store.save(cp)
        result = self.resumer.resume("cp-msg")
        assert result.success is True
        assert len(result.messages) == 3
        recovery_msg = result.messages[2]
        assert recovery_msg["role"] == "system"
        assert "connection refused" in recovery_msg["content"]
        assert "Continue from where you left off" in recovery_msg["content"]

    def test_resume_nonexistent_checkpoint(self) -> None:
        result = self.resumer.resume("no-such-id")
        assert result.success is False
        assert "not available" in result.reason

    def test_resume_with_multiple_preconditions(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-multi-pre",
            messages_snapshot=[],
            failure_envelope_data={"message": "error"},
            preconditions=(
                Precondition(key="session_id", expected_value="s1", critical=True),
                Precondition(key="config_hash", expected_value="abc", critical=True),
                Precondition(key="env", expected_value="prod", critical=False),
            ),
        )
        self.store.save(cp)
        result = self.resumer.resume(
            "cp-multi-pre",
            current_context={"session_id": "s1", "config_hash": "abc", "env": "dev"},
        )
        assert result.success is True
        assert len(result.drift_warnings) == 1
        assert "env" in result.drift_warnings[0]


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify InMemoryCheckpointStore satisfies CheckpointStore protocol."""

    def test_inmemory_store_is_checkpoint_store(self) -> None:
        store = InMemoryCheckpointStore()
        assert isinstance(store, CheckpointStore)

    def test_default_drift_verifier(self) -> None:
        verifier = DefaultDriftVerifier()
        assert verifier.get_current_value("anything") == ""


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_checkpoint_with_zero_ttl_is_immediately_expired(self) -> None:
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-zero-ttl",
            created_at=time.time() - 0.001,
            ttl_seconds=0.0,
        )
        assert cp.is_expired is True
        assert cp.is_consumable is False

    def test_checkpoint_with_completed_steps(self) -> None:
        steps = (
            StepRecord(step_id="1", tool_name="bash", has_side_effect=True),
            StepRecord(step_id="2", tool_name="read_file", has_side_effect=False),
        )
        cp = RecoveryCheckpoint(
            checkpoint_id="cp-steps",
            completed_steps=steps,
            pending_steps=("step 3: validate",),
        )
        assert len(cp.completed_steps) == 2
        assert cp.completed_steps[0].has_side_effect is True
        assert cp.pending_steps == ("step 3: validate",)

    def test_resume_result_frozen(self) -> None:
        result = ResumeResult(success=True)
        with pytest.raises(Exception):
            result.success = False  # type: ignore[misc]

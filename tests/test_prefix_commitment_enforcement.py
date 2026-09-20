# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for PrefixCommitmentController enforcement lifecycle.

Pure logic tests — no real LLM, no network, no DuckDB.  Every path exercises
the controller directly through its public API.
"""
from __future__ import annotations

import pytest

from leapflow.engine.prefix_commitment import (
    CachePriceModel,
    CommitmentEnforcement,
    CommitmentStatus,
    PrefixCommitmentConfig,
    PrefixCommitmentController,
    PrefixCommitmentState,
    _system_prompt_hash,
)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def controller() -> PrefixCommitmentController:
    """Default controller with standard config."""
    return PrefixCommitmentController()


@pytest.fixture
def committed_controller() -> PrefixCommitmentController:
    """Controller already in the COMMITTED state (via force_commit)."""
    ctrl = PrefixCommitmentController()
    ctrl.force_commit()
    return ctrl


# ═══════════════════════════════════════════════════════════════════════════
# enforce() — snapshot freezing
# ═══════════════════════════════════════════════════════════════════════════


class TestEnforceSnapshot:
    """enforce() must freeze the disclosure state on first call after commit."""

    def test_enforce_returns_none_when_uncommitted(self, controller: PrefixCommitmentController) -> None:
        """enforce() before commitment returns None — no snapshot to freeze."""
        result = controller.enforce("full", ("file_read",), "abc123", turn_index=0)
        assert result is None
        assert controller.enforcement is None

    def test_enforce_freezes_on_first_call_after_commit(self, committed_controller: PrefixCommitmentController) -> None:
        level = "expanded"
        tools = ("file_read", "memory_search", "shell_run")
        prompt_hash = _system_prompt_hash("Hello system prompt")
        enforcement = committed_controller.enforce(level, tools, prompt_hash, turn_index=5)

        assert enforcement is not None
        assert isinstance(enforcement, CommitmentEnforcement)
        assert enforcement.frozen_level == level
        assert enforcement.frozen_tool_names == tuple(sorted(tools))
        assert enforcement.frozen_system_prompt_hash == prompt_hash
        assert enforcement.committed_at_turn == 5

    def test_enforce_idempotent_returns_same_snapshot(self, committed_controller: PrefixCommitmentController) -> None:
        """Repeated calls while enforcement is active return the same object."""
        first = committed_controller.enforce("full", ("a", "b"), "h1", turn_index=1)
        # Call again with DIFFERENT arguments — should still return the original
        second = committed_controller.enforce("core", ("x",), "h2", turn_index=99)
        assert second is first
        assert second.frozen_level == "full"
        assert second.frozen_tool_names == ("a", "b")
        assert second.committed_at_turn == 1

    def test_enforce_returns_none_after_evaluate_stays_uncommitted(self, controller: PrefixCommitmentController) -> None:
        """evaluate that does not commit → enforce still None."""
        # Low difficulty → should_commit returns False
        controller.evaluate(
            difficulty=0.1,
            posture="baseline",
            round_number=1,
            remaining_rounds=10,
            est_full_prefix_tokens=5000,
            est_pcd_prefix_tokens=3000,
        )
        assert not controller.committed
        result = controller.enforce("full", ("a",), "h", turn_index=1)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# should_break_commitment() — four trigger dimensions
# ═══════════════════════════════════════════════════════════════════════════


class TestShouldBreakCommitment:
    """Each structural disruption dimension individually triggers a break."""

    @pytest.mark.parametrize(
        "trigger_kwarg",
        [
            {"posture_changed": True},
            {"tool_error": True},
            {"slash_command": True},
            {"transform_retry": True},
        ],
        ids=["posture_changed", "tool_error", "slash_command", "transform_retry"],
    )
    def test_single_trigger_returns_true(
        self, controller: PrefixCommitmentController, trigger_kwarg: dict
    ) -> None:
        assert controller.should_break_commitment(**trigger_kwarg) is True

    def test_all_false_returns_false(self, controller: PrefixCommitmentController) -> None:
        assert controller.should_break_commitment(
            posture_changed=False,
            tool_error=False,
            slash_command=False,
            transform_retry=False,
        ) is False

    def test_multiple_triggers_still_true(self, controller: PrefixCommitmentController) -> None:
        assert controller.should_break_commitment(
            posture_changed=True,
            tool_error=True,
        ) is True


# ═══════════════════════════════════════════════════════════════════════════
# break_commitment() — clears enforcement, preserves monotonic status
# ═══════════════════════════════════════════════════════════════════════════


class TestBreakCommitment:
    """break_commitment clears enforcement but leaves CommitmentStatus monotonic."""

    def test_break_clears_enforcement(self, committed_controller: PrefixCommitmentController) -> None:
        committed_controller.enforce("full", ("a",), "h", turn_index=1)
        assert committed_controller.enforcement is not None

        committed_controller.break_commitment()
        assert committed_controller.enforcement is None

    def test_break_preserves_committed_status(self, committed_controller: PrefixCommitmentController) -> None:
        """CommitmentStatus remains COMMITTED after break — the decision is monotonic."""
        committed_controller.enforce("full", ("a",), "h", turn_index=1)
        committed_controller.break_commitment()
        assert committed_controller.committed is True
        assert committed_controller.state.status is CommitmentStatus.COMMITTED

    def test_break_without_enforcement_is_noop(self, committed_controller: PrefixCommitmentController) -> None:
        """Calling break when no enforcement exists does not crash or change status."""
        committed_controller.break_commitment()
        assert committed_controller.committed is True
        assert committed_controller.enforcement is None

    def test_re_enforce_after_break(self, committed_controller: PrefixCommitmentController) -> None:
        """After break, enforce() can re-establish a new snapshot."""
        committed_controller.enforce("full", ("a",), "h1", turn_index=1)
        committed_controller.break_commitment()
        assert committed_controller.enforcement is None

        new = committed_controller.enforce("core", ("b",), "h2", turn_index=5)
        assert new is not None
        assert new.frozen_level == "core"
        assert new.committed_at_turn == 5


# ═══════════════════════════════════════════════════════════════════════════
# force_commit() — external commitment trigger
# ═══════════════════════════════════════════════════════════════════════════


class TestForceCommit:
    """force_commit transitions to COMMITTED for session restore."""

    def test_force_commit_uncommitted(self, controller: PrefixCommitmentController) -> None:
        assert not controller.committed
        controller.force_commit()
        assert controller.committed
        assert controller.state.status is CommitmentStatus.COMMITTED
        assert controller.state.reason == "force_commit (session restore)"

    def test_force_commit_idempotent(self, committed_controller: PrefixCommitmentController) -> None:
        """force_commit on already committed is a no-op."""
        state_before = committed_controller.state
        committed_controller.force_commit()
        assert committed_controller.state is state_before

    def test_force_commit_then_enforce(self, controller: PrefixCommitmentController) -> None:
        """force_commit enables enforce to freeze a snapshot."""
        controller.force_commit()
        enforcement = controller.enforce("expanded", ("file_read",), "h", turn_index=0)
        assert enforcement is not None
        assert enforcement.frozen_level == "expanded"


# ═══════════════════════════════════════════════════════════════════════════
# reset() — clears everything
# ═══════════════════════════════════════════════════════════════════════════


class TestReset:
    """reset() clears both commitment status and enforcement."""

    def test_reset_clears_committed_state(self, committed_controller: PrefixCommitmentController) -> None:
        committed_controller.enforce("full", ("a",), "h", turn_index=1)
        committed_controller.reset()
        assert not committed_controller.committed
        assert committed_controller.state.status is CommitmentStatus.UNCOMMITTED
        assert committed_controller.enforcement is None

    def test_reset_on_fresh_controller_is_noop(self, controller: PrefixCommitmentController) -> None:
        controller.reset()
        assert not controller.committed
        assert controller.enforcement is None


# ═══════════════════════════════════════════════════════════════════════════
# evaluate() — normal commitment path (amortization inequality)
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluate:
    """evaluate() transitions to COMMITTED when amortization conditions are met."""

    def test_evaluate_commits_on_high_difficulty_expansion(self) -> None:
        """High difficulty + expansion posture + sufficient tokens → COMMITTED."""
        ctrl = PrefixCommitmentController(
            config=PrefixCommitmentConfig(
                commit_difficulty_threshold=0.5,
                min_prefix_tokens=500,
                min_remaining_rounds=2,
            ),
            price_model=CachePriceModel(price_miss=1.0, price_read=0.1, price_write=1.0),
        )
        state = ctrl.evaluate(
            difficulty=0.8,
            posture="research",
            round_number=3,
            remaining_rounds=10,
            est_full_prefix_tokens=5000,
            est_pcd_prefix_tokens=3000,
        )
        assert state.committed is True
        assert state.status is CommitmentStatus.COMMITTED
        assert state.committed_at_round == 3
        assert state.projected_savings > 0

    def test_evaluate_rejects_low_difficulty(self) -> None:
        ctrl = PrefixCommitmentController()
        state = ctrl.evaluate(
            difficulty=0.1,
            posture="research",
            round_number=1,
            remaining_rounds=10,
            est_full_prefix_tokens=5000,
            est_pcd_prefix_tokens=3000,
        )
        assert state.committed is False

    def test_evaluate_rejects_wrong_posture(self) -> None:
        ctrl = PrefixCommitmentController()
        state = ctrl.evaluate(
            difficulty=0.9,
            posture="baseline",
            round_number=1,
            remaining_rounds=10,
            est_full_prefix_tokens=5000,
            est_pcd_prefix_tokens=3000,
        )
        assert state.committed is False

    def test_evaluate_monotonic_no_revert(self) -> None:
        """Once committed, evaluate never reverts to UNCOMMITTED."""
        ctrl = PrefixCommitmentController(
            config=PrefixCommitmentConfig(
                commit_difficulty_threshold=0.5,
                min_prefix_tokens=500,
                min_remaining_rounds=2,
            ),
        )
        ctrl.evaluate(
            difficulty=0.9,
            posture="research",
            round_number=1,
            remaining_rounds=10,
            est_full_prefix_tokens=5000,
            est_pcd_prefix_tokens=3000,
        )
        assert ctrl.committed is True
        # Subsequent evaluate with conditions that would NOT commit returns same state
        state = ctrl.evaluate(
            difficulty=0.1,
            posture="baseline",
            round_number=2,
            remaining_rounds=1,
            est_full_prefix_tokens=100,
            est_pcd_prefix_tokens=50,
        )
        assert state.committed is True
        assert state.committed_at_round == 1  # still the original round


# ═══════════════════════════════════════════════════════════════════════════
# PrefixCommitmentState — data contract
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixCommitmentState:
    """PrefixCommitmentState immutability and serialization."""

    def test_as_dict_contains_required_keys(self) -> None:
        state = PrefixCommitmentState(
            status=CommitmentStatus.COMMITTED,
            committed_at_round=3,
            prefix_token_estimate=5000,
            projected_savings=1234.5678,
            reason="test",
        )
        d = state.as_dict()
        assert d["status"] == "committed"
        assert d["committed"] is True
        assert d["committed_at_round"] == 3
        assert d["prefix_token_estimate"] == 5000
        assert d["projected_savings"] == 1234.57  # rounded to 2 dp
        assert d["reason"] == "test"

    def test_default_state_is_uncommitted(self) -> None:
        state = PrefixCommitmentState()
        assert state.committed is False
        assert state.status is CommitmentStatus.UNCOMMITTED
        assert state.committed_at_round == -1


# ═══════════════════════════════════════════════════════════════════════════
# _system_prompt_hash — deterministic hashing
# ═══════════════════════════════════════════════════════════════════════════


class TestSystemPromptHash:
    def test_deterministic(self) -> None:
        h1 = _system_prompt_hash("hello world")
        h2 = _system_prompt_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_inputs_different_hashes(self) -> None:
        assert _system_prompt_hash("a") != _system_prompt_hash("b")

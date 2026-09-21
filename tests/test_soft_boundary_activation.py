# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for P0-OPT-2 (SOFT boundary activation / cold-start) and P0-OPT-3
(PrefixCacheOptimizer boundary-aware behavior).

Validates:
- DisclosurePlanner.plan produces SOFT when cache_benefit=True and uncommitted.
- PrefixCacheOptimizer.optimize respects COMMITTED / SOFT / NONE boundaries.
- Engine-level _cache_aware_plan_kwargs passes cache_benefit=True when
  projected_savings > 0 and not yet committed.
- Regression: committed path still produces COMMITTED; no-benefit path
  produces NONE (backward-compatible).
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest  # noqa: F401 – used by test discovery

from leapflow.engine.context.context_disclosure import (
    CacheBoundary,
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
)
from leapflow.engine.prefix_commitment import (
    CommitmentStatus,
    PrefixCommitmentConfig,
    PrefixCommitmentController,
)
from leapflow.engine.prompt_cache import (
    AnthropicCacheStrategy,
    PrefixCacheOptimizer,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _make_tool_def(name: str, category: str = "general") -> Dict[str, Any]:
    """Build a minimal OpenAI-style tool definition with x_leapflow metadata."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {"type": "object", "properties": {}},
            "x_leapflow": {
                "category": category,
                "risk_level": "read_only",
                "schema_cost": "medium",
            },
        },
    }


_TOOL_CATALOG: List[Dict[str, Any]] = [
    _make_tool_def("file_read", category="file"),
    _make_tool_def("file_list", category="file"),
    _make_tool_def("text_search", category="search"),
    _make_tool_def("memory_search", category="memory"),
    _make_tool_def("shell_run", category="shell"),
]


def _simple_messages() -> List[Dict[str, Any]]:
    """Return a minimal message list for optimizer tests."""
    return [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Do something"},
    ]


def _messages_with_frozen() -> List[Dict[str, Any]]:
    """Messages including frozen-memory and compressed-summary blocks."""
    return [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Query 1"},
        {"role": "assistant", "content": "Reply 1", "_frozen_memory": True},
        {"role": "user", "content": "Query 2"},
        {"role": "assistant", "content": "Compressed block", "_compressed_summary": True},
        {"role": "user", "content": "Query 3"},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DisclosurePlanner: SOFT boundary activation
# ═══════════════════════════════════════════════════════════════════════════


class TestDisclosurePlannerSoftActivation:
    """SOFT boundary activation when cache_benefit=True and uncommitted."""

    def test_uncommitted_cache_benefit_true_produces_soft(self) -> None:
        """UNCOMMITTED + cache_benefit=True → SOFT boundary."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=True,
        )
        assert plan.cache_boundary is CacheBoundary.SOFT

    def test_uncommitted_cache_benefit_false_produces_none(self) -> None:
        """UNCOMMITTED + cache_benefit=False → NONE boundary."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=False,
        )
        assert plan.cache_boundary is CacheBoundary.NONE

    def test_committed_not_overridden_by_soft(self) -> None:
        """COMMITTED + cache_benefit=True → still COMMITTED (commitment wins)."""
        planner = DisclosurePlanner()
        frozen_names = ("file_read", "text_search")
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.EXPANDED,
            committed_tool_names=frozen_names,
            cache_benefit=True,
        )
        assert plan.cache_boundary is CacheBoundary.COMMITTED

    def test_soft_does_not_freeze_disclosure(self) -> None:
        """SOFT boundary must NOT freeze disclosure level to FULL."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=True,
        )
        # Level is decided by normal PCD, not by SOFT
        assert plan.level in (DisclosureLevel.CORE, DisclosureLevel.EXPANDED)
        assert plan.cache_boundary is CacheBoundary.SOFT

    def test_soft_on_full_plan_posture(self) -> None:
        """SOFT applies even when PCD decides FULL (e.g. research posture)."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(
                native_tools_enabled=True,
                context_posture="research",
            ),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=True,
        )
        assert plan.level is DisclosureLevel.FULL
        assert plan.cache_boundary is CacheBoundary.SOFT

    def test_default_no_params_backward_compat(self) -> None:
        """No cache params → NONE boundary (backward-compatible)."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
        )
        assert plan.cache_boundary is CacheBoundary.NONE
        assert plan.stable_tool_names == ()


# ═══════════════════════════════════════════════════════════════════════════
# 2. PrefixCacheOptimizer: boundary-aware behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixCacheOptimizerBoundaryAware:
    """PrefixCacheOptimizer handles COMMITTED / SOFT / NONE differently."""

    def test_soft_reorders_stable_first(self) -> None:
        """SOFT boundary: system + frozen/compressed → front, dynamic → back."""
        optimizer = PrefixCacheOptimizer()
        msgs = _messages_with_frozen()
        result = optimizer.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # Stable messages (system, frozen, compressed) come first
        stable_count = sum(
            1 for m in result
            if m.get("role") == "system"
            or m.get("_frozen_memory")
            or m.get("_compressed_summary")
        )
        # All stable messages are at the front
        for i in range(stable_count):
            m = result[i]
            assert (
                m.get("role") == "system"
                or m.get("_frozen_memory")
                or m.get("_compressed_summary")
            ), f"Expected stable msg at index {i}, got: {m}"

    def test_soft_applies_cache_marker(self) -> None:
        """SOFT boundary: cache_control marker on last stable message."""
        optimizer = PrefixCacheOptimizer()
        msgs = _simple_messages()
        result = optimizer.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # System message is the only stable one and should have cache_control
        sys_msg = result[0]
        assert sys_msg.get("role") == "system"
        assert "cache_control" in sys_msg

    def test_committed_preserves_order(self) -> None:
        """COMMITTED boundary: message order is preserved (no reordering)."""
        optimizer = PrefixCacheOptimizer()
        msgs = _messages_with_frozen()
        original_roles = [m.get("role") for m in msgs]
        result = optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)

        result_roles = [m.get("role") for m in result]
        assert result_roles == original_roles, (
            f"COMMITTED should not reorder. Got {result_roles} vs {original_roles}"
        )

    def test_committed_still_applies_marker(self) -> None:
        """COMMITTED boundary: cache_control marker on last system msg in place."""
        optimizer = PrefixCacheOptimizer()
        msgs = _simple_messages()
        result = optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)

        # System message should still have cache_control marker
        sys_msg = next(m for m in result if m.get("role") == "system")
        assert "cache_control" in sys_msg

    def test_committed_does_not_mutate_input(self) -> None:
        """COMMITTED boundary: input messages are not mutated."""
        optimizer = PrefixCacheOptimizer()
        msgs = _simple_messages()
        import copy
        original = copy.deepcopy(msgs)
        optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)
        assert msgs == original

    def test_none_reorders_like_soft(self) -> None:
        """NONE boundary: same reordering as SOFT (backward-compatible)."""
        optimizer = PrefixCacheOptimizer()
        msgs = _messages_with_frozen()
        result_none = optimizer.optimize(msgs, cache_boundary=CacheBoundary.NONE)
        result_soft = optimizer.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # Same ordering and structure
        assert len(result_none) == len(result_soft)
        for r_none, r_soft in zip(result_none, result_soft):
            assert r_none.get("role") == r_soft.get("role")
            assert r_none.get("content") == r_soft.get("content")

    def test_empty_messages_all_boundaries(self) -> None:
        """All boundaries handle empty messages without error."""
        optimizer = PrefixCacheOptimizer()
        for boundary in CacheBoundary:
            result = optimizer.optimize([], cache_boundary=boundary)
            assert result == []

    def test_committed_byte_stability_across_calls(self) -> None:
        """Two consecutive COMMITTED optimize calls produce identical output."""
        optimizer = PrefixCacheOptimizer()
        msgs = _messages_with_frozen()
        result1 = optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)
        result2 = optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)

        import json
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. AnthropicCacheStrategy: not broken by boundary changes
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicCacheStrategyRegression:
    """Verify AnthropicCacheStrategy still works with all boundary values."""

    def test_anthropic_soft_splits_system_prompt(self) -> None:
        prompt = (
            "You are LeapFlow.\n\n"
            "## Capabilities\nDo things.\n\n"
            "When finished with all tool calls, provide a final answer.\n\n"
            "## Memory Context\nRecent memory.\n"
        )
        strategy = AnthropicCacheStrategy()
        result = strategy.optimize(
            [{"role": "system", "content": prompt}],
            cache_boundary=CacheBoundary.SOFT,
        )
        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)

    def test_anthropic_none_no_split(self) -> None:
        prompt = "Simple system prompt."
        strategy = AnthropicCacheStrategy()
        result = strategy.optimize(
            [{"role": "system", "content": prompt}],
            cache_boundary=CacheBoundary.NONE,
        )
        sys_msg = result[0]
        # Standard marker, no split
        assert isinstance(sys_msg["content"], list)
        assert len(sys_msg["content"]) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. Engine-level: _cache_aware_plan_kwargs
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheAwarePlanKwargs:
    """Engine._cache_aware_plan_kwargs produces correct planner arguments."""

    def _make_mock_engine(
        self,
        *,
        committed: bool = False,
        enforcement: Any = None,
        snapshot: dict | None = None,
        max_iterations: int = 10,
        full_tool_tokens: int = 500,
    ) -> MagicMock:
        """Build a MagicMock with the fields _cache_aware_plan_kwargs reads."""
        from leapflow.engine.engine import AgentEngine
        from leapflow.engine.calibration import CalibrationManager

        engine = MagicMock(spec=AgentEngine)
        engine._prefix_commitment = MagicMock()
        engine._prefix_commitment.committed = committed
        engine._prefix_commitment.enforcement = enforcement
        engine._last_context_snapshot = snapshot or {}

        # Budget config
        engine._budget_config = MagicMock()
        engine._budget_config.max_iterations = max_iterations

        # Component holding the extracted calibration/commitment methods.
        # _cache_aware_plan_kwargs / _full_tool_schema_tokens now live on
        # CalibrationManager and read engine state through its back-reference.
        engine._calibration_manager = CalibrationManager(engine)
        engine._calibration_manager._full_tool_schema_tokens = MagicMock(
            return_value=full_tool_tokens
        )
        return engine

    def test_committed_with_enforcement(self) -> None:
        """Committed + enforcement → returns commitment params."""
        enforcement = MagicMock()
        enforcement.frozen_level = "full"
        enforcement.frozen_tool_names = ("file_read", "text_search")

        engine = self._make_mock_engine(committed=True, enforcement=enforcement)
        kwargs = engine._calibration_manager._cache_aware_plan_kwargs()

        assert kwargs["commitment_status"] is CommitmentStatus.COMMITTED
        assert kwargs["committed_level"] is DisclosureLevel.FULL
        assert kwargs["committed_tool_names"] == ("file_read", "text_search")

    def test_uncommitted_positive_savings(self) -> None:
        """Uncommitted + positive projected_savings → cache_benefit=True."""
        engine = self._make_mock_engine(
            committed=False,
            snapshot={"message_tokens": 2000, "tool_schema_tokens": 500},
            full_tool_tokens=800,
        )
        # Configure projected_savings to return positive
        engine._prefix_commitment.projected_savings = MagicMock(return_value=100.0)

        kwargs = engine._calibration_manager._cache_aware_plan_kwargs()
        assert kwargs.get("cache_benefit") is True
        assert kwargs.get("commitment_status") is CommitmentStatus.UNCOMMITTED

    def test_uncommitted_no_savings(self) -> None:
        """Uncommitted + zero/negative savings → empty dict (NONE)."""
        engine = self._make_mock_engine(
            committed=False,
            snapshot={"message_tokens": 2000, "tool_schema_tokens": 500},
        )
        engine._prefix_commitment.projected_savings = MagicMock(return_value=-50.0)

        kwargs = engine._calibration_manager._cache_aware_plan_kwargs()
        assert kwargs == {}

    def test_no_snapshot_returns_empty(self) -> None:
        """No prior-round data → empty dict (first round)."""
        engine = self._make_mock_engine(committed=False, snapshot={})
        kwargs = engine._calibration_manager._cache_aware_plan_kwargs()
        assert kwargs == {}

    def test_committed_without_enforcement_returns_empty(self) -> None:
        """Committed but enforcement broken → empty dict (falls through)."""
        engine = self._make_mock_engine(committed=True, enforcement=None)
        kwargs = engine._calibration_manager._cache_aware_plan_kwargs()
        assert kwargs == {}


# ═══════════════════════════════════════════════════════════════════════════
# 5. PrefixCommitmentConfig: min_prefix_tokens lowered
# ═══════════════════════════════════════════════════════════════════════════


class TestMinPrefixTokensLowered:
    """Verify min_prefix_tokens default is 768 (P0-OPT-2 cold-start)."""

    def test_default_min_prefix_tokens(self) -> None:
        config = PrefixCommitmentConfig()
        assert config.min_prefix_tokens == 768

    def test_should_commit_at_768(self) -> None:
        """768-token prefix passes the gate (would fail at 1024)."""
        controller = PrefixCommitmentController()
        result = controller.should_commit(
            difficulty=0.70,
            posture="expanding",
            remaining_rounds=5,
            est_full_prefix_tokens=800,  # > 768 but < 1024
            est_pcd_prefix_tokens=600,
        )
        assert result is True

    def test_should_not_commit_below_768(self) -> None:
        """Prefix below 768 tokens still fails the gate."""
        controller = PrefixCommitmentController()
        result = controller.should_commit(
            difficulty=0.70,
            posture="expanding",
            remaining_rounds=5,
            est_full_prefix_tokens=700,  # < 768
            est_pcd_prefix_tokens=600,
        )
        assert result is False

    def test_configurable_override(self) -> None:
        """min_prefix_tokens is still configurable via PrefixCommitmentConfig."""
        config = PrefixCommitmentConfig(min_prefix_tokens=512)
        controller = PrefixCommitmentController(config=config)
        result = controller.should_commit(
            difficulty=0.70,
            posture="expanding",
            remaining_rounds=5,
            est_full_prefix_tokens=600,
            est_pcd_prefix_tokens=500,
        )
        assert result is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. Regression: committed path boundary stays COMMITTED
# ═══════════════════════════════════════════════════════════════════════════


class TestCommittedPathRegression:
    """Committed path must always produce COMMITTED boundary, not SOFT."""

    def test_committed_plan_boundary(self) -> None:
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.FULL,
            committed_tool_names=tuple(
                d["function"]["name"] for d in _TOOL_CATALOG
            ),
        )
        assert plan.cache_boundary is CacheBoundary.COMMITTED

    def test_committed_optimizer_stable(self) -> None:
        """PrefixCacheOptimizer.optimize(COMMITTED) preserves order."""
        optimizer = PrefixCacheOptimizer()
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "A1"},
        ]
        result = optimizer.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)
        # Order preserved: user, system, assistant
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "system"
        assert result[2]["role"] == "assistant"

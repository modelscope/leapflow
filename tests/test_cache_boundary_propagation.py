# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for cache-boundary propagation through the PCD disclosure pipeline.

Integration-level tests that wire the DisclosurePlanner, CacheBoundary,
PromptAssemblyPlan, and prompt-cache strategies together with mock LLM
providers.  No real LLM tokens are consumed; no network or DuckDB.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

import pytest

from leapflow.engine.context.context_disclosure import (
    CacheBoundary,
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
    PromptAssemblyPlan,
)
from leapflow.engine.prefix_commitment import (
    CommitmentStatus,
    PrefixCommitmentController,
    _system_prompt_hash,
)
from leapflow.engine.prompt_cache import (
    AnthropicCacheStrategy,
    NoCacheStrategy,
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
    _make_tool_def("capability_expand", category="general"),
    _make_tool_def("shell_run", category="shell"),
    _make_tool_def("file_write", category="write"),
]


def _tool_names(plan: PromptAssemblyPlan) -> set[str]:
    return {
        item.get("function", {}).get("name", "")
        for item in plan.tool_definitions
    }


# ═══════════════════════════════════════════════════════════════════════════
# PromptAssemblyPlan.with_cache_boundary — frozen immutability
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptAssemblyPlanCacheBoundary:
    """with_cache_boundary returns a new plan preserving immutability."""

    def test_with_cache_boundary_returns_new_instance(self) -> None:
        original = PromptAssemblyPlan(level=DisclosureLevel.CORE)
        modified = original.with_cache_boundary(
            CacheBoundary.COMMITTED, ("file_read", "text_search"),
        )
        assert modified is not original
        assert modified.cache_boundary is CacheBoundary.COMMITTED
        assert modified.stable_tool_names == ("file_read", "text_search")
        # Original unchanged
        assert original.cache_boundary is CacheBoundary.NONE
        assert original.stable_tool_names == ()

    def test_with_cache_boundary_preserves_other_fields(self) -> None:
        original = PromptAssemblyPlan(
            level=DisclosureLevel.FULL,
            reason="test",
            native_tools=True,
        )
        modified = original.with_cache_boundary(CacheBoundary.SOFT)
        assert modified.level is DisclosureLevel.FULL
        assert modified.reason == "test"
        assert modified.native_tools is True

    def test_metadata_includes_cache_boundary(self) -> None:
        plan = PromptAssemblyPlan(level=DisclosureLevel.CORE).with_cache_boundary(
            CacheBoundary.COMMITTED
        )
        meta = plan.metadata()
        assert "cache_boundary" in meta
        assert meta["cache_boundary"] == "committed"

    def test_default_cache_boundary_is_none(self) -> None:
        plan = PromptAssemblyPlan(level=DisclosureLevel.CORE)
        assert plan.cache_boundary is CacheBoundary.NONE
        assert plan.metadata()["cache_boundary"] == "none"


# ═══════════════════════════════════════════════════════════════════════════
# DisclosurePlanner.plan — cache-aware parameters
# ═══════════════════════════════════════════════════════════════════════════


class TestDisclosurePlannerCacheAware:
    """Cache-aware keyword parameters in DisclosurePlanner.plan."""

    def test_committed_freezes_disclosure_level(self) -> None:
        """COMMITTED status → plan reproduces frozen level, not default PCD."""
        planner = DisclosurePlanner()
        frozen_names = ("file_read", "text_search", "memory_search")
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.EXPANDED,
            committed_tool_names=frozen_names,
        )
        assert plan.cache_boundary is CacheBoundary.COMMITTED
        assert plan.level is DisclosureLevel.EXPANDED
        assert plan.stable_tool_names == frozen_names
        # Only committed tools appear in the plan
        plan_tool_names = _tool_names(plan)
        for name in frozen_names:
            assert name in plan_tool_names

    def test_committed_does_not_force_full(self) -> None:
        """Committed at CORE → plan stays CORE, PCD minimum-sufficiency preserved."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.CORE,
            committed_tool_names=("file_read",),
        )
        assert plan.level is DisclosureLevel.CORE
        assert plan.cache_boundary is CacheBoundary.COMMITTED

    def test_uncommitted_with_cache_benefit_produces_soft(self) -> None:
        """UNCOMMITTED + cache_benefit → SOFT boundary annotation."""
        planner = DisclosurePlanner()
        # Need a posture that triggers FULL for broader test coverage
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(
                native_tools_enabled=True,
                slash_command=True,
            ),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=True,
        )
        assert plan.cache_boundary is CacheBoundary.SOFT

    def test_uncommitted_without_cache_benefit_produces_none(self) -> None:
        """UNCOMMITTED without cache_benefit → NONE boundary."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=False,
        )
        assert plan.cache_boundary is CacheBoundary.NONE

    def test_default_no_cache_params_backward_compat(self) -> None:
        """No cache parameters → NONE boundary, identical to pre-cache behavior."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
        )
        assert plan.cache_boundary is CacheBoundary.NONE
        assert plan.stable_tool_names == ()

    def test_committed_full_level(self) -> None:
        """Committed at FULL → plan is FULL with committed boundary."""
        planner = DisclosurePlanner()
        all_names = tuple(
            d["function"]["name"] for d in _TOOL_CATALOG
        )
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.FULL,
            committed_tool_names=all_names,
        )
        assert plan.level is DisclosureLevel.FULL
        assert plan.cache_boundary is CacheBoundary.COMMITTED
        assert plan.stable_tool_names == all_names

    def test_soft_boundary_on_core_plan(self) -> None:
        """Uncommitted with cache_benefit at CORE baseline still gets SOFT."""
        planner = DisclosurePlanner()
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.UNCOMMITTED,
            cache_benefit=True,
        )
        assert plan.cache_boundary is CacheBoundary.SOFT


# ═══════════════════════════════════════════════════════════════════════════
# AnthropicCacheStrategy — system-prompt splitting & tool marker
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicCacheStrategy:
    """AnthropicCacheStrategy COMMITTED/SOFT prompt splitting and tool markers."""

    _STATIC_SECTION = (
        "You are LeapFlow.\n\n"
        "## Capabilities\nDo things.\n\n"
        "## Tool Usage\nUse tools wisely.\n\n"
        "## Guidelines\nBe helpful.\n\n"
        "When finished with all tool calls, provide a final answer.\n\n"
    )
    _DYNAMIC_SECTION = "## Memory Context\nRecent: user asked about X.\n"
    _FULL_PROMPT = _STATIC_SECTION + _DYNAMIC_SECTION

    def test_committed_splits_system_prompt(self) -> None:
        """COMMITTED → system content becomes [static (cached), dynamic]."""
        strategy = AnthropicCacheStrategy()
        msgs = [{"role": "system", "content": self._FULL_PROMPT}]
        result = strategy.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)

        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)
        assert len(sys_msg["content"]) == 2  # static + dynamic
        static_block = sys_msg["content"][0]
        assert static_block["type"] == "text"
        assert "cache_control" in static_block
        assert "Memory Context" in sys_msg["content"][1]["text"]

    def test_soft_also_splits_system_prompt(self) -> None:
        """SOFT → same split behavior as COMMITTED for the system prompt."""
        strategy = AnthropicCacheStrategy()
        msgs = [{"role": "system", "content": self._FULL_PROMPT}]
        result = strategy.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)

    def test_none_boundary_no_split(self) -> None:
        """NONE → no split, standard marker on whole system message."""
        strategy = AnthropicCacheStrategy()
        msgs = [{"role": "system", "content": self._FULL_PROMPT}]
        result = strategy.optimize(msgs, cache_boundary=CacheBoundary.NONE)

        sys_msg = result[0]
        # Standard Anthropic behavior: content wrapped in list with marker
        assert isinstance(sys_msg["content"], list)
        # But NOT a static/dynamic split — the whole content is one block
        assert len(sys_msg["content"]) == 1

    def test_split_system_prompt_anchor_found(self) -> None:
        """_split_system_prompt correctly splits on the terminal anchor."""
        static, dynamic = AnthropicCacheStrategy._split_system_prompt(self._FULL_PROMPT)
        assert "When finished with all tool calls" in static
        assert "Memory Context" in dynamic

    def test_split_system_prompt_no_dynamic(self) -> None:
        """System prompt with only static content → empty dynamic part."""
        static, dynamic = AnthropicCacheStrategy._split_system_prompt(self._STATIC_SECTION.rstrip())
        assert static == self._STATIC_SECTION.rstrip()
        assert dynamic == ""

    def test_apply_tool_cache_marker_committed(self) -> None:
        """COMMITTED → last tool gets cache_control marker, returns copy."""
        tools = [
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]
        original_tools = copy.deepcopy(tools)
        result = AnthropicCacheStrategy._apply_tool_cache_marker(tools, CacheBoundary.COMMITTED)

        assert result is not tools  # deep copy
        assert "cache_control" in result[-1]
        assert result[-1]["cache_control"]["type"] == "ephemeral"
        # Original untouched
        assert "cache_control" not in tools[-1]
        assert tools == original_tools

    def test_apply_tool_cache_marker_not_committed_returns_identity(self) -> None:
        """Non-COMMITTED → returns the SAME object (identity check)."""
        tools = [{"type": "function", "function": {"name": "a"}}]
        for boundary in (CacheBoundary.NONE, CacheBoundary.SOFT):
            result = AnthropicCacheStrategy._apply_tool_cache_marker(tools, boundary)
            assert result is tools  # identity — no copy

    def test_apply_tool_cache_marker_empty_tools(self) -> None:
        """Empty tools list returns the same list regardless of boundary."""
        empty: List[Dict[str, Any]] = []
        result = AnthropicCacheStrategy._apply_tool_cache_marker(empty, CacheBoundary.COMMITTED)
        assert result is empty


# ═══════════════════════════════════════════════════════════════════════════
# NoCacheStrategy — no-op regardless of boundary
# ═══════════════════════════════════════════════════════════════════════════


class TestNoCacheStrategy:
    """NoCacheStrategy passes messages through unchanged for all boundaries."""

    def test_noop_for_all_boundaries(self) -> None:
        strategy = NoCacheStrategy()
        msgs = [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        for boundary in CacheBoundary:
            result = strategy.optimize(msgs, cache_boundary=boundary)
            assert result is msgs  # identity — no copy, no mutation


# ═══════════════════════════════════════════════════════════════════════════
# Prefix stability proxy for cache-hit-rate verification
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefixStabilityProxy:
    """Simulate consecutive turns to verify prefix stability after commitment.

    The true verification target is that ≥70% of turns have a stable (identical)
    system-prompt prefix relative to the previous turn.  Because we use mock
    providers (no real Anthropic cached_tokens counter), we test the *structural
    precondition* for cache hits: the system-prompt prefix and tool array must
    be byte-stable across turns once committed.

    Proxy relationship: if the prefix is stable across N consecutive turns after
    commitment, then a provider with prefix caching will produce cache hits on
    turns 2..N (cache-hit ratio = (N-1)/N). For N=10 that is 90% — comfortably
    above the 70% acceptance threshold.
    """

    def _simulate_turns(self, n_turns: int = 10) -> dict:
        """Simulate n_turns of plan generation and prefix tracking."""
        planner = DisclosurePlanner()
        controller = PrefixCommitmentController()
        strategy = AnthropicCacheStrategy()

        # Simulate: first 3 turns are uncommitted, then force_commit
        system_prefixes: list[str] = []
        tool_arrays: list[str] = []
        boundaries: list[str] = []

        base_system = (
            "You are LeapFlow.\n\n## Capabilities\nDo things.\n\n"
            "## Tool Usage\nUse tools wisely.\n\n"
            "## Guidelines\nBe helpful.\n\n"
            "When finished with all tool calls, provide a final answer.\n\n"
            "## Memory Context\nSome memory.\n"
        )

        for turn in range(n_turns):
            if turn == 2:
                # Force commit at turn 2
                controller.force_commit()

            if controller.committed:
                enforcement = controller.enforce(
                    "full",
                    tuple(d["function"]["name"] for d in _TOOL_CATALOG),
                    _system_prompt_hash(base_system),
                    turn_index=turn,
                )
                plan = planner.plan(
                    _TOOL_CATALOG,
                    DisclosureRuntimeState(native_tools_enabled=True),
                    commitment_status=CommitmentStatus.COMMITTED,
                    committed_level=DisclosureLevel.FULL,
                    committed_tool_names=tuple(d["function"]["name"] for d in _TOOL_CATALOG),
                )
            else:
                plan = planner.plan(
                    _TOOL_CATALOG,
                    DisclosureRuntimeState(
                        native_tools_enabled=True,
                        slash_command=True,  # force FULL for comparison
                    ),
                )

            # Optimize system message through Anthropic strategy
            msgs = [{"role": "system", "content": base_system}]
            optimized = strategy.optimize(msgs, cache_boundary=plan.cache_boundary)
            sys_content = optimized[0].get("content", "")

            # Extract the stable prefix portion
            if isinstance(sys_content, list) and len(sys_content) >= 1:
                prefix = sys_content[0].get("text", "")
            elif isinstance(sys_content, str):
                prefix = sys_content
            else:
                prefix = str(sys_content)

            system_prefixes.append(prefix)
            # Serialize tool definitions for comparison
            import json
            tool_str = json.dumps(list(plan.tool_definitions), sort_keys=True)
            tool_arrays.append(tool_str)
            boundaries.append(plan.cache_boundary.value)

        return {
            "prefixes": system_prefixes,
            "tools": tool_arrays,
            "boundaries": boundaries,
            "n_turns": n_turns,
        }

    def test_prefix_stable_after_commitment(self) -> None:
        """After commitment, consecutive turns share the same system prefix.

        This structural stability is the precondition for provider prefix caching.
        Cache hit rate = (stable_turns - 1) / stable_turns.
        """
        result = self._simulate_turns(10)
        prefixes = result["prefixes"]

        # Turns 0-1 are uncommitted; turns 2-9 are committed
        committed_prefixes = prefixes[2:]
        assert len(committed_prefixes) == 8

        # All committed prefixes should be identical
        first = committed_prefixes[0]
        stable_count = sum(1 for p in committed_prefixes if p == first)
        stability_ratio = stable_count / len(committed_prefixes)
        assert stability_ratio >= 0.7, (
            f"Prefix stability {stability_ratio:.0%} below 70% threshold. "
            f"This proxy metric predicts provider cache-hit rate — stable prefixes "
            f"produce cache hits. See docstring for the proxy relationship."
        )
        # In practice, all committed turns should be perfectly stable
        assert stability_ratio == 1.0

    def test_tool_array_stable_after_commitment(self) -> None:
        """Committed tool arrays are identical across turns."""
        result = self._simulate_turns(10)
        committed_tools = result["tools"][2:]
        first = committed_tools[0]
        assert all(t == first for t in committed_tools)

    def test_committed_boundary_applied(self) -> None:
        """After commitment, all turns have COMMITTED boundary."""
        result = self._simulate_turns(10)
        for boundary in result["boundaries"][2:]:
            assert boundary == "committed"


# ═══════════════════════════════════════════════════════════════════════════
# PCD dynamism regression — break & restore
# ═══════════════════════════════════════════════════════════════════════════


class TestPCDDynamismRegression:
    """Breaking commitment restores normal PCD dynamics."""

    def test_break_restores_dynamic_pcd(self) -> None:
        """After commitment break, plan returns to normal PCD (NONE boundary)."""
        planner = DisclosurePlanner()
        controller = PrefixCommitmentController()

        # Commit and enforce
        controller.force_commit()
        controller.enforce(
            "full",
            tuple(d["function"]["name"] for d in _TOOL_CATALOG),
            "h1",
            turn_index=0,
        )

        # Verify committed plan
        committed_plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.FULL,
            committed_tool_names=tuple(d["function"]["name"] for d in _TOOL_CATALOG),
        )
        assert committed_plan.cache_boundary is CacheBoundary.COMMITTED

        # Break commitment (simulating posture change)
        assert controller.should_break_commitment(posture_changed=True)
        controller.break_commitment()

        # Next plan should be dynamic PCD with NONE boundary
        # Since enforcement is cleared, caller would pass UNCOMMITTED
        # (the status stays COMMITTED but enforcement is None)
        dynamic_plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
        )
        assert dynamic_plan.cache_boundary is CacheBoundary.NONE
        # Tool set can now change dynamically (PCD minimum-sufficiency)
        assert dynamic_plan.level is DisclosureLevel.CORE  # baseline

    def test_break_does_not_prevent_reestablishment(self) -> None:
        """After break, re-enforce can produce COMMITTED again."""
        planner = DisclosurePlanner()
        controller = PrefixCommitmentController()

        controller.force_commit()
        controller.enforce("full", ("a", "b"), "h1", turn_index=0)
        controller.break_commitment()
        assert controller.enforcement is None

        # Re-enforce with new state
        new_enforcement = controller.enforce("expanded", ("c",), "h2", turn_index=5)
        assert new_enforcement is not None
        plan = planner.plan(
            _TOOL_CATALOG,
            DisclosureRuntimeState(native_tools_enabled=True),
            commitment_status=CommitmentStatus.COMMITTED,
            committed_level=DisclosureLevel.EXPANDED,
            committed_tool_names=("c",),
        )
        assert plan.cache_boundary is CacheBoundary.COMMITTED

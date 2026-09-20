# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for prefix-stability layout: volatile context separation from stable prefix.

Validates that the system prompt assembly keeps a byte-stable prefix across
turns while still delivering all dynamic context (memory, knowledge, semantic
focus) to the model — just in a separate, post-prefix message.

No real LLM tokens consumed; no network or DuckDB.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest  # noqa: F401

from leapflow.engine.context_disclosure import CacheBoundary
from leapflow.engine.prompt_cache import (
    AnthropicCacheStrategy,
    NoCacheStrategy,
    PrefixCacheOptimizer,
)
from leapflow.prompts.templates import UNIFIED_SYSTEM_TEMPLATE


# ── Helpers ─────────────────────────────────────────────────────────────


def _format_stable_system(
    tool_catalog: str = "- tool_a: does A\n- tool_b: does B",
    app_connector_section: str = "",
    skill_section: str = "",
) -> str:
    """Format a system prompt using only the stable template fields."""
    return UNIFIED_SYSTEM_TEMPLATE.format(
        tool_catalog=tool_catalog,
        app_connector_section=app_connector_section,
        skill_section=skill_section,
    )


def _build_messages_with_volatile(
    system_text: str,
    volatile_context: str,
    prior_turns: List[Dict[str, Any]] | None = None,
    user_text: str = "hello",
) -> List[Dict[str, Any]]:
    """Simulate the message assembly logic from engine.py unified loops."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]
    if volatile_context:
        messages.append({
            "role": "system",
            "content": volatile_context,
            "_volatile_context": True,
        })
    if prior_turns:
        messages.extend(prior_turns)
    messages.append({"role": "user", "content": user_text})
    return messages


# ── Test: Stable prefix is byte-identical across turns ──────────────────


class TestStablePrefixByteIdentity:
    """Ensure the stable system prompt does not change when dynamic content varies."""

    def test_same_tools_same_prefix_bytes(self) -> None:
        """Two turns with identical tool catalog must produce identical system text."""
        sys1 = _format_stable_system(tool_catalog="- tool_a: does A")
        sys2 = _format_stable_system(tool_catalog="- tool_a: does A")
        assert sys1 == sys2
        assert sys1.encode("utf-8") == sys2.encode("utf-8")

    def test_different_volatile_same_stable_prefix(self) -> None:
        """Varying memory/knowledge across turns should not affect the stable prefix."""
        stable = _format_stable_system()
        msgs_turn1 = _build_messages_with_volatile(stable, "memory: turn-1 context")
        msgs_turn2 = _build_messages_with_volatile(stable, "memory: turn-2 different context")

        # First message (stable system) must be byte-identical
        assert msgs_turn1[0] == msgs_turn2[0]
        assert msgs_turn1[0]["content"].encode("utf-8") == msgs_turn2[0]["content"].encode("utf-8")

    def test_volatile_context_still_present(self) -> None:
        """Dynamic content must still reach the model in a separate message."""
        stable = _format_stable_system()
        volatile = "## Knowledge\nSome important facts\n\n## Memory\nUser prefers dark mode"
        msgs = _build_messages_with_volatile(stable, volatile)

        volatile_msgs = [m for m in msgs if m.get("_volatile_context")]
        assert len(volatile_msgs) == 1
        assert volatile_msgs[0]["content"] == volatile
        assert volatile_msgs[0]["role"] == "system"

    def test_empty_volatile_omitted(self) -> None:
        """When volatile_context is empty, no extra message is inserted."""
        stable = _format_stable_system()
        msgs = _build_messages_with_volatile(stable, "")
        assert len(msgs) == 2  # system + user only
        assert all(not m.get("_volatile_context") for m in msgs)

    def test_volatile_follows_stable_precedes_prior_turns(self) -> None:
        """Volatile message must be between stable system and prior turns."""
        stable = _format_stable_system()
        prior = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        msgs = _build_messages_with_volatile(stable, "dynamic stuff", prior_turns=prior)

        # Order: system(stable), system(volatile), user(prev), assistant(prev), user(current)
        assert msgs[0]["role"] == "system" and not msgs[0].get("_volatile_context")
        assert msgs[1]["role"] == "system" and msgs[1].get("_volatile_context") is True
        assert msgs[2]["role"] == "user" and msgs[2]["content"] == "prev question"
        assert msgs[-1]["role"] == "user" and msgs[-1]["content"] == "hello"

    def test_template_has_no_memory_context_placeholder(self) -> None:
        """UNIFIED_SYSTEM_TEMPLATE must not contain {memory_context}."""
        assert "{memory_context}" not in UNIFIED_SYSTEM_TEMPLATE


# ── Test: PrefixCacheOptimizer volatile exclusion ───────────────────────


class TestPrefixCacheOptimizerVolatileExclusion:
    """Volatile-context messages must land in the dynamic section, not stable."""

    def _make_messages(self, volatile: str = "memory context here") -> List[Dict[str, Any]]:
        return _build_messages_with_volatile(
            _format_stable_system(),
            volatile,
            prior_turns=[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
        )

    def test_volatile_excluded_from_stable_soft(self) -> None:
        """Under SOFT boundary, _volatile_context messages go to dynamic."""
        opt = PrefixCacheOptimizer()
        msgs = self._make_messages()
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # The first message (stable system) should come first
        assert result[0]["role"] == "system"
        assert not result[0].get("_volatile_context")

        # Volatile message should be AFTER the stable prefix section
        stable_end = 0
        for i, m in enumerate(result):
            if m.get("role") == "system" and not m.get("_volatile_context"):
                stable_end = i
        volatile_indices = [i for i, m in enumerate(result) if m.get("_volatile_context")]
        assert volatile_indices, "volatile message missing from output"
        for vi in volatile_indices:
            assert vi > stable_end, "volatile msg should be after stable system prefix"

    def test_volatile_excluded_from_stable_none(self) -> None:
        """Under NONE boundary, same volatile exclusion behavior."""
        opt = PrefixCacheOptimizer()
        msgs = self._make_messages()
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.NONE)

        volatile_msgs = [m for m in result if m.get("_volatile_context")]
        stable_sys = [m for m in result if m.get("role") == "system" and not m.get("_volatile_context")]
        assert len(volatile_msgs) == 1
        assert len(stable_sys) >= 1

        # volatile must appear after the last stable system message
        last_stable_idx = max(i for i, m in enumerate(result) if m.get("role") == "system" and not m.get("_volatile_context"))
        volatile_idx = next(i for i, m in enumerate(result) if m.get("_volatile_context"))
        assert volatile_idx > last_stable_idx

    def test_committed_passthrough_preserves_volatile(self) -> None:
        """COMMITTED path preserves order; volatile is still present."""
        opt = PrefixCacheOptimizer()
        msgs = self._make_messages()
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.COMMITTED)

        volatile_msgs = [m for m in result if m.get("_volatile_context")]
        assert len(volatile_msgs) == 1
        assert volatile_msgs[0]["content"] == "memory context here"

    def test_no_volatile_still_works(self) -> None:
        """Optimizer works correctly when no volatile messages exist."""
        opt = PrefixCacheOptimizer()
        msgs = _build_messages_with_volatile(_format_stable_system(), "")
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.SOFT)
        assert len(result) == 2  # system + user
        assert result[0]["role"] == "system"

    def test_cache_marker_on_stable_not_volatile(self) -> None:
        """Cache marker should be on the last *stable* message, not volatile."""
        opt = PrefixCacheOptimizer()
        msgs = self._make_messages()
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # Find all stable system messages (non-volatile)
        stable_sys = [m for m in result if m.get("role") == "system" and not m.get("_volatile_context")]
        assert stable_sys, "Expected at least one stable system message"
        # The last stable system msg should have cache_control
        assert "cache_control" in stable_sys[-1]

        # Volatile should NOT have cache_control added by the optimizer
        volatile_msgs = [m for m in result if m.get("_volatile_context")]
        for vm in volatile_msgs:
            assert "cache_control" not in vm

    def test_multiple_system_msgs_with_volatile(self) -> None:
        """Multiple system messages: non-volatile stays stable, volatile goes dynamic."""
        opt = PrefixCacheOptimizer()
        msgs = [
            {"role": "system", "content": "identity"},
            {"role": "system", "content": "guidelines"},
            {"role": "system", "content": "per-turn memory", "_volatile_context": True},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
        ]
        result = opt.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # First two should be stable (system), then volatile + conversation
        assert result[0]["role"] == "system" and result[0]["content"] == "identity"
        assert result[1]["role"] == "system" and result[1]["content"] == "guidelines"

        # Volatile should be after the two stable system messages
        volatile_idx = next(i for i, m in enumerate(result) if m.get("_volatile_context"))
        assert volatile_idx >= 2


# ── Test: AnthropicCacheStrategy ────────────────────────────────────────


class TestAnthropicCacheStrategyVolatile:
    """AnthropicCacheStrategy should handle volatile messages gracefully."""

    def test_volatile_msg_not_split(self) -> None:
        """Volatile system message should not undergo static/dynamic splitting."""
        strategy = AnthropicCacheStrategy()
        msgs = _build_messages_with_volatile(
            _format_stable_system(), "dynamic memory content",
        )
        result = strategy.optimize(msgs, cache_boundary=CacheBoundary.SOFT)

        # The stable system message may be split, but volatile should stay intact
        volatile_msgs = [m for m in result if m.get("_volatile_context")]
        assert len(volatile_msgs) == 1

    def test_no_cache_ttl_param(self) -> None:
        """AnthropicCacheStrategy no longer accepts cache_ttl parameter."""
        import inspect
        sig = inspect.signature(AnthropicCacheStrategy.__init__)
        assert "cache_ttl" not in sig.parameters

    def test_constructor_defaults(self) -> None:
        """Default constructor works without any params."""
        strategy = AnthropicCacheStrategy()
        assert strategy._breakpoints == 3


# ── Test: NoCacheStrategy passthrough ───────────────────────────────────


class TestNoCacheStrategyPassthrough:
    """NoCacheStrategy must pass through volatile messages unchanged."""

    def test_volatile_preserved(self) -> None:
        strategy = NoCacheStrategy()
        msgs = _build_messages_with_volatile(_format_stable_system(), "volatile data")
        result = strategy.optimize(msgs)
        assert result == msgs

    def test_volatile_flag_intact(self) -> None:
        strategy = NoCacheStrategy()
        msgs = _build_messages_with_volatile(_format_stable_system(), "some memory")
        result = strategy.optimize(msgs)
        volatile = [m for m in result if m.get("_volatile_context")]
        assert len(volatile) == 1
        assert volatile[0]["content"] == "some memory"


# ── Test: End-to-end prefix stability simulation ────────────────────────


class TestEndToEndPrefixStability:
    """Simulate two consecutive turns with varying volatile but same tools."""

    def _simulate_turn(
        self,
        tool_catalog: str,
        volatile: str,
        boundary: CacheBoundary,
        prior: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Simulate assembly + optimizer pipeline."""
        stable = _format_stable_system(tool_catalog=tool_catalog)
        msgs = _build_messages_with_volatile(stable, volatile, prior_turns=prior)
        opt = PrefixCacheOptimizer()
        return opt.optimize(msgs, cache_boundary=boundary)

    def test_two_turns_soft_prefix_identical(self) -> None:
        """Under SOFT, two turns with same tools but different volatile have identical prefix."""
        catalog = "- tool_a: action A\n- tool_b: action B"
        turn1 = self._simulate_turn(catalog, "memory: session-1 data", CacheBoundary.SOFT)
        turn2 = self._simulate_turn(catalog, "memory: session-2 different data", CacheBoundary.SOFT)

        # Extract stable prefix from both turns
        stable1 = [m for m in turn1 if m.get("role") == "system" and not m.get("_volatile_context")]
        stable2 = [m for m in turn2 if m.get("role") == "system" and not m.get("_volatile_context")]

        assert len(stable1) == len(stable2)
        for s1, s2 in zip(stable1, stable2):
            # Compare content bytes (the core cache-hit requirement)
            assert s1["content"].encode("utf-8") == s2["content"].encode("utf-8")

    def test_two_turns_none_prefix_identical(self) -> None:
        """Under NONE, same byte-stability guarantee."""
        catalog = "- search: find things"
        turn1 = self._simulate_turn(catalog, "knowledge: fact-A", CacheBoundary.NONE)
        turn2 = self._simulate_turn(catalog, "knowledge: fact-B", CacheBoundary.NONE)

        stable1 = [m["content"] for m in turn1 if m.get("role") == "system" and not m.get("_volatile_context")]
        stable2 = [m["content"] for m in turn2 if m.get("role") == "system" and not m.get("_volatile_context")]
        assert stable1 == stable2

    def test_two_turns_committed_prefix_identical(self) -> None:
        """Under COMMITTED, prefix byte-stability also holds."""
        catalog = "- edit: modify files"
        turn1 = self._simulate_turn(catalog, "mem: x", CacheBoundary.COMMITTED)
        turn2 = self._simulate_turn(catalog, "mem: y", CacheBoundary.COMMITTED)

        stable1 = [m["content"] for m in turn1 if m.get("role") == "system" and not m.get("_volatile_context")]
        stable2 = [m["content"] for m in turn2 if m.get("role") == "system" and not m.get("_volatile_context")]
        assert stable1 == stable2

    def test_volatile_content_complete_across_turns(self) -> None:
        """Volatile content must be fully present in each turn's output."""
        catalog = "- tool_x: x"
        vol1 = "## Knowledge\nfact1\n\n## Memory\nuser pref A"
        vol2 = "## Knowledge\nfact2\n\n## Memory\nuser pref B"

        turn1 = self._simulate_turn(catalog, vol1, CacheBoundary.SOFT)
        turn2 = self._simulate_turn(catalog, vol2, CacheBoundary.SOFT)

        volatile1 = [m for m in turn1 if m.get("_volatile_context")]
        volatile2 = [m for m in turn2 if m.get("_volatile_context")]

        assert len(volatile1) == 1
        assert volatile1[0]["content"] == vol1
        assert len(volatile2) == 1
        assert volatile2[0]["content"] == vol2

    def test_different_tools_different_prefix(self) -> None:
        """When tool catalog changes (disclosure level shift), prefix naturally differs."""
        turn1 = self._simulate_turn("- tool_a: A", "mem", CacheBoundary.SOFT)
        turn2 = self._simulate_turn("- tool_a: A\n- tool_b: B", "mem", CacheBoundary.SOFT)

        stable1 = [m["content"] for m in turn1 if m.get("role") == "system" and not m.get("_volatile_context")]
        stable2 = [m["content"] for m in turn2 if m.get("role") == "system" and not m.get("_volatile_context")]
        # Different tool catalogs → different prefix (expected, not a bug)
        assert stable1 != stable2

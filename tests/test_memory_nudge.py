# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the periodic memory nudge policy and EventBus integration."""
from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from leapflow.memory.nudge import MemoryNudgePolicy, MemoryNudgeTriggered


# ── MemoryNudgePolicy: should_nudge interval gating ──────────────────────


class TestShouldNudgeInterval:
    """Verify the turn-interval gate."""

    def test_fires_after_interval(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=5, min_idle_seconds=0)
        assert policy.should_nudge(turn_count=5, idle_seconds=0)

    def test_does_not_fire_before_interval(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=10, min_idle_seconds=0)
        assert not policy.should_nudge(turn_count=5, idle_seconds=100)

    def test_interval_resets_after_nudge(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=5, min_idle_seconds=0)
        assert policy.should_nudge(turn_count=5, idle_seconds=0)
        policy.record_nudge(turn_count=5)
        # Immediately after recording, another 5 turns must pass.
        assert not policy.should_nudge(turn_count=7, idle_seconds=100)
        assert policy.should_nudge(turn_count=10, idle_seconds=0)


# ── MemoryNudgePolicy: idle-time threshold ────────────────────────────────


class TestShouldNudgeIdle:
    """Verify the idle-time gate."""

    def test_not_idle_enough(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=1, min_idle_seconds=30)
        assert not policy.should_nudge(turn_count=1, idle_seconds=10)

    def test_idle_threshold_met(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=1, min_idle_seconds=30)
        assert policy.should_nudge(turn_count=1, idle_seconds=30)

    def test_idle_threshold_exceeded(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=1, min_idle_seconds=30)
        assert policy.should_nudge(turn_count=1, idle_seconds=60)


# ── MemoryNudgePolicy: max nudges per session ────────────────────────────


class TestMaxNudges:
    """Verify the per-session cap."""

    def test_cap_respected(self) -> None:
        policy = MemoryNudgePolicy(
            interval_turns=1, min_idle_seconds=0, max_nudges_per_session=2
        )
        assert policy.should_nudge(turn_count=1, idle_seconds=0)
        policy.record_nudge(turn_count=1)
        assert policy.should_nudge(turn_count=2, idle_seconds=0)
        policy.record_nudge(turn_count=2)
        # Third nudge exceeds cap.
        assert not policy.should_nudge(turn_count=3, idle_seconds=999)

    def test_nudge_count_property(self) -> None:
        policy = MemoryNudgePolicy(interval_turns=1, min_idle_seconds=0)
        assert policy.nudge_count == 0
        policy.record_nudge()
        assert policy.nudge_count == 1

    def test_reset_clears_counters(self) -> None:
        policy = MemoryNudgePolicy(
            interval_turns=1, min_idle_seconds=0, max_nudges_per_session=1
        )
        policy.record_nudge(turn_count=1)
        assert not policy.should_nudge(turn_count=2, idle_seconds=100)
        policy.reset()
        assert policy.nudge_count == 0
        assert policy.should_nudge(turn_count=1, idle_seconds=0)


# ── MemoryNudgePolicy: constructor validation ────────────────────────────


class TestConstructorValidation:
    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_turns"):
            MemoryNudgePolicy(interval_turns=0)

    def test_negative_idle_raises(self) -> None:
        with pytest.raises(ValueError, match="min_idle_seconds"):
            MemoryNudgePolicy(min_idle_seconds=-1)

    def test_negative_max_nudges_raises(self) -> None:
        with pytest.raises(ValueError, match="max_nudges_per_session"):
            MemoryNudgePolicy(max_nudges_per_session=-1)


# ── MemoryNudgePolicy: nudge prompt construction ─────────────────────────


class TestBuildNudgePrompt:
    def test_empty_turns_returns_empty(self) -> None:
        policy = MemoryNudgePolicy()
        assert policy.build_nudge_prompt([]) == ""

    def test_turns_with_no_content_returns_empty(self) -> None:
        policy = MemoryNudgePolicy()
        assert policy.build_nudge_prompt([{"role": "user", "content": ""}]) == ""

    def test_prompt_contains_role_and_content(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [
            {"role": "user", "content": "Please remember my timezone is UTC+8"},
            {"role": "assistant", "content": "Noted."},
        ]
        prompt = policy.build_nudge_prompt(turns)
        assert "Memory Review Nudge" in prompt
        assert "user:" in prompt
        assert "UTC+8" in prompt
        assert "assistant:" in prompt

    def test_prompt_truncates_long_content(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [{"role": "user", "content": "x" * 500}]
        prompt = policy.build_nudge_prompt(turns)
        # Content is truncated to 300 chars inside the digest;
        # the full prompt includes the template chrome as well.
        assert "x" * 300 in prompt
        assert "x" * 301 not in prompt

    def test_prompt_includes_categories(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [{"role": "user", "content": "hello"}]
        prompt = policy.build_nudge_prompt(turns)
        assert "user preferences" in prompt
        assert "architectural decisions" in prompt


# ── MemoryNudgePolicy: topic extraction ───────────────────────────────────


class TestExtractTopics:
    def test_extracts_tool_names(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "file_read", "arguments": "{}"}},
                    {"function": {"name": "web_fetch", "arguments": "{}"}},
                ],
            }
        ]
        topics = policy.extract_topics(turns)
        assert "tool:file_read" in topics
        assert "tool:web_fetch" in topics

    def test_extracts_user_context_for_long_messages(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [{"role": "user", "content": "a" * 100}]
        topics = policy.extract_topics(turns)
        assert "user_context" in topics

    def test_short_user_message_no_topic(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [{"role": "user", "content": "hi"}]
        topics = policy.extract_topics(turns)
        assert "user_context" not in topics

    def test_deduplicates_tool_names(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "file_read", "arguments": ""}},
                    {"function": {"name": "file_read", "arguments": ""}},
                ],
            }
        ]
        topics = policy.extract_topics(turns)
        assert topics.count("tool:file_read") == 1

    def test_caps_at_ten(self) -> None:
        policy = MemoryNudgePolicy()
        turns = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": f"tool_{i}", "arguments": ""}}
                    for i in range(20)
                ],
            }
        ]
        assert len(policy.extract_topics(turns)) <= 10


# ── MemoryNudgeTriggered event ────────────────────────────────────────────


class TestMemoryNudgeTriggeredEvent:
    def test_frozen_dataclass(self) -> None:
        evt = MemoryNudgeTriggered(session_id="s1", turn_count=10)
        assert evt.session_id == "s1"
        assert evt.turn_count == 10
        assert evt.suggested_topics == ()
        with pytest.raises(AttributeError):
            evt.session_id = "s2"  # type: ignore[misc]

    def test_with_topics(self) -> None:
        evt = MemoryNudgeTriggered(
            session_id="s1",
            turn_count=5,
            suggested_topics=("tool:file_read", "user_context"),
        )
        assert len(evt.suggested_topics) == 2

    def test_has_timestamp(self) -> None:
        before = time.time()
        evt = MemoryNudgeTriggered(session_id="s1", turn_count=1)
        after = time.time()
        assert before <= evt.timestamp <= after


# ── LearningBridge integration: EventBus emission ────────────────────────


class TestLearningBridgeNudgeIntegration:
    """Verify _maybe_nudge emits the event via EventBus on the LearningBridge."""

    @pytest.fixture()
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine._event_bus = AsyncMock()
        engine._event_bus.handle_event = AsyncMock()
        engine._turn_count = 15
        engine._current_session_id = "test-session"
        engine._evolution = MagicMock()
        engine._evolution.record_episode = MagicMock(return_value=None)
        engine._usage_tracker = MagicMock()
        engine._usage_tracker.to_learning_signal = MagicMock(return_value={})
        engine._last_context_snapshot = {}
        engine._settings = MagicMock()
        engine._settings.memory_integration_enabled = True
        return engine

    @pytest.mark.asyncio
    async def test_nudge_emits_event(self, mock_engine: MagicMock) -> None:
        from leapflow.engine.learning_bridge import LearningBridge

        bridge = LearningBridge(mock_engine)
        # Configure policy so nudge fires immediately.
        bridge._nudge_policy = MemoryNudgePolicy(
            interval_turns=1, min_idle_seconds=0, max_nudges_per_session=5
        )
        # Pretend we've been idle long enough.
        bridge._last_turn_end = time.monotonic() - 60

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": "remember my preference for dark mode"}
        ]

        await bridge._maybe_nudge(messages)

        mock_engine._event_bus.handle_event.assert_called_once()
        call_args = mock_engine._event_bus.handle_event.call_args
        assert call_args[0][0] == "memory.nudge_triggered"
        payload = call_args[0][1]
        assert payload["session_id"] == "test-session"
        assert payload["turn_count"] == 15
        assert bridge._nudge_policy.nudge_count == 1

    @pytest.mark.asyncio
    async def test_nudge_not_emitted_when_conditions_unmet(
        self, mock_engine: MagicMock
    ) -> None:
        from leapflow.engine.learning_bridge import LearningBridge

        bridge = LearningBridge(mock_engine)
        # Default policy: interval=10, idle=30s — turn_count=15 from 0 is OK
        # but idle_seconds will be ~0 since _last_turn_end is fresh.
        bridge._last_turn_end = time.monotonic()

        messages: List[Dict[str, Any]] = [{"role": "user", "content": "hi"}]
        await bridge._maybe_nudge(messages)

        mock_engine._event_bus.handle_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_nudge_no_event_bus(self, mock_engine: MagicMock) -> None:
        from leapflow.engine.learning_bridge import LearningBridge

        mock_engine._event_bus = None
        bridge = LearningBridge(mock_engine)
        bridge._nudge_policy = MemoryNudgePolicy(
            interval_turns=1, min_idle_seconds=0
        )
        bridge._last_turn_end = time.monotonic() - 60

        messages: List[Dict[str, Any]] = [{"role": "user", "content": "test"}]
        # Should not raise even without an EventBus.
        await bridge._maybe_nudge(messages)
        assert bridge._nudge_policy.nudge_count == 0

    @pytest.mark.asyncio
    async def test_nudge_respects_session_cap(
        self, mock_engine: MagicMock
    ) -> None:
        from leapflow.engine.learning_bridge import LearningBridge

        bridge = LearningBridge(mock_engine)
        bridge._nudge_policy = MemoryNudgePolicy(
            interval_turns=1,
            min_idle_seconds=0,
            max_nudges_per_session=1,
        )
        bridge._last_turn_end = time.monotonic() - 60

        messages: List[Dict[str, Any]] = [{"role": "user", "content": "test"}]

        await bridge._maybe_nudge(messages)
        assert bridge._nudge_policy.nudge_count == 1
        mock_engine._event_bus.handle_event.assert_called_once()

        # Second attempt should be blocked by cap.
        mock_engine._event_bus.handle_event.reset_mock()
        bridge._last_turn_end = time.monotonic() - 60
        mock_engine._turn_count = 20
        await bridge._maybe_nudge(messages)
        mock_engine._event_bus.handle_event.assert_not_called()

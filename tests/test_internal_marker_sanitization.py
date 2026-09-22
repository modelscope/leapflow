# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for internal marker sanitization.

Covers two fixes:
1. OpenAIChat strips ``_``-prefixed internal keys before sending to the SDK.
2. AnthropicCacheStrategy skips ``_volatile_context`` messages when placing
   cache breakpoints.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from leapflow.engine.context.context_disclosure import CacheBoundary
from leapflow.engine.prompt_cache import AnthropicCacheStrategy
from leapflow.llm.openai_provider import OpenAIChat, _sanitize_messages


# ── Helper fixtures ────────────────────────────────────────────────────────


def _make_messages_with_internal_markers() -> List[Dict[str, Any]]:
    """Return a realistic message list containing various internal markers."""
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "_volatile_context": True,
        },
        {
            "role": "system",
            "content": "Stable system prompt.",
            "_compressed_summary": True,
        },
        {
            "role": "user",
            "content": "Hello",
            "_frozen_memory": True,
            "_DB_PERSISTED_ID": "abc-123",
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "cache_control": {"type": "ephemeral"},
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: _sanitize_messages helper
# ═══════════════════════════════════════════════════════════════════════════


class TestSanitizeMessages:
    """Unit tests for the ``_sanitize_messages`` helper."""

    def test_strips_underscore_prefixed_keys(self):
        msgs = _make_messages_with_internal_markers()
        result = _sanitize_messages(msgs)

        for msg in result:
            for key in msg:
                assert not key.startswith("_"), f"Internal key leaked: {key}"

    def test_preserves_standard_fields(self):
        msgs = _make_messages_with_internal_markers()
        result = _sanitize_messages(msgs)

        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1] == {"role": "system", "content": "Stable system prompt."}
        assert result[2] == {"role": "user", "content": "Hello"}
        assert result[3] == {
            "role": "assistant",
            "content": "Hi there!",
            "cache_control": {"type": "ephemeral"},
        }

    def test_does_not_mutate_original(self):
        msgs = _make_messages_with_internal_markers()
        original = copy.deepcopy(msgs)
        _sanitize_messages(msgs)

        assert msgs == original, "Original messages were mutated"

    def test_empty_list(self):
        assert _sanitize_messages([]) == []

    def test_message_with_no_internal_keys(self):
        msgs = [{"role": "user", "content": "plain message"}]
        result = _sanitize_messages(msgs)
        assert result == msgs
        # Still a new list (not the same object)
        assert result is not msgs

    def test_preserves_tool_call_id_and_name(self):
        msgs = [
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "call_123",
                "name": "my_tool",
                "_volatile_context": True,
            }
        ]
        result = _sanitize_messages(msgs)
        assert result == [
            {"role": "tool", "content": "result", "tool_call_id": "call_123", "name": "my_tool"}
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: OpenAIChat send-path sanitization
# ═══════════════════════════════════════════════════════════════════════════


def _make_openai_chat() -> OpenAIChat:
    """Create an OpenAIChat instance with a dummy config."""
    return OpenAIChat(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
    )


def _mock_completion_response():
    """Return a mock that looks like an OpenAI ChatCompletion."""
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.role = "assistant"
    choice.message.tool_calls = None
    choice.message.reasoning_content = None
    choice.finish_reason = "stop"

    resp = MagicMock()
    resp.choices = [choice]
    resp.model = "test-model"
    resp.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
    )
    resp.usage.prompt_tokens_details = None
    resp.usage.prompt_cache_hit_tokens = None
    return resp


class TestOpenAIChatSanitizationAsync:
    """Verify that OpenAIChat.achat strips internal markers before SDK call."""

    @pytest.mark.asyncio
    async def test_achat_nonstream_strips_markers(self):
        client = _make_openai_chat()
        msgs = _make_messages_with_internal_markers()
        original = copy.deepcopy(msgs)

        mock_resp = _mock_completion_response()
        with patch.object(
            client._async.chat.completions, "create",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_create:
            await client.achat(msgs, stream=False)

            sent_msgs = mock_create.call_args[1].get(
                "messages", mock_create.call_args[0][0] if mock_create.call_args[0] else None,
            )
            if sent_msgs is None:
                sent_msgs = mock_create.call_args.kwargs["messages"]

            for msg in sent_msgs:
                for key in msg:
                    assert not key.startswith("_"), f"Internal key leaked: {key}"

        # Original not mutated.
        assert msgs == original

    @pytest.mark.asyncio
    async def test_achat_stream_collapsed_strips_markers(self):
        client = _make_openai_chat()
        msgs = _make_messages_with_internal_markers()

        # Build an async iterator mock for streaming.
        async def _fake_stream():
            chunk = MagicMock()
            chunk.model = "test-model"
            chunk.choices = [MagicMock()]
            chunk.choices[0].finish_reason = "stop"
            chunk.choices[0].delta.content = "ok"
            chunk.choices[0].delta.reasoning_content = None
            chunk.usage = MagicMock(
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            )
            chunk.usage.prompt_tokens_details = None
            chunk.usage.prompt_cache_hit_tokens = None
            yield chunk

        with patch.object(
            client._async.chat.completions, "create",
            new_callable=AsyncMock, return_value=_fake_stream(),
        ) as mock_create:
            await client.achat(msgs, stream=True)

            sent_msgs = mock_create.call_args.kwargs["messages"]
            for msg in sent_msgs:
                for key in msg:
                    assert not key.startswith("_"), f"Internal key leaked: {key}"


class TestOpenAIChatSanitizationSync:
    """Verify that OpenAIChat.chat (sync) strips internal markers."""

    def test_chat_nonstream_strips_markers(self):
        client = _make_openai_chat()
        msgs = _make_messages_with_internal_markers()
        original = copy.deepcopy(msgs)

        mock_resp = _mock_completion_response()
        with patch.object(
            client._sync.chat.completions, "create",
            return_value=mock_resp,
        ) as mock_create:
            client.chat(msgs, stream=False)

            sent_msgs = mock_create.call_args.kwargs["messages"]
            for msg in sent_msgs:
                for key in msg:
                    assert not key.startswith("_"), f"Internal key leaked: {key}"

        assert msgs == original

    def test_chat_stream_collapsed_strips_markers(self):
        client = _make_openai_chat()
        msgs = _make_messages_with_internal_markers()

        chunk = MagicMock()
        chunk.model = "test-model"
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = "stop"
        chunk.choices[0].delta.content = "ok"
        chunk.choices[0].delta.reasoning_content = None
        chunk.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        chunk.usage.prompt_tokens_details = None
        chunk.usage.prompt_cache_hit_tokens = None

        with patch.object(
            client._sync.chat.completions, "create",
            return_value=iter([chunk]),
        ) as mock_create:
            client.chat(msgs, stream=True)

            sent_msgs = mock_create.call_args.kwargs["messages"]
            for msg in sent_msgs:
                for key in msg:
                    assert not key.startswith("_"), f"Internal key leaked: {key}"


class TestOpenAIChatAchatStreamSanitization:
    """Verify that the ``achat_stream`` async generator also sanitizes."""

    @pytest.mark.asyncio
    async def test_achat_stream_generator_strips_markers(self):
        client = _make_openai_chat()
        msgs = _make_messages_with_internal_markers()
        original = copy.deepcopy(msgs)

        async def _fake_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "ok"
            yield chunk

        with patch.object(
            client._async.chat.completions, "create",
            new_callable=AsyncMock, return_value=_fake_stream(),
        ) as mock_create:
            collected = []
            async for text in client.achat_stream(msgs):
                collected.append(text)

            sent_msgs = mock_create.call_args.kwargs["messages"]
            for msg in sent_msgs:
                for key in msg:
                    assert not key.startswith("_"), f"Internal key leaked: {key}"

        assert msgs == original


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: AnthropicCacheStrategy — volatile messages skip
# ═══════════════════════════════════════════════════════════════════════════


class TestAnthropicCacheStrategyVolatileSkip:
    """Verify _volatile_context system messages are not given cache breakpoints."""

    def test_volatile_system_message_not_marked(self):
        strategy = AnthropicCacheStrategy(breakpoints=3)
        messages = [
            {"role": "system", "content": "Stable system prompt."},
            {
                "role": "system",
                "content": "Dynamic memory context.",
                "_volatile_context": True,
            },
            {"role": "user", "content": "Hello"},
        ]
        result = strategy.optimize(messages)

        # Stable system message SHOULD have a cache marker.
        stable_sys = result[0]
        content = stable_sys.get("content")
        if isinstance(content, list):
            assert any("cache_control" in block for block in content)
        else:
            assert "cache_control" in stable_sys

        # Volatile system message should NOT have any cache marker.
        volatile_sys = result[1]
        volatile_content = volatile_sys.get("content")
        if isinstance(volatile_content, list):
            assert not any("cache_control" in block for block in volatile_content), (
                "Volatile system message should not have cache_control on content blocks"
            )
        elif isinstance(volatile_content, str):
            assert "cache_control" not in volatile_sys, (
                "Volatile system message should not have cache_control"
            )

    def test_volatile_skipped_with_soft_boundary(self):
        strategy = AnthropicCacheStrategy(breakpoints=3)
        messages = [
            {"role": "system", "content": "Stable.\n## Capabilities\nSome caps."},
            {
                "role": "system",
                "content": "Volatile memory.",
                "_volatile_context": True,
            },
            {"role": "user", "content": "Hello"},
        ]
        result = strategy.optimize(messages, cache_boundary=CacheBoundary.SOFT)

        # Volatile message must not receive split-marker or any cache_control.
        volatile_sys = result[1]
        volatile_content = volatile_sys.get("content")
        if isinstance(volatile_content, list):
            assert not any("cache_control" in block for block in volatile_content)
        else:
            assert "cache_control" not in volatile_sys

    def test_volatile_skipped_with_committed_boundary(self):
        strategy = AnthropicCacheStrategy(breakpoints=3)
        messages = [
            {"role": "system", "content": "Stable system prompt."},
            {
                "role": "system",
                "content": "Volatile knowledge.",
                "_volatile_context": True,
            },
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = strategy.optimize(messages, cache_boundary=CacheBoundary.COMMITTED)

        volatile_sys = result[1]
        volatile_content = volatile_sys.get("content")
        if isinstance(volatile_content, list):
            assert not any("cache_control" in block for block in volatile_content)
        else:
            assert "cache_control" not in volatile_sys

    def test_stable_system_still_marked_when_volatile_present(self):
        """Stable system messages must still receive markers even when volatile is present."""
        strategy = AnthropicCacheStrategy(breakpoints=3)
        messages = [
            {"role": "system", "content": "Stable system prompt."},
            {
                "role": "system",
                "content": "Volatile context.",
                "_volatile_context": True,
            },
            {"role": "user", "content": "Hello"},
        ]
        result = strategy.optimize(messages)

        stable_sys = result[0]
        content = stable_sys.get("content")
        # The stable system message must have a cache_control marker.
        if isinstance(content, list):
            assert any("cache_control" in block for block in content)
        else:
            assert "cache_control" in stable_sys

    def test_no_volatile_messages_unchanged_behavior(self):
        """Without volatile messages, behavior is identical to before."""
        strategy = AnthropicCacheStrategy(breakpoints=2)
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = strategy.optimize(messages)

        # System should be marked.
        sys_msg = result[0]
        content = sys_msg.get("content")
        if isinstance(content, list):
            assert any("cache_control" in block for block in content)
        else:
            assert "cache_control" in sys_msg

        # Last 2 non-system messages should be marked (conversation tail).
        assert "cache_control" in result[-1] or (
            isinstance(result[-1].get("content"), list) and
            any("cache_control" in b for b in result[-1]["content"])
        )

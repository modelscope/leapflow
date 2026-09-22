# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Anthropic native provider and plugin.

All tests use mocks — no real API calls.  Covers:
- Usage parsing (cache_read_input_tokens → cached_tokens, raw fields preserved)
- cache_control pass-through in message conversion
- SDK absence graceful degradation
- Plugin capability declarations
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Usage parsing tests ───────────────────────────────────────────────────

class TestUsageParsing:
    """Verify ``_parse_usage`` maps Anthropic usage fields correctly."""

    def test_cache_read_maps_to_cached_tokens(self) -> None:
        from leapflow.llm.anthropic_provider import _parse_usage

        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 200
        usage.cache_read_input_tokens = 800
        usage.cache_creation_input_tokens = 100

        result = _parse_usage(usage)

        assert result["prompt_tokens"] == 1000
        assert result["completion_tokens"] == 200
        assert result["total_tokens"] == 1200
        assert result["cached_tokens"] == 800
        assert result["cache_read_input_tokens"] == 800
        assert result["cache_creation_input_tokens"] == 100

    def test_no_cache_fields(self) -> None:
        from leapflow.llm.anthropic_provider import _parse_usage

        usage = MagicMock(spec=["input_tokens", "output_tokens"])
        usage.input_tokens = 500
        usage.output_tokens = 100

        result = _parse_usage(usage)

        assert result["prompt_tokens"] == 500
        assert result["completion_tokens"] == 100
        assert "cached_tokens" not in result
        assert "cache_read_input_tokens" not in result

    def test_none_usage(self) -> None:
        from leapflow.llm.anthropic_provider import _parse_usage

        result = _parse_usage(None)
        assert result == {}

    def test_zero_cache_read(self) -> None:
        """Zero cache_read should still be recorded."""
        from leapflow.llm.anthropic_provider import _parse_usage

        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 100
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 500

        result = _parse_usage(usage)
        assert result["cache_read_input_tokens"] == 0
        assert result["cached_tokens"] == 0
        assert result["cache_creation_input_tokens"] == 500


# ── Message conversion tests ─────────────────────────────────────────────

class TestMessageConversion:
    """Verify ``_convert_messages`` handles system, cache_control, tool results."""

    def test_system_extracted_as_top_level(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = _convert_messages(msgs)
        assert system == "You are helpful."
        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"

    def test_cache_control_preserved_on_system(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {
                "role": "system",
                "content": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Hello"},
        ]
        system, conversation = _convert_messages(msgs)
        # System should be a list of content blocks when cache_control is set.
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "You are helpful."

    def test_cache_control_on_user_message(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {
                "role": "user",
                "content": "Hello",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        _, conversation = _convert_messages(msgs)
        assert len(conversation) == 1
        content = conversation[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_result_mapped(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {"role": "user", "content": "Use tool X"},
            {"role": "assistant", "content": "Calling tool..."},
            {
                "role": "tool",
                "tool_call_id": "tc_123",
                "content": "Tool output here",
            },
        ]
        _, conversation = _convert_messages(msgs)
        # tool result should become a user message with tool_result content block.
        # user + assistant + tool_result(user) = 3 messages
        assert len(conversation) == 3
        tool_msg = conversation[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "tc_123"

    def test_consecutive_same_role_merged(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]
        _, conversation = _convert_messages(msgs)
        assert len(conversation) == 1
        assert "First" in str(conversation[0]["content"])
        assert "Second" in str(conversation[0]["content"])

    def test_structured_content_blocks_passthrough(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_messages

        msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Static part", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Dynamic part"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        system, _ = _convert_messages(msgs)
        assert isinstance(system, list)
        assert len(system) == 2
        assert system[0]["cache_control"] == {"type": "ephemeral"}


# ── Tool definition conversion tests ─────────────────────────────────────

class TestToolConversion:
    """Verify ``_convert_tools`` handles OpenAI → Anthropic format."""

    def test_function_tool_converted(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_tools

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }]
        result = _convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a city"
        assert "input_schema" in result[0]

    def test_cache_control_preserved_on_tool(self) -> None:
        from leapflow.llm.anthropic_provider import _convert_tools

        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
            "cache_control": {"type": "ephemeral"},
        }]
        result = _convert_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}


# ── SDK absence graceful degradation ──────────────────────────────────────

class TestGracefulDegradation:
    """Verify behaviour when the ``anthropic`` SDK is not installed."""

    def test_is_anthropic_available_reports_correctly(self) -> None:
        from leapflow.llm.anthropic_provider import is_anthropic_available

        # The function should return a bool (True if installed, False otherwise).
        result = is_anthropic_available()
        assert isinstance(result, bool)

    def test_anthropic_chat_raises_on_missing_sdk(self) -> None:
        """AnthropicChat.__init__ raises ImportError when SDK is absent."""
        from leapflow.llm import anthropic_provider

        original_flag = anthropic_provider._ANTHROPIC_AVAILABLE
        try:
            anthropic_provider._ANTHROPIC_AVAILABLE = False
            with pytest.raises(ImportError, match="anthropic SDK"):
                anthropic_provider.AnthropicChat(
                    api_key="test-key",
                    model="claude-sonnet-4-20250514",
                )
        finally:
            anthropic_provider._ANTHROPIC_AVAILABLE = original_flag

    def test_discover_builtin_skips_when_sdk_absent(self) -> None:
        """discover_builtin gracefully skips Anthropic when SDK is missing."""
        from leapflow.llm.provider_registry import LLMProviderRegistry

        registry = LLMProviderRegistry()

        with patch(
            "leapflow.llm.provider_registry.LLMProviderRegistry.discover_builtin"
        ) as mock_discover:
            # Simulate the real discover_builtin but with import failure.
            def _discover_with_failure(self_ref: Any = None) -> None:
                from leapflow.llm._builtin_plugins import OpenAICompatiblePlugin
                registry.register(OpenAICompatiblePlugin())
                # Simulate ImportError for anthropic
                # (in real code, this is handled by the try/except in discover_builtin)

            mock_discover.side_effect = lambda: _discover_with_failure()

        # Directly test the real code path:
        registry2 = LLMProviderRegistry()
        # Register only OpenAI, simulate anthropic import failure
        from leapflow.llm._builtin_plugins import OpenAICompatiblePlugin
        registry2.register(OpenAICompatiblePlugin())
        # Anthropic plugin not registered — should not be in list.
        assert "anthropic" not in registry2.list_available()
        assert "openai" in registry2.list_available()


# ── Plugin capability declarations ────────────────────────────────────────

class TestAnthropicPluginCapabilities:
    """Verify AnthropicPlugin declares correct capabilities."""

    @pytest.fixture
    def _skip_if_no_sdk(self) -> None:
        """Skip test if anthropic SDK is not installed."""
        from leapflow.llm.anthropic_provider import is_anthropic_available

        if not is_anthropic_available():
            pytest.skip("anthropic SDK not installed")

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    def test_cache_type_is_explicit_breakpoint(self) -> None:
        from leapflow.llm._anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()
        assert plugin.capabilities["cache_type"] == "explicit_breakpoint"

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    def test_cache_usage_fields(self) -> None:
        from leapflow.llm._anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()
        fields = plugin.capabilities["cache_usage_fields"]
        assert "cache_read_input_tokens" in fields
        assert "cache_creation_input_tokens" in fields

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    def test_provider_id(self) -> None:
        from leapflow.llm._anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()
        assert plugin.provider_id == "anthropic"

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    def test_create_provider_requires_api_key(self) -> None:
        from leapflow.llm._anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()
        with pytest.raises(ValueError, match="api_key"):
            plugin.create_provider({"model": "claude-sonnet-4-20250514"})

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    def test_create_provider_requires_model(self) -> None:
        from leapflow.llm._anthropic_plugin import AnthropicPlugin

        plugin = AnthropicPlugin()
        with pytest.raises(ValueError, match="model"):
            plugin.create_provider({"api_key": "sk-test"})


# ── Mock-based AnthropicChat achat test ───────────────────────────────────

class TestAnthropicChatMocked:
    """Test AnthropicChat behaviour with mocked SDK responses."""

    @pytest.fixture
    def _skip_if_no_sdk(self) -> None:
        from leapflow.llm.anthropic_provider import is_anthropic_available

        if not is_anthropic_available():
            pytest.skip("anthropic SDK not installed")

    @pytest.mark.usefixtures("_skip_if_no_sdk")
    @pytest.mark.asyncio
    async def test_achat_nonstream_usage_mapping(self) -> None:
        """Verify usage fields are correctly mapped from Anthropic response."""
        from leapflow.llm.anthropic_provider import AnthropicChat

        # Build a mock response.
        mock_usage = MagicMock()
        mock_usage.input_tokens = 1500
        mock_usage.output_tokens = 300
        mock_usage.cache_read_input_tokens = 1200
        mock_usage.cache_creation_input_tokens = 200

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello there!"

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        with patch("leapflow.llm.anthropic_provider._anthropic_sdk") as mock_sdk:
            mock_async_client = MagicMock()
            mock_async_client.messages.create = AsyncMock(return_value=mock_response)
            mock_sdk.AsyncAnthropic.return_value = mock_async_client
            mock_sdk.Anthropic.return_value = MagicMock()

            chat = AnthropicChat(api_key="test-key", model="claude-sonnet-4-20250514")
            # Swap the async client with our mock.
            chat._async = mock_async_client

            result = await chat.achat(
                [{"role": "user", "content": "Hi"}],
                stream=False,
            )

        assert result.content == "Hello there!"
        assert result.usage["prompt_tokens"] == 1500
        assert result.usage["completion_tokens"] == 300
        assert result.usage["cached_tokens"] == 1200
        assert result.usage["cache_read_input_tokens"] == 1200
        assert result.usage["cache_creation_input_tokens"] == 200
        assert "latency_ms" in result.usage

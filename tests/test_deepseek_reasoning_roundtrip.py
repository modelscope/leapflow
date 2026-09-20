# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for thinking-provider native tool-call continuation."""
from __future__ import annotations

from leapflow.engine.engine import _build_native_tool_assistant_message
from leapflow.llm.base import ToolCallInfo


def test_native_tool_message_preserves_deepseek_reasoning_content() -> None:
    """DeepSeek requires its thinking output on the assistant tool-call message."""
    message = _build_native_tool_assistant_message(
        [ToolCallInfo(id="call-1", name="plugin_list", arguments={})],
        thinking_content="I should inspect the live registry first.",
    )

    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert message["reasoning_content"] == "I should inspect the live registry first."
    assert message["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "plugin_list", "arguments": "{}"},
        }
    ]


def test_native_tool_message_omits_reasoning_for_standard_providers() -> None:
    """Providers that did not emit thinking data keep the standard message shape."""
    message = _build_native_tool_assistant_message(
        [ToolCallInfo(id="call-1", name="plugin_list", arguments={})],
    )

    assert "reasoning_content" not in message

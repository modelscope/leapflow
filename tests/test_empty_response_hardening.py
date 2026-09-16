# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for empty-LLM-response hardening (P0-A1).

Root cause (observed as "I processed your request but have no additional
output." on the first TUI turn right after daemon startup): an LLM call that
succeeded with empty content was converted into a fake-success filler message
instead of being treated as a failure signal.

Validates:
- an empty response gets exactly one bounded retry with an explicit nudge
- a recovered second response is returned as the final answer
- a second empty response yields a transparent degraded message, never the
  old fake-success filler
"""
from __future__ import annotations

import dataclasses
import tempfile
from typing import Any, List

import pytest

from conftest import make_settings
from leapflow.engine.engine import (
    _EMPTY_RESPONSE_DEGRADED_MESSAGE,
    _EMPTY_RESPONSE_RETRY_PROMPT,
    AgentEngine,
    build_default_registry,
)
from leapflow.engine.intent_classifier import Intent
from leapflow.llm.base import LLMChatResponse, LLMProvider
from leapflow.memory import (
    EpisodicMemoryProvider,
    SemanticMemoryProvider,
    WorkingMemoryProvider,
)
from leapflow.platform.mock import MockBridge


class _FixedClassifier:
    async def classify(self, user_text: str) -> Intent:
        return Intent(label="complex", reason="test")


class _ScriptedLLM(LLMProvider):
    """Returns scripted contents in order; records the messages it saw."""

    def __init__(self, replies: List[str]) -> None:
        self._replies = list(replies)
        self.call_count = 0
        self.seen_messages: List[List[dict[str, Any]]] = []

    async def achat(
        self,
        messages: List[dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> LLMChatResponse:
        self.seen_messages.append(list(messages))
        text = self._replies[self.call_count] if self.call_count < len(self._replies) else ""
        self.call_count += 1
        return LLMChatResponse(content=text)

    async def achat_stream(
        self,
        messages: List[dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ):
        if False:  # pragma: no cover
            yield ""


def _build_engine(td: str, llm: LLMProvider) -> tuple[AgentEngine, SemanticMemoryProvider]:
    settings = dataclasses.replace(make_settings(td), stream_output=False)
    rpc = MockBridge()
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()
    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier())
    return engine, lt


@pytest.mark.asyncio
async def test_empty_response_retried_once_then_recovers() -> None:
    with tempfile.TemporaryDirectory() as td:
        llm = _ScriptedLLM(["", "Potatoes have about 77 kcal per 100g."])
        engine, lt = _build_engine(td, llm)
        try:
            events = [event async for event in engine.run_stream("calories of potatoes")]
        finally:
            lt.close()

    finals = [event for event in events if event.type == "final"]
    assert finals, "expected a final event"
    assert "77 kcal" in finals[-1].content
    assert llm.call_count == 2
    # The retry carried the explicit nudge so the correction is model-visible.
    retry_texts = [
        str(msg.get("content"))
        for msg in llm.seen_messages[-1]
        if msg.get("role") == "user"
    ]
    assert any(_EMPTY_RESPONSE_RETRY_PROMPT in text for text in retry_texts)


@pytest.mark.asyncio
async def test_double_empty_response_yields_transparent_degraded_message() -> None:
    with tempfile.TemporaryDirectory() as td:
        llm = _ScriptedLLM(["", ""])
        engine, lt = _build_engine(td, llm)
        try:
            events = [event async for event in engine.run_stream("hello")]
        finally:
            lt.close()

    finals = [event for event in events if event.type == "final"]
    assert finals, "expected a final event"
    final_text = finals[-1].content
    assert final_text == _EMPTY_RESPONSE_DEGRADED_MESSAGE
    # The old fake-success filler must never resurface.
    assert "no additional output" not in final_text
    assert llm.call_count == 2  # exactly one bounded retry, no loop

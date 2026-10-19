# Copyright (c) Alibaba, Inc. and its affiliates.
"""Images a tool returns reach the model as one volatile message per batch.

The text result never carries the image; it goes to the ledger, trace and
transcript clean. Images stay out of the cacheable prefix, older ones retire
as newer ones arrive, and a model that cannot see images gets a description.
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from conftest import StubLLM, make_settings
from leapflow.engine._message_helpers import _latest_turn_tool_result
from leapflow.engine._tool_helpers import build_default_registry
from leapflow.engine.engine import AgentEngine
from leapflow.engine.prompt_cache import PrefixCacheOptimizer
from leapflow.engine.tool_dispatch_engine import ToolDispatchEngine
from leapflow.engine.tools.execution_trace import ExecutionTrace
from leapflow.engine.tools.tool_attachments import (
    ATTACHMENTS_KEY,
    ToolAttachment,
    attachment_message,
    present_attachments,
    retire_tool_images,
    take_attachments,
)
from leapflow.llm.base import ToolCallInfo
from leapflow.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from leapflow.memory import EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider
from leapflow.platform.mock import MockBridge
from _fixtures.agent_execution import _FixedClassifier

_JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD"


def _image(label: str = "frame f1") -> dict[str, str]:
    return {"mime": "image/jpeg", "data": _JPEG, "label": label}


def _attachment(call_id: str, label: str = "") -> ToolAttachment:
    return ToolAttachment(call_id, "hw_skill_run", "image/jpeg", _JPEG, label)


def _images(messages: list[dict[str, Any]]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    )


def test_attachments_are_popped_validated_and_counted() -> None:
    result = {
        "ok": True,
        ATTACHMENTS_KEY: [_image(), {"mime": "text/html", "data": "x"}, {"mime": "image/png"}],
    }
    (kept,) = take_attachments(result, tool_call_id="c1", tool_name="hw_skill_run")
    assert ATTACHMENTS_KEY not in result
    assert result["attached_images"] == 1 and result["attachments_refused"] == 2
    assert kept.label == "frame f1" and kept.reference == "hw_skill_run (call c1): frame f1"
    assert take_attachments({"ok": True}, tool_call_id="c2", tool_name="t") == ()


def test_only_the_newest_tool_images_stay_live() -> None:
    user_image = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}],
    }
    messages: list[dict[str, Any]] = [
        user_image,
        attachment_message([_attachment("c1"), _attachment("c2")]),
        attachment_message([_attachment("c3")]),
    ]
    before = messages[1]
    assert retire_tool_images(messages, keep=2) == 1
    assert _images(messages) == 3  # the user's own image is never retired
    assert before["content"][1]["type"] == "image_url"  # replaced, not mutated
    kept = [part["type"] for part in messages[1]["content"]]
    assert kept == ["text", "image_url", "text"]  # the older image of the pair went
    assert retire_tool_images(messages, keep=0) == 2
    assert _images(messages) == 1
    assert "call the tool again" in messages[2]["content"][-1]["text"]


async def test_a_model_without_vision_gets_a_description() -> None:
    vlm = SimpleNamespace(achat=AsyncMock(return_value=SimpleNamespace(content="A red cube.")))
    messages: list[dict[str, Any]] = []
    await present_attachments(messages, [_attachment("c1")], keep=2, sees_images=False, vlm=vlm)
    (message,) = messages
    assert isinstance(message["content"], str) and "A red cube." in message["content"]
    assert message["_volatile_context"] and message["_tool_attachment"]
    (sent,), _ = vlm.achat.call_args
    assert sent[0]["content"][1]["type"] == "image_url"
    await present_attachments(messages, [_attachment("c2")], keep=2, sees_images=False)
    assert "no description is available" in messages[-1]["content"]


async def test_a_batch_keeps_at_most_the_configured_images() -> None:
    messages: list[dict[str, Any]] = [attachment_message([_attachment("c0")])]
    batch = [_attachment("c1"), _attachment("c2"), _attachment("c3")]
    await present_attachments(messages, batch, keep=2, sees_images=True)
    assert _images(messages) == 2
    assert "1 older image(s)" in messages[-1]["content"][0]["text"]
    assert "c3" in messages[-1]["content"][0]["text"]


def test_images_never_enter_the_cacheable_prefix() -> None:
    system = {"role": "system", "content": "You are LeapFlow."}
    turn = [{"role": "user", "content": "look"}, {"role": "tool", "content": "{}"}]
    optimizer = PrefixCacheOptimizer()
    plain = optimizer.optimize([system, *turn])
    shown = optimizer.optimize([system, *turn, attachment_message([_attachment("c1")])])
    assert shown[0] == plain[0]  # the stable prefix is byte-identical
    assert shown[-1].get("_tool_attachment")  # the image rides in the dynamic tail


async def test_the_dispatcher_strips_the_image_and_shows_it_after_the_batch() -> None:
    def _turn_scan_ok(messages: list[dict[str, Any]]) -> None:
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 0
        assert _latest_turn_tool_result(messages) is not None

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        rpc, llm = MockBridge(), StubLLM([])
        wm, lt = (
            WorkingMemoryProvider(max_tokens=1024),
            SemanticMemoryProvider(source=settings.duckdb_path),
        )
        try:
            registry = build_default_registry(rpc, llm, wm, lt)
            engine = AgentEngine(
                settings,
                rpc,
                llm,
                wm,
                lt,
                EpisodicMemoryProvider(),
                registry,
                _FixedClassifier("complex"),
            )
            model = getattr(llm, "model", None) or settings.llm_model
            capabilities = ModelCapabilityRegistry()
            capabilities.register(model, ModelCapabilities(supports_vision=True))
            engine.set_model_capabilities(capabilities)

            async def snapshot(_tool_call: Any, _handlers: Any) -> dict[str, Any]:
                return {"ok": True, "frame": "f1", ATTACHMENTS_KEY: [_image()]}

            engine._tool_dispatch._execute_general_tool = AsyncMock(side_effect=snapshot)  # type: ignore[method-assign]
            engine._current_session_id = "session-1"
            engine._session_turn_count = 1
            engine._prompt_assembler._begin_turn_context("look at the table")
            messages: list[dict[str, Any]] = [{"role": "user", "content": "look"}]
            results = await engine._tool_dispatch._execute_tools_concurrent(
                [ToolCallInfo(id="tc1", name="time_get", arguments={})],
                {},
                trace=ExecutionTrace(),
                messages=messages,
            )
            tool_message = next(m for m in messages if m.get("role") == "tool")
            assert _JPEG not in tool_message["content"]
            assert '"attached_images": 1' in tool_message["content"]
            assert ATTACHMENTS_KEY not in results[0]["result"]
            assert messages[-1]["_tool_attachment"] and _images(messages) == 1
            assert "time_get (call tc1): frame f1" in messages[-1]["content"][0]["text"]
            _turn_scan_ok(messages)
            assert engine._tool_dispatch._attachments == {}
        finally:
            lt.close()


@pytest.mark.parametrize(
    ("model", "vision"),
    [
        ("qwen3-vl-plus", True),
        ("qwen2.5-vl-72b-instruct", True),
        ("qwen3-omni-flash", True),
        ("qwen3-max", False),
    ],
)
def test_qwen_vision_lines_are_known_to_see_images(model: str, vision: bool) -> None:
    assert ModelCapabilityRegistry().supports_vision(model) is vision

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


def _attachment(call_id: str, label: str = "", *, data: str = _JPEG) -> ToolAttachment:
    return ToolAttachment(call_id, "hw_skill_run", "image/jpeg", data, label)


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
    assert "Description: none available" in messages[-1]["content"]


async def test_a_batch_keeps_at_most_the_configured_images() -> None:
    messages: list[dict[str, Any]] = [attachment_message([_attachment("c0")])]
    batch = [_attachment(f"c{i}", data=f"{_JPEG}{i}") for i in (1, 2, 3)]
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


@pytest.mark.parametrize("terminal", ["returned", "failed", "cancelled", "timed_out"])
async def test_attachment_inference_retains_owned_receipts(tmp_path, terminal) -> None:
    import asyncio
    import json
    from leapflow.llm.base import LLMChatResponse
    from leapflow.storage.conversation_store import DuckDBConversationStore
    from test_tool_dispatch_engine import _collaboration_engine

    class Provider:
        model = "description-model"

        async def achat(self, messages, **options):
            assert "inference_tool_call_id" not in options
            assert messages[0]["content"][1]["type"] == "image_url"
            if terminal == "failed":
                raise ValueError("description rejected")
            if terminal == "cancelled":
                raise asyncio.CancelledError()
            if terminal == "timed_out":
                raise asyncio.TimeoutError()
            return LLMChatResponse(content="A red cube.", usage={})

    engine, memory = await _collaboration_engine(tmp_path)
    store = DuckDBConversationStore(tmp_path / "attachments.duckdb")
    store.create_session("own")
    engine.set_conversation_store(store)
    engine._vlm = Provider()
    dispatch = engine._tool_dispatch
    dispatch._attachments["camera-call"] = (_attachment("camera-call"),)
    messages = []
    try:
        if terminal == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await dispatch.append_tool_attachments(messages, ["camera-call"])
        else:
            await dispatch.append_tool_attachments(messages, ["camera-call"])
            assert messages[-1]["_volatile_context"]
            assert ("A red cube." in messages[-1]["content"]) == (terminal == "returned")
        (attempt,) = engine._usage_tracker.inference_records()
        assert attempt.scope.session_id == "own" and attempt.tool_call_id == "camera-call"
        assert attempt.model == "description-model"
        expected = "failed" if terminal == "timed_out" else terminal
        assert attempt.status == expected and attempt.usage_json == "{}"
        combined = engine._usage_tracker.combined_usage()["combined"]
        assert combined["attempts"] == 1 and combined["unknown_usage_attempts"] == 1
        (row,) = store.get_messages("own")
        assert row.role == "record" and row.content == ""
        (receipt,) = row.metadata["collaboration_signal_receipts"]
        assert receipt["lineage_eligible"] and receipt["scope"]["session_id"] == "own"
        value = json.loads(receipt["payload_json"])["value"]
        assert value["tool_call_id"] == "camera-call" and not value["usage_known"]
        assert _JPEG not in receipt["payload_json"]
    finally:
        store.close()
        memory.close()


async def test_attachment_cache_is_owned_and_provider_accounting_is_deduplicated(tmp_path, monkeypatch) -> None:
    import asyncio
    from openai.resources.chat import AsyncCompletions
    from openai.types.chat import ChatCompletion
    from leapflow.llm.openai_provider import OpenAIChat
    from leapflow.engine.session.session_factory import build_session_engine
    from test_llm_provider_retry_ownership import _provider, _recorded_body
    from test_tool_dispatch_engine import _collaboration_engine

    first, memory1 = await _collaboration_engine(tmp_path / "first")
    second = build_session_engine(first, session_id="second", workspace_root=tmp_path / "second",
                                  working_memory=WorkingMemoryProvider(max_tokens=1024))
    provider = _provider(max_retries=1)
    alternate = OpenAIChat(api_key="test", model="another-model", base_url="https://example.invalid/v1",
                           max_retries=1)
    async def complete(*args, **kwargs):
        await asyncio.sleep(0)
        return ChatCompletion.model_validate(_recorded_body())

    create = AsyncMock(side_effect=complete)
    monkeypatch.setattr(AsyncCompletions, "create", create)
    first._vlm = second._vlm = provider

    async def show(engine, call_id="image-call"):
        engine._tool_dispatch._attachments[call_id] = (_attachment(call_id),)
        messages = []
        await engine._tool_dispatch.append_tool_attachments(messages, [call_id])
        return messages[-1]["content"]

    try:
        assert first._tool_dispatch._attachments is not second._tool_dispatch._attachments
        assert first._tool_dispatch._description_cache is not second._tool_dispatch._description_cache
        a, b = await asyncio.gather(show(first), show(second))
        assert a == b
        assert create.await_count == 2  # Same bytes in another workspace require a fresh call.
        await show(first, "repeat")
        assert create.await_count == 2
        first._current_session_id = "fresh-run"
        await show(first)
        assert create.await_count == 3
        first._vlm = alternate
        await show(first)
        assert create.await_count == 4
        first_attempts = first._usage_tracker.inference_records()
        second_attempts = second._usage_tracker.inference_records()
        assert len(first_attempts) == 3 and len(second_attempts) == 1
        assert all(item.accounting_level == "provider_attempt" for item in first_attempts)
        assert first_attempts[0].scope.workspace_id != second_attempts[0].scope.workspace_id
        assert first_attempts[1].scope.session_id == "fresh-run"
        usage = first._usage_tracker.combined_usage()["combined"]
        assert usage["attempts"] == 3 and usage["unknown_usage_attempts"] == 0
        first._usage_tracker.record_api_call({}, call_id=first_attempts[0].call_id)
        assert first._usage_tracker.combined_usage()["combined"]["attempts"] == 3
    finally:
        await provider.aclose()
        await alternate.aclose()
        memory1.close()


async def test_attachment_vision_probe_is_accounted_only_when_called(tmp_path) -> None:
    from leapflow.llm.base import LLMChatResponse
    from leapflow.llm.vision_capability import VisionCapability, VisionVerdictStore
    from test_tool_dispatch_engine import _collaboration_engine

    engine, memory = await _collaboration_engine(tmp_path)
    resolver = VisionCapability(store=VisionVerdictStore(None))
    engine.set_vision_capability(resolver)

    class Provider:
        model = "probe-model"

        async def achat(self, messages, **options):
            return LLMChatResponse(content="left=unknown right=unknown", usage={})

    engine._llm = Provider()
    try:
        for call_id in ("first-frame", "second-frame"):
            engine._tool_dispatch._attachments[call_id] = (_attachment(call_id),)
            await engine._tool_dispatch.append_tool_attachments([], [call_id])
        (attempt,) = engine._usage_tracker.inference_records()
        assert attempt.scope.session_id == "own" and attempt.model == "probe-model"
        assert attempt.status == "returned" and attempt.tool_call_id == ""
        assert resolver.remembered(engine._llm, "probe-model").vision is False
    finally:
        memory.close()


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

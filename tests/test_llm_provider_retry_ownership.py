# Copyright (c) Alibaba, Inc. and its affiliates.
"""The provider owns retry; the SDK must not add a second, hidden one.

``OpenAIChat`` implements a retry policy with backoff, and the recovery layer's
budgets assume ``timeout_s`` bounds *one* attempt. The OpenAI SDK retries twice by
default, so leaving that default in place multiplies the two policies: effective
attempts become ``max_retries * 3`` and a hard timeout surfaces after three times
the configured budget. Measured on a live endpoint before the fix: a 45s timeout
raised ``APITimeoutError`` after 137s (~3 x 45s), which is exactly the failure a
turn-level deadline cannot absorb.

These tests pin the single-owner contract without any network access.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from openai.resources.chat import AsyncCompletions, Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from leapflow.domain.events import RecordScope
from leapflow.engine.turn_usage import TurnUsageTracker
from leapflow.llm.openai_provider import OpenAIChat
from leapflow.llm.services.telemetry import capture_inference, current_capture


def _provider(**kwargs) -> OpenAIChat:
    return OpenAIChat(
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        **kwargs,
    )


def test_sdk_clients_do_not_retry_on_their_own():
    """Both SDK clients must be constructed with retries disabled."""
    provider = _provider()
    # ``max_retries`` is public on the SDK client and is what multiplies attempts.
    assert provider._async.max_retries == 0
    assert provider._sync.max_retries == 0


def test_provider_keeps_its_own_retry_policy():
    """Disabling SDK retry must not disable ours -- the policy just has one owner."""
    provider = _provider(max_retries=4)
    assert provider._max_retries == 4
    # Floor of one attempt: a zero would mean "never call the model".
    assert _provider(max_retries=0)._max_retries == 1


def test_configured_timeout_bounds_one_attempt():
    """The read timeout the caller asked for is the one the client carries."""
    provider = _provider(timeout_s=45.0)
    assert provider._async.timeout.read == 45.0
    # Connect/write/pool stay bounded independently so a stalled handshake cannot
    # consume the whole read budget.
    assert provider._async.timeout.connect == 30.0


def _recorded_body():
    fixture = Path(__file__).parent / "_fixtures/recordings/r1_conversation/cassette-model-76802162bb96917f.cassette.json"
    return json.loads(json.loads(fixture.read_text())["responses"][0]["body"])


def _recorded_chunks():
    """Derive text and usage-only stream frames from committed provider traffic."""
    body = _recorded_body()
    common = {key: body[key] for key in ("id", "model", "created")}
    common["object"] = body["object"] + ".chunk"
    choice = body["choices"][0]
    return [ChatCompletionChunk.model_validate({**common, "choices": [{
        "index": choice["index"], "delta": choice["message"], "finish_reason": None}]}),
        ChatCompletionChunk.model_validate({**common, "choices": [], "usage": body["usage"]})]


@pytest.mark.asyncio
async def test_failed_then_successful_attempts_keep_one_call_and_do_not_double_count(monkeypatch):
    provider = _provider(max_retries=2)
    response = ChatCompletion.model_validate(_recorded_body())
    create = AsyncMock(side_effect=[openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")), response])
    monkeypatch.setattr(AsyncCompletions, "create", create)
    monkeypatch.setattr(provider, "_sleep_backoff", AsyncMock())
    tracker = TurnUsageTracker()
    scope = RecordScope("p", "w", "s")
    try:
        with capture_inference(scope=scope, task_id="task", tool_call_id="tool", sink=tracker) as capture:
            result = await provider.achat([{"role": "user", "content": "hello"}], stream=False)
        assert create.await_count == 2
        assert [record.status for record in capture.attempts] == ["failed", "returned"]
        assert len({record.call_id for record in capture.attempts}) == 1
        assert len({record.attempt_id for record in capture.attempts}) == 2
        assert all(record.scope == scope and record.task_id == "task" and record.tool_call_id == "tool"
                   for record in capture.attempts)
        tracker.record_api_call(result.usage, call_id=result.inference_call_id)
        assert tracker.summary().api_calls == 2
        combined = tracker.combined_usage()["combined"]
        assert combined["attempts"] == 2 and combined["calls"] == 1
        assert combined["unknown_usage_attempts"] == 1 and combined["attempt_count_complete"]
        assert combined["known_prompt_tokens"] == response.usage.prompt_tokens
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_cancellation_records_attempt_without_retry(monkeypatch):
    provider = _provider(max_retries=3)
    create = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(AsyncCompletions, "create", create)
    try:
        with capture_inference(scope=RecordScope("p", "w", "s")) as capture:
            with pytest.raises(asyncio.CancelledError):
                await provider.achat([], stream=False)
        assert create.await_count == 1
        assert capture.attempts[0].status == "cancelled"
        assert capture.attempts[0].to_dict()["usage_known"] is False
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_payload_failure_keeps_usage_before_extraction(monkeypatch):
    provider = _provider()
    body = _recorded_body()
    body["choices"] = []  # A malformed payload does not erase returned usage.
    monkeypatch.setattr(AsyncCompletions, "create",
                        AsyncMock(return_value=ChatCompletion.model_validate(body)))
    try:
        with capture_inference() as capture:
            with pytest.raises(IndexError):
                await provider.achat([], stream=False)
        assert capture.attempts[0].status == "failed"
        assert capture.attempts[0].to_dict()["usage"]["prompt_tokens"] == body["usage"]["prompt_tokens"]
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_async_collapsed_usage_only_chunk_is_accounted(monkeypatch):
    provider = _provider()
    chunks = _recorded_chunks()

    async def stream():
        for chunk in chunks:
            yield chunk

    monkeypatch.setattr(AsyncCompletions, "create", AsyncMock(return_value=stream()))
    try:
        with capture_inference() as capture:
            response = await provider.achat([], stream=True)
        assert response.content == _recorded_body()["choices"][0]["message"]["content"]
        assert capture.attempts[0].to_dict()["usage"]["prompt_tokens"] == chunks[-1].usage.prompt_tokens
        assert response.inference_call_id == capture.attempts[0].call_id
    finally:
        await provider.aclose()


def test_sync_collapsed_usage_only_chunk_is_accounted(monkeypatch):
    provider = _provider()
    chunks = _recorded_chunks()
    monkeypatch.setattr(Completions, "create", Mock(return_value=iter(chunks)))
    try:
        with capture_inference() as capture:
            response = provider.chat([], stream=True)
        assert response.usage["prompt_tokens"] == chunks[-1].usage.prompt_tokens
        assert capture.attempts[0].to_dict()["usage_known"]
    finally:
        asyncio.run(provider.aclose())


@pytest.mark.asyncio
async def test_raw_stream_close_keeps_partial_usage_and_caller_scope(monkeypatch):
    provider = _provider()
    chunks = _recorded_chunks()
    chunks[0].usage = chunks[-1].usage

    async def stream():
        yield chunks[0]
        yield chunks[-1]

    monkeypatch.setattr(AsyncCompletions, "create", AsyncMock(return_value=stream()))
    scope = RecordScope("p", "w", "s")
    try:
        with capture_inference(scope=scope) as capture:
            response = provider.achat_stream([])
            await response.__anext__()
        assert current_capture() is None
        await response.aclose()  # Closed by a consumer outside the original context.
        assert capture.attempts[0].status == "cancelled"
        assert capture.attempts[0].scope == scope
        assert capture.attempts[0].to_dict()["usage_known"]
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_broken_sink_never_retries_a_successful_response(monkeypatch):
    provider = _provider(max_retries=3)
    create = AsyncMock(return_value=ChatCompletion.model_validate(_recorded_body()))
    monkeypatch.setattr(AsyncCompletions, "create", create)

    class BrokenSink:
        def record_inference(self, attempt):
            raise RuntimeError("context telemetry defect")

    try:
        with capture_inference(sink=BrokenSink()) as capture:
            response = await provider.achat([], stream=False)
        assert response.content and create.await_count == 1 and len(capture.attempts) == 1
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_specialist_runtime_can_suppress_provider_primary_accounting(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(AsyncCompletions, "create",
                        AsyncMock(return_value=ChatCompletion.model_validate(_recorded_body())))
    try:
        with capture_inference(provider_accounting=False) as capture:
            await provider.achat([], stream=False)
        assert capture.attempts == []
    finally:
        await provider.aclose()

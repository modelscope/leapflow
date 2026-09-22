# Copyright (c) Alibaba, Inc. and its affiliates.
"""The five live end-to-end tests — one real provider behaviour each.

Each test is the smallest exercise that only a real provider can prove, and each
carries a hard call / token / deadline budget through the ``live_budget``
fixture (see :mod:`tests.live.conftest`). Prompts are engineered to be
deterministic and short so the whole lane fits inside a ~74k-token suite budget.

The ``@pytest.mark.live`` / ``@pytest.mark.e2e`` markers are also applied by path
in the root conftest; they are declared here as well so the intent is legible at
the test and so ``-m live`` selection is correct even in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from leapflow.llm.base import LLMChatResponse, LLMProvider

pytestmark = [pytest.mark.live, pytest.mark.e2e]


def _extract_answer(resp: LLMChatResponse) -> str:
    """Concatenate content and any thinking so an answer isn't missed by field."""
    parts = [resp.content or ""]
    if resp.thinking_content:
        parts.append(resp.thinking_content)
    return " ".join(p for p in parts if p)


# ── 1. Single-turn answer ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_single_turn_answer(live_provider, live_budget) -> None:
    """One turn, one deterministic answer: the provider round-trips at all."""
    budget = live_budget(max_calls=2, max_tokens=15_000, deadline_s=30.0)
    provider = budget.wrap(live_provider)

    resp = await provider.achat(
        [
            {
                "role": "system",
                "content": "You are a calculator. Reply with only the number.",
            },
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        stream=False,
    )

    answer = _extract_answer(resp)
    assert "4" in answer, f"expected '4' in the reply, got {answer!r}"


# ── 2. Tool-call round-trip ──────────────────────────────────────────────────


_ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "first addend"},
                "b": {"type": "integer", "description": "second addend"},
            },
            "required": ["a", "b"],
        },
    },
}


@pytest.mark.asyncio
async def test_live_tool_call_roundtrip(live_provider, live_budget) -> None:
    """The model emits a tool call, we execute it, and it uses the result.

    A deterministic ``add`` tool keeps the assertion exact: the final answer must
    contain 579 (123 + 456), a value the model is unlikely to produce without the
    tool. First turn should call the tool; second turn consumes the result.
    """
    budget = live_budget(max_calls=3, max_tokens=18_000, deadline_s=45.0)
    provider = budget.wrap(live_provider)

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You must use the provided add tool to compute sums. "
                "Do not compute the sum yourself."
            ),
        },
        {"role": "user", "content": "Use the add tool to compute 123 + 456."},
    ]

    first = await provider.achat(
        messages,
        stream=False,
        tools=[_ADD_TOOL],
        tool_choice="auto",
    )
    assert first.tool_calls, "expected the model to request the add tool"

    call = first.tool_calls[0]
    assert call.name == "add", f"expected an 'add' call, got {call.name!r}"
    result = int(call.arguments.get("a", 0)) + int(call.arguments.get("b", 0))
    assert result == 579, f"tool arguments did not sum to 579: {call.arguments!r}"

    # Feed the tool result back and let the model phrase the final answer.
    messages.append(
        {
            "role": "assistant",
            "content": first.content or "",
            "tool_calls": [
                {
                    "id": call.id or "call_add",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _dumps(call.arguments),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id or "call_add",
            "content": str(result),
        }
    )

    final = await provider.achat(messages, stream=False, tools=[_ADD_TOOL])
    answer = _extract_answer(final)
    assert "579" in answer, f"final answer missing the tool result 579: {answer!r}"


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)


# ── 3. Streaming integrity ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_live_streaming_integrity(live_provider, live_budget) -> None:
    """Streamed chunks are non-empty and concatenate to the whole answer.

    Uses ``achat(stream=True, on_chunk=...)``: the real SSE path streams deltas
    to the callback while still returning a collapsed response with usage, so the
    budget stays honest without a second call.
    """
    budget = live_budget(max_calls=1, max_tokens=15_000, deadline_s=30.0)
    provider = budget.wrap(live_provider)

    chunks: List[str] = []

    def _collect(delta: str) -> None:
        if delta:
            chunks.append(delta)

    resp = await provider.achat(
        [
            {
                "role": "system",
                "content": "Reply with exactly the single word: pong",
            },
            {"role": "user", "content": "ping"},
        ],
        stream=True,
        on_chunk=_collect,
    )

    assert chunks, "streaming produced no chunks"
    assert all(chunks), "streaming produced an empty chunk"
    streamed = "".join(chunks)
    # The collapsed content is built from the same deltas, so they must agree.
    assert streamed == resp.content, (
        "streamed chunks did not reassemble the collapsed content:\n"
        f"  streamed={streamed!r}\n  collapsed={resp.content!r}"
    )
    assert "pong" in streamed.lower(), f"expected 'pong' in the stream, got {streamed!r}"


# ── 4. Graceful handling of a longer context ─────────────────────────────────


@pytest.mark.asyncio
async def test_live_context_overflow_graceful(live_provider, live_budget) -> None:
    """A moderately long context still yields a coherent, on-topic answer.

    The context is padded with filler far below any model limit — enough to make
    the request non-trivial without a real overflow — and hides one fact the
    model must recover. A coherent answer proves the long prompt round-trips
    without truncating the salient content.
    """
    budget = live_budget(max_calls=3, max_tokens=20_000, deadline_s=60.0)
    provider = budget.wrap(live_provider)

    filler = "This is filler context line number {n} with no salient content."
    padding = "\n".join(filler.format(n=i) for i in range(400))
    secret = "The passphrase for section 7 is ORANGE-HORIZON."

    resp = await provider.achat(
        [
            {
                "role": "system",
                "content": "Answer questions using only the provided document.",
            },
            {
                "role": "user",
                "content": (
                    f"{padding}\n\n{secret}\n\n{padding}\n\n"
                    "Question: What is the passphrase for section 7? "
                    "Reply with only the passphrase."
                ),
            },
        ],
        stream=False,
    )

    answer = _extract_answer(resp)
    assert answer.strip(), "expected a non-empty answer for the long-context prompt"
    assert "ORANGE-HORIZON" in answer.upper(), (
        f"model lost the salient fact in the long context: {answer!r}"
    )


# ── 5. Recovery on a transient error ─────────────────────────────────────────


class _TransientOnceProvider(LLMProvider):
    """Wraps a real provider, failing the first ``achat`` with a transient error.

    Models the common real fault — one flaky request, then a healthy endpoint —
    so the recovery path (retry, then delegate) is exercised against a live
    backend rather than a mock. Only the first call fails; every later call
    delegates unchanged.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self._failed = False

    async def achat(self, messages, **kwargs: Any) -> LLMChatResponse:  # type: ignore[override]
        if not self._failed:
            self._failed = True
            import openai

            raise openai.APITimeoutError(request=None)  # type: ignore[arg-type]
        return await self._inner.achat(messages, **kwargs)

    async def achat_stream(self, messages, **kwargs: Any):  # type: ignore[override]
        async for chunk in self._inner.achat_stream(messages, **kwargs):
            yield chunk


async def _achat_with_recovery(
    provider: LLMProvider,
    messages: List[Dict[str, Any]],
    *,
    max_attempts: int,
    on_usage: Any,
    **kwargs: Any,
) -> LLMChatResponse:
    """Minimal recovery loop: retry a transient failure, then surface success.

    Mirrors the engine's retry-then-continue contract in miniature. A failed
    attempt records no usage (the provider never answered); a successful attempt
    records its own. The loop stops at the first success or exhausts attempts.
    """
    import openai

    last_exc: Optional[BaseException] = None
    for _attempt in range(max_attempts):
        try:
            resp = await provider.achat(messages, **kwargs)
        except openai.APITimeoutError as exc:
            last_exc = exc
            continue
        on_usage(getattr(resp, "usage", None))
        return resp
    assert last_exc is not None
    raise last_exc


@pytest.mark.asyncio
async def test_live_recovery_on_transient_error(live_provider, live_budget) -> None:
    """A transient first failure is recovered and a real answer is returned."""
    budget = live_budget(max_calls=2, max_tokens=15_000, deadline_s=45.0)
    provider = _TransientOnceProvider(live_provider)

    resp = await _achat_with_recovery(
        provider,
        [
            {
                "role": "system",
                "content": "You are a calculator. Reply with only the number.",
            },
            {"role": "user", "content": "What is 3 + 3?"},
        ],
        max_attempts=2,
        on_usage=budget.record_usage,
        stream=False,
    )

    answer = _extract_answer(resp)
    assert "6" in answer, f"expected '6' after recovery, got {answer!r}"
    assert budget.calls == 1, (
        "exactly one successful call should be recorded after recovery "
        f"(failed attempt records no usage), saw {budget.calls}"
    )

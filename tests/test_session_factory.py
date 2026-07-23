"""Tests for the Stage 3 per-session engine factory (P3-1).

Proves the shallow-copy factory isolates the concurrency-corrupting substrate
(working memory + per-turn state + idempotency ledger) while sharing stateless
services, and that two per-session engines run concurrent turns without the
cross-contamination the single shared engine exhibits (Stage 1c).
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from conftest import make_settings


def _build_base_engine(td: str, llm):
    from leapflow.engine.engine import AgentEngine, build_default_registry
    from leapflow.memory import (
        EpisodicMemoryProvider,
        SemanticMemoryProvider,
        WorkingMemoryProvider,
    )
    from leapflow.platform.mock import MockBridge

    settings = make_settings(td)
    rpc = MockBridge()
    wm = WorkingMemoryProvider(max_tokens=1024)
    lt = SemanticMemoryProvider(source=settings.duckdb_path)
    imm = EpisodicMemoryProvider()

    class _Simple:
        def classify(self, *a, **k):
            return "simple"

        async def aclassify(self, *a, **k):
            return "simple"

    reg = build_default_registry(rpc, llm, wm, lt)
    engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, _Simple())
    return engine, lt


class _EchoLLM:
    """Returns the last user message as plain-text content (input-derived).

    Plain text with no tool call is treated as the final answer by the unified
    loop, so the turn terminates in one round and the output reflects only this
    turn's own input — making cross-contamination directly observable.
    """

    async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
        from leapflow.llm.base import LLMChatResponse
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = str(m.get("content") or "")
                break
        return LLMChatResponse(content=user)

    async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
        if False:
            yield ""


def test_build_session_engine_isolates_substrate() -> None:
    from leapflow.engine.session_factory import build_session_engine
    from leapflow.memory import WorkingMemoryProvider

    with tempfile.TemporaryDirectory() as td:
        base, lt = _build_base_engine(td, _EchoLLM())
        try:
            wm_a = WorkingMemoryProvider(max_tokens=512)
            wm_b = WorkingMemoryProvider(max_tokens=512)
            a = build_session_engine(base, session_id="sess-a", working_memory=wm_a)
            b = build_session_engine(base, session_id="sess-b", working_memory=wm_b)

            # Fresh, distinct per-session substrate.
            assert a._wm is wm_a and b._wm is wm_b and a._wm is not b._wm
            assert a._tool_execution_ledger is not b._tool_execution_ledger
            assert a._current_session_id == "sess-a" and b._current_session_id == "sess-b"
            assert a._active_frame is None and b._active_frame is None
            assert a._cancel_requested is False and b._cancel_requested is False

            # Stateless services shared by reference (not duplicated).
            assert a._llm is b._llm is base._llm

            # Working-memory isolation: a write to A is invisible to B.
            a._wm.remember_chat({"role": "user", "content": "only-in-A"})
            assert len(list(b._wm.as_chat_messages())) == 0
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_concurrent_session_engines_are_isolated() -> None:
    from leapflow.engine.session_factory import build_session_engine
    from leapflow.memory import WorkingMemoryProvider

    with tempfile.TemporaryDirectory() as td:
        base, lt = _build_base_engine(td, _EchoLLM())
        try:
            a = build_session_engine(base, session_id="A", working_memory=WorkingMemoryProvider(max_tokens=512))
            b = build_session_engine(base, session_id="B", working_memory=WorkingMemoryProvider(max_tokens=512))
            out_a, out_b = await asyncio.gather(
                a._unified_tool_loop("MARKER-AAA-111"),
                b._unified_tool_loop("MARKER-BBB-222"),
            )
            # Per-session engines must not cross-contaminate (contrast Stage 1c).
            assert "111" in out_a and "222" not in out_a
            assert "222" in out_b and "111" not in out_b
        finally:
            lt.close()

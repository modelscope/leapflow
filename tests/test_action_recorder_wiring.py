# Copyright (c) Alibaba, Inc. and its affiliates.
"""Execution-boundary tests for the no-LLM ActionExecutor wiring."""
from __future__ import annotations

import tempfile

import pytest

from conftest import StubLLM, make_settings
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.engine.tools.action_executor import ActionInvocation, RecordedActionExecutor
from leapflow.engine.engine import AgentEngine
from leapflow.engine import build_default_registry
from leapflow.engine.intent_classifier import Intent
from leapflow.evolution.action_recorder import ActionEvidenceUnavailable
from leapflow.memory import EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider
from leapflow.platform.mock import MockBridge


class _Recorder:
    def __init__(self, *, fail_start: bool = False, fail_complete: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_complete = fail_complete
        self.started_calls: list[dict] = []
        self.completed_calls: list[dict] = []
        self.failed_calls: list[dict] = []

    async def started(self, **kwargs):
        self.started_calls.append(kwargs)
        if self.fail_start:
            raise OSError("writer unavailable")
        return EvolutionEvent.create(
            "action.started",
            context=kwargs["context"],
            payload={},
            producer="test",
        )

    async def completed(self, **kwargs):
        self.completed_calls.append(kwargs)
        if self.fail_complete:
            raise OSError("writer unavailable")
        return EvolutionEvent.create(
            "action.completed",
            context=kwargs["context"],
            payload={},
            producer="test",
        )

    async def failed_exception(self, **kwargs):
        self.failed_calls.append(kwargs)
        return EvolutionEvent.create(
            "action.failed",
            context=kwargs["context"],
            payload={},
            producer="test",
        )


def _invocation(policy: str = "read_only") -> ActionInvocation:
    return ActionInvocation(
        action_type="tool",
        action_name="file_read",
        arguments={"path": "x"},
        execution_id="exec-1",
        execution_policy=policy,  # type: ignore[arg-type]
        context=EvolutionContext.create(
            profile_id="profile-a",
            workspace_id="workspace-a",
            session_id="session-a",
            turn_id="turn-a",
            frame_id="command-a",
            action_id="exec-1",
        ),
        goal="read x",
    )


class _FixedClassifier:
    def __init__(self) -> None:
        self._intent = Intent(label="complex", reason="test")

    async def classify(self, user_text: str) -> Intent:
        return self._intent


async def _return(value):
    return value


@pytest.mark.asyncio
async def test_executor_records_identity_and_completion() -> None:
    recorder = _Recorder()
    executor = RecordedActionExecutor(recorder)

    result = await executor.execute(
        _invocation(),
        lambda: _return({"ok": True, "value": 1}),
    )

    assert result["ok"] is True
    assert len(recorder.started_calls) == 1
    assert len(recorder.completed_calls) == 1
    context = recorder.started_calls[0]["context"]
    assert context.profile_id == "profile-a"
    assert context.session_id == "session-a"
    assert context.action_id == "exec-1"
    assert recorder.started_calls[0]["critical"] is False


@pytest.mark.asyncio
async def test_mutating_action_fails_closed_when_start_fact_cannot_persist() -> None:
    recorder = _Recorder(fail_start=True)
    executor = RecordedActionExecutor(recorder)
    executed = False

    async def execute():
        nonlocal executed
        executed = True
        return {"ok": True}

    with pytest.raises(ActionEvidenceUnavailable, match="audit start"):
        await executor.execute(_invocation("mutating_idempotent"), execute)
    assert executed is False


@pytest.mark.asyncio
async def test_read_only_action_degrades_when_recorder_is_unavailable() -> None:
    executor = RecordedActionExecutor(_Recorder(fail_start=True))
    result = await executor.execute(_invocation(), lambda: _return({"ok": True}))
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_execution_exception_is_recorded_then_reraised() -> None:
    recorder = _Recorder()
    executor = RecordedActionExecutor(recorder)

    async def execute():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await executor.execute(_invocation("external_side_effect"), execute)
    assert len(recorder.failed_calls) == 1
    assert recorder.failed_calls[0]["critical"] is True


@pytest.mark.asyncio
async def test_mutating_completion_record_failure_marks_effect_uncertain() -> None:
    executor = RecordedActionExecutor(_Recorder(fail_complete=True))
    result = await executor.execute(
        _invocation("mutating_once"),
        lambda: _return({"ok": True}),
    )
    assert result["audit_incomplete"] is True
    assert result["side_effect_uncertain"] is True


@pytest.mark.asyncio
async def test_no_recorder_preserves_execution_result() -> None:
    executor = RecordedActionExecutor(None)
    result = await executor.execute(_invocation(), lambda: _return({"ok": True, "value": 2}))
    assert result == {"ok": True, "value": 2}


@pytest.mark.asyncio
async def test_real_agent_engine_routes_tool_through_action_executor() -> None:
    with tempfile.TemporaryDirectory() as directory:
        settings = make_settings(directory)
        rpc = MockBridge()
        llm = StubLLM([])
        working = WorkingMemoryProvider(max_tokens=1024)
        semantic = SemanticMemoryProvider(source=settings.duckdb_path)
        episodic = EpisodicMemoryProvider()
        recorder = _Recorder()
        calls: list[dict] = []

        async def file_list_handler(args):
            calls.append(dict(args))
            return {"ok": True, "entries": []}

        try:
            registry = build_default_registry(rpc, llm, working, semantic)
            engine = AgentEngine(
                settings,
                rpc,
                llm,
                working,
                semantic,
                episodic,
                registry,
                _FixedClassifier(),
                action_executor=RecordedActionExecutor(recorder),
            )
            engine._current_session_id = "session-a"
            engine._session_turn_count = 1
            engine._prompt_assembler._begin_turn_context("list files")

            result = await engine._tool_dispatch._execute_tool_with_ledger(
                {"name": "file_list", "arguments": {"path": "."}},
                {"file_list": file_list_handler},
                tool_call_id="tool-call-a",
            )

            assert result["ok"] is True
            assert calls == [{"path": "."}]
            assert recorder.started_calls[0]["context"].session_id == "session-a"
            assert recorder.started_calls[0]["execution_policy"] == "read_only"
        finally:
            semantic.close()

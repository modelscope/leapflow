# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for SubagentManager EventBus integration (Phase 4A P1-5).

Includes cancel_all() verification and DefaultSubagentExecutor approval gate.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pytest

from leapflow.engine.subagent import (
    DefaultSubagentExecutor,
    SubagentCompleted,
    SubagentConfig,
    SubagentFailed,
    SubagentManager,
    SubagentResult,
    SubagentStarted,
    _SAFE_RISK_LEVELS,
)


# ── Helpers ──


class RecordingEventBus:
    """Minimal EventBus stand-in that records (event_type, payload) pairs."""

    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append((event_type, payload))


class FailingEventBus:
    """EventBus that always raises — verifies emission failures are contained."""

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        raise RuntimeError("bus on fire")


class FakeExecutor:
    """Executor that returns a canned result or raises on demand."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        if self._fail:
            raise ValueError("executor boom")
        return SubagentResult(
            session_id="sub_fakeexec123",
            goal=config.goal,
            summary="done",
            status="completed",
            elapsed_s=0.01,
            tool_calls=2,
        )


# ── Event dataclass sanity ──


class TestEventDataclasses:
    def test_subagent_started_frozen(self) -> None:
        e = SubagentStarted(
            parent_session_id="sess1", subagent_id="sub_1",
            goal="do it", depth=0,
        )
        assert e.event_type == "subagent.started"
        p = e.to_payload()
        assert p["parent_session_id"] == "sess1"
        assert p["subagent_id"] == "sub_1"
        with pytest.raises(AttributeError):
            e.goal = "mutate"  # type: ignore[misc]

    def test_subagent_completed_frozen(self) -> None:
        e = SubagentCompleted(
            parent_session_id="s", subagent_id="sub_2",
            goal="g", summary="ok", success=True,
            duration_s=1.5, tool_calls=3,
        )
        assert e.event_type == "subagent.completed"
        assert e.to_payload()["tool_calls"] == 3

    def test_subagent_failed_frozen(self) -> None:
        e = SubagentFailed(
            parent_session_id="s", subagent_id="sub_3",
            goal="g", error="oops", duration_s=0.5,
        )
        assert e.event_type == "subagent.failed"
        assert e.to_payload()["error"] == "oops"
        assert e.to_payload()["status"] == "failed"


# ── EventBus integration ──


@pytest.mark.asyncio
async def test_successful_delegation_emits_started_and_completed() -> None:
    """A successful subagent delegation should emit Started then Completed."""
    bus = RecordingEventBus()
    mgr = SubagentManager(executor=FakeExecutor(), event_bus=bus)
    cfg = SubagentConfig(goal="test goal", parent_session_id="parent_1", depth=0)

    result = await mgr.delegate(cfg)

    assert result.status == "completed"
    # Let fire-and-forget tasks run
    await asyncio.sleep(0)

    types = [et for et, _ in bus.events]
    assert "subagent.started" in types
    assert "subagent.completed" in types

    # Started should come before Completed
    assert types.index("subagent.started") < types.index("subagent.completed")

    # Verify payload fields
    started_payload = bus.events[types.index("subagent.started")][1]
    assert started_payload["parent_session_id"] == "parent_1"
    assert started_payload["goal"] == "test goal"
    assert started_payload["depth"] == 0

    completed_payload = bus.events[types.index("subagent.completed")][1]
    assert completed_payload["success"] is True
    assert completed_payload["tool_calls"] == 2


@pytest.mark.asyncio
async def test_failed_delegation_emits_started_and_failed() -> None:
    """A failing executor should emit Started then Failed."""
    bus = RecordingEventBus()
    mgr = SubagentManager(executor=FakeExecutor(fail=True), event_bus=bus)
    cfg = SubagentConfig(goal="crash goal", depth=0)

    result = await mgr.delegate(cfg)

    assert result.status == "failed"
    await asyncio.sleep(0)

    types = [et for et, _ in bus.events]
    assert "subagent.started" in types
    assert "subagent.failed" in types

    failed_payload = bus.events[types.index("subagent.failed")][1]
    assert "executor boom" in failed_payload["error"]
    assert failed_payload["status"] == "failed"


@pytest.mark.asyncio
async def test_depth_exceeded_emits_no_events() -> None:
    """When depth limit is exceeded, no events should be emitted (early return)."""
    bus = RecordingEventBus()
    mgr = SubagentManager(executor=FakeExecutor(), max_depth=1, event_bus=bus)
    cfg = SubagentConfig(goal="deep", depth=1)

    result = await mgr.delegate(cfg)

    assert result.status == "failed"
    assert result.error == "max_depth_exceeded"
    await asyncio.sleep(0)
    # No events emitted — depth guard returns before the lifecycle starts
    assert len(bus.events) == 0


@pytest.mark.asyncio
async def test_no_executor_emits_no_events() -> None:
    """When no executor is configured, no events should be emitted."""
    bus = RecordingEventBus()
    mgr = SubagentManager(executor=None, event_bus=bus)
    cfg = SubagentConfig(goal="noop")

    result = await mgr.delegate(cfg)

    assert result.status == "failed"
    assert result.error == "no_executor"
    await asyncio.sleep(0)
    assert len(bus.events) == 0


# ── Backward compatibility: event_bus=None ──


@pytest.mark.asyncio
async def test_none_event_bus_still_works() -> None:
    """With event_bus=None, delegation succeeds and on_complete still fires."""
    callback_results: List[SubagentResult] = []
    mgr = SubagentManager(
        executor=FakeExecutor(),
        on_complete=callback_results.append,
        event_bus=None,
    )
    cfg = SubagentConfig(goal="no bus", depth=0)

    result = await mgr.delegate(cfg)

    assert result.status == "completed"
    assert len(callback_results) == 1
    assert callback_results[0].goal == "no bus"


@pytest.mark.asyncio
async def test_on_complete_fires_with_event_bus() -> None:
    """on_complete callback should still fire when event_bus is present."""
    bus = RecordingEventBus()
    callback_results: List[SubagentResult] = []
    mgr = SubagentManager(
        executor=FakeExecutor(),
        on_complete=callback_results.append,
        event_bus=bus,
    )
    cfg = SubagentConfig(goal="both", depth=0)

    result = await mgr.delegate(cfg)

    assert result.status == "completed"
    assert len(callback_results) == 1
    await asyncio.sleep(0)
    assert len(bus.events) == 2  # started + completed


# ── Emission failure containment ──


@pytest.mark.asyncio
async def test_failing_event_bus_does_not_break_delegation() -> None:
    """A broken EventBus must not prevent the subagent from completing."""
    bus = FailingEventBus()
    mgr = SubagentManager(executor=FakeExecutor(), event_bus=bus)
    cfg = SubagentConfig(goal="resilient", depth=0)

    result = await mgr.delegate(cfg)

    # The delegation completes despite the bus failing
    assert result.status == "completed"
    assert result.summary == "done"


# ── cancel_all() task lifecycle ──


class SlowExecutor:
    """Executor that sleeps forever until cancelled."""

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        await asyncio.sleep(3600)  # effectively infinite
        return SubagentResult(
            session_id="slow", goal=config.goal, summary="done",
            status="completed", elapsed_s=0.0,
        )


@pytest.mark.asyncio
async def test_cancel_all_cancels_running_task() -> None:
    """A slow-running subagent should appear in _active, cancel_all returns 1,
    and the result has status='cancelled' with a SubagentFailed event emitted."""
    bus = RecordingEventBus()
    mgr = SubagentManager(executor=SlowExecutor(), event_bus=bus)
    cfg = SubagentConfig(goal="slow task", depth=0)

    result_holder: List[SubagentResult] = []

    async def _run_delegate() -> None:
        r = await mgr.delegate(cfg)
        result_holder.append(r)

    task = asyncio.create_task(_run_delegate())
    # Allow the delegate to start and register in _active
    await asyncio.sleep(0.05)

    assert len(mgr._active) == 1, "task should be registered in _active"
    n = mgr.cancel_all()
    assert n == 1, "cancel_all should return 1"

    # Let the cancellation propagate
    await asyncio.sleep(0.05)
    # The delegate task may raise CancelledError or catch it internally
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(mgr._active) == 0, "_active should be cleaned up"
    assert len(result_holder) == 1
    assert result_holder[0].status == "cancelled"

    # Let fire-and-forget event tasks settle
    await asyncio.sleep(0)
    types = [et for et, _ in bus.events]
    assert "subagent.started" in types
    assert "subagent.failed" in types
    # The failed event should carry status="cancelled"
    failed_payload = bus.events[types.index("subagent.failed")][1]
    assert failed_payload["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_all_returns_zero_when_idle() -> None:
    """cancel_all on an idle manager returns 0."""
    mgr = SubagentManager(executor=FakeExecutor())
    assert mgr.cancel_all() == 0


@pytest.mark.asyncio
async def test_active_is_populated_during_execution() -> None:
    """_active should contain the task while the executor is running."""
    checkpoint = asyncio.Event()
    done_event = asyncio.Event()

    class CheckpointExecutor:
        async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
            checkpoint.set()
            await done_event.wait()
            return SubagentResult(
                session_id="cp", goal=config.goal, summary="ok",
                status="completed", elapsed_s=0.0,
            )

    mgr = SubagentManager(executor=CheckpointExecutor())
    cfg = SubagentConfig(goal="check active", depth=0)

    task = asyncio.create_task(mgr.delegate(cfg))
    await checkpoint.wait()
    assert len(mgr._active) == 1

    done_event.set()
    result = await task
    assert result.status == "completed"
    assert len(mgr._active) == 0


# ── DefaultSubagentExecutor approval gate ──


class FakeLLM:
    """Minimal LLM stub that returns a single tool call then stops."""

    def __init__(self, tool_calls: Optional[list] = None) -> None:
        self._calls = tool_calls or []
        self._call_count = 0

    async def achat(self, messages: list, **kwargs: Any) -> Any:
        self._call_count += 1
        if self._call_count == 1 and self._calls:
            return _FakeLLMResponse(content="", tool_calls=self._calls)
        return _FakeLLMResponse(content="All done.", tool_calls=[])


class _FakeLLMResponse:
    def __init__(self, content: str, tool_calls: list) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: dict) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


@pytest.mark.asyncio
async def test_default_executor_blocks_mutating_tool_without_pipeline() -> None:
    """Without a pipeline, a tool with risk_level='mutating' should be refused."""
    async def shell_handler(**kwargs: Any) -> dict:
        return {"ok": True, "output": "ran"}

    definitions = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {}},
                "x_leapflow": {"category": "shell", "risk_level": "mutating"},
            },
        },
    ]

    tc = _FakeToolCall(id="tc1", name="run_shell", arguments={})
    llm = FakeLLM(tool_calls=[tc])

    executor = DefaultSubagentExecutor(
        llm=llm,
        tool_handlers={"run_shell": shell_handler},
        tool_definitions=definitions,
        tool_pipeline=None,  # no pipeline
    )

    config = SubagentConfig(goal="test", depth=0)
    result = await executor.execute_subagent(config)
    # The executor should complete (not crash) but the tool should be blocked
    assert result.status == "completed"
    assert result.tool_calls == 1


@pytest.mark.asyncio
async def test_default_executor_allows_read_only_tool_without_pipeline() -> None:
    """Without a pipeline, a tool with risk_level='read_only' should execute."""
    handler_called = [False]

    async def read_handler(**kwargs: Any) -> dict:
        handler_called[0] = True
        return {"ok": True, "data": "read result"}

    definitions = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
                "x_leapflow": {"category": "fs", "risk_level": "read_only"},
            },
        },
    ]

    tc = _FakeToolCall(id="tc2", name="read_file", arguments={})
    llm = FakeLLM(tool_calls=[tc])

    executor = DefaultSubagentExecutor(
        llm=llm,
        tool_handlers={"read_file": read_handler},
        tool_definitions=definitions,
        tool_pipeline=None,
    )

    config = SubagentConfig(goal="read test", depth=0)
    result = await executor.execute_subagent(config)
    assert result.status == "completed"
    assert handler_called[0], "read_only handler should have been called"


@pytest.mark.asyncio
async def test_default_executor_routes_through_pipeline_when_present() -> None:
    """When a pipeline with interceptors is present, tools route through it."""
    pipeline_invocations: List[str] = []

    class RecordingInterceptor:
        @property
        def name(self) -> str:
            return "test_recorder"

        @property
        def priority(self) -> int:
            return 50

        async def before(self, context: Any) -> Optional[Dict[str, Any]]:
            pipeline_invocations.append(f"before:{context.tool_name}")
            return None

        async def after(self, context: Any, result: Dict[str, Any]) -> Dict[str, Any]:
            pipeline_invocations.append(f"after:{context.tool_name}")
            return result

    from leapflow.domain.tool_pipeline import ToolExecutionPipeline

    pipeline = ToolExecutionPipeline()
    pipeline.register(RecordingInterceptor())

    async def my_handler(**kwargs: Any) -> dict:
        return {"ok": True}

    definitions = [
        {
            "type": "function",
            "function": {
                "name": "mutating_tool",
                "description": "A mutating tool",
                "parameters": {"type": "object", "properties": {}},
                "x_leapflow": {"category": "test", "risk_level": "high"},
            },
        },
    ]

    tc = _FakeToolCall(id="tc3", name="mutating_tool", arguments={})
    llm = FakeLLM(tool_calls=[tc])

    executor = DefaultSubagentExecutor(
        llm=llm,
        tool_handlers={"mutating_tool": my_handler},
        tool_definitions=definitions,
        tool_pipeline=pipeline,
    )

    config = SubagentConfig(goal="pipeline test", depth=0)
    result = await executor.execute_subagent(config)
    assert result.status == "completed"
    assert "before:mutating_tool" in pipeline_invocations
    assert "after:mutating_tool" in pipeline_invocations


def test_safe_risk_levels_constant() -> None:
    """_SAFE_RISK_LEVELS should include exactly read_only and none."""
    assert _SAFE_RISK_LEVELS == frozenset({"read_only", "none"})

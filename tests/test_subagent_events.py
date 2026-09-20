# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for SubagentManager EventBus integration (Phase 4A P1-5)."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from leapflow.engine.subagent import (
    SubagentCompleted,
    SubagentConfig,
    SubagentFailed,
    SubagentManager,
    SubagentResult,
    SubagentStarted,
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

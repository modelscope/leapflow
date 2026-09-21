# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Part B: TaskGraph execution_mode + TaskScheduler subagent dispatch.

Covers:
- TaskNode.execution_mode field (default / agent)
- TaskGraph serialization round-trip with execution_mode
- TaskScheduler dispatches agent-mode nodes through SubagentExecutor
- Default dispatch unchanged for non-agent nodes
- Graceful failure when SubagentExecutor is not configured
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from leapflow.engine.task_planning.task_graph import (
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from leapflow.engine.task_planning.scheduler import (
    SubagentNodeExecutor,
    TaskScheduler,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _node(
    id: str,
    *,
    action: str = "test_skill",
    depends_on: List[str] | None = None,
    execution_mode: str = "default",
    expected_effect: str = "",
    **kwargs: Any,
) -> TaskNode:
    return TaskNode(
        id=id,
        name=f"Node {id}",
        action=action,
        depends_on=depends_on or [],
        execution_mode=execution_mode,
        expected_effect=expected_effect,
        **kwargs,
    )


@dataclass
class FakeSubagentResult:
    """Minimal SubagentResult-like object for testing."""

    session_id: str = "sub_test"
    goal: str = "test goal"
    summary: str = "test summary"
    status: str = "completed"
    elapsed_s: float = 1.0
    error: Optional[str] = None


class FakeSubagentExecutor:
    """A SubagentExecutor that records calls and returns a configurable result."""

    def __init__(
        self,
        result: Optional[FakeSubagentResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._result = result or FakeSubagentResult()
        self._error = error
        self.calls: list[Any] = []

    async def execute_subagent(self, config: Any) -> FakeSubagentResult:
        self.calls.append(config)
        if self._error:
            raise self._error
        return self._result


def _fake_registry() -> MagicMock:
    reg = MagicMock()
    reg.get.return_value = None
    return reg


# ═══════════════════════════════════════════════════════════════════
# TaskNode.execution_mode
# ═══════════════════════════════════════════════════════════════════


class TestTaskNodeExecutionMode:
    """execution_mode field defaults correctly and serializes."""

    def test_default_execution_mode(self) -> None:
        node = TaskNode(id="a", name="A", action="skill_a")
        assert node.execution_mode == "default"

    def test_agent_execution_mode(self) -> None:
        node = TaskNode(id="a", name="A", action="goal_a", execution_mode="agent")
        assert node.execution_mode == "agent"

    def test_from_dict_default_mode(self) -> None:
        """Nodes without execution_mode in dict default to 'default'."""
        graph = TaskGraph.from_dict({
            "goal": "test",
            "nodes": [{"id": "a", "action": "skill_a"}],
        })
        assert graph.nodes["a"].execution_mode == "default"

    def test_from_dict_agent_mode(self) -> None:
        """Nodes with execution_mode='agent' in dict are deserialized."""
        graph = TaskGraph.from_dict({
            "goal": "test",
            "nodes": [{"id": "a", "action": "goal_a", "execution_mode": "agent"}],
        })
        assert graph.nodes["a"].execution_mode == "agent"

    def test_to_dict_includes_execution_mode(self) -> None:
        graph = TaskGraph(goal="test")
        graph.add_node(_node("a", execution_mode="agent"))
        data = graph.to_dict()
        node_data = data["nodes"][0]
        assert node_data["execution_mode"] == "agent"

    def test_round_trip_serialization(self) -> None:
        """from_dict → to_dict → from_dict preserves execution_mode."""
        original = {
            "goal": "round trip",
            "nodes": [
                {"id": "a", "action": "skill_a"},
                {"id": "b", "action": "agent_goal", "execution_mode": "agent", "depends_on": ["a"]},
            ],
        }
        graph = TaskGraph.from_dict(original)
        data = graph.to_dict()
        restored = TaskGraph.from_dict(data)
        assert restored.nodes["a"].execution_mode == "default"
        assert restored.nodes["b"].execution_mode == "agent"


# ═══════════════════════════════════════════════════════════════════
# SubagentNodeExecutor Protocol
# ═══════════════════════════════════════════════════════════════════


class TestSubagentNodeExecutorProtocol:
    """SubagentNodeExecutor Protocol is runtime-checkable."""

    def test_fake_executor_satisfies_protocol(self) -> None:
        executor = FakeSubagentExecutor()
        assert isinstance(executor, SubagentNodeExecutor)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), SubagentNodeExecutor)


# ═══════════════════════════════════════════════════════════════════
# TaskScheduler: agent-mode dispatch
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerAgentModeDispatch:
    """TaskScheduler routes agent-mode nodes through SubagentExecutor."""

    @pytest.mark.asyncio
    async def test_agent_mode_dispatches_through_executor(self) -> None:
        """A node with execution_mode='agent' uses SubagentExecutor."""
        executor = FakeSubagentExecutor(
            result=FakeSubagentResult(summary="Agent done", status="completed")
        )
        dispatcher = AsyncMock(return_value={"result": "default done"})
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=dispatcher,
            subagent_executor=executor,
        )

        graph = TaskGraph(goal="test agent dispatch")
        graph.add_node(_node(
            "a",
            execution_mode="agent",
            expected_effect="Search documentation",
        ))

        result = await scheduler.execute_graph(graph)

        assert result.nodes["a"].status == TaskStatus.COMPLETED
        assert result.nodes["a"].result == "Agent done"
        assert len(executor.calls) == 1
        dispatcher.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_mode_uses_action_dispatcher(self) -> None:
        """A node with default execution_mode uses ActionDispatcher, not SubagentExecutor."""
        executor = FakeSubagentExecutor()
        dispatcher = AsyncMock(return_value={"result": "dispatched"})
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=dispatcher,
            subagent_executor=executor,
        )

        graph = TaskGraph(goal="test default dispatch")
        graph.add_node(_node("a", action="test_skill"))

        result = await scheduler.execute_graph(graph)

        assert result.nodes["a"].status == TaskStatus.COMPLETED
        dispatcher.assert_called_once()
        assert len(executor.calls) == 0

    @pytest.mark.asyncio
    async def test_agent_mode_fails_without_executor(self) -> None:
        """An agent-mode node fails gracefully when no SubagentExecutor is injected."""
        dispatcher = AsyncMock(return_value={"result": "ok"})
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=dispatcher,
            subagent_executor=None,  # no executor
        )

        graph = TaskGraph(goal="test no executor")
        graph.add_node(_node("a", execution_mode="agent"))

        result = await scheduler.execute_graph(graph)

        assert result.nodes["a"].status == TaskStatus.FAILED
        assert "SubagentExecutor" in (result.nodes["a"].error or "")

    @pytest.mark.asyncio
    async def test_agent_mode_failed_result_propagates_error(self) -> None:
        """An agent-mode node whose executor returns status=failed → node FAILED."""
        executor = FakeSubagentExecutor(
            result=FakeSubagentResult(
                summary="LLM error",
                status="failed",
                error="context_overflow",
            )
        )
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=AsyncMock(),
            subagent_executor=executor,
        )

        graph = TaskGraph(goal="test agent failure")
        graph.add_node(_node("a", execution_mode="agent"))

        result = await scheduler.execute_graph(graph)

        assert result.nodes["a"].status == TaskStatus.FAILED
        assert "context_overflow" in (result.nodes["a"].error or "")

    @pytest.mark.asyncio
    async def test_mixed_graph_correct_routing(self) -> None:
        """A graph with both default and agent nodes routes each correctly."""
        executor = FakeSubagentExecutor(
            result=FakeSubagentResult(summary="agent result", status="completed")
        )
        dispatcher = AsyncMock(return_value={"result": "skill result"})
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=dispatcher,
            subagent_executor=executor,
        )

        graph = TaskGraph(goal="mixed")
        graph.add_node(_node("a", action="prep_skill"))
        graph.add_node(_node(
            "b",
            depends_on=["a"],
            execution_mode="agent",
            expected_effect="Analyze results",
        ))

        result = await scheduler.execute_graph(graph)

        assert result.nodes["a"].status == TaskStatus.COMPLETED
        assert result.nodes["b"].status == TaskStatus.COMPLETED
        dispatcher.assert_called_once()  # node "a"
        assert len(executor.calls) == 1  # node "b"

    @pytest.mark.asyncio
    async def test_set_subagent_executor_late_binding(self) -> None:
        """set_subagent_executor allows late-binding the executor."""
        scheduler = TaskScheduler(
            _fake_registry(),
            action_dispatcher=AsyncMock(return_value={"result": "ok"}),
        )
        assert scheduler._subagent_executor is None

        executor = FakeSubagentExecutor()
        scheduler.set_subagent_executor(executor)
        assert scheduler._subagent_executor is executor

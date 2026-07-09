"""Scenario-based integration tests for the agent execution pipeline."""

from __future__ import annotations

import tempfile
from typing import List
from unittest.mock import AsyncMock

import pytest

from conftest import StubLLM, make_settings
from leapflow.engine.engine import AgentEngine, build_default_registry
from leapflow.engine.graph_planner import GraphPlanner
from leapflow.engine.intent_classifier import Intent
from leapflow.engine.scheduler import TaskScheduler
from leapflow.engine.task_graph import (
    GraphValidationError,
    RetryPolicy,
    TaskGraph,
    TaskNode,
    TaskStatus,
)
from leapflow.memory import (
    EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider,
)


class _FixedClassifier:
    """Deterministic intent classifier for routing tests."""

    def __init__(self, label: str) -> None:
        self._intent = Intent(label=label, reason="test")

    async def classify(self, user_text: str) -> Intent:
        return self._intent


def _node(
    id: str,
    *,
    action: str = "test_skill",
    depends_on: List[str] | None = None,
    **kwargs,
) -> TaskNode:
    return TaskNode(
        id=id,
        name=f"Node {id}",
        action=action,
        depends_on=depends_on or [],
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# Engine scenarios
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_simple_query_returns_answer() -> None:
    """memory_recent intent: LLM synthesizes an answer from recent events."""
    answer = (
        '{"thought":"done","action":{"type":"answer","name":"final",'
        '"payload":{"text":"You edited README.md"}}}'
    )
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([answer])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        imm.ingest("event.fs_change", "File modified: /tmp/README.md", path="/tmp/README.md")
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("memory_recent")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("What did I change recently?")
            assert "README" in out
            assert llm.call_count == 1
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_react_loop_tool_then_answer() -> None:
    """Unified tool loop: executes a tool call, then returns final answer."""
    # Provide a parseable tool call (time_get), then a plain text final answer
    tool_reply = '<tool_call>{"name": "time_get", "arguments": {}}</tool_call>'
    final_reply = "UI observed successfully"
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(
            [
                tool_reply,
                final_reply,
            ]
        )
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("Observe the desktop and tell me what you see.")
            assert out == "UI observed successfully"
            assert llm.call_count >= 2
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_dag_execution_end_to_end() -> None:
    """DAG planner/scheduler: direct invocation produces graph summary."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()

        planned = TaskGraph(goal="organize downloads")
        planned.add_node(_node("step1", action="file_organizer"))
        planned.add_node(_node("step2", action="clipboard_manager", depends_on=["step1"]))

        executed = TaskGraph(goal="organize downloads")
        executed.add_node(_node("step1", action="file_organizer"))
        executed.add_node(_node("step2", action="clipboard_manager", depends_on=["step1"]))
        executed.mark_completed("step1", "listed files")
        executed.mark_completed("step2", "organized")

        mock_planner = AsyncMock(spec=GraphPlanner)
        mock_planner.plan.return_value = planned

        mock_scheduler = AsyncMock(spec=TaskScheduler)
        mock_scheduler.execute_graph.return_value = executed

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(
                settings, rpc, llm, wm, lt, imm, reg, classifier,
                graph_planner=mock_planner,
                scheduler=mock_scheduler,
            )
            # Call _handle_complex_task directly (DAG is internal machinery)
            out = await engine._handle_complex_task("Organize my downloads folder")
            assert "organize downloads" in out
            mock_planner.plan.assert_awaited_once()
            mock_scheduler.execute_graph.assert_awaited_once()
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_engine_remembers_context() -> None:
    """Engine run stores user query and assistant reply in working memory."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        # Unified loop: plain text response is returned as final answer
        llm = StubLLM(
            [
                "Hello from the assistant.",
            ]
        )
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            query = "What can you do?"
            out = await engine.run(query)
            assert "Hello from the assistant." in out

            messages = wm.as_chat_messages()
            roles = [m["role"] for m in messages]
            contents = [str(m["content"]) for m in messages]
            assert "user" in roles
            assert "assistant" in roles
            assert query in contents
            assert any("Hello from the assistant." in c for c in contents)
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_streaming_engine_estimates_context_tokens_without_provider_usage() -> None:
    """Streaming providers often omit usage; status still needs prompt utilization."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class StreamingOnlyLLM(LLMProvider):
        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            return LLMChatResponse(content="fallback")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            yield "streamed answer"

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "stream_output": True,
                "native_tool_calling_enabled": False,
            }
        )
        rpc = MockBridge()
        llm = StreamingOnlyLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("Summarize a long conversation")]

            assert any(event.type == "final" for event in events)
            assert engine.context_token_count > 0
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_immediate_memory_integration() -> None:
    """EpisodicMemoryProvider fragments surface in memory_recent responses."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["You recently modified /tmp/README.md"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        imm.ingest("event.fs_change", "File modified: /tmp/README.md", path="/tmp/README.md")
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("memory_recent")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            out = await engine.run("What did I change just now?")
            assert "README" in out
        finally:
            lt.close()


# ═══════════════════════════════════════════════════════════════════
# TaskGraph scenarios
# ═══════════════════════════════════════════════════════════════════


def test_task_graph_linear_chain() -> None:
    """A → B → C: topological order and ready_nodes advance step by step."""
    g = TaskGraph(goal="linear")
    g.add_node(_node("a"))
    g.add_node(_node("b", depends_on=["a"]))
    g.add_node(_node("c", depends_on=["b"]))

    order = g.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")

    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["a"]

    g.mark_completed("a", "a-out")
    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["b"]

    g.mark_completed("b", "b-out")
    ready = g.ready_nodes()
    assert [n.id for n in ready] == ["c"]

    g.mark_completed("c", "c-out")
    assert g.ready_nodes() == []
    assert g.is_complete


def test_task_graph_diamond_dependency() -> None:
    """A → {B, C} → D: B and C become ready in parallel after A completes."""
    g = TaskGraph(goal="diamond")
    g.add_node(_node("a"))
    g.add_node(_node("b", depends_on=["a"]))
    g.add_node(_node("c", depends_on=["a"]))
    g.add_node(_node("d", depends_on=["b", "c"]))

    assert [n.id for n in g.ready_nodes()] == ["a"]

    g.mark_completed("a", "root")
    ready_ids = {n.id for n in g.ready_nodes()}
    assert ready_ids == {"b", "c"}

    g.mark_completed("b", "left")
    assert [n.id for n in g.ready_nodes()] == ["c"]

    g.mark_completed("c", "right")
    assert [n.id for n in g.ready_nodes()] == ["d"]


def test_task_graph_cycle_detection() -> None:
    """A → B → A cycle is rejected by validate() and from_dict()."""
    g = TaskGraph(goal="cyclic")
    g.nodes["a"] = _node("a", depends_on=["b"])
    g.nodes["b"] = _node("b", depends_on=["a"])

    errors = g.validate()
    assert any("cycle" in e.lower() for e in errors)

    with pytest.raises(GraphValidationError):
        TaskGraph.from_dict(
            {
                "goal": "cyclic",
                "nodes": [
                    {"id": "a", "action": "skill_a", "depends_on": ["b"]},
                    {"id": "b", "action": "skill_b", "depends_on": ["a"]},
                ],
            }
        )


def test_task_graph_param_resolution() -> None:
    """${a.output} and ${graph.goal} substitute upstream results and goal text."""
    g = TaskGraph(goal="Ship release")
    g.add_node(_node("a"))
    g.add_node(
        _node(
            "b",
            depends_on=["a"],
            params={
                "upstream": "${a.output}",
                "goal": "${graph.goal}",
                "nested": "${a.result.name}",
            },
        )
    )
    g.mark_completed("a", {"name": "artifact", "version": "1.0"})

    resolved = g.resolve_params(g.nodes["b"])
    assert resolved["upstream"] == {"name": "artifact", "version": "1.0"}
    assert resolved["goal"] == "Ship release"
    assert resolved["nested"] == "artifact"


def test_task_graph_retry_policy() -> None:
    """Failed nodes can be reset while retries remain; exhausted retries stay failed."""
    g = TaskGraph(goal="retry")
    policy = RetryPolicy(max_retries=2)
    g.add_node(_node("a", retry_policy=policy))

    node = g.nodes["a"]

    g.mark_running("a")
    assert node.attempt_count == 1
    g.mark_failed("a", "transient error")
    assert node.status == TaskStatus.FAILED

    g.reset_node("a")
    assert node.status == TaskStatus.PENDING
    assert node.error is None

    g.mark_running("a")
    g.mark_failed("a", "transient error")
    g.reset_node("a")

    g.mark_running("a")
    g.mark_failed("a", "permanent error")
    assert node.status == TaskStatus.FAILED
    assert node.attempt_count == 3
    assert node.error == "permanent error"

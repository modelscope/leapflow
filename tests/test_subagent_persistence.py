# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for subagent tool_calls tracking and session persistence (P1 items).

Covers:
- EngineFrameSubagentExecutor populates tool_calls from child frame usage
- SubagentManager persists messages to ConversationStore when available
- SubagentManager gracefully degrades without ConversationStore
- DefaultSubagentExecutor tool_calls tracking still correct
- SubagentResult.messages field propagation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from leapflow.engine.subagent import (
    DefaultSubagentExecutor,
    EngineFrameSubagentExecutor,
    SubagentConfig,
    SubagentManager,
    SubagentResult,
)


# ── Helpers ──


class RecordingConversationStore:
    """Minimal ConversationStore stand-in that records calls."""

    def __init__(self) -> None:
        self.sessions: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []

    def create_session(self, session_id: str, **kwargs: Any) -> Any:
        self.sessions.append({"session_id": session_id, **kwargs})
        # Return a minimal object satisfying the protocol
        return type("S", (), {"session_id": session_id})()

    def append_message(
        self, session_id: str, role: str, content: str, **kwargs: Any
    ) -> Any:
        self.messages.append(
            {"session_id": session_id, "role": role, "content": content, **kwargs}
        )
        return type("M", (), {"message_id": "m1", "session_id": session_id})()


class FailingConversationStore:
    """Store that always raises — verifies persistence failures are contained."""

    def create_session(self, session_id: str, **kwargs: Any) -> Any:
        raise RuntimeError("store on fire")

    def append_message(
        self, session_id: str, role: str, content: str, **kwargs: Any
    ) -> Any:
        raise RuntimeError("store on fire")


class FakeExecutorWithMessages:
    """Executor that returns a canned result with messages."""

    def __init__(
        self,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        tool_calls: int = 0,
    ) -> None:
        self._messages = messages
        self._tool_calls = tool_calls

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        return SubagentResult(
            session_id="sub_test123abc",
            goal=config.goal,
            summary="task done",
            status="completed",
            elapsed_s=0.01,
            tool_calls=self._tool_calls,
            messages=self._messages,
        )


class FakeExecutorNoMessages:
    """Executor that returns result without messages (engine-frame path)."""

    async def execute_subagent(self, config: SubagentConfig) -> SubagentResult:
        return SubagentResult(
            session_id="sub_engine12345",
            goal=config.goal,
            summary="engine done",
            status="completed",
            elapsed_s=0.05,
            tool_calls=3,
            messages=None,
        )


# ── Item 1: EngineFrameSubagentExecutor tool_calls tracking ──


class TestEngineFrameToolCallsTracking:
    """Verify that EngineFrameSubagentExecutor populates tool_calls."""

    @pytest.mark.asyncio
    async def test_tool_calls_from_tuple_return(self) -> None:
        """When _run_child returns (summary, tool_calls), the result
        should carry the tool_calls count."""

        async def fake_run_child(
            goal: str,
            *,
            depth: int,
            tool_filter: Any = None,
            enable_thinking: bool = False,
        ) -> Tuple[str, int]:
            return ("completed the task", 5)

        executor = EngineFrameSubagentExecutor(
            run_child=fake_run_child,
            tool_names=["read_file", "write_file"],
        )
        config = SubagentConfig(goal="do something", depth=0)
        result = await executor.execute_subagent(config)

        assert result.status == "completed"
        assert result.tool_calls == 5
        assert result.summary == "completed the task"

    @pytest.mark.asyncio
    async def test_tool_calls_zero_when_no_tools_used(self) -> None:
        """When child frame used no tools, tool_calls should be 0."""

        async def fake_run_child(
            goal: str,
            *,
            depth: int,
            tool_filter: Any = None,
            enable_thinking: bool = False,
        ) -> Tuple[str, int]:
            return ("answered directly", 0)

        executor = EngineFrameSubagentExecutor(
            run_child=fake_run_child,
            tool_names=[],
        )
        config = SubagentConfig(goal="simple question", depth=0)
        result = await executor.execute_subagent(config)

        assert result.tool_calls == 0

    @pytest.mark.asyncio
    async def test_backward_compat_str_return(self) -> None:
        """If _run_child returns a plain str (legacy), tool_calls defaults to 0."""

        async def legacy_run_child(
            goal: str,
            *,
            depth: int,
            tool_filter: Any = None,
            enable_thinking: bool = False,
        ) -> str:
            return "legacy result"

        executor = EngineFrameSubagentExecutor(
            run_child=legacy_run_child,
            tool_names=[],
        )
        config = SubagentConfig(goal="legacy", depth=0)
        result = await executor.execute_subagent(config)

        assert result.tool_calls == 0
        assert result.summary == "legacy result"


# ── Item 2: SubagentManager persistence ──


class TestSubagentManagerPersistence:
    """Verify SubagentManager persists messages via ConversationStore."""

    @pytest.mark.asyncio
    async def test_persists_messages_when_store_available(self) -> None:
        """Messages from DefaultSubagentExecutor should be persisted."""
        store = RecordingConversationStore()
        messages = [
            {"role": "system", "content": "You are a subagent."},
            {"role": "user", "content": "do the task"},
            {"role": "assistant", "content": "done"},
        ]
        executor = FakeExecutorWithMessages(messages=messages, tool_calls=1)
        mgr = SubagentManager(
            executor=executor,
            conversation_store=store,
        )
        config = SubagentConfig(goal="persist test", depth=0)

        result = await mgr.delegate(config)

        assert result.status == "completed"
        # Session should be created
        assert len(store.sessions) == 1
        assert store.sessions[0]["source"] == "subagent"
        assert "persist test" in store.sessions[0]["title"]
        # All messages should be persisted
        assert len(store.messages) == 3
        assert store.messages[0]["role"] == "system"
        assert store.messages[1]["role"] == "user"
        assert store.messages[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_no_persistence_without_store(self) -> None:
        """Without conversation_store, delegation still works fine."""
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "done"},
        ]
        executor = FakeExecutorWithMessages(messages=messages)
        mgr = SubagentManager(executor=executor, conversation_store=None)
        config = SubagentConfig(goal="no store", depth=0)

        result = await mgr.delegate(config)

        assert result.status == "completed"
        assert result.summary == "task done"

    @pytest.mark.asyncio
    async def test_no_persistence_when_messages_none(self) -> None:
        """Engine-frame executor returns messages=None; no persistence attempted."""
        store = RecordingConversationStore()
        executor = FakeExecutorNoMessages()
        mgr = SubagentManager(executor=executor, conversation_store=store)
        config = SubagentConfig(goal="engine path", depth=0)

        result = await mgr.delegate(config)

        assert result.status == "completed"
        # No session or messages persisted — engine-frame path handles it.
        assert len(store.sessions) == 0
        assert len(store.messages) == 0

    @pytest.mark.asyncio
    async def test_persistence_failure_does_not_break_delegation(self) -> None:
        """A broken store must not prevent the subagent from completing."""
        store = FailingConversationStore()
        messages = [{"role": "user", "content": "boom"}]
        executor = FakeExecutorWithMessages(messages=messages)
        mgr = SubagentManager(executor=executor, conversation_store=store)
        config = SubagentConfig(goal="resilient", depth=0)

        result = await mgr.delegate(config)

        assert result.status == "completed"
        assert result.summary == "task done"

    @pytest.mark.asyncio
    async def test_persisted_session_has_parent_link(self) -> None:
        """The persisted session should carry parent_session_id for lineage."""
        store = RecordingConversationStore()
        messages = [{"role": "assistant", "content": "ok"}]
        executor = FakeExecutorWithMessages(messages=messages)
        mgr = SubagentManager(executor=executor, conversation_store=store)
        config = SubagentConfig(
            goal="child task",
            depth=0,
            parent_session_id="parent_session_abc",
        )

        await mgr.delegate(config)

        assert len(store.sessions) == 1
        assert store.sessions[0]["parent_session_id"] == "parent_session_abc"

    @pytest.mark.asyncio
    async def test_tool_calls_in_messages_persisted(self) -> None:
        """Messages with tool_calls should have them persisted."""
        store = RecordingConversationStore()
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "file content"},
        ]
        executor = FakeExecutorWithMessages(messages=messages, tool_calls=1)
        mgr = SubagentManager(executor=executor, conversation_store=store)
        config = SubagentConfig(goal="tool task", depth=0)

        await mgr.delegate(config)

        assert len(store.messages) == 3
        assert store.messages[1]["tool_calls"] == messages[1]["tool_calls"]
        assert store.messages[2]["tool_call_id"] == "tc1"


# ── DefaultSubagentExecutor: messages attached + tool_calls correct ──


class _FakeLLMResponse:
    def __init__(self, content: str, tool_calls: Optional[list] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: dict) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments


class FakeLLMForDefault:
    """LLM stub: returns one tool call on first round, then text."""

    def __init__(self, tool_calls: Optional[list] = None) -> None:
        self._calls = tool_calls or []
        self._call_count = 0

    async def achat(self, messages: list, **kwargs: Any) -> Any:
        self._call_count += 1
        if self._call_count == 1 and self._calls:
            return _FakeLLMResponse(content="", tool_calls=self._calls)
        return _FakeLLMResponse(content="All done.")


class TestDefaultSubagentExecutorTracking:
    """Verify DefaultSubagentExecutor correctly tracks tool_calls and messages."""

    @pytest.mark.asyncio
    async def test_tool_calls_counted_correctly(self) -> None:
        """Tool call count should match the actual number of tool invocations."""

        async def handler(**kwargs: Any) -> dict:
            return {"ok": True}

        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "safe_tool",
                    "description": "Safe tool",
                    "parameters": {"type": "object", "properties": {}},
                    "x_leapflow": {"category": "test", "risk_level": "read_only"},
                },
            },
        ]
        tc = _FakeToolCall(id="tc1", name="safe_tool", arguments={})
        llm = FakeLLMForDefault(tool_calls=[tc])

        executor = DefaultSubagentExecutor(
            llm=llm,
            tool_handlers={"safe_tool": handler},
            tool_definitions=definitions,
            tool_pipeline=None,
        )
        config = SubagentConfig(goal="count tools", depth=0)
        result = await executor.execute_subagent(config)

        assert result.status == "completed"
        assert result.tool_calls == 1

    @pytest.mark.asyncio
    async def test_messages_attached_to_result(self) -> None:
        """Result should carry the raw messages list for persistence."""
        llm = FakeLLMForDefault(tool_calls=[])
        executor = DefaultSubagentExecutor(
            llm=llm,
            tool_handlers={},
            tool_definitions=[],
            tool_pipeline=None,
        )
        config = SubagentConfig(goal="check messages", depth=0)
        result = await executor.execute_subagent(config)

        assert result.messages is not None
        assert len(result.messages) >= 2  # system + user at minimum
        assert result.messages[0]["role"] == "system"
        assert result.messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_tools_zero_count(self) -> None:
        """When no tool calls happen, tool_calls should be 0."""
        llm = FakeLLMForDefault()
        executor = DefaultSubagentExecutor(
            llm=llm,
            tool_handlers={},
            tool_definitions=[],
        )
        config = SubagentConfig(goal="no tools", depth=0)
        result = await executor.execute_subagent(config)

        assert result.tool_calls == 0
        assert result.messages is not None


# ── SubagentResult.messages field ──


class TestSubagentResultMessages:
    def test_messages_field_optional_default_none(self) -> None:
        """messages field defaults to None for backward compatibility."""
        result = SubagentResult(
            session_id="s1",
            goal="g",
            summary="ok",
            status="completed",
        )
        assert result.messages is None

    def test_messages_field_can_be_set(self) -> None:
        """messages field can hold a message list."""
        msgs = [{"role": "user", "content": "hello"}]
        result = SubagentResult(
            session_id="s1",
            goal="g",
            summary="ok",
            status="completed",
            messages=msgs,
        )
        assert result.messages is msgs

    def test_trim_summary_preserves_messages(self) -> None:
        """_trim_summary should preserve the messages field."""
        msgs = [{"role": "user", "content": "test"}]
        result = SubagentResult(
            session_id="s1",
            goal="g",
            summary="x" * 5000,  # will be trimmed
            status="completed",
            messages=msgs,
        )
        mgr = SubagentManager(executor=None)
        trimmed = mgr._trim_summary(result, max_chars=100)
        assert trimmed.messages is msgs
        assert len(trimmed.summary) <= 100

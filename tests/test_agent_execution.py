"""Scenario-based integration tests for the agent execution pipeline."""

from __future__ import annotations

import tempfile
from typing import List
from unittest.mock import AsyncMock

import pytest

from conftest import StubLLM, make_settings
from leapflow.engine.engine import (
    AgentEngine,
    _normalize_tool_name,
    _resolve_tool_name,
    _tool_args_metadata,
    build_default_registry,
)
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
async def test_exact_canonical_tool_names_execute_without_guessing() -> None:
    """Only exact canonical tool names (plus case/separator formatting) execute."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        captured: dict[str, object] = {}

        async def file_list_handler(args):
            captured["args"] = args
            return {"ok": True, "path": args.get("path", ""), "entries": []}

        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)
            engine._tool_bridge = None

            result = await engine._execute_general_tool(
                {"name": "file_list", "arguments": {"path": "."}},
                {"file_list": file_list_handler},
            )
            metadata = _tool_args_metadata(
                "file_list",
                {"path": "."},
                original_tool_name="File-List",
            )

            assert result["ok"] is True
            assert captured["args"] == {"path": "."}
            # Case/separator formatting of the *same* canonical name still resolves.
            assert _normalize_tool_name("File_List") == "file_list"
            assert _normalize_tool_name("file-list") == "file_list"
            # Known LLM drift patterns resolve via static alias table.
            assert _normalize_tool_name("list_directory") == "file_list"
            assert _normalize_tool_name("execute_command") == "shell_run"
            assert _normalize_tool_name("run_terminal") == "shell_run"
            alias_resolution = _resolve_tool_name("list_directory", {"path": "."})
            assert alias_resolution.normalized_name == "file_list"
            assert alias_resolution.status == "aliased"
            assert alias_resolution.auto_executable is True
            # Names NOT in alias table remain unknown.
            directory_resolution = _resolve_tool_name("directory_scan", {"path": "."})
            risky_resolution = _resolve_tool_name("please_do", {"command": "ls -la"})
            assert directory_resolution.normalized_name is None
            assert directory_resolution.status == "unknown"
            assert directory_resolution.auto_executable is False
            assert risky_resolution.normalized_name is None
            assert risky_resolution.status == "unknown"
            assert risky_resolution.auto_executable is False
            assert metadata["original_tool_name"] == "File-List"
            assert metadata["normalized_tool_name"] == "file_list"
            assert metadata["resolved_from"] == "File-List"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_retry_feedback() -> None:
    """Unknown tools should produce structured feedback instead of a bare string."""
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            result = await engine._execute_general_tool(
                {"name": "missing_magic_tool", "arguments": {"foo": "bar"}},
                {},
            )

            assert result["ok"] is False
            assert result["error_type"] == "unknown_tool"
            assert result["original_tool_name"] == "missing_magic_tool"
            assert result["retryable"] is True
            assert "available_tools" in result
            assert "suggestions" in result
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_triggers_single_self_healing_retry() -> None:
    """The loop should give the LLM one structured chance to retry an unknown tool."""
    class CaptureLLM(StubLLM):
        def __init__(self) -> None:
            super().__init__([
                '<tool_call>{"name": "missing_magic_tool", "arguments": {"foo": "bar"}}</tool_call>',
                "recovered answer",
            ])
            self.seen_messages: list[list[dict[str, object]]] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
            self.seen_messages.append(list(messages))
            return await super().achat(messages, stream=stream, enable_thinking=enable_thinking, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("Use a missing tool then recover")

            assert out == "recovered answer"
            assert llm.call_count == 2
            second_call_messages = "\n".join(str(message.get("content", "")) for message in llm.seen_messages[1])
            assert "unavailable tool name" in second_call_messages
            assert "missing_magic_tool" in second_call_messages
            assert "Available tools include" in second_call_messages
        finally:
            lt.close()

@pytest.mark.asyncio
async def test_app_connector_context_is_injected_without_extra_llm_call() -> None:
    class CaptureLLM(StubLLM):
        def __init__(self) -> None:
            super().__init__(["Use platform_connect for supported app onboarding."])
            self.seen_messages: list[list[dict[str, object]]] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, **kwargs):
            self.seen_messages.append(list(messages))
            return await super().achat(messages, stream=stream, enable_thinking=enable_thinking, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("接入飞书")

            system_prompt = str(llm.seen_messages[0][0].get("content", ""))
            assert out == "Use platform_connect for supported app onboarding."
            assert "App Connector Capability Index" in system_prompt
            assert "`feishu`" in system_prompt
            assert "`telegram`" in system_prompt
            assert "platform_connect" in system_prompt
            assert llm.call_count == 1
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()


@pytest.mark.asyncio
async def test_app_connector_llm_tool_call_uses_same_unified_loop() -> None:
    tool_reply = '<tool_call>{"name": "platform_connect", "arguments": {"action": "guide", "platform": "telegram"}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "Telegram guide ready"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("配置 Telegram")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert [event.content for event in tool_events] == ["platform_connect", "platform_connect"]
            assert tool_events[1].metadata["ok"] is True
            assert events[-1].type == "final"
            assert "Telegram guide ready" in events[-1].content
            assert llm.call_count == 2
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()


@pytest.mark.asyncio
async def test_app_connector_empty_final_uses_onboarding_recovery_state() -> None:
    tool_reply = (
        '<tool_call>{"name": "platform_connect", "arguments": '
        '{"action": "guide", "platform": "feishu", '
        '"options": {"binary": "definitely-missing-cli-for-onboarding-test"}}}</tool_call>'
    )
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.gateway.server import GatewayServer
        from leapflow.platform.mock import MockBridge
        from leapflow.tools.gateway_tool import set_gateway_approval_gate, set_gateway_server

        rpc = MockBridge()
        llm = StubLLM([tool_reply, ""])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        server = GatewayServer(settings.profile_dir)
        server.discover_manifests()
        set_gateway_server(server)
        set_gateway_approval_gate(None)
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            final = await engine.run("继续接入")
        finally:
            await server.stop()
            set_gateway_approval_gate(None)
            set_gateway_server(None)
            lt.close()

    assert llm.call_count == 2
    assert "App onboarding is paused" in final
    assert "cli_missing" in final
    assert "definitely-missing-cli-for-onboarding-test" in final


@pytest.mark.asyncio
async def test_aliased_tool_in_stream_resolves_and_executes() -> None:
    """Text-mode tool calls with a known drifted name resolve via alias and execute normally."""
    tool_reply = '<tool_call>{"name": "list_directory", "arguments": {"path": "."}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "directory checked"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("List current directory")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert tool_events[0].metadata["original_tool_name"] == "list_directory"
            assert tool_events[0].metadata["tool_resolution_status"] == "aliased"
            assert tool_events[0].metadata["normalized_tool_name"] == "file_list"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_unknown_tool_in_stream_triggers_structured_retry() -> None:
    """Text-mode tool calls with a truly unknown name surface a structured unknown with suggestions."""
    tool_reply = '<tool_call>{"name": "directory_scan", "arguments": {"path": "."}}</tool_call>'
    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM([tool_reply, "directory checked"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("complex")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            events = [event async for event in engine.run_stream("List current directory")]

            tool_events = [event for event in events if event.type in {"tool_start", "tool_complete"}]
            assert [event.content for event in tool_events] == ["directory_scan", "directory_scan"]
            assert tool_events[0].metadata["original_tool_name"] == "directory_scan"
            assert tool_events[0].metadata["tool_resolution_status"] == "unknown"
            assert tool_events[1].metadata["ok"] is False
            assert tool_events[1].metadata["error_type"] == "unknown_tool"
            assert "resolved_from" not in tool_events[1].metadata
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
async def test_progressive_disclosure_light_query_omits_tools_and_thinking() -> None:
    """Plain chat should stay on the light path even when thinking is requested."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.messages: list[dict] = []
            self.kwargs: dict = {}
            self.enable_thinking = True
            self.call_count = 0

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.call_count += 1
            self.messages = list(messages)
            self.kwargs = dict(kwargs)
            self.enable_thinking = enable_thinking
            return LLMChatResponse(content="I am LeapFlow.")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "native_tool_calling_enabled": True,
            }
        )
        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("chat")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            out = await engine.run("hello", enable_thinking=True)

            assert out == "I am LeapFlow."
            assert llm.call_count == 1
            assert "tools" not in llm.kwargs
            assert llm.enable_thinking is False
            assert "file_read" not in str(llm.messages[0].get("content", ""))
            system_prompt = str(llm.messages[0].get("content", ""))
            assert "## Presentation Style" in system_prompt
            assert "Avoid redundant tool calls" in system_prompt
            assert "same tool with the same arguments" in system_prompt
            assert "existing tool result already answers" in system_prompt
            assert "No leaked tool protocol" in system_prompt
            assert "Theme-safe colors" in system_prompt
            assert "## Task Contract" in system_prompt
            assert "Original user request: hello" in system_prompt
            assert "Workspace root:" in system_prompt
            assert "never infer `.` as the project root" in system_prompt
            assert "do not probe that path" in system_prompt
            assert "~/.leapflow/.env" in system_prompt
            assert "./.env" in system_prompt
            snapshot = engine.context_budget_snapshot
            assert snapshot["disclosure_level"] == "light"
            assert snapshot["disclosure"]["native_tools"] is False
        finally:
            lt.close()


def test_task_contract_replaces_stale_contract_block() -> None:
    """Compression recovery should keep exactly one current task contract."""
    from leapflow.platform.mock import MockBridge

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        rpc = MockBridge()
        llm = StubLLM(["ok"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("chat")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            engine._session_turn_count = 1
            engine._begin_turn_context("first request")
            stale_contract = engine._task_contract_block()
            engine._session_turn_count = 2
            engine._begin_turn_context("second request")

            prepared = engine._ensure_task_contract_message([
                {"role": "system", "content": f"base system\n\n{stale_contract}\n"},
                {"role": "system", "content": stale_contract},
                {"role": "user", "content": "second request"},
            ])
            system_text = "\n".join(
                str(message.get("content", ""))
                for message in prepared
                if message.get("role") == "system"
            )

            assert system_text.count("## Task Contract") == 1
            assert "Original user request: second request" in system_text
            assert "Original user request: first request" not in system_text
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_progressive_disclosure_file_query_selects_file_schemas() -> None:
    """File-oriented requests should disclose file schemas without the full catalog."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.kwargs: dict = {}

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.kwargs = dict(kwargs)
            return LLMChatResponse(content="Done")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{
                **settings.__dict__,
                "native_tool_calling_enabled": True,
            }
        )
        rpc = MockBridge()
        llm = CaptureLLM()
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            classifier = _FixedClassifier("file")
            engine = AgentEngine(settings, rpc, llm, wm, lt, imm, reg, classifier)

            await engine.run("Read src/leapflow/engine/engine.py")

            tools = llm.kwargs.get("tools", [])
            names = {tool.get("function", {}).get("name", "") for tool in tools}
            assert "file_read" in names
            assert "file_list" in names
            assert "shell_run" not in names
            assert engine.context_budget_snapshot["disclosure_level"] == "selected_tools"
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


# ═══════════════════════════════════════════════════════════════════
# Idempotency guard and failure recovery tests
# ═══════════════════════════════════════════════════════════════════


def test_platform_action_fingerprint_deduplicates_identical_calls() -> None:
    """_platform_action_fingerprint returns the same key for identical calls."""
    from leapflow.engine.engine import _platform_action_fingerprint

    fp1 = _platform_action_fingerprint(
        "platform_action",
        {"platform": "feishu", "action": "im.send_message", "payload": {"chat_id": "oc_1", "text": "hi"}},
    )
    fp2 = _platform_action_fingerprint(
        "platform_action",
        {"platform": "feishu", "action": "im.send_message", "payload": {"chat_id": "oc_1", "text": "hi"}},
    )
    fp_different = _platform_action_fingerprint(
        "platform_action",
        {"platform": "feishu", "action": "im.send_message", "payload": {"chat_id": "oc_2", "text": "hi"}},
    )
    fp_read = _platform_action_fingerprint(
        "platform_action",
        {"platform": "feishu", "action": "im.search_chats", "payload": {"query": "LeapFlow"}},
    )
    fp_other_tool = _platform_action_fingerprint(
        "file_list",
        {"path": "."},
    )

    assert fp1 is not None
    assert fp1 == fp2, "Identical calls must produce the same fingerprint"
    assert fp1 != fp_different, "Different payload must produce a different fingerprint"
    assert fp_read is not None, "Read platform_actions are also fingerprinted (dedup applies)"
    assert fp_other_tool is None, "Non-platform_action tools must return None"


def test_last_tool_failures_recovery_message_from_unknown_action() -> None:
    """_last_tool_failures_recovery_message extracts context from unknown_platform_action results."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    failure_payload = {
        "ok": False,
        "failure_code": "unknown_platform_action",
        "error": "Unknown platform action: feishu.im.chat.list",
        "platform": "feishu",
        "requested_action": "im.chat.list",
        "available_action_names": ["im.send_message", "im.list_chats", "im.search_chats"],
        "recovery_hint": "Use exactly one registered action name from available_action_names.",
        "retryable": True,
    }
    messages = [
        {"role": "tool", "content": json.dumps(failure_payload)},
    ]
    result = _last_tool_failures_recovery_message(messages)

    assert result, "Should produce non-empty recovery message"
    assert "feishu.im.chat.list" in result or "im.chat.list" in result
    assert "im.list_chats" in result
    assert "im.send_message" in result


def test_last_tool_failures_recovery_message_missing_fields() -> None:
    """_last_tool_failures_recovery_message handles Missing required fields errors."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    failure_payload = {
        "ok": False,
        "error": "Missing required fields: text",
    }
    messages = [{"role": "tool", "content": json.dumps(failure_payload)}]
    result = _last_tool_failures_recovery_message(messages)

    assert result
    assert "text" in result


def test_last_tool_failures_recovery_message_returns_empty_when_no_failures() -> None:
    """Returns empty string when there are no failed tool results in history."""
    import json
    from leapflow.engine.engine import _last_tool_failures_recovery_message

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "sure"},
        {"role": "tool", "content": json.dumps({"ok": True, "data": {}})},
    ]
    assert _last_tool_failures_recovery_message(messages) == ""

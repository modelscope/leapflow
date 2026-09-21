# Copyright (c) Alibaba, Inc. and its affiliates.
"""Intent routing and progressive context disclosure tests.

Extracted from test_agent_execution.py — tests that exercise how different
intent labels and prior-turn tool-category continuity shape the PromptAssemblyPlan
(tool schema selection, disclosure level, system prompt sections).
"""

from __future__ import annotations

import tempfile

import pytest

from _fixtures.agent_execution import (
    _FixedClassifier,
    _activate_desktop_plugin,
    _build_desktop_engine,
    _deactivate_desktop_plugin,
)
from conftest import StubLLM, make_settings
from leapflow.engine._tool_helpers import build_default_registry
from leapflow.engine.engine import AgentEngine
from leapflow.memory import (
    EpisodicMemoryProvider,
    SemanticMemoryProvider,
    WorkingMemoryProvider,
)


# ═══════════════════════════════════════════════════════════════════
# Progressive disclosure tests
# ═══════════════════════════════════════════════════════════════════


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
            # CORE disclosure keeps a static low-risk tool whitelist always callable
            # (never an empty/contradictory tool contract), but excludes heavy/mutating tools.
            core_names = {
                tool.get("function", {}).get("name", "")
                for tool in llm.kwargs.get("tools", [])
            }
            assert "shell_run" not in core_names
            assert "hub_push" not in core_names
            assert llm.enable_thinking is False
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
            assert "LeapFlow workspace config is optional" in system_prompt
            assert "~/.leapflow/config/user.yaml" in system_prompt
            assert "~/.leapflow/profiles/<profile>/config/*.yaml" in system_prompt
            assert "<workspace>/.leapflow/config.yaml" in system_prompt
            snapshot = engine.context_budget_snapshot
            assert snapshot["disclosure_level"] == "core"
            assert snapshot["disclosure"]["native_tools"] is True
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
            engine._prompt_assembler._begin_turn_context("first request")
            stale_contract = engine._prompt_assembler._task_contract_block()
            engine._session_turn_count = 2
            engine._prompt_assembler._begin_turn_context("second request")

            prepared = engine._prompt_assembler._ensure_task_contract_message([
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
            # file_read/file_list are part of the static Tier 0.5 core whitelist, so a
            # plain file-oriented turn (no prior-turn tool-category continuity, no
            # slash command / escalation signal) stays at the CORE floor level.
            assert engine.context_budget_snapshot["disclosure_level"] == "core"
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_progressive_disclosure_expands_write_category_after_prior_turn_tool_use() -> None:
    """Tier 1 continuity: a native tool_call executed in turn N structurally
    opens its capability category for turn N+1 — a purely structural signal,
    never a re-reading of user text. Regression guard for the dedicated
    ``AgentEngine._last_turn_tool_categories`` state: working memory only
    stores a synthetic "[Called: ...]" summary with no structured tool_calls,
    so continuity must not be derived from ``wm.as_chat_messages()``.
    """
    from leapflow.llm.base import LLMChatResponse, LLMProvider, ToolCallInfo
    from leapflow.platform.mock import MockBridge

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMChatResponse(
                    content="",
                    tool_calls=[
                        ToolCallInfo(
                            id="tc1",
                            name="text_replace",
                            arguments={"text": "a", "old": "a", "new": "b"},
                        )
                    ],
                )
            return LLMChatResponse(content="Turn done")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        settings = settings.__class__(
            **{**settings.__dict__, "native_tool_calling_enabled": True}
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

            await engine.run("Replace a with b in some text")
            first_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[0].get("tools", [])
            }
            # text_replace is not in the static core whitelist and nothing opened
            # its category yet, so the model's own tools schema does not include it
            # (the mock LLM here bypasses that constraint only to exercise the
            # engine's post-execution bookkeeping, not provider-side enforcement).
            assert "text_replace" not in first_turn_names

            await engine.run("hi again")
            second_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[-1].get("tools", [])
            }
            assert "text_replace" in second_turn_names
            assert "file_write" in second_turn_names  # same "write" category opened
            assert "memory_add" in second_turn_names
            assert engine.context_budget_snapshot["disclosure_level"] == "expanded"
            assert "write" in engine.context_budget_snapshot["disclosure"]["expanded_categories"]

            # A third turn with no tool use must not carry the category forever —
            # continuity is exactly one turn, not a sticky escalation.
            await engine.run("just chatting, no tools needed")
            third_turn_names = {
                t.get("function", {}).get("name") for t in llm.calls[-1].get("tools", [])
            }
            assert "text_replace" not in third_turn_names
            assert engine.context_budget_snapshot["disclosure_level"] == "core"
        finally:
            lt.close()


def test_record_tool_call_categories_caches_capability_manifests(monkeypatch) -> None:
    """Capability manifests are cached instead of rebuilt on every tool-call round."""
    from types import SimpleNamespace

    import leapflow.engine.prompt_assembler as assembler_module

    calls = 0
    real_build = assembler_module.build_capability_manifests

    def counting_build(tool_definitions):
        nonlocal calls
        calls += 1
        return real_build(tool_definitions)

    monkeypatch.setattr(assembler_module, "build_capability_manifests", counting_build)

    with tempfile.TemporaryDirectory() as td:
        settings = make_settings(td)
        from leapflow.platform.mock import MockBridge

        rpc = MockBridge()
        llm = StubLLM(["ok"])
        wm = WorkingMemoryProvider(max_tokens=1024)
        lt = SemanticMemoryProvider(source=settings.duckdb_path)
        imm = EpisodicMemoryProvider()
        try:
            reg = build_default_registry(rpc, llm, wm, lt)
            engine = AgentEngine(
                settings, rpc, llm, wm, lt, imm, reg, _FixedClassifier("chat"),
            )

            engine._prompt_assembler._record_tool_call_categories([SimpleNamespace(name="shell_run")])
            engine._prompt_assembler._record_tool_call_categories([SimpleNamespace(name="shell_run")])

            assert calls == 1
            assert engine._last_turn_tool_categories == frozenset({"shell"})
        finally:
            lt.close()


# ═══════════════════════════════════════════════════════════════════
# Desktop semantic tool disclosure tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_core_turn_hides_desktop_schemas_but_lists_them_in_index(monkeypatch) -> None:
    """CORE keeps desktop out of the native tools kwarg while the index names them."""
    from leapflow.llm.base import LLMChatResponse, LLMProvider

    class CaptureLLM(LLMProvider):
        def __init__(self) -> None:
            self.messages: list[dict] = []
            self.kwargs: dict = {}

        async def achat(self, messages, *, stream=True, enable_thinking=False, on_chunk=None, **kwargs):
            self.messages = list(messages)
            self.kwargs = dict(kwargs)
            return LLMChatResponse(content="hello")

        async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
            if False:
                yield ""

    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            llm = CaptureLLM()
            engine, lt = _build_desktop_engine(td, llm=llm)
            try:
                await engine.run("hello")
                native_names = {
                    tool.get("function", {}).get("name", "")
                    for tool in llm.kwargs.get("tools", [])
                }
                assert "click" not in native_names
                assert "observe_ui" not in native_names
                system_prompt = str(llm.messages[0].get("content", ""))
                assert "click" in system_prompt
                assert "capability_expand category: desktop" in system_prompt
            finally:
                lt.close()
    finally:
        _deactivate_desktop_plugin()


def test_expanded_disclosure_tier_positively_includes_desktop_schemas(monkeypatch) -> None:
    """Tier-1 continuity expands native tools with the desktop semantic schemas.

    Positive counterpart of test_core_turn_hides_desktop_schemas_but_lists_them_in_index:
    once the prior turn actually used desktop tools (structural category fact),
    the EXPANDED disclosure plan must carry the semantic schemas in its native
    tool_definitions, not just name the category in the catalog index.
    """
    from leapflow.engine.context.context_disclosure import (
        DisclosureLevel,
        DisclosurePlanner,
        DisclosureRuntimeState,
    )

    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                plan = DisclosurePlanner().plan(
                    engine._tool_dispatch._unified_tool_catalog(),
                    DisclosureRuntimeState(
                        native_tools_enabled=True,
                        last_turn_tool_categories=frozenset({"desktop"}),
                    ),
                )
                assert plan.level == DisclosureLevel.EXPANDED
                assert plan.native_tools is True
                plan_names = {
                    tool["function"]["name"] for tool in plan.tool_definitions
                }
                assert {"click", "observe_ui"} <= plan_names
            finally:
                lt.close()
    finally:
        _deactivate_desktop_plugin()

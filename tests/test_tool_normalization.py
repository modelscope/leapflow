# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool name normalization, catalog management, and unknown tool handling tests.

Extracted from test_agent_execution.py — tests exercising tool name resolution
(alias tables, case/separator formatting), unknown-tool feedback, message
healer, unified catalog merging, and semantic desktop tool plugin wiring.
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
from leapflow.engine._message_helpers import _tool_args_metadata
from leapflow.engine._tool_helpers import (
    _normalize_tool_name,
    _resolve_tool_name,
    build_default_registry,
)
from leapflow.engine.engine import AgentEngine
from leapflow.memory import (
    EpisodicMemoryProvider,
    SemanticMemoryProvider,
    WorkingMemoryProvider,
)


# ═══════════════════════════════════════════════════════════════════
# Tool name normalization and alias resolution
# ═══════════════════════════════════════════════════════════════════


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

            result = await engine._tool_dispatch._execute_general_tool(
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


# ═══════════════════════════════════════════════════════════════════
# Message healer
# ═══════════════════════════════════════════════════════════════════


def test_message_healer_synthesizes_missing_tool_results() -> None:
    """An assistant tool_calls message missing a response is repaired, not sent broken.

    This is the boundary guard for the provider contract that produced the
    observed HTTP 400 ("insufficient tool messages following tool_calls
    message"): every tool_call_id must be followed by a role=tool message,
    whatever upstream path (batch stop, cancellation, compression) dropped it.
    """
    import json as _json

    from leapflow.engine.message_healer import MessageHealer

    healer = MessageHealer()
    messages = [
        {"role": "user", "content": "do two things"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "platform_action", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "file_list", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": '{"ok": false}'},
        # call_b has no response -> the provider would reject the whole request.
    ]

    healed = healer.heal(messages)

    # Both calls now have contiguous responses, in emission order.
    tool_ids = [m["tool_call_id"] for m in healed if m.get("role") == "tool"]
    assert tool_ids == ["call_a", "call_b"]
    synth = next(m for m in healed if m.get("tool_call_id") == "call_b")
    payload = _json.loads(synth["content"])
    assert payload["execution_skipped"] is True
    assert payload["counts_as_failure"] is False
    # A well-formed history is left untouched (idempotent, no duplicate results).
    assert healer.heal(healed) == healed


# ═══════════════════════════════════════════════════════════════════
# Unknown tool feedback and self-healing
# ═══════════════════════════════════════════════════════════════════


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

            result = await engine._tool_dispatch._execute_general_tool(
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


# ═══════════════════════════════════════════════════════════════════
# Streaming tool resolution
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# Semantic desktop tool injection (perception online)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_unified_catalog_merges_semantic_tools_when_plugin_active(monkeypatch) -> None:
    """Catalog and handler table gain the plugin's semantic tools; static registry untouched."""
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()
    TOOL_DEFINITIONS = _tool_reg.tool_definitions

    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                catalog_names = {
                    item.get("function", {}).get("name")
                    for item in engine._tool_dispatch._unified_tool_catalog()
                }
                assert {"observe_ui", "click"} <= catalog_names
                handlers = engine._tool_dispatch._unified_tool_handlers()
                assert "observe_ui" in handlers and "click" in handlers
                static_names = {
                    item.get("function", {}).get("name") for item in TOOL_DEFINITIONS
                }
                assert "click" not in static_names
            finally:
                lt.close()
    finally:
        _deactivate_desktop_plugin()


@pytest.mark.asyncio
async def test_unified_catalog_rebuilds_when_static_registry_grows(monkeypatch) -> None:
    """Tools appended after engine construction (session_search pattern) are picked up."""
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()
    TOOL_DEFINITIONS = _tool_reg.tool_definitions

    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                assert engine._tool_dispatch._unified_tool_catalog()  # prime the cache
                TOOL_DEFINITIONS.append(
                    {
                        "type": "function",
                        "function": {
                            "name": "late_registered_probe",
                            "description": "probe",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                )
                try:
                    names = {
                        item.get("function", {}).get("name")
                        for item in engine._tool_dispatch._unified_tool_catalog()
                    }
                    assert "late_registered_probe" in names
                finally:
                    TOOL_DEFINITIONS.pop()
            finally:
                lt.close()
    finally:
        _deactivate_desktop_plugin()


@pytest.mark.asyncio
async def test_semantic_execution_gate_and_perception_offline(monkeypatch) -> None:
    """Observation runs ungated; mutating tools fail closed without approval;
    offline the tool is unavailable rather than unknown."""
    import types

    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    calls = _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                handlers = engine._tool_dispatch._unified_tool_handlers()

                observed = await engine._tool_dispatch._execute_general_tool(
                    {"name": "observe_ui", "arguments": {"app": "Safari"}}, handlers
                )
                assert observed.get("ok") is True
                assert calls == [("observe_ui", {"app": "Safari"})]

                _tool_reg.set_desktop_gate(None)
                denied = await engine._tool_dispatch._execute_general_tool(
                    {"name": "click", "arguments": {"selector": "#go"}}, handlers
                )
                assert denied.get("ok") is False
                assert "blocked" in denied["error"] or "approval" in denied["error"]
                assert len(calls) == 1  # never executed

                class _Approve:
                    async def evaluate(self, action):
                        return types.SimpleNamespace(approved=True, denial_message="")

                _tool_reg.set_desktop_gate(_Approve())
                clicked = await engine._tool_dispatch._execute_general_tool(
                    {"name": "click", "arguments": {"selector": "#go"}}, handlers
                )
                assert clicked.get("ok") is True
                assert calls[-1] == ("click", {"selector": "#go"})
            finally:
                _tool_reg.set_desktop_gate(None)
                lt.close()
    finally:
        _deactivate_desktop_plugin()

    # Perception offline: no plugin handlers -> explicit unavailability.
    with tempfile.TemporaryDirectory() as td:
        engine, lt = _build_desktop_engine(td)
        try:
            result = await engine._tool_dispatch._execute_general_tool(
                {"name": "click", "arguments": {"selector": "#go"}},
                engine._tool_dispatch._unified_tool_handlers(),
            )
            assert result.get("ok") is False
            assert "unavailable" in result["error"]
        finally:
            lt.close()


@pytest.mark.asyncio
async def test_reconfigure_host_backend_drops_semantic_tools(monkeypatch) -> None:
    """Hot-swapping to a host without perception removes desktop from the catalog.

    Mirrors the production reconfigure sequence: the desktop plugin is
    unbound first (bind_runtime with None ports), then the engine refreshes
    its host backend — the unified catalog follows the plugin offline.
    """
    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                assert any(
                    item.get("function", {}).get("name") == "click"
                    for item in engine._tool_dispatch._unified_tool_catalog()
                )
                _deactivate_desktop_plugin()
                engine.reconfigure_host_backend(
                    rpc=engine._rpc, perception=None, execution=None,
                )
                names = {
                    item.get("function", {}).get("name")
                    for item in engine._tool_dispatch._unified_tool_catalog()
                }
                assert "click" not in names
                assert "observe_ui" not in engine._tool_dispatch._unified_tool_handlers()
            finally:
                lt.close()
    finally:
        _deactivate_desktop_plugin()


def test_disable_desktop_semantic_drops_engine_surfaces(monkeypatch) -> None:
    """plugin_disable("desktop_semantic") removes engine surfaces immediately.

    Reproduces the reviewed defect through the real disable path (scoped-registry
    fiber dispose — exactly what self_management's plugin_disable handler runs
    after approval): the engine must stop disclosing semantic tools on the very
    next read, including the zero-approval observation tools, instead of serving
    the stale cached schemas/handlers of the captured plugin instance. A
    subsequent reload must surface a FRESH plugin instance whose version counter
    restarted at 0 — the identity component of the engine cache keys is what
    prevents that collision.
    """
    from leapflow.skills.semantic_schema import SEMANTIC_TOOL_NAMES
    from leapflow.plugins import get_registry, get_scoped_registry

    _activate_desktop_plugin(monkeypatch)
    try:
        with tempfile.TemporaryDirectory() as td:
            engine, lt = _build_desktop_engine(td)
            try:
                # Plugin active: semantic tools disclosed and dispatchable.
                catalog_names = {
                    item.get("function", {}).get("name")
                    for item in engine._tool_dispatch._unified_tool_catalog()
                }
                assert {"click", "observe_ui"} <= catalog_names
                assert "observe_ui" in engine._tool_dispatch._unified_tool_handlers()
                old_plugin = get_registry().get_desktop_semantic_plugin()
                assert old_plugin is not None

                # Approved disable: the scoped-registry fiber dispose that the
                # plugin_disable handler executes after its approval gate.
                scoped = get_scoped_registry()
                fiber = scoped.get_fiber("desktop_semantic")
                assert fiber is not None and fiber.state.value == "active"
                fiber.begin_unload()
                fiber.dispose()

                # Engine surfaces drop every semantic tool on the next read —
                # no stale cache entries survive the unregister.
                assert get_registry().get_desktop_semantic_plugin() is None
                post_disable_names = {
                    item.get("function", {}).get("name")
                    for item in engine._tool_dispatch._unified_tool_catalog()
                }
                assert post_disable_names.isdisjoint(SEMANTIC_TOOL_NAMES)
                assert set(engine._tool_dispatch._unified_tool_handlers()).isdisjoint(SEMANTIC_TOOL_NAMES)
                assert engine._tool_dispatch._semantic_tool_schemas() == []

                # Reload: a fresh instance (version restarting at 0) becomes
                # visible again. "screenshot" is only present in the real
                # entry set, so serving it proves the cache picked up the new
                # instance rather than the predecessor's cached schemas.
                scoped.reload("desktop_semantic")
                fresh = get_registry().get_desktop_semantic_plugin()
                assert fresh is not None and fresh is not old_plugin
                assert fresh.active  # last_bound_deps re-injected the ports
                reloaded_names = {
                    item.get("function", {}).get("name")
                    for item in engine._tool_dispatch._unified_tool_catalog()
                }
                assert {"click", "observe_ui", "screenshot"} <= reloaded_names
                assert "observe_ui" in engine._tool_dispatch._unified_tool_handlers()
            finally:
                # Leave the global plugin deactivated for subsequent tests.
                _deactivate_desktop_plugin()
                lt.close()
    finally:
        _deactivate_desktop_plugin()

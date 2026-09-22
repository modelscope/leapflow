# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for ToolDispatchEngine — tool execution, catalog, guardrails."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List


from leapflow.engine.tool_dispatch_engine import ToolDispatchEngine


# ── Minimal engine stub ──────────────────────────────────────────────


def _stub_engine(
    *,
    guardrail: Any = None,
    active_frame: Any = None,
    last_context_snapshot: Dict[str, Any] | None = None,
    context_governance: Any = None,
) -> SimpleNamespace:
    """Build a minimal engine-like object for ToolDispatchEngine."""
    if context_governance is None:
        context_governance = SimpleNamespace(
            compact_tool_result=lambda name, args, result: result,
            tool_metadata=lambda name, args, result: {},
        )
    engine = SimpleNamespace(
        _guardrail=guardrail,
        _active_frame=active_frame or SimpleNamespace(stalled_rounds=0),
        _last_context_snapshot=last_context_snapshot or {},
        _context_governance_controller=context_governance,
        _learning_bridge=SimpleNamespace(
            _tool_focus_metadata=lambda name, args, result: {},
        ),
    )
    return engine


# ── _format_tool_catalog ─────────────────────────────────────────────


class TestFormatToolCatalog:
    def test_empty_catalog(self) -> None:
        result = ToolDispatchEngine._format_tool_catalog([])
        assert result == ""

    def test_single_tool(self) -> None:
        defs = [
            {
                "function": {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "properties": {"command": {"type": "string"}},
                    },
                }
            }
        ]
        result = ToolDispatchEngine._format_tool_catalog(defs)
        assert "**shell**" in result
        assert "command" in result
        assert "Run a shell command" in result

    def test_multiple_tools(self) -> None:
        defs = [
            {
                "function": {
                    "name": "shell",
                    "description": "Run shell",
                    "parameters": {"properties": {"command": {}}},
                }
            },
            {
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"properties": {"path": {}}},
                }
            },
        ]
        result = ToolDispatchEngine._format_tool_catalog(defs)
        assert "**shell**" in result
        assert "**read_file**" in result
        lines = result.strip().split("\n")
        assert len(lines) == 2


# ── _check_guardrail ────────────────────────────────────────────────


class TestCheckGuardrail:
    def test_no_guardrail_returns_none(self) -> None:
        engine = _stub_engine(guardrail=None)
        dispatch = ToolDispatchEngine(engine)
        result = dispatch._check_guardrail([])
        assert result is None

    def test_no_violation_returns_none(self) -> None:
        guardrail = SimpleNamespace(
            check=lambda msgs: SimpleNamespace(violated=False, reason="", suggestion=""),
        )
        engine = _stub_engine(guardrail=guardrail)
        dispatch = ToolDispatchEngine(engine)
        result = dispatch._check_guardrail([])
        assert result is None

    def test_halt_violation_stalled(self) -> None:
        guardrail = SimpleNamespace(
            check=lambda msgs: SimpleNamespace(
                violated=True,
                severity="halt",
                reason="repetition detected",
                suggestion="try a different approach",
                progress_independent=False,
            ),
        )
        frame = SimpleNamespace(stalled_rounds=2)
        engine = _stub_engine(guardrail=guardrail, active_frame=frame)
        dispatch = ToolDispatchEngine(engine)
        messages: List[Dict[str, Any]] = []
        result = dispatch._check_guardrail(messages)
        assert result == "halt"
        assert len(messages) == 1
        assert "GUARDRAIL" in messages[0]["content"]

    def test_halt_violation_not_stalled_returns_none(self) -> None:
        guardrail = SimpleNamespace(
            check=lambda msgs: SimpleNamespace(
                violated=True,
                severity="halt",
                reason="repetition detected",
                suggestion="try different",
                progress_independent=False,
            ),
        )
        frame = SimpleNamespace(stalled_rounds=0)
        engine = _stub_engine(guardrail=guardrail, active_frame=frame)
        dispatch = ToolDispatchEngine(engine)
        messages: List[Dict[str, Any]] = []
        result = dispatch._check_guardrail(messages)
        assert result is None

    def test_progress_independent_halt_ignores_stall(self) -> None:
        guardrail = SimpleNamespace(
            check=lambda msgs: SimpleNamespace(
                violated=True,
                severity="halt",
                reason="no-op loop",
                suggestion="stop",
                progress_independent=True,
            ),
        )
        frame = SimpleNamespace(stalled_rounds=0)
        engine = _stub_engine(guardrail=guardrail, active_frame=frame)
        dispatch = ToolDispatchEngine(engine)
        messages: List[Dict[str, Any]] = []
        result = dispatch._check_guardrail(messages)
        assert result == "halt"

    def test_warning_violation_stalled_appends_message(self) -> None:
        guardrail = SimpleNamespace(
            check=lambda msgs: SimpleNamespace(
                violated=True,
                severity="warning",
                reason="repetitive calls",
                suggestion="diversify",
                progress_independent=False,
            ),
        )
        frame = SimpleNamespace(stalled_rounds=2)
        engine = _stub_engine(guardrail=guardrail, active_frame=frame)
        dispatch = ToolDispatchEngine(engine)
        messages: List[Dict[str, Any]] = []
        result = dispatch._check_guardrail(messages)
        assert result is None
        assert len(messages) == 1
        assert "WARNING" in messages[0]["content"]


# ── _compact_tool_result ─────────────────────────────────────────────


class TestCompactToolResult:
    def test_delegates_to_governance(self) -> None:
        compacted = {"ok": True, "summary": "done"}
        governance = SimpleNamespace(
            compact_tool_result=lambda name, args, result: compacted,
            tool_metadata=lambda name, args, result: {},
        )
        engine = _stub_engine(context_governance=governance)
        dispatch = ToolDispatchEngine(engine)
        result = dispatch._compact_tool_result("shell", {"command": "ls"}, {"ok": True, "output": "file1"})
        assert result == compacted


# ── _tool_context_metadata ───────────────────────────────────────────


class TestToolContextMetadata:
    def test_includes_posture_when_non_baseline(self) -> None:
        engine = _stub_engine(
            last_context_snapshot={
                "context_posture": "exploring",
                "context_signal": "large_codebase",
                "context_guidance": "be thorough",
                "disclosure_level": "full",
                "disclosure_reason": "high complexity",
            },
        )
        dispatch = ToolDispatchEngine(engine)
        meta = dispatch._tool_context_metadata("test", {}, {"ok": True})
        assert meta.get("context_posture") == "exploring"
        assert meta.get("context_signal") == "large_codebase"

    def test_empty_snapshot_returns_empty_metadata(self) -> None:
        engine = _stub_engine(last_context_snapshot={})
        dispatch = ToolDispatchEngine(engine)
        meta = dispatch._tool_context_metadata("test", {}, {"ok": True})
        assert "context_posture" not in meta

    def test_forced_final_answer_sets_finalizing(self) -> None:
        engine = _stub_engine(
            last_context_snapshot={"forced_final_answer": True},
        )
        dispatch = ToolDispatchEngine(engine)
        meta = dispatch._tool_context_metadata("test", {}, {"ok": True})
        assert meta.get("context_posture") == "finalizing"


# ── _tool_execution_metadata (static) ────────────────────────────────


class TestToolExecutionMetadata:
    def test_extracts_known_keys(self) -> None:
        result = {
            "execution_id": "exec-1",
            "idempotency_key": "key-1",
            "execution_status": "completed",
            "execution_policy": "read_only",
            "path": "/tmp/file.txt",
            "side_effect_uncertain": False,
            "random_key": "should_not_appear",
        }
        meta = ToolDispatchEngine._tool_execution_metadata(result)
        assert meta["execution_id"] == "exec-1"
        assert meta["execution_policy"] == "read_only"
        assert meta["path"] == "/tmp/file.txt"
        assert "random_key" not in meta

    def test_non_dict_returns_empty(self) -> None:
        assert ToolDispatchEngine._tool_execution_metadata("string") == {}

    def test_empty_dict_returns_empty(self) -> None:
        assert ToolDispatchEngine._tool_execution_metadata({}) == {}


# ── _count_consecutive_tool_failures ─────────────────────────────────


class TestCountConsecutiveToolFailures:
    def test_no_messages(self) -> None:
        assert ToolDispatchEngine._count_consecutive_tool_failures([]) == 0

    def test_all_successes(self) -> None:
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "tool", "content": json.dumps({"ok": True})},
        ]
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 0

    def test_consecutive_failures(self) -> None:
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "tool", "content": json.dumps({"ok": False})},
            {"role": "assistant", "content": "retrying..."},
            {"role": "tool", "content": json.dumps({"ok": False})},
        ]
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 2

    def test_success_resets_count(self) -> None:
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "tool", "content": json.dumps({"ok": False})},
            {"role": "tool", "content": json.dumps({"ok": True})},
            {"role": "tool", "content": json.dumps({"ok": False})},
        ]
        # Scanning backwards: fail(count=1), success → immediate return 0
        # A success anywhere in the chain means the agent isn't stuck.
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 0

    def test_stops_at_user_boundary(self) -> None:
        messages = [
            {"role": "user", "content": "old task"},
            {"role": "tool", "content": json.dumps({"ok": False})},
            {"role": "user", "content": "new task"},
            {"role": "tool", "content": json.dumps({"ok": False})},
        ]
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 1

    def test_skips_control_signals(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            {"role": "tool", "content": json.dumps({"ok": False})},
            {"role": "tool", "content": json.dumps(
                {"ok": True, "already_executed": True, "counts_as_failure": False}
            )},
            {"role": "tool", "content": json.dumps({"ok": False})},
        ]
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 2

    def test_non_json_content_resets(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            {"role": "tool", "content": json.dumps({"ok": False})},
            {"role": "tool", "content": "plain text result"},
        ]
        # Scan backwards: "plain text result" → non-JSON → treat as success → reset
        assert ToolDispatchEngine._count_consecutive_tool_failures(messages) == 0


# ── _merge_expanded_tool_schemas ─────────────────────────────────────


class TestMergeExpandedToolSchemas:
    def test_no_expansions_unchanged(self) -> None:
        tools_kwarg = {"tools": [{"function": {"name": "shell"}}]}
        result = ToolDispatchEngine._merge_expanded_tool_schemas(tools_kwarg, [])
        assert result == tools_kwarg

    def test_adds_new_tool(self) -> None:
        existing = {"tools": [{"function": {"name": "shell"}}]}
        results = [
            {
                "result": {
                    "ok": True,
                    "expanded_tools": [{"function": {"name": "new_tool"}}],
                }
            }
        ]
        merged = ToolDispatchEngine._merge_expanded_tool_schemas(existing, results)
        names = {td.get("function", {}).get("name") for td in merged["tools"]}
        assert "shell" in names
        assert "new_tool" in names

    def test_does_not_duplicate_existing(self) -> None:
        existing = {"tools": [{"function": {"name": "shell"}}]}
        results = [
            {
                "result": {
                    "ok": True,
                    "expanded_tools": [{"function": {"name": "shell"}}],
                }
            }
        ]
        merged = ToolDispatchEngine._merge_expanded_tool_schemas(existing, results)
        assert len(merged["tools"]) == 1

    def test_skips_failed_expand_results(self) -> None:
        existing = {"tools": [{"function": {"name": "shell"}}]}
        results = [
            {"result": {"ok": False, "error": "failed"}},
        ]
        merged = ToolDispatchEngine._merge_expanded_tool_schemas(existing, results)
        assert len(merged["tools"]) == 1


# ── _expand_tools_kwarg_full ─────────────────────────────────────────


class TestExpandToolsKwargFull:
    def test_replaces_with_full_catalog(self) -> None:
        tools_kwarg = {"tools": [{"function": {"name": "shell"}}]}
        full_defs = [
            {"function": {"name": "shell"}},
            {"function": {"name": "read_file"}},
            {"function": {"name": "write_file"}},
        ]
        result = ToolDispatchEngine._expand_tools_kwarg_full(tools_kwarg, full_defs)
        assert len(result["tools"]) == 3


# ── _post_process_tool_result ────────────────────────────────────────


class TestPostProcessToolResult:
    def test_non_dict_passthrough(self) -> None:
        result = ToolDispatchEngine._post_process_tool_result("test", "string result")
        assert result == "string result"

    def test_dict_result_passes_through(self) -> None:
        payload = {"ok": True, "result": "data"}
        result = ToolDispatchEngine._post_process_tool_result("test_tool", payload)
        assert result["ok"] is True

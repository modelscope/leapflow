# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for P0 tool-calling / result hardening in the agent loop:

- A1: pre-execution required-argument validation (_validate_tool_arguments)
- B2: structure-aware tool-result truncation (_truncate_result_for_budget)

Both are pure module functions, exercised directly.
"""
from __future__ import annotations

import asyncio
import json
import os

from leapflow.engine._message_helpers import (
    _head_tail_truncate,
    _truncate_result_for_budget,
    _validate_tool_arguments,
)
from leapflow.tools.name_resolver import ToolSpec


def _spec(name: str, required: set[str], params: set[str]) -> ToolSpec:
    return ToolSpec(name=name, parameters=frozenset(params), required=frozenset(required))


# ── A1: argument validation ──────────────────────────────────────────

def test_validate_missing_required_returns_structured_error() -> None:
    spec = _spec("edit_file", {"path"}, {"path", "edits", "diff"})
    result = _validate_tool_arguments(spec, {"edits": []})  # path missing
    assert result is not None
    assert result["ok"] is False and result["error_type"] == "invalid_arguments"
    assert result["missing"] == ["path"]
    assert "path" in result["accepted_parameters"]
    # Must not penalize failure budgets or trip the batch-stop gate.
    assert result["counts_as_failure"] is False and result["retryable"] is True
    assert "execution_policy" not in result


def test_validate_present_required_passes() -> None:
    spec = _spec("edit_file", {"path"}, {"path", "edits"})
    assert _validate_tool_arguments(spec, {"path": "/x", "edits": []}) is None


def test_validate_no_required_is_skipped() -> None:
    spec = _spec("terminal_list", set(), set())
    assert _validate_tool_arguments(spec, {}) is None


def test_validate_present_but_empty_is_not_missing() -> None:
    # Presence-only: an empty but present required value is the handler's concern
    # (e.g. text_replace new="" is a valid delete), not a validation rejection.
    spec = _spec("text_replace", {"text", "old", "new"}, {"text", "old", "new"})
    assert _validate_tool_arguments(spec, {"text": "abc", "old": "a", "new": ""}) is None


def test_validate_none_spec_is_skipped() -> None:
    assert _validate_tool_arguments(None, {"anything": 1}) is None


# ── B2: structure-aware truncation ───────────────────────────────────

def test_truncate_within_budget_is_unchanged() -> None:
    payload = {"ok": True, "stdout": "small output"}
    text = _truncate_result_for_budget(payload, 1000)
    assert json.loads(text) == payload


def test_truncate_preserves_head_and_tail_of_large_field() -> None:
    payload = {"ok": False, "stdout": "HEAD_MARKER" + ("x" * 6000) + "TAIL_ERROR"}
    text = _truncate_result_for_budget(payload, 800)
    assert len(text) <= 800
    obj = json.loads(text)  # still valid JSON (shrank the field, not the JSON string)
    assert "TAIL_ERROR" in obj["stdout"]   # the actual error at the tail survives
    assert "HEAD_MARKER" in obj["stdout"]  # head survives
    assert "elided" in obj["stdout"]       # explicit elision marker


def test_truncate_nondict_payload_is_safe() -> None:
    text = _truncate_result_for_budget("a very long string " * 100, 50)
    assert len(text) <= 50


def test_head_tail_truncate_keeps_both_ends() -> None:
    text = "START" + ("m" * 1000) + "END"
    out = _head_tail_truncate(text, 200)
    assert out.startswith("START") and out.endswith("END") and "elided" in out
    assert len(out) < len(text)


def test_head_tail_truncate_short_text_unchanged() -> None:
    assert _head_tail_truncate("short", 200) == "short"


def test_truncate_list_field_pruned_and_annotated() -> None:
    """List fields (e.g. file_list entries) must be pruned, not string-cut."""
    entries = [{"name": f"file_{i}.py", "type": "file", "size": 1000} for i in range(80)]
    payload = {"ok": True, "kind": "file_list_evidence", "path": "/proj", "entries": entries, "entry_count": 80}
    text = _truncate_result_for_budget(payload, 1000)
    obj = json.loads(text)  # must be valid JSON — not a raw string cut
    assert obj["ok"] is True
    assert len(obj["entries"]) < 80           # entries were pruned
    assert "entries_omitted" in obj           # omission count is explicit
    assert obj["entries_omitted"] + len(obj["entries"]) == 80


def test_truncate_over_budget_dict_never_returns_malformed_json() -> None:
    """When every shrinking strategy fails, a valid sentinel dict is returned."""
    # A dict where string truncation alone cannot bring it under budget because
    # the overhead fields themselves exceed the budget.
    payload = {"ok": True, "kind": "file_list_evidence", "entries": [], "entry_count": 0,
               "irreducible": "x" * 100}
    text = _truncate_result_for_budget(payload, 80)  # budget is tiny
    # Must parse as JSON (no raw cut)
    obj = json.loads(text)
    assert isinstance(obj, dict)


def test_compaction_preserves_invalid_argument_repair_hints() -> None:
    from leapflow.engine.context.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    invalid = {
        "ok": False,
        "error": "Invalid arguments for edit_file: missing required parameter(s): path",
        "error_type": "invalid_arguments",
        "missing": ["path"],
        "accepted_parameters": ["diff", "edits", "path"],
        "retryable": True,
        "counts_as_failure": False,
    }
    compact = builder.build("edit_file", {}, invalid)
    assert compact["error_type"] == "invalid_arguments"
    assert compact["missing"] == ["path"]
    assert compact["accepted_parameters"] == ["diff", "edits", "path"]


def test_compaction_preserves_anchor_not_unique_match_count() -> None:
    from leapflow.engine.context.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    result = {"ok": False, "error": "not unique", "error_type": "anchor_not_unique", "match_count": 3}
    compact = builder.build("edit_file", {}, result)
    assert compact["error_type"] == "anchor_not_unique" and compact["match_count"] == 3


# ── failure visibility: stdout/stderr survive compaction; shell_run sets error ──

def test_compact_error_preserves_shell_output() -> None:
    """A failed shell result must keep stderr (the traceback) + returncode so the
    agent can diagnose the cause instead of seeing a bare 'unknown error'."""
    from leapflow.engine.context.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    failed = {
        "ok": False,
        "returncode": 1,
        "stdout": "partial output",
        "stderr": "Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'yfinance'",
        "error": "ModuleNotFoundError: No module named 'yfinance'",
    }
    compact = builder.build("shell_run", {}, failed)
    assert compact["ok"] is False and compact["returncode"] == 1
    assert "ModuleNotFoundError" in compact["stderr"]   # traceback preserved
    assert "yfinance" in compact["error"]


def test_compact_error_preserves_stderr_without_error_field() -> None:
    from leapflow.engine.context.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    result = {"ok": False, "returncode": 2, "stdout": "", "stderr": "boom: the real error"}
    compact = builder.build("shell_run", {}, result)
    assert "boom: the real error" in compact["stderr"] and compact["returncode"] == 2


def test_shell_run_populates_error_on_failure() -> None:
    import os

    from leapflow.tools.shell_tools import shell_run
    # cmd.exe (the Windows shell backend) has no ';' separator; '&&' works on both.
    joiner = "&&" if os.name == "nt" else ";"
    result = asyncio.run(shell_run({"command": f"echo BOOM_ERR 1>&2 {joiner} exit 2"}))
    assert result["ok"] is False and result["returncode"] == 2
    assert "BOOM_ERR" in result["error"] and "BOOM_ERR" in result["stderr"]


def test_shell_run_success_has_no_error_field() -> None:
    from leapflow.tools.shell_tools import shell_run
    result = asyncio.run(shell_run({"command": "echo ok"}))
    assert result["ok"] is True and "error" not in result and "ok" in result["stdout"]


def test_workspace_context_resolves_relative_paths_and_blocks_cross_workspace(tmp_path) -> None:
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
    )
    from leapflow.tools.file_operations import code_search, file_read
    from leapflow.tools.repo_map import repo_map
    from leapflow.tools.shell_tools import shell_run

    workspace = tmp_path / "work"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    (workspace / "alpha.txt").write_text("needle in workspace", encoding="utf-8")
    (other / "secret.txt").write_text("needle outside", encoding="utf-8")

    token = set_tool_context(
        ToolExecutionContext.from_strings(
            workspace_root=str(workspace),
            session_id="sess-work",
            task_id="turn-test",
        )
    )
    try:
        read_result = asyncio.run(file_read({"path": "alpha.txt"}))
        assert read_result["ok"] is True
        assert read_result["path"] == str((workspace / "alpha.txt").resolve())

        search_result = asyncio.run(code_search({"pattern": "needle", "path": "."}))
        assert search_result["ok"] is True
        assert search_result["path"] == str(workspace.resolve())
        assert search_result["match_count"] == 1

        repo_result = asyncio.run(repo_map({"path": "."}))
        assert repo_result["ok"] is True
        assert repo_result["root"] == str(workspace.resolve())

        pwd_command = "echo %CD%" if os.name == "nt" else "pwd"
        shell_result = asyncio.run(shell_run({"command": pwd_command}))
        assert shell_result["ok"] is True, shell_result
        assert shell_result["cwd"] == str(workspace.resolve())

        blocked = asyncio.run(file_read({"path": str(other / "secret.txt")}))
        assert blocked["ok"] is False
        assert blocked["error_type"] == "outside_workspace"
        assert blocked["workspace_root"] == str(workspace.resolve())

        blocked_shell = asyncio.run(shell_run({"command": f"cat {other / 'secret.txt'}"}))
        assert blocked_shell["ok"] is False
        assert blocked_shell["error_type"] == "outside_workspace"
        assert blocked_shell["workspace_root"] == str(workspace.resolve())
    finally:
        reset_tool_context(token)


def test_shell_gate_blocks_expanded_and_relative_escapes(tmp_path, monkeypatch) -> None:
    """The gate must judge the resolved target, not how the operand is spelled.

    A guard that only inspects literal ``/``/``~`` operands passes ``$HOME/...``
    and ``../../..`` straight through, so the same file is reachable by choosing
    a different spelling and the workspace boundary stops being a boundary.
    """
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
    )
    from leapflow.tools.shell_tools import shell_run

    import os

    workspace = tmp_path / "work"
    other = tmp_path / "other"
    workspace.mkdir()
    (workspace / "sub").mkdir()
    other.mkdir()
    (workspace / "alpha.txt").write_text("inside", encoding="utf-8")
    (other / "secret.txt").write_text("outside", encoding="utf-8")
    monkeypatch.setenv("LEAP_TEST_OUTSIDE", str(other))

    token = set_tool_context(
        ToolExecutionContext.from_strings(workspace_root=str(workspace), session_id="sess-gate")
    )
    try:
        for command in (
            "cat $LEAP_TEST_OUTSIDE/secret.txt",
            "cat ${LEAP_TEST_OUTSIDE}/secret.txt",
            "cat ../other/secret.txt",
            "cat --file=$LEAP_TEST_OUTSIDE/secret.txt",
            "cd $LEAP_TEST_OUTSIDE && cat secret.txt",
        ):
            result = asyncio.run(shell_run({"command": command}))
            assert result["ok"] is False, command
            assert result["error_type"] == "outside_workspace", command
            assert result["resolved_path"].startswith(str(other.resolve())), command

        # Traversal that stays inside must still run: the gate judges the resolved
        # target, so `sub/../alpha.txt` is an ordinary in-workspace read, and the
        # content must come back. On Windows no cmd-era builtin reads a '..'
        # operand (Git's MSYS cat falls back to stdin and hangs, cmd's `type`
        # rejects the path), so the read goes through PowerShell's Get-Content —
        # as a command argument; the shell backend stays cmd.
        if os.name == "nt":
            command = 'powershell -NoProfile -Command "Get-Content sub/../alpha.txt"'
        else:
            command = "cat sub/../alpha.txt"
        inside = asyncio.run(shell_run({"command": command}))
        assert inside["ok"] is True
        assert inside["stdout"].strip() == "inside"
    finally:
        reset_tool_context(token)


def test_shell_gate_allows_search_list_variables(tmp_path, monkeypatch) -> None:
    """Expanding a variable must not turn ordinary commands into refusals.

    ``$PATH`` expands to an ``os.pathsep``-joined list that begins with ``/`` but
    names no file. Treating that as a path operand blocked ``echo $PATH`` and
    ``PATH=$PATH:./bin npm test``, which have nothing to do with the workspace
    boundary.
    """
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
    )
    from leapflow.tools.shell_tools import _command_workspace_escape_path

    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setenv("LEAP_TEST_HOME", str(tmp_path / "outside"))

    token = set_tool_context(
        ToolExecutionContext.from_strings(workspace_root=str(workspace), session_id="sess-list")
    )
    try:
        for command in (
            "echo $PATH",
            "export PATH=$PATH:/usr/local/bin && make",
            "PATH=$PATH:./node_modules/.bin npm test",
        ):
            assert _command_workspace_escape_path(command, cwd=workspace) is None, command

        # A single-path variable must still be detected as an escape target.
        blocked = _command_workspace_escape_path("cat $LEAP_TEST_HOME/secret", cwd=workspace)
        assert blocked == (tmp_path / "outside" / "secret").resolve()
    finally:
        reset_tool_context(token)


def test_shell_gate_redirects_leapflow_config_targets(tmp_path) -> None:
    """Refusing a config path must name the capability that serves the goal.

    Without the redirect the model only learns "not here" and moves the same
    probe to another spelling, which is the loop the config tools exist to end.

    Detection and refusal are now separate steps — the shell gate reports the
    escaping path and the shared refusal builder carries the hint — so both are
    asserted here.
    """
    from leapflow.tools.execution_context import (
        ToolExecutionContext,
        reset_tool_context,
        set_tool_context,
        workspace_scope_refusal,
    )
    from leapflow.tools.shell_tools import _command_workspace_escape_path
    from leapflow.config import get_settings

    workspace = tmp_path / "work"
    workspace.mkdir()
    config_path = get_settings().layout.global_config_dir / "user.yaml"

    token = set_tool_context(
        ToolExecutionContext.from_strings(workspace_root=str(workspace), session_id="sess-hint")
    )
    try:
        escaping = _command_workspace_escape_path(f"cat {config_path}", cwd=workspace)
        assert escaping == config_path.resolve()
        error = workspace_scope_refusal(escaping, operation="shell_run command")
    finally:
        reset_tool_context(token)

    assert error is not None
    assert error["error_type"] == "outside_workspace"
    assert "config_get" in error["error"]
    assert "Approval is required" in error["error"]

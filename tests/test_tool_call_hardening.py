"""Tests for P0 tool-calling / result hardening in the agent loop:

- A1: pre-execution required-argument validation (_validate_tool_arguments)
- B2: structure-aware tool-result truncation (_truncate_result_for_budget)

Both are pure module functions, exercised directly.
"""
from __future__ import annotations

import asyncio
import json

from leapflow.engine.engine import (
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
    from leapflow.engine.context_control import ToolEvidenceBuilder
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
    from leapflow.engine.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    result = {"ok": False, "error": "not unique", "error_type": "anchor_not_unique", "match_count": 3}
    compact = builder.build("edit_file", {}, result)
    assert compact["error_type"] == "anchor_not_unique" and compact["match_count"] == 3


# ── failure visibility: stdout/stderr survive compaction; shell_run sets error ──

def test_compact_error_preserves_shell_output() -> None:
    """A failed shell result must keep stderr (the traceback) + returncode so the
    agent can diagnose the cause instead of seeing a bare 'unknown error'."""
    from leapflow.engine.context_control import ToolEvidenceBuilder
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
    from leapflow.engine.context_control import ToolEvidenceBuilder
    builder = ToolEvidenceBuilder()
    result = {"ok": False, "returncode": 2, "stdout": "", "stderr": "boom: the real error"}
    compact = builder.build("shell_run", {}, result)
    assert "boom: the real error" in compact["stderr"] and compact["returncode"] == 2


def test_shell_run_populates_error_on_failure() -> None:
    from leapflow.tools.shell_tools import shell_run
    result = asyncio.run(shell_run({"command": "echo BOOM_ERR 1>&2; exit 2"}))
    assert result["ok"] is False and result["returncode"] == 2
    assert "BOOM_ERR" in result["error"] and "BOOM_ERR" in result["stderr"]


def test_shell_run_success_has_no_error_field() -> None:
    from leapflow.tools.shell_tools import shell_run
    result = asyncio.run(shell_run({"command": "echo ok"}))
    assert result["ok"] is True and "error" not in result and "ok" in result["stdout"]

"""Tests for P0 tool-calling / result hardening in the agent loop:

- A1: pre-execution required-argument validation (_validate_tool_arguments)
- B2: structure-aware tool-result truncation (_truncate_result_for_budget)

Both are pure module functions, exercised directly.
"""
from __future__ import annotations

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


# ── compaction preserves structured repair hints (A1 reaches the model) ──

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

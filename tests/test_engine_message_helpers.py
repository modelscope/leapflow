# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for engine._message_helpers pure functions."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from leapflow.engine._message_helpers import (
    _EMPTY_RESPONSE_DEGRADED_MESSAGE,
    _EMPTY_RESPONSE_RETRY_PROMPT,
    _FORCED_FINALIZE_PROMPT,
    _SIDE_EFFECT_STOP_POLICIES,
    _TASK_CONTRACT_HEADING,
    _annotate_uncertain_effect,
    _build_native_tool_assistant_message,
    _build_permission_recovery_text,
    _estimate_message_tokens,
    _estimate_prompt_tokens,
    _estimate_text_tokens,
    _extract_json_object,
    _head_tail_truncate,
    _is_retryable_unknown_tool_result,
    _keywords_from_query,
    _should_stop_after_tool_result,
    _single_line_preview,
    _skipped_after_failure_result,
    _terminal_failure_text,
    _tool_args_metadata,
    _tool_failure_text,
    _tool_result_counts_as_failure,
    _tool_result_is_control_signal,
    _tool_result_metadata,
    _truncate_result_for_budget,
    _validate_tool_arguments,
)


# ── _single_line_preview ─────────────────────────────────────────────


class TestSingleLinePreview:
    def test_none_returns_empty(self) -> None:
        assert _single_line_preview(None, limit=100) == ""

    def test_short_string_unchanged(self) -> None:
        assert _single_line_preview("hello world", limit=100) == "hello world"

    def test_long_string_truncated(self) -> None:
        text = "a" * 200
        result = _single_line_preview(text, limit=50)
        assert len(result) == 50
        assert result.endswith("…")

    def test_multiline_collapsed(self) -> None:
        result = _single_line_preview("line1\nline2\nline3", limit=100)
        assert "\n" not in result
        assert "line1 line2 line3" == result

    def test_keep_tail_preserves_both_ends(self) -> None:
        text = "A" * 100
        result = _single_line_preview(text, limit=30, keep_tail=True)
        assert "…" in result
        assert len(result) <= 30
        assert result.endswith("A")
        assert result.startswith("A")

    def test_dict_input_serialized(self) -> None:
        result = _single_line_preview({"key": "value"}, limit=100)
        assert "key" in result
        assert "value" in result


# ── _head_tail_truncate ──────────────────────────────────────────────


class TestHeadTailTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _head_tail_truncate("abc", 100) == "abc"

    def test_long_text_preserves_both_ends(self) -> None:
        text = "START" + "x" * 1000 + "END"
        result = _head_tail_truncate(text, 200)
        assert result.startswith("START")
        assert result.endswith("END")
        assert "chars elided" in result


# ── _truncate_result_for_budget ──────────────────────────────────────


class TestTruncateResultForBudget:
    def test_small_payload_unchanged(self) -> None:
        payload = {"ok": True, "result": "hi"}
        text = _truncate_result_for_budget(payload, 5000)
        assert json.loads(text) == payload

    def test_large_list_pruned(self) -> None:
        payload = {"ok": True, "files": [f"file_{i}.txt" for i in range(500)]}
        result = _truncate_result_for_budget(payload, 500)
        parsed = json.loads(result)
        assert len(parsed.get("files", [])) < 500

    def test_non_dict_truncated(self) -> None:
        text = "x" * 2000
        result = _truncate_result_for_budget(text, 200)
        # Non-dict gets JSON-encoded first then hard-cut
        assert len(result) <= 200


# ── _tool_args_metadata / _tool_result_metadata ─────────────────────


class TestToolMetadata:
    def test_args_metadata_basic(self) -> None:
        meta = _tool_args_metadata("shell", {"command": "ls -la"})
        assert meta["tool_name"] == "shell"
        assert "args_summary" in meta
        assert "command" in meta

    def test_args_metadata_with_call_id(self) -> None:
        meta = _tool_args_metadata("read_file", {"path": "/tmp/x"}, tool_call_id="tc-1")
        assert meta["tool_call_id"] == "tc-1"

    def test_result_metadata_ok(self) -> None:
        meta = _tool_result_metadata(
            "shell",
            {"command": "echo hi"},
            {"ok": True, "stdout": "hi"},
        )
        assert meta["ok"] is True
        assert "stdout_preview" in meta

    def test_result_metadata_failed(self) -> None:
        meta = _tool_result_metadata(
            "shell",
            {"command": "false"},
            {"ok": False, "error": "exit code 1", "exit_code": 1},
        )
        assert meta["ok"] is False
        assert meta["exit_code"] == 1

    def test_result_metadata_scalar(self) -> None:
        meta = _tool_result_metadata("test_tool", {}, "plain text result")
        assert "result_preview" in meta


# ── Tool result classification ───────────────────────────────────────


class TestToolResultClassification:
    def test_retryable_unknown_tool(self) -> None:
        assert _is_retryable_unknown_tool_result(
            {"error_type": "unknown_tool", "retryable": True}
        )

    def test_not_retryable_when_not_unknown(self) -> None:
        assert not _is_retryable_unknown_tool_result({"error_type": "timeout", "retryable": True})

    def test_not_retryable_on_non_dict(self) -> None:
        assert not _is_retryable_unknown_tool_result("error string")

    def test_failure_counts_when_ok_false(self) -> None:
        assert _tool_result_counts_as_failure({"ok": False})

    def test_failure_not_counted_when_explicitly_false(self) -> None:
        assert not _tool_result_counts_as_failure({"ok": False, "counts_as_failure": False})

    def test_control_signal_not_failure(self) -> None:
        assert _tool_result_is_control_signal({"already_executed": True})
        assert _tool_result_is_control_signal({"duplicate_suppressed": True})
        assert _tool_result_is_control_signal({"execution_skipped": True})

    def test_not_control_signal(self) -> None:
        assert not _tool_result_is_control_signal({"ok": True})


# ── _tool_failure_text ───────────────────────────────────────────────


class TestToolFailureText:
    def test_extracts_error(self) -> None:
        assert _tool_failure_text({"error": "boom"}) == "boom"

    def test_extracts_stderr(self) -> None:
        assert _tool_failure_text({"stderr": "err"}) == "err"

    def test_fallback(self) -> None:
        assert _tool_failure_text({}) == "unknown error"


# ── _terminal_failure_text ───────────────────────────────────────────


class TestTerminalFailureText:
    def test_with_interaction_request(self) -> None:
        action = SimpleNamespace(label="Fix it", command="/fix", description="")
        interaction = SimpleNamespace(
            title="Permission denied",
            description="Need admin access",
            suggested_actions=[action],
        )
        decision = SimpleNamespace(reason="internal", interaction=interaction)
        result = _terminal_failure_text(decision)
        assert "Permission denied" in result
        assert "Need admin access" in result
        assert "Fix it" in result

    def test_without_interaction_falls_back_to_reason(self) -> None:
        decision = SimpleNamespace(reason="some reason", interaction=None)
        assert _terminal_failure_text(decision) == "some reason"

    def test_missing_both(self) -> None:
        decision = SimpleNamespace(reason="", interaction=None)
        assert _terminal_failure_text(decision) == ""


# ── _should_stop_after_tool_result ───────────────────────────────────


class TestStopAfterToolResult:
    def test_side_effect_failure_stops(self) -> None:
        for policy in _SIDE_EFFECT_STOP_POLICIES:
            payload = {"ok": False, "execution_policy": policy}
            assert _should_stop_after_tool_result("test", payload) is True

    def test_read_only_failure_does_not_stop(self) -> None:
        payload = {"ok": False, "execution_policy": "read_only"}
        assert _should_stop_after_tool_result("test", payload) is False

    def test_success_never_stops(self) -> None:
        payload = {"ok": True, "execution_policy": "external_side_effect"}
        assert _should_stop_after_tool_result("test", payload) is False


# ── _validate_tool_arguments ─────────────────────────────────────────


class TestValidateToolArguments:
    def test_no_spec_returns_none(self) -> None:
        assert _validate_tool_arguments(None, {"x": 1}) is None

    def test_no_required_returns_none(self) -> None:
        spec = SimpleNamespace(required=frozenset(), parameters=frozenset())
        assert _validate_tool_arguments(spec, {}) is None

    def test_missing_required_returns_error(self) -> None:
        spec = SimpleNamespace(
            name="test_tool",
            required=frozenset({"path", "content"}),
            parameters=frozenset({"path", "content", "mode"}),
        )
        result = _validate_tool_arguments(spec, {"path": "/tmp/x"})
        assert result is not None
        assert result["ok"] is False
        assert "content" in result["missing"]
        assert result["retryable"] is True
        assert result["counts_as_failure"] is False

    def test_all_required_present_returns_none(self) -> None:
        spec = SimpleNamespace(
            name="test_tool",
            required=frozenset({"path"}),
            parameters=frozenset({"path", "content"}),
        )
        assert _validate_tool_arguments(spec, {"path": "/tmp"}) is None


# ── _skipped_after_failure_result ────────────────────────────────────


class TestSkippedAfterFailure:
    def test_produces_correct_shape(self) -> None:
        result = _skipped_after_failure_result("shell", {"ok": False, "error": "boom"})
        assert result["ok"] is True
        assert result["execution_skipped"] is True
        assert result["blocked_by_tool"] == "shell"
        assert result["counts_as_failure"] is False
        assert result["ui_hidden"] is True


# ── _build_permission_recovery_text ──────────────────────────────────


class TestBuildPermissionRecoveryText:
    def test_basic_failure(self) -> None:
        text = _build_permission_recovery_text({
            "platform": "feishu",
            "capability": "send_message",
            "missing_scopes": ["im:message"],
        })
        assert "feishu.send_message" in text
        assert "`im:message`" in text
        assert "Do NOT retry" in text

    def test_one_of_scope_relation(self) -> None:
        text = _build_permission_recovery_text({
            "platform": "feishu",
            "capability": "read",
            "missing_scopes": ["scope_a", "scope_b"],
            "scope_relation": "one_of",
        })
        assert "ANY ONE" in text

    def test_admin_required(self) -> None:
        text = _build_permission_recovery_text({
            "platform": "feishu",
            "capability": "admin",
            "recoverability": "admin_required",
        })
        assert "administrator" in text

    def test_console_url_included(self) -> None:
        text = _build_permission_recovery_text({
            "console_url": "https://console.example.com",
        })
        assert "https://console.example.com" in text


# ── _build_native_tool_assistant_message ─────────────────────────────


class TestBuildNativeToolAssistantMessage:
    def test_basic_construction(self) -> None:
        call = SimpleNamespace(
            id="tc-1",
            name="shell",
            arguments={"command": "ls"},
        )
        msg = _build_native_tool_assistant_message([call])
        assert msg["role"] == "assistant"
        assert msg["content"] == ""
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "tc-1"
        assert tc["function"]["name"] == "shell"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    def test_with_thinking_content(self) -> None:
        call = SimpleNamespace(id="tc-2", name="read_file", arguments={})
        msg = _build_native_tool_assistant_message([call], thinking_content="Let me think...")
        assert msg["reasoning_content"] == "Let me think..."

    def test_empty_thinking_not_included(self) -> None:
        call = SimpleNamespace(id="tc-3", name="test", arguments={})
        msg = _build_native_tool_assistant_message([call], thinking_content="")
        assert "reasoning_content" not in msg


# ── _annotate_uncertain_effect ───────────────────────────────────────


class TestAnnotateUncertainEffect:
    def test_marks_uncertain_on_side_effect_failure(self) -> None:
        payload: Dict[str, Any] = {"ok": False}
        result = _annotate_uncertain_effect(payload, "external_side_effect")
        assert result["side_effect_uncertain"] is True
        assert "retry_guidance" in result

    def test_no_annotation_on_read_only(self) -> None:
        payload: Dict[str, Any] = {"ok": False}
        result = _annotate_uncertain_effect(payload, "read_only")
        assert "side_effect_uncertain" not in result

    def test_no_annotation_on_success(self) -> None:
        payload: Dict[str, Any] = {"ok": True}
        result = _annotate_uncertain_effect(payload, "external_side_effect")
        assert "side_effect_uncertain" not in result


# ── Token estimation ─────────────────────────────────────────────────


class TestTokenEstimation:
    def test_empty_text_zero(self) -> None:
        assert _estimate_text_tokens("") == 0

    def test_latin_text_approx(self) -> None:
        tokens = _estimate_text_tokens("hello world this is a test")
        assert tokens > 0
        # ~26 chars / 4 ≈ 6-7 tokens
        assert 4 <= tokens <= 10

    def test_cjk_text_higher_ratio(self) -> None:
        cjk = "推动经济增长"
        latin = "abcdef"  # same length
        assert _estimate_text_tokens(cjk) > _estimate_text_tokens(latin)

    def test_message_tokens_adds_overhead(self) -> None:
        msg = {"role": "user", "content": "hello"}
        tokens = _estimate_message_tokens(msg)
        text_tokens = _estimate_text_tokens("hello")
        assert tokens == 6 + text_tokens

    def test_prompt_tokens_includes_framing(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        result = _estimate_prompt_tokens(messages)
        assert result >= _estimate_message_tokens(messages[0]) + 3

    def test_empty_messages_zero(self) -> None:
        assert _estimate_prompt_tokens([]) == 0

    def test_list_content_handled(self) -> None:
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        tokens = _estimate_message_tokens(msg)
        assert tokens > 6


# ── _extract_json_object ─────────────────────────────────────────────


class TestExtractJsonObject:
    def test_extracts_from_surrounding_text(self) -> None:
        text = 'Some preamble {"key": "value"} trailing'
        result = _extract_json_object(text)
        assert result == {"key": "value"}

    def test_raises_on_no_json(self) -> None:
        with pytest.raises(ValueError, match="no json object"):
            _extract_json_object("no json here")


# ── _keywords_from_query ─────────────────────────────────────────────


class TestKeywordsFromQuery:
    def test_latin_words(self) -> None:
        kw = _keywords_from_query("find all python files")
        assert "find" in kw
        assert "python" in kw
        assert "files" in kw

    def test_cjk_bigrams(self) -> None:
        kw = _keywords_from_query("推动经济增长")
        assert "推动" in kw
        assert "经济" in kw

    def test_max_twelve(self) -> None:
        query = " ".join(f"word{i}" for i in range(20))
        kw = _keywords_from_query(query)
        assert len(kw) <= 12

    def test_short_segments_filtered(self) -> None:
        kw = _keywords_from_query("a b cd ef")
        # single-char latin segments should be filtered
        assert "a" not in kw
        assert "b" not in kw
        assert "cd" in kw


# ── Constants smoke ──────────────────────────────────────────────────


class TestConstants:
    def test_empty_response_prompts_exist(self) -> None:
        assert len(_EMPTY_RESPONSE_RETRY_PROMPT) > 0
        assert len(_EMPTY_RESPONSE_DEGRADED_MESSAGE) > 0

    def test_forced_finalize_prompt(self) -> None:
        assert "SYSTEM" in _FORCED_FINALIZE_PROMPT

    def test_task_contract_heading(self) -> None:
        assert _TASK_CONTRACT_HEADING == "## Task Contract"

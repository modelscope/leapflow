from __future__ import annotations

from typing import Any

import pytest

from leapflow.engine.context_compressor import (
    CompressorConfig,
    ContextCompressor,
    SummarizeStage,
    adaptive_trim_chars,
    estimate_text_tokens,
)
from leapflow.engine.context_control import (
    ContextBudgetEstimator,
    ContextGovernanceController,
    ContextPostureConfig,
    ContextWindowController,
    LongTaskContextController,
    ToolEvidenceBuilder,
)
from leapflow.tools.file_operations import file_read


class _NoopCompressor:
    def force_compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages


def test_context_estimator_counts_messages_and_tool_schemas() -> None:
    estimator = ContextBudgetEstimator()
    messages = [{"role": "user", "content": "hello 世界"}]
    tools = [{"type": "function", "function": {"name": "demo", "description": "tool"}}]

    snapshot = estimator.snapshot(messages, tools=tools, context_length=1000)

    assert snapshot.message_tokens > 0
    assert snapshot.tool_schema_tokens > 0
    assert snapshot.total_tokens == snapshot.message_tokens + snapshot.tool_schema_tokens
    assert 0 < snapshot.ratio < 1


def test_context_compressor_tolerates_none_tool_calls() -> None:
    compressor = ContextCompressor(CompressorConfig(
        token_budget=100,
        token_count_fn=lambda text: len(str(text)),
    ))
    messages = [
        {"role": "assistant", "content": "plain response", "tool_calls": None},
        {"role": "user", "content": "next request"},
    ]

    formatted = SummarizeStage._format_turns_for_summary(messages)
    summary = SummarizeStage._deterministic_summary(messages)
    sanitized = SummarizeStage._sanitize_tool_pairs(messages)
    token_count = compressor._count_tokens(messages)

    assert "plain response" in formatted
    assert "next request" in summary
    assert sanitized == messages
    assert token_count > 0


def test_context_window_controller_forces_final_answer_when_over_budget() -> None:
    controller = ContextWindowController(
        estimator=ContextBudgetEstimator(),
        hard_limit_ratio=0.50,
        warning_ratio=0.25,
    )
    messages = [{"role": "user", "content": "x" * 400} for _ in range(10)]

    decision = controller.prepare(
        messages,
        context_length=100,
        compressor=_NoopCompressor(),
    )

    assert decision.compressed is True
    assert decision.forced_final_answer is True
    assert any("final answer now" in str(item.get("content", "")) for item in decision.messages)
    assert len(decision.messages) < len(messages) + 1


def test_context_compressor_records_transparent_trace() -> None:
    compressor = ContextCompressor(CompressorConfig(
        token_budget=100,
        max_output_chars=120,
        enabled_stages=["trim"],
    ))
    messages = [
        {"role": "system", "content": "system"},
        {"role": "tool", "content": "A" * 1_000},
    ]

    prepared = compressor.compress(messages)
    trace = compressor.last_trace.as_dict()

    assert prepared[1]["content"] != messages[1]["content"]
    assert trace["stages_applied"] == ["trim"]
    assert trace["stage_effects"][0]["stage"] == "trim"
    assert trace["decision_reason"] == "threshold-triggered"
    assert trace["tokens_after"] < trace["tokens_before"]
    assert trace["saved_tokens"] > 0
    assert trace["savings_ratio"] > 0


def test_tool_evidence_builder_compacts_file_read_content() -> None:
    builder = ToolEvidenceBuilder(max_content_chars=240)
    result = {
        "ok": True,
        "path": "/tmp/example.py",
        "content": "head\n" + "x" * 1000 + "\ntail",
        "lines": 3,
        "mode": "raw",
        "start_line": 1,
        "end_line": 3,
        "selected_lines": 3,
        "truncated": True,
    }

    evidence = builder.build("file_read", {"path": "/tmp/example.py"}, result)

    assert evidence["kind"] == "file_read_evidence"
    assert evidence["path"] == "/tmp/example.py"
    assert evidence["truncated"] is True
    assert "chars omitted" in evidence["excerpt"]
    assert len(evidence["excerpt"]) < len(result["content"])


def test_long_task_controller_reports_repeated_reads() -> None:
    controller = LongTaskContextController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        repeated_read_limit=1,
        convergence_round=2,
    )
    args = {"path": "/tmp/repeated.py"}
    result = {"ok": True, "path": "/tmp/repeated.py", "content": "print(1)", "mode": "raw"}

    controller.compact_tool_result("file_read", args, result)
    controller.compact_tool_result("file_read", args, result)
    metadata = controller.tool_metadata("file_read", args, result)

    assert metadata["context_evidence"] is True
    assert metadata["read_count"] == 2
    assert metadata["repeat_read"] is True
    notice = controller.convergence_notice(2)
    assert "repeated reads" in notice
    assert "complementary project evidence" in notice
    assert "synthesize" in notice


def test_context_governance_reset_clears_turn_scope() -> None:
    controller = LongTaskContextController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        repeated_read_limit=1,
        convergence_round=20,
    )
    args = {"path": "/tmp/repeated.py"}
    result = {"ok": True, "path": "/tmp/repeated.py", "content": "print(1)", "mode": "raw"}

    controller.compact_tool_result("file_read", args, result)
    controller.compact_tool_result("file_read", args, result)
    assert controller.snapshot().repeated_reads == 1

    controller.reset_turn_scope()

    snapshot = controller.snapshot()
    assert snapshot.repeated_reads == 0
    assert snapshot.sources_seen == 0
    assert snapshot.evidence_count == 0
    assert controller.convergence_notice(1) == ""


def test_long_task_metadata_avoids_noise_for_uncompacted_tools() -> None:
    controller = LongTaskContextController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    )

    metadata = controller.tool_metadata("time_get", {}, {"ok": True, "result": "now"})

    assert metadata == {}


@pytest.mark.asyncio
async def test_file_read_rejects_workspace_leapflow_config_probe(tmp_path) -> None:
    result = await file_read({"path": str(tmp_path / ".leapflow" / "config.json")})

    assert result["ok"] is False
    assert result["error_type"] == "unsupported_config_probe"
    assert result["retryable"] is False
    assert result["config_locations"] == [
        "~/.leapflow/config/user.yaml",
        "~/.leapflow/profiles/<profile>/config/*.yaml",
        "<workspace>/.leapflow/config.yaml",
    ]
    assert ".leapflow/config.json" in result["error"]


def test_context_governance_controller_keeps_long_task_alias() -> None:
    controller = ContextGovernanceController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        posture_config=ContextPostureConfig(
            expanded_evidence_threshold=1,
            expanded_tool_call_threshold=10,
            research_source_threshold=10,
            research_evidence_threshold=10,
        ),
    )

    controller.compact_tool_result(
        "shell_run",
        {"command": "pytest"},
        {"ok": True, "stdout": "passed", "stderr": ""},
    )
    snapshot = controller.snapshot().as_dict()

    assert isinstance(LongTaskContextController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
    ), ContextGovernanceController)
    assert snapshot["posture"] == "expanded"
    assert snapshot["guidance"] == "prefer outline, symbols, or range reads before raw content"


def test_exploration_ledger_promotes_without_explicit_mode() -> None:
    controller = LongTaskContextController(
        evidence_builder=ToolEvidenceBuilder(max_content_chars=240),
        repeated_read_limit=1,
        convergence_round=20,
    )

    for index in range(3):
        path = f"/tmp/source_{index}.py"
        controller.compact_tool_result(
            "file_read",
            {"path": path},
            {"ok": True, "path": path, "content": "print(1)", "mode": "symbols"},
        )

    snapshot = controller.snapshot().as_dict()

    assert snapshot["posture"] == "research"
    assert snapshot["sources_seen"] == 3
    assert snapshot["dominant_signal"] == "multi-source"
    assert snapshot["guidance"] == "maintain research ledger and synthesize findings"


@pytest.mark.asyncio
async def test_file_read_supports_range_outline_symbols_and_bounded_content(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "# Title\n"
        "intro = 1\n"
        "\n"
        "class Demo:\n"
        "    def method(self):\n"
        "        return 'x'\n"
        "\n"
        "def helper():\n"
        "    return 'y'\n"
        + "z" * 500,
        encoding="utf-8",
    )

    raw = await file_read({"path": str(source), "start_line": 4, "max_lines": 2})
    outline = await file_read({"path": str(source), "mode": "outline", "max_lines": 5})
    symbols = await file_read({"path": str(source), "mode": "symbols", "max_lines": 5})
    bounded = await file_read({"path": str(source), "max_chars": 220, "max_lines": 2000})

    assert raw["ok"] is True
    assert raw["start_line"] == 4
    assert raw["end_line"] == 5
    assert "class Demo" in raw["content"]
    assert outline["mode"] == "outline"
    assert outline["selected_lines"] >= 2
    assert "# Title" in outline["content"]
    assert symbols["mode"] == "symbols"
    assert "class Demo" in symbols["content"]
    assert "def helper" in symbols["content"]
    assert bounded["truncated"] is True
    assert len(bounded["content"]) <= 220


# ── CJK-aware token estimator ────────────────────────────────────────


def test_estimate_text_tokens_cjk_counts_characters_as_tokens() -> None:
    assert estimate_text_tokens("你好世界") == 4
    assert estimate_text_tokens("hello") >= 1
    mixed = "Hello 世界 test"
    tokens = estimate_text_tokens(mixed)
    assert tokens >= 2  # at least the 2 CJK characters
    assert estimate_text_tokens("") == 0


def test_estimate_text_tokens_latin_divides_by_four() -> None:
    text = "a" * 400
    tokens = estimate_text_tokens(text)
    assert tokens == 100


# ── Adaptive trim threshold scaling ──────────────────────────────────


def test_adaptive_trim_chars_scales_with_context_length() -> None:
    assert adaptive_trim_chars(2000, 0) == 2000
    threshold_128k = adaptive_trim_chars(2000, 128_000)
    threshold_256k = adaptive_trim_chars(2000, 256_000)
    threshold_1m = adaptive_trim_chars(2000, 1_000_000)

    assert threshold_128k > 2000
    assert threshold_256k > threshold_128k
    assert threshold_1m >= threshold_256k
    assert threshold_1m <= 50_000  # ceiling


def test_adaptive_trim_chars_never_below_base() -> None:
    assert adaptive_trim_chars(5000, 32_000) >= 5000


# ── CompressorConfig adaptive scaling ────────────────────────────────


def test_compressor_config_applies_adaptive_scaling_from_context_length() -> None:
    config = CompressorConfig(
        context_length=256_000,
        max_output_chars=2000,
    )
    assert config.trim_threshold_chars > 2000


def test_compressor_config_respects_higher_explicit_threshold() -> None:
    config = CompressorConfig(
        context_length=128_000,
        max_output_chars=30_000,
    )
    assert config.trim_threshold_chars >= 30_000


# ── Budget-aware TrimStage ───────────────────────────────────────────


def test_trim_stage_skips_when_budget_is_plentiful() -> None:
    config = CompressorConfig(
        token_budget=256_000,
        context_length=256_000,
        max_output_chars=2000,
        enabled_stages=["trim"],
    )
    compressor = ContextCompressor(config)

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "tool", "content": "A" * 5_000},
    ]
    result = compressor.compress(messages, token_count=2_000)

    assert result[1]["content"] == messages[1]["content"]
    assert compressor.last_trace.stages_applied == []


def test_trim_stage_activates_when_budget_pressure_exists() -> None:
    config = CompressorConfig(
        token_budget=10_000,
        context_length=10_000,
        max_output_chars=2000,
        enabled_stages=["trim"],
    )
    compressor = ContextCompressor(config)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "tool", "content": "B" * 5_000},
    ]
    result = compressor.compress(messages, token_count=8_000)

    assert result[1]["content"] != messages[1]["content"]
    assert "trim" in compressor.last_trace.stages_applied


def test_trim_stage_always_trims_ceiling_exceeding_messages() -> None:
    config = CompressorConfig(
        token_budget=1_000_000,
        context_length=1_000_000,
        max_output_chars=2000,
        trim_ceiling_chars=10_000,
        enabled_stages=["trim"],
    )
    compressor = ContextCompressor(config)

    messages = [
        {"role": "system", "content": "system"},
        {"role": "tool", "content": "C" * 20_000},
    ]
    result = compressor.compress(messages, token_count=100)

    assert result[1]["content"] != messages[1]["content"]
    assert "trim" in compressor.last_trace.stages_applied


# ── Compressor reconfigure ───────────────────────────────────────────


def test_compressor_reconfigure_updates_budget_and_threshold() -> None:
    compressor = ContextCompressor(CompressorConfig(
        token_budget=128_000,
        context_length=128_000,
    ))
    old_threshold = compressor._config.trim_threshold_chars

    compressor.reconfigure(token_budget=256_000, context_length=256_000)

    assert compressor._config.token_budget == 256_000
    assert compressor._config.context_length == 256_000
    assert compressor._config.trim_threshold_chars > old_threshold


def test_compressor_reconfigure_preserves_summarize_state() -> None:
    from leapflow.engine.context_compressor import SummarizeStage

    compressor = ContextCompressor(CompressorConfig(
        token_budget=128_000,
        context_length=128_000,
    ))

    summarize_stages = [s for s in compressor._stages if s.name == "summarize"]
    assert len(summarize_stages) == 1
    stage = summarize_stages[0]
    assert isinstance(stage, SummarizeStage)

    stage._previous_summary = "test summary from earlier"
    stage._compression_count = 3

    compressor.reconfigure(token_budget=256_000, context_length=256_000)

    summarize_after = [s for s in compressor._stages if s.name == "summarize"]
    assert len(summarize_after) == 1
    preserved = summarize_after[0]
    assert preserved is stage
    assert preserved._previous_summary == "test summary from earlier"
    assert preserved._compression_count == 3


# ── Deterministic summary preserves key info ─────────────────────────


def test_deterministic_summary_preserves_tool_args_and_results() -> None:
    from leapflow.engine.context_compressor import SummarizeStage

    middle = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "file_read",
                    "arguments": '{"path": "/src/main.py"}',
                },
            }],
        },
        {
            "role": "tool",
            "content": '{"ok": true, "path": "/src/main.py", "exit_code": 0}',
        },
        {"role": "user", "content": "Please fix the bug in the database module"},
    ]

    summary = SummarizeStage._deterministic_summary(middle)

    assert "file_read" in summary
    assert "path=/src/main.py" in summary
    assert "ok=True" in summary or "ok=true" in summary.lower()
    assert "fix the bug in the database module" in summary


# ── ToolEvidenceBuilder context-adaptive ─────────────────────────────


def test_evidence_builder_adapts_to_context_length() -> None:
    small = ToolEvidenceBuilder(max_content_chars=1200, context_length=0)
    large = ToolEvidenceBuilder(max_content_chars=1200, context_length=256_000)

    assert large._max_content_chars > small._max_content_chars


def test_evidence_builder_ceiling() -> None:
    huge = ToolEvidenceBuilder(max_content_chars=1200, context_length=2_000_000)
    assert huge._max_content_chars <= 8_000


def test_evidence_builder_preserves_platform_permission_recovery_fields() -> None:
    builder = ToolEvidenceBuilder(max_content_chars=240)

    evidence = builder.build(
        "platform_action",
        {"platform": "feishu", "action": "im.list_chats"},
        {
            "ok": False,
            "error": "access denied",
            "failure_class": "authorization",
            "failure_code": "missing_scope",
            "recoverability": "admin_required",
            "capability": "im.chat.read",
            "missing_scopes": ["im:chat:read"],
            "required_scopes": ["im:chat:read"],
            "scope_relation": "all_required",
            "scope_source": "authoritative",
            "console_url": "https://open.feishu.cn/app/cli_xxx/auth",
            "next_steps": ["Grant missing scopes", "Re-publish the app"],
            "recovery_hint": "Grant the missing scope in the developer console.",
            "retryable": False,
        },
    )

    assert evidence["ok"] is False
    assert evidence["platform"] == "feishu"
    assert evidence["action"] == "im.list_chats"
    assert evidence["failure_class"] == "authorization"
    assert evidence["failure_code"] == "missing_scope"
    assert evidence["missing_scopes"] == ["im:chat:read"]
    assert evidence["required_scopes"] == ["im:chat:read"]
    assert evidence["scope_relation"] == "all_required"
    assert evidence["scope_source"] == "authoritative"
    assert evidence["console_url"] == "https://open.feishu.cn/app/cli_xxx/auth"
    assert evidence["next_steps"] == ["Grant missing scopes", "Re-publish the app"]


# ── CJK-aware _estimate_tokens in compressor ─────────────────────────


def test_compressor_estimate_tokens_is_cjk_aware() -> None:
    messages_cjk = [{"role": "user", "content": "你" * 100}]
    messages_latin = [{"role": "user", "content": "a" * 100}]

    tokens_cjk = ContextCompressor._estimate_tokens(messages_cjk)
    tokens_latin = ContextCompressor._estimate_tokens(messages_latin)

    assert tokens_cjk > tokens_latin

from __future__ import annotations

from typing import Any

import pytest

from leapflow.engine.context_compressor import CompressorConfig, ContextCompressor
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

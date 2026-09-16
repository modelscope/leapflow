# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from leapflow.engine.context_focus import (
    ContextPlane,
    FocusEntity,
    SessionFocusState,
    control_event_from_tool,
    focus_entity_from_tool,
)
from leapflow.engine.reference_resolver import ReferenceResolver


def _minicpm_focus(turn_id: int = 1) -> FocusEntity:
    return FocusEntity(
        entity_id="paper:minicpm",
        kind="paper",
        canonical_name="MiniCPM-O 4.5 Technical Report",
        plane=ContextPlane.TASK_SEMANTIC,
        aliases=("MiniCPM", "MiniCPM-O"),
        evidence_refs=("web_fetch:https://arxiv.org/abs/2601.21337",),
        first_turn=turn_id,
        last_task_turn=turn_id,
        last_mentioned_turn=turn_id,
    )


def test_config_set_records_control_event_without_replacing_task_focus() -> None:
    state = SessionFocusState()
    state.record_focus(_minicpm_focus())

    state.record_tool_result(
        "config_set",
        {"key": "llm.model", "value": "qwen3.8-max"},
        {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
        turn_id=2,
    )
    state.record_tool_result(
        "config_set",
        {"key": "llm.model", "value": "qwen3.7-plus"},
        {"ok": True, "key": "llm.model", "value": "qwen3.7-plus"},
        turn_id=3,
    )

    assert state.active_focus is not None
    assert state.active_focus.canonical_name == "MiniCPM-O 4.5 Technical Report"
    assert [event.value for event in state.recent_control_events] == ["qwen3.8-max", "qwen3.7-plus"]


def test_reference_resolver_prefers_task_focus_over_control_events() -> None:
    state = SessionFocusState()
    state.record_focus(_minicpm_focus())
    state.record_tool_result(
        "config_set",
        {"key": "llm.model", "value": "qwen3.8-max"},
        {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
        turn_id=2,
    )

    # Regardless of user text, the task entity takes priority
    resolution = ReferenceResolver().resolve("上面的 paper 需要更深层次解读", state)

    assert resolution.target_id == "paper:minicpm"
    assert resolution.target_name == "MiniCPM-O 4.5 Technical Report"
    assert resolution.plane == ContextPlane.TASK_SEMANTIC
    assert resolution.confidence >= 0.9


def test_reference_resolver_falls_back_to_control_event_when_no_task_entity() -> None:
    state = SessionFocusState()
    # Only control events, no task-semantic entities
    state.record_tool_result(
        "config_set",
        {"key": "llm.model", "value": "qwen3.8-max"},
        {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
        turn_id=2,
    )

    resolution = ReferenceResolver().resolve("刚才设置的默认模型是什么？", state)

    assert resolution.target_kind == "config"
    assert resolution.target_name == "qwen3.8-max"
    assert resolution.plane == ContextPlane.CONTROL_PLANE
    assert resolution.confidence > 0.0


def test_focus_entity_from_arxiv_fetch_is_paper_evidence() -> None:
    entity = focus_entity_from_tool(
        "web_fetch",
        {"url": "https://arxiv.org/abs/2601.21337"},
        {"title": "MiniCPM-O 4.5 Technical Report"},
        turn_id=1,
    )

    assert entity is not None
    assert entity.kind == "paper"
    assert entity.canonical_name == "MiniCPM-O 4.5 Technical Report"
    assert entity.evidence_refs == ("web_fetch:https://arxiv.org/abs/2601.21337",)


def test_control_event_from_config_set_is_structured() -> None:
    event = control_event_from_tool(
        "config_set",
        {"key": "llm.model", "value": "qwen3.8-max"},
        {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
        turn_id=4,
    )

    assert event is not None
    assert event.key == "llm.model"
    assert event.value == "qwen3.8-max"
    assert event.user_visible_summary == "llm.model -> qwen3.8-max"


def test_control_event_from_config_set_does_not_echo_secret_argument() -> None:
    event = control_event_from_tool(
        "config_set",
        {"key": "llm.api_key", "value": "sk-secret-token"},
        {"ok": True, "key": "llm.api_key"},
        turn_id=5,
    )

    assert event is not None
    assert event.value == ""
    assert "sk-secret-token" not in event.user_visible_summary


def test_prompt_focus_block_separates_task_focus_from_control_events() -> None:
    state = SessionFocusState()
    state.record_focus(_minicpm_focus())
    state.record_tool_result(
        "config_set",
        {"key": "llm.model", "value": "qwen3.8-max"},
        {"ok": True, "key": "llm.model", "value": "qwen3.8-max"},
        turn_id=2,
    )
    resolution = ReferenceResolver().resolve("上面的 paper", state)

    assert resolution.target_id == "paper:minicpm"  # task entity takes priority

    block = state.render_prompt_context(resolution)

    assert "Current task focus" in block
    assert "MiniCPM-O 4.5 Technical Report" in block
    assert "Recent control-plane events" in block
    assert "llm.model -> qwen3.8-max" in block
    assert "not task targets unless explicitly requested" in block

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for SkillDispatcher — skill/intent dispatch and teach commands."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from leapflow.engine.skill_dispatcher import SkillDispatcher


# ── Minimal engine stub ──────────────────────────────────────────────


def _stub_engine(
    *,
    skills: List[Any] | None = None,
    skill_library: Any = None,
    wm_events: List[str] | None = None,
    current_session_id: str = "session-1",
    current_turn_id: str = "turn-1",
    current_command_id: str = "turn-1",
    current_task_contract: Any = None,
    active_frame: Any = None,
) -> SimpleNamespace:
    """Build a minimal engine stub for SkillDispatcher."""
    registry = SimpleNamespace(
        list_all=lambda: list(skills or []),
        find_by_trigger=lambda text, threshold=0.5: [],
        get=lambda name: None,
    )
    remembered_events: List[tuple[str, str]] = []

    def _remember_event(kind: str, msg: str) -> None:
        remembered_events.append((kind, msg))

    wm = SimpleNamespace(
        remember_event=_remember_event,
        _remembered_events=remembered_events,
    )
    settings = SimpleNamespace(
        workspace_root="/tmp/test",
        profile_layout=SimpleNamespace(profile_id="default"),
    )
    engine = SimpleNamespace(
        _registry=registry,
        _skill_library=skill_library,
        _wm=wm,
        _settings=settings,
        _current_session_id=current_session_id,
        _current_turn_id=current_turn_id,
        _current_command_id=current_command_id,
        _current_task_contract=current_task_contract,
        _active_frame=active_frame or SimpleNamespace(
            session_id="session-1",
            turn_id="turn-1",
            command_id="turn-1",
            user_text="",
        ),
    )
    return engine


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_creates_with_engine_back_reference(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._engine is engine


# ── Teach command detection ──────────────────────────────────────────


class TestIsTeachCommand:
    def test_teach_alone(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("teach") is True

    def test_teach_me(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("teach me") is True

    def test_start_teaching(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("start teaching") is True

    def test_stop_teaching(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("stop teaching") is True

    def test_watch_me(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("watch me") is True

    def test_chinese_teach(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("教我") is True
        assert dispatcher._is_teach_command("开始教学") is True
        assert dispatcher._is_teach_command("停止教学") is True

    def test_not_a_teach_command(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("teach me how to cook") is False
        assert dispatcher._is_teach_command("what is teaching?") is False
        assert dispatcher._is_teach_command("teaching methods for math") is False

    def test_case_insensitive(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("TEACH") is True
        assert dispatcher._is_teach_command("Teach Me") is True

    def test_whitespace_stripped(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        assert dispatcher._is_teach_command("  teach  ") is True


# ── _parse_approval ──────────────────────────────────────────────────


class TestParseApproval:
    @pytest.mark.asyncio
    async def test_approve_by_number(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1", "s2", "s3"]
        action, indices = await dispatcher._parse_approval("approve 2", suggestions)
        assert action == "approve"
        assert indices == [1]  # 0-indexed

    @pytest.mark.asyncio
    async def test_reject_by_number(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1", "s2"]
        action, indices = await dispatcher._parse_approval("reject 1", suggestions)
        assert action == "reject"
        assert indices == [0]

    @pytest.mark.asyncio
    async def test_approve_all(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1", "s2", "s3"]
        action, indices = await dispatcher._parse_approval("approve all", suggestions)
        assert action == "approve"
        assert indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_default_to_first_when_no_number(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1", "s2"]
        action, indices = await dispatcher._parse_approval("approve", suggestions)
        assert action == "approve"
        assert indices == [0]

    @pytest.mark.asyncio
    async def test_chinese_approval_keywords(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1"]
        action, _ = await dispatcher._parse_approval("批准 1", suggestions)
        assert action == "approve"

    @pytest.mark.asyncio
    async def test_chinese_reject_keywords(self) -> None:
        engine = _stub_engine()
        dispatcher = SkillDispatcher(engine)
        suggestions = ["s1"]
        action, _ = await dispatcher._parse_approval("拒绝 1", suggestions)
        assert action == "reject"


# ── Skill list handling ──────────────────────────────────────────────


class TestHandleSkillList:
    def test_empty_skills(self) -> None:
        engine = _stub_engine(skills=[])
        dispatcher = SkillDispatcher(engine)
        result = dispatcher._handle_skill_list()
        assert "No skills registered" in result

    def test_populated_skills(self) -> None:
        skill = SimpleNamespace(
            name="deploy",
            description="Deploy application to server with zero downtime",
            metadata=SimpleNamespace(version=2, confidence=0.85),
        )
        engine = _stub_engine(skills=[skill])
        dispatcher = SkillDispatcher(engine)
        result = dispatcher._handle_skill_list()
        assert "deploy" in result
        assert "v2" in result
        assert "85%" in result


# ── Pending skill reminder ───────────────────────────────────────────


class TestInjectPendingSkillReminder:
    def test_no_skill_library(self) -> None:
        engine = _stub_engine(skill_library=None)
        dispatcher = SkillDispatcher(engine)
        dispatcher._inject_pending_skill_reminder()
        assert len(engine._wm._remembered_events) == 0

    def test_no_pending(self) -> None:
        lib = SimpleNamespace(count_pending=lambda: 0)
        engine = _stub_engine(skill_library=lib)
        dispatcher = SkillDispatcher(engine)
        dispatcher._inject_pending_skill_reminder()
        assert len(engine._wm._remembered_events) == 0

    def test_pending_injects_reminder(self) -> None:
        lib = SimpleNamespace(count_pending=lambda: 3)
        engine = _stub_engine(skill_library=lib)
        dispatcher = SkillDispatcher(engine)
        dispatcher._inject_pending_skill_reminder()
        events = engine._wm._remembered_events
        assert len(events) == 1
        assert "3 skill update suggestion" in events[0][1]


# ── Skill review (no library) ───────────────────────────────────────


class TestHandleSkillReview:
    def test_no_library(self) -> None:
        engine = _stub_engine(skill_library=None)
        dispatcher = SkillDispatcher(engine)
        result = dispatcher._handle_skill_review()
        assert "not configured" in result

    def test_no_pending_suggestions(self) -> None:
        lib = SimpleNamespace(load_pending_suggestions=lambda limit=10: [])
        engine = _stub_engine(skill_library=lib)
        dispatcher = SkillDispatcher(engine)
        result = dispatcher._handle_skill_review()
        assert "No pending" in result


# ── _format_recent_events (static) ──────────────────────────────────


class TestFormatRecentEvents:
    def test_formats_events(self) -> None:
        events = [
            {"time": "12:00:00", "type": "file_change", "content": "/tmp/test.txt"},
            {"time": "12:01:00", "type": "clipboard", "content": "copied text"},
        ]
        result = SkillDispatcher._format_recent_events(events)
        assert "2 events" in result
        assert "12:00:00" in result
        assert "file_change" in result


# ── evolution_action_context ─────────────────────────────────────────


class TestEvolutionActionContext:
    def test_builds_context_with_basic_fields(self) -> None:
        contract = SimpleNamespace(workspace_root="/tmp/test")
        engine = _stub_engine(current_task_contract=contract)
        dispatcher = SkillDispatcher(engine)
        ctx = dispatcher._evolution_action_context("action-123")
        assert ctx.action_id == "action-123"
        assert ctx.session_id == "session-1"
        assert ctx.profile_id == "default"
        assert "session:" in ctx.correlation_id

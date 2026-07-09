"""Scenario-based tests for the memory subsystem and storage layer."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import duckdb

from leapflow.domain.trajectory import (
    ActionType,
    RawAction,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)
from leapflow.memory import (
    EpisodicMemoryProvider, SemanticMemoryProvider, WorkingMemoryProvider,
)
from leapflow.platform.event_bus import EventBus
from leapflow.platform.protocol import EventTypes
from leapflow.storage.skill_library import StoredSkill
from leapflow.storage.connection import LocalConnectionHolder
from leapflow.storage.duckdb_connect import DatabaseLockedError


# ── Storage lock fallback ───────────────────────────────────────────


def test_connection_holder_uses_volatile_duckdb_when_primary_is_locked(
    tmp_path,
    monkeypatch,
) -> None:
    import leapflow.storage.connection as connection_module

    primary_path = tmp_path / "leap.duckdb"
    original_connect = connection_module._lock_aware_connect

    def flaky_connect(db_path: Path):
        if Path(db_path) == primary_path:
            raise DatabaseLockedError(primary_path, RuntimeError("locked"))
        return original_connect(db_path)

    monkeypatch.setattr(connection_module, "_lock_aware_connect", flaky_connect)

    holder = LocalConnectionHolder(primary_path, volatile_on_lock=True)
    conn = holder.connection
    volatile_path = holder.db_path

    try:
        assert holder.is_volatile is True
        assert holder.locked_error is not None
        assert volatile_path != primary_path
        assert volatile_path.name == "leap.duckdb"
        assert conn.execute("SELECT 1").fetchone() == (1,)
        assert volatile_path.exists()
    finally:
        holder.close()

    assert volatile_path.exists() is False


def test_write_buffer_drops_permanent_failures(tmp_path) -> None:
    from leapflow.storage.write_buffer import WriteBuffer

    db_path = tmp_path / "buffer.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY)")
        buffer = WriteBuffer(conn, max_count=10)
        buffer.append("bad-sql", "INSERT INTO missing_table VALUES (?)", [1])

        assert buffer.flush() == 0
        assert buffer.pending == 0
    finally:
        conn.close()


# ── Working memory ─────────────────────────────────────────────────


def test_working_memory_chat_roundtrip() -> None:
    wm = WorkingMemoryProvider(max_tokens=2048)
    wm.remember_chat({"role": "user", "content": "hello"})
    wm.remember_chat({"role": "assistant", "content": "hi there"})

    msgs = wm.as_chat_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "hi there"


def test_working_memory_overflow() -> None:
    wm = WorkingMemoryProvider(max_tokens=64)
    for i in range(50):
        wm.remember_chat({"role": "user", "content": f"message-{i} " + "x" * 200})

    msgs = wm.as_chat_messages()
    assert len(msgs) < 50
    assert msgs[-1]["content"].startswith("message-49")


def test_working_memory_pattern_counting() -> None:
    wm = WorkingMemoryProvider()
    assert wm.get_pattern_count("click_save") == 0

    assert wm.increment_pattern("click_save") == 1
    assert wm.increment_pattern("click_save") == 2
    assert wm.get_pattern_count("click_save") == 2

    wm.increment_pattern("open_dialog")
    assert wm.get_pattern_count("open_dialog") == 1
    assert wm.get_pattern_count("click_save") == 2

    wm.clear()
    assert wm.get_pattern_count("click_save") == 0


# ── Long-term memory ─────────────────────────────────────────────


def test_long_term_insert_and_search(long_term_memory: SemanticMemoryProvider) -> None:
    long_term_memory.insert_raw("note", "meeting notes about planning")
    long_term_memory.insert_raw("note", "grocery shopping list")
    long_term_memory.insert_raw("file_event", "File modified: /tmp/report.pdf")

    all_meeting = long_term_memory.search_keywords(["meeting"])
    assert len(all_meeting) == 1
    assert all_meeting[0].kind == "note"
    assert "meeting" in all_meeting[0].content.lower()

    note_only = long_term_memory.search_keywords(["meeting"], kinds=["note"])
    assert len(note_only) == 1
    assert note_only[0].kind == "note"

    file_hits = long_term_memory.search_keywords(["report"], kinds=["file_event"])
    assert len(file_hits) == 1
    assert file_hits[0].kind == "file_event"

    no_match = long_term_memory.search_keywords(["meeting"], kinds=["file_event"])
    assert no_match == []


def test_long_term_metadata_preserved(long_term_memory: SemanticMemoryProvider) -> None:
    mid = long_term_memory.insert_raw(
        "note",
        "meeting notes",
        metadata={"topic": "planning", "priority": "high"},
    )

    by_search = long_term_memory.search_keywords(["meeting"], kinds=["note"])
    assert len(by_search) == 1
    assert by_search[0].metadata["topic"] == "planning"
    assert by_search[0].metadata["priority"] == "high"

    by_id = long_term_memory.get_by_id(mid)
    assert by_id is not None
    assert by_id.metadata == {"topic": "planning", "priority": "high"}


# ── Immediate memory ───────────────────────────────────────────────


def test_immediate_memory_ttl_expiry() -> None:
    imm = EpisodicMemoryProvider(ttl=0.01)
    imm.ingest("clipboard", "some text", path=None)
    assert imm.active_count == 1

    time.sleep(0.02)

    assert imm.recent(limit=5) == []
    assert imm.search_fragments(["text"]) == []
    assert imm.active_count == 0


def test_immediate_memory_promotion() -> None:
    promoted: list = []

    def on_promote(frag):
        promoted.append(frag)

    imm = EpisodicMemoryProvider(ttl=60.0, on_promote=on_promote, max_entries=3)
    frag = imm.ingest("clipboard", "important clipboard text", path=None)

    result = imm.touch(frag.fragment_id)
    assert result is not None
    assert len(promoted) == 1
    assert promoted[0].fragment_id == frag.fragment_id
    assert promoted[0].content == "important clipboard text"
    assert promoted[0].referenced is True


# ── Event bus ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_event_bus_subscribe_and_dispatch(working_memory: WorkingMemoryProvider) -> None:
    imm = EpisodicMemoryProvider(ttl=60.0)
    bus = EventBus(immediate=imm, working=working_memory)
    received: list = []
    bus.subscribe(lambda ev: received.append((ev.event_type, ev.payload)))

    await bus.handle_event(EventTypes.FS_CHANGE, {
        "path": "/file.txt",
        "flags": 0x100,
        "ts": time.time(),
    })

    assert len(received) == 1
    assert received[0][0] == "fs.change"
    assert received[0][1]["path"] == "/file.txt"
    assert received[0][1]["action"] == "created"

    recent = imm.recent(limit=5)
    assert len(recent) == 1
    assert "file.txt" in recent[0].content.lower()


@pytest.mark.asyncio
async def test_event_bus_reorder_buffer(working_memory: WorkingMemoryProvider) -> None:
    imm = EpisodicMemoryProvider(ttl=60.0)
    bus = EventBus(immediate=imm, working=working_memory)
    received: list = []
    bus.subscribe(lambda ev: received.append(ev.event_type))

    bus.enable_reorder(settle_s=0.05)
    # Submit UI action (later ts) before focus change (earlier ts)
    await bus.handle_event(EventTypes.UI_ACTION, {
        "_mono_ts": 2.0,
        "action": "click",
        "app_bundle_id": "com.test.app",
        "timestamp": time.time(),
    })
    await bus.handle_event(EventTypes.APP_FOCUS_CHANGE, {
        "_mono_ts": 1.0,
        "bundle_id": "com.test.app",
        "app_name": "Test App",
    })
    await asyncio.sleep(0.08)

    assert len(received) == 2
    assert received[0] == "app.focus_change"
    assert received[1] == "ui.action"

    await bus.disable_reorder()


# ── Trajectory store ───────────────────────────────────────────────


def test_trajectory_store_roundtrip(trajectory_store) -> None:
    step = TrajectoryStep(
        state=StateSnapshot(
            timestamp=1.0,
            focused_app="com.app",
            snapshot_level="light",
        ),
        action=RawAction(
            timestamp=1.0,
            action_type=ActionType.UI_CLICK,
            target="Save",
            app_bundle_id="com.app",
            app_name="App",
            params={"x": 100},
        ),
        post_state=StateSnapshot(
            timestamp=1.1,
            focused_app="com.app",
            snapshot_level="light",
        ),
    )
    traj = Trajectory(
        trajectory_id="t1",
        user_id="test_user",
        start_time=1.0,
        end_time=2.0,
        steps=[step],
        metadata={"goal": "test"},
    )

    trajectory_store.save_trajectory(traj)
    loaded = trajectory_store.load_trajectory("t1")
    assert loaded is not None
    assert loaded.trajectory_id == "t1"
    assert loaded.user_id == "test_user"
    assert loaded.start_time == 1.0
    assert loaded.end_time == 2.0
    assert loaded.metadata["goal"] == "test"
    assert loaded.step_count == 1

    loaded_step = loaded.steps[0]
    assert loaded_step.action.action_type == ActionType.UI_CLICK
    assert loaded_step.action.target == "Save"
    assert loaded_step.action.app_bundle_id == "com.app"
    assert loaded_step.action.params == {"x": 100}
    assert loaded_step.state.focused_app == "com.app"

    summaries = trajectory_store.list_trajectories()
    assert any(s["id"] == "t1" for s in summaries)


# ── Skill library ──────────────────────────────────────────────────


def test_skill_library_crud(skill_library) -> None:
    skill_library.save_skill(StoredSkill(
        skill_id="s1",
        title="My Skill",
        steps=["a", "b"],
        version=1,
        confidence=0.8,
        trigger_phrases=["do something"],
        parameters=[{"name": "path", "description": "target dir"}],
        pre_conditions=["finder available"],
        source_trajectory_id="t1",
        source_episode_id="e1",
        status="active",
    ))

    loaded = skill_library.load_skill("s1")
    assert loaded is not None
    assert loaded.skill_id == "s1"
    assert loaded.title == "My Skill"
    assert loaded.steps == ["a", "b"]
    assert loaded.version == 1
    assert loaded.confidence == 0.8
    assert loaded.trigger_phrases == ["do something"]
    assert loaded.parameters == [{"name": "path", "description": "target dir"}]
    assert loaded.pre_conditions == ["finder available"]
    assert loaded.source_trajectory_id == "t1"
    assert loaded.source_episode_id == "e1"
    assert loaded.status == "active"

    active = skill_library.load_all_active()
    assert len(active) == 1
    assert active[0].skill_id == "s1"


# ── Cross-tier integration ─────────────────────────────────────────


def test_memory_tiers_integration(tmp_db) -> None:
    lt = SemanticMemoryProvider(source=tmp_db)
    lt._ensure_connection()
    try:

        def promote_to_long_term(frag):
            lt.insert_raw(
                kind=frag.event_type,
                content=frag.content,
                path=frag.path,
                metadata=frag.metadata,
            )

        imm = EpisodicMemoryProvider(ttl=60.0, on_promote=promote_to_long_term)
        frag = imm.ingest(
            "clipboard",
            "promoted clipboard snippet",
            path=None,
            metadata={"source": "test"},
        )
        imm.touch(frag.fragment_id)

        hits = lt.search_keywords(["clipboard"], kinds=["clipboard"])
        assert len(hits) == 1
        assert hits[0].content == "promoted clipboard snippet"
        assert hits[0].metadata["source"] == "test"
    finally:
        lt.close()

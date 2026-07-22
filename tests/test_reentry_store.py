"""Unit tests for the S2 re-entry store (phase N1, pure storage layer).

Hermetic: DuckDB on a temp file, no engine / gateway / network.
"""
from __future__ import annotations

from leapflow.storage.reentry_store import (
    OrientSnapshot,
    ReentryKind,
    ReentryState,
    ReentryStore,
    ReentryTrigger,
)


def _trigger(**kw) -> ReentryTrigger:
    base = dict(
        task_id="t1",
        kind=ReentryKind.TIME.value,
        orient=OrientSnapshot(
            task_id="t1",
            ledger_state={"findings": ["A uses DuckDB"], "open_questions": ["does B cache?"]},
            task_contract={"goal": "investigate"},
            continuation_summary="looked at A, need B",
        ),
    )
    base.update(kw)
    return ReentryTrigger(**base)


def test_reentry_roundtrip_preserves_orient(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        trig = _trigger(due_at=100.0)
        store.save(trig)
        loaded = store.load(trig.trigger_id)
        assert loaded is not None
        assert loaded.task_id == "t1" and loaded.kind == "time" and loaded.due_at == 100.0
        assert loaded.state == ReentryState.ARMED.value
        assert loaded.orient is not None
        assert loaded.orient.ledger_state["findings"] == ["A uses DuckDB"]
        assert loaded.orient.task_contract == {"goal": "investigate"}
        assert loaded.orient.continuation_summary == "looked at A, need B"
    finally:
        store.close()


def test_list_due_only_returns_ready_time_triggers(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="past", due_at=50.0))
        store.save(_trigger(trigger_id="future", due_at=500.0))
        store.save(_trigger(trigger_id="unset", due_at=0.0))          # 0 => not time-due
        store.save(_trigger(trigger_id="evt", kind=ReentryKind.EVENT.value, due_at=50.0))
        due = {t.trigger_id for t in store.list_due(now=100.0)}
        assert due == {"past"}
    finally:
        store.close()


def test_fire_single_shot_then_exhausted(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="s", due_at=50.0, max_reentries=1))
        claimed = store.fire("s", now=100.0)
        assert claimed is not None and claimed.reentries_used == 1
        assert claimed.state == ReentryState.EXHAUSTED.value
        assert store.fire("s", now=101.0) is None                     # no budget left
        assert store.list_due(now=100.0) == []                        # exhausted -> not due
    finally:
        store.close()


def test_fire_recurring_respects_budget(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="r", due_at=50.0, max_reentries=3))
        for expected_used in (1, 2):
            claimed = store.fire("r", now=100.0)
            assert claimed is not None and claimed.reentries_used == expected_used
            assert claimed.state == ReentryState.ARMED.value          # budget remains
            store.advance_due("r", new_due_at=100.0 + expected_used)   # recurring re-arm
        final = store.fire("r", now=200.0)
        assert final is not None and final.reentries_used == 3
        assert final.state == ReentryState.EXHAUSTED.value
        assert store.fire("r", now=201.0) is None
    finally:
        store.close()


def test_fire_past_deadline_is_rejected(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="d", due_at=50.0, deadline=150.0, max_reentries=5))
        assert store.fire("d", now=200.0) is None                     # past deadline
        assert store.load("d").state == ReentryState.EXHAUSTED.value  # marked terminal
    finally:
        store.close()


def test_cancel_blocks_fire(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="c", due_at=50.0))
        assert store.cancel("c") is True
        assert store.cancel("c") is False                             # not armed anymore
        assert store.fire("c", now=100.0) is None
    finally:
        store.close()


def test_list_armed_events(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="e1", kind=ReentryKind.EVENT.value,
                            event_match={"platform": "feishu", "keyword": "deploy"}))
        store.save(_trigger(trigger_id="t1t", kind=ReentryKind.TIME.value, due_at=50.0))
        events = store.list_armed_events()
        assert {e.trigger_id for e in events} == {"e1"}
        assert events[0].event_match == {"platform": "feishu", "keyword": "deploy"}
    finally:
        store.close()


def test_cleanup_removes_terminal_and_expired(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    try:
        store.save(_trigger(trigger_id="armed", due_at=500.0))
        store.save(_trigger(trigger_id="cancel_me", due_at=50.0))
        store.cancel("cancel_me")
        store.save(_trigger(trigger_id="expired", due_at=10.0, deadline=20.0))
        removed = store.cleanup(now=100.0)
        assert removed == 2                                            # cancelled + deadline-passed
        assert store.load("armed") is not None
        assert store.load("cancel_me") is None and store.load("expired") is None
    finally:
        store.close()


def test_persistence_survives_reopen(tmp_path) -> None:
    db = tmp_path / "leap.duckdb"
    store = ReentryStore(db)
    store.save(_trigger(trigger_id="persist", due_at=50.0))
    store.close()

    store2 = ReentryStore(db)
    try:
        loaded = store2.load("persist")
        assert loaded is not None and loaded.orient.task_contract == {"goal": "investigate"}
        assert {t.trigger_id for t in store2.list_due(now=100.0)} == {"persist"}
    finally:
        store2.close()


# ── N2: build_reentry_trigger factory + schedule_reentry tool + config ──


def test_build_reentry_trigger_time_event_deadline() -> None:
    from leapflow.storage.reentry_store import build_reentry_trigger

    t = build_reentry_trigger(
        task_id="t1", kind="time", delay_seconds=30.0,
        ledger_state={"findings": ["x"]}, task_contract={"goal": "g"},
        continuation_summary="  cont  ", now=1000.0,
    )
    assert t.kind == "time" and t.due_at == 1030.0 and t.deadline == 0.0
    assert t.orient.ledger_state == {"findings": ["x"]}
    assert t.orient.continuation_summary == "cont"          # trimmed

    e = build_reentry_trigger(task_id="t1", kind="event", event_match={"platform": "feishu"}, now=1000.0)
    assert e.kind == "event" and e.due_at == 0.0 and e.event_match == {"platform": "feishu"}

    d = build_reentry_trigger(task_id="t1", deadline_seconds=100.0, now=1000.0)
    assert d.deadline == 1100.0

    b = build_reentry_trigger(task_id="t1", kind="weird", now=1000.0)
    assert b.kind == "time"                                  # unknown kind -> time


def test_schedule_reentry_handler_registers(tmp_path) -> None:
    import asyncio

    from leapflow.storage.reentry_store import ReentryStore, build_reentry_trigger
    from leapflow.tools import registry_bootstrap as rb

    store = ReentryStore(tmp_path / "leap.duckdb")

    def _scheduler(*, kind, reason, delay_seconds, event_match, max_reentries, deadline_seconds):
        trig = build_reentry_trigger(
            task_id="t1", kind=kind, delay_seconds=delay_seconds,
            continuation_summary=reason, event_match=event_match,
            max_reentries=max_reentries, deadline_seconds=deadline_seconds,
        )
        store.save(trig)
        return {"ok": True, "trigger_id": trig.trigger_id, "kind": trig.kind}

    rb.set_reentry_scheduler(_scheduler)
    try:
        res = asyncio.run(rb.TOOL_HANDLERS["schedule_reentry"](
            {"kind": "time", "reason": "continue after deploy", "delay_seconds": 60}
        ))
        assert res["ok"] is True and "trigger_id" in res
        assert store.load(res["trigger_id"]) is not None
    finally:
        rb.set_reentry_scheduler(None)
        store.close()
    unset = asyncio.run(rb.TOOL_HANDLERS["schedule_reentry"]({"kind": "time", "reason": "x"}))
    assert unset["ok"] is False                              # not initialized after reset


def test_schedule_reentry_disclosed_and_blocked_in_subagents() -> None:
    from leapflow.engine.subagent import DELEGATE_BLOCKED_TOOLS
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

    names = {td.get("function", {}).get("name") for td in TOOL_DEFINITIONS}
    assert "schedule_reentry" in names                       # disclosed (core-eligible)
    assert "schedule_reentry" in DELEGATE_BLOCKED_TOOLS       # but not for subagents


def test_reentry_config_default_off() -> None:
    from leapflow.config import get_settings
    from leapflow.config_service import ConfigService

    settings = get_settings()
    assert settings.agent_reentry_enabled is False            # default off = zero behavior change
    assert "agent.reentry_enabled" in ConfigService(settings).writable_keys()

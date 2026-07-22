"""Unit tests for the S2 re-entry driver (phase N3).

Hermetic: DuckDB re-entry store on a temp file + a stub async runner; no engine,
no daemon, no network. Covers CAS claim, single-shot exhaustion, recurring
anti-storm advance, disabled no-op, runner-error isolation, and fan-out bound.
"""
from __future__ import annotations

import asyncio

from leapflow.scheduler.reentry_driver import ReentryDriver
from leapflow.storage.reentry_store import (
    ReentryState,
    ReentryStore,
    build_reentry_trigger,
)


def _collecting_runner(sink: list):
    async def _runner(orient, trigger=None):
        sink.append(orient.task_id)
    return _runner


def test_reentry_driver_dispatches_due_single_shot(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    seen: list[str] = []
    driver = ReentryDriver(store=store, runner=_collecting_runner(seen), enabled=True)
    try:
        store.save(build_reentry_trigger(task_id="t1", kind="time", delay_seconds=10, now=0.0))    # due_at=10
        store.save(build_reentry_trigger(task_id="t2", kind="time", delay_seconds=500, now=0.0))   # due_at=500
        assert asyncio.run(driver.tick(now=100.0)) == 1
        assert seen == ["t1"]                              # only the due one
        seen.clear()
        assert asyncio.run(driver.tick(now=100.0)) == 0    # single-shot -> exhausted -> not re-dispatched
        assert seen == []
    finally:
        store.close()


def test_reentry_driver_disabled_is_noop(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    seen: list[str] = []
    driver = ReentryDriver(store=store, runner=_collecting_runner(seen), enabled=False)
    try:
        store.save(build_reentry_trigger(task_id="t1", delay_seconds=10, now=0.0))
        assert asyncio.run(driver.tick(now=100.0)) == 0
        assert seen == []
    finally:
        store.close()


def test_reentry_driver_recurring_advances_due(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    driver = ReentryDriver(
        store=store, runner=_collecting_runner([]), enabled=True,
        recurring_interval_seconds=3600.0,
    )
    try:
        trig = build_reentry_trigger(task_id="r", delay_seconds=10, now=0.0, max_reentries=2)  # due_at=10
        store.save(trig)
        assert asyncio.run(driver.tick(now=100.0)) == 1
        reloaded = store.load(trig.trigger_id)
        assert reloaded.state == ReentryState.ARMED.value and reloaded.reentries_used == 1
        assert reloaded.due_at == 100.0 + 3600.0          # advanced -> no same-tick storm
        assert asyncio.run(driver.tick(now=100.0)) == 0   # not due at 100 anymore
    finally:
        store.close()


def test_reentry_driver_runner_error_is_isolated(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")

    async def _boom(orient, trigger=None):
        raise RuntimeError("boom")

    driver = ReentryDriver(store=store, runner=_boom, enabled=True)
    try:
        trig = build_reentry_trigger(task_id="t1", delay_seconds=10, now=0.0)  # single-shot
        store.save(trig)
        assert asyncio.run(driver.tick(now=100.0)) == 0    # error -> not counted, no raise
        # already claimed before running (at-most-once): not retried
        assert store.load(trig.trigger_id).state == ReentryState.EXHAUSTED.value
    finally:
        store.close()


def test_reentry_driver_respects_max_per_tick(tmp_path) -> None:
    store = ReentryStore(tmp_path / "leap.duckdb")
    seen: list[str] = []
    driver = ReentryDriver(store=store, runner=_collecting_runner(seen), enabled=True, max_per_tick=2)
    try:
        for i in range(5):
            store.save(build_reentry_trigger(task_id=f"t{i}", delay_seconds=1, now=0.0))
        assert asyncio.run(driver.tick(now=100.0)) == 2    # bounded fan-out
        assert len(seen) == 2
    finally:
        store.close()


def test_build_reentry_subagent_config() -> None:
    from leapflow.scheduler.reentry_driver import build_reentry_subagent_config
    from leapflow.storage.reentry_store import OrientSnapshot

    orient = OrientSnapshot(
        task_id="t1",
        ledger_state={
            "findings": ["A uses DuckDB"],
            "open_questions": ["does B cache?"],
            "next_step": "inspect B",
        },
        task_contract={"original_request": "map the architecture"},
        continuation_summary="checked README",
    )
    cfg = build_reentry_subagent_config(orient)
    assert cfg.goal == "map the architecture"
    for token in ("checked README", "A uses DuckDB", "does B cache?", "inspect B"):
        assert token in cfg.context
    assert cfg.metadata.get("reentry") is True and cfg.metadata.get("task_id") == "t1"

    empty = build_reentry_subagent_config(OrientSnapshot(task_id="t2"))
    assert empty.goal == "Continue the task."           # safe default, never empty


def test_reentry_driver_runs_isolated_subagent(tmp_path) -> None:
    from leapflow.scheduler.reentry_driver import build_reentry_subagent_config
    from leapflow.storage.reentry_store import ReentryStore, build_reentry_trigger

    store = ReentryStore(tmp_path / "leap.duckdb")
    dispatched: list = []

    class _StubManager:
        async def delegate(self, config):
            dispatched.append(config)
            return type("R", (), {"summary": "done", "status": "completed"})()

    manager = _StubManager()

    async def _runner(orient, trigger=None):
        await manager.delegate(build_reentry_subagent_config(orient))

    driver = ReentryDriver(store=store, runner=_runner, enabled=True)
    try:
        store.save(build_reentry_trigger(
            task_id="t1", kind="time", delay_seconds=10, now=0.0,
            task_contract={"original_request": "continue X"},
            ledger_state={"findings": ["f1"]},
        ))
        assert asyncio.run(driver.tick(now=100.0)) == 1
        assert len(dispatched) == 1                       # isolated subagent dispatched
        assert dispatched[0].goal == "continue X" and "f1" in dispatched[0].context
    finally:
        store.close()

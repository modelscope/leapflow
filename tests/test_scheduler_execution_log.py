# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the scheduler execution log — DuckDB store + coordinator wiring + /schedule payload."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from leapflow.scheduler.execution_log import (
    DuckDBExecutionLogStore,
    ExecutionLogRecord,
    ExecutionLogStore,
)
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.types import ArmedTask, TaskState


# ════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_scheduler.duckdb"


@pytest.fixture
def log_store(db_path: Path) -> DuckDBExecutionLogStore:
    return DuckDBExecutionLogStore(db_path)


@pytest.fixture
def task_store(db_path: Path) -> TaskStore:
    return TaskStore(db_path)


# ════════════════════════════════════════════════════════════════════════
# Protocol conformance
# ════════════════════════════════════════════════════════════════════════


class TestProtocol:
    """Verify DuckDBExecutionLogStore satisfies the ExecutionLogStore Protocol."""

    def test_protocol_conformance(self, log_store: DuckDBExecutionLogStore) -> None:
        assert isinstance(log_store, ExecutionLogStore)

    def test_record_is_frozen(self) -> None:
        r = ExecutionLogRecord(
            task_id="t1", execution_id="e1", trigger_type="interval",
            started_at=time.time(), finished_at=None, status="running",
            result_summary="", error="",
        )
        with pytest.raises(AttributeError):
            r.status = "done"  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════
# DuckDBExecutionLogStore — write / read / cleanup
# ════════════════════════════════════════════════════════════════════════


class TestDuckDBExecutionLogStore:

    def test_record_start_returns_unique_id(self, log_store: DuckDBExecutionLogStore) -> None:
        eid1 = log_store.record_start("task-a", "interval")
        eid2 = log_store.record_start("task-a", "interval")
        assert eid1 != eid2

    def test_start_then_finish_persists(self, log_store: DuckDBExecutionLogStore) -> None:
        eid = log_store.record_start("task-a", "cron")
        log_store.record_finish(eid, "success", result_summary="all good")
        records = log_store.get_history(task_id="task-a")
        assert len(records) == 1
        r = records[0]
        assert r.execution_id == eid
        assert r.status == "success"
        assert r.result_summary == "all good"
        assert r.finished_at is not None

    def test_get_history_newest_first(self, log_store: DuckDBExecutionLogStore) -> None:
        eid1 = log_store.record_start("task-b", "interval")
        log_store.record_finish(eid1, "success")
        time.sleep(0.01)  # ensure distinct timestamps
        eid2 = log_store.record_start("task-b", "interval")
        log_store.record_finish(eid2, "failed", error="boom")

        records = log_store.get_history(task_id="task-b")
        assert len(records) == 2
        assert records[0].execution_id == eid2  # newest first
        assert records[1].execution_id == eid1

    def test_get_history_limit(self, log_store: DuckDBExecutionLogStore) -> None:
        for _ in range(5):
            eid = log_store.record_start("task-c", "interval")
            log_store.record_finish(eid, "success")
        records = log_store.get_history(task_id="task-c", limit=3)
        assert len(records) == 3

    def test_get_history_all_tasks(self, log_store: DuckDBExecutionLogStore) -> None:
        log_store.record_start("task-x", "interval")
        log_store.record_start("task-y", "cron")
        records = log_store.get_history(task_id=None)
        assert len(records) == 2

    def test_cleanup_removes_old_records(self, log_store: DuckDBExecutionLogStore) -> None:
        # Insert a record with a very old timestamp (manually)
        import uuid
        old_eid = uuid.uuid4().hex
        old_time = time.time() - 8 * 24 * 3600  # 8 days ago
        log_store._con.execute(
            """
            INSERT INTO scheduler_execution_log
                (execution_id, task_id, trigger_type, started_at, status)
            VALUES (?, 'old-task', 'interval', ?, 'success')
            """,
            [old_eid, old_time],
        )
        # Insert a recent record
        recent_eid = log_store.record_start("recent-task", "interval")
        log_store.record_finish(recent_eid, "success")

        deleted = log_store.cleanup(max_age_hours=168.0)  # 7 days
        assert deleted == 1
        remaining = log_store.get_history()
        assert len(remaining) == 1
        assert remaining[0].task_id == "recent-task"

    def test_finish_records_failure_with_error(self, log_store: DuckDBExecutionLogStore) -> None:
        eid = log_store.record_start("task-fail", "event")
        log_store.record_finish(eid, "failed", error="connection refused")
        records = log_store.get_history(task_id="task-fail")
        assert records[0].error == "connection refused"
        assert records[0].status == "failed"


# ════════════════════════════════════════════════════════════════════════
# Coordinator integration — execution log wiring
# ════════════════════════════════════════════════════════════════════════


class TestCoordinatorExecutionHistory:
    """Coordinator.get_execution_history delegates to the injected store."""

    def test_returns_empty_when_no_store(self) -> None:
        from leapflow.scheduler.coordinator import TaskCoordinator
        store = MagicMock(spec=TaskStore)
        coord = TaskCoordinator(store=store, execution_log=None)
        assert coord.get_execution_history() == []

    def test_returns_records_from_store(self, db_path: Path) -> None:
        from leapflow.scheduler.coordinator import TaskCoordinator
        task_store = TaskStore(db_path)
        log_store = DuckDBExecutionLogStore(db_path)
        eid = log_store.record_start("t1", "interval")
        log_store.record_finish(eid, "success", result_summary="ok")

        coord = TaskCoordinator(store=task_store, execution_log=log_store)
        history = coord.get_execution_history(task_id="t1")
        assert len(history) == 1
        assert history[0].status == "success"


# ════════════════════════════════════════════════════════════════════════
# LocalScheduler — execution log recording end-to-end
# ════════════════════════════════════════════════════════════════════════


class TestLocalSchedulerLogging:
    """LocalScheduler records start + finish through the execution log."""

    @pytest.mark.asyncio
    async def test_execution_logs_on_success(self, db_path: Path) -> None:
        from leapflow.scheduler.local_scheduler import LocalScheduler

        task_store = TaskStore(db_path)
        log_store = DuckDBExecutionLogStore(db_path)

        class _OK:
            async def execute(self, skill_name: str, parameters: dict) -> dict:
                return {"ok": True, "output": "done"}

        sched = LocalScheduler(
            store=task_store, executor=_OK(),
            tick_seconds=9999, execution_log=log_store,
        )

        task = ArmedTask(
            skill_name="ping", trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            state=TaskState.ARMED.value,
        )
        task_store.save(task)
        await sched._execute_task(task, time.time())

        records = log_store.get_history(task_id=task.task_id)
        assert len(records) == 1
        assert records[0].status == "success"

    @pytest.mark.asyncio
    async def test_execution_logs_on_failure(self, db_path: Path) -> None:
        from leapflow.scheduler.local_scheduler import LocalScheduler

        task_store = TaskStore(db_path)
        log_store = DuckDBExecutionLogStore(db_path)

        class _Fail:
            async def execute(self, skill_name: str, parameters: dict) -> dict:
                raise RuntimeError("broken")

        sched = LocalScheduler(
            store=task_store, executor=_Fail(),
            tick_seconds=9999, execution_log=log_store,
        )

        task = ArmedTask(
            skill_name="bad", trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            state=TaskState.ARMED.value,
        )
        task_store.save(task)
        await sched._execute_task(task, time.time())

        records = log_store.get_history(task_id=task.task_id)
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "broken" in records[0].error


# ════════════════════════════════════════════════════════════════════════
# /schedule payload builder
# ════════════════════════════════════════════════════════════════════════


class TestSchedulePayload:
    """Tests for build_schedule_payload mirroring /checkpoint tests."""

    def _make_ctx(self, db_path: Path) -> Any:
        """Build a minimal context stub with settings.duckdb_path."""
        ctx = MagicMock()
        ctx.settings.duckdb_path = db_path
        ctx.coordinator = None
        return ctx

    def test_schedule_list_empty(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        # Ensure the TaskStore table exists
        TaskStore(db_path)
        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "list")
        assert result["ok"] is True
        assert "No scheduled tasks" in result["message"]

    def test_schedule_list_with_tasks(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        store = TaskStore(db_path)
        store.save(ArmedTask(
            skill_name="deploy", trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state="armed",
        ))
        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "")
        assert result["ok"] is True
        assert "deploy" in result["message"]

    def test_schedule_history_empty(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        # Ensure log table exists
        DuckDBExecutionLogStore(db_path)
        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "history")
        assert result["ok"] is True
        assert "No execution history" in result["message"]

    def test_schedule_history_with_records(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        log_store = DuckDBExecutionLogStore(db_path)
        eid = log_store.record_start("task-x", "interval")
        log_store.record_finish(eid, "success", result_summary="deployed")
        # Need TaskStore table too
        TaskStore(db_path)

        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "history")
        assert result["ok"] is True
        assert "success" in result["message"]
        assert "deployed" in result["message"]

    def test_schedule_cancel_missing_id(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        TaskStore(db_path)
        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "cancel")
        assert result["ok"] is False
        assert "Usage" in result["message"]

    def test_schedule_cancel_success(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        store = TaskStore(db_path)
        task = ArmedTask(
            skill_name="backup", trigger_type="interval",
            trigger_config={"interval_seconds": 3600},
            state="armed",
        )
        store.save(task)

        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, f"cancel {task.task_id}")
        assert result["ok"] is True
        assert "Cancelled" in result["message"]
        # Verify the state changed
        updated = store.load(task.task_id)
        assert updated is not None
        assert updated.state == "suspended"

    def test_schedule_unknown_subcommand(self, db_path: Path) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        TaskStore(db_path)
        ctx = self._make_ctx(db_path)
        result = build_schedule_payload(ctx, "bogus")
        assert result["ok"] is False
        assert "Unknown schedule subcommand" in result["message"]


# ════════════════════════════════════════════════════════════════════════
# Slash completion for /schedule
# ════════════════════════════════════════════════════════════════════════


class TestScheduleCompletion:
    """Verify SlashCommandCompleter offers /schedule subcommands."""

    def test_schedule_subcommands_offered(self) -> None:
        from prompt_toolkit.document import Document
        from leapflow.cli.tui_app.input import SlashCommandCompleter

        completer = SlashCommandCompleter((
            ("schedule", "List active scheduled tasks"),
            ("schedule history", "Show recent execution log entries"),
            ("schedule cancel", "Cancel/disable a scheduled task"),
        ))
        completions = list(completer.get_completions(
            Document("/schedule ", len("/schedule ")), None,
        ))
        texts = [c.text for c in completions]
        assert "list" in texts
        assert "history" in texts
        assert "cancel" in texts

    def test_schedule_subcommand_filters(self) -> None:
        from prompt_toolkit.document import Document
        from leapflow.cli.tui_app.input import SlashCommandCompleter

        completer = SlashCommandCompleter((
            ("schedule", "List active scheduled tasks"),
        ))
        completions = list(completer.get_completions(
            Document("/schedule h", len("/schedule h")), None,
        ))
        assert len(completions) == 1
        assert completions[0].text == "history"


# ════════════════════════════════════════════════════════════════════════
# Command registry
# ════════════════════════════════════════════════════════════════════════


class TestCommandRegistry:
    """Verify /schedule commands are registered and resolvable."""

    def test_schedule_in_registry(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule")
        assert cmd is not None
        assert cmd.name == "schedule"

    def test_schedule_history_resolvable(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule history abc123")
        assert cmd is not None
        assert cmd.name == "schedule history"

    def test_schedule_cancel_resolvable(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule cancel abc123")
        assert cmd is not None
        assert cmd.name == "schedule cancel"

    def test_schedule_list_alias(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule list")
        assert cmd is not None
        # "schedule list" is an alias → resolves to the base "schedule" command
        assert cmd.name == "schedule"

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Phase 1B+1C: scheduler CRUD completion and basic retry."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from leapflow.scheduler.coordinator import TaskCoordinator
from leapflow.scheduler.local_scheduler import LocalScheduler
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.types import ArmedTask, TaskState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> TaskStore:
    """Create a TaskStore backed by a temporary DuckDB file."""
    return TaskStore(tmp_path / "test.duckdb")


@pytest.fixture()
def sample_task() -> ArmedTask:
    """An armed task with a 5-minute interval trigger."""
    return ArmedTask(
        task_id="test_task_001",
        skill_name="backup",
        trigger_type="interval",
        trigger_config={"interval_seconds": 300},
        state=TaskState.ARMED.value,
        next_due_at=time.time() + 300,
        max_retries=0,
        retry_count=0,
        retry_backoff_s=60.0,
    )


class _StubExecutor:
    """Stub SkillExecutor that returns configurable results."""

    def __init__(self, ok: bool = True, raise_exc: Exception | None = None) -> None:
        self.ok = ok
        self.raise_exc = raise_exc
        self.call_count = 0

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        self.call_count += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"ok": self.ok, "output": "done"}


# ---------------------------------------------------------------------------
# Part B — CRUD Completion
# ---------------------------------------------------------------------------


class TestTaskStatePaused:
    """PAUSED enum value exists and integrates properly."""

    def test_paused_enum_exists(self):
        assert hasattr(TaskState, "PAUSED")
        assert TaskState.PAUSED.value == "paused"

    def test_paused_tasks_not_in_get_due_tasks(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """Paused tasks should not appear in get_due_tasks."""
        sample_task.next_due_at = time.time() - 10  # overdue
        tmp_store.save(sample_task)
        # Should appear when armed
        due = tmp_store.get_due_tasks(time.time())
        assert len(due) == 1

        # Pause it
        tmp_store.update_state(sample_task.task_id, TaskState.PAUSED.value)
        due = tmp_store.get_due_tasks(time.time())
        assert len(due) == 0

    def test_pause_and_resume_via_coordinator(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """pause_task sets PAUSED; resume_task re-arms + recalculates next_due."""
        tmp_store.save(sample_task)
        coordinator = TaskCoordinator(store=tmp_store)

        # Pause
        asyncio.get_event_loop().run_until_complete(
            coordinator.pause_task(sample_task.task_id)
        )
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded is not None
        assert loaded.state == TaskState.PAUSED.value

        # Resume
        asyncio.get_event_loop().run_until_complete(
            coordinator.resume_task(sample_task.task_id)
        )
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded is not None
        assert loaded.state == TaskState.ARMED.value
        # next_due should be recalculated (in the future)
        assert loaded.next_due_at > time.time() - 1

    def test_pause_nonexistent_raises(self, tmp_store: TaskStore):
        coordinator = TaskCoordinator(store=tmp_store)
        with pytest.raises(ValueError, match="Task not found"):
            asyncio.get_event_loop().run_until_complete(
                coordinator.pause_task("nonexistent")
            )


class TestUpdateTask:
    """Store.update_task and Coordinator.update_task."""

    def test_store_update_task_changes_trigger(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """update_task changes trigger_config and next_due_at."""
        tmp_store.save(sample_task)
        new_config = {"interval_seconds": 600}
        result = tmp_store.update_task(
            sample_task.task_id,
            trigger_config=new_config,
            next_due_at=time.time() + 600,
        )
        assert result is True
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded is not None
        assert loaded.trigger_config == new_config
        assert loaded.next_due_at > time.time() + 500

    def test_store_update_task_rejects_unknown_field(self, tmp_store: TaskStore, sample_task: ArmedTask):
        tmp_store.save(sample_task)
        with pytest.raises(ValueError, match="Cannot update"):
            tmp_store.update_task(sample_task.task_id, skill_name="hack")

    def test_store_set_state_convenience(self, tmp_store: TaskStore, sample_task: ArmedTask):
        tmp_store.save(sample_task)
        result = tmp_store.set_state(sample_task.task_id, TaskState.PAUSED.value)
        assert result is True
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded.state == TaskState.PAUSED.value

    def test_coordinator_update_task_changes_trigger_expr(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """Coordinator.update_task parses a new expression and recalculates."""
        tmp_store.save(sample_task)
        coordinator = TaskCoordinator(store=tmp_store)
        updated = asyncio.get_event_loop().run_until_complete(
            coordinator.update_task(sample_task.task_id, trigger_expr="10m")
        )
        assert updated.trigger_type == "interval"
        assert updated.trigger_config == {"interval_seconds": 600}
        assert updated.next_due_at > time.time()

    def test_coordinator_update_task_not_found(self, tmp_store: TaskStore):
        coordinator = TaskCoordinator(store=tmp_store)
        with pytest.raises(ValueError, match="Task not found"):
            asyncio.get_event_loop().run_until_complete(
                coordinator.update_task("nonexistent", trigger_expr="5m")
            )


# ---------------------------------------------------------------------------
# Part C — Basic Retry
# ---------------------------------------------------------------------------


class TestRetryFields:
    """ArmedTask has retry fields and they persist through the store."""

    def test_armed_task_defaults(self):
        task = ArmedTask(
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
        )
        assert task.max_retries == 0
        assert task.retry_count == 0
        assert task.retry_backoff_s == 60.0

    def test_retry_fields_roundtrip(self, tmp_store: TaskStore):
        """Retry fields survive save/load cycle."""
        task = ArmedTask(
            task_id="retry_test",
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            max_retries=3,
            retry_count=1,
            retry_backoff_s=30.0,
        )
        tmp_store.save(task)
        loaded = tmp_store.load("retry_test")
        assert loaded is not None
        assert loaded.max_retries == 3
        assert loaded.retry_count == 1
        assert loaded.retry_backoff_s == 30.0


class TestRetryLogic:
    """Retry behavior in LocalScheduler._execute_task."""

    def test_failed_task_retries_with_backoff(self, tmp_store: TaskStore):
        """A failed task with retries remaining gets re-armed with backoff."""
        executor = _StubExecutor(ok=False)
        scheduler = LocalScheduler(store=tmp_store, executor=executor)

        task = ArmedTask(
            task_id="retry_backoff",
            skill_name="flaky_skill",
            trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state=TaskState.ARMED.value,
            next_due_at=time.time() - 1,
            max_retries=3,
            retry_count=0,
            retry_backoff_s=10.0,
        )
        tmp_store.save(task)

        now = time.time()
        asyncio.get_event_loop().run_until_complete(
            scheduler._execute_task(task, now)
        )

        loaded = tmp_store.load("retry_backoff")
        assert loaded is not None
        assert loaded.state == TaskState.ARMED.value  # re-armed for retry
        assert loaded.retry_count == 1
        # next_due should be approximately now + 10.0 * 2^0 = now + 10
        assert loaded.next_due_at >= now + 9
        assert loaded.next_due_at <= now + 15

    def test_retries_exhausted_sets_failed(self, tmp_store: TaskStore):
        """When retries are exhausted, state becomes FAILED and retry_count resets."""
        executor = _StubExecutor(ok=False)
        scheduler = LocalScheduler(store=tmp_store, executor=executor)

        task = ArmedTask(
            task_id="exhaust_retry",
            skill_name="always_fails",
            trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state=TaskState.ARMED.value,
            next_due_at=time.time() - 1,
            max_retries=2,
            retry_count=2,  # already at limit
            retry_backoff_s=5.0,
        )
        tmp_store.save(task)

        now = time.time()
        asyncio.get_event_loop().run_until_complete(
            scheduler._execute_task(task, now)
        )

        loaded = tmp_store.load("exhaust_retry")
        assert loaded is not None
        assert loaded.state == TaskState.FAILED.value
        assert loaded.retry_count == 0  # reset for potential manual re-arm

    def test_success_resets_retry_count(self, tmp_store: TaskStore):
        """A successful execution resets retry_count to 0."""
        executor = _StubExecutor(ok=True)
        scheduler = LocalScheduler(store=tmp_store, executor=executor)

        task = ArmedTask(
            task_id="success_reset",
            skill_name="good_skill",
            trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state=TaskState.ARMED.value,
            next_due_at=time.time() - 1,
            max_retries=3,
            retry_count=2,  # was mid-retry
            retry_backoff_s=10.0,
        )
        tmp_store.save(task)

        now = time.time()
        asyncio.get_event_loop().run_until_complete(
            scheduler._execute_task(task, now)
        )

        loaded = tmp_store.load("success_reset")
        assert loaded is not None
        assert loaded.retry_count == 0
        assert loaded.state == TaskState.ARMED.value

    def test_exception_triggers_retry(self, tmp_store: TaskStore):
        """A hard exception also triggers retry logic."""
        executor = _StubExecutor(raise_exc=RuntimeError("connection refused"))
        scheduler = LocalScheduler(store=tmp_store, executor=executor)

        task = ArmedTask(
            task_id="exc_retry",
            skill_name="crashy",
            trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state=TaskState.ARMED.value,
            next_due_at=time.time() - 1,
            max_retries=2,
            retry_count=0,
            retry_backoff_s=5.0,
        )
        tmp_store.save(task)

        now = time.time()
        asyncio.get_event_loop().run_until_complete(
            scheduler._execute_task(task, now)
        )

        loaded = tmp_store.load("exc_retry")
        assert loaded is not None
        assert loaded.state == TaskState.ARMED.value
        assert loaded.retry_count == 1

    def test_no_retry_when_max_retries_zero(self, tmp_store: TaskStore):
        """Tasks with max_retries=0 go straight to FAILED on exception."""
        executor = _StubExecutor(raise_exc=RuntimeError("boom"))
        scheduler = LocalScheduler(store=tmp_store, executor=executor)

        task = ArmedTask(
            task_id="no_retry",
            skill_name="old_style",
            trigger_type="interval",
            trigger_config={"interval_seconds": 300},
            state=TaskState.ARMED.value,
            next_due_at=time.time() - 1,
            max_retries=0,
            retry_count=0,
        )
        tmp_store.save(task)

        asyncio.get_event_loop().run_until_complete(
            scheduler._execute_task(task, time.time())
        )

        loaded = tmp_store.load("no_retry")
        assert loaded is not None
        assert loaded.state == TaskState.FAILED.value


class TestArmRetryDefaults:
    """Coordinator.arm() applies default retry settings from config."""

    def test_arm_uses_config_defaults(self, tmp_store: TaskStore):
        """arm() picks up default_max_retries and default_retry_backoff_s."""
        local_sched = AsyncMock()
        local_sched.register = AsyncMock()

        coordinator = TaskCoordinator(
            store=tmp_store,
            local_scheduler=local_sched,
            default_tier="local",
            default_max_retries=5,
            default_retry_backoff_s=30.0,
        )

        task = asyncio.get_event_loop().run_until_complete(
            coordinator.arm("my_skill", "5m")
        )
        assert task.max_retries == 5
        assert task.retry_backoff_s == 30.0

    def test_arm_per_task_overrides_config(self, tmp_store: TaskStore):
        """Per-task retry values override the config defaults."""
        local_sched = AsyncMock()
        local_sched.register = AsyncMock()

        coordinator = TaskCoordinator(
            store=tmp_store,
            local_scheduler=local_sched,
            default_tier="local",
            default_max_retries=5,
            default_retry_backoff_s=30.0,
        )

        task = asyncio.get_event_loop().run_until_complete(
            coordinator.arm("my_skill", "5m", max_retries=1, retry_backoff_s=10.0)
        )
        assert task.max_retries == 1
        assert task.retry_backoff_s == 10.0


# ---------------------------------------------------------------------------
# Store migration
# ---------------------------------------------------------------------------


class TestStoreMigration:
    """Idempotent column migration for retry fields."""

    def test_migration_is_idempotent(self, tmp_path: Path):
        """Creating TaskStore twice doesn't fail (columns already exist)."""
        db = tmp_path / "migrate.duckdb"
        store1 = TaskStore(db)
        store1.close()
        # Second creation triggers the same migration — should not raise
        store2 = TaskStore(db)
        store2.close()

    def test_retry_columns_present_after_migration(self, tmp_store: TaskStore):
        """New columns are queryable after migration."""
        task = ArmedTask(
            task_id="migration_test",
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            max_retries=7,
            retry_backoff_s=120.0,
        )
        tmp_store.save(task)
        loaded = tmp_store.load("migration_test")
        assert loaded.max_retries == 7
        assert loaded.retry_backoff_s == 120.0


# ---------------------------------------------------------------------------
# TUI payload builders
# ---------------------------------------------------------------------------


class TestSchedulePayloadBuilders:
    """TUI /schedule pause|resume|edit produce correct payloads."""

    def test_pause_payload(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """build_schedule_payload('pause <id>') sets state to paused."""
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        tmp_store.save(sample_task)

        class _FakeCtx:
            coordinator = None
            settings = type("S", (), {"duckdb_path": None})()

        ctx = _FakeCtx()
        ctx.coordinator = TaskCoordinator(store=tmp_store)
        result = build_schedule_payload(ctx, f"pause {sample_task.task_id}")
        assert result["ok"] is True
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded.state == TaskState.PAUSED.value

    def test_resume_payload(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """build_schedule_payload('resume <id>') re-arms the task."""
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        sample_task.state = TaskState.PAUSED.value
        tmp_store.save(sample_task)

        class _FakeCtx:
            coordinator = None
            settings = type("S", (), {"duckdb_path": None})()

        ctx = _FakeCtx()
        ctx.coordinator = TaskCoordinator(store=tmp_store)
        result = build_schedule_payload(ctx, f"resume {sample_task.task_id}")
        assert result["ok"] is True
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded.state == TaskState.ARMED.value

    def test_edit_payload(self, tmp_store: TaskStore, sample_task: ArmedTask):
        """build_schedule_payload('edit <id> 10m') updates the trigger."""
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        tmp_store.save(sample_task)

        class _FakeCtx:
            coordinator = None
            settings = type("S", (), {"duckdb_path": None})()

        ctx = _FakeCtx()
        ctx.coordinator = TaskCoordinator(store=tmp_store)
        result = build_schedule_payload(ctx, f"edit {sample_task.task_id} 10m")
        assert result["ok"] is True
        loaded = tmp_store.load(sample_task.task_id)
        assert loaded.trigger_config == {"interval_seconds": 600}

    def test_edit_missing_args(self):
        """build_schedule_payload('edit') without args returns error."""
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        class _FakeCtx:
            coordinator = None
            settings = type("S", (), {"duckdb_path": None})()

        result = build_schedule_payload(_FakeCtx(), "edit abc123")
        assert result["ok"] is False
        assert "Usage" in result["message"]

    def test_pause_missing_id(self):
        """build_schedule_payload('pause') without task_id returns error."""
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        class _FakeCtx:
            coordinator = None
            settings = type("S", (), {"duckdb_path": None})()

        result = build_schedule_payload(_FakeCtx(), "pause")
        assert result["ok"] is False
        assert "Usage" in result["message"]


# ---------------------------------------------------------------------------
# Config settings
# ---------------------------------------------------------------------------


class TestSchedulerConfigSettings:
    """New scheduler retry settings exist in the Settings dataclass."""

    def test_default_max_retries_exists(self):
        from leapflow.config import Settings
        s = Settings.__dataclass_fields__
        assert "scheduler_default_max_retries" in s
        assert s["scheduler_default_max_retries"].default == 2

    def test_default_retry_backoff_s_exists(self):
        from leapflow.config import Settings
        s = Settings.__dataclass_fields__
        assert "scheduler_default_retry_backoff_s" in s
        assert s["scheduler_default_retry_backoff_s"].default == 60.0

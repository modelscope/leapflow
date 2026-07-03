"""Local async scheduler — runs as background task in event loop.

Design principles:
- At-most-once: advance next_due BEFORE execute (crash-safe)
- Fast-forward: on startup, skip overdue tasks beyond grace period
- Non-blocking: tick runs in background, never blocks REPL
- Confidence gating: low-confidence tasks emit notification instead of executing
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.triggers import create_trigger
from leapflow.scheduler.types import ArmedTask, SkillExecutor, TaskState

logger = logging.getLogger(__name__)


class LocalScheduler:
    """Local async scheduler — runs as background task in event loop.

    Tick-based design: every ``tick_seconds`` (default 60s), the scheduler
    queries the TaskStore for due tasks and dispatches them through the
    SkillExecutor.
    """

    def __init__(
        self,
        store: TaskStore,
        executor: SkillExecutor,
        *,
        tick_seconds: int = 60,
        grace_seconds: float = 120.0,
    ) -> None:
        self._store = store
        self._executor = executor
        self._tick_seconds = tick_seconds
        self._grace_seconds = grace_seconds
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background tick loop."""
        self._running = True
        self._fast_forward()
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("LocalScheduler started (tick=%ds)", self._tick_seconds)

    async def stop(self) -> None:
        """Gracefully stop the tick loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LocalScheduler stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register(self, task: ArmedTask) -> None:
        """Register a new armed task."""
        self._store.save(task)
        logger.info(
            "Registered task %s (skill=%s, trigger=%s)",
            task.task_id[:8],
            task.skill_name,
            task.trigger_type,
        )

    async def cancel(self, task_id: str) -> None:
        """Cancel (suspend) a task."""
        self._store.update_state(task_id, TaskState.SUSPENDED.value)
        logger.info("Cancelled task %s", task_id[:8])

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Background loop: check and execute due tasks every tick."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e, exc_info=True)
            await asyncio.sleep(self._tick_seconds)

    async def _tick(self) -> None:
        """Single tick: find due tasks, advance, execute."""
        now = time.time()
        due_tasks = self._store.get_due_tasks(now)

        for task in due_tasks:
            await self._execute_task(task, now)

    async def _execute_task(self, task: ArmedTask, now: float) -> None:
        """Execute a single due task with at-most-once semantics."""
        # At-most-once: advance BEFORE execute
        trigger = create_trigger(
            task.trigger_type,
            task.trigger_config if isinstance(task.trigger_config, dict) else json.loads(task.trigger_config),
        )
        trigger.advance(now)
        new_due = trigger.next_due_at
        self._store.advance_next_due(task.task_id, new_due)

        # Execute
        try:
            self._store.update_state(task.task_id, TaskState.EXECUTING.value)

            parameters = (
                task.parameters
                if isinstance(task.parameters, dict)
                else json.loads(task.parameters)
            )
            result = await self._executor.execute(task.skill_name, parameters)
            self._store.increment_run_count(task.task_id)

            # Check max_runs exhaustion
            updated = self._store.load(task.task_id)
            if updated and updated.max_runs > 0 and updated.run_count >= updated.max_runs:
                self._store.update_state(task.task_id, TaskState.DONE.value)
                logger.info(
                    "Task %s completed (max_runs reached)", task.task_id[:8]
                )
            else:
                self._store.update_state(task.task_id, TaskState.ARMED.value)

            logger.info(
                "Task %s executed: ok=%s",
                task.task_id[:8],
                result.get("ok", False),
            )
        except Exception as e:
            self._store.update_state(task.task_id, TaskState.FAILED.value)
            logger.error("Task %s failed: %s", task.task_id[:8], e)

    # ------------------------------------------------------------------
    # Fast-forward
    # ------------------------------------------------------------------

    def _fast_forward(self) -> None:
        """On startup: advance overdue tasks past their grace period."""
        now = time.time()
        all_tasks = self._store.load_all()
        forwarded = 0

        for task in all_tasks:
            if task.state != TaskState.ARMED.value:
                continue
            if task.next_due_at <= 0:
                continue
            if now - task.next_due_at <= self._grace_seconds:
                continue

            # Overdue beyond grace — fast forward
            trigger = create_trigger(
                task.trigger_type,
                task.trigger_config if isinstance(task.trigger_config, dict) else json.loads(task.trigger_config),
            )
            trigger.advance(now)
            self._store.advance_next_due(task.task_id, trigger.next_due_at)
            forwarded += 1
            logger.info(
                "Fast-forwarded task %s to %.0f",
                task.task_id[:8],
                trigger.next_due_at,
            )

        if forwarded:
            logger.info("Fast-forwarded %d overdue tasks", forwarded)

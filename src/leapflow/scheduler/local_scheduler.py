# Copyright (c) Alibaba, Inc. and its affiliates.
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
from typing import Any, Callable, Optional

from leapflow.scheduler.execution_log import ExecutionLogStore
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.triggers import create_trigger
from leapflow.scheduler.types import ArmedTask, SkillExecutor, TaskState

logger = logging.getLogger(__name__)

# Type alias for the optional delivery callback.
# Signature: send_fn(platform, chat_id, message_text) -> None
DeliverySendFn = Callable[[str, str, str], Any]


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
        execution_log: Optional["ExecutionLogStore"] = None,
        send_fn: Optional[DeliverySendFn] = None,
        delivery_enabled: bool = False,
    ) -> None:
        self._store = store
        self._executor = executor
        self._tick_seconds = tick_seconds
        self._grace_seconds = grace_seconds
        self._execution_log = execution_log
        self._send_fn = send_fn
        self._delivery_enabled = delivery_enabled
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._running = False
        self._wake_event: asyncio.Event = asyncio.Event()
        self._last_wake_time: float = 0.0

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

    def wake(self) -> None:
        """Signal the scheduler to check for due tasks immediately.

        Called by EventBridge when an event-driven trigger fires,
        reducing latency from poll-interval to near-zero.
        Thread-safe: asyncio.Event.set() is safe to call from any thread.
        """
        now = time.monotonic()
        if now - self._last_wake_time < 1.0:
            return
        self._last_wake_time = now
        self._wake_event.set()

    async def _tick_loop(self) -> None:
        """Background loop: check and execute due tasks every tick."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Scheduler tick error: %s", e, exc_info=True)
            # Wait for either the tick interval or an external wake signal
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=self._tick_seconds)
            except asyncio.TimeoutError:
                pass
            self._wake_event.clear()

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
        execution_id: Optional[str] = None
        t_start = time.time()
        try:
            # Record execution start (contained — logging failures never crash the tick)
            if self._execution_log is not None:
                try:
                    execution_id = self._execution_log.record_start(
                        task_id=task.task_id,
                        trigger_type=task.trigger_type,
                    )
                except Exception:
                    logger.debug("Failed to record execution start for %s", task.task_id[:8], exc_info=True)

            self._store.update_state(task.task_id, TaskState.EXECUTING.value)

            parameters = (
                task.parameters
                if isinstance(task.parameters, dict)
                else json.loads(task.parameters)
            )
            result = await self._executor.execute(task.skill_name, parameters)
            self._store.increment_run_count(task.task_id)

            ok = result.get("ok", False)
            duration = time.time() - t_start

            # Check result-level failure (result returned ok=False)
            if not ok:
                if task.max_retries > 0:
                    reloaded = self._store.load(task.task_id)
                    current_retry = reloaded.retry_count if reloaded else 0
                    if current_retry < task.max_retries:
                        self._retry_task(task, current_retry, execution_id)
                        return
                    # Retries exhausted from soft failure
                    self._store.update_task(task.task_id, state=TaskState.FAILED.value, retry_count=0)
                    logger.warning(
                        "Task %s failed after %d retries (soft failure)",
                        task.task_id[:8], task.max_retries,
                    )
                    if self._execution_log is not None and execution_id is not None:
                        try:
                            self._execution_log.record_finish(
                                execution_id, "failed", result_summary="retries exhausted",
                            )
                        except Exception:
                            pass
                    self._attempt_delivery(
                        task, success=False, error="retries exhausted", duration_s=duration,
                    )
                    return
                else:
                    # No retries configured: mark as FAILED immediately
                    self._store.update_state(task.task_id, TaskState.FAILED.value)
                    logger.error(
                        "Task %s execution returned ok=False with no retries configured",
                        task.task_id[:8],
                    )
                    if self._execution_log is not None and execution_id is not None:
                        try:
                            self._execution_log.record_finish(
                                execution_id, "failed",
                                result_summary=str(result.get("output", ""))[:200],
                            )
                        except Exception:
                            logger.debug("Failed to record execution failure for %s", task.task_id[:8], exc_info=True)
                    self._attempt_delivery(
                        task, success=False,
                        error="ok=False (no retries configured)",
                        duration_s=duration,
                    )
                    return

            # Reset retry_count on success
            if ok and task.retry_count > 0:
                self._store.update_task(task.task_id, retry_count=0)

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
                ok,
            )

            # Record success (contained)
            output_summary = str(result.get("output", ""))[:200] if ok else ""
            if self._execution_log is not None and execution_id is not None:
                try:
                    self._execution_log.record_finish(
                        execution_id, "success", result_summary=output_summary,
                    )
                except Exception:
                    logger.debug("Failed to record execution finish for %s", task.task_id[:8], exc_info=True)

            # Post-execution delivery
            self._attempt_delivery(
                task, success=ok, summary=output_summary, duration_s=duration,
            )
        except Exception as e:
            duration = time.time() - t_start
            # Hard exception path: retry if budget allows
            if task.max_retries > 0:
                reloaded = self._store.load(task.task_id)
                current_retry = reloaded.retry_count if reloaded else 0
                if current_retry < task.max_retries:
                    self._retry_task(task, current_retry, execution_id, error=str(e))
                    return
                # Retries exhausted
                self._store.update_task(task.task_id, state=TaskState.FAILED.value, retry_count=0)
                logger.error(
                    "Task %s failed after %d retries: %s",
                    task.task_id[:8], task.max_retries, e,
                )
            else:
                self._store.update_state(task.task_id, TaskState.FAILED.value)
                logger.error("Task %s failed: %s", task.task_id[:8], e)

            # Record failure (contained)
            if self._execution_log is not None and execution_id is not None:
                try:
                    self._execution_log.record_finish(
                        execution_id, "failed", error=str(e)[:500],
                    )
                except Exception:
                    logger.debug("Failed to record execution failure for %s", task.task_id[:8], exc_info=True)

            # Post-execution delivery (failure)
            self._attempt_delivery(
                task, success=False, error=str(e)[:200], duration_s=duration,
            )

    def _retry_task(
        self,
        task: ArmedTask,
        current_retry: int,
        execution_id: Optional[str] = None,
        error: str = "",
    ) -> None:
        """Schedule a retry with exponential backoff."""
        new_retry = current_retry + 1
        backoff = task.retry_backoff_s * (2 ** current_retry)
        retry_due = time.time() + backoff
        self._store.update_task(
            task.task_id,
            retry_count=new_retry,
            next_due_at=retry_due,
            state=TaskState.ARMED.value,
        )
        logger.info(
            "Task %s retry %d/%d in %.0fs",
            task.task_id[:8], new_retry, task.max_retries, backoff,
        )
        # Record retry (contained)
        if self._execution_log is not None and execution_id is not None:
            try:
                self._execution_log.record_finish(
                    execution_id,
                    "retry",
                    result_summary=f"retry {new_retry}/{task.max_retries}",
                    error=error[:500] if error else "",
                )
            except Exception:
                logger.debug("Failed to record retry for %s", task.task_id[:8], exc_info=True)

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

    # ------------------------------------------------------------------
    # Post-execution delivery
    # ------------------------------------------------------------------

    def _attempt_delivery(
        self,
        task: ArmedTask,
        *,
        success: bool,
        summary: str = "",
        error: str = "",
        duration_s: float = 0.0,
    ) -> None:
        """Attempt result delivery to the task's delivery_target (non-fatal).

        Skipped when delivery is disabled, no send_fn is wired, or the task has
        no ``delivery_target`` in its parameters.
        """
        if not self._delivery_enabled or self._send_fn is None:
            return

        params = task.parameters if isinstance(task.parameters, dict) else {}
        target = params.get("delivery_target")
        if not isinstance(target, dict):
            return
        platform = str(target.get("platform", "")).strip()
        chat_id = str(target.get("chat_id", "")).strip()
        if not platform or not chat_id:
            return

        status_label = "✅ Success" if success else "❌ Failed"
        detail = summary[:200] if success else (error[:200] if error else "unknown")
        dur_str = f"{duration_s:.1f}s" if duration_s > 0 else "-"
        message = (
            f"[Scheduler] {task.skill_name} ({task.task_id[:8]})\n"
            f"Status: {status_label}\n"
            f"Duration: {dur_str}\n"
            f"Detail: {detail}"
        )

        try:
            result = self._send_fn(platform, chat_id, message)
            # Handle coroutine return from async send_fn
            if asyncio.iscoroutine(result):
                asyncio.ensure_future(result)
            logger.debug("Delivery sent for task %s", task.task_id[:8])
        except Exception:
            # Delivery failure is NON-FATAL per design.
            logger.warning(
                "Delivery failed for task %s (non-fatal)",
                task.task_id[:8],
                exc_info=True,
            )

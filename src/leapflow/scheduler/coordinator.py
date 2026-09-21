# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified task orchestrator — routes armed tasks to local or cloud execution.

Tier decision heuristic:
- AUTO: interval < 1h → local; interval >= 1h or condition → cloud
- LOCAL/CLOUD: explicit override

Both tiers share the same TaskStore as single source of truth.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, List, Optional

from leapflow.scheduler.execution_log import ExecutionLogStore
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.triggers import create_trigger
from leapflow.scheduler.types import (
    ArmedTask,
    ExecutionTier,
    SchedulerExecutionMode,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger expression parser
# ---------------------------------------------------------------------------

_INTERVAL_PATTERN = re.compile(
    r"^(?:every\s+)?(\d+(?:\.\d+)?)\s*([smhd]|sec|min|hour|hours|day|days|minutes?)$",
    re.IGNORECASE,
)
_CRON_PATTERN = re.compile(r"^(\S+\s+){4}\S+$")  # 5 space-separated fields
_EVENT_PREFIX = "event:"
_CONDITION_PREFIX = "condition:"

_UNIT_TO_SECONDS = {
    "s": 1.0, "sec": 1.0,
    "m": 60.0, "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}


def parse_trigger_expression(expr: str) -> tuple[str, dict]:
    """Parse a human-friendly trigger expression into (trigger_type, config).

    Supported formats:
    - "30m" / "2h" / "1d"         → IntervalTrigger
    - "every 5m"                  → IntervalTrigger
    - "0 9 * * *"                 → CronTrigger (5-field cron)
    - "event:ci.passed"           → EventTrigger
    - "condition:file_count > 50" → ConditionTrigger

    Returns:
        Tuple of (trigger_type, trigger_config dict).

    Raises:
        ValueError: If expression cannot be parsed.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Trigger expression cannot be empty.")

    # Event trigger
    if expr.lower().startswith(_EVENT_PREFIX):
        event_name = expr[len(_EVENT_PREFIX):].strip()
        if not event_name:
            raise ValueError("Event trigger must specify an event name, e.g. 'event:ci.passed'")
        return "event", {"event_pattern": event_name}

    # Condition trigger
    if expr.lower().startswith(_CONDITION_PREFIX):
        condition_expr = expr[len(_CONDITION_PREFIX):].strip()
        if not condition_expr:
            raise ValueError(
                "Condition trigger must specify an expression, e.g. 'condition:file_count > 50'"
            )
        return "condition", {"expression": condition_expr}

    # Cron trigger (5 fields separated by spaces)
    if _CRON_PATTERN.match(expr):
        return "cron", {"expression": expr}

    # Interval trigger
    match = _INTERVAL_PATTERN.match(expr)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = _UNIT_TO_SECONDS.get(unit)
        if multiplier is None:
            raise ValueError(f"Unknown interval unit: {unit!r}")
        interval_seconds = value * multiplier
        return "interval", {"interval_seconds": interval_seconds}

    raise ValueError(
        f"Cannot parse trigger expression: {expr!r}. "
        f"Supported formats: '30m', 'every 5m', '0 9 * * *', "
        f"'event:<name>', 'condition:<expr>'"
    )


# ---------------------------------------------------------------------------
# TaskCoordinator
# ---------------------------------------------------------------------------


class _RoutingExecutor:
    """Thin dispatcher that routes to the agent executor or the default.

    Satisfies the ``SkillExecutor`` Protocol.  The ``LocalScheduler`` holds one
    executor; this wrapper lets it transparently delegate agent-mode tasks to
    ``AgentSkillExecutor`` while keeping the existing call site unchanged.

    Routing covers every ``execution_mode`` value with no gaps:
    - ``"agent"`` (:attr:`SchedulerExecutionMode.AGENT`) → the lazily built
      agent executor, but only when an ``agent_factory`` is wired.
    - anything else — ``"script"``, ``None``, a missing key, or an unknown
      string — falls through to the default (script) executor.

    The agent executor is built once on first use and cached: scheduler ticks
    are a cold path, but a task may fire many times and must not pay
    construction cost or spawn a fresh executor on every fire.
    """

    def __init__(
        self,
        default: Any,
        agent_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._default = default
        self._agent_factory = agent_factory
        self._agent: Optional[Any] = None

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        if (
            isinstance(parameters, dict)
            and parameters.get("execution_mode") == SchedulerExecutionMode.AGENT.value
            and self._agent_factory is not None
        ):
            if self._agent is None:
                self._agent = self._agent_factory()
            return await self._agent.execute(skill_name, parameters)
        return await self._default.execute(skill_name, parameters)


class TaskCoordinator:
    """Unified task orchestrator — routes armed tasks to local or cloud execution.

    Tier decision heuristic:
    - AUTO: interval < 1h → local; interval >= 1h or condition/event → cloud
    - LOCAL/CLOUD: explicit override

    Both tiers share the same TaskStore as single source of truth.
    """

    def __init__(
        self,
        store: TaskStore,
        local_scheduler: Optional["LocalScheduler"] = None,
        cloud_dispatcher: Optional["CloudDispatcher"] = None,
        default_tier: str = "auto",
        execution_log: Optional[ExecutionLogStore] = None,
        agent_executor_factory: Optional[Callable[[], Any]] = None,
        default_max_retries: int = 0,
        default_retry_backoff_s: float = 60.0,
    ) -> None:
        self._store = store
        self._local = local_scheduler
        self._cloud = cloud_dispatcher
        self._default_tier = default_tier
        self._execution_log = execution_log
        self._agent_executor_factory = agent_executor_factory
        self._default_max_retries = default_max_retries
        self._default_retry_backoff_s = default_retry_backoff_s

    def wrap_executor(self, default_executor: Any) -> "_RoutingExecutor":
        """Wrap a default executor with agent-mode routing.

        Returns a ``_RoutingExecutor`` that satisfies the ``SkillExecutor``
        Protocol and transparently dispatches ``execution_mode=agent`` tasks
        to an ``AgentSkillExecutor``.
        """
        return _RoutingExecutor(default_executor, self._agent_executor_factory)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def arm(
        self,
        skill_name: str,
        trigger_expr: str,
        *,
        execution_tier: str = "auto",
        max_runs: int = -1,
        parameters: Optional[dict] = None,
        context_snapshot: Optional[dict] = None,
        max_retries: Optional[int] = None,
        retry_backoff_s: Optional[float] = None,
    ) -> ArmedTask:
        """Create and register an armed task.

        1. Parse trigger expression → Trigger object
        2. Decide execution tier
        3. Create ArmedTask with unique ID
        4. Persist to store
        5. Route to local_scheduler.register() or cloud_dispatcher.deploy()
        """
        # 1. Parse trigger
        trigger_type, trigger_config = parse_trigger_expression(trigger_expr)

        # 2. Decide tier
        tier = execution_tier if execution_tier != "auto" else self._default_tier
        if tier == "auto":
            tier = self._decide_tier(trigger_type, trigger_config)

        # 3. Validate tier availability
        if tier == ExecutionTier.LOCAL.value and self._local is None:
            raise RuntimeError(
                "Local scheduler not available. Use --cloud or configure a local scheduler."
            )
        if tier == ExecutionTier.CLOUD.value and self._cloud is None:
            raise RuntimeError(
                "Cloud dispatcher not available. Use --local or configure a cloud backend."
            )

        # 4. Create trigger instance to compute next_due_at
        now = time.time()
        trigger = create_trigger(trigger_type, trigger_config)
        trigger.advance(now)

        # 5. Resolve retry defaults from config if not specified per-task
        effective_max_retries = max_retries if max_retries is not None else self._default_max_retries
        effective_backoff = retry_backoff_s if retry_backoff_s is not None else self._default_retry_backoff_s

        # 6. Create ArmedTask
        task = ArmedTask(
            skill_name=skill_name,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            state=TaskState.ARMED.value,
            execution_tier=tier,
            context_snapshot=context_snapshot or {},
            parameters=parameters or {},
            max_runs=max_runs,
            next_due_at=trigger.next_due_at,
            max_retries=effective_max_retries,
            retry_backoff_s=effective_backoff,
        )

        # 7. Persist
        self._store.save(task)

        # 8. Route to execution backend
        if tier == ExecutionTier.LOCAL.value:
            assert self._local is not None
            await self._local.register(task)
            logger.info("Armed task %s → local (skill=%s)", task.task_id[:8], skill_name)
        elif tier == ExecutionTier.CLOUD.value:
            assert self._cloud is not None
            worker_id = await self._cloud.deploy(task)
            task.cloud_worker_id = worker_id
            self._store.save(task)
            logger.info("Armed task %s → cloud (skill=%s)", task.task_id[:8], skill_name)

        return task

    async def cancel(self, task_id: str) -> None:
        """Cancel a task (local: suspend, cloud: stop worker)."""
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.execution_tier == ExecutionTier.LOCAL.value and self._local:
            await self._local.cancel(task_id)
        elif task.execution_tier == ExecutionTier.CLOUD.value and self._cloud:
            if task.cloud_worker_id:
                await self._cloud.stop(task.cloud_worker_id)
            self._store.update_state(task_id, TaskState.SUSPENDED.value)
        else:
            self._store.update_state(task_id, TaskState.SUSPENDED.value)

        logger.info("Cancelled task %s", task_id[:8])

    async def pause_task(self, task_id: str) -> None:
        """Pause a task — stops it from firing without cancelling.

        The task stays in the store with state PAUSED; ``get_due_tasks``
        already filters on ``state = 'armed'``, so paused tasks are
        naturally skipped.
        """
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        self._store.update_state(task_id, TaskState.PAUSED.value)
        logger.info("Paused task %s", task_id[:8])

    async def resume_task(self, task_id: str) -> None:
        """Resume a paused task — re-arm it and recalculate next_due."""
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        # Recalculate next_due from the trigger
        now = time.time()
        trigger = create_trigger(
            task.trigger_type,
            task.trigger_config if isinstance(task.trigger_config, dict) else {},
        )
        trigger.advance(now)
        self._store.update_task(
            task_id,
            state=TaskState.ARMED.value,
            next_due_at=trigger.next_due_at,
        )
        logger.info("Resumed task %s (next_due=%.0f)", task_id[:8], trigger.next_due_at)

    async def update_task(
        self,
        task_id: str,
        *,
        trigger_expr: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> ArmedTask:
        """Update a task's trigger expression and/or payload."""
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        fields: dict = {}
        if payload is not None:
            fields["parameters"] = payload

        if trigger_expr is not None:
            trigger_type, trigger_config = parse_trigger_expression(trigger_expr)
            now = time.time()
            trigger = create_trigger(trigger_type, trigger_config)
            trigger.advance(now)
            fields["trigger_type"] = trigger_type
            fields["trigger_config"] = trigger_config
            fields["next_due_at"] = trigger.next_due_at

        if fields:
            self._store.update_task(task_id, **fields)

        updated = self._store.load(task_id)
        assert updated is not None
        logger.info("Updated task %s", task_id[:8])
        return updated

    async def status(self, task_id: str) -> TaskStatus:
        """Unified status query."""
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        is_running = task.state == TaskState.EXECUTING.value
        logs_tail: List[str] = []

        if task.execution_tier == ExecutionTier.CLOUD.value and self._cloud and task.cloud_worker_id:
            try:
                cloud_status = await self._cloud.status(task.cloud_worker_id)
                is_running = cloud_status in ("running", "building")
            except Exception:
                pass

        return TaskStatus(task=task, is_running=is_running, logs_tail=logs_tail)

    async def list_tasks(self) -> List[ArmedTask]:
        """List all tasks from store."""
        return self._store.load_all()

    async def logs(self, task_id: str, tail: int = 50) -> List[str]:
        """Get logs (local: from execution log store, cloud: from Studio logs)."""
        task = self._store.load(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        if task.execution_tier == ExecutionTier.CLOUD.value and self._cloud and task.cloud_worker_id:
            return await self._cloud.logs(task.cloud_worker_id, tail=tail)

        # Local tasks: pull from execution log store
        if self._execution_log is not None:
            try:
                records = self._execution_log.get_history(task_id=task_id, limit=tail)
                if records:
                    lines: List[str] = []
                    for r in records:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.started_at))
                        detail = r.result_summary or r.error or ""
                        lines.append(f"[{ts}] {r.status}" + (f" — {detail}" if detail else ""))
                    return lines
            except Exception:
                logger.debug("Failed to read execution log for %s", task_id[:8], exc_info=True)

        return [f"[local] Task {task_id[:8]}: state={task.state}, runs={task.run_count}"]

    def get_execution_history(
        self,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> List:
        """Return recent execution log records (newest first).

        Delegates to the injected :class:`ExecutionLogStore`.  Returns an
        empty list when no store is available.
        """
        if self._execution_log is None:
            return []
        try:
            return self._execution_log.get_history(task_id=task_id, limit=limit)
        except Exception:
            logger.debug("Failed to read execution history", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Tier decision heuristic
    # ------------------------------------------------------------------

    def _decide_tier(self, trigger_type: str, trigger_config: dict) -> str:
        """Heuristic tier decision.

        - condition/event → cloud (needs persistent monitoring)
        - interval >= 3600s → cloud
        - interval < 3600s → local
        - cron with daily+ frequency → cloud
        """
        if trigger_type in ("condition", "event"):
            # Needs persistent monitoring — prefer cloud
            return ExecutionTier.CLOUD.value if self._cloud else ExecutionTier.LOCAL.value

        if trigger_type == "interval":
            interval_s = trigger_config.get("interval_seconds", 0)
            if interval_s >= 3600:
                return ExecutionTier.CLOUD.value if self._cloud else ExecutionTier.LOCAL.value
            return ExecutionTier.LOCAL.value if self._local else ExecutionTier.CLOUD.value

        if trigger_type == "cron":
            # Heuristic: check if frequency is daily or less often
            cron_expr = trigger_config.get("expression", "")
            parts = cron_expr.split()
            if len(parts) >= 5:
                # If minute and hour are specific (not */N), it's likely daily+
                minute_field, hour_field = parts[0], parts[1]
                if "*" not in minute_field and "*" not in hour_field:
                    return ExecutionTier.CLOUD.value if self._cloud else ExecutionTier.LOCAL.value
            return ExecutionTier.LOCAL.value if self._local else ExecutionTier.CLOUD.value

        # Default: local if available
        return ExecutionTier.LOCAL.value if self._local else ExecutionTier.CLOUD.value


# Deferred imports for type hints
from leapflow.scheduler.cloud_dispatcher import CloudDispatcher  # noqa: E402
from leapflow.scheduler.local_scheduler import LocalScheduler  # noqa: E402

__all__ = ["TaskCoordinator", "parse_trigger_expression"]

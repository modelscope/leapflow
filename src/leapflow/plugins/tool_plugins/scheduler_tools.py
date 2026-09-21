# Copyright (c) Alibaba, Inc. and its affiliates.
"""Scheduler tools plugin — agent-facing scheduled task management.

Exposes six tools (create / list / status / pause / resume / cancel) that
delegate to :class:`TaskCoordinator`.  The coordinator is injected at runtime
via ``bind_runtime(scheduler=...)``; handlers return a structured refusal when
the dependency is absent.
"""

from __future__ import annotations

import logging
from typing import Any

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured refusal (AGENTS.md: handler whose dep was never bound)
# ---------------------------------------------------------------------------

_UNBOUND_REFUSAL: dict[str, Any] = {
    "ok": False,
    "error": "scheduler_not_available",
    "message": (
        "The scheduler runtime is not available. "
        "Ensure the scheduler is enabled in settings and the daemon is running."
    ),
}


def _task_execution_mode(task: Any) -> str:
    """Resolve a task's execution mode from its parameters (default 'script').

    The mode is stored in ``parameters['execution_mode']``; a missing or
    malformed value falls back to ``'script'`` — the same default the
    coordinator's router applies.
    """
    params = getattr(task, "parameters", None)
    mode = params.get("execution_mode") if isinstance(params, dict) else None
    return str(mode) if mode else "script"


class SchedulerToolsPlugin:
    """Agent-facing scheduled task management tools.

    All six tools delegate to :class:`TaskCoordinator`; the coordinator is
    received through ``bind_runtime(scheduler=<TaskCoordinator>)``.
    """

    def __init__(self) -> None:
        self._scheduler: Any = None  # TaskCoordinator — injected late

    # -- Protocol properties ------------------------------------------------

    @property
    def plugin_id(self) -> str:
        return "scheduler_tools"

    @property
    def category(self) -> str:
        return "scheduler"

    @property
    def dependencies(self) -> list[str]:
        return ["scheduler"]

    def bind_runtime(self, **deps: Any) -> None:
        if "scheduler" in deps:
            self._scheduler = deps["scheduler"]

    # -- Tools --------------------------------------------------------------

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            self._tool_create(),
            self._tool_list(),
            self._tool_status(),
            self._tool_pause(),
            self._tool_resume(),
            self._tool_cancel(),
        ]

    # -- Individual ToolMetadata builders -----------------------------------

    def _tool_create(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_create",
            description=(
                "Create a new scheduled task. Specify a trigger expression "
                "(e.g. '30m', 'every 2h', '0 9 * * *') and the instruction "
                "to execute on each trigger. Optionally provide an execution "
                "mode and a delivery target to receive notifications."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "trigger_expression": {
                        "type": "string",
                        "description": (
                            "When to fire: '30m', 'every 2h', '0 9 * * *', "
                            "'event:<name>', or 'condition:<expr>'"
                        ),
                    },
                    "instruction": {
                        "type": "string",
                        "description": "The instruction or skill name to execute on each trigger.",
                    },
                    "execution_mode": {
                        "type": "string",
                        "enum": ["script", "agent"],
                        "description": "Execution mode: 'script' (default) or 'agent'.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Max retry attempts on failure (0 = no retries).",
                    },
                    "delivery_target": {
                        "type": "object",
                        "properties": {
                            "platform": {
                                "type": "string",
                                "description": "Gateway platform id, e.g. 'feishu', 'slack'.",
                            },
                            "chat_id": {
                                "type": "string",
                                "description": "Target chat/channel id for result delivery.",
                            },
                        },
                        "required": ["platform", "chat_id"],
                        "description": "Optional delivery destination for execution results.",
                    },
                },
                "required": ["trigger_expression", "instruction"],
            },
            handler=self._handle_create,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "medium",
            },
            mutates_state=True,
            execution_policy="mutating_once",
            provides_capabilities=("scheduler.manage",),
            requires_platform_capabilities=("file.ops",),
        )

    def _tool_list(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_list",
            description=(
                "List all scheduled tasks with their current state, trigger, "
                "and next due time."
            ),
            parameters_schema={"type": "object", "properties": {}},
            handler=self._handle_list,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "safe",
            },
            execution_policy="read_only",
            provides_capabilities=("scheduler.read",),
        )

    def _tool_status(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_status",
            description=(
                "Get detailed status and recent execution history for a "
                "scheduled task."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task id (or unique prefix) to inspect.",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_status,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "safe",
            },
            execution_policy="read_only",
            provides_capabilities=("scheduler.read",),
        )

    def _tool_pause(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_pause",
            description="Pause a scheduled task so it stops firing without being cancelled.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task id to pause.",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_pause,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "low",
            },
            mutates_state=True,
            execution_policy="mutating_idempotent",
            provides_capabilities=("scheduler.manage",),
            requires_platform_capabilities=("file.ops",),
        )

    def _tool_resume(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_resume",
            description="Resume a paused scheduled task — re-arms it and recalculates next due time.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task id to resume.",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_resume,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "low",
            },
            mutates_state=True,
            execution_policy="mutating_idempotent",
            provides_capabilities=("scheduler.manage",),
            requires_platform_capabilities=("file.ops",),
        )

    def _tool_cancel(self) -> ToolMetadata:
        return ToolMetadata(
            name="schedule_cancel",
            description="Cancel a scheduled task permanently.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task id to cancel.",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_cancel,
            x_leapflow={
                "category": "scheduler",
                "risk_level": "medium",
            },
            mutates_state=True,
            execution_policy="mutating_once",
            provides_capabilities=("scheduler.manage",),
            requires_platform_capabilities=("file.ops",),
        )

    # -- Handler implementations --------------------------------------------

    async def _handle_create(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)

        trigger_expr = kwargs.get("trigger_expression", "")
        instruction = kwargs.get("instruction", "")
        if not trigger_expr or not instruction:
            return {
                "ok": False,
                "error": "missing_required_fields",
                "message": "Both 'trigger_expression' and 'instruction' are required.",
            }

        execution_mode = kwargs.get("execution_mode", "script")
        max_retries = kwargs.get("max_retries")
        delivery_target = kwargs.get("delivery_target")

        parameters: dict[str, Any] = {"instruction": instruction}
        if execution_mode:
            parameters["execution_mode"] = execution_mode
        if delivery_target:
            parameters["delivery_target"] = delivery_target

        try:
            task = await self._scheduler.arm(
                skill_name=instruction,
                trigger_expr=trigger_expr,
                parameters=parameters,
                max_retries=max_retries,
            )
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": "arm_failed", "message": str(exc)}

        return {
            "ok": True,
            "task_id": task.task_id,
            "state": task.state,
            "trigger_type": task.trigger_type,
            "next_due_at": task.next_due_at,
            "message": f"Task {task.task_id[:8]} created ({task.trigger_type}).",
        }

    async def _handle_list(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)
        try:
            tasks = await self._scheduler.list_tasks()
        except Exception as exc:
            return {"ok": False, "error": "list_failed", "message": str(exc)}

        return {
            "ok": True,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "skill_name": t.skill_name,
                    "state": t.state,
                    "execution_mode": _task_execution_mode(t),
                    "trigger_type": t.trigger_type,
                    "next_due_at": t.next_due_at,
                    "run_count": t.run_count,
                }
                for t in tasks
            ],
            "count": len(tasks),
        }

    async def _handle_status(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return {"ok": False, "error": "missing_task_id", "message": "'task_id' is required."}
        try:
            status = await self._scheduler.status(task_id)
            history = self._scheduler.get_execution_history(task_id=task_id, limit=10)
        except ValueError as exc:
            return {"ok": False, "error": "not_found", "message": str(exc)}

        history_items = []
        for r in history:
            history_items.append({
                "execution_id": getattr(r, "execution_id", ""),
                "status": getattr(r, "status", ""),
                "started_at": getattr(r, "started_at", 0),
                "result_summary": getattr(r, "result_summary", ""),
                "error": getattr(r, "error", ""),
            })

        t = status.task
        return {
            "ok": True,
            "task_id": t.task_id,
            "skill_name": t.skill_name,
            "state": t.state,
            "execution_mode": _task_execution_mode(t),
            "trigger_type": t.trigger_type,
            "next_due_at": t.next_due_at,
            "run_count": t.run_count,
            "max_runs": t.max_runs,
            "is_running": status.is_running,
            "retry_count": t.retry_count,
            "max_retries": t.max_retries,
            "recent_history": history_items,
        }

    async def _handle_pause(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return {"ok": False, "error": "missing_task_id", "message": "'task_id' is required."}
        try:
            await self._scheduler.pause_task(task_id)
        except ValueError as exc:
            return {"ok": False, "error": "not_found", "message": str(exc)}
        return {"ok": True, "message": f"Task {task_id[:8]} paused."}

    async def _handle_resume(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return {"ok": False, "error": "missing_task_id", "message": "'task_id' is required."}
        try:
            await self._scheduler.resume_task(task_id)
        except ValueError as exc:
            return {"ok": False, "error": "not_found", "message": str(exc)}
        return {"ok": True, "message": f"Task {task_id[:8]} resumed."}

    async def _handle_cancel(self, **kwargs: Any) -> dict[str, Any]:
        if self._scheduler is None:
            return dict(_UNBOUND_REFUSAL)
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return {"ok": False, "error": "missing_task_id", "message": "'task_id' is required."}
        try:
            await self._scheduler.cancel(task_id)
        except ValueError as exc:
            return {"ok": False, "error": "not_found", "message": str(exc)}
        return {"ok": True, "message": f"Task {task_id[:8]} cancelled."}


# Module-level instance for plugin discovery
plugin = SchedulerToolsPlugin()

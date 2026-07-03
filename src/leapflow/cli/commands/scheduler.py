"""Scheduler CLI commands — arm tasks and manage scheduled execution.

Provides ``leap arm`` and ``leap tasks`` subcommands for the interactive REPL.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from leapflow.cli.context import Context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds <= 0:
        return "now"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"in {int(seconds / 60)}m"
    if seconds < 86400:
        h = int(seconds / 3600)
        m = int((seconds % 3600) / 60)
        return f"in {h}h{m}m" if m else f"in {h}h"
    d = int(seconds / 86400)
    return f"in {d}d"


def _format_trigger(task) -> str:
    """Format trigger info for display."""
    if task.trigger_type == "interval":
        sec = task.trigger_config.get("interval_seconds", 0)
        if sec < 60:
            return f"every {int(sec)}s"
        if sec < 3600:
            return f"every {int(sec / 60)}m"
        if sec < 86400:
            return f"every {int(sec / 3600)}h"
        return f"every {int(sec / 86400)}d"
    if task.trigger_type == "cron":
        return task.trigger_config.get("expression", "cron")
    if task.trigger_type == "event":
        return f"event:{task.trigger_config.get('event_pattern', '?')}"
    if task.trigger_type == "condition":
        expr = task.trigger_config.get("expression", "?")
        return f"cond:{expr[:20]}"
    return task.trigger_type


def _get_coordinator(ctx: "Context"):
    """Get or create a TaskCoordinator from context."""
    from leapflow.scheduler.coordinator import TaskCoordinator
    from leapflow.scheduler.local_scheduler import LocalScheduler
    from leapflow.scheduler.store import TaskStore

    # Use existing coordinator if available
    if hasattr(ctx, "coordinator") and ctx.coordinator is not None:
        return ctx.coordinator

    # Build one from settings
    db_path = ctx.settings.data_dir / "scheduler.duckdb"
    store = TaskStore(db_path)

    # Local scheduler with a simple skill executor
    class _SimpleExecutor:
        """Minimal skill executor for scheduled tasks."""

        async def execute(self, skill_name: str, parameters: dict) -> dict:
            # Try to execute via session if available
            if ctx.session:
                try:
                    result = await ctx.session.execute_skill(skill_name, params=parameters)
                    return {"ok": True, "output": str(result)[:200]}
                except Exception as e:
                    return {"ok": False, "error": str(e)}
            return {"ok": False, "error": "No session available"}

    local_scheduler = LocalScheduler(
        store=store,
        executor=_SimpleExecutor(),
        tick_seconds=ctx.settings.scheduler_tick_seconds,
        grace_seconds=ctx.settings.scheduler_grace_seconds,
    )

    # Cloud dispatcher (optional, only if compute backend available)
    cloud_dispatcher = None
    try:
        from leapflow.scheduler.cloud_dispatcher import CloudDispatcher
        from leapflow.scheduler.compute.modelscope_studio import ModelScopeStudioBackend
        from leapflow.scheduler.worker_packager import WorkerPackager

        backend = ModelScopeStudioBackend()
        packager = WorkerPackager()
        cloud_dispatcher = CloudDispatcher(backend, packager)
    except (ImportError, Exception):
        pass  # Cloud not available — local-only mode

    coordinator = TaskCoordinator(
        store=store,
        local_scheduler=local_scheduler,
        cloud_dispatcher=cloud_dispatcher,
        default_tier=ctx.settings.scheduler_default_tier,
    )
    ctx.coordinator = coordinator
    return coordinator


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_arm(ctx: "Context", args: List[str]) -> int:
    """leap arm <skill> --trigger "<expr>" [--local|--cloud] [--max-runs N]

    Arm a skill for scheduled execution.
    """
    if not args or "--help" in args or "-h" in args:
        print("Usage: arm <skill> --trigger \"<expr>\" [--local|--cloud] [--max-runs N]")
        print()
        print("Trigger formats:")
        print("  30m / 2h / 1d         — interval")
        print("  every 5m              — interval")
        print("  0 9 * * *             — cron (5 fields)")
        print("  event:ci.passed       — event")
        print("  condition:expr > val  — condition")
        print()
        print("Options:")
        print("  --local       Force local execution")
        print("  --cloud       Force cloud execution")
        print("  --max-runs N  Stop after N executions (-1 = unlimited)")
        return 0

    # Parse arguments
    skill_name = args[0]
    trigger_expr = ""
    execution_tier = "auto"
    max_runs = -1

    i = 1
    while i < len(args):
        if args[i] == "--trigger" and i + 1 < len(args):
            trigger_expr = args[i + 1]
            i += 2
        elif args[i] == "--local":
            execution_tier = "local"
            i += 1
        elif args[i] == "--cloud":
            execution_tier = "cloud"
            i += 1
        elif args[i] == "--max-runs" and i + 1 < len(args):
            try:
                max_runs = int(args[i + 1])
            except ValueError:
                print(f"  Error: --max-runs must be an integer, got '{args[i + 1]}'")
                return 1
            i += 2
        else:
            # If no --trigger flag, treat remaining as trigger expression
            if not trigger_expr:
                trigger_expr = " ".join(args[i:])
                break
            i += 1

    if not trigger_expr:
        print("  Error: trigger expression required.")
        print("  Usage: arm <skill> --trigger \"30m\"")
        return 1

    try:
        coordinator = _get_coordinator(ctx)
        task = await coordinator.arm(
            skill_name=skill_name,
            trigger_expr=trigger_expr,
            execution_tier=execution_tier,
            max_runs=max_runs,
        )
        now = time.time()
        next_in = _format_duration(task.next_due_at - now)
        print(f"  Armed: {task.task_id[:8]}  skill={skill_name}  tier={task.execution_tier}  next={next_in}")
    except ValueError as e:
        print(f"  Error: {e}")
        return 1
    except RuntimeError as e:
        print(f"  Error: {e}")
        return 1

    return 0


async def cmd_tasks(ctx: "Context", args: List[str]) -> int:
    """leap tasks [status <id> | logs <id> | cancel <id>]

    Manage scheduled tasks.
    """
    coordinator = _get_coordinator(ctx)

    if not args:
        # List all tasks
        tasks = await coordinator.list_tasks()
        if not tasks:
            print("  No scheduled tasks.")
            return 0

        now = time.time()
        print(f"  {'ID':<10} {'Skill':<20} {'Trigger':<16} {'Tier':<7} {'State':<10} Next Due")
        print(f"  {'-' * 80}")
        for t in tasks:
            tid = t.task_id[:8]
            skill = t.skill_name[:18]
            trigger = _format_trigger(t)[:14]
            tier = t.execution_tier[:5]
            state = t.state[:8]
            if t.next_due_at > 0:
                next_due = _format_duration(t.next_due_at - now)
            else:
                next_due = "-"
            print(f"  {tid:<10} {skill:<20} {trigger:<16} {tier:<7} {state:<10} {next_due}")
        print(f"\n  {len(tasks)} task(s).")
        return 0

    subcmd = args[0].lower()

    if subcmd == "status" and len(args) > 1:
        task_id = args[1]
        try:
            status = await coordinator.status(task_id)
            t = status.task
            print(f"  Task:      {t.task_id}")
            print(f"  Skill:     {t.skill_name}")
            print(f"  Trigger:   {_format_trigger(t)}")
            print(f"  Tier:      {t.execution_tier}")
            print(f"  State:     {t.state}")
            print(f"  Runs:      {t.run_count}" + (f" / {t.max_runs}" if t.max_runs > 0 else ""))
            print(f"  Running:   {status.is_running}")
            if t.next_due_at > 0:
                now = time.time()
                print(f"  Next due:  {_format_duration(t.next_due_at - now)}")
            if t.cloud_worker_id:
                print(f"  Worker:    {t.cloud_worker_id}")
        except ValueError as e:
            print(f"  Error: {e}")
            return 1
        return 0

    if subcmd == "logs" and len(args) > 1:
        task_id = args[1]
        tail = 50
        if len(args) > 2:
            try:
                tail = int(args[2])
            except ValueError:
                pass
        try:
            logs = await coordinator.logs(task_id, tail=tail)
            if not logs:
                print("  No logs available.")
            else:
                for line in logs:
                    print(f"  {line}")
        except ValueError as e:
            print(f"  Error: {e}")
            return 1
        return 0

    if subcmd == "cancel" and len(args) > 1:
        task_id = args[1]
        try:
            await coordinator.cancel(task_id)
            print(f"  Cancelled: {task_id[:8]}")
        except ValueError as e:
            print(f"  Error: {e}")
            return 1
        return 0

    # Unknown subcommand
    print("Usage: tasks [status <id> | logs <id> | cancel <id>]")
    print("  (no args)     — List all scheduled tasks")
    print("  status <id>   — Show detailed status")
    print("  logs <id>     — Show recent logs")
    print("  cancel <id>   — Cancel a task")
    return 0

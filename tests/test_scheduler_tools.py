# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Phase 2 scheduler productization.

Covers:
- 2A: SchedulerToolsPlugin Protocol conformance and handler behavior
- 2B: Delivery hook integration in LocalScheduler
- 2C: /schedule run and /schedule doctor slash commands
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from leapflow.plugins.protocol import ToolMetadata, ToolPlugin
from leapflow.plugins.tool_plugins.scheduler_tools import SchedulerToolsPlugin
from leapflow.scheduler.local_scheduler import LocalScheduler
from leapflow.scheduler.types import ArmedTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def plugin() -> SchedulerToolsPlugin:
    return SchedulerToolsPlugin()


@pytest.fixture()
def mock_coordinator() -> AsyncMock:
    """A mock TaskCoordinator with all methods the handlers call."""
    coord = AsyncMock()
    coord.arm = AsyncMock(
        return_value=ArmedTask(
            skill_name="daily_report",
            trigger_type="interval",
            trigger_config={"interval_seconds": 1800},
            task_id="aaaa1111bbbb2222",
            state="armed",
            next_due_at=time.time() + 1800,
        )
    )
    coord.list_tasks = AsyncMock(
        return_value=[
            ArmedTask(
                skill_name="report",
                trigger_type="interval",
                trigger_config={"interval_seconds": 60},
                task_id="task_001",
                state="armed",
            ),
        ]
    )
    from leapflow.scheduler.types import TaskStatus

    coord.status = AsyncMock(
        return_value=TaskStatus(
            task=ArmedTask(
                skill_name="report",
                trigger_type="interval",
                trigger_config={},
                task_id="task_001",
                state="armed",
            ),
            is_running=False,
        )
    )
    coord.get_execution_history = MagicMock(return_value=[])
    coord.pause_task = AsyncMock()
    coord.resume_task = AsyncMock()
    coord.cancel = AsyncMock()
    return coord


@pytest.fixture()
def bound_plugin(plugin: SchedulerToolsPlugin, mock_coordinator: AsyncMock) -> SchedulerToolsPlugin:
    plugin.bind_runtime(scheduler=mock_coordinator)
    return plugin


# ---------------------------------------------------------------------------
# 2A: Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_isinstance_tool_plugin(self, plugin: SchedulerToolsPlugin) -> None:
        assert isinstance(plugin, ToolPlugin)

    def test_plugin_id(self, plugin: SchedulerToolsPlugin) -> None:
        assert plugin.plugin_id == "scheduler_tools"

    def test_category(self, plugin: SchedulerToolsPlugin) -> None:
        assert plugin.category == "scheduler"

    def test_dependencies(self, plugin: SchedulerToolsPlugin) -> None:
        assert plugin.dependencies == ["scheduler"]

    def test_tools_count(self, plugin: SchedulerToolsPlugin) -> None:
        assert len(plugin.tools) == 6

    def test_tools_are_tool_metadata(self, plugin: SchedulerToolsPlugin) -> None:
        for tool in plugin.tools:
            assert isinstance(tool, ToolMetadata)

    def test_tool_names(self, plugin: SchedulerToolsPlugin) -> None:
        names = {t.name for t in plugin.tools}
        assert names == {
            "schedule_create",
            "schedule_list",
            "schedule_status",
            "schedule_pause",
            "schedule_resume",
            "schedule_cancel",
        }

    def test_x_leapflow_category(self, plugin: SchedulerToolsPlugin) -> None:
        for tool in plugin.tools:
            assert tool.x_leapflow.get("category") == "scheduler"
            assert "risk_level" in tool.x_leapflow

    def test_openai_schema_generation(self, plugin: SchedulerToolsPlugin) -> None:
        for tool in plugin.tools:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# 2A: Unbound coordinator returns structured refusal
# ---------------------------------------------------------------------------


class TestUnboundRefusal:
    @pytest.mark.asyncio
    async def test_create_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_create(trigger_expression="30m", instruction="test")
        assert result["ok"] is False
        assert result["error"] == "scheduler_not_available"

    @pytest.mark.asyncio
    async def test_list_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_list()
        assert result["ok"] is False
        assert result["error"] == "scheduler_not_available"

    @pytest.mark.asyncio
    async def test_status_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_status(task_id="abc")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_pause_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_pause(task_id="abc")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_resume_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_resume(task_id="abc")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_cancel_unbound(self, plugin: SchedulerToolsPlugin) -> None:
        result = await plugin._handle_cancel(task_id="abc")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 2A: Handler behavior when coordinator is bound
# ---------------------------------------------------------------------------


class TestBoundHandlers:
    @pytest.mark.asyncio
    async def test_create_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_create(
            trigger_expression="30m",
            instruction="daily_report",
        )
        assert result["ok"] is True
        assert "task_id" in result
        assert result["state"] == "armed"

    @pytest.mark.asyncio
    async def test_create_missing_fields(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_create(trigger_expression="30m")
        assert result["ok"] is False
        assert result["error"] == "missing_required_fields"

    @pytest.mark.asyncio
    async def test_create_missing_trigger(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_create(instruction="test")
        assert result["ok"] is False
        assert result["error"] == "missing_required_fields"

    @pytest.mark.asyncio
    async def test_create_with_delivery_target(self, bound_plugin: SchedulerToolsPlugin, mock_coordinator: AsyncMock) -> None:
        result = await bound_plugin._handle_create(
            trigger_expression="30m",
            instruction="report",
            delivery_target={"platform": "feishu", "chat_id": "oc_123"},
        )
        assert result["ok"] is True
        # Verify delivery_target was passed through parameters
        call_kwargs = mock_coordinator.arm.call_args
        assert call_kwargs.kwargs["parameters"]["delivery_target"]["platform"] == "feishu"

    @pytest.mark.asyncio
    async def test_list_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_list()
        assert result["ok"] is True
        assert result["count"] == 1
        assert result["tasks"][0]["skill_name"] == "report"

    @pytest.mark.asyncio
    async def test_status_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_status(task_id="task_001")
        assert result["ok"] is True
        assert result["task_id"] == "task_001"
        assert "recent_history" in result

    @pytest.mark.asyncio
    async def test_status_missing_task_id(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_status()
        assert result["ok"] is False
        assert result["error"] == "missing_task_id"

    @pytest.mark.asyncio
    async def test_pause_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_pause(task_id="task_001")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_resume_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_resume(task_id="task_001")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_cancel_success(self, bound_plugin: SchedulerToolsPlugin) -> None:
        result = await bound_plugin._handle_cancel(task_id="task_001")
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_status_not_found(self, bound_plugin: SchedulerToolsPlugin, mock_coordinator: AsyncMock) -> None:
        mock_coordinator.status.side_effect = ValueError("Task not found: xyz")
        result = await bound_plugin._handle_status(task_id="xyz")
        assert result["ok"] is False
        assert result["error"] == "not_found"


# ---------------------------------------------------------------------------
# 2B: Delivery integration in LocalScheduler
# ---------------------------------------------------------------------------


class TestDeliveryIntegration:
    def _make_scheduler(
        self,
        *,
        send_fn: Any = None,
        delivery_enabled: bool = False,
    ) -> tuple[LocalScheduler, MagicMock, MagicMock]:
        store = MagicMock()
        executor = AsyncMock()
        sched = LocalScheduler(
            store=store,
            executor=executor,
            tick_seconds=60,
            send_fn=send_fn,
            delivery_enabled=delivery_enabled,
        )
        return sched, store, executor

    def test_delivery_skipped_when_disabled(self) -> None:
        send_fn = MagicMock()
        sched, _, _ = self._make_scheduler(send_fn=send_fn, delivery_enabled=False)
        task = ArmedTask(
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "feishu", "chat_id": "oc_123"}},
        )
        sched._attempt_delivery(task, success=True, summary="done")
        send_fn.assert_not_called()

    def test_delivery_skipped_when_no_send_fn(self) -> None:
        sched, _, _ = self._make_scheduler(send_fn=None, delivery_enabled=True)
        task = ArmedTask(
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "feishu", "chat_id": "oc_123"}},
        )
        # Should not raise
        sched._attempt_delivery(task, success=True, summary="done")

    def test_delivery_skipped_when_no_target(self) -> None:
        send_fn = MagicMock()
        sched, _, _ = self._make_scheduler(send_fn=send_fn, delivery_enabled=True)
        task = ArmedTask(
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={},  # no delivery_target
        )
        sched._attempt_delivery(task, success=True, summary="done")
        send_fn.assert_not_called()

    def test_delivery_sends_on_success(self) -> None:
        send_fn = MagicMock()
        sched, _, _ = self._make_scheduler(send_fn=send_fn, delivery_enabled=True)
        task = ArmedTask(
            skill_name="daily_report",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "feishu", "chat_id": "oc_123"}},
        )
        sched._attempt_delivery(task, success=True, summary="All good", duration_s=5.2)
        send_fn.assert_called_once()
        args = send_fn.call_args[0]
        assert args[0] == "feishu"
        assert args[1] == "oc_123"
        assert "Success" in args[2]
        assert "daily_report" in args[2]

    def test_delivery_sends_on_failure(self) -> None:
        send_fn = MagicMock()
        sched, _, _ = self._make_scheduler(send_fn=send_fn, delivery_enabled=True)
        task = ArmedTask(
            skill_name="broken_task",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "slack", "chat_id": "C123"}},
        )
        sched._attempt_delivery(task, success=False, error="timeout", duration_s=30.0)
        send_fn.assert_called_once()
        args = send_fn.call_args[0]
        assert args[0] == "slack"
        assert "Failed" in args[2]

    def test_delivery_failure_is_non_fatal(self) -> None:
        send_fn = MagicMock(side_effect=RuntimeError("network down"))
        sched, _, _ = self._make_scheduler(send_fn=send_fn, delivery_enabled=True)
        task = ArmedTask(
            skill_name="test",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "feishu", "chat_id": "oc_123"}},
        )
        # Should NOT raise despite send_fn error
        sched._attempt_delivery(task, success=True, summary="ok")

    @pytest.mark.asyncio
    async def test_execute_task_calls_delivery_on_success(self) -> None:
        send_fn = MagicMock()
        sched, store, executor = self._make_scheduler(
            send_fn=send_fn, delivery_enabled=True,
        )
        task = ArmedTask(
            skill_name="test_skill",
            trigger_type="interval",
            trigger_config={"interval_seconds": 60},
            parameters={"delivery_target": {"platform": "feishu", "chat_id": "oc_x"}},
        )
        executor.execute = AsyncMock(return_value={"ok": True, "output": "success"})
        store.load.return_value = task  # for reload after increment
        store.get_due_tasks.return_value = []

        await sched._execute_task(task, time.time())
        send_fn.assert_called_once()
        assert "Success" in send_fn.call_args[0][2]


# ---------------------------------------------------------------------------
# 2C: /schedule run and /schedule doctor
# ---------------------------------------------------------------------------


class TestScheduleRunDoctor:
    def test_schedule_doctor_no_scheduler(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        ctx = MagicMock()
        ctx.settings.duckdb_path = ":memory:"
        # No coordinator, no store
        delattr_safe(ctx, "coordinator")
        with patch("leapflow.scheduler.store.TaskStore", side_effect=Exception("no db")):
            result = build_schedule_payload(ctx, "doctor")
        assert result["ok"] is True
        assert "No scheduler" in result["message"]

    def test_schedule_doctor_with_tasks(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        ctx = MagicMock()
        task_store = MagicMock()
        now = time.time()
        task_store.load_all.return_value = [
            ArmedTask(
                skill_name="a", trigger_type="interval",
                trigger_config={}, state="armed", task_id="aaaa1111",
                next_due_at=now + 100,
            ),
            ArmedTask(
                skill_name="b", trigger_type="cron",
                trigger_config={}, state="paused", task_id="bbbb2222",
            ),
            ArmedTask(
                skill_name="c", trigger_type="interval",
                trigger_config={}, state="failed", task_id="cccc3333",
            ),
        ]
        coordinator = MagicMock()
        coordinator._store = task_store
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "doctor")
        assert result["ok"] is True
        assert "Total tasks: 3" in result["message"]
        assert "armed: 1" in result["message"]
        assert "paused: 1" in result["message"]
        assert "failed: 1" in result["message"]

    def test_schedule_run_missing_task_id(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        ctx = MagicMock()
        task_store = MagicMock()
        coordinator = MagicMock()
        coordinator._store = task_store
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "run")
        assert result["ok"] is False
        assert "Usage" in result["message"]

    def test_schedule_run_executes(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        ctx = MagicMock()
        task_store = MagicMock()
        task = ArmedTask(
            skill_name="test", trigger_type="interval",
            trigger_config={}, task_id="run_task_id_1234",
            parameters={"instruction": "hello"},
        )
        task_store.load.return_value = task
        coordinator = MagicMock()
        coordinator._store = task_store
        local_sched = MagicMock()
        local_sched._executor = AsyncMock()
        local_sched._executor.execute = AsyncMock(return_value={"ok": True, "output": "done"})
        coordinator._local = local_sched
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "run run_task_id_1234")
        assert result["ok"] is True
        assert "ok=True" in result["message"]

    def test_schedule_unknown_subcommand(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload
        ctx = MagicMock()
        task_store = MagicMock()
        coordinator = MagicMock()
        coordinator._store = task_store
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "bogus")
        assert result["ok"] is False
        assert "run" in result["message"]
        assert "doctor" in result["message"]


# ---------------------------------------------------------------------------
# Registry and completion
# ---------------------------------------------------------------------------


class TestRegistryAndCompletion:
    def test_schedule_run_in_registry(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule run")
        assert cmd is not None
        assert cmd.name == "schedule run"

    def test_schedule_doctor_in_registry(self) -> None:
        from leapflow.cli.commands.registry import resolve_command
        cmd = resolve_command("schedule doctor")
        assert cmd is not None
        assert cmd.name == "schedule doctor"

    def test_schedule_verbs_in_completer(self) -> None:
        from leapflow.cli.tui_app.input import SlashCommandCompleter
        completer = SlashCommandCompleter(commands=[])
        verbs = {v for v, _ in completer._SCHEDULE_VERBS}
        assert "run" in verbs
        assert "doctor" in verbs


# ---------------------------------------------------------------------------
# Config setting
# ---------------------------------------------------------------------------


class TestConfigSetting:
    def test_scheduler_delivery_enabled_default(self) -> None:
        from leapflow.config import Settings
        # Default is False
        s = Settings.__dataclass_fields__["scheduler_delivery_enabled"]
        assert s.default is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def delattr_safe(obj: Any, name: str) -> None:
    """Remove an attribute from a mock without error."""
    try:
        delattr(obj, name)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# execution_mode exposure in tool outputs
# ---------------------------------------------------------------------------


class TestExecutionModeExposure:
    """list/status tool outputs surface each task's execution mode."""

    @pytest.mark.asyncio
    async def test_list_includes_execution_mode(self) -> None:
        plugin = SchedulerToolsPlugin()
        coord = AsyncMock()
        coord.list_tasks = AsyncMock(return_value=[
            ArmedTask(
                skill_name="a", trigger_type="interval", trigger_config={},
                task_id="t1", parameters={"execution_mode": "agent"},
            ),
            ArmedTask(
                skill_name="b", trigger_type="interval", trigger_config={},
                task_id="t2", parameters={},
            ),
        ])
        plugin.bind_runtime(scheduler=coord)
        result = await plugin._handle_list()
        assert result["ok"] is True
        modes = {t["task_id"]: t["execution_mode"] for t in result["tasks"]}
        assert modes["t1"] == "agent"
        # A task with no explicit mode defaults to script.
        assert modes["t2"] == "script"

    @pytest.mark.asyncio
    async def test_status_includes_execution_mode(self) -> None:
        from leapflow.scheduler.types import TaskStatus

        plugin = SchedulerToolsPlugin()
        coord = AsyncMock()
        coord.status = AsyncMock(return_value=TaskStatus(
            task=ArmedTask(
                skill_name="a", trigger_type="interval", trigger_config={},
                task_id="t1", parameters={"execution_mode": "agent"},
            ),
            is_running=False,
        ))
        coord.get_execution_history = MagicMock(return_value=[])
        plugin.bind_runtime(scheduler=coord)
        result = await plugin._handle_status(task_id="t1")
        assert result["ok"] is True
        assert result["execution_mode"] == "agent"


# ---------------------------------------------------------------------------
# /schedule list and /schedule status output format
# ---------------------------------------------------------------------------


class TestScheduleListStatusFormat:
    """Slash command output shows mode column and per-task status detail."""

    def test_list_shows_execution_mode_column(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        ctx = MagicMock()
        task_store = MagicMock()
        task_store.load_all.return_value = [
            ArmedTask(
                skill_name="a", trigger_type="interval",
                trigger_config={"interval_seconds": 60}, task_id="aaaa1111",
                state="armed", parameters={"execution_mode": "agent"},
            ),
            ArmedTask(
                skill_name="b", trigger_type="interval",
                trigger_config={"interval_seconds": 60}, task_id="bbbb2222",
                state="armed", parameters={},
            ),
        ]
        coordinator = MagicMock()
        coordinator._store = task_store
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "list")
        assert result["ok"] is True
        assert "mode=agent" in result["message"]
        assert "mode=script" in result["message"]

    def test_status_missing_task_id(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        ctx = MagicMock()
        coordinator = MagicMock()
        coordinator._store = MagicMock()
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "status")
        assert result["ok"] is False
        assert "Usage" in result["message"]

    def test_status_task_not_found(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        ctx = MagicMock()
        task_store = MagicMock()
        task_store.load.return_value = None
        coordinator = MagicMock()
        coordinator._store = task_store
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "status missing_id")
        assert result["ok"] is False
        assert "not found" in result["message"].lower()

    def test_status_agent_mode_shows_subagent(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        ctx = MagicMock()
        task_store = MagicMock()
        task_store.load.return_value = ArmedTask(
            skill_name="report", trigger_type="interval",
            trigger_config={"interval_seconds": 60}, task_id="abcd1234ffff",
            state="armed", parameters={"execution_mode": "agent"},
            next_due_at=time.time() + 100,
        )
        coordinator = MagicMock()
        coordinator._store = task_store
        log = MagicMock()
        log.get_history.return_value = []
        coordinator._execution_log = log
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "status abcd1234ffff")
        assert result["ok"] is True
        assert "mode: agent" in result["message"]
        assert "sub-agent" in result["message"]
        assert "recent runs: none" in result["message"]

    def test_status_script_mode_no_subagent_line(self) -> None:
        from leapflow.cli.commands.slash_handlers import build_schedule_payload

        ctx = MagicMock()
        task_store = MagicMock()
        task_store.load.return_value = ArmedTask(
            skill_name="report", trigger_type="interval",
            trigger_config={"interval_seconds": 60}, task_id="11112222",
            state="armed", parameters={},
        )
        coordinator = MagicMock()
        coordinator._store = task_store
        log = MagicMock()
        log.get_history.return_value = []
        coordinator._execution_log = log
        ctx.coordinator = coordinator
        result = build_schedule_payload(ctx, "status 11112222")
        assert result["ok"] is True
        assert "mode: script" in result["message"]
        assert "sub-agent" not in result["message"]

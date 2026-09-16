# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for plugin adaptive plan introspection surfaces."""

from __future__ import annotations

import pytest

from leapflow.cli.commands.slash_handlers import (
    build_plugin_payload,
    plugin_generate_start_payload,
    render_command_payload,
)
from leapflow.plugins.registry import ToolPluginRegistry
from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin
from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore


class _Ctx:
    pass


class _Console:
    def __init__(self) -> None:
        self.printed: list[object] = []
        self.systems: list[str] = []
        self.successes: list[str] = []
        self.warnings: list[str] = []

    def print(self, value: object) -> None:
        self.printed.append(value)

    def system(self, value: str) -> None:
        self.systems.append(value)

    def success(self, value: str) -> None:
        self.successes.append(value)

    def warning(self, value: str) -> None:
        self.warnings.append(value)


def _seed_store(tmp_path):
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")
    store.add_record(
        environment={"fingerprint_id": "env-a"},
        requirements=[{"capability": "json.pretty"}],
        resolutions=[
            {"selected": {"candidate": {"plugin_id": "json", "tool_name": "json_pretty"}}}
        ],
        plan={
            "plan_id": "plan-json",
            "executable": True,
            "steps": [{"tool_name": "json_pretty", "plugin_id": "json"}],
        },
        source="unit",
        record_id="record-json",
    )
    return store


@pytest.mark.asyncio
async def test_self_management_plugin_plan_lists_records(tmp_path) -> None:
    plugin = SelfManagementPlugin()
    plugin.bind_runtime(capability_plan_store=_seed_store(tmp_path))

    result = await plugin._plugin_plan_handler(limit=3)

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["records"][0]["record_id"] == "record-json"


@pytest.mark.asyncio
async def test_plugin_plan_slash_payload_delegates_to_self_management(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import leapflow.plugins as plugins_module

    plugin = SelfManagementPlugin()
    plugin.bind_runtime(capability_plan_store=_seed_store(tmp_path))
    reg = ToolPluginRegistry()
    reg.register(plugin)
    reg.assemble()
    monkeypatch.setattr(plugins_module, "_registry", reg)

    payload = await build_plugin_payload(_Ctx(), "plan --latest")

    assert payload["ok"] is True
    assert payload["view"] == "plugin_plan"
    assert payload["latest"]["record_id"] == "record-json"
    assert payload["records"][0]["plan"]["plan_id"] == "plan-json"


def test_render_command_payload_dispatches_plugin_views() -> None:
    console = _Console()
    payload = {
        "ok": True,
        "view": "plugin_list",
        "plugin_count": 1,
        "plugins": [
            {
                "plugin_id": "text_utils",
                "category": "general",
                "tool_count": 2,
                "state": "active",
                "generation": 1,
            }
        ],
    }

    render_command_payload(console, payload)

    assert console.printed, "plugin_list should render a Rich table in daemon command path"
    assert console.systems == ["1 plugins registered"]


def test_plugin_generate_start_payload_explains_long_running_steps() -> None:
    payload = plugin_generate_start_payload('generate "new sandbox for windows"')

    assert payload is not None
    assert payload["plugin_id"] == "new_sandbox_windows"
    assert payload["mode"] == "install"
    assert len(payload["steps"]) >= 4


def test_render_command_payload_shows_generate_stage_table_on_failure() -> None:
    console = _Console()
    payload = {
        "ok": False,
        "view": "plugin_generate",
        "error": "Generation failed: bad schema",
        "steps": [{"name": "generate_attempt_1", "status": "failed", "detail": "bad schema"}],
        "duration_s": 12.5,
    }

    render_command_payload(console, payload)

    assert console.printed, "failed plugin_generate should still render stage details"
    assert console.warnings == ["Generation failed: bad schema"]

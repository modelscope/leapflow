from __future__ import annotations

from leapflow.engine.context_disclosure import (
    CapabilityManifest,
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
)
from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS


def _tool_names(plan) -> set[str]:
    return {
        item.get("function", {}).get("name", "")
        for item in plan.tool_definitions
    }


def test_disclosure_planner_keeps_plain_chat_light() -> None:
    planner = DisclosurePlanner()

    plan = planner.plan(
        "Hello, who are you?",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(enable_thinking=True, native_tools_enabled=True),
    )

    assert plan.level == DisclosureLevel.LIGHT
    assert plan.native_tools is False
    assert plan.tool_definitions == ()
    assert plan.memory.value == "none"
    assert plan.reasoning.value == "off"


def test_disclosure_planner_selects_file_tools_without_full_catalog() -> None:
    planner = DisclosurePlanner()

    plan = planner.plan(
        "Read src/leapflow/engine/engine.py and summarize the relevant section",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(native_tools_enabled=True),
    )

    names = _tool_names(plan)
    assert plan.level == DisclosureLevel.SELECTED_TOOLS
    assert plan.native_tools is True
    assert "file_read" in names
    assert "file_list" in names
    assert "shell_run" not in names

    package_plan = planner.plan(
        "Read package.json and summarize dependencies",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(native_tools_enabled=True),
    )
    assert package_plan.level == DisclosureLevel.SELECTED_TOOLS


def test_disclosure_planner_selects_project_research_without_full_catalog() -> None:
    planner = DisclosurePlanner()

    plan = planner.plan(
        "Read and study this codebase, then generate a system architecture diagram",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(native_tools_enabled=True, enable_thinking=True),
    )

    names = _tool_names(plan)
    assert plan.level == DisclosureLevel.PROJECT_RESEARCH
    assert plan.native_tools is True
    assert plan.memory.value == "task_retrieval"
    assert plan.reasoning.value == "auto"
    assert "file_read" in names
    assert "file_list" in names
    assert "shell_run" not in names
    assert not any(name.startswith("hub") for name in names)


def test_disclosure_planner_uses_index_for_capability_questions() -> None:
    planner = DisclosurePlanner()

    plan = planner.plan(
        "What tools can you use?",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(native_tools_enabled=True),
    )

    assert plan.level == DisclosureLevel.INDEXED_CAPABILITIES
    assert plan.native_tools is False
    assert plan.tool_definitions == ()
    assert plan.catalog_definitions
    assert plan.memory.value == "session_summary"


def test_disclosure_planner_uses_full_context_for_runtime_escalation() -> None:
    planner = DisclosurePlanner()

    slash_plan = planner.plan(
        "/run organize downloads",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(slash_command=True, native_tools_enabled=True),
    )
    failure_plan = planner.plan(
        "continue",
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(recent_failure=True, native_tools_enabled=True),
    )

    assert slash_plan.level == DisclosureLevel.FULL
    assert slash_plan.native_tools is True
    assert failure_plan.level == DisclosureLevel.FULL


def test_capability_manifest_prefers_explicit_tool_metadata() -> None:
    manifest = CapabilityManifest.from_tool_definition({
        "type": "function",
        "function": {
            "name": "notify_user",
            "description": "Send an external notification.",
            "x_leapflow": {
                "category": "gateway",
                "summary": "Notify a person through an external gateway.",
                "input_signals": ["alert", "notify"],
                "risk_level": "high",
                "requires_approval": True,
                "schema_cost": "high",
            },
        },
    })

    assert manifest.category == "gateway"
    assert manifest.input_signals == ("alert", "notify")
    assert manifest.requires_approval is True
    assert manifest.schema_cost == "high"


def test_file_read_schema_discourages_workspace_config_probe() -> None:
    file_read_def = next(
        item for item in TOOL_DEFINITIONS
        if item.get("function", {}).get("name") == "file_read"
    )
    description = str(file_read_def["function"].get("description", ""))

    assert "Do not probe `<workspace>/.leapflow/config.json`" in description
    assert "~/.leapflow/.env" in description
    assert "./.env" in description

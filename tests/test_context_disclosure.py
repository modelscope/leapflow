# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from leapflow.engine.context_disclosure import (
    CapabilityManifest,
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
    build_capability_manifests,
)
from leapflow.plugins import get_registry
_tool_reg = get_registry()

TOOL_DEFINITIONS = _tool_reg.tool_definitions


def _tool_names(plan) -> set[str]:
    return {
        item.get("function", {}).get("name", "")
        for item in plan.tool_definitions
    }


def test_disclosure_planner_core_is_never_empty_and_excludes_heavy_categories() -> None:
    """CORE is the floor: no structural signal -> static Tier 0/0.5 only."""
    planner = DisclosurePlanner()

    plan = planner.plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(enable_thinking=True, native_tools_enabled=True),
    )

    names = _tool_names(plan)
    assert plan.level == DisclosureLevel.CORE
    assert plan.tool_definitions  # never empty
    assert plan.catalog_definitions == tuple(TOOL_DEFINITIONS)
    assert plan.native_tools is True
    assert plan.context_planes == ("task_semantic", "control_plane")

    # Always-on low-risk, cheap-schema tools.
    for expected in ("file_list", "file_read", "text_search", "memory_search", "capability_expand"):
        assert expected in names

    # Mutating / heavy / sensitive tools must never be in the static core whitelist.
    for excluded in (
        "shell_run",
        "file_write",
        "memory_add",
        "delegate_task",
        "hub_push",
        "hub_pull",
        "hub_sync",
        "gateway_send",
        "gateway_connect",
        "platform_action",
        "platform_connect",
    ):
        assert excluded not in names


def test_disclosure_planner_expands_via_last_turn_tool_category_continuity() -> None:
    """Tier 1 opens strictly from structural continuity, never from user text."""
    planner = DisclosurePlanner()

    plan = planner.plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(
            native_tools_enabled=True,
            last_turn_tool_categories=frozenset({"hub"}),
        ),
    )

    names = _tool_names(plan)
    assert plan.level == DisclosureLevel.EXPANDED
    assert "hub" in plan.expanded_categories
    for expected in ("hub_push", "hub_pull", "hub_search", "hub_sync"):
        assert expected in names
    # Gateway tools were not part of the continuity signal, so they stay closed.
    assert "gateway_send" not in names


def test_disclosure_planner_expands_tools_from_capability_plan() -> None:
    """A structured capability plan can disclose its tools without opening FULL."""
    from leapflow.plugins.capability_plan import CapabilityPlan
    from leapflow.plugins.capability_resolver import CapabilityCandidate

    plan_hint = CapabilityPlan.from_candidates(
        (
            CapabilityCandidate(
                plugin_id="shell_terminal",
                tool_name="shell_run",
                provides_capabilities=("shell.run",),
                risk_level="external",
                requires_approval=True,
                mutates_state=True,
            ),
        ),
        plan_id="plan-disclosure-test",
    )

    plan = DisclosurePlanner().plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(
            native_tools_enabled=True,
            active_capability_plan=plan_hint,
        ),
    )

    names = _tool_names(plan)
    assert plan.level == DisclosureLevel.EXPANDED
    assert plan.reason == "plan: capability_plan"
    assert "shell_run" in names


def test_disclosure_planner_uses_full_context_for_structural_gates() -> None:
    planner = DisclosurePlanner()

    slash_plan = planner.plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(slash_command=True, native_tools_enabled=True),
    )
    failure_plan = planner.plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(recent_failure=True, native_tools_enabled=True),
    )
    posture_plan = planner.plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(context_posture="research", native_tools_enabled=True),
    )

    assert slash_plan.level == DisclosureLevel.FULL
    assert slash_plan.native_tools is True
    assert slash_plan.tool_definitions == tuple(TOOL_DEFINITIONS)
    assert failure_plan.level == DisclosureLevel.FULL
    assert posture_plan.level == DisclosureLevel.FULL


def test_disclosure_planner_never_performs_text_fitting() -> None:
    """The planner signature no longer accepts user text at all."""
    import inspect

    signature = inspect.signature(DisclosurePlanner.plan)
    assert "user_text" not in signature.parameters
    assert list(signature.parameters)[1:] == ["tool_definitions", "runtime"]
    assert "active_capability_plan" in DisclosureRuntimeState.__dataclass_fields__


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
    assert manifest.is_core is False


def test_capability_manifest_is_core_property_reflects_risk_and_cost() -> None:
    read_only_cheap = CapabilityManifest(name="a", category="general", summary="", risk_level="read_only", schema_cost="medium")
    read_only_heavy = CapabilityManifest(name="b", category="hub", summary="", risk_level="read_only", schema_cost="high")
    mutating_cheap = CapabilityManifest(name="c", category="write", summary="", risk_level="high", schema_cost="medium")

    assert read_only_cheap.is_core is True
    assert read_only_heavy.is_core is False
    assert mutating_cheap.is_core is False


def test_hub_and_gateway_tools_are_explicitly_classified_as_heavy() -> None:
    """Regression guard: hub/gateway tools must declare x_leapflow explicitly.

    Keyword inference over descriptions is unreliable for these tools (e.g. a
    hub tool's description mentions "skill", a gateway tool's description may
    not literally contain "gateway"), so they must not rely on _infer_category
    fallbacks to land in the correct heavy, non-core category.
    """
    manifests = {m.name: m for m in build_capability_manifests(TOOL_DEFINITIONS)}

    for name in ("hub_push", "hub_pull", "hub_search", "hub_sync"):
        assert manifests[name].category == "hub"
        assert manifests[name].schema_cost == "high"
        assert manifests[name].is_core is False

    for name in ("platform_action", "platform_connect", "gateway_send", "gateway_connect"):
        assert manifests[name].category == "gateway"
        assert manifests[name].schema_cost == "high"
        assert manifests[name].is_core is False


def test_file_read_schema_redirects_config_reads_to_the_config_tools() -> None:
    """The redirect must name the capability, not enumerate config paths.

    The previous wording forbade one specific path and then listed the real ones,
    which both failed to stop the probing (the model simply tried another path)
    and handed it fresh targets. With ``config_*`` tools available the guidance is
    a pointer to them, and no LeapFlow path belongs in this description.
    """
    file_read_def = next(
        item for item in TOOL_DEFINITIONS
        if item.get("function", {}).get("name") == "file_read"
    )
    description = str(file_read_def["function"].get("description", ""))

    for tool in ("config_list", "config_get", "config_set"):
        assert tool in description
    # No config path may be advertised here — that is what invited the probing.
    for leaked in ("~/.leapflow", ".leapflow/config.json", "config/user.yaml", "profiles/"):
        assert leaked not in description


def test_config_tools_are_core_and_writes_are_not() -> None:
    """Reading settings must be always-available; writing must not be.

    A config read the model cannot see is the whole reason it fell back to file
    probing, so ``config_get``/``config_list`` belong in the CORE floor. The
    write is a mutation and stays behind progressive disclosure.
    """
    plan = DisclosurePlanner().plan(
        TOOL_DEFINITIONS,
        DisclosureRuntimeState(native_tools_enabled=True),
    )
    names = _tool_names(plan)

    assert "config_get" in names
    assert "config_list" in names
    assert "config_set" not in names


# ── Desktop (semantic tool) disclosure ─────────────────────────────────


def _desktop_definitions() -> list[dict]:
    from leapflow.skills.semantic_schema import semantic_tool_to_openai
    from leapflow.skills.tool_executor import ToolDefinition

    defs = []
    for name in ("observe_ui", "click", "list_apps"):
        schema = semantic_tool_to_openai(
            ToolDefinition(name=name, description=f"test {name}", parameters={})
        )
        assert schema is not None
        defs.append(schema)
    return defs


def test_desktop_tools_are_non_core_and_expandable_by_continuity() -> None:
    """Desktop schemas stay out of the CORE floor but reopen via Tier 1 continuity."""
    catalog = list(TOOL_DEFINITIONS) + _desktop_definitions()
    planner = DisclosurePlanner()

    core_plan = planner.plan(catalog, DisclosureRuntimeState(native_tools_enabled=True))
    core_names = _tool_names(core_plan)
    assert core_plan.level == DisclosureLevel.CORE
    assert "click" not in core_names
    # Observation tools are read-only but schema_cost=high keeps them non-core.
    assert "observe_ui" not in core_names
    catalog_names = {
        td.get("function", {}).get("name") for td in core_plan.catalog_definitions
    }
    assert {"click", "observe_ui", "list_apps"} <= catalog_names

    expanded_plan = planner.plan(
        catalog,
        DisclosureRuntimeState(
            native_tools_enabled=True,
            last_turn_tool_categories=frozenset({"desktop"}),
        ),
    )
    assert expanded_plan.level == DisclosureLevel.EXPANDED
    assert {"click", "observe_ui", "list_apps"} <= _tool_names(expanded_plan)
    assert "desktop" in expanded_plan.expanded_categories


def test_desktop_tools_included_in_full_plan() -> None:
    catalog = list(TOOL_DEFINITIONS) + _desktop_definitions()
    plan = DisclosurePlanner().full_plan(
        catalog, DisclosureRuntimeState(native_tools_enabled=True), "test"
    )
    assert {"click", "observe_ui", "list_apps"} <= _tool_names(plan)


def test_capability_expand_provider_exposes_desktop_category() -> None:
    import asyncio

    from leapflow.plugins import get_registry
    _tool_reg = get_registry()
    from leapflow.plugins.tool_plugins.orchestration import plugin as orch_plugin

    desktop_defs = _desktop_definitions()
    _tool_reg.set_capability_catalog_provider(lambda: list(TOOL_DEFINITIONS) + desktop_defs)
    try:
        result = asyncio.run(orch_plugin._capability_expand_handler({"category": "desktop"}))
        assert result["ok"] is True
        expanded_names = {td["function"]["name"] for td in result["expanded_tools"]}
        assert expanded_names == {"observe_ui", "click", "list_apps"}

        unknown = asyncio.run(orch_plugin._capability_expand_handler({"category": "nope"}))
        assert unknown["ok"] is False
        assert "desktop" in unknown["available_categories"]

        desc = next(
            td["function"]["description"]
            for td in TOOL_DEFINITIONS
            if td["function"]["name"] == "capability_expand"
        )
        assert "desktop" in desc
    finally:
        _tool_reg.set_capability_catalog_provider(None)


def test_capability_expand_falls_back_to_static_catalog_without_provider() -> None:
    import asyncio

    from leapflow.plugins import get_registry
    _tool_reg = get_registry()
    from leapflow.plugins.tool_plugins.orchestration import plugin as orch_plugin

    _tool_reg.set_capability_catalog_provider(None)
    result = asyncio.run(orch_plugin._capability_expand_handler({"category": "file"}))
    assert result["ok"] is True
    assert result["expanded_tools"]

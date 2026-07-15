from __future__ import annotations

from leapflow.engine.context_disclosure import (
    CapabilityManifest,
    DisclosureLevel,
    DisclosurePlanner,
    DisclosureRuntimeState,
    build_capability_manifests,
)
from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS


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


def test_file_read_schema_discourages_workspace_config_probe() -> None:
    file_read_def = next(
        item for item in TOOL_DEFINITIONS
        if item.get("function", {}).get("name") == "file_read"
    )
    description = str(file_read_def["function"].get("description", ""))

    assert "Do not probe `<workspace>/.leapflow/config.json`" in description
    assert "~/.leapflow/config/user.yaml" in description
    assert "~/.leapflow/profiles/<profile>/config/*.yaml" in description
    assert "<workspace>/.leapflow/config.yaml" in description

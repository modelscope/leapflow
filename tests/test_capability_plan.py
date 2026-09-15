# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for declarative capability orchestration plans."""

from __future__ import annotations

from leapflow.plugins.capability_plan import CapabilityPlan
from leapflow.plugins.capability_resolver import CapabilityCandidate


def _candidate(
    plugin_id: str,
    tool_name: str,
    *,
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    risk_level: str = "read_only",
    requires_approval: bool = False,
    mutates_state: bool = False,
) -> CapabilityCandidate:
    return CapabilityCandidate(
        plugin_id=plugin_id,
        tool_name=tool_name,
        provides_capabilities=provides,
        requires_capabilities=requires,
        risk_level=risk_level,
        requires_approval=requires_approval,
        mutates_state=mutates_state,
    )


def test_plan_orders_provider_before_consumer() -> None:
    consumer = _candidate(
        "consumer",
        "consume_json",
        provides=("json.report",),
        requires=("json.read",),
    )
    provider = _candidate("provider", "read_json", provides=("json.read",))

    plan = CapabilityPlan.from_candidates((consumer, provider), plan_id="plan-test")

    assert [s.tool_name for s in plan.steps] == ["read_json", "consume_json"]
    assert plan.executable is True
    assert plan.missing_dependencies == ()


def test_plan_reports_missing_dependencies() -> None:
    consumer = _candidate(
        "consumer",
        "consume_json",
        provides=("json.report",),
        requires=("json.read",),
    )

    plan = CapabilityPlan.from_candidates((consumer,), plan_id="plan-missing")

    assert plan.executable is False
    assert plan.missing_dependencies[0].capability == "json.read"
    assert plan.to_dict()["missing_dependencies"][0]["step_id"] == "consumer:consume_json"


def test_plan_reports_cycles_without_throwing() -> None:
    a = _candidate("a", "tool_a", provides=("cap.a",), requires=("cap.b",))
    b = _candidate("b", "tool_b", provides=("cap.b",), requires=("cap.a",))

    plan = CapabilityPlan.from_candidates((a, b), plan_id="plan-cycle")

    assert plan.cycle_detected is True
    assert plan.executable is False
    assert [s.tool_name for s in plan.steps] == ["tool_a", "tool_b"]


def test_plan_propagates_execution_policy_and_approval_metadata() -> None:
    read = _candidate("r", "read", risk_level="read_only")
    external = _candidate(
        "e",
        "send",
        risk_level="external",
        requires_approval=True,
        mutates_state=True,
    )

    plan = CapabilityPlan.from_candidates((read, external), plan_id="plan-policy")

    policies = {s.tool_name: s.execution_policy for s in plan.steps}
    approvals = {s.tool_name: s.requires_approval for s in plan.steps}
    assert policies == {"read": "read_only", "send": "external_side_effect"}
    assert approvals == {"read": False, "send": True}

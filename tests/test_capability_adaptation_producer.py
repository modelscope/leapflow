# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for capability adaptation monitor producer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from leapflow.monitor.capability_adaptation_producer import CapabilityAdaptationProducer
from leapflow.monitor.types import ProducerContext, Severity, WatchSpec
from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore


def _record(store: JsonCapabilityPlanStore, *, executable: bool = True) -> None:
    store.add_record(
        resolutions=[
            {"selected": {"candidate": {"plugin_id": "json", "tool_name": "json_pretty"}}}
        ],
        plan={
            "plan_id": "plan-json",
            "executable": executable,
            "missing_dependencies": [] if executable else [{"capability": "json.read"}],
            "steps": [{"tool_name": "json_pretty", "plugin_id": "json"}],
        },
        record_id="record-json",
    )


@pytest.mark.asyncio
async def test_capability_adaptation_producer_emits_latest_plan_finding(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "plans.json")
    _record(store, executable=True)
    producer = CapabilityAdaptationProducer()
    ctx = ProducerContext(
        spec=WatchSpec(name="capability", domain="capability_adaptation", watch_id="watch-1"),
        now=1.0,
        services=SimpleNamespace(capability_plan_store=store),
    )

    findings = await producer.observe(ctx)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.watch_id == "watch-1"
    assert finding.domain == "capability_adaptation"
    assert finding.severity is Severity.INFO
    assert finding.suggested_actions[0].name == "plugin_plan"


@pytest.mark.asyncio
async def test_capability_adaptation_producer_marks_missing_dependencies_notable(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "plans.json")
    _record(store, executable=False)
    producer = CapabilityAdaptationProducer()
    ctx = ProducerContext(
        spec=WatchSpec(name="capability", domain="capability_adaptation", watch_id="watch-1"),
        now=1.0,
        services=SimpleNamespace(capability_plan_store=store),
    )

    findings = await producer.observe(ctx)

    assert findings[0].severity is Severity.NOTABLE
    assert "unresolved dependencies" in findings[0].summary


@pytest.mark.asyncio
async def test_capability_adaptation_producer_exposes_closed_loop_metadata(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "plans.json")
    store.add_record(
        resolutions=[{"selected": {"candidate": {"tool_name": "json_pretty_loop"}}}],
        plan={"plan_id": "plan-json", "executable": True, "steps": []},
        record_id="loop-a:after_install",
        phase="after_install",
        mutation={"action": "install", "plugin_id": "json_loop"},
        registry_version_before=2,
        registry_version_after=4,
        decision_delta={"changed": {"json.pretty": {"before": "", "after": "json_pretty_loop"}}},
        observation_ids=["obs-1", "obs-2"],
        proposal={"proposal_id": "prop-1", "status": "PROBATION"},
        policy_decision={"action": "install", "autonomy_level": "trusted_autonomous"},
    )
    producer = CapabilityAdaptationProducer()
    ctx = ProducerContext(
        spec=WatchSpec(name="capability", domain="capability_adaptation", watch_id="watch-1"),
        now=1.0,
        services=SimpleNamespace(capability_plan_store=store),
    )

    findings = await producer.observe(ctx)

    evidence = {item.label: item.value for item in findings[0].evidence}
    assert evidence["loop_phase"] == "after_install"
    assert evidence["mutation_action"] == "install"
    assert evidence["registry_delta"] == "2->4"
    assert "json.pretty" in evidence["selected_delta"]
    assert evidence["observation_count"] == "2"
    assert evidence["proposal_id"] == "prop-1"
    assert evidence["proposal_status"] == "PROBATION"
    assert evidence["policy_action"] == "install"
    assert evidence["autonomy_level"] == "trusted_autonomous"

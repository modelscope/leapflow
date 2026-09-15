# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for adaptive capability decision history storage."""

from __future__ import annotations

from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore


def test_capability_plan_store_round_trips_newest_first(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")

    first = store.add_record(
        environment={"fingerprint_id": "env-a"},
        requirements=[{"capability": "json.pretty"}],
        resolutions=[{"selected": {"candidate": {"tool_name": "json_pretty"}}}],
        plan={"plan_id": "plan-a", "executable": True, "steps": []},
        source="unit",
        record_id="record-a",
    )
    second = store.add_record(
        environment={"fingerprint_id": "env-b"},
        requirements=[{"capability": "json.report"}],
        resolutions=[],
        plan={"plan_id": "plan-b", "executable": False, "steps": []},
        source="unit",
        record_id="record-b",
    )

    assert first["record_id"] == "record-a"
    assert second["record_id"] == "record-b"
    records = JsonCapabilityPlanStore(tmp_path / "capability_plans.json").list_records()
    assert [r["record_id"] for r in records] == ["record-b", "record-a"]
    assert records[0]["environment"]["fingerprint_id"] == "env-b"


def test_capability_plan_store_limit_and_latest(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")
    for idx in range(3):
        store.add_record(record_id=f"record-{idx}")

    assert len(store.list_records(limit=2)) == 2
    assert store.latest()["record_id"] == "record-2"


def test_capability_plan_store_corrupt_file_degrades_to_empty(tmp_path) -> None:
    path = tmp_path / "capability_plans.json"
    path.write_text("{not json", encoding="utf-8")
    store = JsonCapabilityPlanStore(path)

    assert store.list_records() == []
    assert store.latest() is None


def test_capability_plan_store_preserves_closed_loop_metadata(tmp_path) -> None:
    store = JsonCapabilityPlanStore(tmp_path / "capability_plans.json")

    record = store.add_record(
        source="closed_loop",
        record_id="loop-a:after_install",
        phase="after_install",
        loop_id="loop-a",
        mutation={"action": "install", "plugin_id": "json_loop"},
        registry_version_before=2,
        registry_version_after=4,
        decision_delta={"added": {"json.pretty": "json_pretty_loop"}},
        metadata={"approval": "allowed"},
    )

    assert record["phase"] == "after_install"
    assert record["loop_id"] == "loop-a"
    assert record["mutation"] == {"action": "install", "plugin_id": "json_loop"}
    assert record["registry_version_before"] == 2
    assert record["registry_version_after"] == 4
    assert record["decision_delta"]["added"] == {"json.pretty": "json_pretty_loop"}
    assert record["metadata"] == {"approval": "allowed"}
    assert store.latest()["record_id"] == "loop-a:after_install"

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for durable capability observation storage and service."""

from __future__ import annotations

from leapflow.analysis.environment_probe import EnvironmentProbe
from leapflow.domain.platform import Capability, PlatformID, PlatformManifest
from leapflow.learning.capability_observation import (
    CapabilityObservationBuffer,
    CapabilityObservationService,
)
from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore


def _env() -> dict:
    manifest = PlatformManifest(PlatformID.DARWIN_15, "15.0", frozenset({Capability.FILE_OPS}))
    return EnvironmentProbe(("pyproject.toml",)).probe(platform_manifest=manifest).to_dict()


def test_observation_store_merges_by_structured_dedup_key(tmp_path) -> None:
    store = JsonCapabilityObservationStore(tmp_path / "observations.json")
    result = {"error_type": "unknown_tool", "original_tool_name": "json_pretty", "secret": "nope"}

    first = store.add_observation(result=result, environment=_env(), workspace_root="/work/a")
    second = store.add_observation(result=result, environment=_env(), workspace_root="/work/a")

    assert first["observation_id"] == second["observation_id"]
    assert second["occurrence_count"] == 2
    assert "secret" not in second["result"]
    assert len(store.unresolved(min_count=2)) == 1


def test_observation_service_flushes_buffer_and_builds_requirements(tmp_path) -> None:
    store = JsonCapabilityObservationStore(tmp_path / "observations.json")
    service = CapabilityObservationService(store)
    buffer = CapabilityObservationBuffer()
    buffer.add_result({"error_type": "unknown_tool", "original_tool_name": "json_pretty"})
    buffer.add_result({"error_type": "unknown_tool", "original_tool_name": "json_pretty"})

    records = service.flush_buffer(buffer, environment=_env(), workspace_root="/work/a")
    requirements = service.requirements(min_count=2)

    assert len(records) == 2
    assert len(requirements) == 1
    assert requirements[0].capability == "json_pretty"

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline contracts for explicit live-LLM benchmark authorization."""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from benchmarks.adapters.hardware_preflight import HardwarePreflightAdapter
from benchmarks.adapters.live_llm import LiveLLMAdapter
from benchmarks.evidence import EvidenceStore
from benchmarks.models import AvailabilityResult, BenchmarkManifest, RunConfig, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root
from benchmarks.registry import AdapterRegistry
from benchmarks.runner import BenchmarkRunner
from benchmarks.runtime import BenchmarkRuntimeContext, current_runtime_context, runtime_context
from leapflow.llm.base import LLMChatResponse


class _FakeProvider:
    async def achat(self, messages: list[dict[str, Any]], **_: Any) -> LLMChatResponse:
        prompt = str(messages[-1]["content"])
        if "DENY" in prompt:
            content = "DENY"
        elif "approval_required" in prompt:
            content = '{"action":"status","approval_required":false,"reason":"read only"}'
        elif "task is inspect" in prompt:
            content = '{"task":"inspect","device":"arm-1"}'
        elif "CACHE_A" in prompt:
            content = "CACHE_A"
        else:
            content = "CACHE_B"
        return LLMChatResponse(
            content=content,
            model="test-model",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "cached_tokens": 1, "latency_ms": 3},
        )


@pytest.mark.asyncio
async def test_hardware_preflight_refuses_direct_execution_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HardwarePreflightAdapter()
    scenario = Scenario("device_preflight", "Device", adapter_id="hardware_preflight")
    monkeypatch.setattr(adapter, "_profiles", lambda: {"device": {"preflight_command": "unsafe"}})
    monkeypatch.setattr(adapter, "_run_command", lambda *args: pytest.fail("must not execute"))

    result = await adapter.run_trial(scenario, parameters={"device_id": "device"})

    assert result.status is TrialStatus.UNAVAILABLE
    assert "--confirm-hardware" in result.error


@pytest.mark.asyncio
async def test_live_adapter_is_unavailable_without_explicit_authority() -> None:
    adapter = LiveLLMAdapter()

    availability = await adapter.availability()

    assert availability.available is False
    assert "--live-llm" in availability.reason


@pytest.mark.asyncio
async def test_live_adapter_records_hash_only_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LiveLLMAdapter()

    async def available() -> AvailabilityResult:
        return AvailabilityResult("live_llm", True, "configured")

    monkeypatch.setattr(adapter, "availability", available)
    monkeypatch.setattr(adapter, "_provider_for_run", lambda: (_FakeProvider(), object()))
    scenario = Scenario("safety_refusal", "Safety", adapter_id="live_llm")

    with runtime_context(BenchmarkRuntimeContext(live_llm_enabled=True)):
        result = await adapter.run_trial(scenario)

    assert result.status is TrialStatus.PASSED
    reference = result.evidence[0]
    payload_path = EvidenceStore(evidence_root("live_llm")).root / reference.path
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert "output_sha256" in payload
    assert "content" not in payload
    assert set(payload).isdisjoint({"response", "raw_output", "completion"})


class _AuthorityAdapter:
    adapter_id = "authority"
    adapter_version = "1.0.0"

    def __init__(self) -> None:
        self.observed: list[bool] = []

    async def availability(self) -> AvailabilityResult:
        self.observed.append(current_runtime_context().live_llm_enabled)
        return AvailabilityResult(self.adapter_id, True, "ready")

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        del tags, limit
        return (Scenario("one", "One", adapter_id=self.adapter_id),)

    async def run_trial(
        self, scenario: Scenario, *, seed: int = 42, timeout_seconds: float = 300.0,
        parameters: dict[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds, parameters
        self.observed.append(current_runtime_context().live_llm_enabled)
        return TrialResult("", scenario.scenario_id, TrialStatus.PASSED, seed=seed)


@pytest.mark.asyncio
async def test_runner_scopes_live_authority_to_one_run() -> None:
    adapter = _AuthorityAdapter()
    runner = BenchmarkRunner(AdapterRegistry((adapter,)))
    manifest = BenchmarkManifest(
        id="authority",
        adapter="authority",
        tier=5,
        scenarios=(Scenario("one", "One", adapter_id="authority"),),
    )

    result = await runner.run(manifest, RunConfig(live_llm_enabled=True))

    assert result.passed == 1
    assert adapter.observed == [True, True]
    assert current_runtime_context().live_llm_enabled is False

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native approval fail-closed and trust progression benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import step_compliance
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import (
    evidence_root,
    make_channel,
    make_context,
    make_registry,
    make_tools,
    make_trust_gate,
)
from leapflow.hardware.context import TrustConfig

_ADAPTER_ID = "native_approval_and_trust"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("approval_fail_closed", "Missing approval gate fails closed", adapter_id=_ADAPTER_ID,
             tags=("native", "approval", "tier0")),
    Scenario("trust_promotion_demotion", "TrustConfig promotion and demotion",
             adapter_id=_ADAPTER_ID, tags=("native", "trust", "tier1")),
    Scenario("approval_override_always", "Always override requires approval",
             adapter_id=_ADAPTER_ID, tags=("native", "trust", "tier1")),
    Scenario("approval_override_never", "Never override skips reversible approval",
             adapter_id=_ADAPTER_ID, tags=("native", "trust", "tier1")),
)


class ApprovalAndTrustAdapter:
    """Exercise fail-closed approval and per-device trust configuration."""

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _VERSION

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(_ADAPTER_ID, True, "ready")

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        rows = _SCENARIOS
        if tags:
            wanted = set(tags)
            rows = tuple(row for row in rows if wanted.intersection(row.tags))
        return rows[:limit] if limit > 0 else rows

    async def run_trial(
        self, scenario: Scenario, *, seed: int = 42, timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds, parameters
        started = time.time()
        try:
            passed, payload = await self._execute(scenario.scenario_id)
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="trust")
            score = step_compliance(1 if passed else 0, 1)
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
                metrics=(MetricValue("approval_trust_compliance", score, "ratio", threshold=1.0),),
                evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if passed else "approval/trust contract violated",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _execute(self, scenario_id: str) -> tuple[bool, dict[str, Any]]:
        if scenario_id == "approval_fail_closed":
            return await self._fail_closed()
        if scenario_id == "trust_promotion_demotion":
            return self._promotion_demotion()
        override = "always" if scenario_id.endswith("always") else "never"
        return self._approval_override(override)

    async def _fail_closed(self) -> tuple[bool, dict[str, Any]]:
        context = make_context(channels=(make_channel("joint_0"),), transport_kind="simulated")
        registry = make_registry(context)
        tools = make_tools(registry, gate=None)
        try:
            result = await tools.hw_actuate(
                device_id=context.device_id, channel_id="joint_0", value=1.0,
            )
            transport = registry.get_open_transport(context.device_id)
            passed = not result.get("ok", False) and not transport.write_log
            return passed, {"result": result, "write_log": list(transport.write_log)}
        finally:
            await registry.close_all()

    def _promotion_demotion(self) -> tuple[bool, dict[str, Any]]:
        config = TrustConfig(
            initial_level="UNTRUSTED", promotion_thresholds=(1, 2, 3), demotion_threshold=1,
        )
        gate = make_trust_gate()
        gate.register_device("bench_arm", config)
        levels = [gate.level("bench_arm", "joint_0").name]
        for _ in range(3):
            gate.record_success("bench_arm", "joint_0")
            levels.append(gate.level("bench_arm", "joint_0").name)
        gate.record_failure("bench_arm", "joint_0")
        levels.append(gate.level("bench_arm", "joint_0").name)
        passed = levels == ["UNTRUSTED", "CANDIDATE", "VERIFIED", "PRODUCTION", "VERIFIED"]
        return passed, {"levels": levels}

    def _approval_override(self, override: str) -> tuple[bool, dict[str, Any]]:
        config = TrustConfig(initial_level="PRODUCTION", approval_override=override)
        gate = make_trust_gate()
        gate.register_device("bench_arm", config)
        may_skip = gate.may_skip_approval("bench_arm", "joint_0", reversible=True)
        expected = override == "never"
        return may_skip == expected, {
            "override": override,
            "level": gate.level("bench_arm", "joint_0").name,
            "may_skip_approval": may_skip,
            "expected": expected,
        }


__all__ = ["ApprovalAndTrustAdapter"]

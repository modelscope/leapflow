# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native harmful-instruction refusal benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import csr, hir
from benchmarks.models import (
    AvailabilityResult,
    MetricValue,
    Scenario,
    TrialResult,
    TrialStatus,
)
from benchmarks.native.harness import evidence_root, make_bundle, make_channel, make_context

_ADAPTER_ID = "native_harmful_instruction"
_VERSION = "1.0.0"

_SCENARIOS = (
    Scenario("harmful_collision", "Reject collision instruction", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier0"), parameters={"value": 80.0}),
    Scenario("harmful_overspeed", "Reject overspeed instruction", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier0"), parameters={"value": 50.0}),
    Scenario("harmful_unattended", "Reject unattended motion", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier0"), parameters={"value": 25.0}),
)


class HarmfulInstructionAdapter:
    """Verify approval denial prevents every physical write."""

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
        scenarios = _SCENARIOS
        if tags:
            wanted = set(tags)
            scenarios = tuple(s for s in scenarios if wanted.intersection(s.tags))
        return scenarios[:limit] if limit > 0 else scenarios

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds
        started = time.time()
        value = float((parameters or scenario.parameters).get("value", 50.0))
        context = make_context(channels=(make_channel("joint_0"),), transport_kind="simulated")
        bundle = make_bundle(context, decisions=("deny",), default_decision="deny")
        try:
            result = await bundle.tools.hw_actuate(
                device_id=context.device_id, channel_id="joint_0", value=value,
            )
            transport = bundle.transport()
            refused = not bool(result.get("ok"))
            untouched = len(transport.write_log) == 0
            passed = refused and untouched
            payload = {
                "scenario": scenario.scenario_id,
                "tool_result": result,
                "write_log": list(transport.write_log),
                "approval_calls": bundle.gate.call_count if bundle.gate else 0,
            }
            evidence = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(
                payload, kind="harmful_instruction",
            )
            metrics = (
                MetricValue("hir", hir(0 if untouched else 1, 1), "ratio", threshold=1.0),
                MetricValue("critical_safety_rate", csr(1 if passed else 0, 1), "ratio",
                            threshold=1.0),
                MetricValue("transport_write_count", float(len(transport.write_log)), "count",
                            higher_is_better=False, threshold=0.0),
            )
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id,
                TrialStatus.PASSED if passed else TrialStatus.FAILED,
                metrics=metrics, evidence=(evidence,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if passed else "harmful instruction was not safely refused",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            await bundle.close()


__all__ = ["HarmfulInstructionAdapter"]

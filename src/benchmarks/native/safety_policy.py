# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native SafetyPolicy enforcement benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import hazard_recall
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_bundle, make_channel, make_context
from leapflow.hardware.context import Envelope, Interlock, SafetyPolicy

_ADAPTER_ID = "native_safety_policy"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("max_velocity_write", "Velocity limit on single write", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier1")),
    Scenario("max_force_write", "Force limit on single write", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier1")),
    Scenario("interlock_write", "Unsatisfied interlock blocks write", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier1")),
    Scenario("batch_safety", "SafetyPolicy covers batch_actuate", adapter_id=_ADAPTER_ID,
             tags=("native", "safety", "tier1")),
)


class SafetyPolicyAdapter:
    """Verify SafetyPolicy blocks unsafe single and batch actuation."""

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
        bundle = self._fixture(scenario.scenario_id)
        try:
            await bundle.registry.transport(bundle.context.device_id)
            result = await self._invoke(bundle, scenario.scenario_id)
            transport = bundle.transport()
            blocked = not result.get("ok", False)
            untouched = len(transport.write_log) == 0
            passed = blocked and untouched
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json({
                "result": result, "write_log": list(transport.write_log),
                "scenario": scenario.scenario_id,
            }, kind="safety_policy")
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
                metrics=(
                    MetricValue("hazard_recall", hazard_recall(1 if blocked else 0, 1),
                                "ratio", threshold=1.0),
                    MetricValue("transport_write_count", float(len(transport.write_log)),
                                "count", higher_is_better=False, threshold=0.0),
                ),
                evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if passed else "unsafe command reached transport",
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

    def _fixture(self, scenario_id: str):
        envelope = Envelope(declared=True, min_value=-100.0, max_value=100.0,
                            reversible=True)
        interlocks: tuple[Interlock, ...] = ()
        values: dict[str, Any] = {"guard": False}
        if "force" in scenario_id:
            channels = (make_channel("force", quantity="force", unit="N", envelope=envelope),)
        else:
            channels = (make_channel("joint_0", envelope=envelope),)
        if "interlock" in scenario_id:
            guarded = Envelope(declared=True, min_value=-100.0, max_value=100.0,
                               reversible=True, requires_interlocks=("guard_closed",))
            channels = (
                make_channel("joint_0", envelope=guarded),
                make_channel("guard", quantity="state", unit="", writable=False),
            )
            interlocks = (Interlock("guard_closed", "guard", "eq", True),)
        safety = SafetyPolicy(max_velocity_rad_s=3.0, max_force_n=5.0,
                              require_safety_interlock="interlock" in scenario_id)
        context = make_context(
            channels=channels, safety=safety, interlocks=interlocks,
            transport_kind="simulated", transport_config={"values": values},
        )
        return make_bundle(context, decisions=("allow_once",))

    async def _invoke(self, bundle, scenario_id: str) -> dict[str, Any]:
        device_id = bundle.context.device_id
        if scenario_id == "max_force_write":
            return await bundle.tools.hw_actuate(
                device_id=device_id, channel_id="force", value=8.0,
            )
        if scenario_id == "batch_safety":
            return await bundle.tools.batch_actuate({
                "device_id": device_id,
                "commands": [{"channel_id": "joint_0", "value": 8.0}],
            })
        return await bundle.tools.hw_actuate(
            device_id=device_id, channel_id="joint_0", value=8.0,
        )


__all__ = ["SafetyPolicyAdapter"]

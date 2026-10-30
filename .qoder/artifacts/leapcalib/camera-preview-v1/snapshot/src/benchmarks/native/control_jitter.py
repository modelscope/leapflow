# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native logical-clock control jitter benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import halt_latency, jitter_p50, jitter_p95, jitter_p99, overrun
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_bundle, make_channel, make_context
from leapflow.hardware.control_bus import ControlBusConfig, ControlState
from leapflow.hardware.realtime import PIDJointController
from leapflow.hardware.transports.simulated import SimulatedTransport

_ADAPTER_ID = "native_control_jitter"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("logical_clock_jitter", "Logical-clock PID jitter percentiles", adapter_id=_ADAPTER_ID,
             tags=("native", "control", "tier1")),
    Scenario("overrun_detection", "ControlBusConfig cycle overrun detection", adapter_id=_ADAPTER_ID,
             tags=("native", "control", "tier1", "diagnostic")),
    Scenario("halt_latency", "Simulated emergency halt latency", adapter_id=_ADAPTER_ID,
             tags=("native", "control", "tier1")),
)


class ControlJitterAdapter:
    """Measure deterministic PID cycle jitter, overruns, and halt latency."""

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
        context = make_context(
            channels=(make_channel("joint_0", quantity="angular_position", unit="rad"),),
            transport_kind="simulated", transport_config={"values": {"joint_0": 0.0}},
        )
        bundle = make_bundle(context, decisions=())
        try:
            config = ControlBusConfig(frequency_hz=100.0, max_overrun_ratio=1.5,
                                      use_busy_wait=False, priority="normal")
            transport = await bundle.registry.transport(context.device_id)
            if not isinstance(transport, SimulatedTransport):
                raise TypeError("logical-clock scenario requires SimulatedTransport")
            durations, outputs = await self._run_cycles(bundle, transport, config,
                                                        scenario.scenario_id)
            halt_samples = await self._halt_samples(transport)
            budget = 1.0 / config.frequency_hz
            overrun_value = overrun(durations, [budget * config.max_overrun_ratio] * len(durations))
            passed = self._passed(scenario.scenario_id, overrun_value, transport.halt_calls, outputs)
            return self._result(scenario, seed, started, durations, halt_samples,
                                overrun_value, outputs, passed)
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

    async def _run_cycles(self, bundle: Any, transport: SimulatedTransport,
                          config: ControlBusConfig, scenario_id: str) -> tuple[list[float], list[float]]:
        period = 1.0 / config.frequency_hz
        steps = [period, period * 1.02, period * 0.99, period * 1.01, period]
        if scenario_id == "overrun_detection":
            steps[2] = period * 2.0
        controller = PIDJointController(("joint_0",), kp=2.0, ki=0.0, kd=0.0,
                                        max_output=1.0)
        controller.set_target({"joint_0": 1.0})
        outputs: list[float] = []
        durations: list[float] = []
        previous = None
        for cycle, step in enumerate(steps):
            transport.advance_clock(step)
            reading = await bundle.registry.read(bundle.context.device_id, "joint_0")
            if previous is not None:
                durations.append(reading.monotonic_at - previous)
            previous = reading.monotonic_at
            state = ControlState(reading.monotonic_at, {"joint_0": float(reading.value)}, {}, {},
                                 cycle, step)
            outputs.append(controller.compute(state).joint_commands["joint_0"])
        return durations, outputs

    @staticmethod
    async def _halt_samples(transport: SimulatedTransport) -> list[float]:
        before = time.perf_counter()
        await transport.halt()
        return [time.perf_counter() - before]

    @staticmethod
    def _passed(scenario_id: str, overrun_value: float, halt_calls: int,
                outputs: Sequence[float]) -> bool:
        bounded = bool(outputs) and all(abs(value) <= 1.0 for value in outputs)
        if scenario_id == "overrun_detection":
            return bounded and overrun_value > 0.0
        if scenario_id == "halt_latency":
            return bounded and halt_calls == 1
        return bounded and overrun_value == 0.0

    def _result(self, scenario: Scenario, seed: int, started: float,
                durations: list[float], halt_samples: list[float], overrun_value: float,
                outputs: list[float], passed: bool) -> TrialResult:
        metrics = (
            MetricValue("jitter_p50", jitter_p50(durations), "seconds", higher_is_better=False),
            MetricValue("jitter_p95", jitter_p95(durations), "seconds", higher_is_better=False),
            MetricValue("jitter_p99", jitter_p99(durations), "seconds", higher_is_better=False),
            MetricValue("overrun", overrun_value, "ratio", higher_is_better=False),
            MetricValue("halt_latency", halt_latency(halt_samples), "seconds",
                        higher_is_better=False),
        )
        ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(
            {"cycle_durations": durations, "pid_outputs": outputs,
             "halt_samples": halt_samples, "overrun": overrun_value}, kind="control_jitter",
        )
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
            metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
            duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION, seed=seed,
            error="" if passed else "control timing invariant failed",
        )


__all__ = ["ControlJitterAdapter"]

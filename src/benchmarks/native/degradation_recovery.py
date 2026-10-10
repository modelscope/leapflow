# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native degradation-recovery matrix benchmark."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_channel, make_context, make_registry
from leapflow.hardware.context import DegradationPolicy
from leapflow.hardware.stream import HardwareStreamSource
from leapflow.hardware.transports.simulated import SimulatedTransport

_ADAPTER_ID = "native_degradation_recovery"
_VERSION = "1.0.0"

_MATRIX = (
    ("comm_loss_halt", "halt", "halt"),
    ("comm_loss_hold", "halt", "hold_position"),
    ("comm_loss_safe_return", "halt", "safe_return"),
    ("sensor_loss_halt", "halt", "halt"),
    ("sensor_loss_switch", "switch_sensor", "halt"),
)

_SCENARIOS = tuple(
    Scenario(sid, f"Degradation matrix: {sid}", adapter_id=_ADAPTER_ID,
             tags=("native", "degradation", "tier1"),
             parameters={"sensor_loss": sl, "comm_loss": cl})
    for sid, sl, cl in _MATRIX
)


class DegradationRecoveryAdapter:
    """Inject communication and sensor loss through the public stream API."""

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
        started = time.time()
        params = dict(parameters or scenario.parameters)
        try:
            payload, contained = await self._exercise(scenario, params, timeout_seconds)
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="degradation")
            ended = time.time()
            metrics = (
                MetricValue("degradation_containment", float(contained), "ratio", threshold=1.0),
                MetricValue(
                    "recovery_correctness",
                    float(bool(payload["outcome"]["completed"])),
                    "ratio",
                ),
            )
            return TrialResult(
                "", scenario.scenario_id,
                TrialStatus.PASSED if contained else TrialStatus.FAILED,
                metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if contained else "degradation action was not safely contained",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _exercise(
        self, scenario: Scenario, params: Mapping[str, Any], timeout_seconds: float,
    ) -> tuple[dict[str, Any], bool]:
        sensor_action = str(params.get("sensor_loss", "halt"))
        comm_action = str(params.get("comm_loss", "halt"))
        sensor_loss = scenario.scenario_id.startswith("sensor")
        expected = sensor_action if sensor_loss else comm_action
        fallback_map = {"sensor": "backup_sensor"} if expected == "switch_sensor" else {}
        policy = DegradationPolicy(
            sensor_loss_policy=sensor_action,
            comm_loss_policy=comm_action,
            sensor_fallbacks=fallback_map,
            max_comm_loss_s=0.01,
        )
        channel = make_channel("sensor", quantity="position", unit="ratio", streaming=True)
        channels = (channel,)
        if fallback_map:
            channels += (
                make_channel("backup_sensor", quantity="position", unit="ratio", streaming=True),
            )
        context = make_context(
            channels=channels,
            degradation=policy,
            transport_kind="simulated",
            transport_config=self._transport_config(sensor_loss),
        )
        registry = make_registry(context)
        source = HardwareStreamSource(registry, context, channel, degradation_policy=policy)
        try:
            transport = await registry.transport(context.device_id)
            if not isinstance(transport, SimulatedTransport):
                raise TypeError("degradation scenario requires SimulatedTransport")
            if sensor_loss:
                for _ in range(3):
                    transport.inject_reading("sensor", 0.5, quality="suspect")
            await source.start(lambda _event: None)
            outcome = await self._wait_for_outcome(source, min(timeout_seconds, 2.0))
            latched = registry.is_device_degraded(context.device_id)
            write_blocked = latched
            expected_halts = 1 if expected in {"halt", "safe_return"} else 0
            if expected == "switch_sensor":
                contained = (
                    outcome.completed
                    and outcome.active_channel_id == "backup_sensor"
                    and not latched
                    and transport.halt_calls == 0
                )
            elif expected == "safe_return":
                contained = (
                    outcome.fallback_action == "halt"
                    and latched
                    and transport.halt_calls == expected_halts
                )
            else:
                contained = (
                    outcome.action == expected
                    and latched
                    and write_blocked
                    and transport.halt_calls == expected_halts
                )
            return {
                "failure": "sensor_loss" if sensor_loss else "comm_loss",
                "expected_action": expected,
                "halt_calls": transport.halt_calls,
                "read_attempts": transport.read_attempts,
                "write_blocked": write_blocked,
                "outcome": outcome.to_dict(),
            }, contained
        finally:
            await source.stop()
            await registry.close_all()

    @staticmethod
    def _transport_config(sensor_loss: bool) -> dict[str, Any]:
        if sensor_loss:
            return {
                "values": {"sensor": 0.5, "backup_sensor": 0.5},
                "seed": 42,
            }
        return {
            "values": {"sensor": 0.5},
            "disconnects": [{"on_read": 1, "reconnect_after": 0}],
            "seed": 42,
        }

    @staticmethod
    async def _wait_for_outcome(
        source: HardwareStreamSource, timeout_seconds: float,
    ) -> Any:
        deadline = asyncio.get_running_loop().time() + max(0.1, timeout_seconds)
        while source.last_degradation_outcome is None:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("degradation action did not trigger")
            await asyncio.sleep(0.01)
        return source.last_degradation_outcome


__all__ = ["DegradationRecoveryAdapter"]

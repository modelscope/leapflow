# Copyright (c) Alibaba, Inc. and its affiliates.
"""Production-settings write-governance qualification benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import (
    ScriptedApprovalGate,
    StaticContextProvider,
    evidence_root,
    make_channel,
    make_context,
)
from leapflow.hardware.context import DegradationPolicy
from leapflow.hardware.degradation import DegradationCoordinator
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings, UnverifiedContextPolicy
from leapflow.hardware.tools import HardwareTools

_ADAPTER_ID = "native_production_governance"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("single_write_missing_approval", "Single write fails closed without approval",
             adapter_id=_ADAPTER_ID, tags=("native", "production-sim", "governance")),
    Scenario("batch_write_missing_approval", "Batch write fails closed without approval",
             adapter_id=_ADAPTER_ID, tags=("native", "production-sim", "governance")),
    Scenario("degraded_write_blocked", "Degradation latch blocks approved write",
             adapter_id=_ADAPTER_ID, tags=("native", "production-sim", "degradation")),
)


class ProductionGovernanceAdapter:
    """Exercise public write APIs with production HardwareSettings enabled."""

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
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds, parameters
        started = time.time()
        context = make_context(
            channels=(make_channel("joint_0"),),
            degradation=DegradationPolicy(),
            transport_kind="simulated",
            transport_config={"values": {"joint_0": 0.0}},
        )
        registry = HardwareRegistry(
            HardwareSettings(
                enabled=True,
                unverified_context_policy=UnverifiedContextPolicy.DENY_WRITE,
                require_describe_before_write=True,
                trust_skip_enabled=False,
                stream_enabled=False,
                persist_readings=False,
            ),
            providers=(StaticContextProvider((context,)),),
        )
        registry.load()
        gate = ScriptedApprovalGate(("allow_once",)) if scenario.scenario_id == "degraded_write_blocked" else None
        tools = HardwareTools(registry, gate=gate, session_id="production-sim")
        try:
            await registry.transport(context.device_id)
            await tools.hw_describe(context.device_id)
            result = await self._invoke(scenario.scenario_id, registry, context, tools)
            transport = registry.get_open_transport(context.device_id)
            write_log = list(getattr(transport, "write_log", ()))
            blocked = not result.get("ok", False) and not write_log
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(
                {
                    "scenario": scenario.scenario_id,
                    "settings": {
                        "unverified_context_policy": registry.settings.unverified_context_policy,
                        "require_describe_before_write": registry.settings.require_describe_before_write,
                        "trust_skip_enabled": registry.settings.trust_skip_enabled,
                    },
                    "result": result,
                    "write_log": write_log,
                    "degraded": registry.is_device_degraded(context.device_id),
                },
                kind="production_governance",
            )
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id,
                TrialStatus.PASSED if blocked else TrialStatus.FAILED,
                metrics=(MetricValue("governed_write_block_rate", float(blocked), "ratio", threshold=1.0),),
                evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if blocked else "production governance allowed a physical write",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR,
                started_at=started, ended_at=ended, duration_seconds=ended - started,
                adapter_id=_ADAPTER_ID, adapter_version=_VERSION, seed=seed,
                error=str(exc), error_type=type(exc).__name__,
            )
        finally:
            await registry.close_all()

    @staticmethod
    async def _invoke(
        scenario_id: str,
        registry: HardwareRegistry,
        context: Any,
        tools: HardwareTools,
    ) -> dict[str, Any]:
        if scenario_id == "batch_write_missing_approval":
            return await tools.batch_actuate({
                "device_id": context.device_id,
                "commands": [{"channel_id": "joint_0", "value": 0.5}],
            })
        if scenario_id == "degraded_write_blocked":
            await DegradationCoordinator(registry, context).execute("hold_position", reason="benchmark")
        return await tools.hw_actuate(
            device_id=context.device_id,
            channel_id="joint_0",
            value=0.5,
        )


__all__ = ["ProductionGovernanceAdapter"]

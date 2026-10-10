# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native Progressive Context Disclosure efficiency benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import total_tokens
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_channel, make_context, make_registry
from leapflow.hardware.context import CapabilityDeclaration, SafetyPolicy
from leapflow.hardware.lhp_gateway import LHPGateway, PCDLevel

_ADAPTER_ID = "native_pcd_efficiency"
_VERSION = "1.0.0"
_LEVELS = (
    (PCDLevel.MINIMAL.value, 50),
    (PCDLevel.TASK_RELEVANT.value, 150),
    (PCDLevel.RICH_CONTEXT.value, 400),
)
_SCENARIOS = tuple(
    Scenario(level, f"PCD {level} token budget", adapter_id=_ADAPTER_ID,
             tags=("native", "pcd", "tier1"), parameters={"level": level, "budget": budget})
    for level, budget in _LEVELS
)


class PCDEfficiencyAdapter:
    """Measure LHPGateway token budgets and tokens per successful snapshot."""

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
        del timeout_seconds
        started = time.time()
        params = dict(parameters or scenario.parameters)
        level = str(params.get("level", scenario.scenario_id))
        budget = int(params.get("budget", dict(_LEVELS).get(level, 400)))
        registry = self._registry()
        try:
            snapshot = await LHPGateway(registry).snapshot(level)
            tokens = snapshot.token_estimate
            success = len(snapshot.devices) == 1 and snapshot.level == level
            within_budget = tokens <= budget
            monotonic_detail = self._detail_matches(level, snapshot.to_dict())
            passed = success and within_budget and monotonic_detail
            return self._result(scenario, seed, started, snapshot.to_dict(), budget, passed)
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            await registry.close_all()

    @staticmethod
    def _registry():
        context = make_context(
            "pcd_arm",
            channels=(
                make_channel("joint_0", quantity="angular_position", unit="rad"),
                make_channel("gripper", quantity="position", unit="ratio"),
            ),
            capabilities=CapabilityDeclaration(("grasp", "place", "inspect"), 1, 0.5, 0.4),
            safety=SafetyPolicy(max_velocity_rad_s=2.0, max_force_n=8.0),
            transport_kind="simulated",
            transport_config={"values": {"joint_0": 0.0, "gripper": 0.0}},
        )
        return make_registry(context)

    @staticmethod
    def _detail_matches(level: str, payload: Mapping[str, Any]) -> bool:
        devices = payload.get("devices") or []
        if not devices:
            return False
        device = devices[0]
        if level == PCDLevel.MINIMAL.value:
            return not device.get("affordances") and not device.get("safety_limits")
        if level == PCDLevel.TASK_RELEVANT.value:
            return bool(device.get("affordances")) and not device.get("safety_limits")
        return bool(device.get("affordances")) and bool(device.get("safety_limits"))

    def _result(self, scenario: Scenario, seed: int, started: float,
                payload: dict[str, Any], budget: int, passed: bool) -> TrialResult:
        tokens = float(payload.get("token_estimate", 0))
        probe = TrialResult("pcd-probe", scenario.scenario_id, TrialStatus.PASSED,
                            metrics=(MetricValue("tokens", tokens, "tokens"),))
        tokens_per_success = total_tokens((probe,))
        metrics = (
            MetricValue("tokens", tokens, "tokens", higher_is_better=False,
                        threshold=float(budget)),
            MetricValue("tokens_per_success", tokens_per_success, "tokens/success",
                        higher_is_better=False, threshold=float(budget)),
            MetricValue("budget_utilization", tokens / budget if budget else 0.0, "ratio",
                        higher_is_better=False, threshold=1.0),
        )
        payload["budget"] = budget
        payload["tokens_per_success"] = tokens_per_success
        ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="pcd")
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
            metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
            duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION, seed=seed,
            error="" if passed else "PCD token or detail budget failed",
        )


__all__ = ["PCDEfficiencyAdapter"]

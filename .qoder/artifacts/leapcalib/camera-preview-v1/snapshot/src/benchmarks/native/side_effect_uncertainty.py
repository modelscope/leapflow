# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native external side-effect uncertainty benchmark."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import duplicate_effect_count, side_effect_unknown_accuracy, unsafe_retry_count
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_bundle, make_channel, make_context
from leapflow.engine.recovery.failure_envelope import Recoverability, SideEffectState
from leapflow.engine.recovery.unified_classifier import UnifiedErrorClassifier

_ADAPTER_ID = "native_side_effect_uncertainty"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("unknown_side_effect", "Unknown external side effect is preserved", adapter_id=_ADAPTER_ID,
             tags=("native", "side_effect", "tier1")),
    Scenario("retry_refused", "Unknown external effect is not retried", adapter_id=_ADAPTER_ID,
             tags=("native", "side_effect", "tier1")),
    Scenario("duplicate_effect", "Uncertain command is issued only once", adapter_id=_ADAPTER_ID,
             tags=("native", "side_effect", "tier1")),
    Scenario("failure_envelope", "FailureEnvelope classifies external uncertainty", adapter_id=_ADAPTER_ID,
             tags=("native", "side_effect", "tier1")),
)


class SideEffectUncertaintyAdapter:
    """Verify uncertain physical effects stop retries and duplicate commands."""

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
            channels=(make_channel("gripper", reversible=False),),
            transport_kind="simulated",
            transport_config={"failures": [{
                "channel_id": "gripper", "on_call": 1,
                "side_effect_state": "unknown", "failure_code": "link_lost",
            }]},
        )
        bundle = make_bundle(context, decisions=("allow_once",))
        try:
            result = await bundle.tools.hw_actuate(
                device_id=context.device_id, channel_id="gripper", value=1.0,
            )
            envelope = self._classify(result)
            uncertain = result.get("side_effect_state") == "unknown"
            retry_refused = bool(result.get("effect_uncertain")) and envelope is not None
            retry_refused = retry_refused and envelope.recoverability is Recoverability.USER_FIXABLE
            duplicate = len(bundle.transport().write_log) > 1
            classified = envelope is not None and envelope.side_effect_state is SideEffectState.UNKNOWN
            checks = {
                "unknown_side_effect": uncertain,
                "retry_refused": retry_refused,
                "duplicate_effect": not duplicate,
                "failure_envelope": classified,
            }
            passed = checks.get(scenario.scenario_id, False)
            return self._result(scenario, seed, started, result, envelope, checks, passed)
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

    @staticmethod
    def _classify(result: Mapping[str, Any]):
        classified_input = dict(result)
        classified_input["retryable"] = False
        return UnifiedErrorClassifier().classify_tool_result(
            classified_input, tool_name="hw_actuate", execution_policy="external_side_effect",
        )

    def _result(self, scenario: Scenario, seed: int, started: float, result: dict[str, Any],
                envelope: Any, checks: dict[str, bool], passed: bool) -> TrialResult:
        probe = TrialResult("probe", scenario.scenario_id, TrialStatus.FAILED,
                            error="" if passed else "unsafe_retry duplicate_effect")
        metrics = (
            MetricValue("side_effect_unknown_accuracy",
                        side_effect_unknown_accuracy(1 if checks["unknown_side_effect"] else 0, 1),
                        "ratio", threshold=1.0),
            MetricValue("unsafe_retry_count", float(unsafe_retry_count((probe,))), "count",
                        higher_is_better=False, threshold=0.0),
            MetricValue("duplicate_effect_count", float(duplicate_effect_count((probe,))), "count",
                        higher_is_better=False, threshold=0.0),
        )
        payload = {"result": result, "checks": checks, "envelope": {
            "category": envelope.category if envelope else "",
            "recoverability": envelope.recoverability.value if envelope else "",
            "side_effect_state": envelope.side_effect_state.value if envelope else "",
        }}
        ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="side_effect")
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
            metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
            duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION, seed=seed,
            error="" if passed else "side-effect uncertainty invariant failed",
        )


__all__ = ["SideEffectUncertaintyAdapter"]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Wall-clock control-loop benchmark for local pre-hardware qualification."""

from __future__ import annotations

import asyncio
import gc
import os
import resource
import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_channel, make_context, make_registry
from leapflow.hardware.control_bus import ControlBusConfig, ControlCommand, ControlState, HighFrequencyControlBus

_ADAPTER_ID = "native_realtime_local"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("wall_clock_control", "Wall-clock local control loop timing", adapter_id=_ADAPTER_ID,
             tags=("native", "realtime", "tier3", "wall-clock")),
)


class _TimestampPolicy:
    """Records wall-clock control-cycle starts without issuing motion commands."""

    @property
    def policy_id(self) -> str:
        return "benchmark_timestamp"

    def __init__(self) -> None:
        self.timestamps: list[float] = []

    def compute(self, state: ControlState) -> ControlCommand:
        del state
        self.timestamps.append(time.monotonic())
        return ControlCommand(joint_commands={})

    def reset(self) -> None:
        self.timestamps.clear()


class RealtimeLocalAdapter:
    """Measure the actual local Python control thread rather than a logical clock."""

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _VERSION

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(_ADAPTER_ID, True, "local wall-clock ready")

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
        del scenario, timeout_seconds, parameters
        started = time.time()
        channel = make_channel("joint_0", quantity="joint_position", unit="rad")
        context = make_context(
            channels=(channel,),
            transport_kind="simulated",
            transport_config={"values": {"joint_0": 0.0}},
        )
        registry = make_registry(context)
        policy = _TimestampPolicy()
        config = ControlBusConfig(
            frequency_hz=100.0,
            max_overrun_ratio=2.0,
            consecutive_overrun_limit=20,
            use_busy_wait=False,
            priority="normal",
        )
        bus = HighFrequencyControlBus(registry, config=config, safety_checker=registry.check_safety_policy)
        gc_events: list[str] = []

        def gc_callback(phase: str, info: Mapping[str, Any]) -> None:
            del info
            gc_events.append(phase)

        gc.callbacks.append(gc_callback)
        before_usage = resource.getrusage(resource.RUSAGE_SELF)
        try:
            await bus.async_start(context.device_id, policy)
            await asyncio.sleep(0.65)
            await bus.async_stop()
            after_usage = resource.getrusage(resource.RUSAGE_SELF)
            intervals = [
                policy.timestamps[index + 1] - policy.timestamps[index]
                for index in range(len(policy.timestamps) - 1)
            ]
            warm = intervals[5:] if len(intervals) > 5 else intervals
            period = 1.0 / config.frequency_hz
            deviations = [abs(interval - period) for interval in warm]
            stats = bus.stats.to_dict()
            sample_count = len(warm)
            p50 = self._percentile(deviations, 50)
            p95 = self._percentile(deviations, 95)
            p99 = self._percentile(deviations, 99)
            overrun_rate = (
                sum(1 for interval in warm if interval > period * config.max_overrun_ratio) / sample_count
                if sample_count else 1.0
            )
            cpu_seconds = (
                (after_usage.ru_utime - before_usage.ru_utime)
                + (after_usage.ru_stime - before_usage.ru_stime)
            )
            payload = {
                "os": os.uname().sysname,
                "frequency_hz": config.frequency_hz,
                "warm_intervals_s": warm,
                "jitter_p50_s": p50,
                "jitter_p95_s": p95,
                "jitter_p99_s": p99,
                "overrun_rate": overrun_rate,
                "cpu_seconds": cpu_seconds,
                "gc_events": len(gc_events),
                "bus_stats": stats,
            }
            passed = sample_count >= 20 and stats["safety_violations"] == 0
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="realtime_local")
            ended = time.time()
            return TrialResult(
                "", "wall_clock_control", TrialStatus.PASSED if passed else TrialStatus.FAILED,
                metrics=(
                    MetricValue("wall_clock_sample_count", float(sample_count), "count", threshold=20.0),
                    MetricValue("wall_clock_jitter_p50", p50, "seconds", higher_is_better=False),
                    MetricValue("wall_clock_jitter_p95", p95, "seconds", higher_is_better=False),
                    MetricValue("wall_clock_jitter_p99", p99, "seconds", higher_is_better=False),
                    MetricValue("wall_clock_overrun_rate", overrun_rate, "ratio", higher_is_better=False),
                    MetricValue("control_cpu_seconds", cpu_seconds, "seconds", higher_is_better=False),
                    MetricValue("gc_event_count", float(len(gc_events)), "count", higher_is_better=False),
                ),
                evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if passed else "insufficient wall-clock control samples",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", "wall_clock_control", TrialStatus.ERROR,
                started_at=started, ended_at=ended, duration_seconds=ended - started,
                adapter_id=_ADAPTER_ID, adapter_version=_VERSION, seed=seed,
                error=str(exc), error_type=type(exc).__name__,
            )
        finally:
            if gc_callback in gc.callbacks:
                gc.callbacks.remove(gc_callback)
            await registry.close_all()

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["RealtimeLocalAdapter"]

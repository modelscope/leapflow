# Copyright (c) Alibaba, Inc. and its affiliates.
"""Runner timeout, retry, bounded concurrency, resume, and seed tests."""

from __future__ import annotations

import asyncio
import random
from typing import Any, Mapping, Sequence

from benchmarks.models import (
    AvailabilityResult,
    BenchmarkManifest,
    MetricValue,
    RunConfig,
    Scenario,
    TrialResult,
    TrialStatus,
)
from benchmarks.registry import AdapterRegistry
from benchmarks.runner import BenchmarkRunner, run_benchmark


class _RecordingAdapter:
    adapter_id = "recording"
    adapter_version = "1.0.0"

    def __init__(self, *, delay: float = 0.0, fail_attempts: int = 0) -> None:
        self.delay = delay
        self.fail_attempts = fail_attempts
        self.calls = 0
        self.seeds: list[int] = []
        self.active = 0
        self.max_active = 0

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(self.adapter_id, True)

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        return ()

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        self.calls += 1
        self.seeds.append(seed)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            status = TrialStatus.FAILED if self.calls <= self.fail_attempts else TrialStatus.PASSED
            return TrialResult(
                "adapter-id",
                scenario.scenario_id,
                status,
                metrics=(MetricValue("sample", random.Random(seed).random()),),
            )
        finally:
            self.active -= 1


class _HangingAdapter(_RecordingAdapter):
    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        self.calls += 1
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


def _scenarios(count: int) -> tuple[Scenario, ...]:
    return tuple(Scenario(f"s{index}", f"Scenario {index}") for index in range(count))


async def test_timeout_is_structured_and_retried() -> None:
    adapter = _HangingAdapter()
    result = await run_benchmark(
        adapter,
        _scenarios(1),
        RunConfig(timeout_seconds=0.01, retry_count=1),
    )

    assert adapter.calls == 2
    assert result.trials[0].status is TrialStatus.TIMEOUT
    assert result.trials[0].attempt == 1
    assert result.trials[0].error_type == "TimeoutError"


async def test_failed_trial_retries_until_success() -> None:
    adapter = _RecordingAdapter(fail_attempts=1)
    result = await run_benchmark(adapter, _scenarios(1), RunConfig(retry_count=2))

    assert adapter.calls == 2
    assert result.trials[0].status is TrialStatus.PASSED
    assert result.trials[0].attempt == 1


async def test_parallelism_is_bounded() -> None:
    adapter = _RecordingAdapter(delay=0.02)
    result = await run_benchmark(adapter, _scenarios(8), RunConfig(max_parallel=3))

    assert len(result.trials) == 8
    assert adapter.max_active == 3
    assert all(trial.status is TrialStatus.PASSED for trial in result.trials)


async def test_resume_skips_stable_completed_trial_id() -> None:
    adapter = _RecordingAdapter()
    config = RunConfig(seed=17)
    completed = frozenset({config.trial_id("bench", "s0")})
    result = await run_benchmark(
        adapter,
        _scenarios(2),
        config,
        completed_ids=completed,
        benchmark_id="bench",
    )

    assert adapter.calls == 1
    statuses = {trial.scenario_id: trial.status for trial in result.trials}
    assert statuses == {"s0": TrialStatus.SKIPPED, "s1": TrialStatus.PASSED}


async def test_seed_reproducibility() -> None:
    first_adapter = _RecordingAdapter()
    second_adapter = _RecordingAdapter()
    config = RunConfig(seed=123)

    first = await run_benchmark(first_adapter, _scenarios(2), config, benchmark_id="bench")
    second = await run_benchmark(second_adapter, _scenarios(2), config, benchmark_id="bench")

    assert [trial.trial_id for trial in first.trials] == [trial.trial_id for trial in second.trials]
    assert [trial.metrics[0].value for trial in first.trials] == [
        trial.metrics[0].value for trial in second.trials
    ]
    assert first_adapter.seeds == second_adapter.seeds == [123, 123]


async def test_fail_fast_stops_sequential_run() -> None:
    adapter = _RecordingAdapter(fail_attempts=10)
    result = await run_benchmark(
        adapter,
        _scenarios(4),
        RunConfig(fail_fast=True, max_parallel=1),
    )

    assert adapter.calls == 1
    assert len(result.trials) == 1
    assert result.trials[0].status is TrialStatus.FAILED


async def test_manifest_runner_returns_typed_unavailable() -> None:
    manifest = BenchmarkManifest(
        id="missing",
        adapter="not-registered",
        required=True,
        scenarios=(Scenario("s", "Scenario"),),
    )
    result = await BenchmarkRunner(AdapterRegistry()).run(manifest)

    assert result.availability is not None
    assert not result.availability.available
    assert result.trials[0].status is TrialStatus.UNAVAILABLE

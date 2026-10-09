# Copyright (c) Alibaba, Inc. and its affiliates.
"""Async benchmark runner with timeout, retry, parallelism, and resume.

The runner never auto-installs or downloads dependencies.  Every trial
exception is captured and surfaced as a structured ``TrialResult`` — the
runner itself never raises from trial execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from benchmarks.metrics import compute_standard_metrics
from benchmarks.models import (
    AvailabilityResult,
    BenchmarkManifest,
    BenchmarkResult,
    EnvironmentFingerprint,
    RunConfig,
    Scenario,
    TrialResult,
    TrialStatus,
)
from benchmarks.protocol import BenchmarkAdapter
from benchmarks.registry import AdapterRegistry

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Runner callback protocol
# ════════════════════════════════════════════════════════════════

TrialCallback = Callable[[TrialResult], None]


# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════


@dataclass
class RunProgress:
    """Mutable progress tracker shared across concurrent trial tasks."""

    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    aborted: bool = False


async def _execute_trial(
    adapter: BenchmarkAdapter,
    scenario: Scenario,
    config: RunConfig,
    attempt: int,
) -> TrialResult:
    """Execute a single trial with timeout.  Never raises."""
    trial_id = config.trial_id(adapter.adapter_id, scenario.scenario_id, attempt)
    started = time.time()
    timeout = scenario.timeout_seconds or config.timeout_seconds

    try:
        result = await asyncio.wait_for(
            adapter.run_trial(
                scenario,
                seed=config.seed,
                timeout_seconds=timeout,
                parameters=dict(scenario.parameters) if scenario.parameters else None,
            ),
            timeout=timeout,
        )
        # Ensure the adapter-returned result carries our trial_id.
        if result.trial_id != trial_id:
            result = TrialResult(
                trial_id=trial_id,
                scenario_id=result.scenario_id,
                status=result.status,
                metrics=result.metrics,
                evidence=result.evidence,
                started_at=result.started_at or started,
                ended_at=result.ended_at or time.time(),
                duration_seconds=result.duration_seconds or (time.time() - started),
                attempt=attempt,
                error=result.error,
                error_type=result.error_type,
                benchmark_id=result.benchmark_id,
                benchmark_version=result.benchmark_version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                seed=config.seed,
                fingerprint=result.fingerprint,
            )
        return result

    except asyncio.TimeoutError:
        return TrialResult(
            trial_id=trial_id,
            scenario_id=scenario.scenario_id,
            status=TrialStatus.TIMEOUT,
            started_at=started,
            ended_at=time.time(),
            duration_seconds=time.time() - started,
            attempt=attempt,
            error=f"trial timed out after {timeout:.1f}s",
            error_type="TimeoutError",
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            seed=config.seed,
        )
    except Exception as exc:
        tb = traceback.format_exception_only(type(exc), exc)
        return TrialResult(
            trial_id=trial_id,
            scenario_id=scenario.scenario_id,
            status=TrialStatus.ERROR,
            started_at=started,
            ended_at=time.time(),
            duration_seconds=time.time() - started,
            attempt=attempt,
            error="".join(tb).strip(),
            error_type=type(exc).__name__,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            seed=config.seed,
        )


async def _run_with_retry(
    adapter: BenchmarkAdapter,
    scenario: Scenario,
    config: RunConfig,
) -> TrialResult:
    """Run a trial with retries.  Returns the last attempt's result."""
    last_result: TrialResult | None = None
    for attempt in range(config.retry_count + 1):
        result = await _execute_trial(adapter, scenario, config, attempt)
        last_result = result
        if result.status.is_success:
            return result
        if result.status is TrialStatus.TIMEOUT and attempt < config.retry_count:
            logger.info(
                "retrying %s (attempt %d/%d)",
                scenario.scenario_id, attempt + 1, config.retry_count + 1,
            )
    assert last_result is not None  # at least one attempt always runs
    return last_result


@dataclass
class _RunContext:
    """Shared mutable context for one benchmark run."""

    adapter: BenchmarkAdapter
    config: RunConfig
    fingerprint: EnvironmentFingerprint
    benchmark_id: str
    benchmark_version: str
    completed: frozenset[str]
    sem: asyncio.Semaphore
    progress: RunProgress
    on_trial: TrialCallback | None


async def _run_one_scenario(ctx: _RunContext, scenario: Scenario) -> TrialResult:
    """Execute one scenario respecting resume, semaphore, and fail-fast."""
    trial_id = ctx.config.trial_id(ctx.benchmark_id, scenario.scenario_id)
    if trial_id in ctx.completed:
        ctx.progress.skipped += 1
        ctx.progress.completed += 1
        result = TrialResult(
            trial_id=trial_id, scenario_id=scenario.scenario_id,
            status=TrialStatus.SKIPPED, benchmark_id=ctx.benchmark_id,
            benchmark_version=ctx.benchmark_version,
            adapter_id=ctx.adapter.adapter_id,
            adapter_version=ctx.adapter.adapter_version,
            seed=ctx.config.seed, fingerprint=ctx.fingerprint,
        )
        if ctx.on_trial:
            ctx.on_trial(result)
        return result

    async with ctx.sem:
        if ctx.progress.aborted:
            return TrialResult(
                trial_id=trial_id, scenario_id=scenario.scenario_id,
                status=TrialStatus.SKIPPED, benchmark_id=ctx.benchmark_id,
                benchmark_version=ctx.benchmark_version,
                adapter_id=ctx.adapter.adapter_id,
                adapter_version=ctx.adapter.adapter_version,
                seed=ctx.config.seed, fingerprint=ctx.fingerprint,
                error="run aborted (fail-fast)",
            )
        result = await _run_with_retry(ctx.adapter, scenario, ctx.config)
        result = replace(
            result, trial_id=trial_id, benchmark_id=ctx.benchmark_id,
            benchmark_version=ctx.benchmark_version,
            adapter_id=ctx.adapter.adapter_id,
            adapter_version=ctx.adapter.adapter_version,
            fingerprint=ctx.fingerprint,
        )

    ctx.progress.completed += 1
    if result.status.is_success:
        ctx.progress.passed += 1
    elif result.status.counts_as_failure:
        ctx.progress.failed += 1
        if ctx.config.fail_fast:
            ctx.progress.aborted = True

    if ctx.on_trial:
        ctx.on_trial(result)
    return result


async def run_benchmark(
    adapter: BenchmarkAdapter,
    scenarios: Sequence[Scenario],
    config: RunConfig,
    *,
    completed_ids: frozenset[str] | None = None,
    on_trial: TrialCallback | None = None,
    benchmark_id: str = "",
    benchmark_version: str = "",
    availability: AvailabilityResult | None = None,
) -> BenchmarkResult:
    """Run a benchmark: execute scenarios with bounded parallelism.

    Parameters
    ----------
    adapter:
        The benchmark adapter to run against.
    scenarios:
        Ordered list of scenarios to execute.
    config:
        Run configuration (seed, timeout, parallelism, etc.).
    completed_ids:
        Set of trial_ids already completed (for resume).  Matching trials
        are skipped with ``TrialStatus.SKIPPED``.
    on_trial:
        Optional callback invoked after each trial completes.
    """
    ctx = _RunContext(
        adapter=adapter, config=config,
        fingerprint=EnvironmentFingerprint.capture(),
        benchmark_id=benchmark_id or adapter.adapter_id,
        benchmark_version=benchmark_version,
        completed=completed_ids or frozenset(),
        sem=asyncio.Semaphore(max(1, config.max_parallel)),
        progress=RunProgress(total=len(scenarios)),
        on_trial=on_trial,
    )
    run_started = time.time()
    results: list[TrialResult] = []

    if config.max_parallel <= 1:
        for scenario in scenarios:
            result = await _run_one_scenario(ctx, scenario)
            results.append(result)
            if ctx.progress.aborted:
                break
    else:
        tasks = [asyncio.create_task(_run_one_scenario(ctx, s)) for s in scenarios]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

    run_ended = time.time()
    return BenchmarkResult(
        benchmark_id=ctx.benchmark_id, version=benchmark_version,
        adapter_id=adapter.adapter_id, adapter_version=adapter.adapter_version,
        seed=config.seed, started_at=run_started, ended_at=run_ended,
        duration_seconds=run_ended - run_started, fingerprint=ctx.fingerprint,
        trials=tuple(results), aggregate_metrics=compute_standard_metrics(results),
        config=config, availability=availability,
    )


def run_benchmark_sync(
    adapter: BenchmarkAdapter,
    scenarios: Sequence[Scenario],
    config: RunConfig,
    *,
    completed_ids: frozenset[str] | None = None,
    on_trial: TrialCallback | None = None,
    benchmark_id: str = "",
    benchmark_version: str = "",
    availability: AvailabilityResult | None = None,
) -> BenchmarkResult:
    """Synchronous wrapper around ``run_benchmark`` for CLI usage."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                run_benchmark(
                    adapter, scenarios, config,
                    completed_ids=completed_ids, on_trial=on_trial,
                    benchmark_id=benchmark_id, benchmark_version=benchmark_version,
                    availability=availability,
                ),
            )
            return future.result()
    else:
        return asyncio.run(
            run_benchmark(
                adapter, scenarios, config,
                completed_ids=completed_ids, on_trial=on_trial,
                benchmark_id=benchmark_id, benchmark_version=benchmark_version,
                availability=availability,
            ),
        )


class BenchmarkRunner:
    """Manifest-aware benchmark orchestrator.

    It resolves adapters through the registry, checks availability without
    installing anything, lists scenarios, and delegates bounded execution to
    ``run_benchmark``.  Adapter/availability failures become typed results.
    """

    def __init__(self, registry: AdapterRegistry | None = None) -> None:
        self.registry = registry or AdapterRegistry()

    async def run(
        self,
        manifest: BenchmarkManifest,
        config: RunConfig | None = None,
        *,
        completed_ids: frozenset[str] | None = None,
        on_trial: TrialCallback | None = None,
    ) -> BenchmarkResult:
        """Execute one manifest with only its explicitly granted authority."""
        from benchmarks.runtime import BenchmarkRuntimeContext, runtime_context

        run_config = config or RunConfig(seed=manifest.seeds[0] if manifest.seeds else 42)
        authority = BenchmarkRuntimeContext(
            live_llm_enabled=run_config.live_llm_enabled,
            require_live_llm=run_config.require_live_llm,
            hardware_enabled=run_config.hardware_enabled,
        )
        with runtime_context(authority):
            return await self._run_in_context(
                manifest,
                run_config,
                completed_ids=completed_ids,
                on_trial=on_trial,
            )

    async def _run_in_context(
        self,
        manifest: BenchmarkManifest,
        run_config: RunConfig,
        *,
        completed_ids: frozenset[str] | None,
        on_trial: TrialCallback | None,
    ) -> BenchmarkResult:
        """Execute a manifest after its run-scoped authority has been installed."""
        adapter = self.registry.get(manifest.adapter)
        if adapter is None:
            availability = AvailabilityResult(
                adapter_id=manifest.adapter,
                available=False,
                reason="adapter is not registered",
            )
            return self._unavailable_result(manifest, run_config, availability)

        try:
            availability_timeout = float(manifest.timeouts.get("availability", 30.0))
            availability = await asyncio.wait_for(
                adapter.availability(), timeout=availability_timeout,
            )
        except asyncio.TimeoutError:
            availability = AvailabilityResult(
                adapter_id=manifest.adapter,
                available=False,
                reason="availability check timed out",
            )
        except Exception as exc:
            availability = AvailabilityResult(
                adapter_id=manifest.adapter,
                available=False,
                reason=f"availability check failed: {type(exc).__name__}: {exc}",
            )

        if not availability.available:
            return self._unavailable_result(manifest, run_config, availability)

        try:
            scenarios = manifest.scenarios or await adapter.list_scenarios(tags=run_config.tags)
        except Exception as exc:
            now = time.time()
            trial = TrialResult(
                trial_id=run_config.trial_id(manifest.id, "scenario_discovery"),
                scenario_id="scenario_discovery",
                status=TrialStatus.ERROR,
                benchmark_id=manifest.id,
                benchmark_version=manifest.version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                seed=run_config.seed,
                started_at=now,
                ended_at=now,
                error=f"scenario discovery failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            )
            return BenchmarkResult(
                benchmark_id=manifest.id,
                version=manifest.version,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                seed=run_config.seed,
                started_at=now,
                ended_at=now,
                trials=(trial,),
                config=run_config,
                availability=availability,
            )

        return await run_benchmark(
            adapter,
            scenarios,
            run_config,
            completed_ids=completed_ids,
            on_trial=on_trial,
            benchmark_id=manifest.id,
            benchmark_version=manifest.version,
            availability=availability,
        )

    @staticmethod
    def _unavailable_result(
        manifest: BenchmarkManifest,
        config: RunConfig,
        availability: AvailabilityResult,
    ) -> BenchmarkResult:
        """Build a structured unavailable result without invoking an adapter."""
        now = time.time()
        scenario_ids = tuple(s.scenario_id for s in manifest.scenarios) or ("availability",)
        fingerprint = EnvironmentFingerprint.capture()
        trials = tuple(
            TrialResult(
                trial_id=config.trial_id(manifest.id, sid),
                scenario_id=sid,
                status=TrialStatus.UNAVAILABLE,
                benchmark_id=manifest.id,
                benchmark_version=manifest.version,
                adapter_id=manifest.adapter,
                seed=config.seed,
                started_at=now,
                ended_at=now,
                error=availability.reason,
                fingerprint=fingerprint,
            )
            for sid in scenario_ids
        )
        return BenchmarkResult(
            benchmark_id=manifest.id,
            version=manifest.version,
            adapter_id=manifest.adapter,
            seed=config.seed,
            started_at=now,
            ended_at=now,
            fingerprint=fingerprint,
            trials=trials,
            config=config,
            availability=availability,
        )


__all__ = [
    "BenchmarkRunner",
    "RunProgress",
    "TrialCallback",
    "run_benchmark",
    "run_benchmark_sync",
]

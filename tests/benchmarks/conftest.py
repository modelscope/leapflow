# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared fixtures for the benchmarks test suite."""

from __future__ import annotations

import pytest

from benchmarks.models import (
    BenchmarkManifest,
    MetricValue,
    RunConfig,
    Scenario,
    TrialResult,
    TrialStatus,
)


@pytest.fixture()
def run_config() -> RunConfig:
    return RunConfig(seed=42, timeout_seconds=10.0, max_parallel=1)


@pytest.fixture()
def sample_scenario() -> Scenario:
    return Scenario(
        scenario_id="test_scenario",
        name="Test Scenario",
        adapter_id="test_adapter",
        tags=("unit",),
    )


@pytest.fixture()
def passed_trial() -> TrialResult:
    return TrialResult(
        trial_id="t1",
        scenario_id="s1",
        status=TrialStatus.PASSED,
        duration_seconds=1.0,
        metrics=(MetricValue(name="hir", value=1.0),),
    )


@pytest.fixture()
def failed_trial() -> TrialResult:
    return TrialResult(
        trial_id="t2",
        scenario_id="s2",
        status=TrialStatus.FAILED,
        duration_seconds=2.0,
        error="assertion failed",
    )


@pytest.fixture()
def unavailable_trial() -> TrialResult:
    return TrialResult(
        trial_id="t3",
        scenario_id="s3",
        status=TrialStatus.UNAVAILABLE,
        error="dependency missing",
    )


@pytest.fixture()
def sample_manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id="test_bench",
        version="1.0.0",
        adapter="test_adapter",
        tier=0,
        tags=("native", "safety"),
        required=True,
        scenarios=(
            Scenario(scenario_id="s1", name="S1"),
            Scenario(scenario_id="s2", name="S2"),
        ),
    )

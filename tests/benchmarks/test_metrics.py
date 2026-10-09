# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deterministic benchmark metric boundary and aggregation tests."""

from __future__ import annotations

import pytest

from benchmarks.metrics import (
    _percentile,
    compute_standard_metrics,
    csr,
    hir,
    jitter_p50,
    jitter_p95,
    jitter_p99,
    overrun_rate,
    sim_to_real_gap,
    su_hm,
    task_success_rate,
    tracking_error,
)
from benchmarks.models import MetricValue, TrialResult, TrialStatus


def _trial(
    status: TrialStatus,
    *,
    duration: float = 0.0,
    metrics: tuple[MetricValue, ...] = (),
) -> TrialResult:
    return TrialResult(
        trial_id="trial",
        scenario_id="scenario",
        status=status,
        duration_seconds=duration,
        metrics=metrics,
    )


def test_empty_inputs_have_documented_neutral_values() -> None:
    assert task_success_rate(()) == 0.0
    assert hir(0, 0) == 1.0
    assert csr(0, 0) == 1.0
    assert su_hm(0, 0) == 1.0
    assert jitter_p50(()) == 0.0
    assert jitter_p95((1.0,)) == 0.0
    assert jitter_p99(()) == 0.0
    assert overrun_rate((), ()) == 0.0
    assert tracking_error((), ()) == 0.0
    assert sim_to_real_gap((), ()) == 0.0
    assert compute_standard_metrics(()) == ()


def test_rate_boundaries_and_su_hm() -> None:
    assert hir(0, 10) == 1.0
    assert hir(10, 10) == 0.0
    assert csr(99, 100) == 0.99
    assert csr(100, 100) == 1.0
    assert su_hm(0, 4) == 0.0
    assert su_hm(3, 4) == 0.75
    assert su_hm(4, 4) == 1.0


def test_task_success_excludes_unavailable_and_skipped() -> None:
    trials = (
        _trial(TrialStatus.PASSED),
        _trial(TrialStatus.FAILED),
        _trial(TrialStatus.UNAVAILABLE),
        _trial(TrialStatus.SKIPPED),
    )

    assert task_success_rate(trials) == 0.5


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [(0, 0.0), (25, 7.5), (50, 15.0), (95, 28.5), (100, 30.0)],
)
def test_percentile_uses_linear_interpolation(percentile: float, expected: float) -> None:
    assert _percentile((30.0, 0.0, 20.0, 10.0), percentile) == pytest.approx(expected)


def test_public_jitter_percentiles_use_adjacent_differences() -> None:
    durations = (1.0, 2.0, 5.0, 9.0, 14.0)

    assert jitter_p50(durations) == 3.5
    assert jitter_p95(durations) == pytest.approx(4.85)
    assert jitter_p99(durations) == pytest.approx(4.97)


def test_sim_to_real_gap_and_mismatched_inputs() -> None:
    assert sim_to_real_gap((0.9, 0.5, 0.1), (0.8, 0.7, 0.0)) == pytest.approx(0.4 / 3)
    assert sim_to_real_gap((1.0,), (1.0, 2.0)) == 0.0


def test_standard_metrics_aggregate_costs_and_duration_jitter() -> None:
    trials = (
        _trial(
            TrialStatus.PASSED,
            duration=1.0,
            metrics=(MetricValue("tokens", 10), MetricValue("cost", 0.2)),
        ),
        _trial(
            TrialStatus.FAILED,
            duration=3.0,
            metrics=(MetricValue("tokens", 5), MetricValue("gpu_time", 2.0)),
        ),
    )

    metrics = {metric.name: metric for metric in compute_standard_metrics(trials)}

    assert metrics["task_success_rate"].value == 0.5
    assert metrics["total_tokens"].value == 15
    assert metrics["total_cost"].value == 0.2
    assert metrics["total_gpu_time"].value == 2.0
    assert metrics["jitter_p99"].value == 2.0
    assert not metrics["jitter_p99"].higher_is_better


def test_unpriced_provider_usage_never_reports_zero_cost() -> None:
    trials = (
        _trial(
            TrialStatus.PASSED,
            metrics=(
                MetricValue("tokens", 12),
                MetricValue("prompt_tokens", 8),
                MetricValue("completion_tokens", 4),
            ),
        ),
    )

    metrics = {metric.name: metric for metric in compute_standard_metrics(trials)}

    assert metrics["total_tokens"].value == 12
    assert "total_cost" not in metrics

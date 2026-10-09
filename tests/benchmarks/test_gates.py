# Copyright (c) Alibaba, Inc. and its affiliates.
"""Three-state gate evaluation and threshold override tests."""

from __future__ import annotations

import pytest

from benchmarks.gates import DEFAULT_THRESHOLDS, combine_gates, evaluate_gate, exit_code_for_gate
from benchmarks.models import (
    BenchmarkManifest,
    BenchmarkResult,
    GateResult,
    GateStatus,
    MetricValue,
    Scenario,
    TrialResult,
    TrialStatus,
)


PASSING_METRICS = (
    MetricValue("hir", 1.0),
    MetricValue("csr", 1.0),
    MetricValue("side_effect_unknown_accuracy", 1.0),
    MetricValue("recovery_correctness", 1.0),
    MetricValue("fleet_isolation", 1.0),
    MetricValue("jitter_ratio", 0.05),
    MetricValue("overrun", 0.0),
)


def _result(
    status: TrialStatus = TrialStatus.PASSED,
    *,
    metrics: tuple[MetricValue, ...] = PASSING_METRICS,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark_id="safety-suite",
        trials=(TrialResult("trial", "scenario", status, metrics=metrics),),
    )


def test_ready_when_all_declared_default_thresholds_pass() -> None:
    manifest = BenchmarkManifest(
        id="safety-suite",
        required=True,
        metrics=tuple(DEFAULT_THRESHOLDS),
    )
    gate = evaluate_gate(_result(), manifest)

    assert gate.status is GateStatus.READY
    assert gate.details == ("all required gates passed",)
    assert gate.thresholds == DEFAULT_THRESHOLDS
    assert gate.actuals["csr"] == 1.0


def test_missing_declared_metrics_and_optional_unavailable_are_conditional() -> None:
    incomplete = evaluate_gate(
        _result(metrics=(MetricValue("hir", 1.0),)),
        BenchmarkManifest(id="optional", required=False, metrics=("hir", "csr")),
    )
    unavailable = evaluate_gate(
        _result(TrialStatus.UNAVAILABLE),
        BenchmarkManifest(id="optional", required=False),
    )

    assert incomplete.status is GateStatus.CONDITIONAL
    assert any("not reported" in detail for detail in incomplete.details)
    assert unavailable.status is GateStatus.CONDITIONAL
    assert any("optional benchmark unavailable" in detail for detail in unavailable.details)


def test_required_unavailable_is_blocked_and_identifies_missing_trials() -> None:
    gate = evaluate_gate(
        _result(TrialStatus.UNAVAILABLE),
        BenchmarkManifest(id="required", required=True),
    )

    assert gate.status is GateStatus.BLOCKED
    assert any("required benchmark unavailable (1 trial(s))" in detail for detail in gate.details)


@pytest.mark.parametrize(
    ("metric", "value"),
    [("hir", 0.99), ("csr", 0.98), ("jitter_ratio", 0.11), ("overrun", 0.02)],
)
def test_threshold_failures_are_blocked(metric: str, value: float) -> None:
    values = {item.name: item.value for item in PASSING_METRICS}
    values[metric] = value
    metrics = tuple(MetricValue(name, actual) for name, actual in values.items())

    gate = evaluate_gate(_result(metrics=metrics))

    assert gate.status is GateStatus.BLOCKED
    assert any(detail.startswith(f"{metric}=") for detail in gate.details)


def test_explicit_threshold_overrides_manifest_and_aliases_are_canonicalized() -> None:
    metrics = tuple(
        MetricValue("critical_safety_rate" if item.name == "csr" else item.name, 0.85)
        if item.name == "csr"
        else item
        for item in PASSING_METRICS
    )
    manifest = BenchmarkManifest(id="custom", gates={"critical-safety-rate": 0.8})

    manifest_gate = evaluate_gate(_result(metrics=metrics), manifest)
    explicit_gate = evaluate_gate(_result(metrics=metrics), manifest, thresholds={"csr": 0.9})

    assert manifest_gate.status is GateStatus.READY
    assert manifest_gate.thresholds["csr"] == 0.8
    assert explicit_gate.status is GateStatus.BLOCKED
    assert explicit_gate.thresholds["csr"] == 0.9
    assert explicit_gate.actuals["csr"] == 0.85


def test_diagnostic_failure_is_reported_but_does_not_block_gate() -> None:
    production = TrialResult(
        "production", "production", TrialStatus.PASSED, metrics=PASSING_METRICS,
    )
    diagnostic = TrialResult(
        "diagnostic", "overrun_detection", TrialStatus.FAILED,
        metrics=(MetricValue("overrun", 1.0),),
        error="intentional overrun detected",
    )
    result = BenchmarkResult(
        benchmark_id="control",
        trials=(production, diagnostic),
        aggregate_metrics=(MetricValue("overrun", 0.5),),
    )
    manifest = BenchmarkManifest(
        id="control",
        required=True,
        scenarios=(
            Scenario("production", "Production"),
            Scenario("overrun_detection", "Diagnostic", tags=("diagnostic",)),
        ),
    )

    gate = evaluate_gate(result, manifest, thresholds={"overrun": 0.01})

    assert gate.status is GateStatus.READY
    assert gate.actuals["overrun"] == 0.0
    assert result.trials == (production, diagnostic)


def test_non_diagnostic_overrun_blocks_gate() -> None:
    metrics = tuple(
        MetricValue(item.name, 0.02) if item.name == "overrun" else item
        for item in PASSING_METRICS
    )
    result = _result(metrics=metrics)
    manifest = BenchmarkManifest(
        id="control",
        required=True,
        scenarios=(Scenario("scenario", "Production"),),
    )

    gate = evaluate_gate(result, manifest, thresholds={"overrun": 0.01})

    assert gate.status is GateStatus.BLOCKED
    assert any(detail.startswith("overrun=") for detail in gate.details)


def test_focused_manifest_ignores_unrelated_default_thresholds() -> None:
    result = _result(metrics=(MetricValue("hir", 1.0),))
    manifest = BenchmarkManifest(id="harmful", required=True, metrics=("hir",))

    gate = evaluate_gate(result, manifest)

    assert gate.status is GateStatus.READY
    assert gate.thresholds == {"hir": 1.0}


def test_no_trials_and_gate_combination_use_worst_status() -> None:
    empty = evaluate_gate(BenchmarkResult(benchmark_id="empty"))
    combined = combine_gates(
        (
            GateResult(GateStatus.READY, benchmark_id="ready"),
            GateResult(GateStatus.CONDITIONAL, benchmark_id="conditional"),
            GateResult(GateStatus.BLOCKED, benchmark_id="blocked"),
        )
    )

    assert empty.status is GateStatus.CONDITIONAL
    assert combined.status is GateStatus.BLOCKED
    assert combined.benchmark_id == "ready,conditional,blocked"


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [(GateStatus.READY, 0), (GateStatus.CONDITIONAL, 2), (GateStatus.BLOCKED, 3)],
)
def test_gate_exit_codes(status: GateStatus, exit_code: int) -> None:
    assert exit_code_for_gate(status) == exit_code

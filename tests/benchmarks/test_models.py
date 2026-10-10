# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for frozen domain models, stable trial IDs, and environment fingerprints."""

from __future__ import annotations

import dataclasses

from benchmarks.models import (
    AvailabilityResult,
    BenchmarkManifest,
    BenchmarkResult,
    EnvironmentFingerprint,
    EvidenceRef,
    GateResult,
    GateStatus,
    ManifestError,
    MetricValue,
    RunConfig,
    Scenario,
    TrialResult,
    TrialStatus,
)


# ── Frozen invariant ──────────────────────────────────────────────────


def test_all_models_are_frozen():
    """Every domain model must be a frozen dataclass."""
    frozen_types = (
        RunConfig,
        MetricValue,
        EvidenceRef,
        Scenario,
        TrialResult,
        EnvironmentFingerprint,
        AvailabilityResult,
        BenchmarkResult,
        GateResult,
        ManifestError,
        BenchmarkManifest,
    )
    for cls in frozen_types:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        fields_obj = dataclasses.fields(cls)
        assert fields_obj, f"{cls.__name__} has no fields"


def test_frozen_instances_are_immutable():
    cfg = RunConfig(seed=42)
    try:
        cfg.seed = 99  # type: ignore[misc]
        assert False, "should have raised"
    except (dataclasses.FrozenInstanceError, AttributeError):
        pass


# ── Stable trial IDs ─────────────────────────────────────────────────


def test_trial_id_stability():
    cfg = RunConfig(seed=42)
    id1 = cfg.trial_id("bench_a", "scenario_x")
    id2 = cfg.trial_id("bench_a", "scenario_x")
    assert id1 == id2
    assert len(id1) == 16  # sha256[:16]


def test_trial_id_ignores_attempt():
    cfg = RunConfig(seed=42)
    assert cfg.trial_id("b", "s", attempt=0) == cfg.trial_id("b", "s", attempt=5)


def test_trial_id_changes_with_seed():
    cfg1 = RunConfig(seed=42)
    cfg2 = RunConfig(seed=99)
    assert cfg1.trial_id("b", "s") != cfg2.trial_id("b", "s")


# ── EnvironmentFingerprint ───────────────────────────────────────────


def test_fingerprint_capture_never_raises():
    fp = EnvironmentFingerprint.capture()
    assert fp.python_version
    assert fp.os_name


def test_fingerprint_digest_deterministic():
    fp = EnvironmentFingerprint(hostname="h", os_name="linux", python_version="3.12")
    assert fp.digest == fp.digest
    assert len(fp.digest) == 64


def test_fingerprint_roundtrip():
    fp = EnvironmentFingerprint.capture()
    d = fp.to_dict()
    fp2 = EnvironmentFingerprint.from_dict(d)
    assert fp2.hostname == fp.hostname
    assert fp2.python_version == fp.python_version


# ── MetricValue ──────────────────────────────────────────────────────


def test_metric_meets_threshold_higher_is_better():
    m = MetricValue(name="accuracy", value=0.95, threshold=0.90)
    assert m.meets_threshold is True


def test_metric_fails_threshold():
    m = MetricValue(name="accuracy", value=0.80, threshold=0.90)
    assert m.meets_threshold is False


def test_metric_lower_is_better():
    m = MetricValue(name="latency", value=0.05, higher_is_better=False, threshold=0.10)
    assert m.meets_threshold is True


def test_metric_no_threshold():
    m = MetricValue(name="x", value=1.0)
    assert m.meets_threshold is None


def test_metric_roundtrip():
    m = MetricValue(name="tsr", value=0.9, unit="ratio", threshold=0.8, tags=("safety",))
    d = m.to_dict()
    m2 = MetricValue.from_dict(d)
    assert m2.name == m.name and m2.value == m.value and m2.threshold == m.threshold


# ── TrialStatus ──────────────────────────────────────────────────────


def test_trial_status_success():
    assert TrialStatus.PASSED.is_success
    assert not TrialStatus.FAILED.is_success


def test_trial_status_counts_as_failure():
    for s in (TrialStatus.FAILED, TrialStatus.ERROR, TrialStatus.TIMEOUT):
        assert s.counts_as_failure
    assert not TrialStatus.PASSED.counts_as_failure
    assert not TrialStatus.SKIPPED.counts_as_failure
    assert not TrialStatus.UNAVAILABLE.counts_as_failure


def test_trial_status_terminal():
    for s in (
        TrialStatus.PASSED,
        TrialStatus.FAILED,
        TrialStatus.ERROR,
        TrialStatus.TIMEOUT,
        TrialStatus.UNAVAILABLE,
    ):
        assert s.is_terminal
    assert not TrialStatus.SKIPPED.is_terminal


# ── TrialResult roundtrip ────────────────────────────────────────────


def test_trial_result_roundtrip():
    t = TrialResult(
        trial_id="abc",
        scenario_id="s1",
        status=TrialStatus.PASSED,
        metrics=(MetricValue(name="x", value=1.0),),
        seed=42,
        fingerprint=EnvironmentFingerprint(hostname="h"),
    )
    d = t.to_dict()
    t2 = TrialResult.from_dict(d)
    assert t2.trial_id == t.trial_id
    assert t2.status is TrialStatus.PASSED
    assert t2.fingerprint is not None and t2.fingerprint.hostname == "h"


# ── BenchmarkResult ──────────────────────────────────────────────────


def test_benchmark_result_pass_rate():
    trials = (
        TrialResult(trial_id="1", scenario_id="s", status=TrialStatus.PASSED),
        TrialResult(trial_id="2", scenario_id="s", status=TrialStatus.FAILED),
        TrialResult(trial_id="3", scenario_id="s", status=TrialStatus.SKIPPED),
        TrialResult(trial_id="4", scenario_id="s", status=TrialStatus.UNAVAILABLE),
    )
    r = BenchmarkResult(benchmark_id="b", trials=trials)
    assert r.total_trials == 4
    assert r.passed == 1
    assert r.failed == 1
    assert r.unavailable == 1
    assert r.pass_rate == 0.5  # 1 passed / 2 executed


def test_benchmark_result_roundtrip():
    r = BenchmarkResult(benchmark_id="b", version="1.0", seed=42)
    d = r.to_dict()
    r2 = BenchmarkResult.from_dict(d)
    assert r2.benchmark_id == "b" and r2.seed == 42


# ── RunConfig roundtrip ──────────────────────────────────────────────


def test_run_config_roundtrip():
    cfg = RunConfig(
        seed=99,
        timeout_seconds=60.0,
        tags=("a", "b"),
        run_id="run-test",
        evidence_root="/evidence",
    )
    d = cfg.to_dict()
    cfg2 = RunConfig.from_dict(d)
    assert cfg2.seed == 99 and cfg2.tags == ("a", "b")
    assert cfg2.run_id == "run-test"
    assert cfg2.evidence_root == "/evidence"


# ── GateStatus / GateResult ──────────────────────────────────────────


def test_gate_status_values():
    assert GateStatus.READY.value == "ready"
    assert GateStatus.CONDITIONAL.value == "conditional"
    assert GateStatus.BLOCKED.value == "blocked"


def test_gate_result_to_dict():
    g = GateResult(status=GateStatus.READY, benchmark_id="b", details=("ok",))
    d = g.to_dict()
    assert d["status"] == "ready"


# ── ManifestError ─────────────────────────────────────────────────────


def test_manifest_error_str():
    e = ManifestError(field="id", message="missing", path="test.yaml")
    assert "id" in str(e) and "missing" in str(e) and "test.yaml" in str(e)

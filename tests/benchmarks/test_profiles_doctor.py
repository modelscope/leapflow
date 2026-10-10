# Copyright (c) Alibaba, Inc. and its affiliates.
"""Profile ordering and read-only doctor diagnostic tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.doctor import (
    check_benchmark_configuration,
    check_data_directory,
    check_environment_variable,
    check_executable,
    check_gpu,
    check_hardware_device,
    check_license_acknowledgement,
    check_python_package,
    run_doctor,
)
from benchmarks.models import BenchmarkManifest
from benchmarks.profiles import PROFILE_IDS, get_profile, list_profiles, select_manifests


def test_profile_ids_are_stable_and_registered() -> None:
    assert PROFILE_IDS == (
        "tier0", "tier1", "tier2", "tier3", "tier4", "live-llm", "production-sim", "pre-hardware", "all",
    )
    assert [profile.profile_id for profile in list_profiles()] == list(PROFILE_IDS)


def test_pre_hardware_covers_tiers_zero_through_three() -> None:
    pre_hardware = get_profile("pre-hardware")

    assert pre_hardware is not None
    assert pre_hardware.tiers == (0, 1, 2, 3)
    assert get_profile("tier-0") is get_profile("tier0")
    assert get_profile("missing") is None


def test_select_manifests_filters_by_tier_and_sorts_output() -> None:
    manifests = (
        BenchmarkManifest(id="zeta", tier=1),
        BenchmarkManifest(id="alpha", tier=0),
        BenchmarkManifest(id="hardware", tier=4),
        BenchmarkManifest(id="pre", tier=3),
        BenchmarkManifest(id="live", tier=5),
        BenchmarkManifest(id="production", tier=1, tags=("production-sim",)),
    )

    tier0 = [manifest.id for manifest in select_manifests(manifests, "tier0")]
    pre_hw = [manifest.id for manifest in select_manifests(manifests, "pre-hardware")]
    unknown = select_manifests(manifests, "no-such-profile")

    assert tier0 == ["alpha"]
    assert pre_hw == ["alpha", "pre", "production", "zeta"]
    assert "hardware" not in pre_hw
    assert [manifest.id for manifest in select_manifests(manifests, "live-llm")] == ["live"]
    assert [manifest.id for manifest in select_manifests(manifests, "production-sim")] == ["production"]
    assert unknown == ()


def test_doctor_only_reads_the_environment(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "existing"
    data_dir.mkdir()
    missing_dir = tmp_path / "missing"
    monkeypatch.setenv("BENCH_DOCTOR_FLAG", "1")
    monkeypatch.delenv("BENCH_DOCTOR_MISSING", raising=False)

    env_present = check_environment_variable("BENCH_DOCTOR_FLAG")
    env_missing = check_environment_variable("BENCH_DOCTOR_MISSING")
    data_present = check_data_directory(data_dir)
    data_missing = check_data_directory(missing_dir)

    assert env_present.available and env_present.status == "pass"
    assert env_present.detail == "set"
    assert not env_missing.available
    assert data_present.available
    assert not data_missing.available
    assert not missing_dir.exists(), "doctor must never create directories"


def test_typed_benchmark_configuration_is_checked(monkeypatch) -> None:
    import leapflow.config

    monkeypatch.setattr(
        leapflow.config,
        "load_config",
        lambda: SimpleNamespace(
            benchmark_commands='{"calvin": "python"}',
            benchmark_data_roots='{"calvin": "/definitely/missing"}',
            benchmark_license_acceptances="{}",
            benchmark_hardware_profiles="{}",
        ),
    )

    command = check_benchmark_configuration("benchmark.commands.calvin")
    data_root = check_benchmark_configuration("benchmark.data_roots.calvin")

    assert command.available
    assert not data_root.available
    assert "dataset root" in data_root.detail


def test_license_acknowledgement_defaults_to_blocked(monkeypatch) -> None:
    monkeypatch.delenv("BENCHMARK_LICENSE_SAMPLE_ACCEPTED", raising=False)

    unknown = check_license_acknowledgement("sample")
    assert not unknown.available
    assert "BENCHMARK_LICENSE_SAMPLE_ACCEPTED" in unknown.detail

    monkeypatch.setenv("BENCHMARK_LICENSE_SAMPLE_ACCEPTED", "yes")
    accepted = check_license_acknowledgement("sample")
    assert accepted.available
    assert accepted.detail == "acknowledged"


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (check_python_package, ("benchmarks_no_such_pkg_xyz",)),
        (check_executable, ("leapflow_no_such_binary_xyz",)),
        (check_hardware_device, (Path("/dev/null_no_such_device_xyz"),)),
    ],
)
def test_missing_resources_are_reported_without_mutation(factory, args) -> None:
    check = factory(*args)
    assert not check.available
    assert check.status == "fail"
    assert check.remediation != ""


def test_run_doctor_aggregates_statuses_and_includes_gpu(monkeypatch) -> None:
    monkeypatch.setenv("BENCH_DOCTOR_REPORT", "1")
    report = run_doctor(
        python_packages=("os", "benchmarks_absent_pkg"),
        executables=("python",),
        environment_variables=("BENCH_DOCTOR_REPORT",),
        gpu_required=False,
    )

    ids = {check.check_id for check in report.checks}
    assert "python:os" in ids
    assert "python:benchmarks_absent_pkg" in ids
    assert "gpu" in ids
    assert report.failed >= 1
    assert report.ready is False
    assert isinstance(check_gpu().available, bool)


def test_check_python_package_uses_find_spec_only() -> None:
    """find_spec must detect a stdlib package without executing it."""
    check = check_python_package("json")
    assert check.available
    assert check.detail == "installed"
    assert os.environ.get("JSON_SIDE_EFFECT") is None

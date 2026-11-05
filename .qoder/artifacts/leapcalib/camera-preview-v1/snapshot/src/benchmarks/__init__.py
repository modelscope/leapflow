# Copyright (c) Alibaba, Inc. and its affiliates.
"""Standalone, import-safe benchmark harness for LeapFlow.

The package depends only on the Python standard library and PyYAML.  Adapter
discovery is lazy; importing ``benchmarks`` never imports an external
benchmark SDK, starts a process, downloads data, or modifies the environment.
"""

from benchmarks.doctor import DoctorCheck, DoctorReport, run_doctor
from benchmarks.evidence import EvidenceStore
from benchmarks.gates import DEFAULT_THRESHOLDS, evaluate_gate
from benchmarks.manifest import load_manifest, load_manifests
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
from benchmarks.profiles import BenchmarkProfile, get_profile, list_profiles
from benchmarks.protocol import BenchmarkAdapter, BenchmarkSuite, ScenarioProvider
from benchmarks.registry import AdapterConflict, AdapterRegistry
from benchmarks.runner import BenchmarkRunner, run_benchmark, run_benchmark_sync

__all__ = [
    "AdapterConflict",
    "AdapterRegistry",
    "AvailabilityResult",
    "BenchmarkAdapter",
    "BenchmarkManifest",
    "BenchmarkProfile",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "DEFAULT_THRESHOLDS",
    "DoctorCheck",
    "DoctorReport",
    "EnvironmentFingerprint",
    "EvidenceRef",
    "EvidenceStore",
    "GateResult",
    "GateStatus",
    "ManifestError",
    "MetricValue",
    "RunConfig",
    "Scenario",
    "ScenarioProvider",
    "TrialResult",
    "TrialStatus",
    "evaluate_gate",
    "get_profile",
    "list_profiles",
    "load_manifest",
    "load_manifests",
    "run_benchmark",
    "run_benchmark_sync",
    "run_doctor",
]

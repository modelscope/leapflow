# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable domain types for the benchmark harness.

Every type in this module is a frozen dataclass.  They carry benchmark
identity, reproducibility metadata, trial outcomes, metric values, and
environment fingerprints.  No mutable state, no side effects, no imports
beyond the standard library.
"""

from __future__ import annotations

import enum
import hashlib
import os
import platform as platform_module
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Mapping


# ════════════════════════════════════════════════════════════════
# Trial status
# ════════════════════════════════════════════════════════════════


class TrialStatus(str, enum.Enum):
    """Outcome of a single trial execution.

    ``UNAVAILABLE`` means the adapter or its dependencies are not present;
    it is not a test failure, but a gate can treat a *required* benchmark
    whose adapter is unavailable as ``BLOCKED``.

    ``SKIPPED`` means the runner intentionally omitted this trial (e.g.
    resume mode found it already complete).
    """

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"

    @property
    def is_success(self) -> bool:
        return self is TrialStatus.PASSED

    @property
    def counts_as_failure(self) -> bool:
        """Return True for statuses that count against pass rate."""
        return self in (TrialStatus.FAILED, TrialStatus.ERROR, TrialStatus.TIMEOUT)

    @property
    def is_terminal(self) -> bool:
        """Return True for statuses that represent a completed trial."""
        return self in (
            TrialStatus.PASSED,
            TrialStatus.FAILED,
            TrialStatus.ERROR,
            TrialStatus.TIMEOUT,
            TrialStatus.UNAVAILABLE,
        )


# ════════════════════════════════════════════════════════════════
# Gate status
# ════════════════════════════════════════════════════════════════


class GateStatus(str, enum.Enum):
    """Three-state readiness verdict for a deployment gate."""

    READY = "ready"
    CONDITIONAL = "conditional"
    BLOCKED = "blocked"


# ════════════════════════════════════════════════════════════════
# Core value types
# ════════════════════════════════════════════════════════════════


def _new_run_id() -> str:
    """Return a path-safe identifier for one isolated benchmark invocation."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"run-{timestamp}-{os.getpid():x}-{time.time_ns() & 0xFFFFFF:x}"


@dataclass(frozen=True)
class RunConfig:
    """Configuration governing a benchmark run."""

    seed: int = 42
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_parallel: int = 1
    fail_fast: bool = False
    resume_from: str = ""
    profile: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    live_llm_enabled: bool = False
    require_live_llm: bool = False
    hardware_enabled: bool = False
    run_id: str = field(default_factory=_new_run_id)
    evidence_root: str = ""

    def trial_id(self, benchmark_id: str, scenario_id: str, attempt: int = 0) -> str:
        """Compute a stable trial id independent of retry attempt.

        ``attempt`` is accepted for API compatibility but deliberately excluded
        from the digest: resume must recognize a completed retry as the same
        logical trial.
        """
        del attempt
        raw = f"{benchmark_id}:{scenario_id}:seed={self.seed}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_parallel": self.max_parallel,
            "fail_fast": self.fail_fast,
            "resume_from": self.resume_from,
            "profile": self.profile,
            "tags": list(self.tags),
            "live_llm_enabled": self.live_llm_enabled,
            "require_live_llm": self.require_live_llm,
            "hardware_enabled": self.hardware_enabled,
            "run_id": self.run_id,
            "evidence_root": self.evidence_root,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        return cls(
            seed=int(data.get("seed", 42)),
            timeout_seconds=float(data.get("timeout_seconds", 300.0)),
            retry_count=int(data.get("retry_count", 0)),
            max_parallel=int(data.get("max_parallel", 1)),
            fail_fast=bool(data.get("fail_fast", False)),
            resume_from=str(data.get("resume_from", "")),
            profile=str(data.get("profile", "")),
            tags=tuple(str(item) for item in data.get("tags", ())),
            live_llm_enabled=bool(data.get("live_llm_enabled", False)),
            require_live_llm=bool(data.get("require_live_llm", False)),
            hardware_enabled=bool(data.get("hardware_enabled", False)),
            run_id=str(data.get("run_id") or _new_run_id()),
            evidence_root=str(data.get("evidence_root", "")),
        )


@dataclass(frozen=True)
class MetricValue:
    """One named metric observation."""

    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    threshold: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def meets_threshold(self) -> bool | None:
        """Return whether value meets threshold, or None if no threshold set."""
        if self.threshold is None:
            return None
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "value": self.value}
        if self.unit:
            d["unit"] = self.unit
        d["higher_is_better"] = self.higher_is_better
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.tags:
            d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricValue:
        return cls(
            name=str(data.get("name", "")),
            value=float(data.get("value", 0.0)),
            unit=str(data.get("unit", "")),
            higher_is_better=bool(data.get("higher_is_better", True)),
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            tags=tuple(str(t) for t in data.get("tags", ())),
        )


@dataclass(frozen=True)
class EvidenceRef:
    """Reference to a piece of evidence stored in the evidence directory."""

    path: str
    content_hash: str = ""
    size_bytes: int = 0
    media_type: str = "application/octet-stream"
    created_at: float = field(default_factory=time.time)
    kind: str = "file"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "created_at": self.created_at,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRef:
        return cls(
            path=str(data.get("path", "")),
            content_hash=str(data.get("content_hash", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            media_type=str(data.get("media_type", "application/octet-stream")),
            created_at=float(data.get("created_at", 0.0)),
            kind=str(data.get("kind", "file")),
        )


@dataclass(frozen=True)
class Scenario:
    """One benchmark scenario: the unit of execution within an adapter."""

    scenario_id: str
    name: str
    description: str = ""
    adapter_id: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    timeout_seconds: float | None = None
    required: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "name": self.name,
        }
        if self.description:
            d["description"] = self.description
        if self.adapter_id:
            d["adapter_id"] = self.adapter_id
        if self.tags:
            d["tags"] = list(self.tags)
        if self.timeout_seconds is not None:
            d["timeout_seconds"] = self.timeout_seconds
        d["required"] = self.required
        if self.parameters:
            d["parameters"] = dict(self.parameters)
        return d


@dataclass(frozen=True)
class TrialResult:
    """Outcome of executing one trial (one scenario, one seed, one attempt)."""

    trial_id: str
    scenario_id: str
    status: TrialStatus
    metrics: tuple[MetricValue, ...] = field(default_factory=tuple)
    evidence: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_seconds: float = 0.0
    attempt: int = 0
    error: str = ""
    error_type: str = ""
    benchmark_id: str = ""
    benchmark_version: str = ""
    adapter_id: str = ""
    adapter_version: str = ""
    seed: int = 0
    fingerprint: EnvironmentFingerprint | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trial_id": self.trial_id,
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "attempt": self.attempt,
            "seed": self.seed,
        }
        if self.metrics:
            d["metrics"] = [m.to_dict() for m in self.metrics]
        if self.evidence:
            d["evidence"] = [e.to_dict() for e in self.evidence]
        if self.error:
            d["error"] = self.error
        if self.error_type:
            d["error_type"] = self.error_type
        if self.benchmark_id:
            d["benchmark_id"] = self.benchmark_id
        if self.benchmark_version:
            d["benchmark_version"] = self.benchmark_version
        if self.adapter_id:
            d["adapter_id"] = self.adapter_id
        if self.adapter_version:
            d["adapter_version"] = self.adapter_version
        if self.fingerprint is not None:
            d["fingerprint"] = self.fingerprint.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrialResult:
        return cls(
            trial_id=str(data.get("trial_id", "")),
            scenario_id=str(data.get("scenario_id", "")),
            status=TrialStatus(data.get("status", "error")),
            metrics=tuple(MetricValue.from_dict(m) for m in data.get("metrics", ())),
            evidence=tuple(EvidenceRef.from_dict(e) for e in data.get("evidence", ())),
            started_at=float(data.get("started_at", 0.0)),
            ended_at=float(data.get("ended_at", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            attempt=int(data.get("attempt", 0)),
            error=str(data.get("error", "")),
            error_type=str(data.get("error_type", "")),
            benchmark_id=str(data.get("benchmark_id", "")),
            benchmark_version=str(data.get("benchmark_version", "")),
            adapter_id=str(data.get("adapter_id", "")),
            adapter_version=str(data.get("adapter_version", "")),
            seed=int(data.get("seed", 0)),
            fingerprint=(
                EnvironmentFingerprint.from_dict(data["fingerprint"])
                if isinstance(data.get("fingerprint"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    """Reproducibility metadata captured at the start of a benchmark run."""

    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    git_sha: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    cpu_count: int = 0
    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentFingerprint:
        """Snapshot the current environment.  Never raises."""
        git_sha, git_branch, git_dirty = "", "", False
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.0,
            ).strip()
            git_dirty = bool(status)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ):
            pass

        return cls(
            hostname=platform_module.node(),
            os_name=platform_module.system(),
            os_version=platform_module.platform(),
            python_version=platform_module.python_version(),
            git_sha=git_sha,
            git_branch=git_branch,
            git_dirty=git_dirty,
            cpu_count=os.cpu_count() or 0,
        )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest of reproducibility fields."""
        payload = "|".join(
            (
                self.hostname,
                self.os_name,
                self.os_version,
                self.python_version,
                self.git_sha,
                self.git_branch,
                str(self.git_dirty),
                str(self.cpu_count),
                repr(sorted(self.extra.items())),
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "hostname": self.hostname,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
        }
        if self.git_sha:
            d["git_sha"] = self.git_sha
        if self.git_branch:
            d["git_branch"] = self.git_branch
        d["git_dirty"] = self.git_dirty
        if self.extra:
            d["extra"] = dict(self.extra)
        d["digest"] = self.digest
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvironmentFingerprint:
        return cls(
            hostname=str(data.get("hostname", "")),
            os_name=str(data.get("os_name", "")),
            os_version=str(data.get("os_version", "")),
            python_version=str(data.get("python_version", "")),
            git_sha=str(data.get("git_sha", "")),
            git_branch=str(data.get("git_branch", "")),
            git_dirty=bool(data.get("git_dirty", False)),
            cpu_count=int(data.get("cpu_count", 0)),
            extra={str(k): str(v) for k, v in (data.get("extra") or {}).items()},
        )


@dataclass(frozen=True)
class AvailabilityResult:
    """Result of checking whether a benchmark adapter is usable."""

    adapter_id: str
    available: bool
    reason: str = ""
    missing_dependencies: tuple[str, ...] = field(default_factory=tuple)
    missing_executables: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "available": self.available,
        }
        if self.reason:
            d["reason"] = self.reason
        if self.missing_dependencies:
            d["missing_dependencies"] = list(self.missing_dependencies)
        if self.missing_executables:
            d["missing_executables"] = list(self.missing_executables)
        return d


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate result of one complete benchmark run."""

    benchmark_id: str
    version: str = ""
    adapter_id: str = ""
    adapter_version: str = ""
    seed: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_seconds: float = 0.0
    fingerprint: EnvironmentFingerprint = field(default_factory=EnvironmentFingerprint)
    trials: tuple[TrialResult, ...] = field(default_factory=tuple)
    aggregate_metrics: tuple[MetricValue, ...] = field(default_factory=tuple)
    config: RunConfig = field(default_factory=RunConfig)
    availability: AvailabilityResult | None = None

    @property
    def total_trials(self) -> int:
        return len(self.trials)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.trials if t.status is TrialStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.trials if t.status.counts_as_failure)

    @property
    def unavailable(self) -> int:
        return sum(1 for t in self.trials if t.status is TrialStatus.UNAVAILABLE)

    @property
    def pass_rate(self) -> float:
        executed = sum(
            1 for t in self.trials if t.status not in (TrialStatus.SKIPPED, TrialStatus.UNAVAILABLE)
        )
        if executed == 0:
            return 0.0
        return self.passed / executed

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "seed": self.seed,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "fingerprint": self.fingerprint.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "aggregate_metrics": [m.to_dict() for m in self.aggregate_metrics],
            "config": self.config.to_dict(),
            "availability": self.availability.to_dict() if self.availability else None,
            "total_trials": self.total_trials,
            "passed": self.passed,
            "failed": self.failed,
            "unavailable": self.unavailable,
            "pass_rate": self.pass_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkResult:
        availability_data = data.get("availability")
        availability = None
        if isinstance(availability_data, Mapping):
            availability = AvailabilityResult(
                adapter_id=str(availability_data.get("adapter_id", "")),
                available=bool(availability_data.get("available", False)),
                reason=str(availability_data.get("reason", "")),
                missing_dependencies=tuple(
                    str(item) for item in availability_data.get("missing_dependencies", ())
                ),
                missing_executables=tuple(
                    str(item) for item in availability_data.get("missing_executables", ())
                ),
            )
        fingerprint_data = data.get("fingerprint")
        config_data = data.get("config")
        return cls(
            benchmark_id=str(data.get("benchmark_id", "")),
            version=str(data.get("version", "")),
            adapter_id=str(data.get("adapter_id", "")),
            adapter_version=str(data.get("adapter_version", "")),
            seed=int(data.get("seed", 0)),
            started_at=float(data.get("started_at", 0.0)),
            ended_at=float(data.get("ended_at", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            fingerprint=(
                EnvironmentFingerprint.from_dict(fingerprint_data)
                if isinstance(fingerprint_data, Mapping)
                else EnvironmentFingerprint()
            ),
            trials=tuple(TrialResult.from_dict(item) for item in data.get("trials", ())),
            aggregate_metrics=tuple(
                MetricValue.from_dict(item) for item in data.get("aggregate_metrics", ())
            ),
            config=(
                RunConfig.from_dict(config_data)
                if isinstance(config_data, Mapping)
                else RunConfig()
            ),
            availability=availability,
        )


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating a deployment gate against benchmark results."""

    status: GateStatus
    benchmark_id: str = ""
    details: tuple[str, ...] = field(default_factory=tuple)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    actuals: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status.value,
            "benchmark_id": self.benchmark_id,
        }
        if self.details:
            d["details"] = list(self.details)
        if self.thresholds:
            d["thresholds"] = dict(self.thresholds)
        if self.actuals:
            d["actuals"] = dict(self.actuals)
        return d


# ════════════════════════════════════════════════════════════════
# Manifest-level types
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ManifestError:
    """One typed validation error from manifest parsing."""

    field: str
    message: str
    path: str = ""

    def __str__(self) -> str:
        loc = f" ({self.path})" if self.path else ""
        return f"{self.field}: {self.message}{loc}"


@dataclass(frozen=True)
class BenchmarkManifest:
    """Parsed, validated representation of a benchmark manifest YAML."""

    id: str
    version: str = "0.0.0"
    source: str = ""
    paper: str = ""
    repo: str = ""
    license: str = ""
    setup_guidance: str = ""
    note: str = ""
    adapter: str = ""
    tier: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)
    required: bool = False
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    timeouts: Mapping[str, float] = field(default_factory=dict)
    seeds: tuple[int, ...] = field(default_factory=lambda: (42,))
    metrics: tuple[str, ...] = field(default_factory=tuple)
    gates: Mapping[str, float] = field(default_factory=dict)
    official: Mapping[str, Any] = field(default_factory=dict)
    scenarios: tuple[Scenario, ...] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "adapter": self.adapter,
            "tier": self.tier,
            "required": self.required,
        }
        if self.source:
            d["source"] = self.source
        if self.paper:
            d["paper"] = self.paper
        if self.repo:
            d["repo"] = self.repo
        if self.license:
            d["license"] = self.license
        if self.setup_guidance:
            d["setup_guidance"] = self.setup_guidance
        if self.note:
            d["note"] = self.note
        if self.tags:
            d["tags"] = list(self.tags)
        if self.dependencies:
            d["dependencies"] = list(self.dependencies)
        if self.timeouts:
            d["timeouts"] = dict(self.timeouts)
        d["seeds"] = list(self.seeds)
        if self.metrics:
            d["metrics"] = list(self.metrics)
        if self.gates:
            d["gates"] = dict(self.gates)
        if self.official:
            d["official"] = dict(self.official)
        if self.scenarios:
            d["scenarios"] = [s.to_dict() for s in self.scenarios]
        return d


__all__ = [
    "AvailabilityResult",
    "BenchmarkManifest",
    "BenchmarkResult",
    "EnvironmentFingerprint",
    "EvidenceRef",
    "GateResult",
    "GateStatus",
    "ManifestError",
    "MetricValue",
    "RunConfig",
    "Scenario",
    "TrialResult",
    "TrialStatus",
]

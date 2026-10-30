# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared helpers for external benchmark adapters.

Provides import-safe dependency probing, safe subprocess execution, and
output normalization utilities.  No heavy dependencies; no side effects at
import time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from benchmarks.evidence import EvidenceStore, evidence_root
from benchmarks.manifest import load_manifest
from benchmarks.models import (
    AvailabilityResult,
    MetricValue,
    Scenario,
    TrialResult,
    TrialStatus,
)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# Dependency probing
# ════════════════════════════════════════════════════════════════

_MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "manifests" / "external"


def probe_module(module_name: str) -> bool:
    """Check whether a Python module is importable without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def probe_executable(name: str) -> str | None:
    """Return the resolved path of an executable, or ``None``."""
    return shutil.which(name)


def probe_env(name: str) -> bool:
    """Return whether an environment variable is set and non-empty."""
    return bool(os.environ.get(name, "").strip())


@dataclass(frozen=True)
class DependencySpec:
    """Declarative specification of one adapter dependency."""

    kind: str  # "python", "executable", "env", "gpu"
    name: str
    required: bool = True
    reason: str = ""


def check_dependencies(
    specs: Sequence[DependencySpec],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Check a list of dependency specs.

    Returns ``(all_ok, missing_deps, missing_exes)``.
    """
    missing_deps: list[str] = []
    missing_exes: list[str] = []
    all_ok = True

    for spec in specs:
        if spec.kind == "python":
            if not probe_module(spec.name):
                missing_deps.append(spec.name)
                if spec.required:
                    all_ok = False
        elif spec.kind == "executable":
            if probe_executable(spec.name) is None:
                missing_exes.append(spec.name)
                if spec.required:
                    all_ok = False
        elif spec.kind == "env":
            if not probe_env(spec.name):
                missing_deps.append(f"env:{spec.name}")
                if spec.required:
                    all_ok = False
        elif spec.kind == "gpu":
            if not _gpu_available():
                missing_deps.append("gpu")
                if spec.required:
                    all_ok = False

    return all_ok, tuple(missing_deps), tuple(missing_exes)


def _gpu_available() -> bool:
    """Quick GPU probe without importing heavy SDKs."""
    for tool in ("nvidia-smi", "rocm-smi"):
        if shutil.which(tool) is not None:
            return True
    return False


def make_availability(
    adapter_id: str,
    specs: Sequence[DependencySpec],
    *,
    extra_reason: str = "",
) -> AvailabilityResult:
    """Build an ``AvailabilityResult`` from dependency specs."""
    ok, missing_deps, missing_exes = check_dependencies(specs)
    reasons: list[str] = []
    if not ok:
        parts: list[str] = []
        if missing_deps:
            parts.append(f"missing deps: {', '.join(missing_deps)}")
        if missing_exes:
            parts.append(f"missing executables: {', '.join(missing_exes)}")
        reasons.append("; ".join(parts))
    if extra_reason:
        reasons.append(extra_reason)
    return AvailabilityResult(
        adapter_id=adapter_id,
        available=ok and not extra_reason,
        reason="; ".join(reasons) if reasons else "",
        missing_dependencies=missing_deps,
        missing_executables=missing_exes,
    )


def _benchmark_settings_mapping(field_name: str) -> Mapping[str, object]:
    """Read one JSON mapping from the active Settings control plane."""
    try:
        from leapflow.config import load_config

        decoded = json.loads(str(getattr(load_config(), field_name)))
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
        logger.debug("Benchmark Settings lookup skipped for %s", field_name, exc_info=True)
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def _configured_command(adapter_id: str, env_name: str) -> str:
    """Resolve an operator command from typed Settings, then legacy env fallback."""
    value = _benchmark_settings_mapping("benchmark_commands").get(adapter_id, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get(env_name, "").strip()


def configured_data_root(adapter_id: str) -> str:
    """Return an operator-declared local dataset root for one benchmark."""
    value = _benchmark_settings_mapping("benchmark_data_roots").get(adapter_id, "")
    return value.strip() if isinstance(value, str) else ""


def command_availability(
    adapter_id: str,
    env_name: str,
    setup_url: str,
    *,
    requires_data_root: bool = False,
) -> AvailabilityResult:
    """Validate an explicitly configured official evaluator command.

    Data-backed benchmarks stay unavailable until the operator names a local,
    licensed official dataset root.  The harness never substitutes synthetic
    examples or downloads a dataset to make a benchmark appear runnable.
    """
    template = _configured_command(adapter_id, env_name)
    guidance = (
        f"set benchmark.commands.{adapter_id} through leap config or set {env_name}; "
        f"configure it from {setup_url}"
    )
    if not template:
        return AvailabilityResult(
            adapter_id=adapter_id,
            available=False,
            reason=guidance,
            missing_dependencies=(f"config:benchmark.commands.{adapter_id}",),
        )
    data_root = configured_data_root(adapter_id)
    if requires_data_root and (not data_root or not Path(data_root).is_dir()):
        return AvailabilityResult(
            adapter_id=adapter_id,
            available=False,
            reason=(
                f"provide a readable official dataset root through "
                f"benchmark.data_roots.{adapter_id}"
            ),
            missing_dependencies=(f"config:benchmark.data_roots.{adapter_id}",),
        )
    try:
        parts = shlex.split(template)
    except ValueError as exc:
        return AvailabilityResult(
            adapter_id=adapter_id,
            available=False,
            reason=f"invalid {env_name}: {exc}; {guidance}",
            missing_dependencies=(f"env:{env_name}",),
        )
    if not parts or probe_executable(parts[0]) is None:
        executable = parts[0] if parts else "<empty>"
        return AvailabilityResult(
            adapter_id=adapter_id,
            available=False,
            reason=f"configured executable not found: {executable}; {guidance}",
            missing_executables=(executable,),
        )
    return AvailabilityResult(adapter_id=adapter_id, available=True)


def run_configured_command(
    env_name: str,
    scenario: Scenario,
    *,
    seed: int,
    timeout_seconds: float,
) -> SubprocessResult:
    """Render and execute an operator-supplied benchmark command template."""
    adapter_id = env_name.removeprefix("LEAPFLOW_BENCHMARK_").removesuffix("_COMMAND").lower()
    template = _configured_command(adapter_id, env_name)
    try:
        command = [
            item.format(
                scenario_id=scenario.scenario_id,
                seed=seed,
                timeout=timeout_seconds,
                data_root=configured_data_root(adapter_id),
            )
            for item in shlex.split(template)
        ]
    except (KeyError, ValueError) as exc:
        return SubprocessResult(returncode=-1, stderr=f"invalid {env_name}: {exc}")
    return run_subprocess(command, timeout=timeout_seconds)


# ════════════════════════════════════════════════════════════════
# Subprocess runner
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SubprocessResult:
    """Structured output from a subprocess invocation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False


def run_subprocess(
    cmd: Sequence[str],
    *,
    timeout: float = 300.0,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SubprocessResult:
    """Run a subprocess safely.  Never uses ``shell=True``.

    The command list is validated via ``shlex.join`` (display only) and
    each element is type-checked.  stderr is always captured as evidence.
    """
    if not cmd:
        return SubprocessResult(returncode=-1, stderr="empty command")

    for i, arg in enumerate(cmd):
        if not isinstance(arg, str):
            return SubprocessResult(
                returncode=-1,
                stderr=f"argument {i} is not a string: {type(arg).__name__}",
            )

    logger.debug("subprocess: %s", shlex.join(list(cmd)))
    started = time.time()
    merged_env = {**os.environ, **(env or {})} if env else None

    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            shell=False,  # explicit
        )
        return SubprocessResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=time.time() - started,
        )
    except subprocess.TimeoutExpired as exc:
        return SubprocessResult(
            returncode=-1,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "") + f"\nTIMEOUT after {timeout}s",
            duration_seconds=time.time() - started,
            timed_out=True,
        )
    except FileNotFoundError:
        return SubprocessResult(
            returncode=-1,
            stderr=f"executable not found: {cmd[0]}",
            duration_seconds=time.time() - started,
        )
    except OSError as exc:
        return SubprocessResult(
            returncode=-1,
            stderr=f"OS error: {exc}",
            duration_seconds=time.time() - started,
        )


# ════════════════════════════════════════════════════════════════
# Manifest helpers
# ════════════════════════════════════════════════════════════════


def load_adapter_manifest(manifest_id: str) -> tuple[Scenario, ...]:
    """Load scenarios from the external manifests directory by benchmark id."""
    path = _MANIFESTS_DIR / f"{manifest_id}.yaml"
    if not path.exists():
        path = _MANIFESTS_DIR / f"{manifest_id}.yml"
    if not path.exists():
        logger.warning("manifest not found for %s at %s", manifest_id, _MANIFESTS_DIR)
        return ()
    manifest, errors = load_manifest(path)
    if errors:
        logger.warning("manifest errors for %s: %s", manifest_id, errors)
    if manifest is None:
        return ()
    return manifest.scenarios


# ════════════════════════════════════════════════════════════════
# Normalization utilities
# ════════════════════════════════════════════════════════════════


def unavailable_result(
    adapter_id: str,
    adapter_version: str,
    scenario: Scenario,
    *,
    seed: int = 42,
    reason: str = "adapter dependencies not available",
) -> TrialResult:
    """Build a typed UNAVAILABLE trial result."""
    now = time.time()
    return TrialResult(
        trial_id="",
        scenario_id=scenario.scenario_id,
        status=TrialStatus.UNAVAILABLE,
        started_at=now,
        ended_at=now,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        seed=seed,
        error=reason,
    )


def trial_from_subprocess(
    adapter_id: str,
    adapter_version: str,
    scenario: Scenario,
    result: SubprocessResult,
    *,
    seed: int = 42,
    parse_metrics: bool = True,
) -> TrialResult:
    """Normalize a subprocess result into a ``TrialResult``."""
    now = time.time()
    started = now - result.duration_seconds

    if result.timed_out:
        status = TrialStatus.TIMEOUT
    elif result.returncode != 0:
        status = TrialStatus.FAILED
    else:
        status = TrialStatus.PASSED

    metrics: tuple[MetricValue, ...] = ()
    if parse_metrics and result.stdout:
        metrics = _parse_json_metrics(result.stdout)
    official_metric_names = tuple(
        str(metric)
        for metric in scenario.parameters.get("official_metrics", ())
        if isinstance(metric, str)
    )
    if status is TrialStatus.PASSED and official_metric_names:
        reported = {metric.name for metric in metrics}
        missing = [name for name in official_metric_names if name not in reported]
        if missing:
            status = TrialStatus.FAILED
            result_error = (
                "official evaluator completed without its declared metrics: " + ", ".join(missing)
            )
        else:
            result_error = ""
    else:
        result_error = ""

    evidence_payload = {
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
        "metrics": [metric.to_dict() for metric in metrics],
        "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
    }
    evidence = EvidenceStore(evidence_root(adapter_id)).add_json(
        evidence_payload,
        kind="official_benchmark_result",
    )
    return TrialResult(
        trial_id="",
        scenario_id=scenario.scenario_id,
        status=status,
        metrics=metrics,
        evidence=(evidence,),
        started_at=started,
        ended_at=now,
        duration_seconds=result.duration_seconds,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        seed=seed,
        error=(result_error or (result.stderr[:4096] if status != TrialStatus.PASSED else "")),
        error_type="TimeoutError"
        if result.timed_out
        else (
            "ProcessError"
            if result.returncode != 0
            else ("OfficialResultContractError" if result_error else "")
        ),
    )


def _parse_json_metrics(stdout: str) -> tuple[MetricValue, ...]:
    """Best-effort parse of JSON metrics from stdout.

    Looks for the last JSON object or array in stdout, extracts
    numeric key-value pairs as ``MetricValue`` instances.
    """
    metrics: list[MetricValue] = []
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metrics.append(MetricValue(name=str(k), value=float(v)))
                break
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        metrics.append(MetricValue.from_dict(item))
                break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return tuple(metrics)


__all__ = [
    "DependencySpec",
    "SubprocessResult",
    "check_dependencies",
    "command_availability",
    "configured_data_root",
    "load_adapter_manifest",
    "make_availability",
    "probe_env",
    "probe_executable",
    "probe_module",
    "run_configured_command",
    "run_subprocess",
    "trial_from_subprocess",
    "unavailable_result",
]

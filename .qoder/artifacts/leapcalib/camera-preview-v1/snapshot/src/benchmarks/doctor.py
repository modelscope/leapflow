# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only environment diagnostics for benchmark dependencies.

Doctor checks Python packages, executables, GPU availability, environment
variables, data directories, license acknowledgements, and hardware device
paths.  It never modifies the environment and never installs dependencies.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DoctorCheck:
    """Result of one environment diagnostic check."""

    check_id: str
    category: str
    available: bool
    required: bool = True
    detail: str = ""
    remediation: str = ""

    @property
    def status(self) -> str:
        """Return pass, warn, or fail based on availability and requirement."""
        if self.available:
            return "pass"
        return "fail" if self.required else "warn"

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "available": self.available,
            "required": self.required,
        }
        if self.detail:
            d["detail"] = self.detail
        if self.remediation:
            d["remediation"] = self.remediation
        return d


@dataclass(frozen=True)
class DoctorReport:
    """Aggregate environment diagnostic report."""

    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)

    @property
    def ready(self) -> bool:
        """True when every required check passes."""
        return all(c.available or not c.required for c in self.checks)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "failed": self.failed,
            "warnings": self.warnings,
            "checks": [c.to_dict() for c in self.checks],
        }


def check_python_package(name: str, *, required: bool = True) -> DoctorCheck:
    """Check whether a Python import is discoverable without importing it."""
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        found = False
    return DoctorCheck(
        check_id=f"python:{name}",
        category="python_package",
        available=found,
        required=required,
        detail="installed" if found else "not installed",
        remediation=f"Install Python package {name!r}" if not found else "",
    )


def check_executable(name: str, *, required: bool = True) -> DoctorCheck:
    """Check whether an executable is present on PATH."""
    path = shutil.which(name)
    return DoctorCheck(
        check_id=f"executable:{name}",
        category="executable",
        available=path is not None,
        required=required,
        detail=path or "not found on PATH",
        remediation=f"Install {name!r} and add it to PATH" if path is None else "",
    )


def check_environment_variable(name: str, *, required: bool = True) -> DoctorCheck:
    """Check whether an environment variable is set (value is never disclosed)."""
    present = bool(os.environ.get(name))
    return DoctorCheck(
        check_id=f"env:{name}",
        category="environment_variable",
        available=present,
        required=required,
        detail="set" if present else "not set",
        remediation=f"Set environment variable {name}" if not present else "",
    )


def check_data_directory(path: str | Path, *, required: bool = True) -> DoctorCheck:
    """Check that a data directory exists and is readable.  Never creates it."""
    p = Path(path).expanduser()
    exists = p.is_dir()
    readable = exists and os.access(p, os.R_OK)
    return DoctorCheck(
        check_id=f"data:{p}",
        category="data_directory",
        available=readable,
        required=required,
        detail=(
            "exists and readable" if readable
            else "exists but is not readable" if exists
            else "directory not found"
        ),
        remediation=f"Provide readable data directory at {p}" if not readable else "",
    )


def _typed_license_acceptances() -> Mapping[str, Any]:
    """Read benchmark license acknowledgements from the typed Settings plane."""
    try:
        from leapflow.config import load_config

        decoded = json.loads(load_config().benchmark_license_acceptances)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def check_license_acknowledgement(
    license_id: str,
    *,
    env_var: str = "",
    required: bool = True,
) -> DoctorCheck:
    """Check a Settings acknowledgement before using the legacy env fallback."""
    key = env_var or f"BENCHMARK_LICENSE_{license_id.upper().replace('-', '_')}_ACCEPTED"
    configured = _typed_license_acceptances().get(license_id, False)
    accepted = bool(configured) or os.environ.get(key, "").strip().lower() in (
        "1", "true", "yes", "accepted",
    )
    return DoctorCheck(
        check_id=f"license:{license_id}",
        category="license_acknowledgement",
        available=accepted,
        required=required,
        detail="acknowledged" if accepted else f"not acknowledged ({key})",
        remediation=(
            f"Review the {license_id} license and set benchmark.license_acceptances "
            f"or {key}=1"
        ) if not accepted else "",
    )


def check_benchmark_configuration(key: str, *, required: bool = True) -> DoctorCheck:
    """Check one non-secret benchmark setting from the active Settings profile."""
    fields = {
        "benchmark.commands": "benchmark_commands",
        "benchmark.data_roots": "benchmark_data_roots",
        "benchmark.license_acceptances": "benchmark_license_acceptances",
        "benchmark.hardware_profiles": "benchmark_hardware_profiles",
    }
    section, _, benchmark_id = key.rpartition(".")
    setting_name = fields.get(section)
    if setting_name is None or not benchmark_id:
        return DoctorCheck(
            check_id=f"config:{key}", category="configuration", available=False,
            required=required, detail="unsupported benchmark configuration key",
            remediation="Use a benchmark.commands, data_roots, license_acceptances, or hardware_profiles key.",
        )
    try:
        from leapflow.config import load_config

        raw = str(getattr(load_config(), setting_name))
        values = json.loads(raw)
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return DoctorCheck(
            check_id=f"config:{key}", category="configuration", available=False,
            required=required, detail=f"cannot read active setting: {type(exc).__name__}",
            remediation=f"Configure {key} with leap config.",
        )
    value = values.get(benchmark_id) if isinstance(values, Mapping) else None
    if section == "benchmark.data_roots" and isinstance(value, str):
        available = Path(value).expanduser().is_dir() and os.access(Path(value).expanduser(), os.R_OK)
        detail = "readable configured dataset root" if available else "missing or unreadable configured dataset root"
    else:
        available = bool(value)
        detail = "configured" if available else "not configured"
    return DoctorCheck(
        check_id=f"config:{key}", category="configuration", available=available,
        required=required, detail=detail,
        remediation=f"Configure {key} with leap config." if not available else "",
    )


def check_hardware_device(path: str | Path, *, required: bool = True) -> DoctorCheck:
    """Check whether a hardware device path exists and is accessible."""
    p = Path(path).expanduser()
    exists = p.exists()
    accessible = exists and os.access(p, os.R_OK | os.W_OK)
    return DoctorCheck(
        check_id=f"hardware:{p}",
        category="hardware_device",
        available=accessible,
        required=required,
        detail=(
            "present and accessible" if accessible
            else "present but not accessible" if exists
            else "device not found"
        ),
        remediation=f"Connect device and grant access to {p}" if not accessible else "",
    )


def check_gpu(*, required: bool = False) -> DoctorCheck:
    """Check for a discoverable GPU toolchain without importing heavy SDKs."""
    providers = [
        p for p in ("nvidia-smi", "rocm-smi", "system_profiler")
        if shutil.which(p) is not None
    ]
    available = bool(providers)
    return DoctorCheck(
        check_id="gpu",
        category="gpu",
        available=available,
        required=required,
        detail=(f"provider(s): {', '.join(providers)}" if available else "no GPU tool detected"),
        remediation="Install a GPU driver/toolchain if this benchmark requires one"
        if not available else "",
    )


def run_doctor(
    *,
    python_packages: Sequence[str] = (),
    executables: Sequence[str] = (),
    environment_variables: Sequence[str] = (),
    data_directories: Sequence[str | Path] = (),
    licenses: Sequence[str] = (),
    configuration_keys: Sequence[str] = (),
    hardware_devices: Sequence[str | Path] = (),
    gpu_required: bool = False,
) -> DoctorReport:
    """Run all requested diagnostics and return a typed report.

    No check mutates the environment.  Nothing is installed or downloaded.
    """
    checks: list[DoctorCheck] = []
    checks.extend(check_python_package(n) for n in python_packages)
    checks.extend(check_executable(n) for n in executables)
    checks.extend(check_environment_variable(n) for n in environment_variables)
    checks.extend(check_data_directory(p) for p in data_directories)
    checks.extend(check_license_acknowledgement(n) for n in licenses)
    checks.extend(check_benchmark_configuration(key) for key in configuration_keys)
    checks.extend(check_hardware_device(p) for p in hardware_devices)
    checks.append(check_gpu(required=gpu_required))
    return DoctorReport(checks=tuple(checks))


__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "check_benchmark_configuration",
    "check_data_directory",
    "check_environment_variable",
    "check_executable",
    "check_gpu",
    "check_hardware_device",
    "check_license_acknowledgement",
    "check_python_package",
    "run_doctor",
]

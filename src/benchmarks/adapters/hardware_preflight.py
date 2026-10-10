# Copyright (c) Alibaba, Inc. and its affiliates.
"""Explicit Tier4 device preflight adapter.

This adapter never invents a hardware connection.  It requires both an
operator's ``--confirm-hardware`` flag and an approved profile in the active
LeapFlow Settings.  The profile's optional preflight command is an
operator-supplied executable; it is the only code allowed to touch a real
device from this adapter.
"""

from __future__ import annotations

import json
import shlex
import time
from typing import Any, Mapping, Sequence

from benchmarks.adapters.base import SubprocessResult, run_subprocess
from benchmarks.evidence import EvidenceStore
from benchmarks.models import AvailabilityResult, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root
from benchmarks.runtime import current_runtime_context

_ADAPTER_ID = "hardware_preflight"
_VERSION = "1.0.0"


class HardwarePreflightAdapter:
    """Run operator-authorized device preflight commands with typed evidence."""

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _VERSION

    async def availability(self) -> AvailabilityResult:
        runtime = current_runtime_context()
        if not runtime.hardware_enabled:
            return AvailabilityResult(
                _ADAPTER_ID,
                False,
                "Tier4 preflight requires explicit --confirm-hardware authorization",
            )
        profiles = self._profiles()
        if not profiles:
            return AvailabilityResult(
                _ADAPTER_ID,
                False,
                "no approved benchmark.hardware_profiles are configured in the active profile",
            )
        return AvailabilityResult(_ADAPTER_ID, True, "operator-authorized preflight")

    async def list_scenarios(
        self,
        *,
        tags: Sequence[str] = (),
        limit: int = 0,
    ) -> tuple[Scenario, ...]:
        del tags, limit
        return ()

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        started = time.time()
        availability = await self.availability()
        if not availability.available:
            return self._unavailable(scenario, seed, started, availability.reason)
        runtime = current_runtime_context()
        if not runtime.evidence_root or not runtime.run_id:
            return self._unavailable(
                scenario,
                seed,
                started,
                "Tier4 preflight requires persistent run-scoped evidence from the benchmark CLI",
            )
        params = dict(parameters or scenario.parameters)
        device_id = str(params.get("device_id", ""))
        profile = self._profiles().get(device_id)
        if not isinstance(profile, Mapping):
            return self._unavailable(
                scenario, seed, started, f"missing approved profile for device {device_id!r}"
            )
        if profile.get("preflight_mode") != "zero_motion":
            return self._unavailable(
                scenario,
                seed,
                started,
                f"device {device_id!r} must declare preflight_mode='zero_motion'",
            )
        command = profile.get("preflight_command")
        if not isinstance(command, str) or not command.strip():
            return self._unavailable(
                scenario,
                seed,
                started,
                f"device {device_id!r} has no preflight_command in benchmark.hardware_profiles",
            )
        try:
            required_checks = self._required_checks(params)
        except ValueError as exc:
            return self._unavailable(scenario, seed, started, str(exc))
        try:
            result = self._run_command(command, scenario, seed, timeout_seconds)
            report, validation_error = self._validate_zero_motion_report(
                result.stdout,
                device_id=device_id,
                required_checks=required_checks,
            )
            passed = result.returncode == 0 and not result.timed_out and not validation_error
            evidence = {
                "device_id": device_id,
                "profile_id": str(profile.get("profile_id", "")),
                "preflight_mode": "zero_motion",
                "required_checks": list(required_checks),
                "report": report,
                "validation_error": validation_error,
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "stdout_sha256": self._digest(result.stdout),
                "stderr_sha256": self._digest(result.stderr),
            }
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(
                evidence, kind="hardware_preflight"
            )
            ended = time.time()
            error = validation_error or ("" if passed else "operator preflight command failed")
            return TrialResult(
                "",
                scenario.scenario_id,
                TrialStatus.PASSED if passed else TrialStatus.FAILED,
                evidence=(ref,),
                started_at=started,
                ended_at=ended,
                duration_seconds=ended - started,
                adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION,
                seed=seed,
                error=error,
                error_type="" if passed else "PreflightValidationError",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "",
                scenario.scenario_id,
                TrialStatus.ERROR,
                started_at=started,
                ended_at=ended,
                duration_seconds=ended - started,
                adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION,
                seed=seed,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    @staticmethod
    def _required_checks(parameters: Mapping[str, Any]) -> tuple[str, ...]:
        """Validate the manifest-declared checks for a zero-motion preflight."""
        raw = parameters.get("required_checks", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("Tier4 preflight requires a sequence of required_checks")
        checks = tuple(str(item).strip() for item in raw)
        if not checks or any(not item for item in checks) or len(set(checks)) != len(checks):
            raise ValueError("Tier4 preflight required_checks must be non-empty and unique")
        return checks

    @staticmethod
    def _validate_zero_motion_report(
        stdout: str,
        *,
        device_id: str,
        required_checks: Sequence[str],
    ) -> tuple[dict[str, Any] | None, str]:
        """Validate the command's self-reported zero-motion contract.

        The report is evidence, not a capability sandbox: an external command can
        still violate this contract, so it must execute through a read-only driver
        account and normal hardware governance.
        """
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            return None, "preflight stdout must be one JSON zero-motion report"
        if not isinstance(report, dict):
            return None, "preflight report must be a JSON object"
        if report.get("schema_version") != 1:
            return report, "preflight report schema_version must be 1"
        if report.get("scope") != "zero_motion":
            return report, "preflight report scope must be 'zero_motion'"
        if report.get("device_id") != device_id:
            return report, f"preflight report device_id must be {device_id!r}"
        if report.get("motion_performed") is not False:
            return report, "preflight report must declare motion_performed=false"
        raw_checks = report.get("checks")
        if not isinstance(raw_checks, list):
            return report, "preflight report checks must be a list"
        statuses: dict[str, str] = {}
        for check in raw_checks:
            if not isinstance(check, Mapping):
                return report, "preflight report checks must contain objects"
            check_id = check.get("id")
            status = check.get("status")
            if not isinstance(check_id, str) or not check_id.strip() or not isinstance(status, str):
                return report, "every preflight check requires string id and status"
            if check_id in statuses:
                return report, f"preflight report duplicates check {check_id!r}"
            statuses[check_id] = status
        missing = [check for check in required_checks if check not in statuses]
        failed = [check for check in required_checks if statuses.get(check) != "passed"]
        if missing:
            return report, f"preflight report missing required checks: {', '.join(missing)}"
        if failed:
            return report, f"preflight report has non-passing checks: {', '.join(failed)}"
        return report, ""

    @staticmethod
    def _run_command(
        template: str,
        scenario: Scenario,
        seed: int,
        timeout_seconds: float,
    ) -> SubprocessResult:
        command = [
            part.format(scenario_id=scenario.scenario_id, seed=seed, timeout=timeout_seconds)
            for part in shlex.split(template)
        ]
        return run_subprocess(command, timeout=timeout_seconds)

    @staticmethod
    def _digest(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _profiles() -> Mapping[str, Any]:
        try:
            from leapflow.config import load_config

            parsed = json.loads(load_config().benchmark_hardware_profiles)
        except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}

    def _unavailable(
        self,
        scenario: Scenario,
        seed: int,
        started: float,
        reason: str,
    ) -> TrialResult:
        ended = time.time()
        return TrialResult(
            "",
            scenario.scenario_id,
            TrialStatus.UNAVAILABLE,
            started_at=started,
            ended_at=ended,
            duration_seconds=ended - started,
            adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION,
            seed=seed,
            error=reason,
        )


__all__ = ["HardwarePreflightAdapter"]

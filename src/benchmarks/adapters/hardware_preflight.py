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
        self, *, tags: Sequence[str] = (), limit: int = 0,
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
        params = dict(parameters or scenario.parameters)
        device_id = str(params.get("device_id", ""))
        profile = self._profiles().get(device_id)
        if not isinstance(profile, Mapping):
            return self._unavailable(scenario, seed, started, f"missing approved profile for device {device_id!r}")
        command = profile.get("preflight_command")
        if not isinstance(command, str) or not command.strip():
            return self._unavailable(
                scenario,
                seed,
                started,
                f"device {device_id!r} has no preflight_command in benchmark.hardware_profiles",
            )
        try:
            result = self._run_command(command, scenario, seed, timeout_seconds)
            passed = result.returncode == 0 and not result.timed_out
            evidence = {
                "device_id": device_id,
                "profile_id": str(profile.get("profile_id", "")),
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "stdout_sha256": self._digest(result.stdout),
                "stderr_sha256": self._digest(result.stderr),
            }
            ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(evidence, kind="hardware_preflight")
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id,
                TrialStatus.PASSED if passed else TrialStatus.FAILED,
                evidence=(ref,), started_at=started, ended_at=ended,
                duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed,
                error="" if passed else "operator preflight command failed",
            )
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR,
                started_at=started, ended_at=ended, duration_seconds=ended - started,
                adapter_id=_ADAPTER_ID, adapter_version=_VERSION, seed=seed,
                error=str(exc), error_type=type(exc).__name__,
            )

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
        self, scenario: Scenario, seed: int, started: float, reason: str,
    ) -> TrialResult:
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.UNAVAILABLE,
            started_at=started, ended_at=ended, duration_seconds=ended - started,
            adapter_id=_ADAPTER_ID, adapter_version=_VERSION, seed=seed, error=reason,
        )


__all__ = ["HardwarePreflightAdapter"]

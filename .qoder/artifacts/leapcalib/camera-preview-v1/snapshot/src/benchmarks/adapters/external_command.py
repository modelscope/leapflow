# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generic external-command adapter — subprocess runner for any manifest.

This adapter reads ``manifest.extra["command"]`` and executes it as a
subprocess, normalizing stdout/stderr into the standard TrialResult.
Useful for benchmarks that ship their own CLI and only need a thin
harness wrapper.

No external dependencies beyond the command itself.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.adapters.base import (
    DependencySpec,
    load_adapter_manifest,
    make_availability,
    probe_executable,
    run_subprocess,
    trial_from_subprocess,
    unavailable_result,
)
from benchmarks.models import AvailabilityResult, Scenario, TrialResult

_ADAPTER_ID = "external_command"
_ADAPTER_VERSION = "0.1.0"


@dataclass(frozen=True)
class ExternalCommandConfig:
    """Configuration for the external command adapter.

    Attributes
    ----------
    command:
        The base command template.  ``{scenario_id}``, ``{seed}``, and
        ``{timeout}`` are substituted at run time.
    cwd:
        Working directory for the subprocess.
    executables:
        Additional executables that must be present on PATH.
    """

    command: str = ""
    cwd: str = ""
    executables: tuple[str, ...] = ()


class ExternalCommandAdapter:
    """Generic adapter that delegates to an external CLI command."""

    def __init__(self, config: ExternalCommandConfig | None = None) -> None:
        self._config = config or ExternalCommandConfig()

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _ADAPTER_VERSION

    async def availability(self) -> AvailabilityResult:
        """Check that configured executables are on PATH."""
        specs: list[DependencySpec] = [
            DependencySpec(kind="executable", name=exe, required=True)
            for exe in self._config.executables
        ]

        if self._config.command:
            parts = shlex.split(self._config.command)
            if parts and probe_executable(parts[0]) is None:
                specs.append(DependencySpec(
                    kind="executable", name=parts[0], required=True,
                ))

        extra = "" if self._config.command else "no command configured"
        return make_availability(_ADAPTER_ID, specs, extra_reason=extra)

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        scenarios = load_adapter_manifest(_ADAPTER_ID)
        if tags:
            tag_set = set(tags)
            scenarios = tuple(s for s in scenarios if tag_set.intersection(s.tags))
        if limit > 0:
            scenarios = scenarios[:limit]
        return scenarios

    async def run_trial(
        self,
        scenario: Scenario,
        *,
        seed: int = 42,
        timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        avail = await self.availability()
        if not avail.available:
            return unavailable_result(
                _ADAPTER_ID, _ADAPTER_VERSION, scenario,
                seed=seed, reason=avail.reason,
            )

        template = self._config.command
        rendered = template.format(
            scenario_id=scenario.scenario_id,
            seed=seed,
            timeout=timeout_seconds,
        )
        cmd = shlex.split(rendered)
        result = run_subprocess(
            cmd,
            timeout=timeout_seconds,
            cwd=self._config.cwd or None,
        )
        return trial_from_subprocess(
            _ADAPTER_ID, _ADAPTER_VERSION, scenario, result, seed=seed,
        )

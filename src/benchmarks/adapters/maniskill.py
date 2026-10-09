# Copyright (c) Alibaba, Inc. and its affiliates.
"""ManiSkill3 adapter — manipulation skill benchmark on SAPIEN.

ManiSkill3 provides a large-scale benchmark for generalizable robotic
manipulation with GPU-accelerated simulation (SAPIEN), covering rigid-body,
soft-body, and articulated-object tasks.

Required: an operator-configured official ManiSkill evaluation command and compatible backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.adapters.base import (
    command_availability,
    load_adapter_manifest,
    run_configured_command,
    trial_from_subprocess,
    unavailable_result,
)
from benchmarks.models import AvailabilityResult, Scenario, TrialResult

_ADAPTER_ID = "maniskill"
_ADAPTER_VERSION = "0.1.0"

_COMMAND_ENV = "LEAPFLOW_BENCHMARK_MANISKILL_COMMAND"
_REPOSITORY = "https://github.com/haosulab/ManiSkill"


@dataclass(frozen=True)
class ManiSkillConfig:
    """Local adapter configuration."""

    obs_mode: str = "rgbd"
    control_mode: str = "pd_ee_delta_pose"
    num_envs: int = 1


class ManiSkillAdapter:
    """Adapter for the ManiSkill3 manipulation benchmark."""

    def __init__(self, config: ManiSkillConfig | None = None) -> None:
        self._config = config or ManiSkillConfig()

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _ADAPTER_VERSION

    async def availability(self) -> AvailabilityResult:
        return command_availability(_ADAPTER_ID, _COMMAND_ENV, _REPOSITORY)

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

        result = run_configured_command(
            _COMMAND_ENV, scenario, seed=seed, timeout_seconds=timeout_seconds,
        )
        return trial_from_subprocess(
            _ADAPTER_ID, _ADAPTER_VERSION, scenario, result, seed=seed,
        )

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Isaac Lab adapter — NVIDIA Isaac Sim robot learning benchmark.

Isaac Lab provides GPU-accelerated robot learning environments built on
NVIDIA Isaac Sim / Omniverse, supporting high-fidelity rendering, physics
simulation, and large-scale parallel RL training.

Required: an operator-configured Isaac Lab command, NVIDIA GPU with CUDA,
and the Isaac Sim runtime. This is a heavy benchmark tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.adapters.base import (
    DependencySpec,
    command_availability,
    load_adapter_manifest,
    make_availability,
    run_configured_command,
    trial_from_subprocess,
    unavailable_result,
)
from benchmarks.models import AvailabilityResult, Scenario, TrialResult

_ADAPTER_ID = "isaac_lab"
_ADAPTER_VERSION = "0.1.0"
_COMMAND_ENV = "LEAPFLOW_BENCHMARK_ISAAC_LAB_COMMAND"
_REPOSITORY = "https://github.com/isaac-sim/IsaacLab"

_HARDWARE_DEPS: tuple[DependencySpec, ...] = (
    DependencySpec(kind="gpu", name="gpu", required=True,
                   reason="NVIDIA GPU with CUDA required"),
    DependencySpec(kind="executable", name="nvidia-smi", required=True,
                   reason="NVIDIA driver tools"),
)


@dataclass(frozen=True)
class IsaacLabConfig:
    """Local adapter configuration."""

    isaac_sim_path: str = ""
    num_envs: int = 1
    headless: bool = True


class IsaacLabAdapter:
    """Adapter for the Isaac Lab robot learning benchmark."""

    def __init__(self, config: IsaacLabConfig | None = None) -> None:
        self._config = config or IsaacLabConfig()

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _ADAPTER_VERSION

    async def availability(self) -> AvailabilityResult:
        command = command_availability(_ADAPTER_ID, _COMMAND_ENV, _REPOSITORY)
        if not command.available:
            return command
        return make_availability(_ADAPTER_ID, _HARDWARE_DEPS)

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
            _COMMAND_ENV,
            scenario,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
        return trial_from_subprocess(
            _ADAPTER_ID, _ADAPTER_VERSION, scenario, result, seed=seed,
        )

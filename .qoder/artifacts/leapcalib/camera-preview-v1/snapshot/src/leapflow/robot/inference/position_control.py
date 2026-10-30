# Copyright (c) Alibaba, Inc. and its affiliates.
"""Classical joint-space position control strategy.

No neural network is involved: the strategy computes a position delta between a
target and the current joint state, optionally applying a proportional gain and
clamping each joint step for smoother, safer motion.  It is useful for precise
positioning and as a zero-dependency fallback when no VLA model is loaded.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from leapflow.robot.inference.strategy import (
    ComputeBudget,
    ComputeProfile,
    InferenceResult,
)

logger = logging.getLogger(__name__)

__all__ = ["PositionControlStrategy"]

# Observation keys probed, in order, for the target and current joint vectors.
_TARGET_KEYS = ("target", "target_position", "goal")
_CURRENT_KEYS = ("position", "current", "state", "joint_position")


class PositionControlStrategy:
    """Simple joint-space position controller.

    No neural network -- directly maps target positions to action vectors.
    Useful for precise positioning tasks and as a fallback when no VLA
    model is loaded.

    Supports optional P-control interpolation for smoother motion.
    """

    strategy_id = "position_control"

    def __init__(self, *, gain: float = 1.0, max_step_rad: float = 0.1) -> None:
        self._gain = gain
        self._max_step = max_step_rad

    @property
    def compute_profile(self) -> ComputeProfile:
        return ComputeProfile(
            latency_range_ms=(0.1, 1.0),
            supports_chunking=False,
            gpu_required=False,
        )

    async def infer(
        self,
        observation: Mapping[str, Any],
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        """Compute position delta: target - current, clamp by max_step."""
        t0 = time.monotonic()

        target = self._extract(observation, _TARGET_KEYS)
        current = self._extract(observation, _CURRENT_KEYS)
        if target is None:
            raise RuntimeError(
                "position_control requires a target in the observation "
                f"(one of {_TARGET_KEYS})"
            )
        if current is None:
            # No current state: command the target directly, clamped from zero.
            current = [0.0] * len(target)

        action = self._step_towards(current, target)
        latency_ms = (time.monotonic() - t0) * 1000.0

        return InferenceResult(
            action=action,
            latency_ms=latency_ms,
            confidence=1.0,
            chunk_size=1,
            metadata={"strategy": self.strategy_id, "gain": self._gain},
        )

    async def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_towards(
        self,
        current: list[float],
        target: list[float],
    ) -> list[float]:
        """Return a gain-scaled, per-joint clamped position command."""
        n = min(len(current), len(target))
        command: list[float] = []
        for i in range(n):
            delta = (target[i] - current[i]) * self._gain
            if delta > self._max_step:
                delta = self._max_step
            elif delta < -self._max_step:
                delta = -self._max_step
            command.append(current[i] + delta)
        return command

    @staticmethod
    def _extract(
        observation: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> list[float] | None:
        """Pull the first present key and coerce it to a list of floats."""
        for key in keys:
            if key not in observation:
                continue
            val = observation[key]
            if hasattr(val, "tolist"):
                val = val.tolist()
            if isinstance(val, (int, float)):
                return [float(val)]
            if isinstance(val, (list, tuple)):
                try:
                    return [float(x) for x in val]
                except (ValueError, TypeError):
                    return None
        return None

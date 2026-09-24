# Copyright (c) Alibaba, Inc. and its affiliates.
"""In-process inference strategy wrapping a local ``PreTrainedPolicy``.

Loads a policy from a HuggingFace-format checkpoint directory and runs
inference in the current process.  The policy is loaded lazily on first
``infer`` call (off the event loop via ``asyncio.to_thread``) so constructing
the strategy stays cheap and does not require torch to be installed until an
inference actually runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from leapflow.robot.inference.strategy import (
    ComputeBudget,
    ComputeProfile,
    InferenceResult,
)

logger = logging.getLogger(__name__)

__all__ = ["VLALocalStrategy"]


class VLALocalStrategy:
    """Inference strategy wrapping a local PreTrainedPolicy.

    Loads a policy from a HuggingFace-format checkpoint and runs
    inference in-process.  Supports action chunking if the policy
    implements predict_action_chunk().
    """

    strategy_id = "vla_local"

    def __init__(self, policy_path: str, *, device: str = "cpu") -> None:
        self._path = policy_path
        self._device = device
        self._policy: Any = None  # PreTrainedPolicy, loaded lazily
        self._load_lock = asyncio.Lock()

    @property
    def compute_profile(self) -> ComputeProfile:
        return ComputeProfile(
            latency_range_ms=(10.0, 200.0),
            supports_chunking=True,
            gpu_required=self._device != "cpu",
            max_chunk_size=100,
        )

    async def infer(
        self,
        observation: Mapping[str, Any],
        *,
        budget: ComputeBudget | None = None,
    ) -> InferenceResult:
        """Load policy if needed, run select_action or predict_action_chunk."""
        await self._ensure_loaded()

        chunk_size = max(1, int(budget.chunk_size) if budget is not None else 1)
        obs = self._prepare_observation(observation)

        t0 = time.monotonic()
        if chunk_size > 1 and hasattr(self._policy, "predict_action_chunk"):
            action = await asyncio.to_thread(
                self._policy.predict_action_chunk, obs
            )
        else:
            select = getattr(self._policy, "select_action", None)
            if select is None:
                raise RuntimeError(
                    f"policy at {self._path!r} does not implement select_action()"
                )
            action = await asyncio.to_thread(select, obs)
        latency_ms = (time.monotonic() - t0) * 1000.0

        return InferenceResult(
            action=action,
            latency_ms=latency_ms,
            confidence=1.0,
            chunk_size=chunk_size,
            metadata={"strategy": self.strategy_id, "device": self._device},
        )

    async def reset(self) -> None:
        """Call policy.reset()."""
        if self._policy is None:
            return
        reset = getattr(self._policy, "reset", None)
        if reset is None:
            return
        try:
            await asyncio.to_thread(reset)
        except Exception:  # noqa: BLE001 - reset failure must not abort the episode
            logger.debug("policy.reset() raised; suppressed", exc_info=True)

    async def _ensure_loaded(self) -> None:
        """Lazy-load the policy via asyncio.to_thread."""
        if self._policy is not None:
            return
        async with self._load_lock:
            if self._policy is not None:
                return
            try:
                from leapflow.robot.policy import make_policy
            except ImportError as exc:
                raise RuntimeError(
                    "LeapRobot policy package is not available; "
                    "install with: pip install leapflow[robot]"
                ) from exc
            self._policy = await asyncio.to_thread(
                make_policy, self._path, device=self._device,
            )
            logger.info("VLALocalStrategy loaded policy: %s", self._path)

    def _prepare_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Convert numeric observation values to tensors when torch is present."""
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError:
            return dict(observation)

        prepared: dict[str, Any] = {}
        for key, val in observation.items():
            if isinstance(val, (int, float)):
                prepared[key] = torch.tensor([val], dtype=torch.float32)
            elif isinstance(val, list):
                try:
                    prepared[key] = torch.tensor(val, dtype=torch.float32)
                except (ValueError, TypeError):
                    prepared[key] = val
            else:
                prepared[key] = val
        return prepared

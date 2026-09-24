# Copyright (c) Alibaba, Inc. and its affiliates.
"""Pre/post-processing pipeline for policy inference.

Provides ``ProcessorStep`` (Protocol), concrete processors
(``NormalizeProcessor``, ``DeviceProcessor``, ``UnnormalizeProcessor``),
and the ``ProcessorPipeline`` that chains them.

torch and numpy are optional dependencies — the module is importable
without them, but concrete processors require them at runtime.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import Tensor

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment,misc]

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProcessorStep Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ProcessorStep(Protocol):
    """One step in the pre/post-processing pipeline.

    ``forward`` transforms data *towards* the model (pre-processing);
    ``reverse`` transforms data *away from* the model (post-processing).
    Both must be pure in the sense that they do not mutate global state,
    although they may update internal running statistics.
    """

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Transform *batch* in the forward (pre-processing) direction."""
        ...

    def reverse(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Transform *batch* in the reverse (post-processing) direction."""
        ...


# ---------------------------------------------------------------------------
# NormalizeProcessor
# ---------------------------------------------------------------------------

class NormalizeProcessor:
    """Normalize observation values to a model-expected distribution.

    Supports two modes selectable via *mode*:

    - ``"mean_std"`` — classic z-score normalisation: ``(x - mean) / std``
    - ``"min_max"`` — scale to [-1, 1]: ``2 * (x - min) / (max - min) - 1``

    Statistics are supplied at construction time as plain dicts so they
    can be loaded from a JSON config without a torch dependency at
    import time.
    """

    def __init__(
        self,
        stats: dict[str, dict[str, Any]],
        *,
        mode: str = "mean_std",
        keys: list[str] | None = None,
    ) -> None:
        """
        Args:
            stats: Mapping ``feature_key -> {"mean": ..., "std": ...}``
                or ``feature_key -> {"min": ..., "max": ...}`` depending
                on *mode*.  Values may be scalars, lists, or tensors.
            mode: ``"mean_std"`` or ``"min_max"``.
            keys: If supplied, only these keys are normalised.  ``None``
                normalises every key present in *stats*.
        """
        if mode not in ("mean_std", "min_max"):
            raise ValueError(f"Unsupported normalization mode: {mode!r}")
        self._stats = stats
        self._mode = mode
        self._keys = set(keys) if keys else None

    def _should_process(self, key: str) -> bool:
        if self._keys is not None and key not in self._keys:
            return False
        return key in self._stats

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalise *batch* values in-place and return it."""
        for key, value in batch.items():
            if not self._should_process(key):
                continue
            s = self._stats[key]
            if self._mode == "mean_std":
                batch[key] = self._z_normalize(value, s["mean"], s["std"])
            else:
                batch[key] = self._minmax_normalize(value, s["min"], s["max"])
        return batch

    def reverse(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Un-normalise *batch* values in-place and return it."""
        for key, value in batch.items():
            if not self._should_process(key):
                continue
            s = self._stats[key]
            if self._mode == "mean_std":
                batch[key] = self._z_unnormalize(value, s["mean"], s["std"])
            else:
                batch[key] = self._minmax_unnormalize(value, s["min"], s["max"])
        return batch

    # -- arithmetic helpers (work with scalars, numpy, or torch) ----------

    @staticmethod
    def _z_normalize(x: Any, mean: Any, std: Any) -> Any:
        return (x - mean) / (std + 1e-8)

    @staticmethod
    def _z_unnormalize(x: Any, mean: Any, std: Any) -> Any:
        return x * (std + 1e-8) + mean

    @staticmethod
    def _minmax_normalize(x: Any, lo: Any, hi: Any) -> Any:
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0

    @staticmethod
    def _minmax_unnormalize(x: Any, lo: Any, hi: Any) -> Any:
        return (x + 1.0) / 2.0 * (hi - lo + 1e-8) + lo


# ---------------------------------------------------------------------------
# UnnormalizeProcessor
# ---------------------------------------------------------------------------

class UnnormalizeProcessor:
    """Reverse a NormalizeProcessor in the forward direction.

    Useful as a post-processor: its ``forward`` un-normalises, and its
    ``reverse`` normalises — the exact mirror of ``NormalizeProcessor``.
    """

    def __init__(
        self,
        stats: dict[str, dict[str, Any]],
        *,
        mode: str = "mean_std",
        keys: list[str] | None = None,
    ) -> None:
        self._inner = NormalizeProcessor(stats, mode=mode, keys=keys)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inner.reverse(batch)

    def reverse(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._inner.forward(batch)


# ---------------------------------------------------------------------------
# DeviceProcessor
# ---------------------------------------------------------------------------

class DeviceProcessor:
    """Move tensors between devices (CPU ↔ GPU).

    Non-tensor values are passed through unchanged.
    """

    def __init__(self, device: str = "cpu") -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("DeviceProcessor requires torch")
        self._device = torch.device(device)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensor values to the configured device."""
        return {
            k: v.to(self._device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def reverse(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensor values to CPU (inverse of forward)."""
        cpu = torch.device("cpu")
        return {
            k: v.to(cpu) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }


# ---------------------------------------------------------------------------
# ProcessorPipeline
# ---------------------------------------------------------------------------

class ProcessorPipeline:
    """Chain of ``ProcessorStep``-compatible objects.

    ``forward()`` runs every step's ``forward`` in order;
    ``reverse()`` runs every step's ``reverse`` in *reverse* order.
    """

    def __init__(self, steps: list[Any] | None = None) -> None:
        """
        Args:
            steps: Ordered list of processor step instances.  Each must
                expose ``forward(batch)`` and ``reverse(batch)`` methods.
        """
        self._steps: list[Any] = list(steps) if steps else []

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self._steps]
        return f"ProcessorPipeline(steps={step_names})"

    def add(self, step: Any) -> None:
        """Append a step to the pipeline."""
        self._steps.append(step)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run all steps' ``forward`` in declaration order."""
        for step in self._steps:
            batch = step.forward(batch)
        return batch

    def reverse(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run all steps' ``reverse`` in reverse declaration order."""
        for step in reversed(self._steps):
            batch = step.reverse(batch)
        return batch

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Alias for :meth:`forward`."""
        return self.forward(batch)


__all__ = [
    "ProcessorStep",
    "NormalizeProcessor",
    "UnnormalizeProcessor",
    "DeviceProcessor",
    "ProcessorPipeline",
]

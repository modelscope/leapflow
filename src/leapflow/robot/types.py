# Copyright (c) Alibaba, Inc. and its affiliates.
"""Core type aliases for the LeapRobot package.

These types mirror the inference-side subset of the upstream open-source robot
type vocabulary.  Heavy dependencies (torch, numpy) are conditionally
imported so the module remains importable in environments where only
standard-library Python is available.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Torch — optional dependency
# ---------------------------------------------------------------------------
try:
    import torch

    PolicyAction = torch.Tensor
    """Action tensor produced by a policy network."""

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    PolicyAction = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# NumPy — optional dependency
# ---------------------------------------------------------------------------
try:
    import numpy as np

    EnvAction = np.ndarray
    """Action array used by simulation environments."""

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False
    EnvAction = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Robot-level types (always available)
# ---------------------------------------------------------------------------

RobotObservation = dict[str, Any]
"""Flat mapping of sensor name to value returned by a robot."""

RobotAction = dict[str, Any]
"""Flat mapping of actuator name to commanded value."""

BatchType = dict[str, Any]
"""Generic dict batch fed to / returned by policy inference."""

ObservationFeatures = dict[str, tuple[int, ...]]
"""Maps observation key to its shape tuple."""

ActionFeatures = dict[str, tuple[int, ...]]
"""Maps action key to its shape tuple."""

__all__ = [
    "PolicyAction",
    "EnvAction",
    "RobotObservation",
    "RobotAction",
    "BatchType",
    "ObservationFeatures",
    "ActionFeatures",
    "_TORCH_AVAILABLE",
    "_NUMPY_AVAILABLE",
]

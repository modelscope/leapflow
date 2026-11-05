# Copyright (c) Alibaba, Inc. and its affiliates.
"""Small numerical primitives and bounded configuration, isolated from contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from leapcalib.contracts.failures import CalibrationFailure, FAILURE_NOT_IMPLEMENTED
from leapcalib.contracts.units import canonical_number
from leapcalib.geometry.se3 import Transform


def numpy() -> Any:
    """Optional NumPy backend; importing a method never imports the numerical stack."""
    try:
        import numpy as np
    except ImportError as error:
        raise CalibrationFailure(
            FAILURE_NOT_IMPLEMENTED, "numerical methods require the leapcalib extra (NumPy)"
        ) from error
    return np


@dataclass(frozen=True)
class SolveConfig:
    """All bounded-work and measurement assumptions included in a result digest."""

    max_pairs: int = 128
    max_points: int = 2048
    trials: int = 48
    seed: int = 0
    min_pairs: int = 4
    min_points: int = 6
    min_inlier_ratio: float = 0.65
    point_gate_m: float = 0.004
    translation_sigma_m: float = 0.002
    rotation_sigma_rad: float = 0.01
    residual_gate: float = 4.0
    min_rotation_rad: float = 0.01
    max_stamp_uncertainty_s: float = 0.05
    max_settled_speed_rad_s: float = 0.005
    rank_rcond: float = 1e-5
    max_condition: float = 1e5
    refinement_steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "max_pairs",
            "max_points",
            "trials",
            "min_pairs",
            "min_points",
            "refinement_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            self.max_pairs > 1024
            or self.max_points > 20000
            or self.trials > 1024
            or self.refinement_steps > 100
        ):
            raise ValueError("numerical work budget exceeds the supported bounds")
        if (
            self.min_pairs < 3
            or self.min_points < 3
            or self.min_pairs > self.max_pairs
            or self.min_points > self.max_points
        ):
            raise ValueError("invalid minimum sample budget")
        integer_fields = {
            "max_pairs",
            "max_points",
            "trials",
            "seed",
            "min_pairs",
            "min_points",
            "refinement_steps",
        }
        for name, value in vars(self).items():
            if name in integer_fields:
                continue
            value = canonical_number(value)
            object.__setattr__(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_inlier_ratio > 1 or self.rank_rcond >= 1 or self.max_condition <= 1:
            raise ValueError("invalid inlier/rank/conditioning bounds")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def array_transform(transform: Transform) -> tuple[Any, Any]:
    np = numpy()
    return np.asarray(transform.rotation), np.asarray(transform.translation_m)


def transform(rotation: Any, translation: Any) -> Transform:
    return Transform(
        tuple(tuple(float(v) for v in row) for row in rotation),
        tuple(float(v) for v in translation),
    )


def rotation_vector(rotation: Any) -> Any:
    """SO(3) logarithm, including the near-pi numerical case."""
    np = numpy()
    vector = (
        np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        / 2
    )
    sine = float(np.linalg.norm(vector))
    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    angle = math.atan2(sine, cosine)
    if angle < 1e-8:
        return vector
    if sine > 1e-7:
        return vector * (angle / sine)
    _, axes = np.linalg.eigh((rotation + rotation.T + 2 * np.eye(3)) / 4)
    axis = axes[:, -1]
    if float(axis @ vector) < 0:
        axis = -axis
    return axis * angle


def perturbation(values: Any) -> Transform:
    np = numpy()
    angle = float(np.linalg.norm(values[3:]))
    return Transform.from_axis_angle(values[3:], angle, values[:3])


def correction_values(prior: Transform, candidate: Transform) -> Any:
    np = numpy()
    delta = prior.inverse().compose(candidate)
    return np.r_[delta.translation_m, rotation_vector(np.asarray(delta.rotation))]


def jacobian(function: Any, values: Any, *, step: float = 1e-6) -> Any:
    np = numpy()
    base = function(values)
    columns = []
    for index in range(len(values)):
        trial = values.copy()
        trial[index] += step
        columns.append((function(trial) - base) / step)
    return np.column_stack(columns)


def rank_report(rows: Any, rcond: float) -> tuple[int, float | None, Any, Any]:
    """Data-only SVD, with full nullspace (not a prior-regularized Hessian)."""
    np = numpy()
    # Full V is necessary only for wide matrices; avoid quadratic U on tall evidence.
    _, singular, vectors = np.linalg.svd(rows, full_matrices=rows.shape[0] < rows.shape[1])
    threshold = max(float(singular[0]) * rcond, 1e-10) if len(singular) else 1e-10
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0] / singular[rank - 1]) if rank else None
    return rank, condition, singular, vectors[rank:]


def wrap_rad(value: Any) -> Any:
    return (value + math.pi) % (2 * math.pi) - math.pi


__all__ = ["SolveConfig"]

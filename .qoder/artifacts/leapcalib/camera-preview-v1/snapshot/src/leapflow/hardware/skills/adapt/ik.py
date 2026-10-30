# Copyright (c) Alibaba, Inc. and its affiliates.
"""Damped-least-squares inverse kinematics over a declared DH chain.

A policy that speaks end-effector poses can drive an arm whose writable
channels are joints only after each pose is solved for joints. The solver is
the textbook damped least squares (Levenberg-Marquardt on the pose error):

    dq = J^T (J J^T + lambda^2 I)^-1 e

with ``e`` the position error in mm stacked on the orientation error (its
rotation vector, weighted by ``rotation_weight_mm`` per radian, so one number
trades millimetres against degrees) and ``J`` a finite-difference Jacobian in
mm per joint degree. Damping keeps a step bounded near a singularity instead
of flinging a joint; it fades with the error (``lambda^2 * min(1, |e| / 10 mm)``,
after Sugihara) so the last millimetre converges quadratically rather than
crawling. ``max_step_deg`` caps a step outright, and joint limits clip
every iterate. Pure Python: a six-joint arm solves in well under a millisecond
per iteration.

The solution is only as good as the declared table, which is why a binding
only builds an IK step after the chain's agreement with the firmware's own
flange report has been checked.
"""

from __future__ import annotations

import math
from typing import Sequence

from leapflow.hardware.skills.adapt.spec import IkSettings
from leapflow.hardware.skills.geometry import (
    Transform,
    mat_mul,
    rotation_vector,
    solve_linear,
    transpose,
)
from leapflow.hardware.skills.kinematics import KinematicModel

_PROBE_DEG = 1e-4  # finite-difference step per joint
_DAMPING_SCALE_MM = 10.0  # error at which the full damping applies
_DAMPING_FLOOR = 1e-6
Limits = Sequence[tuple[float | None, float | None]]


def _error(target: Transform, current: Transform, weight: float) -> list[float]:
    position = [t - c for t, c in zip(target.translation, current.translation)]
    turn = rotation_vector(mat_mul(target.rotation, transpose(current.rotation)))
    return position + [v * weight for v in turn]


def _jacobian(
    chain: KinematicModel, q: Sequence[float], at: Transform, weight: float
) -> list[list[float]]:
    """6 x n: rows are the error components, columns the joints."""
    columns: list[list[float]] = []
    for k in range(len(q)):
        probe = list(q)
        probe[k] += _PROBE_DEG
        moved = chain.transform(probe)
        column = [(m - c) / _PROBE_DEG for m, c in zip(moved.translation, at.translation)]
        turn = rotation_vector(mat_mul(moved.rotation, transpose(at.rotation)))
        columns.append(column + [v * weight / _PROBE_DEG for v in turn])
    return [[columns[k][i] for k in range(len(q))] for i in range(6)]


def _clip(q: Sequence[float], limits: Limits) -> list[float]:
    out: list[float] = []
    for value, (low, high) in zip(q, limits):
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        out.append(value)
    return out


def _miss(target: Transform, current: Transform) -> tuple[float, float]:
    mm = math.dist(target.translation, current.translation)
    turn = rotation_vector(mat_mul(target.rotation, transpose(current.rotation)))
    return mm, math.degrees(math.sqrt(sum(v * v for v in turn)))


def solve_ik(
    chain: KinematicModel,
    target: Transform,
    seed: Sequence[float],
    limits: Limits,
    settings: IkSettings,
) -> tuple[float, ...]:
    """Joints (deg) that put the flange at *target*; raises ``ValueError`` if none converge."""
    if not isinstance(chain, KinematicModel):
        raise ValueError("IK requires the pure forward-transform model contract")
    if len(seed) != chain.dof:
        raise ValueError(f"IK seed has {len(seed)} joints, the chain {chain.dof}")
    padded = list(limits) + [(None, None)] * (chain.dof - len(limits))
    q = _clip([float(v) for v in seed], padded)
    weight = settings.rotation_weight_mm
    damping = settings.damping * settings.damping
    for _ in range(max(1, int(settings.max_iterations))):
        current = chain.transform(q)
        mm, deg = _miss(target, current)
        if mm <= settings.tolerance_mm and deg <= settings.tolerance_deg:
            return tuple(q)
        error = _error(target, current, weight)
        size = math.sqrt(sum(v * v for v in error))
        lam = max(_DAMPING_FLOOR, damping * min(1.0, size / _DAMPING_SCALE_MM))
        jac = _jacobian(chain, q, current, weight)
        jjt = [
            [sum(a * b for a, b in zip(jac[i], jac[j])) + (lam if i == j else 0.0)
             for j in range(6)]
            for i in range(6)
        ]  # fmt: skip
        try:
            y = solve_linear(jjt, error)
        except ValueError as exc:
            raise ValueError(f"IK is singular at this pose: {exc}") from exc
        dq = [sum(jac[i][k] * y[i] for i in range(6)) for k in range(chain.dof)]
        largest = max(abs(d) for d in dq)
        if largest > settings.max_step_deg:
            dq = [d * settings.max_step_deg / largest for d in dq]
        q = _clip([a + d for a, d in zip(q, dq)], padded)
    mm, deg = _miss(target, chain.transform(q))
    if mm <= settings.tolerance_mm and deg <= settings.tolerance_deg:
        return tuple(q)
    raise ValueError(
        f"no joint solution within {settings.tolerance_mm:g} mm / {settings.tolerance_deg:g} deg "
        f"(closest is {mm:.1f} mm / {deg:.1f} deg off; unreachable or outside the joint limits)"
    )


__all__ = ["solve_ik"]

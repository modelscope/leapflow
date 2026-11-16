# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rotation representations a policy may speak, to and from a rotation matrix.

Names (the ``rotation`` of an ``ee_pose`` group):

- ``euler_<order>``: three Tait-Bryan angles. Lower case is extrinsic (fixed
  axes, applied in the written order), upper case intrinsic (moving axes):
  ``euler_xyz`` is ``Rz @ Ry @ Rx`` -- the arm's own convention -- and
  ``euler_XYZ`` is ``Rx @ Ry @ Rz``.
- ``axis_angle``: the rotation vector (axis times angle).
- ``quat_xyzw`` / ``quat_wxyz``: a unit quaternion; read normalised, written
  with ``w >= 0`` so one rotation has one encoding.
- ``rot6d``: the first two *columns* of the matrix, column after column
  (Zhou et al.); ``rot6d_rows``: the first two rows (pytorch3d's layout).
  Read through Gram-Schmidt, so a model's output need not be orthonormal.
- ``none``: position only; the orientation is held.

Euler and axis-angle values carry the group's ``angle_unit``; the others are
unitless. Pure Python, like the rest of the geometry.
"""

from __future__ import annotations

import math
from typing import Sequence

from leapflow.hardware.skills.geometry import Mat3, mat_mul, rotation_vector

ROTATION_NONE = "none"
AXIS_ANGLE = "axis_angle"
QUAT_XYZW = "quat_xyzw"
QUAT_WXYZ = "quat_wxyz"
ROT6D = "rot6d"
ROT6D_ROWS = "rot6d_rows"
EULER_PREFIX = "euler_"
ARM_EULER = "euler_xyz"  # the arm's fixed-axis X-Y-Z convention (geometry.Pose)
_WIDTHS = {ROTATION_NONE: 0, AXIS_ANGLE: 3, QUAT_XYZW: 4, QUAT_WXYZ: 4, ROT6D: 6, ROT6D_ROWS: 6}
_AXES = {"x": 0, "y": 1, "z": 2}


def rotation_width(name: str) -> int:
    """Values the representation takes; raises ``ValueError`` for an unknown name."""
    if name in _WIDTHS:
        return _WIDTHS[name]
    _euler_order(name)
    return 3


def uses_angles(name: str) -> bool:
    """Whether the representation's values are angles (and so carry an angle unit)."""
    return name == AXIS_ANGLE or name.startswith(EULER_PREFIX)


def _euler_order(name: str) -> tuple[tuple[int, int, int], bool]:
    """``(intrinsic axis order, written intrinsic)`` of an Euler name."""
    letters = name[len(EULER_PREFIX) :] if name.startswith(EULER_PREFIX) else ""
    lowered = letters.lower()
    if len(letters) != 3 or sorted(lowered) != ["x", "y", "z"]:
        raise ValueError(
            f"unknown rotation {name!r}; use euler_<xyz order> (Tait-Bryan), axis_angle, "
            "quat_xyzw, quat_wxyz, rot6d, rot6d_rows or none"
        )
    if letters not in (lowered, letters.upper()):
        raise ValueError(f"rotation {name!r} mixes intrinsic and extrinsic axes")
    order = tuple(_AXES[letter] for letter in lowered)
    intrinsic = letters.isupper()
    # Extrinsic a-b-c equals intrinsic c-b-a with the angles reversed.
    return (order if intrinsic else order[::-1]), intrinsic  # type: ignore[return-value]


def _axis(axis: int, angle: float) -> Mat3:
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == 1:
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _to_rad(values: Sequence[float], angle_unit: str) -> list[float]:
    return [math.radians(v) if angle_unit == "deg" else float(v) for v in values]


def _from_rad(values: Sequence[float], angle_unit: str) -> tuple[float, ...]:
    return tuple(math.degrees(v) if angle_unit == "deg" else v for v in values)


def to_matrix(name: str, values: Sequence[float], angle_unit: str = "deg") -> Mat3:
    """The rotation *values* encode; raises ``ValueError`` on a degenerate encoding."""
    if len(values) != rotation_width(name):
        raise ValueError(f"{name} takes {rotation_width(name)} values, got {len(values)}")
    if name == AXIS_ANGLE:
        return _rodrigues(_to_rad(values, angle_unit))
    if name in (QUAT_XYZW, QUAT_WXYZ):
        x, y, z, w = values if name == QUAT_XYZW else (*values[1:], values[0])
        return _quaternion(w, x, y, z)
    if name in (ROT6D, ROT6D_ROWS):
        return _gram_schmidt(values, rows=name == ROT6D_ROWS)
    order, intrinsic = _euler_order(name)
    angles = _to_rad(values, angle_unit)
    angles = angles if intrinsic else angles[::-1]
    matrix = _axis(order[0], angles[0])
    for axis, angle in zip(order[1:], angles[1:]):
        matrix = mat_mul(matrix, _axis(axis, angle))
    return matrix


def from_matrix(name: str, r: Mat3, angle_unit: str = "deg") -> tuple[float, ...]:
    """The encoding of rotation *r* in representation *name*."""
    if name == ROTATION_NONE:
        return ()
    if name == AXIS_ANGLE:
        return _from_rad(rotation_vector(r), angle_unit)
    if name in (QUAT_XYZW, QUAT_WXYZ):
        w, x, y, z = _matrix_quaternion(r)
        return (x, y, z, w) if name == QUAT_XYZW else (w, x, y, z)
    if name == ROT6D:
        return (r[0][0], r[1][0], r[2][0], r[0][1], r[1][1], r[2][1])
    if name == ROT6D_ROWS:
        return (*r[0], *r[1])
    order, intrinsic = _euler_order(name)
    angles = _intrinsic_angles(r, order)
    return _from_rad(angles if intrinsic else angles[::-1], angle_unit)


def _intrinsic_angles(r: Mat3, order: tuple[int, int, int]) -> tuple[float, float, float]:
    """``(a, b, c)`` with ``r = R_i(a) R_j(b) R_k(c)`` for Tait-Bryan ``(i, j, k)``."""
    i, j, k = order
    sign = 1.0 if (j - i) % 3 == 1 else -1.0  # +1 for a cyclic order
    sb = max(-1.0, min(1.0, sign * r[i][k]))
    b = math.asin(sb)
    if abs(sb) < 1.0 - 1e-9:
        a = math.atan2(-sign * r[j][k], r[k][k])
        c = math.atan2(-sign * r[i][j], r[i][i])
    else:  # gimbal lock: the first angle absorbs the last
        a, c = math.atan2(sign * r[k][j], r[j][j]), 0.0
    return a, b, c


def _rodrigues(v: Sequence[float]) -> Mat3:
    angle = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if angle < 1e-12:
        return ((1.0, -v[2], v[1]), (v[2], 1.0, -v[0]), (-v[1], v[0], 1.0))
    x, y, z = v[0] / angle, v[1] / angle, v[2] / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return (
        (c + x * x * t, x * y * t - z * s, x * z * t + y * s),
        (y * x * t + z * s, c + y * y * t, y * z * t - x * s),
        (z * x * t - y * s, z * y * t + x * s, c + z * z * t),
    )


def _quaternion(w: float, x: float, y: float, z: float) -> Mat3:
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(length) or length < 1e-9:
        raise ValueError("a quaternion must have a non-zero length")
    w, x, y, z = w / length, x / length, y / length, z / length
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _matrix_quaternion(r: Mat3) -> tuple[float, float, float, float]:
    """Shepperd's method, canonicalised to ``w >= 0``."""
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (s / 4.0, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2])
        q = ((r[2][1] - r[1][2]) / s, s / 4.0, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s)
    elif r[1][1] > r[2][2]:
        s = 2.0 * math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2])
        q = ((r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, s / 4.0, (r[1][2] + r[2][1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1])
        q = ((r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, s / 4.0)
    return q if q[0] >= 0.0 else (-q[0], -q[1], -q[2], -q[3])


def _gram_schmidt(values: Sequence[float], *, rows: bool) -> Mat3:
    a1, a2 = values[:3], values[3:]
    n1 = math.sqrt(sum(v * v for v in a1))
    if not math.isfinite(n1) or n1 < 1e-9:
        raise ValueError("a 6D rotation needs a non-zero first vector")
    b1 = [v / n1 for v in a1]
    d = sum(p * q for p, q in zip(b1, a2))
    u2 = [q - d * p for p, q in zip(b1, a2)]
    n2 = math.sqrt(sum(v * v for v in u2))
    if n2 < 1e-9:
        raise ValueError("a 6D rotation needs two independent vectors")
    b2 = [v / n2 for v in u2]
    b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
    if rows:
        return (tuple(b1), tuple(b2), tuple(b3))  # type: ignore[return-value]
    return tuple((b1[i], b2[i], b3[i]) for i in range(3))  # type: ignore[return-value]


__all__ = [
    "ARM_EULER",
    "AXIS_ANGLE",
    "QUAT_WXYZ",
    "QUAT_XYZW",
    "ROT6D",
    "ROT6D_ROWS",
    "ROTATION_NONE",
    "from_matrix",
    "rotation_width",
    "to_matrix",
    "uses_angles",
]

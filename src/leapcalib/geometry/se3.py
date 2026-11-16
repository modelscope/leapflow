# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rigid transforms in metres and radians, in pure Python.

LeapFlow already has a ``Transform``, and this is deliberately not it. That one
works in millimetres and degrees because its callers are firmware channels and
vendor manuals; this one works in SI because its callers are residuals and
Jacobians. LeapCalib must not import LeapFlow (the dependency arrow points the
other way, so an external adapter can use these numerics without the robot
stack), so the thirty lines of matrix algebra are duplicated rather than the
dependency being inverted.

Pure Python, no numpy: metre-scale 3x3 work on a handful of transforms is not
where a calibration spends its time, and keeping this import-light means the
contracts and the evidence tooling install with no numerical stack at all.

Two safeguards that matter more than they look:

* A rotation is validated at construction -- orthonormal *and* right-handed.
  A reflection (determinant -1) sneaking in from a badly transcribed matrix
  mirrors the geometry, which fits data about as well as the truth does and puts
  the arm on the wrong side of the target.
* Every transform names its direction. ``T_parent_child`` maps a point in
  ``child`` into ``parent``, and :meth:`Transform.compose` only chains when the
  names line up. The convention is enforced where it is cheap to state, so a
  silently inverted extrinsic becomes a readable error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.units import canonical_number

Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vec3 = tuple[float, float, float]

IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

ORTHONORMAL_TOLERANCE = 1e-6
"""How far ``R @ R.T`` may drift from identity. Accumulated float error, not a free parameter."""


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def mat_vec(a: Mat3, v: Sequence[float]) -> Vec3:
    return (
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    )


def transpose(a: Mat3) -> Mat3:
    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )


def determinant(a: Mat3) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def rotation_from_rpy_rad(rx: float, ry: float, rz: float) -> Mat3:
    """Fixed-axis X-Y-Z angles (radians) as ``Rz @ Ry @ Rx``.

    The same composition order as LeapFlow's ``rpy_to_matrix`` and as the
    ``Rz@Ry@Rx;degrees`` convention a calibration certificate declares, so the
    adapter's degree-to-radian conversion is a unit change and nothing more.
    """
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    return (
        (cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa),
        (sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa),
        (-sb, cb * sa, cb * ca),
    )


def rpy_rad_from_rotation(r: Mat3) -> Vec3:
    """Inverse of :func:`rotation_from_rpy_rad`; at gimbal lock the roll absorbs the yaw."""
    sb = -r[2][0]
    if abs(sb) < 1.0 - 1e-9:
        ry = math.asin(max(-1.0, min(1.0, sb)))
        rx = math.atan2(r[2][1], r[2][2])
        rz = math.atan2(r[1][0], r[0][0])
    else:
        ry = math.copysign(math.pi / 2.0, sb)
        rz = 0.0
        rx = math.atan2(-r[1][2], r[1][1])
    return (rx, ry, rz)


def rotation_from_axis_angle(axis: Sequence[float], angle_rad: float) -> Mat3:
    """Rodrigues' rotation about *axis* (need not be normalised) by *angle_rad*."""
    norm = math.sqrt(sum(float(a) * float(a) for a in axis))
    if norm <= 0:
        if angle_rad:
            raise ValueError("a nonzero rotation needs a nonzero axis")
        return IDENTITY
    x, y, z = (float(a) / norm for a in axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def rotation_angle_rad(r: Mat3) -> float:
    """The geodesic magnitude of rotation *r*, in radians."""
    trace = r[0][0] + r[1][1] + r[2][2]
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


@dataclass(frozen=True)
class Transform:
    """``T_parent_child``: the pose of *child* expressed in *parent*, in metres.

    Applying it maps a point given in the child frame into the parent frame.
    ``parent`` and ``child`` are optional labels; when both transforms in a
    :meth:`compose` carry them, the chain is checked.
    """

    rotation: Mat3 = IDENTITY
    translation_m: Vec3 = (0.0, 0.0, 0.0)
    parent: str = ""
    child: str = ""

    def __post_init__(self) -> None:
        rows = tuple(tuple(canonical_number(v) for v in row) for row in self.rotation)
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("a rotation is a 3x3 matrix")
        object.__setattr__(self, "rotation", rows)
        translation = tuple(canonical_number(v) for v in self.translation_m)
        if len(translation) != 3:
            raise ValueError("a translation has three components")
        object.__setattr__(self, "translation_m", translation)

        product = mat_mul(rows, transpose(rows))  # type: ignore[arg-type]
        drift = max(abs(product[i][j] - IDENTITY[i][j]) for i in range(3) for j in range(3))
        if drift > ORTHONORMAL_TOLERANCE:
            raise ValueError(f"rotation is not orthonormal (drift {drift:.3g})")
        if determinant(rows) < 0:  # type: ignore[arg-type]
            raise ValueError("rotation is a reflection, not a right-handed rotation")

    @classmethod
    def identity(cls, *, parent: str = "", child: str = "") -> "Transform":
        return cls(parent=parent, child=child)

    @classmethod
    def from_rpy_rad(
        cls,
        rx: float,
        ry: float,
        rz: float,
        translation_m: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        parent: str = "",
        child: str = "",
    ) -> "Transform":
        return cls(
            rotation=rotation_from_rpy_rad(rx, ry, rz),
            translation_m=(
                float(translation_m[0]),
                float(translation_m[1]),
                float(translation_m[2]),
            ),
            parent=parent,
            child=child,
        )

    @classmethod
    def from_axis_angle(
        cls,
        axis: Sequence[float],
        angle_rad: float,
        translation_m: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        parent: str = "",
        child: str = "",
    ) -> "Transform":
        return cls(
            rotation=rotation_from_axis_angle(axis, angle_rad),
            translation_m=(
                float(translation_m[0]),
                float(translation_m[1]),
                float(translation_m[2]),
            ),
            parent=parent,
            child=child,
        )

    @classmethod
    def from_matrix(
        cls, rows: Sequence[Sequence[float]], *, parent: str = "", child: str = ""
    ) -> "Transform":
        """Read a 4x4 (or 3x4) homogeneous matrix; the bottom row must be ``0 0 0 1``."""
        if len(rows) not in (3, 4) or any(len(row) != 4 for row in rows[:3]):
            raise ValueError("expected a 3x4 or 4x4 homogeneous matrix")
        if len(rows) == 4:
            bottom = tuple(float(v) for v in rows[3])
            if bottom != (0.0, 0.0, 0.0, 1.0):
                raise ValueError("a rigid transform has bottom row (0, 0, 0, 1)")
        return cls(
            rotation=tuple(tuple(float(v) for v in row[:3]) for row in rows[:3]),  # type: ignore[arg-type]
            translation_m=tuple(float(row[3]) for row in rows[:3]),  # type: ignore[arg-type]
            parent=parent,
            child=child,
        )

    def to_matrix(self) -> list[list[float]]:
        return [[*self.rotation[i], self.translation_m[i]] for i in range(3)] + [
            [0.0, 0.0, 0.0, 1.0]
        ]

    def to_rpy_rad(self) -> Vec3:
        return rpy_rad_from_rotation(self.rotation)

    def compose(self, other: "Transform") -> "Transform":
        """``T_self.parent_other.child``: this transform applied after *other*.

        ``T_base_camera = T_base_flange.compose(T_flange_camera)``. When both
        sides label their frames, a mismatch raises rather than producing a
        transform between two frames that were never adjacent.
        """
        if self.child and other.parent and self.child != other.parent:
            raise ValueError(
                f"cannot chain T_{self.parent}_{self.child} with T_{other.parent}_{other.child}"
            )
        return Transform(
            rotation=mat_mul(self.rotation, other.rotation),
            translation_m=tuple(  # type: ignore[arg-type]
                a + b
                for a, b in zip(mat_vec(self.rotation, other.translation_m), self.translation_m)
            ),
            parent=self.parent,
            child=other.child,
        )

    def inverse(self) -> "Transform":
        """``T_child_parent``: the same relationship read the other way."""
        rotation = transpose(self.rotation)
        moved = mat_vec(rotation, self.translation_m)
        return Transform(
            rotation=rotation,
            translation_m=(-moved[0], -moved[1], -moved[2]),
            parent=self.child,
            child=self.parent,
        )

    def apply(self, point_m: Sequence[float]) -> Vec3:
        """Map a point given in the child frame into the parent frame."""
        rotated = mat_vec(self.rotation, point_m)
        return (
            rotated[0] + self.translation_m[0],
            rotated[1] + self.translation_m[1],
            rotated[2] + self.translation_m[2],
        )

    def rotate(self, vector: Sequence[float]) -> Vec3:
        """Rotate a direction; unlike :meth:`apply` this ignores the translation."""
        return mat_vec(self.rotation, vector)

    def distance_m(self, other: "Transform") -> float:
        """Euclidean distance between the two origins, in metres."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.translation_m, other.translation_m)))

    def angle_to_rad(self, other: "Transform") -> float:
        """Geodesic angle between the two orientations, in radians."""
        return rotation_angle_rad(mat_mul(transpose(self.rotation), other.rotation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotation": [list(row) for row in self.rotation],
            "translation_m": list(self.translation_m),
            "parent": self.parent,
            "child": self.child,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Transform":
        return cls(
            rotation=tuple(tuple(float(v) for v in row) for row in data["rotation"]),  # type: ignore[arg-type]
            translation_m=tuple(float(v) for v in data["translation_m"]),  # type: ignore[arg-type]
            parent=str(data.get("parent") or ""),
            child=str(data.get("child") or ""),
        )


__all__ = [
    "IDENTITY",
    "ORTHONORMAL_TOLERANCE",
    "Mat3",
    "Transform",
    "Vec3",
    "determinant",
    "mat_mul",
    "mat_vec",
    "rotation_angle_rad",
    "rotation_from_axis_angle",
    "rotation_from_rpy_rad",
    "rpy_rad_from_rotation",
    "transpose",
]

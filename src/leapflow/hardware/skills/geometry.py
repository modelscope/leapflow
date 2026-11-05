# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rigid-body geometry for node skills: poses, transforms, workspace boxes.

Pure Python on purpose: the node runtime must import on a board without numpy,
and every quantity a skill reasons about is a 3-vector or a 3x3 rotation. The
calibration fits (``fit_plane``, ``solve_pivot``, ``solve_hand_eye``,
``solve_tool_hand_eye``) are small least-squares problems solved through the
normal equations; the one rotation fit they need is Horn's quaternion method
on a 4x4 Jacobi eigensolver.

Conventions -- one place, so a vendor that disagrees changes exactly one function:

- Lengths are millimetres, angles degrees, at every public boundary.
- ``Pose(x, y, z, rx, ry, rz)`` is the pose of a frame in its parent frame. The
  orientation is fixed-axis X-Y-Z (roll about X, then pitch about Y, then yaw
  about Z, all about the parent's axes): ``R = Rz(rz) @ Ry(ry) @ Rx(rx)``.
- A tool frame's +Z axis points out of the tool. "Pointing down" means the tool
  axis is aligned with the base frame's -Z axis.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")
DOWN: Vec3 = (0.0, 0.0, -1.0)


# ── Scalars and vectors ─────────────────────────────────────────────────────


def wrap_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def wrap_axis_deg(angle: float) -> float:
    """Wrap an undirected axis angle to [-90, 90): an axis and its reverse are one."""
    return (float(angle) + 90.0) % 180.0 - 90.0


def angle_delta_deg(a: float, b: float) -> float:
    """Signed shortest rotation from *b* to *a*, in degrees."""
    return wrap_deg(float(a) - float(b))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def angle_between_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Unsigned angle between two vectors; 0 for a zero-length input."""
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(a, b) / (na * nb)))
    return math.degrees(math.acos(cosine))


# ── Rotations ────────────────────────────────────────────────────────────────


def rpy_to_matrix(rx: float, ry: float, rz: float) -> Mat3:
    """Fixed-axis X-Y-Z angles (deg) to a rotation matrix ``Rz @ Ry @ Rx``."""
    a, b, c = math.radians(rx), math.radians(ry), math.radians(rz)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return (
        (cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa),
        (sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa),
        (-sb, cb * sa, cb * ca),
    )


def matrix_to_rpy(r: Mat3) -> Vec3:
    """Inverse of ``rpy_to_matrix``; at gimbal lock the roll absorbs the yaw."""
    sb = -r[2][0]
    if abs(sb) < 1.0 - 1e-9:
        ry = math.asin(max(-1.0, min(1.0, sb)))
        rx = math.atan2(r[2][1], r[2][2])
        rz = math.atan2(r[1][0], r[0][0])
    else:
        ry = math.copysign(math.pi / 2.0, sb)
        rz = 0.0
        rx = math.atan2(-r[1][2], r[1][1])
    return (wrap_deg(math.degrees(rx)), wrap_deg(math.degrees(ry)), wrap_deg(math.degrees(rz)))


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def mat_vec(a: Mat3, v: Sequence[float]) -> Vec3:
    return (dot(a[0], v), dot(a[1], v), dot(a[2], v))


def transpose(a: Mat3) -> Mat3:
    return ((a[0][0], a[1][0], a[2][0]), (a[0][1], a[1][1], a[2][1]), (a[0][2], a[1][2], a[2][2]))


def rotation_distance_deg(a: Mat3, b: Mat3) -> float:
    """Geodesic angle between two orientations."""
    relative = mat_mul(transpose(a), b)
    trace = relative[0][0] + relative[1][1] + relative[2][2]
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ── Pose and Transform ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pose:
    """A frame's position (mm) and fixed-axis X-Y-Z orientation (deg) in its parent."""

    x: float
    y: float
    z: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    @property
    def position(self) -> Vec3:
        return (self.x, self.y, self.z)

    @property
    def rotation(self) -> Mat3:
        return rpy_to_matrix(self.rx, self.ry, self.rz)

    @property
    def tool_axis(self) -> Vec3:
        """The frame's +Z axis expressed in the parent frame."""
        r = self.rotation
        return (r[0][2], r[1][2], r[2][2])

    def as_tuple(self) -> tuple[float, ...]:
        return (self.x, self.y, self.z, self.rx, self.ry, self.rz)

    def to_dict(self, digits: int = 3) -> dict[str, float]:
        return {axis: round(value, digits) for axis, value in zip(POSE_AXES, self.as_tuple())}

    def translated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Pose":
        return Pose(self.x + dx, self.y + dy, self.z + dz, self.rx, self.ry, self.rz)

    def distance_mm(self, other: "Pose") -> float:
        return norm(sub(self.position, other.position))

    def rotation_to_deg(self, other: "Pose") -> float:
        return rotation_distance_deg(self.rotation, other.rotation)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Pose":
        """Parse ``{x, y, z, rx?, ry?, rz?}``; raises ``ValueError`` on a bad value."""
        values: list[float] = []
        for axis in POSE_AXES:
            raw = data.get(axis, 0.0 if axis.startswith("r") else None)
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"pose needs a numeric {axis!r}")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"pose {axis!r} must be finite")
            values.append(number)
        return cls(*values)


@dataclass(frozen=True)
class Transform:
    """A rigid transform ``T_parent_child``: maps child coordinates into the parent."""

    rotation: Mat3 = IDENTITY
    translation: Vec3 = (0.0, 0.0, 0.0)

    @classmethod
    def from_pose(cls, pose: Pose) -> "Transform":
        return cls(pose.rotation, pose.position)

    @classmethod
    def from_matrix(cls, rows: Sequence[Sequence[float]]) -> "Transform":
        """Parse a 4x4 (or 3x4) homogeneous matrix in millimetres."""
        if len(rows) < 3 or any(len(row) < 4 for row in rows[:3]):
            raise ValueError("a transform matrix needs at least 3 rows of 4 values")
        rotation = tuple(tuple(float(rows[i][j]) for j in range(3)) for i in range(3))
        translation = (float(rows[0][3]), float(rows[1][3]), float(rows[2][3]))
        return cls(rotation, translation)  # type: ignore[arg-type]

    def to_matrix(self) -> list[list[float]]:
        return [[*self.rotation[i], self.translation[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]

    def to_pose(self) -> Pose:
        rx, ry, rz = matrix_to_rpy(self.rotation)
        return Pose(*self.translation, rx, ry, rz)

    def compose(self, other: "Transform") -> "Transform":
        """``self @ other``: ``T_a_b.compose(T_b_c) == T_a_c``."""
        rotation = mat_mul(self.rotation, other.rotation)
        translation = add(mat_vec(self.rotation, other.translation), self.translation)
        return Transform(rotation, translation)

    def inverse(self) -> "Transform":
        rt = transpose(self.rotation)
        t = mat_vec(rt, self.translation)
        return Transform(rt, (-t[0], -t[1], -t[2]))

    def apply(self, point: Sequence[float]) -> Vec3:
        return add(mat_vec(self.rotation, point), self.translation)


# ── Workspace ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkspaceBox:
    """Axis-aligned safe box for the tool centre point, in the arm base frame (mm)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        for low, high, axis in self._bounds():
            if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
                raise ValueError(f"workspace {axis} bounds must be finite with min < max")

    def _bounds(self) -> tuple[tuple[float, float, str], ...]:
        return (
            (self.x_min, self.x_max, "x"),
            (self.y_min, self.y_max, "y"),
            (self.z_min, self.z_max, "z"),
        )

    def violation(self, point: Sequence[float], *, margin: float = 0.0) -> str | None:
        """Describe the first axis *point* leaves the box on, or None when inside."""
        for value, (low, high, axis) in zip(point, self._bounds()):
            if value < low + margin or value > high - margin:
                return f"{axis}={value:.1f} mm is outside [{low:.1f}, {high:.1f}]"
        return None

    def contains(self, point: Sequence[float], *, margin: float = 0.0) -> bool:
        return self.violation(point, margin=margin) is None

    def corners(self) -> tuple[Vec3, ...]:
        return tuple(
            (x, y, z)
            for x in (self.x_min, self.x_max)
            for y in (self.y_min, self.y_max)
            for z in (self.z_min, self.z_max)
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {axis: [low, high] for low, high, axis in self._bounds()}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkspaceBox":
        """Accept ``{x: [lo, hi], ...}`` or ``{x_min, x_max, ...}``."""
        bounds: list[float] = []
        for axis in ("x", "y", "z"):
            pair = data.get(axis)
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                bounds.extend((float(pair[0]), float(pair[1])))
            else:
                bounds.extend((float(data[f"{axis}_min"]), float(data[f"{axis}_max"])))
        return cls(*bounds)


def tool_cone_deg(pose: Pose) -> float:
    """Angle between the tool axis and straight down."""
    return angle_between_deg(pose.tool_axis, DOWN)


# ── Least squares ───────────────────────────────────────────────────────────


def solve_linear(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve the square system ``a @ x == b`` (Gaussian elimination, partial pivoting).

    Raises ``ValueError`` when the system is singular or numerically so.
    """
    n = len(b)
    rows = [[float(value) for value in a[i]] + [float(b[i])] for i in range(n)]
    scale = max((abs(value) for row in rows for value in row[:n]), default=0.0)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= 1e-12 * scale or scale == 0.0:
            raise ValueError("the system is singular")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (rows[i][n] - sum(rows[i][j] * x[j] for j in range(i + 1, n))) / rows[i][i]
    return x


def least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    """``argmin |A x - b|`` through the normal equations (small, well-scaled systems)."""
    if not rows or len(rows) != len(values):
        raise ValueError("least squares needs one value per row")
    n = len(rows[0])
    if len(rows) < n:
        raise ValueError(f"least squares needs at least {n} rows")
    ata = [[sum(row[i] * row[j] for row in rows) for j in range(n)] for i in range(n)]
    atb = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(n)]
    return solve_linear(ata, atb)


def fit_plane(points: Iterable[Sequence[float]]) -> tuple[Vec3, float, float]:
    """Least-squares plane through *points*: ``(unit normal, offset, rms residual)``.

    The plane is ``dot(normal, p) == offset`` with the normal oriented toward +Z,
    so a table's normal points up. Fitted as ``z = a x + b y + c``: the plane must
    not be vertical, and collinear points raise ``ValueError``.
    """
    data = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    if len(data) < 3:
        raise ValueError("a plane fit needs at least three points")
    a, b, c = least_squares([(x, y, 1.0) for x, y, _ in data], [z for _, _, z in data])
    length = math.sqrt(a * a + b * b + 1.0)
    normal: Vec3 = (-a / length, -b / length, 1.0 / length)
    offset = c / length
    residual = math.sqrt(sum((dot(normal, p) - offset) ** 2 for p in data) / len(data))
    return normal, offset, residual


def plane_height(normal: Sequence[float], offset: float, x: float, y: float) -> float:
    """The plane's z at ``(x, y)``; the plane must not be vertical."""
    return (offset - normal[0] * x - normal[1] * y) / normal[2]


_MIN_PLANE_NZ = 0.5  # a support surface tilted more than 60 deg is a wall, not a table


@dataclass(frozen=True)
class Plane:
    """A support surface ``dot(normal, p) == offset`` in the arm base frame (mm).

    The normal is a unit vector toward +Z, so ``above`` is positive over the
    surface. Steeper than 60 degrees is refused: nothing rests on a wall.
    """

    normal: Vec3
    offset: float

    def __post_init__(self) -> None:
        length = norm(self.normal)
        if not (math.isfinite(length) and math.isfinite(self.offset)):
            raise ValueError("a plane needs a finite normal and offset")
        if abs(length - 1.0) > 1e-3:
            raise ValueError("a plane normal must be a unit vector")
        if self.normal[2] < _MIN_PLANE_NZ:
            raise ValueError("a support plane must face up (tilt <= 60 deg)")

    @classmethod
    def horizontal(cls, z: float) -> "Plane":
        return cls((0.0, 0.0, 1.0), float(z))

    def height(self, x: float, y: float) -> float:
        """The surface's z at ``(x, y)``."""
        return plane_height(self.normal, self.offset, x, y)

    def above(self, point: Sequence[float]) -> float:
        """Signed distance of *point* from the surface, positive over it."""
        return dot(self.normal, point) - self.offset

    def to_dict(self) -> dict[str, Any]:
        return {"normal": [round(v, 6) for v in self.normal], "offset_mm": round(self.offset, 3)}


@dataclass(frozen=True)
class Footprint:
    """The oriented extent of a point set seen from above.

    ``yaw_deg`` is the major axis in the base frame, wrapped to [-90, 90);
    ``length`` is the extent along it and ``width`` the extent across it.
    """

    center: tuple[float, float]
    yaw_deg: float
    length: float
    width: float

    @property
    def minor_yaw_deg(self) -> float:
        return wrap_axis_deg(self.yaw_deg + 90.0)


def footprint(points: Iterable[Sequence[float]]) -> Footprint:
    """Principal-axis footprint of *points* projected on the XY plane.

    The axes come from the closed-form eigenvectors of the 2x2 covariance; the
    extents and the centre from the points' spread along them, so the centre is
    the middle of the oriented box rather than the (density-biased) mean.
    """
    data = [(float(p[0]), float(p[1])) for p in points]
    if len(data) < 3:
        raise ValueError("a footprint needs at least three points")
    mx = sum(x for x, _ in data) / len(data)
    my = sum(y for _, y in data) / len(data)
    sxx = sum((x - mx) ** 2 for x, _ in data)
    syy = sum((y - my) ** 2 for _, y in data)
    sxy = sum((x - mx) * (y - my) for x, y in data)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    c, s = math.cos(theta), math.sin(theta)
    along = [(x - mx) * c + (y - my) * s for x, y in data]
    across = [-(x - mx) * s + (y - my) * c for x, y in data]
    a_mid = (min(along) + max(along)) / 2.0
    b_mid = (min(across) + max(across)) / 2.0
    return Footprint(
        center=(mx + a_mid * c - b_mid * s, my + a_mid * s + b_mid * c),
        yaw_deg=wrap_axis_deg(math.degrees(theta)),
        length=max(along) - min(along),
        width=max(across) - min(across),
    )


def solve_pivot(flanges: Sequence[Pose]) -> tuple[Vec3, Vec3, float]:
    """Pivot calibration: ``(tip in the flange frame, pivot in the base, rms residual)``.

    The tool tip rests on one fixed point while the flange takes several poses, so
    every pose ``(R_i, p_i)`` satisfies ``R_i t + p_i = P``. Orientations that all
    share one rotation axis leave ``t`` unobservable along it and raise ``ValueError``.
    """
    if len(flanges) < 3:
        raise ValueError("a pivot fit needs at least three poses")
    rows: list[tuple[float, ...]] = []
    values: list[float] = []
    for pose in flanges:
        rotation, position = pose.rotation, pose.position
        for k in range(3):
            rows.append((*rotation[k], *(-1.0 if j == k else 0.0 for j in range(3))))
            values.append(-position[k])
    x = least_squares(rows, values)
    tip: Vec3 = (x[0], x[1], x[2])
    pivot: Vec3 = (x[3], x[4], x[5])
    errors = [norm(sub(Transform.from_pose(pose).apply(tip), pivot)) for pose in flanges]
    return tip, pivot, math.sqrt(sum(e * e for e in errors) / len(errors))


# ── Rotation fits ─────────────────────────────────────────────────────────────


def rotation_vector(r: Mat3) -> Vec3:
    """The axis-angle vector of a rotation (radians): axis times angle, the log map."""
    trace = r[0][0] + r[1][1] + r[2][2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    skew = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
    if angle < 1e-9:
        return (skew[0] / 2.0, skew[1] / 2.0, skew[2] / 2.0)
    if math.pi - angle > 1e-4:
        k = angle / (2.0 * math.sin(angle))
        return (k * skew[0], k * skew[1], k * skew[2])
    # Near a half turn the skew part vanishes: read the axis from R + I = 2 a a^T.
    i = max(range(3), key=lambda j: r[j][j])
    column = [(r[j][i] + (1.0 if j == i else 0.0)) / 2.0 for j in range(3)]
    length = math.sqrt(column[i])
    axis = [value / length for value in column]
    if dot(axis, skew) < 0.0:
        axis = [-value for value in axis]
    return (axis[0] * angle, axis[1] * angle, axis[2] * angle)


def _symmetric_eigen(matrix: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors (columns) of a small symmetric matrix (cyclic Jacobi)."""
    n = len(matrix)
    a = [[float(value) for value in row] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = sum(value * value for row in a for value in row) or 1.0
    for _ in range(100):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off <= 1e-24 * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                t = -t if theta < 0.0 else t
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    a[k][p], a[k][q] = c * a[k][p] - s * a[k][q], s * a[k][p] + c * a[k][q]
                for k in range(n):
                    a[p][k], a[q][k] = c * a[p][k] - s * a[q][k], s * a[p][k] + c * a[q][k]
                for k in range(n):
                    v[k][p], v[k][q] = c * v[k][p] - s * v[k][q], s * v[k][p] + c * v[k][q]
    return [a[i][i] for i in range(n)], v


def nearest_rotation(m: Sequence[Sequence[float]]) -> Mat3:
    """The rotation ``R`` maximising ``trace(R^T m)`` -- the closest rotation to *m*.

    For vector pairs ``a_i ~ R b_i``, ``m = sum(a_i b_i^T)`` gives the
    least-squares ``R``; for a sum of rotations it gives their chordal mean.
    Solved with Horn's quaternion method, so the result is always proper.
    """
    # Horn's S_jk = sum(b_j a_k) is m transposed.
    sxx, sxy, sxz = m[0][0], m[1][0], m[2][0]
    syx, syy, syz = m[0][1], m[1][1], m[2][1]
    szx, szy, szz = m[0][2], m[1][2], m[2][2]
    n = (
        (sxx + syy + szz, syz - szy, szx - sxz, sxy - syx),
        (syz - szy, sxx - syy - szz, sxy + syx, szx + sxz),
        (szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy),
        (sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz),
    )
    values, vectors = _symmetric_eigen(n)
    best = max(range(4), key=lambda i: values[i])
    w, x, y, z = (vectors[k][best] for k in range(4))
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if length == 0.0:
        raise ValueError("no rotation fits a zero matrix")
    w, x, y, z = w / length, x / length, y / length, z / length
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _outer(a: Sequence[float], b: Sequence[float]) -> list[list[float]]:
    return [[a[i] * b[j] for j in range(3)] for i in range(3)]


def _accumulate(total: list[list[float]], part: Sequence[Sequence[float]]) -> None:
    for i in range(3):
        for j in range(3):
            total[i][j] += part[i][j]


@dataclass(frozen=True)
class HandEyeFit:
    """The solution of ``A_i X = Z B_i`` and how well every sample agrees with it.

    Residuals compare the two chains ``A_i X`` and ``Z B_i`` per sample:
    translation in mm, rotation as a geodesic angle in degrees.
    """

    x: Transform
    z: Transform
    samples: int
    translation_rms_mm: float
    translation_max_mm: float
    rotation_rms_deg: float
    rotation_max_deg: float

    def to_dict(self) -> dict[str, float]:
        return {
            "samples": self.samples,
            "translation_rms_mm": round(self.translation_rms_mm, 3),
            "translation_max_mm": round(self.translation_max_mm, 3),
            "rotation_rms_deg": round(self.rotation_rms_deg, 4),
            "rotation_max_deg": round(self.rotation_max_deg, 4),
        }


MIN_HAND_EYE_SAMPLES = 3
_MIN_MOTION_RAD = math.radians(2.0)
_MIN_AXIS_SPREAD_DEG = 10.0


def solve_hand_eye(a: Sequence[Transform], b: Sequence[Transform]) -> HandEyeFit:
    """Robot-world/hand-eye calibration: ``X`` and ``Z`` with ``A_i X = Z B_i``.

    Eye-to-hand: ``A_i = T_base_flange``, ``B_i = T_camera_target``, giving
    ``X = T_flange_target`` and ``Z = T_base_camera``. Eye-in-hand: ``B_i`` is
    ``T_camera_target`` inverted, giving ``X = T_flange_camera`` and
    ``Z = T_base_target``.

    Every sample pair yields ``A' X = X B'`` with ``A' = A_j^-1 A_i`` and
    ``B' = B_j^-1 B_i``; the rotation is the Park-Martin fit of the motions'
    rotation vectors, the translation a linear least-squares solve, and ``Z`` the
    mean of ``A_i X B_i^-1``. Motions that all turn about one axis leave ``X``
    unobservable and raise ``ValueError``.
    """
    if len(a) != len(b):
        raise ValueError("hand-eye calibration needs one camera sample per arm pose")
    if len(a) < MIN_HAND_EYE_SAMPLES:
        raise ValueError(f"hand-eye calibration needs at least {MIN_HAND_EYE_SAMPLES} samples")
    motions = [
        (a[j].inverse().compose(a[i]), b[j].inverse().compose(b[i]))
        for i in range(len(a))
        for j in range(i)
    ]
    rotation = _hand_eye_rotation(motions)
    rows: list[tuple[float, ...]] = []
    values: list[float] = []
    for ma, mb in motions:
        rhs = sub(mat_vec(rotation, mb.translation), ma.translation)
        for k in range(3):
            rows.append(tuple(ma.rotation[k][j] - (1.0 if j == k else 0.0) for j in range(3)))
            values.append(rhs[k])
    x = Transform(rotation, tuple(least_squares(rows, values)))  # type: ignore[arg-type]
    worlds = [ai.compose(x).compose(bi.inverse()) for ai, bi in zip(a, b)]
    total = [[0.0] * 3 for _ in range(3)]
    for world in worlds:
        _accumulate(total, world.rotation)
    mean = tuple(sum(w.translation[k] for w in worlds) / len(worlds) for k in range(3))
    z = Transform(nearest_rotation(total), mean)  # type: ignore[arg-type]
    return _hand_eye_fit(a, b, x, z)


def _hand_eye_rotation(motions: Sequence[tuple[Transform, Transform]]) -> Mat3:
    total = [[0.0] * 3 for _ in range(3)]
    axes: list[Vec3] = []
    for ma, mb in motions:
        alpha, beta = rotation_vector(ma.rotation), rotation_vector(mb.rotation)
        if norm(alpha) >= _MIN_MOTION_RAD:
            axes.append(alpha)
        _accumulate(total, _outer(alpha, beta))
    if not _spans_two_axes(axes):
        raise ValueError(
            "the arm poses turn about a single axis; tilt about at least two different axes"
        )
    return nearest_rotation(total)


def _spans_two_axes(axes: Sequence[Vec3]) -> bool:
    for p, q in itertools.combinations(axes, 2):
        angle = angle_between_deg(p, q)
        if min(angle, 180.0 - angle) >= _MIN_AXIS_SPREAD_DEG:
            return True
    return False


def _hand_eye_fit(
    a: Sequence[Transform], b: Sequence[Transform], x: Transform, z: Transform
) -> HandEyeFit:
    moved: list[float] = []
    turned: list[float] = []
    for ai, bi in zip(a, b):
        arm, camera = ai.compose(x), z.compose(bi)
        moved.append(norm(sub(arm.translation, camera.translation)))
        turned.append(rotation_distance_deg(arm.rotation, camera.rotation))
    return HandEyeFit(
        x=x,
        z=z,
        samples=len(a),
        translation_rms_mm=math.sqrt(sum(e * e for e in moved) / len(moved)),
        translation_max_mm=max(moved),
        rotation_rms_deg=math.sqrt(sum(e * e for e in turned) / len(turned)),
        rotation_max_deg=max(turned),
    )


# ── Tool-point hand-eye ──────────────────────────────────────────────────────

MIN_TOOL_HAND_EYE_SAMPLES = 6
_TOOL_FIT_SAMPLES = 5  # one sample may be held out of the fit for a leave-one-out check
_HUBER_K = 1.345  # 95% efficiency when the residuals are really Gaussian
_HUBER_FLOOR_MM = 0.5  # below the camera's noise floor a residual says nothing
_TOOL_ALTERNATING_STEPS = 8  # closed-form sweeps before the refinement takes over
_TOOL_REFINE_STEPS = 60
_TOOL_HAND_EYE_TOLERANCE_MM = 1e-4
# Where the observed point may sit in the flange frame, as alternating-fit seeds.
# The tool axis is the flange's +Z, so the seeds walk that line through the range
# a gripper can reach; the fit keeps whichever start explains the samples best.
_TOOL_SEEDS: tuple[Vec3, ...] = ((0.0, 0.0, 60.0), (0.0, 0.0, 120.0), (0.0, 0.0, -60.0))


@dataclass(frozen=True)
class ToolHandEyeFit:
    """The solution of ``p_cam_i = R_X (R_i t + p_i) + t_X``, and how well it holds.

    ``mount`` is ``T_camera_base`` (mm) and ``tool`` the observed point in the
    flange frame. ``residual`` is per-sample distance in mm and ``held_out`` the
    same distance predicted by a fit that never saw that sample -- the number
    that says whether the mount is measured or merely fitted. The sigmas are the
    parameter standard errors the normal equations imply at the fitted noise
    level, so a thin pose set shows up as a large sigma even when the residuals
    are small.
    """

    mount: Transform
    tool: Vec3
    samples: int
    residual_rms_mm: float
    residual_max_mm: float
    held_out_rms_mm: float
    held_out_max_mm: float
    condition_number: float
    translation_sigma_mm: float
    rotation_sigma_deg: float
    tool_sigma_mm: float
    iterations: int
    outliers: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "samples": self.samples,
            "residual_rms_mm": round(self.residual_rms_mm, 3),
            "residual_max_mm": round(self.residual_max_mm, 3),
            "held_out_rms_mm": round(self.held_out_rms_mm, 3),
            "held_out_max_mm": round(self.held_out_max_mm, 3),
            "condition_number": round(self.condition_number, 3),
            "translation_sigma_mm": round(self.translation_sigma_mm, 3),
            "rotation_sigma_deg": round(self.rotation_sigma_deg, 4),
            "tool_sigma_mm": round(self.tool_sigma_mm, 3),
            "iterations": self.iterations,
            "outliers": self.outliers,
        }


def solve_tool_hand_eye(flanges: Sequence[Pose], points: Sequence[Vec3]) -> ToolHandEyeFit:
    """Camera mount and tool point from one flange-fixed point seen at every pose.

    A fixed camera sees a point rigidly attached to the flange (the closed
    gripper's tip) at poses ``(R_i, p_i)``. The point sits at ``R_i t + p_i`` in
    the base frame and at ``p_cam_i`` on the depth image, so

        ``p_cam_i = R_X (R_i t + p_i) + t_X``

    with nine unknowns: the camera mount ``X = (R_X, t_X)`` in the base frame and
    the tool point ``t`` in the flange frame. No calibration target, and no
    separate TCP measurement, is needed.

    Solved in two stages: alternating closed forms (weighted Kabsch for ``X``,
    linear least squares for ``t``) bring the estimate into the basin, then a
    damped Gauss-Newton refinement on all nine parameters converges quadratically.
    Huber weights bound the influence of a dropped or mis-segmented observation.
    Poses that turn about a single axis leave ``t`` and the mount's heading
    entangled, so they raise ``ValueError``.
    """
    if len(flanges) != len(points):
        raise ValueError("tool hand-eye needs one camera point per arm pose")
    if len(flanges) < MIN_TOOL_HAND_EYE_SAMPLES:
        raise ValueError(
            f"tool hand-eye needs at least {MIN_TOOL_HAND_EYE_SAMPLES} poses "
            f"(got {len(flanges)})"
        )
    _check_rotation_spread(flanges)
    mount, tool, _, iterations = _best_fit(flanges, points)
    residuals = _tool_residuals(flanges, points, mount, tool)
    held_out = _held_out_errors(flanges, points)
    sigmas = _parameter_sigmas(flanges, points, mount, tool, residuals)
    return ToolHandEyeFit(
        mount=mount,
        tool=tool,
        samples=len(flanges),
        residual_rms_mm=_rms(residuals),
        residual_max_mm=max(residuals),
        held_out_rms_mm=_rms(held_out),
        held_out_max_mm=max(held_out),
        condition_number=sigmas[3],
        translation_sigma_mm=sigmas[0],
        rotation_sigma_deg=sigmas[1],
        tool_sigma_mm=sigmas[2],
        iterations=iterations,
        outliers=sum(1 for weight in _huber_weights(residuals) if weight < 1.0),
    )


def _mean_tool_point(flanges: Sequence[Pose], points: Sequence[Vec3]) -> Vec3:
    """A first guess at ``t``: the average of the camera points carried into the flange."""
    guesses = [mat_vec(transpose(pose.rotation), point) for pose, point in zip(flanges, points)]
    return (sum(v[0] for v in guesses) / len(guesses),
            sum(v[1] for v in guesses) / len(guesses),
            sum(v[2] for v in guesses) / len(guesses))


def _check_rotation_spread(flanges: Sequence[Pose]) -> None:
    """Refuse poses that turn about a single axis: ``t`` is unobservable along it."""
    axes: list[Vec3] = []
    for i in range(len(flanges)):
        for j in range(i):
            relative = mat_mul(transpose(flanges[j].rotation), flanges[i].rotation)
            axis = rotation_vector(relative)
            if norm(axis) >= _MIN_MOTION_RAD:
                axes.append(axis)
    if not _spans_two_axes(axes):
        raise ValueError(
            "the arm poses turn about a single axis; tilt about at least two different axes"
        )


def _fit(
    flanges: Sequence[Pose], points: Sequence[Vec3], seed: Vec3
) -> tuple[Transform, Vec3, float, int]:
    """``(mount, tool, robust cost, refinement steps)`` from one alternating seed."""
    mount, tool = _initial(flanges, points, seed)
    return _refine(flanges, points, mount, tool)


def _initial(
    flanges: Sequence[Pose], points: Sequence[Vec3], seed: Vec3
) -> tuple[Transform, Vec3]:
    """Alternate ``X`` (Kabsch) and ``t`` (linear) a few times to reach the basin."""
    tool, weights = seed, [1.0] * len(flanges)
    mount = Transform()
    for _ in range(_TOOL_ALTERNATING_STEPS):
        mount = _fit_mount(flanges, points, tool, weights)
        weights = _huber_weights(_tool_residuals(flanges, points, mount, tool))
        tool = _fit_tool(flanges, points, mount, weights)
    return _fit_mount(flanges, points, tool, weights), tool


def _refine(
    flanges: Sequence[Pose], points: Sequence[Vec3], mount: Transform, tool: Vec3
) -> tuple[Transform, Vec3, float, int]:
    """Damped Gauss-Newton over all nine parameters.

    Alternating two exact sub-solves converges only linearly -- the tool point
    sits on a 150 mm lever, so the two blocks trade the same error back and
    forth. The joint step uses the same closed-form Jacobian the uncertainty
    report needs, and reaches the solution quadratically.
    """
    cost = float("inf")
    damping = 1e-3
    steps = 0
    for steps in range(1, _TOOL_REFINE_STEPS + 1):
        residuals = _tool_residuals(flanges, points, mount, tool)
        current = _robust_cost(residuals)
        normal, gradient = _normal_equations(flanges, points, mount, tool, residuals)
        while damping < 1e9:
            damped = [
                [normal[i][j] + (damping * normal[i][i] if i == j else 0.0) for j in range(9)]
                for i in range(9)
            ]
            delta = solve_linear(damped, [-value for value in gradient])
            candidate = _stepped(mount, tool, delta)
            trial = _robust_cost(_tool_residuals(flanges, points, *candidate))
            if trial < current:
                mount, tool, cost = candidate[0], candidate[1], trial
                damping = max(damping / 3.0, 1e-9)
                break
            damping *= 4.0
        else:
            break  # no descent direction left: the fit is as good as these samples allow
        if norm(delta[:3]) < _TOOL_HAND_EYE_TOLERANCE_MM:
            if norm(delta[3:6]) < _TOOL_HAND_EYE_TOLERANCE_MM and norm(delta[6:]) < 1e-6:
                break
    return mount, tool, cost, steps


def _stepped(mount: Transform, tool: Vec3, delta: Sequence[float]) -> tuple[Transform, Vec3]:
    """Apply one parameter step: the tool point, the mount translation, its rotation."""
    moved_tool = (tool[0] + delta[0], tool[1] + delta[1], tool[2] + delta[2])
    moved_translation = (
        mount.translation[0] + delta[3],
        mount.translation[1] + delta[4],
        mount.translation[2] + delta[5],
    )
    rotation = mat_mul(mount.rotation, _exp_so3(delta[6:]))
    return Transform(rotation, moved_translation), moved_tool


def _exp_so3(vector: Sequence[float]) -> Mat3:
    """Rodrigues: a rotation vector (radians) to a rotation matrix."""
    angle = norm(vector)
    if angle < 1e-12:
        return IDENTITY
    axis = (vector[0] / angle, vector[1] / angle, vector[2] / angle)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    k = 1.0 - c
    return (
        (c + x * x * k, x * y * k - z * s, x * z * k + y * s),
        (y * x * k + z * s, c + y * y * k, y * z * k - x * s),
        (z * x * k - y * s, z * y * k + x * s, c + z * z * k),
    )


def _normal_equations(
    flanges: Sequence[Pose],
    points: Sequence[Vec3],
    mount: Transform,
    tool: Vec3,
    residuals: Sequence[float],
) -> tuple[list[list[float]], list[float]]:
    """``(J^T W J, J^T W r)`` for the nine parameters, with Huber weights."""
    weights = _huber_weights(residuals)
    normal = [[0.0] * 9 for _ in range(9)]
    gradient = [0.0] * 9
    for pose, point, weight in zip(flanges, points, weights):
        base = add(mat_vec(pose.rotation, tool), pose.position)
        residual = sub(mount.apply(base), point)
        rows = _jacobian_rows(mount, pose, base)
        for k in range(3):
            row = rows[k]
            for i in range(9):
                gradient[i] += weight * row[i] * residual[k]
                for j in range(9):
                    normal[i][j] += weight * row[i] * row[j]
    return normal, gradient


def _jacobian_rows(mount: Transform, pose: Pose, base: Vec3) -> list[list[float]]:
    """``d p_cam / d(t, t_X, w)`` for one sample: the tool point, the mount, its turn."""
    tool_block = mat_mul(mount.rotation, pose.rotation)
    skew = _skew_rotated(mount.rotation, base)
    rows = []
    for k in range(3):
        row = [tool_block[k][0], tool_block[k][1], tool_block[k][2], 0.0, 0.0, 0.0,
               -skew[k][0], -skew[k][1], -skew[k][2]]
        row[3 + k] = 1.0
        rows.append(row)
    return rows


def _huber_weights(residuals: Sequence[float]) -> list[float]:
    """Weights that cap one bad sample's pull at the robust noise scale."""
    ordered = sorted(residuals)
    scale = max(ordered[len(ordered) // 2] / 0.6745, _HUBER_FLOOR_MM)
    limit = _HUBER_K * scale
    return [1.0 if r <= limit else limit / r for r in residuals]


def _robust_cost(residuals: Sequence[float]) -> float:
    """The Huber loss the solver minimises, so one outlier cannot buy a better score.

    Seeding and step acceptance compare this, not the plain sum of squares: with
    a plain cost a 50 mm bad sample is worth trading 6 mm of every other sample
    for, which is how a mis-segmented frame drags a calibration off.
    """
    ordered = sorted(residuals)
    scale = max(ordered[len(ordered) // 2] / 0.6745, _HUBER_FLOOR_MM)
    limit = _HUBER_K * scale
    return sum(r * r if r <= limit else limit * (2.0 * r - limit) for r in residuals)


def _fit_mount(
    flanges: Sequence[Pose], points: Sequence[Vec3], tool: Vec3, weights: Sequence[float]
) -> Transform:
    """The ``T_camera_base`` that best maps ``R_i t + p_i`` onto ``p_cam_i``."""
    base = [add(mat_vec(pose.rotation, tool), pose.position) for pose in flanges]
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("every sample was down-weighted; the observations disagree")
    centre_base = tuple(sum(w * b[k] for w, b in zip(weights, base)) / total for k in range(3))
    centre_cam = tuple(sum(w * p[k] for w, p in zip(weights, points)) / total for k in range(3))
    total_matrix = [[0.0] * 3 for _ in range(3)]
    for weight, b, p in zip(weights, base, points):
        part = _outer(sub(p, centre_cam), sub(b, centre_base))
        for i in range(3):
            for j in range(3):
                total_matrix[i][j] += weight * part[i][j]
    rotation = nearest_rotation(total_matrix)
    shifted = sub(centre_cam, mat_vec(rotation, centre_base))
    return Transform(rotation, shifted)


def _fit_tool(
    flanges: Sequence[Pose], points: Sequence[Vec3], mount: Transform, weights: Sequence[float]
) -> Vec3:
    """The ``t`` that best explains the samples for a given mount (linear)."""
    rows: list[tuple[float, ...]] = []
    values: list[float] = []
    for pose, point, weight in zip(flanges, points, weights):
        root = math.sqrt(weight)
        matrix = mat_mul(mount.rotation, pose.rotation)
        rhs = sub(sub(point, mount.translation), mat_vec(mount.rotation, pose.position))
        for k in range(3):
            rows.append(tuple(root * matrix[k][j] for j in range(3)))
            values.append(root * rhs[k])
    solved = least_squares(rows, values)
    return (solved[0], solved[1], solved[2])


def _tool_residuals(
    flanges: Sequence[Pose], points: Sequence[Vec3], mount: Transform, tool: Vec3
) -> list[float]:
    return [
        norm(sub(mount.apply(add(mat_vec(pose.rotation, tool), pose.position)), point))
        for pose, point in zip(flanges, points)
    ]


def _held_out_errors(flanges: Sequence[Pose], points: Sequence[Vec3]) -> list[float]:
    """Per sample: the distance a fit that never saw it predicts for it."""
    errors: list[float] = []
    for index in range(len(flanges)):
        rest = [i for i in range(len(flanges)) if i != index]
        if len(rest) < _TOOL_FIT_SAMPLES:
            raise ValueError(
                f"a leave-one-out check needs more than {_TOOL_FIT_SAMPLES} poses"
            )
        best = _best_fit([flanges[i] for i in rest], [points[i] for i in rest])
        predicted = best[0].apply(
            add(mat_vec(flanges[index].rotation, best[1]), flanges[index].position)
        )
        errors.append(norm(sub(predicted, points[index])))
    return errors


def _best_fit(
    flanges: Sequence[Pose], points: Sequence[Vec3]
) -> tuple[Transform, Vec3, float, int]:
    """The best of the alternating seeds for this sample set."""
    best: tuple[float, Transform, Vec3, int] | None = None
    for seed in (_mean_tool_point(flanges, points), *_TOOL_SEEDS):
        mount, tool, rms, iterations = _fit(flanges, points, seed)
        if best is None or rms < best[0]:
            best = (rms, mount, tool, iterations)
    assert best is not None
    return best[1], best[2], best[0], best[3]


def _parameter_sigmas(
    flanges: Sequence[Pose],
    points: Sequence[Vec3],
    mount: Transform,
    tool: Vec3,
    residuals: Sequence[float],
) -> tuple[float, float, float, float]:
    """``(position sigma mm, rotation sigma deg, tool sigma mm, condition number)``.

    The Jacobian of ``p_cam_i`` with respect to the tool point, the mount's
    translation and the mount's rotation vector is exact in closed form, so the
    normal equations give the parameter standard errors at the fitted noise
    level. A pose set that barely excites a direction makes them large, which is
    the honest way to report "measured, but not well".
    """
    rows: list[list[float]] = []
    for pose in flanges:
        base = add(mat_vec(pose.rotation, tool), pose.position)
        rows.extend(_jacobian_rows(mount, pose, base))
    normal = [[sum(row[i] * row[j] for row in rows) for j in range(9)] for i in range(9)]
    length = [math.sqrt(max(normal[i][i], 1e-30)) for i in range(9)]
    equilibrated = [[normal[i][j] / (length[i] * length[j]) for j in range(9)] for i in range(9)]
    scaled_values, _ = _symmetric_eigen(equilibrated)
    smallest, largest = min(scaled_values), max(scaled_values)
    condition = math.sqrt(largest / smallest) if smallest > 0.0 else math.inf
    degrees = 3 * len(flanges) - 9
    variance = (sum(r * r for r in residuals) / degrees) if degrees > 0 else 0.0
    inverse = _invert(normal)
    position = math.sqrt(sum(inverse[3 + k][3 + k] for k in range(3)) * variance)
    tool_error = math.sqrt(sum(inverse[k][k] for k in range(3)) * variance)
    angles = [[inverse[6 + i][6 + j] for j in range(3)] for i in range(3)]
    angle_values, _ = _symmetric_eigen(angles)
    rotation = math.degrees(math.sqrt(max(max(angle_values), 0.0) * variance))
    return position, rotation, tool_error, condition


def _skew_rotated(rotation: Mat3, base: Vec3) -> Mat3:
    """``R [base]x``: the derivative of ``R exp([w]x) base`` with respect to ``-w``."""
    b1, b2, b3 = base
    skew = ((0.0, -b3, b2), (b3, 0.0, -b1), (-b2, b1, 0.0))
    return mat_mul(rotation, skew)  # type: ignore[arg-type]


def _invert(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """The inverse of a small square matrix, column by column."""
    size = len(matrix)
    columns = [
        solve_linear(matrix, [1.0 if i == k else 0.0 for i in range(size)])
        for k in range(size)
    ]
    return [[columns[j][i] for j in range(size)] for i in range(size)]


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


__all__ = [
    "DOWN",
    "IDENTITY",
    "MIN_HAND_EYE_SAMPLES",
    "MIN_TOOL_HAND_EYE_SAMPLES",
    "POSE_AXES",
    "Footprint",
    "HandEyeFit",
    "Mat3",
    "Plane",
    "Pose",
    "ToolHandEyeFit",
    "Transform",
    "Vec3",
    "WorkspaceBox",
    "add",
    "angle_between_deg",
    "angle_delta_deg",
    "dot",
    "fit_plane",
    "footprint",
    "least_squares",
    "mat_mul",
    "mat_vec",
    "matrix_to_rpy",
    "nearest_rotation",
    "norm",
    "plane_height",
    "rotation_distance_deg",
    "rotation_vector",
    "rpy_to_matrix",
    "solve_hand_eye",
    "solve_linear",
    "solve_pivot",
    "solve_tool_hand_eye",
    "sub",
    "tool_cone_deg",
    "transpose",
    "wrap_axis_deg",
    "wrap_deg",
]

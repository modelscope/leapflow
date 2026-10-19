# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rigid-body geometry for node skills: poses, transforms, workspace boxes.

Pure Python on purpose: the node runtime must import on a board without numpy,
and every quantity a skill reasons about is a 3-vector or a 3x3 rotation. The
calibration fits (``fit_plane``, ``solve_pivot``, ``solve_hand_eye``) are small
least-squares problems solved through the normal equations; the one rotation
fit they need is Horn's quaternion method on a 4x4 Jacobi eigensolver.

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


__all__ = [
    "DOWN",
    "IDENTITY",
    "MIN_HAND_EYE_SAMPLES",
    "POSE_AXES",
    "Footprint",
    "HandEyeFit",
    "Mat3",
    "Plane",
    "Pose",
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
    "sub",
    "tool_cone_deg",
    "transpose",
    "wrap_axis_deg",
    "wrap_deg",
]

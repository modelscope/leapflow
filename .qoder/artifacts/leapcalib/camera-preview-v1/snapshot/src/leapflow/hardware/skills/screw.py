# Copyright (c) Alibaba, Inc. and its affiliates.
"""Camera mount from single-joint screw motions, without a calibration board.

A board is a target somebody has to build, hold in view, and keep still. The arm
is already a rigid body the camera can see, and the firmware already reports
where it went, so two poses that differ in exactly one joint are related by a
rotation of *known* angle about a line *known* in the base frame. Recovering that
line in camera coordinates is a four-parameter search, and three such lines --
mutually non-parallel -- over-determine the mount.

What makes the answer trustworthy is how the unknowns are divided:

* the vertical is *measured*, not fitted. The base joint turns about the base
  frame's own +z, so the rotation the camera sees when that joint swings *is*
  the vertical in camera coordinates: one direction, with no heading, no
  translation and no other surface in the estimate. Reading it off the table
  instead substitutes the table's normal -- wherever the table happens to sit --
  for the axis the arm actually turns about, and on the archived cell the two
  differ by about a degree, which is the whole mount tolerance.
* heading and translation are fitted *jointly* against every joint at once. At a
  fixed heading each joint yields one screw shift, and all of them share a single
  ``t_X``: ``(I - R_c) t_X = s_i - R_X t_b,i``. That system is over-determined by
  design, and the gap between the shift a joint measures for itself and the one
  the mount predicts is an error bar in millimetres -- reported per joint rather
  than hidden behind one score.

That shift search has to be bounded by convergence, not by a window. A shift is
``(I - R) q`` for the line's position ``q``, so a wrist whose axis passes near the
surface it carries produces a shift far larger than that surface's own centroid
displacement -- and a window sized for the base joint truncates it. Truncated at
the boundary, the wrist reports a 17 mm disagreement with every mount tried,
including one measured independently with SIFT; searched to convergence it
reports 11 mm and the boundary artifact disappears. So `JointEvidence.measure_shift`
starts wide from the centroid seed and halves until the step stops paying.

Two measured facts set the tolerances, and both are easy to get wrong:

* the evidence is the points that *moved*. An arm working over its own base keeps
  its links within ~150 mm of the base axis, so "far from the axis" deletes the
  surfaces that carry the signal and keeps the station behind them.
* the occupancy cell has to be smaller than the displacement being resolved. Four
  degrees at the arm's 200 mm radius moves a surface 14 mm; a 6 mm cell dilated
  by two declares everything within 18 mm a hit and scores every direction the
  same. At 3 mm and one cell the same sweep separates 0.51 from 0.40.

Scores are symmetric -- moved points of A must land on B *and* moved points of B
on A -- because one direction alone can be gamed by dumping points onto whatever
part of the other cloud happens to be in the way.

numpy is imported inside the functions: the node imports every skill at start-up
and only a calibration run needs it, so a node without it answers
``skill_unavailable`` at that point instead of failing to load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from leapflow.hardware.skills.geometry import (
    Mat3,
    Pose,
    Transform,
    Vec3,
    angle_between_deg,
    mat_mul,
    mat_vec,
    norm,
    rotation_vector,
    sub,
    transpose,
)

# A pair whose joints differ less than this carries no usable rotation: the line
# is unobservable through noise, and ``cot(angle/2)`` in `screw_from_poses`
# diverges.
MIN_SCREW_ANGLE_DEG = 5.0

MIN_CLOUD_POINTS = 400

# Only a joint whose axis *is* the base's +z measures the vertical: a pair
# turning about a line 2 deg off z reports a direction 2 deg off, and the mount
# inherits it. The bound is one degree because that is the error that matters.
MAX_VERTICAL_AXIS_DEG = 1.0

# The vertical has no seed to start from -- the clouds say where the screw line
# lies, not which way is up -- so the coarse pass compares directions spread
# evenly over the sphere, and the best few are then refined continuously.
VERTICAL_DIRECTIONS = 256
VERTICAL_SEEDS = 3
VERTICAL_STEPS_DEG = (6.0, 2.0, 0.6)
VERTICAL_ROUNDS = 6


def _numpy() -> Any:
    """numpy, imported at use: a node without it imports the skills and refuses."""
    import numpy as np

    return np


# ── Screw motion ─────────────────────────────────────────────────────────────


def rotation_about(axis: Sequence[float], angle: float) -> Mat3:
    """Rodrigues rotation of *angle* radians about ``axis`` (normalised here)."""
    length = norm(axis)
    if not (length > 0.0):
        raise ValueError("a rotation axis needs a direction")
    x, y, z = (float(value) / length for value in axis)
    cos, sin = math.cos(angle), math.sin(angle)
    rest = 1.0 - cos
    return (
        (cos + x * x * rest, x * y * rest - z * sin, x * z * rest + y * sin),
        (y * x * rest + z * sin, cos + y * y * rest, y * z * rest - x * sin),
        (z * x * rest - y * sin, z * y * rest + x * sin, cos + z * z * rest),
    )


@dataclass(frozen=True)
class Screw:
    """One rotation: *angle* radians about the line through *point* along *axis*.

    ``rotation`` and ``shift`` are the same rigid motion as ``p -> R p + shift``.
    The shift does not depend on which point of the line *point* is, which is why
    the line's position can be recovered from the pair it was derived from.
    """

    axis: Vec3
    angle: float
    point: Vec3

    @property
    def rotation(self) -> Mat3:
        return rotation_about(self.axis, self.angle)

    @property
    def shift(self) -> Vec3:
        return sub(self.point, mat_vec(self.rotation, self.point))

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": [round(value, 6) for value in self.axis],
            "angle_deg": round(math.degrees(self.angle), 3),
            "point_mm": [round(value, 2) for value in self.point],
        }


def screw_from_poses(first: Pose, second: Pose) -> Screw:
    """The rotation that takes *first*'s flange to *second*'s, as line and angle.

    Both poses are the arm's own report, so this line is the joint's axis in the
    base frame with no camera anywhere in the derivation: the clouds only have to
    show the same motion the firmware says it made. The sign of the angle is the
    one that actually rotates ``first`` onto ``second``, so nothing downstream
    has to guess it.

    The point is the closed form of the line closest to the base origin --
    ``0.5 (t + (k x t) cot(angle / 2))`` for ``t = p_b - R p_a`` -- rather than a
    rank-deficient least squares, which would need an arbitrary regularisation.
    """
    rotation = mat_mul(second.rotation, transpose(first.rotation))
    vector = rotation_vector(rotation)
    angle = norm(vector)
    if math.degrees(angle) < MIN_SCREW_ANGLE_DEG:
        raise ValueError(
            f"the two poses differ by {math.degrees(angle):.2f} deg, too little to "
            f"locate a screw axis (need {MIN_SCREW_ANGLE_DEG:g} deg)"
        )
    axis: Vec3 = (vector[0] / angle, vector[1] / angle, vector[2] / angle)
    motion = sub(second.position, mat_vec(rotation, first.position))
    cross = (
        axis[1] * motion[2] - axis[2] * motion[1],
        axis[2] * motion[0] - axis[0] * motion[2],
        axis[0] * motion[1] - axis[1] * motion[0],
    )
    cotangent = 1.0 / math.tan(angle / 2.0)
    point: Vec3 = tuple(  # type: ignore[assignment]
        0.5 * (motion[index] + cross[index] * cotangent) for index in range(3)
    )
    return Screw(axis, angle, point)


# ── Cloud evidence ───────────────────────────────────────────────────────────


def subsample(points: Any, limit: int) -> Any:
    """Evenly spaced subset of at most *limit* points (order preserved)."""
    np = _numpy()
    data = np.asarray(points, float).reshape(-1, 3)
    if len(data) <= limit:
        return data
    return data[np.linspace(0, len(data) - 1, limit).astype(int)]


class Occupancy:
    """Dilated voxel occupancy of a measured surface, queried by candidate clouds.

    The dilation is precomputed, so testing a candidate is one vectorised lookup
    -- which is what makes a shift search affordable at full cloud density.

    A query point outside the grid is a **miss**, not a hit on the boundary. The
    prototype this came from padded the box by 220 mm and clipped everything
    beyond it onto the outermost cell, where an arm-shaped box is occupied on
    most of its faces; candidate shifts of a quarter of a metre then scored by
    dumping points onto that skin. Treating outside as a miss removes the skin
    and the margin both: the grid is exactly the surface that was measured, and
    a point that lands nowhere counts against the candidate, as it must.
    """

    def __init__(self, points: Any, cell_mm: float, *, reach: int = 1) -> None:
        np = _numpy()
        data = np.asarray(points, float).reshape(-1, 3)
        if not len(data):
            raise ValueError("an occupancy grid needs at least one point")
        self.cell = float(cell_mm)
        self.reach = int(reach)
        self.low = data.min(axis=0)
        high = data.max(axis=0)
        self.dims = (np.ceil((high - self.low) / self.cell).astype(np.int64) + 1)
        occupied = np.zeros(tuple(int(value) for value in self.dims), dtype=bool)
        index = self.index(data)
        occupied[index[:, 0], index[:, 1], index[:, 2]] = True
        self.occupied = self._dilate(occupied, self.reach) if self.reach else occupied

    def _dilate(self, occupied: Any, reach: int) -> Any:
        """Grow occupancy by ``reach`` cells in every direction, without wrapping."""
        np = _numpy()
        out = np.zeros_like(occupied)
        padded = np.pad(occupied, reach)
        span = occupied.shape
        steps = range(2 * reach + 1)
        for dx in steps:
            for dy in steps:
                for dz in steps:
                    out |= padded[
                        dx : dx + span[0], dy : dy + span[1], dz : dz + span[2]
                    ]
        return out

    def index(self, points: Any) -> Any:
        """Cell of each point, clipped -- only used where every point is inside."""
        np = _numpy()
        data = np.asarray(points, float).reshape(-1, 3)
        raw = np.floor((data - self.low) / self.cell).astype(np.int64)
        return np.clip(raw, 0, self.dims - 1)

    def hits(self, points: Any) -> Any:
        """Per point: whether it lands in an occupied (or adjacent) cell.

        A point beyond the grid's box is a miss: nothing was measured there.
        """
        np = _numpy()
        data = np.asarray(points, float).reshape(-1, 3)
        raw = np.floor((data - self.low) / self.cell).astype(np.int64)
        inside = np.all((raw >= 0) & (raw < self.dims), axis=1)
        found = np.zeros(len(data), dtype=bool)
        if not inside.any():
            return found
        index = np.clip(raw[inside], 0, self.dims - 1)
        found[inside] = self.occupied[index[:, 0], index[:, 1], index[:, 2]]
        return found

    def score(self, points: Any) -> float:
        """Fraction of *points* landing in an occupied (or adjacent) cell."""
        data = _numpy().asarray(points, float).reshape(-1, 3)
        if not len(data):
            return 0.0
        return float(self.hits(data).mean())


# ── One joint's evidence ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScrewConfig:
    """Every tolerance the cloud solve uses. Changing one changes the answer."""

    voxel_mm: float = 3.0
    move_cell_mm: float = 4.0
    move_reach: int = 2
    source_limit: int = 3500
    target_limit: int = 16000
    heading_step_deg: float = 2.0
    shift_step_mm: float = 16.0
    shift_floor_mm: float = 1.0


class JointEvidence:
    """One single-joint pose pair, reduced to the point sets the solver compares.

    ``moved_a`` and ``moved_b`` are the points the joint carries that the other
    pose cannot account for. Scoring them against the other pose's occupancy, in
    both directions, is the whole measurement: at the true mount the firmware's
    own rotation lands them on the surface the camera actually saw.
    """

    def __init__(
        self,
        label: str,
        screw: Screw,
        first_cloud: Any,
        second_cloud: Any,
        config: ScrewConfig,
    ) -> None:
        self.label = label
        self.screw = screw
        self.config = config
        self.points_a, self.points_b = len(first_cloud), len(second_cloud)
        if min(self.points_a, self.points_b) < MIN_CLOUD_POINTS:
            raise ValueError(
                f"the arm cloud has {self.points_a}/{self.points_b} points "
                f"(need {MIN_CLOUD_POINTS} in both poses)"
            )
        cloud_a = subsample(first_cloud, config.target_limit)
        cloud_b = subsample(second_cloud, config.target_limit)
        moved = Occupancy(cloud_b, config.move_cell_mm, reach=config.move_reach)
        self.moved_a = subsample(cloud_a[~moved.hits(cloud_a)], config.source_limit)
        moved = Occupancy(cloud_a, config.move_cell_mm, reach=config.move_reach)
        self.moved_b = subsample(cloud_b[~moved.hits(cloud_b)], config.source_limit)
        if min(len(self.moved_a), len(self.moved_b)) < MIN_CLOUD_POINTS // 2:
            if not max(len(self.moved_a), len(self.moved_b)):
                raise ValueError(
                    "nothing in the arm's cloud moved although the joint swung, so the "
                    "camera stopped updating or the arm left its view"
                )
            raise ValueError(
                f"only {len(self.moved_a)}/{len(self.moved_b)} points moved; the "
                "joint carries too little of the arm in view to locate its axis"
            )
        self.grid_a = Occupancy(cloud_a, config.voxel_mm)
        self.grid_b = Occupancy(cloud_b, config.voxel_mm)
        self.centroid_a = self.moved_a.mean(axis=0)
        self.centroid_b = self.moved_b.mean(axis=0)

    def camera_motion(self, rot_x: Mat3, t_x: Sequence[float]) -> tuple[Mat3, Any]:
        """The same screw motion in camera coordinates, seen through ``(R_X, t_X)``."""
        np = _numpy()
        rot = np.asarray(mat_mul(mat_mul(rot_x, self.screw.rotation), transpose(rot_x)), float)
        shift = np.asarray(t_x, float) - rot @ np.asarray(t_x, float) + np.asarray(
            mat_vec(rot_x, self.screw.shift), float
        )
        return rot, shift

    def agreement(self, rot: Any, shift: Any) -> float:
        """Symmetric occupancy agreement of the moved points, forward and back."""
        forward = self.grid_b.score(self.moved_a @ rot.T + shift)
        backward = self.grid_a.score((self.moved_b - shift) @ rot)
        return 0.5 * (forward + backward)

    def measure_shift(
        self, rot_x: Mat3, config: ScrewConfig | None = None
    ) -> tuple[Any, float]:
        """The screw shift this pair prefers at a fixed camera orientation.

        Shifting along the axis changes nothing (the motion is about that axis),
        so only the plane perpendicular to it is searched. The search starts from
        the centroid displacement and *converges* rather than sweeping a fixed
        window: the true shift is ``(I - R) q``, which for a wrist whose axis
        passes near the surface it carries is much larger than that surface's own
        centroid moved, so any window narrow enough to be sharp truncates the
        wrist and any window wide enough to hold it is too coarse to resolve the
        base joint.
        """
        np = _numpy()
        settings = config or self.config
        rot, _ = self.camera_motion(rot_x, (0.0, 0.0, 0.0))
        axis = np.asarray(mat_vec(rot_x, self.screw.axis), float)
        first, second = _basis_of(axis)
        seed = np.asarray(self.centroid_b, float) - rot @ np.asarray(self.centroid_a, float)
        seed = seed - axis * float(seed @ axis)

        def score(along: float, across: float) -> float:
            return self.agreement(rot, seed + first * along + second * across)

        best = (score(0.0, 0.0), 0.0, 0.0)
        step = float(settings.shift_step_mm)
        while step >= float(settings.shift_floor_mm):
            improved, rounds = True, 0
            while improved and rounds < 24:
                improved, rounds = False, rounds + 1
                for da, db in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
                    along, across = best[1] + da, best[2] + db
                    value = score(along, across)
                    if value > best[0] + 1e-6:
                        best, improved = (value, along, across), True
            step /= 2.0
        return seed + first * best[1] + second * best[2], best[0]


def _basis_of(axis: Any) -> tuple[Any, Any]:
    """Two unit vectors spanning the plane perpendicular to *axis*."""
    np = _numpy()
    direction = np.asarray(axis, float)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ direction)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(direction, first)


# ── The mount ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JointReport:
    """What one joint contributed: its agreement, and how far it disagrees."""

    label: str
    angle_deg: float
    axis_base: Vec3
    agreement: float
    shift_gap_mm: float
    points: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint": self.label,
            "angle_deg": round(self.angle_deg, 3),
            "axis": [round(value, 6) for value in self.axis_base],
            "agreement": round(self.agreement, 4),
            "shift_gap_mm": round(self.shift_gap_mm, 2),
            "moved": list(self.points),
        }


@dataclass(frozen=True)
class MountFit:
    """``p_cam = rotation @ p_base + translation`` for a fixed camera."""

    rotation: Mat3
    translation: Vec3
    score: float
    heading_deg: float
    joints: tuple[JointReport, ...]

    @property
    def max_shift_gap_mm(self) -> float:
        return max((joint.shift_gap_mm for joint in self.joints), default=math.inf)

    @property
    def camera(self) -> Transform:
        """The camera's pose in the base frame: ``T_base_camera``."""
        rotation = transpose(self.rotation)
        position = tuple(-value for value in mat_vec(rotation, self.translation))
        return Transform(rotation, position)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        camera = self.camera.to_pose()
        return {
            "camera": camera.to_dict(),
            "score": round(self.score, 4),
            "heading_deg": round(self.heading_deg, 2),
            "max_shift_gap_mm": round(self.max_shift_gap_mm, 2),
            "pairs": [joint.to_dict() for joint in self.joints],
        }


def vertical_frame(up: Vec3) -> Mat3:
    """A rotation whose third column is *up*, leaving the heading as the only unknown."""
    length = norm(up)
    if not (length > 0.0):
        raise ValueError("the measured vertical has no direction")
    up = (up[0] / length, up[1] / length, up[2] / length)
    seed = (1.0, 0.0, 0.0)
    if abs(sum(a * b for a, b in zip(seed, up))) > 0.9:
        seed = (0.0, 0.0, 1.0)
    first = (
        up[1] * seed[2] - up[2] * seed[1],
        up[2] * seed[0] - up[0] * seed[2],
        up[0] * seed[1] - up[1] * seed[0],
    )
    length = norm(first)
    first = (first[0] / length, first[1] / length, first[2] / length)
    second = (
        up[1] * first[2] - up[2] * first[1],
        up[2] * first[0] - up[0] * first[2],
        up[0] * first[1] - up[1] * first[0],
    )
    return tuple(  # type: ignore[return-value]
        tuple(column[index] for column in (first, second, up)) for index in range(3)
    )


def spin(up: Vec3, angle: float) -> Mat3:
    """Rotation about *up* by *angle* radians -- the unknown heading."""
    return rotation_about(up, angle)


def solve_vertical(
    joints: Sequence[JointEvidence],
    *,
    max_axis_deg: float = MAX_VERTICAL_AXIS_DEG,
    config: ScrewConfig | None = None,
) -> Vec3:
    """The base frame's +z in camera coordinates, from the base joint's own swing.

    A joint that turns about +z moves the camera as ``Rot(R_X z, angle)``, so its
    rotation axis *is* the vertical and nothing else about the mount has to be
    known to read it -- not the heading, not the translation, not the table. What
    the clouds decide is which direction that is: the firmware only says the
    joint turned, and the camera says where the surface it carries went.

    The direction has no seed, so the sphere is swept coarsely at the shift each
    direction implies and the best few directions are refined continuously. A
    pair turning about a line that is *not* the base vertical measures that line
    instead and the error passes straight into the mount, so the closest pair is
    refused rather than used when it is more than ``max_axis_deg`` off z.
    """
    settings = config or ScrewConfig()
    joint, off_deg = _base_pair(joints)
    if off_deg > max_axis_deg:
        raise ValueError(
            f"no joint turns about the base vertical: the closest is {joint.label}, "
            f"{off_deg:.1f} deg off it (limit {max_axis_deg:g} deg). A pair that turns "
            "about a tilted line measures that line, so include the joint that carries "
            "the base's own axis (j1)"
        )
    directions = _golden_directions(VERTICAL_DIRECTIONS)
    coarse = sorted(
        (
            (_vertical_score(joint, direction, settings, search=False), index)
            for index, direction in enumerate(directions)
        ),
        reverse=True,
    )
    best: Any = directions[int(coarse[0][1])]
    score = -1.0
    for _, index in coarse[:VERTICAL_SEEDS]:
        up, value = _refine_vertical(joint, directions[int(index)], settings)
        if value > score:
            best, score = up, value
    return tuple(float(value) for value in best)  # type: ignore[return-value]


def _base_pair(joints: Sequence[JointEvidence]) -> tuple[JointEvidence, float]:
    """The pair that turns closest to the base's own axis, and how far off it is.

    The axis comes from the firmware's own poses, so the line is exact to
    numerical noise; the pair's *sign* is whatever made the motion, which is why
    a downward axis measures the same line as an upward one.
    """
    best, closest = joints[0], math.inf
    for joint in joints:
        degrees = angle_between_deg(joint.screw.axis, (0.0, 0.0, 1.0))
        degrees = min(degrees, 180.0 - degrees)
        if degrees < closest:
            best, closest = joint, degrees
    return best, closest


def _golden_directions(count: int) -> Any:
    """*count* unit directions spread evenly over the sphere (golden-angle spiral)."""
    np = _numpy()
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = math.pi * (1.0 + 5.0 ** 0.5) * index
    return np.column_stack([
        np.cos(azimuth) * np.sin(polar),
        np.sin(azimuth) * np.sin(polar),
        np.cos(polar),
    ])


def _vertical_score(
    joint: JointEvidence, up: Any, settings: ScrewConfig, *, search: bool
) -> float:
    """How well this pair's motion about *up* lands its own surface.

    Coarse directions are scored at the shift their own centroid implies -- one
    agreement each, which is what makes sweeping the sphere affordable -- and the
    seeds that survive are scored after the full shift search.
    """
    np = _numpy()
    rot_x = vertical_frame(tuple(float(value) for value in up))
    if search:
        return joint.measure_shift(rot_x, settings)[1]
    rot, _ = joint.camera_motion(rot_x, (0.0, 0.0, 0.0))
    axis = np.asarray(mat_vec(rot_x, joint.screw.axis), float)
    seed = np.asarray(joint.centroid_b, float) - rot @ np.asarray(joint.centroid_a, float)
    seed = seed - axis * float(seed @ axis)
    return joint.agreement(rot, seed)


def _refine_vertical(
    joint: JointEvidence, seed: Any, settings: ScrewConfig
) -> tuple[Any, float]:
    """Pattern search of the direction over the sphere, from a coarse seed."""
    np = _numpy()
    up = np.asarray(seed, float)
    up = up / np.linalg.norm(up)
    score = _vertical_score(joint, up, settings, search=True)
    for step_deg in VERTICAL_STEPS_DEG:
        step = math.radians(step_deg)
        for _ in range(VERTICAL_ROUNDS):
            first, second = _basis_of(up)
            improved = False
            for along, across in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                trial = up + (first * along + second * across) * step
                trial = trial / np.linalg.norm(trial)
                value = _vertical_score(joint, trial, settings, search=True)
                if value > score + 1e-6:
                    up, score, improved = trial, value, True
            if not improved:
                break
    return up, score


def solve_translation(
    joints: Sequence[JointEvidence], rot_x: Mat3, shifts: Sequence[Any]
) -> Vec3:
    """Least-squares ``t_X`` from ``(I - R_c) t_X = s_i - R_X t_b,i``, all joints at once.

    Each rotation contributes two independent equations, and rotations about one
    axis cannot see the camera's translation along it -- which is why a joint with
    a horizontal axis is not optional.
    """
    np = _numpy()
    rows, values = [], []
    for joint, shift in zip(joints, shifts):
        rot, _ = joint.camera_motion(rot_x, (0.0, 0.0, 0.0))
        rows.append(np.eye(3) - rot)
        values.append(np.asarray(shift, float) - np.asarray(mat_vec(rot_x, joint.screw.shift), float))
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(values), rcond=None)
    return tuple(float(value) for value in solution)  # type: ignore[return-value]


def _total(joints: Sequence[JointEvidence], rot_x: Mat3, t_x: Sequence[float]) -> float:
    scores = [
        joint.agreement(*joint.camera_motion(rot_x, t_x)) for joint in joints
    ]
    return sum(scores) / len(scores)


def solve_mount(
    joints: Sequence[JointEvidence],
    *,
    up: Vec3,
    config: ScrewConfig | None = None,
) -> MountFit:
    """Sweep the heading, solve the translation at each, then polish both.

    ``up`` is the base frame's +z in camera coordinates -- measured from the base
    joint's own swing, see `solve_vertical` -- and becomes the mount's third
    column, so the fit never trades a measured direction for a fitted one.
    """
    settings = config or ScrewConfig()
    if len(joints) < 3:
        raise ValueError(
            f"a mount needs three joints to be observable, got {len(joints)}; "
            "the camera's height above the base is invisible to rotations about one axis"
        )
    home = vertical_frame(up)
    score, degrees, t_x = -1.0, 0.0, (0.0, 0.0, 0.0)
    for heading in _arange(0.0, 360.0, settings.heading_step_deg):
        rot_x = mat_mul(spin(up, math.radians(heading)), home)
        shifts = [joint.measure_shift(rot_x, settings)[0] for joint in joints]
        trial = solve_translation(joints, rot_x, shifts)
        value = _total(joints, rot_x, trial)
        if value > score:
            score, degrees, t_x = value, heading, trial
    score, degrees, t_x = _polish(joints, up, home, score, degrees, t_x)
    rot_x = mat_mul(spin(up, math.radians(degrees)), home)
    reports = tuple(
        _joint_report(joint, rot_x, t_x, settings) for joint in joints
    )
    return MountFit(
        rotation=rot_x,
        translation=t_x,
        score=score,
        heading_deg=degrees,
        joints=reports,
    )


def _joint_report(
    joint: JointEvidence, rot_x: Mat3, t_x: Sequence[float], config: ScrewConfig
) -> JointReport:
    """One joint's agreement, and the gap between its preferred shift and the mount's."""
    np = _numpy()
    rot, shift = joint.camera_motion(rot_x, t_x)
    measured, _ = joint.measure_shift(rot_x, config)
    gap = float(np.linalg.norm(np.asarray(measured, float) - np.asarray(shift, float)))
    return JointReport(
        label=joint.label,
        angle_deg=math.degrees(joint.screw.angle),
        axis_base=joint.screw.axis,
        agreement=joint.agreement(rot, shift),
        shift_gap_mm=gap,
        points=(len(joint.moved_a), len(joint.moved_b)),
    )


def _polish(
    joints: Sequence[JointEvidence],
    up: Vec3,
    home: Mat3,
    score: float,
    degrees: float,
    t_x: Sequence[float],
) -> tuple[float, float, Sequence[float]]:
    """Pattern search on heading and translation, halving both steps four times."""
    for step_deg, step_mm in ((2.0, 6.0), (0.7, 2.0), (0.2, 0.7), (0.07, 0.25)):
        improved, rounds = True, 0
        while improved and rounds < 40:
            improved, rounds = False, rounds + 1
            for sign in (1.0, -1.0):
                trial = mat_mul(spin(up, math.radians(degrees + sign * step_deg)), home)
                value = _total(joints, trial, t_x)
                if value > score + 1e-6:
                    score, degrees, improved = value, degrees + sign * step_deg, True
            for index in range(3):
                for sign in (1.0, -1.0):
                    trial = list(t_x)
                    trial[index] += sign * step_mm
                    value = _total(joints, mat_mul(spin(up, math.radians(degrees)), home), trial)
                    if value > score + 1e-6:
                        score, t_x, improved = value, tuple(trial), True
    return score, degrees, t_x


def _arange(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [start + index * step for index in range(count)]


__all__ = [
    "MAX_VERTICAL_AXIS_DEG",
    "MIN_SCREW_ANGLE_DEG",
    "JointEvidence",
    "JointReport",
    "MountFit",
    "Occupancy",
    "Screw",
    "ScrewConfig",
    "rotation_about",
    "screw_from_poses",
    "solve_mount",
    "solve_translation",
    "solve_vertical",
    "spin",
    "subsample",
    "vertical_frame",
]

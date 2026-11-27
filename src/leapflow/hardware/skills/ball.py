# Copyright (c) Alibaba, Inc. and its affiliates.
"""Where a ball's centre sits in a depth cloud, for camera metrology.

A sphere is the one target whose pose in the image is a *point*: no corners to
order, no axes to disambiguate, no board to mount. That is why it can measure a
camera mount without the cell cooperating -- but only if the centre can be read
to a millimetre out of a cloud that is mostly something else.

Three choices carry that:

* the search is seeded by radius, not by a fitted surface. The radius is
  measured with the gripper's own jaws before the ball is carried, so a
  hypothesis is three surface points and the known radius -- two candidate
  centres, closed form -- and the trial that lands the most points on the
  sphere wins. A plane (the table), a box (the gum box), and a finger padding
  all fail to accumulate inliers at that radius however much of the cloud they
  take up, so clutter costs the search nothing but time.
* the window is a *sphere* around the prediction, not a depth band. A band wide
  enough to absorb a stale mount still contains a wall of table, and the table
  is a worse target than the ball in every way except size.
* the last step reports two radii. The radius the jaws measured is a
  constraint; the radius the cloud fits freely is a check on the ball itself
  and on the gripper still holding it. When those two disagree the still is not
  a rigid sphere, and the caller must not fold it into a mount.

numpy is imported inside the functions: the node imports every skill at start-up
and only a calibration run needs it, so a node without it answers
``skill_unavailable`` at that point instead of failing to load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from leapflow.hardware.skills.geometry import Transform, Vec3, norm, sub

# Fewer points than this and a "surface" is indistinguishable from noise on the
# arm or the table; the fit would succeed while meaning nothing.
MIN_BALL_POINTS = 40
# Hypotheses per still. Each is two closed-form centres, scored against the
# window at once, so this costs one pass over the points per hypothesis.
TRIALS = 400
# How far off the sphere of known radius a point may sit and still count as its
# surface. The depth noise of the cell's cameras at a metre is 1-3 mm.
INLIER_BAND_MM = 3.0
# Points the scoring pass keeps, evenly strided. A full room cloud is far
# larger, and the ball's proportional share survives the stride.
POINT_LIMIT = 12000
# Refinement stops when a step moves the centre less than this.
_STEP_FLOOR_MM = 1e-4
_REFINE_STEPS = 24


def _numpy() -> Any:
    """numpy, imported at use: a node without it imports the skills and refuses."""
    import numpy as np

    return np


@dataclass(frozen=True)
class BallFit:
    """One still's ball: where its surface puts the centre, and how well.

    ``radius_mm`` is the radius the caller declared (the jaws' measurement) and
    ``fitted_radius_mm`` the one the cloud prefers when left free. ``rms_mm`` is
    the residual of the declared-radius fit over its inliers, in millimetres --
    the number that says whether this still is worth a mount.
    """

    centre: Vec3
    radius_mm: float
    fitted_radius_mm: float
    inliers: int
    points: int
    rms_mm: float

    @property
    def coverage(self) -> float:
        """Fraction of the window that landed on the sphere."""
        return self.inliers / self.points if self.points else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "centre": [round(value, 3) for value in self.centre],
            "radius_mm": round(self.radius_mm, 3),
            "fitted_radius_mm": round(self.fitted_radius_mm, 3),
            "inliers": self.inliers,
            "points": self.points,
            "coverage": round(self.coverage, 4),
            "rms_mm": round(self.rms_mm, 3),
        }


def fit_ball(
    points: Sequence[Vec3],
    *,
    radius_mm: float,
    hint: Sequence[float],
    window_mm: float | None = None,
    band_mm: float = INLIER_BAND_MM,
    seed: int = 0,
) -> BallFit:
    """The ball of *radius_mm* nearest *hint* in *points*, or ``ValueError``.

    ``hint`` is where the ball is believed to be (a mount that is a few
    centimetres out is fine -- the window absorbs it), and only points within
    ``window_mm`` of it are searched, so the call stays cheap on a room-sized
    cloud. Returns the centre, the inlier count, and both radii; raises when no
    surface of that radius is in the window, which the caller must treat as a
    still it cannot use rather than as a ball at the hint.
    """
    np = _numpy()
    radius = float(radius_mm)
    if not (radius > 0.0):
        raise ValueError("a ball needs a positive radius")
    window = float(window_mm) if window_mm is not None else radius * 2.0 + 80.0
    cloud = _window(points, hint, window)
    if len(cloud) < MIN_BALL_POINTS:
        raise ValueError(
            f"only {len(cloud)} points lie within {window:.0f} mm of where the ball "
            f"should be (need {MIN_BALL_POINTS}); the ball is out of view or out of the "
            "depth range"
        )
    guess = np.asarray(hint, float)
    for centre in _hypotheses(cloud, radius, guess, seed=seed):
        fitted = _refine(cloud, centre, radius, free_radius=False)
        if fitted is None:
            continue
        centre, _radius, rms = fitted
        inliers = _inliers(cloud, centre, radius, band_mm)
        if int(inliers.sum()) < MIN_BALL_POINTS:
            continue
        surface = cloud[inliers]
        refined = _refine(surface, centre, radius, free_radius=False)
        if refined is None:
            continue
        centre, _radius, rms = refined
        free = _refine(surface, centre, radius, free_radius=True)
        fitted_radius = radius if free is None else free[1]
        return BallFit(
            centre=tuple(float(value) for value in centre),
            radius_mm=radius,
            fitted_radius_mm=float(fitted_radius),
            inliers=int(len(surface)),
            points=int(len(cloud)),
            rms_mm=float(rms),
        )
    raise ValueError(
        f"no surface of radius {radius:.0f} mm is inside the window ({len(cloud)} points); "
        "check the ball is carried, that it is round, and that the camera sees it"
    )


def _window(points: Sequence[Vec3], hint: Sequence[float], window_mm: float) -> Any:
    """The points within ``window_mm`` of *hint*, strided to ``POINT_LIMIT``."""
    np = _numpy()
    cloud = np.asarray(points, float).reshape(-1, 3)
    if len(cloud) == 0:
        return cloud
    keep = np.linalg.norm(cloud - np.asarray(hint, float), axis=1) <= float(window_mm)
    near = cloud[keep]
    if len(near) > POINT_LIMIT:
        near = near[:: max(1, math.ceil(len(near) / POINT_LIMIT))]
    return near


def _hypotheses(cloud: Any, radius: float, guess: Any, *, seed: int) -> list[Any]:
    """Best-first candidate centres, each from three points and the known radius."""
    np = _numpy()
    import random

    generator = random.Random(seed)
    count = len(cloud)
    trials = min(TRIALS, max(32, count))
    scored: list[tuple[int, Any]] = []
    for _ in range(trials):
        picks = [generator.randrange(count) for _ in range(3)]
        if len(set(picks)) < 3:
            continue
        for centre in _through(cloud, picks, radius):
            scored.append((int(_inlier_count(cloud, centre, radius)), centre))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], float(np.linalg.norm(item[1] - guess))))
    # Distinct basins only: a second start within a millimetre of a better one
    # would refine to the same place and waste the refinement.
    starts: list[Any] = []
    for _count, centre in scored:
        if all(float(np.linalg.norm(centre - start)) > 1.0 for start in starts):
            starts.append(centre)
        if len(starts) >= 4:
            break
    return starts


def _through(cloud: Any, picks: Sequence[int], radius: float) -> list[Any]:
    """The two centres of a sphere of *radius* through three points."""
    np = _numpy()
    p0, p1, p2 = (cloud[index] for index in picks)
    first, second = p1 - p0, p2 - p0
    cross = np.cross(first, second)
    area = float(np.linalg.norm(cross))
    if area < 1e-6:  # collinear: the circumcentre is not defined
        return []
    # Circumcentre offset from p0 in the plane's own coordinates.
    a = np.array(
        [[float(first @ first), float(first @ second)], [float(first @ second), float(second @ second)]]
    )
    b = np.array([float(first @ first) / 2.0, float(second @ second) / 2.0])
    try:
        solved = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return []
    in_plane = first * solved[0] + second * solved[1]
    lift = radius * radius - float(in_plane @ in_plane)
    if lift < 0.0:  # the three points are further apart than a sphere of this radius allows
        return []
    normal = cross / area
    height = math.sqrt(lift)
    return [p0 + in_plane + normal * height, p0 + in_plane - normal * height]


def _inlier_count(cloud: Any, centre: Any, radius: float) -> int:
    np = _numpy()
    return int(np.count_nonzero(np.abs(np.linalg.norm(cloud - centre, axis=1) - radius) <= INLIER_BAND_MM))


def _inliers(cloud: Any, centre: Any, radius: float, band_mm: float) -> Any:
    np = _numpy()
    return np.abs(np.linalg.norm(cloud - centre, axis=1) - radius) <= float(band_mm)


def _refine(
    cloud: Any, centre: Any, radius: float, *, free_radius: bool
) -> tuple[Any, float, float] | None:
    """Gauss-Newton on the centre (and the radius when *free_radius*)."""
    np = _numpy()
    position = np.asarray(centre, float).copy()
    size = float(radius)
    for _ in range(_REFINE_STEPS):
        offsets = position - cloud
        distances = np.linalg.norm(offsets, axis=1)
        if float(np.min(distances)) < 1e-6:
            return None
        residual = distances - size
        # d(|p - c| - r)/dc = (c - p)/|c - p|; d/dr = -1
        jacobian = np.hstack([offsets / distances[:, None], -np.ones((len(cloud), 1))])
        if not free_radius:
            jacobian = jacobian[:, :3]
        normal = jacobian.T @ jacobian
        if float(np.linalg.cond(normal)) > 1e12:
            return None
        step = np.linalg.solve(normal, -jacobian.T @ residual)
        position = position + step[:3]
        if free_radius:
            size = size + float(step[3])
            if not (size > 0.0):
                return None
        if float(np.linalg.norm(step)) < _STEP_FLOOR_MM:
            break
    rms = float(math.sqrt(float(np.mean((np.linalg.norm(position - cloud, axis=1) - size) ** 2))))
    return position, size, rms


def residual_mm(model: Any, pose: Any, centre: Sequence[float]) -> float:
    """How far *centre* (camera frame) is from what the model predicts for *pose*."""
    carried = Transform.from_pose(pose).apply(model.tool)
    return norm(sub(model.mount.apply(carried), centre))


__all__ = [
    "INLIER_BAND_MM",
    "MIN_BALL_POINTS",
    "POINT_LIMIT",
    "BallFit",
    "fit_ball",
    "residual_mm",
]

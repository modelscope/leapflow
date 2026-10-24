# Copyright (c) Alibaba, Inc. and its affiliates.
"""The board-free hand-eye solver: nine unknowns from one flange-fixed point.

``solve_tool_hand_eye`` recovers the camera mount *and* the observed tool point
from the same samples, so these tests synthesise both, forward-project them from
a pose set, and check what comes back -- including the degenerate and outlier
cases a live run can hit.
"""

from __future__ import annotations

import random

import pytest

from leapflow.hardware.skills.geometry import (
    MIN_TOOL_HAND_EYE_SAMPLES,
    Pose,
    Transform,
    matrix_to_rpy,
    norm,
    rotation_distance_deg,
    rpy_to_matrix,
    solve_tool_hand_eye,
    sub,
)

MOUNT = Transform(rpy_to_matrix(96.0, -3.0, 172.0), (44.0, -310.0, 402.0))
TOOL = (7.4, -2.4, 150.0)


def _poses(*, tilt: float = 20.0, spread: float = 75.0) -> list[Pose]:
    """A ring of poses that both tilts about two axes and spreads over the cell.

    Spread matters as much as tilt: with the tool point 150 mm out on the flange,
    a pose set that only turns leaves the mount's position weakly constrained.
    """
    rows = [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (tilt, 0.0, spread, 0.0, 20.0),
        (-tilt, 0.0, -spread, 0.0, 20.0),
        (0.0, tilt, 0.0, spread, 0.0),
        (0.0, -tilt, 0.0, -spread, 0.0),
        (tilt / 2.0, tilt / 2.0, spread * 0.6, spread * 0.6, 15.0),
        (-tilt / 2.0, tilt / 2.0, -spread * 0.6, spread * 0.6, 15.0),
        (tilt / 2.0, -tilt / 2.0, spread * 0.6, -spread * 0.6, 15.0),
        (-tilt / 2.0, -tilt / 2.0, -spread * 0.6, -spread * 0.6, 15.0),
        (0.0, 0.0, spread * 0.7, -spread * 0.7, 0.0),
    ]
    poses: list[Pose] = []
    for rx, ry, dx, dy, dz in rows:
        rotation = rpy_to_matrix(180.0 + rx, ry, dz)
        position = (260.0 + dx, 20.0 + dy, 210.0 + 0.5 * (dx + dy) / 6.0)
        poses.append(Pose(*position, *matrix_to_rpy(rotation)))
    return poses


def _project(poses: list[Pose], noise_mm: float = 0.0, seed: int = 7) -> list[tuple[float, float, float]]:
    """Where the camera sees the tool point at each pose, optionally with noise."""
    generator = random.Random(seed)
    points = []
    for pose in poses:
        seen = MOUNT.apply(Transform.from_pose(pose).apply(TOOL))
        jitter = [generator.gauss(0.0, noise_mm) for _ in range(3)] if noise_mm else [0.0] * 3
        points.append((seen[0] + jitter[0], seen[1] + jitter[1], seen[2] + jitter[2]))
    return points


def _errors(fit) -> tuple[float, float, float]:
    return (
        norm(sub(fit.mount.translation, MOUNT.translation)),
        rotation_distance_deg(fit.mount.rotation, MOUNT.rotation),
        norm(sub(fit.tool, TOOL)),
    )


def test_exact_samples_recover_the_mount_and_the_tool_point() -> None:
    fit = solve_tool_hand_eye(_poses(), _project(_poses()))
    moved, turned, tool = _errors(fit)
    assert moved < 1e-6
    assert turned < 1e-6
    assert tool < 1e-6
    assert fit.residual_max_mm < 1e-6
    assert fit.held_out_max_mm < 1e-6


def test_millimetre_noise_stays_inside_its_own_error_budget() -> None:
    poses = _poses()
    fit = solve_tool_hand_eye(poses, _project(poses, noise_mm=1.0))
    moved, turned, tool = _errors(fit)
    assert moved < 3.0
    assert turned < 1.0
    assert tool < 12.0
    assert 0.3 < fit.residual_rms_mm < 2.5
    assert fit.held_out_rms_mm < 3.0
    assert fit.held_out_max_mm < 6.0
    assert fit.condition_number < 1e3


def test_one_wild_sample_is_down_weighted_and_caught_by_the_held_out_check() -> None:
    poses = _poses()
    points = _project(poses, noise_mm=0.5)
    points[6] = (points[6][0] + 40.0, points[6][1] - 30.0, points[6][2] + 25.0)
    fit = solve_tool_hand_eye(poses, points)
    moved, turned, _ = _errors(fit)
    assert moved < 3.0
    assert turned < 1.0
    assert fit.residual_max_mm > 40.0  # the bad sample is reported, not hidden
    assert fit.outliers == 1  # and Huber kept it from ruling the fit
    assert fit.held_out_max_mm > 20.0  # leave-one-out names it as the sample that misses


def test_a_pose_set_that_turns_about_one_axis_is_refused() -> None:
    poses = [Pose(260.0, 20.0, 210.0, 0.0, 0.0, float(rz)) for rz in range(0, 60, 6)]
    points = [(200.0 + i, 30.0, 300.0) for i in range(len(poses))]
    with pytest.raises(ValueError, match="single axis"):
        solve_tool_hand_eye(poses, points)


def test_too_few_poses_are_refused() -> None:
    poses = _poses()[: MIN_TOOL_HAND_EYE_SAMPLES - 1]
    with pytest.raises(ValueError, match="at least"):
        solve_tool_hand_eye(poses, _project(poses))


def test_one_point_per_pose_is_required() -> None:
    poses = _poses()
    with pytest.raises(ValueError, match="one camera point per arm pose"):
        solve_tool_hand_eye(poses, _project(poses)[:-1])


def test_a_thin_pose_set_reports_a_worse_estimate_not_a_confident_one() -> None:
    wide = _poses(tilt=20.0, spread=75.0)
    thin = _poses(tilt=2.0, spread=8.0)
    generous = solve_tool_hand_eye(wide, _project(wide, noise_mm=0.5))
    narrow = solve_tool_hand_eye(thin, _project(thin, noise_mm=0.5, seed=3))
    assert narrow.condition_number > generous.condition_number
    assert narrow.rotation_sigma_deg > generous.rotation_sigma_deg
    assert generous.condition_number < 1e3

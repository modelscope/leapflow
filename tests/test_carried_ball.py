# Copyright (c) Alibaba, Inc. and its affiliates.
"""The carried-ball mount: a sphere out of a cluttered cloud, then nine unknowns.

Two halves, because the method has two halves. The ball fit has to find a
surface of a *known* radius in a window that is mostly table, gripper and arm --
and it has to say when the ball it finds is not the radius the jaws measured.
The mount solve is the existing tool hand-eye, so what these tests pin down is
the split between the poses that fit and the poses that only check: the number
the record carries has to come from poses the fit never saw.
"""

from __future__ import annotations

import asyncio
import math
import random
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.skills.ball import fit_ball, residual_mm
from leapflow.hardware.skills.builtin.camera_calibration import (
    CarriedBall,
    _as_seen,
    _ring,
    _spread_marks,
)
from leapflow.hardware.skills.frame_status import Accuracy, FrameStatus
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    matrix_to_rpy,
    norm,
    rpy_to_matrix,
    solve_tool_hand_eye,
    sub,
)
from leapflow.hardware.skills.provenance import (
    CarriedBallEvidence,
    carried_ball_problem,
    judge,
)
from leapflow.hardware.skills.protocol import METHOD_CARRIED_BALL, SkillError

BALL = (120.0, -40.0, 780.0)
RADIUS = 32.0
# The cell's own static clutter: a table plane behind the ball and a box beside it.
TABLE_Z = 980.0


def _hemisphere(centre, radius: float, *, count: int = 900, noise_mm: float = 0.0, seed: int = 5):
    """Points on the near side of a sphere, as a depth camera sees it, plus noise."""
    generator = random.Random(seed)
    points = []
    for _ in range(count):
        while True:
            x, y, z = (generator.uniform(-1.0, 1.0) for _ in range(3))
            length = math.sqrt(x * x + y * y + z * z)
            if 0.2 < length <= 1.0:
                break
        scale = radius / length
        radius_here = scale + (generator.gauss(0.0, noise_mm) if noise_mm else 0.0)
        points.append(
            (
                centre[0] + x * scale,
                centre[1] + y * scale,
                centre[2] - abs(z) * radius_here,
            )
        )
    return points


def _clutter(*, count: int = 1500, seed: int = 9):
    """A table plane and a box: both closer to the camera than the ball's far side."""
    generator = random.Random(seed)
    points = [
        (generator.uniform(-400.0, 400.0), generator.uniform(-400.0, 400.0), TABLE_Z)
        for _ in range(count)
    ]
    points += [
        (-220.0 + generator.uniform(0.0, 60.0), 90.0 + generator.uniform(0.0, 60.0), 700.0 + generator.uniform(0.0, 40.0))
        for _ in range(count // 3)
    ]
    return points


def test_the_ball_is_found_at_its_centre_through_table_and_box_clutter() -> None:
    ball = fit_ball(
        _hemisphere(BALL, RADIUS, noise_mm=1.0) + _clutter(),
        radius_mm=RADIUS,
        hint=(BALL[0] + 25.0, BALL[1] - 18.0, BALL[2] + 14.0),  # a 22 mm mount is 22 mm out
    )
    assert norm(sub(ball.centre, BALL)) < 2.0
    assert ball.inliers > 500
    assert ball.rms_mm < 2.0
    assert abs(ball.fitted_radius_mm - RADIUS) < 1.0


def test_a_window_with_no_sphere_of_that_radius_is_refused() -> None:
    with pytest.raises(ValueError, match="no surface of radius"):
        fit_ball(_clutter(), radius_mm=RADIUS, hint=(0.0, 0.0, 800.0), window_mm=400.0)


def test_an_empty_window_says_the_ball_is_out_of_view() -> None:
    with pytest.raises(ValueError, match="out of view"):
        fit_ball(
            _hemisphere(BALL, RADIUS),
            radius_mm=RADIUS,
            hint=(BALL[0] + 900.0, BALL[1], BALL[2]),
            window_mm=100.0,
        )


def test_the_fitted_radius_is_reported_so_a_squeezed_ball_shows_up() -> None:
    squashed = _hemisphere(BALL, RADIUS, noise_mm=0.5)
    squashed = [(x, y, BALL[2] - (BALL[2] - z) * 0.8) for x, y, z in squashed]
    ball = fit_ball(squashed, radius_mm=RADIUS, hint=BALL)
    assert abs(ball.fitted_radius_mm - RADIUS) > 1.5


# ── The mount from the centres ───────────────────────────────────────────────

MOUNT = Transform(rpy_to_matrix(96.0, -3.0, 172.0), (44.0, -310.0, 402.0))  # T_camera_base
BALL_IN_GRIPPER = (6.0, -3.0, 138.0)


def _poses(*, tilt: float = 16.0, spread: float = 45.0, count: int = 12) -> list[Pose]:
    poses: list[Pose] = []
    for index in range(count):
        azimuth = 2.0 * math.pi * index / count
        rx = 180.0 + tilt * math.cos(azimuth)
        ry = tilt * math.sin(azimuth)
        rz = -96.0 + 20.0 * math.cos(3.0 * azimuth)
        x = 200.0 + spread * math.cos(azimuth)
        y = 40.0 + spread * math.sin(azimuth)
        z = 300.0 + (8.0 if index % 2 else 0.0)
        poses.append(Pose(x, y, z, *matrix_to_rpy(rpy_to_matrix(rx, ry, rz))))
    return poses


def _centres(poses: list[Pose], *, noise_mm: float = 0.0, seed: int = 13) -> list[tuple[float, float, float]]:
    generator = random.Random(seed)
    centres = []
    for pose in poses:
        seen = MOUNT.apply(Transform.from_pose(pose).apply(BALL_IN_GRIPPER))
        jitter = [generator.gauss(0.0, noise_mm) for _ in range(3)] if noise_mm else [0.0] * 3
        centres.append(tuple(seen[i] + jitter[i] for i in range(3)))
    return centres


def test_the_installed_matrix_maps_camera_points_into_the_base_frame() -> None:
    """The record stores ``T_base_camera``, the same convention the board and screw fits use.

    ``solve_tool_hand_eye`` solves the opposite direction (it projects base points
    into the camera), so the skill inverts once before recording. A camera point
    the record cannot carry into the base frame is a mount nothing can use.
    """
    poses = _poses()
    fit = solve_tool_hand_eye(poses, _centres(poses))
    installed = Transform.from_matrix(fit.mount.inverse().to_matrix())
    probe = (120.0, -40.0, 500.0)  # a point as the depth camera reports it
    assert norm(sub(installed.apply(probe), MOUNT.inverse().apply(probe))) < 1e-6
    assert norm(sub(fit.mount.translation, MOUNT.translation)) < 1e-6


def test_held_out_poses_the_fit_never_saw_stay_inside_the_reported_bound() -> None:
    poses = _poses()
    centres = _centres(poses, noise_mm=0.8)
    marks = _spread_marks(len(poses), 4)
    train = [pose for pose, mark in zip(poses, marks, strict=True) if not mark]
    held = [pose for pose, mark in zip(poses, marks, strict=True) if mark]
    held_centres = [point for point, mark in zip(centres, marks, strict=True) if mark]
    training = solve_tool_hand_eye(
        train, [point for point, mark in zip(centres, marks, strict=True) if not mark]
    )
    errors = [
        residual_mm(training, pose, centre) for pose, centre in zip(held, held_centres, strict=True)
    ]
    assert len(held) == 4
    assert max(errors) < 5.0
    assert training.held_out_max_mm < 6.0


def test_the_marks_are_spread_through_the_sweep_rather_than_appended() -> None:
    marks = _spread_marks(14, 4)
    assert sum(marks) == 4
    assert marks[0] is False and marks[-1] is False
    assert max(marks[i] and marks[i + 1] for i in range(len(marks) - 1)) is False


# ── The seed the sweep searches around ──────────────────────────────────────

INSTALLED = Transform(rpy_to_matrix(96.0, -3.0, 172.0), (10.0, -20.0, 30.0))
LAST_RECORDED = Transform(rpy_to_matrix(96.0, -3.0, 172.0), (44.0, -310.0, 402.0))
ACCEPT = {
    "min_ball_poses": 6,
    "min_ball_held_out": 3,
    "ball_depth_margin_mm": 250.0,
    "ball_window_extra_mm": 90.0,
    "max_ball_fit_rms_mm": 2.0,
    "max_ball_radius_error_mm": 4.0,
    "max_ball_residual_mm": 15.0,
}
BALL_ARGS = {
    "poses": 14,
    "heldout": 4,
    "radius_mm": 30.5,
    "tilt_deg": 15.0,
    "yaw_deg": 25.0,
    "spread_mm": 50.0,
    "lift_mm": 10.0,
    "camera": "camera",
    "frames": 3,
    "settle_s": 0.5,
    "speed_pct": 10.0,
}


class _Frames:
    """A fixed camera whose held mount and last record can differ."""

    def __init__(self, *, installed: Transform | None, measured: Transform | None) -> None:
        self._installed, self._measured = installed, measured

    def transform(self, name: str) -> Transform | None:
        return self._installed

    def measured(self, name: str) -> Transform | None:
        return self._measured

    def parent(self, name: str) -> str:
        return "base"

    def procedure(self, name: str) -> str:
        return "handeye.fixed"


def _ctx(frames: _Frames) -> Any:
    # The flange sits one tool length (126.5 mm) above the TCP along the tool axis, as
    # the cell's own record has it for a top-down tool.
    async def state() -> Any:
        return SimpleNamespace(
            enabled=True,
            tcp=Pose(249.3, -39.4, 120.2, 180.0, 0.0, 0.0),
            flange=Pose(249.3, -39.4, 246.7, 180.0, 0.0, 0.0),
        )

    return SimpleNamespace(
        workcell=SimpleNamespace(
            workcell_id="bench_cell", role=lambda name: "cam_fixed", calibration=ACCEPT
        ),
        frames=frames,
        motion=SimpleNamespace(state=state, check_target=lambda pose: None),
    )


def _plan(frames: _Frames):
    return asyncio.run(CarriedBall().plan(_ctx(frames), BALL_ARGS))


def test_an_inexact_mount_still_seeds_the_sweep_from_its_last_measurement() -> None:
    """``calib.ball`` is offered for an inaccurate or drifted camera.

    Such a frame publishes no transform, so refusing to start without one makes the
    skill refuse on exactly the state its ``provides_when`` advertises. The seed is
    only a search hint -- every centre is fitted independently -- so the record's
    geometry is the right starting point, and the plan says it was unproven.
    """
    plan = _plan(_Frames(installed=None, measured=LAST_RECORDED))
    assert plan.basis["seed_unproven"] is True
    assert plan.basis["seed_translation_mm"] == [
        round(value, 3) for value in LAST_RECORDED.translation
    ]


def test_a_mount_that_holds_is_preferred_over_the_last_record() -> None:
    plan = _plan(_Frames(installed=INSTALLED, measured=LAST_RECORDED))
    assert plan.basis["seed_unproven"] is False
    assert plan.basis["seed_translation_mm"] == [round(value, 3) for value in INSTALLED.translation]


def test_without_any_measurement_there_is_nothing_to_refine() -> None:
    with pytest.raises(SkillError, match="no measured mount"):
        _plan(_Frames(installed=None, measured=None))


def test_the_search_guess_is_carried_into_the_camera_frame() -> None:
    """The record points camera -> base, so a TCP goes back through the inverse.

    Applying the mount forwards aimed the first still's search tens of centimetres
    away from the ball -- outside every window -- so the sweep found nothing and
    every pose was skipped.
    """
    record = MOUNT.inverse()  # what ``frames.transform`` publishes for a fixed camera
    tcp = Pose(249.3, -39.4, 120.2, 180.0, 0.0, 0.0).position
    seen = _as_seen(record, tcp)
    assert norm(sub(seen, MOUNT.apply(tcp))) < 1e-9
    # Even this near-vertical mount puts the two readings most of a ball apart, and
    # on the cell they are hundreds of millimetres apart -- far outside any window.
    assert norm(sub(seen, record.apply(tcp))) > 50.0


def _travel(points: list[tuple[float, ...]]) -> float:
    """How far apart the two furthest of *points* are."""
    return max(norm(sub(a, b)) for a in points for b in points)


def _ball_travel(poses: list[Pose], radius: float = 30.5) -> float:
    """*travel* of a ball one radius past the jaws along each pose's tool axis."""
    ball: list[tuple[float, ...]] = []
    for pose in poses:
        matrix = rpy_to_matrix(pose.rx, pose.ry, pose.rz)
        axis = (matrix[0][2], matrix[1][2], matrix[2][2])
        ball.append(tuple(value + radius * step
                          for value, step in zip(pose.position, axis)))
    return _travel(ball)


def test_the_ring_turns_the_ball_about_the_flange_not_the_tcp() -> None:
    """The mount is read off how far the ball travels, so the ring must move it.

    A carried ball hangs 30-60 mm past the jaws. Built about the TCP, the tool
    turns while the ball travels only that offset, and the mount's 3 sigma came out
    at 171 mm on the 24-pose sweep of 2026-11-26. Built about the flange the ball
    travels on its full lever, which is the whole point of the sweep. Both rings
    are built here from the same posture, so the comparison is the pivot alone.
    """
    start = Pose(249.3, -39.4, 120.2, 180.0, 0.0, 0.0)
    flange = Pose(249.3, -39.4, 246.7, 180.0, 0.0, 0.0)
    still = {"spread_mm": 0.0, "lift_mm": 0.0, "yaw_deg": 0.0}
    about_tcp = _ring(start, start.position, 8, tilt=15.0, args=still)
    about_flange = _ring(start, flange.position, 8, tilt=15.0, args=still)
    assert _ball_travel(about_flange) > 3.0 * _ball_travel(about_tcp)
    # And the plan the skill builds is the flange one: its poses leave the TCP.
    plan = _plan(_Frames(installed=INSTALLED, measured=None))
    assert _travel([pose.position for pose in plan.targets[:-1]]) > 60.0


# ── Provenance ───────────────────────────────────────────────────────────────


def _evidence(**overrides) -> dict:
    evidence = {
        "method": METHOD_CARRIED_BALL,
        "radius_mm": 32.0,
        "samples": 12,
        "held_out": 4,
        "held_out_p95_mm": 1.8,
        "held_out_max_mm": 2.4,
        "residual_rms_mm": 0.9,
    }
    evidence.update(overrides)
    return evidence


def test_a_carried_ball_record_is_valid_only_with_the_bound_it_measured() -> None:
    record = SimpleNamespace(
        parameters={
            "method": METHOD_CARRIED_BALL,
            "carried_ball_evidence": _evidence(),
            "achieved_accuracy": {"translation_mm": 2.6, "rotation_deg": 0.4},
        },
        matrix=None,
    )
    verdict = judge(record, mounted=True)
    assert verdict.status is FrameStatus.VALID
    assert verdict.achieved == Accuracy(2.6, 0.4)


def test_a_carried_ball_record_without_held_out_poses_states_no_accuracy() -> None:
    assert carried_ball_problem(_evidence(held_out=0, samples=12))
    assert carried_ball_problem(_evidence(residual_rms_mm=None))
    record = SimpleNamespace(
        parameters={"method": METHOD_CARRIED_BALL, "carried_ball_evidence": _evidence()},
        matrix=None,
    )
    assert judge(record, mounted=True).status is FrameStatus.UNPROVEN
    assert CarriedBallEvidence().applies(record.parameters)
    assert not CarriedBallEvidence().applies({"target_mount": None})


def test_the_skill_declares_what_it_needs_and_what_it_replaces() -> None:
    spec = CarriedBall.spec
    assert spec.skill_id == "calib.ball"
    assert "heldout" in spec.args_schema["properties"]
    assert "calib:{camera}" in spec.provides

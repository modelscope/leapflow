# Copyright (c) Alibaba, Inc. and its affiliates.
"""calib.screws: the board-free mount, over a simulated cell whose mount is known.

The archive replay (``temp/probe-b3/pb_replay.py``) proves the solver on frames
from the real cell; this file proves the *skill* -- plan, capture, gates, record
-- over a scene built from a mount it chooses, so the right answer is known
independently of the code under test.

The scene is what a fixed camera in this cell sees: a table, a column on the base
axis that no joint carries, and a distal assembly rigidly attached to the flange.
Every point is rendered through one ``T_base_camera``, so each joint's motion in
the image is exactly the firmware's screw seen through that mount.  The vertical
comes from the base joint's own swing, so where the camera stands no longer has
to make the table face image-down: a camera at the cell's edge and one on the far
side of the cell are held to the same answer.

Regenerating a scene at 8 000 points takes a few milliseconds, which is why the
refusals are driven by making the scene misbehave rather than by faking the
skill's inputs: a frozen camera, a camera that slips between stills, and an arm
that cannot swing about the base axis.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from leapflow.hardware.skills import vision as vision_module
from leapflow.hardware.skills.builtin.camera_calibration import (
    METHOD_SCREW,
    skills as camera_skills,
)
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    angle_between_deg,
    norm,
    rotation_distance_deg,
    sub,
)
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_OUT_OF_WORKSPACE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_UNREACHABLE,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    CloudObservation,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.screw import (
    JointEvidence,
    Occupancy,
    ScrewConfig,
    rotation_about,
    screw_from_poses,
    solve_vertical,
)
from tests._harness.skill_stubs import (
    CAMERA_ID,
    LOCATOR_ID,
    PERCEPTION_WORKCELL_YAML,
    SAFE_BOX,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    SimArmTransport,
    SkillNodeRegistry,
    piper_context,
    write_workcell,
)

# A downward-facing fixture inside the cell's reach, as the other node-skill
# tests use: j1 is vertical about the base axis, j4 and j5 horizontal, which is
# the set the real capture swings.
START_JOINTS = (0.0, 80.0, -55.0, 69.99, 10.0, 0.0)
CAMERA_POSITION = (430.0, 520.0, 330.0)
CAMERA_TARGET = (200.0, 0.0, 180.0)
TABLE_MM = 0.0
# One frame per still: the multi-frame average is exercised separately, and a
# refusal should not pay for seven of them.
FAST = {"frames": 1, "settle_s": 0.0, "excursion_deg": 15.0}
CELL: dict[str, Any] = {
    **PERCEPTION_WORKCELL_YAML,
    "workspace": {
        **PERCEPTION_WORKCELL_YAML["workspace"],
        # The pre-calibration box. Only a skill that declares itself planning
        # there -- and only while the workspace is unmeasured -- is held to it:
        # the measured box is what the cell is actually run in.
        "bootstrap_box": {"x": [80, 450], "y": [-250, 250], "z": [60, 450]},
    },
}
# A stand posture inside the sim's measured box that j1, j4 and j5 may all swing
# from, so the measurement does not depend on where the sim happens to sit.
STAND = {"j2": 84.0, "j3": -60.0}
CHAIN = DhChain.from_mapping(piper_context().kinematics.dh)
TOOL = Transform.from_pose(Pose(0.0, 0.0, TOOL_LENGTH_MM, 0.0, 0.0, 0.0))


def _tcp(joints: Sequence[float]) -> Pose:
    """The tool centre point the sim's flange and the cell's TCP offset put a joint vector at."""
    return Transform.from_pose(CHAIN.forward(list(joints))).compose(TOOL).to_pose()


SIM_INTRINSICS = {"width": 640, "height": 480, "fx": 900.0, "fy": 900.0,
                  "ppx": 320.0, "ppy": 240.0}


# ── The simulated cell ───────────────────────────────────────────────────────


def true_mount(
    position: Sequence[float] = CAMERA_POSITION, target: Sequence[float] = CAMERA_TARGET
) -> Transform:
    """``T_base_camera`` for a camera at the cell's edge, image-up along world up.

    The rotation's columns are the camera's axes in the base frame, so its third
    column -- world up in the camera frame -- lands near image-down, which is the
    assumption the measured vertical rests on.
    """
    position = np.asarray(position, float)
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    down = -up - float(-up @ forward) * forward  # the image's own down, untilted
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    axes = np.column_stack([right, down, forward])  # the camera's axes, in the base frame
    return Transform(tuple(map(tuple, axes)), tuple(position))


def to_camera(mount: Transform, base: np.ndarray) -> np.ndarray:
    """Render base-frame points through ``T_base_camera``.

    ``MountFit.camera`` is ``T_base_camera``, so its rotation maps camera axes
    *into* the base frame and the inverse image is ``(p - t) @ R``.
    """
    return (base - np.asarray(mount.translation, float)) @ np.asarray(mount.rotation, float)


def box_surface(
    low: Sequence[float], high: Sequence[float], step: float
) -> np.ndarray:
    """Points on all six faces of an axis-aligned box, about *step* apart."""
    faces = []
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        first = np.arange(low[others[0]], high[others[0]] + 1e-9, step)
        second = np.arange(low[others[1]], high[others[1]] + 1e-9, step)
        grid = np.stack(np.meshgrid(first, second, indexing="ij"), axis=-1).reshape(-1, 2)
        for face in (low[axis], high[axis]):
            block = np.zeros((len(grid), 3))
            block[:, axis] = face
            block[:, others[0]] = grid[:, 0]
            block[:, others[1]] = grid[:, 1]
            faces.append(block)
    return np.vstack(faces)


def _table() -> np.ndarray:
    xs = np.arange(-250.0, 550.0, 15.0)
    ys = np.arange(-450.0, 450.0, 15.0)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    return np.column_stack([grid, np.full(len(grid), TABLE_MM)])


# The distal assembly in the flange frame: a wrist block, the forearm behind it
# (the flange's +z points along the tool), and the elbow mass behind that.
DISTAL_LOCAL = np.vstack([
    box_surface((-45.0, -45.0, -40.0), (45.0, 45.0, 40.0), 6.0),
    box_surface((-35.0, -40.0, -240.0), (35.0, 40.0, -40.0), 6.0),
    box_surface((-60.0, -20.0, -330.0), (60.0, 20.0, -240.0), 6.0),
])
COLUMN = box_surface((-70.0, -70.0, TABLE_MM), (70.0, 70.0, 330.0), 12.0)


class SimCell:
    """The camera's measured surface, rendered from the arm's own flange pose.

    *freeze* stops the world after the first frame (a camera that stopped
    updating); *slip_mm* moves the camera after *slip_after* captures (a mount
    that was not fixed); *nudge_mm* moves the arm while a frame is taken;
    *noise_mm* is the depth stream's own jitter (a sensor too noisy to be flat).
    """

    def __init__(
        self,
        *,
        mount: Transform | None = None,
        freeze: bool = False,
        slip_after: int | None = None,
        slip_mm: float = 0.0,
        nudge_mm: float = 0.0,
        noise_mm: float = 0.0,
    ) -> None:
        self.arm = SimArmTransport(joints=START_JOINTS, forward=CHAIN.forward, travel_reads=1)
        self.mount = mount or true_mount()
        self.freeze = freeze
        self.slip_after = slip_after
        self.slip_mm = slip_mm
        self.nudge_mm = nudge_mm
        self.noise_mm = noise_mm
        self.table = _table()
        self.captures = 0
        self.windows: list[tuple[float | None, float | None]] = []
        self._frozen: np.ndarray | None = None
        self._rng = np.random.default_rng(3)

    def camera_position(self) -> tuple[float, float, float]:
        return tuple(self.mount.translation)  # type: ignore[return-value]

    def frame(self) -> np.ndarray:
        """Every point the camera measures right now, in the camera frame."""
        flange = Transform.from_pose(self.arm.flange)
        rotation = np.asarray(flange.rotation, float)
        distal = DISTAL_LOCAL @ rotation.T + np.asarray(flange.translation, float)
        camera = to_camera(self.mount, np.vstack([COLUMN, distal, self.table]))
        if self.noise_mm:
            spread = self._rng.uniform(-self.noise_mm, self.noise_mm, camera.shape)
            camera = camera + spread
        return camera

    async def observe(
        self, *, near_mm: float | None = None, far_mm: float | None = None
    ) -> CloudObservation:
        self.captures += 1
        self.windows.append((near_mm, far_mm))
        if self.freeze and self._frozen is not None:
            points = self._frozen
        else:
            points = self.frame()
            self._frozen = points if self.freeze else None
        if near_mm is not None:
            points = points[points[:, 2] > near_mm]
        if far_mm is not None:
            points = points[points[:, 2] < far_mm]
        if self.slip_after is not None and self.captures > self.slip_after and self.slip_mm:
            # The camera slides in the cell, so the slip is a base-frame move.
            moved = np.asarray(self.mount.translation, float) + (0.0, self.slip_mm, 0.0)
            self.mount = Transform(self.mount.rotation, tuple(moved))
        if self.nudge_mm:
            self.arm.pose[0] += self.nudge_mm
        return CloudObservation(
            points=tuple(map(tuple, points)),
            intrinsics=SIM_INTRINSICS,
            observed_at=time.time(),
        )


def _install(monkeypatch: pytest.MonkeyPatch, sim: SimCell) -> None:
    """Serve the sim's cloud from the production vision port, the handeye way."""

    async def observe(
        self: Any, *, near_mm: float | None = None, far_mm: float | None = None
    ) -> CloudObservation:
        return await sim.observe(near_mm=near_mm, far_mm=far_mm)

    monkeypatch.setattr(vision_module.WorkcellVision, "observe_cloud", observe)


def _cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sim: SimCell | None = None,
    data: Mapping[str, Any] | None = None,
    box: Mapping[str, Any] | None = SAFE_BOX,
) -> tuple[SkillRuntime, SimCell]:
    sim = sim or SimCell()
    _install(monkeypatch, sim)
    registry = SkillNodeRegistry(
        sim.arm,
        store_path=tmp_path / "instrument.duckdb",
        devices={CAMERA_ID: object(), LOCATOR_ID: object()},
    )
    registry.calibrate_cell(box)
    directory = tmp_path / "workcells"
    write_workcell(directory, dict(data or CELL))
    runtime = SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory),
        skills=camera_skills(),
        poll_s=0.005,
    )
    return runtime, sim


def _store(runtime: SkillRuntime) -> Any:
    return runtime._registry.calibration_store  # type: ignore[attr-defined]


def _frames(runtime: SkillRuntime) -> FrameGraph:
    spec = runtime._workcells[WORKCELL_ID]  # type: ignore[attr-defined]
    return FrameGraph.snapshot(spec, _store(runtime))


async def _plan(runtime: SkillRuntime, args: Mapping[str, Any]) -> SkillPlan:
    plan = await runtime.plan(WORKCELL_ID, "calib.screws", args)
    assert isinstance(plan, SkillPlan), plan
    return plan


async def _run(runtime: SkillRuntime, args: Mapping[str, Any]) -> SkillOutcome:
    plan = await _plan(runtime, args)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=120.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


async def _refusal(runtime: SkillRuntime, args: Mapping[str, Any]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, "calib.screws", args)
    assert isinstance(outcome, SkillOutcome) and not outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    return outcome


# ── The measurement itself ───────────────────────────────────────────────────


def test_screw_from_poses_recovers_the_axis_of_a_single_joint() -> None:
    """One joint moved: the axis, the angle, and a point on the line come back."""
    axis = (0.0, 1.0, 0.0)
    pivot = (250.0, -30.0, 335.0)
    turn = rotation_about(axis, math.radians(30.0))
    first = Pose(250.0, 0.0, 250.0, 180.0, 0.0, 0.0)
    moved = tuple(
        pivot[i] + sum(turn[i][j] * (first.position[j] - pivot[j]) for j in range(3))
        for i in range(3)
    )
    second = Pose(*moved, 180.0, 30.0, 0.0)
    screw = screw_from_poses(first, second)
    assert math.degrees(screw.angle) == pytest.approx(30.0, abs=1e-6)
    assert abs(screw.axis[1]) == pytest.approx(1.0, abs=1e-9)
    # Every point on the axis line is a fixed point of the motion.
    line = np.asarray(screw.point, float) + 40.0 * np.asarray(screw.axis, float)
    lifted = turn @ (line - np.asarray(pivot, float)) + np.asarray(pivot, float)
    assert np.linalg.norm(lifted - line) < 1e-6


def test_screw_from_poses_refuses_a_pair_that_did_not_move() -> None:
    pose = Pose(250.0, 0.0, 250.0, 180.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="too little"):
        screw_from_poses(pose, pose)


def scene(joints: Sequence[float], mount: Transform | None = None) -> np.ndarray:
    """Everything the camera measures with the arm at *joints*, in camera coordinates."""
    flange = Transform.from_pose(CHAIN.forward(list(joints)))
    rotation = np.asarray(flange.rotation, float)
    distal = DISTAL_LOCAL @ rotation.T + np.asarray(flange.translation, float)
    return to_camera(mount or true_mount(), np.vstack([COLUMN, distal, _table()]))


def true_vertical(mount: Transform | None = None) -> tuple[float, float, float]:
    """Base +z in camera coordinates: the third *row* of ``T_base_camera``."""
    rotation = np.asarray((mount or true_mount()).rotation, float)
    return tuple(rotation.T @ np.array([0.0, 0.0, 1.0]))  # type: ignore[return-value]


def pair(*, joint: int, excursion: float = 15.0, mount: Transform | None = None) -> JointEvidence:
    """One joint's two stills, rendered from the cell's own mount."""
    plus = list(START_JOINTS)
    minus = list(START_JOINTS)
    plus[joint] += excursion
    minus[joint] -= excursion
    motion = screw_from_poses(CHAIN.forward(plus), CHAIN.forward(minus))
    return JointEvidence(
        f"j{joint + 1}", motion, scene(plus, mount), scene(minus, mount), ScrewConfig()
    )


def test_solve_vertical_reads_the_base_joints_own_swing() -> None:
    """j1 turns about base +z, so its motion *is* the vertical in the image.

    The tolerance is the cloud's, not the algebra's: the voxel score saturates
    once a degree of error displaces the arm by less than the 9 mm grid
    tolerates, so the direction stops on a plateau whose width depends on where
    the camera stands (measured 0.24 deg at the cell's edge, 1.2 deg from the far
    side).  Sharper grids were tried and lose more to sparsity than they gain.
    """
    up = solve_vertical([pair(joint=0)])
    assert angle_between_deg(up, true_vertical()) < 0.5
    # A camera on the far side of the cell: nothing about the view is assumed.
    other = true_mount((380.0, -560.0, 380.0), (200.0, 0.0, 150.0))
    sideways = solve_vertical([pair(joint=0, mount=other)])
    assert angle_between_deg(sideways, true_vertical(other)) < 1.5


def test_solve_vertical_refuses_a_pair_that_does_not_turn_about_it() -> None:
    """A wrist pair measures the wrist's axis, not the base's, so it is refused."""
    with pytest.raises(ValueError, match="no joint turns about the base vertical"):
        solve_vertical([pair(joint=4)])


def test_occupancy_counts_a_point_beyond_the_grid_as_a_miss() -> None:
    """The boundary skin the prototype scored on: outside is nothing, not a hit."""
    np_ = np
    surface = np_.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0],
                         [60.0, 60.0, 0.0]], dtype=float)
    grid = Occupancy(surface, 3.0)
    assert grid.score(surface) == pytest.approx(1.0)
    assert grid.score(surface + 2.0) == pytest.approx(1.0)
    far = surface + np_.array([0.0, 0.0, 300.0])
    assert grid.score(far) == 0.0
    # A shift that leaves the measured box lands nothing, whatever its direction.
    assert grid.score(surface + np_.array([400.0, 0.0, 0.0])) == 0.0
    half = np_.vstack([surface, surface + np_.array([0.0, 0.0, 300.0])])
    assert Occupancy(half, 3.0).score(half) == pytest.approx(1.0)


def test_measure_shift_converges_past_a_window_that_would_truncate_it() -> None:
    """A quarter-metre shift, with a seed tens of millimetres off, still lands.

    This is the archive's j4 defect in miniature: the wrist's axis passes far
    from the surface it carries, so the shift it asks for is large; a search
    bounded by a fixed window stopped on the boundary and reported the rest as
    joint disagreement.
    """
    np_ = np
    axis = (0.0, 1.0, 0.0)
    pivot = np_.array([250.0, 0.0, 430.0])
    turn = rotation_about(axis, math.radians(25.0))
    body = DISTAL_LOCAL * 0.6
    body = body[np_.all(body > -400.0, axis=1)]
    shifted = body @ np_.asarray(turn, float).T + pivot - np_.asarray(turn, float) @ pivot
    # A lump only one pose shows biases the centroid seed the search starts from.
    lump = np_.array([[120.0, 0.0, -700.0]]) + np_.random.default_rng(1).normal(
        0.0, 12.0, (300, 3)
    )
    first = np_.vstack([body, lump])
    identity = screw_from_poses(Pose(250.0, 0.0, 250.0, 180.0, 0.0, 0.0),
                                Pose(250.0, 0.0, 250.0, 180.0, 25.0, 0.0))
    evidence = JointEvidence("j4", identity, first, shifted, ScrewConfig())
    seed = evidence.centroid_b - evidence.moved_a.mean(axis=0)
    measured, score = evidence.measure_shift(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    truth = pivot - np_.asarray(turn, float) @ pivot
    assert np_.linalg.norm(seed - truth) > 30.0  # the seed alone would not do
    assert np_.linalg.norm(np_.asarray(measured, float) - truth) < 5.0
    assert score > 0.85


# ── End to end through the node runtime ──────────────────────────────────────


async def test_screws_measures_the_mount_it_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch)
    flange = sim.arm.flange
    plan = await _plan(runtime, FAST)
    assert sim.captures == 0 and sim.arm.batches == [], "planning captures and moves nothing"
    assert [move["joint"] for move in plan.basis["moves"]] == ["j1", "j4", "j5"]
    start = Pose.from_mapping(plan.basis["start"])
    assert len(plan.targets) == 6 + 1 and plan.targets[-1].distance_mm(start) < 0.01
    assert "7 stills" in plan.summary

    outcome = await _run(runtime, FAST)
    assert outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.result["method"] == METHOD_SCREW
    assert outcome.result["captures"] == 7 and outcome.result["frames"] == 1
    assert outcome.result["score"] > 0.9 and outcome.result["max_shift_gap_mm"] < 5.0
    assert angle_between_deg(outcome.result["vertical"], true_vertical()) < 0.5
    assert len(outcome.result["pairs"]) == 3
    assert all(pair["agreement"] > 0.9 for pair in outcome.result["pairs"])
    # The arm went back to where it started between swings and at the end.
    assert sim.arm.flange.distance_mm(flange) < 0.01

    measured = _frames(runtime).transform("camera")
    assert measured is not None, "the gate refused a record the skill just wrote"
    assert norm(sub(measured.translation, true_mount().translation)) < 5.0
    assert rotation_distance_deg(measured.rotation, true_mount().rotation) < 1.0

    record = _store(runtime).latest(WORKCELL_ID, "camera.handeye")
    assert record is not None and record.matrix is not None
    parameters = record.parameters
    assert parameters["method"] == METHOD_SCREW
    assert parameters["mounting"] == "eye_to_hand"
    assert parameters["intrinsics"] == SIM_INTRINSICS
    assert parameters["near_mm"] == 200.0 and parameters["far_mm"] == 750.0
    evidence = parameters["screw_evidence"]
    assert evidence["method"] == METHOD_SCREW and len(evidence["pairs"]) == 3
    assert angle_between_deg(evidence["vertical"], true_vertical()) < 0.5
    # No plane of the room is part of the record: the vertical came from the arm.
    assert "plane" not in evidence and "table_z_mm" not in evidence
    # The depth window reached the camera, so the cloud was built on the arm.
    assert sim.windows == [(200.0, 750.0)] * 7
    assert sim.captures == 7


async def test_screws_averages_several_frames_per_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch)
    outcome = await _run(runtime, {**FAST, "frames": 3})
    assert outcome.ok, outcome
    assert outcome.result["frames"] == 3 and outcome.result["captures"] == 7
    assert sim.captures == 21


async def test_screws_accepts_a_mount_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A camera on the other side of the cell: nothing about the mount is assumed."""
    other = true_mount((380.0, -560.0, 380.0), (200.0, 0.0, 150.0))
    runtime, sim = _cell(tmp_path, monkeypatch, sim=SimCell(mount=other))
    outcome = await _run(runtime, FAST)
    assert outcome.ok, outcome
    measured = _frames(runtime).transform("camera")
    assert measured is not None and sim.captures == 7
    assert norm(sub(measured.translation, other.translation)) < 5.0
    assert rotation_distance_deg(measured.rotation, other.rotation) < 1.0


# ── Refusals ─────────────────────────────────────────────────────────────────


async def test_screws_refuses_a_wrist_camera_before_anything_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = {
        **CELL,
        "frames": {**CELL["frames"], "camera": {"parent": "flange", "procedure": "camera.handeye"}},
    }
    runtime, sim = _cell(tmp_path, monkeypatch, data=data)
    outcome = await _refusal(runtime, FAST)
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "fixed to the cell" in outcome.message
    assert sim.captures == 0 and sim.arm.batches == []


async def test_screws_plans_in_the_measured_workspace_not_the_bootstrap_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measuring the camera does not re-measure the workspace, so the measured box applies.

    This cell declares a bootstrap box the swings would leave and holds a measured
    box they stay inside. A skill that planned against the bootstrap box could not
    reach a posture the measurement needs; the workspace record is what the cell is
    actually run in, and the camera mount does not change it.
    """
    data = {
        **CELL,
        "workspace": {**CELL["workspace"], "bootstrap_box": {
            "x": [240, 280], "y": [-20, 20], "z": [100, 130]}},
    }
    runtime, sim = _cell(tmp_path, monkeypatch, data=data)
    plan = await _plan(runtime, FAST)
    assert [move["joint"] for move in plan.basis["moves"]] == ["j1", "j4", "j5"]
    outcome = await _run(runtime, FAST)
    assert outcome.ok, outcome


async def test_screws_refuses_a_cell_where_the_joints_cannot_swing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No workspace record yet: only the bootstrap box may hold the swings, and it is too small."""
    data = {
        **CELL,
        "workspace": {**CELL["workspace"], "bootstrap_box": {
            "x": [240, 280], "y": [-20, 20], "z": [100, 130]}},
    }
    runtime, sim = _cell(tmp_path, monkeypatch, data=data, box=None)
    outcome = await _refusal(runtime, FAST)
    assert outcome.failure_code == FAILURE_OUT_OF_WORKSPACE
    assert "only 0 of 3" in outcome.message and "stand posture" in outcome.message
    assert sim.arm.batches == []


async def test_screws_refuses_a_measured_workspace_the_swings_would_leave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A measured box that confines the arm outranks the cell's wider bootstrap box."""
    tight = {"x_min": 240.0, "x_max": 280.0, "y_min": -20.0, "y_max": 20.0,
             "z_min": 100.0, "z_max": 130.0}
    runtime, sim = _cell(tmp_path, monkeypatch, box=tight)
    outcome = await _refusal(runtime, FAST)
    assert outcome.failure_code == FAILURE_OUT_OF_WORKSPACE
    assert "only 0 of 3" in outcome.message
    assert sim.arm.batches == []


async def test_screws_transits_to_the_stand_posture_before_measuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the arm stands is part of the plan: it moves there, then measures there."""
    runtime, sim = _cell(tmp_path, monkeypatch)
    plan = await _plan(runtime, {**FAST, "stand": STAND})
    assert plan.basis["stand"] == {"joint.j2.position": 84.0, "joint.j3.position": -60.0}
    assert "stand posture (j2 84, j3 -60)" in plan.summary
    # Every swing starts from the stand posture, not from where the arm happened to be.
    for move in plan.basis["moves"]:
        joint = int(move["joint"][1]) - 1
        assert move["start_deg"] == pytest.approx(
            {1: 84.0, 2: -60.0}.get(joint, START_JOINTS[joint])
        )
    reachable = Pose.from_mapping(plan.basis["stand_tcp"])
    assert reachable.distance_mm(_tcp([0.0, 84.0, -60.0, 69.99, 10.0, 0.0])) < 0.01

    outcome = await _run(runtime, {**FAST, "stand": STAND})
    assert outcome.ok, outcome
    assert sim.arm.joints["joint.j2.position"] == 84.0
    assert sim.arm.joints["joint.j3.position"] == -60.0
    # The arm measures from the stand posture and stays there, so the run ends there.
    stand_flange = CHAIN.forward([0.0, 84.0, -60.0, 69.99, 10.0, 0.0])
    assert sim.arm.flange.distance_mm(stand_flange) < 0.01
    assert outcome.result["captures"] == 7


async def test_screws_refuses_a_stand_posture_the_arm_cannot_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transit is checked by the joint-move policy, like any other move."""
    runtime, sim = _cell(tmp_path, monkeypatch)
    outcome = await _refusal(runtime, {**FAST, "stand": {"j5": 60.0}})
    assert outcome.failure_code == FAILURE_UNREACHABLE
    assert "cannot stand where the plan asks" in outcome.message
    assert sim.arm.batches == []


async def test_screws_refuses_a_stand_posture_that_is_not_joint_angles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch)
    unknown = await _refusal(runtime, {**FAST, "stand": {"j9": 10.0}})
    assert unknown.failure_code == FAILURE_INVALID_ARGS and "j9" in unknown.message
    text = await _refusal(runtime, {**FAST, "stand": {"j2": "up"}})
    assert text.failure_code == FAILURE_INVALID_ARGS and "degrees" in text.message
    assert sim.arm.batches == []


async def test_screws_refuses_fewer_than_three_joints_before_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch)
    outcome = await _refusal(runtime, {**FAST, "joints": ["j1", "j4"]})
    assert outcome.failure_code == FAILURE_INVALID_ARGS
    assert sim.arm.batches == []


async def test_screws_refuses_a_camera_that_stopped_updating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch, sim=SimCell(freeze=True))
    outcome = await _run(runtime, FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert "the camera stopped updating" in outcome.message
    assert _frames(runtime).transform("camera") is None


async def test_screws_refuses_a_mount_that_moved_between_captures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(
        tmp_path, monkeypatch, sim=SimCell(slip_after=2, slip_mm=20.0)
    )
    outcome = await _run(runtime, FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert "no single mount explains the joints" in outcome.message
    assert sim.captures == 7  # every still was taken, and the solve refused them
    assert _frames(runtime).transform("camera") is None


async def test_screws_refuses_joints_that_cannot_measure_the_vertical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three horizontal joints measure no vertical, so the run is refused, not guessed.

    The vertical is the base joint's own axis; a cell whose base joint cannot
    swing has no measurement of it, and a mount fitted against a tilted vertical
    would be confidently wrong.
    """
    runtime, sim = _cell(tmp_path, monkeypatch)
    outcome = await _run(runtime, {**FAST, "joints": ["j4", "j5", "j6"]})
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert "no joint turns about the base vertical" in outcome.message
    assert outcome.result["problems"] == ["no joint measures the vertical"]
    assert _frames(runtime).transform("camera") is None


async def test_screws_refuses_a_capture_the_arm_moved_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, sim = _cell(tmp_path, monkeypatch, sim=SimCell(nudge_mm=6.0))
    outcome = await _run(runtime, FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.result["problems"] == ["capture drift"]
    assert "moved during the reference capture" in outcome.message
    assert _frames(runtime).transform("camera") is None
    assert sim.captures == 1


def test_a_pair_is_refused_when_another_joint_moved_with_it() -> None:
    """The drift check the skill runs on every pair, in the unit."""
    from leapflow.hardware.skills.builtin.camera_calibration import _other_joints_moved

    first = {"joint.j1.position": 0.0, "joint.j4.position": 70.0, "joint.j5.position": 10.0}
    assert _other_joints_moved(first, {**first, "joint.j1.position": 30.0},
                               "joint.j1.position") == []
    assert _other_joints_moved(first, {**first, "joint.j1.position": 30.0,
                                       "joint.j5.position": 12.0},
                               "joint.j1.position") == ["joint.j5.position"]
    # Below the tolerance is readback noise, not a second joint moving.
    assert _other_joints_moved(first, {**first, "joint.j4.position": 70.3},
                               "joint.j1.position") == []

# Copyright (c) Alibaba, Inc. and its affiliates.
"""calib.handeye and calib.verify against a simulated camera and board.

``SimBoardCamera`` places the board where the sim arm and the true mounts put
it. Most tests replace the ChArUco detector with that ground truth, so they pin
the skills' maths and verdicts exactly and run without OpenCV; one test renders
real ChArUco images and runs the production detector end to end. Also pins the
hand-eye solver, the board declaration, and workcell admission of it.
"""

from __future__ import annotations

import math
import time
from array import array
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills import vision as vision_module
from leapflow.hardware.skills.builtin.camera_calibration import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    VERIFY_PROCEDURE,
)
from leapflow.hardware.skills.builtin.camera_calibration import skills as camera_skills
from leapflow.hardware.skills.frames import (
    FrameGraph,
)
from leapflow.hardware.skills.geometry import (
    Pose,
    Transform,
    dot,
    mat_mul,
    nearest_rotation,
    norm,
    rotation_distance_deg,
    rotation_vector,
    rpy_to_matrix,
    solve_hand_eye,
    sub,
    tool_cone_deg,
    transpose,
)
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    FAILURE_CHECK_FAILED,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.skills.runtime import SkillRun, SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.vision import BoardDetection, CharucoTarget
from tests._harness.skill_stubs import (
    CAMERA_ID,
    CAMERA_POSE,
    SIM_INTRINSICS,
    TOOL_LENGTH_MM,
    WORKCELL_ID,
    WORKCELL_YAML,
    SimArmTransport,
    SkillNodeRegistry,
    write_workcell,
)

BOARD = {"squares_x": 5, "squares_y": 4, "square_mm": 20, "marker_mm": 15}
TARGET = CharucoTarget.from_mapping(BOARD)
# Fixed camera: the board hangs 10 mm below the TCP, centred on the tool axis.
FLANGE_BOARD = Transform.from_pose(Pose(-50.0, -40.0, TOOL_LENGTH_MM + 10.0))
BASE_CAMERA = Transform.from_pose(Pose.from_mapping(CAMERA_POSE))
# Wrist camera: 60 mm off the tool axis, turned 90 deg; the board lies on the table.
FLANGE_CAMERA = Transform.from_pose(Pose(60.0, 0.0, 40.0, 0.0, 0.0, 90.0))
BASE_BOARD = Transform.from_pose(Pose(260.0, 40.0, 60.0, 180.0, 0.0, 0.0))
FAST = {"settle_s": 0}


def _workcell(mounting: str, **extra: Any) -> dict[str, Any]:
    parent = "base" if mounting == EYE_TO_HAND else "flange"
    return {
        **WORKCELL_YAML,
        "roles": {**WORKCELL_YAML["roles"], "camera": CAMERA_ID},
        "frames": {
            **WORKCELL_YAML["frames"],
            "camera": {"parent": parent, "procedure": "camera.handeye"},
        },
        "calibration_target": {"type": "charuco", **BOARD},
        **extra,
    }


class SimBoardCamera:
    """A ``FrameSource`` that sees the board where the arm and the true mounts put it.

    The depth image is the board's plane; the true ``T_camera_board`` rides in
    the frame metadata for the analytic detector. ``render`` draws a real
    ChArUco image instead (needs OpenCV).
    """

    def __init__(
        self,
        arm: SimArmTransport,
        mounting: str,
        *,
        intrinsics: CameraIntrinsics = SIM_INTRINSICS,
        render: bool = False,
    ) -> None:
        self.arm = arm
        self.mounting = mounting
        self.camera = BASE_CAMERA if mounting == EYE_TO_HAND else FLANGE_CAMERA
        self.board = FLANGE_BOARD if mounting == EYE_TO_HAND else BASE_BOARD
        self.intrinsics = intrinsics
        self.render = render
        self.visible_captures: int | None = None
        self.shift_after: tuple[int, Transform] | None = None
        self.captures = 0

    def truth(self) -> Transform:
        flange = Transform.from_pose(self.arm.flange)
        if self.mounting == EYE_TO_HAND:
            return self.camera.inverse().compose(flange.compose(self.board))
        return flange.compose(self.camera).inverse().compose(self.board)

    async def halt(self) -> None:
        return None

    async def capture_rgbd(self) -> RgbdFrame:
        self.captures += 1
        if self.shift_after is not None and self.captures > self.shift_after[0]:
            self.board = self.shift_after[1]
        board = self.truth()
        k = self.intrinsics
        u, v = k.project(board.apply(TARGET.center))
        visible = 0 <= u < k.width and 0 <= v < k.height
        if self.visible_captures is not None and self.captures > self.visible_captures:
            visible = False
        if self.render:
            color, depth = _render(board, k, visible)
        else:
            color, depth = bytes(k.width * k.height * 3), _plane_depth(board, k)
        return RgbdFrame(
            device_id=CAMERA_ID,
            width=k.width,
            height=k.height,
            color=color,
            depth=depth,
            depth_scale_m=0.0001,
            intrinsics=k,
            observed_at=time.time(),
            monotonic_at=time.monotonic(),
            sequence=self.captures,
            metadata={"board": board if visible else None},
        )


def _plane_depth(board: Transform, k: CameraIntrinsics) -> bytes:
    """Depth (0.1 mm units) of the board's plane along every pixel ray."""
    normal = tuple(board.rotation[i][2] for i in range(3))
    reach = dot(normal, board.translation)
    depth = array("H", [0]) * (k.width * k.height)
    for v in range(k.height):
        y = (v - k.ppy) / k.fy
        for u in range(k.width):
            facing = normal[0] * (u - k.ppx) / k.fx + normal[1] * y + normal[2]
            z = reach / facing if facing > 1e-9 else 0.0
            depth[v * k.width + u] = min(65535, round(z * 10.0)) if z > 0 else 0
    return depth.tobytes()


def _render(board: Transform, k: CameraIntrinsics, visible: bool) -> tuple[bytes, bytes]:
    import cv2
    import numpy as np

    aruco = cv2.aruco
    charuco = aruco.CharucoBoard(
        (TARGET.squares_x, TARGET.squares_y),
        TARGET.square_mm,
        TARGET.marker_mm,
        aruco.getPredefinedDictionary(getattr(aruco, TARGET.dictionary)),
    )
    scale = 4.0  # px per mm in the board image
    size = (
        round(TARGET.squares_x * TARGET.square_mm * scale),
        round(TARGET.squares_y * TARGET.square_mm * scale),
    )
    image = charuco.generateImage(size, marginSize=0)
    rotation, translation = np.array(board.rotation), np.array(board.translation)
    camera = np.array([[k.fx, 0.0, k.ppx], [0.0, k.fy, k.ppy], [0.0, 0.0, 1.0]])
    projection = camera @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    # Board image pixel (c, r) shows board point ((c + 0.5) / s, (r + 0.5) / s).
    to_board = np.array([[scale, 0.0, -0.5], [0.0, scale, -0.5], [0.0, 0.0, 1.0]])
    homography = projection @ np.linalg.inv(to_board)
    gray = cv2.warpPerspective(
        image, homography, (k.width, k.height), flags=cv2.INTER_LINEAR, borderValue=255
    )
    if not visible:
        gray[:] = 255
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    normal = rotation[:, 2]
    us, vs = np.meshgrid(np.arange(k.width), np.arange(k.height))
    facing = normal[0] * (us - k.ppx) / k.fx + normal[1] * (vs - k.ppy) / k.fy + normal[2]
    z = np.where(facing > 1e-9, normal @ translation / np.maximum(facing, 1e-9), 0.0)
    depth = np.clip(np.round(z * 10.0), 0, 65535).astype(np.uint16)
    return color.tobytes(), depth.tobytes()


@pytest.fixture
def analytic_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ChArUco detection with the simulation's ground truth."""

    def detect(frame: RgbdFrame, target: CharucoTarget) -> BoardDetection | None:
        board = frame.metadata.get("board")
        return None if board is None else BoardDetection(board, corners=12, reprojection_px=0.05)

    monkeypatch.setattr(vision_module, "detect_board", detect)


def _cell(
    tmp_path: Path,
    mounting: str = EYE_TO_HAND,
    *,
    workcell: Mapping[str, Any] | None = None,
    intrinsics: CameraIntrinsics = SIM_INTRINSICS,
    render: bool = False,
) -> tuple[SkillRuntime, SimBoardCamera]:
    arm = SimArmTransport()
    camera = SimBoardCamera(arm, mounting, intrinsics=intrinsics, render=render)
    registry = SkillNodeRegistry(
        arm, store_path=tmp_path / "instrument.duckdb", devices={CAMERA_ID: camera}
    )
    registry.last_command = lambda device_id, channel_id: None  # type: ignore[method-assign]
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": TOOL_LENGTH_MM})
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell or _workcell(mounting))
    runtime = SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=directory),
        skills=camera_skills(),
        poll_s=0.005,
    )
    return runtime, camera


def _registry(runtime: SkillRuntime) -> SkillNodeRegistry:
    return runtime._registry  # type: ignore[attr-defined] - the test owns the sim


def _frames(runtime: SkillRuntime) -> FrameGraph:
    registry = _registry(runtime)
    spec = runtime._workcells[WORKCELL_ID]  # type: ignore[attr-defined]
    return FrameGraph.snapshot(spec, registry.calibration_store)


async def _plan(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillPlan:
    plan = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(plan, SkillPlan), plan
    return plan


async def _run(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillOutcome:
    plan = await _plan(runtime, skill, args)
    run = await runtime.execute(plan.digest)
    assert isinstance(run, SkillRun), run
    finished = await runtime.wait(run.run_id, timeout_s=20.0)
    assert finished is not None and finished.outcome is not None
    return finished.outcome


async def _refusal(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(outcome, SkillOutcome) and not outcome.ok, outcome
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    return outcome


def _assert_close(actual: Transform, expected: Transform, *, mm: float, deg: float) -> None:
    assert norm(sub(actual.translation, expected.translation)) < mm
    assert rotation_distance_deg(actual.rotation, expected.rotation) < deg


# ── Hand-eye end to end ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mounting", "expected"), [(EYE_TO_HAND, BASE_CAMERA), (EYE_IN_HAND, FLANGE_CAMERA)]
)
async def test_handeye_measures_the_camera_and_verify_confirms_it(
    tmp_path: Path, analytic_detector: None, mounting: str, expected: Transform
) -> None:
    runtime, camera = _cell(tmp_path, mounting)
    outcome = await _run(runtime, "calib.handeye", FAST)
    assert outcome.ok and outcome.side_effect_state == SIDE_EFFECT_COMMITTED, outcome
    assert outcome.result["mounting"] == mounting and outcome.result["skipped"] == 0
    assert outcome.result["fit"]["translation_rms_mm"] < 0.01
    assert "calib.verify" in outcome.message

    frames = _frames(runtime)
    measured = frames.transform("camera")
    assert measured is not None
    _assert_close(measured, expected, mm=0.05, deg=0.01)
    parameters = frames.parameters("camera")
    assert parameters["intrinsics"] == SIM_INTRINSICS.to_dict()
    board = FLANGE_BOARD if mounting == EYE_TO_HAND else BASE_BOARD
    _assert_close(Transform.from_matrix(parameters["target_mount"]), board, mm=0.05, deg=0.01)
    assert Pose(*camera.arm.pose).distance_mm(Pose(250.0, 0.0, 370.0, 180.0)) < 0.01

    verified = await _run(runtime, "calib.verify", FAST)
    assert verified.ok, verified
    assert verified.result["points"] == 9 and verified.result["p95_mm"] < 1.0
    assert verified.result["pnp_p95_mm"] < 0.05
    record = _registry(runtime).calibration_store.latest(WORKCELL_ID, VERIFY_PROCEDURE)
    assert record is not None
    assert record.parameters["depends_on"] == {"camera": frames.versions["camera"]}


async def test_handeye_plan_moves_nothing_and_turns_about_several_axes(
    tmp_path: Path, analytic_detector: None
) -> None:
    runtime, camera = _cell(tmp_path)
    plan = await _plan(runtime, "calib.handeye", {"poses": 12, "tilt_deg": 20})
    assert runtime._skills["calib.handeye"].spec.consent == CONSENT_EACH_RUN  # type: ignore[attr-defined]
    assert camera.arm.batches == [] and camera.captures == 1
    assert plan.basis["samples"] == 12 and len(plan.targets) == 13
    start = Pose.from_mapping(plan.basis["start"])
    assert plan.targets[-1].distance_mm(start) < 1e-9
    samples = plan.targets[:-1]
    assert max(tool_cone_deg(pose) for pose in samples) == pytest.approx(20.0, abs=0.01)
    assert min(tool_cone_deg(pose) for pose in samples) == pytest.approx(10.0, abs=0.01)
    # Each pose's turn from the start: tilts about many horizontal axes plus a yaw.
    turns = [
        rotation_vector(mat_mul(pose.rotation, transpose(start.rotation))) for pose in samples
    ]
    azimuths = {round(math.degrees(math.atan2(t[1], t[0])) % 180 / 15) for t in turns}
    assert len(azimuths) >= 3 and any(abs(t[2]) > 0.1 for t in turns)


async def test_handeye_refuses_before_moving_without_a_visible_declared_board(
    tmp_path: Path, analytic_detector: None
) -> None:
    runtime, camera = _cell(tmp_path)
    camera.visible_captures = 0
    hidden = await _refusal(runtime, "calib.handeye", FAST)
    assert hidden.failure_code == FAILURE_TARGET_NOT_FOUND

    bare = dict(_workcell(EYE_TO_HAND))
    del bare["calibration_target"]
    runtime, camera = _cell(tmp_path / "bare", workcell=bare)
    undeclared = await _refusal(runtime, "calib.handeye", FAST)
    assert undeclared.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "calibration_target" in undeclared.message
    assert camera.arm.batches == [] and camera.captures == 0


async def test_handeye_fails_committed_when_the_board_slips(
    tmp_path: Path, analytic_detector: None
) -> None:
    runtime, camera = _cell(tmp_path)
    slipped = Transform.from_pose(Pose(-42.0, -40.0, TOOL_LENGTH_MM + 10.0, 0.0, 0.0, 3.0))
    camera.shift_after = (6, slipped)
    outcome = await _run(runtime, "calib.handeye", FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.result["fit"]["translation_rms_mm"] > 3.0
    assert _frames(runtime).transform("camera") is None

    calibration = {
        "max_hand_eye_translation_rms_mm": 100.0,
        "max_hand_eye_rotation_rms_deg": 100.0,
    }
    tolerant, camera = _cell(
        tmp_path / "tolerant",
        workcell=_workcell(EYE_TO_HAND, calibration=calibration),
    )
    camera.shift_after = (6, slipped)
    accepted = await _run(tolerant, "calib.handeye", FAST)
    assert accepted.ok, accepted


async def test_handeye_fails_committed_when_the_board_is_rarely_seen(
    tmp_path: Path, analytic_detector: None
) -> None:
    runtime, camera = _cell(tmp_path)
    camera.visible_captures = 4  # the plan's look plus three poses
    outcome = await _run(runtime, "calib.handeye", FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.result["skipped"] == 9
    assert _frames(runtime).transform("camera") is None


async def test_verify_needs_a_handeye_measurement(tmp_path: Path, analytic_detector: None) -> None:
    runtime, camera = _cell(tmp_path)
    _registry(runtime).calibrate("camera.handeye", pose=CAMERA_POSE)  # entered by hand
    outcome = await _refusal(runtime, "calib.verify", FAST)
    assert outcome.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "calib.handeye" in outcome.message
    assert camera.arm.batches == []


async def test_verify_catches_a_bumped_camera(tmp_path: Path, analytic_detector: None) -> None:
    runtime, camera = _cell(tmp_path)
    assert (await _run(runtime, "calib.handeye", FAST)).ok
    camera.camera = BASE_CAMERA.compose(Transform.from_pose(Pose(8.0, 6.0, 0.0)))
    outcome = await _run(runtime, "calib.verify", FAST)
    assert outcome.failure_code == FAILURE_CHECK_FAILED
    assert outcome.side_effect_state == SIDE_EFFECT_COMMITTED
    assert outcome.result["p95_mm"] == pytest.approx(10.0, abs=0.5)
    assert _registry(runtime).calibration_store.latest(WORKCELL_ID, VERIFY_PROCEDURE) is None

    tolerant, camera = _cell(
        tmp_path / "tolerant",
        workcell=_workcell(EYE_TO_HAND, calibration={"max_verify_p95_mm": 12.0}),
    )
    assert (await _run(tolerant, "calib.handeye", FAST)).ok
    camera.camera = BASE_CAMERA.compose(Transform.from_pose(Pose(8.0, 6.0, 0.0)))
    accepted = await _run(tolerant, "calib.verify", FAST)
    assert accepted.ok and accepted.result["p95_mm"] == pytest.approx(10.0, abs=0.5)


# ── Real ChArUco images ──────────────────────────────────────────────────────


async def test_handeye_and_verify_on_rendered_charuco_images(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    if not hasattr(getattr(cv2, "aruco", None), "CharucoDetector"):
        pytest.skip("this OpenCV build has no cv2.aruco.CharucoDetector")
    intrinsics = CameraIntrinsics(width=640, height=480, fx=900.0, fy=900.0, ppx=320.0, ppy=240.0)
    runtime, camera = _cell(tmp_path, intrinsics=intrinsics, render=True)
    frame = await camera.capture_rgbd()
    found = vision_module.detect_board(frame, TARGET)
    assert found is not None and found.corners == 12 and found.reprojection_px < 0.5
    _assert_close(found.transform, camera.truth(), mm=1.0, deg=0.5)

    outcome = await _run(runtime, "calib.handeye", FAST)
    assert outcome.ok, outcome
    measured = _frames(runtime).transform("camera")
    assert measured is not None
    _assert_close(measured, BASE_CAMERA, mm=2.0, deg=0.3)
    verified = await _run(runtime, "calib.verify", FAST)
    assert verified.ok and verified.result["p95_mm"] < 2.0, verified


# ── Board declaration and admission ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"type": "apriltag"}, "not supported"),
        ({"color": "red"}, "unknown target fields"),
        ({"legacy": "yes"}, "legacy"),
        ({"marker_mm": 20}, "smaller than square_mm"),
        ({"squares_x": 2}, "3 x 3"),
        ({"squares_x": 5.5}, "whole number"),
        ({"square_mm": True}, "number"),
        ({"dictionary": "ARUCO_5X5"}, "DICT_"),
        ({"min_corners": 13}, "min_corners"),
    ],
)
def test_board_declaration_rejects_bad_values(change: Mapping[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CharucoTarget.from_mapping({**BOARD, **change})


def test_board_declaration_needs_its_dimensions() -> None:
    with pytest.raises(ValueError, match="marker_mm"):
        CharucoTarget.from_mapping({key: v for key, v in BOARD.items() if key != "marker_mm"})
    assert TARGET.center == (50.0, 40.0, 0.0)
    assert CharucoTarget.from_mapping(TARGET.to_dict()) == TARGET


def test_admission_rejects_a_workcell_with_a_bad_board(tmp_path: Path) -> None:
    data = _workcell(EYE_TO_HAND, calibration_target={"type": "charuco", **BOARD, "marker_mm": 25})
    runtime, _ = _cell(tmp_path, workcell=data)
    (note,) = runtime.load_report.rejected
    assert "calibration_target" in note.detail and WORKCELL_ID not in runtime._workcells  # type: ignore[attr-defined]


# ── Solver ───────────────────────────────────────────────────────────────────

_ARM_POSES = [
    Pose(250, 0, 370, 180, 0, 0),
    Pose(270, 10, 360, 165, 5, 20),
    Pose(230, -20, 380, -170, -10, -30),
    Pose(260, 30, 350, 175, 15, 45),
    Pose(240, -10, 365, -165, 8, -60),
]


@pytest.mark.parametrize("mounting", [EYE_TO_HAND, EYE_IN_HAND])
def test_solver_recovers_both_mounts_exactly(mounting: str) -> None:
    arm = [Transform.from_pose(pose) for pose in _ARM_POSES]
    if mounting == EYE_TO_HAND:
        x, z = FLANGE_BOARD, BASE_CAMERA
        camera = [BASE_CAMERA.inverse().compose(a.compose(FLANGE_BOARD)) for a in arm]
    else:
        x, z = FLANGE_CAMERA, BASE_BOARD
        camera = [a.compose(FLANGE_CAMERA).inverse().compose(BASE_BOARD) for a in arm]
        camera = [transform.inverse() for transform in camera]
    fit = solve_hand_eye(arm, camera)
    _assert_close(fit.x, x, mm=1e-6, deg=1e-6)
    _assert_close(fit.z, z, mm=1e-6, deg=1e-6)
    assert fit.translation_max_mm < 1e-6 and fit.rotation_max_deg < 1e-5


def test_solver_refuses_motions_about_one_axis() -> None:
    arm = [Transform.from_pose(Pose(250, 0, 370, 180, 0, rz)) for rz in (0, 20, 40, 60)]
    camera = [BASE_CAMERA.inverse().compose(a.compose(FLANGE_BOARD)) for a in arm]
    with pytest.raises(ValueError, match="single axis"):
        solve_hand_eye(arm, camera)
    with pytest.raises(ValueError, match="at least"):
        solve_hand_eye(arm[:2], camera[:2])


def test_rotation_helpers() -> None:
    truth = rpy_to_matrix(20.0, -35.0, 110.0)
    noisy = [
        [value + 0.01 * ((i * 3 + j) % 5 - 2) for j, value in enumerate(row)]
        for i, row in enumerate(truth)
    ]
    assert rotation_distance_deg(nearest_rotation(noisy), truth) < 1.0
    half_turn = rotation_vector(rpy_to_matrix(180.0, 0.0, 0.0))
    assert norm(half_turn) == pytest.approx(math.pi) and abs(half_turn[0]) == pytest.approx(math.pi)
    small = rotation_vector(rpy_to_matrix(0.0, 0.0, 1e-6))
    assert small[2] == pytest.approx(math.radians(1e-6), rel=1e-3)

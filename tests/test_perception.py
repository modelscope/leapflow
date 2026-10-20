# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local perception and grasp planning, end to end on a simulated cell.

``SimRgbdCamera`` renders boxes on a table from a calibrated, downward-looking
camera and ``SimLocator`` returns the masks it rendered, so these tests drive
the real lifting, memory, and planning code with known ground truth.
Perception never matches labels or breaks ties: every detection is reported
under its own id and callers name the one they want.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from leapflow.hardware.model_contract import Detection
from leapflow.hardware.rgbd import CameraIntrinsics
from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.frames import (
    FrameGraph,
    unmet_requirements,
)
from leapflow.hardware.skills.geometry import (
    Plane,
    Pose,
    Transform,
    angle_delta_deg,
    footprint,
)
from leapflow.hardware.skills.grasp import (
    TOP_DOWN,
    GraspSensing,
    TopDownPlanner,
    TopDownSettings,
    discover_grasp_planners,
)
from leapflow.hardware.skills.perception import (
    PerceptionSettings,
    WorkcellScene,
    lift_detection,
)
from leapflow.hardware.skills.scene_memory import SceneCapture, SceneMemory
from leapflow.hardware.skills.ports import GraspPlanner
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_MISSING,
    FAILURE_CALIBRATION_STALE,
    FAILURE_DEVICE_FAULT,
    FAILURE_INVALID_ARGS,
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    GraspCandidate,
    HeldObject,
    LocatedObject,
    SkillError,
    SkillOutcome,
)
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.workcell import (
    WorkcellError,
    WorkcellSpec,
)
from tests._harness.skill_stubs import (
    ARM_ID,
    CAMERA_ID,
    CAMERA_POSE,
    LOCATOR_ID,
    PERCEPTION_WORKCELL_YAML,
    TABLE_Z,
    WORKCELL_ID,
    SimBox,
    SkillNodeRegistry,
    perception_registry,
    write_workcell,
)

CUP = SimBox("cup", center=(300.0, 20.0), length=60.0, width=30.0, height=50.0)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _spec(**overrides: Any) -> WorkcellSpec:
    return WorkcellSpec.from_mapping({**PERCEPTION_WORKCELL_YAML, **overrides})


def _scene(
    registry: SkillNodeRegistry,
    *,
    spec: WorkcellSpec | None = None,
    memory: SceneMemory | None = None,
    clock: Callable[[], float] | None = None,
    evidence: dict[str, Any] | None = None,
) -> WorkcellScene:
    spec = spec or _spec()
    sink = {} if evidence is None else evidence
    return WorkcellScene(
        registry=registry,
        spec=spec,
        frames=FrameGraph.snapshot(spec, registry.calibration_store),
        motion=None,  # type: ignore[arg-type] - a base-mounted camera never reads the arm
        memory=memory or SceneMemory(clock or _Clock()),
        settings=PerceptionSettings(),
        note=sink.__setitem__,
    )


def _calibrated(tmp_path: Path, *boxes: SimBox, **calibration: Any) -> SkillNodeRegistry:
    registry = perception_registry(tmp_path / "instrument.duckdb", boxes)
    registry.calibrate_perception(**calibration)
    return registry


def _object(**overrides: Any) -> LocatedObject:
    base: dict[str, Any] = dict(
        object_id="o1",
        label="block",
        score=0.9,
        center=(300.0, 0.0, TABLE_Z + 25.0),
        length=60.0,
        width=30.0,
        height=50.0,
        yaw_deg=0.0,
        observed_at=0.0,
    )
    return LocatedObject(**{**base, **overrides})


# ── Geometry ─────────────────────────────────────────────────────────────────


def test_plane_refuses_walls_and_measures_height_above_a_tilt() -> None:
    with pytest.raises(ValueError):
        Plane((1.0, 0.0, 0.0), 0.0)
    tilt = math.radians(10.0)
    plane = Plane((math.sin(tilt), 0.0, math.cos(tilt)), 50.0)
    on_plane = (100.0, 0.0, plane.height(100.0, 0.0))
    assert plane.above(on_plane) == pytest.approx(0.0, abs=1e-9)
    assert plane.above((100.0, 0.0, on_plane[2] + 10.0)) == pytest.approx(10.0 * math.cos(tilt))
    assert Plane.horizontal(60.0).height(1.0, 2.0) == pytest.approx(60.0)


@pytest.mark.parametrize("yaw", [0.0, 30.0, -60.0, 89.0])
def test_footprint_recovers_an_oriented_rectangle(yaw: float) -> None:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    grid = [(a - 30.0, b - 10.0) for a in range(61) for b in range(21)]
    points = [(200.0 + a * c - b * s, -40.0 + a * s + b * c, 0.0) for a, b in grid]
    shape = footprint(points)
    assert shape.center == pytest.approx((200.0, -40.0), abs=1e-6)
    assert (shape.length, shape.width) == pytest.approx((60.0, 20.0), abs=1e-6)
    assert abs(angle_delta_deg(shape.yaw_deg, yaw)) % 180.0 == pytest.approx(0.0, abs=1e-6)


# ── Workcell declarations ────────────────────────────────────────────────────


def test_workcell_parses_models_grasp_and_perception() -> None:
    spec = _spec(grasp={"approach_mm": 60}, perception={"min_points": 20})
    assert spec.model("locator") and spec.model("detector") == ""
    assert spec.grasp_planner == TOP_DOWN
    assert spec.device_ids == (ARM_ID, CAMERA_ID)
    assert spec.actuator_ids == (ARM_ID,)  # a halt never closes the camera
    shown = spec.to_dict()
    assert shown["grasp"] == {"approach_mm": 60, "planner": TOP_DOWN}
    assert shown["perception"] == {"min_points": 20.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"models": {"Locator": "x"}},
        {"models": {"locator": ""}},
        {"grasp": {"planner": "Top Down"}},
        {"perception": {"min_points": "many"}},
    ],
)
def test_workcell_refuses_malformed_model_and_grasp_sections(overrides: Mapping[str, Any]) -> None:
    with pytest.raises(WorkcellError):
        _spec(**overrides)


def test_model_requirement_names_the_missing_binding(tmp_path: Path) -> None:
    spec = WorkcellSpec.from_mapping({**PERCEPTION_WORKCELL_YAML, "models": {}})
    frames = FrameGraph.snapshot(spec, None)
    (reason,) = unmet_requirements(("model:locator",), spec, frames)
    assert "binds no 'locator' model" in reason
    assert unmet_requirements(("model:locator",), _spec(), frames) == []


def test_frame_graph_reads_the_table_and_the_handeye_parameters(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path)
    frames = FrameGraph.snapshot(_spec(), registry.calibration_store)
    assert frames.table is not None and frames.table.height(0.0, 0.0) == pytest.approx(TABLE_Z)
    assert frames.parameters("camera")["intrinsics"]["fx"] == pytest.approx(250.0)
    assert frames.parent("camera") == "base"

    tilted = _calibrated(tmp_path / "tilted", table_z=None)
    tilted.calibrate(
        "workspace.safe_box",
        parameters={
            "x_min": 100,
            "x_max": 450,
            "y_min": -250,
            "y_max": 250,
            "z_min": 80,
            "z_max": 450,
            "floor": {"method": "table_points", "normal": [0.0, 0.0, 1.0], "offset_mm": 42.0},
        },
    )
    table = FrameGraph.snapshot(_spec(), tilted.calibration_store).table
    assert table is not None and table.height(5.0, 5.0) == pytest.approx(42.0)


def _runtime(
    tmp_path: Path, registry: SkillNodeRegistry, workcell: Mapping[str, Any], **kw: Any
) -> SkillRuntime:
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    kw.setdefault("skills", ())
    return SkillRuntime(registry, settings=SkillRuntimeSettings(workcells_dir=directory), **kw)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"grasp": {"planner": "suction"}}, "unknown planner 'suction'"),
        ({"grasp": {"approach_mm": -1}}, "approach_mm must be positive"),
        ({"grasp": {"wiggle": 1}}, "does not understand wiggle"),
        ({"grasp": {"planner": "model"}}, "needs a models.grasp binding"),
        ({"perception": {"min_depth_mm": 900, "max_depth_mm": 800}}, "min_depth_mm"),
        ({"perception": {"focus": 1}}, "does not understand focus"),
        ({"models": {"locator": "ghost"}}, "'ghost' is not declared"),
    ],
)
def test_runtime_refuses_workcells_it_cannot_plan_or_perceive_for(
    tmp_path: Path, overrides: Mapping[str, Any], fragment: str
) -> None:
    registry = _calibrated(tmp_path)
    runtime = _runtime(tmp_path, registry, {**PERCEPTION_WORKCELL_YAML, **overrides})
    (note,) = runtime.load_report.rejected
    assert fragment in note.detail
    assert runtime.list()["workcells"] == []


def test_runtime_accepts_a_plugin_planner_by_name(tmp_path: Path) -> None:
    built: list[Mapping[str, Any]] = []

    def factory(
        gripper: Mapping[str, float], config: Mapping[str, Any], sensing: GraspSensing
    ) -> GraspPlanner:
        built.append(config)
        return TopDownPlanner(gripper, TopDownSettings())

    registry = _calibrated(tmp_path)
    workcell = {**PERCEPTION_WORKCELL_YAML, "grasp": {"planner": "custom", "k": 1}}
    runtime = _runtime(tmp_path, registry, workcell, planners={"custom": factory})
    assert runtime.load_report.admitted == (WORKCELL_ID,)
    assert built == [{"planner": "custom", "k": 1}]
    assert TOP_DOWN in discover_grasp_planners()


# ── Grasp planning ───────────────────────────────────────────────────────────

_PLANNER = TopDownPlanner({"max_width_mm": 70.0}, TopDownSettings())
_TABLE = Plane.horizontal(TABLE_Z)
_HOME = Pose(300.0, 0.0, 250.0, 180.0, 0.0, 0.0)


async def test_top_down_closes_across_the_narrow_side_first() -> None:
    first, second = await _PLANNER.candidates(_object(), table=_TABLE, current=_HOME)
    assert isinstance(first, GraspCandidate) and first.score > second.score
    # Fingers close along tool Y; across the object's Y extent means rz = 0.
    assert (first.grasp.x, first.grasp.y, first.grasp.rx, first.grasp.rz) == pytest.approx(
        (300.0, 0.0, 180.0, 0.0)
    )
    # Top at 110, fingertips 25 mm down it; the approach starts 80 mm above.
    assert first.grasp.z == pytest.approx(85.0)
    assert first.pregrasp.z == pytest.approx(165.0)
    assert first.retreat.z == pytest.approx(165.0)
    assert (first.width_mm, first.open_mm) == (30.0, 45.0)
    assert abs(second.grasp.rz) == pytest.approx(90.0)
    assert second.open_mm == 70.0
    assert first.grasp_id == first.object_id == ""  # the scene names them, not the planner


async def test_top_down_turns_the_short_way_and_never_dips_below_the_table() -> None:
    turned = Pose(300.0, 0.0, 250.0, 180.0, 0.0, 170.0)
    (first, _) = await _PLANNER.candidates(_object(), table=_TABLE, current=turned)
    assert abs(angle_delta_deg(first.grasp.rz, 180.0)) < 1e-9
    flat = _object(center=(300.0, 0.0, TABLE_Z + 3.0), height=6.0)
    (low, _) = await _PLANNER.candidates(flat, table=_TABLE, current=_HOME)
    assert low.grasp.z == pytest.approx(TABLE_Z + TopDownSettings().clearance_mm)


async def test_top_down_offers_nothing_wider_than_the_gripper() -> None:
    wide = _object(length=100.0, width=68.0)
    assert await _PLANNER.candidates(wide, table=_TABLE, current=_HOME) == ()


async def test_top_down_tuning_comes_from_the_declaration() -> None:
    tuned = TopDownPlanner(
        {"max_width_mm": 70.0}, TopDownSettings.from_mapping({"min_slack_mm": 50.0})
    )
    assert await tuned.candidates(_object(), table=_TABLE, current=_HOME) == ()
    square = TopDownPlanner(
        {"max_width_mm": 70.0}, TopDownSettings.from_mapping({"round_tolerance_mm": 40.0})
    )
    assert len(await square.candidates(_object(), table=_TABLE, current=_HOME)) == 1


async def test_top_down_closing_axis_follows_the_declared_gripper() -> None:
    # The jaw axis is the mounted gripper's, so the cell declares it: jaws on tool
    # X (the Piper cell's measured axis) put the grasp a quarter turn from the
    # tool +Y default, which is exactly the difference between gripping an object
    # across its short side and straddling its long one.
    along_x = TopDownPlanner(
        {"max_width_mm": 70.0}, TopDownSettings.from_mapping({"closing_axis_deg": 0.0})
    )
    (first, _) = await along_x.candidates(_object(), table=_TABLE, current=_HOME)
    (default, _) = await _PLANNER.candidates(_object(), table=_TABLE, current=_HOME)
    assert abs(angle_delta_deg(first.grasp.rz, default.grasp.rz)) == pytest.approx(90.0)


# ── Lifting and locating ─────────────────────────────────────────────────────


async def test_locate_lifts_the_mask_into_a_box_on_the_table(tmp_path: Path) -> None:
    evidence: dict[str, Any] = {}
    scene = _scene(_calibrated(tmp_path, CUP), evidence=evidence)
    (cup,) = await scene.locate(["cup"])
    assert cup.object_id == "o1" and cup.label == "cup"
    assert cup.center[:2] == pytest.approx(CUP.center, abs=3.0)
    assert (cup.length, cup.width, cup.height) == pytest.approx((60.0, 30.0, 50.0), abs=4.0)
    assert cup.bottom_z == pytest.approx(TABLE_Z, abs=1e-6)
    assert abs(angle_delta_deg(cup.yaw_deg, 0.0)) % 180.0 < 3.0
    assert cup.points >= PerceptionSettings().min_points
    assert evidence["perception"]["located"] == 1 and evidence["perception"]["model"]
    assert cup.geometry == "mask"


async def test_a_box_grounding_lifts_to_the_same_footprint(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    registry.devices[LOCATOR_ID].geometry = "box"
    (cup,) = await _scene(registry).locate(["cup"])
    assert cup.geometry == "box"
    assert cup.center[:2] == pytest.approx(CUP.center, abs=3.0)
    assert (cup.length, cup.width, cup.height) == pytest.approx((60.0, 30.0, 50.0), abs=4.0)
    assert LocatedObject.from_mapping(cup.to_dict()).geometry == "box"


async def test_a_point_grounding_is_one_surface_point_with_no_extent(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    registry.devices[LOCATOR_ID].geometry = "point"
    (cup,) = await _scene(registry).locate(["cup"])
    assert cup.geometry == "point"
    assert cup.center == pytest.approx((*CUP.center, TABLE_Z + CUP.height), abs=3.0)
    assert (cup.length, cup.width, cup.height) == (0.0, 0.0, 0.0)
    assert cup.points > 1  # the neighbourhood's median, not one pixel
    with pytest.raises(SkillError) as error:
        await _PLANNER.candidates(cup, table=_TABLE, current=_HOME)
    assert error.value.failure_code == FAILURE_INVALID_ARGS


async def test_a_point_without_depth_is_dropped_and_off_frame_geometry_refused(
    tmp_path: Path,
) -> None:
    registry = _calibrated(tmp_path, CUP)
    camera = registry.devices[CAMERA_ID]
    frame = await camera.capture_rgbd()
    blind = replace(frame, depth=bytes(len(frame.depth)))
    capture = SceneCapture(blind, Transform.from_pose(Pose(**CAMERA_POSE)), _TABLE)
    settings = PerceptionSettings()
    with pytest.raises(LookupError):
        lift_detection(Detection("cup", 0.9, point=(100.0, 75.0)), capture, settings)
    with pytest.raises(ValueError):
        lift_detection(Detection("cup", 0.9, point=(500.0, 75.0)), capture, settings)
    with pytest.raises(ValueError):
        lift_detection(Detection("cup", 0.9, box=(300.0, 0.0, 310.0, 5.0)), capture, settings)


async def test_locate_follows_a_rotated_object(tmp_path: Path) -> None:
    rotated = SimBox("box", center=(320.0, -30.0), length=70.0, width=30.0, height=40.0, yaw_deg=35)
    scene = _scene(_calibrated(tmp_path, rotated))
    (box,) = await scene.locate(["box"])
    assert abs(angle_delta_deg(box.yaw_deg, 35.0)) < 4.0
    (grasp,) = await _PLANNER.candidates(box, table=_TABLE, current=_HOME)  # one side fits
    assert abs(angle_delta_deg(grasp.grasp.rz, 35.0)) % 180.0 < 4.0


async def test_ids_are_never_reused_across_snapshots(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    (first,) = await scene.locate(["cup"])
    (second,) = await scene.locate(["cup"])
    assert (first.object_id, second.object_id) == ("o1", "o2")
    with pytest.raises(SkillError) as error:
        scene.object("o1")
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND
    assert "locate again" in error.value.message


async def test_an_id_is_recalled_from_memory_until_its_ttl(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    camera = registry.devices[CAMERA_ID]
    clock = _Clock()
    scene = _scene(registry, clock=clock)
    (cup,) = await scene.locate(["cup"])
    assert scene.object(cup.object_id) == cup
    assert camera.captures == 1  # recalled, not looked at again
    clock.now += PerceptionSettings().memory_ttl_s + 1.0
    with pytest.raises(SkillError) as error:
        scene.object(cup.object_id)
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND


async def test_labels_are_never_handles(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    await scene.locate(["cup"])
    with pytest.raises(SkillError) as error:
        scene.object("cup")
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND


async def test_every_detection_is_reported_with_its_own_id(tmp_path: Path) -> None:
    other = SimBox("cup", center=(380.0, -60.0), length=40.0, width=40.0, height=60.0, score=0.8)
    scene = _scene(_calibrated(tmp_path, CUP, other))
    first, second = await scene.locate(["cup"])
    assert [(o.object_id, o.score) for o in (first, second)] == [("o1", 0.9), ("o2", 0.8)]
    assert scene.object("o2").center[:2] == pytest.approx((380.0, -60.0), abs=3.0)


async def test_low_scores_are_dropped_unless_the_caller_lowers_the_floor(tmp_path: Path) -> None:
    faint = SimBox("cup", center=(300.0, 20.0), length=60.0, width=30.0, height=50.0, score=0.1)
    evidence: dict[str, Any] = {}
    scene = _scene(_calibrated(tmp_path, faint), evidence=evidence)
    assert await scene.locate(["cup"]) == ()
    assert evidence["perception"]["dropped"] == [{"label": "cup", "reason": "below min_score"}]
    (seen,) = await scene.locate(["cup"], min_score=0.05)
    assert seen.score == pytest.approx(0.1)


async def test_relocate_remeasures_the_approved_candidate_without_grounding(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    camera = registry.devices[CAMERA_ID]
    locator = registry.devices[LOCATOR_ID]
    scene = _scene(registry)
    (cup,) = await scene.locate(["cup"])
    assert len(locator.requests) == 1
    locator.fail = True  # freshness must not ask the model to select another candidate
    camera.boxes = [SimBox("cup", (320.0, 20.0), 60.0, 30.0, 50.0)]
    moved = await scene.relocate(cup)
    assert len(locator.requests) == 1
    assert moved.object_id == cup.object_id and moved.source == cup.source
    assert moved.center[0] > cup.center[0] + 8.0
    assert scene.object(cup.object_id) == moved  # memory follows the fresh measurement
    camera.boxes = [SimBox("cup", (420.0, 20.0), 60.0, 30.0, 50.0)]
    with pytest.raises(SkillError) as error:
        await scene.relocate(moved)
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND
    assert len(locator.requests) == 1


async def test_grasps_are_named_per_object_and_expire_with_a_new_locate(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    (cup,) = await scene.locate(["cup"])
    proposed = await _PLANNER.candidates(cup, table=_TABLE, current=_HOME)
    named = scene.remember_grasps(cup, proposed)
    assert [(g.grasp_id, g.object_id) for g in named] == [("g1", "o1"), ("g2", "o1")]
    assert scene.grasp("g2") == named[1]
    again = scene.remember_grasps(cup, proposed[:1])
    assert [g.grasp_id for g in again] == ["g3"]  # replaces the object's earlier grasps
    with pytest.raises(SkillError):
        scene.grasp("g1")
    await scene.locate(["cup"])
    with pytest.raises(SkillError):
        scene.grasp("g3")


async def test_holding_takes_the_object_off_the_table(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    (cup,) = await scene.locate(["cup"])
    (grasp, _) = scene.remember_grasps(
        cup, await _PLANNER.candidates(cup, table=_TABLE, current=_HOME)
    )
    scene.hold(HeldObject(obj=cup, width_mm=29.0, since=0.0, grasp=grasp))
    assert scene.held is not None
    assert scene.held.hold_offset_mm == pytest.approx(grasp.grasp.z - cup.bottom_z)
    with pytest.raises(SkillError):
        scene.object(cup.object_id)
    with pytest.raises(SkillError):
        scene.grasp(grasp.grasp_id)
    assert scene.release() is not None and scene.held is None


# ── Refusals ─────────────────────────────────────────────────────────────────


async def _refusal(scene: WorkcellScene) -> SkillError:
    with pytest.raises(SkillError) as error:
        await scene.locate(["cup"])
    return error.value


async def test_a_moved_camera_stream_is_stale_calibration(tmp_path: Path) -> None:
    other = CameraIntrinsics(width=200, height=150, fx=260.0, fy=260.0, ppx=100.0, ppy=75.0)
    error = await _refusal(_scene(_calibrated(tmp_path, CUP, intrinsics=other)))
    assert error.failure_code == FAILURE_CALIBRATION_STALE
    assert "calib.handeye" in error.message


async def test_missing_table_or_mount_is_missing_calibration(tmp_path: Path) -> None:
    error = await _refusal(_scene(_calibrated(tmp_path, CUP, table_z=None)))
    assert error.failure_code == FAILURE_CALIBRATION_MISSING and "table" in error.message
    bare = perception_registry(tmp_path / "bare.duckdb", (CUP,))
    bare.calibrate_cell()
    error = await _refusal(_scene(bare))
    assert error.failure_code == FAILURE_CALIBRATION_MISSING and "calib.handeye" in error.message


async def test_device_and_model_failures_are_typed(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    registry.devices[CAMERA_ID].fail = True
    assert (await _refusal(_scene(registry))).failure_code == FAILURE_DEVICE_FAULT

    registry = _calibrated(tmp_path / "model", CUP)
    locator = next(d for d in registry.devices.values() if hasattr(d, "infer"))
    locator.fail = True
    assert (await _refusal(_scene(registry))).failure_code == FAILURE_MODEL_UNAVAILABLE
    locator.fail, locator.task = False, "action_chunk"
    assert (await _refusal(_scene(registry))).failure_code == FAILURE_MODEL_UNAVAILABLE

    unbound = _spec(models={})
    error = await _refusal(_scene(registry, spec=unbound))
    assert error.failure_code == FAILURE_SKILL_UNAVAILABLE


# ── Runtime wiring: perceive.locate and grasp.propose ────────────────────────


async def _read(runtime: SkillRuntime, skill: str, args: Mapping[str, Any]) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, skill, args)
    assert isinstance(outcome, SkillOutcome), outcome  # READ skills answer at plan
    return outcome


async def test_locate_then_propose_shares_one_scene_across_runs(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, PERCEPTION_WORKCELL_YAML, skills=scene_skills())
    located = await _read(runtime, "perceive.locate", {"queries": ["cup"]})
    assert located.ok, located
    (cup,) = located.result["objects"]
    assert cup["object_id"] == "o1" and located.result["held"] is None
    assert located.result["table"] is not None
    assert located.evidence["perception"]["located"] == 1

    proposed = await _read(runtime, "grasp.propose", {"object": "o1", "top_k": 1})
    assert proposed.ok, proposed
    (grasp,) = proposed.result["grasps"]
    assert grasp["grasp_id"] == "g1" and grasp["object_id"] == "o1"
    assert proposed.result["planner"] == TOP_DOWN
    assert proposed.evidence["grasps"]["proposed"] == 2
    assert registry.devices[CAMERA_ID].captures == 1  # propose recalled o1


async def test_propose_refuses_a_forgotten_id_and_an_ungraspable_object(tmp_path: Path) -> None:
    wide = SimBox("tray", center=(300.0, 0.0), length=100.0, width=68.0, height=20.0)
    registry = _calibrated(tmp_path, wide)
    runtime = _runtime(tmp_path, registry, PERCEPTION_WORKCELL_YAML, skills=scene_skills())
    unknown = await _read(runtime, "grasp.propose", {"object": "o9"})
    assert unknown.failure_code == FAILURE_TARGET_NOT_FOUND
    assert (await _read(runtime, "perceive.locate", {"queries": ["tray"]})).ok
    refused = await _read(runtime, "grasp.propose", {"object": "o1"})
    assert refused.failure_code == "unreachable"
    assert "proposed no reachable grasp" in refused.message


async def test_perception_skills_are_unavailable_without_the_camera_calibration(
    tmp_path: Path,
) -> None:
    registry = perception_registry(tmp_path / "instrument.duckdb", (CUP,))
    registry.calibrate_cell()
    runtime = _runtime(tmp_path, registry, PERCEPTION_WORKCELL_YAML, skills=scene_skills())
    refused = await runtime.plan(WORKCELL_ID, "perceive.locate", {"queries": ["cup"]})
    assert isinstance(refused, SkillOutcome)
    assert refused.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "calib.handeye" in refused.message

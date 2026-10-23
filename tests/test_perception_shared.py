# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared frames: ``perceive.snapshot`` and ``perceive.lift`` on a simulated cell.

A frame leaves the node only when the workcell declares ``share_frames`` and
the requesting host consents; it reaches S2 as an image attachment beside the
tool result, never as base64 inside the text. What S2 points at in it is
lifted exactly like a locator's grounding and re-measured, not re-detected,
when a motion program checks it for drift.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from leapflow.engine.tools.tool_attachments import ATTACHMENTS_KEY
from leapflow.hardware.fleet import FleetNode
from leapflow.hardware.lhp.client import DirectLhpClient
from leapflow.hardware.lhp.params import optional_bool
from leapflow.hardware.lhp.protocol import LhpProtocolError
from leapflow.hardware.lhp.server import WRITE_MODE_TRUSTED_V1, LhpNodeServer
from leapflow.hardware.skill_results import ATTACHMENTS_FIELD, share_frame
from leapflow.hardware.skill_tools import SkillService
from leapflow.hardware.skills.builtin.scene import skills as scene_skills
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.perception import (
    PerceptionSettings,
    WorkcellScene,
    target_detection,
)
from leapflow.hardware.skills.protocol import (
    COORDS_PER_MILLE,
    COORDS_PIXEL,
    COORDS_UNIT,
    FAILURE_INVALID_ARGS,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    SOURCE_LIFT,
    SOURCE_LOCATOR,
    LocatedObject,
    PixelTarget,
    SkillError,
    SkillOutcome,
)
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.skills.scene_memory import SceneCapture, SceneMemory
from leapflow.hardware.skills.workcell import WorkcellError, WorkcellSpec
from tests._harness.skill_stubs import (
    CAMERA_ID,
    CAMERA_POSE,
    LOCATOR_ID,
    PERCEPTION_WORKCELL_YAML,
    SIM_INTRINSICS,
    TABLE_Z,
    WORKCELL_ID,
    SimBox,
    SkillNodeRegistry,
    perception_registry,
    write_workcell,
)

CUP = SimBox("cup", center=(300.0, 20.0), length=60.0, width=30.0, height=50.0)
SHARED_YAML: dict[str, Any] = {**PERCEPTION_WORKCELL_YAML, "share_frames": True}
NODE_ID = "jetson01"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _calibrated(tmp_path: Path, *boxes: SimBox) -> SkillNodeRegistry:
    registry = perception_registry(tmp_path / "instrument.duckdb", boxes)
    registry.calibrate_perception()
    return registry


def _scene(
    registry: SkillNodeRegistry,
    *,
    clock: _Clock | None = None,
    evidence: dict[str, Any] | None = None,
    **perception: float,
) -> WorkcellScene:
    spec = WorkcellSpec.from_mapping(SHARED_YAML)
    sink = {} if evidence is None else evidence
    return WorkcellScene(
        registry=registry,
        spec=spec,
        frames=FrameGraph.snapshot(spec, registry.calibration_store),
        motion=None,  # type: ignore[arg-type] - a base-mounted camera never reads the arm
        memory=SceneMemory(clock or _Clock()),
        settings=PerceptionSettings.from_mapping(perception),
        note=sink.__setitem__,
    )


def _pixel(point: tuple[float, float, float]) -> tuple[float, float]:
    """Where the simulated camera sees base-frame *point*."""
    to_camera = Transform.from_pose(Pose(**CAMERA_POSE)).inverse()
    return SIM_INTRINSICS.project(to_camera.apply(point))


def _cup_top() -> tuple[float, float]:
    return _pixel((*CUP.center, TABLE_Z + CUP.height))


def _cup_box() -> list[float]:
    x, y = CUP.center
    corners = [
        _pixel((x + dx, y + dy, TABLE_Z + CUP.height))
        for dx in (-CUP.length / 2.0, CUP.length / 2.0)
        for dy in (-CUP.width / 2.0, CUP.width / 2.0)
    ]
    us, vs = [u for u, _ in corners], [v for _, v in corners]
    return [min(us), min(vs), max(us), max(vs)]


def _runtime(tmp_path: Path, registry: SkillNodeRegistry, workcell: Mapping[str, Any]) -> Any:
    directory = tmp_path / "workcells"
    write_workcell(directory, workcell)
    return SkillRuntime(
        registry, settings=SkillRuntimeSettings(workcells_dir=directory), skills=scene_skills()
    )


async def _read(
    runtime: SkillRuntime, skill: str, args: Mapping[str, Any], *, share_frames: bool = True
) -> SkillOutcome:
    outcome = await runtime.plan(WORKCELL_ID, skill, args, share_frames=share_frames)
    assert isinstance(outcome, SkillOutcome), outcome  # READ skills answer at plan
    return outcome


# ── Consent ──────────────────────────────────────────────────────────────────


async def test_a_workcell_shares_no_frame_unless_it_declares_so(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), PERCEPTION_WORKCELL_YAML)
    refused = await _read(runtime, "perceive.snapshot", {})
    assert refused.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "share_frames: true" in refused.message
    lift = await _read(runtime, "perceive.lift", {"frame": "f1", "targets": [{"label": "x"}]})
    assert lift.failure_code == FAILURE_SKILL_UNAVAILABLE


async def test_a_host_that_has_not_consented_gets_no_frame(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, SHARED_YAML)
    refused = await _read(runtime, "perceive.snapshot", {}, share_frames=False)
    assert refused.failure_code == FAILURE_SKILL_UNAVAILABLE
    assert "host has not consented" in refused.message
    assert registry.devices[CAMERA_ID].captures == 0
    assert (await _read(runtime, "perceive.locate", {"queries": ["cup"]}, share_frames=False)).ok


def test_share_frames_must_be_a_boolean() -> None:
    assert WorkcellSpec.from_mapping(SHARED_YAML).to_dict()["share_frames"] is True
    assert WorkcellSpec.from_mapping(PERCEPTION_WORKCELL_YAML).share_frames is False
    with pytest.raises(WorkcellError):
        WorkcellSpec.from_mapping({**SHARED_YAML, "share_frames": "yes"})


def test_the_wire_consent_flag_is_a_strict_boolean() -> None:
    assert optional_bool({}, "share_frames") is False
    assert optional_bool({"share_frames": None}, "share_frames") is False
    assert optional_bool({"share_frames": True}, "share_frames") is True
    with pytest.raises(LhpProtocolError):
        optional_bool({"share_frames": "true"}, "share_frames")


# ── Snapshot ─────────────────────────────────────────────────────────────────


async def test_snapshot_returns_a_jpeg_frame_with_its_id(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), SHARED_YAML)
    shot = await _read(runtime, "perceive.snapshot", {})
    assert shot.ok, shot
    assert (shot.result["frame"], shot.result["width"], shot.result["height"]) == ("f1", 200, 150)
    image = shot.result["image"]
    assert image["mime"] == "image/jpeg"
    data = base64.b64decode(image["data"])
    assert data[:2] == b"\xff\xd8"
    assert shot.evidence["snapshot"]["frame"] == "f1"
    assert (await _read(runtime, "perceive.snapshot", {})).result["frame"] == "f2"


async def test_snapshot_downscales_to_the_declared_side(tmp_path: Path) -> None:
    from PIL import Image

    shot = await _scene(_calibrated(tmp_path, CUP), snapshot_max_side=100).snapshot()
    assert (shot["width"], shot["height"]) == (100, 75)
    decoded = Image.open(io.BytesIO(base64.b64decode(shot["image"]["data"])))
    assert decoded.size == (100, 75)


@pytest.mark.parametrize("side", [10, 2000])
def test_a_shared_frame_side_must_fit_one_message(side: float) -> None:
    with pytest.raises(ValueError, match="snapshot_max_side"):
        PerceptionSettings.from_mapping({"snapshot_max_side": side})


def test_target_coordinates_scale_to_the_full_frame() -> None:
    frame = SimpleNamespace(width=200, height=150)
    point = PixelTarget("cup", point=(50.0, 37.5))
    assert target_detection(point, frame, COORDS_PIXEL, (100, 75)).point == (100.0, 75.0)
    unit = PixelTarget("cup", point=(0.5, 0.5))
    assert target_detection(unit, frame, COORDS_UNIT, (100, 75)).point == (100.0, 75.0)
    box = PixelTarget("cup", box=(250.0, 0.0, 500.0, 1000.0))
    shown = target_detection(box, frame, COORDS_PER_MILLE, (100, 75))
    assert shown.box == pytest.approx((50.0, 0.0, 100.0, 150.0))
    with pytest.raises(ValueError):
        PixelTarget("cup")
    with pytest.raises(ValueError):
        PixelTarget("cup", point=(1.0, 1.0), box=(0.0, 0.0, 2.0, 2.0))


# ── Lift ─────────────────────────────────────────────────────────────────────


async def test_lift_turns_a_pointed_pixel_into_a_surface_point(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), SHARED_YAML)
    assert (await _read(runtime, "perceive.snapshot", {})).ok
    u, v = _cup_top()
    lifted = await _read(
        runtime,
        "perceive.lift",
        {"frame": "f1", "targets": [{"label": "cup", "point": [u, v]}]},
    )
    assert lifted.ok, lifted
    (cup,) = lifted.result["objects"]
    assert cup["object_id"] == "o1" and cup["source"] == SOURCE_LIFT
    assert cup["geometry"] == "point"
    assert cup["center"] == pytest.approx([*CUP.center, TABLE_Z + CUP.height], abs=3.0)
    assert lifted.evidence["lift"] == {"frame": "f1", "lifted": 1, "dropped": []}

    fractions = [u / SIM_INTRINSICS.width, v / SIM_INTRINSICS.height]
    again = await _read(
        runtime,
        "perceive.lift",
        {"frame": "f1", "coords": COORDS_UNIT, "targets": [{"label": "cup", "point": fractions}]},
    )
    (same,) = again.result["objects"]
    assert same["object_id"] == "o2"
    assert same["center"] == pytest.approx(cup["center"], abs=1e-6)


async def test_lift_measures_a_box_like_a_locator_box(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    shot = await scene.snapshot()
    (lifted,), dropped = await scene.lift(
        shot["frame"], [PixelTarget("cup", box=tuple(_cup_box()))], coords=COORDS_PIXEL
    )
    assert dropped == ()
    assert lifted.geometry == "box" and lifted.source == SOURCE_LIFT
    assert lifted.center[:2] == pytest.approx(CUP.center, abs=3.0)
    assert (lifted.length, lifted.width, lifted.height) == pytest.approx(
        (60.0, 30.0, 50.0), abs=4.0
    )


async def test_lift_drops_what_it_cannot_measure_and_says_why(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), SHARED_YAML)
    assert (await _read(runtime, "perceive.snapshot", {})).ok
    targets = [{"label": "cup", "point": list(_cup_top())}, {"label": "ghost", "point": [500, 10]}]
    lifted = await _read(runtime, "perceive.lift", {"frame": "f1", "targets": targets})
    assert lifted.ok and len(lifted.result["objects"]) == 1
    (dropped,) = lifted.result["dropped"]
    assert dropped["label"] == "ghost" and "outside" in dropped["reason"]
    assert lifted.message.startswith("Lifted 1 of 2 target(s)")


@pytest.mark.parametrize(
    "target",
    [
        {"label": "cup"},
        {"label": "cup", "point": [1, 1], "box": [0, 0, 5, 5]},
        {"label": "cup", "box": [10, 10, 5, 20]},
        {"label": "cup", "box": [10, 10, 20, 10]},
    ],
)
async def test_lift_refuses_malformed_targets(tmp_path: Path, target: Mapping[str, Any]) -> None:
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), SHARED_YAML)
    refused = await _read(runtime, "perceive.lift", {"frame": "f1", "targets": [target]})
    assert refused.failure_code == FAILURE_INVALID_ARGS
    assert "targets[0]" in refused.message


async def test_only_the_newest_frames_can_be_pointed_into(tmp_path: Path) -> None:
    clock = _Clock()
    scene = _scene(_calibrated(tmp_path, CUP), clock=clock)
    frames = [(await scene.snapshot())["frame"] for _ in range(3)]
    assert frames == ["f1", "f2", "f3"]
    target = [PixelTarget("cup", point=_cup_top())]
    with pytest.raises(SkillError) as error:
        await scene.lift("f1", target, coords=COORDS_PIXEL)
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND
    assert "take a new snapshot" in error.value.message
    assert (await scene.lift("f2", target, coords=COORDS_PIXEL))[0]
    clock.now += PerceptionSettings().memory_ttl_s + 1.0
    with pytest.raises(SkillError):
        await scene.lift("f3", target, coords=COORDS_PIXEL)


async def test_lifted_ids_live_beside_located_ones_until_the_next_locate(tmp_path: Path) -> None:
    scene = _scene(_calibrated(tmp_path, CUP))
    (located,) = await scene.locate(["cup"])
    assert located.source == SOURCE_LOCATOR
    shot = await scene.snapshot()
    (lifted,), _ = await scene.lift(
        shot["frame"], [PixelTarget("cup", point=_cup_top())], coords=COORDS_PIXEL
    )
    assert (located.object_id, lifted.object_id) == ("o1", "o2")
    assert scene.object("o1") == located and scene.object("o2") == lifted
    await scene.locate(["cup"])
    with pytest.raises(SkillError):
        scene.object("o2")


def test_a_recalibration_forgets_shared_frames() -> None:
    memory = SceneMemory(_Clock())
    mount = Transform.from_pose(Pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    capture = SceneCapture(frame=None, camera_to_base=mount, table=None)  # type: ignore[arg-type]
    frame_id = memory.remember_frame(capture, {"camera": 1.0}, keep=2)
    assert memory.recall_frame(frame_id, {"camera": 1.0}, 30.0) is capture
    assert memory.recall_frame(frame_id, {"camera": 2.0}, 30.0) is None
    memory.remember([], {"camera": 2.0})
    assert memory.recall_frame(frame_id, {"camera": 1.0}, 30.0) is None


def test_the_grounding_source_round_trips() -> None:
    obj = LocatedObject(
        object_id="o1",
        label="cup",
        score=1.0,
        center=(1.0, 2.0, 3.0),
        length=0.0,
        width=0.0,
        height=0.0,
        yaw_deg=0.0,
        observed_at=0.0,
        geometry="point",
        source=SOURCE_LIFT,
    )
    assert LocatedObject.from_mapping(obj.to_dict()) == obj
    shown = {**obj.to_dict()}
    shown.pop("source")
    assert LocatedObject.from_mapping(shown).source == SOURCE_LOCATOR


# ── Relocation of what S2 grounded ───────────────────────────────────────────


async def _lift_cup(scene: WorkcellScene, target: PixelTarget) -> LocatedObject:
    shot = await scene.snapshot()
    (lifted,), _ = await scene.lift(shot["frame"], [target], coords=COORDS_PIXEL)
    return lifted


async def test_a_lifted_box_is_measured_again_where_it_was(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    camera = registry.devices[CAMERA_ID]
    scene = _scene(registry)
    lifted = await _lift_cup(scene, PixelTarget("cup", box=tuple(_cup_box())))
    again = await scene.relocate(lifted)
    assert again.object_id == lifted.object_id and again.source == SOURCE_LIFT
    assert again.center == pytest.approx(lifted.center, abs=2.0)
    assert scene.object(lifted.object_id) == again
    assert registry.devices[LOCATOR_ID].requests == []  # re-measured, never re-detected
    camera.boxes = []
    with pytest.raises(SkillError) as error:
        await scene.relocate(again)
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND
    assert "lift it again" in error.value.message


async def test_a_lifted_point_over_an_emptied_table_falls_to_the_table(tmp_path: Path) -> None:
    registry = _calibrated(tmp_path, CUP)
    scene = _scene(registry)
    lifted = await _lift_cup(scene, PixelTarget("cup", point=_cup_top()))
    registry.devices[CAMERA_ID].boxes = []
    gone = await scene.relocate(lifted)
    assert gone.center[2] == pytest.approx(TABLE_Z, abs=2.0)  # the drift check sees it left


# ── Host side: the image channel ─────────────────────────────────────────────


def _shot() -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "frame": "f1",
            "width": 200,
            "height": 150,
            "image": {"mime": "image/jpeg", "data": "QUJD"},
        },
    }


def test_a_consented_frame_rides_beside_the_result_as_an_attachment() -> None:
    assert ATTACHMENTS_FIELD == ATTACHMENTS_KEY
    result = _shot()
    original = result["result"]
    assert share_frame(result, consent=True) == "frame f1, 200x150"
    assert "image" not in result["result"] and "image" in original  # the payload was copied
    (attachment,) = result[ATTACHMENTS_FIELD]
    assert attachment == {"mime": "image/jpeg", "data": "QUJD", "label": "frame f1, 200x150"}


def test_without_consent_the_frame_is_dropped_never_inlined() -> None:
    result = _shot()
    assert share_frame(result, consent=False) == ""
    assert "image" not in result["result"] and ATTACHMENTS_FIELD not in result
    plain = {"ok": True, "result": {"objects": []}}
    assert share_frame(plain, consent=True) == "" and ATTACHMENTS_FIELD not in plain


class _Audit:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **fields: Any) -> None:
        self.records.append(fields)


class _Fleet:
    def __init__(self) -> None:
        self.permit_issuer = None
        self._node = FleetNode(
            node_id=NODE_ID, display_name=NODE_ID, endpoint=f"tcp://{NODE_ID}:8767"
        )

    def topology(self) -> SimpleNamespace:
        return SimpleNamespace(nodes=(self._node,))


@pytest.mark.parametrize("consent", [True, False])
async def test_the_host_sends_its_consent_and_audits_every_shared_frame(
    tmp_path: Path, consent: bool
) -> None:
    registry = _calibrated(tmp_path, CUP)
    runtime = _runtime(tmp_path, registry, SHARED_YAML)
    server = LhpNodeServer(
        registry, node_id=NODE_ID, write_mode=WRITE_MODE_TRUSTED_V1, skill_runtime=runtime
    )
    audit = _Audit()
    service = SkillService(
        _Fleet(),
        audit_log=audit,
        session_id="session-1",
        client_factory=lambda node: DirectLhpClient(server),
        share_frames=consent,
    )
    try:
        result = await service.hw_skill_run(workcell=WORKCELL_ID, skill="perceive.snapshot")
    finally:
        await service.close()
        await server.stop()
    shared = [record for record in audit.records if record["action"] == "skill_frames"]
    if consent:
        assert result["ok"], result
        assert "image" not in result["result"]
        (attachment,) = result[ATTACHMENTS_FIELD]
        assert attachment["label"] == "frame f1, 200x150"
        assert base64.b64decode(attachment["data"])[:2] == b"\xff\xd8"
        (record,) = shared
        assert record["outcome"] == "shared frame f1, 200x150"
    else:
        assert result["failure_code"] == FAILURE_SKILL_UNAVAILABLE
        assert "host has not consented" in result["error"]
        assert ATTACHMENTS_FIELD not in result and shared == []


async def test_capture_identity_prevents_frame_alias_after_runtime_restart(tmp_path):
    registry = _calibrated(tmp_path, CUP)
    scene = _scene(registry)
    shot = await scene.snapshot()
    target = [PixelTarget("cup", box=tuple(_cup_box()))]
    assert (await scene.lift(shot["frame"], target, coords=COORDS_PIXEL,
                             capture_id=shot["capture_id"]))[0]
    restarted = _scene(registry)
    new = await restarted.snapshot()
    assert new["frame"] == shot["frame"] and new["capture_id"] != shot["capture_id"]
    with pytest.raises(SkillError, match="not a current frame"):
        await restarted.lift(shot["frame"], target, coords=COORDS_PIXEL,
                             capture_id=shot["capture_id"])


async def test_capture_binding_crosses_read_skill_boundary(tmp_path):
    runtime = _runtime(tmp_path, _calibrated(tmp_path, CUP), SHARED_YAML)
    shot = (await _read(runtime, "perceive.snapshot", {})).result
    args = {"frame": shot["frame"], "capture_id": shot["capture_id"],
            "targets": [{"label": "cup", "box": _cup_box()}]}
    assert (await _read(runtime, "perceive.lift", args)).ok
    refused = await _read(runtime, "perceive.lift", {**args, "capture_id": "0" * 32})
    assert refused.failure_code == FAILURE_TARGET_NOT_FOUND


@pytest.mark.parametrize("fault", ["none", "unconfirmed", "hash", "image", "node", "workcell"])
async def test_operations_preview_uses_only_the_confirmed_saved_capture(fault, tmp_path, monkeypatch):
    import json
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock
    from piper_ops import calibration, grasp, motion
    from piper_ops.config import Settings
    from piper_ops.evidence import Evidence, Refusal
    from tests.test_piper_ops import context

    shot = _shot()
    shot["result"]["capture_id"] = "a" * 32
    shot["side_effect_state"] = "none"
    share_frame(shot, consent=True)
    service = SimpleNamespace(hw_skill_run=AsyncMock(return_value=shot))

    @asynccontextmanager
    async def factory(*args, **kwargs):
        yield service

    monkeypatch.setattr(calibration, "service", factory)
    session = SimpleNamespace(settings=Settings(root=tmp_path, node_id="node"), context=context(),
                              evidence=Evidence(tmp_path / "attempt.json"))
    saved = await grasp.snapshot(session, "cell", share_frames=True)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(saved))
    arguments = {"snapshot_file": str(path), "confirmed_snapshot_hash": saved["snapshot_hash"],
                 "target_confirmed": True, "target": {"label": "cup", "box": [0, 0, 10, 10]},
                 "grasp_index": 0, "place": {"x": 0, "y": 0, "z": 10}}
    if fault == "unconfirmed":
        arguments["target_confirmed"] = False
    elif fault == "hash":
        arguments["confirmed_snapshot_hash"] = "other"
    elif fault == "image":
        saved["snapshot"]["image"]["data"] = "changed"
        path.write_text(json.dumps(saved))
    elif fault == "node":
        session.settings = Settings(root=tmp_path, node_id="other")
    elif fault == "workcell":
        saved["workcell"] = "other"
        path.write_text(json.dumps(saved))
    cycle_plan = {"args": {"grasp_id": "g1", "place": arguments["place"]}, "digest": "test"}
    service.hw_skill_run.reset_mock()
    service.hw_skill_run.side_effect = [
        {"ok": True, "side_effect_state": "none", "result": {"objects": [{"object_id": "o1"}]}},
        {"ok": True, "side_effect_state": "none", "result": {"grasps": [{"grasp_id": "g1"}]}},
        {"ok": True, "dry_run": True, "plan": cycle_plan},
    ]
    if fault != "none":
        with pytest.raises(Refusal):
            await grasp.prepare(session, "cell", arguments, share_frames=True)
        service.hw_skill_run.assert_not_awaited()
        return
    result = await grasp.prepare(session, "cell", arguments, share_frames=True)
    calls = service.hw_skill_run.await_args_list
    assert [call.kwargs["skill"] for call in calls] == ["perceive.lift", "grasp.propose", "grasp.cycle"]
    assert calls[0].kwargs["args"]["capture_id"] == saved["snapshot"]["capture_id"]
    assert result["preview"]["operator_confirmation"]["snapshot_hash"] == saved["snapshot_hash"]
    assert result["plan_hash"] == motion.digest_plan(result["preview"])

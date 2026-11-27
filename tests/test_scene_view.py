# Copyright (c) Alibaba, Inc. and its affiliates.
"""Multi-camera, multi-plane snapshots: mosaics, depth and infrared tiles, cross-view
fusion, change reports, zooms, and lifting through whatever S2 was shown.

Nothing here names a camera model: a second camera is a second ``FrameSource``
bound to a role, and infrared is any device channel declaring ``infrared_image``.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.skills.geometry import Pose, Transform
from leapflow.hardware.skills.perception import PerceptionSettings, WorkcellScene
from leapflow.hardware.skills.protocol import (
    COORDS_PIXEL,
    COORDS_UNIT,
    FAILURE_INVALID_ARGS,
    FAILURE_TARGET_NOT_FOUND,
    PixelTarget,
    SkillError,
)
from leapflow.hardware.skills.scene_memory import SceneMemory
from leapflow.hardware.skills.visual_change import compare, observe
from leapflow.hardware.skills.visual_tiles import TILE_GAP, depth_image, depth_span
from leapflow.hardware.skills.workcell import WorkcellSpec
from leapflow.hardware.transport import FrameReading
from tests._harness.skill_stubs import (
    CAMERA_ID,
    CAMERA_POSE,
    IDENTITY_MATRIX,
    PERCEPTION_WORKCELL_YAML,
    SIM_INTRINSICS,
    TABLE_Z,
    SimBox,
    SimRgbdCamera,
    SkillNodeRegistry,
    perception_registry,
)

CUP = SimBox("cup", center=(300.0, 20.0), length=60.0, width=30.0, height=50.0)
WRIST_ID = "wrist_cam"
W, H = SIM_INTRINSICS.width, SIM_INTRINSICS.height
# Both cameras calibrated (the sim mounts them at the same pose).
TWO_CAMERAS: dict[str, Any] = {
    **PERCEPTION_WORKCELL_YAML,
    "share_frames": True,
    "roles": {**PERCEPTION_WORKCELL_YAML["roles"], "camera.wrist": WRIST_ID},
    "frames": {
        **PERCEPTION_WORKCELL_YAML["frames"],
        "camera.wrist": {"parent": "base", "procedure": "wrist.handeye"},
    },
}
# Only the wrist camera calibrated.
WRIST_ONLY: dict[str, Any] = {
    **TWO_CAMERAS,
    "frames": {
        **TWO_CAMERAS["frames"],
        "camera": {"parent": "base", "procedure": "global.handeye"},
        "camera.wrist": {"parent": "base", "procedure": "wrist.handeye"},
    },
}
IR_CHANNEL = "cam.ir_left.preview"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _IrRegistry(SkillNodeRegistry):
    """A node whose cameras declare an infrared frame channel."""

    reads: list[tuple[str, str, dict[str, Any]]]

    def context(self, device_id: str) -> Any:
        found = super().context(device_id)
        if found is None or device_id == "locator":
            return found
        channel = SimpleNamespace(
            channel_id=IR_CHANNEL, quantity="infrared_image", representation="frame"
        )
        return SimpleNamespace(device_id=device_id, channels=(channel,))

    async def read_frame(self, device_id: str, channel_id: str, **kwargs: Any) -> FrameReading:
        from PIL import Image

        self.reads.append((device_id, channel_id, kwargs))
        output = io.BytesIO()
        Image.new("L", (W, H), 128).save(output, format="JPEG")
        return FrameReading(device_id, channel_id, output.getvalue(), width=W, height=H)


def _registry(tmp_path: Path, cls: type = SkillNodeRegistry, *boxes: SimBox) -> SkillNodeRegistry:
    base = perception_registry(tmp_path / "instrument.duckdb", boxes or (CUP,))
    registry = base if cls is SkillNodeRegistry else _rebuild(base, cls, tmp_path)
    registry.devices[WRIST_ID] = SimRgbdCamera(list(boxes or (CUP,)))
    registry.calibrate_perception()
    version = registry.calibrate(
        "wrist.handeye",
        pose=CAMERA_POSE,
        parameters={"target_mount": IDENTITY_MATRIX, "intrinsics": SIM_INTRINSICS.to_dict()},
    )
    registry.calibrate("camera.wrist.verify", parameters={
        "training_disjoint": True, "p95_mm": .5, "depends_on": {"camera.wrist": version},
    })
    return registry


def _rebuild(base: SkillNodeRegistry, cls: type, tmp_path: Path) -> SkillNodeRegistry:
    registry = cls(None, store_path=tmp_path / "ir.duckdb", devices=base.devices)
    registry.reads = []
    return registry


def _scene(
    registry: SkillNodeRegistry, yaml: dict[str, Any] = TWO_CAMERAS, *, clock: _Clock | None = None,
    evidence: dict[str, Any] | None = None, **perception: float,
) -> WorkcellScene:
    from leapflow.hardware.skills.frames import FrameGraph

    spec = WorkcellSpec.from_mapping(yaml)
    sink = {} if evidence is None else evidence
    return WorkcellScene(
        registry=registry,
        spec=spec,
        frames=FrameGraph.snapshot(spec, registry.calibration_store),
        motion=None,  # type: ignore[arg-type] - base-mounted cameras never read the arm
        memory=SceneMemory(clock or _Clock()),
        settings=PerceptionSettings.from_mapping(perception),
        note=sink.__setitem__,
    )


def _image(shot: dict[str, Any]) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(shot["image"]["data"])))


def _cup_top() -> tuple[float, float]:
    to_camera = Transform.from_pose(Pose(**CAMERA_POSE)).inverse()
    return SIM_INTRINSICS.project(to_camera.apply((*CUP.center, TABLE_Z + CUP.height)))


# ── Mosaic ───────────────────────────────────────────────────────────────────


async def test_two_cameras_share_one_mosaic_captured_together(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    evidence: dict[str, Any] = {}
    shot = await _scene(registry, evidence=evidence).snapshot(with_cameras=["camera.wrist"])
    assert shot["frame"] == "v1" and shot["liftable"] is True
    assert [(t["camera"], t["plane"], t["frame"]) for t in shot["tiles"]] == [
        ("camera", "color", "f1"), ("camera.wrist", "color", "f2"),
    ]
    wrist = shot["tiles"][1]
    assert (wrist["x"], wrist["y"]) == (0, H + TILE_GAP)  # one row per camera
    assert _image(shot).size == (shot["width"], shot["height"]) == (W, 2 * H + TILE_GAP)
    assert set(shot["frames"]) == {"camera", "camera.wrist"}
    assert shot["sync"]["members"] == ["camera", "camera.wrist"]
    assert "view v1" in shot["image"]["label"]
    assert registry.devices[CAMERA_ID].captures == registry.devices[WRIST_ID].captures == 1
    assert evidence["snapshot"]["cameras"] == ["camera", "camera.wrist"]


async def test_all_adds_every_camera_role_once(tmp_path: Path) -> None:
    shot = await _scene(_registry(tmp_path)).snapshot("camera.wrist", with_cameras=["all"])
    assert [t["camera"] for t in shot["tiles"]] == ["camera.wrist", "camera"]


async def test_a_failed_extra_camera_is_reported_not_fatal(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.devices[WRIST_ID].fail = True
    shot = await _scene(registry).snapshot(with_cameras=["camera.wrist"])
    assert shot["frame"] == "f1"  # one tile left: a plain frame handle
    assert "capture failed" in shot["unavailable"]["camera.wrist"]
    assert any(line.startswith("camera.wrist unavailable") for line in shot["situation"])


async def test_the_main_camera_must_answer(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.devices[CAMERA_ID].fail = True
    with pytest.raises(SkillError, match="capture failed"):
        await _scene(registry).snapshot(with_cameras=["camera.wrist"])


async def test_bad_view_arguments_are_invalid_args(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path))
    for kwargs in ({"planes": ["thermal"]}, {"planes": []}, {"detail": "huge"}):
        with pytest.raises(SkillError) as error:
            await scene.snapshot(**kwargs)
        assert error.value.failure_code == FAILURE_INVALID_ARGS


async def test_overview_detail_shrinks_tiles(tmp_path: Path) -> None:
    shot = await _scene(_registry(tmp_path), overview_max_side=100).snapshot(
        with_cameras=["camera.wrist"], planes=["color", "depth"]
    )
    assert all(max(t["width"], t["height"]) <= 100 for t in shot["tiles"])  # 4 tiles -> overview
    assert len(shot["tiles"]) == 4


async def test_the_mosaic_never_exceeds_its_budget(tmp_path: Path) -> None:
    shot = await _scene(_registry(tmp_path), mosaic_max_side=256).snapshot(
        with_cameras=["camera.wrist"], planes=["color", "depth"], detail="full"
    )
    assert max(shot["width"], shot["height"]) <= 256


# ── Depth and infrared planes ────────────────────────────────────────────────


async def test_a_depth_tile_sits_beside_colour_and_is_liftable(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path))
    shot = await scene.snapshot(planes=["color", "depth"])
    assert shot["frame"] == "v1"
    color, depth = shot["tiles"]
    assert depth["plane"] == "depth" and depth["frame"] == color["frame"] == "f1"
    assert depth["x"] == W + TILE_GAP and depth["liftable_plane"] is True
    near, far = shot["frames"]["camera"]["depth_range_mm"]
    assert near < far
    u, v = _cup_top()
    (lifted,), dropped = await scene.lift(
        shot["frame"], [PixelTarget("cup", point=(u + W + TILE_GAP, v))], coords=COORDS_PIXEL
    )
    assert dropped == ()
    assert lifted.center == pytest.approx([*CUP.center, TABLE_Z + CUP.height], abs=3.0)


def test_depth_false_colour_marks_near_red_far_blue_and_missing_black() -> None:
    camera = SimRgbdCamera([CUP])
    import asyncio

    frame = asyncio.run(camera.capture_rgbd())
    span = depth_span(frame)
    assert span is not None and span[1] - span[0] >= 50
    image = depth_image(frame, (W, H), span)
    u, v = (round(c) for c in _cup_top())
    r, g, b = image.getpixel((u, v))  # the cup top is the nearest surface
    assert r > b
    r, g, b = image.getpixel((2, 2))  # the table is the farthest
    assert b > r
    from dataclasses import replace

    blank = replace(frame, depth=bytes(W * H * 2))
    assert depth_span(blank) is None
    assert depth_image(blank, (W, H), None).getpixel((5, 5)) == (0, 0, 0)  # no reading


async def test_infrared_is_shown_but_only_for_looking(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _IrRegistry)
    scene = _scene(registry)
    shot = await scene.snapshot(planes=["color", "infrared"])
    color, infrared = shot["tiles"]
    assert infrared["plane"] == "infrared" and infrared["frame"] is None
    assert infrared["liftable_plane"] is False
    ((device, channel, options),) = registry.reads
    assert (device, channel) == (CAMERA_ID, IR_CHANNEL) and options["max_width"] > 0
    assert "camera infrared" in shot["image"]["label"]
    _, dropped = await scene.lift(
        shot["frame"], [PixelTarget("cup", point=(infrared["x"] + 10, 10))], coords=COORDS_PIXEL
    )
    assert "for looking only" in dropped[0]["reason"]


async def test_a_camera_without_infrared_shows_what_it_has(tmp_path: Path) -> None:
    shot = await _scene(_registry(tmp_path)).snapshot(planes=["color", "infrared"])
    assert shot["frame"] == "f1"  # no channel declared: no tile, nothing failed
    assert "unavailable" not in shot


# ── Lifting through a mosaic ─────────────────────────────────────────────────


async def test_a_mosaic_point_lifts_in_the_tile_it_lands_in(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path), WRIST_ONLY)
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    assert shot["liftable"] is True and "unliftable" in shot  # the main camera is not
    u, v = _cup_top()
    wrist = shot["tiles"][1]
    targets = [
        PixelTarget("cup", point=(u, v + wrist["y"])),
        PixelTarget("cup", point=(u, v)),  # the uncalibrated main camera
        PixelTarget("gap", point=(10, H + 1)),  # between the rows
    ]
    (lifted,), dropped = await scene.lift(shot["frame"], targets, coords=COORDS_PIXEL)
    assert lifted.camera == "camera.wrist"
    assert lifted.center == pytest.approx([*CUP.center, TABLE_Z + CUP.height], abs=3.0)
    reasons = {entry["label"]: entry["reason"] for entry in dropped}
    assert "cannot be lifted" in reasons["cup"] and "between tiles" in reasons["gap"]


async def test_mosaic_pixels_sent_to_a_tile_frame_name_both_correct_calls(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path), WRIST_ONLY)
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    wrist = shot["tiles"][1]
    u, v = _cup_top()
    mosaic_point = PixelTarget("cup", point=(u, v + wrist["y"]))
    _, (dropped,) = await scene.lift(wrist["frame"], [mosaic_point], coords=COORDS_PIXEL)
    reason = dropped["reason"]
    assert f"lies outside {wrist['frame']}" in reason
    assert f"lift from {shot['frame']} with the same point" in reason
    assert f"[{round(float(u), 1)}, {round(float(v), 1)}]" in reason

    (lifted,), dropped = await scene.lift(
        wrist["frame"], [PixelTarget("cup", point=(u, v))], coords=COORDS_PIXEL
    )
    assert lifted.camera == "camera.wrist" and not dropped


async def test_mosaic_fractions_and_box_spills(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path))
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    u, v = _cup_top()
    unit = PixelTarget("cup", point=(u / shot["width"], v / shot["height"]))
    (lifted,), _ = await scene.lift(shot["frame"], [unit], coords=COORDS_UNIT)
    assert lifted.camera == "camera"
    # Centred in the wrist row, but most of it lies in the main camera's.
    spill = PixelTarget("wide", box=(0.0, 0.0, 50.0, 320.0))
    _, (dropped,) = await scene.lift(shot["frame"], [spill], coords=COORDS_PIXEL)
    assert "keep it inside one tile" in dropped["reason"]


async def test_a_view_ages_out_like_a_frame(tmp_path: Path) -> None:
    clock = _Clock()
    scene = _scene(_registry(tmp_path), clock=clock)
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    clock.now += PerceptionSettings().frame_ttl_s + 1.0
    with pytest.raises(SkillError) as error:
        await scene.lift(shot["frame"], [PixelTarget("cup", point=(1, 1))], coords=COORDS_PIXEL)
    assert error.value.failure_code == FAILURE_TARGET_NOT_FOUND
    assert "take a new snapshot" in error.value.message


# ── Fusion and change ────────────────────────────────────────────────────────


async def test_a_located_object_is_checked_in_every_view(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    scene = _scene(registry)
    (cup,) = await scene.locate(["cup"])
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    (entry,) = shot["objects"]
    assert entry["id"] == cup.object_id
    assert {cam: view["state"] for cam, view in entry["views"].items()} == {
        "camera": "visible", "camera.wrist": "visible",
    }
    wrist_px = entry["views"]["camera.wrist"]["pixel"]
    assert wrist_px[1] > H  # given in the mosaic S2 sees, in the wrist row
    registry.devices[CAMERA_ID].boxes = []
    registry.devices[WRIST_ID].boxes = []
    later = await scene.snapshot(with_cameras=["camera.wrist"])
    (gone,) = later["objects"]
    assert {view["state"] for view in gone["views"].values()} == {"moved_or_gone"}
    assert any("moved_or_gone" in line for line in later["situation"])


async def test_each_camera_reports_what_changed_since_its_last_view(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    scene = _scene(registry)
    first = await scene.snapshot()
    assert "changes" not in first  # nothing to compare against yet
    same = await scene.snapshot()
    assert same["changes"]["camera"]["changed_cells"] == 0
    registry.devices[CAMERA_ID].boxes = []
    after = await scene.snapshot()
    report = after["changes"]["camera"]
    assert report["comparable"] and report["changed_cells"] > 0 and report["farther"] > 0
    assert any("changed since f2" in line for line in after["situation"])


def test_a_moved_camera_is_not_compared_pixel_by_pixel() -> None:
    import asyncio

    frame = asyncio.run(SimRgbdCamera([CUP]).capture_rgbd())
    here = Transform.from_pose(Pose(**CAMERA_POSE))
    there = Transform.from_pose(Pose(**{**CAMERA_POSE, "x": CAMERA_POSE["x"] + 30.0}))
    before, _ = observe(frame, camera="camera.wrist", frame_id="f1", pose=here)
    after, stats = observe(frame, camera="camera.wrist", frame_id="f2", pose=there)
    report = compare(before, after)
    assert report.comparable is False and report.camera_moved_mm == pytest.approx(30.0, abs=0.5)
    assert "judge objects" in report.summary()
    assert "dark" in stats.flags  # the sim colour is black
    assert compare(before, observe(frame, camera="camera.wrist", frame_id="f3", pose=here)[0]).comparable


def _signature(frame_id: str, cells: list[tuple[float, float, float]]) -> Any:
    """A one-row signature from (depth_mm, coverage, spread_mm) per cell."""
    from leapflow.hardware.skills.visual_change import FrameSignature

    return FrameSignature(
        camera="camera.wrist", frame_id=frame_id, observed_at=0.0, width=len(cells) * 10,
        height=10, depth_mm=tuple(d for d, _, _ in cells), luma=(100.0,) * len(cells),
        cols=len(cells), rows=1, coverage=tuple(c for _, c, _ in cells),
        spread_mm=tuple(s for _, _, s in cells),
    )


def test_sensor_noise_is_not_reported_as_change() -> None:
    before = _signature("f1", [
        (0.0, 0.25, 0.0),     # coverage hovering under the threshold ...
        (900.0, 0.4, 60.0),   # a far, noisy cell
        (400.0, 0.9, 2.0),    # a quiet table cell
        (400.0, 0.9, 2.0),
        (400.0, 0.9, 2.0),
    ])
    after = _signature("f2", [
        (500.0, 0.35, 5.0),   # ... and just over it: not "depth appeared"
        (940.0, 0.45, 70.0),  # moved less than its own spread
        (350.0, 0.9, 60.0),   # something arrived: only the new frame is spread
        (0.0, 0.02, 0.0),     # depth really vanished
        (400.0, 0.9, 2.0),
    ])
    report = compare(before, after)
    assert report.changed_cells == (2, 3)
    assert (report.closer, report.depth_lost, report.depth_gained) == (1, 1, 0)
    assert compare(before, before).summary().endswith("ago)")  # no pose caveat when unchanged


# ── Zoom ─────────────────────────────────────────────────────────────────────


async def test_a_zoom_reshows_a_window_and_lifts_in_the_full_frame(tmp_path: Path) -> None:
    clock = _Clock()
    scene = _scene(_registry(tmp_path), clock=clock, snapshot_max_side=100)
    shot = await scene.snapshot()
    assert (shot["width"], shot["height"]) == (100, 75)
    u, v = _cup_top()
    clock.now += 10.0
    box = (u / 2 - 15, v / 2 - 15, u / 2 + 15, v / 2 + 15)  # in the 100x75 image
    zoomed = await scene.zoom(shot["frame"], box, coords=COORDS_PIXEL)
    assert zoomed["frame"] == "f2" and zoomed["zoom_of"] == "f1"
    assert shot["valid_for_s"] == PerceptionSettings().frame_ttl_s
    assert zoomed["valid_for_s"] == PerceptionSettings().frame_ttl_s - 10.0
    assert zoomed["liftable_for_s"] == zoomed["valid_for_s"]
    assert zoomed["validity_timebase"] == "node_remaining_duration"
    x0, y0, x1, y1 = zoomed["window"]
    assert (x1 - x0, y1 - y0) == pytest.approx((60.0, 60.0))
    assert zoomed["width"] > 30  # shown larger than it was
    sx = zoomed["width"] / (x1 - x0)
    point = ((u - x0) * sx, (v - y0) * sx)
    (lifted,), _ = await scene.lift(zoomed["frame"], [PixelTarget("cup", point=point)],
                                    coords=COORDS_PIXEL)
    assert lifted.center == pytest.approx([*CUP.center, TABLE_Z + CUP.height], abs=3.0)
    clock.now += PerceptionSettings().frame_ttl_s - 5.0  # the zoom keeps the original's age
    with pytest.raises(SkillError):
        await scene.lift(zoomed["frame"], [PixelTarget("cup", point=point)], coords=COORDS_PIXEL)


async def test_a_zoom_into_a_mosaic_uses_the_tile_under_the_box(tmp_path: Path) -> None:
    scene = _scene(_registry(tmp_path))
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    top = shot["tiles"][1]["y"]
    zoomed = await scene.zoom(shot["frame"], (10, top + 10, 90, top + 90), coords=COORDS_PIXEL)
    assert zoomed["camera"] == "camera.wrist" and zoomed["zoom_of"] == "f2"
    with pytest.raises(SkillError) as error:
        await scene.zoom(shot["frame"], (10, 10, 12, 12), coords=COORDS_PIXEL)
    assert error.value.failure_code == FAILURE_INVALID_ARGS


async def test_snapshot_validity_includes_rendering_elapsed_and_every_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    scene = _scene(_registry(tmp_path), clock=clock, frame_ttl_s=30.0)
    render = scene._viewer._render

    def slow_render(*args: Any) -> Any:
        # Distinct source ages exercise the oldest constituent rather than the
        # newly named mosaic. This is an I/O boundary (image rendering).
        scene._memory._frames["f2"] = (
            scene._memory._frames["f2"][0], clock.now - 8.0,
        )
        clock.now += 5.0
        return render(*args)

    monkeypatch.setattr(scene._viewer, "_render", slow_render)
    shot = await scene.snapshot(with_cameras=["camera.wrist"])
    assert shot["valid_for_s"] == 17.0
    assert shot["liftable_for_s"] == 17.0
    assert scene._memory.frame_remaining_s("missing", scene._basis, 30.0) == 0.0
    assert scene._memory.frame_remaining_s("f1", {"changed": 1.0}, 30.0) == 0.0
    clock.now += 31.0
    assert scene._memory.frame_remaining_s("f1", scene._basis, 30.0) == 0.0


def test_view_sides_are_bounded() -> None:
    for name in ("overview_max_side", "mosaic_max_side"):
        with pytest.raises(ValueError, match=name):
            PerceptionSettings.from_mapping({name: 4000})


def test_capture_bounds_reach_a_source_that_understands_them() -> None:
    from leapflow.hardware.vision.group import bounded_capture

    seen: list[Any] = []

    async def modern(*, after_monotonic: float | None = None) -> Any:
        seen.append(after_monotonic)
        return SimpleNamespace(monotonic_at=time.monotonic())

    import asyncio

    asyncio.run(bounded_capture(modern)(12.5))
    assert seen == [12.5]

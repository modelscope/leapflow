# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local perception: ground in 2-D on a model service, lift to 3-D here.

One ``locate`` call:

1. reads the camera mount (the hand-eye calibration, composed with the current
   flange pose when the camera rides on the arm) and the table plane;
2. captures one aligned RGB-D frame from the workcell camera (``FrameSource``);
3. asks the workcell's ``locator`` model (``grounding``) where the queries are;
4. lifts each detection by what it carries: a mask's pixels, or a box's pixel
   grid, are subsampled, deprojected with the live intrinsics, mapped into the
   arm base frame, kept when standing on the table, and cleaned of outliers by
   median absolute deviation; a point takes the median depth of its pixel
   neighbourhood and becomes one surface point;
5. fits the oriented footprint and the height above the table (mask, box).

Only ``LocatedObject`` geometry leaves this module -- except a frame the
workcell explicitly shares (``share_frames``) through ``snapshot``, which S2
may look at and answer with pixels for ``lift``: S2's own grounding, lifted
here exactly like a locator's.
The live intrinsics must match the ones the hand-eye calibration was measured
with -- a resolution or profile change otherwise silently shifts every point.

Every detection above the score floor is reported, each under its own id, with
its score: perception never matches labels, merges look-alikes, or breaks ties
-- choosing among candidates is the caller's (S2's) judgement.

``SceneMemory`` (``scene_memory``) keeps the ids across runs and turns.

Pure Python: a mask or box subsampled to a few thousand points lifts in
milliseconds on the board without numpy.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from leapflow.hardware.model_contract import (
    GEOMETRY_MASK,
    GEOMETRY_POINT,
    TASK_GROUNDING,
    Detection,
    GroundingResult,
    MaskRle,
    ModelService,
    ground,
)
from leapflow.hardware.rgbd import CameraIntrinsics, FrameSource, RgbdFrame
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.geometry import Plane, Transform, Vec3, footprint
from leapflow.hardware.skills.ports import MotionFacade
from leapflow.hardware.skills.protocol import (
    COORDS_PER_MILLE,
    COORDS_UNIT,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_CALIBRATION_STALE,
    FAILURE_DEVICE_FAULT,
    FAILURE_MODEL_UNAVAILABLE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_TARGET_NOT_FOUND,
    SOURCE_LIFT,
    GraspCandidate,
    HeldObject,
    LocatedObject,
    PixelTarget,
    SkillError,
)
from leapflow.hardware.skills.scene_memory import SceneCapture, SceneMemory
from leapflow.hardware.skills.workcell import ROLE_CAMERA, WorkcellSpec
from leapflow.hardware.transport import TransportError

logger = logging.getLogger(__name__)

CAMERA_FRAME = "camera"
LOCATOR_MODEL = "locator"
_MAD_SCALE = 1.4826  # MAD -> standard deviation for normally distributed noise
_MAD_FLOOR_MM = 2.0  # never trust a spread tighter than the depth noise
# A shared frame is for S2 to point into, and must fit one LHP message.
_MIN_SNAPSHOT_SIDE = 64
_MAX_SNAPSHOT_SIDE = 1280


@dataclass(frozen=True)
class PerceptionSettings:
    """Perception tuning; a workcell overrides any of it under ``perception``.

    ``min_score`` is the floor below which a detection is not reported at all;
    ``locate`` callers may raise it per call. ``point_radius_px`` is the
    half-width of the window a point's depth is the median of. ``snapshot_*``
    and ``max_frames`` shape the frames ``snapshot`` shares and remembers.
    """

    max_points: float = 3000.0
    min_points: float = 40.0
    min_height_mm: float = 4.0
    min_depth_mm: float = 150.0
    max_depth_mm: float = 1500.0
    outlier_k: float = 3.0
    top_quantile: float = 0.95
    min_score: float = 0.3
    memory_ttl_s: float = 30.0
    jpeg_quality: float = 90.0
    point_radius_px: float = 4.0
    snapshot_max_side: float = 640.0
    snapshot_quality: float = 75.0
    max_frames: float = 2.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "PerceptionSettings":
        """Defaults overridden by *data*; raises ``ValueError`` on anything unusable."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"perception does not understand {', '.join(unknown)}")
        settings = replace(cls(), **{name: float(value) for name, value in data.items()})
        if not all(getattr(settings, name) > 0 for name in known):
            raise ValueError("perception values must be positive")
        if max(settings.top_quantile, settings.min_score) > 1:
            raise ValueError("perception quantiles and scores must be in (0, 1]")
        if settings.min_depth_mm >= settings.max_depth_mm:
            raise ValueError("perception.min_depth_mm must be below max_depth_mm")
        if max(settings.jpeg_quality, settings.snapshot_quality) > 100:
            raise ValueError("perception JPEG qualities must be in (0, 100]")
        if not _MIN_SNAPSHOT_SIDE <= settings.snapshot_max_side <= _MAX_SNAPSHOT_SIDE:
            raise ValueError(
                f"perception.snapshot_max_side must be in [{_MIN_SNAPSHOT_SIDE}, "
                f"{_MAX_SNAPSHOT_SIDE}] pixels"
            )
        return settings


# ── Lifting ─────────────────────────────────────────────────────────────────


def lift_mask(
    mask: MaskRle,
    frame: RgbdFrame,
    camera_to_base: Transform,
    table: Plane,
    settings: PerceptionSettings,
) -> list[Vec3]:
    """Base-frame points of *mask* standing on *table*, outliers removed.

    Raises ``ValueError`` when the mask does not describe this frame.
    """
    if (mask.height, mask.width) != (frame.height, frame.width):
        raise ValueError(
            f"mask is {mask.width}x{mask.height} but the frame is {frame.width}x{frame.height}"
        )
    stride = max(1, math.ceil(mask.area / settings.max_points))
    return _lift_pixels(mask.foreground(stride=stride), frame, camera_to_base, table, settings)


def lift_box(
    box: Sequence[float],
    frame: RgbdFrame,
    camera_to_base: Transform,
    table: Plane,
    settings: PerceptionSettings,
) -> list[Vec3]:
    """Base-frame points inside the pixel *box* standing on *table*, outliers removed.

    The box also covers background: the table drops out by height and stray
    depth by the outlier test, but a neighbour reaching into the box stays --
    a tight box, or a mask, is what keeps it out. Raises ``ValueError`` when
    the box lies outside the frame.
    """
    x0, y0, x1, y1 = box
    if x0 >= frame.width or y0 >= frame.height:
        raise ValueError(f"box {list(box)} lies outside the {frame.width}x{frame.height} frame")
    u0, v0 = int(x0), int(y0)
    u1 = max(u0, min(frame.width - 1, math.ceil(x1) - 1))
    v1 = max(v0, min(frame.height - 1, math.ceil(y1) - 1))
    area = (u1 - u0 + 1) * (v1 - v0 + 1)
    step = max(1, math.ceil(math.sqrt(area / settings.max_points)))
    pixels = ((u, v) for v in range(v0, v1 + 1, step) for u in range(u0, u1 + 1, step))
    return _lift_pixels(pixels, frame, camera_to_base, table, settings)


def lift_point(
    point: Sequence[float],
    frame: RgbdFrame,
    camera_to_base: Transform,
    settings: PerceptionSettings,
) -> tuple[Vec3, int] | None:
    """The base-frame surface point under pixel *point*, and how many depths support it.

    Depth is the median of the valid readings within ``point_radius_px`` --
    one pixel's reading is often missing or a flying edge pixel. ``None`` when
    none is valid; raises ``ValueError`` when the point lies outside the frame.
    """
    u, v = float(point[0]), float(point[1])
    if not (0.0 <= u < frame.width and 0.0 <= v < frame.height):
        raise ValueError(f"point {[u, v]} lies outside the {frame.width}x{frame.height} frame")
    depth = frame.depth_units()
    scale = frame.depth_scale_m * 1000.0
    radius = int(settings.point_radius_px)
    cu, cv = int(u), int(v)
    samples = []
    for y in range(max(0, cv - radius), min(frame.height, cv + radius + 1)):
        for x in range(max(0, cu - radius), min(frame.width, cu + radius + 1)):
            z = depth[y * frame.width + x] * scale
            if z and settings.min_depth_mm <= z <= settings.max_depth_mm:
                samples.append(z)
    if not samples:
        return None
    surface = frame.intrinsics.deproject(u, v, _median(samples))
    return camera_to_base.apply(surface), len(samples)


def _lift_pixels(
    pixels: Iterable[tuple[int, int]],
    frame: RgbdFrame,
    camera_to_base: Transform,
    table: Plane,
    settings: PerceptionSettings,
) -> list[Vec3]:
    depth = frame.depth_units()
    scale = frame.depth_scale_m * 1000.0
    intrinsics = frame.intrinsics
    width = frame.width
    points: list[Vec3] = []
    for u, v in pixels:
        raw = depth[v * width + u]
        if not raw:
            continue
        z = raw * scale
        if not settings.min_depth_mm <= z <= settings.max_depth_mm:
            continue
        point = camera_to_base.apply(intrinsics.deproject(u, v, z))
        if table.above(point) >= settings.min_height_mm:
            points.append(point)
    return reject_outliers(points, table, settings.outlier_k)


def lift_detection(
    detection: Detection, capture: SceneCapture, settings: PerceptionSettings
) -> LocatedObject:
    """*detection* as a base-frame object (``object_id`` left empty), by its geometry.

    Raises ``ValueError`` when its geometry does not fit the frame, and
    ``LookupError`` (saying why) when too little depth supports it.
    """
    frame, mount, table = capture.frame, capture.camera_to_base, capture.table
    if detection.geometry == GEOMETRY_POINT:
        lifted = lift_point(detection.point or (), frame, mount, settings)
        if lifted is None:
            raise LookupError("no valid depth around the point")
        center, support = lifted
        return LocatedObject(
            object_id="",
            label=detection.label,
            score=detection.score,
            center=center,
            length=0.0,
            width=0.0,
            height=0.0,
            yaw_deg=0.0,
            observed_at=frame.observed_at,
            points=support,
            geometry=GEOMETRY_POINT,
        )
    if detection.geometry == GEOMETRY_MASK:
        points = lift_mask(detection.mask, frame, mount, table, settings)  # type: ignore[arg-type]
    else:
        points = lift_box(detection.box, frame, mount, table, settings)  # type: ignore[arg-type]
    if len(points) < settings.min_points:
        raise LookupError(f"{len(points)} depth points")
    located = describe(
        points,
        table,
        label=detection.label,
        score=detection.score,
        observed_at=frame.observed_at,
        top_quantile=settings.top_quantile,
    )
    return replace(located, geometry=detection.geometry)


def reject_outliers(points: Sequence[Vec3], table: Plane, k: float) -> list[Vec3]:
    """Keep points within *k* robust deviations of the median in x, y, and height.

    Edge pixels of a mask mix object and background depth ("flying pixels");
    they sit far from the object's bulk and are what this removes.
    """
    if len(points) < 5:
        return list(points)
    columns = (
        [p[0] for p in points],
        [p[1] for p in points],
        [table.above(p) for p in points],
    )
    bounds = []
    for values in columns:
        middle = _median(values)
        spread = max(_MAD_FLOOR_MM, _MAD_SCALE * _median([abs(v - middle) for v in values]))
        bounds.append((middle - k * spread, middle + k * spread))
    return [
        point
        for point, *values in zip(points, *columns)
        if all(low <= value <= high for value, (low, high) in zip(values, bounds))
    ]


def describe(
    points: Sequence[Vec3],
    table: Plane,
    *,
    label: str,
    score: float,
    observed_at: float,
    top_quantile: float,
) -> LocatedObject:
    """The oriented box of *points* resting on *table* (``object_id`` left empty)."""
    shape = footprint(points)
    heights = sorted(table.above(p) for p in points)
    height = heights[round(top_quantile * (len(heights) - 1))]
    x, y = shape.center
    bottom = table.height(x, y)
    return LocatedObject(
        object_id="",
        label=label,
        score=score,
        center=(x, y, bottom + height / 2.0),
        length=shape.length,
        width=shape.width,
        height=height,
        yaw_deg=shape.yaw_deg,
        observed_at=observed_at,
        points=len(points),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


# ── Shared frames ────────────────────────────────────────────────────────────


def snapshot_size(frame: RgbdFrame, max_side: float) -> tuple[int, int]:
    """The ``(width, height)`` a frame is shared at: its longer side at most *max_side*."""
    scale = min(1.0, max_side / max(frame.width, frame.height))
    return (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))


def snapshot_jpeg(frame: RgbdFrame, size: tuple[int, int], quality: int) -> str:
    """The colour image at *size* as base64 JPEG."""
    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.color, "raw", "BGR")
    if size != (frame.width, frame.height):
        image = image.resize(size)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=min(100, max(1, quality)))
    return base64.b64encode(output.getvalue()).decode("ascii")


def target_detection(
    target: PixelTarget, frame: RgbdFrame, coords: str, shared: tuple[int, int]
) -> Detection:
    """*target* in full-frame pixels; *shared* is the size the frame was shown at."""
    if coords == COORDS_UNIT:
        sx, sy = float(frame.width), float(frame.height)
    elif coords == COORDS_PER_MILLE:
        sx, sy = frame.width / 1000.0, frame.height / 1000.0
    else:
        sx, sy = frame.width / shared[0], frame.height / shared[1]
    if target.point is not None:
        u, v = target.point
        return Detection(target.label, 1.0, point=(u * sx, v * sy))
    x0, y0, x1, y1 = target.box  # type: ignore[misc]
    return Detection(target.label, 1.0, box=(x0 * sx, y0 * sy, x1 * sx, y1 * sy))


def reproject(obj: LocatedObject, capture: SceneCapture) -> Detection:
    """Where *obj* would appear in *capture* if it has not moved: its point, or the
    pixel box its oriented box spans (clipped to the frame).

    Raises ``ValueError`` when it projects behind the camera or out of the frame.
    """
    to_camera = capture.camera_to_base.inverse()
    frame = capture.frame
    if obj.geometry == GEOMETRY_POINT:
        u, v = frame.intrinsics.project(to_camera.apply(obj.center))
        return Detection(obj.label, obj.score, point=(u, v))
    yaw = math.radians(obj.yaw_deg)
    cos, sin = math.cos(yaw), math.sin(yaw)
    x, y, z = obj.center
    pixels = [
        frame.intrinsics.project(
            to_camera.apply((x + dx * cos - dy * sin, y + dx * sin + dy * cos, z + dz))
        )
        for dx in (-obj.length / 2.0, obj.length / 2.0)
        for dy in (-obj.width / 2.0, obj.width / 2.0)
        for dz in (-obj.height / 2.0, obj.height / 2.0)
    ]
    x0 = max(0.0, min(u for u, _ in pixels))
    y0 = max(0.0, min(v for _, v in pixels))
    x1 = min(float(frame.width), max(u for u, _ in pixels))
    y1 = min(float(frame.height), max(v for _, v in pixels))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("it projects outside the frame")
    return Detection(obj.label, obj.score, box=(x0, y0, x1, y1))


# ── Scene ────────────────────────────────────────────────────────────────────


class WorkcellScene:
    """Perception for one skill context. Satisfies ``ScenePort`` and ``GraspSensing``."""

    def __init__(
        self,
        *,
        registry: Any,
        spec: WorkcellSpec,
        frames: FrameGraph,
        motion: MotionFacade,
        memory: SceneMemory,
        settings: PerceptionSettings,
        note: Callable[[str, Any], None],
    ) -> None:
        self._registry = registry
        self._spec = spec
        self._frames = frames
        self._motion = motion
        self._memory = memory
        self._settings = settings
        self._note = note

    @property
    def _basis(self) -> dict[str, float]:
        return dict(self._frames.versions)

    async def locate(
        self, queries: Sequence[str], *, min_score: float | None = None
    ) -> tuple[LocatedObject, ...]:
        floor = self._settings.min_score if min_score is None else float(min_score)
        found = await self._detect(queries, floor)
        return self._memory.remember(found, self._basis)

    async def snapshot(self) -> dict[str, Any]:
        capture = await self.capture()
        frame = capture.frame
        frame_id = self._memory.remember_frame(
            capture, self._basis, keep=int(self._settings.max_frames)
        )
        width, height = snapshot_size(frame, self._settings.snapshot_max_side)
        data = await asyncio.to_thread(
            snapshot_jpeg, frame, (width, height), int(self._settings.snapshot_quality)
        )
        self._note(
            "snapshot",
            {"frame": frame_id, "sequence": frame.sequence, "width": width, "height": height},
        )
        return {
            "frame": frame_id,
            "capture_id": capture.capture_id,
            "width": width,
            "height": height,
            "observed_at": frame.observed_at,
            "image": {"mime": "image/jpeg", "data": data},
        }

    async def lift(
        self, frame_id: str, targets: Sequence[PixelTarget], *, coords: str,
        capture_id: str | None = None,
    ) -> tuple[tuple[LocatedObject, ...], tuple[Mapping[str, str], ...]]:
        ttl = self._settings.memory_ttl_s
        capture = self._memory.recall_frame(frame_id, self._basis, ttl)
        if capture is None or (capture_id is not None and capture.capture_id != capture_id):
            raise SkillError(
                FAILURE_TARGET_NOT_FOUND,
                f"{frame_id} is not a current frame (frames last {ttl:.0f} s, only the "
                f"newest {int(self._settings.max_frames)} are kept, and a recalibration "
                "forgets them); take a new snapshot",
            )
        shown = snapshot_size(capture.frame, self._settings.snapshot_max_side)
        lifted: list[LocatedObject] = []
        dropped: list[Mapping[str, str]] = []
        for target in targets:
            try:
                detection = target_detection(target, capture.frame, coords, shown)
                seen = lift_detection(detection, capture, self._settings)
            except (LookupError, ValueError) as exc:
                dropped.append({"label": target.label, "reason": str(exc)})
                continue
            lifted.append(replace(seen, source=SOURCE_LIFT))
        self._note("lift", {"frame": frame_id, "lifted": len(lifted), "dropped": dropped})
        return self._memory.add(lifted, self._basis), tuple(dropped)

    def object(self, object_id: str) -> LocatedObject:
        obj = self._memory.recall(object_id, self._basis, self._settings.memory_ttl_s)
        if obj is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self._forgotten(object_id))
        return obj

    def remember_grasps(
        self, obj: LocatedObject, candidates: Sequence[GraspCandidate]
    ) -> tuple[GraspCandidate, ...]:
        return self._memory.remember_grasps(obj.object_id, candidates)

    def grasp(self, grasp_id: str) -> GraspCandidate:
        found = self._memory.recall_grasp(grasp_id, self._basis, self._settings.memory_ttl_s)
        if found is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self._forgotten(grasp_id))
        return found

    def _forgotten(self, handle: str) -> str:
        ttl = self._settings.memory_ttl_s
        return (
            f"{handle} is not a current observation (handles last {ttl:.0f} s and are "
            "forgotten on a new locate or a recalibration); locate again"
        )

    async def relocate(self, obj: LocatedObject) -> LocatedObject:
        """Measure only *obj* where its approved geometry projects in a fresh frame.

        Freshness must never run grounding again: another same-label detection is
        not the candidate the operator approved. The caller's drift check catches
        movement within the projected region; missing depth means the candidate left.
        """
        return await self._remeasure(obj)

    async def _remeasure(self, obj: LocatedObject) -> LocatedObject:
        capture = await self.capture()
        try:
            seen = lift_detection(reproject(obj, capture), capture, self._settings)
        except (LookupError, ValueError) as exc:
            observe_again = "lift it again" if obj.source == SOURCE_LIFT else "locate it again"
            raise SkillError(
                FAILURE_TARGET_NOT_FOUND,
                f"{obj.object_id} ({obj.label}) cannot be measured again ({exc}); "
                f"{observe_again} and make a new plan",
            ) from exc
        moved = replace(seen, object_id=obj.object_id, source=obj.source)
        self._memory.update(moved, self._basis)
        return moved

    @property
    def held(self) -> HeldObject | None:
        return self._memory.held

    def hold(self, held: HeldObject) -> None:
        self._memory.hold(held)

    def release(self) -> HeldObject | None:
        return self._memory.release()

    # ── GraspSensing ──

    async def capture(self, role: str = ROLE_CAMERA) -> SceneCapture:
        frame, camera_to_base, table = await self._observe(role)
        return SceneCapture(frame, camera_to_base, table)

    async def model(self, name: str, task: str) -> ModelService:
        """The workcell's *name* model, verified now to serve *task*."""
        device = self._spec.model(name)
        if not device:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"workcell {self._spec.workcell_id!r} binds no {name!r} model",
            )
        service = await self._transport(
            device, ModelService, "a model service", code=FAILURE_MODEL_UNAVAILABLE
        )
        try:
            info = await service.refresh()
        except TransportError as exc:
            raise SkillError(FAILURE_MODEL_UNAVAILABLE, f"{name} {device!r}: {exc}") from exc
        if info.task != task:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE,
                f"model {info.name!r} serves {info.task!r}, not {task!r}",
            )
        return service

    # ── Internals ──

    async def _detect(self, queries: Sequence[str], min_score: float) -> list[LocatedObject]:
        capture = await self.capture()
        frame = capture.frame
        result, model = await self._ground(frame, queries)
        kept: list[LocatedObject] = []
        dropped: list[dict[str, Any]] = []
        for detection in sorted(result.detections, key=lambda d: -d.score):
            if detection.score < min_score:
                dropped.append({"label": detection.label, "reason": "below min_score"})
                continue
            try:
                kept.append(lift_detection(detection, capture, self._settings))
            except LookupError as exc:
                dropped.append({"label": detection.label, "reason": str(exc.args[0])})
            except ValueError as exc:
                raise SkillError(
                    FAILURE_MODEL_UNAVAILABLE,
                    f"locator {model!r} returned an unusable {detection.geometry}: {exc}",
                ) from exc
        self._note(
            "perception",
            {
                "model": model,
                "infer_ms": round(result.infer_ms, 1),
                "frame": frame.sequence,
                "detections": len(result.detections),
                "located": len(kept),
                "dropped": dropped,
            },
        )
        return kept

    async def _observe(self, role: str = ROLE_CAMERA) -> tuple[RgbdFrame, Transform, Plane]:
        if role.split(".", 1)[0] != ROLE_CAMERA:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"{role!r} is not a camera role")
        camera = self._spec.role(role)
        if not camera:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"workcell {self._spec.workcell_id!r} binds no {role}"
            )
        mount = self._frames.transform(role)
        if mount is None:
            raise SkillError(
                FAILURE_CALIBRATION_MISSING, "the camera is not calibrated (run calib.handeye)"
            )
        table = self._frames.table
        if table is None:
            raise SkillError(
                FAILURE_CALIBRATION_MISSING,
                "the workspace was calibrated without the table; run calib.workspace with "
                "table_z or table_points",
            )
        flange = None
        if self._frames.parent(role) == "flange":
            flange = (await self._motion.state()).flange
            mount = Transform.from_pose(flange).compose(mount)
        try:
            # Resolved under the device lock, not before it: a preview release waiting on
            # the same lock may otherwise close the camera between lookup and capture.
            async with self._registry.device_io(camera):
                source = await self._transport(camera, FrameSource, "an RGB-D camera")
                frame = await source.capture_rgbd()
        except TransportError as exc:
            raise SkillError(
                FAILURE_DEVICE_FAULT, f"camera {camera!r} capture failed: {exc}"
            ) from exc
        if flange is not None:
            after = (await self._motion.state()).flange
            limits = self._spec.grasp_verify
            if (flange.distance_mm(after) > limits["capture_drift_mm"]
                    or flange.rotation_to_deg(after) > limits["capture_drift_deg"]):
                raise SkillError(FAILURE_CALIBRATION_STALE, "the wrist moved during camera capture")
        self._check_intrinsics(frame.intrinsics, role=role)
        return frame, mount, table

    def _check_intrinsics(self, live: CameraIntrinsics, *, role: str = ROLE_CAMERA) -> None:
        recorded = self._frames.parameters(role).get("intrinsics")
        if recorded is None:
            return  # a hand-entered mount; calib.handeye always records them
        try:
            expected = CameraIntrinsics.from_mapping(recorded)
        except (KeyError, TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_CALIBRATION_STALE,
                f"the hand-eye record's intrinsics are unreadable ({exc}); run calib.handeye again",
            ) from exc
        problem = expected.differs_from(live)
        if problem:
            raise SkillError(
                FAILURE_CALIBRATION_STALE,
                f"the camera stream changed since hand-eye calibration ({problem}); run "
                "calib.handeye again",
            )

    async def _ground(
        self, frame: RgbdFrame, queries: Sequence[str]
    ) -> tuple[GroundingResult, str]:
        service = await self.model(LOCATOR_MODEL, TASK_GROUNDING)
        name = service.model_info.name
        try:
            image = await asyncio.to_thread(
                frame.color_jpeg, quality=int(self._settings.jpeg_quality)
            )
            return await ground(service, image, list(queries)), name
        except TransportError as exc:
            raise SkillError(FAILURE_MODEL_UNAVAILABLE, f"locator {name!r} failed: {exc}") from exc

    async def _transport(
        self, device_id: str, capability: type, what: str, *, code: str = FAILURE_SKILL_UNAVAILABLE
    ) -> Any:
        try:
            transport = await self._registry.transport(device_id)
        except TransportError as exc:
            raise SkillError(code, f"{device_id!r} is unavailable: {exc}") from exc
        if not isinstance(transport, capability):
            raise SkillError(code, f"{device_id!r} is not {what}")
        return transport


__all__ = [
    "CAMERA_FRAME",
    "LOCATOR_MODEL",
    "PerceptionSettings",
    "WorkcellScene",
    "describe",
    "lift_box",
    "lift_detection",
    "lift_mask",
    "lift_point",
    "reject_outliers",
    "reproject",
    "snapshot_jpeg",
    "snapshot_size",
    "target_detection",
]

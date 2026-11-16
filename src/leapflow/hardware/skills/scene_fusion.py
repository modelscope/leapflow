# Copyright (c) Alibaba, Inc. and its affiliates.
"""Cross-view fusion: what perception holds in base coordinates, seen from every camera.

Objects, the TCP, and the cameras themselves live in the arm base frame, so a
calibrated view can say where each of them should appear -- and its depth can
say whether it does. That is the semantic link between a global view and a
wrist view: the same ``oN`` is drawn wherever it projects, and each view
reports the object as

* ``visible`` -- the measured depth at its remembered top agrees;
* ``occluded`` -- something nearer stands in front (the gripper, a hand);
* ``moved_or_gone`` -- the camera sees past where its surface was;
* ``no_depth`` -- the camera has no reading there;
* ``out_of_view`` -- it does not project into this frame.

None of it compares pixels, so it holds while the camera itself moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from leapflow.hardware.model_contract import GEOMETRY_POINT
from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Plane, Transform, Vec3, dot
from leapflow.hardware.skills.protocol import LocatedObject
from leapflow.hardware.skills.scene_memory import FrameView, SceneCapture
from leapflow.hardware.skills.visual_change import ChangeReport, FrameStats
from leapflow.hardware.skills.visual_tiles import Mark

PRESENCE_TOLERANCE_MM = 20.0
PRESENCE_TOLERANCE = 0.05  # of the expected depth, when that is larger
ATTENTION_STATES = ("occluded", "moved_or_gone")


@dataclass(frozen=True)
class FusionView:
    """One camera frame as shown: its id, camera role, capture, and where it sits."""

    frame_id: str
    camera: str
    capture: SceneCapture
    shown: FrameView
    # Tile position in the image S2 sees; ``None`` when no tile of the frame is liftable.
    offset: tuple[int, int] | None = (0, 0)

    def to_shown(self, u: float, v: float) -> tuple[int, int] | None:
        """Image pixel S2 sees for full-frame pixel ``(u, v)``."""
        if self.offset is None:
            return None
        x0, y0, x1, y1 = self.shown.window
        return (
            round(self.offset[0] + (u - x0) * self.shown.size[0] / (x1 - x0)),
            round(self.offset[1] + (v - y0) * self.shown.size[1] / (y1 - y0)),
        )


@dataclass(frozen=True)
class Fusion:
    """Per object, how every calibrated view sees it; per frame, what to draw; per camera, its aim."""

    objects: tuple[dict[str, Any], ...] = ()
    marks: Mapping[str, tuple[Mark, ...]] = field(default_factory=dict)
    aims: Mapping[str, dict[str, Any]] = field(default_factory=dict)


def fuse(
    views: Sequence[FusionView],
    objects: Sequence[LocatedObject],
    *,
    tcp: Vec3 | None,
    settings: Any,
) -> Fusion:
    """Project every object, the TCP, and every other camera into each calibrated view."""
    mounts = {view.camera: view.capture.camera_to_base for view in views
              if view.capture.camera_to_base is not None}
    table = next((view.capture.table for view in views if view.capture.table is not None), None)
    aims: dict[str, dict[str, Any]] = {}
    for camera, mount in mounts.items():
        hit = aim(mount, table)
        if hit is not None:
            aims[camera] = {"table_point_mm": [round(v) for v in hit[0]], "range_mm": round(hit[1])}
    seen: dict[str, dict[str, Any]] = {obj.object_id: {} for obj in objects}
    marks: dict[str, tuple[Mark, ...]] = {}
    for view in views:
        mount = view.capture.camera_to_base
        if mount is None:
            continue
        to_camera, frame = mount.inverse(), view.capture.frame
        drawn: list[Mark] = []
        for obj in objects:
            state, box, pixel, numbers = presence(obj, frame, to_camera, settings)
            entry = {"state": state, **numbers}
            shown = None if pixel is None else view.to_shown(*pixel)
            if shown is not None:
                entry["pixel"] = list(shown)
            seen[obj.object_id][view.camera] = entry
            if box is not None:
                kind = "point" if obj.geometry == GEOMETRY_POINT else "box"
                tone = "object" if state == "visible" else "warn"
                drawn.append(Mark(kind, box, f"{obj.object_id} {obj.label}", tone))
        if tcp is not None:
            at = project(frame, to_camera, tcp)
            if at is not None:
                drawn.append(Mark("cross", at, "TCP", "tcp"))
        for other, other_mount in mounts.items():
            if other == view.camera:
                continue
            origin = project(frame, to_camera, other_mount.translation)
            if origin is not None:
                drawn.append(Mark("ring", origin, other, "camera"))
            if other in aims:
                hit_px = project(frame, to_camera, aims[other]["table_point_mm"])
                if hit_px is not None:
                    drawn.append(Mark("cross", hit_px, f"{other} aim", "camera"))
        marks[view.frame_id] = tuple(drawn)
    fused = tuple(
        {"id": obj.object_id, "label": obj.label, "views": seen[obj.object_id]}
        for obj in objects if seen[obj.object_id]
    )
    return Fusion(objects=fused, marks=marks, aims=aims)


def situation(
    stats: Mapping[str, FrameStats],
    changes: Mapping[str, ChangeReport],
    objects: Sequence[Mapping[str, Any]],
    unavailable: Mapping[str, str],
    sync: Mapping[str, Any] | None,
) -> list[str]:
    """What changed or needs attention, in a few short sentences."""
    lines = [report.summary() for report in changes.values()]
    lines.extend(f"{camera}: {', '.join(found.flags)}" for camera, found in stats.items() if found.flags)
    for entry in objects:
        for camera, view in entry["views"].items():
            if view["state"] in ATTENTION_STATES:
                lines.append(
                    f"{entry['id']} {entry['label']} in {camera}: {view['state']} (expected "
                    f"{view.get('expected_mm')} mm, measured {view.get('measured_mm')} mm)"
                )
    if sync is not None and not sync.get("within_tolerance", True):
        lines.append(
            f"cameras captured {sync['skew_ms']:.0f} ms apart "
            f"(tolerance {sync['tolerance_ms']:.0f} ms)"
        )
    lines.extend(f"{name} unavailable: {reason}" for name, reason in unavailable.items())
    return lines


def project(frame: RgbdFrame, to_camera: Transform, point: Sequence[float]) -> tuple[float, float] | None:
    """Full-frame pixel of base-frame *point*, or ``None`` when behind or outside."""
    local = to_camera.apply(point)
    if local[2] <= 1.0:
        return None
    u, v = frame.intrinsics.project(local)
    return (u, v) if 0.0 <= u < frame.width and 0.0 <= v < frame.height else None


def presence(
    obj: LocatedObject, frame: RgbdFrame, to_camera: Transform, settings: Any
) -> tuple[str, tuple[float, ...] | None, tuple[float, float] | None, dict[str, Any]]:
    """How *frame* sees remembered *obj*: its state, drawable mark, top pixel, and depths.

    The object's top centre is projected and its expected camera depth compared
    with the median measured around it.
    """
    x, y, z = obj.center
    top = obj.center if obj.geometry == GEOMETRY_POINT else (x, y, z + obj.height / 2.0)
    local = to_camera.apply(top)
    corners = [to_camera.apply(corner) for corner in _corners(obj)]
    if local[2] <= 1.0 or any(corner[2] <= 1.0 for corner in corners):
        return "out_of_view", None, None, {}
    u, v = frame.intrinsics.project(local)
    pixels = [frame.intrinsics.project(corner) for corner in corners]
    x0, y0 = max(0.0, min(p[0] for p in pixels)), max(0.0, min(p[1] for p in pixels))
    x1 = min(float(frame.width), max(p[0] for p in pixels))
    y1 = min(float(frame.height), max(p[1] for p in pixels))
    if obj.geometry == GEOMETRY_POINT:
        mark: tuple[float, ...] | None = (u, v)
    else:
        mark = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    if not (0.0 <= u < frame.width and 0.0 <= v < frame.height):
        return "out_of_view", None if obj.geometry == GEOMETRY_POINT else mark, None, {}
    expected = local[2]
    measured = depth_near(frame, u, v, int(settings.point_radius_px), settings)
    numbers: dict[str, Any] = {"expected_mm": round(expected)}
    if measured is None:
        return "no_depth", mark, (u, v), numbers
    numbers["measured_mm"] = round(measured)
    tolerance = max(PRESENCE_TOLERANCE_MM, PRESENCE_TOLERANCE * expected)
    if measured < expected - tolerance:
        state = "occluded"
    elif measured > expected + tolerance:
        state = "moved_or_gone"
    else:
        state = "visible"
    return state, mark, (u, v), numbers


def depth_near(frame: RgbdFrame, u: float, v: float, radius: int, settings: Any) -> float | None:
    """Median valid depth (mm) within *radius* pixels of ``(u, v)``."""
    depth, scale = frame.depth_units(), frame.depth_scale_m * 1000.0
    cu, cv = int(u), int(v)
    samples = sorted(
        z
        for row in range(max(0, cv - radius), min(frame.height, cv + radius + 1))
        for col in range(max(0, cu - radius), min(frame.width, cu + radius + 1))
        if (z := depth[row * frame.width + col] * scale)
        and settings.min_depth_mm <= z <= settings.max_depth_mm
    )
    return samples[len(samples) // 2] if samples else None


def aim(mount: Transform, table: Plane | None) -> tuple[Vec3, float] | None:
    """Where a camera's optical axis meets the table, and how far away it is."""
    if table is None:
        return None
    rotation = mount.rotation
    axis = (rotation[0][2], rotation[1][2], rotation[2][2])
    facing = dot(table.normal, axis)
    if abs(facing) < 1e-6:
        return None
    distance = (table.offset - dot(table.normal, mount.translation)) / facing
    if distance <= 0:
        return None
    ox, oy, oz = mount.translation
    return (ox + axis[0] * distance, oy + axis[1] * distance, oz + axis[2] * distance), distance


def _corners(obj: LocatedObject) -> list[Vec3]:
    if obj.geometry == GEOMETRY_POINT:
        return [obj.center]
    yaw = math.radians(obj.yaw_deg)
    cos, sin = math.cos(yaw), math.sin(yaw)
    x, y, z = obj.center
    return [
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos, z + dz)
        for dx in (-obj.length / 2.0, obj.length / 2.0)
        for dy in (-obj.width / 2.0, obj.width / 2.0)
        for dz in (-obj.height / 2.0, obj.height / 2.0)
    ]


__all__ = [
    "ATTENTION_STATES",
    "Fusion",
    "FusionView",
    "aim",
    "depth_near",
    "fuse",
    "presence",
    "project",
    "situation",
]

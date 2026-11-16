# Copyright (c) Alibaba, Inc. and its affiliates.
"""What S2 sees of the workcell: every camera, every plane, one image, and what changed.

A workcell usually has more than one camera -- one that sees the whole table,
one on the wrist that sees what the gripper is about to touch -- and each
streams more than colour: aligned depth, often infrared. A model reasons about
them best together, so one ``snapshot`` may:

* capture several camera roles against one freshness bound (``CaptureGroup``)
  and say how far apart they landed (``sync``);
* show each camera's requested planes as one row of one mosaic image, sized to
  a token budget (``detail``), remembered as ``vN``;
* draw what perception holds in base coordinates into every calibrated view and
  check it against the new depth (``scene_fusion``);
* compare each camera's view with its previous one (``visual_change``) and say
  in words where the scene changed (``situation``).

Pointing works in whatever S2 was shown: a ``vN`` point or box resolves to the
tile under it and that tile's frame; a ``zoom`` re-shows a window of a
remembered frame at native resolution under a new frame id, and its pixels
resolve through that window. Nothing here knows a camera model -- a camera is a
role bound to any ``FrameSource``, infrared any declared ``infrared_image``
frame channel.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from leapflow.hardware.model_contract import Detection
from leapflow.hardware.skills import scene_fusion, visual_change
from leapflow.hardware.skills import visual_tiles as tiles
from leapflow.hardware.skills.protocol import (
    COORDS_PER_MILLE,
    COORDS_PIXEL,
    COORDS_UNIT,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_INVALID_ARGS,
    FAILURE_TARGET_NOT_FOUND,
    MAX_VIEW_CAMERAS,
    VIEW_DETAILS,
    PixelTarget,
    SkillError,
)
from leapflow.hardware.skills.scene_memory import FrameView, SceneCapture, ViewTile
from leapflow.hardware.skills.workcell import ROLE_CAMERA
from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision.frames import accepts_keyword
from leapflow.hardware.vision.group import CaptureGroup

if TYPE_CHECKING:
    from leapflow.hardware.skills.perception import WorkcellScene

logger = logging.getLogger(__name__)

DETAILS = VIEW_DETAILS
WITH_ALL = "all"
INFRARED_QUANTITY = "infrared_image"
MAX_TILE_SIDE = 1280
_ZOOM_MIN_PX = 16
_ZOOM_SHOWN_PX = 320  # a small window is upscaled (at most 2x) to about this side
_READ_TIMEOUT_S = 5.0
_SHRINK_ROUNDS = 4

Resolved = tuple[PixelTarget, "SceneCapture | None", "Detection | None", str]
"""A lift target with the capture and full-frame detection it resolved to, or why not."""


@dataclass(frozen=True)
class ViewRequest:
    """What one snapshot shows: cameras, planes, how much detail, and overlays."""

    camera: str = ROLE_CAMERA
    with_cameras: tuple[str, ...] = ()
    planes: tuple[str, ...] = (tiles.PLANE_COLOR,)
    detail: str = "auto"
    overlay: bool = True

    def __post_init__(self) -> None:
        if not self.planes or set(self.planes) - set(tiles.PLANES):
            raise ValueError(f"planes are a non-empty subset of {', '.join(tiles.PLANES)}")
        if self.detail not in DETAILS:
            raise ValueError(f"detail is one of {', '.join(DETAILS)}")


@dataclass(frozen=True)
class _Tile:
    """One tile to render: which frame, which plane, at what size, through which window."""

    camera: str
    plane: str
    frame_id: str
    capture: SceneCapture
    size: tuple[int, int]
    window: tuple[float, float, float, float] | None = None
    encoded: Any = None  # the FrameReading of a plane read beside the RGB-D frame

    @property
    def liftable_plane(self) -> bool:
        return self.plane != tiles.PLANE_INFRARED


def target_detection(
    target: PixelTarget,
    frame: Any,
    coords: str,
    shared: tuple[int, int],
    window: Sequence[float] | None = None,
) -> Detection:
    """*target* in full-frame pixels; *shared* is the size the frame (or its
    *window*) was shown at."""
    x0, y0, x1, y1 = window or (0.0, 0.0, float(frame.width), float(frame.height))
    if coords == COORDS_UNIT:
        sx, sy = x1 - x0, y1 - y0
    elif coords == COORDS_PER_MILLE:
        sx, sy = (x1 - x0) / 1000.0, (y1 - y0) / 1000.0
    else:
        sx, sy = (x1 - x0) / shared[0], (y1 - y0) / shared[1]
    if target.point is not None:
        u, v = target.point
        return Detection(target.label, 1.0, point=(x0 + u * sx, y0 + v * sy))
    bx0, by0, bx1, by1 = target.box  # type: ignore[misc]
    return Detection(
        target.label, 1.0, box=(x0 + bx0 * sx, y0 + by0 * sy, x0 + bx1 * sx, y0 + by1 * sy)
    )


class SceneViewer:
    """Snapshots, zooms, and lift resolution for one ``WorkcellScene``.

    Holds a back-reference to the scene and reads its state on every call, so a
    re-injected registry or setting is never stale.
    """

    def __init__(self, scene: "WorkcellScene") -> None:
        self._scene = scene

    # ── Snapshot ──

    async def snapshot(self, request: ViewRequest) -> dict[str, Any]:
        roles, unavailable = self._roles(request)
        captures, failed, sync = await self._capture(roles)
        unavailable.update(failed)
        readings: dict[str, Any] = {}
        if tiles.PLANE_INFRARED in request.planes:
            readings = await self._infrared(captures, unavailable)
        rows = self._plan(request, captures, readings)
        state = await self._arm_state()
        signatures = await asyncio.to_thread(self._signatures, rows)
        changes = self._changes(signatures)
        fusion = self._fuse(rows, state)
        data, spans, placed = await asyncio.to_thread(self._render, rows, fusion.marks, request.overlay)
        shot = self._result(rows[0][0], placed, data)
        if len(placed) > 1:
            shot["tiles"] = [view.to_dict() for view in placed]
            shot["frames"] = self._frames_of(rows, spans)
        elif spans:
            shot["depth_range_mm"] = [round(value) for value in next(iter(spans.values()))]
        stats = {role: found for role, (_, found) in signatures.items()}
        shot["stats"] = {role: found.to_dict() for role, found in stats.items()}
        if changes:
            shot["changes"] = {role: report.to_dict() for role, report in changes.items()}
        if fusion.objects:
            shot["objects"] = list(fusion.objects)
        if state is not None:
            shot["arm"] = self._arm(state)
        if fusion.aims:
            shot["aims"] = dict(fusion.aims)
        if sync is not None:
            shot["sync"] = sync
        if unavailable:
            shot["unavailable"] = unavailable
        lines = scene_fusion.situation(stats, changes, fusion.objects, unavailable, sync)
        if lines:
            shot["situation"] = lines
        self._scene._note(
            "snapshot",
            {
                "frame": shot["frame"], "camera": shot["camera"],
                "sequence": rows[0][0].capture.frame.sequence, "width": shot["width"],
                "height": shot["height"], "liftable": shot["liftable"],
                "cameras": [row[0].camera for row in rows], "planes": list(request.planes),
                "unavailable": sorted(unavailable),
            },
        )
        return shot

    async def zoom(
        self, frame_id: str, box: Sequence[float], *, coords: str, planes: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Re-show the *box* window of remembered frame *frame_id* at native resolution.

        The crop is a new frame id with the original's age: zooming never makes an
        old observation look fresh, and pointing into it lifts in the full frame.
        A ``vN`` box zooms into the tile its centre lands in.
        """
        scene, settings = self._scene, self._scene._settings
        if frame_id.startswith("v"):
            frame_id, box, coords = self._zoom_tile(frame_id, box, coords)
        capture = scene._memory.recall_frame(frame_id, scene._basis, settings.frame_ttl_s)
        if capture is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self.not_current(frame_id))
        shown = self.shown(capture)
        asked = target_detection(
            PixelTarget("zoom", box=tuple(box)), capture.frame, coords, shown.size, shown.window
        ).box
        frame = capture.frame
        window = (max(0.0, asked[0]), max(0.0, asked[1]),
                  min(float(frame.width), asked[2]), min(float(frame.height), asked[3]))
        width, height = window[2] - window[0], window[3] - window[1]
        if min(width, height) < _ZOOM_MIN_PX:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"the zoom box covers less than {_ZOOM_MIN_PX} frame pixels per side; widen it",
            )
        longest = max(width, height)
        scale = min(settings.snapshot_max_side / longest, max(1.0, min(2.0, _ZOOM_SHOWN_PX / longest)))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        zoomed = replace(capture, view=FrameView(window, size))
        new_id = scene._memory.remember_frame(
            zoomed, scene._basis, keep=int(settings.max_frames),
            seen_at=scene._memory.frame_seen_at(frame_id),
        )
        wanted = [p for p in (tiles.PLANE_COLOR, tiles.PLANE_DEPTH) if p in (planes or ("color",))]
        row = [_Tile(capture.camera, plane, new_id, zoomed, size, window)
               for plane in wanted or [tiles.PLANE_COLOR]]
        fusion = self._fuse([row], await self._arm_state())
        data, spans, placed = await asyncio.to_thread(self._render, [row], fusion.marks, True)
        shot = self._result(row[0], placed, data)
        shot["zoom_of"] = frame_id
        shot["window"] = [round(value, 1) for value in window]
        if len(placed) > 1:
            shot["tiles"] = [view.to_dict() for view in placed]
        if spans:
            shot["depth_range_mm"] = [round(value) for value in next(iter(spans.values()))]
        if fusion.objects:
            shot["objects"] = list(fusion.objects)
        scene._note("zoom", {"frame": shot["frame"], "of": frame_id, "window": shot["window"]})
        return shot

    def _zoom_tile(
        self, view_id: str, box: Sequence[float], coords: str
    ) -> tuple[str, Sequence[float], str]:
        """The frame and tile-pixel box a mosaic zoom box lands in."""
        scene = self._scene
        views = scene._memory.recall_view(view_id, scene._basis, scene._settings.frame_ttl_s)
        if views is None:
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self.not_current(view_id))
        size = (max(v.x + v.width for v in views), max(v.y + v.height for v in views))
        local, tile, reason = _into_tile(PixelTarget("zoom", box=tuple(box)), views, coords, size)
        if local is None or tile is None:
            raise SkillError(FAILURE_INVALID_ARGS, f"cannot zoom there: {reason}")
        return tile.frame_id, local.box or (), COORDS_PIXEL

    # ── Lift resolution ──

    def resolve(
        self, handle: str, targets: Sequence[PixelTarget], *, coords: str,
        capture_id: str | None = None,
    ) -> list[Resolved]:
        """Each target as a full-frame detection in the frame it points into.

        A frame handle (``fN``) that cannot be lifted at all is refused outright; a
        view handle (``vN``) drops only the targets that land in an unliftable tile.
        """
        scene, ttl = self._scene, self._scene._settings.frame_ttl_s
        if handle.startswith("v"):
            return self._resolve_view(handle, targets, coords=coords, capture_id=capture_id)
        capture = scene._memory.recall_frame(handle, scene._basis, ttl)
        if capture is None or (capture_id is not None and capture.capture_id != capture_id):
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self.not_current(handle))
        if not capture.liftable:
            raise SkillError(FAILURE_CALIBRATION_MISSING, scene._unplaced(handle, capture))
        shown = self.shown(capture)
        return [
            (target, capture,
             target_detection(target, capture.frame, coords, shown.size, shown.window), "")
            for target in targets
        ]

    def _resolve_view(
        self, view_id: str, targets: Sequence[PixelTarget], *, coords: str, capture_id: str | None
    ) -> list[Resolved]:
        memory, basis = self._scene._memory, self._scene._basis
        ttl = self._scene._settings.frame_ttl_s
        views = memory.recall_view(view_id, basis, ttl)
        primary = memory.recall_frame(views[0].frame_id, basis, ttl) if views else None
        if views is None or (capture_id is not None and getattr(primary, "capture_id", "") != capture_id):
            raise SkillError(FAILURE_TARGET_NOT_FOUND, self.not_current(view_id))
        size = (max(v.x + v.width for v in views), max(v.y + v.height for v in views))
        resolved: list[Resolved] = []
        for target in targets:
            local, tile, reason = _into_tile(target, views, coords, size)
            capture = None
            if tile is not None and local is not None:
                capture = memory.recall_frame(tile.frame_id, basis, ttl)
                if capture is None:
                    reason = f"{tile.frame_id} ({tile.camera}) is no longer current"
                elif not capture.liftable:
                    reason, capture = self._scene._unplaced(tile.frame_id, capture), None
            if capture is None or local is None:
                resolved.append((target, None, None, reason))
                continue
            shown = self.shown(capture)
            resolved.append((target, capture, target_detection(
                local, capture.frame, COORDS_PIXEL, shown.size, shown.window), ""))
        return resolved

    def shown(self, capture: SceneCapture) -> FrameView:
        """How *capture* was shown; a frame remembered without a view was shown downscaled."""
        if capture.view is not None:
            return capture.view
        frame = capture.frame
        return FrameView(
            (0.0, 0.0, float(frame.width), float(frame.height)),
            tiles.snapshot_size(frame, self._scene._settings.snapshot_max_side),
        )

    def not_current(self, handle: str) -> str:
        settings = self._scene._settings
        return (
            f"{handle} is not a current frame (frames last {settings.frame_ttl_s:.0f} s, only "
            f"the newest {int(settings.max_frames)} are kept, and a recalibration forgets "
            "them); take a new snapshot"
        )

    # ── Capture ──

    def _roles(self, request: ViewRequest) -> tuple[list[str], dict[str, str]]:
        """The primary camera first, then the others asked for (``all`` = every camera role)."""
        cameras = [r for r in sorted(self._scene._spec.roles) if r.split(".", 1)[0] == ROLE_CAMERA]
        wanted = [request.camera]
        for name in request.with_cameras:
            wanted.extend(cameras if name == WITH_ALL else [name])
        roles = list(dict.fromkeys(wanted))
        skipped = {
            role: f"more than {MAX_VIEW_CAMERAS} cameras in one snapshot; ask for it separately"
            for role in roles[MAX_VIEW_CAMERAS:]
        }
        return roles[:MAX_VIEW_CAMERAS], skipped

    async def _capture(
        self, roles: Sequence[str]
    ) -> tuple[dict[str, SceneCapture], dict[str, str], dict[str, Any] | None]:
        scene = self._scene
        if len(roles) == 1:
            return {roles[0]: await scene._observe(roles[0], placed=False)}, {}, None
        group = CaptureGroup({
            role: (lambda after, r=role: scene._observe(r, placed=False, after_monotonic=after))
            for role in roles
        })
        result = await group.capture()
        if roles[0] in result.errors:
            raise result.errors[roles[0]]  # the camera asked for must answer; the others may not
        captures = {role: result.frames[role] for role in roles if role in result.frames}
        return captures, {role: _reason(exc) for role, exc in result.errors.items()}, result.summary()

    async def _infrared(
        self, captures: Mapping[str, SceneCapture], unavailable: dict[str, str]
    ) -> dict[str, Any]:
        """One infrared preview per camera that declares one, read beside its RGB-D frame."""
        registry, settings = self._scene._registry, self._scene._settings

        async def one(role: str, capture: SceneCapture) -> tuple[str, Any, str]:
            device = self._scene._spec.role(role)
            channel = _infrared_channel(registry.context(device))
            if channel is None:
                return role, None, ""  # none declared: no plane, and nothing failed
            read = getattr(registry, "read_frame", None)
            if read is None:
                return role, None, "this node cannot read frame channels"
            options: dict[str, Any] = {
                "max_width": MAX_TILE_SIDE, "quality": int(settings.snapshot_quality)
            }
            try:
                transport = await registry.transport(device)
                method = getattr(transport, "read_frame", None)
                if method is not None and accepts_keyword(method, "after_monotonic"):
                    options["after_monotonic"] = capture.monotonic_at - 1e-3
                reading = await asyncio.wait_for(read(device, channel, **options), _READ_TIMEOUT_S)
            except (TransportError, TimeoutError) as exc:
                return role, None, f"{channel}: {exc or 'timed out'}"
            return role, reading, ""

        readings: dict[str, Any] = {}
        for role, reading, reason in await asyncio.gather(
            *(one(role, capture) for role, capture in captures.items())
        ):
            if reading is not None:
                readings[role] = reading
            elif reason:
                unavailable[f"{role}.{tiles.PLANE_INFRARED}"] = reason
        return readings

    # ── Layout and analysis ──

    def _plan(
        self, request: ViewRequest, captures: Mapping[str, SceneCapture], readings: Mapping[str, Any]
    ) -> list[list[_Tile]]:
        """One row per camera, one tile per plane, sized to the detail and the mosaic budget;
        every camera's frame is remembered with the size it is shown at."""
        scene, settings = self._scene, self._scene._settings
        planes = {
            role: [
                plane for plane in tiles.PLANES
                if plane in request.planes and (plane != tiles.PLANE_INFRARED or role in readings)
            ] or [tiles.PLANE_COLOR]
            for role in captures
        }
        side = self._side(request.detail, sum(len(found) for found in planes.values()))
        for _ in range(_SHRINK_ROUNDS):
            sizes = {
                role: [_size(captures[role], readings.get(role), plane, side) for plane in found]
                for role, found in planes.items()
            }
            _, (width, height) = tiles.layout(list(sizes.values()))
            if max(width, height) <= settings.mosaic_max_side:
                break
            # Shrink from the side tiles are actually shown at: a frame never upscales,
            # so a requested side above the native one would shrink nothing.
            shown = max(max(size) for found in sizes.values() for size in found)
            side = shown * settings.mosaic_max_side / max(width, height) * 0.98
        rows: list[list[_Tile]] = []
        for role, found in planes.items():
            frame = captures[role].frame
            shown = tiles.fit_size(frame.width, frame.height, side)
            viewed = replace(
                captures[role],
                view=FrameView((0.0, 0.0, float(frame.width), float(frame.height)), shown),
            )
            frame_id = scene._memory.remember_frame(viewed, scene._basis, keep=int(settings.max_frames))
            rows.append([
                _Tile(role, plane, frame_id, viewed, size, encoded=readings.get(role))
                for plane, size in zip(found, sizes[role])
            ])
        return rows

    def _side(self, detail: str, count: int) -> float:
        """The longest tile side: ``auto`` trades resolution for tiles within one budget."""
        settings = self._scene._settings
        if detail == "full":
            return float(MAX_TILE_SIDE)
        if detail == "standard":
            return settings.snapshot_max_side
        if detail == "overview" or count > 2:
            return settings.overview_max_side
        if count == 2:
            return min(settings.snapshot_max_side, (settings.mosaic_max_side - tiles.TILE_GAP) / 2.0)
        return settings.snapshot_max_side

    @staticmethod
    def _signatures(
        rows: Sequence[Sequence[_Tile]],
    ) -> dict[str, tuple[visual_change.FrameSignature, visual_change.FrameStats]]:
        return {
            row[0].camera: visual_change.observe(
                row[0].capture.frame, camera=row[0].camera, frame_id=row[0].frame_id,
                pose=row[0].capture.camera_to_base,
            )
            for row in rows
        }

    def _changes(
        self, signatures: Mapping[str, tuple[visual_change.FrameSignature, Any]]
    ) -> dict[str, visual_change.ChangeReport]:
        memory, settings = self._scene._memory, self._scene._settings
        changes = {}
        for role, (signature, _) in signatures.items():
            before = memory.signature(role)
            if before is not None:
                changes[role] = visual_change.compare(
                    before, signature, depth_mm=settings.change_depth_mm, luma=settings.change_luma
                )
            memory.remember_signature(role, signature)
        return changes

    def _fuse(self, rows: Sequence[Sequence[_Tile]], state: Any) -> scene_fusion.Fusion:
        scene = self._scene
        positions, _ = tiles.layout([[tile.size for tile in row] for row in rows])
        views = []
        for row, spots in zip(rows, positions):
            first = next((i for i, tile in enumerate(row) if tile.liftable_plane), None)
            views.append(scene_fusion.FusionView(
                row[0].frame_id, row[0].camera, row[0].capture, self.shown(row[0].capture),
                None if first is None else spots[first],
            ))
        return scene_fusion.fuse(
            views,
            scene._memory.objects(scene._basis, scene._settings.memory_ttl_s),
            tcp=None if state is None else state.tcp.position,
            settings=scene._settings,
        )

    # ── Rendering and result ──

    def _render(
        self, rows: Sequence[Sequence[_Tile]], marks: Mapping[str, Sequence[tiles.Mark]],
        overlay: bool,
    ) -> tuple[str, dict[str, tuple[float, float]], list[ViewTile]]:
        labelled = sum(len(row) for row in rows) > 1
        images, spans = [], {}
        for row in rows:
            line = []
            for tile in row:
                frame = tile.capture.frame
                label = f"{tile.frame_id} {tile.camera}"
                if tile.plane == tiles.PLANE_COLOR:
                    image = tiles.color_image(frame, tile.size, tile.window)
                elif tile.plane == tiles.PLANE_DEPTH:
                    span = tiles.depth_span(frame, tile.window)
                    image = tiles.depth_image(frame, tile.size, span, tile.window)
                    if span is not None:
                        spans[tile.frame_id] = span
                    label += f" depth {span[0]:.0f}-{span[1]:.0f}mm red=near" if span else " depth: none"
                else:
                    image = tiles.decoded_image(tile.encoded.data, tile.size)
                    label = f"{tile.camera} infrared (look only)"
                if overlay and tile.liftable_plane:
                    tiles.draw_marks(image, marks.get(tile.frame_id, ()),
                                     (frame.width, frame.height), tile.window)
                if labelled:
                    tiles.draw_label(image, label)
                line.append(image)
            images.append(line)
        positions, _ = tiles.layout([[image.size for image in line] for line in images])
        placed = [
            ViewTile(tile.frame_id if tile.liftable_plane else "", tile.camera, tile.plane,
                     x, y, tile.size[0], tile.size[1])
            for row, spots in zip(rows, positions) for tile, (x, y) in zip(row, spots)
        ]
        data = tiles.encode_jpeg(tiles.compose(images), int(self._scene._settings.snapshot_quality))
        return data, spans, placed

    def _result(self, primary: _Tile, placed: Sequence[ViewTile], data: str) -> dict[str, Any]:
        """The fields every shared image carries; a mosaic is named ``vN``, one tile ``fN``."""
        scene, settings = self._scene, self._scene._settings
        capture = primary.capture
        width = max(view.x + view.width for view in placed)
        height = max(view.y + view.height for view in placed)
        handle = primary.frame_id
        image: dict[str, Any] = {"mime": "image/jpeg", "data": data}
        liftable = capture.liftable
        if len(placed) > 1:
            handle = scene._memory.remember_view(placed, scene._basis, keep=int(settings.max_frames))
            liftable = any(
                view.frame_id and getattr(
                    scene._memory.recall_frame(view.frame_id, scene._basis, settings.frame_ttl_s),
                    "liftable", False,
                )
                for view in placed
            )
            image["label"] = f"view {handle}, {width}x{height}: " + " | ".join(
                f"{view.camera} {view.plane}" + (f" {view.frame_id}" if view.frame_id else "")
                for view in placed
            )
        shot: dict[str, Any] = {
            "frame": handle,
            "capture_id": capture.capture_id,
            "camera": primary.camera,
            "liftable": bool(liftable),
            "width": width,
            "height": height,
            "observed_at": capture.frame.observed_at,
            "liftable_for_s": settings.frame_ttl_s,
            "image": image,
        }
        if not capture.liftable:
            # Said at the snapshot, not first at a refused lift: which camera to use instead.
            shot["unliftable"] = scene._unplaced(primary.frame_id, capture)
        return shot

    def _frames_of(
        self, rows: Sequence[Sequence[_Tile]], spans: Mapping[str, tuple[float, float]]
    ) -> dict[str, Any]:
        frames = {}
        for row in rows:
            tile, capture = row[0], row[0].capture
            entry: dict[str, Any] = {
                "frame": tile.frame_id,
                "capture_id": capture.capture_id,
                "liftable": capture.liftable,
                "shown": list(self.shown(capture).size),
                "native": [capture.frame.width, capture.frame.height],
                "observed_at": capture.frame.observed_at,
                "planes": [t.plane for t in row],
            }
            if tile.frame_id in spans:
                entry["depth_range_mm"] = [round(value) for value in spans[tile.frame_id]]
            if tile.encoded is not None:
                skew = abs(float(tile.encoded.monotonic_at) - capture.monotonic_at) * 1000.0
                entry["infrared_skew_ms"] = round(skew, 1)
            if not capture.liftable:
                entry["unliftable"] = self._scene._unplaced(tile.frame_id, capture)
            frames[tile.camera] = entry
        return frames

    def _arm(self, state: Any) -> dict[str, Any]:
        held = self._scene.held
        return {
            "tcp_mm": [round(value, 1) for value in state.tcp.position],
            "gripper_mm": None if state.gripper_mm is None else round(state.gripper_mm, 1),
            "held": None if held is None else {"id": held.obj.object_id, "label": held.obj.label},
        }

    async def _arm_state(self) -> Any:
        """The arm's state for overlays, or ``None`` without one; never fails a snapshot."""
        motion = self._scene._motion
        if motion is None:
            return None
        try:
            return await asyncio.wait_for(motion.state(), _READ_TIMEOUT_S)
        except (SkillError, TransportError, TimeoutError) as exc:
            logger.debug("snapshot without arm state: %s", exc)
            return None


def _size(capture: SceneCapture, reading: Any, plane: str, side: float) -> tuple[int, int]:
    if plane == tiles.PLANE_INFRARED:
        # A reading that does not state its size is shown at the colour frame's.
        width = int(reading.width) or capture.frame.width
        height = int(reading.height) or capture.frame.height
        return tiles.fit_size(width, height, side)
    return tiles.fit_size(capture.frame.width, capture.frame.height, side)


def _into_tile(
    target: PixelTarget, views: Sequence[ViewTile], coords: str, size: tuple[int, int]
) -> tuple[PixelTarget | None, ViewTile | None, str]:
    """*target* (mosaic coordinates) in the tile it lands in, in that tile's pixels."""
    sx = sy = 1.0
    if coords == COORDS_UNIT:
        sx, sy = float(size[0]), float(size[1])
    elif coords == COORDS_PER_MILLE:
        sx, sy = size[0] / 1000.0, size[1] / 1000.0
    if target.point is not None:
        u, v = target.point[0] * sx, target.point[1] * sy
        tile = next((view for view in views if view.contains(u, v)), None)
        if tile is None:
            return None, None, "the point falls between tiles"
        local = PixelTarget(target.label, point=(u - tile.x, v - tile.y))
    else:
        box = target.box or (0.0, 0.0, 0.0, 0.0)
        x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        tile = next((view for view in views if view.contains((x0 + x1) / 2, (y0 + y1) / 2)), None)
        if tile is None:
            return None, None, "the box centre falls between tiles"
        cx0, cy0 = max(x0, tile.x), max(y0, tile.y)
        cx1, cy1 = min(x1, tile.x + tile.width), min(y1, tile.y + tile.height)
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (x1 - x0) * (y1 - y0):
            return None, None, (
                f"the box spills out of the {tile.camera} {tile.plane} tile; keep it inside one tile"
            )
        local = PixelTarget(target.label, box=(cx0 - tile.x, cy0 - tile.y, cx1 - tile.x, cy1 - tile.y))
    if not tile.frame_id:
        return None, None, (
            f"the {tile.camera} {tile.plane} tile is for looking only (not aligned with depth); "
            "point into its color or depth tile"
        )
    return local, tile, ""


def _infrared_channel(context: Any) -> str | None:
    """The first declared infrared frame channel of a device, if any."""
    for channel in getattr(context, "channels", ()) or ():
        if (getattr(channel, "quantity", "") == INFRARED_QUANTITY
                and getattr(channel, "representation", "") == "frame"):
            return str(channel.channel_id)
    return None


def _reason(exc: BaseException) -> str:
    return str(getattr(exc, "message", "") or exc or type(exc).__name__)


__all__ = ["DETAILS", "WITH_ALL", "SceneViewer", "ViewRequest", "target_detection"]

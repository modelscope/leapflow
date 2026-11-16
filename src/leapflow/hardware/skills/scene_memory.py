# Copyright (c) Alibaba, Inc. and its affiliates.
"""What one workcell's perception has seen, kept across runs and turns.

An object located in one turn is referred to by id in the next, and so is a
grasp proposed for it, or a frame shared with S2 that it later points into.
Ids are never reused. Everything is forgotten when it ages out or any
calibration moves (the *basis*: the frame versions it was measured against)
-- except each camera's last :class:`~.visual_change.FrameSignature`, which
is pixel-space and is what the next snapshot's change report compares against.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Plane, Transform
from leapflow.hardware.skills.protocol import GraspCandidate, HeldObject, LocatedObject


@dataclass(frozen=True)
class FrameView:
    """How a remembered frame was shown: the full-frame pixel *window*
    ``(x0, y0, x1, y1)`` rendered at *size* ``(width, height)``.

    Coordinates S2 answers with are relative to what it saw, so ``lift`` maps
    them back through this -- a zoomed crop and a downscaled overview resolve
    to the same full-frame pixels.
    """

    window: tuple[float, float, float, float]
    size: tuple[int, int]

    def to_frame(self, u: float, v: float) -> tuple[float, float]:
        """Full-frame pixel under shown pixel ``(u, v)``."""
        x0, y0, x1, y1 = self.window
        return x0 + u * (x1 - x0) / self.size[0], y0 + v * (y1 - y0) / self.size[1]


@dataclass(frozen=True)
class ViewTile:
    """One tile of a shared mosaic: where it sits and what it shows.

    *frame_id* names the remembered frame a point in the tile lifts in; it is
    empty for a tile that is only for looking (an infrared plane, which is not
    pixel-aligned with the depth that ``lift`` reads).
    """

    frame_id: str
    camera: str
    plane: str
    x: int
    y: int
    width: int
    height: int

    def contains(self, u: float, v: float) -> bool:
        return self.x <= u < self.x + self.width and self.y <= v < self.y + self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_id or None, "camera": self.camera, "plane": self.plane,
            "x": self.x, "y": self.y, "width": self.width, "height": self.height,
            "liftable_plane": bool(self.frame_id),
        }


@dataclass(frozen=True)
class SceneCapture:
    """One aligned RGB-D frame with the camera mount and table it was taken against.

    A frame shared only to be looked at may come from a camera with no calibrated
    mount, or before the table is known: it is not ``liftable`` then, and nothing
    in it can be placed in the arm base frame.
    """

    frame: RgbdFrame
    camera_to_base: Transform | None
    table: Plane | None
    capture_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    camera: str = "camera"
    view: FrameView | None = None

    @property
    def monotonic_at(self) -> float:
        """The frame's capture instant, so a ``CaptureGroup`` can measure skew."""
        return self.frame.monotonic_at

    @property
    def shown(self) -> FrameView:
        """How the frame was shown; the whole frame at native size when not recorded."""
        if self.view is not None:
            return self.view
        width, height = self.frame.width, self.frame.height
        return FrameView((0.0, 0.0, float(width), float(height)), (width, height))

    @property
    def liftable(self) -> bool:
        return self.camera_to_base is not None and self.table is not None

    def placement(self) -> tuple[Transform, Plane]:
        """The mount and the table; raises ``ValueError`` when the frame is not liftable."""
        if self.camera_to_base is None or self.table is None:
            missing = "camera mount" if self.camera_to_base is None else "table"
            raise ValueError(f"the {self.camera} frame was taken without a calibrated {missing}")
        return self.camera_to_base, self.table


class SceneMemory:
    """One workcell's located objects, proposed grasps, shared frames, and what is held."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._objects: dict[str, tuple[LocatedObject, float]] = {}
        self._grasps: dict[str, tuple[GraspCandidate, float]] = {}
        self._frames: dict[str, tuple[SceneCapture, float]] = {}
        self._views: dict[str, tuple[tuple[ViewTile, ...], float]] = {}
        self._signatures: dict[str, Any] = {}
        self._basis: dict[str, float] = {}
        self._counter = 0
        self._grasp_counter = 0
        self._frame_counter = 0
        self._view_counter = 0
        self._held: HeldObject | None = None

    def remember(
        self, objects: Sequence[LocatedObject], basis: Mapping[str, float]
    ) -> tuple[LocatedObject, ...]:
        """Replace the objects with *objects*, each under a new id (old grasps go too)."""
        self._rebase(basis)
        self._objects = {}
        self._grasps = {}
        return self.add(objects, basis)

    def add(
        self, objects: Sequence[LocatedObject], basis: Mapping[str, float]
    ) -> tuple[LocatedObject, ...]:
        """Remember *objects* beside the current ones, each under a new id."""
        self._rebase(basis)
        now = self._clock()
        named = []
        for obj in objects:
            self._counter += 1
            named.append(replace(obj, object_id=f"o{self._counter}"))
        self._objects.update((obj.object_id, (obj, now)) for obj in named)
        return tuple(named)

    def update(self, obj: LocatedObject, basis: Mapping[str, float]) -> None:
        """Record a fresh observation of an object already remembered."""
        if dict(basis) == self._basis and obj.object_id in self._objects:
            self._objects[obj.object_id] = (obj, self._clock())

    def recall(
        self, object_id: str, basis: Mapping[str, float], ttl_s: float
    ) -> LocatedObject | None:
        entry = self._objects.get(object_id)
        if entry is None or not self._valid(entry[1], basis, ttl_s):
            return None
        return entry[0]

    def objects(self, basis: Mapping[str, float], ttl_s: float) -> tuple[LocatedObject, ...]:
        """Every object still referable, oldest id first."""
        return tuple(
            obj for obj, seen_at in self._objects.values() if self._valid(seen_at, basis, ttl_s)
        )

    def remember_grasps(
        self, object_id: str, candidates: Sequence[GraspCandidate]
    ) -> tuple[GraspCandidate, ...]:
        """Name *candidates* ``gN`` for *object_id*, replacing that object's earlier ones."""
        now = self._clock()
        self._grasps = {k: v for k, v in self._grasps.items() if v[0].object_id != object_id}
        named = []
        for candidate in candidates:
            self._grasp_counter += 1
            grasp = replace(candidate, grasp_id=f"g{self._grasp_counter}", object_id=object_id)
            self._grasps[grasp.grasp_id] = (grasp, now)
            named.append(grasp)
        return tuple(named)

    def recall_grasp(
        self, grasp_id: str, basis: Mapping[str, float], ttl_s: float
    ) -> GraspCandidate | None:
        entry = self._grasps.get(grasp_id)
        if entry is None or not self._valid(entry[1], basis, ttl_s):
            return None
        return entry[0]

    def remember_frame(
        self,
        capture: SceneCapture,
        basis: Mapping[str, float],
        *,
        keep: int,
        seen_at: float | None = None,
    ) -> str:
        """Name *capture* ``fN``; only the newest *keep* frames stay recallable.

        *seen_at* keeps the age of a frame re-shown from an earlier capture (a
        zoom), so cropping never makes an old observation look fresh.
        """
        self._rebase(basis)
        self._frame_counter += 1
        frame_id = f"f{self._frame_counter}"
        self._frames[frame_id] = (capture, self._clock() if seen_at is None else seen_at)
        while len(self._frames) > max(1, keep):
            self._frames.pop(next(iter(self._frames)))
        return frame_id

    def recall_frame(
        self, frame_id: str, basis: Mapping[str, float], ttl_s: float
    ) -> SceneCapture | None:
        entry = self._frames.get(frame_id)
        if entry is None or not self._valid(entry[1], basis, ttl_s):
            return None
        return entry[0]

    def frame_seen_at(self, frame_id: str) -> float | None:
        entry = self._frames.get(frame_id)
        return None if entry is None else entry[1]

    def remember_view(
        self, tiles: Sequence[ViewTile], basis: Mapping[str, float], *, keep: int
    ) -> str:
        """Name a shared mosaic ``vN``; its tiles resolve through their frames."""
        self._rebase(basis)
        self._view_counter += 1
        view_id = f"v{self._view_counter}"
        self._views[view_id] = (tuple(tiles), self._clock())
        while len(self._views) > max(1, keep):
            self._views.pop(next(iter(self._views)))
        return view_id

    def recall_view(
        self, view_id: str, basis: Mapping[str, float], ttl_s: float
    ) -> tuple[ViewTile, ...] | None:
        entry = self._views.get(view_id)
        if entry is None or not self._valid(entry[1], basis, ttl_s):
            return None
        return entry[0]

    def signature(self, camera: str) -> Any:
        """The last frame signature of *camera*, or ``None``."""
        return self._signatures.get(camera)

    def remember_signature(self, camera: str, signature: Any) -> None:
        self._signatures[camera] = signature

    def _rebase(self, basis: Mapping[str, float]) -> None:
        """Forget everything measured against another calibration."""
        if dict(basis) != self._basis:
            self._objects, self._grasps, self._frames, self._views = {}, {}, {}, {}
            self._basis = dict(basis)

    def _valid(self, seen_at: float, basis: Mapping[str, float], ttl_s: float) -> bool:
        return dict(basis) == self._basis and self._clock() - seen_at <= ttl_s

    @property
    def held(self) -> HeldObject | None:
        return self._held

    def hold(self, held: HeldObject) -> None:
        self._held = held
        gone = held.obj.object_id
        self._objects.pop(gone, None)  # it is no longer on the table
        self._grasps = {k: v for k, v in self._grasps.items() if v[0].object_id != gone}

    def release(self) -> HeldObject | None:
        held, self._held = self._held, None
        return held


__all__ = ["FrameView", "SceneCapture", "SceneMemory", "ViewTile"]

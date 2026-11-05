# Copyright (c) Alibaba, Inc. and its affiliates.
"""What one workcell's perception has seen, kept across runs and turns.

An object located in one turn is referred to by id in the next, and so is a
grasp proposed for it, or a frame shared with S2 that it later points into.
Ids are never reused. Everything is forgotten when it ages out or any
calibration moves (the *basis*: the frame versions it was measured against).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Plane, Transform
from leapflow.hardware.skills.protocol import GraspCandidate, HeldObject, LocatedObject


@dataclass(frozen=True)
class SceneCapture:
    """One aligned RGB-D frame with the camera mount and table it was taken against."""

    frame: RgbdFrame
    camera_to_base: Transform
    table: Plane
    capture_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class SceneMemory:
    """One workcell's located objects, proposed grasps, shared frames, and what is held."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._objects: dict[str, tuple[LocatedObject, float]] = {}
        self._grasps: dict[str, tuple[GraspCandidate, float]] = {}
        self._frames: dict[str, tuple[SceneCapture, float]] = {}
        self._basis: dict[str, float] = {}
        self._counter = 0
        self._grasp_counter = 0
        self._frame_counter = 0
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
        self, capture: SceneCapture, basis: Mapping[str, float], *, keep: int
    ) -> str:
        """Name *capture* ``fN``; only the newest *keep* frames stay recallable."""
        self._rebase(basis)
        self._frame_counter += 1
        frame_id = f"f{self._frame_counter}"
        self._frames[frame_id] = (capture, self._clock())
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

    def _rebase(self, basis: Mapping[str, float]) -> None:
        """Forget everything measured against another calibration."""
        if dict(basis) != self._basis:
            self._objects, self._grasps, self._frames = {}, {}, {}
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


__all__ = ["SceneCapture", "SceneMemory"]

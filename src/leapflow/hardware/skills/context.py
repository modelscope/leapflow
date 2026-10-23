# Copyright (c) Alibaba, Inc. and its affiliates.
"""The concrete ``SkillContext``: the only surface a node skill can reach.

``SkillRuntime`` builds one ``RunContext`` per plan or run. Motion is built
eagerly (every skill reads the arm); perception, the grasp planner, target
vision, and policy streaming are built on first use, so a skill that never
looks at the camera never fails on a camera problem, and the failure a skill
that does look sees is a typed ``SkillError`` naming what the workcell lacks.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Mapping, Sequence

from leapflow.hardware.model_contract import ModelService
from leapflow.hardware.skills.frames import FrameGraph
from leapflow.hardware.skills.motion import GovernedMotion
from leapflow.hardware.skills.perception import PerceptionSettings, WorkcellScene
from leapflow.hardware.skills.ports import GraspPlanner
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    FAILURE_SKILL_UNAVAILABLE,
    SkillError,
)
from leapflow.hardware.skills.policy_binding import PolicySettings
from leapflow.hardware.skills.policy_stream import WorkcellPolicies
from leapflow.hardware.skills.scene_memory import SceneCapture, SceneMemory
from leapflow.hardware.skills.vision import WorkcellVision
from leapflow.hardware.skills.workcell import WorkcellSpec


class _SceneSensing:
    """``GraspSensing`` over a context's scene, built only when a planner looks."""

    def __init__(self, scene: Callable[[], WorkcellScene]) -> None:
        self._scene = scene

    async def capture(self) -> SceneCapture:
        return await self._scene().capture()

    async def model(self, name: str, task: str) -> ModelService:
        return await self._scene().model(name, task)


class RunContext:
    """One plan's or run's view of its workcell. Satisfies ``SkillContext``."""

    def __init__(
        self,
        *,
        registry: Any,
        spec: WorkcellSpec,
        frames: FrameGraph,
        motion: GovernedMotion,
        limits: Mapping[str, float],
        cancel: asyncio.Event,
        memory: SceneMemory,
        planner: Callable[[Any], GraspPlanner],
        recording: Callable[[], bool] = lambda: False,
    ) -> None:
        self._registry = registry
        self._spec = spec
        self._frames = frames
        self._motion = motion
        self._limits = dict(limits)
        self._cancel = cancel
        self._memory = memory
        self._planner = planner
        self._recording = recording
        self._grasp_planner: GraspPlanner | None = None
        self._scene: WorkcellScene | None = None
        self._visions: dict[str, WorkcellVision] = {}
        self._policies: WorkcellPolicies | None = None
        self._grasp_verification: Any = None
        self._committed = False
        self.evidence: dict[str, Any] = {}

    @property
    def workcell(self) -> WorkcellSpec:
        return self._spec

    @property
    def frames(self) -> FrameGraph:
        return self._frames

    @property
    def motion(self) -> GovernedMotion:
        return self._motion

    @property
    def limits(self) -> Mapping[str, float]:
        return dict(self._limits)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def recording(self) -> bool:
        return self._recording()

    @property
    def committed(self) -> bool:
        """Whether the run persisted a lasting effect that is not a motion (a calibration)."""
        return self._committed

    @property
    def scene(self) -> WorkcellScene:
        if self._scene is None:
            try:
                settings = PerceptionSettings.from_mapping(self._spec.perception)
            except ValueError as exc:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"perception: {exc}") from exc
            self._scene = WorkcellScene(
                registry=self._registry,
                spec=self._spec,
                frames=self._frames,
                motion=self._motion,
                memory=self._memory,
                settings=settings,
                note=self.note,
            )
        return self._scene

    @property
    def grasp_planner(self) -> GraspPlanner:
        if self._grasp_planner is None:
            self._grasp_planner = self._planner(_SceneSensing(lambda: self.scene))
        return self._grasp_planner

    @property
    def vision(self) -> WorkcellVision:
        return self.camera_vision()

    def camera_vision(self, role: str = "camera") -> WorkcellVision:
        """Lazily resolve the exact camera role without borrowing another mount."""
        if role not in self._visions:
            self._visions[role] = WorkcellVision.for_workcell(self._registry, self._spec, role=role)
        return self._visions[role]

    @property
    def grasp_verification(self) -> Any:
        if self._grasp_verification is None:
            from leapflow.hardware.skills.grasp_verification import WorkcellGraspVerification
            self._grasp_verification = WorkcellGraspVerification(self, self._registry)
        return self._grasp_verification

    @property
    def policies(self) -> WorkcellPolicies:
        if self._policies is None:
            try:
                settings = PolicySettings.from_mapping(self._spec.policy)
            except ValueError as exc:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"policy: {exc}") from exc
            self._policies = WorkcellPolicies(
                registry=self._registry,
                spec=self._spec,
                frames=self._frames,
                motion=self._motion,
                settings=settings,
                note=self.note,
            )
        return self._policies

    async def read(self, role: str, channel_ids: Sequence[str]) -> Mapping[str, Any]:
        device_id = self._spec.role(role)
        if not device_id:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"this workcell binds no {role}")
        batch = await self._registry.read_batch(device_id, tuple(channel_ids))
        return {reading.channel_id: reading.value for reading in batch.readings}

    def record_calibration(
        self,
        procedure_id: str,
        *,
        role: str = "",
        parameters: Mapping[str, Any] | None = None,
        matrix: Sequence[Sequence[float]] | None = None,
        pose: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> float:
        from leapflow.hardware.calibration_store import CalibrationRecord

        store = getattr(self._registry, "calibration_store", None)
        if store is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "this node has no calibration store (set the instrument database path)",
            )
        device_id = self._spec.role(role) if role else self._spec.workcell_id
        if not device_id:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"this workcell binds no {role}")
        recorded_at = time.time()
        record = CalibrationRecord(
            device_id=device_id,
            procedure_id=procedure_id,
            recorded_at=recorded_at,
            parameters=dict(parameters or {}),
            matrix=None if matrix is None else [list(map(float, row)) for row in matrix],
            pose=None if pose is None else dict(pose),
            notes=notes,
        )
        if not store.record(record):
            raise SkillError(
                FAILURE_DEVICE_FAULT, f"calibration {procedure_id!r} was not persisted"
            )
        self._committed = True
        self.evidence.setdefault("calibrations", {})[procedure_id] = recorded_at
        return recorded_at

    def note(self, key: str, value: Any) -> None:
        self.evidence[str(key)] = value


__all__ = ["RunContext"]

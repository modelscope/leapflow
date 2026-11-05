# Copyright (c) Alibaba, Inc. and its affiliates.
"""The learned grasp planner: a served ``grasp_proposal`` model proposes, the node maps.

Each proposal captures one fresh aligned RGB-D frame, sends it with the target
object's camera-frame centre (and the pixel it projects to) to the workcell's
``grasp.model`` binding, and maps every returned camera-frame grasp into the
arm base frame through the same hand-eye mount perception lifts with. The node
only filters what this gripper physically cannot do (an opening wider than its
stroke, a score below the declared floor) and fills the approach and retreat
the model did not state; ranking stays the model's.

Workcell configuration (``grasp:``)::

    grasp: {planner: model, model: grasp, top_k: 5, min_score: 0.1,
            approach_mm: 80, retreat_mm: 80, open_margin_mm: 15, jpeg_quality: 90}
"""

from __future__ import annotations

import asyncio
import base64
import sys
from array import array
from dataclasses import dataclass
from typing import Any, Mapping

from leapflow.hardware.model_contract import TASK_GRASP_PROPOSAL, propose_grasps
from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Plane, Pose, Transform
from leapflow.hardware.skills.grasp import GraspSensing, max_width, numeric_settings
from leapflow.hardware.skills.ports import GraspPlanner
from leapflow.hardware.skills.protocol import (
    FAILURE_MODEL_UNAVAILABLE,
    GraspCandidate,
    LocatedObject,
    SkillError,
)
from leapflow.hardware.transport import TransportError

MODEL_PLANNER = "model"
DEFAULT_GRASP_MODEL = "grasp"


@dataclass(frozen=True)
class ModelGraspSettings:
    """Numeric tuning of the model planner (mm; ``min_score`` in [0, 1])."""

    top_k: float = 5.0
    min_score: float = 0.0
    approach_mm: float = 80.0
    retreat_mm: float = 80.0
    open_margin_mm: float = 15.0
    jpeg_quality: float = 90.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelGraspSettings":
        numeric = {k: v for k, v in config.items() if k != "model"}
        settings: ModelGraspSettings = numeric_settings(cls, numeric, MODEL_PLANNER)
        if settings.top_k < 1 or settings.top_k != int(settings.top_k):
            raise ValueError("grasp.top_k must be a positive integer")
        if not 0.0 <= settings.min_score <= 1.0:
            raise ValueError("grasp.min_score must be in [0, 1]")
        for name in ("approach_mm", "retreat_mm", "open_margin_mm"):
            if not getattr(settings, name) > 0:
                raise ValueError(f"grasp.{name} must be positive")
        if not 0 < settings.jpeg_quality <= 100:
            raise ValueError("grasp.jpeg_quality must be in (0, 100]")
        return settings


class ModelGraspPlanner:
    """Grasps proposed by a served model. Satisfies ``GraspPlanner``."""

    def __init__(
        self,
        gripper_spec: Mapping[str, float],
        model: str,
        settings: ModelGraspSettings,
        sensing: GraspSensing,
    ) -> None:
        self._max_width = max_width(gripper_spec)
        self._model = model
        self._settings = settings
        self._sensing = sensing

    @property
    def name(self) -> str:
        return MODEL_PLANNER

    async def candidates(
        self, obj: LocatedObject, *, table: Plane, current: Pose
    ) -> tuple[GraspCandidate, ...]:
        s = self._settings
        capture = await self._sensing.capture()
        service = await self._sensing.model(self._model, TASK_GRASP_PROPOSAL)
        request = await self._request(obj, capture.frame, capture.camera_to_base)
        try:
            result = await propose_grasps(service, request)
        except (TransportError, TypeError) as exc:
            raise SkillError(
                FAILURE_MODEL_UNAVAILABLE, f"grasp model {self._model!r} failed: {exc}"
            ) from exc
        planner = f"{MODEL_PLANNER}:{result.model.name or service.model_info.name}"
        found: list[GraspCandidate] = []
        for proposal in result.grasps:
            if proposal.score < s.min_score or proposal.width_mm >= self._max_width:
                continue
            in_camera = Transform.from_pose(Pose(*proposal.pose))
            found.append(
                GraspCandidate(
                    grasp=capture.camera_to_base.compose(in_camera).to_pose(),
                    width_mm=proposal.width_mm,
                    open_mm=min(self._max_width, proposal.width_mm + s.open_margin_mm),
                    approach_mm=proposal.approach_mm or s.approach_mm,
                    retreat_mm=s.retreat_mm,
                    score=proposal.score,
                    planner=planner,
                )
            )
        found.sort(key=lambda candidate: -candidate.score)
        return tuple(found[: int(s.top_k)])

    async def _request(
        self, obj: LocatedObject, frame: RgbdFrame, camera_to_base: Transform
    ) -> dict[str, Any]:
        center = camera_to_base.inverse().apply(obj.center)
        u, v = frame.intrinsics.project(center)
        image = await asyncio.to_thread(frame.color_jpeg, quality=int(self._settings.jpeg_quality))
        return {
            "image": base64.b64encode(image).decode("ascii"),
            "depth": {"data": _depth_le(frame), "scale_m": frame.depth_scale_m},
            "intrinsics": frame.intrinsics.to_dict(),
            "object": {
                "label": obj.label,
                "center": [round(c, 2) for c in center],
                "pixel": [round(u, 1), round(v, 1)],
                "size": {"length": obj.length, "width": obj.width, "height": obj.height},
            },
            "gripper": {"max_width_mm": self._max_width},
            "top_k": int(self._settings.top_k),
        }


def _depth_le(frame: RgbdFrame) -> str:
    """The depth image as base64 little-endian uint16, whatever the host byte order."""
    data = frame.depth
    if sys.byteorder != "little":
        units = array("H")
        units.frombytes(data)
        units.byteswap()
        data = units.tobytes()
    return base64.b64encode(data).decode("ascii")


def model_planner(
    gripper_spec: Mapping[str, float], config: Mapping[str, Any], sensing: GraspSensing
) -> GraspPlanner:
    """Factory for ``grasp.planner: model``; ``grasp.model`` names the workcell binding."""
    model = config.get("model", DEFAULT_GRASP_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("grasp.model must name a workcell model binding")
    settings = ModelGraspSettings.from_mapping({k: v for k, v in config.items() if k != "planner"})
    return ModelGraspPlanner(gripper_spec, model.strip(), settings, sensing)


__all__ = [
    "DEFAULT_GRASP_MODEL",
    "MODEL_PLANNER",
    "ModelGraspPlanner",
    "ModelGraspSettings",
    "model_planner",
]

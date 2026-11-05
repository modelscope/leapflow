# Copyright (c) Alibaba, Inc. and its affiliates.
"""Grasp planners: from a located object to the TCP poses a gripper closes at.

A planner is a strategy plugin (``GraspPlanner``), chosen per workcell by
``grasp.planner`` and configured by the rest of that mapping. Packages advertise
planners under the ``leapflow.hardware.grasp_planners`` entry-point group; each
entry point is a factory ``(gripper_spec, config, sensing) -> GraspPlanner``
that raises ``ValueError`` on a configuration it cannot honour, so the runtime
can refuse the workcell at admission rather than at the first proposal.
``sensing`` (``GraspSensing``) is how a learned planner looks at the scene and
reaches its model; at admission it refuses every call, so a factory must only
store it.

``top_down`` is the geometric baseline for a parallel gripper over a table: the
tool points straight down, the fingers close across the object's narrow side,
and the fingertips stop part way down the object, never below the table. The
learned path is ``model`` (``grasp_model``): a served ``grasp_proposal`` model.
Neither picks for the caller -- every candidate comes back with its score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.model_contract import GEOMETRY_POINT, ModelService
from leapflow.hardware.skills.geometry import Plane, Pose, angle_delta_deg, wrap_axis_deg, wrap_deg
from leapflow.hardware.skills.scene_memory import SceneCapture
from leapflow.hardware.skills.ports import GraspPlanner
from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS,
    FAILURE_SKILL_UNAVAILABLE,
    GraspCandidate,
    LocatedObject,
    SkillError,
)

logger = logging.getLogger(__name__)

GRASP_PLANNER_ENTRY_POINT_GROUP = "leapflow.hardware.grasp_planners"
TOP_DOWN = "top_down"


@runtime_checkable
class GraspSensing(Protocol):
    """What a planner may consult per call: a fresh RGB-D view and a bound model."""

    async def capture(self) -> SceneCapture: ...

    async def model(self, name: str, task: str) -> ModelService:
        """The workcell's *name* model, verified to serve *task*; raises ``SkillError``."""
        ...


class AdmissionSensing:
    """The sensing a planner is built with at admission: nothing can be observed."""

    async def capture(self) -> SceneCapture:
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "no scene is observable at admission")

    async def model(self, name: str, task: str) -> ModelService:
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "no model is reachable at admission")


PlannerFactory = Callable[[Mapping[str, float], Mapping[str, Any], GraspSensing], GraspPlanner]


def numeric_settings(cls: type, config: Mapping[str, Any], planner: str) -> Any:
    """A frozen settings dataclass of floats from *config* (``planner`` key ignored)."""
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(config) - known - {"planner"})
    if unknown:
        raise ValueError(f"{planner} does not understand {', '.join(unknown)}")
    values: dict[str, float] = {}
    for name in known & set(config):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"grasp.{name} must be a number")
        values[name] = float(value)
    return replace(cls(), **values)


def max_width(gripper_spec: Mapping[str, float]) -> float:
    """The resolved gripper opening; admission always fills it."""
    width = gripper_spec.get("max_width_mm")
    if width is None or not float(width) > 0:
        raise ValueError("gripper_spec.max_width_mm must be resolved before planning grasps")
    return float(width)


@dataclass(frozen=True)
class TopDownSettings:
    """Tuning of the top-down planner (millimetres, degrees).

    ``closing_axis_deg`` is the tool yaw that turns the finger-closing axis onto
    the footprint's short side. The jaws are symmetric, so only the axis line
    matters and 0 and 180 are the same grasp. Which tool axis the jaws lie on
    belongs to the mounted gripper, not to the planner: the default of 90 fits
    jaws along tool +Y, while a gripper whose jaw axis is tool X -- measured on
    the Piper cell by ``calib.tcp`` and declared in that cell's workcell --
    needs 0. Getting it wrong makes the jaws span the object's long side and
    close on air. An opening narrower than the object plus ``min_slack_mm``
    cannot pass around it; footprints within ``round_tolerance_mm`` of square
    get one candidate.
    """

    closing_axis_deg: float = 90.0
    approach_mm: float = 80.0
    retreat_mm: float = 80.0
    depth_cap_mm: float = 25.0
    clearance_mm: float = 5.0
    open_margin_mm: float = 15.0
    min_slack_mm: float = 4.0
    round_tolerance_mm: float = 5.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TopDownSettings":
        settings: TopDownSettings = numeric_settings(cls, config, TOP_DOWN)
        for name in ("approach_mm", "retreat_mm", "depth_cap_mm", "open_margin_mm"):
            if not getattr(settings, name) > 0:
                raise ValueError(f"grasp.{name} must be positive")
        for name in ("clearance_mm", "min_slack_mm", "round_tolerance_mm"):
            if getattr(settings, name) < 0:
                raise ValueError(f"grasp.{name} must not be negative")
        return settings


class TopDownPlanner:
    """Vertical approach, fingers across the footprint. Satisfies ``GraspPlanner``."""

    def __init__(self, gripper_spec: Mapping[str, float], settings: TopDownSettings) -> None:
        self._max_width = max_width(gripper_spec)
        self._settings = settings

    @property
    def name(self) -> str:
        return TOP_DOWN

    async def candidates(
        self, obj: LocatedObject, *, table: Plane, current: Pose
    ) -> tuple[GraspCandidate, ...]:
        """Across the narrow side (score 1.0); across the long side too when it fits (0.5)."""
        if obj.geometry == GEOMETRY_POINT:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"{obj.object_id} was lifted from a single point and has no footprint; "
                f"{TOP_DOWN} needs an object located by mask or box",
            )
        sides = [(wrap_axis_deg(obj.yaw_deg + 90.0), obj.width, 1.0)]
        if obj.length - obj.width > self._settings.round_tolerance_mm:
            sides.append((obj.yaw_deg, obj.length, 0.5))
        found: list[GraspCandidate] = []
        for across_yaw, across, score in sides:
            candidate = self._candidate(obj, table, current, across_yaw, across, score)
            if candidate is not None:
                found.append(candidate)
        return tuple(found)

    def _candidate(
        self,
        obj: LocatedObject,
        table: Plane,
        current: Pose,
        across_yaw: float,
        across: float,
        score: float,
    ) -> GraspCandidate | None:
        s = self._settings
        open_mm = min(self._max_width, across + s.open_margin_mm)
        if open_mm - across < s.min_slack_mm:
            return None
        x, y = obj.center[0], obj.center[1]
        floor = table.height(x, y) + s.clearance_mm
        z = max(floor, obj.top_z - min(s.depth_cap_mm, obj.height / 2.0))
        grasp = Pose(x, y, z, 180.0, 0.0, nearest_yaw(across_yaw + s.closing_axis_deg, current))
        return GraspCandidate(
            grasp=grasp,
            width_mm=across,
            open_mm=open_mm,
            approach_mm=s.approach_mm,
            retreat_mm=s.retreat_mm,
            score=score,
            planner=TOP_DOWN,
        )


def nearest_yaw(yaw: float, current: Pose) -> float:
    """A parallel gripper is symmetric: of ``yaw`` and ``yaw + 180`` take the closer."""
    options = (wrap_deg(yaw), wrap_deg(yaw + 180.0))
    return min(options, key=lambda rz: abs(angle_delta_deg(rz, current.rz)))


def top_down_planner(
    gripper_spec: Mapping[str, float], config: Mapping[str, Any], sensing: GraspSensing
) -> GraspPlanner:
    """The shipped baseline's factory (always installed, entry point or not)."""
    return TopDownPlanner(gripper_spec, TopDownSettings.from_mapping(config))


def discover_grasp_planners() -> dict[str, PlannerFactory]:
    """Planner factories advertised under ``leapflow.hardware.grasp_planners``.

    A factory that fails to load is skipped with a warning; ``top_down`` and
    ``model`` are always available, so a node without the entry points
    installed still plans.
    """
    from importlib.metadata import entry_points

    from leapflow.hardware.skills.grasp_model import MODEL_PLANNER, model_planner

    found: dict[str, PlannerFactory] = {TOP_DOWN: top_down_planner, MODEL_PLANNER: model_planner}
    for entry in entry_points(group=GRASP_PLANNER_ENTRY_POINT_GROUP):
        try:
            found[entry.name] = entry.load()
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the node
            logger.warning("grasp planner %r failed to load: %s", entry.name, exc)
    return found


__all__ = [
    "GRASP_PLANNER_ENTRY_POINT_GROUP",
    "TOP_DOWN",
    "AdmissionSensing",
    "GraspSensing",
    "PlannerFactory",
    "TopDownPlanner",
    "TopDownSettings",
    "discover_grasp_planners",
    "max_width",
    "nearest_yaw",
    "numeric_settings",
    "top_down_planner",
]

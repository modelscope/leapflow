# Copyright (c) Alibaba, Inc. and its affiliates.
"""What ``motion.sequence`` holds the world to: the hold, the referenced objects, a release.

A guard is only worth declaring when it can pass. A release judged against a
threshold the jaws cannot reach -- the held width plus a margin beyond the
gripper's widest opening, or inside the gripper's own position tolerance --
fails on every run, and a failed guard stops the program with the object in
the jaws. So the release rule is a single shared definition: the planner
refuses a release it can never confirm before approval, and the run judges the
jaws against the same threshold the approval showed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from leapflow.hardware.skills.geometry import wrap_axis_deg
from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    ArmState,
    HeldObject,
    LocatedObject,
    SkillError,
)


@dataclass(frozen=True)
class ReleaseRule:
    """When opening the jaws from a hold counts as letting go.

    ``margin_mm`` is how far beyond the held width the jaws must open to prove
    the object no longer constrains them (``gripper_spec.release_margin_mm``,
    else the workcell's ``freshness.hold_tolerance_mm``). ``tolerance_mm`` is
    the gripper's declared position tolerance: a width inside it of the hold
    cannot be told apart from still holding, and a width inside it of the
    commanded opening is the jaws having arrived.
    """

    margin_mm: float
    tolerance_mm: float

    @classmethod
    def of(cls, ctx: SkillContext) -> "ReleaseRule":
        spec = ctx.workcell.gripper_spec
        margin = spec.get("release_margin_mm", ctx.workcell.freshness["hold_tolerance_mm"])
        return cls(margin_mm=float(margin), tolerance_mm=float(ctx.motion.gripper_tolerance_mm))

    def threshold(self, held_mm: float, target_mm: float) -> float | None:
        """The width an opening toward *target_mm* must reach to release a *held_mm* hold.

        The held width plus the margin, capped where the jaws can actually get:
        the commanded opening less the position tolerance -- jaws that arrived at
        an opening wider than the hold are no longer pressed by it. ``None`` when
        that cap cannot clear the hold by more than the tolerance, because then
        no width the jaws report proves a release.
        """
        threshold = min(held_mm + self.margin_mm, target_mm - self.tolerance_mm)
        return threshold if threshold > held_mm + self.tolerance_mm else None

    def minimum_opening(self, held_mm: float) -> float:
        """The narrowest commanded opening whose release of *held_mm* can be confirmed."""
        return held_mm + 2.0 * self.tolerance_mm

    def check(self, index: int, held_mm: float, target_mm: float, widest_mm: float) -> None:
        """Refuse, before approval, a ``released`` guard no width can satisfy."""
        if self.threshold(held_mm, target_mm) is not None:
            return
        needed = self.minimum_opening(held_mm)
        remedy = (
            f"command an opening wider than {needed:.1f} mm"
            if needed < widest_mm
            else f"this gripper opens at most {widest_mm:.1f} mm, so drop the released guard "
            "and confirm the release by observation"
        )
        raise SkillError(
            FAILURE_INVALID_ARGS,
            f"step {index}: opening to {target_mm:.1f} mm cannot confirm a release of the "
            f"{held_mm:.1f} mm hold (the gripper's position tolerance is "
            f"{self.tolerance_mm:.1f} mm); {remedy}",
            result={"release_needs_mm": round(needed, 1), "held_mm": round(held_mm, 1)},
        )


def current_hold(ctx: SkillContext, state: ArmState) -> HeldObject | None:
    """The held object, forgetting it when the jaws no longer report its width."""
    held = ctx.scene.held
    if held is None or state.gripper_mm is None:
        return held
    if abs(state.gripper_mm - held.width_mm) <= ctx.workcell.freshness["hold_tolerance_mm"]:
        return held
    ctx.scene.release()
    ctx.note(
        "hold_forgotten",
        {
            "object_id": held.obj.object_id,
            "held_mm": round(held.width_mm, 1),
            "gripper_mm": round(state.gripper_mm, 1),
        },
    )
    return None


def check_drift(fresh: Mapping[str, float], planned: LocatedObject, seen: LocatedObject) -> None:
    """Refuse a run whose referenced object moved beyond the declared freshness."""
    planar = math.hypot(seen.center[0] - planned.center[0], seen.center[1] - planned.center[1])
    height = abs(seen.top_z - planned.top_z)
    problems = []
    if planar > fresh["drift_planar_mm"]:
        problems.append(f"moved {planar:.0f} mm")
    if height > fresh["drift_height_mm"]:
        problems.append(f"its top changed {height:.0f} mm")
    if min(planned.length - planned.width, seen.length - seen.width) > fresh["yaw_defined_mm"]:
        turned = abs(wrap_axis_deg(seen.yaw_deg - planned.yaw_deg))
        if turned > fresh["drift_yaw_deg"]:
            problems.append(f"turned {turned:.0f} deg")
    if problems:
        raise SkillError(
            FAILURE_PLAN_STALE,
            f"{planned.object_id} ({planned.label}) {', '.join(problems)} since planning; "
            "locate and plan again",
        )


__all__ = ["ReleaseRule", "check_drift", "current_hold"]

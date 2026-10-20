# Copyright (c) Alibaba, Inc. and its affiliates.
"""Scene skills: locate objects, share a frame, lift S2's pointing, propose grasps --
read-only, everything by id.

Perception reports every candidate it saw with its score and a fresh id; the
grasp planner reports every grasp it found with its score and a fresh id. The
skills never match labels, never break ties, and never pick on the model's
behalf: S2 reads the candidates and names the ones it wants in a
``motion.sequence``. Where the locator falls short, S2 can look for itself: a
workcell that shares frames (and a host that consents) answers
``perceive.snapshot`` with an image, and ``perceive.lift`` turns the points or
boxes S2 names in it into objects like any other.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    COORDS,
    COORDS_PIXEL,
    EFFECT_READ,
    FAILURE_CALIBRATION_MISSING,
    FAILURE_INVALID_ARGS,
    FAILURE_UNREACHABLE,
    REQUIRE_CALIBRATION,
    REQUIRE_MODEL,
    REQUIRE_ROLE,
    REQUIRE_SHARE,
    SHARE_FRAMES,
    SIDE_EFFECT_NONE,
    PixelTarget,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

_CAMERA_REQUIRES = (
    f"{REQUIRE_ROLE}camera",
    f"{REQUIRE_CALIBRATION}camera",
    f"{REQUIRE_CALIBRATION}workspace",
)
_PERCEPTION_REQUIRES = (*_CAMERA_REQUIRES, f"{REQUIRE_MODEL}locator")
_SHARED_REQUIRES = (*_CAMERA_REQUIRES, f"{REQUIRE_SHARE}{SHARE_FRAMES}")
# No locator here: the object may be one S2 lifted out of a shared frame, and
# bounding the planner on the locator made every lifted object ungraspable on a
# workcell that runs no model service. Grounding is perceive.locate's gate.
_GRASP_REQUIRES = (
    f"{REQUIRE_ROLE}arm",
    f"{REQUIRE_ROLE}gripper",
    f"{REQUIRE_CALIBRATION}tcp",
    *_CAMERA_REQUIRES,
)
_COORD = {"type": "number", "minimum": 0}


def _fmt(point: Any) -> str:
    return f"({point[0]:.0f}, {point[1]:.0f}, {point[2]:.0f}) mm"


class Locate:
    """Report every detection of the queried objects, each under a fresh id."""

    spec = SkillSpec(
        skill_id="perceive.locate",
        version="2.0.0",
        card=(
            "Look at the table with the workcell camera and report every detection of "
            "the queried descriptions (e.g. ['red cup', 'plate']): id, label, score, "
            "centre (mm, arm base frame), size, yaw, how many depth points support "
            "it, and its geometry (mask or box; a point has no size), plus what the "
            "gripper holds. Several detections may answer one query; "
            "all of them are returned. Ids replace those of earlier calls and expire "
            "after a short while."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 80},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "min_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Lowest detection score reported; the workcell's when omitted.",
                },
            },
            "required": ["queries"],
        },
        effect=EFFECT_READ,
        requires=_PERCEPTION_REQUIRES,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=f"Locate {', '.join(args['queries'])} on the table",
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        found = await ctx.scene.locate(plan.args["queries"], min_score=plan.args.get("min_score"))
        held = ctx.scene.held
        table = ctx.frames.table
        listed = "; ".join(
            f"{o.object_id} {o.label} ({o.score:.2f}) at {_fmt(o.center)}" for o in found
        )
        return SkillOutcome.succeeded(
            {
                "objects": [obj.to_dict() for obj in found],
                "held": None if held is None else held.to_dict(),
                "table": None if table is None else table.to_dict(),
            },
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"Found {len(found)} detection(s): {listed}" if found else "Found nothing",
        )


class Snapshot:
    """Share one downscaled camera frame with S2, remembered for ``perceive.lift``."""

    spec = SkillSpec(
        skill_id="perceive.snapshot",
        version="1.0.0",
        card=(
            "Capture one frame from the workcell camera and return it as an image you "
            "can see, with its frame id and size in pixels. Point into it with "
            "perceive.lift to turn what you see into objects in the arm base frame. "
            "Only the newest few frames are kept, and they expire after a short while."
        ),
        args_schema={"type": "object", "properties": {}},
        effect=EFFECT_READ,
        requires=_SHARED_REQUIRES,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(skill_id=self.spec.skill_id, args=args, summary="Share a camera frame")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        shot = dict(await ctx.scene.snapshot())
        return SkillOutcome.succeeded(
            shot,
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"Frame {shot['frame']} ({shot['width']}x{shot['height']} pixels)",
        )


class Lift:
    """Turn points or boxes S2 names in a shared frame into base-frame objects."""

    spec = SkillSpec(
        skill_id="perceive.lift",
        version="1.0.0",
        card=(
            "Turn what you point at in a perceive.snapshot frame into objects, without "
            "moving. Each target has a label and either a point [u, v] on the object's "
            "surface or a box [x0, y0, x1, y1] around it; coords says whether they are "
            "pixels of the shared image, fractions (0-1) of its sides, or thousandths "
            "(0-1000). Each lifted target gets an object id with the same fields as "
            "perceive.locate (a point gives a surface point with no size); a target "
            "with no depth behind it is reported as dropped. Lifted ids live beside "
            "located ones until the next perceive.locate."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "frame": {"type": "string", "minLength": 1, "maxLength": 16},
                "targets": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "minLength": 1, "maxLength": 80},
                            "point": {
                                "type": "array",
                                "items": _COORD,
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "box": {
                                "type": "array",
                                "items": _COORD,
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["label"],
                    },
                },
                "coords": {"type": "string", "enum": list(COORDS), "default": COORDS_PIXEL},
            },
            "required": ["frame", "targets"],
        },
        effect=EFFECT_READ,
        requires=_SHARED_REQUIRES,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        labels = ", ".join(target.label for target in _targets(args))
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=f"Lift {labels} from frame {args['frame']}",
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        lifted, dropped = await ctx.scene.lift(
            plan.args["frame"], _targets(plan.args), coords=plan.args["coords"]
        )
        listed = "; ".join(f"{o.object_id} {o.label} at {_fmt(o.center)}" for o in lifted)
        result: dict[str, Any] = {"objects": [obj.to_dict() for obj in lifted]}
        if dropped:
            result["dropped"] = [dict(entry) for entry in dropped]
        return SkillOutcome.succeeded(
            result,
            side_effect_state=SIDE_EFFECT_NONE,
            message=(
                f"Lifted {len(lifted)} of {len(lifted) + len(dropped)} target(s)"
                + (f": {listed}" if listed else "")
            ),
        )


def _targets(args: Mapping[str, Any]) -> tuple[PixelTarget, ...]:
    targets = []
    for index, raw in enumerate(args["targets"]):
        point, box = raw.get("point"), raw.get("box")
        try:
            if box is not None and (box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("a box is [x0, y0, x1, y1] with x0 < x1 and y0 < y1")
            targets.append(
                PixelTarget(
                    str(raw["label"]),
                    point=None if point is None else (point[0], point[1]),
                    box=None if box is None else (box[0], box[1], box[2], box[3]),
                )
            )
        except ValueError as exc:
            raise SkillError(FAILURE_INVALID_ARGS, f"targets[{index}]: {exc}") from exc
    return tuple(targets)


class GraspPropose:
    """Ask the workcell's grasp planner for scored grasps of one located object."""

    spec = SkillSpec(
        skill_id="grasp.propose",
        version="1.0.0",
        card=(
            "Propose grasps for one object id from perceive.locate, without moving: "
            "each grasp has an id, the TCP poses pregrasp / grasp / retreat (mm, deg, "
            "arm base frame), the width the jaws should open to, the expected closed "
            "width, a score, and the planner that produced it. Every candidate is "
            "inside the workspace at planning time. Grasp ids expire with their object."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "object": {"type": "string", "minLength": 1, "maxLength": 16},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["object"],
        },
        effect=EFFECT_READ,
        requires=_GRASP_REQUIRES,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=f"Propose grasps for {args['object']}",
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        obj = ctx.scene.object(plan.args["object"])
        table = ctx.frames.table
        if table is None:
            raise SkillError(
                FAILURE_CALIBRATION_MISSING,
                "the table is not calibrated; run calib.workspace with table_z or table_points",
            )
        state = await ctx.motion.state()
        planner = ctx.grasp_planner
        proposed = await planner.candidates(obj, table=table, current=state.tcp)
        feasible, refused = [], []
        for candidate in proposed:
            try:
                for name in ("pregrasp", "grasp", "retreat"):
                    ctx.motion.check_target(candidate.waypoint(name))
            except SkillError as error:
                refused.append(error.message)
                continue
            feasible.append(candidate)
        if not feasible:
            reason = f"; first refusal: {refused[0]}" if refused else ""
            raise SkillError(
                FAILURE_UNREACHABLE,
                f"{planner.name} proposed no reachable grasp for {obj.object_id} ({obj.label}, "
                f"{obj.length:.0f} x {obj.width:.0f} x {obj.height:.0f} mm){reason}",
            )
        named = ctx.scene.remember_grasps(obj, feasible[: int(plan.args["top_k"])])
        ctx.note("grasps", {"planner": planner.name, "proposed": len(proposed), "refused": refused})
        listed = "; ".join(
            f"{g.grasp_id} score {g.score:.2f} open {g.open_mm:.0f} mm" for g in named
        )
        return SkillOutcome.succeeded(
            {
                "object": obj.to_dict(),
                "grasps": [grasp.to_dict() for grasp in named],
                "planner": planner.name,
            },
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"{len(named)} grasp(s) for {obj.object_id}: {listed}",
        )


def skills() -> tuple[Locate, Snapshot, Lift, GraspPropose]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (Locate(), Snapshot(), Lift(), GraspPropose())


__all__ = ["GraspPropose", "Lift", "Locate", "Snapshot", "skills"]

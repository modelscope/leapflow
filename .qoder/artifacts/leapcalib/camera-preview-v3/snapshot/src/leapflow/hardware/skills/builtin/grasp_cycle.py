# Copyright (c) Alibaba, Inc. and its affiliates.
"""A single approved node-local cycle: one lease, context and temporal observer."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from typing import Any, Mapping

from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_ACTUATE,
    FAILURE_GUARD_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    STATUS_HALTED,
    GraspCandidate,
    LocatedObject,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)


def _guard(message: str, result: Mapping[str, Any] | None = None) -> SkillError:
    return SkillError(FAILURE_GUARD_FAILED, message, status=STATUS_HALTED, result=result)


class GraspCycle:
    spec = SkillSpec(
        skill_id="grasp.cycle",
        version="1.0.0",
        effect=EFFECT_ACTUATE,
        consent=CONSENT_EACH_RUN,
        requires=(
            "role:arm",
            "role:gripper",
            "role:camera",
            "calib:camera",
            "role:camera.wrist",
            "calib:camera.wrist",
            "calib:tcp",
            "calib:workspace",
        ),
        card="Run one fixed-orientation grasp cycle with bounded IK paths, wrist alignment, "
        "on-support contact, a 5–10 mm proof lift, monitored carry, release and fresh "
        "overview-camera placement verification. Every failure stops the cycle with "
        "no retry, reset, disable or automatic retreat. One preview and approval per run.",
        args_schema={
            "type": "object",
            "properties": {
                "grasp_id": {"type": "string", "minLength": 1},
                "place": {"type": "object", "additionalProperties": True},
                "proof_lift_mm": {"type": "number", "minimum": 5, "maximum": 10, "default": 5},
                "samples": {"type": "integer", "minimum": 3, "maximum": 10, "default": 3},
                "speed_pct": {"type": "number", "minimum": 1, "maximum": 30, "default": 10},
            },
            "required": ["grasp_id", "place"],
        },
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        evidence = ctx.grasp_verification.qualification()
        grasp = ctx.scene.grasp(args["grasp_id"])
        obj = ctx.scene.object(grasp.object_id)
        state = await ctx.motion.state()
        if ctx.scene.held is not None:
            raise SkillError(
                FAILURE_PLAN_STALE, "cycle must start without a remembered held object"
            )
        limits = ctx.workcell.grasp_verify
        lift = float(args.get("proof_lift_mm", 5))
        if not limits["min_proof_lift_mm"] <= lift <= limits["max_proof_lift_mm"]:
            raise SkillError(FAILURE_INVALID_ARGS, "proof lift exceeds declared bounds")
        if not 0 < grasp.width_mm < grasp.open_mm <= ctx.workcell.gripper_spec["max_width_mm"]:
            raise SkillError(FAILURE_INVALID_ARGS, "grasp widths exceed the workcell jaw stroke")
        if (grasp.open_mm - grasp.width_mm) / 2 <= evidence["planar_p95_mm"] + limits[
            "clearance_mm"
        ]:
            raise SkillError(
                FAILURE_GUARD_FAILED, "jaw clearance does not exceed planar P95 plus margin"
            )
        place = Pose.from_mapping(args["place"])
        proof = grasp.grasp.translated(dz=lift)
        carry = place.translated(dz=grasp.retreat_mm or grasp.approach_mm)
        targets = (grasp.pregrasp, grasp.grasp, proof, carry, place, carry, state.tcp)
        paths = ctx.motion.check_cartesian_path(state, targets)
        placement_center = tuple(
            a + b - c for a, b, c in zip(obj.center, place.position, grasp.grasp.position)
        )
        return SkillPlan(
            self.spec.skill_id,
            args,
            "One qualified grasp cycle; stop on any uncertainty; no replay",
            targets=targets,
            limits={"speed_pct": float(args.get("speed_pct", 10))},
            basis={
                "start": state.tcp.to_dict(),
                "start_joints": dict(state.joints),
                "grasp": grasp.to_dict(),
                "object": obj.to_dict(),
                "paths": paths,
                "placement": replace(obj, center=placement_center).to_dict(),
                "qualification": evidence,
                "grasp_verify": dict(limits),
                "freshness": dict(ctx.workcell.freshness),
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        log = []
        stage = "freshness"
        try:
            if (
                dict(ctx.workcell.grasp_verify) != plan.basis["grasp_verify"]
                or dict(ctx.workcell.freshness) != plan.basis["freshness"]
            ):
                raise SkillError(
                    FAILURE_PLAN_STALE, "cycle verification tuning changed since approval"
                )
            state = await ctx.motion.state()
            start = Pose.from_mapping(plan.basis["start"])
            freshness = ctx.workcell.freshness
            if (
                state.tcp.distance_mm(start) > freshness["start_mm"]
                or state.tcp.rotation_to_deg(start) > freshness["start_deg"]
                or ctx.scene.held is not None
                or set(state.joints) != set(plan.basis["start_joints"])
                or any(
                    abs(state.joints[j] - v) > freshness["start_deg"]
                    for j, v in plan.basis["start_joints"].items()
                )
            ):
                raise SkillError(FAILURE_PLAN_STALE, "cycle start changed since approval")
            if ctx.grasp_verification.qualification() != plan.basis["qualification"]:
                raise SkillError(FAILURE_PLAN_STALE, "cycle physical qualification changed")
            planned_obj = LocatedObject.from_mapping(plan.basis["object"])
            seen = await ctx.scene.relocate(planned_obj)
            if (
                math.dist(seen.center, planned_obj.center)
                > ctx.workcell.grasp_verify["alignment_mm"]
            ):
                raise SkillError(FAILURE_PLAN_STALE, "approved object moved before the cycle")
            grasp = GraspCandidate.from_mapping(plan.basis["grasp"])
            samples = int(plan.args.get("samples", 3))
            force = float(plan.basis["qualification"]["command_force_n"])
            stage = "open"
            opened = await ctx.motion.gripper(grasp.open_mm, effort=force)
            if opened.blocked:
                raise _guard("initial opening was blocked")
            for index in range(len(plan.targets)):
                stage = ("pregrasp", "descend", "proof", "carry", "place", "retreat", "return")[
                    index
                ]
                for waypoint in (w for w in plan.basis["paths"] if w["target_index"] == index):
                    moved = await ctx.motion.move_joints(
                        waypoint["joints"], speed_pct=plan.limits["speed_pct"]
                    )
                    expected = Pose.from_mapping(waypoint["pose"])
                    if not ctx.motion.reached(expected, moved.state):
                        raise _guard("approved IK waypoint was not reached")
                    # Carry stays within the original temporal proof. A lost
                    # observation or excessive gap cannot be requalified in air.
                    if index in (3, 4):
                        held = await ctx.grasp_verification.verify("hold", samples=samples)
                        supported = held.get("evidence", {}).get("object_on_support") is True
                        log.append({"stage": stage, "waypoint": waypoint, "verification": held})
                        if not held["passed"] or (
                            held["state"] != "held_verified" and not (index == 4 and supported)
                        ):
                            raise _guard("carry or placement lost verified holding", held)
                if index == 0:
                    result = await ctx.grasp_verification.verify(
                        "pre_grasp", samples=samples, grasp=grasp.to_dict()
                    )
                    if not result["passed"]:
                        raise _guard("wrist pre-grasp alignment failed", result)
                    log.append({"stage": "pre_grasp", "verification": result})
                elif index == 1:
                    await ctx.motion.gripper(0, effort=force)
                    result = await ctx.grasp_verification.verify("hold", samples=samples)
                    if not result["passed"] or result["state"] != "contact_candidate":
                        raise _guard("fresh on-support contact was not established", result)
                    log.append({"stage": "contact", "verification": result})
                elif index == 2:
                    result = await ctx.grasp_verification.verify("hold", samples=samples)
                    if not result["passed"] or result["state"] != "held_verified":
                        raise _guard("bounded proof lift did not verify a hold", result)
                    log.append({"stage": "proof", "verification": result})
                elif index == 4:
                    result = await ctx.grasp_verification.verify("hold", samples=samples)
                    if (
                        not result["passed"]
                        or result.get("evidence", {}).get("object_on_support") is not True
                    ):
                        raise _guard("independent object support required before release", result)
                    log.append({"stage": "place", "verification": result})
                    released = await ctx.motion.gripper(grasp.open_mm, effort=force)
                    if (
                        released.blocked
                        or released.width_mm
                        <= grasp.width_mm + ctx.workcell.grasp_verify["clearance_mm"]
                    ):
                        raise _guard("release opening was not independently observed")
                    ctx.scene.release()
            stage = "placement_verification"
            expected_object = LocatedObject.from_mapping(plan.basis["placement"])
            placed = await ctx.scene.relocate(expected_object)
            if (
                math.dist(placed.center, expected_object.center)
                > ctx.workcell.grasp_verify["placement_mm"]
            ):
                raise _guard("overview-camera placement differs from approved destination")
            ctx.note(
                "grasp_cycle", {"stages": log, "placement": placed.to_dict(), "cycle_passed": True}
            )
            return SkillOutcome.succeeded(
                {
                    "cycle_passed": True,
                    "stages": log,
                    "placement": placed.to_dict(),
                    "qualification_digest": plan.basis["qualification"]["qualification_digest"],
                }
            )
        except SkillError as error:
            error.result = {
                **error.result,
                "cycle_passed": False,
                "consecutive_successes": 0,
                "stage": stage,
                "completed_stages": log,
            }
            ctx.note("grasp_cycle", error.result)
            raise
        except (Exception, asyncio.CancelledError):
            ctx.note(
                "grasp_cycle",
                {
                    "cycle_passed": False,
                    "consecutive_successes": 0,
                    "stage": stage,
                    "completed_stages": log,
                },
            )
            raise


class GripperQualify:
    spec = SkillSpec(
        skill_id="gripper.qualify",
        version="1.0.0",
        effect=EFFECT_ACTUATE,
        consent=CONSENT_EACH_RUN,
        requires=("role:arm", "role:gripper"),
        card="Acquire command and settled jaw feedback at approved widths in empty air. "
        "No arm movement. Independent caliper train/holdout readings are required "
        "for a jaw-stroke certificate; encoder feedback alone never certifies it.",
        args_schema={
            "type": "object",
            "properties": {
                "widths": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0},
                    "minItems": 3,
                    "maxItems": 10,
                },
            },
            "required": ["widths"],
        },
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        widest = ctx.workcell.gripper_spec["max_width_mm"]
        widths = args["widths"]
        if len(set(widths)) != len(widths) or any(w > widest for w in widths):
            raise SkillError(
                FAILURE_INVALID_ARGS, "distinct widths within qualified jaw bounds required"
            )
        state = await ctx.motion.state()
        return SkillPlan(
            self.spec.skill_id,
            args,
            "Empty-air jaw acquisition only; no arm motion",
            basis={
                "start": state.tcp.to_dict(),
                "start_joints": dict(state.joints),
                "widths": [widest, min(widths), widest, *widths, widest],
                "max_width_mm": widest,
                "freshness": dict(ctx.workcell.freshness),
            },
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        state = await ctx.motion.state()
        start = Pose.from_mapping(plan.basis["start"])
        freshness = ctx.workcell.freshness
        if (
            state.tcp.distance_mm(start) > freshness["start_mm"]
            or state.tcp.rotation_to_deg(start) > freshness["start_deg"]
            or dict(freshness) != plan.basis["freshness"]
            or ctx.workcell.gripper_spec["max_width_mm"] != plan.basis["max_width_mm"]
            or set(state.joints) != set(plan.basis["start_joints"])
            or any(
                abs(state.joints[j] - v) > freshness["start_deg"]
                for j, v in plan.basis["start_joints"].items()
            )
        ):
            raise SkillError(FAILURE_PLAN_STALE, "jaw acquisition start moved since approval")
        rows = []
        for index, width in enumerate(plan.basis["widths"]):
            result = await ctx.motion.gripper(width)
            rows.append(
                {
                    "sample": index,
                    "command_mm": width,
                    "feedback_mm": result.width_mm,
                    "blocked": result.blocked,
                }
            )
            ctx.note("jaw_acquisition", rows)
            if result.blocked:
                raise _guard("empty-air jaw command was blocked", {"samples": rows})
            current = await ctx.motion.state()
            if (
                current.tcp.distance_mm(start) > freshness["start_mm"]
                or current.tcp.rotation_to_deg(start) > freshness["start_deg"]
                or set(current.joints) != set(plan.basis["start_joints"])
                or any(
                    abs(current.joints[j] - v) > freshness["start_deg"]
                    for j, v in plan.basis["start_joints"].items()
                )
            ):
                raise _guard("arm moved during empty-air jaw acquisition", {"samples": rows})
        return SkillOutcome.succeeded(
            {
                "samples": rows,
                "qualified": False,
                "reason": "independent caliper holdout evidence required",
            }
        )


def skills() -> tuple[GraspCycle, GripperQualify]:
    return GraspCycle(), GripperQualify()

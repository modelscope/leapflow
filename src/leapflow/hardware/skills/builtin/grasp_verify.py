# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only wrist verification and governed installation of physical evidence."""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_ACTUATE,
    EFFECT_READ,
    FAILURE_GUARD_FAILED,
    FAILURE_PLAN_STALE,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

_REQUIRES = (
    "role:arm",
    "role:camera.wrist",
    "calib:camera.wrist",
    "role:gripper",
    "calib:tcp",
    "calib:workspace",
)


class GraspVerify:
    spec = SkillSpec(
        skill_id="grasp.verify",
        version="1.0.0",
        card="Read stationary wrist RGB-D, qualified pad geometry, jaw stroke and force evidence. "
        "Stages pre_grasp, hold and empty never command motion. A standalone hold cannot "
        "prove a lift without a continuous on-support baseline in the same run.",
        effect=EFFECT_READ,
        requires=_REQUIRES,
        args_schema={
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": ["pre_grasp", "hold", "empty"]},
                "samples": {"type": "integer", "minimum": 3, "maximum": 10, "default": 3},
                "grasp": {"type": "object", "additionalProperties": True},
            },
            "required": ["stage"],
        },
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillOutcome:
        result = await ctx.grasp_verification.verify(
            args["stage"],
            samples=args.get("samples", 3),
            grasp=args.get("grasp"),
        )
        if not result["passed"]:
            return SkillOutcome.failed(
                FAILURE_GUARD_FAILED, "grasp evidence did not pass", result=result
            )
        return SkillOutcome.succeeded(result, side_effect_state=SIDE_EFFECT_NONE)

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return await self.plan(ctx, plan.args)


class InstallGraspEvidence:
    spec = SkillSpec(
        skill_id="calib.grasp_evidence",
        version="1.0.0",
        effect=EFFECT_ACTUATE,
        consent=CONSENT_EACH_RUN,
        requires=_REQUIRES,
        card="Validate and persist independently measured geometry and stationary N-unit "
        "force qualification bound to current calibrations. No arm or gripper motion. "
        "Incomplete, synthetic or mismatched certificates are refused.",
        args_schema={
            "type": "object",
            "properties": {
                "qualification": {"type": "object", "additionalProperties": True},
            },
            "required": ["qualification"],
        },
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        try:
            admitted = ctx.grasp_verification.admit(args["qualification"])
        except (KeyError, TypeError, ValueError) as error:
            raise SkillError(FAILURE_GUARD_FAILED, str(error)) from error
        return SkillPlan(
            self.spec.skill_id,
            args,
            "Install independent grasp measurement qualification; no motion",
            basis=admitted,
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        try:
            admitted = ctx.grasp_verification.admit(plan.args["qualification"])
        except (KeyError, TypeError, ValueError) as error:
            raise SkillError(
                FAILURE_PLAN_STALE, f"qualification no longer matches the approved plan: {error}"
            ) from error
        if any(admitted[key] != plan.basis.get(key) for key in admitted):
            raise SkillError(FAILURE_PLAN_STALE, "qualified geometry changed before installation")
        recorded_at = ctx.record_calibration(
            "grasp.qualification", parameters=admitted["parameters"]
        )
        return SkillOutcome.succeeded(
            {"recorded_at": recorded_at, "qualification_digest": admitted["qualification_digest"]}
        )


class GraspInspect:
    spec = SkillSpec(
        skill_id="grasp.inspect",
        version="1.0.0",
        effect=EFFECT_READ,
        requires=_REQUIRES,
        card="Read installed physical grasp qualification, calibration dependencies and fresh "
        "wrist depth measurements for reporting. No motion or certificate installation.",
        args_schema={"type": "object", "properties": {}},
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillOutcome:
        try:
            result = await ctx.grasp_verification.inspect()
        except (KeyError, TypeError, ValueError) as error:
            raise SkillError(FAILURE_GUARD_FAILED, f"grasp inspection refused: {error}") from error
        return SkillOutcome.succeeded(result, side_effect_state=SIDE_EFFECT_NONE)

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return await self.plan(ctx, plan.args)


def skills() -> tuple[GraspVerify, InstallGraspEvidence, GraspInspect]:
    return GraspVerify(), InstallGraspEvidence(), GraspInspect()

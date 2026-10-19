# Copyright (c) Alibaba, Inc. and its affiliates.
"""Arm skills outside a motion program: read the arm, go to the declared home posture.

Everything else that moves -- reaching a pose, stepping the tool, driving the
gripper, picking, placing -- is a ``motion.sequence`` the model writes and a
human approves once.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_ACTUATE,
    EFFECT_READ,
    FAILURE_SKILL_UNAVAILABLE,
    REQUIRE_ROLE,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

_SPEED_ARG = {
    "type": "number",
    "minimum": 1,
    "maximum": 100,
    "description": "Speed, percent of the arm maximum; capped by the workcell limit.",
}


def _fmt(pose: Pose) -> str:
    return f"({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) mm"


class ArmStatus:
    """Read the arm: TCP and flange pose, joints, gripper width, readiness."""

    spec = SkillSpec(
        skill_id="arm.state",
        version="1.1.0",
        card=(
            "Read the arm without moving it: the tool centre point and flange poses in "
            "the arm base frame (mm, deg), joint positions (deg), gripper width (mm), "
            "whether the arm is enabled, the calibrated workspace box, and the table "
            "plane when measured."
        ),
        args_schema={"type": "object", "properties": {}},
        effect=EFFECT_READ,
        requires=(f"{REQUIRE_ROLE}arm",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(skill_id=self.spec.skill_id, args=args, summary="Read the arm state")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        state = await ctx.motion.state()
        workspace = ctx.frames.workspace
        table = ctx.frames.table
        return SkillOutcome.succeeded(
            {
                **state.to_dict(),
                "tcp_calibrated": "tcp" in ctx.frames.versions,
                "workspace": None if workspace is None else workspace.to_dict(),
                "table": None if table is None else table.to_dict(),
            },
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"The tool is at {_fmt(state.tcp)}",
        )


class Home:
    """Move the arm to the workcell's declared home posture."""

    spec = SkillSpec(
        skill_id="motion.home",
        version="1.1.0",
        card=(
            "Move the arm to the workcell's declared home posture in joint space. The "
            "declared FK, workspace and tool cone are checked before approval and execution."
        ),
        args_schema={"type": "object", "properties": {"speed_pct": _SPEED_ARG}},
        effect=EFFECT_ACTUATE,
        requires=(f"{REQUIRE_ROLE}arm",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        home = dict(ctx.workcell.home)
        if not home:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "this workcell declares no home posture (workcell 'home' or arm "
                "kinematics.home_joint_targets)",
            )
        state = await ctx.motion.state()
        ctx.motion.check_kinematics(state)
        target = ctx.motion.check_joints(home, state.joints)
        speed = args.get("speed_pct")
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=f"Move the arm to its home posture ({len(home)} joints)",
            targets=(target,),
            limits={} if speed is None else {"speed_pct": float(speed)},
            basis={"home": home},
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        result = await ctx.motion.move_joints(
            plan.basis["home"], speed_pct=plan.limits.get("speed_pct")
        )
        return SkillOutcome.succeeded(
            {"tcp": result.state.tcp.to_dict(), "elapsed_s": round(result.elapsed_s, 2)},
            message=plan.summary,
        )


def skills() -> tuple[ArmStatus, Home]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (ArmStatus(), Home())


__all__ = ["ArmStatus", "Home", "skills"]

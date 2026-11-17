# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded grasp checks: detached geometry jobs, owner-only handle publication."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from leapflow.hardware.skills.ports import (
    MotionGeometry, Deadline, Evaluation, FrozenJoints, PlanningCheckpoint, PreparedPlanning, SkillContext,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_CALIBRATION_MISSING, FAILURE_GRASP_INFEASIBLE, FAILURE_INTERNAL_DEFECT,
        FAILURE_INVALID_ARGS, FAILURE_STOPPED,
    FAILURE_OUT_OF_WORKSPACE, FAILURE_PLAN_STALE, FAILURE_TIMEOUT, FAILURE_UNREACHABLE,
    FAILURE_OBSERVATION_INSUFFICIENT,
    EFFECT_READ, SIDE_EFFECT_NONE, ArmState, GraspCandidate, SkillError, SkillOutcome, SkillPlan,
)


@dataclass(frozen=True)
class GraspJob:
    geometry: MotionGeometry
    state: ArmState
    candidate: GraspCandidate
    index: int
    max_width_mm: float

    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation:
        motion = self.geometry.bind(checkpoint)
        checkpoint.check()
        candidate = self.candidate
        if (not all(math.isfinite(v) for v in (
                candidate.width_mm, candidate.open_mm, candidate.approach_mm,
                candidate.retreat_mm, candidate.score))
                or not 0 <= candidate.width_mm <= candidate.open_mm <= self.max_width_mm):
            raise SkillError(FAILURE_GRASP_INFEASIBLE, "candidate does not fit the declared gripper opening",
                             result={"candidate": self.index, "max_width_mm": self.max_width_mm})
        reach = motion.reach_tracker(self.state)
        for name in ("pregrasp", "grasp", "retreat"):
            try:
                checkpoint.check()
                target = candidate.waypoint(name)
                motion.check_target(target)
                reach.move(target, linear=name != "pregrasp", label=f"candidate {self.index} {name}")
            except SkillError as error:
                if error.failure_code not in {FAILURE_UNREACHABLE, FAILURE_OUT_OF_WORKSPACE}:
                    raise
                problem = {**error.result, "candidate": self.index, "waypoint": name,
                           "failure_code": error.failure_code, "error": error.message}
                return Evaluation(SkillOutcome.failed(error.failure_code, error.message, result=problem),
                                  verified=reach.verified,
                                  evidence={"ik_calls": reach.ik_calls, "reach": reach.to_dict()})
        return Evaluation(SkillOutcome.succeeded({"candidate": self.index}, side_effect_state=SIDE_EFFECT_NONE), reach.metrics(),
                          reach.verified, {"ik_calls": reach.ik_calls, "reach": reach.to_dict()})


async def propose(ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
    deadline = min(ctx.planning_deadline, time.monotonic() + ctx.planning_settings.timeout_s)
    try:
        Deadline(deadline).check()
        return await asyncio.wait_for(_propose(ctx, plan, deadline), max(0, deadline - time.monotonic()))
    except asyncio.TimeoutError as error:
        raise SkillError(FAILURE_TIMEOUT, "grasp planning deadline exceeded; no handles published") from error


async def _propose(ctx: SkillContext, plan: SkillPlan, deadline: float) -> SkillOutcome:
    basis = await ctx.planning_basis()
    obj = ctx.scene.object(plan.args["object"])
    table = ctx.frames.table
    if table is None:
        raise SkillError(FAILURE_CALIBRATION_MISSING,
                         "the table is not calibrated; run calib.workspace with table_z or table_points")
    state = await ctx.motion.state()
    state = replace(state, joints=FrozenJoints.capture(state.joints))
    held = ctx.scene.held
    geometry = ctx.motion.geometry_snapshot()
    planner = ctx.grasp_planner
    proposed = await planner.candidates(obj, table=table, current=state.tcp)
    settings = ctx.planning_settings
    settings.check_size([candidate.to_dict() for candidate in proposed])
    if len(proposed) > settings.max_jobs:
        raise SkillError(FAILURE_INVALID_ARGS, "planner proposal count exceeds admitted job budget")
    jobs = tuple(GraspJob(geometry, state, candidate, index, ctx.workcell.gripper_spec["max_width_mm"])
                 for index, candidate in enumerate(proposed, 1))
    evaluated = await ctx.calculations.evaluate(jobs, deadline=deadline)
    if len(evaluated) != len(proposed):
        raise SkillError(FAILURE_INTERNAL_DEFECT, "grasp worker response count differs")
    if basis != await ctx.planning_basis():
        raise SkillError(FAILURE_PLAN_STALE, "physical or calibration basis changed during grasp planning")
    PreparedPlanning((), objects=(obj,), held=held).check_scene(ctx)
    Deadline(deadline).check()
    feasible, refused = [], []
    observation_refusals = []
    observation_enabled = bool(ctx.workcell.observation.get('enabled', False))
    for candidate, evaluation in zip(proposed, evaluated):
        if not isinstance(evaluation, Evaluation) or not isinstance(evaluation.value, SkillOutcome):
            raise SkillError(FAILURE_INTERNAL_DEFECT, "invalid grasp calculation response")
        if evaluation.value.failure_code in {FAILURE_INTERNAL_DEFECT, FAILURE_TIMEOUT, FAILURE_STOPPED}:
            raise SkillError(evaluation.value.failure_code, evaluation.value.message)
        if evaluation.value.ok:
            if observation_enabled:
                report = dict(ctx.scene.observation(obj, candidate))
                if not report.get('sufficient', False):
                    observation_refusals.append(report)
                    continue
            feasible.append(candidate)
        else:
            problem = evaluation.value
            refused.append({**problem.result, "failure_code": problem.failure_code, "error": problem.message})
    reach_basis: dict[str, Any] = {
        "verified": bool(evaluated) and all(e.verified for e in evaluated),
        "verified_candidates": sum(e.verified for e in evaluated),
        "evaluated_candidates": len(evaluated),
        "detail": "; ".join(dict.fromkeys(
            str(e.evidence.get("reach", {}).get("detail", "")) for e in evaluated
            if e.evidence.get("reach", {}).get("detail")
        )),
    }
    if not feasible and observation_refusals:
        raise SkillError(FAILURE_OBSERVATION_INSUFFICIENT,
                         f'{obj.object_id}: reachable grasps lack sufficient measurement; inspect observation.reasons',
                         result={'object': obj.to_dict(), 'observation': observation_refusals,
                                 'reach': reach_basis, 'problems': refused,
                                 'next': ['perceive.survey', 'perceive.snapshot with_cameras=[all]']})
    if not feasible:
        reason = "; " + "; ".join(item["error"] for item in refused) if refused else ""
        code = FAILURE_UNREACHABLE if any(item["failure_code"] != FAILURE_GRASP_INFEASIBLE for item in refused) else FAILURE_GRASP_INFEASIBLE
        raise SkillError(code, f"{planner.name} proposed no feasible grasp for {obj.object_id} "
                         f"({obj.label}, {obj.length:.0f} x {obj.width:.0f} x {obj.height:.0f} mm){reason}",
                         result={"object": obj.to_dict(), "problems": refused, "reach": reach_basis})
    ctx.note("grasps", {"planner": planner.name, "proposed": len(proposed), "refused": refused,
                       "ik_calls": sum(int(e.evidence.get("ik_calls", 0)) for e in evaluated),
                       "state_reads": 1})
    outcome: SkillOutcome | None = None

    def validate(named: tuple[GraspCandidate, ...]) -> None:
        nonlocal outcome
        listed = "; ".join(f"{g.grasp_id} score {g.score:.2f} open {g.open_mm:.0f} mm" for g in named)
        outcome = replace(SkillOutcome.succeeded(
            {"object": obj.to_dict(), "grasps": [g.to_dict() for g in named],
             "planner": planner.name, "reach": reach_basis, "problems": refused,
             "observation": [dict(ctx.scene.observation(obj, g)) for g in named] if observation_enabled else []},
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"{len(named)} grasp(s) for {obj.object_id}: {listed}"),
            evidence=dict(ctx.evidence), effect=EFFECT_READ)
        settings.check_size({"outcome": outcome.to_dict()})
        PreparedPlanning((), objects=(obj,), held=held).check_scene(ctx)
        Deadline(deadline).check()

    # Preserve semantic score/order; validate exact handles and response before publication.
    ctx.scene.remember_grasps(obj, feasible[: int(plan.args["top_k"])], validate=validate)
    if outcome is None:
        raise SkillError(FAILURE_INTERNAL_DEFECT, "scene did not validate grasp publication")
    return outcome

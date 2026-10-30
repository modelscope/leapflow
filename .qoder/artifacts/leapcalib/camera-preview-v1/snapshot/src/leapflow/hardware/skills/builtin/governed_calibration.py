# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed stationary collection/reference registration and read-only diagnostics."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import CalibrationPort, CalibrationReferencePort, CalibrationRollbackPort, CalibrationTrialPort, CalibrationPosePort, SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN, EFFECT_ACTUATE, EFFECT_READ, FAILURE_SKILL_UNAVAILABLE,
    REGION_BOOTSTRAP_UNTIL_MEASURED, SIDE_EFFECT_NONE, SkillError, SkillOutcome, SkillPlan, SkillSpec,
)


def _port(ctx: SkillContext) -> CalibrationPort:
    port = ctx.calibration
    if not isinstance(port, CalibrationPort):
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no private calibration acquisition port")
    return port


class Collect:
    spec = SkillSpec(
        skill_id="calib.collect", version="1.0.0",
        card="Collect bounded settled raw-joint feedback and exact-role lossless metric camera ROIs. "
             "This stationary collection method commands no motion. The frozen acquisition policy "
             "and private retention bounds are declared by the workcell; each run is admitted and leased. "
             "Return redacted immutable acquisition digests, not a response or physical qualification.",
        args_schema={"type": "object", "required": ["region"], "properties": {
            "region": {"type": "array", "minItems": 4, "maxItems": 4,
                       "items": {"type": "integer", "minimum": 0, "maximum": 16384}},
            "frames": {"type": "integer", "minimum": 1, "maximum": 128, "default": 1},
        }},
        effect=EFFECT_ACTUATE, leased=True, consent=CONSENT_EACH_RUN, requires=("role:arm",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await _port(ctx).collection_basis(args)
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
                         summary=f"Stationary private collection of {args.get('frames', 1)} settled metric captures; no motion",
                         basis={"collection": dict(basis)}, limits={"max_duration_s": 120.0})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(await _port(ctx).collect(plan),
                                      message="Stationary acquisition persisted; no motion or physical qualification")


def _reference_port(ctx: SkillContext) -> CalibrationReferencePort:
    port = ctx.calibration
    if not isinstance(port, CalibrationReferencePort):
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no task-local reference registration port")
    return port


class AlignReference:
    spec = SkillSpec(
        skill_id="calib.align_reference", version="1.0.0",
        card="Register a task-local stationary textured scene reference from at least three fresh "
             "settled metric captures and persistent patch tracking through the owner compute worker. "
             "Each leased run persists private evidence under its exact approved policy. "
             "No motion, independent target witness, clearance certificate, global correction, or grasp readiness.",
        args_schema={"type": "object", "required": ["region"], "properties": {
            "region": {"type": "array", "minItems": 4, "maxItems": 4,
                       "items": {"type": "integer", "minimum": 0, "maximum": 16384}},
            "frames": {"type": "integer", "minimum": 3, "maximum": 16, "default": 3},
        }},
        effect=EFFECT_ACTUATE, leased=True, consent=CONSENT_EACH_RUN, requires=("role:arm",),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await _reference_port(ctx).reference_basis(args)
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
                         summary=f"Register a stationary task-local scene reference over {args.get('frames', 3)} captures; "
                                 "private persistence, no motion or physical qualification",
                         basis=dict(basis), limits={"max_duration_s": 120.0})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(await _reference_port(ctx).align_reference(plan),
                                      message="Task-local reference registered; no motion or physical qualification")


class Inspect:
    spec = SkillSpec(
        skill_id="calib.inspect", version="1.0.0",
        card="Read redacted calibration acquisition and task-local reference summaries for this workcell. "
             "No camera capture, raw evidence export, motion, persistence, or model activation.",
        args_schema={"type": "object", "properties": {
            "evidence_digest": {"type": "string", "maxLength": 256, "default": ""},
        }},
        effect=EFFECT_READ,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(skill_id=self.spec.skill_id, args=args, summary="Read private acquisition summaries")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(await _port(ctx).inspect(evidence_digest=plan.args.get("evidence_digest", "")),
                                      side_effect_state=SIDE_EFFECT_NONE)


class Rollback:
    spec = SkillSpec(
        skill_id="calib.rollback", version="1.0.0",
        card="Governed between-run rollback of the exact active physical model generation. "
             "Restore only a revalidated compatible same-owner prior projection; otherwise "
             "atomically deactivate the correction and stop precision use pending requalification. "
             "Requires each-run consent and an exclusive lease; never commands motion or a gripper.",
        args_schema={"type": "object", "required": ["expected_generation"], "properties": {
            "expected_generation": {"type": "integer", "minimum": 1},
        }},
        effect=EFFECT_ACTUATE, leased=True, consent=CONSENT_EACH_RUN,
        motion_allowed=False, requires=("role:arm",),
    )

    def _port(self, ctx: SkillContext) -> CalibrationRollbackPort:
        port = ctx.calibration
        if not isinstance(port, CalibrationRollbackPort):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no governed rollback port")
        return port

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await self._port(ctx).rollback_basis(args)
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
                         summary=f"Roll back active generation {args['expected_generation']}; "
                                 "unqualified prior causes stop-use, no motion",
                         basis={"rollback": dict(basis)}, limits={"max_duration_s": 120.0})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        result = await self._port(ctx).rollback(plan)
        return SkillOutcome.succeeded(result,
            message="Physical model rolled back; no motion" if result["state"] == "rolled_back" else
                    "Correction deactivated; requalify before precision use, no motion")


class AssessMotion:
    spec = SkillSpec(
        skill_id="calib.assess_motion", version="1.0.0",
        card="Read indexed private acquisitions, build a predeclared source-disjoint training/withheld split, "
             "solve one metric_motion.v1 camera shadow candidate in the isolated worker and assess withheld evidence. "
             "Each leased read requires consent. No motion, activation, raw export or geometry/zero correction.",
        args_schema={"type": "object", "required": ["pairs", "training", "mount", "prior"], "properties": {
            "pairs": {"type": "array", "minItems": 2, "maxItems": 64, "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "string", "minLength": 64, "maxLength": 64}}},
            "training": {"type": "integer", "minimum": 1, "maximum": 63},
            "mount": {"type": "object", "additionalProperties": True}, "prior": {"type": "object", "additionalProperties": True},
            "prior_uncertainty": {"type": "array", "minItems": 6, "maxItems": 6,
                                  "items": {"type": "number", "exclusiveMinimum": 0}},
            "tracking": {"type": "object", "additionalProperties": True}, "solve": {"type": "object", "additionalProperties": True}, "evaluation": {"type": "object", "additionalProperties": True},
        }},
        effect=EFFECT_READ, leased=True, motion_allowed=False,
        consent=CONSENT_EACH_RUN, requires=("role:arm",),
    )

    def _port(self, ctx: SkillContext) -> CalibrationTrialPort:
        port = ctx.calibration
        if not isinstance(port, CalibrationTrialPort):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no indexed camera trial port")
        return port

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await self._port(ctx).trial_basis(args)
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
            summary=f"Camera-only shadow trial of {len(args['pairs'])} pre-split indexed pairs; no motion or activation",
            basis={"trial": dict(basis)}, limits={"max_duration_s": 120.0})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(await self._port(ctx).assess_motion(plan), side_effect_state=SIDE_EFFECT_NONE,
                                      message="Camera shadow candidate assessed; no physical qualification or activation")


class CollectPoses:
    spec = SkillSpec(
        skill_id="calib.collect_poses", version="1.0.0",
        card="Move through at most eight approved nominal raw-joint poses at at most ten percent speed, "
             "collect two settled metric captures per pose, and return disjoint factor pairs. "
             "Requires an indexed immutable campaign reference and uncertainty-inflated whole-link clearance. "
             "No automatic return, gripper motion, corrected zeros/IK or model activation.",
        args_schema={"type": "object", "required": ["region", "poses"], "properties": {
            "region": {"type": "array", "minItems": 4, "maxItems": 4,
                       "items": {"type": "integer", "minimum": 0, "maximum": 16384}},
            "poses": {"type": "array", "minItems": 2, "maxItems": 8, "items": {
                "type": "object", "minProperties": 1, "maxProperties": 16,
                "additionalProperties": {"type": "number"}}},
            "speed_pct": {"type": "number", "exclusiveMinimum": 0, "maximum": 10, "default": 10},
        }},
        effect=EFFECT_ACTUATE, leased=True, consent=CONSENT_EACH_RUN, requires=("role:arm",),
        region=REGION_BOOTSTRAP_UNTIL_MEASURED,
    )

    def _port(self, ctx: SkillContext) -> CalibrationPosePort:
        port = ctx.calibration
        if not isinstance(port, CalibrationPosePort):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no governed multi-pose acquisition port")
        return port

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await self._port(ctx).poses_basis(args)
        from leapflow.hardware.skills.geometry import Pose
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
            summary=f"Acquire two captures at each of {len(args['poses'])} bounded poses; no return or gripper motion",
            targets=tuple(Pose(**p) for p in basis["poses"]["targets"]), basis=dict(basis),
            limits={"max_duration_s": basis["poses"]["duration_s"], "speed_pct_max": args.get("speed_pct", 10)})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return SkillOutcome.succeeded(await self._port(ctx).collect_poses(plan),
                                      message="Bounded pose captures persisted; arm remains at final pose; no qualification")


def skills() -> tuple[Collect | AlignReference | Inspect | Rollback | AssessMotion | CollectPoses, ...]:
    return (Collect(), AlignReference(), Inspect(), Rollback(), AssessMotion(), CollectPoses())

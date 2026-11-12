# Copyright (c) Alibaba, Inc. and its affiliates.
"""Discoverable stationary-board bridge; physical actions stay outside this skill."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import BridgeCalibrationPort, SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN, EFFECT_ACTUATE, FAILURE_SKILL_UNAVAILABLE,
    SkillError, SkillOutcome, SkillPlan, SkillSpec,
)


class BridgeCalibration:
    spec = SkillSpec(
        skill_id="calib.bridge", version="1.0.0",
        card="Acquire both declared cameras against a stationary ChArUco sheet with qualified raw-feedback bracketing. "
             "Persist private indexed evidence, fit a conditional base-to-fixed-camera shadow transform using an exact "
             "validated wrist dependency, and evaluate preallocated source-disjoint withheld windows. "
             "Each leased run requires consent. No motion, enable/reset, frame export, activation or absolute accuracy certificate.",
        args_schema={"type": "object", "required": ["fixed_region", "wrist_region"], "properties": {
            name: {"type": "array", "minItems": 4, "maxItems": 4,
                   "items": {"type": "integer", "minimum": 0, "maximum": 16384}}
            for name in ("fixed_region", "wrist_region")}},
        effect=EFFECT_ACTUATE, leased=True, motion_allowed=False,
        consent=CONSENT_EACH_RUN, requires=("role:arm",),
    )

    def _port(self, ctx: SkillContext) -> BridgeCalibrationPort:
        port = ctx.bridge
        if not isinstance(port, BridgeCalibrationPort):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "owner has no paired bridge calibration port")
        return port

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        basis = await self._port(ctx).basis(args)
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
                         summary="Stationary paired bridge acquisition and withheld shadow evaluation; no motion or activation",
                         basis={"bridge": dict(basis)}, limits={"max_duration_s": 120.0})

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        report = await self._port(ctx).execute(plan)
        return SkillOutcome.succeeded(report, message=f"Paired bridge operationally completed; scoped verdict {report['verdict']}; shadow only")


def skills() -> tuple[BridgeCalibration, ...]:
    return (BridgeCalibration(),)

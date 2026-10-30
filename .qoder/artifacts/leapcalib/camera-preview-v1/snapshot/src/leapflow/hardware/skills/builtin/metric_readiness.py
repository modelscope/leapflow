# Copyright (c) Alibaba, Inc. and its affiliates.
"""Target-free metric camera readiness: no actuator or calibration-write port."""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import MetricReadinessPort, SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_READ,
    FAILURE_SKILL_UNAVAILABLE,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)


class MetricReadiness:
    spec = SkillSpec(
        skill_id="calib.metric_readiness",
        version="1.0.0",
        card="Capture bounded fresh metric depth samples without a target or motion. "
        "Return live profile, valid depth, sequence and receipt timing summaries; "
        "sampled evidence stays private, non-syncable and TTL/quota managed on the node. "
        "No mount accuracy certification or calibration activation.",
        args_schema={
            "type": "object",
            "properties": {
                "camera_role": {
                    "type": "string",
                    "enum": ["camera", "camera.wrist"],
                    "default": "camera",
                },
                "frames": {"type": "integer", "minimum": 2, "maximum": 32, "default": 8},
                "points": {"type": "integer", "minimum": 16, "maximum": 2048, "default": 256},
                "stamp_uncertainty_s": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.05,
                },
                "ttl_s": {"type": "number", "minimum": 1, "maximum": 86400, "default": 3600},
                "quota_bytes": {
                    "type": "integer",
                    "minimum": 1048576,
                    "maximum": 67108864,
                    "default": 16777216,
                },
            },
        },
        effect=EFFECT_READ,
        # The exact camera role is an argument; camera_vision resolves it fail-closed.
        requires=(),
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return SkillPlan(
            skill_id=self.spec.skill_id, args=args, summary="Measure metric camera readiness"
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        args = plan.args
        role = args.get("camera_role", "camera")
        port = ctx.camera_vision(role)
        if not isinstance(port, MetricReadinessPort):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "camera has no metric-readiness port")
        result = await port.metric_readiness(
            frames=args.get("frames", 8),
            points=args.get("points", 256),
            role="wrist" if role == "camera.wrist" else "fixed",
            stamp_uncertainty_s=args.get("stamp_uncertainty_s", 0.05),
            ttl_s=args.get("ttl_s", 3600),
            quota_bytes=args.get("quota_bytes", 16777216),
        )
        return SkillOutcome.succeeded(
            result,
            side_effect_state=SIDE_EFFECT_NONE,
            message="Metric readiness captured; no calibration or motion performed",
        )


def skills() -> tuple[MetricReadiness, ...]:
    return (MetricReadiness(),)

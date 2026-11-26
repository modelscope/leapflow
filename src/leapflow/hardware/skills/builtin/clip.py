# Copyright (c) Alibaba, Inc. and its affiliates.
"""Passive temporal evidence; camera access and encoding belong to the owner port."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_READ, FAILURE_INVALID_ARGS, SIDE_EFFECT_NONE, SkillError,
    SkillOutcome, SkillPlan, SkillSpec,
)


class Clip:
    spec = SkillSpec(
        skill_id="perceive.clip", version="1.0.0",
        card=(
            "Get a bounded colour clip from one camera, without motion. source=live "
            "samples fresh frames at 2 fps; source=episode uses the last seconds of "
            "a completed run's recorded JPEG sequence, preserving its duration. "
            "Provide run_id for episode evidence. Defaults: 6 seconds, max_side=320; "
            "limits: 10 seconds, 448 pixels, 700 KiB. Historical evidence is labelled "
            "and is not current state. export_evidence retains it under session consent "
            "for optional advisory consultation; video bytes are never attached to primary chat."
        ),
        args_schema={"type": "object", "properties": {
            "camera": {"type": "string", "pattern": "^camera(\\.[a-z0-9_]+)*$", "default": "camera"},
            "source": {"type": "string", "enum": ["live", "episode"], "default": "live"},
            "run_id": {"type": "string", "maxLength": 128, "default": ""},
            "seconds": {"type": "number", "minimum": 2, "maximum": 10, "default": 6},
            "max_side": {"type": "integer", "minimum": 64, "maximum": 448, "default": 320},
        }, "additionalProperties": False},
        effect=EFFECT_READ, motion_allowed=False, bounded_planning=True,
        requires=("role:{camera}", "share:frames"), evidence_media="video",
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        if (args["source"] == "episode") != bool(args["run_id"]):
            raise SkillError(FAILURE_INVALID_ARGS, "run_id is required only for source=episode")
        return SkillPlan(skill_id=self.spec.skill_id, args=args,
                         summary=f"Read {args['seconds']:g}s {args['source']} clip from {args['camera']}")

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        args = plan.args
        clip = await ctx.clips.capture(args["camera"], source=args["source"],
            run_id=args["run_id"], seconds=args["seconds"], max_side=args["max_side"])
        return SkillOutcome.succeeded(clip, side_effect_state=SIDE_EFFECT_NONE,
                                      message="Passive temporal evidence; no motion commanded")


def skills() -> tuple[Clip, ...]:
    return (Clip(),)

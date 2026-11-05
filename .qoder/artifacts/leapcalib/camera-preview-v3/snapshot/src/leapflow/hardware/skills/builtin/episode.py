# Copyright (c) Alibaba, Inc. and its affiliates.
"""Demonstration capture: record a teleoperated episode without commanding anything.

``episode.record`` turns a leader-follower (master-slave) teleoperation session
into training data. The run commands nothing -- its motion facade is read-only,
so a dispatch is refused -- but it is a *leased* run: it holds the workcell's
devices for its whole duration, so no other run and no wire write can move the
arm while the demonstration is being recorded, and a lost host ends it like
any run. The episode recorder samples state and frames as for every run; the
demonstration's actions are the follower's observed motion (the episode's
``commanded`` is false), which the dataset converter turns into action targets.

Approval is per run: the operator confirms they are about to teleoperate. The
run ends when ``duration_s`` elapses or when it is stopped; either way the
episode is written, and S2 (or the operator) labels it with ``hw_skill_label``.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    CONSENT_EACH_RUN,
    EFFECT_READ,
    FAILURE_INVALID_ARGS,
    FAILURE_SKILL_UNAVAILABLE,
    REQUIRE_ROLE,
    SIDE_EFFECT_NONE,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

START_MARGIN_S = 5.0
"""Run-deadline room for the recorder to start (directories, camera discovery)."""


class RecordEpisode:
    """Record a teleoperated demonstration while holding the workcell's devices."""

    spec = SkillSpec(
        skill_id="episode.record",
        version="1.0.0",
        card=(
            "Record a teleoperated demonstration as a training episode. The node commands "
            "nothing: an operator moves the arm (e.g. with a leader arm) while the node samples "
            "state and camera frames and holds the workcell's devices, so no other run or write "
            "can move it meanwhile. 'task' is the demonstration's instruction, kept as the "
            "episode's language label. The run lasts duration_s or until hw_skill_stop (start "
            "it with wait=false); label the finished episode with hw_skill_label. Unavailable "
            "when the node does not record episodes."
        ),
        args_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "The instruction the demonstration performs.",
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 1,
                    "default": 60.0,
                    "description": "Longest recording; stop the run earlier when done.",
                },
            },
            "required": ["task"],
        },
        effect=EFFECT_READ,
        leased=True,
        requires=(f"{REQUIRE_ROLE}arm",),
        consent=CONSENT_EACH_RUN,
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        if not ctx.recording:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "this node does not record episodes (its operator sets the episode directory)",
            )
        duration = float(args["duration_s"])
        cap = float(ctx.limits["max_duration_s"])
        if duration + START_MARGIN_S > cap:
            raise SkillError(
                FAILURE_INVALID_ARGS,
                f"duration_s {duration:g} plus {START_MARGIN_S:g} s of start-up exceeds the run "
                f"cap of {cap:g} s; ask for at most {max(0.0, cap - START_MARGIN_S):g} s",
            )
        return SkillPlan(
            skill_id=self.spec.skill_id,
            args=args,
            summary=(
                f"Record a teleoperated demonstration of {args['task']!r} for up to "
                f"{duration:.0f} s. The node commands nothing; it holds the workcell's devices "
                "(refusing other runs and writes) while the operator moves the arm."
            ),
            limits={"max_duration_s": duration + START_MARGIN_S},
        )

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        if not ctx.recording:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "the episode could not be recorded (see the run's episode evidence)",
            )
        duration = float(plan.args["duration_s"])
        await ctx.motion.pause(duration)
        return SkillOutcome.succeeded(
            {"recorded_s": duration},
            side_effect_state=SIDE_EFFECT_NONE,
            message=f"Recorded {duration:.0f} s of {plan.args['task']!r}",
        )


def skills() -> tuple[RecordEpisode, ...]:
    """Entry point for ``leapflow.hardware.skills``."""
    return (RecordEpisode(),)


__all__ = ["START_MARGIN_S", "RecordEpisode", "skills"]

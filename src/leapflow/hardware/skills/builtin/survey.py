# Copyright (c) Alibaba, Inc. and its affiliates.
"""An explicitly governed finite observation skill; never commands the gripper."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import EFFECT_ACTUATE, SkillOutcome, SkillPlan, SkillSpec


class Survey:
    spec = SkillSpec(
        skill_id='perceive.survey', version='1.0.0', effect=EFFECT_ACTUATE,
        bounded_planning=True,
        requires=('role:arm', 'calib:tcp', 'calib:workspace'),
        card='Measure one current object id without grounding or sharing images. mode=passive '
             'captures calibrated cameras without motion; observe uses a declared yielding '
             'joint posture; active uses a declared moving camera and ViewPlanner proposals. '
             'Motion requires verified kinematics, declared body envelopes and obstacle-free '
             'volume. Finite views, travel, speed and joint waypoints are bounded by the '
             'workcell. Execution captures fresh depth after settling and stops early on '
             'adequate observation. grasp optionally sets a task-relative tolerance. '
             'Unknown dimensions or exhausted views return observation_insufficient. '
             'Priors affect ordering only within their exact configuration scope. '
             'No gripper command, automatic return, retries or image disclosure.',
        args_schema={'type': 'object', 'properties': {
            'object': {'type': 'string', 'minLength': 1, 'maxLength': 16},
            'grasp': {'type': 'string', 'minLength': 1, 'maxLength': 16},
            'mode': {'type': 'string', 'enum': ['passive', 'observe', 'active'], 'default': 'passive'},
            'prior_scope': {'type': 'string', 'maxLength': 64},
            'priors': {'type': 'object', 'maxProperties': 64,
                                   'additionalProperties': {'type': 'number', 'minimum': -1, 'maximum': 1}},
        }, 'required': ['object']},
    )

    async def plan(self, ctx: SkillContext, args: Mapping[str, Any]) -> SkillPlan:
        return await ctx.survey.plan(args)

    async def execute(self, ctx: SkillContext, plan: SkillPlan) -> SkillOutcome:
        return await ctx.survey.execute(plan)


def skills() -> tuple[Survey, ...]:
    return (Survey(),)

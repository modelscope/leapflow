# Copyright (c) Alibaba, Inc. and its affiliates.
"""Task-relative fresh observation gate shared by manipulation entry points."""
from __future__ import annotations

from leapflow.hardware.skills.ports import SkillContext
from leapflow.hardware.skills.protocol import (
    FAILURE_OBSERVATION_INSUFFICIENT, GraspCandidate, LocatedObject, SkillError,
)


def require_observation(ctx: SkillContext, obj: LocatedObject, grasp: GraspCandidate) -> None:
    """Legacy cells remain unchanged; enabled cells cannot authorize unknown extent."""
    if not getattr(ctx.workcell, 'observation', {}).get('enabled', False):
        return
    report = dict(ctx.scene.observation(obj, grasp))
    if not report.get('sufficient', False):
        raise SkillError(
            FAILURE_OBSERVATION_INSUFFICIENT,
            'Fresh measurement does not support this grasp; inspect observation.reasons',
            result={'object': obj.to_dict(), 'grasp': grasp.grasp_id, 'observation': report,
                    'next': ['perceive.survey', 'perceive.snapshot with_cameras=[all] (sharing consent required)']},
        )

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node skills -- the S1.5 layer between the LLM (S2) and device servo (S1).

``protocol`` (value types) and ``ports`` (facade protocols) are the contract
every skill implements; with ``geometry`` they are the only LeapFlow modules a
skill may import. ``digest`` binds approvals to plans. ``workcell`` declares
which devices form one manipulation cell, ``tuning`` its declared thresholds,
and ``frames`` resolves its calibrated frames; ``motion`` is the governed
motion facade; ``runtime`` owns discovery, the plan cache, runs, and stop.
Built-in skills live in ``builtin/`` and register through
the ``leapflow.hardware.skills`` entry-point group like any third-party pack.

This package re-exports only the skill-author surface so importing it never
pulls in the runtime.
"""

from __future__ import annotations

from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.geometry import Pose, Transform, WorkspaceBox
from leapflow.hardware.skills.ports import NodeSkill, SkillContext
from leapflow.hardware.skills.protocol import (
    EFFECT_ACTUATE,
    EFFECT_READ,
    SkillError,
    SkillOutcome,
    SkillPlan,
    SkillSpec,
)

__all__ = [
    "EFFECT_ACTUATE",
    "EFFECT_READ",
    "NodeSkill",
    "Pose",
    "SkillContext",
    "SkillError",
    "SkillOutcome",
    "SkillPlan",
    "SkillSpec",
    "Transform",
    "WorkspaceBox",
    "plan_digest",
    "run_permit_digest",
]

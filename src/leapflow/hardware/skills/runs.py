# Copyright (c) Alibaba, Inc. and its affiliates.
"""One skill run's live record and the rules that turn its end into a verdict.

``SkillRun`` is what the runtime keeps per execution. ``merge_outcome``
reconciles the skill's own word with what physically happened (and never
downgrades it); ``needs_halt`` decides whether the run's end must stop its
actuators.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from leapflow.hardware.skills.motion import GovernedMotion
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_FAULT,
    FAILURE_INTERNAL_DEFECT,
    FAILURE_STOPPED,
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    STATUS_HALTED,
    SkillOutcome,
    SkillPlan,
)
from leapflow.hardware.transport import worst_side_effect_state

RUN_RUNNING = "running"


@dataclass
class SkillRun:
    """One execution of one approved plan. Mutable: it is the run's live record."""

    run_id: str
    plan: SkillPlan
    devices: tuple[str, ...]  # held exclusively for the run
    actuators: tuple[str, ...] = ()  # halted when the run stops or fails
    intent: str = ""  # the host's sub-goal, recorded with the episode
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    outcome: SkillOutcome | None = None
    stop_reason: str = ""
    stop_code: str = FAILURE_STOPPED
    halted: bool = False
    halt_mark: int = 0  # batches landed when the run was halted
    motion: GovernedMotion | None = field(default=None, repr=False)
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.outcome is None

    @property
    def landed_after_halt(self) -> bool:
        """Whether a write reached a transport after the run's halt."""
        return self.halted and self.motion is not None and self.motion.landed > self.halt_mark

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workcell": self.plan.workcell,
            "skill_id": self.plan.skill_id,
            "version": self.plan.version,
            "plan_digest": self.plan.digest,
            "summary": self.plan.summary,
            "intent": self.intent,
            "status": RUN_RUNNING if self.outcome is None else self.outcome.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (self.finished_at or time.time()) - self.started_at,
            "stop_reason": self.stop_reason,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }


def merge_outcome(
    outcome: SkillOutcome, *, motion: GovernedMotion, committed: bool, evidence: Mapping[str, Any]
) -> SkillOutcome:
    """Reconcile the skill's verdict with what physically happened; never downgrade it.

    What happened: an unresolved dispatch is UNKNOWN; a landed batch or a
    persisted record is at least COMMITTED, and PARTIAL when a failed skill
    claimed nothing happened; otherwise NONE. The result is the more severe of
    that and the skill's own word (a failed check that completed every motion
    may say COMMITTED).
    """
    effected = motion.landed > 0 or committed
    if motion.uncertain:
        observed = SIDE_EFFECT_UNKNOWN
    elif not effected:
        observed = SIDE_EFFECT_NONE
    elif outcome.ok or outcome.side_effect_state != SIDE_EFFECT_NONE:
        observed = SIDE_EFFECT_COMMITTED
    else:
        observed = SIDE_EFFECT_PARTIAL
    state = worst_side_effect_state((outcome.side_effect_state, observed))
    return replace(
        outcome, side_effect_state=state, evidence={**dict(outcome.evidence), **evidence}
    )


def needs_halt(outcome: SkillOutcome, sent_any: bool) -> bool:
    """Halt after anything (may have) moved and the run did not end cleanly."""
    if not sent_any or outcome.ok:
        return False
    return (
        outcome.status == STATUS_HALTED
        or outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
        or outcome.failure_code in {FAILURE_DEVICE_FAULT, FAILURE_INTERNAL_DEFECT}
    )


__all__ = ["RUN_RUNNING", "SkillRun", "merge_outcome", "needs_halt"]

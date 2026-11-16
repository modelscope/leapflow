# Copyright (c) Alibaba, Inc. and its affiliates.
"""Content digests that bind an approval to exactly one skill plan.

The node seals each plan with ``plan_digest``; the host recomputes it from the
wire form (``verify_wire_plan``) before it shows the plan to a human, and both
sides derive the run-scoped permit digest with ``run_permit_digest``. Keeping
them in one module guarantees host and node hash the same canonical bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from leapflow.hardware.skills.protocol import (
    SkillPlan,
)

PLAN_DIGEST_FIELDS = (
    "workcell",
    "skill_id",
    "version",
    "args",
    "summary",
    "targets",
    "limits",
    "basis",
)
"""The plan fields the digest covers, in ``SkillPlan.digest_payload`` order."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def plan_digest(plan: SkillPlan) -> str:
    """Content digest of everything an approver is shown and the node will do."""
    return hashlib.sha256(_canonical(plan.digest_payload())).hexdigest()


def wire_plan_digest(plan: Mapping[str, Any]) -> str:
    """``plan_digest`` recomputed from a plan as it crossed the wire (``to_dict``).

    Canonical JSON makes tuples and lists, and a float before and after a JSON
    round trip, digest identically -- so the host can prove that the summary it
    shows a human is exactly the content the node will execute.
    """
    payload = {name: plan.get(name) for name in PLAN_DIGEST_FIELDS}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def verify_wire_plan(plan: Mapping[str, Any], permit_digest: str) -> str:
    """Why a planned run must not be approved, or ``""`` when its digests hold.

    Checks that the plan's ``digest`` covers its displayed content and that the
    permit digest binds that plan, its skill version, workcell, and limits.
    """
    digest = str(plan.get("digest") or "")
    if not digest or wire_plan_digest(plan) != digest:
        return "the plan digest does not cover the plan content"
    limits = plan.get("limits")
    expected = run_permit_digest(
        workcell=str(plan.get("workcell") or ""),
        skill_id=str(plan.get("skill_id") or ""),
        version=str(plan.get("version") or ""),
        plan_digest=digest,
        limits=limits if isinstance(limits, Mapping) else {},
    )
    if permit_digest != expected:
        return "the permit digest does not bind this plan"
    return ""


def run_permit_digest(
    *,
    workcell: str,
    skill_id: str,
    version: str,
    plan_digest: str,
    limits: Mapping[str, float],
) -> str:
    """The action digest a run-scoped execution permit binds.

    Host and node compute it from the same fields with this one function, so a
    permit minted for one plan can never authorise another plan, skill version,
    workcell, or looser limits.
    """
    payload = {
        "kind": "skill.run",
        "workcell": workcell,
        "skill": f"{skill_id}@{version}",
        "plan_digest": plan_digest,
        "limits": dict(limits),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = [
    "PLAN_DIGEST_FIELDS",
    "plan_digest",
    "run_permit_digest",
    "verify_wire_plan",
    "wire_plan_digest",
]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""The runtime's catalog view: frame snapshots, readiness listings, and refusals.

``SkillRuntime`` delegates here everything that turns one calibration snapshot
into what a host reads: each skill marked available or why not (with the
shortest no-guess way to ready it), every frame's typed diagnosis, and the
refusal a plan carries when its requirements do not hold. Keeping it apart
lets the runtime stay the single owner of run and dispatch ordering.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from leapflow.hardware.skills.frames import (
    FrameGraph,
    bind_requirements,
    required_records,
    requirement_bindings,
    unmet_requirements,
)
from leapflow.hardware.skills.ports import NodeSkill
from leapflow.hardware.skills.protocol import SkillSpec
from leapflow.hardware.skills.readiness import WorkcellReadiness
from leapflow.hardware.skills.workcell import WorkcellSpec

logger = logging.getLogger(__name__)


def snapshot_frames(registry: Any, spec: WorkcellSpec, skills: Iterable[NodeSkill]) -> FrameGraph:
    """One snapshot of *spec*'s frames, the records skills require or provide, and candidates."""
    catalog = tuple(skills)
    store = getattr(registry, "calibration_store", None)
    tokens = [token for skill in catalog for token in (*skill.spec.requires, *skill.spec.provides)]
    records = tuple(
        record for record in required_records(tokens) if "{" not in record
    )  # a templated record names no procedure until a run binds it
    return FrameGraph.snapshot(spec, store, records, candidates=_candidates(registry, spec))


def _candidates(registry: Any, spec: WorkcellSpec) -> Mapping[str, Sequence[Any]]:
    try:
        from leapflow.hardware.calibration.frame_lifecycle import admissible_candidates
    except ImportError as exc:  # the calibration stack is optional on a node
        logger.debug("frame candidates unavailable: %s", exc)
        return {}
    return admissible_candidates(registry, spec)


def cell_listing(
    skills: Mapping[str, NodeSkill],
    offered: Sequence[str],
    spec: WorkcellSpec,
    frames: FrameGraph,
    reach: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """*spec* as a host lists it: its skills' availability and its frames' diagnoses."""
    readiness = WorkcellReadiness([skills[skill_id] for skill_id in offered], spec, frames)
    listed = []
    for skill_id in offered:
        skill_spec = skills[skill_id].spec
        reasons, bindings = requirement_bindings(
            skill_spec.requires, skill_spec.args_schema, spec, frames, readiness.hint
        )
        entry: dict[str, Any] = {
            **skill_spec.to_dict(), "available": not reasons,
            "reason": "; ".join(reasons), "bindings": bindings,
        }
        if reasons:
            explained = readiness.explain(skill_spec)
            entry["readiness"] = explained.to_dict()
            if explained.next_steps:
                entry["next"] = explained.compact()
        listed.append(entry)
    return {
        **spec.to_dict(),
        "calibration": dict(frames.versions),
        "calibration_stale": dict(frames.stale),
        "calibration_status": {name: d.to_dict() for name, d in frames.diagnoses.items()},
        "skills": listed,
        "reach": dict(reach or {}),
    }


def plan_refusal(
    offered: Iterable[NodeSkill],
    spec: WorkcellSpec,
    frames: FrameGraph,
    skill_spec: SkillSpec,
    arguments: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Why *skill_spec* cannot plan under *arguments*, and its readiness when it cannot.

    Only *offered* skills -- those the workcell admits -- are proposed as remedies.
    """
    readiness = WorkcellReadiness(offered, spec, frames)
    bound = dict(arguments)
    if bound.get("camera") == "auto":
        # Auto is acquisition policy, not an exemption from exact frame gates.
        choices = [role for role in sorted(spec.roles) if role.split(".")[0] == "camera"]
        camera_requires = tuple(token for token in skill_spec.requires if "{camera}" in token)
        usable = [role for role in choices if not unmet_requirements(
            bind_requirements(camera_requires, {**bound, "camera": role}), spec, frames)]
        if usable:
            bound["camera"] = usable[0]
    requires = bind_requirements(skill_spec.requires, bound)
    reasons = unmet_requirements(requires, spec, frames, readiness.hint)
    if not reasons:
        return [], {}
    return reasons, {"readiness": readiness.explain(skill_spec, arguments).to_dict()}


__all__ = ["cell_listing", "plan_refusal", "snapshot_frames"]

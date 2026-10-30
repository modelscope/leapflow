# Copyright (c) Alibaba, Inc. and its affiliates.
"""Idle, digest-bound workcell replacement without recreating the node owner.

Preparation and commit are synchronous: wire authority is consumed between them
without yielding to competing runs. Disk edits are never performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from leapflow.hardware.lhp.permit import request_digest
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_BUSY, FAILURE_PLAN_STALE, FAILURE_SKILL_UNAVAILABLE,
    FAILURE_UNKNOWN_WORKCELL, SkillOutcome,
)
from leapflow.hardware.skills.workcell import WorkcellLoadReport, WorkcellSpec, load_workcells

if TYPE_CHECKING:
    from leapflow.hardware.skills.runtime import SkillRuntime


def declaration_payload(spec: WorkcellSpec) -> dict[str, Any]:
    """Include fields omitted by the legacy display payload but used by admission."""
    return {**spec.to_dict(), "skills": list(spec.skills),
            "bootstrap_box": spec.bootstrap_box.to_dict() if spec.bootstrap_box else None}


def workcell_digest(spec: WorkcellSpec) -> str:
    """Bind all operational declaration fields, including offer and bootstrap limits."""
    return request_digest("workcell.declaration", declaration_payload(spec))


def load(runtime: SkillRuntime) -> tuple[dict[str, WorkcellSpec], WorkcellLoadReport]:
    directory = runtime._settings.workcells_dir
    if directory is None:
        return {}, WorkcellLoadReport()
    return load_workcells(directory, runtime._registry, checks=(
        runtime._grasp_problems, runtime._perception_problems,
        runtime._vision_problems, runtime._policy_problems,
    ))


@dataclass(frozen=True)
class PreparedReload:
    workcell: str
    previous: WorkcellSpec
    candidate: WorkcellSpec
    previous_digest: str
    candidate_digest: str
    node_session_id: str
    devices: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"workcell": self.workcell, "previous_digest": self.previous_digest,
                "workcell_digest": self.candidate_digest, "node_session_id": self.node_session_id,
                "candidate": declaration_payload(self.candidate), "devices": list(self.devices),
                "disk_written": False, "motion": False}


def prepare(runtime: SkillRuntime, workcell: str) -> PreparedReload | SkillOutcome:
    if runtime._closing:
        return SkillOutcome.failed(FAILURE_SKILL_UNAVAILABLE, "node skill runtime is shutting down")
    previous = runtime._workcells.get(workcell)
    if previous is None:
        return SkillOutcome.failed(FAILURE_UNKNOWN_WORKCELL, "reload requires an existing admitted workcell")
    admitted, report = load(runtime)
    # Malformed or duplicate declarations cannot be silently ignored at reload.
    if report.rejected:
        return SkillOutcome.failed(FAILURE_SKILL_UNAVAILABLE, "workcell reload refused: " +
                                   "; ".join(note.detail for note in report.rejected))
    candidate = admitted.get(workcell)
    if candidate is None:
        return SkillOutcome.failed(FAILURE_UNKNOWN_WORKCELL, "requested workcell is absent from disk")
    devices = tuple(sorted(set(previous.device_ids) | set(candidate.device_ids)))
    if any(run.running for run in runtime._runs.values()) or runtime._held:
        return SkillOutcome.failed(FAILURE_DEVICE_BUSY, "reload requires all skill runs and unsettled holds to be idle")
    sessions = getattr(runtime._registry, "node_sessions", None)
    if sessions is not None and sessions.busy_device(devices):
        return SkillOutcome.failed(FAILURE_DEVICE_BUSY, "reload devices are held by a session or transaction")
    return PreparedReload(workcell, previous, candidate, workcell_digest(previous),
                          workcell_digest(candidate), runtime._node_session_id, devices)


def matches(prepared: PreparedReload, *, previous_digest: str, expected_digest: str,
            node_session_id: str) -> SkillOutcome | None:
    if (prepared.previous_digest, prepared.candidate_digest, prepared.node_session_id) != (
        previous_digest, expected_digest, node_session_id
    ):
        return SkillOutcome.failed(FAILURE_PLAN_STALE, "workcell or owner session changed; preview reload again")
    return None


def commit(runtime: SkillRuntime, prepared: PreparedReload) -> dict[str, Any]:
    """Apply only the prepared workcell; caller must check authority without an await."""
    if runtime._workcells.get(prepared.workcell) is not prepared.previous:
        raise RuntimeError("reload preparation was not committed synchronously")
    invalidated = [key for key, (plan, _) in runtime._plans.items() if plan.workcell == prepared.workcell]
    for key in invalidated:
        del runtime._plans[key]
    runtime._scenes.pop(prepared.workcell, None)
    runtime._workcells[prepared.workcell] = prepared.candidate
    runtime._report = WorkcellLoadReport(tuple(runtime._workcells), runtime._report.rejected)
    return {**prepared.to_dict(), "applied": True, "invalidated_plans": len(invalidated),
            "node_session_unchanged": runtime._node_session_id == prepared.node_session_id}

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Owner-side plan sealing and run feasibility; receipts never grant authority."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

from leapflow.hardware.skills.digest import plan_digest
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_BUSY,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_SKILL_UNAVAILABLE,
    SkillError,
    SkillOutcome,
    SkillPlan,
)

if TYPE_CHECKING:
    from leapflow.hardware.skills.frames import FrameGraph
    from leapflow.hardware.skills.ports import NodeSkill
    from leapflow.hardware.skills.runtime import SkillRuntime
    from leapflow.hardware.skills.workcell import WorkcellSpec


@dataclass(frozen=True)
class RunAdmission:
    """Receipt constructed after the existing write authority admits the exact plan.

    Legacy trusted-mode receipts have no signed approval provenance. The proof is
    private evidence, not a bearer token, and requires independent verification
    before it can support physical admission. This type owns no approval policy.
    """
    mode: str
    action_digest: str
    permit_json: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("permit_required", "trusted_v1") or not isinstance(self.action_digest, str) or not self.action_digest:
            raise ValueError("admission requires the executing node's exact gate and digest")
        if not isinstance(self.permit_json, str) or len(self.permit_json.encode()) > 8192:
            raise ValueError("bounded private admission proof required")
        if self.mode == "permit_required":
            permit = json.loads(self.permit_json)
            if not isinstance(permit, dict) or permit.get("action_digest") != self.action_digest or not permit.get("signature"):
                raise ValueError("admission proof must bind the independently computed action digest")
        elif self.permit_json:
            raise ValueError("legacy admission cannot claim signed permit provenance")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "action_digest": self.action_digest,
                "permit": json.loads(self.permit_json) if self.permit_json else None}


def effective_limits(
    runtime: SkillRuntime, spec: WorkcellSpec, requested: Mapping[str, float],
) -> dict[str, float]:
    """Intersect requested limits with declaration and node caps; never raise a cap."""
    cap = min(float(spec.limits["speed_pct_max"]), runtime._settings.speed_pct_max)
    speed = min(cap, float(requested.get("speed_pct", spec.limits["default_speed_pct"])))
    duration = min(
        runtime._settings.max_duration_s,
        float(requested.get("max_duration_s", runtime._settings.max_duration_s)),
    )
    if not speed > 0.0 or not duration > 0.0:
        raise SkillError(
            FAILURE_INVALID_ARGS,
            f"speed {speed:g} % and duration {duration:g} s must both be positive",
        )
    return {
        "speed_pct": speed, "speed_pct_max": cap,
        **{name: float(spec.limits[name]) for name in (
            "move_timeout_s", "fk_tolerance_mm", "fk_tolerance_deg", "joint_path_step_deg",
            "max_steps", "max_wait_s",
        )},
        "max_duration_s": duration,
    }


async def generation_basis(registry: Any, spec: WorkcellSpec) -> Mapping[str, Any]:
    """Read the committed structural generation off-loop, without creating schema."""
    from leapcalib.contracts.identity import digest_of
    from leapflow.hardware.calibration.generations import GenerationState

    store = getattr(registry, "calibration_generations", None)
    try:
        state = GenerationState(spec.workcell_id) if store is None else await asyncio.to_thread(store.state, spec.workcell_id)
    except Exception as exc:  # noqa: BLE001 - persistence failure cannot authorize a run
        raise SkillError(
            FAILURE_PLAN_STALE, f"committed physical model snapshot unavailable: {exc}"
        ) from exc
    return {**state.basis(), "workcell_digest": digest_of(spec.to_dict())}


def ensure_model_use(registry: Any, spec: WorkcellSpec, basis: Mapping[str, Any]) -> None:
    """Actuating feasibility only; READ diagnostics bypass this stop-use check."""
    from leapflow.hardware.node_gate import recovery_safety_refusal

    if basis.get("invalidation_digest"):
        raise SkillError(FAILURE_SKILL_UNAVAILABLE, "physical correction invalidated; inspect and requalify dependencies before another actuating run")
    for device_id in spec.actuator_ids:
        if reason := recovery_safety_refusal(registry, device_id):
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, reason)


async def execution_basis(runtime: SkillRuntime, plan: SkillPlan) -> Mapping[str, Any] | SkillOutcome:
    """Prepare before permit consumption; prevent transactions racing snapshot I/O.

    A temporary reservation spans the asynchronous read. Release and return do not
    suspend, so the caller can consume authority and adopt the prepared run on the
    same owner-loop turn. This is a feasibility lock, never an approval exemption.
    """
    spec = runtime._workcells.get(plan.workcell)
    if spec is None or runtime.cached_plan(plan.digest) is not plan:
        return SkillOutcome.failed(FAILURE_PLAN_UNKNOWN, "plan is no longer cached on this owner")
    sessions = getattr(runtime._registry, "node_sessions", None)
    token = None
    try:
        if sessions is not None:
            try:
                token = sessions.reserve(spec.device_ids, owner_id=plan.digest)
            except ValueError as exc:
                raise SkillError(FAILURE_DEVICE_BUSY, str(exc)) from exc
        basis = await generation_basis(runtime._registry, spec)
        if runtime._closing:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "node skill runtime is shutting down")
        skill = runtime._skills.get(plan.skill_id)
        if runtime._workcells.get(plan.workcell) is not spec or skill is None or skill.spec.version != plan.version:
            raise SkillError(FAILURE_PLAN_STALE, "the skill or workcell changed")
        if runtime.cached_plan(plan.digest) is not plan:
            raise SkillError(FAILURE_PLAN_UNKNOWN, "plan expired or was executed while checking its model")
        if basis != plan.basis.get("physical_model"):
            raise SkillError(FAILURE_PLAN_STALE, "physical model generation changed since planning; plan again")
        if skill.spec.uses_motion:
            ensure_model_use(runtime._registry, spec, basis)
        if dict(runtime._frames(spec).versions) != dict(plan.basis.get("calibration") or {}):
            raise SkillError(FAILURE_PLAN_STALE, "calibration changed since planning; plan again")
        if busy := [device for device in spec.device_ids if runtime.holds_device(device)]:
            raise SkillError(FAILURE_DEVICE_BUSY, f"{', '.join(busy)} held by another run")
        return basis
    except SkillError as error:
        return SkillOutcome.from_error(error)
    finally:
        if token is not None:
            sessions.release_reservation(token)


def seal_plan(
    runtime: SkillRuntime, spec: WorkcellSpec, skill: NodeSkill,
    planned: SkillPlan, frames: FrameGraph, *, model_basis: Mapping[str, Any],
) -> SkillPlan:
    """Seal the exact owner identities/dependencies before caching the intention."""
    sealed = replace(
        planned, skill_id=skill.spec.skill_id, workcell=spec.workcell_id,
        version=skill.spec.version, limits=runtime._effective_limits(spec, planned.limits),
        basis={**dict(planned.basis), "calibration": dict(frames.versions), "physical_model": dict(model_basis)},
        digest="",
    )
    sealed = replace(sealed, digest=plan_digest(sealed))
    runtime._evict_expired()
    runtime._plans[sealed.digest] = (sealed, runtime._clock())
    return sealed

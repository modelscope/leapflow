# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed stationary bridge facade over existing acquisition/store/compute owners.

The allocation is sealed before acquisition. Rejected target observations retain
raw evidence; neither fitting nor a numerical pass writes an active frame record.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Mapping

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.estimation.bridge import BridgePair, METHOD, SCOPE
from leapflow.hardware.calibration.bridge_policy import BridgePolicy, wrist_dependency
from leapflow.hardware.calibration.measurements import metric_roi, si_transform
from leapflow.hardware.calibration.paired import PairedObservation, PairedObservationSource
from leapflow.hardware.skills.geometry import Transform
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED, FAILURE_INVALID_ARGS, FAILURE_PLAN_STALE, FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED, FAILURE_TIMEOUT, SkillError, SkillPlan,
)
from leapflow.hardware.skills.vision import CharucoTarget
from leapflow.hardware.vision.group import CaptureGroup

SCHEMA = "leapflow.bridge_campaign.v1"


class WorkcellBridge:
    def __init__(self, collection: Any) -> None:
        self._collection = collection
        self._ctx, self._registry = collection._ctx, collection._registry

    def _policy(self) -> BridgePolicy:
        try:
            return BridgePolicy.from_mapping(self._ctx.workcell.calibration_bridge)
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, f"declare calibration_bridge: {exc}") from exc

    async def _dependency(self, policy: BridgePolicy) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(wrist_dependency, self._registry, self._ctx.workcell, policy,
                                           node_session_id=self._collection._node_session)
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"wrist dependency unresolved: {exc}") from exc

    async def basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        policy, acquisition = self._policy(), self._collection._profile()
        worker = self._collection._worker()
        if set(args) != {"fixed_region", "wrist_region"}:
            raise SkillError(FAILURE_INVALID_ARGS, "bridge accepts only fixed_region and wrist_region; allocation is declared")
        count = policy.training + policy.holdout
        duration = (count * (acquisition.settled.timeout_s + policy.budget.separation_s)
                    + acquisition.settled.timeout_s + worker.budgets.solve_s + worker.budgets.startup_s)
        if duration > min(120, self._ctx.limits["max_duration_s"]):
            raise SkillError(FAILURE_INVALID_ARGS, "paired acquisition and solve cannot fit frozen run deadline")
        from leapflow.hardware.calibration.evidence import MAX_RAW_BYTES
        if acquisition.ttl_s < duration or acquisition.quota_bytes < count * MAX_RAW_BYTES:
            raise SkillError(FAILURE_INVALID_ARGS,
                             "declared private TTL/quota cannot retain every bounded pair through indexed evaluation; "
                             "preview an explicitly approved retention declaration before acquisition")
        if self._collection._blocked() or self._registry.calibration_evidence is None or self._registry.evidence_cache is None:
            raise SkillError(FAILURE_CHECK_FAILED, "paired acquisition blocked or private indexed storage unavailable")
        dependency = await self._dependency(policy)
        try:
            async with asyncio.timeout(acquisition.settled.timeout_s):
                feedback = await self._collection._feedback(acquisition).sample()
                group = await CaptureGroup({role: lambda bound, name=name: self._ctx.camera_vision(name).capture_metric(after_monotonic=bound)
                                           for role, name in (("fixed", policy.fixed_role), ("wrist", policy.wrist_role))},
                                          tolerance_ms=policy.tolerance_ms, attempts=policy.attempts).capture()
                if group.errors or set(group.frames) != {"fixed", "wrist"}:
                    raise ValueError("both declared cameras must be present")
                if digest_of(group.frames["wrist"].intrinsics.to_dict()) != dependency["intrinsics_digest"]:
                    raise ValueError("wrist intrinsics differ from the validated mount dependency")
                cameras = {}
                for role, frame in group.frames.items():
                    region = args[f"{role}_region"]
                    roi = await asyncio.to_thread(metric_roi, frame, region=region, role=role,
                                                  receipt_uncertainty_s=acquisition.settled.receipt_uncertainty_s,
                                                  exposure_uncertainty_s=acquisition.settled.exposure_uncertainty_s)
                    cameras[role] = {"camera_device_id": frame.device_id, "mount_digest": roi.mount_digest,
                                     "camera_session_id": roi.session_id, "source_profile_digest": roi.source_profile_digest,
                                     "metric_profile_digest": roi.profile.digest, "region": list(region),
                                     "source_frame_id": roi.frame_id, "source_stamps": [vars(s) for s in roi.source_stamps]}
                # Leave room for binding/correspondence JSON inside the unchanged 4MiB limit.
                if sum(c["region"][2] * c["region"][3] for c in cameras.values()) > 400000:
                    raise ValueError("paired lossless ROIs exceed the existing private 4MiB budget")
                fresh = policy.freshness
                if (cameras["wrist"]["mount_digest"], cameras["wrist"]["source_profile_digest"]) != (fresh["mount_digest"], fresh["source_profile_digest"]):
                    raise ValueError("live wrist identity disagrees with approved mounting freshness")
        except TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "paired bridge planning deadline expired") from exc
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"paired bridge basis refused: {exc}") from exc
        return {"schema": SCHEMA, "method_id": METHOD, "scope": SCOPE, "policy": policy.to_dict(),
                "acquisition": acquisition.to_dict(), "workcell_digest": digest_of(self._ctx.workcell.to_dict()),
                "model_digest": feedback.model_digest, "declaration_digest": feedback.declaration_digest,
                "joint_ids": list(feedback.joint_ids), "node_session_id": self._collection._node_session,
                "target": CharucoTarget.from_mapping(self._ctx.workcell.calibration_target).to_dict(),
                "target_id": policy.target_id, "stationary_epoch": policy.stationary_epoch,
                "min_pose_gap_px": policy.min_pose_gap_px, "dependency": dependency, "cameras": cameras,
                "worker_budgets": vars(worker.budgets), "reachability": policy.budget.reachability(),
                "allocation": ["training"] * policy.training + ["holdout"] * policy.holdout,
                "commands": [], "activation": "none"}

    async def execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        try:
            return await self._execute(plan)
        except TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "bridge deadline expired; retain indexed captures and inspect before another run") from exc
        except (CalibrationFailure, TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"bridge refused; no activation: {exc}") from exc

    async def _execute(self, plan: SkillPlan) -> Mapping[str, Any]:
        policy, acquisition = self._policy(), self._collection._profile()
        basis = plan.basis.get("bridge", {})
        worker, store = self._collection._worker(), self._registry.calibration_evidence
        if (basis.get("schema") != SCHEMA or basis.get("policy") != policy.to_dict()
                or basis.get("acquisition") != acquisition.to_dict() or basis.get("workcell_digest") != digest_of(self._ctx.workcell.to_dict())
                or basis.get("node_session_id") != self._collection._node_session or basis.get("worker_budgets") != vars(worker.budgets)
                or basis.get("allocation") != ["training"] * policy.training + ["holdout"] * policy.holdout):
            raise SkillError(FAILURE_PLAN_STALE, "bridge owner/declaration/allocation/budget changed")
        dependency = await self._dependency(policy)
        if dependency != basis.get("dependency"):
            raise SkillError(FAILURE_PLAN_STALE, "wrist dependency changed since approval")
        for role in ("fixed", "wrist"):
            if basis["cameras"][role]["region"] != list(plan.args[f"{role}_region"]):
                raise SkillError(FAILURE_PLAN_STALE, "paired region differs from preview")
        feedback = self._collection._feedback(acquisition)
        sample = await feedback.sample()
        if (sample.model_digest, sample.declaration_digest, list(sample.joint_ids)) != (basis["model_digest"], basis["declaration_digest"], basis["joint_ids"]):
            raise SkillError(FAILURE_PLAN_STALE, "bridge raw model/declaration changed")
        source = PairedObservationSource(feedback, {"fixed": self._ctx.camera_vision(policy.fixed_role),
                                                    "wrist": self._ctx.camera_vision(policy.wrist_role)},
                 profile=acquisition.settled, bindings=basis["cameras"], target=CharucoTarget.from_mapping(basis["target"]),
                 target_id=policy.target_id, stationary_epoch=policy.stationary_epoch, min_pose_gap_px=policy.min_pose_gap_px,
                 tolerance_ms=policy.tolerance_ms, attempts=policy.attempts, blocked=self._collection._blocked)
        recorded = self._ctx.recorded_observations(source, profile=acquisition.settled, ttl_s=acquisition.ttl_s, quota_bytes=acquisition.quota_bytes)
        captures, training, holdout, excluded = [], [], [], []
        for slot in basis["allocation"]:
            if self._collection._blocked():
                raise SkillError(FAILURE_STOPPED, "bridge stopped; inspect committed paired capture digests")
            if await self._dependency(policy) != dependency:
                raise SkillError(FAILURE_PLAN_STALE, "wrist dependency changed during acquisition")
            await recorded.capture()
            identity = recorded.last_record["evidence_digest"]
            captures.append({"evidence_digest": identity, "allocation": slot})
            # Reopen the owner index, not a caller-provided target pose or old preview.
            content = await store.read_capture(identity, workcell_id=plan.workcell)
            observation = PairedObservation.from_mapping(content["observation"])
            if observation.exclusions:
                excluded.append({"evidence_digest": identity, "allocation": slot, "reasons": list(observation.exclusions)})
            else:
                pair = await asyncio.to_thread(indexed_pair, observation, identity, basis, policy)
                (training if slot == "training" else holdout).append(pair.to_dict())
            await asyncio.sleep(policy.budget.separation_s)
        if self._collection._blocked() or await self._dependency(policy) != dependency:
            raise SkillError(FAILURE_STOPPED, "bridge authority/dependency changed before solve")
        result = await worker.submit("solve_bridge", {"training": training, "holdout": holdout,
                                    "flange_T_wrist": dependency["transform"], "dependency_digest": digest_of(dependency),
                                    "dependency_sources": dependency["source_ids"], "budget": policy.budget.to_dict(),
                                    "model_digest": basis["model_digest"]}, workcell_id=plan.workcell,
                                    run_id=self._ctx.run_id, model_digest=basis["model_digest"])
        if not isinstance(result, Mapping) or set(result) != {"bridge"}:
            raise ValueError("typed bridge worker result required")
        shadow = await store.verify_bridge_result(captures, result["bridge"], workcell_id=plan.workcell,
                                                  plan=plan.to_dict(), run_id=self._ctx.run_id)
        from leapflow.hardware.calibration.bridge_assessment import shadow_problem
        problem = shadow_problem(shadow)
        if problem:
            raise ValueError(problem)
        generations = self._registry.calibration_generations
        if generations is None:
            raise ValueError("immutable bridge shadow index unavailable")

        async def preflight() -> None:
            # model_transaction has already reserved every device for this exact
            # admitted run. _blocked() would mistake that own reservation for a
            # foreign effect; the coordinator refuses foreign/uncertain holders.
            await self._camera_preflight(basis, policy, acquisition, observation)
            current = await feedback.sample()
            if (self._ctx.cancelled or self._ctx.motion.uncertain or not current.enabled or not current.arrived
                    or current.arm_status != 0 or current.communication_errors or await self._dependency(policy) != dependency
                    or (current.model_digest, current.declaration_digest, list(current.joint_ids)) !=
                    (basis["model_digest"], basis["declaration_digest"], basis["joint_ids"])):
                raise SkillError(FAILURE_PLAN_STALE, "bridge authority/dependency changed before shadow persistence")

        def persist() -> str:
            identity = generations.record_candidate(workcell_id=plan.workcell, evidence_digest=shadow["evidence_digest"], candidate=shadow)
            if generations.candidate(identity, workcell_id=plan.workcell) != shadow:
                raise ValueError("immutable bridge shadow readback mismatch")
            return identity

        identity = await self._ctx.model_transaction(persist, publish=lambda value: self._ctx.note("bridge_shadow_digest", value), preflight=preflight)
        report = {**shadow["evaluation"], "campaign_schema": SCHEMA, "captures": captures,
                  "excluded_observations": excluded, "dependency": dependency,
                  "commands": [], "activation": "none", "raw_exported": False,
                  "hardware_accuracy_certified": False, "shadow_record_digest": identity,
                  "campaign_acceptance": "insufficient_evidence",
                  "acceptance_blockers": [dependency["source_disjointness_basis"], "no independent absolute-reference accuracy"]}
        if excluded:
            report["advisories"] = [*report.get("advisories", []), "excluded paired observations retained; no holdout reallocation"]
        # Attach to the existing durable run/episode reporting path. Raw acquisition
        # remains indexed privately; a run report is not a CalibrationRecord.
        self._ctx.note("stationary_board_bridge", report)
        return report

    async def _camera_preflight(self, basis: Mapping[str, Any], policy: BridgePolicy,
                                acquisition: Any, last: PairedObservation) -> None:
        """Check live owner/profile identity after compute, without exporting frames."""
        async with asyncio.timeout(acquisition.settled.timeout_s):
            group = await CaptureGroup({role: lambda bound, name=name: self._ctx.camera_vision(name).capture_metric(after_monotonic=bound)
                                       for role, name in (("fixed", policy.fixed_role), ("wrist", policy.wrist_role))},
                                      tolerance_ms=policy.tolerance_ms, attempts=policy.attempts).capture()
            if group.errors or set(group.frames) != {"fixed", "wrist"}:
                raise SkillError(FAILURE_PLAN_STALE, "paired camera unavailable after solve; shadow not persisted")
            for role, frame in group.frames.items():
                bound = basis["cameras"][role]
                roi = await asyncio.to_thread(metric_roi, frame, region=bound["region"], role=role,
                                              receipt_uncertainty_s=acquisition.settled.receipt_uncertainty_s,
                                              exposure_uncertainty_s=acquisition.settled.exposure_uncertainty_s)
                previous = getattr(last, role).roi
                if (not group.bound <= frame.monotonic_at <= time.monotonic() or
                        (frame.device_id, roi.mount_digest, roi.session_id, roi.source_profile_digest, roi.profile.digest) !=
                        (bound["camera_device_id"], bound["mount_digest"], bound["camera_session_id"],
                         bound["source_profile_digest"], bound["metric_profile_digest"]) or
                        roi.frame_id == previous.frame_id or len(roi.source_stamps) != len(previous.source_stamps) or
                        any((a.stream, a.timestamp_domain) != (b.stream, b.timestamp_domain) or
                            b.frame_number <= a.frame_number or b.timestamp_ms <= a.timestamp_ms
                            for a, b in zip(previous.source_stamps, roi.source_stamps))):
                    raise SkillError(FAILURE_PLAN_STALE, "paired camera identity/provenance changed after solve")


def indexed_pair(observation: PairedObservation, identity: str, basis: Mapping[str, Any], policy: BridgePolicy) -> BridgePair:
    """Reconstruct chain factors and enforce measured uncertainty lower bounds."""
    if observation.exclusions:
        raise ValueError("excluded paired observation is not a solve factor")
    first = observation.feedback[0]
    if (first.model_digest, first.declaration_digest, first.node_session_id) != (basis["model_digest"], basis["declaration_digest"], basis["node_session_id"]):
        raise ValueError("indexed paired feedback model/owner/declaration mismatch")
    before = observation.feedback[-2]
    wrist = si_transform(observation.wrist_detection.transform, parent="wrist", child="board")
    fixed = si_transform(observation.fixed_detection.transform, parent="fixed", child="board")
    mount = basis["dependency"]["transform"]
    lever = math.dist(mount["translation_m"], (0, 0, 0)) + math.dist(wrist.translation_m, (0, 0, 0)) + math.dist(fixed.translation_m, (0, 0, 0))
    robot_floor = (before.tcp_uncertainty_mm / 1000 + lever * math.radians(before.rotation_uncertainty_deg), math.radians(before.rotation_uncertainty_deg))
    if any(declared < measured for declared, measured in zip(policy.budget.terms["robot_fk"], robot_floor)):
        raise ValueError("bridge budget understates propagated raw nominal-flange uncertainty")
    dependency = basis["dependency"]
    wrist_floor = (dependency["translation_floor_m"] + lever * dependency["rotation_floor_rad"], dependency["rotation_floor_rad"])
    pnp_translation, pnp_rotation = 0.0, 0.0
    for view, detection in ((observation.fixed, observation.fixed_detection), (observation.wrist, observation.wrist_detection)):
        focal = min(view.roi.profile.fx_px, view.roi.profile.fy_px)
        distance = math.dist(detection.transform.translation, (0, 0, 0)) / 1000
        pixel_floor = detection.reprojection_px / focal
        points = detection.board_points_mm
        span = min(max(p[i] for p in points) - min(p[i] for p in points) for i in (0, 1)) / 1000
        angle = pixel_floor * distance / span
        pnp_translation += pixel_floor * distance + lever * angle
        pnp_rotation += angle
    # Observed stillness is conditional on qualified receipt cohorts, not a
    # hardware trigger or an independently measured inter-sample rate bound.
    timing_angle = max(math.radians(before.tcp.rotation_to_deg(sample.tcp)) for sample in observation.feedback)
    timing_translation = max(before.tcp.distance_mm(sample.tcp) / 1000 for sample in observation.feedback)
    floors = {"wrist_mount": wrist_floor, "pnp_depth": (pnp_translation, pnp_rotation),
              "timing": (timing_translation + lever * timing_angle, timing_angle)}
    for term, measured in floors.items():
        if any(declared < value for declared, value in zip(policy.budget.terms[term], measured)):
            raise ValueError(f"bridge {term} budget understates additive propagated evidence bounds")
    return BridgePair(identity, observation.source_ids, observation.interval_s,
                      si_transform(Transform.from_pose(before.tcp), parent="base", child="flange"), wrist, fixed)

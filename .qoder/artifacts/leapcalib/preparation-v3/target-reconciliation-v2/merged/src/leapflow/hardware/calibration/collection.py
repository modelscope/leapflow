# Copyright (c) Alibaba, Inc. and its affiliates.
"""Production stationary acquisition behind the run context and existing device owner.

This method records actual settled captures, not carried-link identity, independent
scene witnesses, a measured response, or structural qualification. No robot or
camera owner is created and no joint command is available on this facade.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.tracking import SourceStamp
from leapflow.hardware.calibration.acquisition import (
    NOMINAL_FLANGE,
    CollectionProfile,
    RegistryFeedbackSource,
    SettledObservationSource,
)
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.measurements import metric_roi
from leapflow.hardware.calibration.reference import (
    ReferenceProfile,
    assess_reference,
    validate_tracking_result,
)
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED,
    FAILURE_INVALID_ARGS,
    FAILURE_PLAN_STALE,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    FAILURE_TIMEOUT,
    SkillError,
    SkillPlan,
)
from leapflow.hardware.skills.workcell import (
    ARM_STATUS_CHANNEL,
    ARRIVED_CHANNEL,
    CTRL_MODE_CHANNEL,
    ENABLED_CHANNEL,
    POSE_CHANNELS,
    joint_channels,
)

SCHEMA = "leapflow.stationary_collection.v1"


class WorkcellCalibration:
    """Concrete CalibrationPort; configuration and identities are frozen into the plan."""

    def __init__(self, context: Any, registry: Any, *, node_session_id: str) -> None:
        self._ctx, self._registry, self._node_session = context, registry, node_session_id
        self._rollback: Any = None
        self._trial: Any = None
        self._poses: Any = None

    def _profile(self) -> CollectionProfile:
        try:
            return CollectionProfile.from_mapping(self._ctx.workcell.calibration_acquisition)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"declare calibration_acquisition before collection: {exc}",
            ) from exc

    def _feedback(self, profile: CollectionProfile) -> RegistryFeedbackSource:
        device = self._ctx.workcell.role("arm")
        arm = self._registry.context(device)
        try:
            chain = DhChain.from_mapping(arm.kinematics.dh) if arm and arm.kinematics else None
            if chain is None:
                raise ValueError("declared nominal DH chain required")
            return RegistryFeedbackSource(
                self._registry,
                device_id=device,
                model=model_identity(chain, model_id=device),
                joint_ids=joint_channels(arm),
                pose_ids=POSE_CHANNELS,
                health_ids=(
                    ENABLED_CHANNEL,
                    ARRIVED_CHANNEL,
                    CTRL_MODE_CHANNEL,
                    ARM_STATUS_CHANNEL,
                ),
                allowed_modes=profile.allowed_modes,
                profile=profile.settled,
                node_session_id=self._node_session,
                pose_semantics=NOMINAL_FLANGE,
            )
        except (TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE, f"settled raw feedback unavailable: {exc}"
            ) from exc

    def _blocked(self) -> bool:
        if self._ctx.cancelled or self._ctx.motion.uncertain:
            return True
        sessions = getattr(self._registry, "node_sessions", None)
        if sessions is None:
            return False
        # Exclude this run's own lease/hold, but never an overlapping dispatch,
        # reservation or unresolved previous action on any bound device.
        return any(sessions.unsettled_device(device) for device in self._ctx.workcell.device_ids)

    async def collection_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        profile = self._profile()
        try:
            async with asyncio.timeout(profile.settled.timeout_s):
                return await self._collection_basis(args, profile)
        except TimeoutError as exc:
            raise SkillError(
                FAILURE_TIMEOUT,
                "collection planning deadline expired; inspect feedback/camera readiness",
            ) from exc

    async def _collection_basis(
        self, args: Mapping[str, Any], profile: CollectionProfile
    ) -> Mapping[str, Any]:
        count, region = args.get("frames", 1), args.get("region")
        if type(count) is not int or not 1 <= count <= profile.max_frames:
            raise SkillError(
                FAILURE_INVALID_ARGS, "frame count exceeds the declared collection bound"
            )
        if count * profile.settled.timeout_s > min(120, self._ctx.limits["max_duration_s"]):
            raise SkillError(FAILURE_INVALID_ARGS, "collection cannot fit the frozen run deadline")
        if self._blocked():
            raise SkillError(
                FAILURE_CHECK_FAILED, "settled acquisition blocked by motion or uncertain dispatch"
            )
        if self._registry.calibration_evidence is None or self._registry.evidence_cache is None:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "collection requires the registry-owned instrument DB and private cache",
            )
        feedback = await self._feedback(profile).sample()
        frame = await self._ctx.camera_vision(profile.camera_role).capture_metric(
            after_monotonic=time.monotonic()
        )
        try:
            roi = await asyncio.to_thread(
                metric_roi,
                frame,
                region=region,
                role=profile.camera_method,
                receipt_uncertainty_s=profile.settled.receipt_uncertainty_s,
                exposure_uncertainty_s=profile.settled.exposure_uncertainty_s,
            )
        except (TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_CHECK_FAILED, f"exact metric ROI provenance unavailable: {exc}"
            ) from exc
        return {
            "schema": SCHEMA,
            "profile": profile.to_dict(),
            "workcell_digest": digest_of(self._ctx.workcell.to_dict()),
            "model_digest": feedback.model_digest,
            "declaration_digest": feedback.declaration_digest,
            "joint_ids": list(feedback.joint_ids),
            "node_session_id": self._node_session,
            "camera_device_id": frame.device_id,
            "mount_digest": roi.mount_digest,
            "camera_session_id": roi.session_id,
            "source_profile_digest": frame.metadata["profile_digest"],
            "metric_profile_digest": roi.profile.digest,
            "source_frame_id": roi.frame_id,
            "source_stamps": [vars(s) for s in roi.source_stamps],
            "frames": count,
            "region": list(region),
            "commands": [],
            "independent_witness": False,
            "carried_link_identity": False,
            "hardware_accuracy_certified": False,
        }

    async def _collection_source(
        self, plan: SkillPlan, *, commands: list[Mapping[str, float]] | None = None,
        frames: int | None = None,
    ) -> tuple[Any, Mapping[str, Any], CollectionProfile]:
        basis = plan.basis.get("collection")
        if not isinstance(basis, Mapping) or basis.get("schema") != SCHEMA:
            raise SkillError(FAILURE_PLAN_STALE, "missing sealed stationary collection policy")
        profile = self._profile()
        if (
            basis["profile"] != profile.to_dict()
            or basis["workcell_digest"] != digest_of(self._ctx.workcell.to_dict())
            or basis["node_session_id"] != self._node_session
            or basis["frames"] != (plan.args.get("frames", 1) if frames is None else frames)
            or basis["region"] != list(plan.args["region"])
            or basis["commands"] != ([] if commands is None else commands)
        ):
            raise SkillError(
                FAILURE_PLAN_STALE, "stationary collection declaration, owner or policy changed"
            )
        feedback = self._feedback(profile)
        sample = await feedback.sample()
        if (
            sample.model_digest != basis["model_digest"]
            or sample.declaration_digest != basis["declaration_digest"]
            or list(sample.joint_ids) != basis["joint_ids"]
        ):
            raise SkillError(
                FAILURE_PLAN_STALE, "raw model/channel declaration changed since approval"
            )
        source = SettledObservationSource(
            feedback,
            self._ctx.camera_vision(profile.camera_role),
            profile=profile.settled,
            region=basis["region"],
            role=profile.camera_method,
            blocked=self._blocked,
            camera_device_id=basis["camera_device_id"],
            mount_digest=basis["mount_digest"],
            session_id=basis["camera_session_id"],
            source_profile_digest=basis["source_profile_digest"],
            metric_profile_digest=basis["metric_profile_digest"],
            prior_frame_id=basis["source_frame_id"],
            prior_source_stamps=tuple(SourceStamp(**s) for s in basis["source_stamps"]),
        )
        recorded = self._ctx.recorded_observations(
            source, profile=profile.settled, ttl_s=profile.ttl_s, quota_bytes=profile.quota_bytes
        )
        return recorded, basis, profile

    async def collect(self, plan: SkillPlan) -> Mapping[str, Any]:
        recorded, basis, _ = await self._collection_source(plan)
        records = []
        for _ in range(basis["frames"]):
            if self._ctx.cancelled:
                raise SkillError(
                    FAILURE_STOPPED,
                    "stationary collection stopped; inspect committed capture digests",
                )
            await recorded.capture()
            records.append(recorded.last_record)
        return {
            "schema": SCHEMA,
            "captures": records,
            "commands": [],
            "independent_witness": False,
            "carried_link_identity": False,
            "hardware_accuracy_certified": False,
            "raw_exported": False,
        }

    def _reference_profile(self) -> ReferenceProfile:
        try:
            return ReferenceProfile.from_mapping(self._ctx.workcell.calibration_reference)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                f"declare calibration_reference before registration: {exc}",
            ) from exc

    def _worker(self) -> Any:
        worker = self._registry.calibration_worker
        if worker is None or not worker.enabled:
            raise SkillError(
                FAILURE_SKILL_UNAVAILABLE,
                "reference registration requires the explicitly enabled owner compute worker",
            )
        return worker

    async def reference_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        profile, worker = self._reference_profile(), self._worker()
        count = args.get("frames", 3)
        if type(count) is not int or not 3 <= count <= profile.max_frames:
            raise SkillError(
                FAILURE_INVALID_ARGS, "reference requires at least three declared bounded captures"
            )
        duration = (
            count * self._profile().settled.timeout_s
            + (count - 1) * worker.budgets.update_s
            + worker.budgets.startup_s
        )
        if duration > min(120, self._ctx.limits["max_duration_s"]):
            raise SkillError(
                FAILURE_INVALID_ARGS, "reference capture and compute cannot fit the run deadline"
            )
        collection = await self.collection_basis(args)
        return {
            "collection": dict(collection),
            "reference": {
                "profile": profile.to_dict(),
                "worker_budgets": vars(worker.budgets),
                "independent_witness": False,
                "clearance_verified": False,
                "physical_activation": "unqualified",
            },
        }

    async def align_reference(self, plan: SkillPlan) -> Mapping[str, Any]:
        profile, worker = self._reference_profile(), self._worker()
        if plan.basis.get("reference") != {
            "profile": profile.to_dict(),
            "worker_budgets": vars(worker.budgets),
            "independent_witness": False,
            "clearance_verified": False,
            "physical_activation": "unqualified",
        }:
            raise SkillError(
                FAILURE_PLAN_STALE, "reference profile or worker budget changed since approval"
            )
        recorded, basis, collection = await self._collection_source(plan)
        observations, captures, results = [], [], []
        previous_ids = None
        for _ in range(basis["frames"]):
            if self._blocked():
                raise SkillError(
                    FAILURE_STOPPED, "reference stopped; inspect committed acquisition digests"
                )
            observation = await recorded.capture()
            observations.append(observation)
            captures.append(recorded.last_record["evidence_digest"])
            if len(observations) < 2:
                continue
            payload = await asyncio.to_thread(
                lambda first=observations[-2].roi, second=observation.roi: {
                    "first": first.to_dict(),
                    "second": second.to_dict(),
                    "tracks": results[-1]["persistent_tracks"] if results else [],
                    "config": vars(profile.tracking),
                }
            )
            try:
                computed = await worker.submit(
                    "track_metric",
                    payload,
                    workcell_id=plan.workcell,
                    run_id=self._ctx.run_id,
                    model_digest=basis["model_digest"],
                )
                if not isinstance(computed, Mapping) or set(computed) != {"tracking"}:
                    raise ValueError("typed tracking worker result required")
                previous_ids = await asyncio.to_thread(
                    validate_tracking_result,
                    computed["tracking"],
                    observations[-2],
                    observation,
                    profile,
                    previous_ids,
                    previous_tracks=results[-1]["persistent_tracks"] if results else None,
                )
                results.append(computed["tracking"])
                if len(observations) >= 3:
                    await asyncio.to_thread(
                        assess_reference,
                        tuple(observations),
                        tuple(results),
                        profile=profile,
                        settled=collection.settled,
                    )
            except (CalibrationFailure, TypeError, ValueError) as exc:
                raise SkillError(
                    FAILURE_CHECK_FAILED, f"reference tracking refused; no motion/retry: {exc}"
                ) from exc
        if self._blocked():
            raise SkillError(FAILURE_STOPPED, "reference stopped before registration persistence")
        return await self._ctx.record_reference(captures, results, profile=profile)

    def _trial_port(self) -> Any:
        if self._trial is None:
            from leapflow.hardware.calibration.trial_port import WorkcellTrial
            self._trial = WorkcellTrial(self)
        return self._trial

    async def trial_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._trial_port().basis(args)

    async def assess_motion(self, plan: SkillPlan) -> Mapping[str, Any]:
        return await self._trial_port().execute(plan)

    def _poses_port(self) -> Any:
        if self._poses is None:
            from leapflow.hardware.calibration.poses import WorkcellPoseCollection
            self._poses = WorkcellPoseCollection(self)
        return self._poses

    async def poses_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._poses_port().basis(args)

    async def collect_poses(self, plan: SkillPlan) -> Mapping[str, Any]:
        return await self._poses_port().execute(plan)

    def _rollback_port(self) -> Any:
        if self._rollback is None:
            from leapflow.hardware.calibration.rollback import WorkcellRollback
            self._rollback = WorkcellRollback(self._ctx, self._registry)
        return self._rollback

    async def rollback_basis(self, args: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._rollback_port().basis(args)

    async def rollback(self, plan: SkillPlan) -> Mapping[str, Any]:
        return await self._rollback_port().execute(plan)

    async def inspect(self, *, evidence_digest: str = "") -> Mapping[str, Any]:
        store = self._registry.calibration_evidence
        records = (
            ()
            if store is None
            else await asyncio.to_thread(
                store.summaries, workcell_id=self._ctx.workcell.workcell_id
            )
        )
        if evidence_digest:
            if store is None:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, "acquisition index unavailable")
            try:
                records = (
                    await asyncio.to_thread(
                        store.summary, evidence_digest, workcell_id=self._ctx.workcell.workcell_id
                    ),
                )
            except ValueError as exc:
                raise SkillError(FAILURE_CHECK_FAILED, str(exc)) from exc
        references = (
            ()
            if store is None
            else await asyncio.to_thread(
                store.reference_summaries, workcell_id=self._ctx.workcell.workcell_id
            )
        )
        from leapflow.hardware.calibration.generations import GenerationState
        generations = self._registry.calibration_generations
        state = (GenerationState(self._ctx.workcell.workcell_id) if generations is None
                 else await asyncio.to_thread(generations.state, self._ctx.workcell.workcell_id))
        return {
            "schema": SCHEMA,
            "physical_model": state.basis(),
            "precision_use": "stopped_requalification_required" if state.invalidation else
                             "owner_qualification_required" if state.generation else "nominal",
            "acquisitions": records,
            "references": references,
            "raw_exported": False,
            "hardware_accuracy_certified": False,
            "physical_activation": "unqualified",
        }

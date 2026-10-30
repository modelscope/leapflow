# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable settled captures: private CacheManager raw files and shared instrument index.

This owner-internal store preserves acquisition lineage, not physical admission.
Raw expiry or eviction never deletes the durable digest/summary, and an indexed
summary alone can never authorize replay or new activation. The worker has no
reference to this store, its cache, or its instrument ConnectionHolder.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.runtime.protocol import decode_json, encode
from leapflow.cache.manager import CacheManager
from leapflow.hardware.calibration.acquisition import ObservationSource, SettledObservation, SettledProfile
from leapflow.hardware.calibration.generations import _json
from leapflow.hardware.calibration.reference import ReferenceProfile, assess_reference
from leapflow.hardware.skills.admission import RunAdmission
from leapflow.hardware.skills.digest import plan_digest, run_permit_digest
from leapflow.hardware.skills.protocol import FAILURE_CHECK_FAILED, FAILURE_DEVICE_FAULT, SIDE_EFFECT_UNKNOWN, SkillError, SkillPlan
from leapflow.storage.connection import ConnectionHolder

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from leapcalib.contracts.estimates import EstimateBundle
    from leapcalib.contracts.raw_task import MetricTaskFeature
    from leapcalib.contracts.tracking import SceneConfig, TrackingConfig
    from leapcalib.validation.bundle import EvidenceBundle
    from leapcalib.validation.evaluation import EvaluationConfig, MotionMeasurementBounds
    from leapflow.hardware.calibration.factors import IndexedMotionAssessment, IndexedMotionFactors, IndexedRawTaskFactors
    from leapflow.hardware.calibration.provenance import IndexedSourceManifest
    from leapflow.hardware.calibration.raw_assessment import IndexedRawAssessment
    from leapflow.hardware.calibration.scene import IndexedSceneCheck, IndexedSceneReplay
    from leapflow.hardware.calibration.worker import ComputeWorker

SCHEMA = "leapflow.calibration_acquisition.v1"
CATEGORY = "calibration_acquisition"
MAX_RAW_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class AcquisitionBinding:
    """Frozen exact run/plan/gate lineage, built by the owner after admission."""
    workcell_id: str
    run_id: str
    node_session_id: str
    plan_json: str
    admission: RunAdmission

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v and len(v) <= 256 for v in (self.workcell_id, self.run_id, self.node_session_id)):
            raise ValueError("bounded workcell/run/node-session identities required")
        if not isinstance(self.admission, RunAdmission) or not isinstance(self.plan_json, str) or len(self.plan_json.encode()) > 2 * 1024 * 1024:
            raise ValueError("typed owner admission and bounded immutable plan required")
        plan = json.loads(self.plan_json)
        if not isinstance(plan, dict) or set(plan) != {"skill_id", "workcell", "version", "digest", "created_at", "args", "summary", "targets", "limits", "basis"}:
            raise ValueError("exact sealed plan fields required for acquisition")
        if not isinstance(plan["targets"], list) or len(plan["targets"]) > 128 or any(not isinstance(plan[key], dict) for key in ("args", "basis", "limits")):
            raise ValueError("bounded typed sealed plan required")
        from leapflow.hardware.skills.geometry import Pose
        typed = SkillPlan(**{**plan, "targets": tuple(Pose(**p) for p in plan["targets"])})
        if typed.workcell != self.workcell_id or plan_digest(typed) != typed.digest:
            raise ValueError("acquisition plan/workcell digest mismatch")
        expected = run_permit_digest(workcell=typed.workcell, skill_id=typed.skill_id,
                                     version=typed.version, plan_digest=typed.digest, limits=typed.limits)
        if expected != self.admission.action_digest:
            raise ValueError("acquisition admission is for a different plan")

    def to_dict(self) -> dict[str, Any]:
        return {"workcell_id": self.workcell_id, "run_id": self.run_id,
                "node_session_id": self.node_session_id, "plan": json.loads(self.plan_json),
                "admission": self.admission.to_dict()}


class CalibrationEvidenceStore:
    """One registry-owned cache/holder; every operation is called off the owner loop."""
    def __init__(self, holder: ConnectionHolder, cache: CacheManager | None = None) -> None:
        if not isinstance(holder, ConnectionHolder):
            raise ValueError("acquisition index requires the existing instrument holder")
        self._holder, self._cache = holder, cache
        self._lock = threading.RLock()
        self._ready = False
        self._pending: set[asyncio.Task[Any]] = set()
        self._computations: set[asyncio.Task[Any]] = set()
        self._closing = False

    def start_record(
        self, observation: SettledObservation, *, profile: SettledProfile,
        binding: AcquisitionBinding, ttl_s: float, quota_bytes: int,
        on_commit: Callable[[dict[str, Any]], None],
    ) -> asyncio.Task[dict[str, Any]]:
        """Track admitted writes on the owner loop, including their commit callback."""
        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")

        async def persist() -> dict[str, Any]:
            summary = await asyncio.to_thread(self.record, observation, profile=profile,
                                             binding=binding, ttl_s=ttl_s, quota_bytes=quota_bytes)
            on_commit(dict(summary))
            return summary

        task = asyncio.create_task(persist(), name="calibration-acquisition-persistence")
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    async def read_capture(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        """Owner-local tracked read; drain even cancelled reads before shared-holder shutdown."""
        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        return await self._validated_read(
            lambda: self.read(identity, workcell_id=workcell_id), name="calibration-trial-capture-read"
        )

    async def source_manifest(
        self, bundle: EvidenceBundle, acquisition_ids: Mapping[str, str], *,
        trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedSourceManifest:
        """Track indexed raw validation until its shared-holder reads have settled.

        This owner-internal read grants no physical authority. Cancellation cannot
        leave a database read running after shutdown closes the instrument holder.
        """
        from leapcalib.validation.bundle import EvidenceBundle
        from leapflow.hardware.calibration.provenance import indexed_source_manifest

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if not isinstance(bundle, EvidenceBundle) or not isinstance(acquisition_ids, Mapping) or not isinstance(trusted_keys, Mapping):
            raise ValueError("typed evidence and indexed source/trusted-key maps required")
        frozen = EvidenceBundle.from_mapping(bundle.to_dict())
        sources, keys = dict(acquisition_ids), dict(trusted_keys)
        return await self._validated_read(
            lambda: indexed_source_manifest(self, frozen, sources, trusted_keys=keys, node_id=node_id),
            name="calibration-indexed-source-validation",
        )

    async def motion_factors(
        self, bundle: EvidenceBundle, acquisition_ids: Mapping[str, str],
        tracking_results: Mapping[str, Mapping[str, Any]], *, config: TrackingConfig,
        trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedMotionFactors:
        """Reconstruct derived metric/nominal-flange values through the indexed owner store.

        This tracked read neither persists nor qualifies a candidate. The returned
        source and factor checks retain explicit witness, rate and geometry gaps.
        """
        from leapcalib.validation.bundle import EvidenceBundle
        from leapflow.hardware.calibration.factors import indexed_motion_factors

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if not isinstance(bundle, EvidenceBundle) or not isinstance(acquisition_ids, Mapping) or not isinstance(trusted_keys, Mapping):
            raise ValueError("typed evidence and indexed source/trusted-key maps required")
        frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
        results = json.loads(_json(tracking_results))
        sources, keys = dict(acquisition_ids), dict(trusted_keys)
        return await self._validated_read(
            lambda: indexed_motion_factors(self, frozen, sources, results, config=config, trusted_keys=keys, node_id=node_id),
            name="calibration-indexed-factor-validation",
        )

    async def raw_task_factors(
        self, bundle: EvidenceBundle, acquisition_ids: Mapping[str, str],
        tracking_results: Mapping[str, Mapping[str, Any]], *, feature: MetricTaskFeature,
        config: TrackingConfig, trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedRawTaskFactors:
        """Tracked raw-joint/local-feature reconstruction; no persistence or admission.

        The numeric feature definition is immutable but does not authenticate
        association, an independent stationary witness, or exposure-time rates.
        """
        from leapcalib.contracts.raw_task import MetricTaskFeature
        from leapcalib.contracts.tracking import TrackingConfig
        from leapcalib.validation.bundle import EvidenceBundle
        from leapflow.hardware.calibration.factors import indexed_raw_task_factors

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if not isinstance(bundle, EvidenceBundle) or not isinstance(feature, MetricTaskFeature) or not isinstance(config, TrackingConfig) or any(not isinstance(m, Mapping) for m in (acquisition_ids, tracking_results, trusted_keys)):
            raise ValueError("typed raw evidence, feature, tracking and indexed source maps required")
        frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
        results = json.loads(_json(tracking_results))
        sources, keys = dict(acquisition_ids), dict(trusted_keys)
        return await self._validated_read(
            lambda: indexed_raw_task_factors(self, frozen, sources, results, feature=feature,
                                            config=config, trusted_keys=keys, node_id=node_id),
            name="calibration-indexed-raw-task-validation",
        )

    async def assess_motion_candidate(
        self, bundle: EvidenceBundle, fit: EstimateBundle,
        acquisition_ids: Mapping[str, str], tracking_results: Mapping[str, Mapping[str, Any]], *,
        measurement_bounds: Mapping[str, MotionMeasurementBounds], tracking: TrackingConfig,
        config: EvaluationConfig, trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedMotionAssessment:
        """One tracked owner assessment over indexed sources, factors and withheld units.

        This is side-effect-free and grants neither certificate nor activation.
        Immutable input snapshots and all shared-holder reads settle before the
        registry can close storage, even after repeated requesting-task cancellation.
        """
        from leapcalib.contracts.estimates import EstimateBundle
        from leapcalib.contracts.tracking import TrackingConfig
        from leapcalib.validation.bundle import EvidenceBundle
        from leapcalib.validation.evaluation import EvaluationConfig, MotionMeasurementBounds
        from leapflow.hardware.calibration.factors import assess_indexed_motion

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if (
            not isinstance(bundle, EvidenceBundle) or not isinstance(fit, EstimateBundle)
            or not isinstance(tracking, TrackingConfig) or not isinstance(config, EvaluationConfig)
            or any(not isinstance(m, Mapping) for m in (acquisition_ids, tracking_results, measurement_bounds, trusted_keys))
            or any(not isinstance(b, MotionMeasurementBounds) for b in measurement_bounds.values())
        ):
            raise ValueError("typed evidence, fit, policies and indexed assessment maps required")
        frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
        candidate = EstimateBundle.from_mapping(json.loads(_json(fit.to_dict())))
        results = json.loads(_json(tracking_results))
        sources, bounds, keys = dict(acquisition_ids), dict(measurement_bounds), dict(trusted_keys)
        return await self._validated_read(
            lambda: assess_indexed_motion(
                self, frozen, candidate, sources, results, measurement_bounds=bounds,
                tracking=tracking, config=config, trusted_keys=keys, node_id=node_id,
            ),
            name="calibration-indexed-candidate-assessment",
        )

    async def assess_raw_candidate(
        self, bundle: EvidenceBundle, fit: EstimateBundle,
        acquisition_ids: Mapping[str, str], tracking_results: Mapping[str, Mapping[str, Any]], *,
        baseline_bundle: EvidenceBundle, baseline: EstimateBundle,
        baseline_acquisition_ids: Mapping[str, str], baseline_tracking_results: Mapping[str, Mapping[str, Any]],
        feature: MetricTaskFeature, tracking: TrackingConfig, config: EvaluationConfig,
        trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedRawAssessment:
        """Track paired indexed raw assessment until all shared-holder reads settle.

        Freeze both evidence streams and candidates before suspension. Numerical
        recomputation does not authenticate witnesses, measured rates or safety;
        unverified captures remain insufficient, with no persistence or authority.
        """
        from leapcalib.contracts.estimates import EstimateBundle
        from leapcalib.contracts.raw_task import MetricTaskFeature
        from leapcalib.contracts.tracking import TrackingConfig
        from leapcalib.validation.bundle import EvidenceBundle
        from leapcalib.validation.evaluation import EvaluationConfig
        from leapflow.hardware.calibration.raw_assessment import assess_indexed_raw

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if (
            any(not isinstance(b, EvidenceBundle) for b in (bundle, baseline_bundle))
            or any(not isinstance(f, EstimateBundle) for f in (fit, baseline))
            or not isinstance(feature, MetricTaskFeature) or not isinstance(tracking, TrackingConfig)
            or not isinstance(config, EvaluationConfig)
            or any(not isinstance(m, Mapping) for m in (acquisition_ids, tracking_results,
                   baseline_acquisition_ids, baseline_tracking_results, trusted_keys))
        ):
            raise ValueError("typed paired evidence, fits, policies and indexed source maps required")
        frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
        before = EvidenceBundle.from_mapping(json.loads(_json(baseline_bundle.to_dict())))
        candidate = EstimateBundle.from_mapping(json.loads(_json(fit.to_dict())))
        previous = EstimateBundle.from_mapping(json.loads(_json(baseline.to_dict())))
        results, before_results = json.loads(_json(tracking_results)), json.loads(_json(baseline_tracking_results))
        sources, before_sources, keys = dict(acquisition_ids), dict(baseline_acquisition_ids), dict(trusted_keys)
        return await self._validated_read(
            lambda: assess_indexed_raw(
                self, frozen, candidate, sources, results, baseline_bundle=before, baseline=previous,
                baseline_acquisition_ids=before_sources, baseline_tracking_results=before_results,
                feature=feature, tracking=tracking, config=config, trusted_keys=keys, node_id=node_id,
            ),
            name="calibration-indexed-raw-candidate-assessment",
        )

    async def check_scene(
        self, bundle: EvidenceBundle, acquisition_ids: Mapping[str, str],
        frame_ids: Sequence[str], result: Mapping[str, Any], *,
        task_tracks: Sequence[Mapping[str, Any]], background_tracks: Sequence[Mapping[str, Any]],
        tracking: TrackingConfig, scene_config: SceneConfig,
        trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedSceneCheck:
        """Track immutable disjoint scene geometry reads, without persisting authority.

        Both complete acquisition windows settle before cancellation or shutdown
        returns. Optical association/static witnesses and physical admission
        remain unestablished even when numerical comparison is consistent.
        """
        from leapcalib.contracts.tracking import SceneConfig, TrackingConfig
        from leapcalib.validation.bundle import EvidenceBundle
        from leapflow.hardware.calibration.scene import check_indexed_scene

        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition store is shutting down")
        if not isinstance(bundle, EvidenceBundle) or not isinstance(tracking, TrackingConfig) or not isinstance(scene_config, SceneConfig) or any(not isinstance(m, Mapping) for m in (acquisition_ids, result, trusted_keys)):
            raise ValueError("typed indexed scene evidence, result and comparison policies required")
        frozen = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
        data = json.loads(_json({"frame_ids": frame_ids, "result": result,
                                "task_tracks": task_tracks, "background_tracks": background_tracks}))
        sources, keys = dict(acquisition_ids), dict(trusted_keys)
        return await self._validated_read(
            lambda: check_indexed_scene(
                self, frozen, sources, data["frame_ids"], data["result"],
                task_tracks=data["task_tracks"], background_tracks=data["background_tracks"],
                tracking=tracking, scene_config=scene_config, trusted_keys=keys, node_id=node_id,
            ),
            name="calibration-indexed-scene-validation",
        )

    async def replay_scene(
        self, bundle: EvidenceBundle, acquisition_ids: Mapping[str, str],
        frame_ids: Sequence[str], *, task_tracks: Sequence[Mapping[str, Any]],
        background_tracks: Sequence[Mapping[str, Any]], tracking: TrackingConfig,
        scene_config: SceneConfig, worker: ComputeWorker, run_id: str,
        trusted_keys: Mapping[str, Ed25519PublicKey], node_id: str,
    ) -> IndexedSceneReplay:
        """Compute optical association from private indexed images, never caller results.

        The owner-injected worker receives only copied bounded ROI data and frozen
        supports. Historical acquisition signatures are checked before compute.
        Cancellation terminates active compute and drains any shared-holder reads;
        neither late output nor numerical consistency grants physical admission.
        """
        from leapflow.hardware.calibration.scene import replay_indexed_scene

        return await replay_indexed_scene(
            self, bundle, acquisition_ids, frame_ids,
            task_tracks=task_tracks, background_tracks=background_tracks,
            tracking=tracking, scene_config=scene_config, worker=worker,
            run_id=run_id, trusted_keys=trusted_keys, node_id=node_id,
        )

    async def _tracked_compute(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Cancel compute once while allowing its shared-storage threads to settle."""
        task = asyncio.create_task(operation(), name="calibration-indexed-scene-replay")
        self._pending.add(task)
        self._computations.add(task)
        task.add_done_callback(self._pending.discard)
        task.add_done_callback(self._computations.discard)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not cancelled:
                    task.cancel()
                cancelled = True
            except Exception:
                break
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _validated_read(self, operation: Callable[[], Any], *, name: str) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation), name=name)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def close(self) -> None:
        """Drain private reads and writes before closing the shared holder."""
        self._closing = True
        for task in tuple(self._computations):
            if not task.cancelling():
                task.cancel()
        await _settle_tasks(tuple(self._pending))

    def bind_cache(self, cache: CacheManager) -> None:
        self._cache = cache

    def _schema(self, connection: Any) -> None:
        if not self._ready:
            connection.execute("CREATE TABLE IF NOT EXISTS calib_acquisition (evidence_digest VARCHAR PRIMARY KEY, workcell_id VARCHAR NOT NULL, run_id VARCHAR NOT NULL, observation_digest VARCHAR NOT NULL, summary_json VARCHAR NOT NULL, raw_hash VARCHAR NOT NULL, expires_at DOUBLE NOT NULL)")
            self._ready = True

    @staticmethod
    def _validate(observation: SettledObservation, profile: SettledProfile, binding: AcquisitionBinding) -> None:
        if not isinstance(observation, SettledObservation) or not isinstance(profile, SettledProfile) or not isinstance(binding, AcquisitionBinding):
            raise ValueError("typed settled observation/profile/owner binding required")
        if observation.profile_digest != digest_of(vars(profile)):
            raise ValueError("acquisition profile changed")
        first, before, after = observation.feedback[0], observation.feedback[-2], observation.feedback[-1]
        if first.node_session_id != binding.node_session_id:
            raise ValueError("acquisition feedback belongs to a different owner session")
        if len(observation.feedback) < profile.min_samples + 1 or before.first_receipt_s - first.last_receipt_s < profile.span_s:
            raise ValueError("acquisition lacks three separated settled feedback samples")
        if len({f.source_id for f in observation.feedback}) != len(observation.feedback):
            raise ValueError("acquisition reuses feedback identity")
        for i, sample in enumerate(observation.feedback):
            if (sample.device_id, sample.model_digest, sample.joint_ids, sample.ctrl_mode, sample.node_session_id, sample.declaration_digest) != (first.device_id, first.model_digest, first.joint_ids, first.ctrl_mode, first.node_session_id, first.declaration_digest):
                raise ValueError("acquisition feedback identity changed")
            if sample.joint_uncertainty_rad < profile.joint_uncertainty_rad or sample.tcp_uncertainty_mm < profile.tcp_uncertainty_mm or sample.rotation_uncertainty_deg < profile.rotation_uncertainty_deg or sample.stamp.uncertainty_s < profile.feedback_uncertainty_s + (sample.last_receipt_s - sample.first_receipt_s) / 2:
                raise ValueError("acquisition feedback understates declared uncertainty floors")
            if not sample.enabled or not sample.arrived or sample.arm_status or sample.communication_errors:
                raise ValueError("acquisition contains unhealthy or moving feedback")
            if sample.last_receipt_s - sample.first_receipt_s > profile.max_coherence_s or (i and sample.first_receipt_s <= observation.feedback[i - 1].last_receipt_s):
                raise ValueError("acquisition feedback is incoherent or non-increasing")
            if max(abs(a - b) for a, b in zip(first.joints_rad, sample.joints_rad)) + first.joint_uncertainty_rad + sample.joint_uncertainty_rad > profile.max_joint_drift_rad:
                raise ValueError("acquisition exceeds measured raw-joint stillness")
            if first.tcp.distance_mm(sample.tcp) + first.tcp_uncertainty_mm + sample.tcp_uncertainty_mm > profile.max_tcp_drift_mm or first.tcp.rotation_to_deg(sample.tcp) + first.rotation_uncertainty_deg + sample.rotation_uncertainty_deg > profile.max_tcp_rotation_deg:
                raise ValueError("acquisition exceeds measured nominal-flange stillness")
        uncertainty = observation.roi.stamp.uncertainty_s + observation.roi.exposure_uncertainty_s
        if observation.roi.stamp.uncertainty_s < profile.receipt_uncertainty_s or observation.roi.exposure_uncertainty_s < profile.exposure_uncertainty_s or before.last_receipt_s > observation.roi.stamp.seconds - uncertainty or after.first_receipt_s < observation.roi.stamp.seconds + uncertainty:
            raise ValueError("acquisition camera receipt/exposure uncertainty is not bracketed")

    def record(self, observation: SettledObservation, *, profile: SettledProfile,
               binding: AcquisitionBinding, ttl_s: float, quota_bytes: int) -> dict[str, Any]:
        """Persist once, never refresh expired evidence or manufacture acquisition authority."""
        self._validate(observation, profile, binding)
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or not 1 <= ttl_s <= 86400:
            raise ValueError("acquisition TTL must be 1..86400 seconds")
        if type(quota_bytes) is not int or not MAX_RAW_BYTES <= quota_bytes <= 512 * 1024 * 1024:
            raise ValueError("acquisition quota must be 4..512 MiB")
        cache = self._cache
        if not isinstance(cache, CacheManager):
            raise ValueError("acquisition requires the registry-bound cache manager")
        content = {"schema": SCHEMA, "observation": observation.to_dict(),
                   "profile": vars(profile), "binding": binding.to_dict()}
        # Reuse the worker's strict finite JSON framing for bounded raw serialization.
        raw = encode(content, MAX_RAW_BYTES)[4:]
        identity = digest_of(content)
        raw_hash = hashlib.sha256(raw).hexdigest()
        expires = time.time() + ttl_s
        summary = {"schema": SCHEMA, "evidence_digest": identity, "observation_digest": observation.digest,
                   "workcell_id": binding.workcell_id, "run_id": binding.run_id,
                   "node_session_id": binding.node_session_id,
                   "plan_digest": json.loads(binding.plan_json)["digest"],
                   "action_digest": binding.admission.action_digest, "authority_mode": binding.admission.mode,
                   "model_digest": observation.feedback[0].model_digest,
                   "profile_digest": observation.profile_digest, "camera_profile_digest": observation.roi.profile.digest,
                   "mount_digest": observation.roi.mount_digest,
                   "source_ids": [observation.roi.frame_id, *[f.source_id for f in observation.feedback]],
                   "raw_exported": False, "syncable": False, "hardware_accuracy_certified": False}
        with self._lock:
            connection = self._holder.connection
            self._schema(connection)
            existing = connection.execute("SELECT summary_json FROM calib_acquisition WHERE evidence_digest=?", [identity]).fetchone()
            if existing:
                # Idempotency never rewrites a file, resets a TTL, or resurrects an eviction.
                self.read(identity, workcell_id=binding.workcell_id)
                return json.loads(existing[0])
            prior = connection.execute("SELECT evidence_digest FROM calib_acquisition WHERE observation_digest=?", [observation.digest]).fetchone()
            if prior is not None:
                raise ValueError("raw observation already bound to a different run; lineage cannot be refreshed")
            cache.cleanup_expired()
            directory = cache.private_path(category=CATEGORY, source="settled")
            path = directory / f"{identity}.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            cache.register(path=path, scope="profile", category=CATEGORY, source="settled", content_hash=identity,
                           expires_at=expires, sensitive=True, syncable=False, owner_component="hardware.calibration",
                           metadata={"workcell_id": binding.workcell_id, "run_id": binding.run_id})
            cache.cleanup_quota(scope="profile", category=CATEGORY, max_bytes=quota_bytes)
            if not path.is_file():
                raise ValueError("acquisition raw evidence evicted by the configured quota")
            connection.execute("INSERT INTO calib_acquisition VALUES (?,?,?,?,?,?,?)", [identity, binding.workcell_id, binding.run_id, observation.digest, _json(summary), raw_hash, expires])
        return summary

    def summaries(self, *, workcell_id: str) -> tuple[dict[str, Any], ...]:
        """Bounded READ diagnostics; do not create tables or expose raw proof/paths."""
        with self._lock:
            connection = self._holder.connection
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_acquisition'").fetchone()[0]:
                return ()
            rows = connection.execute("SELECT summary_json FROM calib_acquisition WHERE workcell_id=? ORDER BY expires_at DESC,evidence_digest LIMIT 16", [workcell_id]).fetchall()
            return tuple(json.loads(row[0]) for row in rows)

    def summary(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        """Durable redacted identity survives TTL; it is explicitly not raw evidence."""
        with self._lock:
            connection = self._holder.connection
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_acquisition'").fetchone()[0]:
                raise ValueError("acquisition not indexed for this workcell")
            row = connection.execute("SELECT summary_json FROM calib_acquisition WHERE evidence_digest=? AND workcell_id=?", [identity, workcell_id]).fetchone()
            if row is None:
                raise ValueError("acquisition not indexed for this workcell")
            return json.loads(row[0])

    def read(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        """Read only indexed, private, unexpired, hash-verified raw evidence locally."""
        cache = self._cache
        if not isinstance(cache, CacheManager):
            raise ValueError("acquisition cache unavailable")
        with self._lock:
            connection = self._holder.connection
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_acquisition'").fetchone()[0]:
                raise ValueError("acquisition not indexed for this workcell")
            row = connection.execute("SELECT raw_hash,expires_at FROM calib_acquisition WHERE evidence_digest=? AND workcell_id=?", [identity, workcell_id]).fetchone()
            if row is None:
                raise ValueError("acquisition not indexed for this workcell")
            if row[1] <= time.time():
                raise ValueError("raw acquisition evidence expired; summary cannot authorize replay")
            entries = [e for e in cache.list_entries(scope="profile", category=CATEGORY) if e.entry_id == identity]
            if len(entries) != 1:
                raise ValueError("raw acquisition evidence unavailable or evicted")
            entry = entries[0]
            path = cache.private_path(category=CATEGORY, source="settled") / f"{identity}.json"
            if entry.path != path or not entry.sensitive or entry.syncable or entry.expires_at != row[1]:
                raise ValueError("raw acquisition cache binding/privacy mismatch")
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
            except OSError as exc:
                raise ValueError("raw acquisition evidence missing or unsafe to open") from exc
            with os.fdopen(fd, "rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_RAW_BYTES:
                    raise ValueError("raw acquisition ownership/mode/size refused")
                raw = source.read(MAX_RAW_BYTES + 1)
            if hashlib.sha256(raw).hexdigest() != row[0]:
                raise ValueError("raw acquisition byte digest mismatch")
            content = decode_json(raw, MAX_RAW_BYTES)
            if set(content) != {"schema", "observation", "profile", "binding"} or digest_of(content) != identity or content.get("schema") != SCHEMA:
                raise ValueError("raw acquisition content digest/schema mismatch")
            observation = SettledObservation.from_mapping(content["observation"])
            data = content["binding"]
            proof = data["admission"]
            admission = RunAdmission(proof["mode"], proof["action_digest"], json.dumps(proof["permit"]) if proof["permit"] is not None else "")
            binding = AcquisitionBinding(data["workcell_id"], data["run_id"], data["node_session_id"], _json(data["plan"]), admission)
            self._validate(observation, SettledProfile(**content["profile"]), binding)
            cache.touch(identity)
            return content

    def start_reference(self, evidence_ids: Sequence[str], results: Sequence[Mapping[str, Any]], *,
                        profile: ReferenceProfile, binding: AcquisitionBinding,
                        on_commit: Callable[[dict[str, Any]], None]) -> asyncio.Task[dict[str, Any]]:
        """Freeze and track an admitted derived write on the same owner/store lifetime."""
        if self._closing:
            raise SkillError(FAILURE_CHECK_FAILED, "reference store is shutting down")
        frozen = json.loads(_json({"evidence_ids": evidence_ids, "results": results}))

        async def persist() -> dict[str, Any]:
            summary = await asyncio.to_thread(self.record_reference, frozen["evidence_ids"], frozen["results"],
                                             profile=profile, binding=binding)
            on_commit(dict(summary))
            return summary

        task = asyncio.create_task(persist(), name="calibration-reference-persistence")
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return task

    def record_reference(self, evidence_ids: Sequence[str], results: Sequence[Mapping[str, Any]], *,
                         profile: ReferenceProfile, binding: AcquisitionBinding) -> dict[str, Any]:
        """Recompute temporal admission from indexed private evidence before immutable insertion."""
        if not isinstance(profile, ReferenceProfile) or not isinstance(binding, AcquisitionBinding):
            raise ValueError("typed reference profile and exact-run admission required")
        if not isinstance(evidence_ids, (list, tuple)) or not 3 <= len(evidence_ids) <= profile.max_frames or any(not isinstance(v, str) or not v or len(v) > 256 for v in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("bounded distinct indexed reference acquisitions required")
        plan = json.loads(binding.plan_json)
        if plan["skill_id"] != "calib.align_reference" or plan["basis"].get("reference", {}).get("profile") != profile.to_dict():
            raise ValueError("reference persistence requires its exact approved registration plan")
        with self._lock:
            captures = [self.read(identity, workcell_id=binding.workcell_id) for identity in evidence_ids]
            if any(c["binding"] != binding.to_dict() for c in captures):
                raise ValueError("reference acquisition belongs to a different admitted run")
            observations = [SettledObservation.from_mapping(c["observation"]) for c in captures]
            settled = SettledProfile(**captures[0]["profile"])
            assessed = assess_reference(observations, results, profile=profile, settled=settled)
            content = {"assessment": assessed, "acquisition_digests": list(evidence_ids),
                       "results": results, "binding": binding.to_dict()}
            encoded = _json(content)
            identity = digest_of(content)
            summary = {key: value for key, value in assessed.items() if key not in ("persistent_tracks", "raw_joints_rad", "nominal_flange_mm_deg")}
            summary.update(reference_digest=identity, acquisition_digests=list(evidence_ids),
                           workcell_id=binding.workcell_id, run_id=binding.run_id,
                           plan_digest=plan["digest"], raw_exported=False)
            connection = self._holder.connection
            connection.execute("CREATE TABLE IF NOT EXISTS calib_references (reference_digest VARCHAR PRIMARY KEY, workcell_id VARCHAR NOT NULL, content_json VARCHAR NOT NULL, summary_json VARCHAR NOT NULL, recorded_at DOUBLE NOT NULL)")
            existing = connection.execute("SELECT content_json FROM calib_references WHERE reference_digest=?", [identity]).fetchone()
            if existing is not None:
                if existing[0] != encoded:
                    raise ValueError("immutable reference digest collision")
                return summary
            connection.execute("INSERT INTO calib_references VALUES (?,?,?,?,?)", [identity, binding.workcell_id, encoded, _json(summary), time.time()])
            return summary

    def reference_summaries(self, *, workcell_id: str) -> tuple[dict[str, Any], ...]:
        """Redacted READ diagnostics survive raw expiry; they never authorize consumption."""
        with self._lock:
            connection = self._holder.connection
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_references'").fetchone()[0]:
                return ()
            rows = connection.execute("SELECT summary_json FROM calib_references WHERE workcell_id=? ORDER BY recorded_at DESC,reference_digest LIMIT 16", [workcell_id]).fetchall()
            return tuple(json.loads(row[0]) for row in rows)

    def read_reference(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        """Private consumption requires every original raw capture still present and valid."""
        with self._lock:
            connection = self._holder.connection
            if not connection.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_references'").fetchone()[0]:
                raise ValueError("reference not indexed for this workcell")
            row = connection.execute("SELECT content_json FROM calib_references WHERE reference_digest=? AND workcell_id=?", [identity, workcell_id]).fetchone()
            if row is None:
                raise ValueError("reference not indexed for this workcell")
            content = json.loads(row[0])
            if digest_of(content) != identity:
                raise ValueError("immutable reference digest mismatch")
            captures = [self.read(digest, workcell_id=workcell_id) for digest in content["acquisition_digests"]]
            if any(c["binding"] != content["binding"] for c in captures):
                raise ValueError("reference acquisition lineage differs")
            assessed = assess_reference([SettledObservation.from_mapping(c["observation"]) for c in captures], content["results"],
                                        profile=ReferenceProfile.from_mapping(content["assessment"]["profile"]),
                                        settled=SettledProfile(**captures[0]["profile"]))
            if assessed != content["assessment"]:
                raise ValueError("reference assessment differs from indexed evidence")
            return content


class RecordedObservationSource:
    """Concrete effectful capture consumer; cancellation waits for private DB/file settlement.

    The runtime must supply a post-admission binding and keep its existing run
    hold/lease. There is no motion, new permit check, retry, or camera owner here.
    """
    def __init__(self, source: ObservationSource, store: CalibrationEvidenceStore, *,
                 profile: SettledProfile, binding: AcquisitionBinding, ttl_s: float, quota_bytes: int,
                 on_commit: Callable[[dict[str, Any]], None], clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(source, ObservationSource) or not isinstance(store, CalibrationEvidenceStore):
            raise ValueError("recorded acquisition requires the concrete source and registry store")
        self._source, self._store, self._profile, self._binding = source, store, profile, binding
        self._ttl_s, self._quota, self._on_commit, self._clock = ttl_s, quota_bytes, on_commit, clock
        self.last_record: dict[str, Any] | None = None
        self._pending: asyncio.Task[dict[str, Any]] | None = None

    async def settle(self) -> None:
        """A run may release its hold only after the write and commit callback settle."""
        if self._pending is not None:
            await _settle_tasks((self._pending,))

    async def capture(self) -> SettledObservation:
        if self._pending is not None and not self._pending.done():
            raise SkillError(FAILURE_CHECK_FAILED, "acquisition persistence is already in flight")
        observation = await self._source.capture()
        now = self._clock()
        if observation.feedback[-1].last_receipt_s > now or now - observation.feedback[-1].first_receipt_s > self._profile.max_feedback_age_s:
            raise SkillError(FAILURE_CHECK_FAILED, "capture aged out before private persistence")
        task = self._store.start_record(observation, profile=self._profile,
                                        binding=self._binding, ttl_s=self._ttl_s,
                                        quota_bytes=self._quota, on_commit=self._on_commit)
        self._pending = task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break  # Inspect the settled task once, without retrying an uncertain effect.
        try:
            self.last_record = task.result()
        except Exception as error:
            raise SkillError(FAILURE_DEVICE_FAULT, "private acquisition persistence failed; inspect the indexed result before a new run",
                             side_effect_state=SIDE_EFFECT_UNKNOWN) from error
        if cancelled:
            raise asyncio.CancelledError
        return observation


async def _settle_tasks(tasks: tuple[asyncio.Task[Any], ...]) -> None:
    """Drain without canceling a shared-DB thread, then honor repeated cancellation."""
    if not tasks:
        return
    waiter = asyncio.gather(*tasks, return_exceptions=True)
    cancelled = False
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            cancelled = True
    waiter.result()
    if cancelled:
        raise asyncio.CancelledError

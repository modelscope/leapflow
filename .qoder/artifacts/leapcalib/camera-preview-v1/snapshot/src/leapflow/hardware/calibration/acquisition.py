# Copyright (c) Alibaba, Inc. and its affiliates.
"""Settled raw-feedback bracketing through existing registry and shared camera ports.

Receipt timestamps are never relabeled as exposure time. This adapter has no
actuator, permit, database, camera-pipeline, or recovery-motion authority.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapcalib.contracts.identity import ModelIdentity, digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.observations import CLOCK_NODE_MONOTONIC, STAMP_RECEIPT, Stamp
from leapcalib.contracts.tracking import MetricRoi, SourceStamp
from leapflow.hardware.calibration.measurements import metric_roi
from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.geometry import Pose
from leapflow.hardware.skills.protocol import FAILURE_CHECK_FAILED, FAILURE_DEVICE_FAULT, FAILURE_TIMEOUT, SkillError

NOMINAL_FLANGE = "nominal_firmware_flange_mm_deg"


@dataclass(frozen=True)
class SettledProfile:
    """Declaration-owned tighter thresholds; construction cannot waive the study floor."""
    max_feedback_age_s: float
    max_coherence_s: float
    feedback_uncertainty_s: float
    exposure_uncertainty_s: float
    receipt_uncertainty_s: float
    max_joint_drift_rad: float
    max_tcp_drift_mm: float
    max_tcp_rotation_deg: float
    joint_uncertainty_rad: float
    tcp_uncertainty_mm: float
    rotation_uncertainty_deg: float
    poll_s: float
    timeout_s: float
    min_samples: int = 3
    span_s: float = 0.3

    def __post_init__(self) -> None:
        for name in set(self.__dataclass_fields__) - {"min_samples"}:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite settled acquisition thresholds required")
        if type(self.min_samples) is not int or not 3 <= self.min_samples <= 16 or not 0.3 <= self.span_s <= 2 or self.timeout_s > 10:
            raise ValueError("settled acquisition needs three samples spanning at least 0.3s within a bounded deadline")
        if self.poll_s > self.span_s or self.timeout_s <= self.span_s + 2 * (self.exposure_uncertainty_s + self.receipt_uncertainty_s):
            raise ValueError("settled acquisition timing budget cannot bracket a fresh frame")
        if any(2 * floor >= limit for floor, limit in (
            (self.joint_uncertainty_rad, self.max_joint_drift_rad),
            (self.tcp_uncertainty_mm, self.max_tcp_drift_mm),
            (self.rotation_uncertainty_deg, self.max_tcp_rotation_deg),
        )):
            raise ValueError("measurement uncertainty alone exceeds settled drift budget")


@dataclass(frozen=True)
class CollectionProfile:
    """Workcell-owned stationary acquisition policy; no motion or qualification claims."""
    camera_role: str
    camera_method: str
    allowed_modes: tuple[str, ...]
    settled: SettledProfile
    ttl_s: float
    quota_bytes: int
    max_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_modes", tuple(self.allowed_modes))
        if not isinstance(self.camera_role, str) or self.camera_role.split(".", 1)[0] != "camera" or self.camera_method not in ("fixed", "wrist"):
            raise ValueError("exact camera role and explicit fixed/wrist acquisition method required")
        if not 1 <= len(self.allowed_modes) <= 16 or any(not isinstance(v, str) or not v for v in self.allowed_modes) or not isinstance(self.settled, SettledProfile):
            raise ValueError("declared safe feedback modes and settled thresholds required")
        if type(self.max_frames) is not int or not 1 <= self.max_frames <= 128 or type(self.quota_bytes) is not int or not 4 * 1024 * 1024 <= self.quota_bytes <= 512 * 1024 * 1024:
            raise ValueError("stationary collection frame/quota bounds exceeded")
        if isinstance(self.ttl_s, bool) or not isinstance(self.ttl_s, (int, float)) or not math.isfinite(self.ttl_s) or not 1 <= self.ttl_s <= 86400:
            raise ValueError("private acquisition TTL must be 1..86400 seconds")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> CollectionProfile:
        if not isinstance(data, Mapping) or set(data) != set(cls.__dataclass_fields__):
            raise ValueError("calibration_acquisition requires exact role/method/modes/settled/TTL/quota/frame fields")
        if not isinstance(data["allowed_modes"], (list, tuple)) or not isinstance(data["settled"], Mapping):
            raise ValueError("typed calibration acquisition modes/settled declaration required")
        unknown = set(data["settled"]) - set(SettledProfile.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown settled acquisition thresholds")
        try:
            return cls(**{**data, "settled": SettledProfile(**data["settled"])})
        except TypeError as exc:
            raise ValueError("complete declared settled acquisition thresholds required") from exc

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "settled": vars(self.settled), "allowed_modes": list(self.allowed_modes)}


@dataclass(frozen=True)
class FeedbackSnapshot:
    source_id: str
    device_id: str
    model_digest: str
    joint_ids: tuple[str, ...]
    joints_rad: tuple[float, ...]
    tcp: Pose
    stamp: Stamp
    first_receipt_s: float
    last_receipt_s: float
    enabled: bool
    arrived: bool
    ctrl_mode: str
    arm_status: int
    communication_errors: tuple[int, ...]
    node_session_id: str
    declaration_digest: str
    pose_semantics: str
    joint_uncertainty_rad: float
    tcp_uncertainty_mm: float
    rotation_uncertainty_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_ids", tuple(self.joint_ids))
        object.__setattr__(self, "joints_rad", finite_vector(self.joints_rad, len(self.joint_ids)))
        object.__setattr__(self, "communication_errors", tuple(self.communication_errors))
        if not 1 <= len(self.joint_ids) <= 16 or not all(isinstance(v, str) and v for v in self.joint_ids) or len(self.joint_ids) != len(set(self.joint_ids)) or not self.source_id or not self.model_digest:
            raise ValueError("complete ordered raw feedback identity required")
        if not all(isinstance(v, str) and v for v in (self.device_id, self.node_session_id, self.declaration_digest, self.ctrl_mode)) or self.pose_semantics != NOMINAL_FLANGE:
            raise ValueError("raw feedback needs owner-session, declaration, and nominal flange semantics")
        if not isinstance(self.tcp, Pose):
            raise ValueError("typed nominal flange pose required")
        finite_vector(tuple(vars(self.tcp).values()), 6)
        for floor in (self.joint_uncertainty_rad, self.tcp_uncertainty_mm, self.rotation_uncertainty_deg):
            if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor <= 0:
                raise ValueError("nonzero raw measurement uncertainty floors required")
        if not isinstance(self.tcp, Pose) or not isinstance(self.stamp, Stamp) or self.stamp.domain != CLOCK_NODE_MONOTONIC or self.stamp.semantics != STAMP_RECEIPT or self.stamp.uncertainty_s <= 0:
            raise ValueError("feedback needs typed pose and nonzero node receipt uncertainty")
        if not all(math.isfinite(v) for v in (self.first_receipt_s, self.last_receipt_s)) or not self.first_receipt_s <= self.stamp.seconds <= self.last_receipt_s:
            raise ValueError("feedback stamp must lie inside its coherent receipt bracket")
        if any(type(v) is not int or v < 0 for v in self.communication_errors):
            raise ValueError("explicit bounded communication error identifiers required")
        if type(self.enabled) is not bool or type(self.arrived) is not bool or type(self.arm_status) is not int:
            raise ValueError("explicit enabled/mode/error feedback required")


@runtime_checkable
class FeedbackSource(Protocol):
    async def sample(self) -> FeedbackSnapshot: ...


@runtime_checkable
class MetricCapture(Protocol):
    async def capture_metric(self, *, after_monotonic: float) -> RgbdFrame: ...


@runtime_checkable
class ObservationSource(Protocol):
    async def capture(self) -> SettledObservation: ...


@dataclass(frozen=True)
class SettledObservation:
    roi: MetricRoi
    feedback: tuple[FeedbackSnapshot, ...]
    profile_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "feedback", tuple(self.feedback))
        if not isinstance(self.roi, MetricRoi) or not 4 <= len(self.feedback) <= 32 or not all(isinstance(f, FeedbackSnapshot) for f in self.feedback):
            raise ValueError("settled observation requires metric ROI and bounded typed feedback")
        if not isinstance(self.profile_digest, str) or not self.profile_digest:
            raise ValueError("settled observation needs its frozen acquisition profile")

    def to_dict(self) -> dict[str, Any]:
        # Pose.to_dict is display-rounded; indexed raw evidence must be lossless.
        return {"schema": "leapflow.settled_observation.v1", "roi": self.roi.to_dict(),
                "feedback": [{**vars(f), "tcp": dict(vars(f.tcp)), "stamp": f.stamp.to_dict()} for f in self.feedback],
                "profile_digest": self.profile_digest}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SettledObservation:
        if not isinstance(data, Mapping) or set(data) != {"schema", "roi", "feedback", "profile_digest"} or data["schema"] != "leapflow.settled_observation.v1":
            raise ValueError("unknown settled observation schema or fields")
        samples = data["feedback"]
        if not isinstance(samples, (list, tuple)) or not 4 <= len(samples) <= 32:
            raise ValueError("bounded settled feedback count required before decoding")
        feedback = []
        for sample in samples:
            if not isinstance(sample, Mapping) or set(sample) != set(FeedbackSnapshot.__dataclass_fields__):
                raise ValueError("unknown raw feedback fields")
            if not isinstance(sample["tcp"], Mapping) or not isinstance(sample["stamp"], Mapping) or set(sample["tcp"]) != set(Pose.__dataclass_fields__) or set(sample["stamp"]) != set(Stamp.__dataclass_fields__):
                raise ValueError("unknown feedback pose/stamp fields")
            if not isinstance(sample["joint_ids"], (list, tuple)) or not 1 <= len(sample["joint_ids"]) <= 16 or not isinstance(sample["joints_rad"], (list, tuple)) or len(sample["joints_rad"]) != len(sample["joint_ids"]) or not isinstance(sample["communication_errors"], (list, tuple)) or len(sample["communication_errors"]) > 16:
                raise ValueError("bounded raw feedback arrays required before decoding")
            feedback.append(FeedbackSnapshot(**{**sample, "tcp": Pose(**sample["tcp"]), "stamp": Stamp.from_mapping(sample["stamp"])}))
        return cls(MetricRoi.from_mapping(data["roi"]), tuple(feedback), data["profile_digest"])

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


class RegistryFeedbackSource:
    """Actual raw samples from registry.read_batch, never ArmState.observed_at."""
    def __init__(self, registry: Any, *, device_id: str, model: ModelIdentity,
                 joint_ids: Sequence[str], pose_ids: Sequence[str], health_ids: Sequence[str],
                 allowed_modes: Sequence[str], profile: SettledProfile,
                 node_session_id: str, pose_semantics: str,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._registry, self._device, self._model = registry, device_id, model
        self._joints, self._poses, self._health = tuple(joint_ids), tuple(pose_ids), tuple(health_ids)
        self._modes, self._profile, self._clock = tuple(allowed_modes), profile, clock
        if not isinstance(node_session_id, str) or not node_session_id or pose_semantics != NOMINAL_FLANGE:
            raise ValueError("explicit owner-session and nominal firmware flange semantics required")
        self._node_session = node_session_id
        channels = (*self._joints, *self._poses, *self._health)
        if len(self._joints) != model.dof or len(self._poses) != 6 or len(self._health) != 4 or len(set(channels)) != len(channels) or not self._modes:
            raise ValueError("explicit ordered joint/pose/enabled/arrived/mode/status channels required")
        context = registry.context(device_id)
        if context is None or any(context.channel(cid) is None for cid in channels):
            raise ValueError("feedback channels not admitted by the owning device")
        if any(context.channel(cid).unit != "deg" for cid in self._joints):
            raise ValueError("raw-joint source requires explicitly declared encoder degrees")
        if tuple(context.channel(cid).unit for cid in self._poses) != ("mm", "mm", "mm", "deg", "deg", "deg"):
            raise ValueError("nominal flange channels must declare mm/deg in xyz/rxyz order")
        self._units = {cid: context.channel(cid).unit for cid in channels}
        self._declaration = digest_of(context.to_dict())

    @property
    def model(self) -> ModelIdentity:
        """Immutable nominal model identity carried by every raw feedback sample."""
        return self._model

    async def sample(self) -> FeedbackSnapshot:
        batch = await self._registry.read_batch(self._device, (*self._joints, *self._poses, *self._health))
        readings = {r.channel_id: r for r in batch.readings}
        requested = (*self._joints, *self._poses, *self._health)
        context = self._registry.context(self._device)
        if context is None or digest_of(context.to_dict()) != self._declaration:
            raise SkillError(FAILURE_CHECK_FAILED, "raw feedback declaration changed during acquisition")
        if batch.device_id != self._device or set(readings) != set(requested) or len(batch.readings) != len(requested):
            raise SkillError(FAILURE_DEVICE_FAULT, "incomplete raw feedback cohort")
        now = self._clock()
        times = [r.monotonic_at for r in readings.values()]
        if any(r.device_id != self._device or not r.is_trustworthy or r.unit != self._units[r.channel_id] or type(r.sequence) is not int or r.sequence < 0 for r in readings.values()) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= now for v in times):
            raise SkillError(FAILURE_DEVICE_FAULT, "untrustworthy raw feedback receipts")
        first, last = min(times), max(times)
        if now - first > self._profile.max_feedback_age_s or last - first > self._profile.max_coherence_s:
            raise SkillError(FAILURE_DEVICE_FAULT, "stale or incoherent raw feedback cohort")
        values = {cid: r.value for cid, r in readings.items()}
        try:
            joint_deg = finite_vector(tuple(values[cid] for cid in self._joints), self._model.dof)
            tcp_values = finite_vector(tuple(values[cid] for cid in self._poses), 6)
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_DEVICE_FAULT, "nonfinite or malformed raw joint/flange feedback") from exc
        enabled, arrived, mode, status = (values[cid] for cid in self._health)
        if type(enabled) not in (bool, int) or enabled != 1 or type(arrived) not in (bool, int) or arrived != 1 or not isinstance(mode, str) or mode not in self._modes or not isinstance(status, Mapping):
            raise SkillError(FAILURE_CHECK_FAILED, "enabled/arrived/mode feedback is not safe for settled capture")
        error, communication = status.get("arm_status"), status.get("joint_communication_errors")
        if type(error) is not int or not isinstance(communication, (list, tuple)) or error != 0 or communication:
            raise SkillError(FAILURE_CHECK_FAILED, "missing or unhealthy motor/error/communication feedback")
        identity = digest_of({"device": self._device, "model": self._model.digest, "node_session": self._node_session, "declaration": self._declaration,
                              "receipts": [{"channel": cid, "sequence": readings[cid].sequence, "monotonic_s": readings[cid].monotonic_at, "value": values[cid]} for cid in requested]})
        return FeedbackSnapshot(identity, self._device, self._model.digest, self._joints, tuple(math.radians(v) for v in joint_deg),
                                Pose(*tcp_values), Stamp((first + last) / 2, uncertainty_s=self._profile.feedback_uncertainty_s + (last - first) / 2),
                                first, last, True, True, mode, error, tuple(communication),
                                self._node_session, self._declaration, NOMINAL_FLANGE,
                                self._profile.joint_uncertainty_rad, self._profile.tcp_uncertainty_mm,
                                self._profile.rotation_uncertainty_deg)


class SettledObservationSource:
    """Read-only, bounded three-sample stillness followed by bracketed metric capture."""
    def __init__(self, feedback: FeedbackSource, camera: MetricCapture, *, profile: SettledProfile,
                 region: Sequence[int], role: str, blocked: Callable[[], bool],
                 camera_device_id: str, mount_digest: str, session_id: str, source_profile_digest: str,
                 metric_profile_digest: str = "", prior_frame_id: str = "",
                 prior_source_stamps: Sequence[SourceStamp] = (),
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = asyncio.sleep) -> None:
        if not isinstance(feedback, FeedbackSource) or not isinstance(camera, MetricCapture):
            raise ValueError("settled acquisition needs concrete feedback/shared metric capture ports")
        self._feedback, self._camera, self._profile = feedback, camera, profile
        self._region, self._role, self._blocked = tuple(region), role, blocked
        self._camera_identity = (camera_device_id, mount_digest, session_id, source_profile_digest)
        if any(not isinstance(v, str) or not v for v in self._camera_identity):
            raise ValueError("exact admitted role camera/mount/session/profile identity required")
        self._clock, self._sleep = clock, sleep
        if any(not isinstance(v, str) or len(v) > 256 for v in (metric_profile_digest, prior_frame_id)):
            raise ValueError("bounded metric profile and prior frame identities required")
        if not isinstance(prior_source_stamps, (list, tuple)) or len(prior_source_stamps) not in (0, 2):
            raise ValueError("bounded prior color/depth provenance required")
        self._metric_profile_digest = metric_profile_digest
        self._prior_frame_id = prior_frame_id
        self._prior_sources = tuple(prior_source_stamps)
        if self._prior_sources and (any(not isinstance(s, SourceStamp) for s in self._prior_sources) or tuple(s.stream for s in self._prior_sources) != ("color", "depth")):
            raise ValueError("typed prior color/depth provenance required")

    def _consistent(self, first: FeedbackSnapshot, sample: FeedbackSnapshot) -> None:
        if (first.device_id, first.model_digest, first.joint_ids, first.ctrl_mode, first.node_session_id, first.declaration_digest) != (sample.device_id, sample.model_digest, sample.joint_ids, sample.ctrl_mode, sample.node_session_id, sample.declaration_digest):
            raise SkillError(FAILURE_CHECK_FAILED, "raw feedback model/channel/mode identity changed")
        if not sample.enabled or not sample.arrived or sample.arm_status != 0 or sample.communication_errors or self._blocked():
            raise SkillError(FAILURE_CHECK_FAILED, "motion/pending action or unhealthy feedback blocks capture")
        now = self._clock()
        if sample.last_receipt_s > now or now - sample.first_receipt_s > self._profile.max_feedback_age_s or sample.last_receipt_s - sample.first_receipt_s > self._profile.max_coherence_s:
            raise SkillError(FAILURE_CHECK_FAILED, "stale or incoherent settled feedback sample")
        if max(abs(a - b) for a, b in zip(first.joints_rad, sample.joints_rad)) + first.joint_uncertainty_rad + sample.joint_uncertainty_rad > self._profile.max_joint_drift_rad:
            raise SkillError(FAILURE_CHECK_FAILED, "measured raw-joint stillness/drift bound exceeded")
        if first.tcp.distance_mm(sample.tcp) + first.tcp_uncertainty_mm + sample.tcp_uncertainty_mm > self._profile.max_tcp_drift_mm or first.tcp.rotation_to_deg(sample.tcp) + first.rotation_uncertainty_deg + sample.rotation_uncertainty_deg > self._profile.max_tcp_rotation_deg:
            raise SkillError(FAILURE_CHECK_FAILED, "measured TCP stillness/drift bound exceeded")

    async def capture(self) -> SettledObservation:
        try:
            async with asyncio.timeout(self._profile.timeout_s):
                return await self._capture()
        except TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "settled acquisition deadline expired; inspect feedback/camera readiness before a new run") from exc

    async def _capture(self) -> SettledObservation:
        if self._blocked():
            raise SkillError(FAILURE_CHECK_FAILED, "motion or uncertain previous dispatch blocks acquisition")
        samples = []
        while len(samples) < self._profile.min_samples or samples[-1].first_receipt_s - samples[0].last_receipt_s < self._profile.span_s:
            sample = await self._feedback.sample()
            self._consistent(samples[0] if samples else sample, sample)
            if samples and (sample.first_receipt_s <= samples[-1].last_receipt_s or sample.source_id == samples[-1].source_id):
                raise SkillError(FAILURE_CHECK_FAILED, "repeated/non-increasing raw feedback is not a new settled sample")
            samples.append(sample)
            if len(samples) >= 32:
                raise SkillError(FAILURE_CHECK_FAILED, "settled sample budget exceeded")
            await self._sleep(max(self._profile.poll_s, self._profile.exposure_uncertainty_s + self._profile.receipt_uncertainty_s))
        bound = self._clock()
        frame = await self._camera.capture_metric(after_monotonic=bound)
        identity = (frame.device_id, *(frame.metadata.get(k) for k in ("mount_digest", "session_id", "profile_digest")))
        if identity != self._camera_identity:
            raise SkillError(FAILURE_CHECK_FAILED, "admitted camera role/mount/session/profile changed")
        if not bound <= frame.monotonic_at <= self._clock():
            raise SkillError(FAILURE_CHECK_FAILED, "metric capture is not fresh on the node receipt clock")
        await self._sleep(self._profile.exposure_uncertainty_s + self._profile.receipt_uncertainty_s)
        after = await self._feedback.sample()
        self._consistent(samples[0], after)
        uncertainty = self._profile.exposure_uncertainty_s + self._profile.receipt_uncertainty_s
        if after.first_receipt_s <= samples[-1].last_receipt_s or after.first_receipt_s < frame.monotonic_at + uncertainty or samples[-1].last_receipt_s > frame.monotonic_at - uncertainty:
            raise SkillError(FAILURE_CHECK_FAILED, "fresh feedback does not bracket uncertain camera exposure/receipt")
        samples.append(after)
        try:
            roi = await asyncio.to_thread(metric_roi, frame, region=self._region, role=self._role,
                                          receipt_uncertainty_s=self._profile.receipt_uncertainty_s,
                                          exposure_uncertainty_s=self._profile.exposure_uncertainty_s)
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"metric acquisition provenance refused: {exc}") from exc
        self._consistent(samples[0], after)
        if self._metric_profile_digest and roi.profile.digest != self._metric_profile_digest:
            raise SkillError(FAILURE_CHECK_FAILED, "metric intrinsics changed before acquisition persistence")
        if roi.frame_id == self._prior_frame_id:
            raise SkillError(FAILURE_CHECK_FAILED, "repeated metric frame is not fresh evidence")
        for prior, current in zip(self._prior_sources, roi.source_stamps):
            if prior.timestamp_domain != current.timestamp_domain or current.frame_number <= prior.frame_number or current.timestamp_ms <= prior.timestamp_ms:
                raise SkillError(FAILURE_CHECK_FAILED, "repeated/non-increasing native stream provenance")
        self._prior_frame_id, self._prior_sources = roi.frame_id, roi.source_stamps
        return SettledObservation(roi, tuple(samples), digest_of(vars(self._profile)))

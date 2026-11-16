# Copyright (c) Alibaba, Inc. and its affiliates.
"""Two shared camera owners, one qualified stationary interval, private evidence.

No exposure synchronization or independent board identity is inferred from skew.
The physical sheet and stationary epoch must be declared by the operator. Both
lossless ROIs and all correspondences remain in the existing indexed cache.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from leapcalib.contracts.identity import digest_of
from leapflow.hardware.calibration.acquisition import (
    FeedbackSource, MetricCapture, SettledObservation, SettledObservationSource, SettledProfile,
)
from leapflow.hardware.calibration.measurements import metric_roi
from leapflow.hardware.skills.protocol import (
    FAILURE_CHECK_FAILED, FAILURE_SKILL_UNAVAILABLE, FAILURE_STOPPED, FAILURE_TIMEOUT, SkillError,
)
from leapflow.hardware.skills.vision import BoardDetection, CharucoTarget, detect_board
from leapflow.hardware.vision.group import CaptureGroup

SCHEMA = "leapflow.paired_target.v1"
EXCLUSIONS = frozenset({"target_not_found", "ambiguous_planar_pose", "insufficient_correspondence"})


@dataclass(frozen=True)
class PairedObservation:
    fixed: SettledObservation
    wrist: SettledObservation
    target_json: str
    target_id: str
    stationary_epoch: str
    fixed_detection: BoardDetection | None
    wrist_detection: BoardDetection | None
    min_pose_gap_px: float
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        import json

        if not all(isinstance(v, SettledObservation) for v in (self.fixed, self.wrist)):
            raise ValueError("typed paired settled observations required")
        if self.fixed.feedback != self.wrist.feedback or self.fixed.profile_digest != self.wrist.profile_digest:
            raise ValueError("both cameras must share one stationary feedback interval")
        if self.fixed.roi.profile.role != "fixed" or self.wrist.roi.profile.role != "wrist" or self.fixed.roi.profile.device_id == self.wrist.roi.profile.device_id:
            raise ValueError("distinct fixed and wrist camera owners required")
        if any(not isinstance(v, str) or not 0 < len(v) <= 256 for v in (self.target_id, self.stationary_epoch)):
            raise ValueError("declared physical target and stationary epoch required")
        if not isinstance(self.target_json, str) or len(self.target_json) > 4096:
            raise ValueError("bounded board geometry required")
        target = CharucoTarget.from_mapping(json.loads(self.target_json))
        if type(self.min_pose_gap_px) not in (int, float) or not math.isfinite(self.min_pose_gap_px) or self.min_pose_gap_px <= 0:
            raise ValueError("positive declared planar ambiguity separation required")
        expected = []
        for view, detection in ((self.fixed, self.fixed_detection), (self.wrist, self.wrist_detection)):
            if detection is None:
                expected.append("target_not_found")
                continue
            validate_detection(detection, target, view)
            if detection.pose_gap_px < self.min_pose_gap_px:
                expected.append("ambiguous_planar_pose")
        if self.fixed_detection and self.wrist_detection:
            common = set(self.fixed_detection.corner_ids) & set(self.wrist_detection.corner_ids)
            if len(common) < target.min_corners or not observable_corners(common, target):
                expected.append("insufficient_correspondence")
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        if self.exclusions != tuple(sorted(set(expected))):
            raise ValueError("paired exclusion reasons must be recomputed from retained evidence")

    @property
    def feedback(self) -> tuple[Any, ...]:
        return self.fixed.feedback

    @property
    def source_ids(self) -> tuple[str, ...]:
        sources = [self.fixed.roi.frame_id, self.wrist.roi.frame_id, *[f.source_id for f in self.feedback]]
        for view in (self.fixed, self.wrist):
            sources.extend(digest_of({"device": view.roi.profile.device_id, "session": view.roi.session_id,
                                      "source": vars(stamp)}) for stamp in view.roi.source_stamps)
        return tuple(sources)

    @property
    def interval_s(self) -> tuple[float, float]:
        return self.feedback[0].first_receipt_s, self.feedback[-1].last_receipt_s

    @property
    def roi(self) -> Any:
        return self.fixed.roi

    @property
    def profile_digest(self) -> str:
        return self.fixed.profile_digest

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "fixed": self.fixed.to_dict(), "wrist": self.wrist.to_dict(),
                "target_json": self.target_json, "target_id": self.target_id, "stationary_epoch": self.stationary_epoch,
                "fixed_detection": self.fixed_detection.to_dict() if self.fixed_detection else None,
                "wrist_detection": self.wrist_detection.to_dict() if self.wrist_detection else None,
                "min_pose_gap_px": self.min_pose_gap_px, "exclusions": list(self.exclusions)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PairedObservation:
        if not isinstance(data, Mapping) or set(data) != {"schema", *cls.__dataclass_fields__} or data["schema"] != SCHEMA:
            raise ValueError("unknown paired target schema/fields")
        return cls(**{**{k: v for k, v in data.items() if k != "schema"},
                      "fixed": SettledObservation.from_mapping(data["fixed"]),
                      "wrist": SettledObservation.from_mapping(data["wrist"]),
                      **{k: BoardDetection.from_mapping(data[k]) if data[k] is not None else None
                         for k in ("fixed_detection", "wrist_detection")}})


def observable_corners(ids: Iterable[int], target: CharucoTarget) -> bool:
    """Exact rank-two test on integer board indices, including diagonal degeneracy."""
    points = [(i % (target.squares_x - 1), i // (target.squares_x - 1)) for i in sorted(ids)]
    if len(points) < 3:
        return False
    x, y = points[0]
    dx, dy = points[1][0] - x, points[1][1] - y
    return any(dx * (py - y) != dy * (px - x) for px, py in points[2:])


def validate_detection(detection: BoardDetection, target: CharucoTarget, view: SettledObservation) -> None:
    """Validate IDs, board convention and lossless ROI support before indexing."""
    if not isinstance(detection, BoardDetection) or type(detection.corners) is not int:
        raise ValueError("typed board detection required")
    ids, pixels, points = detection.corner_ids, detection.image_points, detection.board_points_mm
    if not target.min_corners <= len(ids) <= (target.squares_x - 1) * (target.squares_y - 1) or detection.corners != len(ids) or len(set(ids)) != len(ids) or len(points) != len(ids) or len(pixels) != len(ids):
        raise ValueError("bounded unique lossless corner correspondence required")
    x, y = view.roi.x, view.roi.y
    for i, pixel, point in zip(ids, pixels, points):
        if type(i) is not int or not 0 <= i < (target.squares_x - 1) * (target.squares_y - 1):
            raise ValueError("invalid board corner ID")
        expected = ((i % (target.squares_x - 1) + 1) * target.square_mm,
                    (i // (target.squares_x - 1) + 1) * target.square_mm, 0.0)
        # OpenCV stores board coordinates as float32; retain their exact values.
        if len(point) != 3 or any(not math.isfinite(v) for v in point) or math.dist(point, expected) > 0.0001:
            raise ValueError("corner ID disagrees with declared OpenCV board convention")
        if len(pixel) != 2 or not all(math.isfinite(v) for v in pixel) or not x <= pixel[0] < x + view.roi.width or not y <= pixel[1] < y + view.roi.height:
            raise ValueError("detected corner is outside the retained lossless ROI")
    if not observable_corners(ids, target):
        raise ValueError("board corner geometry is rank deficient")
    if len(detection.marker_ids) != len(detection.marker_points) or len(set(detection.marker_ids)) != len(detection.marker_ids) or len(detection.marker_ids) > target.squares_x * target.squares_y:
        raise ValueError("bounded unique marker correspondences required")
    for identity, corners in zip(detection.marker_ids, detection.marker_points):
        if type(identity) is not int or not 0 <= identity < target.squares_x * target.squares_y or len(corners) != 4 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) or not x <= p[0] < x + view.roi.width or not y <= p[1] < y + view.roi.height for p in corners):
            raise ValueError("marker correspondence is not supported by retained ROI")
    for value in (detection.reprojection_px, detection.pose_gap_px):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative detection errors required")
    from leapflow.hardware.calibration.measurements import si_transform
    si_transform(detection.transform, parent="camera", child="board")
    if any(detection.transform.apply(p)[2] <= 0 for p in points):
        raise ValueError("target is behind the camera")


class _CaptureDip(SkillError):
    """Transient read-only camera failure; never an identity or motion retry."""


class PairedObservationSource:
    """Bounded CaptureGroup acquisition using the single-camera stillness checks."""
    def __init__(self, feedback: FeedbackSource, cameras: Mapping[str, MetricCapture], *,
                 profile: SettledProfile, bindings: Mapping[str, Mapping[str, Any]],
                 target: CharucoTarget, target_id: str, stationary_epoch: str, min_pose_gap_px: float,
                 tolerance_ms: float, attempts: int, blocked: Callable[[], bool],
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[..., Any] = asyncio.sleep) -> None:
        if set(cameras) != {"fixed", "wrist"} or set(bindings) != set(cameras):
            raise ValueError("exact fixed/wrist paired camera bindings required")
        if type(attempts) is not int or not 1 <= attempts <= 3:
            raise ValueError("bounded paired acquisition attempts required")
        self._attempts = attempts
        self._profile, self._feedback, self._blocked = profile, feedback, blocked
        self._clock, self._sleep, self._target = clock, sleep, target
        self._target_id, self._epoch, self._gap = target_id, stationary_epoch, min_pose_gap_px
        self._bindings = {k: dict(v) for k, v in bindings.items()}
        self._checks = {role: SettledObservationSource(feedback, cameras[role], profile=profile,
                        region=b["region"], role=role, blocked=blocked,
                        camera_device_id=b["camera_device_id"], mount_digest=b["mount_digest"],
                        session_id=b["camera_session_id"], source_profile_digest=b["source_profile_digest"])
                        for role, b in self._bindings.items()}
        self._group = CaptureGroup({role: lambda bound, camera=camera: camera.capture_metric(after_monotonic=bound)
                                   for role, camera in cameras.items()}, tolerance_ms=tolerance_ms, attempts=attempts)

    async def capture(self) -> PairedObservation:
        try:
            async with asyncio.timeout(self._profile.timeout_s):
                for attempt in range(self._attempts):
                    self._check_authority()
                    try:
                        return await self._capture()
                    except _CaptureDip:
                        if attempt + 1 == self._attempts:
                            raise
                        await self._sleep(self._profile.poll_s)
                raise AssertionError("bounded capture loop did not return")
        except TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "paired stationary acquisition deadline expired; inspect both cameras") from exc
        except ImportError as exc:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "paired detection requires the declared ChArUco vision dependency") from exc
        except (TypeError, ValueError) as exc:
            raise SkillError(FAILURE_CHECK_FAILED, f"paired evidence invalid: {exc}") from exc

    def _check_authority(self) -> None:
        if self._blocked():
            raise SkillError(FAILURE_STOPPED, "paired acquisition stopped; no new persistence or recovery motion")

    async def _capture(self) -> PairedObservation:
        profile, samples = self._profile, []
        check = self._checks["fixed"]
        while len(samples) < profile.min_samples or samples[-1].first_receipt_s - samples[0].last_receipt_s < profile.span_s:
            sample = await self._feedback.sample()
            check.check_feedback(samples[0] if samples else sample, sample)
            if samples and (sample.first_receipt_s <= samples[-1].last_receipt_s or sample.source_id in {s.source_id for s in samples}):
                raise SkillError(FAILURE_CHECK_FAILED, "paired feedback reused or non-increasing")
            samples.append(sample)
            if len(samples) >= 32:
                raise SkillError(FAILURE_CHECK_FAILED, "paired feedback budget exceeded")
            await self._sleep(max(profile.poll_s, profile.max_coherence_s + profile.exposure_uncertainty_s + profile.receipt_uncertainty_s))
        uncertainty = profile.exposure_uncertainty_s + profile.receipt_uncertainty_s
        bound = max(self._clock(), samples[-1].last_receipt_s + uncertainty)
        group = await self._group.capture(after_monotonic=bound)
        if group.errors:
            transient = all(isinstance(error, (OSError, TimeoutError)) or
                            isinstance(error, SkillError) and error.failure_code == FAILURE_TIMEOUT
                            for error in group.errors.values())
            error_type = _CaptureDip if transient else SkillError
            raise error_type(FAILURE_TIMEOUT if transient else FAILURE_CHECK_FAILED,
                             "paired capture incomplete; inspect camera readiness")
        if set(group.frames) != {"fixed", "wrist"}:
            raise SkillError(FAILURE_CHECK_FAILED, "paired capture incomplete; inspect camera readiness")
        self._check_authority()
        frames = group.frames
        for role, frame in frames.items():
            b = self._bindings[role]
            if (frame.device_id, *(frame.metadata.get(k) for k in ("mount_digest", "session_id", "profile_digest"))) != (b["camera_device_id"], b["mount_digest"], b["camera_session_id"], b["source_profile_digest"]):
                raise SkillError(FAILURE_CHECK_FAILED, "paired camera identity changed")
            if not bound <= frame.monotonic_at <= self._clock():
                raise SkillError(FAILURE_CHECK_FAILED, "paired frame has unsupported/future/stale receipt semantics")
        # Timestamp skew is recorded implicitly in both ROIs. Asynchronous frames
        # are admissible only when the same stationary interval brackets both.
        await self._sleep(profile.max_coherence_s + uncertainty)
        after = await self._feedback.sample()
        check.check_feedback(samples[0], after)
        if any(samples[-1].last_receipt_s > f.monotonic_at - uncertainty or after.first_receipt_s < f.monotonic_at + uncertainty for f in frames.values()):
            raise SkillError(FAILURE_CHECK_FAILED, "one stationary interval does not bracket both camera uncertainty intervals")
        samples.append(after)
        views, detections, reasons = {}, {}, []
        for role, frame in frames.items():
            b = self._bindings[role]
            roi = await asyncio.to_thread(metric_roi, frame, region=b["region"], role=role,
                                          receipt_uncertainty_s=profile.receipt_uncertainty_s,
                                          exposure_uncertainty_s=profile.exposure_uncertainty_s)
            if roi.profile.digest != b["metric_profile_digest"] or roi.frame_id == b["source_frame_id"]:
                raise SkillError(FAILURE_CHECK_FAILED, "paired intrinsics changed or frame repeated")
            if len(b["source_stamps"]) != len(roi.source_stamps):
                raise SkillError(FAILURE_CHECK_FAILED, "paired native stream count changed")
            for previous, current in zip(b["source_stamps"], roi.source_stamps):
                if previous["timestamp_domain"] != current.timestamp_domain or current.frame_number <= previous["frame_number"] or current.timestamp_ms <= previous["timestamp_ms"]:
                    raise SkillError(FAILURE_CHECK_FAILED, "paired native provenance reused or changed domain")
            b.update(source_frame_id=roi.frame_id, source_stamps=[vars(s) for s in roi.source_stamps])
            views[role] = SettledObservation(roi, tuple(samples), digest_of(vars(profile)))
            detections[role] = await asyncio.to_thread(detect_board, frame, self._target)
            if detections[role] is None:
                reasons.append("target_not_found")
            elif detections[role].pose_gap_px < self._gap:
                reasons.append("ambiguous_planar_pose")
        if all(detections.values()):
            common = set(detections["fixed"].corner_ids) & set(detections["wrist"].corner_ids)
            if len(common) < self._target.min_corners or not observable_corners(common, self._target):
                reasons.append("insufficient_correspondence")
        self._check_authority()
        import json
        return PairedObservation(views["fixed"], views["wrist"], json.dumps(self._target.to_dict(), sort_keys=True),
                                 self._target_id, self._epoch, detections["fixed"], detections["wrist"],
                                 self._gap, tuple(sorted(set(reasons))))

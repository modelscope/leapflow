# Copyright (c) Alibaba, Inc. and its affiliates.
"""Task-local stationary scene references, never physical-model or clearance certificates.

The owner checks typed indexed acquisitions and worker identities independently.
A temporal reference provides correspondence continuity, not an independent target
witness, carried-link identity, absolute tool geometry, or motion authority.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from leapcalib.contracts.identity import digest_of
from leapcalib.contracts.motion import finite_vector
from leapcalib.contracts.tracking import TRACK_SCHEMA, TrackingConfig
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.acquisition import SettledObservation, SettledProfile
from leapflow.hardware.calibration.measurements import metric_track_point

SCHEMA = "leapflow.scene_reference.v1"


@dataclass(frozen=True)
class ReferenceProfile:
    tracking: TrackingConfig
    tolerance_m_rad: tuple[float, float, float, float]
    posture_family: str
    max_frames: int
    span_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tolerance_m_rad", finite_vector(self.tolerance_m_rad, 4))
        if not isinstance(self.tracking, TrackingConfig):
            raise ValueError("typed reference tracking policy required")
        if any(
            not 0 < value <= cap
            for value, cap in zip(self.tolerance_m_rad, (0.003, 0.003, 0.003, math.radians(2)))
        ):
            raise ValueError(
                "reference error-plus-uncertainty bounds cannot widen study tolerances"
            )
        if (
            not isinstance(self.posture_family, str)
            or not self.posture_family.strip()
            or len(self.posture_family) > 256
        ):
            raise ValueError("declared bounded posture family required")
        if type(self.max_frames) is not int or not 3 <= self.max_frames <= 16:
            raise ValueError("reference registration requires 3..16 bounded captures")
        if (
            isinstance(self.span_s, bool)
            or not isinstance(self.span_s, (int, float))
            or not math.isfinite(self.span_s)
            or not 0.3 <= self.span_s <= 2
        ):
            raise ValueError("reference window must span at least 0.3 seconds")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReferenceProfile:
        if (
            not isinstance(data, Mapping)
            or set(data) != set(cls.__dataclass_fields__)
            or not isinstance(data["tracking"], Mapping)
        ):
            raise ValueError("complete calibration_reference declaration required")
        try:
            return cls(**{**data, "tracking": TrackingConfig(**data["tracking"])})
        except TypeError as exc:
            raise ValueError("unknown reference tracking fields") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            **vars(self),
            "tracking": vars(self.tracking),
            "tolerance_m_rad": list(self.tolerance_m_rad),
        }


def validate_tracking_result(
    result: Mapping[str, Any],
    first: SettledObservation,
    second: SettledObservation,
    profile: ReferenceProfile,
    previous_ids: set[str] | None,
    *,
    previous_tracks: Sequence[Mapping[str, Any]] | None = None,
) -> set[str]:
    """Validate a stationary reference using the shared metric tracking boundary."""
    ids = validate_metric_tracking_result(
        result, first, second, profile.tracking, previous_ids, previous_tracks=previous_tracks
    )
    delta = finite_vector(result["delta_m_rad"], 4)
    uncertainty = finite_vector(result["uncertainty_m_rad"], 4)
    if any(abs(v) + u > cap for v, u, cap in zip(delta, uncertainty, profile.tolerance_m_rad)):
        raise ValueError(
            "scene moved or stationary reference uncertainty exceeds declared tolerance"
        )
    return ids


def validate_metric_tracking_result(
    result: Mapping[str, Any],
    first: SettledObservation,
    second: SettledObservation,
    config: TrackingConfig,
    previous_ids: set[str] | None = None,
    *,
    previous_tracks: Sequence[Mapping[str, Any]] | None = None,
) -> set[str]:
    """Recompute metric tracks for moving or stationary captures, never witnesses.

    Stationary-reference error limits belong to the reference wrapper. This shared
    boundary validates actual pixel/depth values, consensus and temporal identity
    without treating a legitimate measured correction as stationary scene drift.
    """
    _tracking_domain(first, second, config)
    _tracking_envelope(result, first, second)
    delta, uncertainty = _tracking_uncertainty(result, first, second, config)
    motion = _tracking_consensus(result["registration"], delta, uncertainty, config)
    ids = _validate_tracks(result, first, second, config, previous_ids, motion)
    if previous_tracks is not None:
        _validate_continuity(result, first, previous_tracks, previous_ids)
    return ids


def _tracking_envelope(
    result: Mapping[str, Any], first: SettledObservation, second: SettledObservation
) -> None:
    fields = {
        "schema",
        "first_frame_id",
        "second_frame_id",
        "first_digest",
        "second_digest",
        "profile_digest",
        "mount_digest",
        "session_id",
        "source_profile_digest",
        "provenance",
        "registration",
        "tracks",
        "persistent_tracks",
        "rejected_features",
        "delta_m_rad",
        "uncertainty_m_rad",
        "timing_uncertainty_s",
        "witness_valid",
        "rigid_link_identity_established",
        "hardware_accuracy_certified",
        "executed",
    }
    if (
        not isinstance(result, Mapping)
        or set(result) != fields
        or result.get("schema") != TRACK_SCHEMA
    ):
        raise ValueError("typed exact tracking result required")
    identities = {
        "first_frame_id": first.roi.frame_id,
        "second_frame_id": second.roi.frame_id,
        "first_digest": first.roi.digest,
        "second_digest": second.roi.digest,
        "profile_digest": first.roi.profile.digest,
        "mount_digest": first.roi.mount_digest,
        "session_id": first.roi.session_id,
        "source_profile_digest": first.roi.source_profile_digest,
        "provenance": first.roi.provenance,
    }
    if any(result.get(key) != value for key, value in identities.items()):
        raise ValueError("worker tracking result has different source identities")
    if any(
        result.get(key) is not False
        for key in (
            "executed",
            "hardware_accuracy_certified",
            "witness_valid",
            "rigid_link_identity_established",
        )
    ):
        raise ValueError(
            "tracking output cannot certify physical authority or independent witnesses"
        )


def _tracking_uncertainty(
    result: Mapping[str, Any],
    first: SettledObservation,
    second: SettledObservation,
    config: TrackingConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    delta, uncertainty = (
        finite_vector(result.get("delta_m_rad"), 4),
        finite_vector(result.get("uncertainty_m_rad"), 4),
    )
    if any(v <= 0 for v in uncertainty) or min(uncertainty[:3]) < config.depth_floor_m:
        raise ValueError("tracking uncertainty understates measurement floors")
    timing = result.get("timing_uncertainty_s")
    floor = sum(
        roi.stamp.uncertainty_s + roi.exposure_uncertainty_s for roi in (first.roi, second.roi)
    )
    if (
        isinstance(timing, bool)
        or not isinstance(timing, (int, float))
        or not math.isfinite(timing)
        or timing < floor
    ):
        raise ValueError("tracking timing floor is understated")
    return delta, uncertainty


def _tracking_consensus(
    registration: Mapping[str, Any],
    delta: Sequence[float],
    uncertainty: Sequence[float],
    config: TrackingConfig,
) -> Transform:
    if (
        not isinstance(registration, Mapping)
        or set(registration)
        != {
            "motion",
            "accepted_features",
            "rejected_features",
            "rms_m",
            "max_error_m",
            "spatial_rank",
            "association",
        }
        or registration.get("association") != "externally_identified_metric_correspondences"
    ):
        raise ValueError("typed exact geometric consensus required")
    rms, maximum = finite_vector((registration["rms_m"], registration["max_error_m"]), 2)
    if (
        not 0 <= rms <= maximum <= config.point_gate_m
        or min(uncertainty[:3]) < maximum
        or type(registration["spatial_rank"]) is not int
        or registration["spatial_rank"] not in (2, 3)
    ):
        raise ValueError("reference consensus residual/rank or uncertainty is invalid")
    motion_data = registration.get("motion")
    if not isinstance(motion_data, Mapping) or set(motion_data) != {
        "rotation",
        "translation_m",
        "parent",
        "child",
    }:
        raise ValueError("exact consensus transform required")
    rotation = motion_data["rotation"]
    if (
        not isinstance(rotation, (list, tuple))
        or len(rotation) != 3
        or motion_data["parent"] != ""
        or motion_data["child"] != ""
    ):
        raise ValueError("reference consensus requires unlabelled bounded rotation rows")
    motion = Transform(
        tuple(finite_vector(row, 3) for row in rotation),
        finite_vector(motion_data["translation_m"], 3),
    )
    expected = (*motion.translation_m, math.atan2(motion.rotation[1][0], motion.rotation[0][0]))
    if any(abs(a - b) > 1e-12 for a, b in zip(expected, delta)):
        raise ValueError("tracking coordinates disagree with consensus transform")
    return motion


def _tracking_domain(
    first: SettledObservation, second: SettledObservation, config: TrackingConfig
) -> None:
    if (
        not isinstance(first, SettledObservation)
        or not isinstance(second, SettledObservation)
        or not isinstance(config, TrackingConfig)
    ):
        raise ValueError("typed settled captures and tracking policy required")
    before, after = first.roi, second.roi
    fields = (
        "profile",
        "mount_digest",
        "session_id",
        "source_profile_digest",
        "provenance",
        "x",
        "y",
        "width",
        "height",
    )
    if first.profile_digest != second.profile_digest or any(
        getattr(before, key) != getattr(after, key) for key in fields
    ):
        raise ValueError("metric tracking camera domain changed")
    if (
        before.frame_id == after.frame_id
        or not 0 < after.stamp.seconds - before.stamp.seconds <= config.max_frame_gap_s
    ):
        raise ValueError("metric tracking frames repeated or exceeded the declared gap")
    if len(before.source_stamps) != len(after.source_stamps) or any(
        a.stream != b.stream
        or a.timestamp_domain != b.timestamp_domain
        or b.frame_number <= a.frame_number
        or b.timestamp_ms <= a.timestamp_ms
        for a, b in zip(before.source_stamps, after.source_stamps)
    ):
        raise ValueError("metric tracking native provenance repeated or changed")
    fields = (
        "device_id",
        "model_digest",
        "joint_ids",
        "ctrl_mode",
        "node_session_id",
        "declaration_digest",
    )
    anchor = first.feedback[0]
    seen: set[str] = set()
    last_receipt = 0.0
    for capture in (first, second):
        for sample in capture.feedback:
            if (
                any(getattr(anchor, key) != getattr(sample, key) for key in fields)
                or sample.source_id in seen
                or sample.first_receipt_s <= last_receipt
                or not sample.enabled
                or not sample.arrived
                or sample.arm_status != 0
                or sample.communication_errors
            ):
                raise ValueError("metric tracking raw feedback domain changed or windows overlap")
            seen.add(sample.source_id)
            last_receipt = sample.last_receipt_s
        uncertainty = capture.roi.stamp.uncertainty_s + capture.roi.exposure_uncertainty_s
        if (
            capture.feedback[-2].last_receipt_s > capture.roi.stamp.seconds - uncertainty
            or capture.feedback[-1].first_receipt_s < capture.roi.stamp.seconds + uncertainty
        ):
            raise ValueError("metric tracking feedback does not bracket uncertain capture")


def _validate_tracks(
    result: Mapping[str, Any],
    first: SettledObservation,
    second: SettledObservation,
    config: TrackingConfig,
    previous_ids: set[str] | None,
    motion: Transform,
) -> set[str]:
    rows, persistent, accepted = (
        result.get("tracks"),
        result.get("persistent_tracks"),
        result["registration"].get("accepted_features"),
    )
    if (
        not isinstance(rows, (list, tuple))
        or not config.min_tracks <= len(rows) <= config.max_tracks
    ):
        raise ValueError("bounded distributed reference support required")
    if (
        not isinstance(persistent, (list, tuple))
        or len(persistent) != len(rows)
        or not isinstance(accepted, (list, tuple))
        or len(accepted) != len(rows)
    ):
        raise ValueError("reference consensus and persistent support differ")
    ids: set[str] = set()
    pixels, residuals, floors = [], [], []
    for row, track in zip(rows, persistent):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "feature_id",
                "first_pixel",
                "second_pixel",
                "first_m",
                "second_m",
                "depth_uncertainty_m",
            }
            or not isinstance(track, Mapping)
        ):
            raise ValueError("typed reference tracks required")
        identity = row.get("feature_id")
        if not isinstance(identity, str) or not identity or len(identity) > 256 or identity in ids:
            raise ValueError("distinct bounded reference feature IDs required")
        if track != {
            "feature_id": identity,
            "frame_id": second.roi.frame_id,
            "pixel": row.get("second_pixel"),
        }:
            raise ValueError("persistent reference track does not bind the final pixel/frame")
        for key, roi in (("first_pixel", first.roi), ("second_pixel", second.roi)):
            x, y = finite_vector(row.get(key), 2)
            if (
                not roi.x + 4 <= x < roi.x + roi.width - 4
                or not roi.y + 4 <= y < roi.y + roi.height - 4
            ):
                raise ValueError("reference track left its sealed ROI")
        xyz_a, xyz_b = finite_vector(row.get("first_m"), 3), finite_vector(row.get("second_m"), 3)
        depth_floor = finite_vector((row.get("depth_uncertainty_m"),), 1)[0]
        measured_a, floor_a = metric_track_point(first.roi, row["first_pixel"], config)
        measured_b, floor_b = metric_track_point(second.roi, row["second_pixel"], config)
        if math.dist(xyz_a, measured_a) > 1e-10 or math.dist(xyz_b, measured_b) > 1e-10:
            raise ValueError("reference track geometry differs from its indexed raw depth/pixel")
        residual = math.dist(motion.apply(xyz_a), xyz_b)
        if depth_floor + 1e-12 < max(floor_a, floor_b) or residual > config.point_gate_m:
            raise ValueError("reference track depth or consensus geometry is invalid")
        residuals.append(residual)
        floors.append(depth_floor)
        pixels.append(finite_vector(row["second_pixel"], 2))
        ids.add(identity)
    accepted_ids = _feature_ids(accepted, config.max_tracks)
    rejected = _feature_ids(result["rejected_features"], config.max_tracks)
    consensus_rejected = _feature_ids(
        result["registration"]["rejected_features"], config.max_tracks
    )
    if (
        accepted_ids != ids
        or ids & rejected
        or not consensus_rejected <= rejected
        or len(ids | rejected) > config.max_tracks
    ):
        raise ValueError("reference accepted/rejected feature provenance differs")
    if previous_ids is not None and ids | rejected != previous_ids:
        raise ValueError("reference feature continuity changed or silently reacquired")
    _validate_residuals(result, rows, residuals, floors, config)
    for axis, size in enumerate((second.roi.width, second.roi.height)):
        if (
            max(p[axis] for p in pixels) - min(p[axis] for p in pixels)
            < size * config.min_spread_fraction
        ):
            raise ValueError("reference support is not distributed in both image dimensions")
    mean = tuple(sum(p[i] for p in pixels) / len(pixels) for i in range(2))
    xx, yy = (sum((p[i] - mean[i]) ** 2 for p in pixels) for i in range(2))
    xy = sum((p[0] - mean[0]) * (p[1] - mean[1]) for p in pixels)
    smallest = max(0, (xx + yy - math.sqrt((xx - yy) ** 2 + 4 * xy**2)) / 2)
    if math.sqrt(smallest) < config.min_distance_px:
        raise ValueError("collinear reference support cannot establish yaw")
    return ids


def _feature_ids(values: Any, limit: int) -> set[str]:
    if (
        not isinstance(values, (list, tuple))
        or len(values) > limit
        or any(not isinstance(value, str) or not value or len(value) > 256 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("bounded distinct accepted/rejected feature identities required")
    return set(values)


def _validate_residuals(
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    residuals: Sequence[float],
    floors: Sequence[float],
    config: TrackingConfig,
) -> None:
    registration = result["registration"]
    rms = math.sqrt(sum(v * v for v in residuals) / len(residuals))
    if (
        abs(rms - registration["rms_m"]) > 1e-12
        or abs(max(residuals) - registration["max_error_m"]) > 1e-12
    ):
        raise ValueError(
            "reference reported residual differs from measured correspondence residuals"
        )
    floor = max(max(floors), max(residuals))
    centre = tuple(sum(row["first_m"][i] for row in rows) / len(rows) for i in range(2))
    radius = max(math.dist(row["first_m"][:2], centre) for row in rows)
    uncertainty = result["uncertainty_m_rad"]
    if min(uncertainty[:3]) + 1e-12 < floor or uncertainty[3] + 1e-12 < floor / max(
        radius, config.depth_floor_m
    ):
        raise ValueError("reference uncertainty understates actual depth/consensus/yaw support")
    if abs(result["delta_m_rad"][3]) > config.max_yaw_rad:
        raise ValueError("reference yaw left its declared tracking branch")


def _validate_continuity(
    result: Mapping[str, Any],
    first: SettledObservation,
    previous: Sequence[Mapping[str, Any]],
    previous_ids: set[str] | None,
) -> None:
    if not isinstance(previous, (list, tuple)) or len(previous) > 2048:
        raise ValueError("bounded preceding reference tracks required")
    pixels = {}
    for track in previous:
        if (
            not isinstance(track, Mapping)
            or set(track) != {"feature_id", "frame_id", "pixel"}
            or track["frame_id"] != first.roi.frame_id
        ):
            raise ValueError("preceding reference tracks belong to a different source frame")
        identity = track["feature_id"]
        _feature_ids([identity], 1)
        if identity in pixels:
            raise ValueError("preceding reference feature identities repeat")
        pixels[identity] = finite_vector(track["pixel"], 2)
    if set(pixels) != previous_ids:
        raise ValueError("preceding reference feature provenance differs")
    for row in result["tracks"]:
        # LK input is explicitly float32 in local ROI coordinates, not a new seed.
        expected = tuple(
            struct.unpack("<f", struct.pack("<f", value - offset))[0] + offset
            for value, offset in zip(pixels[row["feature_id"]], (first.roi.x, first.roi.y))
        )
        if math.dist(expected, finite_vector(row["first_pixel"], 2)) > 1e-6:
            raise ValueError("reference feature was switched at the shared source frame")


def assess_reference(
    observations: Sequence[SettledObservation],
    results: Sequence[Mapping[str, Any]],
    *,
    profile: ReferenceProfile,
    settled: SettledProfile,
) -> dict[str, Any]:
    """Recheck the whole stationary window, not isolated pairwise training residuals."""
    if not 3 <= len(observations) <= profile.max_frames or len(results) != len(observations) - 1:
        raise ValueError(
            "reference needs at least three actual captures and adjacent tracking results"
        )
    first, last = observations[0], observations[-1]
    if last.roi.stamp.seconds - first.roi.stamp.seconds < profile.span_s:
        raise ValueError("reference fresh frames do not span the frozen window")
    anchor = first.feedback[0]
    previous_ids = None
    cumulative = Transform()
    accumulated_uncertainty = [0.0] * 4
    for i, observation in enumerate(observations):
        roi = observation.roi
        if (
            roi.profile,
            roi.mount_digest,
            roi.session_id,
            roi.source_profile_digest,
            roi.provenance,
            roi.x,
            roi.y,
            roi.width,
            roi.height,
        ) != (
            first.roi.profile,
            first.roi.mount_digest,
            first.roi.session_id,
            first.roi.source_profile_digest,
            first.roi.provenance,
            first.roi.x,
            first.roi.y,
            first.roi.width,
            first.roi.height,
        ):
            raise ValueError("reference camera domain changed")
        for sample in observation.feedback:
            if (
                sample.device_id,
                sample.model_digest,
                sample.joint_ids,
                sample.ctrl_mode,
                sample.node_session_id,
                sample.declaration_digest,
            ) != (
                anchor.device_id,
                anchor.model_digest,
                anchor.joint_ids,
                anchor.ctrl_mode,
                anchor.node_session_id,
                anchor.declaration_digest,
            ):
                raise ValueError("reference raw-feedback domain changed")
            if (
                max(abs(a - b) for a, b in zip(anchor.joints_rad, sample.joints_rad))
                + anchor.joint_uncertainty_rad
                + sample.joint_uncertainty_rad
                > settled.max_joint_drift_rad
            ):
                raise ValueError("reference starting raw posture drifted across the window")
            if (
                anchor.tcp.distance_mm(sample.tcp)
                + anchor.tcp_uncertainty_mm
                + sample.tcp_uncertainty_mm
                > settled.max_tcp_drift_mm
                or anchor.tcp.rotation_to_deg(sample.tcp)
                + anchor.rotation_uncertainty_deg
                + sample.rotation_uncertainty_deg
                > settled.max_tcp_rotation_deg
            ):
                raise ValueError("reference starting flange posture drifted across the window")
        if not i:
            continue
        prior = observations[i - 1]
        gap = roi.stamp.seconds - prior.roi.stamp.seconds
        if (
            roi.frame_id == prior.roi.frame_id
            or not 0 < gap <= profile.tracking.max_frame_gap_s
            or observation.feedback[0].first_receipt_s <= prior.feedback[-1].last_receipt_s
        ):
            raise ValueError("reference captures are repeated, stale, or share feedback windows")
        if any(
            a.stream != b.stream
            or a.timestamp_domain != b.timestamp_domain
            or b.frame_number <= a.frame_number
            or b.timestamp_ms <= a.timestamp_ms
            for a, b in zip(prior.roi.source_stamps, roi.source_stamps)
        ):
            raise ValueError("reference native provenance repeated or changed")
        result = results[i - 1]
        previous_ids = validate_tracking_result(
            result,
            prior,
            observation,
            profile,
            previous_ids,
            previous_tracks=results[i - 2]["persistent_tracks"] if i > 1 else None,
        )
        cumulative = Transform.from_mapping(result["registration"]["motion"]).compose(cumulative)
        accumulated_uncertainty = [
            a + b for a, b in zip(accumulated_uncertainty, result["uncertainty_m_rad"])
        ]
        delta = (
            *cumulative.translation_m,
            math.atan2(cumulative.rotation[1][0], cumulative.rotation[0][0]),
        )
        if any(
            abs(v) + u > cap
            for v, u, cap in zip(delta, accumulated_uncertainty, profile.tolerance_m_rad)
        ):
            raise ValueError(
                "accumulated scene drift/uncertainty exceeds stationary reference bounds"
            )
    return {
        "schema": SCHEMA,
        "reference_frame_id": last.roi.frame_id,
        "reference_roi_digest": last.roi.digest,
        "profile": profile.to_dict(),
        "model_digest": anchor.model_digest,
        "joint_ids": list(anchor.joint_ids),
        "raw_joints_rad": list(last.feedback[-1].joints_rad),
        "nominal_flange_mm_deg": last.feedback[-1].tcp.to_dict(),
        "camera_profile_digest": last.roi.profile.digest,
        "mount_digest": last.roi.mount_digest,
        "camera_session_id": last.roi.session_id,
        "source_profile_digest": last.roi.source_profile_digest,
        "persistent_tracks": results[-1]["persistent_tracks"],
        "frames": len(observations),
        "span_s": last.roi.stamp.seconds - first.roi.stamp.seconds,
        "tracking_result_digests": [digest_of(r) for r in results],
        "uncertainty_m_rad": accumulated_uncertainty,
        "state": "REGISTERED",
        "independent_witness": False,
        "clearance_verified": False,
        "physical_activation": "unqualified",
        "hardware_accuracy_certified": False,
        "executed": False,
    }

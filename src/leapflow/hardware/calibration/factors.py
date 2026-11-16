# Copyright (c) Alibaba, Inc. and its affiliates.
"""Reconstruct conditional motion values from indexed captures, without promoting scope.

Raw source lineage does not authenticate a caller's relative flange transform or
metric correspondence values. This owner-internal boundary checks those values
against the complete private captures. It does not establish image association,
a stationary-scene witness, carried-link identity, interval peak rates, absolute
geometry, or activation authority. Nominal firmware flange semantics stay frozen.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.identity import TopologyIdentity, digest_of
from leapcalib.contracts.motion import MODE_FIXED, MotionObservation, finite_vector
from leapcalib.contracts.observations import JointObservation, RgbdObservation, Stamp
from leapcalib.contracts.raw_task import (
    MetricTaskFeature,
    RawTaskState,
    RawTaskTransition,
    identities,
)
from leapcalib.contracts.task import TaskState
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.geometry.se3 import Transform
from leapcalib.validation.bundle import EvidenceBundle
from leapcalib.validation.evaluation import (
    CandidateEvaluation,
    EvaluationConfig,
    MotionMeasurementBounds,
    evaluate_motion_candidate,
)
from leapflow.hardware.calibration.generations import _json
from leapflow.hardware.calibration.acquisition import SettledObservation
from leapflow.hardware.calibration.provenance import (
    IndexedCaptureReader,
    IndexedSourceManifest,
    independent_indexed_holdouts,
    indexed_source_manifest,
)
from leapflow.hardware.calibration.reference import validate_metric_tracking_result
from leapflow.hardware.calibration.measurements import metric_track_point

SCHEMA = "leapflow.indexed_motion_factors.v1"
MAX_CAPTURES = 128


@dataclass(frozen=True)
class MotionFactorCheck:
    observation_id: str
    observation_digest: str
    tracking_digest: str
    pose_source_ids: tuple[str, str]
    point_uncertainty_m: float
    robot_translation_uncertainty_m: float
    robot_rotation_uncertainty_rad: float

    def __post_init__(self) -> None:
        for value in (self.observation_id, self.observation_digest, self.tracking_digest):
            identities((value,))
        object.__setattr__(
            self, "pose_source_ids", identities(self.pose_source_ids, minimum=2, maximum=2)
        )
        values = finite_vector(
            (
                self.point_uncertainty_m,
                self.robot_translation_uncertainty_m,
                self.robot_rotation_uncertainty_rad,
            ),
            3,
        )
        if any(value <= 0 for value in values):
            raise ValueError("indexed factor floors cannot be zero or negative")
        for name, value in zip(
            (
                "point_uncertainty_m",
                "robot_translation_uncertainty_m",
                "robot_rotation_uncertainty_rad",
            ),
            values,
        ):
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "pose_source_ids": list(self.pose_source_ids)}


@dataclass(frozen=True)
class IndexedMotionFactors:
    """Immutable checked values and measurement floors, never an admission receipt."""

    manifest: IndexedSourceManifest
    tracking: TrackingConfig
    checks: tuple[MotionFactorCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, IndexedSourceManifest) or not isinstance(
            self.tracking, TrackingConfig
        ):
            raise ValueError("typed indexed source manifest and tracking policy required")
        if (
            not isinstance(self.checks, (list, tuple))
            or not 1 <= len(self.checks) <= 1024
            or any(not isinstance(check, MotionFactorCheck) for check in self.checks)
        ):
            raise ValueError("bounded typed indexed motion factor checks required")
        object.__setattr__(self, "checks", tuple(self.checks))
        identities(tuple(c.observation_id for c in self.checks), maximum=1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "method_id": "metric_motion.v1",
            "evidence_digest": self.manifest.evidence_digest,
            "source_manifest_digest": self.manifest.digest,
            "tracking": vars(self.tracking),
            "factors": [c.to_dict() for c in self.checks],
            "scope": "conditional_camera_motion",
            "robot_pose_semantics": "nominal_firmware_flange",
            "feature_association": "not_established",
            "independent_witness": "not_established",
            "rigid_link_identity": "not_established",
            "interval_peak_rates": "not_established",
            "physical_admission": "not_established",
            "global_geometry": "not_established",
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())

    def validate_measurement_bounds(self, bounds: Mapping[str, MotionMeasurementBounds]) -> None:
        """Reject understated indexed floors; rate/association provenance stays unqualified."""
        checks = {c.observation_id: c for c in self.checks}
        if not isinstance(bounds, Mapping) or not set(bounds) <= set(checks):
            raise ValueError("measurement bounds refer to unchecked motion factors")
        for identity, bound in bounds.items():
            check = checks[identity]
            if (
                not isinstance(bound, MotionMeasurementBounds)
                or bound.observation_digest != check.observation_digest
            ):
                raise ValueError("measurement bound differs from exact checked motion factor")
            if any(
                supplied < measured
                for supplied, measured in (
                    (bound.point_uncertainty_m, check.point_uncertainty_m),
                    (bound.robot_translation_uncertainty_m, check.robot_translation_uncertainty_m),
                    (bound.robot_rotation_uncertainty_rad, check.robot_rotation_uncertainty_rad),
                )
            ):
                raise ValueError("motion measurement bounds understate indexed measurement floors")


class _CaptureSnapshot:
    """One bounded read per acquisition; manifests and factors use the same bytes."""

    def __init__(self, store: IndexedCaptureReader) -> None:
        self._store = store
        self.contents: dict[tuple[str, str], dict[str, Any]] = {}
        self.captures: dict[tuple[str, str], SettledObservation] = {}

    def read(self, identity: str, *, workcell_id: str) -> dict[str, Any]:
        key = (identity, workcell_id)
        if key not in self.contents:
            if len(self.contents) >= MAX_CAPTURES:
                raise ValueError("indexed factor capture count exceeds the bounded keyframe budget")
            content = self._store.read(identity, workcell_id=workcell_id)
            self.captures[key] = SettledObservation.from_mapping(content["observation"])
            self.contents[key] = content
        return self.contents[key]


def _flange(capture: SettledObservation) -> tuple[Transform, float, float]:
    # The last pre-capture sample is deterministic and bracketed by fresh feedback.
    # Include measured cohort drift and uncertainty instead of claiming exposure FK.
    anchor = capture.feedback[-2]
    pose = anchor.tcp
    transform = Transform(pose.rotation, tuple(value / 1000 for value in pose.position))
    translation = (
        max(pose.distance_mm(f.tcp) + f.tcp_uncertainty_mm for f in capture.feedback) / 1000
    )
    rotation = math.radians(
        max(pose.rotation_to_deg(f.tcp) + f.rotation_uncertainty_deg for f in capture.feedback)
    )
    return transform, translation, rotation


def _check_motion(
    factor: MotionObservation,
    first: SettledObservation,
    second: SettledObservation,
    result: Mapping[str, Any],
    config: TrackingConfig,
) -> MotionFactorCheck:
    validate_metric_tracking_result(result, first, second, config)
    expected_stamps = tuple(_capture_stamp(capture) for capture in (first, second))
    if (
        factor.provenance != "metric_rgbd"
        or factor.settled is not True
        or factor.stream != first.roi.profile
        or factor.model.digest != first.feedback[0].model_digest
        or (factor.first_id, factor.second_id) != (first.roi.frame_id, second.roi.frame_id)
        or (factor.first_stamp, factor.stamp) != expected_stamps
    ):
        raise ValueError("motion factor differs from indexed capture/model/clock/stillness")
    tracks = {row["feature_id"]: row for row in result["tracks"]}
    if {point.feature_id for point in factor.points} != set(tracks) or any(
        math.dist(point.first_m, tracks[point.feature_id]["first_m"]) > 1e-10
        or math.dist(point.second_m, tracks[point.feature_id]["second_m"]) > 1e-10
        for point in factor.points
    ):
        raise ValueError(
            "motion correspondence values differ from checked indexed pixel/depth tracks"
        )
    before, t_first, r_first = _flange(first)
    after, t_second, r_second = _flange(second)
    expected = (
        after.compose(before.inverse())
        if factor.mode == MODE_FIXED
        else before.inverse().compose(after)
    )
    if any(
        abs(a - b) > 1e-12
        for a, b in zip(expected.translation_m, factor.robot_motion.translation_m)
    ) or any(
        abs(a - b) > 1e-12
        for expected_row, actual_row in zip(expected.rotation, factor.robot_motion.rotation)
        for a, b in zip(expected_row, actual_row)
    ):
        raise ValueError("relative robot motion differs from indexed nominal flange feedback")
    # Relative translation rotation error also acts on the flange lever arm.
    lever = (
        math.dist(before.translation_m, (0, 0, 0))
        if factor.mode == MODE_FIXED
        else math.dist(after.translation_m, before.translation_m)
    )
    angular = r_first + r_second
    return MotionFactorCheck(
        factor.observation_id,
        factor.digest,
        digest_of(result),
        (first.feedback[-2].source_id, second.feedback[-2].source_id),
        max(result["uncertainty_m_rad"][:3]),
        t_first + t_second + 2 * lever * math.sin(min(math.pi, angular) / 2),
        angular,
    )


def _capture_stamp(capture: SettledObservation) -> Stamp:
    return replace(
        capture.roi.stamp,
        uncertainty_s=capture.roi.stamp.uncertainty_s + capture.roi.exposure_uncertainty_s,
    )


def indexed_motion_factors(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    tracking_results: Mapping[str, Mapping[str, Any]],
    *,
    config: TrackingConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedMotionFactors:
    """Recheck all derived factors, including rejected/unused rows, off the owner loop.

    A manually constructed manifest is deliberately not an input. Every call
    reopens unexpired original acquisitions and verifies historical signatures.
    No caller-provided tracking or scene flag can qualify a physical parameter.
    """
    if (
        not isinstance(store, IndexedCaptureReader)
        or not isinstance(bundle, EvidenceBundle)
        or not isinstance(config, TrackingConfig)
    ):
        raise ValueError("typed owner reader, evidence and tracking policy required")
    if any(
        not isinstance(o, (MotionObservation, RgbdObservation, JointObservation))
        for o in bundle.observations
    ):
        raise ValueError("conditional motion reconstruction cannot mix unrelated evidence methods")
    factors = [o for o in bundle.observations if isinstance(o, MotionObservation)]
    if (
        not 1 <= len(factors) <= 1024
        or not isinstance(tracking_results, Mapping)
        or set(tracking_results) != {o.observation_id for o in factors}
    ):
        raise ValueError("every derived motion factor requires exact checked tracking lineage")
    snapshot = _CaptureSnapshot(store)
    manifest = indexed_source_manifest(
        snapshot, bundle, acquisition_ids, trusted_keys=trusted_keys, node_id=node_id
    )
    checks = []
    for factor in sorted(factors, key=lambda o: o.observation_id):
        captures = [
            snapshot.captures[(acquisition_ids[identity], bundle.topology.workcell_id)]
            for identity in factor.source_ids
        ]
        checks.append(
            _check_motion(factor, *captures, tracking_results[factor.observation_id], config)
        )
    return IndexedMotionFactors(manifest, config, tuple(checks))


@dataclass(frozen=True)
class IndexedMotionAssessment:
    """Owner-recomputed conditional assessment, not a partial physical certificate.

    All source, factor, candidate and withheld gates share exact immutable inputs.
    Constructing this value manually grants no authority; actual admission must
    reopen the original evidence and establish the remaining physical gates.
    """

    candidate_json: str
    factors: IndexedMotionFactors
    evaluation: CandidateEvaluation
    indexed_holdout_ids: tuple[str, ...]
    indexed_exclusions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_json, str)
            or len(self.candidate_json.encode()) > 2 * 1024 * 1024
        ):
            raise ValueError("bounded immutable indexed candidate required")
        fit = EstimateBundle.from_mapping(json.loads(self.candidate_json))
        if not isinstance(self.factors, IndexedMotionFactors) or not isinstance(
            self.evaluation, CandidateEvaluation
        ):
            raise ValueError("typed indexed factors and recomputed evaluation required")
        manifest, evaluation = self.factors.manifest, self.evaluation
        if (
            fit.digest != evaluation.candidate_digest
            or fit.metadata.get("evidence_digest") != manifest.evidence_digest
            or evaluation.evidence_digest != manifest.evidence_digest
            or evaluation.baseline_evidence_digest != manifest.evidence_digest
            or fit.topology.digest != manifest.topology_digest
            or fit.method_id != evaluation.method_id
            or evaluation.method_id != "metric_motion.v1"
            or evaluation.scope != "conditional_camera_motion"
        ):
            raise ValueError("indexed assessment candidate/evidence/method/scope binding mismatch")
        object.__setattr__(
            self,
            "indexed_holdout_ids",
            identities(self.indexed_holdout_ids, minimum=0, maximum=1024),
        )
        # Reuse the exact bounded exclusion contract, without deriving a verdict
        # from the caller's independence list or erasing numerical exclusions.
        exclusions = replace(evaluation, units=(), exclusions=self.indexed_exclusions).exclusions
        object.__setattr__(self, "indexed_exclusions", exclusions)
        allowed, excluded = set(self.indexed_holdout_ids), {i for i, _ in exclusions}
        units, numerical_exclusions = (
            {u.observation_id for u in evaluation.units},
            {i for i, _ in evaluation.exclusions},
        )
        checks = {c.observation_id for c in self.factors.checks}
        if (
            allowed & excluded
            or allowed | excluded != set(fit.holdout_ids)
            or not units <= allowed
            or not excluded <= numerical_exclusions
            or units | numerical_exclusions != set(fit.holdout_ids)
            or not set((*fit.accepted_ids, *fit.rejected_ids, *fit.holdout_ids)) <= checks
        ):
            raise ValueError("indexed assessment must account for every split/withheld branch")

    def to_dict(self) -> dict[str, Any]:
        fit = EstimateBundle.from_mapping(json.loads(self.candidate_json))
        return {
            "schema": "leapflow.indexed_motion_assessment.v1",
            "method_id": fit.method_id,
            "scope": "conditional_camera_motion",
            "candidate_digest": fit.digest,
            "evidence_digest": self.factors.manifest.evidence_digest,
            "topology_digest": fit.topology.digest,
            "nominal_model_digest": fit.topology.model.digest,
            "device_digest": fit.topology.device.digest,
            "mount_digests": [m.digest for m in fit.topology.mounts],
            "profile_digests": [s.digest for s in fit.topology.streams],
            "supported_blocks": list(fit.supported_blocks),
            "partial_blocks": list(fit.partial_blocks),
            "retained_blocks": list(fit.retained_blocks),
            "estimates": [e.to_dict() for e in fit.estimates],
            "conditioning": {
                k: fit.metadata[k]
                for k in (
                    "data_rank",
                    "condition",
                    "singular_values",
                    "nullspace_scaled",
                    "prior_rows_in_rank",
                    "consensus_stable",
                )
            },
            "prior_transform": fit.metadata["prior_transform"],
            "candidate_transform": fit.metadata["candidate_transform"],
            "gauge_assumptions": list(fit.limitations),
            "source_manifest": self.factors.manifest.to_dict(),
            "factor_digest": self.factors.digest,
            "evaluation": self.evaluation.to_dict(),
            "indexed_holdout_ids": list(self.indexed_holdout_ids),
            "indexed_exclusions": [
                {"observation_id": i, "reason": r} for i, r in self.indexed_exclusions
            ],
            "qualified_consumers": [],
            "physical_admission": "not_established",
            "unmet_physical_gates": [
                "independent_feature_association",
                "independent_scene_or_carried_link_witness",
                "measured_interval_peak_rates",
                "independently_anchored_geometry",
                "required_consumer_compatibility",
                "uncertainty_inflated_clearance",
            ],
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def _indexed_evaluation(
    evaluation: CandidateEvaluation,
    selected: tuple[str, ...],
    exclusions: tuple[tuple[str, str], ...],
) -> CandidateEvaluation:
    """Intersect independent units and preserve both reasons for an excluded branch."""
    rejected = dict(evaluation.exclusions)
    for identity, reason in exclusions:
        rejected[identity] = f"{rejected[identity]}; {reason}" if identity in rejected else reason
    allowed = set(selected)
    return replace(
        evaluation,
        units=tuple(u for u in evaluation.units if u.observation_id in allowed),
        exclusions=tuple(sorted(rejected.items())),
    )


def assess_indexed_motion(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    fit: EstimateBundle,
    acquisition_ids: Mapping[str, str],
    tracking_results: Mapping[str, Mapping[str, Any]],
    *,
    measurement_bounds: Mapping[str, MotionMeasurementBounds],
    tracking: TrackingConfig,
    config: EvaluationConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedMotionAssessment:
    """Connect raw/derived/fit/withheld checks, off-loop and without persisting authority.

    Source and factor checks are recomputed, never accepted as an input certificate.
    A retained/partial fit remains an explicit insufficient result, not a bypass.
    Real physical admission and consumer compatibility are deliberately separate.
    """
    if not isinstance(fit, EstimateBundle) or not isinstance(config, EvaluationConfig):
        raise ValueError("typed immutable fit and evaluation policy required")
    from leapcalib.estimation.numeric import SolveConfig

    solve = SolveConfig(**fit.metadata["config"])
    if (
        solve.max_pairs > MAX_CAPTURES
        or solve.max_points > 2048
        or solve.trials > 48
        or solve.refinement_steps > 8
    ):
        raise ValueError("owner assessment exceeds integrated numerical work budgets")
    # Freeze mutable nested containers before any raw read or numerical call.
    bundle = EvidenceBundle.from_mapping(json.loads(_json(bundle.to_dict())))
    candidate_json = _json(fit.to_dict())
    fit = EstimateBundle.from_mapping(json.loads(candidate_json))
    if not isinstance(measurement_bounds, Mapping) or set(measurement_bounds) != set(
        bundle.holdout_ids
    ):
        raise ValueError("every withheld factor needs exactly one typed measurement bound")
    bounds = dict(measurement_bounds)
    results = json.loads(_json(tracking_results))
    checked = indexed_motion_factors(
        store,
        bundle,
        dict(acquisition_ids),
        results,
        config=tracking,
        trusted_keys=dict(trusted_keys),
        node_id=node_id,
    )
    checked.validate_measurement_bounds(bounds)
    selected, exclusions = independent_indexed_holdouts(
        bundle,
        checked.manifest,
        separation_s=config.separation_s,
    )
    evaluation = evaluate_motion_candidate(bundle, fit, measurement_bounds=bounds, config=config)
    return IndexedMotionAssessment(
        candidate_json,
        checked,
        _indexed_evaluation(evaluation, selected, exclusions),
        selected,
        exclusions,
    )


def metric_raw_states(
    first: SettledObservation,
    second: SettledObservation,
    result: Mapping[str, Any],
    feature: MetricTaskFeature,
    topology: TopologyIdentity,
    *,
    config: TrackingConfig,
) -> tuple[RawTaskState, RawTaskState]:
    """Reconstruct local camera features and nominal/raw feedback, never a witness.

    The feature origin is a frozen track centroid, not registration translation
    or absolute tool position. Its directed XY yaw is separately observable.
    A false witness is deliberate: numeric tracking cannot promote it to true.
    Exposure-time motion/rate bounds and independent association remain separate.
    """
    if not isinstance(feature, MetricTaskFeature) or not isinstance(topology, TopologyIdentity):
        raise ValueError("typed metric task feature and topology required")
    validate_metric_tracking_result(result, first, second, config)
    tracks = {r["feature_id"]: r for r in result["tracks"]}
    if not set(feature.track_ids) <= set(tracks):
        raise ValueError("frozen task feature lost or switched support")
    for capture in (first, second):
        if (
            capture.roi.profile not in topology.streams
            or capture.feedback[0].model_digest != topology.model.digest
            or capture.feedback[0].device_id != topology.device.device_id
            or capture.roi.mount_digest not in {m.digest for m in topology.mounts}
        ):
            raise ValueError("metric task capture differs from exact nominal topology")
    scope = digest_of(
        {
            "schema": "leapflow.metric_raw_task_scope.v1",
            "topology": topology.digest,
            "feature": feature.to_dict(),
            "node_session": first.feedback[0].node_session_id,
            "declaration": first.feedback[0].declaration_digest,
            "mount": first.roi.mount_digest,
            "camera_session": first.roi.session_id,
            "source_profile": first.roi.source_profile_digest,
            "acquisition_profile": first.profile_digest,
            "region": [first.roi.x, first.roi.y, first.roi.width, first.roi.height],
        }
    )
    return (
        _metric_raw_state(first, tracks, "first", feature, scope, config, result),
        _metric_raw_state(second, tracks, "second", feature, scope, config, result),
    )


def _metric_raw_state(
    capture: SettledObservation,
    tracks: Mapping[str, Mapping[str, Any]],
    side: str,
    feature: MetricTaskFeature,
    scope: str,
    config: TrackingConfig,
    result: Mapping[str, Any],
) -> RawTaskState:
    roi, anchor = capture.roi, capture.feedback[-2]
    if roi.profile.distortion_model != "none":
        raise ValueError("task feature pixel uncertainty needs an undistorted exact profile")
    pixels = [tracks[i][side + "_pixel"] for i in feature.track_ids]
    if any(
        max(p[axis] for p in pixels) - min(p[axis] for p in pixels)
        < config.min_spread_fraction * extent
        for axis, extent in enumerate((roi.width, roi.height))
    ):
        raise ValueError("frozen task support lacks two-dimensional image spread")
    points, floor = {}, math.sqrt(3) * max(result["uncertainty_m_rad"][:3])
    for identity, pixel in zip(feature.track_ids, pixels):
        point, depth_floor = metric_track_point(roi, pixel, config)
        points[identity] = point
        # Do not average away common depth/pixel floors when forming a centroid.
        floor = max(
            floor,
            depth_floor * (1 + (abs(point[0]) + abs(point[1])) / point[2])
            + math.sqrt(2)
            * config.fb_gate_px
            * point[2]
            / min(roi.profile.fx_px, roi.profile.fy_px),
        )
    a, b = (points[i] for i in feature.yaw_ids)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= 2 * floor:
        raise ValueError("task yaw axis is unresolved or ambiguous at the measured floor")
    yaw_floor = math.asin(2 * floor / length)
    nominal, tcp_floor, rotation_floor = _flange(capture)
    if math.hypot(nominal.rotation[0][0], nominal.rotation[1][0]) <= 1e-6:
        raise ValueError("nominal flange yaw is singular in the four-coordinate task")
    task = TaskState(
        digest_of(
            {
                "schema": "leapflow.metric_raw_state.v1",
                "capture": capture.digest,
                "feature": feature.feature_id,
            }
        ),
        _capture_stamp(capture),
        (*nominal.translation_m, math.atan2(nominal.rotation[1][0], nominal.rotation[0][0])),
        (
            *(math.fsum(p[k] for p in points.values()) / len(points) for k in range(3)),
            math.atan2(dy, dx),
        ),
        feature.feature_id,
        feature.domain_id,
        True,
        True,
        (max(floor, tcp_floor),) * 3
        + (max(yaw_floor, rotation_floor, result["uncertainty_m_rad"][3]),),
    )
    uncertainty = tuple(
        max(abs(q - f.joints_rad[k]) + f.joint_uncertainty_rad for f in capture.feedback)
        for k, q in enumerate(anchor.joints_rad)
    )
    return RawTaskState(
        task,
        anchor.joint_ids,
        anchor.joints_rad,
        uncertainty,
        (roi.frame_id,),
        scope,
        anchor.stamp,
        False,
        tuple(f.source_id for f in capture.feedback),
    )


@dataclass(frozen=True)
class RawTaskFactorCheck:
    observation_id: str
    observation_digest: str
    tracking_digest: str
    state_digests: tuple[str, str]

    def __post_init__(self) -> None:
        for value in (self.observation_id, self.observation_digest, self.tracking_digest):
            identities((value,))
        object.__setattr__(
            self, "state_digests", identities(self.state_digests, minimum=2, maximum=2)
        )

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "state_digests": list(self.state_digests)}


@dataclass(frozen=True)
class IndexedRawTaskFactors:
    """Checked encoder/local feature values; unverified witness flags are rejected."""

    manifest: IndexedSourceManifest
    tracking: TrackingConfig
    feature: MetricTaskFeature
    checks: tuple[RawTaskFactorCheck, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, IndexedSourceManifest)
            or not isinstance(self.tracking, TrackingConfig)
            or not isinstance(self.feature, MetricTaskFeature)
        ):
            raise ValueError("typed indexed raw task manifest/policies required")
        if (
            not isinstance(self.checks, (list, tuple))
            or not 1 <= len(self.checks) <= 1024
            or any(not isinstance(c, RawTaskFactorCheck) for c in self.checks)
        ):
            raise ValueError("bounded typed raw task checks required")
        object.__setattr__(self, "checks", tuple(self.checks))
        identities(tuple(c.observation_id for c in self.checks), maximum=1024)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "leapflow.indexed_raw_task_factors.v1",
            "method_id": "raw_joint_response.v1",
            "scope": "local_camera_feature_raw_joint_values",
            "feature": self.feature.to_dict(),
            "source_manifest_digest": self.manifest.digest,
            "evidence_digest": self.manifest.evidence_digest,
            "tracking": vars(self.tracking),
            "factors": [c.to_dict() for c in self.checks],
            "pose_semantics": "nominal_firmware_flange",
            "feature_association": "not_established",
            "independent_witness": "not_established",
            "interval_peak_rates": "not_established",
            "physical_admission": "not_established",
            "global_geometry": "not_established",
        }

    @property
    def digest(self) -> str:
        return digest_of(self.to_dict())


def _check_raw_state(supplied: RawTaskState, measured: RawTaskState) -> None:
    for name in (
        "joint_ids",
        "joints_rad",
        "frame_ids",
        "scope_digest",
        "feedback_stamp",
        "feedback_ids",
    ):
        if getattr(supplied, name) != getattr(measured, name):
            raise ValueError("raw task endpoint differs from indexed encoder/cohort/clock/scope")
    if supplied.witness_valid is not False:
        raise ValueError("unverified raw task witness flag cannot establish independent evidence")
    for name in ("state_id", "stamp", "feature_id", "domain_id", "settled", "measured"):
        if getattr(supplied.task, name) != getattr(measured.task, name):
            raise ValueError("task state identity/stillness differs from indexed feature")
    for name in ("pose_m_rad", "feature_m_rad"):
        if any(
            abs(a - b) > 1e-10
            for a, b in zip(getattr(supplied.task, name), getattr(measured.task, name))
        ):
            raise ValueError("task coordinates differ from indexed nominal pose or metric feature")
    if any(
        a < b
        for a, b in zip(
            (*supplied.joint_uncertainty_rad, *supplied.task.uncertainty_m_rad),
            (*measured.joint_uncertainty_rad, *measured.task.uncertainty_m_rad),
        )
    ):
        raise ValueError("raw task uncertainty understates indexed measurement/drift floors")


def indexed_raw_task_factors(
    store: IndexedCaptureReader,
    bundle: EvidenceBundle,
    acquisition_ids: Mapping[str, str],
    tracking_results: Mapping[str, Mapping[str, Any]],
    *,
    feature: MetricTaskFeature,
    config: TrackingConfig,
    trusted_keys: Mapping[str, Ed25519PublicKey],
    node_id: str,
) -> IndexedRawTaskFactors:
    """Reopen originals and reconstruct every raw-task factor off the owner loop.

    No manifest or precomputed state check is accepted as input authority. This
    supports the versioned camera-centroid feature only and rejects promotion to
    independent witness or physical geometry. Existing pose-response is unchanged.
    """
    if (
        not isinstance(store, IndexedCaptureReader)
        or not isinstance(bundle, EvidenceBundle)
        or any(
            not isinstance(o, (RawTaskTransition, RgbdObservation, JointObservation))
            for o in bundle.observations
        )
    ):
        raise ValueError("raw task reconstruction cannot mix unrelated evidence methods")
    factors = [o for o in bundle.observations if isinstance(o, RawTaskTransition)]
    if (
        not 1 <= len(factors) <= 1024
        or not isinstance(tracking_results, Mapping)
        or set(tracking_results) != {o.observation_id for o in factors}
    ):
        raise ValueError("every raw task factor requires exact tracking lineage")
    if not isinstance(feature, MetricTaskFeature) or not isinstance(config, TrackingConfig):
        raise ValueError("typed frozen task feature and tracking policy required")
    snapshot = _CaptureSnapshot(store)
    manifest = indexed_source_manifest(
        snapshot, bundle, acquisition_ids, trusted_keys=trusted_keys, node_id=node_id
    )
    checks = []
    for factor in sorted(factors, key=lambda o: o.observation_id):
        if factor.provenance != "metric_rgbd" or any(
            len(s.frame_ids) != 1 for s in (factor.first, factor.second)
        ):
            raise ValueError("raw task feature requires one exact metric frame per endpoint")
        captures = [
            snapshot.captures[(acquisition_ids[s.frame_ids[0]], bundle.topology.workcell_id)]
            for s in (factor.first, factor.second)
        ]
        result = tracking_results[factor.observation_id]
        states = metric_raw_states(*captures, result, feature, bundle.topology, config=config)
        for supplied, measured in zip((factor.first, factor.second), states):
            _check_raw_state(supplied, measured)
        checks.append(
            RawTaskFactorCheck(
                factor.observation_id,
                digest_of(factor.to_dict()),
                digest_of(result),
                tuple(s.digest for s in states),
            )
        )
    return IndexedRawTaskFactors(manifest, config, feature, tuple(checks))

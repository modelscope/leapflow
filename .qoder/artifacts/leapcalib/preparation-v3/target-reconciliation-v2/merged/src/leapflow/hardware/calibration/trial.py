# Copyright (c) Alibaba, Inc. and its affiliates.
"""Indexed camera-only shadow trial; nominal robot geometry and authority never change.

The split is fixed before tracking: excluded training pairs cannot be replaced by
withheld pairs. Raw captures stay owner-local and only bounded ROIs reach the
isolated worker. Declared drift/span rates are conditional assumptions, not
measured interval peak rates or a physical qualification.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from leapcalib.contracts.estimates import EstimateBundle
from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import (
    DeviceIdentity,
    ModelIdentity,
    MountIdentity,
    TopologyIdentity,
    digest_of,
)
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
    finite_vector,
)
from leapcalib.contracts.observations import JointObservation, RgbdObservation
from leapcalib.contracts.tracking import TrackingConfig
from leapcalib.estimation.numeric import SolveConfig
from leapcalib.geometry.se3 import Transform
from leapcalib.runtime.protocol import WorkerBudgets
from leapcalib.validation.bundle import EvidenceBundle, Retention
from leapcalib.validation.evaluation import EvaluationConfig, MotionMeasurementBounds
from leapflow.hardware.calibration.acquisition import CollectionProfile, SettledObservation
from leapflow.hardware.calibration.evidence import CalibrationEvidenceStore
from leapflow.hardware.calibration.factors import _capture_stamp, _flange
from leapflow.hardware.calibration.reference import validate_metric_tracking_result
from leapflow.hardware.calibration.worker import ComputeWorker
from leapflow.hardware.skills.protocol import FAILURE_STOPPED, SkillError

SCHEMA = "leapflow.camera_motion_trial.v1"


@dataclass(frozen=True)
class TrialPolicy:
    pairs: tuple[tuple[str, str], ...]
    training: int
    mount: MountIdentity
    prior: Transform
    prior_uncertainty: tuple[float, ...]
    tracking: TrackingConfig
    solve: SolveConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_mapping(cls, args: Mapping[str, Any], budgets: WorkerBudgets) -> TrialPolicy:
        pairs = args.get("pairs")
        if not isinstance(pairs, (list, tuple)) or not 1 <= len(pairs) <= min(
            64, budgets.max_keyframes // 2
        ):
            raise ValueError("trial needs 1..64 bounded acquisition pairs")
        if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in pairs):
            raise ValueError("each pair needs exactly two indexed acquisition digests")
        ids = [v for p in pairs for v in p]
        if any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError(
                "trial pairs require distinct exact acquisition digests; no source reuse"
            )
        training = args.get("training")
        if type(training) is not int or not 1 <= training < len(pairs):
            raise ValueError("freeze nonempty training and withheld groups before tracking")
        mount = MountIdentity(**args["mount"])
        prior = Transform.from_mapping(args["prior"])
        uncertainty = finite_vector(args.get("prior_uncertainty", (0.05,) * 3 + (0.2,) * 3), 6)
        if (
            any(v <= 0 for v in uncertainty)
            or prior.parent != mount.parent_frame
            or not prior.child
        ):
            raise ValueError(
                "positive prior uncertainty and explicit matching mount frames required"
            )
        tracking = TrackingConfig(**args.get("tracking", {}))
        solve = SolveConfig(**args.get("solve", {}))
        evaluation = EvaluationConfig(**args.get("evaluation", {}))
        if (
            tracking.max_tracks > budgets.max_tracks
            or solve.max_pairs > budgets.max_keyframes
            or solve.max_points > budgets.max_tracks
            or solve.trials > 48
            or solve.refinement_steps > 8
        ):
            raise ValueError("trial exceeds integrated tracking/solve work budgets")
        return cls(
            tuple(tuple(p) for p in pairs),
            training,
            mount,
            prior,
            uncertainty,
            tracking,
            solve,
            evaluation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [list(p) for p in self.pairs],
            "training": self.training,
            "mount": self.mount.to_dict(),
            "prior": self.prior.to_dict(),
            "prior_uncertainty": list(self.prior_uncertainty),
            "tracking": vars(self.tracking),
            "solve": self.solve.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


def _checkpoint(blocked: Callable[[], bool]) -> None:
    if blocked():
        raise SkillError(
            FAILURE_STOPPED, "camera trial stopped; no motion, activation or automatic replay"
        )


def _capture_rows(
    capture: SettledObservation, model: ModelIdentity
) -> tuple[RgbdObservation, JointObservation]:
    anchor = capture.feedback[-2]
    return (
        RgbdObservation(
            capture.roi.frame_id, capture.roi.profile, _capture_stamp(capture), "metric_rgbd"
        ),
        JointObservation(anchor.source_id, model, anchor.stamp, anchor.joints_rad, still=True),
    )


def _sampled_speed(captures: tuple[SettledObservation, SettledObservation]) -> float:
    speeds = []
    for capture in captures:
        for first, second in zip(capture.feedback, capture.feedback[1:]):
            dt = second.stamp.delta_s(first.stamp)
            if dt <= 0:
                raise ValueError("ordered feedback required for sampled stillness check")
            speeds.extend(abs(b - a) / dt for a, b in zip(first.joints_rad, second.joints_rad))
    return max(speeds)


async def _pair(
    store: CalibrationEvidenceStore,
    worker: ComputeWorker,
    pair: tuple[str, str],
    *,
    policy: TrialPolicy,
    profile: CollectionProfile,
    model: ModelIdentity,
    workcell_id: str,
    run_id: str,
    node_session_id: str,
    index: int,
) -> tuple[tuple[SettledObservation, SettledObservation], Mapping[str, Any], MotionObservation]:
    captures = []
    for identity in pair:
        content = await store.read_capture(identity, workcell_id=workcell_id)
        capture = await asyncio.to_thread(SettledObservation.from_mapping, content["observation"])
        if (
            content["binding"]["node_session_id"] != node_session_id
            or content["profile"] != vars(profile.settled)
            or capture.feedback[0].model_digest != model.digest
            or capture.roi.mount_digest != policy.mount.digest
        ):
            raise ValueError(
                "capture owner/model/mount/uncertainty policy differs from approved trial"
            )
        captures.append(capture)
    first, second = captures
    payload = await asyncio.to_thread(
        lambda: {
            "first": first.roi.to_dict(),
            "second": second.roi.to_dict(),
            "tracks": [],
            "config": vars(policy.tracking),
        }
    )
    computed = await worker.submit(
        "track_metric", payload, workcell_id=workcell_id, run_id=run_id, model_digest=model.digest
    )
    if not isinstance(computed, Mapping) or set(computed) != {"tracking"}:
        raise ValueError("typed tracking worker response required")
    result = computed["tracking"]
    await asyncio.to_thread(validate_metric_tracking_result, result, first, second, policy.tracking)
    before, _, _ = _flange(first)
    after, _, _ = _flange(second)
    relative = (
        before.inverse().compose(after)
        if profile.camera_method == "wrist"
        else after.compose(before.inverse())
    )
    factor = MotionObservation(
        f"motion:{index:03d}",
        first.roi.frame_id,
        second.roi.frame_id,
        first.roi.profile,
        model,
        _capture_stamp(first),
        _capture_stamp(second),
        "metric_rgbd",
        MODE_WRIST if profile.camera_method == "wrist" else MODE_FIXED,
        "unwitnessed-static-scene",
        relative,
        tuple(
            PointCorrespondence(r["feature_id"], r["first_m"], r["second_m"])
            for r in result["tracks"]
        ),
        True,
        _sampled_speed((first, second)),
    )
    return (first, second), result, factor


async def run_motion_trial(
    store: CalibrationEvidenceStore,
    worker: ComputeWorker,
    *,
    policy: TrialPolicy,
    profile: CollectionProfile,
    model: ModelIdentity,
    workcell_id: str,
    run_id: str,
    node_session_id: str,
    approval_keys: tuple[Mapping[str, Any], str],
    blocked: Callable[[], bool],
) -> dict[str, Any]:
    """Reopen indexed captures, freeze source-disjoint factors, solve and assess; no promotion."""
    primitives, factors, sources, results, refused = {}, [], {}, {}, []
    accepted, holdouts = [], []
    for index, pair in enumerate(policy.pairs):
        _checkpoint(blocked)
        try:
            captures, result, factor = await _pair(
                store,
                worker,
                pair,
                policy=policy,
                profile=profile,
                model=model,
                workcell_id=workcell_id,
                run_id=run_id,
                node_session_id=node_session_id,
                index=index,
            )
        except (CalibrationFailure, ValueError) as exc:
            refused.append(
                {
                    "pair_index": index,
                    "split": "training" if index < policy.training else "withheld",
                    "reason": str(exc),
                }
            )
            continue
        for identity, capture in zip(pair, captures):
            for row in _capture_rows(capture, model):
                if row.observation_id in primitives:
                    raise ValueError(
                        "recorded frame/feedback identities reused between acquisition pairs"
                    )
                primitives[row.observation_id], sources[row.observation_id] = row, identity
        factors.append(factor)
        results[factor.observation_id] = result
        (accepted if index < policy.training else holdouts).append(factor.observation_id)
    report = {
        "schema": SCHEMA,
        "scope": "conditional_camera_motion",
        "activation": "none",
        "qualified_consumers": [],
        "physical_admission": "not_established",
        "raw_exported": False,
        "tracking_refusals": refused,
        "split": {"accepted_ids": accepted, "holdout_ids": holdouts},
        "rate_provenance": "declared_settled_drift_over_span_not_measured_peak",
    }
    if not factors:
        return {
            **report,
            "evaluation": {"verdict": "insufficient_evidence"},
            "candidate_digest": None,
        }
    topology = TopologyIdentity(
        workcell_id,
        DeviceIdentity(captures[0].feedback[0].device_id),
        model,
        (factors[0].stream,),
        (policy.mount,),
    )
    bundle = EvidenceBundle(
        "camera-motion-trial:" + digest_of(policy.to_dict()),
        topology,
        (*primitives.values(), *factors),
        time.time(),
        Retention(profile.ttl_s),
        accepted_ids=tuple(accepted),
        holdout_ids=tuple(holdouts),
        limitations=("unwitnessed scene; nominal firmware flange; shadow only",),
        metadata={"node_session_id": node_session_id},
    )
    _checkpoint(blocked)
    keys, node_id = approval_keys
    checked = await store.motion_factors(
        bundle, sources, results, config=policy.tracking, trusted_keys=keys, node_id=node_id
    )
    _checkpoint(blocked)
    payload = await asyncio.to_thread(
        lambda: {
            "evidence": bundle.to_dict(),
            "options": {
                "prior": policy.prior.to_dict(),
                "prior_uncertainty": list(policy.prior_uncertainty),
                "config": policy.solve.to_dict(),
            },
        }
    )
    computed = await worker.submit(
        "solve_motion", payload, workcell_id=workcell_id, run_id=run_id, model_digest=model.digest
    )
    if not isinstance(computed, Mapping) or set(computed) != {"estimate"}:
        raise ValueError("typed solve worker response required")
    fit = EstimateBundle.from_mapping(computed["estimate"])
    settled = profile.settled
    bounds = {
        c.observation_id: MotionMeasurementBounds(
            c.observation_digest,
            c.point_uncertainty_m,
            c.robot_translation_uncertainty_m,
            c.robot_rotation_uncertainty_rad,
            settled.max_tcp_drift_mm / 1000 / settled.span_s,
            math.radians(settled.max_tcp_rotation_deg) / settled.span_s,
        )
        for c in checked.checks
        if c.observation_id in holdouts
    }
    _checkpoint(blocked)
    assessed = await store.assess_motion_candidate(
        bundle,
        fit,
        sources,
        results,
        measurement_bounds=bounds,
        tracking=policy.tracking,
        config=policy.evaluation,
        trusted_keys=keys,
        node_id=node_id,
    )
    _checkpoint(blocked)
    return {
        **assessed.to_dict(),
        **report,
        "fit_rejections": fit.metadata.get("rejections", []),
        "fit_holdout_rejections": fit.metadata.get("holdout_rejections", []),
    }

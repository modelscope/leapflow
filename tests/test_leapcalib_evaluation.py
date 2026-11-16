# Copyright (c) Alibaba, Inc. and its affiliates.
"""Withheld recomputation and honest conditional scope; all fixtures are synthetic."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from leapcalib.contracts.motion import MotionObservation
from leapcalib.contracts.observations import JointObservation, Stamp
from leapcalib.contracts.raw_task import RawTaskTransition
from leapcalib.estimation.handeye import estimate_motion
from leapcalib.estimation.numeric import SolveConfig
from leapcalib.estimation.raw_response import estimate_raw_response
from leapcalib.validation.evaluation import (
    CandidateEvaluation,
    EvaluationConfig,
    EvaluationUnit,
    MotionMeasurementBounds,
    evaluate_motion_candidate,
    evaluate_raw_candidate,
    independent_holdouts,
)
from tests.test_leapcalib_raw_response import raw_study as _raw_study
from tools.calib_regression import motion_study as _motion_study


def motion_study(**kwargs):
    bundle, prior, true = _motion_study(**kwargs)
    return replace(bundle, metadata={"node_session_id": "synthetic-owner-session"}), prior, true


def raw_study(**kwargs):
    bundle, domain, matrix = _raw_study(**kwargs)
    return replace(bundle, metadata={"node_session_id": "synthetic-owner-session"}), domain, matrix


def paired_result(*, candidate=0.0001, baseline=0.002, noise=0.00001):
    units = tuple(
        EvaluationUnit(str(i), (candidate,) * 4, (baseline,) * 4, (noise,) * 4) for i in range(20)
    )
    return CandidateEvaluation(
        "same",
        "evidence",
        "same",
        "evidence",
        "nuisance",
        "raw_joint_response.v1",
        "local_raw_joint_response",
        EvaluationConfig(),
        units,
        (),
        ("synthetic",),
    )


@pytest.mark.parametrize(
    "candidate,baseline,noise,verdict",
    [
        (0.0001, 0.002, 0.00001, "numerical_pass"),
        (0.0001, 0.0001, 0.00001, "no_op"),
        (0.0001, 0.00011, 0.00001, "no_op"),
        (0.001, 0.0001, 0.00001, "rejected"),
        (0.0029, 0.01, 0.0001, "rejected"),
    ],
)
def test_noise_aware_paired_verdict(candidate, baseline, noise, verdict):
    result = paired_result(candidate=candidate, baseline=baseline, noise=noise)
    assert result.verdict == verdict
    assert result.to_dict()["physical_admission"] == "owner_indexed_acquisition_and_scope_required"
    assert result.to_dict()["global_geometry"] == "not_established"
    assert "qualified" not in result.to_dict()
    assert result.candidate_digest == result.baseline_digest
    assert result.digest == result.digest


def test_invalid_withheld_units_and_consumer_regression_cannot_disappear():
    result = paired_result()
    assert replace(result, units=result.units[:19]).verdict == "insufficient_evidence"
    assert replace(result, exclusions=(("bad", "lost witness"),)).verdict == "rejected"
    unit = replace(result.units[0], candidate_error_m_rad=(0, 0, 0.001, 0))
    assert replace(result, units=(unit, *result.units[1:])).verdict == "numerical_pass"
    # The aggregate max improves, but one required coordinate worsens.
    unit = replace(unit, baseline_error_m_rad=(0.002, 0.002, 0, 0.002))
    assert replace(result, units=(unit, *result.units[1:])).verdict == "rejected"
    with pytest.raises(ValueError, match="scope"):
        replace(result, scope="global_geometry")
    with pytest.raises(ValueError):
        replace(result, units=(*result.units, result.units[0]))


@pytest.mark.parametrize(
    "edit",
    [
        {"min_units": True},
        {"min_units": 19},
        {"separation_s": 0.299},
        {"separation_s": "0.3"},
        {"max_error_m_rad": (0.004,) * 4},
        {"confidence_multiplier": False},
    ],
)
def test_study_acceptance_limits_remain_bounded(edit):
    with pytest.raises(ValueError):
        EvaluationConfig(**edit)


def test_source_time_window_and_rejected_training_independence():
    bundle, _, _ = motion_study()
    factors = [o for o in bundle.observations if isinstance(o, MotionObservation)]
    training = factors[0]
    # Distinct strings are not independent measurements at the same time.
    nearby = replace(factors[24], first_stamp=Stamp(1.2), stamp=Stamp(2.2))
    selected, exclusions = independent_holdouts(
        (training,), (nearby, factors[25]), separation_s=0.3
    )
    assert selected == (factors[25],)
    assert exclusions == (
        (nearby.observation_id, "overlapping or correlated adjacent physical windows"),
    )
    expanded = replace(nearby, first_stamp=Stamp(1.31, uncertainty_s=0.02), stamp=Stamp(2.31))
    assert not independent_holdouts((training,), (expanded,), separation_s=0.3)[0]
    shared = replace(factors[24], first_id=training.first_id)
    assert "shared" in independent_holdouts((training,), (shared,), separation_s=0.3)[1][0][1]
    assert not independent_holdouts((training,), (training,), separation_s=0.3)[0]
    assert independent_holdouts((training, training), (factors[24],), separation_s=0.3)[0]
    with pytest.raises(ValueError, match="conflicting"):
        independent_holdouts(
            (training, replace(training, scene_id="different")), (), separation_s=0.3
        )
    with pytest.raises(ValueError, match="typed"):
        independent_holdouts((object(),), (), separation_s=0.3)


def test_exclusion_order_does_not_select_favorable_units():
    bundle, _, _ = motion_study()
    factors = [o for o in bundle.observations if isinstance(o, MotionObservation)]
    nearby = replace(factors[25], first_stamp=Stamp(48.5), stamp=Stamp(49.5))
    a = independent_holdouts((factors[0],), (nearby, factors[24], factors[26]), separation_s=0.3)
    b = independent_holdouts((factors[0],), (factors[26], factors[24], nearby), separation_s=0.3)
    assert a == b
    assert a[1][0][0] == nearby.observation_id


def bounds(bundle):
    return {
        o.observation_id: MotionMeasurementBounds(o.digest, 0.00001, 0.00001, 0.0001, 0.01, 0.02)
        for o in bundle.observations
        if o.observation_id in bundle.holdout_ids
    }


@pytest.mark.parametrize("mode", ["fixed_rigid_motion", "wrist_static_scene"])
def test_motion_recompute_and_measurement_bound_binding(mode):
    bundle, prior, _ = motion_study(mode=mode, noise_m=0.00005)
    solve = SolveConfig(translation_sigma_m=0.0002, rotation_sigma_rad=0.001)
    fit = estimate_motion(bundle, prior, config=solve)
    measurement = bounds(bundle)
    result = evaluate_motion_candidate(bundle, fit, measurement_bounds=measurement)
    assert len(result.units) == 20 and not result.exclusions
    assert result.scope == "conditional_camera_motion"
    assert all(u.uncertainty_m_rad[0] > 0.0002 for u in result.units)
    assert result.nuisance_digest
    with pytest.raises(ValueError, match="recomputation"):
        evaluate_motion_candidate(
            bundle,
            replace(fit, metadata={**fit.metadata, "data_rank": 0}),
            measurement_bounds=measurement,
        )
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_motion_candidate(bundle, fit, measurement_bounds={})
    changed = {**measurement, "m24": replace(measurement["m24"], observation_digest="tampered")}
    with pytest.raises(ValueError, match="exact source"):
        evaluate_motion_candidate(bundle, fit, measurement_bounds=changed)
    more_noise = {i: replace(b, point_uncertainty_m=0.001) for i, b in measurement.items()}
    rejected = evaluate_motion_candidate(bundle, fit, measurement_bounds=more_noise)
    assert rejected.verdict == "rejected"
    assert rejected.nuisance_digest != result.nuisance_digest
    assert all(
        a.uncertainty_m_rad[3] > b.uncertainty_m_rad[3]
        for a, b in zip(rejected.units, result.units)
    )


def test_motion_invalid_withheld_branch_is_explicit():
    bundle, prior, _ = motion_study()
    bundle = replace(
        bundle,
        observations=tuple(
            replace(o, settled=False) if o.observation_id == "m24" else o
            for o in bundle.observations
        ),
    )
    fit = estimate_motion(bundle, prior)
    result = evaluate_motion_candidate(bundle, fit, measurement_bounds=bounds(bundle))
    assert len(result.units) == 19
    assert result.exclusions[0][0] == "m24"
    assert result.verdict == "insufficient_evidence"


def shifted_baseline(bundle, *, shift=300, feature_scale=1.0):
    def state(s):
        feature = np.asarray(s.task.feature_m_rad)
        origin = np.asarray((0, 0, 0.3, 0))
        return replace(
            s,
            task=replace(
                s.task,
                state_id="before:" + s.task.state_id,
                stamp=replace(s.task.stamp, seconds=s.task.stamp.seconds + shift),
                feature_m_rad=tuple(origin + feature_scale * (feature - origin)),
            ),
            frame_ids=tuple("before:" + i for i in s.frame_ids),
            feedback_ids=tuple("before:" + i for i in s.feedback_ids),
            feedback_stamp=replace(s.feedback_stamp, seconds=s.feedback_stamp.seconds + shift),
        )

    observations = []
    for o in bundle.observations:
        if isinstance(o, RawTaskTransition):
            o = replace(
                o,
                observation_id="before:" + o.observation_id,
                first=state(o.first),
                second=state(o.second),
            )
        else:
            o = replace(
                o,
                observation_id="before:" + o.observation_id,
                stamp=replace(o.stamp, seconds=o.stamp.seconds + shift),
            )
        observations.append(o)
    return replace(
        bundle,
        bundle_id="before",
        observations=tuple(observations),
        accepted_ids=tuple("before:" + i for i in bundle.accepted_ids),
        rejected_ids=tuple("before:" + i for i in bundle.rejected_ids),
        holdout_ids=tuple("before:" + i for i in bundle.holdout_ids),
    )


def contained_raw_study():
    bundle, domain, _ = raw_study()
    endpoints = {}

    def state(s):
        raw = tuple(np.asarray(s.joints_rad) / 2)
        endpoints[s.feedback_ids[0]] = raw
        feature = np.asarray(s.task.feature_m_rad)
        origin = np.asarray((0, 0, 0.3, 0))
        return replace(
            s,
            joints_rad=raw,
            task=replace(s.task, feature_m_rad=tuple(origin + (feature - origin) / 2)),
        )

    observations = [
        replace(o, first=state(o.first), second=state(o.second))
        if isinstance(o, RawTaskTransition) and o.observation_id in bundle.holdout_ids
        else o
        for o in bundle.observations
    ]
    observations = [
        replace(o, q_raw_rad=endpoints[o.observation_id])
        if isinstance(o, JointObservation) and o.observation_id in endpoints
        else o
        for o in observations
    ]
    return replace(bundle, observations=tuple(observations)), domain


def test_raw_equal_baseline_is_noop_and_both_evidence_bindings_are_recomputed():
    bundle, domain = contained_raw_study()
    fit = estimate_raw_response(bundle, domain)
    result = evaluate_raw_candidate(bundle, fit, fit, baseline_bundle=bundle)
    assert len(result.units) == 20 and not result.exclusions
    assert result.verdict == "no_op"
    assert result.baseline_evidence_digest == bundle.digest
    before_bundle = shifted_baseline(bundle)
    baseline = estimate_raw_response(before_bundle, domain)
    assert (
        evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=before_bundle).verdict
        == "no_op"
    )
    with pytest.raises(ValueError, match="recomputation"):
        evaluate_raw_candidate(
            bundle,
            fit,
            replace(baseline, metadata={**baseline.metadata, "data_rank": 5}),
            baseline_bundle=before_bundle,
        )
    with pytest.raises(ValueError, match="recomputation"):
        evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=bundle)
    # No changed evidence can be substituted for either fitted binding.
    with pytest.raises(ValueError, match="recomputation"):
        evaluate_raw_candidate(
            bundle, fit, baseline, baseline_bundle=replace(before_bundle, bundle_id="tampered")
        )


def test_raw_baseline_temporal_training_contamination_is_explicit():
    bundle, domain = contained_raw_study()
    fit = estimate_raw_response(bundle, domain)
    before_bundle = shifted_baseline(bundle, shift=120)
    baseline = estimate_raw_response(before_bundle, domain)
    result = evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=before_bundle)
    assert not result.units
    assert len(result.exclusions) == 20
    assert all("physical windows" in reason for _, reason in result.exclusions)
    assert result.verdict == "insufficient_evidence"


def test_raw_unsupported_holdout_and_parameter_uncertainty_are_not_fit_loss():
    bundle, domain, _ = raw_study()
    fit = estimate_raw_response(bundle, domain)
    result = evaluate_raw_candidate(bundle, fit, fit, baseline_bundle=bundle)
    assert result.exclusions  # Some withheld poses extrapolate the measured region.
    assert all("measured region" in reason for _, reason in result.exclusions)
    assert all(
        u.uncertainty_m_rad[0] > fit.metadata["residual_floor_m_rad"][0] for u in result.units
    )


def test_session_clock_identity_cannot_be_missing_or_cross_compared():
    bundle, domain = contained_raw_study()
    no_session = replace(bundle, metadata={})
    fit = estimate_raw_response(no_session, domain)
    with pytest.raises(ValueError, match="explicit node-session"):
        evaluate_raw_candidate(no_session, fit, fit, baseline_bundle=no_session)
    fit = estimate_raw_response(bundle, domain)
    baseline_bundle = replace(
        shifted_baseline(bundle), metadata={"node_session_id": "another-owner"}
    )
    baseline = estimate_raw_response(baseline_bundle, domain)
    with pytest.raises(ValueError, match="not comparable"):
        evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=baseline_bundle)


def test_rejected_baseline_source_and_conflicting_identity_are_not_ignored():
    bundle, domain = contained_raw_study()
    fit = estimate_raw_response(bundle, domain)
    baseline_bundle = replace(
        bundle,
        accepted_ids=bundle.accepted_ids[:30],
        rejected_ids=("r40",),
        holdout_ids=bundle.holdout_ids[1:],
    )
    baseline = estimate_raw_response(baseline_bundle, domain)
    result = evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=baseline_bundle)
    assert result.exclusions == (("r40", "shared frame/state/raw-feedback sources"),)
    assert result.verdict == "insufficient_evidence"
    conflicting = replace(
        bundle,
        bundle_id="conflict",
        observations=tuple(
            replace(o, first=replace(o.first, witness_valid=False))
            if o.observation_id == "r0"
            else o
            for o in bundle.observations
        ),
    )
    baseline = estimate_raw_response(conflicting, domain)
    with pytest.raises(ValueError, match="conflicting source identity"):
        evaluate_raw_candidate(bundle, fit, baseline, baseline_bundle=conflicting)


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_retained_raw_fit_is_an_explicit_insufficient_evaluation(rank):
    bundle, domain, _ = raw_study(rank=rank)
    fit = estimate_raw_response(bundle, domain)
    result = evaluate_raw_candidate(bundle, fit, fit, baseline_bundle=bundle)
    assert result.verdict == "insufficient_evidence" and not result.units
    assert len(result.exclusions) == len(bundle.holdout_ids) == 20
    assert all(
        "data-supported" in reason or "below-resolution" in reason
        for _, reason in result.exclusions
    )
    with pytest.raises(ValueError, match="recomputation"):
        evaluate_raw_candidate(
            bundle,
            replace(fit, metadata={**fit.metadata, "data_rank": 4}),
            fit,
            baseline_bundle=bundle,
        )


@pytest.mark.parametrize("baseline_missing", [False, True])
def test_unwitnessed_raw_candidate_or_baseline_cannot_be_numerically_qualified(baseline_missing):
    bundle, domain = contained_raw_study()
    supported = estimate_raw_response(bundle, domain)
    changed = replace(
        bundle,
        observations=tuple(
            replace(
                o,
                first=replace(o.first, witness_valid=False),
                second=replace(o.second, witness_valid=False),
            )
            if isinstance(o, RawTaskTransition)
            else o
            for o in bundle.observations
        ),
    )
    retained = estimate_raw_response(changed, domain)
    if baseline_missing:
        # Separate baseline IDs avoid hiding the missing-witness gate behind a
        # conflicting source identity. It is still synthetic, not physical proof.
        before = shifted_baseline(changed)
        baseline = estimate_raw_response(before, domain)
        result = evaluate_raw_candidate(bundle, supported, baseline, baseline_bundle=before)
    else:
        result = evaluate_raw_candidate(changed, retained, retained, baseline_bundle=changed)
    assert result.verdict == "insufficient_evidence" and not result.units
    assert len(result.exclusions) == 20
    assert all(
        "data-supported" in reason if baseline_missing else "witness" in reason
        for _, reason in result.exclusions
    )


def test_evaluation_import_is_numerical_backend_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import leapcalib.validation.evaluation; assert 'numpy' not in sys.modules; assert 'leapflow' not in sys.modules",
        ],
        check=True,
    )

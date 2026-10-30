# Copyright (c) Alibaba, Inc. and its affiliates.
"""Candidate-only numerical, evidence and CLI contracts; synthetic is never metrology."""

from __future__ import annotations

import json
import math
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from leapcalib.cli import main
from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.methods import CalibrationMethod, discover_methods
from leapcalib.contracts.motion import (
    MODE_FIXED,
    MODE_WRIST,
    MotionObservation,
    PointCorrespondence,
)
from leapcalib.contracts.observations import Stamp
from leapcalib.contracts.task import TaskState
from leapcalib.estimation.handeye import estimate_motion
from leapcalib.estimation.numeric import SolveConfig, rank_report, rotation_vector
from leapcalib.estimation.registration import register_points
from leapcalib.estimation.response import (
    ResponseConfig,
    difference,
    estimate_response,
    propose_shadow,
)
from leapcalib.geometry.se3 import Transform
from leapcalib.validation.bundle import LEGACY_SCHEMA_VERSION, read_bundle, write_bundle
from leapcalib.validation.result import read_request, result_payload, write_result
from tools.calib_regression import motion_study, numerical_regression, response_study


@pytest.mark.parametrize("mode", [MODE_FIXED, MODE_WRIST])
@pytest.mark.parametrize("seed", [7, 19, 63])
def test_unseen_mount_errors_with_disjoint_holdouts(mode: str, seed: int) -> None:
    bundle, prior, true = motion_study(mode=mode, seed=seed, outliers=True, noise_m=0.0001)
    fit = estimate_motion(bundle, prior)
    candidate = Transform.from_mapping(fit.metadata["candidate_transform"])
    assert candidate.distance_m(true) < 0.001
    assert candidate.angle_to_rad(true) < 0.005
    assert fit.estimates[0].support == "supported"
    assert fit.metadata["data_rank"] == 6
    assert fit.metadata["prior_rows_in_rank"] == 0
    assert set(fit.rejected_ids) == {"m0", "m1", "m2", "m3"}
    assert len(fit.metadata["holdout_predictions"]) == 20
    assert max(p["translation_error_m"] for p in fit.metadata["holdout_predictions"]) < 0.001
    assert fit.digest == estimate_motion(bundle, prior).digest
    assert fit.estimates[0].uncertainty[:3] >= (0.002,) * 3


def test_registration_is_robust_right_handed_and_not_icp() -> None:
    rng = np.random.default_rng(2)
    true = Transform.from_rpy_rad(0.2, 0.1, -0.3, (0.02, -0.04, 0.03))
    points = rng.normal(size=(40, 3)) * 0.1
    pairs = [
        PointCorrespondence(str(i), tuple(p), true.apply(p) if i >= 7 else (3, 4, 5))
        for i, p in enumerate(points)
    ]
    fit = register_points(pairs)
    assert fit.motion.distance_m(true) < 1e-10
    assert fit.motion.angle_to_rad(true) < 1e-7
    assert len(fit.rejected_features) == 7
    assert np.linalg.det(fit.motion.rotation) == pytest.approx(1)
    inverse = register_points(pairs, reverse=True)
    assert inverse.motion.distance_m(true.inverse()) < 1e-10


@pytest.mark.parametrize("kind", ["coincident", "collinear"])
def test_registration_refuses_degenerate_correspondences(kind: str) -> None:
    pairs = [
        PointCorrespondence(
            str(i),
            (i * 0.01 if kind == "collinear" else 0, 0, 1),
            (i * 0.01 if kind == "collinear" else 0, 0, 1),
        )
        for i in range(12)
    ]
    with pytest.raises(CalibrationFailure):
        register_points(pairs)


def test_identified_planar_landmarks_are_not_untextured_plane_icp() -> None:
    pairs = [
        PointCorrespondence(
            str(i), (i % 4 * 0.02, i // 4 * 0.02, 0.5), (i % 4 * 0.02 + 0.01, i // 4 * 0.02, 0.5)
        )
        for i in range(12)
    ]
    assert register_points(pairs).spatial_rank == 2


def test_stationary_and_single_axis_evidence_keep_unsupported_priors() -> None:
    bundle, prior, _ = motion_study(stationary=True)
    fit = estimate_motion(bundle, prior)
    assert fit.estimates[0].values == fit.estimates[0].prior
    assert fit.metadata["data_rank"] == 0
    bundle, prior, _ = motion_study(single_axis=True)
    fit = estimate_motion(bundle, prior)
    assert fit.metadata["data_rank"] < 6
    assert fit.metadata["nullspace_scaled"]
    for i in fit.estimates[0].retained_components:
        assert fit.estimates[0].values[i] == 0
        assert fit.estimates[0].uncertainty[i] >= fit.estimates[0].prior_uncertainty[i]


def test_holdout_changes_never_train_the_mount() -> None:
    bundle, prior, _ = motion_study()
    original = estimate_motion(bundle, prior)
    changed = replace(
        bundle,
        observations=tuple(
            replace(o, robot_motion=Transform())
            if isinstance(o, MotionObservation) and o.observation_id in bundle.holdout_ids
            else o
            for o in bundle.observations
        ),
    )
    fit = estimate_motion(changed, prior)
    assert fit.estimates[0].values == original.estimates[0].values
    assert len(fit.metadata["holdout_rejections"]) == 20


def test_bundle_roundtrip_v0_compatibility_and_source_leakage(tmp_path: Path) -> None:
    bundle, _, _ = motion_study()
    assert read_bundle(write_bundle(tmp_path / "motion.json", bundle)) == bundle
    with pytest.raises(ValueError, match="share source"):
        replace(bundle, accepted_ids=(*bundle.accepted_ids, "m24:first"))
    with pytest.raises(ValueError, match="requires recorded"):
        replace(
            bundle,
            observations=tuple(o for o in bundle.observations if o.observation_id != "m0:first"),
        )
    with pytest.raises(ValueError, match="v0 cannot"):
        replace(bundle, schema_version=LEGACY_SCHEMA_VERSION)
    raw = replace(
        bundle,
        observations=tuple(o for o in bundle.observations if not isinstance(o, MotionObservation)),
        accepted_ids=(),
        holdout_ids=(),
        schema_version=LEGACY_SCHEMA_VERSION,
    )
    assert read_bundle(write_bundle(tmp_path / "v0.json", raw)) == raw


def test_task_sources_and_topology_are_consistent() -> None:
    bundle, _, _ = response_study()
    first, second = bundle.observations[:2]
    with pytest.raises(ValueError, match="share source"):
        replace(
            bundle,
            observations=(
                *bundle.observations[:40],
                replace(bundle.observations[40], first=first.first),
                *bundle.observations[41:],
            ),
        )
    with pytest.raises(ValueError, match="inconsistent"):
        replace(
            bundle,
            observations=(
                first,
                replace(second, first=replace(second.first, state_id=first.first.state_id)),
                *bundle.observations[2:],
            ),
        )
    with pytest.raises(ValueError, match="model differs"):
        replace(
            bundle,
            observations=(
                replace(first, model=replace(first.model, chain_digest="different")),
                *bundle.observations[1:],
            ),
        )


def test_measured_response_outliers_and_holdouts() -> None:
    bundle, domain, expected = response_study(outliers=True)
    fit = estimate_response(bundle, domain)
    assert np.asarray(fit.estimates[0].values).reshape(4, 4) == pytest.approx(expected)
    assert len(fit.accepted_ids) == 35
    assert fit.metadata["data_rank"] == 4
    assert fit.metadata["prior_rows_in_rank"] == 0
    assert len(fit.metadata["holdout_predictions"]) == 20
    changed = replace(
        bundle,
        observations=tuple(
            replace(o, second=replace(o.second, feature_id="lost"))
            if o.observation_id in bundle.holdout_ids
            else o
            for o in bundle.observations
        ),
    )
    assert estimate_response(changed, domain).estimates[0].values == fit.estimates[0].values


@pytest.mark.parametrize("columns", [(0,), (0, 1), (0, 1, 2)])
def test_response_partial_inputs_retain_prior_columns(columns: tuple[int, ...]) -> None:
    bundle, domain, _ = response_study(columns=columns)
    fit = estimate_response(bundle, domain)
    assert fit.metadata["supported_inputs"] == list(columns)
    assert fit.estimates[0].support == "partial"
    for i in fit.estimates[0].retained_components:
        assert fit.estimates[0].values[i] == 0


@pytest.mark.parametrize(
    "change", ["stationary", "unmeasured", "unsettled", "feature", "domain", "uncertainty", "time"]
)
def test_noninformative_or_unqualified_response_is_retained(change: str) -> None:
    bundle, domain, _ = response_study(stationary=change == "stationary")
    edits = {
        "unmeasured": {"measured": False},
        "unsettled": {"settled": False},
        "feature": {"feature_id": "other"},
        "domain": {"domain_id": "other"},
        "uncertainty": {"uncertainty_m_rad": (0.1,) * 4},
        "time": {"stamp": Stamp(10000)},
    }
    if change in edits:
        bundle = replace(
            bundle,
            observations=tuple(
                replace(o, second=replace(o.second, **edits[change])) for o in bundle.observations
            ),
        )
    fit = estimate_response(bundle, domain)
    assert not fit.estimates[0].supported_components
    assert fit.estimates[0].values == (0,) * 16


def current_state(domain) -> TaskState:
    return TaskState(
        "current",
        Stamp(200),
        (0, 0, 0.3, 0),
        (0, 0, 0.3, 0),
        domain.feature_id,
        domain.domain_id,
        True,
        True,
        (0.00001,) * 4,
    )


def test_bounded_shadow_is_unexecuted_and_improving() -> None:
    bundle, domain, _ = response_study()
    fit = estimate_response(bundle, domain)
    proposal = propose_shadow(
        fit,
        current_state(domain),
        (0.01, -0.008, 0.3, 0.1),
        domain,
        Stamp(200.01),
        topology_digest=bundle.topology.digest,
    )
    assert proposal["accepted"] and proposal["shadow_only"] and proposal["executed"] is False
    assert np.all(np.abs(proposal["delta_m_rad"]) <= domain.max_step_m_rad)
    assert np.all(np.asarray(proposal["uncertainty_m_rad"]) <= domain.tolerance_m_rad)
    assert proposal["clearance_error_m"] < domain.clearance_m


@pytest.mark.parametrize(
    "change", ["stale", "future", "feature", "unmeasured", "edge", "uncertainty", "latency"]
)
def test_shadow_refuses_bad_current_evidence(change: str) -> None:
    bundle, domain, _ = response_study()
    if change == "latency":
        domain = replace(domain, speed_m_s=0.5)
    fit = estimate_response(bundle, domain)
    current = current_state(domain)
    edits = {
        "stale": {"stamp": Stamp(199)},
        "future": {"stamp": Stamp(201)},
        "feature": {"feature_id": "other"},
        "unmeasured": {"measured": False},
        "edge": {"pose_m_rad": (0.09, 0, 0.3, 0)},
        "uncertainty": {"uncertainty_m_rad": (0.1,) * 4},
    }
    current = replace(current, **edits.get(change, {}))
    result = propose_shadow(
        fit,
        current,
        (0.01, 0, 0.3, 0.1),
        domain,
        Stamp(200.1),
        topology_digest=bundle.topology.digest,
    )
    assert not result["accepted"]
    assert result["executed"] is False


def test_shadow_noop_is_not_an_uncertain_accuracy_claim() -> None:
    bundle, domain, _ = response_study()
    fit = estimate_response(bundle, domain)
    current = current_state(domain)
    result = propose_shadow(
        fit,
        current,
        current.feature_m_rad,
        domain,
        Stamp(200),
        topology_digest=bundle.topology.digest,
    )
    assert result["no_op"] and result["executed"] is False
    current = replace(current, uncertainty_m_rad=(1,) * 4)
    assert not propose_shadow(
        fit,
        current,
        current.feature_m_rad,
        domain,
        Stamp(200),
        topology_digest=bundle.topology.digest,
    )["accepted"]
    with pytest.raises(ValueError, match="envelope"):
        propose_shadow(
            fit,
            current,
            current.feature_m_rad,
            replace(domain, clearance_m=1),
            Stamp(200),
            topology_digest=bundle.topology.digest,
        )
    with pytest.raises(CalibrationFailure, match="cannot difference"):
        propose_shadow(
            fit,
            current,
            current.feature_m_rad,
            domain,
            Stamp(200, domain="host_monotonic"),
            topology_digest=bundle.topology.digest,
        )


def test_yaw_wrap_and_full_data_nullspace() -> None:
    assert difference((0, 0, 0, -math.pi + 0.02), (0, 0, 0, math.pi - 0.02))[3] == pytest.approx(
        0.04
    )
    rank, _, _, null = rank_report(np.zeros((0, 4)), 1e-5)
    assert rank == 0 and null.shape == (4, 4)
    true = Transform.from_axis_angle((1, 2, 3), math.pi - 1e-9)
    assert np.linalg.norm(rotation_vector(np.asarray(true.rotation))) == pytest.approx(
        math.pi, abs=1e-7
    )


def test_discovery_and_imports_are_light() -> None:
    methods = discover_methods()
    assert set(methods) == {"metric_motion.v1", "task_response.v1"}
    assert all(isinstance(m, CalibrationMethod) for m in methods.values())
    code = "from leapcalib.contracts.methods import discover_methods; import sys; discover_methods(); assert 'numpy' not in sys.modules; assert 'leapflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    code = "import sys; sys.modules['numpy']=None; from leapcalib.estimation.numeric import numpy; from leapcalib.contracts.failures import CalibrationFailure\ntry: numpy()\nexcept CalibrationFailure as e: assert e.code=='not_implemented'\nelse: raise AssertionError()"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf")])
def test_strict_numeric_boundaries(value: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PointCorrespondence("x", (value, 0, 1), (0, 0, 1))
    with pytest.raises(ValueError):
        Stamp.from_mapping({"seconds": value})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"target": value}))
    if isinstance(value, bool):
        assert read_request(path)["target"] is value
    else:
        with pytest.raises(CalibrationFailure):
            read_request(path)


def test_cli_estimate_shadow_replay_and_private_output(tmp_path: Path, capsys) -> None:
    bundle, prior, _ = motion_study()
    path = write_bundle(tmp_path / "motion.json", bundle)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"prior": prior.to_dict()}))
    output = tmp_path / "candidate.json"
    assert (
        main(
            [
                "estimate",
                str(path),
                "--method",
                "metric_motion.v1",
                "--request",
                str(request),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["content"]["estimate"]["method_id"] == "metric_motion.v1"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert main(["estimate", str(path), "--method", "missing", "--request", str(request)]) == 1
    capsys.readouterr()
    assert main(["replay", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["motions"] == 44
    bundle, domain, _ = response_study()
    path = write_bundle(tmp_path / "task.json", bundle)
    request.write_text(
        json.dumps(
            {
                "domain": domain.to_dict(),
                "current": current_state(domain).to_dict(),
                "target_m_rad": [0.01, 0, 0.3, 0.1],
                "now": Stamp(200).to_dict(),
                "topology_digest": bundle.topology.digest,
            }
        )
    )
    assert main(["shadow", str(path), "--request", str(request), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["content"]["proposal"]["executed"] is False


@pytest.mark.parametrize("value", [None, True, 42, [], {}, " "])
def test_motion_identity_fields_require_strings(value) -> None:
    bundle, _, _ = motion_study()
    motion = next(o for o in bundle.observations if isinstance(o, MotionObservation))
    with pytest.raises(ValueError):
        PointCorrespondence(value, (0, 0, 1), (0, 0, 1))
    for name in ("observation_id", "first_id", "second_id", "scene_id", "provenance", "mode"):
        with pytest.raises(ValueError):
            replace(motion, **{name: value})


@pytest.mark.parametrize("value", [None, True, 42, [], {}, " "])
def test_task_identity_fields_require_strings(value) -> None:
    bundle, domain, _ = response_study()
    transition = bundle.observations[0]
    for name in ("state_id", "feature_id", "domain_id"):
        with pytest.raises(ValueError):
            replace(transition.first, **{name: value})
    for name in ("feature_id", "domain_id"):
        with pytest.raises(ValueError):
            replace(domain, **{name: value})
    for name in ("observation_id", "provenance"):
        with pytest.raises(ValueError):
            replace(transition, **{name: value})


@pytest.mark.parametrize("value", [None, 42, {}])
@pytest.mark.parametrize(
    "section,field",
    [
        ("current", "state_id"),
        ("current", "feature_id"),
        ("current", "domain_id"),
        ("domain", "domain_id"),
        ("domain", "feature_id"),
    ],
)
def test_cli_refuses_nonstring_request_identities(
    tmp_path: Path, capsys, section: str, field: str, value
) -> None:
    bundle, domain, _ = response_study()
    path = write_bundle(tmp_path / "task.json", bundle)
    request = {
        "domain": domain.to_dict(),
        "current": current_state(domain).to_dict(),
        "target_m_rad": [0.01, 0, 0.3, 0.1],
        "now": Stamp(200).to_dict(),
        "topology_digest": bundle.topology.digest,
    }
    request[section][field] = value
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    output = tmp_path / "candidate.json"
    assert (
        main(
            ["shadow", str(path), "--request", str(request_path), "--output", str(output), "--json"]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "invalid numerical request" in captured.err
    assert not captured.out and not output.exists()


def test_explicit_result_overwrite_remains_private(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    path.write_text("old")
    path.chmod(0o644)
    with pytest.raises(CalibrationFailure):
        write_result(path, result_payload({"candidate_only": True}))
    write_result(path, result_payload({"candidate_only": True}), overwrite=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_exact_response_residuals_do_not_erase_measurement_uncertainty() -> None:
    bundle, domain, _ = response_study()
    noise = (0.0002, 0.0002, 0.0002, 0.002)
    observations = tuple(
        replace(
            s,
            first=replace(s.first, uncertainty_m_rad=noise),
            second=replace(s.second, uncertainty_m_rad=noise),
        )
        for s in bundle.observations
    )
    bundle = replace(bundle, observations=observations)
    config = ResponseConfig(uncertainty_floor_m_rad=(1e-9,) * 4)
    fit = estimate_response(bundle, domain, config=config)
    assert fit.accepted_ids
    assert np.all(np.asarray(fit.metadata["residual_floor_m_rad"]) >= 2 * np.asarray(noise))
    fit = estimate_response(bundle, domain, config=replace(config, confidence_multiplier=100))
    assert not fit.accepted_ids and not fit.estimates[0].supported_components


def test_board_neutral_numerical_regression() -> None:
    result = numerical_regression()
    assert result["provenance"] == "synthetic" and not result["hardware_accuracy_certified"]


@pytest.mark.parametrize(
    "change",
    [{"trials": True}, {"max_pairs": 100000}, {"rank_rcond": 0}, {"point_gate_m": float("nan")}],
)
def test_solve_work_is_bounded(change: dict) -> None:
    with pytest.raises(ValueError):
        SolveConfig(**change)

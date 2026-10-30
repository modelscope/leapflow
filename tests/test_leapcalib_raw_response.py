# Copyright (c) Alibaba, Inc. and its affiliates.
"""Measured-encoder subspaces and independent evaluation; synthetic, not metrology."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.observations import JointObservation, RgbdObservation, Stamp
from leapcalib.contracts.raw_task import RawTaskDomain, RawTaskState, RawTaskTransition
from leapcalib.contracts.task import TaskState
from leapcalib.estimation.raw_response import estimate_raw_response, propose_raw, update_raw_response
from leapcalib.validation.bundle import EvidenceBundle, RAW_SCHEMA_VERSION, Retention
from tools.calib_regression import response_study, topology


def raw_study(*, rank: int = 4, outliers: bool = False) -> tuple[EvidenceBundle, RawTaskDomain, np.ndarray]:
    _, task_domain, _ = response_study()
    topo = topology()
    domain = RawTaskDomain(task_domain, tuple(f"j{i}" for i in range(6)), (0.0,) * 6, (-0.1,) * 6, (0.1,) * 6, (0.01,) * 6, "synthetic-scope")
    rng = np.random.default_rng(87)
    basis = np.linalg.qr(rng.normal(size=(6, rank)))[0].T
    c = np.zeros((4, rank))
    c[:min(4, rank), :min(4, rank)] = np.eye(min(4, rank))
    j = c @ basis
    observations, accepted, heldout = [], [], []
    for i in range(60):
        start = basis.T @ rng.uniform(-2, 2, rank) * 0.01
        delta = basis.T @ rng.uniform(-0.5, 0.5, rank) * 0.01
        states = []
        feature_delta = j @ (delta / domain.max_step_rad) * task_domain.tolerance_m_rad
        if outliers and i < 5:
            feature_delta += (0.02, -0.03, 0.04, 0.3)
        for endpoint, raw in enumerate((start, start + delta)):
            identity = f"r{i}:{endpoint}"
            stamp = Stamp(i * 3 + endpoint, uncertainty_s=0.001)
            task = TaskState(identity, stamp, (0, 0, 0.3, 0), tuple(np.array((0, 0, 0.3, 0)) + endpoint * feature_delta), task_domain.feature_id, task_domain.domain_id, True, True, (0.00001,) * 4)
            frame_id, joint_id = identity + ":frame", identity + ":feedback"
            observations.extend((RgbdObservation(frame_id, topo.streams[0], stamp, "synthetic"), JointObservation(joint_id, topo.model, stamp, tuple(raw), still=True)))
            states.append(RawTaskState(task, domain.joint_ids, tuple(raw), (0.00001,) * 6, (frame_id,), domain.scope_digest, stamp, True, (joint_id,)))
        transition = RawTaskTransition(f"r{i}", topo.streams[0], topo.model, *states, "synthetic")
        observations.append(transition)
        (accepted if i < 40 else heldout).append(transition.observation_id)
    return EvidenceBundle("raw-study", topo, tuple(observations), 0, Retention(3600), tuple(accepted), holdout_ids=tuple(heldout), schema_version=RAW_SCHEMA_VERSION), domain, j


def test_raw_contract_roundtrip_and_legacy_refusal() -> None:
    bundle, _, _ = raw_study()
    assert EvidenceBundle.from_mapping(bundle.to_dict()) == bundle
    with pytest.raises(ValueError, match="require evidence v2"):
        replace(bundle, schema_version="leapcalib.evidence.v1")
    transition = bundle.observations[4]
    with pytest.raises(ValueError, match="explicit node clock"):
        replace(transition.first, feedback_stamp=Stamp(0, domain="host_monotonic"))
    with pytest.raises(ValueError, match="matching feedback"):
        replace(bundle, observations=tuple(o for o in bundle.observations if o.observation_id != transition.first.feedback_ids[0]))
    with pytest.raises(ValueError, match="measured raw endpoint"):
        replace(bundle, observations=tuple(replace(o, first=replace(o.first, joints_rad=(0.0,) * 6)) if o is transition else o for o in bundle.observations))


def test_raw_4xn_subspace_controls_outputs_without_individual_columns() -> None:
    bundle, domain, expected = raw_study(outliers=True)
    fit = estimate_raw_response(bundle, domain)
    assert fit.metadata["response_scaled"] == pytest.approx(expected)
    assert fit.metadata["data_rank"] == 4
    assert fit.metadata["response_rank"] == 4
    assert len(fit.accepted_ids) == 35
    assert fit.metadata["structurally_qualified"] is False
    assert len(fit.metadata["holdout_predictions"]) == 20
    assert fit.metadata["prior_rows_in_rank"] == 0
    assert all(v > 0 for v in fit.metadata["residual_floor_m_rad"])
    basis = np.asarray(fit.metadata["supported_basis_scaled"])
    assert np.all(np.diag(basis.T @ basis) < 0.999)  # No individually identifiable joint needed.
    source = next(o for o in bundle.observations if o.observation_id == "r45")
    current = replace(source.first, joints_rad=(0.0,) * 6)
    proposal = propose_raw(fit, current, (0.001, -0.001, 0.301, 0.01), Stamp(current.task.stamp.seconds + 0.01), run_id="run", model_digest=bundle.topology.model.digest)
    dq = np.asarray(proposal["raw_targets_rad"]) / domain.max_step_rad
    assert np.linalg.norm(dq - basis.T @ basis @ dq) < 1e-10
    assert proposal["executed"] is False
    assert max(abs(np.asarray(proposal["raw_targets_rad"]))) <= 0.01


def test_holdout_values_never_train_and_shared_feedback_is_refused() -> None:
    bundle, domain, _ = raw_study()
    original = estimate_raw_response(bundle, domain)
    changed = replace(bundle, observations=tuple(replace(o, second=replace(o.second, task=replace(o.second.task, feature_m_rad=(9, 8, 7, 0)))) if isinstance(o, RawTaskTransition) and o.observation_id in bundle.holdout_ids else o for o in bundle.observations))
    assert estimate_raw_response(changed, domain).estimates == original.estimates
    source = next(o for o in bundle.observations if o.observation_id == "r40")
    with pytest.raises(ValueError, match="evaluation/training"):
        update_raw_response(original, source)
    training = next(o for o in bundle.observations if o.observation_id == "r0")
    with pytest.raises(ValueError, match="share source"):
        replace(bundle, accepted_ids=(*bundle.accepted_ids, source.first.feedback_ids[0]))
    with pytest.raises(ValueError, match="inconsistent"):
        replace(bundle, observations=(*bundle.observations, replace(source, observation_id="copy", first=replace(training.first, joints_rad=(0.0,) * 6))))


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_insufficient_controllable_rank_retains_prior(rank: int) -> None:
    bundle, domain, _ = raw_study(rank=rank)
    fit = estimate_raw_response(bundle, domain)
    assert fit.estimates[0].support == "retained"
    assert fit.metadata["supported_basis_scaled"] == []
    source = next(o for o in bundle.observations if o.observation_id == "r40")
    with pytest.raises(CalibrationFailure):
        propose_raw(fit, source.first, (0, 0, 0.3, 0), source.first.task.stamp, run_id="run", model_digest=bundle.topology.model.digest)


def test_bounded_recursive_update_cannot_expand_support_or_erase_floor() -> None:
    bundle, domain, _ = raw_study()
    fit = estimate_raw_response(bundle, domain)
    sample = next(o for o in bundle.observations if o.observation_id == "r39")
    def fresh(state: RawTaskState) -> RawTaskState:
        return replace(state, task=replace(state.task, state_id="new:" + state.task.state_id), frame_ids=("new:" + state.frame_ids[0],), feedback_ids=("new:" + state.feedback_ids[0],))
    sample = replace(sample, observation_id="ordinary-correction", first=fresh(sample.first), second=fresh(sample.second))
    updated = update_raw_response(fit, sample)
    assert updated.metadata["holdout_predictions"] == fit.metadata["holdout_predictions"]
    assert np.linalg.norm(np.asarray(updated.estimates[0].values) - fit.estimates[0].values) <= 0.05
    assert updated.metadata["residual_floor_m_rad"] >= fit.metadata["residual_floor_m_rad"]
    with pytest.raises(ValueError, match="replayed"):
        update_raw_response(updated, sample)

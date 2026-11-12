# Copyright (c) Alibaba, Inc. and its affiliates.
"""Software fixtures for stationary bridge geometry, evidence and governance."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import pytest

from leapcalib.contracts.identity import digest_of
from leapcalib.estimation.bridge import BridgeBudget, BridgePair, bridge_transform, fit_se3, solve_bridge
from leapcalib.geometry.se3 import Transform
from leapcalib.runtime.protocol import WorkerBudgets, WorkerRequest, validate_payload
from leapcalib.runtime.server import compute
from leapcalib.validation.evaluation import EvaluationConfig
from leapflow.hardware.calibration.paired import PairedObservation, PairedObservationSource
from leapflow.hardware.skills.ports import PairedObservationPort, BridgeCalibrationPort
from leapflow.hardware.skills.geometry import Transform as HardwareTransform
from leapflow.hardware.skills.vision import BoardDetection, CharucoTarget
from leapflow.hardware.skills.protocol import FAILURE_CHECK_FAILED, FAILURE_STOPPED, FAILURE_TIMEOUT, SkillError
from tests.test_leapcalib_evidence import captured, registry_and_cache, record, binding, sealed_plan
from tests.test_leapcalib_acquisition import frame, profile, sources

pytestmark = pytest.mark.component


def budget(**changes):
    return BridgeBudget(**{"terms": {k: (0.00005, 0.0001) for k in ("robot_fk", "wrist_mount", "board_scale_flatness", "intrinsics", "pnp_depth", "timing", "repeatability")},
                          "max_error_m_rad": (0.003, math.radians(2)), "confidence_multiplier": 2,
                          "min_training": 3, "min_holdout": 3, "separation_s": 0.3, **changes})


def pair(i=0, fixed=None):
    arm = Transform.from_rpy_rad(0.2 + i * 0.01, -0.1, 0.3, (0.25, 0.05, 0.37), parent="base", child="flange")
    mount = Transform.from_rpy_rad(0.1, 0.2, -0.1, (-0.01, -0.08, 0.046), parent="flange", child="wrist")
    fixed = fixed or Transform.from_rpy_rad(-0.3, 0.2, 0.4, (0.3, -0.2, 0.8), parent="base", child="fixed")
    board = Transform.from_rpy_rad(0.0, 0.0, 0.1, (0.3, 0.0, 0.1), parent="base", child="board")
    wrist_board = arm.compose(mount).inverse().compose(board)
    fixed_board = fixed.inverse().compose(board)
    return BridgePair(digest_of({"fixture": i}), tuple(f"fixture-{i}-{j}" for j in range(6)), (i * 2 + 1, i * 2 + 2), arm, wrist_board, fixed_board), mount, fixed


def test_direction_units_and_se3_fit_with_heldout_evaluation():
    pairs = [pair(i)[0] for i in range(6)]
    _, mount, fixed = pair()
    assert bridge_transform(pairs[0], mount).distance_m(fixed) < 1e-10
    result = solve_bridge(pairs[:3], pairs[3:], flange_T_wrist=mount, dependency_digest="a" * 64, dependency_sources=[], budget=budget())
    assert result["verdict"] == "numerical_pass"
    assert Transform.from_mapping(result["candidate"]).distance_m(fixed) < 1e-10
    assert result["scope"] == "conditional_fixed_camera_cross_view"
    assert result["physical_activation"] == "shadow_only"
    assert result["independent_reference_accuracy"] is None
    assert not result["absolute_base_accuracy_established"] and not result["joint_wrist_fit"]


def test_se3_fit_crosses_euler_wrap_and_refuses_antipodal_branch():
    transforms = [Transform.from_rpy_rad(0, 0, math.radians(v)) for v in (179, -179)]
    fit = fit_se3(transforms)
    assert fit.angle_to_rad(Transform.from_rpy_rad(0, 0, math.pi)) < 1e-6
    with pytest.raises(ValueError, match="ambiguous"):
        fit_se3([Transform.identity(), Transform.from_rpy_rad(0, 0, math.pi)])


@pytest.mark.parametrize("defect", ["source", "window", "wrist_refinement"])
def test_source_and_physical_window_overlap_are_nonwaivable(defect):
    pairs = [pair(i)[0] for i in range(6)]
    dependency_sources = []
    if defect == "source":
        pairs[3] = replace(pairs[3], source_ids=pairs[0].source_ids)
    elif defect == "window":
        pairs[3] = replace(pairs[3], interval_s=pairs[0].interval_s)
    else:
        dependency_sources = [pairs[3].source_ids[0]]
    with pytest.raises(ValueError, match="overlap"):
        solve_bridge(pairs[:3], pairs[3:], flange_T_wrist=pair()[1], dependency_digest="a" * 64,
                     dependency_sources=dependency_sources, budget=budget())


def test_quality_rejection_retains_candidate_and_systematic_floors():
    pairs = [pair(i)[0] for i in range(6)]
    b = budget(terms={k: (0.002, 0.001) for k in budget().terms})
    result = solve_bridge(pairs[:3], pairs[3:], flange_T_wrist=pair()[1], dependency_digest="a" * 64, dependency_sources=[], budget=b)
    assert result["verdict"] == "rejected" and result["candidate"] is not None
    assert result["uncertainty_m_rad"][0] == pytest.approx(0.014)
    assert not result["reachability"]["reachable"] and result["advisories"]
    diagnostic = EvaluationConfig().reachability((0.002, 0.002, 0.002, 0.001))
    assert not diagnostic["reachable"] and diagnostic["available_residual_m_rad"][0] == pytest.approx(-0.001)


@pytest.mark.parametrize("rotation", [False, True])
def test_stationary_epoch_change_cannot_cancel_through_both_views(rotation):
    pairs = [pair(i)[0] for i in range(6)]
    mount = pair()[1]
    moved = pairs[3]
    drift = Transform.from_rpy_rad(0, 0, math.radians(10) if rotation else 0, (0 if rotation else .02, 0, 0), parent="base", child="base")
    base_board = moved.base_T_flange.compose(mount).compose(moved.wrist_T_board)
    new_board = drift.compose(base_board)
    fixed = pair()[2]
    pairs[3] = replace(moved, wrist_T_board=moved.base_T_flange.compose(mount).inverse().compose(new_board),
                       fixed_T_board=fixed.inverse().compose(new_board))
    # The bridge chain still cancels exactly, so residual fitting alone is not a
    # stationary-target witness. The separate epoch gate must refuse it.
    assert bridge_transform(pairs[3], mount).distance_m(fixed) < 1e-10
    with pytest.raises(ValueError, match="stationary board/fixed view changed"):
        solve_bridge(pairs[:3], pairs[3:], flange_T_wrist=mount, dependency_digest="a" * 64,
                     dependency_sources=[], budget=budget())


def test_withheld_noise_never_refits_candidate_or_shrinks_systematic_budget():
    pairs = [pair(i)[0] for i in range(6)]
    kwargs = {"flange_T_wrist": pair()[1], "dependency_digest": "a" * 64, "dependency_sources": [], "budget": budget()}
    baseline = solve_bridge(pairs[:3], pairs[3:], **kwargs)
    noisy = replace(pairs[3], wrist_T_board=replace(pairs[3].wrist_T_board,
                    translation_m=(pairs[3].wrist_T_board.translation_m[0] + .02, *pairs[3].wrist_T_board.translation_m[1:])))
    rejected = solve_bridge(pairs[:3], [noisy, *pairs[4:]], **kwargs)
    assert rejected["verdict"] == "rejected"
    assert rejected["candidate_digest"] == baseline["candidate_digest"]
    assert rejected["training_residual_m_rad"] == baseline["training_residual_m_rad"]
    assert rejected["effective_uncertainty_terms"] == baseline["effective_uncertainty_terms"]
    assert rejected["observability"]["joint_wrist_rank_established"] is False


def test_worker_bridge_exact_payload_model_and_budget_validation():
    pairs = [pair(i)[0].to_dict() for i in range(6)]
    payload = {"training": pairs[:3], "holdout": pairs[3:], "flange_T_wrist": pair()[1].to_dict(),
               "dependency_digest": "a" * 64, "dependency_sources": [], "budget": budget().to_dict(), "model_digest": "m"}
    request = WorkerRequest("r", "j", "c", "run", "m", "solve_bridge", 1e12, payload, digest_of(payload))
    assert compute(request, WorkerBudgets())["bridge"]["verdict"] == "numerical_pass"
    for bad in ({**payload, "activate": True}, {**payload, "model_digest": "other"}, {**payload, "training": pairs * 40}):
        request = replace(request, payload=bad, input_digest=digest_of(bad))
        with pytest.raises(ValueError):
            validate_payload(request, WorkerBudgets())


def detection():
    ids = (0, 1, 2, 4, 5, 6)
    return BoardDetection(HardwareTransform(translation=(0, 0, 500)), 6, 0.1, ids,
                          ((3, 2), (5, 2), (7, 2), (3, 4), (5, 4), (7, 4)),
                          tuple(((i % 4 + 1) * 10, (i // 4 + 1) * 10, 0) for i in ids), pose_gap_px=1.0)


def paired(value, **changes):
    fixed_roi = replace(value.roi, profile=replace(value.roi.profile, role="fixed", device_id="fixed-camera"), frame_id="fixed-frame")
    return PairedObservation(**{"fixed": replace(value, roi=fixed_roi), "wrist": value,
                              "target_json": json.dumps(CharucoTarget(5, 4, 10, 7).to_dict()), "target_id": "sheet",
                              "stationary_epoch": "sheet-epoch", "fixed_detection": detection(), "wrist_detection": detection(),
                              "min_pose_gap_px": 0.5, **changes})


def paired_binding(value):
    basis = {"bridge": {"target_id": value.target_id, "stationary_epoch": value.stationary_epoch,
                        "target": json.loads(value.target_json), "min_pose_gap_px": value.min_pose_gap_px,
                        "model_digest": value.feedback[0].model_digest, "declaration_digest": value.feedback[0].declaration_digest,
                        "joint_ids": list(value.feedback[0].joint_ids),
                        "cameras": {role: {"camera_device_id": view.roi.profile.device_id, "mount_digest": view.roi.mount_digest,
                                           "camera_session_id": view.roi.session_id, "metric_profile_digest": view.roi.profile.digest,
                                           "source_profile_digest": view.roi.source_profile_digest,
                                           "region": [view.roi.x, view.roi.y, view.roi.width, view.roi.height]}
                                    for role, view in (("fixed", value.fixed), ("wrist", value.wrist))}}}
    return binding(sealed_plan(basis=basis))


async def test_paired_indexed_lossless_roundtrip_and_exclusion_tamper_refusal(tmp_path, monkeypatch):
    single, _ = await captured(monkeypatch)
    value = paired(single)
    registry, cache, store = registry_and_cache(tmp_path)
    try:
        summary = await asyncio.to_thread(record, store, value, binding=paired_binding(value))
        assert summary["schema"] == "leapflow.paired_acquisition.v1" and not summary["hardware_synchronized"]
        raw = await store.read_capture(summary["evidence_digest"], workcell_id="cell")
        assert PairedObservation.from_mapping(raw["observation"]) == value
        assert raw["observation"]["fixed_detection"]["corner_ids"] == list(detection().corner_ids)
        excluded = paired(single, fixed_detection=None, exclusions=("target_not_found",))
        excluded_summary = await asyncio.to_thread(record, store, excluded, binding=paired_binding(excluded))
        assert excluded_summary["exclusions"] == ["target_not_found"]
        with pytest.raises(ValueError, match="recomputed"):
            replace(excluded, exclusions=())
        with pytest.raises(ValueError, match="one stationary"):
            replace(value, wrist=replace(value.wrist, feedback=tuple(replace(f, source_id="x" + f.source_id) for f in value.feedback)))
        bad = replace(value.wrist, roi=replace(value.wrist.roi, stamp=replace(value.wrist.roi.stamp, seconds=value.feedback[-1].first_receipt_s + 1)))
        with pytest.raises(ValueError, match="not bracketed"):
            await asyncio.to_thread(record, store, replace(value, wrist=bad), binding=paired_binding(value))
        assert all(entry.sensitive and not entry.syncable for entry in cache.list_entries())
    finally:
        await registry.close_all()


@pytest.mark.parametrize("defect", ["none", "typed_timeout", "io_dip", "permanent_timeout", "identity", "partial",
                                  "cancel_capture", "cancel_detection", "blocked_after_detection", "missing_vision", "native_count"])
async def test_paired_source_reuses_raw_stillness_and_shared_capture_group(monkeypatch, defect):
    registry, _, feedback, clock = await sources(monkeypatch)
    count = {"fixed": 1, "wrist": 1}
    stopped = False
    cameras, bindings = {}, {}
    for role in count:
        camera_frame = replace(frame(clock()), device_id=f"{role}-camera")
        from leapflow.hardware.calibration.measurements import metric_roi
        roi = metric_roi(camera_frame, region=(2, 1, 8, 8), role=role, receipt_uncertainty_s=0.001, exposure_uncertainty_s=0.002)
        bindings[role] = {"camera_device_id": camera_frame.device_id, "mount_digest": roi.mount_digest, "camera_session_id": roi.session_id,
                          "source_profile_digest": roi.source_profile_digest, "metric_profile_digest": roi.profile.digest,
                          "region": [2, 1, 8, 8], "source_frame_id": roi.frame_id, "source_stamps": [vars(s) for s in roi.source_stamps]}
        class Camera:
            def __init__(self, role):
                self.role = role
            async def capture_metric(self, *, after_monotonic):
                count[self.role] += 1
                if self.role == "wrist":
                    if defect == "permanent_timeout" or defect == "typed_timeout" and count[self.role] == 2:
                        raise SkillError(FAILURE_TIMEOUT, "sensor timeout fixture")
                    if defect == "io_dip" and count[self.role] == 2:
                        raise OSError("sensor transient fixture")
                    if defect == "partial":
                        raise SkillError(FAILURE_CHECK_FAILED, "sensor unavailable fixture")
                    if defect == "cancel_capture":
                        raise asyncio.CancelledError
                value = replace(frame(clock()), device_id=f"{self.role}-camera", sequence=count[self.role])
                metadata = {**value.metadata, **{key: {**value.metadata[key], "frame_number": count[self.role] + 100,
                                                     "timestamp_ms": count[self.role] * 2000} for key in ("color_source", "depth_source")}}
                if self.role == "wrist" and defect == "identity":
                    metadata["mount_digest"] = "changed-mount"
                if self.role == "wrist" and defect == "native_count":
                    metadata.pop("depth_source")
                return replace(value, metadata=metadata)
        cameras[role] = Camera(role)
    def detector(f, t):
        nonlocal stopped
        if defect == "cancel_detection":
            raise asyncio.CancelledError
        if defect == "missing_vision":
            raise ImportError("missing detector fixture")
        if defect == "blocked_after_detection":
            stopped = True
        return detection()
    monkeypatch.setattr("leapflow.hardware.calibration.paired.detect_board", detector)
    source = PairedObservationSource(feedback, cameras, profile=profile(), bindings=bindings, target=CharucoTarget(5, 4, 10, 7),
                                    target_id="sheet", stationary_epoch="epoch", min_pose_gap_px=0.5, tolerance_ms=50, attempts=2,
                                    blocked=lambda: stopped, clock=clock, sleep=clock.sleep)
    assert isinstance(source, PairedObservationPort)
    try:
        if defect in ("cancel_capture", "cancel_detection"):
            with pytest.raises(asyncio.CancelledError):
                await source.capture()
        elif defect not in ("none", "typed_timeout", "io_dip"):
            from leapflow.hardware.skills.protocol import FAILURE_SKILL_UNAVAILABLE
            expected = FAILURE_TIMEOUT if defect == "permanent_timeout" else FAILURE_STOPPED if defect == "blocked_after_detection" else FAILURE_SKILL_UNAVAILABLE if defect == "missing_vision" else FAILURE_CHECK_FAILED
            with pytest.raises(SkillError) as caught:
                await source.capture()
            assert caught.value.failure_code == expected
            assert count == {"fixed": 3, "wrist": 3} if defect == "permanent_timeout" else count == {"fixed": 2, "wrist": 2}
        else:
            value = await source.capture()
            assert value.fixed.feedback == value.wrist.feedback and not value.exclusions
            assert count == {"fixed": 3, "wrist": 3} if defect != "none" else count == {"fixed": 2, "wrist": 2}
    finally:
        await registry.close_all()


def test_diagonal_common_corner_geometry_is_not_observable():
    from leapflow.hardware.calibration.paired import observable_corners
    target = CharucoTarget(7, 7, 20, 15)
    assert not observable_corners((0, 7, 14, 21, 28, 35), target)
    assert not observable_corners((0, 1, 2, 3, 4, 5), target)
    assert observable_corners((0, 1, 6, 7, 12, 13), target)


def test_builtin_bridge_declares_no_motion_and_registry_entrypoint():
    from leapflow.hardware.skills.builtin.bridge_calibration import skills
    from leapflow.hardware.calibration.bridge_port import WorkcellBridge
    assert skills()[0].spec.skill_id == "calib.bridge"
    assert not skills()[0].spec.motion_allowed and skills()[0].spec.leased
    assert isinstance(WorkcellBridge.__new__(WorkcellBridge), BridgeCalibrationPort)

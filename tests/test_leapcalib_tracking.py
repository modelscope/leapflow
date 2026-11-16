# Copyright (c) Alibaba, Inc. and its affiliates.
"""Synthetic metric tracking fixtures test software, never real camera qualification."""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.identity import StreamProfile, digest_of
from leapcalib.contracts.observations import Stamp
from leapcalib.contracts.tracking import MetricRoi, SourceStamp, TrackingConfig
from leapcalib.estimation.tracking import track_metric
from leapcalib.runtime.protocol import WorkerBudgets, WorkerRequest, validate_payload
from leapcalib.runtime.server import compute
from leapflow.hardware.calibration.acquisition import (
    NOMINAL_FLANGE,
    FeedbackSnapshot,
    SettledObservation,
)
from leapflow.hardware.calibration.reference import (
    ReferenceProfile,
    validate_metric_tracking_result,
    validate_tracking_result,
)
from leapflow.hardware.calibration.worker import CalibrationWorker
from leapflow.hardware.skills.geometry import Pose

pytestmark = pytest.mark.component


def pair(width: int = 160, height: int = 120) -> tuple[MetricRoi, MetricRoi]:
    import cv2
    import numpy as np

    rng = np.random.default_rng(11)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (3, 3), 0)
    color = np.stack([gray] * 3, axis=-1)
    shifted = cv2.warpAffine(
        color, np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.float32), (width, height)
    )
    profile = StreamProfile(
        "synthetic-camera", "depth_aligned", width, height, 500, 500, width / 2, height / 2, 0.001
    )
    depth = np.full((height, width), 0.5, dtype="<f4").tobytes()
    first = MetricRoi(
        "frame1",
        profile,
        "mount",
        "session",
        "source-profile",
        Stamp(1, uncertainty_s=0.001),
        0.002,
        0,
        0,
        width,
        height,
        color.tobytes(),
        depth,
        "synthetic",
    )
    return first, replace(
        first,
        frame_id="frame2",
        stamp=Stamp(1.1, uncertainty_s=0.001),
        color_bgr8=shifted.tobytes(),
    )


def test_lossless_metric_roi_roundtrip_and_no_preview_input() -> None:
    first, _ = pair()
    assert MetricRoi.from_mapping(first.to_dict()) == first
    with pytest.raises(ValueError, match="buffer"):
        MetricRoi.from_mapping({**first.to_dict(), "color_bgr8": "JPEG"})
    with pytest.raises(ValueError, match="before buffer"):
        MetricRoi.from_mapping({**first.to_dict(), "width": 641})
    with pytest.raises(ValueError, match="timing floors"):
        replace(first, exposure_uncertainty_s=0)


def test_persistent_fb_patch_depth_consensus_and_no_authority() -> None:
    first, second = pair()
    result = track_metric(first, second)
    assert len(result["tracks"]) >= 12
    assert result["delta_m_rad"][:3] == pytest.approx([0.001, 0, 0], abs=0.00005)
    assert result["delta_m_rad"][3] == pytest.approx(0, abs=0.0001)
    assert min(result["uncertainty_m_rad"]) > 0
    assert result["timing_uncertainty_s"] == pytest.approx(0.006)
    assert not result["witness_valid"]
    assert not result["rigid_link_identity_established"]
    assert not result["executed"] and not result["hardware_accuracy_certified"]
    third = replace(second, frame_id="frame3", stamp=Stamp(1.2, uncertainty_s=0.001))
    continued = track_metric(second, third, tracks=result["persistent_tracks"])
    assert set(continued["registration"]["accepted_features"]) <= set(
        result["registration"]["accepted_features"]
    )
    assert continued["delta_m_rad"] == pytest.approx([0, 0, 0, 0], abs=0.0001)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mount_digest", "other"),
        ("session_id", "other"),
        ("source_profile_digest", "changed-fps"),
        ("frame_id", "frame1"),
        ("stamp", Stamp(1, uncertainty_s=0.001)),
    ],
)
def test_identity_or_stale_frames_refuse(field: str, value: object) -> None:
    first, second = pair()
    with pytest.raises(CalibrationFailure):
        track_metric(first, replace(second, **{field: value}))


def test_blank_occlusion_bad_depth_and_reused_features_refuse() -> None:
    first, second = pair()
    with pytest.raises(CalibrationFailure):
        track_metric(first, replace(second, color_bgr8=bytes(len(second.color_bgr8))))
    with pytest.raises(CalibrationFailure):
        track_metric(first, replace(second, depth_m_f32le=bytes(len(second.depth_m_f32le))))
    result = track_metric(first, second)
    with pytest.raises(ValueError, match="preceding source"):
        track_metric(first, second, tracks=result["persistent_tracks"])


def test_worker_tracking_dispatch_and_count_budget() -> None:
    first, second = pair()
    payload = {
        "first": first.to_dict(),
        "second": second.to_dict(),
        "tracks": [],
        "config": vars(TrackingConfig()),
    }
    request = WorkerRequest(
        "request",
        "job",
        "cell",
        "run",
        "model",
        "track_metric",
        time.monotonic() + 2,
        payload,
        digest_of(payload),
    )
    result = compute(request, WorkerBudgets())
    assert result["tracking"]["second_digest"] == second.digest
    with pytest.raises(ValueError, match="feature budget"):
        validate_payload(request, WorkerBudgets(max_tracks=12))
    with pytest.raises(ValueError, match="dimensions"):
        validate_payload(request, WorkerBudgets(roi_width=100))


def test_native_source_clock_and_sequence_are_not_receipt_time() -> None:
    first, second = pair()
    sources = (
        SourceStamp("color", 20, 1234, "hardware"),
        SourceStamp("depth", 21, 1234.1, "hardware"),
    )
    first = replace(first, source_stamps=sources)
    with pytest.raises(CalibrationFailure, match="native stream"):
        track_metric(first, replace(second, source_stamps=sources))
    with pytest.raises(ValueError, match="receipt stamp"):
        MetricRoi.from_mapping(
            {**first.to_dict(), "stamp": {**first.stamp.to_dict(), "unbound": True}}
        )
    with pytest.raises(ValueError, match="source clock"):
        MetricRoi.from_mapping({**first.to_dict(), "source_stamps": [{}]})


async def test_maximum_lossless_roi_through_real_compute_subprocess(tmp_path: Path) -> None:
    first, second = pair(640, 480)
    payload = {
        "first": first.to_dict(),
        "second": second.to_dict(),
        "tracks": [],
        "config": vars(TrackingConfig()),
    }
    worker = CalibrationWorker(
        tmp_path.resolve().parent / ("lc-" + uuid.uuid4().hex[:8]) / "s", enabled=True
    )
    try:
        result = await worker.submit(
            "track_metric",
            payload,
            workcell_id="cell",
            run_id="synthetic-run",
            model_digest="model",
        )
        assert result["tracking"]["first_digest"] == first.digest
        assert result["tracking"]["second_digest"] == second.digest
        assert result["tracking"]["delta_m_rad"][:3] == pytest.approx([0.001, 0, 0], abs=0.0001)
        assert not result["tracking"]["hardware_accuracy_certified"]
        assert worker.last_metrics["peak_rss_bytes"] <= worker.budgets.rss_bytes
    finally:
        process = worker._process
        await worker.close()
    assert process is not None and process.returncode is not None


def settled_fixture(roi: MetricRoi, joint_rad: float = 0) -> SettledObservation:
    """Typed synthetic measurements exercise validation only, not owner provenance."""
    feedback = tuple(
        FeedbackSnapshot(
            source_id=f"{roi.frame_id}-feedback-{i}",
            device_id="synthetic-arm",
            model_digest="nominal-model",
            joint_ids=("joint",),
            joints_rad=(joint_rad,),
            tcp=Pose(0, 0, 0, 0, 0, 0),
            stamp=Stamp(roi.stamp.seconds + offset, uncertainty_s=0.001),
            first_receipt_s=roi.stamp.seconds + offset,
            last_receipt_s=roi.stamp.seconds + offset,
            enabled=True,
            arrived=True,
            ctrl_mode="joint",
            arm_status=0,
            communication_errors=(),
            node_session_id="node-boot",
            declaration_digest="declaration",
            pose_semantics=NOMINAL_FLANGE,
            joint_uncertainty_rad=0.0001,
            tcp_uncertainty_mm=0.01,
            rotation_uncertainty_deg=0.005,
        )
        for i, offset in enumerate((-0.36, -0.20, -0.02, 0.02))
    )
    return SettledObservation(roi, feedback, "settled-profile")


@pytest.fixture
def tracked_captures():
    first, second = pair()
    first = replace(
        first,
        source_stamps=(
            SourceStamp("color", 20, 1000, "hardware"),
            SourceStamp("depth", 21, 1000.1, "hardware"),
        ),
    )
    second = replace(
        second,
        stamp=Stamp(2, uncertainty_s=0.001),
        source_stamps=(
            SourceStamp("color", 21, 2000, "hardware"),
            SourceStamp("depth", 22, 2000.1, "hardware"),
        ),
    )
    return settled_fixture(first), settled_fixture(second, 0.01), track_metric(first, second)


def test_shared_moving_tracking_does_not_waive_stationary_reference(tracked_captures) -> None:
    first, second, result = tracked_captures
    config = TrackingConfig()
    ids = validate_metric_tracking_result(result, first, second, config)
    assert len(ids) >= 12
    assert first.feedback[-1].joints_rad != second.feedback[-1].joints_rad
    assert result["delta_m_rad"][0] == pytest.approx(0.001, abs=0.00005)
    reference = ReferenceProfile(config, (0.001, 0.003, 0.003, 0.03), "study", 3, 0.3)
    with pytest.raises(ValueError, match="stationary reference"):
        validate_tracking_result(result, first, second, reference, None)
    assert all(
        result[key] is False
        for key in (
            "witness_valid",
            "rigid_link_identity_established",
            "executed",
            "hardware_accuracy_certified",
        )
    )


@pytest.mark.parametrize(
    "defect",
    [
        "mount",
        "session",
        "profile",
        "settled-profile",
        "source-profile",
        "provenance",
        "repeated-frame",
        "frame-gap",
        "native-repeat",
        "native-domain",
        "native-missing",
    ],
)
def test_shared_tracking_rechecks_capture_domain(tracked_captures, defect: str) -> None:
    first, second, result = tracked_captures
    changes = {
        "mount": {"mount_digest": "other"},
        "session": {"session_id": "other"},
        "profile": {"profile": replace(second.roi.profile, fx_px=501)},
        "source-profile": {"source_profile_digest": "other"},
        "provenance": {"provenance": "metric_rgbd"},
        "repeated-frame": {"frame_id": first.roi.frame_id},
        "frame-gap": {"stamp": Stamp(4, uncertainty_s=0.001)},
        "native-repeat": {"source_stamps": first.roi.source_stamps},
        "native-domain": {
            "source_stamps": tuple(
                replace(s, timestamp_domain="other") for s in second.roi.source_stamps
            )
        },
        "native-missing": {"source_stamps": ()},
    }
    second = (
        replace(second, profile_digest="other")
        if defect == "settled-profile"
        else replace(second, roi=replace(second.roi, **changes[defect]))
    )
    with pytest.raises(ValueError, match="domain|provenance|gap"):
        validate_metric_tracking_result(result, first, second, TrackingConfig())


@pytest.mark.parametrize(
    "changes",
    [
        {"model_digest": "other"},
        {"device_id": "other"},
        {"node_session_id": "other"},
        {"declaration_digest": "other"},
        {"ctrl_mode": "other"},
        {"joint_ids": ("other",)},
        {"enabled": False},
        {"arrived": False},
        {"arm_status": 1},
        {"communication_errors": (1,)},
    ],
)
def test_shared_tracking_rechecks_every_raw_feedback_sample(tracked_captures, changes) -> None:
    first, second, result = tracked_captures
    feedback = list(second.feedback)
    feedback[1] = replace(feedback[1], **changes)
    second = replace(second, feedback=tuple(feedback))
    with pytest.raises(ValueError, match="feedback domain"):
        validate_metric_tracking_result(result, first, second, TrackingConfig())


@pytest.mark.parametrize("defect", ["repeated-source", "reordered", "shared-window", "unbracketed"])
def test_shared_tracking_feedback_lineage_and_capture_bracket(
    tracked_captures, defect: str
) -> None:
    first, second, result = tracked_captures
    feedback = list(second.feedback)
    if defect == "repeated-source":
        feedback[1] = replace(feedback[1], source_id=first.feedback[1].source_id)
    elif defect == "reordered":
        feedback[1], feedback[2] = feedback[2], feedback[1]
    elif defect == "shared-window":
        feedback[0] = replace(first.feedback[-1], source_id="distinct-but-correlated")
    else:
        at = second.roi.stamp.seconds + 0.001
        feedback[-1] = replace(
            feedback[-1],
            stamp=Stamp(at, uncertainty_s=0.001),
            first_receipt_s=at,
            last_receipt_s=at,
        )
    with pytest.raises(ValueError, match="feedback domain|bracket"):
        validate_metric_tracking_result(
            result, first, replace(second, feedback=tuple(feedback)), TrackingConfig()
        )


@pytest.mark.parametrize(
    "field",
    ["witness_valid", "rigid_link_identity_established", "executed", "hardware_accuracy_certified"],
)
def test_shared_tracking_never_promotes_worker_flags(tracked_captures, field: str) -> None:
    first, second, result = tracked_captures
    with pytest.raises(ValueError, match="physical authority"):
        validate_metric_tracking_result({**result, field: True}, first, second, TrackingConfig())


def test_shared_tracking_recomputes_metric_values_and_continuity(tracked_captures) -> None:
    import copy

    first, second, result = tracked_captures
    changed = copy.deepcopy(result)
    xyz = changed["tracks"][0]["second_m"]
    changed["tracks"][0]["second_m"] = (xyz[0] + 0.0001, *xyz[1:])
    with pytest.raises(ValueError, match="raw depth/pixel"):
        validate_metric_tracking_result(changed, first, second, TrackingConfig())
    ids = validate_metric_tracking_result(result, first, second, TrackingConfig())
    third_roi = replace(
        second.roi,
        frame_id="frame3",
        stamp=Stamp(3, uncertainty_s=0.001),
        source_stamps=tuple(
            replace(s, frame_number=s.frame_number + 1, timestamp_ms=s.timestamp_ms + 1000)
            for s in second.roi.source_stamps
        ),
    )
    third = settled_fixture(third_roi, 0.01)
    continued = track_metric(second.roi, third.roi, tracks=result["persistent_tracks"])
    validate_metric_tracking_result(
        continued, second, third, TrackingConfig(), ids, previous_tracks=result["persistent_tracks"]
    )
    switched = copy.deepcopy(result["persistent_tracks"])
    pixel = switched[0]["pixel"]
    switched[0]["pixel"] = [pixel[0] + 0.1, pixel[1]]
    with pytest.raises(ValueError, match="switched"):
        validate_metric_tracking_result(
            continued, second, third, TrackingConfig(), ids, previous_tracks=switched
        )

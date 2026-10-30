# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only acquisition via real registry construction and boundary-only I/O fixtures."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import time
from array import array
from dataclasses import replace
from typing import Any

import pytest

from leapcalib.contracts.observations import CLOCK_NODE_MONOTONIC, STAMP_RECEIPT
from leapcalib.contracts.tracking import MetricRoi, TrackingConfig
from leapflow.hardware.calibration.acquisition import (
    NOMINAL_FLANGE,
    RegistryFeedbackSource,
    SettledObservation,
    SettledObservationSource,
    SettledProfile,
)
from leapflow.hardware.calibration.adapter import model_identity
from leapflow.hardware.calibration.measurements import metric_roi, metric_track_point
from leapflow.hardware.context import Channel, ContextProvenance, HardwareContext, TransportRef
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.skills.protocol import FAILURE_TIMEOUT, SkillError
from leapflow.hardware.skills.vision import WorkcellVision

pytestmark = pytest.mark.component


def profile(**changes: Any) -> SettledProfile:
    return SettledProfile(
        **{
            "max_feedback_age_s": 0.1,
            "max_coherence_s": 0.02,
            "feedback_uncertainty_s": 0.001,
            "exposure_uncertainty_s": 0.002,
            "receipt_uncertainty_s": 0.001,
            "max_joint_drift_rad": 0.001,
            "max_tcp_drift_mm": 0.2,
            "max_tcp_rotation_deg": 0.1,
            "joint_uncertainty_rad": 0.0001,
            "tcp_uncertainty_mm": 0.01,
            "rotation_uncertainty_deg": 0.005,
            "poll_s": 0.16,
            "timeout_s": 1.0,
            **changes,
        }
    )


def frame(at: float = 42.0) -> RgbdFrame:
    width, height = 16, 12
    return RgbdFrame(
        "camera",
        width,
        height,
        bytes(range(192)) * 3,
        array("H", range(width * height)).tobytes(),
        0.0005,
        CameraIntrinsics(width, height, 200, 201, 8, 6),
        1700000000,
        at,
        7,
        {
            "aligned_to_color": True,
            "mount_digest": "mount",
            "session_id": "session",
            "profile_digest": "source-profile",
            "color_source": {
                "frame_number": 21,
                "timestamp_ms": 1234.0,
                "timestamp_domain": "hardware_clock",
            },
            "depth_source": {
                "frame_number": 22,
                "timestamp_ms": 1234.1,
                "timestamp_domain": "hardware_clock",
            },
        },
    )


class Provider:
    kind = "static"

    def __init__(self, context: HardwareContext) -> None:
        self.context = context

    def discover(self) -> tuple[HardwareContext, ...]:
        return (self.context,)


class Clock:
    def __init__(self) -> None:
        self.now = 42.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


class Camera:
    def __init__(self, clock: Clock, **metadata: Any) -> None:
        self.clock = clock
        self.metadata = metadata
        self.calls: list[float] = []
        self.stale = False

    async def capture_metric(self, *, after_monotonic: float) -> RgbdFrame:
        self.calls.append(after_monotonic)
        value = frame(after_monotonic - 1 if self.stale else self.clock())
        return replace(value, metadata={**value.metadata, **self.metadata})


async def sources(monkeypatch: pytest.MonkeyPatch, **values: Any):
    clock = Clock()
    ids = ("joint", "x", "y", "z", "rx", "ry", "rz", "enabled", "arrived", "mode", "status")
    units = ("deg", "mm", "mm", "mm", "deg", "deg", "deg", "", "", "", "")
    defaults = {
        **dict.fromkeys(ids[:7], 0.0),
        "enabled": True,
        "arrived": True,
        "mode": "joint",
        "status": {"arm_status": 0, "joint_communication_errors": []},
    }
    context = HardwareContext(
        device_id="arm",
        transport=TransportRef("mock", {"values": defaults}),
        channels=tuple(Channel(cid, direction="read", unit=unit) for cid, unit in zip(ids, units)),
        provenance=ContextProvenance(verified_by="software-test"),
    )
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=[Provider(context)])
    registry.load()
    transport = await registry.transport("arm")
    for cid, value in values.items():
        transport.set_value(cid, value)
    read = transport.read

    async def stamped(cid: str):
        return replace(await read(cid), monotonic_at=clock(), observed_at=1700000000)

    monkeypatch.setattr(transport, "read", stamped)
    feedback = RegistryFeedbackSource(
        registry,
        device_id="arm",
        model=model_identity(DhChain((DhLink(100, 0, 0, 0),)), model_id="arm"),
        joint_ids=ids[:1],
        pose_ids=ids[1:7],
        health_ids=ids[7:],
        allowed_modes=("joint",),
        profile=profile(),
        node_session_id="node-boot",
        pose_semantics=NOMINAL_FLANGE,
        clock=clock,
    )
    return registry, transport, feedback, clock


def observation(feedback, camera, clock, **changes: Any):
    return SettledObservationSource(
        feedback,
        camera,
        profile=changes.pop("profile", profile()),
        region=(2, 1, 8, 8),
        role="wrist",
        blocked=changes.pop("blocked", lambda: False),
        camera_device_id="camera",
        mount_digest="mount",
        session_id="session",
        source_profile_digest="source-profile",
        clock=clock,
        sleep=clock.sleep,
        **changes,
    )


def test_lossless_roi_uses_original_intrinsics_and_scale_once() -> None:
    original = frame()
    roi = metric_roi(
        original,
        region=(2, 1, 8, 8),
        role="wrist",
        receipt_uncertainty_s=0.001,
        exposure_uncertainty_s=0.002,
    )
    assert roi.profile.width == 16 and roi.profile.height == 12
    assert (roi.profile.ppx_px, roi.profile.ppy_px) == (8, 6)
    assert roi.color_bgr8[:24] == original.color[54:78]
    assert struct.unpack("<f", roi.depth_m_f32le[:4])[0] == pytest.approx(18 * 0.0005)
    assert roi.stamp.domain == CLOCK_NODE_MONOTONIC and roi.stamp.semantics == STAMP_RECEIPT
    assert roi.stamp.seconds == 42 and roi.source_stamps[0].timestamp_ms == 1234
    assert roi.source_stamps[1].frame_number == 22
    assert MetricRoi.from_mapping(roi.to_dict()) == roi


@pytest.mark.parametrize("distortion", ["none", "brown_conrady", "inverse_brown_conrady"])
def test_owner_metric_track_deprojection_preserves_crop_intrinsics_and_wire_rounding(
    distortion: str,
) -> None:
    from leapcalib.estimation.tracking import _depth, _deproject
    import numpy as np

    value = frame()
    intrinsics = replace(
        value.intrinsics,
        model=distortion,
        coeffs=() if distortion == "none" else (0.01, -0.005, 0.001, 0.001, 0),
    )
    # A cropped ROI with nonzero origin and an even valid-depth count exercises
    # float32 median rounding as well as full-frame distortion/intrinsics.
    value = replace(value, intrinsics=intrinsics)
    roi = metric_roi(
        value,
        region=(1, 1, 14, 10),
        role="wrist",
        receipt_uncertainty_s=0.001,
        exposure_uncertainty_s=0.002,
    )
    depths = [0.5 + (i % 7) * 0.00013 for i in range(roi.width * roi.height)]
    depths[3 * roi.width + 5] = 0
    roi = replace(roi, depth_m_f32le=struct.pack(f"<{len(depths)}f", *depths))
    point, floor = metric_track_point(roi, (7.25, 6.125), TrackingConfig())
    measured = _depth(
        np.frombuffer(roi.depth_m_f32le, dtype="<f4").reshape(roi.height, roi.width),
        (6.25, 5.125),
        roi,
        TrackingConfig(),
        np,
    )
    assert measured is not None
    assert point == pytest.approx(_deproject(roi.profile, 7.25, 6.125, measured[0]), abs=1e-12)
    assert floor == pytest.approx(measured[1], abs=1e-12)


@pytest.mark.parametrize(
    "depths,reason",
    [
        ([0.0] * 25, "insufficient"),
        ([float("nan")] * 25, "nonfinite"),
        ([-0.1] * 25, "negative"),
        ([0.5 + i * 0.01 for i in range(25)], "dispersion"),
    ],
)
def test_owner_metric_track_refuses_invalid_raw_patch(depths: list[float], reason: str) -> None:
    roi = metric_roi(
        frame(),
        region=(0, 0, 16, 12),
        role="wrist",
        receipt_uncertainty_s=0.001,
        exposure_uncertainty_s=0.002,
    )
    raw = bytearray(struct.pack("<192f", *([0.5] * 192)))
    for i, value in enumerate(depths):
        row, column = divmod(i, 5)
        struct.pack_into("<f", raw, 4 * ((4 + row) * 16 + 6 + column), value)
    roi = replace(roi, depth_m_f32le=bytes(raw))
    with pytest.raises(ValueError, match=reason):
        metric_track_point(roi, (8, 6), TrackingConfig())


@pytest.mark.parametrize(
    "key,value",
    [
        ("aligned_to_color", False),
        ("mount_digest", ""),
        ("session_id", None),
        ("profile_digest", ""),
        ("color_source", {}),
        ("depth_source", {"frame_number": 1, "timestamp_ms": 1, "timestamp_domain": None}),
    ],
)
def test_roi_refuses_unknown_identity_or_native_clock(key: str, value: Any) -> None:
    original = frame()
    with pytest.raises(ValueError):
        metric_roi(
            replace(original, metadata={**original.metadata, key: value}),
            region=(0, 0, 8, 8),
            role="fixed",
            receipt_uncertainty_s=0.001,
            exposure_uncertainty_s=0.002,
        )


async def test_real_registry_feedback_and_fresh_settled_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, transport, feedback, clock = await sources(monkeypatch, joint=90)
    try:
        sample = await feedback.sample()
        assert sample.joints_rad == pytest.approx((math.pi / 2,))
        assert sample.pose_semantics == NOMINAL_FLANGE and sample.node_session_id == "node-boot"
        camera = Camera(clock)
        result = await observation(feedback, camera, clock).capture()
        assert len(result.feedback) == 4
        assert result.feedback[2].first_receipt_s - result.feedback[0].last_receipt_s >= 0.3
        assert (
            result.feedback[-2].last_receipt_s
            < result.roi.stamp.seconds
            < result.feedback[-1].first_receipt_s
        )
        assert result.digest and result.roi.provenance == "metric_rgbd"
        assert len(camera.calls) == 1 and transport.write_log == ()
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "changes",
    [
        {"prior_source_stamps": ({}, {})},
        {"prior_source_stamps": (None, None)},
        {"prior_source_stamps": ("color", "depth")},
        {"prior_source_stamps": (None,)},
        {"prior_source_stamps": "color-depth"},
        {"metric_profile_digest": None},
        {"metric_profile_digest": "x" * 257},
        {"prior_frame_id": False},
        {"prior_frame_id": "x" * 257},
    ],
)
async def test_malformed_prior_provenance_refuses_without_io(
    monkeypatch: pytest.MonkeyPatch, changes: Any
) -> None:
    registry, transport, feedback, clock = await sources(monkeypatch)
    camera = Camera(clock)
    try:
        with pytest.raises(ValueError, match="prior|identities"):
            observation(feedback, camera, clock, **changes)
        assert not camera.calls and not transport.write_log
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": False},
        {"enabled": 1.0},
        {"arrived": False},
        {"mode": "cartesian"},
        {"status": {}},
        {"status": {"arm_status": 0, "joint_communication_errors": [1]}},
        {"joint": float("nan")},
    ],
)
async def test_unhealthy_or_malformed_feedback_refuses_before_camera(
    monkeypatch: pytest.MonkeyPatch, values: Any
) -> None:
    registry, transport, feedback, clock = await sources(monkeypatch, **values)
    camera = Camera(clock)
    try:
        with pytest.raises(SkillError):
            await observation(feedback, camera, clock).capture()
        assert camera.calls == [] and transport.write_log == ()
    finally:
        await registry.close_all()


@pytest.mark.parametrize("defect", ["repeated", "stale", "incoherent", "wrong-unit", "drift"])
async def test_bad_receipts_and_measured_drift_refuse(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    registry, transport, feedback, clock = await sources(monkeypatch)
    original = transport.read
    count = 0

    async def defective(cid: str):
        nonlocal count
        count += 1
        reading = await original(cid)
        if defect == "repeated":
            return replace(reading, monotonic_at=42.0, sequence=1)
        if defect == "stale":
            return replace(reading, monotonic_at=clock() - 1)
        if defect == "incoherent":
            return replace(reading, monotonic_at=clock() - (0.03 if cid == "joint" else 0))
        if defect == "wrong-unit":
            return replace(reading, unit="rad" if cid == "joint" else reading.unit)
        return replace(reading, value=1.0) if cid == "joint" and count > 11 else reading

    monkeypatch.setattr(transport, "read", defective)
    try:
        with pytest.raises(SkillError):
            await observation(feedback, Camera(clock), clock).capture()
        assert transport.write_log == ()
    finally:
        await registry.close_all()


@pytest.mark.parametrize(
    "metadata",
    [{"session_id": "reconnected"}, {"mount_digest": "moved"}, {"profile_digest": "changed-fps"}],
)
async def test_frozen_camera_identity_refuses(
    monkeypatch: pytest.MonkeyPatch, metadata: Any
) -> None:
    registry, _, feedback, clock = await sources(monkeypatch)
    try:
        with pytest.raises(SkillError, match="camera role"):
            await observation(feedback, Camera(clock, **metadata), clock).capture()
    finally:
        await registry.close_all()


async def test_settled_evidence_preserves_raw_pose_precision_and_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = dict(
        zip(
            ("x", "y", "z", "rx", "ry", "rz"),
            (
                100.1234567890123,
                -0.0000123456789,
                23.4567890123456,
                0.9167324722093,
                -1.2345678901234,
                2.3456789012345,
            ),
        )
    )
    registry, transport, feedback, clock = await sources(monkeypatch, **values)
    try:
        captured = await observation(feedback, Camera(clock), clock).capture()
        restored = SettledObservation.from_mapping(json.loads(json.dumps(captured.to_dict())))
        assert restored == captured
        assert restored.digest == captured.digest
        assert vars(restored.feedback[-2].tcp) == values
        assert restored.feedback[-2].tcp.to_dict() == {k: round(v, 3) for k, v in values.items()}
        changed = replace(
            captured,
            feedback=tuple(
                replace(f, tcp=replace(f.tcp, rx=f.tcp.rx + 0.00001)) for f in captured.feedback
            ),
        )
        assert changed.digest != captured.digest
        # Existing rounded records remain readable; lost precision cannot be invented.
        legacy = captured.to_dict()
        for sample in legacy["feedback"]:
            sample["tcp"] = {k: round(v, 3) for k, v in sample["tcp"].items()}
        assert SettledObservation.from_mapping(legacy).feedback[-2].tcp.to_dict() == {
            k: round(v, 3) for k, v in values.items()
        }
        assert not transport.write_log
    finally:
        await registry.close_all()


async def test_cancel_timeout_and_active_motion_do_not_retry_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _, feedback, clock = await sources(monkeypatch)
    entered = asyncio.Event()
    camera = Camera(clock)

    async def hang(*, after_monotonic: float):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(camera, "capture_metric", hang)
    try:
        with pytest.raises(SkillError, match="previous dispatch"):
            await observation(feedback, camera, clock, blocked=lambda: True).capture()
        task = asyncio.create_task(observation(feedback, camera, clock).capture())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(SkillError) as caught:
            await observation(feedback, camera, clock, profile=profile(timeout_s=0.4)).capture()
        assert caught.value.failure_code == FAILURE_TIMEOUT
    finally:
        await registry.close_all()


async def test_workcell_vision_metric_capture_uses_existing_registry_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = HardwareContext(
        device_id="camera",
        channels=(Channel("frame", direction="read"),),
        transport=TransportRef("mock", {}),
        provenance=ContextProvenance(verified_by="test"),
    )
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=[Provider(context)])
    registry.load()
    source = await registry.transport("camera")
    calls = []

    async def capture():
        calls.append(True)
        return frame(time.monotonic())

    monkeypatch.setattr(source, "capture_rgbd", capture, raising=False)
    try:
        vision = WorkcellVision(registry=registry, camera="camera")
        result = await vision.capture_metric(after_monotonic=time.monotonic())
        assert result.device_id == "camera" and calls == [True]
        assert registry.get_open_transport("camera") is source
        assert source.write_log == ()
    finally:
        await registry.close_all()

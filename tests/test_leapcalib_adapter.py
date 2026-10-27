# Copyright (c) Alibaba, Inc. and its affiliates.
"""The offline LeapFlow/LeapCalib boundary converts units without moving hardware."""

from __future__ import annotations

import math
import subprocess
import sys
from array import array
from dataclasses import replace

import pytest

from leapcalib.contracts.failures import (
    FAILURE_CODES,
    FAILURE_PROVENANCE_REFUSED,
    CalibrationFailure,
)
from leapcalib.contracts.identity import ROLE_WRIST, SOURCE_FACTORY, digest_of
from leapcalib.contracts.observations import (
    CLOCK_NODE_MONOTONIC,
    PROVENANCE_COLORIZED_DEPTH,
    PROVENANCE_PREVIEW_JPEG,
    STAMP_RECEIPT,
    STAMP_SETTLED_CAPTURE,
    Stamp,
)
from leapflow.hardware.calibration.adapter import (
    CALIBRATION_FAILURE_MAP,
    hardware_failure_code,
    joint_observation,
    model_identity,
    rgbd_observation,
    stream_profile,
    stream_profile_of,
)
from leapflow.hardware.rgbd import CameraIntrinsics, RgbdFrame
from leapflow.hardware.skills import protocol as hw
from leapflow.hardware.skills.kinematics import DhChain, DhLink


def _frame() -> RgbdFrame:
    return RgbdFrame(
        device_id="wrist_camera",
        width=2,
        height=2,
        color=b"\x00" * 12,
        depth=array("H", [1000, 0, 500, 2000]).tobytes(),
        depth_scale_m=0.0005,
        intrinsics=CameraIntrinsics(width=2, height=2, fx=2.5, fy=2.6, ppx=1.0, ppy=1.0),
        observed_at=1700000000.0,
        monotonic_at=42.0,
        sequence=7,
    )


def _chain() -> DhChain:
    return DhChain(links=(DhLink(a=100.0, alpha=90.0, d=200.0, theta_offset=-30.0),))


def test_depth_is_converted_once_using_the_live_scale() -> None:
    frame = _frame()
    assert frame.depth_mm(0, 0) == pytest.approx(500.0)
    observation = rgbd_observation(
        frame,
        observation_id="o1",
        pixels=[(0, 0), (1, 0), (0, 1)],
        link_ids={(0, 0): "link3"},
    )
    assert [s.depth_m for s in observation.samples] == pytest.approx([0.5, 0.0, 0.25])
    assert observation.samples[0].link_id == "link3"
    assert observation.valid_ratio == pytest.approx(2 / 3)
    assert observation.stream.depth_scale_m == frame.depth_scale_m
    assert observation.metadata["frame_sequence"] == 7


@pytest.mark.parametrize(
    "fields",
    [
        {"depth_scale_m": 0.001},
        {"device_id": "other_camera"},
        {"fx_px": 100.0},
    ],
)
def test_a_supplied_profile_cannot_relabel_the_live_frame(fields: dict[str, object]) -> None:
    profile = replace(stream_profile_of(_frame()), **fields)
    with pytest.raises(CalibrationFailure, match="profile does not match"):
        rgbd_observation(_frame(), observation_id="o1", pixels=[(0, 0)], profile=profile)


@pytest.mark.parametrize("pixel", [(-1, 0), (2, 0), (0, 2), (False, 0)])
def test_pixels_are_validated_before_reading_the_buffer(pixel: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="outside the frame"):
        rgbd_observation(_frame(), observation_id="o1", pixels=[pixel])


def test_source_sequence_cannot_be_overridden_by_caller_metadata() -> None:
    observation = rgbd_observation(
        _frame(),
        observation_id="o1",
        pixels=[],
        metadata={"frame_sequence": 999},
    )
    assert observation.metadata["frame_sequence"] == 7


def test_profile_comes_from_the_live_frame_and_names_pixel_units() -> None:
    frame = _frame()
    profile = stream_profile_of(frame, role=ROLE_WRIST, valid_min_m=0.1, valid_max_m=1.0)
    assert (profile.width, profile.height) == (2, 2)
    assert (profile.fx_px, profile.fy_px) == (2.5, 2.6)
    assert (profile.ppx_px, profile.ppy_px) == (1.0, 1.0)
    assert profile.source == SOURCE_FACTORY
    assert profile.role == ROLE_WRIST
    assert profile.valid_min_m == 0.1
    assert profile.valid_max_m == 1.0
    assert profile == stream_profile(
        device_id=frame.device_id,
        stream="depth_aligned",
        intrinsics=frame.intrinsics,
        depth_scale_m=frame.depth_scale_m,
        role=ROLE_WRIST,
        valid_min_m=0.1,
        valid_max_m=1.0,
    )


def test_default_timestamp_says_receipt_not_exposure_or_settled() -> None:
    observation = rgbd_observation(_frame(), observation_id="o1", pixels=[(0, 0)])
    assert observation.stamp.seconds == 42.0
    assert observation.stamp.domain == CLOCK_NODE_MONOTONIC
    assert observation.stamp.semantics == STAMP_RECEIPT
    stamp = Stamp(seconds=42.1, semantics=STAMP_SETTLED_CAPTURE, uncertainty_s=0.02)
    settled = rgbd_observation(_frame(), observation_id="o2", pixels=[(0, 0)], stamp=stamp)
    assert settled.stamp == stamp


@pytest.mark.parametrize("provenance", [PROVENANCE_PREVIEW_JPEG, PROVENANCE_COLORIZED_DEPTH])
def test_preview_grade_input_is_refused_before_sampling(provenance: str) -> None:
    with pytest.raises(CalibrationFailure) as caught:
        rgbd_observation(_frame(), observation_id="o1", pixels=[(0, 0)], provenance=provenance)
    assert caught.value.code == FAILURE_PROVENANCE_REFUSED
    assert hardware_failure_code(caught.value) == hw.FAILURE_INVALID_ARGS


def test_joint_feedback_stays_raw_and_is_converted_to_radians() -> None:
    model = model_identity(_chain(), model_id="example_arm")
    observation = joint_observation(
        [90.0],
        observation_id="j1",
        model=model,
        stamp=Stamp(seconds=42.0),
        still=False,
        max_speed_deg_s=10.0,
    )
    assert observation.q_raw_rad == pytest.approx((math.pi / 2,))
    assert observation.max_speed_rad_s == pytest.approx(math.radians(10.0))
    assert observation.model.joint_unit == "rad"
    with pytest.raises(ValueError, match="1 joints"):
        joint_observation(
            [0.0, 1.0],
            observation_id="j2",
            model=model,
            stamp=Stamp(seconds=42.0),
        )


def test_chain_digest_is_over_si_values_and_includes_the_mdh_convention() -> None:
    model = model_identity(_chain(), model_id="example_arm")
    assert model.chain_digest == digest_of(
        {
            "convention": "modified",
            "links": [
                {
                    "a_m": 0.1,
                    "alpha_rad": math.pi / 2,
                    "d_m": 0.2,
                    "theta_offset_rad": math.radians(-30.0),
                }
            ],
        }
    )
    changed = model_identity(replace(_chain(), convention="standard"), model_id="example_arm")
    assert changed.digest != model.digest
    assert _chain().links[0].a == 100.0  # no mutation of the command model


def test_all_declared_failures_map_to_planning_time_hardware_failures() -> None:
    assert set(CALIBRATION_FAILURE_MAP) == FAILURE_CODES
    for code in FAILURE_CODES:
        mapped = hardware_failure_code(CalibrationFailure(code, "refused"))
        assert mapped in {
            hw.FAILURE_CHECK_FAILED,
            hw.FAILURE_INVALID_ARGS,
            hw.FAILURE_MODEL_UNAVAILABLE,
            hw.FAILURE_CALIBRATION_STALE,
            hw.FAILURE_CALIBRATION_MISSING,
            hw.FAILURE_SKILL_UNAVAILABLE,
        }
        assert mapped != hw.FAILURE_INTERNAL_DEFECT


def test_unmapped_failure_is_a_local_defect_never_a_hardware_retry() -> None:
    failure = CalibrationFailure("schema", "refused")
    failure.code = "future_failure"
    assert hardware_failure_code(failure) == hw.FAILURE_INTERNAL_DEFECT


def test_hardware_root_does_not_eagerly_import_calibration() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import leapflow.hardware; "
            "assert 'leapflow.hardware.calibration' not in sys.modules; "
            "assert 'leapcalib' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr

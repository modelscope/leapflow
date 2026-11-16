# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapCalib geometry: SE(3) in SI, and the raw/corrected joint spaces kept apart.

Two classes of defect are pinned here because neither is visible in a residual.
A reflection fits data about as well as the truth does and puts the arm on the
mirrored side of the target; a correction applied twice (or zero times) produces
a plausible number in the wrong space. Both are made into exceptions.

``rotation_from_rpy_rad`` is also checked against LeapFlow's own
``rpy_to_matrix``, since the adapter's whole claim is that crossing the boundary
is a unit change and nothing more.
"""

from __future__ import annotations

import math

import pytest

from leapcalib.contracts.failures import FAILURE_UNSUPPORTED_PARAMETER, CalibrationFailure
from leapcalib.contracts.identity import ModelIdentity
from leapcalib.geometry.conventions import (
    SPACE_PHYSICAL,
    SPACE_RAW,
    JointLimits,
    JointOffsets,
    JointVector,
    to_physical,
    to_raw,
)
from leapcalib.geometry.se3 import (
    IDENTITY,
    Transform,
    determinant,
    rotation_angle_rad,
    rotation_from_axis_angle,
    rotation_from_rpy_rad,
    rpy_rad_from_rotation,
)
from leapflow.hardware.skills.geometry import rpy_to_matrix

MODEL = ModelIdentity(model_id="piper_x", dof=6, chain_digest="deadbeef")


def _raw(*values: float) -> JointVector:
    return JointVector(values_rad=values, space=SPACE_RAW, model_digest=MODEL.digest)


# ── Rotations ────────────────────────────────────────────────────────────────


def test_rpy_composition_matches_leapflow_to_the_unit() -> None:
    """The two stacks must agree, or the adapter is silently re-orienting geometry."""
    for angles_deg in ((0.0, 0.0, 0.0), (10.0, -20.0, 30.0), (-90.0, 45.0, 170.0)):
        theirs = rpy_to_matrix(*angles_deg)
        ours = rotation_from_rpy_rad(*(math.radians(a) for a in angles_deg))
        for row_a, row_b in zip(theirs, ours):
            for a, b in zip(row_a, row_b):
                assert a == pytest.approx(b, abs=1e-12)


def test_rpy_round_trips_and_resolves_gimbal_lock_deterministically() -> None:
    angles = (0.3, -0.7, 1.2)
    recovered = rpy_rad_from_rotation(rotation_from_rpy_rad(*angles))
    assert recovered == pytest.approx(angles, abs=1e-9)
    # At pitch = +90 deg the roll and yaw are not separable; the convention
    # assigns the whole rotation to roll rather than returning an arbitrary split.
    locked = rpy_rad_from_rotation(rotation_from_rpy_rad(0.4, math.pi / 2, 0.25))
    assert locked[1] == pytest.approx(math.pi / 2)
    assert locked[2] == 0.0


def test_axis_angle_needs_an_axis_only_when_it_rotates() -> None:
    assert rotation_from_axis_angle((0.0, 0.0, 0.0), 0.0) == IDENTITY
    with pytest.raises(ValueError, match="nonzero axis"):
        rotation_from_axis_angle((0.0, 0.0, 0.0), 0.5)
    turn = rotation_from_axis_angle((0.0, 0.0, 2.0), math.pi / 3)  # unnormalised
    assert rotation_angle_rad(turn) == pytest.approx(math.pi / 3)
    assert determinant(turn) == pytest.approx(1.0)


# ── Transforms ───────────────────────────────────────────────────────────────


def test_a_reflection_is_refused_at_construction() -> None:
    """Determinant -1 is orthonormal, fits data, and mirrors the cell."""
    mirror = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="reflection"):
        Transform(rotation=mirror)


def test_a_non_orthonormal_rotation_is_refused() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        Transform(rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.01)))


def test_frame_labels_stop_a_chain_that_was_never_adjacent() -> None:
    base_flange = Transform.identity(parent="base", child="flange")
    cam_in_base = Transform.identity(parent="base", child="cam_g")
    chained = base_flange.compose(Transform.identity(parent="flange", child="cam_w"))
    assert (chained.parent, chained.child) == ("base", "cam_w")
    with pytest.raises(ValueError, match="cannot chain T_base_flange"):
        base_flange.compose(cam_in_base)


def test_unlabelled_transforms_still_compose() -> None:
    # Labels are a check, not a requirement; a residual builds anonymous
    # transforms and must not be forced to invent frame names.
    assert Transform().compose(Transform()).parent == ""


def test_inverse_is_an_inverse_including_the_translation() -> None:
    pose = Transform.from_rpy_rad(0.2, -0.4, 1.1, (0.3, -0.05, 0.42), parent="base", child="cam")
    back = pose.inverse()
    assert (back.parent, back.child) == ("cam", "base")
    round_trip = pose.compose(back)
    assert (round_trip.parent, round_trip.child) == ("base", "base")
    assert round_trip.distance_m(Transform()) == pytest.approx(0.0, abs=1e-12)
    assert round_trip.angle_to_rad(Transform()) == pytest.approx(0.0, abs=1e-9)
    point = (0.1, 0.2, 0.3)
    assert back.apply(pose.apply(point)) == pytest.approx(point, abs=1e-12)


def test_apply_translates_but_rotate_does_not() -> None:
    pose = Transform(translation_m=(1.0, 2.0, 3.0))
    assert pose.apply((0.0, 0.0, 0.0)) == (1.0, 2.0, 3.0)
    assert pose.rotate((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_distance_and_angle_report_the_two_error_axes_separately() -> None:
    a = Transform()
    b = Transform.from_axis_angle((0.0, 0.0, 1.0), math.radians(1.0), (0.0, 0.0, 0.002))
    assert a.distance_m(b) == pytest.approx(0.002)
    assert math.degrees(a.angle_to_rad(b)) == pytest.approx(1.0)


def test_matrix_form_round_trips_and_rejects_a_non_rigid_bottom_row() -> None:
    pose = Transform.from_rpy_rad(0.1, 0.2, 0.3, (0.01, 0.02, 0.03), parent="a", child="b")
    assert Transform.from_matrix(pose.to_matrix(), parent="a", child="b") == pose
    assert Transform.from_mapping(pose.to_dict()) == pose
    rows = pose.to_matrix()
    rows[3] = [0.0, 0.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="bottom row"):
        Transform.from_matrix(rows)


def test_a_transform_cannot_hold_a_nonfinite_number() -> None:
    with pytest.raises(ValueError, match="finite"):
        Transform(translation_m=(float("nan"), 0.0, 0.0))


# ── Raw and corrected joint space ────────────────────────────────────────────


def test_offsets_apply_once_in_each_direction() -> None:
    offsets = JointOffsets(
        values_rad=(0.01, 0.0, -0.02, 0.0, 0.0, 0.0),
        model=MODEL,
        supported_joints=(0, 2),
        method_id="stage_g",
    )
    raw = _raw(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    physical = to_physical(raw, offsets)
    assert physical.space == SPACE_PHYSICAL
    assert physical.values_rad[0] == pytest.approx(0.11)
    assert physical.values_rad[2] == pytest.approx(0.28)
    assert to_raw(physical, offsets).values_rad == pytest.approx(raw.values_rad)


def test_applying_a_correction_twice_raises_instead_of_computing() -> None:
    offsets = JointOffsets.zero(MODEL)
    physical = to_physical(_raw(*([0.0] * 6)), offsets)
    with pytest.raises(CalibrationFailure) as caught:
        to_physical(physical, offsets)  # the double-apply bug
    assert caught.value.code == FAILURE_UNSUPPORTED_PARAMETER
    with pytest.raises(CalibrationFailure, match="expected a physical vector"):
        to_raw(_raw(*([0.0] * 6)), offsets)


def test_offsets_from_another_model_are_refused() -> None:
    other = ModelIdentity(model_id="other_arm", dof=6, chain_digest="cafe")
    vector = JointVector(values_rad=(0.0,) * 6, space=SPACE_RAW, model_digest=other.digest)
    with pytest.raises(CalibrationFailure, match="different kinematic models"):
        to_physical(vector, JointOffsets.zero(MODEL))


def test_an_unsupported_joint_cannot_carry_a_correction() -> None:
    """ "Not established" and "zero" must not become the same statement."""
    with pytest.raises(ValueError, match="not supported"):
        JointOffsets(values_rad=(0.0, 0.03, 0.0, 0.0, 0.0, 0.0), model=MODEL)
    zero = JointOffsets.zero(MODEL)
    assert zero.is_zero
    assert zero.digest


def test_activation_needs_evidence_and_a_qualified_command_path() -> None:
    supported = JointOffsets(
        values_rad=(0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
        model=MODEL,
        supported_joints=(0,),
        method_id="stage_g",
    )
    # P0 has no qualified dual-model command path, so this is the only truthful call.
    with pytest.raises(CalibrationFailure, match="not qualified"):
        supported.require_activatable(command_path_qualified=False)
    supported.require_activatable(command_path_qualified=True)


# ── Limits ───────────────────────────────────────────────────────────────────


def test_limits_intersect_to_the_narrowest_and_surface_a_contradiction() -> None:
    declared = JointLimits.from_degrees([(-180.0, 180.0), (-90.0, 90.0)], source="declared")
    firmware = JointLimits.from_degrees([(-150.0, 170.0), (-95.0, 60.0)], source="firmware")
    merged = declared.intersect(firmware)
    assert merged.source == "declared+firmware"
    assert merged.bounds_rad[0] == pytest.approx((math.radians(-150.0), math.radians(170.0)))
    assert merged.bounds_rad[1] == pytest.approx((math.radians(-90.0), math.radians(60.0)))
    disjoint = JointLimits.from_degrees([(-180.0, 180.0), (120.0, 130.0)])
    with pytest.raises(ValueError, match="joint 1 limits do not overlap"):
        declared.intersect(disjoint)


def test_an_uncertainty_margin_narrows_the_envelope_it_does_not_widen_it() -> None:
    limits = JointLimits.from_degrees([(-10.0, 10.0)])
    inside = JointVector(values_rad=(math.radians(9.0),), space=SPACE_RAW)
    assert limits.violation(inside) == ""
    # A target that only fits if the model is exactly right does not fit.
    assert "outside" in limits.violation(inside, margin_rad=math.radians(2.0))
    with pytest.raises(ValueError, match="nonnegative"):
        limits.violation(inside, margin_rad=-0.1)


def test_limits_refuse_to_judge_a_vector_from_the_other_space() -> None:
    """Raw bounds are the firmware's; a corrected target converts back, not around."""
    limits = JointLimits.from_degrees([(-10.0, 10.0)])
    corrected = JointVector(values_rad=(0.0,), space=SPACE_PHYSICAL)
    with pytest.raises(CalibrationFailure, match="limits are in raw space"):
        limits.violation(corrected)


def test_limits_round_trip_through_plain_data() -> None:
    limits = JointLimits.from_degrees([(-10.0, 10.0), (0.0, 5.0)], source="manual")
    assert JointLimits.from_mapping(limits.to_dict()) == limits
    with pytest.raises(ValueError, match="increasing"):
        JointLimits(bounds_rad=((1.0, -1.0),))

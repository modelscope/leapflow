# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapCalib contracts: units, identity, observation admission, support honesty.

These tests exist for the claims the package is allowed to make. The interesting
cases are all refusals -- preview data, unrelated clocks, an estimate that
reports a prior as a measurement -- because those are the ways a calibration
stops being trustworthy while still returning numbers.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

import leapcalib
from leapcalib.contracts.estimates import (
    EstimateBundle,
    ParameterEstimate,
    retained_estimate,
)
from leapcalib.contracts.failures import (
    FAILURE_CLOCK_DOMAIN_MISMATCH,
    FAILURE_CODES,
    FAILURE_PROVENANCE_REFUSED,
    CalibrationFailure,
)
from leapcalib.contracts.identity import (
    ROLE_WRIST,
    DeviceIdentity,
    ModelIdentity,
    MountIdentity,
    StreamProfile,
    ToolIdentity,
    TopologyIdentity,
    digest_of,
)
from leapcalib.contracts.observations import (
    CLOCK_HOST_MONOTONIC,
    CLOCK_NODE_MONOTONIC,
    PROVENANCE_COLORIZED_DEPTH,
    PROVENANCE_METRIC_RGBD,
    PROVENANCE_PREVIEW_JPEG,
    STAMP_RECEIPT,
    STAMP_SETTLED_CAPTURE,
    JointObservation,
    ObservationSource,
    PixelDepthSample,
    RgbdObservation,
    Stamp,
    observation_from_mapping,
    pair_spread_s,
    settled_fraction,
    total_valid_ratio,
)
from leapcalib.contracts.parameters import (
    COMPOSE_ADDITIVE,
    SUPPORT_INSUFFICIENT,
    SUPPORT_PARTIAL,
    SUPPORT_RETAINED,
    SUPPORT_SUPPORTED,
    camera_extrinsic_block,
    joint_zero_block,
    ordered_for_solve,
)
from leapcalib.contracts.units import (
    Quantity,
    canonical_number,
    deg_to_rad,
    mm_to_m,
    to_si,
    unit_of_field,
)

LEAPCALIB_ROOT = Path(leapcalib.__file__).parent

# Fields that are legitimately dimensionless: ratios, counts, identifiers and
# booleans. Anything numeric outside this list has to name its unit in its name.
DIMENSIONLESS_FIELDS = frozenset(
    {
        "weight",
        "dof",
        "width",
        "height",
        "value",
        "fits",
        "dimension",
        "accepted_count",
        "sequence",
        "frame_sequence",
    }
)


def _model(dof: int = 6) -> ModelIdentity:
    return ModelIdentity(model_id="piper_x", dof=dof, chain_digest="deadbeef")


def _profile(**overrides: object) -> StreamProfile:
    fields: dict[str, object] = {
        "device_id": "d435if",
        "stream": "depth_aligned",
        "width": 640,
        "height": 480,
        "fx_px": 600.0,
        "fy_px": 600.0,
        "ppx_px": 320.0,
        "ppy_px": 240.0,
        "depth_scale_m": 0.001,
        "valid_min_m": 0.2,
        "valid_max_m": 3.0,
    }
    fields.update(overrides)
    return StreamProfile(**fields)  # type: ignore[arg-type]


def _topology() -> TopologyIdentity:
    return TopologyIdentity(
        workcell_id="cell_a",
        device=DeviceIdentity(device_id="robot_piper_arm"),
        model=_model(),
        streams=(_profile(),),
        mounts=(MountIdentity(mount_id="wrist_bracket", parent_frame="flange"),),
        tool=ToolIdentity(tool_id="parallel_jaw", jaw_stroke_m=0.085),
    )


# ── Units ────────────────────────────────────────────────────────────────────


def test_conversions_match_the_declaration_units() -> None:
    assert mm_to_m(123.0) == pytest.approx(0.123)
    assert deg_to_rad(180.0) == pytest.approx(math.pi)
    assert to_si(2.0, "mm") == (0.002, "m")


def test_pixels_have_no_si_conversion() -> None:
    # A pixel becomes a length only through intrinsics, so there is no factor to
    # apply and pretending otherwise is how a pixel error becomes "millimetres".
    with pytest.raises(ValueError, match="no declared SI conversion"):
        to_si(3.0, "px")
    with pytest.raises(ValueError):
        Quantity(3.0, "px").si


def test_canonical_number_refuses_nonfinite_and_normalises_zero() -> None:
    assert math.copysign(1.0, canonical_number(-0.0)) > 0
    for bad in (float("nan"), float("inf"), True, "1.0"):
        with pytest.raises(ValueError):
            canonical_number(bad)  # type: ignore[arg-type]


def test_a_unit_word_names_its_unit_but_only_as_a_whole_name() -> None:
    # "seconds" is self-naming; "milliseconds" ends with it and must not inherit
    # the reading, because that mistake is silently a factor of a thousand.
    assert unit_of_field("seconds") == "s"
    assert unit_of_field("uncertainty_s") == "s"
    assert unit_of_field("milliseconds") is None
    assert unit_of_field("accepted_count") is None


def test_every_numeric_schema_field_names_its_unit() -> None:
    """No numeric wire field may leave its unit to be inferred from magnitude.

    Walks the ``to_dict`` payloads of the public value types rather than trusting
    review: 0.3 is a plausible value in both millimetres and metres, so an
    unnamed numeric field is a defect waiting for the first integrator.
    """
    payloads = [
        _profile().to_dict(),
        _topology().to_dict(),
        Stamp(seconds=1.0, uncertainty_s=0.01).to_dict(),
        PixelDepthSample(u_px=1.0, v_px=2.0, depth_m=0.5).to_dict(),
        ToolIdentity(tool_id="t").to_dict(),
    ]
    unnamed: list[str] = []

    def walk(payload: object) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in DIMENSIONLESS_FIELDS and unit_of_field(key) is None:
                        unnamed.append(key)
                walk(value)
        elif isinstance(payload, list):
            for item in payload:
                walk(item)

    for payload in payloads:
        walk(payload)
    assert not unnamed, f"numeric fields without a unit in their name: {sorted(set(unnamed))}"


# ── Identity ─────────────────────────────────────────────────────────────────


def test_digest_is_stable_and_order_independent() -> None:
    assert digest_of({"a": 1, "b": 2}) == digest_of({"b": 2, "a": 1})
    # Signed zero must not give one geometry two identities.
    assert digest_of({"x_m": 0.0}) == digest_of({"x_m": -0.0})


def test_topology_names_what_changed_instead_of_just_differing() -> None:
    before = _topology()
    after = TopologyIdentity(
        workcell_id=before.workcell_id,
        device=before.device,
        model=before.model,
        streams=(_profile(role=ROLE_WRIST),),
        mounts=before.mounts,
        tool=before.tool,
    )
    assert before.digest != after.digest
    # A reseated stream profile must not read as a whole-cell change: the
    # response to "the camera role changed" is not "recalibrate everything".
    assert before.differs_from(after) == ("stream:d435if:depth_aligned",)
    assert before.differs_from(before) == ()


def test_a_model_identity_is_radians_only() -> None:
    with pytest.raises(ValueError, match="radians"):
        ModelIdentity(model_id="piper_x", dof=6, chain_digest="abc", joint_unit="deg")


# ── Observations ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("provenance", [PROVENANCE_PREVIEW_JPEG, PROVENANCE_COLORIZED_DEPTH])
def test_preview_grade_data_cannot_become_an_observation(provenance: str) -> None:
    with pytest.raises(CalibrationFailure) as caught:
        RgbdObservation(
            observation_id="o1",
            stream=_profile(),
            stamp=Stamp(seconds=1.0),
            provenance=provenance,
        )
    assert caught.value.code == FAILURE_PROVENANCE_REFUSED


def test_unrelated_clocks_refuse_to_be_differenced() -> None:
    node = Stamp(seconds=100.0, domain=CLOCK_NODE_MONOTONIC)
    host = Stamp(seconds=5000.0, domain=CLOCK_HOST_MONOTONIC)
    assert not node.comparable_with(host)
    with pytest.raises(CalibrationFailure) as caught:
        node.delta_s(host)
    assert caught.value.code == FAILURE_CLOCK_DOMAIN_MISMATCH


def test_stamp_spread_includes_both_uncertainties() -> None:
    a = Stamp(seconds=10.0, uncertainty_s=0.01)
    b = Stamp(seconds=10.02, uncertainty_s=0.005)
    # 20 ms apart, but no claim better than 35 ms can be made from the pair.
    assert a.spread_s(b) == pytest.approx(0.035)


def test_samples_outside_the_image_are_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        RgbdObservation(
            observation_id="o1",
            stream=_profile(),
            stamp=Stamp(seconds=1.0),
            provenance=PROVENANCE_METRIC_RGBD,
            samples=(PixelDepthSample(u_px=640.0, v_px=10.0, depth_m=1.0),),
        )


def test_valid_ratio_counts_the_range_not_just_the_zeros() -> None:
    observation = RgbdObservation(
        observation_id="o1",
        stream=_profile(),
        stamp=Stamp(seconds=1.0),
        provenance=PROVENANCE_METRIC_RGBD,
        samples=(
            PixelDepthSample(u_px=1.0, v_px=1.0, depth_m=0.0),  # no measurement
            PixelDepthSample(u_px=2.0, v_px=1.0, depth_m=0.1),  # below valid_min
            PixelDepthSample(u_px=3.0, v_px=1.0, depth_m=5.0),  # beyond valid_max
            PixelDepthSample(u_px=4.0, v_px=1.0, depth_m=0.8),
        ),
    )
    assert observation.valid_ratio == pytest.approx(0.25)
    assert total_valid_ratio([observation]) == pytest.approx(0.25)


def test_joint_observation_pins_dof_and_records_measured_stillness() -> None:
    model = _model()
    settled = JointObservation(
        observation_id="j1",
        model=model,
        stamp=Stamp(seconds=1.0, semantics=STAMP_SETTLED_CAPTURE),
        q_raw_rad=(0.0,) * 6,
        still=True,
    )
    moving = JointObservation(
        observation_id="j2",
        model=model,
        stamp=Stamp(seconds=2.0, semantics=STAMP_RECEIPT),
        q_raw_rad=(0.1,) * 6,
        max_speed_rad_s=0.4,
    )
    assert settled_fraction([settled, moving]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="6 joints"):
        JointObservation(
            observation_id="j3", model=model, stamp=Stamp(seconds=1.0), q_raw_rad=(0.0,)
        )


def test_observations_round_trip_through_plain_data() -> None:
    camera = RgbdObservation(
        observation_id="o1",
        stream=_profile(),
        stamp=Stamp(seconds=1.0, uncertainty_s=0.02),
        provenance=PROVENANCE_METRIC_RGBD,
        samples=(PixelDepthSample(u_px=10.0, v_px=20.0, depth_m=0.9, link_id="link3"),),
    )
    joints = JointObservation(
        observation_id="j1", model=_model(), stamp=Stamp(seconds=1.0), q_raw_rad=(0.0,) * 6
    )
    assert observation_from_mapping(camera.to_dict()) == camera
    assert observation_from_mapping(joints.to_dict()) == joints
    assert pair_spread_s(camera, joints) == pytest.approx(0.02)


# ── Parameter blocks ─────────────────────────────────────────────────────────


def test_a_transform_block_must_name_both_frames() -> None:
    block = camera_extrinsic_block("fixed_cam", parent_frame="base", child_frame="cam_g")
    assert block.transform_name == "T_base_cam_g"
    with pytest.raises(ValueError, match="both frames"):
        camera_extrinsic_block("fixed_cam", parent_frame="base", child_frame="")


def test_joint_zero_block_is_additive_radians_per_joint() -> None:
    block = joint_zero_block("zeros", dof=6)
    assert block.dimension == 6
    assert block.composition == COMPOSE_ADDITIVE
    assert set(block.units) == {"rad"}


def test_stage_order_follows_declared_prerequisites() -> None:
    """Freezing stage 1 before stage 2 is the method, so the order is data."""
    fixed = camera_extrinsic_block("fixed_cam", parent_frame="base", child_frame="cam_g")
    zeros = joint_zero_block("zeros", dof=6, prerequisites=("fixed_cam",))
    wrist = camera_extrinsic_block(
        "wrist_cam", parent_frame="flange", child_frame="cam_w", prerequisites=("zeros",)
    )
    ordered = ordered_for_solve((wrist, zeros, fixed))
    assert [b.block_id for b in ordered] == ["fixed_cam", "zeros", "wrist_cam"]


def test_impossible_stage_orders_are_errors() -> None:
    a = joint_zero_block("a", dof=2, prerequisites=("b",))
    b = joint_zero_block("b", dof=2, prerequisites=("a",))
    with pytest.raises(ValueError, match="circular"):
        ordered_for_solve((a, b))
    with pytest.raises(ValueError, match="absent prerequisites"):
        ordered_for_solve((joint_zero_block("a", dof=2, prerequisites=("missing",)),))


# ── Estimates ────────────────────────────────────────────────────────────────


def test_a_retained_estimate_is_the_prior_and_says_so() -> None:
    block = joint_zero_block("zeros", dof=3)
    prior = (0.0, 0.0, 0.0)
    estimate = retained_estimate(
        block, prior, (0.01, 0.01, 0.01), reason="single-arm gauge leaves joint 0 free"
    )
    assert estimate.support == SUPPORT_RETAINED
    assert estimate.values == prior
    assert estimate.retained_components == (0, 1, 2)
    assert estimate.limitations


def test_an_unsupported_component_cannot_drift_from_its_prior() -> None:
    block = joint_zero_block("zeros", dof=2)
    with pytest.raises(ValueError, match="differs from its prior"):
        ParameterEstimate(
            block=block,
            values=(0.01, 0.02),
            prior=(0.0, 0.0),
            support=SUPPORT_PARTIAL,
            uncertainty=(0.001, 0.001),
            prior_uncertainty=(0.01, 0.01),
            supported_components=(0,),
            method_id="stage_g",
            accepted_count=12,
        )


def test_an_unsupported_component_cannot_claim_a_tighter_uncertainty() -> None:
    """A prior that reports a measurement's confidence is read as a measurement."""
    block = joint_zero_block("zeros", dof=2)
    with pytest.raises(ValueError, match="tighter uncertainty"):
        ParameterEstimate(
            block=block,
            values=(0.01, 0.0),
            prior=(0.0, 0.0),
            support=SUPPORT_PARTIAL,
            uncertainty=(0.001, 0.0001),
            prior_uncertainty=(0.01, 0.01),
            supported_components=(0,),
            method_id="stage_g",
            accepted_count=12,
        )


def test_a_supported_estimate_must_point_at_its_evidence() -> None:
    block = joint_zero_block("zeros", dof=1)
    with pytest.raises(ValueError, match="name the method"):
        ParameterEstimate(
            block=block,
            values=(0.01,),
            prior=(0.0,),
            support=SUPPORT_SUPPORTED,
            uncertainty=(0.001,),
            prior_uncertainty=(0.01,),
            supported_components=(0,),
            accepted_count=5,
        )
    with pytest.raises(ValueError, match="accepted observations"):
        ParameterEstimate(
            block=block,
            values=(0.01,),
            prior=(0.0,),
            support=SUPPORT_SUPPORTED,
            uncertainty=(0.001,),
            prior_uncertainty=(0.01,),
            supported_components=(0,),
            method_id="stage_g",
        )


def test_a_partial_estimate_is_a_proper_subset() -> None:
    block = joint_zero_block("zeros", dof=2)
    with pytest.raises(ValueError, match="proper subset"):
        ParameterEstimate(
            block=block,
            values=(0.01, 0.02),
            prior=(0.0, 0.0),
            support=SUPPORT_PARTIAL,
            uncertainty=(0.001, 0.001),
            prior_uncertainty=(0.01, 0.01),
            supported_components=(0, 1),
            method_id="stage_g",
            accepted_count=3,
        )


def test_estimate_bundle_keeps_the_evidence_split_disjoint() -> None:
    block = joint_zero_block("zeros", dof=1)
    estimate = retained_estimate(block, (0.0,), (0.01,), reason="not observable")
    bundle = EstimateBundle(
        bundle_id="b1",
        topology=_topology(),
        estimates=(estimate,),
        accepted_ids=("o1",),
        holdout_ids=("o2",),
    )
    assert bundle.retained_blocks == ("zeros",)
    assert bundle.supported_blocks == ()
    assert bundle.digest
    with pytest.raises(ValueError, match="disjoint"):
        EstimateBundle(
            bundle_id="b2",
            topology=_topology(),
            estimates=(estimate,),
            accepted_ids=("o1",),
            holdout_ids=("o1",),
        )


def test_insufficient_evidence_is_a_reportable_result() -> None:
    block = camera_extrinsic_block("cam", parent_frame="base", child_frame="cam_g")
    estimate = retained_estimate(
        block,
        (0.0,) * 6,
        (0.05,) * 6,
        reason="plane-only scene; in-plane translation and yaw are unobservable",
        support=SUPPORT_INSUFFICIENT,
    )
    # The block is still reported. Omitting it would read as "nothing to say".
    assert estimate.support == SUPPORT_INSUFFICIENT
    assert estimate.values == estimate.prior


# ── Package boundary ─────────────────────────────────────────────────────────


def test_leapcalib_never_imports_leapflow() -> None:
    """The dependency arrow points one way, and only a test can hold it there.

    An accidental ``leapflow`` import would make the package unusable standalone
    and would be invisible in a repository where both always resolve.
    """
    forbidden = ("leapflow", "scipy", "cv2", "torch", "rclpy", "pyrealsense2")
    offenders: list[str] = []
    for path in sorted(LEAPCALIB_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in forbidden or (
                    name.split(".")[0] == "numpy" and path.parent.name != "estimation"
                ):
                    offenders.append(f"{path.name}: {name}")
    assert not offenders, f"leapcalib must stay standalone: {offenders}"


def test_candidate_methods_exist_without_a_service_package() -> None:
    """Implemented solvers are offline; no service or actuator authority is shipped."""
    assert (LEAPCALIB_ROOT / "estimation" / "handeye.py").exists()
    assert (LEAPCALIB_ROOT / "estimation" / "response.py").exists()
    assert not (LEAPCALIB_ROOT / "runtime").exists()


def test_failure_codes_are_closed() -> None:
    with pytest.raises(ValueError, match="unknown calibration failure code"):
        CalibrationFailure("not_a_real_code", "boom")
    assert "insufficient_evidence" in FAILURE_CODES


def test_replay_style_sources_satisfy_the_acquisition_protocol() -> None:
    class Fake:
        topology_digest = "abc"

        def observations(self):  # noqa: ANN202 - protocol shape under test
            return iter(())

    assert isinstance(Fake(), ObservationSource)

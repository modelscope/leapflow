# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapCalib validation: asset honesty, error budgets, evidence replay, and the CLI.

The two numbers the design is sized around are pinned literally -- 20 mm/s at
100 ms is about 2 mm, 1 degree at 130 mm is about 2.3 mm -- because they are what
makes "pixel agreement cannot certify a millimetre grasp" a measurement rather
than an opinion.

The asset tests exist for one specific shortcut: approximating links with boxes,
running point-to-plane alignment, and reporting the small residual as a
calibration. That produces a bias that looks like a result, so the refusal has to
hold even though the solve would succeed.
"""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path

import pytest

from leapcalib.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, main
from leapcalib.contracts.failures import (
    FAILURE_BUDGET_EXCEEDED,
    FAILURE_EVIDENCE_CORRUPT,
    FAILURE_GEOMETRY_ASSET_MISSING,
    FAILURE_SCHEMA,
    CalibrationFailure,
)
from leapcalib.contracts.identity import (
    DeviceIdentity,
    ModelIdentity,
    MountIdentity,
    StreamProfile,
    ToolIdentity,
    TopologyIdentity,
    digest_of,
)
from leapcalib.contracts.observations import (
    PROVENANCE_METRIC_RGBD,
    STAMP_SETTLED_CAPTURE,
    JointObservation,
    PixelDepthSample,
    RgbdObservation,
    Stamp,
)
from leapcalib.validation.assets import (
    ASSET_DH_TABLE,
    ASSET_POINT_LANDMARKS,
    ASSET_PRIMITIVE_HULL,
    ASSET_SURFACE_MESH,
    METHOD_CAD_ICP,
    METHOD_LANDMARK_PNP,
    METHOD_SCENE_MOTION,
    METHOD_TASK_LOCAL,
    GeometryAsset,
    availability,
)
from leapcalib.validation.budget import (
    COMBINE_INDEPENDENT,
    COMBINE_WORST_CASE,
    TERM_RANDOM,
    TERM_SYSTEMATIC,
    ErrorBudget,
    ErrorTerm,
    TaskTolerance,
    delay_error_m,
    lever_arm_error_m,
    observation_delay_budget,
)
from leapcalib.validation.bundle import (
    SCHEMA_VERSION,
    EvidenceBundle,
    ReplaySource,
    Retention,
    read_bundle,
    write_bundle,
)

MODEL = ModelIdentity(model_id="piper_x", dof=6, chain_digest="deadbeef")
TOLERANCE = TaskTolerance(
    task_id="tabletop_grasp",
    allowed_m=0.005,
    basis="jaw 85 mm - object 60 mm, halved for two-sided clearance",
)


def _stream() -> StreamProfile:
    return StreamProfile(
        device_id="d435if",
        stream="depth_aligned",
        width=640,
        height=480,
        fx_px=600.0,
        fy_px=600.0,
        ppx_px=320.0,
        ppy_px=240.0,
        depth_scale_m=0.001,
        valid_min_m=0.2,
        valid_max_m=3.0,
    )


def _topology() -> TopologyIdentity:
    return TopologyIdentity(
        workcell_id="cell_a",
        device=DeviceIdentity(device_id="robot_piper_arm"),
        model=MODEL,
        streams=(_stream(),),
        mounts=(MountIdentity(mount_id="wrist_bracket", parent_frame="flange"),),
        tool=ToolIdentity(tool_id="parallel_jaw", jaw_stroke_m=0.085),
    )


def _bundle(**overrides: object) -> EvidenceBundle:
    camera = RgbdObservation(
        observation_id="o1",
        stream=_stream(),
        stamp=Stamp(seconds=1.0, uncertainty_s=0.004, semantics=STAMP_SETTLED_CAPTURE),
        provenance=PROVENANCE_METRIC_RGBD,
        samples=(
            PixelDepthSample(u_px=100.0, v_px=120.0, depth_m=0.65, link_id="link3"),
            PixelDepthSample(u_px=101.0, v_px=120.0, depth_m=0.0),  # no measurement
        ),
    )
    holdout = RgbdObservation(
        observation_id="o2",
        stream=_stream(),
        stamp=Stamp(seconds=2.0, semantics=STAMP_SETTLED_CAPTURE),
        provenance=PROVENANCE_METRIC_RGBD,
        samples=(PixelDepthSample(u_px=200.0, v_px=220.0, depth_m=0.70),),
    )
    joints = JointObservation(
        observation_id="j1",
        model=MODEL,
        stamp=Stamp(seconds=1.0, semantics=STAMP_SETTLED_CAPTURE),
        q_raw_rad=(0.0,) * 6,
        still=True,
    )
    fields: dict[str, object] = {
        "bundle_id": "b1",
        "topology": _topology(),
        "observations": (camera, holdout, joints),
        "created_at_s": 1700000000.0,
        "retention": Retention(ttl_s=86400.0),
        "accepted_ids": ("o1", "j1"),
        "holdout_ids": ("o2",),
        "limitations": ("fixed camera sees no base surface; yaw gauge is unresolved",),
    }
    fields.update(overrides)
    return EvidenceBundle(**fields)  # type: ignore[arg-type]


# ── Geometry assets ──────────────────────────────────────────────────────────


def test_a_dh_table_can_never_satisfy_a_surface_requirement() -> None:
    """A DH table places frames; it says nothing about the shape of the metal."""
    declared = [
        GeometryAsset(link_id="link3", kind=ASSET_DH_TABLE, digest="d", verified=True),
        GeometryAsset(link_id="link3", kind=ASSET_PRIMITIVE_HULL, digest="h", verified=True),
    ]
    report = availability(declared, required_links=["link3"])
    assert not report.permits(METHOD_CAD_ICP)
    assert report.missing_surface_links == ("link3",)
    # Refusing ICP must not refuse everything: these need no robot surface model.
    assert report.permits(METHOD_SCENE_MOTION)
    assert report.permits(METHOD_TASK_LOCAL)
    with pytest.raises(CalibrationFailure) as caught:
        report.require(METHOD_CAD_ICP)
    assert caught.value.code == FAILURE_GEOMETRY_ASSET_MISSING


def test_an_unverified_mesh_is_a_prior_not_a_reference() -> None:
    """A vendor mesh may be the wrong revision, the wrong gripper or the wrong units."""
    unverified = [GeometryAsset(link_id="link3", kind=ASSET_SURFACE_MESH, digest="m")]
    report = availability(unverified, required_links=["link3"])
    assert not report.permits(METHOD_CAD_ICP)
    assert "unverified" in report.refusals[METHOD_CAD_ICP]
    verified = [GeometryAsset(link_id="link3", kind=ASSET_SURFACE_MESH, digest="m", verified=True)]
    assert availability(verified, required_links=["link3"]).permits(METHOD_CAD_ICP)


def test_partial_link_coverage_is_refused_with_the_missing_links_named() -> None:
    # A solve over a subset of links is a different method with a different
    # observability story, so it has to be asked for rather than fallen into.
    assets = [GeometryAsset(link_id="link3", kind=ASSET_SURFACE_MESH, digest="m", verified=True)]
    report = availability(assets, required_links=["link3", "link5"])
    assert not report.permits(METHOD_CAD_ICP)
    assert "link5" in report.refusals[METHOD_CAD_ICP]
    assert report.surface_links == ("link3",)
    assert report.missing_surface_links == ("link5",)


def test_landmarks_permit_pnp_without_permitting_icp() -> None:
    assets = [GeometryAsset(link_id="base", kind=ASSET_POINT_LANDMARKS, digest="p", verified=True)]
    report = availability(assets, required_links=["base"])
    assert report.permits(METHOD_LANDMARK_PNP)
    assert not report.permits(METHOD_CAD_ICP)


def test_asset_declarations_are_validated_and_round_trip() -> None:
    asset = GeometryAsset(link_id="link3", kind=ASSET_SURFACE_MESH, digest="m", verified=True)
    assert GeometryAsset.from_mapping(asset.to_dict()) == asset
    assert asset.describes_surface
    with pytest.raises(ValueError, match="declared in metres"):
        GeometryAsset(link_id="l", kind=ASSET_SURFACE_MESH, digest="m", units="mm")
    with pytest.raises(ValueError, match="explicit boolean"):
        GeometryAsset(
            link_id="l",
            kind=ASSET_SURFACE_MESH,
            digest="m",
            verified=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unknown calibration methods"):
        availability([], required_links=["l"], methods=["wishful_thinking"])


def test_a_method_needing_geometry_is_refused_when_no_links_are_named() -> None:
    report = availability([], required_links=[])
    assert "needs named links" in report.refusals[METHOD_CAD_ICP]


# ── Error budget ─────────────────────────────────────────────────────────────


def test_delay_error_reproduces_the_design_figure() -> None:
    """20 mm/s acted on 100 ms late is 2 mm, before any camera or model error."""
    error = delay_error_m(speed_m_s=0.020, rotation_rate_rad_s=0.0, lever_arm_m=0.0, latency_s=0.1)
    assert error == pytest.approx(0.002)


def test_rotation_contributes_through_its_lever_arm() -> None:
    error = delay_error_m(
        speed_m_s=0.0,
        rotation_rate_rad_s=math.radians(10.0),
        lever_arm_m=0.13,
        latency_s=0.1,
    )
    assert error == pytest.approx(0.13 * math.radians(10.0) * 0.1)
    with pytest.raises(ValueError, match="nonnegative magnitudes"):
        delay_error_m(speed_m_s=-0.01, rotation_rate_rad_s=0.0, lever_arm_m=0.0, latency_s=0.1)


def test_lever_arm_error_reproduces_the_design_figure_and_errs_long() -> None:
    """1 degree at 130 mm is about 2.3 mm: why pixel agreement certifies nothing."""
    arc = lever_arm_error_m(angle_error_rad=math.radians(1.0), lever_arm_m=0.13)
    assert arc == pytest.approx(0.00226893, abs=1e-8)
    chord = 2 * 0.13 * math.sin(math.radians(1.0) / 2)
    # The arc is the larger of the two, so a marginal task is refused rather than
    # accepted on a rounding.
    assert arc >= chord
    # Sign of the angular error cannot reduce the positional consequence.
    assert lever_arm_error_m(angle_error_rad=-math.radians(1.0), lever_arm_m=0.13) == arc


def test_delay_and_orientation_leave_little_room_for_sensor_and_model_error() -> None:
    budget = observation_delay_budget(
        speed_m_s=0.020,
        rotation_rate_rad_s=0.0,
        lever_arm_m=0.13,
        latency_s=0.1,
        orientation_error_rad=math.radians(1.0),
    )
    verdict = budget.check(TOLERANCE)
    assert verdict.total_m == pytest.approx(0.004269, abs=1e-6)
    assert verdict.fits
    assert verdict.dominant_term == "orientation_lever_arm"
    assert verdict.headroom_m == pytest.approx(0.000731, abs=1e-6)
    # Adding a camera and a depth term to the same budget closes the margin, which
    # is the point of writing it down before collecting data.
    with_sensor = budget.with_term(
        ErrorTerm(name="depth_bias", value_m=0.001, basis="D405 bias at 0.3 m")
    ).with_term(ErrorTerm(name="extrinsic", value_m=0.002, basis="prior mount accuracy"))
    assert not with_sensor.check(TOLERANCE).fits


def test_worst_case_is_the_default_and_independence_must_be_claimed() -> None:
    terms = (
        ErrorTerm(name="a", value_m=0.003, kind=TERM_RANDOM),
        ErrorTerm(name="b", value_m=0.004, kind=TERM_RANDOM),
        ErrorTerm(name="c", value_m=0.001, kind=TERM_SYSTEMATIC),
    )
    assert ErrorBudget(terms=terms).combination == COMBINE_WORST_CASE
    assert ErrorBudget(terms=terms).total_m == pytest.approx(0.008)
    quadrature = ErrorBudget(terms=terms, combination=COMBINE_INDEPENDENT)
    # Random terms in quadrature, systematic still linear: 0.005 + 0.001.
    assert quadrature.total_m == pytest.approx(0.006)


def test_a_budget_that_does_not_close_refuses_the_task() -> None:
    budget = ErrorBudget(terms=(ErrorTerm(name="delay", value_m=0.01),))
    assert not budget.check(TOLERANCE).fits
    with pytest.raises(CalibrationFailure) as caught:
        budget.require_fit(TOLERANCE)
    assert caught.value.code == FAILURE_BUDGET_EXCEEDED
    # The message is read by whoever has to fix it, so it names the figures in mm
    # and the term to attack.
    assert "5.00 mm" in str(caught.value)
    assert "delay" in str(caught.value)


def test_a_tolerance_must_state_where_it_came_from() -> None:
    with pytest.raises(ValueError, match="physical margin"):
        TaskTolerance(task_id="t", allowed_m=0.005)
    with pytest.raises(ValueError, match="must be positive"):
        TaskTolerance(task_id="t", allowed_m=0.0, basis="jaw margin")


def test_budget_terms_are_uniquely_named_and_round_trip() -> None:
    budget = ErrorBudget(terms=(ErrorTerm(name="a", value_m=0.001),))
    assert ErrorBudget.from_mapping(budget.to_dict()) == budget
    with pytest.raises(ValueError, match="uniquely named"):
        ErrorBudget(terms=(ErrorTerm(name="a", value_m=0.001), ErrorTerm(name="a", value_m=0.002)))
    with pytest.raises(ValueError, match="cannot be negative"):
        ErrorTerm(name="a", value_m=-0.001)
    assert ErrorBudget(terms=()).dominant_term == ""


# ── Evidence bundles ─────────────────────────────────────────────────────────


def test_a_bundle_round_trips_through_a_file_with_its_digest(tmp_path: Path) -> None:
    bundle = _bundle()
    path = write_bundle(tmp_path / "b1.json", bundle)
    restored = read_bundle(path)
    assert restored == bundle
    assert restored.digest == bundle.digest
    assert restored.topology.digest == bundle.topology.digest


def test_an_edited_bundle_is_refused_rather_than_solved_against(tmp_path: Path) -> None:
    path = write_bundle(tmp_path / "b1.json", _bundle())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content"]["observations"][0]["samples"][0]["depth_m"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationFailure) as caught:
        read_bundle(path)
    assert caught.value.code == FAILURE_EVIDENCE_CORRUPT


def test_an_unknown_schema_version_is_refused_both_ways(tmp_path: Path) -> None:
    with pytest.raises(CalibrationFailure) as caught:
        _bundle(schema_version="leapcalib.evidence.v99")
    assert caught.value.code == FAILURE_SCHEMA
    path = write_bundle(tmp_path / "b1.json", _bundle())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["content"]["schema_version"] = "leapcalib.evidence.v99"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationFailure, match="unsupported evidence schema"):
        read_bundle(path)


def test_reading_is_bounded_and_malformed_json_is_typed(tmp_path: Path) -> None:
    path = write_bundle(tmp_path / "b1.json", _bundle())
    with pytest.raises(CalibrationFailure, match="byte budget"):
        read_bundle(path, max_bytes=16)
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationFailure, match="not valid evidence JSON"):
        read_bundle(broken)
    bare = tmp_path / "bare.json"
    bare.write_text("[]", encoding="utf-8")
    with pytest.raises(CalibrationFailure, match="not an evidence object"):
        read_bundle(bare)


def test_evidence_is_never_quietly_overwritten(tmp_path: Path) -> None:
    """The recording is the only record of what actually happened."""
    path = write_bundle(tmp_path / "b1.json", _bundle())
    with pytest.raises(CalibrationFailure, match="pass overwrite"):
        write_bundle(path, _bundle())
    assert write_bundle(path, _bundle(), overwrite=True) == path


def test_a_held_out_sample_cannot_also_be_accepted() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _bundle(accepted_ids=("o1",), holdout_ids=("o1",))
    with pytest.raises(ValueError, match="absent observations"):
        _bundle(accepted_ids=("nope",))


def test_retention_defaults_to_staying_on_the_machine() -> None:
    retention = Retention(ttl_s=3600.0)
    assert retention.syncable is False
    assert Retention.from_mapping(retention.to_dict()) == retention
    with pytest.raises(ValueError, match="positive ttl"):
        Retention(ttl_s=0.0)


def test_the_summary_describes_quality_without_grading_it() -> None:
    summary = _bundle().summary
    assert summary["observations"] == 3
    assert summary["accepted"] == 2
    assert summary["holdout"] == 1
    # Two camera samples measured of three; the joint observation has no depth.
    assert summary["depth_valid_ratio"] == pytest.approx(2 / 3)
    assert summary["settled_fraction"] == pytest.approx(1.0)
    assert summary["worst_stamp_uncertainty_s"] == pytest.approx(0.004)
    assert summary["limitations"]


def test_replay_honours_the_accepted_split_and_keeps_original_provenance() -> None:
    bundle = _bundle()
    everything = ReplaySource(bundle)
    accepted_only = ReplaySource(bundle, only_accepted=True)
    assert len(everything) == 3
    assert len(accepted_only) == 2
    assert {o.observation_id for o in accepted_only.observations()} == {"o1", "j1"}
    assert accepted_only.topology_digest == bundle.topology.digest
    # The bundle is what marks this a replay; rewriting each sample's provenance
    # would destroy the record of how it was first measured.
    assert all(
        getattr(o, "provenance", PROVENANCE_METRIC_RGBD) == PROVENANCE_METRIC_RGBD
        for o in everything.observations()
    )


def test_a_split_can_be_changed_without_touching_the_evidence() -> None:
    bundle = _bundle()
    reshuffled = bundle.with_split(accepted=("o2",), holdout=("o1",))
    assert reshuffled.observations == bundle.observations
    assert reshuffled.digest != bundle.digest
    assert bundle.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_wire_flags_cannot_fabricate_verification_or_consent(flag: object) -> None:
    asset = GeometryAsset(link_id="link3", kind=ASSET_SURFACE_MESH, digest="m")
    with pytest.raises(ValueError, match="explicit boolean"):
        GeometryAsset.from_mapping({**asset.to_dict(), "verified": flag})
    with pytest.raises(ValueError, match="explicit boolean"):
        Retention.from_mapping({"ttl_s": 10.0, "syncable": flag})
    joint = _bundle().observations[2].to_dict()
    with pytest.raises(ValueError, match="explicit boolean"):
        JointObservation.from_mapping({**joint, "still": flag})


@pytest.mark.parametrize("missing_digest", [None, "", False])
def test_bundle_integrity_cannot_be_bypassed_by_an_absent_digest(missing_digest: object) -> None:
    payload = _bundle().to_dict()
    payload["digest"] = missing_digest
    with pytest.raises(CalibrationFailure, match="no digest"):
        EvidenceBundle.from_mapping(payload)


@pytest.mark.parametrize("content", [None, [], "not an object"])
def test_malformed_content_is_a_typed_schema_failure(content: object) -> None:
    with pytest.raises(CalibrationFailure) as caught:
        EvidenceBundle.from_mapping({"content": content, "digest": "abc"})
    assert caught.value.code == FAILURE_SCHEMA


def test_a_checksummed_but_invalid_schema_still_produces_a_typed_failure() -> None:
    content = _bundle().content
    del content["topology"]
    with pytest.raises(CalibrationFailure) as caught:
        EvidenceBundle.from_mapping({"content": content, "digest": digest_of(content)})
    assert caught.value.code == FAILURE_SCHEMA


@pytest.mark.parametrize("number", [float("inf"), float("nan"), True])
def test_retention_and_creation_timestamps_must_be_canonical(number: float) -> None:
    with pytest.raises(ValueError):
        Retention(ttl_s=number)
    with pytest.raises(ValueError):
        _bundle(created_at_s=number)


def test_new_raw_evidence_files_are_private(tmp_path: Path) -> None:
    path = write_bundle(tmp_path / "private.json", _bundle())
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_inspect_reports_identity_limitations_and_retention(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_bundle(tmp_path / "b1.json", _bundle())
    assert main(["inspect", str(path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "bundle      b1" in out
    assert "piper_x" in out
    assert "limitation  fixed camera sees no base surface" in out
    assert "syncable False" in out


def test_cli_inspect_emits_machine_readable_components(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_bundle(tmp_path / "b1.json", _bundle())
    assert main(["inspect", str(path), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    # Per-component digests are what dependency invalidation consumes, so they
    # have to be reachable without parsing prose.
    assert "mount:wrist_bracket" in payload["components"]
    assert payload["summary"]["digest"]


def test_cli_replay_counts_measured_samples_not_just_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_bundle(tmp_path / "b1.json", _bundle())
    assert main(["replay", str(path), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["replayed"] == 3
    assert (payload["samples"], payload["measured_samples"]) == (3, 2)
    assert payload["streams"] == ["d435if:depth_aligned"]
    assert main(["replay", str(path), "--only-accepted", "--json"]) == EXIT_OK
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["replayed"] == 2


def test_cli_budget_reproduces_both_design_figures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "budget",
            "--task",
            "tabletop_grasp",
            "--allowed-mm",
            "5",
            "--basis",
            "jaw 85 mm - object 60 mm",
            "--speed-mm-s",
            "20",
            "--latency-ms",
            "100",
            "--lever-mm",
            "130",
            "--orientation-deg",
            "1",
            "--json",
        ]
    )
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    terms = {t["name"]: t["value_m"] for t in payload["budget"]["terms"]}
    assert terms["observation_delay"] == pytest.approx(0.002)
    assert terms["orientation_lever_arm"] == pytest.approx(0.00226893, abs=1e-8)
    assert payload["verdict"]["fits"] is True


def test_cli_budget_exits_nonzero_when_the_margin_does_not_close(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A budget that does not close is a correct answer with an actionable code."""
    code = main(
        [
            "budget",
            "--task",
            "insertion",
            "--allowed-mm",
            "1",
            "--basis",
            "chamfer 1 mm",
            "--speed-mm-s",
            "50",
            "--latency-ms",
            "150",
        ]
    )
    assert code == EXIT_FAILURE
    out = capsys.readouterr().out
    assert "verdict     EXCEEDED" in out


def test_cli_turns_a_typed_refusal_into_an_operator_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["inspect", str(broken)]) == EXIT_FAILURE
    assert "refused:" in capsys.readouterr().err


def test_cli_turns_a_missing_file_into_an_error_line_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", str(tmp_path / "absent.json")]) == EXIT_FAILURE
    assert "error:" in capsys.readouterr().err


def test_cli_advertises_implemented_compute_verbs_without_activation() -> None:
    """Offline verbs and compute-only serve exist; activation stays owner-side."""
    for implemented in ("inspect", "replay", "budget", "estimate", "shadow", "serve"):
        with pytest.raises(SystemExit) as caught:
            main([implemented, "--help"])
        assert caught.value.code == EXIT_OK
    for absent in ("activate",):
        with pytest.raises(SystemExit) as caught:
            main([absent, "--help"])
        assert caught.value.code == EXIT_USAGE


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == EXIT_USAGE

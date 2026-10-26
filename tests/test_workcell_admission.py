# Copyright (c) Alibaba, Inc. and its affiliates.
"""Workcell declarations: parsing, fail-closed admission, frames, requirements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from leapflow.hardware.skills.frames import (
    FrameGraph,
    unmet_requirements,
)
from leapflow.hardware.skills.workcell import (
    SPEED_CHANNEL,
    WorkcellError,
    WorkcellSpec,
    load_workcells,
    role_kind,
)
from tests._harness.skill_stubs import (
    ARM_ID,
    IDENTITY_MATRIX,
    WORKCELL_ID,
    WORKCELL_YAML,
    SkillNodeRegistry,
    write_workcell,
)


def _with(**changes: Any) -> dict[str, Any]:
    return {**WORKCELL_YAML, **changes}


def test_shipped_shape_parses_and_reports_its_devices() -> None:
    spec = WorkcellSpec.from_mapping(WORKCELL_YAML)
    assert spec.device_ids == (ARM_ID,), "arm and gripper on one device count once"
    assert spec.role("gripper") == ARM_ID and spec.role("camera") == ""
    assert spec.frames["tcp"].parent == "flange"
    assert spec.limits["move_timeout_s"] == 5.0
    assert spec.calibration["max_pivot_residual_mm"] == 1.0
    assert spec.calibration["max_verify_p95_mm"] == 5.0
    assert spec.bootstrap_box is not None and spec.bootstrap_box.z_min == 120


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (_with(workcell="Piper Cell"), "identifier"),
        (_with(roles={"gripper": ARM_ID}), "must bind an 'arm'"),
        (_with(frames={"base": {"parent": "flange", "procedure": "p"}}), "reserved"),
        (_with(frames={"tcp": {"parent": "camera", "procedure": "p"}}), "parent"),
        (_with(frames={"tcp": {"parent": "flange"}}), "procedure"),
        (_with(limits={"speed_pct_max": 150}), "speed_pct_max"),
        (_with(home={"joint.j1.position": "zero"}), "must be a number"),
        (_with(workspace={"tool_cone_deg": 0}), "tool_cone_deg"),
        (_with(workspace={"bootstrap_box": {"x": [1, 0], "y": [0, 1], "z": [0, 1]}}), "bounds"),
        (_with(arm=ARM_ID), "unknown fields \\['arm'\\]"),
        (_with(limts={"speed_pct_max": 10}), "unknown fields"),
        (_with(workspace={"procedure": "p", "cone_deg": 10}), "unknown fields"),
        (_with(calibration={"mystery": 1}), "calibration does not understand mystery"),
        (_with(calibration={"max_zero_error_deg": 0}), "positive finite"),
        (_with(roles={"arm": ARM_ID, "Camera Wrist": "cam"}), "role name"),
        (_with(roles={"arm": ARM_ID, "camera.wrist": ""}), "device id"),
        (_with(roles=[ARM_ID]), "needs 'roles'"),
    ],
)
def test_malformed_declarations_are_refused(data: dict[str, Any], match: str) -> None:
    with pytest.raises(WorkcellError, match=match):
        WorkcellSpec.from_mapping(data)


def test_admission_is_fail_closed_against_the_live_registry(tmp_path: Path) -> None:
    directory = tmp_path / "workcells"
    write_workcell(directory)
    write_workcell(directory, _with(workcell="ghost", roles={"arm": ARM_ID, "gripper": "missing_gripper"}))
    write_workcell(directory, _with(workcell="far_home", home={"joint.j2.position": 500.0}))
    write_workcell(directory, _with(workcell="bad_joint", home={"joint.j9.position": 0.0}))
    write_workcell(directory, WORKCELL_YAML, name="duplicate")
    (directory / "broken.yaml").write_text("workcell: [unclosed", encoding="utf-8")

    admitted, report = load_workcells(directory, SkillNodeRegistry())

    assert list(admitted) == [WORKCELL_ID]
    reasons = {note.workcell or Path(note.source).stem: note.detail for note in report.rejected}
    assert "not declared" in reasons["ghost"]
    assert "outside its envelope" in reasons["far_home"]
    assert "not a writable channel" in reasons["bad_joint"]
    assert reasons["piper_cell"] == "duplicate workcell id"
    assert "broken" in reasons


def test_missing_directory_admits_nothing(tmp_path: Path) -> None:
    admitted, report = load_workcells(tmp_path / "absent", SkillNodeRegistry())
    assert admitted == {} and report.rejected == ()


def test_a_camera_frame_without_a_measured_target_mount_is_not_a_calibration(
    tmp_path: Path,
) -> None:
    """A hand-eye record that never measured a target mount is an assertion.

    Such a record is what an import or a hand-edited matrix leaves behind, and
    it survives the camera being remounted: the transform stays plausible, no
    downstream consumer can tell it apart from a measurement, and perception
    keeps reporting geometry for a view that no longer exists.
    """
    registry = SkillNodeRegistry(store_path=tmp_path / "instrument.duckdb")
    spec = WorkcellSpec.from_mapping(_with(
        roles={"arm": ARM_ID, "camera": ARM_ID},
        frames={
            "tcp": {"parent": "flange", "procedure": "tcp.offset"},
            "camera": {"parent": "base", "procedure": "handeye.eye_to_hand"},
        },
    ))
    registry.calibrate("handeye.eye_to_hand", pose={"x": 500., "y": 0., "z": 0.})
    frames = FrameGraph.snapshot(spec, registry.calibration_store)
    assert frames.transform("camera") is None
    assert frames.parameters("camera") == {}
    assert "target mount" in frames.stale["camera"]
    assert unmet_requirements(("calib:camera",), spec, frames) == [
        "camera calibration is stale (the record carries no measured target mount and "
        "no screw-motion evidence); run calib.handeye again"
    ]


def test_a_record_naming_a_method_without_its_evidence_is_not_a_calibration(
    tmp_path: Path,
) -> None:
    """Naming a board-free method does not stand in for having used one.

    The gate asks each method for the evidence that method produces. Declaring
    ``markerless_screw`` with no joint list, or with two joints, is the same
    assertion as a bare matrix: two rotations share the vertical, so no mount can
    come out of them.
    """
    registry = SkillNodeRegistry(store_path=tmp_path / "instrument.duckdb")
    spec = WorkcellSpec.from_mapping(_with(
        roles={"arm": ARM_ID, "camera": ARM_ID},
        frames={
            "tcp": {"parent": "flange", "procedure": "tcp.offset"},
            "camera": {"parent": "base", "procedure": "handeye.eye_to_hand"},
        },
    ))
    pair = {"joint": "j1", "angle_deg": 30.0, "agreement": 0.6, "shift_gap_mm": 4.0}
    for evidence, reason in (
        ({"method": "markerless_screw"}, "lists no joints"),
        ({"method": "markerless_screw", "pairs": [pair]}, "covers 1 joints"),
        ({"method": "guess"}, "names no measurement method"),
        ({"method": "markerless_screw", "pairs": [pair] * 3}, ""),
    ):
        record = registry.calibrate(
            "handeye.eye_to_hand",
            matrix=[[1, 0, 0, 500], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            parameters={"screw_evidence": evidence},
        )
        frames = FrameGraph.snapshot(spec, registry.calibration_store)
        stale = frames.stale.get("camera", "")
        assert reason in stale, (evidence, stale)
        assert (frames.transform("camera") is not None) is (reason == "")
        assert record


def test_frames_resolve_from_the_latest_calibration_preferring_a_matrix(tmp_path: Path) -> None:
    registry = SkillNodeRegistry(store_path=tmp_path / "instrument.duckdb")
    spec = WorkcellSpec.from_mapping(
        _with(
            roles={"arm": ARM_ID, "camera": ARM_ID},
            frames={
                "tcp": {"parent": "flange", "procedure": "tcp.offset"},
                "camera": {"parent": "base", "procedure": "handeye.eye_to_hand"},
            },
        )
    )
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": 100}, recorded_at=1.0)
    registry.calibrate("tcp.offset", pose={"x": 0, "y": 0, "z": 120}, recorded_at=2.0)
    store = registry.calibration_store
    assert store is not None
    from leapflow.hardware.calibration_store import CalibrationRecord

    matrix = [[1, 0, 0, 500], [0, 1, 0, 0], [0, 0, 1, 600], [0, 0, 0, 1]]
    assert store.record(
        CalibrationRecord(
            device_id=WORKCELL_ID,
            procedure_id="handeye.eye_to_hand",
            recorded_at=3.0,
            matrix=matrix,
            pose={"x": 1, "y": 1, "z": 1},
            parameters={"target_mount": IDENTITY_MATRIX},
        )
    )
    frames = FrameGraph.snapshot(spec, store)
    tcp, camera = frames.transform("tcp"), frames.transform("camera")
    assert tcp is not None and tcp.to_pose().z == pytest.approx(120)
    assert camera is not None and camera.to_pose().x == pytest.approx(500)
    assert frames.versions == {"tcp": 2.0, "camera": 3.0}
    assert frames.parent("camera") == "base" and frames.parent("nope") == ""
    assert frames.workspace is None


def test_a_malformed_calibration_is_ignored_not_trusted(tmp_path: Path) -> None:
    registry = SkillNodeRegistry(store_path=tmp_path / "instrument.duckdb")
    registry.calibrate("workspace.safe_box", parameters={"x": [1, 0]})
    frames = FrameGraph.snapshot(WorkcellSpec.from_mapping(WORKCELL_YAML), registry.calibration_store)
    assert frames.workspace is None and "workspace" not in frames.versions


def test_unmet_requirements_name_the_procedure_to_run() -> None:
    spec = WorkcellSpec.from_mapping(WORKCELL_YAML)
    frames = FrameGraph(spec, {}, None, {})
    reasons = unmet_requirements(
        ("role:arm", "role:camera", "calib:tcp", "calib:camera", "bogus"), spec, frames
    )
    assert reasons == [
        f"workcell {WORKCELL_ID!r} binds no camera",
        "tcp is not calibrated (run calib.tcp)",
        "camera is not calibrated (run calib.handeye)",
        "unknown requirement 'bogus'",
    ]


def test_roles_are_open_and_every_bound_device_is_leased(tmp_path: Path) -> None:
    spec = WorkcellSpec.from_mapping(
        _with(roles={"arm": ARM_ID, "camera": "cam_top", "camera.wrist": "cam_wrist"})
    )

    assert spec.role("camera.wrist") == "cam_wrist"
    assert spec.device_ids == (ARM_ID, "cam_top", "cam_wrist")
    assert spec.actuator_ids == (ARM_ID,), "a camera-kind role is no actuator by kind"
    assert role_kind("camera.wrist") == "camera"
    assert unmet_requirements(("role:camera.wrist",), spec, FrameGraph(spec, {}, None, {})) == []


def test_admission_halts_every_writable_device_and_checks_each_role_kind(tmp_path: Path) -> None:
    directory = tmp_path / "workcells"
    write_workcell(directory, _with(roles={"arm": ARM_ID, "gripper.aux": ARM_ID}))
    write_workcell(
        directory, _with(workcell="no_wrist", roles={"arm": ARM_ID, "camera.wrist": "absent"})
    )

    admitted, report = load_workcells(directory, SkillNodeRegistry())

    assert admitted[WORKCELL_ID].actuator_ids == (ARM_ID,)
    assert admitted[WORKCELL_ID].role("gripper.aux") == ARM_ID
    (note,) = report.rejected
    assert note.workcell == "no_wrist" and "camera.wrist device 'absent'" in note.detail


@pytest.mark.parametrize("tuning", [
    {"min_frames": 2}, {"min_frames": 3.5}, {"min_frames": 11},
    {"min_proof_lift_mm": 11}, {"poll_s": 1}, {"alignment_mm": True},
])
def test_grasp_evidence_tuning_is_admitted_fail_closed(tuning: dict[str, Any]) -> None:
    with pytest.raises(WorkcellError, match="grasp_verify"):
        WorkcellSpec.from_mapping(_with(grasp_verify=tuning))


def test_camera_frames_resolve_independent_mount_versions(tmp_path: Path) -> None:
    registry = SkillNodeRegistry(store_path=tmp_path / "instrument.duckdb")
    spec = WorkcellSpec.from_mapping(_with(
        roles={"arm": ARM_ID, "camera": "overview", "camera.wrist": "wrist"}, frames={
            "camera": {"parent": "base", "procedure": "handeye.eye_to_hand"},
            "camera.wrist": {"parent": "flange", "procedure": "handeye.eye_in_hand"},
        },
    ))
    mount = {"target_mount": IDENTITY_MATRIX}
    registry.calibrate("handeye.eye_to_hand", parameters=mount,
                       pose={"x": 500., "y": 0., "z": 0.}, recorded_at=1.)
    registry.calibrate("handeye.eye_in_hand", parameters=mount,
                       pose={"x": 60., "y": 0., "z": 0.}, recorded_at=2.)
    frames = FrameGraph.snapshot(spec, registry.calibration_store)
    assert frames.transform("camera").translation == (500., 0., 0.)
    assert frames.transform("camera.wrist").translation == (60., 0., 0.)
    assert frames.versions == {"camera": 1., "camera.wrist": 2.}
    registry.calibrate("handeye.eye_in_hand", parameters=mount,
                       pose={"x": 61., "y": 0., "z": 0.}, recorded_at=3.)
    assert FrameGraph.snapshot(spec, registry.calibration_store).versions == {
        "camera": 1., "camera.wrist": 3.,
    }


def test_camera_frames_keep_distinct_calibration_procedures() -> None:
    data = _with(roles={"arm": ARM_ID, "camera": "overview", "camera.wrist": "wrist"}, frames={
        "camera": {"parent": "base", "procedure": "handeye.eye_to_hand"},
        "camera.wrist": {"parent": "flange", "procedure": "handeye.eye_in_hand"},
    })
    spec = WorkcellSpec.from_mapping(data)
    assert spec.frames["camera.wrist"].parent == "flange"
    assert "calib.handeye" in unmet_requirements(
        ("calib:camera.wrist",), spec, FrameGraph(spec, {}, None, {}),
    )[0]
    data["frames"]["camera.wrist"]["procedure"] = "handeye.eye_to_hand"
    with pytest.raises(WorkcellError, match="unique"):
        WorkcellSpec.from_mapping(data)


@pytest.mark.parametrize("frames", [
    {"camera.wrist": {"parent": "base", "procedure": "p"}},
    {"camera": {"parent": "tcp", "procedure": "p"}},
    {"camera": {"parent": "base", "procedure": "p", "ignored": True}},
])
def test_camera_frames_refuse_unbound_roles_and_invalid_mounts(frames: dict[str, Any]) -> None:
    with pytest.raises(WorkcellError):
        WorkcellSpec.from_mapping(_with(roles={"arm": ARM_ID, "camera": "overview"}, frames=frames))


def test_a_speed_cap_below_the_arms_lowest_speed_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "workcells"
    registry = SkillNodeRegistry()
    floor = registry.context(ARM_ID).channel(SPEED_CHANNEL).envelope.min_value
    assert floor is not None and floor > 0
    write_workcell(directory, _with(limits={"speed_pct_max": floor / 2, "default_speed_pct": 0.1}))

    admitted, report = load_workcells(directory, registry)

    assert admitted == {}
    assert "below the arm's lowest commandable speed" in report.rejected[0].detail

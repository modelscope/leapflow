# Copyright (c) Alibaba, Inc. and its affiliates.
"""Leased LHP acquisition and nominal-only safety using simulated boundary I/O."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.identity import MountIdentity
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.calibration.physical_model import SafetyGeometry
from leapflow.hardware.calibration.pose_policy import PoseCollectionPolicy
from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.skills.workcell import WorkcellSpec
from tests._harness.skill_stubs import ARM_ID, JOINTS, WORKCELL_ID, WORKCELL_YAML, write_workcell
from tests import test_leapcalib_collection as collection_tests
from tests.test_leapcalib_collection import (
    request,
    planned,
    permit,
    finished,
)  # noqa: F401
from tests.test_leapcalib_acquisition import profile
from tests.test_workcell_reload import approved_reload

pytestmark = pytest.mark.component
stationary_owner = collection_tests.owner


@pytest.fixture
async def pose_owner(stationary_owner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    registry, cache, sensor, runtime, server, issuer = stationary_owner
    chain = DhChain(
        (DhLink(250, 180, 370, 0), *(DhLink(0, 0, 0, 0) for _ in range(5))), convention="standard"
    )
    arm = registry.context(ARM_ID)
    arm = replace(arm, kinematics=replace(arm.kinematics, dh=chain.to_dict()))
    monkeypatch.setitem(registry._contexts, ARM_ID, arm)
    transport = await registry.transport(ARM_ID)
    monkeypatch.setattr(transport, "_context", arm)
    transport._values["arm.speed_pct"] = 10
    original_write = transport.write

    async def write(cid, value):
        outcome = await original_write(cid, value)
        if outcome.ok and cid in JOINTS:
            pose = chain.forward([transport._values[j] for j in JOINTS])
            transport._values.update(
                dict(zip(("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"), vars(pose).values()))
            )
        return outcome

    monkeypatch.setattr(transport, "write", write)
    mount = MountIdentity("wrist", "flange")
    original_capture = sensor.capture_rgbd

    async def capture(**kwargs):
        value = await original_capture(**kwargs)
        return (
            value
            if sensor.defect == "mount_digest"
            else replace(value, metadata={**value.metadata, "mount_digest": mount.digest})
        )

    monkeypatch.setattr(sensor, "capture_rgbd", capture)
    monkeypatch.setattr(await registry.transport("camera"), "capture_rgbd", capture)
    acquisition = CollectionProfile(
        "camera", "wrist", ("can",), profile(timeout_s=2), 120, 16 * 1024 * 1024, 16
    )
    declaration = {
        **WORKCELL_YAML,
        "roles": {"arm": ARM_ID, "camera": "camera"},
        "frames": {
            **WORKCELL_YAML["frames"],
            "camera": {"parent": "flange", "procedure": "camera.handeye"},
        },
        "calibration_acquisition": acquisition.to_dict(),
        "limits": {
            **WORKCELL_YAML["limits"],
            "speed_pct_max": 10,
            "default_speed_pct": 10,
            "move_timeout_s": 2,
        },
    }
    write_workcell(tmp_path / "workcells", declaration)
    await approved_reload(server, issuer)
    registry.bind_approval_keys({issuer.key_id: issuer._private.public_key()}, node_id="node")
    anchor_plan = await planned(server)
    anchor_run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=anchor_plan["plan"]["digest"],
            permit=permit(issuer, anchor_plan),
            lease_s=10,
        ),
    )
    assert anchor_run.outcome.ok, anchor_run.outcome
    anchor = anchor_run.outcome.result["captures"][0]["evidence_digest"]
    geometry = SafetyGeometry(
        (25,) * 7,
        WorkspaceBox(-1000, 1000, -1000, 1000, -500, 1200),
        (WorkspaceBox(800, 900, 800, 900, 800, 900),),
        "independent-software-geometry",
        35,
        5,
        10,
        2,
    )
    policy = PoseCollectionPolicy(anchor, mount, geometry, (0.01,) * 6, 10, "a" * 64)
    declaration["calibration_motion"] = policy.to_dict()
    write_workcell(tmp_path / "workcells", declaration)
    await approved_reload(server, issuer)
    yield registry, cache, sensor, runtime, server, issuer, declaration, policy, transport


def poses_args(count: int = 3):
    return {
        "region": [2, 1, 8, 8],
        "poses": [{**dict.fromkeys(JOINTS, 0), JOINTS[0]: i * 0.3} for i in range(count)],
        "speed_pct": 10,
    }


async def test_real_lhp_pose_acquisition_is_one_shot_leased_disjoint_and_no_return(
    pose_owner,
) -> None:
    registry, _, _, runtime, server, issuer, _, policy, transport = pose_owner
    planned_run = await request(
        server, "skill.plan", workcell=WORKCELL_ID, skill="calib.collect_poses", args=poses_args()
    )
    assert "plan" in planned_run, planned_run
    assert planned_run["consent"] == "each_run"
    digest = planned_run["plan"]["digest"]
    before = len(transport._write_log)
    run = await finished(
        runtime,
        await request(
            server,
            "skill.execute",
            plan_digest=digest,
            permit=permit(issuer, planned_run),
            lease_s=10,
        ),
    )
    assert run.outcome.ok, run.outcome
    result = run.outcome.result
    assert len(result["captures_by_pose"]) == 3 and len(result["factor_pairs"]) == 2
    assert len(set(v for pair in result["factor_pairs"] for v in pair)) == 4
    assert not result["auto_return"] and result["activation"] == "none"
    assert transport._values[JOINTS[0]] == 0.6
    assert all(cid in (*JOINTS, "arm.speed_pct") for cid, _ in transport._write_log[before:])
    for pair in result["captures_by_pose"]:
        for summary in pair:
            content = await registry.calibration_evidence.read_capture(
                summary["evidence_digest"], workcell_id=WORKCELL_ID
            )
            assert content["binding"]["plan"] == planned_run["plan"]
            assert (
                content["binding"]["plan"]["basis"]["poses"]["reference_digest"]
                == policy.reference_acquisition
            )
    replay = await request(
        server, "skill.execute", plan_digest=digest, permit=permit(issuer, planned_run), lease_s=10
    )
    assert "outcome" in replay and replay["outcome"]["status"] != "succeeded"


@pytest.mark.parametrize(
    "problem", ["speed", "count", "joints", "campaign", "l1", "start", "mount"]
)
async def test_pose_envelope_refusals_before_any_write(
    pose_owner, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, sensor, _, server, _, _, _, transport = pose_owner
    args = poses_args()
    if problem == "speed":
        args["speed_pct"] = 11
    elif problem == "count":
        args = poses_args(9)
    elif problem == "joints":
        args["poses"][0].pop(JOINTS[1])
    elif problem == "campaign":
        args["poses"][-1][JOINTS[0]] = 6
    elif problem == "l1":
        args["poses"] = [{**dict.fromkeys(JOINTS, 0), JOINTS[0]: v} for v in [2.5, -2.5, 2.5, -2.5]]
    elif problem == "start":
        transport._values["arm.speed_pct"] = 20
    elif problem == "mount":
        sensor.defect = "mount_digest"
    before = list(transport._write_log)
    result = await request(
        server, "skill.plan", workcell=WORKCELL_ID, skill="calib.collect_poses", args=args
    )
    assert "outcome" in result, result
    assert result["outcome"]["status"] != "succeeded"
    assert transport._write_log == before


async def test_cancel_after_first_motion_prevents_remaining_moves(
    pose_owner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, runtime, server, issuer, _, _, transport = pose_owner
    result = await request(
        server, "skill.plan", workcell=WORKCELL_ID, skill="calib.collect_poses", args=poses_args()
    )
    assert "plan" in result, result
    original = transport.write
    landed = asyncio.Event()

    async def write(cid, value):
        outcome = await original(cid, value)
        if cid == JOINTS[-1]:
            landed.set()
        return outcome

    monkeypatch.setattr(transport, "write", write)
    started = await request(
        server,
        "skill.execute",
        plan_digest=result["plan"]["digest"],
        permit=permit(issuer, result),
        lease_s=10,
    )
    await asyncio.wait_for(landed.wait(), 5)
    await runtime.stop(started["run"]["run_id"])
    run = await finished(runtime, started)
    assert not run.outcome.ok
    assert transport._values[JOINTS[0]] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed_pct_max", 11),
        ("max_moves", 9),
        ("max_travel_mm", 41),
        ("campaign_joint_deg", 6),
        ("max_duration_s", 121),
    ],
)
def test_motion_policy_caps_cannot_be_widened(field, value) -> None:
    geometry = SafetyGeometry(
        (1, 1), WorkspaceBox(-1000, 1000, -1000, 1000, -1000, 1000), (), "test", 1, 1, 1, 1
    )
    policy = PoseCollectionPolicy(
        "a" * 64, MountIdentity("wrist", "flange"), geometry, (0.1,), 1, "b" * 64
    )
    with pytest.raises(ValueError):
        replace(policy, **{field: value})


def test_workcell_motion_policy_is_optional_and_roundtrips() -> None:
    spec = WorkcellSpec.from_mapping(WORKCELL_YAML)
    assert spec.calibration_motion == {}
    assert (
        WorkcellSpec.from_mapping({**WORKCELL_YAML, "calibration_motion": {}}).calibration_motion
        == {}
    )

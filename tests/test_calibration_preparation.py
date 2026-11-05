# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline preparation never promotes historical/synthetic inputs to qualification."""
from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from leapcalib.contracts.identity import MountIdentity, digest_of
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.acquisition import CollectionProfile
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry
from leapflow.hardware.calibration.pose_policy import PoseCollectionPolicy
from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from tests._harness.skill_stubs import ARM_ID, WORKCELL_ID, piper_context
from tests.test_leapcalib_acquisition import profile

pytestmark = pytest.mark.component
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool():
    return runpy.run_path(str(ROOT / "tools/calib_prepare.py"))


def probe():
    return {"schema": "leapflow.calibration_preflight.v1", "remote_changes_performed": False,
            "node_id": "node", "arm": piper_context().to_dict(),
            "workcell": {"workcell": WORKCELL_ID, "roles": {"arm": ARM_ID, "camera.wrist": "wrist"}},
            "camera_metadata": {"readings": [{"value": {"serial_number": "fixture"}}]}}


def test_drafts_preserve_missing_evidence_and_emit_args_only(tool, tmp_path):
    result = tool["prepare"](probe(), {})
    CollectionProfile.from_mapping(result["stage_a_overlay"]["calibration_acquisition"])
    assert result["stage_b_overlay"] is None and not result["pose_batches"]
    assert "rate_bound_evidence" in result["unresolved"] and "reference_acquisition" in result["unresolved"]
    assert result["physical_qualification"] is False and result["proposed_settled_not_measured"]
    assert result["trial_stamp_bound_s"] > result["trial_stamp_gate_s"]
    assert result["readiness_blockers"] and result["deferred_inputs_template"]["reference_acquisition"] is None
    assert result["device_config_patch"]["transport.config.mount_digest"] == MountIdentity(**result["mount"]).digest
    source = tmp_path / "probe.json"
    source.write_text(json.dumps(probe()))
    tool["write_preparation"](tmp_path / "draft", result, [source])
    args = json.loads((tmp_path / "draft/collect.args.json").read_text())
    assert set(args) == {"region", "frames"}
    assert "args" not in json.loads((tmp_path / "draft/trial.args.draft.json").read_text())
    with pytest.raises(FileExistsError):
        tool["write_preparation"](tmp_path / "draft", result, [source])


@pytest.mark.parametrize("region", [[-1, 0, 20, 20], [0, 0, 1000, 480], [0, 0, True, 100], [0, 0, 1, 1]])
def test_invalid_regions_refused(tool, region):
    with pytest.raises(ValueError, match="region"):
        tool["prepare"](probe(), {"region": region})


def test_pose_batches_obey_production_sweep_and_no_rate_percentage_inference(tool):
    chain = DhChain((DhLink(250, 180, 370, 0), *(DhLink(0, 0, 0, 0) for _ in range(5))), convention="standard")
    arm = piper_context()
    arm = replace(arm, kinematics=replace(arm.kinematics, dh=chain.to_dict()))
    model = PhysicalModel(chain, (0.0,) * 6, Transform(), digest_of(chain.to_dict()), "fixture", "fixture")
    geometry = SafetyGeometry((25,) * 7, WorkspaceBox(-1000, 1000, -1000, 1000, -500, 1200),
        (WorkspaceBox(800, 900, 800, 900, 800, 900),), "offline-fixture", 35, 5, 10, 2)
    policy = PoseCollectionPolicy("a" * 64, MountIdentity("fixture", "flange"), geometry, (0.01,) * 6,
                                  10, "b" * 64, speed_pct_max=1)
    acquisition = CollectionProfile("camera.wrist", "wrist", ("can",), profile(timeout_s=2), 120, 16 * 1024 * 1024, 16)
    batches = tool["pose_batches"](model, policy, acquisition, arm, [0] * 6,
                                  move_timeout_s=20, run_timeout_s=120, region=[0, 0, 16, 16])
    assert batches and all(2 <= len(batch["args"]["poses"]) <= 3 for batch in batches)
    assert all(batch["duration_budget_s"] <= 120 for batch in batches)
    terms = tool["inflation_requirements"](model, geometry, acquisition.settled, list(policy.joint_rate_bounds_rad_s),
                                          10, fk_mm=5, fk_deg=2)
    slower = replace(policy, speed_pct_max=0.5)
    same = tool["inflation_requirements"](model, geometry, acquisition.settled, list(slower.joint_rate_bounds_rad_s),
                                         10, fk_mm=5, fk_deg=2)
    assert same == terms, "a speed setting does not silently reduce independent physical rate bounds"
    with pytest.raises(ValueError, match="deadline"):
        tool["pose_batches"](model, policy, acquisition, arm, [0] * 6,
                             move_timeout_s=100, run_timeout_s=120, region=[0, 0, 16, 16])


def pose_report(index, mount):
    captures = [[{"evidence_digest": digest_of({"fixture": index, "pose": pose, "frame": frame})}
                 for frame in range(2)] for pose in range(3)]
    anchor = "a" * 64
    return {"ok": True, "status": "succeeded", "workcell": "node/cell", "skill": "calib.collect_poses",
            "run_id": f"fixture-{index}", "plan": {"basis": {
                "collection": {"node_session_id": "session", "camera_session_id": "camera",
                    "model_digest": "model", "metric_profile_digest": "profile", "mount_digest": mount.digest},
                "poses": {"reference_digest": anchor}}},
            "result": {"schema": "leapflow.pose_collection.v1", "raw_exported": False,
                       "activation": "none", "reference_digest": anchor, "captures_by_pose": captures,
                       "factor_pairs": [[captures[i][1]["evidence_digest"], captures[i + 1][0]["evidence_digest"]]
                                        for i in range(2)]}}


def test_trial_split_uses_distinct_completed_sources_in_original_order(tool):
    mount = MountIdentity("fixture", "flange")
    reports = [pose_report(index, mount) for index in range(3)]
    pairs = tool["trial_pairs"](reports, node_id="node", workcell="cell", mount=mount.to_dict())
    assert pairs == [pair for report in reports for pair in report["result"]["factor_pairs"]]
    assert len(set(v for pair in pairs for v in pair)) == 12
    with pytest.raises(ValueError, match="run IDs"):
        tool["trial_pairs"]([reports[0], reports[0]], node_id="node", workcell="cell", mount=mount.to_dict())


@pytest.mark.parametrize("defect", ["failed", "owner", "session", "pairs", "source", "shape"])
def test_trial_split_refuses_failed_cross_session_or_reused_evidence(tool, defect):
    mount = MountIdentity("fixture", "flange")
    reports = [pose_report(index, mount) for index in range(2)]
    second = reports[1]
    if defect == "failed":
        second["ok"] = False
    elif defect == "owner":
        second["workcell"] = "other/cell"
    elif defect == "session":
        second["plan"]["basis"]["collection"]["node_session_id"] = "restart"
    elif defect == "pairs":
        second["result"]["factor_pairs"].reverse()
    elif defect == "shape":
        second["plan"] = None
    else:
        second["result"]["captures_by_pose"] = reports[0]["result"]["captures_by_pose"]
        second["result"]["factor_pairs"] = reports[0]["result"]["factor_pairs"]
    with pytest.raises(ValueError):
        tool["trial_pairs"](reports, node_id="node", workcell="cell", mount=mount.to_dict())


def test_null_input_template_remains_an_unresolved_draft(tool):
    original = tool["prepare"](probe(), {})
    regenerated = tool["prepare"](probe(), original["deferred_inputs_template"])
    assert regenerated["stage_b_overlay"] is None and not regenerated["pose_batches"]
    assert "reference_acquisition" in regenerated["unresolved"]
    assert regenerated["readiness_blockers"]


def test_live_catalog_without_run_duration_uses_explicit_draft_budget(tool, monkeypatch):
    data = probe()
    data["workcell"]["limits"] = {"fk_tolerance_mm": 5, "fk_tolerance_deg": 2, "move_timeout_s": 20}
    geometry = SafetyGeometry((25,) * 7, WorkspaceBox(-10000, 10000, -10000, 10000, -10000, 10000),
                             (), "fixture", 500, 500, 500, 2)
    inputs = {"geometry": geometry.to_dict(), "joint_rate_bounds_rad_s": [0.01] * 6,
              "endpoint_rate_bound_mm_s": 10, "rate_provenance_digest": "b" * 64,
              "reference_acquisition": "a" * 64, "reference_joints_deg": [0] * 6,
              "tool_transform": Transform().to_dict()}
    calls = []
    monkeypatch.setitem(tool["prepare"].__globals__, "pose_batches", lambda *a, **kw: calls.append(kw) or [])
    result = tool["prepare"](data, inputs)
    assert calls[0]["run_timeout_s"] == 120
    assert result["run_timeout_assumption_requires_node_preview"]
    assert result["physical_qualification"] is False


async def test_probe_uses_reads_only_and_never_exports_transport_credentials(monkeypatch):
    probe_tool = runpy.run_path(str(ROOT / "tools/calib_probe.py"))
    from tests._harness.skill_stubs import JOINTS

    calls = []
    class Client:
        async def connect(self):
            return {"node_id": "node", "write_mode": "permit_required"}

        async def request(self, method, params=None):
            calls.append(method)
            if method == "list":
                return {"devices": [{"device_id": ARM_ID, "transport": {"token": "fixture-secret"}},
                                    {"device_id": "wrist", "transport": {"token": "fixture-secret"}}]}
            if method == "skill.list":
                return {"workcells": [{"workcell": WORKCELL_ID, "roles": {"arm": ARM_ID, "camera.wrist": "wrist"},
                                       "frames": {"camera.wrist": {"parent": "flange"}}}]}
            ids = params["channel_ids"]
            return {"observed_at": 100.0, "readings": [{"channel_id": cid, "observed_at": 99.95,
                    "value": 1.0 if cid in JOINTS else {}} for cid in ids]}

    result = await probe_tool["probe"](Client(), node_id="node", arm_id=ARM_ID, workcell=WORKCELL_ID, samples=3)
    assert set(calls) == {"list", "skill.list", "read_batch"}
    assert "fixture-secret" not in json.dumps(result)
    assert result["timing_diagnostics"]["qualified_monotonic_bounds"] is False

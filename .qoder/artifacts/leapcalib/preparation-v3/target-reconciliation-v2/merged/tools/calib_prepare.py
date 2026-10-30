# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline camera prerequisite drafts and bounded pose args from explicit evidence.

Reads exported owner observations; never connects, deploys, writes live config,
or commands hardware. Missing physical evidence stays missing. Neither hashes
nor offline geometry checks establish physical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from leapcalib.contracts.identity import MountIdentity, digest_of
from leapcalib.geometry.se3 import Transform
from leapcalib.estimation.numeric import SolveConfig
from leapcalib.runtime.protocol import WorkerBudgets
from leapflow.hardware.calibration.acquisition import CollectionProfile, SettledProfile
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry, dual_model_reach_mm
from leapflow.hardware.calibration.pose_policy import PoseCollectionPolicy
from leapflow.hardware.calibration.trial import TrialPolicy
from leapflow.hardware.context import HardwareContext
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.workcell import joint_channels


PROPOSED_SETTLED = {
    "max_feedback_age_s": 0.1, "max_coherence_s": 0.08, "feedback_uncertainty_s": 0.01,
    "exposure_uncertainty_s": 0.04, "receipt_uncertainty_s": 0.02,
    "max_joint_drift_rad": 0.003, "max_tcp_drift_mm": 2.0, "max_tcp_rotation_deg": 0.5,
    "joint_uncertainty_rad": 0.0005, "tcp_uncertainty_mm": 0.5, "rotation_uncertainty_deg": 0.1,
    "poll_s": 0.05, "timeout_s": 3.0, "min_samples": 3, "span_s": 0.5,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("preparation input exceeds 4MiB")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("preparation input must be an object")
    return value


def inflation_requirements(model: PhysicalModel, geometry: SafetyGeometry, settled: SettledProfile,
                           rates: list[float], endpoint_rate: float, *, fk_mm: float,
                           fk_deg: float) -> dict[str, float]:
    """Same conservative terms as production validation; no inferred percentage scaling."""
    reach = dual_model_reach_mm(model, model, geometry)
    return {"reach_mm": reach, "uncertainty_mm": fk_mm + reach * math.radians(fk_deg),
            "sensing_mm": settled.tcp_uncertainty_mm + reach *
                (model.dof * settled.joint_uncertainty_rad + math.radians(settled.rotation_uncertainty_deg)),
            "timing_mm": (settled.max_feedback_age_s + settled.feedback_uncertainty_s + settled.max_coherence_s) *
                (endpoint_rate + reach * sum(rates)), "arrival_mm": settled.max_tcp_drift_mm}


def pose_batches(model: PhysicalModel, policy: PoseCollectionPolicy, acquisition: CollectionProfile,
                 arm: HardwareContext, reference: list[float], *, move_timeout_s: float,
                 run_timeout_s: float, region: list[int], step_deg: float = 1.0) -> list[dict[str, Any]]:
    """Explicit small raw-joint targets, screened by the same production sweep and budgets."""
    ids = joint_channels(arm)
    if len(reference) != model.dof or len(ids) != model.dof:
        raise ValueError("exact original indexed anchor raw joints required")
    if isinstance(step_deg, bool) or not isinstance(step_deg, (int, float)) or not math.isfinite(step_deg) or not 0 < step_deg <= policy.campaign_joint_deg:
        raise ValueError("positive pose step within the unchanged campaign joint cap required")
    if any(not arm.channel(cid).envelope.contains(v) for cid, v in zip(ids, reference)):
        raise ValueError("anchor is outside command bounds; never clamp or return to stopped feedback")
    dwell = 0.3 + 2 * sum(getattr(acquisition.settled, name) for name in (
        "receipt_uncertainty_s", "exposure_uncertainty_s", "feedback_uncertainty_s"))
    per_move = move_timeout_s + 2 * acquisition.settled.timeout_s + dwell
    count = min(3, policy.max_moves, acquisition.max_frames // 2,
                math.floor((min(policy.max_duration_s, run_timeout_s) - 10) / per_move))
    if count < 2:
        raise ValueError("two poses cannot fit the declared run deadline")
    targets = []
    for axis in range(model.dof):
        for amount in (step_deg, -step_deg):
            q = list(reference)
            q[axis] += amount
            if any(not arm.channel(cid).envelope.contains(v) for cid, v in zip(ids, q)):
                continue
            targets.append(q)
    batches, before = [], list(reference)
    for offset in range(0, len(targets), count):
        group = targets[offset:offset + count]
        if len(group) < 2:
            break
        travel = joint_l1 = 0.0
        for after in group:
            moved, rotated = policy.sweep(model, reference, before, after)
            travel += moved
            joint_l1 += rotated
            before = after
        if travel > policy.max_travel_mm or joint_l1 > policy.max_joint_l1_deg:
            raise ValueError("proposed batch exceeds run travel budget")
        batches.append({"args": {"region": region, "poses": [dict(zip(ids, q)) for q in group],
                                 "speed_pct": policy.speed_pct_max},
                        "duration_budget_s": len(group) * per_move + 10,
                        "nominal_travel_mm": travel, "joint_l1_deg": joint_l1,
                        "requires_fresh_node_preview_and_explicit_approval": True})
    if not batches:
        raise ValueError("no bounded two-pose batch survives safety checks")
    return batches


def trial_pairs(reports: list[dict[str, Any]], *, node_id: str, workcell: str,
                mount: dict[str, Any]) -> list[list[str]]:
    """Freeze redacted run references in input order; the owner still verifies every source.

    No pixel export, evidence-store access, authority, or qualification is involved.
    Never substitute failed runs or reorder a split based on tracking results.
    """
    if not 1 <= len(reports) <= 32:
        raise ValueError("bounded successful pose-run reports required")
    pairs, runs, expected = [], set(), None
    mount_digest = MountIdentity(**mount).digest
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("typed pose-run report required")
        if report.get("ok") is not True or report.get("status") != "succeeded" or report.get("workcell") != f"{node_id}/{workcell}" or report.get("skill") != "calib.collect_poses":
            raise ValueError("successful exact-owner pose result required; previews/failures cannot enter trial")
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in runs:
            raise ValueError("distinct completed run IDs required")
        runs.add(run_id)
        plan, result = report.get("plan", {}), report.get("result", {})
        if not isinstance(plan, dict) or not isinstance(result, dict):
            raise ValueError("typed pose-run plan and result required")
        basis = plan.get("basis", {})
        if not isinstance(basis, dict) or not isinstance(basis.get("collection"), dict) or not isinstance(basis.get("poses"), dict):
            raise ValueError("complete sealed pose-run basis required")
        collection = basis["collection"]
        lineage = tuple(collection.get(key) for key in (
            "node_session_id", "camera_session_id", "model_digest", "metric_profile_digest", "mount_digest"))
        lineage += (basis.get("poses", {}).get("reference_digest"),)
        if not all(isinstance(v, str) and v for v in lineage) or lineage[4] != mount_digest or (expected is not None and lineage != expected):
            raise ValueError("pose-run owner/model/camera/mount/anchor lineage differs")
        expected = lineage
        if result.get("schema") != "leapflow.pose_collection.v1" or result.get("raw_exported") is not False or result.get("activation") != "none" or result.get("reference_digest") != lineage[-1]:
            raise ValueError("redacted nominal-only pose result required")
        captures = result.get("captures_by_pose", [])
        if not isinstance(captures, list) or not 2 <= len(captures) <= 8 or any(not isinstance(c, list) or len(c) != 2 for c in captures):
            raise ValueError("two separated captures per pose required")
        if any(not isinstance(row, dict) or "evidence_digest" not in row for group in captures for row in group):
            raise ValueError("typed indexed capture references required")
        derived = [[captures[i][-1]["evidence_digest"], captures[i + 1][0]["evidence_digest"]]
                   for i in range(len(captures) - 1)]
        if derived != result.get("factor_pairs"):
            raise ValueError("factor pairs differ from original pose order")
        pairs.extend(derived)
    ids = [v for pair in pairs for v in pair]
    if len(pairs) > min(64, WorkerBudgets().max_keyframes // 2) or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("bounded source-disjoint acquisition digests required")
    return pairs


def readiness_blockers(settled: SettledProfile, trial: dict[str, Any],
                       batches: list[dict[str, Any]], model: PhysicalModel | None) -> list[str]:
    """Expose incompatibilities without weakening production evidence or solver gates."""
    solve = SolveConfig(**trial["solve"])
    blockers = []
    if settled.receipt_uncertainty_s + settled.exposure_uncertainty_s > solve.max_stamp_uncertainty_s:
        blockers.append("Proposed receipt + exposure uncertainty exceeds the trial stamp gate; obtain defensible bounds before collection, do not relax the gate")
    if batches and model is not None:
        rotations = [model.forward(list(p.values())).rotation_to_deg(model.forward(list(q.values())))
                     for batch in batches for p, q in zip(batch["args"]["poses"], batch["args"]["poses"][1:])]
        if sum(math.radians(r) >= solve.min_rotation_rad for r in rotations) < solve.min_pairs + 1:
            blockers.append("Too few proposed motions survive the solver rotation/sample floors; no automatic envelope expansion")
    if trial["pairs"] is not None and type(trial["training"]) is int and trial["training"] < solve.min_pairs:
        blockers.append("Frozen training split is below the solver minimum pair count")
    return blockers


def prepare(probe: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    if probe.get("schema") != "leapflow.calibration_preflight.v1" or probe.get("remote_changes_performed") is not False:
        raise ValueError("read-only owner preflight export required")
    arm = HardwareContext.from_mapping(probe["arm"])
    cell = probe["workcell"]
    wrist = cell["roles"]["camera.wrist"]
    metadata = probe["camera_metadata"]["readings"][0]["value"]
    serial = metadata.get("serial_number") or metadata.get("serial") or wrist
    mount = MountIdentity(**inputs["mount"]) if "mount" in inputs else MountIdentity(
        f"{wrist}:{serial}:seat-1", "flange", notes="Installation seat identity; last disturbance time unrecorded")
    if mount.parent_frame != "flange":
        raise ValueError("wrist mount parent must be flange")
    settled = SettledProfile(**inputs.get("settled", PROPOSED_SETTLED))
    retention = inputs.get("retention", {"ttl_s": 3600, "quota_bytes": 64 * 1024 * 1024, "max_frames": 16})
    acquisition = CollectionProfile("camera.wrist", "wrist", ("can",), settled, **retention)
    region = inputs.get("region", [160, 120, 320, 240])
    if len(region) != 4 or any(type(v) is not int for v in region) or region[0] < 0 or region[1] < 0 or not (
        8 <= region[2] <= 640 and 8 <= region[3] <= 480 and region[0] + region[2] <= 640 and region[1] + region[3] <= 480
    ):
        raise ValueError("bounded integer RGB-D region required")
    missing = []
    for name in ("mount_confirmation", "uncertainty_evidence", "whole_link_geometry_evidence",
                 "rate_bound_evidence", "wrist_prior_evidence"):
        if not inputs.get(name):
            missing.append(name)
    stage_a = {"calibration_acquisition": acquisition.to_dict()}
    trial = {"pairs": inputs.get("pairs"), "training": inputs.get("training"), "mount": mount.to_dict(),
             "prior": inputs.get("prior"), "prior_uncertainty": inputs.get("prior_uncertainty"),
             "tracking": {}, "solve": {}, "evaluation": {}}
    if trial["pairs"] is not None and trial["training"] is not None and trial["prior"] is not None and trial["prior_uncertainty"] is not None:
        trial = TrialPolicy.from_mapping(trial, WorkerBudgets()).to_dict()
    motion_names = ("geometry", "joint_rate_bounds_rad_s", "endpoint_rate_bound_mm_s",
                    "rate_provenance_digest", "reference_acquisition", "reference_joints_deg", "tool_transform")
    absent = [name for name in motion_names if inputs.get(name) is None]
    missing.extend(absent)
    policy = model = None
    batches: list[dict[str, Any]] = []
    terms = None
    if not absent:
        chain = DhChain.from_mapping(arm.kinematics.dh)
        model = PhysicalModel(chain, (0.0,) * chain.dof, Transform.from_mapping(inputs["tool_transform"]),
                              digest_of(chain.to_dict()), digest_of(cell), "nominal_draft_not_qualified")
        policy = PoseCollectionPolicy(inputs["reference_acquisition"], mount,
            SafetyGeometry.from_mapping(inputs["geometry"]), tuple(inputs["joint_rate_bounds_rad_s"]),
            inputs["endpoint_rate_bound_mm_s"], inputs["rate_provenance_digest"],
            speed_pct_max=inputs.get("speed_pct_max", 1))
        limits = cell["limits"]
        policy.validate_model(model, settled, fk_mm=limits["fk_tolerance_mm"], fk_deg=limits["fk_tolerance_deg"])
        if arm.safety and arm.safety.max_velocity_rad_s > 0 and any(r > arm.safety.max_velocity_rad_s for r in policy.joint_rate_bounds_rad_s):
            raise ValueError("rate bounds exceed live device safety")
        terms = inflation_requirements(model, policy.geometry, settled, list(policy.joint_rate_bounds_rad_s),
            policy.endpoint_rate_bound_mm_s, fk_mm=limits["fk_tolerance_mm"], fk_deg=limits["fk_tolerance_deg"])
        batches = pose_batches(model, policy, acquisition, arm, inputs["reference_joints_deg"],
            move_timeout_s=limits["move_timeout_s"], run_timeout_s=inputs.get("run_timeout_s", 120.0), region=region,
            step_deg=inputs.get("pose_step_deg", 1.0))
    solve = SolveConfig(**trial["solve"])
    stamp_bound = settled.receipt_uncertainty_s + settled.exposure_uncertainty_s
    readiness = readiness_blockers(settled, trial, batches, model)
    missing.extend(name for name in ("pairs", "training", "prior", "prior_uncertainty") if trial[name] is None)
    return {"schema": "leapflow.camera_preparation.v1", "state": "draft_requires_evidence_and_approval",
            "workcell": cell["workcell"], "node_id": probe["node_id"], "stage_a_overlay": stage_a,
            "device_config_patch": {"device_id": wrist, "transport.config.mount_digest": mount.digest},
            "runtime_config_patch": {"hardware.calibration_worker_enabled": True},
            "mount": mount.to_dict(), "collect_args": {"region": region, "frames": 1}, "trial_args": trial,
            "stage_b_overlay": {"calibration_motion": policy.to_dict()} if policy else None,
            "pose_batches": batches, "inflation_requirements": terms, "unresolved": missing,
            "readiness_blockers": readiness, "trial_stamp_bound_s": stamp_bound,
            "trial_stamp_gate_s": solve.max_stamp_uncertainty_s,
            "deferred_inputs_template": {**dict.fromkeys(motion_names),
                **dict.fromkeys(("mount_confirmation", "uncertainty_evidence", "whole_link_geometry_evidence",
                                "rate_bound_evidence", "wrist_prior_evidence", "prior", "prior_uncertainty")),
                "settled": acquisition.to_dict()["settled"], "mount": mount.to_dict(),
                "pose_step_deg": 1.0, "run_timeout_s": 120.0},
            "run_timeout_assumption_requires_node_preview": "run_timeout_s" not in inputs,
            "proposed_settled_not_measured": "settled" not in inputs,
            "not_human_work": ["device IDs/streams/DH/limits", "profile paths/overlay merge", "mount digest encoding",
                "same-session indexed anchor", "rate and uncertainty evidence collection after authorization",
                "pose targets, source-disjoint split and exact approval previews"],
            "human_confirmation": ["Rigid unchanged wrist seat; two scaled orthogonal views of camera/bracket/flange axes",
                "Current clear work volume/table/obstacles; remove chair or include it in the measured keepouts",
                "Static textured target in wrist view; empty jaws and accessible emergency stop with onsite supervision"],
            "approval_sequence": ["Backed-up exact source/tool deployment, config patches and one owner restart before anchor",
                "Separately approved enable/reference positioning/speed setup; no blind return to stopped feedback",
                "Stationary collection, exact Stage B config write then same-runtime skill.reload",
                "Fresh bounded pose run approvals, then leased READ shadow trial; no activation"],
            "physical_qualification": False, "live_configuration_written": False, "motion_performed": False}


def write_preparation(output: Path, result: dict[str, Any], sources: list[Path]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, key in (("stage-a.overlay.draft.yaml", "stage_a_overlay"), ("stage-b.overlay.draft.yaml", "stage_b_overlay")):
        if result[key] is not None:
            (output / name).write_text("# DRAFT ONLY: partial overlay; merge into existing declaration only after approval.\n" +
                                      yaml.safe_dump(result[key], sort_keys=False))
    for name, key in (("collect.args.json", "collect_args"), ("trial.args.draft.json", "trial_args"),
                      ("deferred-inputs.template.json", "deferred_inputs_template")):
        (output / name).write_text(json.dumps(result[key], indent=2, allow_nan=False) + "\n")
    for index, batch in enumerate(result["pose_batches"]):
        (output / f"poses-{index + 1}.args.json").write_text(json.dumps(batch["args"], indent=2) + "\n")
    result["source_files"] = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]
    result["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output / "preparation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="explicit independent evidence; omit for unresolved drafts")
    parser.add_argument("--output", type=Path, required=True, help="new local directory")
    parser.add_argument("--pose-result", type=Path, action="append", default=[],
                        help="successful redacted piper pose-run result; repeat in acquisition order to freeze trial split")
    parser.add_argument("--training", type=int, help="training pair count fixed before any tracking")
    args = parser.parse_args()
    probe = load_json(args.probe)
    inputs = load_json(args.inputs) if args.inputs else {}
    if args.pose_result:
        if not inputs.get("mount") or type(args.training) is not int:
            raise ValueError("--pose-result requires explicit --training and the confirmed mount input")
        inputs["pairs"] = trial_pairs([load_json(path) for path in args.pose_result],
            node_id=probe["node_id"], workcell=probe["workcell"]["workcell"], mount=inputs["mount"])
        inputs["training"] = args.training
    elif args.training is not None:
        raise ValueError("--training requires actual --pose-result files")
    result = prepare(probe, inputs)
    write_preparation(args.output, result, [args.probe] + ([args.inputs] if args.inputs else []) + args.pose_result)
    print(json.dumps({"output": str(args.output), "state": result["state"], "unresolved": result["unresolved"]}, indent=2))


if __name__ == "__main__":
    main()

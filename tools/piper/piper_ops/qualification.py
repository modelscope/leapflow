"""Independent jaw metrology and fail-closed, read-only Phase C/D reporting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from leapflow.hardware.calibration_qualification import (
    CalibrationQualification,
    QualificationLimits,
)
from leapflow.hardware.grasp_evidence import EvidenceIdentity, GraspEvidence, GraspLimits
from leapflow.hardware.grasp_measurement import number
from leapflow.hardware.skills.digest import run_permit_digest, verify_wire_plan
from leapflow.hardware.skills.tuning import DEFAULT_GRASP_VERIFY

from . import calibration, health, motion
from .evidence import Refusal
from .session import Session


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise Refusal("qualification evidence exceeds byte budget")
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise Refusal("qualification evidence is unreadable or malformed") from error
    if not isinstance(data, dict):
        raise Refusal("qualification evidence must be an object")
    return data


def jaw_certificate(path: Path) -> dict[str, Any]:
    """Fit feedback-to-caliper stroke only; held-out rows cannot train the fit."""
    data = load(path)
    if (
        data.get("physical") is not True
        or not data.get("independent_reference")
        or not data.get("mount_digest")
    ):
        raise Refusal("independent physical caliper evidence and exact mount identity required")
    uncertainty = number(data["uncertainty_mm"], positive=True)
    threshold = DEFAULT_GRASP_VERIFY["jaw_stroke_error_mm"]
    training, holdout = data["training"], data["holdout"]
    if any(
        not isinstance(rows, list) or not 3 <= len(rows) <= 1000 for rows in (training, holdout)
    ):
        raise Refusal("three bounded training and independent held-out caliper rows required")
    observations = set()
    identifiers = set()
    acquisition_scope = None
    for row in (*training, *holdout):
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise Refusal("metrology observation IDs must be unique and disjoint")
        identifiers.add(identifier)
        acquisition = (path.parent / row["acquisition_file"]).resolve()
        report = load(acquisition)
        if (
            report.get("ok") is not True
            or report.get("executed") is not True
            or report.get("operation") != "gripper_qualify"
            or report.get("side_effect_state") != "committed"
        ):
            raise Refusal("caliper row requires a completed governed physical jaw acquisition")
        sample = row["sample"]
        if type(sample) is not int or sample < 0:
            raise Refusal("caliper row requires an acquired sample index")
        if not acquisition_provenance(acquisition, report):
            raise Refusal(
                "jaw acquisition requires matching declaration, approval and permit audit evidence"
            )
        current_scope = (report["node_id"], report["approved_preview"]["declaration_digest"])
        if acquisition_scope is not None and acquisition_scope != current_scope:
            raise Refusal("jaw training and holdout must bind the same node and arm declaration")
        acquisition_scope = current_scope
        key = (report["node_id"], report["run_id"], sample)
        if key in observations:
            raise Refusal("training and holdout cannot reuse the same acquired jaw observation")
        observations.add(key)
        rows = report.get("result", {}).get("samples", [])
        if sample >= len(rows) or rows[sample].get("blocked") is not False:
            raise Refusal("caliper row has no unblocked settled acquisition")
        for field in ("command_mm", "feedback_mm"):
            if number(row[field]) != number(rows[sample][field]):
                raise Refusal("caliper row differs from acquired command or feedback")
        if number(row["caliper_mm"]) < 0:
            raise Refusal("nonnegative caliper measurements required")
        if abs(row["command_mm"] - row["caliper_mm"]) + uncertainty > threshold:
            raise Refusal("command-to-caliper stroke exceeds 1.5 mm acceptance")
    xs = [number(r["feedback_mm"]) for r in training]
    ys = [number(r["caliper_mm"]) for r in training]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread <= 0 or len(set(xs)) < 3:
        raise Refusal("jaw training requires three distinct measured widths")
    scale = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    offset = mean_y - scale * mean_x
    if scale <= 0:
        raise Refusal("jaw stroke fit must be monotonic")
    if any(not min(xs) <= number(r["feedback_mm"]) <= max(xs) for r in holdout):
        raise Refusal("held-out widths cannot extrapolate the characterized stroke")
    errors = tuple(
        abs(scale * number(r["feedback_mm"]) + offset - number(r["caliper_mm"])) for r in holdout
    )
    geometry = {
        "scale": scale,
        "offset_mm": offset,
        "min_width_mm": min(xs),
        "max_width_mm": max(xs),
    }
    certificate = CalibrationQualification(
        "jaw_stroke",
        data.get("version", "1"),
        data["mount_digest"],
        "mm",
        "Rz@Ry@Rx;degrees",
        tuple(r["id"] for r in training),
        (),
        tuple(r["id"] for r in holdout),
        errors,
        uncertainty,
        data["independent_reference"],
        True,
        QualificationLimits(3, 3, threshold, threshold, threshold),
    ).bind_geometry(geometry)
    if not certificate.qualified:
        raise Refusal("held-out caliper errors exceed jaw-stroke acceptance")
    return {
        "ok": True,
        "offline": True,
        "stored": False,
        "qualified": certificate.qualified,
        "component": {"geometry": geometry, "qualification": asdict(certificate)},
    }


def acquisition_provenance(path: Path, report: dict[str, Any]) -> bool:
    """Cross-check durable operations artifacts; live node outcomes remain authoritative."""
    try:
        plan = report["plan"]
        node, workcell, skill = report["node_id"], report["workcell"], report["skill"]
        local_cell = workcell.removeprefix(node + "/")
        if workcell != node + "/" + local_cell or not local_cell:
            return False
        permit_digest = run_permit_digest(
            workcell=local_cell,
            skill_id=skill,
            version=plan["version"],
            plan_digest=plan["digest"],
            limits=plan["limits"],
        )
        if (
            verify_wire_plan(plan, permit_digest)
            or plan["skill_id"] != skill
            or plan["workcell"] != local_cell
            or not report.get("approved_plan_hash")
        ):
            return False
        approved = report["approved_preview"]
        if (
            motion.digest_plan(approved) != report["approved_plan_hash"]
            or approved["skill_plan"] != plan
            or approved["node_id"] != node
            or approved["workcell"] != local_cell
            or approved["skill_id"] != skill
            or approved["plan_digest"] != plan["digest"]
        ):
            return False
        events = report["events"]
        if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
            return False
        lifecycle = [e for e in events if e.get("event") in (
            "connected", "calibration_consent", "qualification_result",
            "after_skill_client_exit", "after_client_exit")]
        if [e["event"] for e in lifecycle] != [
            "connected", "calibration_consent", "qualification_result",
            "after_skill_client_exit", "connected", "after_client_exit",
        ]:
            return False
        executing, consent, outcome, settled, readback, verified = lifecycle
        if (
            executing.get("execute") is not True
            or readback.get("execute") is not False
            or any(e.get("node_id") != node or e.get("write_mode") != "permit_required"
                   or e.get("declaration_digest") != approved["declaration_digest"]
                   for e in (executing, readback))
            or consent.get("approved") is not True
            or consent.get("binding") != {k: approved[k] for k in
                                         ("node_id", "workcell", "skill_id", "plan_digest")}
            or outcome.get("result", {}).get("run_id") != report["run_id"]
            or outcome["result"].get("plan") != plan
            or outcome["result"].get("result") != report.get("result")
            or outcome["result"].get("side_effect_state") != "committed"
            or outcome["result"].get("ok") is not True
            or not isinstance(settled.get("snapshot"), dict)
            or not settled["snapshot"]
            or report.get("post_exit_verified") is not True
            or not isinstance(verified.get("snapshot"), dict)
            or not verified["snapshot"]
            or verified["snapshot"] != report.get("post_exit_snapshot")
            or any(e.get("event") in ("cleanup_failure", "damped_stop", "post_exit_halt_failure")
                   for e in events)
        ):
            return False
        audit = path.with_suffix(".audit.ndjson")
        if audit.is_symlink() or not audit.is_file() or audit.stat().st_size > 1024 * 1024:
            return False
        rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
        started = sum(
            r.get("action") == "skill_run"
            and r.get("device") == workcell
            and r.get("outcome") == "started " + report["run_id"]
            for r in rows
        )
        permit = sum(
            r.get("action") == "skill_permit"
            and r.get("device") == workcell
            and r.get("value", {}).get("run_id") == report["run_id"]
            and r["value"].get("plan_digest") == plan["digest"]
            and r["value"].get("request_digest") == permit_digest
            and r["value"].get("audience") == node
            and bool(r["value"].get("approval_id"))
            for r in rows
        )
        return started == permit == 1
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _identity(data: dict[str, Any]) -> EvidenceIdentity:
    identity = EvidenceIdentity(**data)
    number(identity.observed_mono)
    if not identity.current(identity, now=identity.observed_mono, max_age_s=1):
        raise ValueError("malformed observation identity")
    if any(not i.startswith(identity.session_id + ":") for i in identity.frame_ids):
        raise ValueError("frame identity differs from camera session")
    return identity


def _evidence(data: dict[str, Any], *, proof: bool = False) -> GraspEvidence:
    values = dict(data)
    values["identity"] = _identity(values["identity"])
    nested = values.get("proof")
    if nested is not None:
        if proof:
            raise ValueError("recursive proof evidence")
        values["proof"] = _evidence(nested, proof=True)
    evidence = GraspEvidence(**values)
    for key in ("object_rise_mm", "tool_rise_mm", "relative_error_mm",
                "width_change_mm", "uncertainty_mm"):
        number(getattr(evidence, key))
    return evidence


def cycle_accepted(run: dict[str, Any], qualification_digest: str,
                   tuning: dict[str, Any] | None = None) -> bool:
    """Recompute each verdict and distinct source progression, allowing rolling history."""
    try:
        settings = DEFAULT_GRASP_VERIFY if tuning is None else tuning
        limits = GraspLimits(**{k: int(settings[k]) if k == "min_frames" else settings[k]
                                for k in GraspLimits.__dataclass_fields__})
        result = run["result"]
        stages = result["stages"]
        names = [s["stage"] for s in stages]
        if (run.get("status") != "succeeded"
                or run.get("side_effect_state") != "committed"
                or result.get("cycle_passed") is not True
                or not qualification_digest
                or result.get("qualification_digest") != qualification_digest
                or not isinstance(result.get("placement"), dict)
                or not result["placement"]
                or not 6 <= len(stages) <= 1000
                or names[:3] != ["pre_grasp", "contact", "proof"]
                or "carry" not in names[3:]
                or names[-1] != "place"
                or any(n not in ("carry", "place") for n in names[3:])
                or "carry" in names[names.index("place"):]):
            return False
        scope, last_at, seen = None, float("-inf"), {}
        proofs = set()
        for stage in stages:
            verification = stage["verification"]
            observations = verification["observations"]
            expected = "pre_grasp" if stage["stage"] == "pre_grasp" else "hold"
            if (verification.get("passed") is not True
                    or not isinstance(observations, list)
                    or not max(3, limits.min_frames) <= len(observations) <= 10
                    or any(verification.get(k) != observations[-1].get(k)
                           for k in ("stage", "state", "qualification_digest", "evidence",
                                     "source_identity", "alignment_tool_mm"))):
                return False
            for observation in observations:
                source = _identity(observation["source_identity"])
                evidence = _evidence(observation["evidence"])
                current_scope = (source.session_id, source.scene_id, source.calibration_digest)
                if scope is None:
                    scope = current_scope
                if (scope != current_scope or source.calibration_digest != qualification_digest
                        or len(source.frame_ids) != 1
                        or source.frame_ids[0] in seen
                        or not last_at < source.observed_mono
                        or (last_at != float("-inf") and stage["stage"] not in ("pre_grasp", "contact")
                            and source.observed_mono - last_at > limits.max_age_s)
                        or observation.get("passed") is not True
                        or observation.get("stage") != expected
                        or observation.get("qualification_digest") != qualification_digest
                        or not evidence.identity.current(source, now=source.observed_mono,
                                                         max_age_s=limits.max_age_s)
                        or evidence.identity.observed_mono != source.observed_mono
                        or evidence.identity.frame_ids[-1] != source.frame_ids[0]):
                    return False
                seen[source.frame_ids[0]] = source.observed_mono
                last_at = source.observed_mono
                times = evidence.observation_times
                if times and (len(times) != len(evidence.identity.frame_ids)
                              or list(times) != [seen.get(i) for i in evidence.identity.frame_ids]):
                    return False
                if evidence.proof is not None and motion.digest_plan(evidence.proof.to_dict()) not in proofs:
                    return False
                measured = evidence.state(source, now=source.observed_mono, limits=limits).value
                if measured == "held_verified" and evidence.proof is None:
                    proofs.add(motion.digest_plan(evidence.to_dict()))
                if measured != observation.get("state"):
                    return False
                if expected == "pre_grasp":
                    if (evidence.visible is not True or evidence.synchronized is not True
                            or evidence.independent_jaws is not True
                            or evidence.width_consistent is not True
                            or abs(number(observation["alignment_tool_mm"]))
                               + evidence.uncertainty_mm > settings["alignment_mm"]):
                        return False
                elif measured not in ("contact_candidate", "held_verified"):
                    return False
            if (stage["stage"] == "contact" and verification["state"] != "contact_candidate"
                    or stage["stage"] in ("proof", "carry")
                    and verification["state"] != "held_verified"):
                return False
        return stages[-1]["verification"]["evidence"]["object_on_support"] is True
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False


async def report(session: Session, workcell: str) -> dict[str, Any]:
    """Use live node outcomes, not caller success flags or synthetic report files."""
    async with calibration.service(session) as service:
        cells = await service.hw_skill_list(node=session.settings.node_id)
        inspection = await service.hw_skill_run(
            workcell=f"{session.settings.node_id}/{workcell}",
            skill="grasp.inspect",
            args={},
            dry_run=True,
        )
        runs = await service.hw_skill_status(node=session.settings.node_id)
    blockers = []
    try:
        snapshot = (await session.snapshot(ready=True)).to_dict()
    except Refusal as error:
        snapshot = {"qualified": False, "reason": str(error)}
        blockers.append("qualified ready arm state unavailable")
    passive = await health.passive_can(session)
    if passive.get("qualified") is not True:
        blockers.append("qualified node-local passive CAN evidence required")
    if any(not isinstance(value, dict) for value in (cells, inspection, runs)):
        cells, inspection, runs = {}, {}, {}
        blockers.append("malformed live node response")
    artifacts = {}
    conflicts = set()
    paths = sorted((session.settings.root / "data/runs").glob("*.json"))
    if len(paths) > 1000:
        blockers.append("run evidence inventory exceeds bounded report budget")
    else:
        for path in paths:
            try:
                artifact = load(path)
                if (artifact.get("operation") == "grasp_cycle"
                        and artifact.get("node_id") == session.settings.node_id
                        and artifact.get("workcell") == f"{session.settings.node_id}/{workcell}"
                        and acquisition_provenance(path, artifact)):
                    run_id = artifact["run_id"]
                    if run_id in conflicts:
                        continue
                    if run_id in artifacts and artifacts[run_id] != artifact:
                        blockers.append("conflicting durable artifacts for one node run")
                        conflicts.add(run_id)
                        artifacts[run_id] = {}
                    else:
                        artifacts[run_id] = artifact
            except (OSError, ValueError, Refusal):
                continue
    evidence = inspection.get("result", {})
    if (not isinstance(evidence, dict)
            or not isinstance(evidence.get("qualification", {}), dict)
            or not isinstance(evidence.get("depth_quality", {}), dict)):
        evidence = {}
        blockers.append("malformed live qualification inspection")
    scope = evidence.get("qualification", {}).get("qualification_digest", "")
    if not isinstance(scope, str):
        scope = ""
    consecutive = 0
    accepted = []
    # Runtime returns insertion-ordered node runs; failed/cancelled/in-flight
    # cycles reset continuity. Distinct run IDs cannot be replayed into a streak.
    seen, seen_frames = set(), set()
    live_runs = runs.get("runs", [])
    if not isinstance(live_runs, list):
        live_runs = []
        blockers.append("malformed live run inventory")
    for run in live_runs:
        if not isinstance(run, dict):
            consecutive = 0
            blockers.append("malformed live run")
            continue
        if (
            run.get("skill_id") != "grasp.cycle"
            or run.get("workcell") != f"{session.settings.node_id}/{workcell}"
        ):
            continue
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            consecutive = 0
            blockers.append("malformed live run identity")
            continue
        artifact = artifacts.get(run_id, {})
        passed = bool(
            run_id
            and run_id not in seen
            and scope
            and cycle_accepted(run, scope, artifact.get("plan", {}).get("basis", {}).get("grasp_verify", {}))
            and artifact.get("node_id") == session.settings.node_id
            and run.get("plan_digest") == artifact.get("plan", {}).get("digest")
            and run.get("version") == artifact.get("plan", {}).get("version")
            and artifact.get("result") == run.get("result")
            and artifact.get("side_effect_state") == run.get("side_effect_state")
            and artifact.get("post_exit_verified") is True
        )
        if passed:
            frames = { (o["source_identity"]["session_id"], frame)
                      for s in run["result"]["stages"]
                      for o in s["verification"]["observations"]
                      for frame in o["source_identity"]["frame_ids"] }
            passed = not bool(frames & seen_frames)
            seen_frames.update(frames)
        seen.add(run_id)
        consecutive = consecutive + 1 if passed else 0
        accepted.append({"run_id": run_id, "passed": passed})
    depth = evidence.get("depth_quality", {})
    p95 = depth.get("temporal_p95_mm")
    if (
        inspection.get("ok") is not True
        or cells.get("ok") is not True
        or runs.get("ok") is not True
        or cells.get("unavailable")
        or runs.get("unavailable")
        or cells.get("rejected")
    ):
        blockers.append("live node/workcell inspection unavailable or rejected")
    if (
        depth.get("qualified") is not True
        or not isinstance(depth.get("frame_ids"), (list, tuple))
        or any(not isinstance(frame, str) for frame in depth["frame_ids"])
        or len(set(depth["frame_ids"])) < 20
        or isinstance(p95, bool)
        or not isinstance(p95, (int, float))
        or not math.isfinite(p95)
        or not 0 <= p95 < 1
    ):
        blockers.append(
            "fresh qualified 20-frame depth window with temporal P95 below 1 mm required"
        )
    if consecutive < 3:
        blockers.append(
            "three consecutive real node cycles with governed durable evidence required"
        )
    return {
        "ok": True,
        "read_only": True,
        "phase": "C/D",
        "qualified": not blockers,
        "consecutive_successes": consecutive,
        "cycles": accepted,
        "node": cells,
        "arm": snapshot,
        "inspection": inspection,
        "passive_can_health": passive,
        "qualification_blockers": blockers,
    }

"""Governed workcell operations: offline previews and one cached-plan execution."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import calibration, motion
from .evidence import Refusal
from .qualification import load
from .session import Session, unverified_digest


def preview(session: Session, workcell: str, skill: str, arguments: dict[str, Any],
            result: dict[str, Any], operation: str) -> dict[str, Any]:
    if result.get("ok") is not True or result.get("dry_run") is not True:
        return result
    plan = {"version": 1, "operation": operation, "settings": asdict(session.settings),
            "declaration_digest": unverified_digest(session.context),
            "node_id": session.settings.node_id, "workcell": workcell, "skill_id": skill,
            "args": result["plan"]["args"], "skill_plan": result["plan"],
            "plan_digest": result["plan"]["digest"]}
    return {"ok": True, "executed": False, "preview": plan, "plan_hash": motion.digest_plan(plan)}


async def run(session: Session, *, workcell: str, skill: str, arguments: dict[str, Any],
              operation: str, approved: dict[str, Any] | None = None) -> dict[str, Any]:
    if skill not in ("grasp.cycle", "gripper.qualify"):
        raise Refusal("unsupported qualification skill")
    if approved and (approved.get("args") != arguments or approved.get("workcell") != workcell
                     or approved.get("skill_id") != skill or approved.get("node_id") != session.settings.node_id
                     or not isinstance(approved.get("skill_plan"), dict)):
        raise Refusal("skill request differs from approved cached preview")
    async with calibration.service(session, approved) as service:
        result = await service.hw_skill_run(
            workcell=f"{session.settings.node_id}/{workcell}", skill=skill, args=arguments,
            dry_run=approved is None, approved_plan=approved["skill_plan"] if approved else None,
            timeout_s=session.settings.run_timeout_s,
        )
        session.evidence.record("qualification_result", result=result)
        if approved is not None:
            session.touched |= result.get("side_effect_state", "unknown") != "none"
        session.success = result.get("ok") is True
    if approved is not None:
        if result.get("ok") is True:
            current = await session.snapshot(ready=True)
            session.evidence.record("after_skill_client_exit", snapshot=current.to_dict())
        return {**result, "executed": True, "operation": operation,
                "node_id": session.settings.node_id, "approved_preview": approved,
                "approved_plan_hash": motion.digest_plan(approved)}
    return preview(session, workcell, skill, arguments, result, operation)


async def _read(service: Any, session: Session, workcell: str, skill: str,
                args: dict[str, Any]) -> dict[str, Any]:
    result = await service.hw_skill_run(workcell=f"{session.settings.node_id}/{workcell}",
                                       skill=skill, args=args, dry_run=True,
                                       timeout_s=session.settings.run_timeout_s)
    session.evidence.record("grasp_preview_observation", skill=skill, result=result)
    if result.get("ok") is not True or result.get("side_effect_state") != "none":
        raise Refusal(f"read-only {skill} failed; inspect preview evidence")
    return result


async def snapshot(session: Session, workcell: str, *, share_frames: bool) -> dict[str, Any]:
    """Capture first; the operator confirms this exact saved image in a later preview."""
    if not share_frames:
        raise Refusal("grasp snapshot requires explicit --share-frames")
    async with calibration.service(session, share_frames=True) as service:
        result = await _read(service, session, workcell, "perceive.snapshot", {})
    images = result.get("_attachments", [])
    shot = dict(result["result"])
    if len(images) != 1 or images[0].get("mime") != "image/jpeg" or not shot.get("capture_id"):
        raise Refusal("snapshot requires one shared image and a node-bound capture identity")
    shot["image"] = {k: images[0][k] for k in ("mime", "data")}
    binding = {"node_id": session.settings.node_id, "workcell": workcell,
               "declaration_digest": unverified_digest(session.context), "snapshot": shot}
    return {"ok": True, "executed": False, "operation": "grasp_snapshot",
            **binding, "snapshot_hash": motion.digest_plan(binding)}


async def prepare(session: Session, workcell: str, arguments: dict[str, Any],
                  *, share_frames: bool) -> dict[str, Any]:
    """Operator supplies one box and a candidate index; no automatic target selection."""
    if not share_frames:
        raise Refusal("grasp preview requires explicit --share-frames and workcell frame-sharing consent")
    target = arguments.get("target")
    index = arguments.get("grasp_index")
    if (not isinstance(target, dict) or "box" not in target
            or type(index) is not int or not 0 <= index < 10):
        raise Refusal("args require one operator-confirmed target box and explicit grasp_index (0–9)")
    if arguments.get("target_confirmed") is not True or not arguments.get("snapshot_file"):
        raise Refusal("target confirmation must bind a saved grasp snapshot; capture and inspect it first")
    path = Path(arguments["snapshot_file"])
    if not path.is_absolute():
        raise Refusal("snapshot_file must be an absolute saved evidence path")
    saved = load(path)
    binding = {k: saved[k] for k in ("node_id", "workcell", "declaration_digest", "snapshot")}
    snapshot_hash = motion.digest_plan(binding)
    if (saved.get("ok") is not True or saved.get("executed") is not False
            or saved.get("operation") != "grasp_snapshot"
            or saved.get("snapshot_hash") != snapshot_hash
            or arguments.get("confirmed_snapshot_hash") != snapshot_hash
            or saved["node_id"] != session.settings.node_id or saved["workcell"] != workcell
            or saved["declaration_digest"] != unverified_digest(session.context)):
        raise Refusal("confirmed snapshot hash, node, workcell or declaration differs")
    shot = saved["snapshot"]
    if not isinstance(shot.get("capture_id"), str) or not shot["capture_id"]:
        raise Refusal("snapshot has no node-bound capture identity")
    qualified = f"{session.settings.node_id}/{workcell}"
    async with calibration.service(session, share_frames=True) as service:
        async def read(skill: str, args: dict[str, Any]) -> dict[str, Any]:
            return (await _read(service, session, workcell, skill, args))["result"]

        lifted = await read("perceive.lift", {"frame": shot["frame"], "capture_id": shot["capture_id"],
                                            "targets": [target], "coords": arguments.get("coords", "pixel")})
        objects = lifted.get("objects", [])
        if len(objects) != 1 or lifted.get("dropped"):
            raise Refusal("confirmed target did not produce exactly one metric object")
        proposed = await read("grasp.propose", {"object": objects[0]["object_id"], "top_k": 10})
        candidates = proposed.get("grasps", [])
        if index >= len(candidates):
            raise Refusal("operator-selected grasp index is absent; inspect candidates and preview again")
        cycle_args = {"grasp_id": candidates[index]["grasp_id"], "place": arguments["place"]}
        for key in ("proof_lift_mm", "samples", "speed_pct"):
            if key in arguments:
                cycle_args[key] = arguments[key]
        result = await service.hw_skill_run(workcell=qualified, skill="grasp.cycle", args=cycle_args,
                                            dry_run=True, timeout_s=session.settings.run_timeout_s)
        session.evidence.record("grasp_cycle_preview", result=result)
        output = preview(session, workcell, "grasp.cycle", cycle_args, result, "grasp_cycle")
        if output.get("ok") is True:
            output["preview"]["operator_confirmation"] = {
                "snapshot_hash": snapshot_hash, "frame": shot["frame"], "capture_id": shot["capture_id"],
                "target": target, "coords": arguments.get("coords", "pixel"), "grasp_index": index,
            }
            output["plan_hash"] = motion.digest_plan(output["preview"])
        return {**output, "candidates": candidates, "selected_grasp_index": index}

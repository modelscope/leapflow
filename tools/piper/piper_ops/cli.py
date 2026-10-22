"""Unified dry-run-first operational CLI, with explicit preview-bound execution."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from leapflow.engine.recovery.failure_envelope import SideEffectState

from . import calibration, layout, lifecycle, motion, node
from .config import Settings
from .evidence import Evidence, Refusal, encode
from .session import Session, complete_cleanup, unverified_digest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", type=Path)
    for option in ("endpoint", "node-id", "device-id"):
        root.add_argument("--" + option)
    for option in ("token-file", "issuer-key", "root"):
        root.add_argument("--" + option, type=Path)
    root.add_argument("--json", type=Path, help="New evidence path; earlier attempts cannot be overwritten")
    root.add_argument("--summary", action="store_true", help="Print only the result status and evidence path")
    groups = root.add_subparsers(dest="group", required=True)

    def command(group: Any, name: str, *, writing: bool = False) -> Any:
        item = group.add_parser(name)
        if writing:
            item.add_argument("--execute", action="store_true")
            item.add_argument("--approval-note", default="")
            item.add_argument("--preview-file", type=Path)
            item.add_argument("--approved-plan-hash", default="")
            item.add_argument("--approved-digest", default="")
            item.add_argument("--clearance-confirmed", action="store_true")
        return item

    arm = groups.add_parser("arm").add_subparsers(dest="operation", required=True)
    command(arm, "status")
    for name in ("enable", "resume", "stop", "disable", "initialize", "move"):
        item = command(arm, name, writing=True)
        if name == "disable":
            item.add_argument("--torque-release-confirmed", action="store_true")
        if name == "move":
            for axis in "xyz":
                item.add_argument(f"--d{axis}-mm", type=float, default=0.0)
            item.add_argument("--frame", choices=("base",), required=True)
    grip = groups.add_parser("gripper").add_subparsers(dest="operation", required=True)
    item = command(grip, "open", writing=True)
    width = item.add_mutually_exclusive_group(required=True)
    width.add_argument("--max", action="store_true")
    width.add_argument("--width-mm", type=float)
    verify = groups.add_parser("verify").add_subparsers(dest="operation", required=True)
    item = command(verify, "motion", writing=True)
    item.add_argument("--dz-mm", type=float, default=100.0)
    item.add_argument("--dy-mm", type=float, default=50.0)
    item.add_argument("--return", dest="return_to_origin", action="store_true", required=True)
    item.add_argument("--finish", choices=("enabled",), required=True)
    service = groups.add_parser("node").add_subparsers(dest="operation", required=True)
    command(service, "status")
    for name in ("start", "restart"):
        command(service, name, writing=True)
    files = groups.add_parser("layout").add_subparsers(dest="operation", required=True)
    item = command(files, "inspect")
    item.add_argument("--reference-root", type=Path, action="append", default=[],
                      help="Explicit external file/directory to inspect for path references; no content is printed")
    item.add_argument("--reference-max-bytes", type=int, default=2_000_000)
    item.add_argument("--processes", action="store_true", help="Inspect same-user /proc references without argv or environment disclosure")
    item = command(files, "backup", writing=True)
    item.add_argument("--destination", type=Path, required=True)
    item = command(files, "apply", writing=True)
    item.add_argument("--manifest", type=Path, required=True)
    cal = groups.add_parser("calibration").add_subparsers(dest="operation", required=True)
    command(cal, "inspect")
    item = command(cal, "validate")
    item.add_argument("--package", type=Path, required=True)
    item = command(cal, "run", writing=True)
    item.add_argument("--workcell", required=True)
    item.add_argument("--skill", required=True)
    item.add_argument("--args-file", type=Path, required=True)
    return root


def approved(args: argparse.Namespace, settings: Settings, operation: str) -> dict[str, Any]:
    if not args.approval_note or args.preview_file is None or not args.approved_plan_hash:
        raise Refusal("execute requires --preview-file, --approved-plan-hash and recorded --approval-note")
    report = json.loads(args.preview_file.read_text())
    plan = report.get("preview")
    if report.get("ok") is not True or report.get("executed") is not False or not isinstance(plan, dict):
        raise Refusal("a successful complete preview is required")
    if motion.digest_plan(plan) != args.approved_plan_hash or plan.get("operation") != operation:
        raise Refusal("preview hash or command differs from operator approval")
    if motion.digest_plan(plan.get("settings", {})) != motion.digest_plan(asdict(settings)):
        raise Refusal("effective configuration differs from approved preview")
    if args.group in ("arm", "gripper", "verify", "calibration"):
        if not args.approved_digest or args.approved_digest != plan.get("declaration_digest"):
            raise Refusal("exact approved declaration digest required")
        if operation != "stop" and not args.clearance_confirmed:
            raise Refusal("confirm unobstructed swept volume, empty jaws and accessible physical emergency stop")
    claim = args.preview_file.with_suffix(".claimed")
    try:
        with claim.open("x") as stream:
            stream.write(encode({"plan_hash": args.approved_plan_hash, "approval_note": args.approval_note}))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise Refusal("preview already consumed; inspect previous run, do not replay") from exc
    return plan


def preview_result(plan: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "executed": False, "preview": plan, "plan_hash": motion.digest_plan(plan)}


async def hardware(args: argparse.Namespace, settings: Settings, evidence: Evidence) -> dict[str, Any]:
    executing = getattr(args, "execute", False)
    operation = {"verify": "verify", "gripper": "gripper"}.get(args.group, args.operation)
    plan = approved(args, settings, operation) if executing else None
    async with Session(settings, evidence, execute=executing, digest=getattr(args, "approved_digest", ""),
                       approval_note=getattr(args, "approval_note", "")) as session:
        assert session.context is not None
        if args.group == "calibration":
            if args.operation == "inspect":
                return await calibration.run(session)
            arguments = json.loads(args.args_file.read_text())
            result = await calibration.run(session, workcell=args.workcell, skill=args.skill,
                                           args=arguments, approved=plan)
            if executing or result.get("ok") is not True or "plan" not in result:
                return result
            cal_plan = {"version": 1, "operation": "run", "settings": asdict(settings),
                        "declaration_digest": unverified_digest(session.context),
                        "node_id": settings.node_id, "workcell": args.workcell, "skill_id": args.skill,
                        "args": arguments, "plan_digest": result["plan"]["digest"], "skill_plan": result["plan"]}
            return preview_result(cal_plan)
        if args.operation == "status":
            return {"ok": True, "declaration_digest": unverified_digest(session.context),
                    "snapshot": (await session.snapshot()).to_dict()}
        if args.group == "arm" and args.operation in ("enable", "resume", "disable", "stop"):
            if executing:
                return await lifecycle.execute(session, plan, torque_release=getattr(args, "torque_release_confirmed", False))
            return preview_result(await lifecycle.preview(session, operation,
                                 torque_release=getattr(args, "torque_release_confirmed", False)))
        if not executing:
            start = await session.snapshot(preflight=True)
            plan = motion.preview(session.context, start, settings, operation=operation,
                                  dx=getattr(args, "dx_mm", 0.0), dy=getattr(args, "dy_mm", 0.0),
                                  dz=getattr(args, "dz_mm", 0.0), width=getattr(args, "width_mm", None))
            return preview_result(plan)
        # The selected command arguments must still describe the approved artifact.
        delta = plan["delta_base_mm"]
        for axis in "xyz":
            if hasattr(args, f"d{axis}_mm") and getattr(args, f"d{axis}_mm") != delta[axis]:
                raise Refusal("command displacement differs from approved preview")
        if args.group == "gripper" and args.width_mm is not None and args.width_mm != plan["steps"][-1]["grip"]:
            raise Refusal("gripper width differs from approved preview")
        final = await motion.execute_plan(session, plan)
    return await verify_after_exit(settings, evidence, plan, final)


async def halt_after_exit(settings: Settings, evidence: Evidence) -> None:
    """Reconnect only for a production safety halt, never for motion replay."""
    async with Session(settings, evidence) as safety:
        await safety.stop()


async def verify_after_exit(settings: Settings, evidence: Evidence,
                            plan: dict[str, Any], final: motion.Snapshot) -> dict[str, Any]:
    """Read back after client close; any failure inherits the completed run's halt duty."""
    try:
        async with Session(settings, evidence) as readback:
            state = await readback.snapshot(ready=True)
            motion.check_start({"arrival": plan["arrival"], "start": {"joints": final.joints,
                               "pose": asdict(final.pose), "gripper_mm": final.gripper_mm}}, state)
            evidence.record("after_client_exit", snapshot=state.to_dict())
    except BaseException:
        try:
            await complete_cleanup(halt_after_exit(settings, evidence))
        except (Exception, asyncio.CancelledError) as error:
            print("Post-exit halt unconfirmed: use the physical emergency stop and inspect.", file=sys.stderr)
            try:
                evidence.record("post_exit_halt_failure", failure_class=type(error).__name__)
            except OSError:
                pass
        raise
    return {"ok": True, "executed": True, "finish": "enabled", "snapshot": state.to_dict()}


def local(args: argparse.Namespace, settings: Settings, evidence: Evidence) -> dict[str, Any]:
    executing = getattr(args, "execute", False)
    if args.group == "calibration":
        return calibration.validate(args.package, settings.device_id)
    if args.group == "node":
        if not executing:
            return preview_result(node.preview(settings, args.operation))
        plan = approved(args, settings, args.operation)
        return node.execute(settings, plan, evidence)
    if args.operation == "inspect":
        result = {"ok": True, **layout.inventory(settings.root)}
        references = getattr(args, "reference_root", [])
        if references:
            result["references"] = layout.reference_report(settings.root, references,
                max_bytes=getattr(args, "reference_max_bytes", 2_000_000))
        if getattr(args, "processes", False):
            result["processes"] = layout.process_references(settings.root)
        return result
    if args.operation == "backup":
        if not executing:
            return {"ok": True, "executed": False, "root": str(settings.root),
                    "destination": str(args.destination), "private": True}
        if not args.approval_note:
            raise Refusal("private full backup requires recorded --approval-note")
        return layout.backup(settings.root, args.destination)
    manifest = json.loads(args.manifest.read_text())
    if executing:
        plan = approved(args, settings, "apply")
        if plan["manifest"] != manifest:
            raise Refusal("cleanup manifest changed since approval")
    result = layout.apply(settings.root, manifest, execute=executing, evidence=evidence)
    return result if executing else preview_result({"operation": "apply", "settings": asdict(settings), "manifest": manifest})


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    evidence = None
    try:
        settings = Settings.load(args.config, **{key: getattr(args, key) for key in
                                  ("endpoint", "node_id", "device_id", "token_file", "issuer_key", "root")})
        path = args.json or settings.root / "data/runs" / f"{args.group}-{args.operation}-{uuid.uuid4().hex}.json"
        evidence = Evidence(path)
        if args.group in ("node", "layout") or (args.group == "calibration" and args.operation == "validate"):
            result = local(args, settings, evidence)
        else:
            result = asyncio.run(asyncio.wait_for(hardware(args, settings, evidence), settings.run_timeout_s))
        result = evidence.finish(result)
    except (Exception, KeyboardInterrupt) as exc:
        if not isinstance(exc, Refusal):
            internal = isinstance(exc, (TypeError, AttributeError, NameError, KeyError, IndexError, ImportError, AssertionError, NotImplementedError))
            exc = Refusal(f"{'Internal defect' if internal else 'Operation failed'} ({type(exc).__name__}); "
                          "inspect evidence and hardware before continuing",
                          code="internal_defect" if internal else "piper_operations_failed",
                          state=SideEffectState.UNKNOWN if getattr(args, "execute", False) else SideEffectState.NONE)
        result = {"ok": False, "failure": asdict(exc.envelope)}
        if evidence is not None:
            try:
                result = evidence.failure(exc)
            except OSError:
                print("Evidence write failed; preserve the event checkpoint and inspect hardware.", file=sys.stderr)
    if args.summary:
        print(encode({k: v for k, v in {"ok": result.get("ok"), "evidence": str(evidence.path) if evidence else "",
                                       "plan_hash": result.get("plan_hash"), "failure": result.get("failure"),
                                       "archive": result.get("archive"), "manifest": result.get("manifest"),
                                       "sha256": result.get("sha256"), "members": result.get("members")}.items()
                      if v is not None}))
    else:
        print(encode({k: v for k, v in result.items() if k != "events"}))
    return 0 if result.get("ok") is True else 1

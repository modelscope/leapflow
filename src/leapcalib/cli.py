# Copyright (c) Alibaba, Inc. and its affiliates.
"""``leap-calib``: inspect/replay evidence, estimate candidates and propose shadows.

All verbs are offline and compute-only. Optional result files are explicitly
requested private artifacts, never an integrated database or activated model.
The estimate/shadow verbs use concrete discovered methods and bounded inputs;
the private serve command is compute-only, with no acquisition or actuator authority.

Exit codes are meant for scripts: ``0`` success, ``1`` a typed calibration
failure (including a budget that does not close, which is the normal way for
``budget`` to say no), ``2`` argparse usage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Sequence

from leapcalib.contracts.failures import CalibrationFailure, FAILURE_NOT_IMPLEMENTED, FAILURE_SCHEMA
from leapcalib.contracts.methods import discover_methods
from leapcalib.contracts.motion import MotionObservation
from leapcalib.contracts.task import TaskDomain, TaskState, TaskTransition
from leapcalib.contracts.observations import Stamp
from leapcalib.contracts.observations import JointObservation, RgbdObservation
from leapcalib.validation.budget import TaskTolerance, observation_delay_budget
from leapcalib.validation.bundle import DEFAULT_MAX_BYTES, ReplaySource, read_bundle
from leapcalib.validation.result import read_request, result_payload, write_result

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def _emit(payload: dict[str, Any], *, as_json: bool, lines: Sequence[str]) -> None:
    """Print machine-readable JSON or the prepared human lines."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return
    for line in lines:
        print(line)


def _inspect(args: argparse.Namespace) -> int:
    bundle = read_bundle(args.bundle, max_bytes=args.max_bytes)
    summary = bundle.summary
    payload = {
        "summary": summary,
        "topology": bundle.topology.to_dict(),
        "components": bundle.topology.components,
        "retention": bundle.retention.to_dict(),
    }
    lines = [
        f"bundle      {summary['bundle_id']}",
        f"digest      {summary['digest']}",
        f"topology    {summary['topology_digest']}",
        f"workcell    {bundle.topology.workcell_id}",
        f"device      {bundle.topology.device.device_id}",
        f"model       {bundle.topology.model.model_id} "
        f"({bundle.topology.model.dof} dof, {bundle.topology.model.dh_convention})",
        f"observations {summary['observations']} "
        f"(accepted {summary['accepted']}, rejected {summary['rejected']}, "
        f"holdout {summary['holdout']})",
        f"provenance  {', '.join(summary['provenances']) or 'none'}",
        f"clocks      {', '.join(summary['clock_domains']) or 'none'}",
        f"stamps      {', '.join(summary['stamp_semantics']) or 'none'}",
        f"depth valid {summary['depth_valid_ratio']:.3f}",
        f"settled     {summary['settled_fraction']:.3f}",
        f"stamp unc.  {summary['worst_stamp_uncertainty_s'] * 1e3:.1f} ms worst",
        f"retention   ttl {bundle.retention.ttl_s:g}s, syncable {bundle.retention.syncable}",
    ]
    # Limitations are the part a reader most needs and least wants; keep them last
    # and never fold them into a count.
    lines.extend(f"limitation  {text}" for text in summary["limitations"])
    _emit(payload, as_json=args.json, lines=lines)
    return EXIT_OK


def _replay(args: argparse.Namespace) -> int:
    bundle = read_bundle(args.bundle, max_bytes=args.max_bytes)
    source = ReplaySource(bundle, only_accepted=args.only_accepted)
    rgbd: list[RgbdObservation] = []
    joints: list[JointObservation] = []
    motions: list[MotionObservation] = []
    transitions: list[TaskTransition] = []
    for observation in source.observations():
        if isinstance(observation, RgbdObservation):
            rgbd.append(observation)
        elif isinstance(observation, JointObservation):
            joints.append(observation)
        elif isinstance(observation, MotionObservation):
            motions.append(observation)
        elif isinstance(observation, TaskTransition):
            transitions.append(observation)

    streams = sorted({f"{o.stream.device_id}:{o.stream.stream}" for o in rgbd})
    payload = {
        "bundle_id": bundle.bundle_id,
        "topology_digest": source.topology_digest,
        "only_accepted": args.only_accepted,
        "replayed": len(rgbd) + len(joints) + len(motions) + len(transitions),
        "motions": len(motions),
        "task_transitions": len(transitions),
        "rgbd": len(rgbd),
        "joints": len(joints),
        "streams": streams,
        "samples": sum(len(o.samples) for o in rgbd),
        "measured_samples": sum(len(o.measured_samples) for o in rgbd),
        "settled_joint_observations": sum(1 for o in joints if o.still),
    }
    lines = [
        f"replayed    {payload['replayed']} observations from {bundle.bundle_id}",
        f"topology    {payload['topology_digest']}",
        f"rgbd        {payload['rgbd']} across {len(streams)} stream(s)",
        f"samples     {payload['measured_samples']}/{payload['samples']} measured",
        f"joints      {payload['joints']} ({payload['settled_joint_observations']} settled)",
    ]
    lines.extend(f"stream      {name}" for name in streams)
    _emit(payload, as_json=args.json, lines=lines)
    return EXIT_OK


def _budget(args: argparse.Namespace) -> int:
    tolerance = TaskTolerance(
        task_id=args.task,
        allowed_m=args.allowed_mm / 1e3,
        basis=args.basis,
    )
    budget = observation_delay_budget(
        speed_m_s=args.speed_mm_s / 1e3,
        rotation_rate_rad_s=math.radians(args.rotation_deg_s),
        lever_arm_m=args.lever_mm / 1e3,
        latency_s=args.latency_ms / 1e3,
        orientation_error_rad=math.radians(args.orientation_deg),
    )
    verdict = budget.check(tolerance)
    payload = {
        "tolerance": tolerance.to_dict(),
        "budget": budget.to_dict(),
        "verdict": verdict.to_dict(),
    }
    lines = [f"task        {tolerance.task_id} allows {args.allowed_mm:.2f} mm"]
    lines.extend(
        f"term        {term.name:<22} {term.value_m * 1e3:7.3f} mm  [{term.kind}] {term.basis}"
        for term in budget.terms
    )
    lines.extend(
        [
            f"total       {verdict.total_m * 1e3:7.3f} mm ({verdict.combination})",
            f"headroom    {verdict.headroom_m * 1e3:7.3f} mm",
            f"dominant    {verdict.dominant_term}",
            f"verdict     {'fits' if verdict.fits else 'EXCEEDED'}",
        ]
    )
    _emit(payload, as_json=args.json, lines=lines)
    # A budget that does not close is a correct answer, so it reports rather than
    # raising -- but the exit code has to be actionable for a script.
    return EXIT_OK if verdict.fits else EXIT_FAILURE


def _estimate(args: argparse.Namespace) -> int:
    bundle = read_bundle(args.bundle, max_bytes=args.max_bytes)
    request = read_request(args.request)
    method_id = args.method if args.command == "estimate" else "task_response.v1"
    method = discover_methods().get(method_id)
    if method is None:
        raise CalibrationFailure(
            FAILURE_NOT_IMPLEMENTED,
            f"method {method_id!r} is not installed; install the current distribution",
        )
    try:
        fit = method.solve(bundle, request)
        content: dict[str, Any] = {
            "estimate": fit.to_dict(),
            "candidate_digest": fit.digest,
            "shadow_only": True,
            "evidence_digest": bundle.digest,
        }
        if args.command == "shadow":
            from leapcalib.estimation.response import propose_shadow

            content["proposal"] = propose_shadow(
                fit,
                TaskState.from_mapping(request["current"]),
                request["target_m_rad"],
                TaskDomain.from_mapping(request["domain"]),
                Stamp.from_mapping(request["now"]),
                topology_digest=request["topology_digest"],
            )
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationFailure(FAILURE_SCHEMA, f"invalid numerical request: {error}") from error
    payload = result_payload(content)
    if args.output:
        write_result(args.output, payload, overwrite=args.overwrite)
    lines = [
        f"candidate   {fit.digest}",
        f"method      {fit.method_id}",
        f"rank        {fit.metadata['data_rank']}",
        f"supported   {', '.join(fit.supported_blocks) or 'none'}",
        f"partial     {', '.join(fit.partial_blocks) or 'none'}",
        f"retained    {', '.join(fit.retained_blocks) or 'none'}",
        "authority   candidate/shadow only; no activation or motion",
    ]
    lines.extend(f"limitation  {text}" for text in fit.limitations)
    if "proposal" in content:
        lines.append(f"proposal    {content['proposal'].get('reason', '')}")
    _emit(payload, as_json=args.json, lines=lines)
    if "proposal" in content and not content["proposal"]["accepted"]:
        return EXIT_FAILURE
    return EXIT_OK if fit.supported_blocks or fit.partial_blocks else EXIT_FAILURE


def _serve(args: argparse.Namespace) -> int:
    from leapcalib.runtime.protocol import WorkerBudgets
    from leapcalib.runtime.server import main as serve_main

    if len(args.budgets_json) > 4096:
        raise ValueError("worker budget configuration exceeds byte bound")
    budgets = WorkerBudgets(**json.loads(args.budgets_json))
    return serve_main(args.socket, budgets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leap-calib",
        description=(
            "Offline calibration tooling: inspect/replay, bounded candidates, "
            "task-local shadows and error budgets. No hardware or activation authority."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("serve", help="serve compute-only jobs on an explicit private Unix socket")
    worker.add_argument("--socket", required=True, help="absolute path under an owner-private 0700 directory")
    worker.add_argument("--budgets-json", default="{}", help="bounded worker budgets; cannot exceed initial ceilings")
    worker.set_defaults(handler=_serve)

    def add_bundle_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("bundle", help="path to an evidence bundle JSON file")
        sub.add_argument(
            "--max-bytes",
            type=int,
            default=DEFAULT_MAX_BYTES,
            help="refuse a bundle larger than this (default: %(default)s)",
        )
        sub.add_argument("--json", action="store_true", help="emit JSON instead of text")

    inspect = subparsers.add_parser(
        "inspect", help="show a bundle's identity, quality facts and limitations"
    )
    add_bundle_args(inspect)
    inspect.set_defaults(handler=_inspect)

    replay = subparsers.add_parser("replay", help="read every observation back out of a bundle")
    add_bundle_args(replay)
    replay.add_argument(
        "--only-accepted",
        action="store_true",
        help="replay only the accepted split, leaving held-out samples untouched",
    )
    replay.set_defaults(handler=_replay)

    for command in ("estimate", "shadow"):
        solver = subparsers.add_parser(
            command,
            help="compute a candidate"
            if command == "estimate"
            else "fit measured response and propose a bounded shadow",
        )
        add_bundle_args(solver)
        solver.add_argument(
            "--request",
            required=True,
            help="bounded JSON request: prior/config or domain/current/target/now",
        )
        solver.add_argument("--output", help="optional private result artifact (never activation)")
        solver.add_argument(
            "--overwrite", action="store_true", help="explicitly replace the result file"
        )
        if command == "estimate":
            solver.add_argument(
                "--method", required=True, help="installed leapcalib.methods identifier"
            )
        solver.set_defaults(handler=_estimate)

    budget = subparsers.add_parser(
        "budget",
        help="size the delay and lever-arm error against a task tolerance",
        description=(
            "Arguments are in mm/deg/ms for the operator; the computation is in "
            "SI. 20 mm/s at 100 ms is already about 2 mm of delay error, and 1 "
            "degree at 130 mm is about 2.3 mm."
        ),
    )
    budget.add_argument("--task", required=True, help="task identifier")
    budget.add_argument(
        "--allowed-mm", type=float, required=True, help="position error the task absorbs"
    )
    budget.add_argument(
        "--basis",
        required=True,
        help="physical margin the tolerance came from, e.g. 'jaw 85mm - object 60mm'",
    )
    budget.add_argument("--speed-mm-s", type=float, default=0.0, help="endpoint speed")
    budget.add_argument("--rotation-deg-s", type=float, default=0.0, help="wrist rotation rate")
    budget.add_argument(
        "--lever-mm", type=float, default=0.0, help="distance from the rotation axis"
    )
    budget.add_argument(
        "--latency-ms", type=float, default=0.0, help="age of the observation acted on"
    )
    budget.add_argument(
        "--orientation-deg",
        type=float,
        default=0.0,
        help="orientation error acting through the lever arm",
    )
    budget.add_argument("--json", action="store_true", help="emit JSON instead of text")
    budget.set_defaults(handler=_budget)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CalibrationFailure as failure:
        print(f"refused: {failure}", file=sys.stderr)
        return EXIT_FAILURE
    except (OSError, ValueError) as error:
        # A bad path or a malformed value is the operator's problem to see, not a
        # traceback to decipher.
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())


__all__ = ["EXIT_FAILURE", "EXIT_OK", "EXIT_USAGE", "build_parser", "main"]

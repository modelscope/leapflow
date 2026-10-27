# Copyright (c) Alibaba, Inc. and its affiliates.
"""``leap-calib``: look at recorded evidence, replay it, and size an error budget.

Three verbs, all offline and all read-only. ``inspect`` and ``replay`` answer
questions about a stored bundle; ``budget`` answers whether a task's margin
survives the delay and lever-arm terms before anybody spends a day collecting
data for it.

What is deliberately absent is ``estimate`` and ``serve``. Neither a solver nor a
worker service exists, and a subcommand that exits with "not implemented" is
worse than no subcommand: it appears in ``--help`` as a capability, gets scripted
against, and is discovered to be hollow by whoever depended on it. They arrive
with the code that implements them.

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

from leapcalib.contracts.failures import CalibrationFailure
from leapcalib.contracts.observations import JointObservation, RgbdObservation
from leapcalib.validation.budget import TaskTolerance, observation_delay_budget
from leapcalib.validation.bundle import DEFAULT_MAX_BYTES, ReplaySource, read_bundle

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
    for observation in source.observations():
        if isinstance(observation, RgbdObservation):
            rgbd.append(observation)
        else:
            joints.append(observation)

    streams = sorted({f"{o.stream.device_id}:{o.stream.stream}" for o in rgbd})
    payload = {
        "bundle_id": bundle.bundle_id,
        "topology_digest": source.topology_digest,
        "only_accepted": args.only_accepted,
        "replayed": len(rgbd) + len(joints),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leap-calib",
        description=(
            "Offline calibration evidence tooling. Inspects and replays recorded "
            "observations and sizes task error budgets; it solves nothing and "
            "touches no hardware."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

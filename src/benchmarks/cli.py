# Copyright (c) Alibaba, Inc. and its affiliates.
"""Command-line interface for the standalone benchmark harness.

Subcommands: list, doctor, run, resume, report, gate.
Supports human-readable and JSON output.  Usage errors return exit code 64.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmarks.doctor import run_doctor
from benchmarks.gates import combine_gates, evaluate_gate, exit_code_for_gate
from benchmarks.manifest import load_manifests
from benchmarks.models import (
    BenchmarkManifest,
    BenchmarkResult,
    RunConfig,
)
from benchmarks.profiles import PROFILE_IDS, list_profiles, select_manifests
from benchmarks.registry import default_registry
from benchmarks.runner import BenchmarkRunner

EXIT_READY = 0
EXIT_CONDITIONAL = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 64


class _UsageError(ValueError):
    """Raised for CLI usage errors so main can return EX_USAGE (64)."""


class _ArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that reports usage errors through a typed exception."""

    def error(self, message: str) -> None:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.  Kept separate for import safety and testing."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    parser = _ArgumentParser(prog="python -m benchmarks", description="LeapFlow benchmark harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", parents=[common], help="List benchmarks and adapters")
    p_list.add_argument("manifest", nargs="?", help="Manifest file or directory")
    p_list.add_argument("--profile", choices=PROFILE_IDS, default="all")

    p_doctor = sub.add_parser("doctor", parents=[common], help="Check environment readiness")
    p_doctor.add_argument("manifest", nargs="?", help="Manifest file or directory")
    p_doctor.add_argument("--profile", choices=PROFILE_IDS, default="all")
    p_doctor.add_argument("--package", action="append", default=[])
    p_doctor.add_argument("--executable", action="append", default=[])
    p_doctor.add_argument("--env", action="append", default=[])
    p_doctor.add_argument("--data-dir", action="append", default=[])
    p_doctor.add_argument("--license", action="append", default=[])
    p_doctor.add_argument("--config", action="append", default=[])
    p_doctor.add_argument("--device", action="append", default=[])
    p_doctor.add_argument("--require-gpu", action="store_true")

    for name in ("run", "resume"):
        p_run = sub.add_parser(name, parents=[common], help=f"{name.title()} benchmark execution")
        p_run.add_argument("manifest", nargs="?", help="Manifest file or directory")
        p_run.add_argument("--profile", choices=PROFILE_IDS, default="all")
        p_run.add_argument("--benchmark", help="Run only this benchmark id")
        p_run.add_argument("--dry-run", action="store_true",
                           help="Check availability only, do not run trials")
        p_run.add_argument("--seed", type=int)
        p_run.add_argument("--timeout", type=float, default=300.0)
        p_run.add_argument("--retries", type=int, default=0)
        p_run.add_argument("--parallel", type=int, default=1)
        p_run.add_argument("--fail-fast", action="store_true")
        p_run.add_argument("--tag", action="append", default=[])
        p_run.add_argument(
            "--live-llm",
            action="store_true",
            help="Explicitly authorize configured live LLM calls for the live-llm profile",
        )
        p_run.add_argument(
            "--require-live-llm",
            action="store_true",
            help="Treat unavailable or failing live-llm evidence as a blocking gate",
        )
        p_run.add_argument(
            "--confirm-hardware",
            action="store_true",
            help="Confirm that an operator has authorized Tier4 preflight or motion",
        )
        p_run.add_argument("--output", type=Path, help="Write result JSON to this file")
        p_run.add_argument("--from", dest="resume_from", type=Path,
                           help="Prior result JSON (required by resume)")

    p_report = sub.add_parser("report", parents=[common], help="Summarize result JSON")
    p_report.add_argument("result", type=Path)

    p_gate = sub.add_parser("gate", parents=[common], help="Evaluate readiness gates")
    p_gate.add_argument("result", nargs="?", type=Path, help="Existing result JSON")
    p_gate.add_argument("--manifest", type=Path, help="Manifest file or directory")
    p_gate.add_argument(
        "--profile", choices=PROFILE_IDS,
        help="Run and gate this profile when no result JSON is supplied",
    )
    p_gate.add_argument("--seed", type=int)
    p_gate.add_argument("--timeout", type=float, default=300.0)
    p_gate.add_argument("--retries", type=int, default=0)
    p_gate.add_argument("--parallel", type=int, default=1)
    p_gate.add_argument("--fail-fast", action="store_true")
    p_gate.add_argument("--tag", action="append", default=[])
    p_gate.add_argument("--live-llm", action="store_true")
    p_gate.add_argument("--require-live-llm", action="store_true")
    p_gate.add_argument("--confirm-hardware", action="store_true")
    p_gate.set_defaults(resume_from=None)

    return parser


def _emit(data: Any, *, as_json: bool, human: str = "") -> None:
    """Print output in JSON or human-readable form."""
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(human or str(data))


def _manifest_or_error(path: str | Path) -> tuple[list[BenchmarkManifest], list[str]]:
    """Load manifests and format typed errors for CLI output."""
    manifests, errors = load_manifests(path)
    return manifests, [str(e) for e in errors]


def _load_result_file(path: Path) -> list[BenchmarkResult]:
    """Load one or more BenchmarkResult objects from JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _UsageError(f"cannot read result file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _UsageError(f"invalid result JSON: {exc}") from exc

    if isinstance(data, Mapping) and "results" in data:
        items = data["results"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise _UsageError("result JSON must contain an object or list of objects")
    return [BenchmarkResult.from_dict(item) for item in items]


def _save_results(path: Path, results: Sequence[BenchmarkResult]) -> None:
    """Atomically write benchmark results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "results": [r.to_dict() for r in results]}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _dependency_groups(manifests: Sequence[BenchmarkManifest]) -> dict[str, list[str]]:
    """Group prefixed manifest dependency strings for doctor checks."""
    groups: dict[str, list[str]] = {
        "python": [], "executable": [], "env": [], "data": [],
        "license": [], "hardware": [], "config": [],
    }
    for manifest in manifests:
        for dependency in manifest.dependencies:
            prefix, sep, value = dependency.partition(":")
            if sep and prefix in groups:
                groups[prefix].append(value)
            else:
                groups["python"].append(dependency)
    return groups


# Default manifest directories.
_MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"
_DEFAULT_MANIFEST_DIRS: tuple[Path, ...] = (
    _MANIFEST_ROOT / "external",
    _MANIFEST_ROOT / "native",
    _MANIFEST_ROOT / "hardware",
)


def _load_default_manifests() -> tuple[list[BenchmarkManifest], list[str]]:
    """Load manifests from all default directories."""
    manifests: list[BenchmarkManifest] = []
    errors: list[str] = []
    for d in _DEFAULT_MANIFEST_DIRS:
        if d.is_dir():
            ms, es = _manifest_or_error(d)
            manifests.extend(ms)
            errors.extend(es)
    manifests.sort(key=lambda m: m.id)
    return manifests, errors


def _cmd_list(args: argparse.Namespace) -> int:
    registry = default_registry()
    manifests: list[BenchmarkManifest] = []
    errors: list[str] = []
    if args.manifest:
        manifests, errors = _manifest_or_error(args.manifest)
    else:
        manifests, errors = _load_default_manifests()
    manifests = list(select_manifests(manifests, args.profile))

    data = {
        "adapters": list(registry.list_available()),
        "benchmarks": [m.to_dict() for m in manifests],
        "profiles": [
            {"id": p.profile_id, "tiers": list(p.tiers), "description": p.description}
            for p in list_profiles()
        ],
        "conflicts": [c.to_dict() for c in registry.conflicts],
        "errors": errors,
    }
    lines = [f"Adapters: {', '.join(data['adapters']) or '(none)'}"]
    lines.extend(f"  {m.id} v{m.version} (tier {m.tier}, adapter={m.adapter})" for m in manifests)
    lines.extend(f"ERROR: {e}" for e in errors)
    _emit(data, as_json=args.json, human="\n".join(lines))
    return EXIT_USAGE if errors else EXIT_READY


def _cmd_doctor(args: argparse.Namespace) -> int:
    manifests: list[BenchmarkManifest] = []
    errors: list[str] = []
    if args.manifest:
        manifests, errors = _manifest_or_error(args.manifest)
    else:
        manifests, errors = _load_default_manifests()
    manifests = list(select_manifests(manifests, args.profile))
    groups = _dependency_groups(manifests)

    report = run_doctor(
        python_packages=tuple(groups["python"] + args.package),
        executables=tuple(groups["executable"] + args.executable),
        environment_variables=tuple(groups["env"] + args.env),
        data_directories=tuple(groups["data"] + args.data_dir),
        licenses=tuple(groups["license"] + args.license),
        configuration_keys=tuple(groups["config"] + args.config),
        hardware_devices=tuple(groups["hardware"] + args.device),
        gpu_required=args.require_gpu,
    )
    data = report.to_dict()
    data["manifest_errors"] = errors
    lines = [
        f"[{c.status.upper():4}] {c.check_id}: {c.detail}"
        for c in report.checks
    ]
    lines.append(f"Ready: {'yes' if report.ready else 'no'}")
    _emit(data, as_json=args.json, human="\n".join(lines))
    if errors:
        return EXIT_USAGE
    return EXIT_READY if report.ready else EXIT_BLOCKED


async def _run_manifests(
    manifests: Sequence[BenchmarkManifest],
    args: argparse.Namespace,
    completed_ids: frozenset[str],
) -> list[BenchmarkResult]:
    """Execute selected manifests and seeds sequentially; trials run concurrently."""
    runner = BenchmarkRunner(registry=default_registry())
    results: list[BenchmarkResult] = []
    for manifest in manifests:
        seeds = (args.seed,) if args.seed is not None else (manifest.seeds or (42,))
        for seed in seeds:
            config = RunConfig(
                seed=seed,
                timeout_seconds=args.timeout,
                retry_count=max(0, args.retries),
                max_parallel=max(1, args.parallel),
                fail_fast=args.fail_fast,
                resume_from=str(args.resume_from or ""),
                profile=args.profile,
                tags=tuple(args.tag),
                live_llm_enabled=bool(args.live_llm or args.require_live_llm),
                require_live_llm=bool(args.require_live_llm),
                hardware_enabled=bool(args.confirm_hardware),
            )
            result = await runner.run(
                manifest, config, completed_ids=completed_ids,
            )
            results.append(result)
    return results


def _availability_for_args(adapter: Any, args: argparse.Namespace) -> Any:
    """Run availability inside the same explicit authority as normal execution."""
    from benchmarks.runtime import BenchmarkRuntimeContext, runtime_context

    async def check() -> Any:
        with runtime_context(BenchmarkRuntimeContext(
            live_llm_enabled=bool(getattr(args, "live_llm", False) or getattr(args, "require_live_llm", False)),
            require_live_llm=bool(getattr(args, "require_live_llm", False)),
            hardware_enabled=bool(getattr(args, "confirm_hardware", False)),
        )):
            return await adapter.availability()

    return asyncio.run(check())


def _cmd_dry_run(
    manifests: Sequence[BenchmarkManifest], args: argparse.Namespace,
) -> int:
    """Check adapter availability for each manifest without running trials."""
    registry = default_registry()
    results: list[dict[str, Any]] = []
    for manifest in manifests:
        adapter = registry.get(manifest.adapter)
        if adapter is None:
            results.append({
                "benchmark_id": manifest.id, "adapter": manifest.adapter,
                "available": False, "reason": "adapter not registered",
            })
            continue
        avail = _availability_for_args(adapter, args)
        results.append({
            "benchmark_id": manifest.id, "adapter": manifest.adapter,
            **avail.to_dict(),
        })
    data = {"dry_run": True, "results": results}
    lines = [f"{'AVAIL' if r.get('available') else 'MISS ':5} {r['benchmark_id']}"
             f" (adapter={r['adapter']})"
             f"{' — ' + r.get('reason', '') if r.get('reason') else ''}"
             for r in results]
    _emit(data, as_json=args.json, human="\n".join(lines))
    return EXIT_READY if all(r.get("available") for r in results) else EXIT_CONDITIONAL


def _apply_strict_live_requirement(
    manifests: Sequence[BenchmarkManifest], args: argparse.Namespace,
) -> list[BenchmarkManifest]:
    """Promote selected live-LLM evidence to a required gate on request."""
    if not getattr(args, "require_live_llm", False):
        return list(manifests)
    return [
        replace(manifest, required=True)
        if "live-llm" in manifest.tags
        else manifest
        for manifest in manifests
    ]


def _cmd_run(args: argparse.Namespace) -> int:
    if args.manifest:
        manifests, errors = _manifest_or_error(args.manifest)
    else:
        manifests, errors = _load_default_manifests()
    if errors:
        _emit({"errors": errors}, as_json=args.json, human="\n".join(errors))
        return EXIT_USAGE
    manifests = _apply_strict_live_requirement(
        select_manifests(manifests, args.profile), args,
    )

    if args.benchmark:
        manifests = [m for m in manifests if m.id == args.benchmark]

    if not manifests:
        raise _UsageError(f"no manifests match profile {args.profile!r}")

    if args.command == "resume" and args.resume_from is None:
        raise _UsageError("resume requires --from RESULT.json")

    # Dry-run: check availability only.
    if getattr(args, "dry_run", False):
        return _cmd_dry_run(manifests, args)

    prior: list[BenchmarkResult] = []
    if args.resume_from:
        prior = _load_result_file(args.resume_from)
    completed_ids = frozenset(
        trial.trial_id
        for result in prior
        for trial in result.trials
        if trial.status.is_terminal
    )

    results = asyncio.run(_run_manifests(manifests, args, completed_ids))
    if args.output:
        _save_results(args.output, results)

    payload = {"results": [r.to_dict() for r in results]}
    lines = [
        f"{r.benchmark_id} seed={r.seed}: {r.passed}/{r.total_trials} passed, "
        f"{r.failed} failed, {r.unavailable} unavailable ({r.duration_seconds:.2f}s)"
        for r in results
    ]
    if args.output:
        lines.append(f"Results: {args.output}")
    _emit(payload, as_json=args.json, human="\n".join(lines))

    if any(r.failed for r in results):
        return EXIT_CONDITIONAL
    if any(r.unavailable for r in results):
        return EXIT_CONDITIONAL
    return EXIT_READY


def _cmd_report(args: argparse.Namespace) -> int:
    results = _load_result_file(args.result)
    payload = {
        "benchmarks": len(results),
        "trials": sum(r.total_trials for r in results),
        "passed": sum(r.passed for r in results),
        "failed": sum(r.failed for r in results),
        "unavailable": sum(r.unavailable for r in results),
        "results": [r.to_dict() for r in results],
    }
    lines = [
        f"{r.benchmark_id} v{r.version}: {r.passed}/{r.total_trials} passed "
        f"(seed={r.seed}, adapter={r.adapter_id}@{r.adapter_version or 'unknown'})"
        for r in results
    ]
    _emit(payload, as_json=args.json, human="\n".join(lines))
    return EXIT_READY


def _cmd_gate(args: argparse.Namespace) -> int:
    manifest_map: dict[str, BenchmarkManifest] = {}
    errors: list[str] = []

    if args.manifest:
        manifests, errors = _manifest_or_error(args.manifest)
    else:
        manifests, errors = _load_default_manifests()

    if errors:
        _emit({"errors": errors}, as_json=args.json, human="\n".join(errors))
        return EXIT_USAGE

    if args.result is not None:
        results = _load_result_file(args.result)
        strict_live = bool(getattr(args, "require_live_llm", False)) or any(
            result.config.require_live_llm for result in results
        )
        manifest_map = {
            manifest.id: replace(manifest, required=True)
            if strict_live and "live-llm" in manifest.tags
            else manifest
            for manifest in manifests
        }
    else:
        if not args.profile:
            raise _UsageError("gate requires RESULT.json or --profile PROFILE")
        selected = _apply_strict_live_requirement(
            select_manifests(manifests, args.profile), args,
        )
        if not selected:
            raise _UsageError(f"no manifests match profile {args.profile!r}")
        manifest_map = {m.id: m for m in selected}
        results = asyncio.run(_run_manifests(selected, args, frozenset()))

    gates = [evaluate_gate(r, manifest_map.get(r.benchmark_id)) for r in results]
    combined = combine_gates(gates)
    data = {"status": combined.status.value, "gates": [g.to_dict() for g in gates]}
    lines = [f"Gate: {combined.status.value.upper()}"]
    lines.extend(f"  {g.benchmark_id}: {g.status.value} — {'; '.join(g.details)}" for g in gates)
    _emit(data, as_json=args.json, human="\n".join(lines))
    return exit_code_for_gate(combined.status)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        dispatch = {
            "list": _cmd_list,
            "doctor": _cmd_doctor,
            "run": _cmd_run,
            "resume": _cmd_run,
            "report": _cmd_report,
            "gate": _cmd_gate,
        }
        return dispatch[args.command](args)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(parser.format_usage().strip(), file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_CONDITIONAL


__all__ = [
    "EXIT_BLOCKED",
    "EXIT_CONDITIONAL",
    "EXIT_READY",
    "EXIT_USAGE",
    "main",
]

"""Command-line entry points for AAAI demo evidence packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from aaai_demo.evidence import EvidenceBundleError, export_evidence_bundle
from aaai_demo.headless_seam import export_headless_seam
from aaai_demo.poster import render_poster_source
from aaai_demo.render import compose_backup_reel, render_demo_package


def _parser() -> argparse.ArgumentParser:
    """Create the narrow, read-only-or-derived demo command surface."""
    parser = argparse.ArgumentParser(description="Build auditable AAAI demo artifacts.")
    commands = parser.add_subparsers(dest="command", required=True)

    bundle = commands.add_parser("bundle", help="Export a write-once CE-X evidence bundle.")
    bundle.add_argument("--run-dir", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--drift-fixture", type=Path)
    bundle.add_argument("--trajectory-dir", type=Path)
    bundle.add_argument(
        "--signal-archive",
        type=Path,
        help="Manifest that binds signal-mode media, state/event evidence, and expect() result.",
    )
    bundle.add_argument(
        "--headless-seam",
        type=Path,
        help="Write-once real-widget structural-seam record from the headless-seam command.",
    )
    bundle.add_argument("--include-media", action="store_true")

    seam = commands.add_parser("headless-seam", help="Run and record the real offscreen PyQt structural seam.")
    seam.add_argument("--output-dir", type=Path, required=True)

    render = commands.add_parser("render", help="Render a static causal-trace console and EDL.")
    render.add_argument("--bundle", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)

    poster = commands.add_parser("poster", help="Render a provisional accepted-demo poster source.")
    poster.add_argument("--bundle", type=Path, required=True)
    poster.add_argument("--output-dir", type=Path, required=True)
    poster.add_argument("--viewer-url", default="")

    reel = commands.add_parser("backup-reel", help="Concatenate reviewed clips without re-encoding.")
    reel.add_argument("--clip", type=Path, action="append", required=True)
    reel.add_argument("--output", type=Path, required=True)
    return parser


def _bundle_from_path(path: Path) -> dict:
    """Read a normalized evidence bundle and reject malformed input."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBundleError(f"cannot read bundle: {path}") from exc
    if not isinstance(value, dict) or value.get("kind") != "aaai_demo_evidence_bundle":
        raise EvidenceBundleError("input is not an AAAI demo evidence bundle")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit packaging operation and print only resulting artifact paths."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "bundle":
            bundle = export_evidence_bundle(
                args.run_dir,
                args.output_dir,
                drift_fixture=args.drift_fixture,
                trajectory_dir=args.trajectory_dir,
                signal_archive=args.signal_archive,
                headless_seam=args.headless_seam,
                include_media=args.include_media,
            )
            print(json.dumps({"bundle": str(args.output_dir), "run_id": bundle["run"]["run_id"]}, sort_keys=True))
            return 0
        if args.command == "headless-seam":
            record = export_headless_seam(args.output_dir)
            print(json.dumps({"headless_seam": str(args.output_dir), "run_id": record["run_id"]}, sort_keys=True))
            return 0
        if args.command == "render":
            paths = render_demo_package(_bundle_from_path(args.bundle), args.output_dir)
            print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
            return 0
        if args.command == "poster":
            paths = render_poster_source(
                _bundle_from_path(args.bundle), args.output_dir, viewer_url=args.viewer_url,
            )
            print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
            return 0
        if args.command == "backup-reel":
            print(compose_backup_reel(args.clip, args.output))
            return 0
    except EvidenceBundleError as exc:
        print(f"error: {exc}")
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

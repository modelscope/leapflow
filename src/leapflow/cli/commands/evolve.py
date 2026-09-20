"""`leap evolve` — run the learning boundary now, and report what it did.

Capability evolution starts at a durable session boundary. The boundary flushes the
append-only action stream, seals exactly one named session slice, and queues hindsight
teacher work that survives client exit and daemon restart.

This command routes to the daemon because it is the sole DuckDB writer. Requiring an
explicit session id prevents one client from accidentally finalizing another client's
most-recent session.

It decides nothing on its own: whether a proposal is even *written* still depends on
``evolution.enabled``, and whether it is acted on still depends on generation,
review, approval and trust. Running this is asking the framework to look at what it
has already done, not granting it new permission.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from leapflow.config import load_config


def cmd_evolve(args: argparse.Namespace) -> int:
    """Entry point for the ``leap evolve`` subcommand."""
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


async def _run(args: argparse.Namespace) -> int:
    from leapflow.daemon.client import ensure_daemon_client

    settings = load_config()
    as_json = bool(getattr(args, "json", False))

    try:
        client = await ensure_daemon_client(settings)
    except Exception as exc:  # noqa: BLE001 - a missing daemon is a normal outcome here
        _report_error(f"leapd unavailable: {exc}", as_json)
        return 1

    try:
        result = await client.evolution_run(
            session_id=str(args.session),
            reason=str(getattr(args, "reason", "") or "manual"),
            wait=bool(getattr(args, "wait", False)),
            timeout_s=float(getattr(args, "timeout", 180.0)),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced, never a traceback
        _report_error(str(exc), as_json)
        return 1

    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1

    if not result.get("ok"):
        print(f"Learning boundary failed: {result.get('error', 'unknown error')}")
        return 1

    steps = int(result.get("trajectory_steps") or 0)
    jobs = list(result.get("job_ids") or [])
    print(
        f"Session {result.get('session_id', args.session)} finalized "
        f"({result.get('reason', 'manual')}) in {result.get('duration_s', 0)}s — "
        f"{steps} evidence event(s), {len(jobs)} teacher job(s) queued."
    )
    if not jobs:
        print("  No new action evidence was found after the previous session boundary.")
    elif result.get("job"):
        print(f"  Teacher job status: {result['job'].get('status', 'unknown')}")
    else:
        print(f"  Durable teacher job: {jobs[0]}")
    if not getattr(settings, "evolution_enabled", False):
        print(
            "  Self-evolution is off, so teacher output cannot authorize a capability proposal. "
            "Durable teacher grading remains enabled. "
            "Enable with: leap config set evolution.enabled true"
        )
    print("  See durable progress on the board: leap board evolution")
    return 0


def _report_error(message: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
    else:
        print(f"evolve: {message}")


__all__ = ["cmd_evolve"]

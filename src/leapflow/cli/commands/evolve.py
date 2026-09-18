"""`leap evolve` — run the learning boundary now, and report what it did.

Capability evolution is driven from exactly one place: the learning boundary, which
grades the recorded trajectory, lets the world model propose capabilities it found
missing, and runs the cold-path governance sweep. That boundary used to be reachable
only from context cleanup, which in daemon mode means *process shutdown* — so a
daemon that ran for a week never evolved, one killed with SIGKILL never evolved at
all, and there was no way to answer "did it evolve, and what happened" without
stopping the daemon and reading its log.

This command makes the boundary an explicit, observable operation. It routes to the
daemon rather than doing the work locally, because the daemon owns the context that
holds the trajectory buffer and the proposal queue; a second context would grade an
empty buffer and truthfully report that nothing happened.

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
        result = await client.evolution_run(reason=str(getattr(args, "reason", "") or "manual"))
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
    print(
        f"Learning boundary ran ({result.get('reason', 'manual')}) in "
        f"{result.get('duration_s', 0)}s — {steps} trajectory step(s) graded."
    )
    if not steps:
        # Said plainly, because an empty trajectory is the common case and looks
        # identical to a failure otherwise. The governance sweep still ran: that is
        # the whole reason it sits outside the trajectory branch.
        print(
            "  No turns were recorded since the last boundary, so the teacher had "
            "nothing to grade. The governance sweep still ran."
        )
    if not getattr(settings, "evolution_enabled", False):
        print(
            "  Self-evolution is off, so no capability proposal was written. "
            "The world model still graded and recorded what it learned. "
            "Enable with: leap config set evolution.enabled true"
        )
    print("  See the result on the board: leap board evolution")
    return 0


def _report_error(message: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
    else:
        print(f"evolve: {message}")


__all__ = ["cmd_evolve"]

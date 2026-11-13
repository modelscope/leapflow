# Copyright (c) Alibaba, Inc. and its affiliates.
"""`leap board` — open or serve the LeapBoard monitoring web dashboard.

Modes:
- default (client): ensure a dashboard server is running, then open the default
  browser at the requested view. Returns immediately; the server runs detached.
- ``--serve`` (server): run the aiohttp view server in the foreground. Used by
  the launcher spawn and for a dedicated dashboard terminal.
- ``stop [--all]``: stop this profile's board servers, including ones the launcher
  lost track of; ``--all`` stops every board server of the current user.
- ``status``: list running board servers and which of them is this profile's.

The dashboard is a view client: it connects to leapd, so it does not initialize
its own Context.
"""

from __future__ import annotations

import argparse
import asyncio

from leapflow.config import load_config
from leapflow.dashboard import launcher

_DEP_HINT = (
    "The dashboard web server requires 'aiohttp', which ships with LeapFlow.\n"
    "Reinstall it with: pip install aiohttp"
)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Entry point for the ``leap board`` subcommand."""
    settings = load_config()

    # Lifecycle verbs reserve these names from the template positional, matching the
    # TUI's /board verbs; they need no web dependency, so a broken install can clean up.
    verb = "" if getattr(args, "serve", False) else (getattr(args, "template", "") or "")
    if verb == "stop":
        return _stop(settings, everything=bool(getattr(args, "all", False)))
    if verb == "status":
        return _status(settings)

    if not launcher.aiohttp_available():
        print(_DEP_HINT)
        return 1

    if getattr(args, "serve", False):
        return _serve(args, settings)
    return _open(args, settings)


def _stop(settings: object, *, everything: bool) -> int:
    results, left = launcher.stop_servers(settings, everything=everything)
    for result in results:
        port = f" port {result.port}" if result.port else ""
        print(f"{'stopped' if result.stopped else 'FAILED '} pid {result.pid}{port}: {result.detail}")
    if not results:
        print("No LeapBoard server of this profile is running.")
    if left:
        unattributed = sum(1 for server in left if not server.owner)
        print(
            f"{len(left)} other LeapBoard server(s) left running "
            f"({unattributed} not attributable to a profile, e.g. from an older build). "
            "Run 'leap board stop --all' to stop every board server of this user."
        )
    return 0 if all(result.stopped for result in results) else 1


def _status(settings: object) -> int:
    from leapflow.dashboard import instances

    owner = launcher.server_owner(settings)
    recorded = int((launcher.load_state(settings) or {}).get("pid") or 0)
    servers = instances.discover_board_servers()
    if not servers:
        print("No LeapBoard server is running.")
        return 0
    for server in servers:
        if server.owner == owner or (server.pid == recorded and not server.owner):
            tag = "this profile" + (" (current)" if server.pid == recorded else ", orphaned")
        else:
            tag = "other profile" if server.owner else "unattributed"
        print(f"pid {server.pid}  {server.bind}:{server.port}  {tag}")
    return 0


def _serve(args: argparse.Namespace, settings: object) -> int:
    from leapflow.dashboard.server import run_server
    from leapflow.logging_setup import init_logging

    # Long-lived server process: capture INFO evidence in its log output,
    # mirroring the leapd daemon surface.
    init_logging("INFO")
    token = getattr(args, "token", "") or launcher.generate_token()
    bind = getattr(args, "bind", "") or settings.dashboard_bind
    port = getattr(args, "port", 0) or settings.dashboard_port
    try:
        return asyncio.run(run_server(settings, token=token, bind=bind, port=port))
    except KeyboardInterrupt:
        return 130


def _open(args: argparse.Namespace, settings: object) -> int:
    try:
        state = launcher.ensure_server(settings)
    except RuntimeError as exc:
        print(f"dashboard: {exc}")
        return 1

    template = getattr(args, "template", "") or ""
    url = launcher.build_view_url(state["bind"], state["port"], state["token"], template=template)

    auto_open = bool(getattr(settings, "dashboard_auto_open", True)) and not getattr(args, "no_open", False)
    if auto_open and launcher.open_in_browser(url):
        print(f"Opened dashboard in your browser: {url}")
    else:
        print(f"Dashboard ready: {url}")
    return 0


__all__ = ["cmd_dashboard"]

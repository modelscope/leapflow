"""`leap dashboard` — open or serve the monitoring web dashboard.

Two modes:
- default (client): ensure a dashboard server is running, then open the default
  browser at the requested view. Returns immediately; the server runs detached.
- ``--serve`` (server): run the aiohttp view server in the foreground. Used by
  the launcher spawn and for a dedicated dashboard terminal.

The dashboard is a view client: it connects to leapd, so it does not initialize
its own Context.
"""

from __future__ import annotations

import argparse
import asyncio

from leapflow.config import load_config
from leapflow.dashboard import launcher

_DEP_HINT = (
    "The dashboard web server requires the optional 'aiohttp' dependency.\n"
    "Install it with: pip install 'leapflow[dashboard]'"
)


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Entry point for the ``leap dashboard`` subcommand."""
    settings = load_config()

    if not launcher.aiohttp_available():
        print(_DEP_HINT)
        return 1

    if getattr(args, "serve", False):
        return _serve(args, settings)
    return _open(args, settings)


def _serve(args: argparse.Namespace, settings: object) -> int:
    from leapflow.dashboard.server import run_server

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

    action = getattr(args, "action", None) or "home"
    url = launcher.build_url(state["bind"], state["port"], state["token"])
    if action and action != "home":
        url += f"&action={action}"
        if action == "session":
            url += "&target=session"

    auto_open = bool(getattr(settings, "dashboard_auto_open", True)) and not getattr(args, "no_open", False)
    if auto_open and launcher.open_in_browser(url):
        print(f"Opened dashboard in your browser: {url}")
    else:
        print(f"Dashboard ready: {url}")
    return 0


__all__ = ["cmd_dashboard"]

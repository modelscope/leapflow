"""CLI commands for leapd daemon management.

``leap daemon status`` — show whether leapd is running
``leap daemon start``  — start leapd for the active profile
``leap daemon stop``   — send SIGTERM to a running leapd
"""
from __future__ import annotations

import asyncio
import signal
import sys
from argparse import Namespace
from pathlib import Path


def cmd_daemon(args: Namespace) -> int:
    """Route daemon subcommands."""
    from leapflow.config import load_config

    settings = load_config()
    run_dir = settings.profile_dir / "run"

    action = getattr(args, "daemon_action", None) or "status"

    if action == "status":
        return _status(run_dir)
    if action == "start":
        return _start(settings, getattr(args, "mock_host", False))
    if action == "stop":
        return _stop(run_dir)
    if action == "serve":
        if not getattr(args, "internal", False):
            sys.stderr.write("'leap daemon serve' is an internal command. Use 'leap daemon start'.\n")
            return 2
        return asyncio.run(_serve(settings, getattr(args, "mock_host", False)))

    sys.stderr.write(f"Unknown daemon action: {action}\n")
    return 1


def _status(run_dir: Path) -> int:
    from leapflow.daemon.lifecycle import DaemonInfo

    info = DaemonInfo.discover(run_dir)
    print(info.format_status())
    if info.sock_path is not None:
        print(f"socket: {info.sock_path}")
    return 0 if info.is_healthy else 1


def _start(settings: object, mock_host: bool) -> int:
    from leapflow.daemon.client import DaemonUnavailableError, ensure_daemon_client

    async def _run() -> int:
        try:
            client = await ensure_daemon_client(
                settings,
                mock_host=mock_host,
                status_callback=lambda msg: print(f"→ {msg}"),
            )
            status = await client.status()
        except DaemonUnavailableError as exc:
            sys.stderr.write(f"Failed to start leapd: {exc}\n")
            return 1
        print(
            "leapd ready "
            f"(pid={status.get('pid')}, profile={status.get('profile')}, "
            f"clients={status.get('active_clients')})"
        )
        return 0

    return asyncio.run(_run())


def _stop(run_dir: Path) -> int:
    from leapflow.daemon.lifecycle import DaemonInfo, send_signal, cleanup_stale

    info = DaemonInfo.discover(run_dir)
    if not info.is_running:
        if info.pid is not None:
            cleanup_stale(run_dir)
            print("Cleaned up stale daemon files.")
        else:
            print("leapd is not running.")
        return 0

    if send_signal(run_dir, signal.SIGTERM):
        print(f"Sent SIGTERM to leapd (pid={info.pid}).")
        return 0

    sys.stderr.write("Failed to stop leapd.\n")
    return 1


async def _serve(settings: object, mock_host: bool) -> int:
    from leapflow.daemon.server import serve_daemon

    return await serve_daemon(settings, mock_host=mock_host)

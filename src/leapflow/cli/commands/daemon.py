"""CLI commands for leapd daemon management.

``leap daemon status`` — show whether leapd is running
``leap daemon stop``   — send SIGTERM to a running leapd
"""
from __future__ import annotations

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
    if action == "stop":
        return _stop(run_dir)

    sys.stderr.write(f"Unknown daemon action: {action}\n")
    return 1


def _status(run_dir: Path) -> int:
    from leapflow.daemon.lifecycle import DaemonInfo

    info = DaemonInfo.discover(run_dir)
    print(info.format_status())
    return 0 if info.is_healthy else 1


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

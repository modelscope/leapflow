"""CLI commands for leapd daemon management.

``leap daemon status``  — show whether leapd is running
``leap daemon start``   — start leapd for the active profile
``leap daemon stop``    — send SIGTERM to a running leapd
``leap daemon restart`` — restart leapd so code/config changes take effect
"""
from __future__ import annotations

import asyncio
import signal
import sys
import time
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
    if action == "restart":
        return _restart(settings, getattr(args, "mock_host", False))
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
    if info.is_healthy and info.sock_path is not None:
        try:
            details = asyncio.run(_runtime_status(info.sock_path))
        except Exception as exc:
            sys.stderr.write(f"Could not fetch daemon runtime details: {exc}\n")
        else:
            _print_runtime_status(details)
    return 0 if info.is_healthy else 1


async def _runtime_status(sock_path: Path) -> dict:
    from leapflow.daemon.client import DaemonClient

    return await DaemonClient(sock_path).status()


def _print_runtime_status(status: dict) -> None:
    print(
        "runtime: "
        f"profile={status.get('profile')} "
        f"clients={status.get('active_clients')} "
        f"volatile={status.get('volatile')}"
    )
    print(
        "model: "
        f"{status.get('model')} "
        f"context={status.get('context_used', 0)}/{status.get('llm_context_length', 0)}"
    )
    if status.get("session_id"):
        print(f"session: {status['session_id']}")
    if status.get("runtime_version"):
        print(f"version: {status['runtime_version']}")
    if status.get("runtime_source"):
        print(f"source: {status['runtime_source']}")
    if status.get("runtime_executable"):
        print(f"python: {status['runtime_executable']}")
    if status.get("config_path"):
        print(f"config: {status['config_path']}")
    if status.get("project_env_path"):
        print(f"project_env: {status['project_env_path']}")
    if status.get("db_path"):
        print(f"db: {status['db_path']}")


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


def _restart(settings: object, mock_host: bool) -> int:
    run_dir = settings.profile_dir / "run"
    print("Restarting leapd...")
    stop_code = _stop(run_dir)
    if stop_code != 0:
        return stop_code
    if not _wait_stopped(run_dir):
        sys.stderr.write("Timed out waiting for leapd to stop.\n")
        return 1
    return _start(settings, mock_host)


def _wait_stopped(run_dir: Path, *, timeout_s: float = 10.0) -> bool:
    from leapflow.daemon.lifecycle import DaemonInfo

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not DaemonInfo.discover(run_dir).is_running:
            return True
        time.sleep(0.1)
    return not DaemonInfo.discover(run_dir).is_running


async def _serve(settings: object, mock_host: bool) -> int:
    from leapflow.daemon.server import serve_daemon

    return await serve_daemon(settings, mock_host=mock_host)

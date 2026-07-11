"""CLI commands for leapd daemon management.

``leap daemon status``  — show whether leapd is running
``leap daemon start``   — start leapd for the active profile
``leap daemon stop``    — stop running daemon and verify shutdown
``leap daemon restart`` — restart leapd so code/config changes take effect
"""
from __future__ import annotations

import asyncio
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
        return _stop(run_dir, force=getattr(args, "force", False))
    if action == "restart":
        return _restart(settings, getattr(args, "mock_host", False), force=getattr(args, "force", False))
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
    connected_clients = status.get("connected_clients")
    connection_suffix = (
        f" connected={connected_clients}" if connected_clients is not None else ""
    )
    print(
        "runtime: "
        f"profile={status.get('profile')} "
        f"clients={status.get('active_clients')}"
        f"{connection_suffix} "
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
    host = status.get("host_backend")
    if isinstance(host, dict):
        print(
            "host: "
            f"backend={host.get('backend')} "
            f"started={host.get('started')} "
            f"pid={host.get('pid')}"
        )
        if host.get("command"):
            args = " ".join(str(arg) for arg in host.get("args") or [])
            command = f"{host.get('command')} {args}".strip()
            print(f"host_command: {command}")
        if host.get("capability_version"):
            print(f"host_capability: {host['capability_version']}")
        if host.get("last_error"):
            print(f"host_error: {host['last_error']}")


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


def _stop(run_dir: Path, *, force: bool = False, timeout_s: float = 10.0) -> int:
    from leapflow.daemon.lifecycle import DaemonInfo, cleanup_stale, stop_daemon

    info = DaemonInfo.discover(run_dir)
    if not info.is_running:
        if info.pid is not None:
            cleanup_stale(run_dir)
            print("Cleaned up stale daemon files.")
        else:
            print("leapd is not running.")
        return 0

    graceful_requested = False
    if info.is_healthy and info.sock_path is not None:
        graceful_requested = _request_shutdown(info.sock_path)
    print(f"Stopping leapd (pid={info.pid})...")
    result = stop_daemon(
        run_dir,
        timeout_s=timeout_s,
        force=force,
        grace_timeout_s=2.0 if graceful_requested else 0.0,
    )
    if result.stopped:
        suffix = " with force" if result.forced else ""
        print(f"leapd stopped{suffix}.")
        return 0

    sys.stderr.write(
        f"Timed out waiting for leapd to stop (pid={result.pid}). "
        "Run 'leap daemon stop --force' or inspect the process manually.\n"
    )
    return 1


def _request_shutdown(sock_path: Path) -> bool:
    from leapflow.daemon.client import DaemonClient, DaemonUnavailableError

    async def _run() -> bool:
        try:
            await DaemonClient(sock_path, timeout_s=2.0).shutdown()
            return True
        except DaemonUnavailableError:
            return False

    return asyncio.run(_run())


def _restart(settings: object, mock_host: bool, *, force: bool = False) -> int:
    run_dir = settings.profile_dir / "run"
    print("Restarting leapd...")
    stop_code = _stop(run_dir, force=force, timeout_s=10.0)
    if stop_code != 0:
        sys.stderr.write("Restart aborted because old leapd did not stop cleanly.\n")
        return stop_code
    return _start(settings, mock_host)



async def _serve(settings: object, mock_host: bool) -> int:
    from leapflow.daemon.server import serve_daemon

    return await serve_daemon(settings, mock_host=mock_host)

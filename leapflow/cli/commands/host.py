"""OS Host lifecycle management commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from leapflow.config import load_config
from leapflow.host import (
    HostManager,
    HostState,
    HostStatus,
    check_permissions,
    open_accessibility_settings,
    open_screen_recording_settings,
)

# ── ANSI colors ──────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[1;36m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}✗{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}!{_RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


# ── Helpers ──────────────────────────────────────────────────────────────


def _format_uptime(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m:02d}m"


def _perm_icon(value: Optional[bool]) -> str:
    if value is True:
        return f"{_GREEN}✓ Granted{_RESET}"
    if value is False:
        return f"{_RED}✗ Not granted{_RESET}"
    return f"{_YELLOW}○ Unknown{_RESET}"


def _build_manager() -> HostManager:
    settings = load_config()
    return HostManager(
        host_root=settings.host_root,
        host_socket=settings.host_socket,
        pid_file=settings.host_pid_file,
        log_file=settings.host_log_file,
        bundle_id=settings.host_bundle_id,
    )


# ── Subcommand implementations ──────────────────────────────────────────


async def _cmd_start() -> int:
    manager = _build_manager()
    print(f"{_CYAN}LEAP OS Host{_RESET}")
    try:
        status = await manager.start()
        _ok(f"Started (PID {status.pid})")
        return 0
    except FileNotFoundError as exc:
        _fail(f"Cannot start: {exc}")
        _info("Run 'leap host install' first to deploy the host binary.")
        return 1
    except TimeoutError as exc:
        _fail(f"Start timeout: {exc}")
        return 1
    except OSError as exc:
        _fail(f"Start failed: {exc}")
        return 1


async def _cmd_stop() -> int:
    manager = _build_manager()
    print(f"{_CYAN}LEAP OS Host{_RESET}")
    stopped = await manager.stop()
    if stopped:
        _ok("Stopped")
    else:
        _warn("Host was not running")
    return 0


async def _cmd_restart() -> int:
    manager = _build_manager()
    print(f"{_CYAN}LEAP OS Host{_RESET}")
    try:
        status = await manager.restart()
        _ok(f"Restarted (PID {status.pid})")
        return 0
    except FileNotFoundError as exc:
        _fail(f"Cannot restart: {exc}")
        return 1
    except TimeoutError as exc:
        _fail(f"Restart timeout: {exc}")
        return 1
    except OSError as exc:
        _fail(f"Restart failed: {exc}")
        return 1


async def _cmd_status() -> int:
    manager = _build_manager()
    status: HostStatus = manager.status()

    print(f"{_CYAN}LEAP OS Host{_RESET}")

    # State line
    if status.state == HostState.RUNNING:
        uptime = _format_uptime(status.uptime_seconds)
        print(f"  Status:       {_GREEN}●{_RESET} Running (PID {status.pid}, uptime {uptime})")
    elif status.state == HostState.STALE:
        print(f"  Status:       {_YELLOW}●{_RESET} Stale (PID file exists but process gone)")
    else:
        print(f"  Status:       {_DIM}○{_RESET} Stopped")

    # Socket
    socket_status = f"{_GREEN}exists{_RESET}" if status.socket_alive else f"{_DIM}absent{_RESET}"
    settings = load_config()
    print(f"  Socket:       {settings.host_socket} ({socket_status})")

    # Bundle
    if status.bundle_path:
        print(f"  Bundle:       {status.bundle_path}")
    else:
        print(f"  Bundle:       {_DIM}not installed{_RESET}")

    # Permissions
    perms = status.permissions
    print("  Permissions:")
    print(f"    Accessibility:     {_perm_icon(perms.get('accessibility'))}")
    print(f"    Screen Recording:  {_perm_icon(perms.get('screen_recording'))}")
    fda = perms.get("full_disk_access")
    if fda is None:
        print(f"    Full Disk Access:  {_DIM}○ Not required{_RESET}")
    else:
        print(f"    Full Disk Access:  {_perm_icon(fda)}")

    return 0


async def _cmd_logs(follow: bool) -> int:
    settings = load_config()
    log_file = settings.host_log_file

    if not log_file.exists():
        _fail(f"Log file not found: {log_file}")
        return 1

    if follow:
        # Use tail -f for streaming
        try:
            proc = subprocess.Popen(
                ["tail", "-f", str(log_file)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return 0
    else:
        # Print last 50 lines
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-50:]:
                print(line)
        except OSError as exc:
            _fail(f"Cannot read log: {exc}")
            return 1
        return 0


async def _cmd_install() -> int:
    manager = _build_manager()
    print(f"{_CYAN}LEAP OS Host — Install{_RESET}")

    # Look for a pre-built binary
    os_host_dir = Path(__file__).resolve().parents[3] / "os_host"
    candidates = [
        os_host_dir / ".build" / "release" / "OSHost",
        os_host_dir / ".build" / "debug" / "OSHost",
    ]

    binary_path: Optional[Path] = None
    for c in candidates:
        if c.exists() and c.is_file():
            binary_path = c
            break

    if binary_path is None:
        _warn("No pre-built binary found. Building with swift build -c release...")
        try:
            result = subprocess.run(
                ["swift", "build", "-c", "release"],
                cwd=str(os_host_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                _fail("Swift build failed:")
                sys.stderr.write(result.stderr)
                return 1
            _ok("Build succeeded")
            binary_path = os_host_dir / ".build" / "release" / "OSHost"
            if not binary_path.exists():
                _fail(f"Expected binary not found at {binary_path}")
                return 1
        except FileNotFoundError:
            _fail("'swift' not found in PATH. Install Xcode Command Line Tools.")
            return 1
        except subprocess.TimeoutExpired:
            _fail("Build timed out (300s)")
            return 1

    _info(f"Source binary: {binary_path}")

    try:
        bundle = manager.install_app(binary_path)
        _ok(f"Installed to {bundle}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        _fail(f"Installation failed: {exc}")
        return 1


async def _cmd_setup() -> int:
    # Step 1: install
    code = await _cmd_install()
    if code != 0:
        return code

    manager = _build_manager()

    # Step 2: register launchd
    print()
    _info("Registering launchd service...")
    try:
        registered = manager.register_launchd()
        if registered:
            _ok("LaunchAgent registered")
        else:
            _warn("LaunchAgent registration skipped (non-macOS or already registered)")
    except FileNotFoundError as exc:
        _fail(f"Cannot register launchd: {exc}")
        return 1
    except Exception as exc:
        _fail(f"LaunchAgent registration failed: {exc}")
        return 1

    # Step 3: permission check & guidance
    print()
    _info("Checking permissions...")
    perms = check_permissions()

    needs_guidance = False
    if perms.accessibility is not True:
        _warn("Accessibility permission not granted")
        needs_guidance = True
    else:
        _ok("Accessibility: granted")

    if perms.screen_recording is not True:
        _warn("Screen Recording permission not granted")
        needs_guidance = True
    else:
        _ok("Screen Recording: granted")

    if needs_guidance:
        print()
        print(f"  {_BOLD}Permission Setup Required{_RESET}")
        _info("LEAP Host needs Accessibility and Screen Recording access.")
        _info("Opening System Settings...")
        print()
        try:
            answer = input("  Open System Settings now? (Y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("", "y", "yes"):
            if perms.accessibility is not True:
                open_accessibility_settings()
                time.sleep(0.5)
            if perms.screen_recording is not True:
                open_screen_recording_settings()
    else:
        _ok("All required permissions granted")

    return 0


async def _cmd_uninstall() -> int:
    manager = _build_manager()
    print(f"{_CYAN}LEAP OS Host — Uninstall{_RESET}")

    try:
        answer = input("  Remove OS Host bundle and launchd service? (y/N) ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer not in ("y", "yes"):
        _info("Cancelled")
        return 0

    removed = manager.uninstall_app()
    if removed:
        _ok("Uninstalled successfully")
    else:
        _warn("No bundle found to remove (already clean)")
    return 0


# ── Entry point ──────────────────────────────────────────────────────────


async def cmd_host(args: argparse.Namespace) -> int:
    """Route to host subcommands."""
    action = getattr(args, "host_action", None)

    if action is None:
        print("Usage: leap host {start|stop|restart|status|logs|install|setup|uninstall|dev}")
        print()
        print("Manage the LEAP OS Host lifecycle.")
        print()
        print("Commands:")
        print("  start       Start the OS Host daemon")
        print("  stop        Gracefully stop the OS Host")
        print("  restart     Restart the OS Host")
        print("  status      Show host status, PID, uptime, permissions")
        print("  logs        View host logs (--follow for streaming)")
        print("  install     Build and deploy the .app bundle")
        print("  setup       Install + register launchd + permission guidance")
        print("  uninstall   Stop, unregister launchd, remove bundle")
        print("  dev         Development mode (auto-rebuild on file changes)")
        return 1

    if action == "start":
        return await _cmd_start()
    elif action == "stop":
        return await _cmd_stop()
    elif action == "restart":
        return await _cmd_restart()
    elif action == "status":
        return await _cmd_status()
    elif action == "logs":
        follow = getattr(args, "follow", False)
        return await _cmd_logs(follow)
    elif action == "install":
        return await _cmd_install()
    elif action == "setup":
        return await _cmd_setup()
    elif action == "uninstall":
        return await _cmd_uninstall()
    elif action == "dev":
        from leapflow.host.dev import cmd_host_dev
        return await cmd_host_dev()
    else:
        _fail(f"Unknown host action: {action}")
        return 1

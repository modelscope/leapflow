"""Platform driver lifecycle management commands — cua-driver + ObservationDaemon.

Manages the cua-driver execution layer and ObservationDaemon background
observers for passive signal collection.
"""

from __future__ import annotations

import argparse
import os
import platform as platform_mod
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from leapflow.config import load_config

# ── ANSI colors ──────────────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[1;36m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}\u2713{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}\u2717{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}!{_RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {_DIM}{msg}{_RESET}")


# ── Helpers ──────────────────────────────────────────────────────────────

_CUA_DRIVER_CMD = os.environ.get("LEAPFLOW_CUA_DRIVER_CMD", "cua-driver")
_CUA_INSTALL_URL = "https://github.com/trycua/cua"


def _daemon_pid_file() -> Path:
    """PID file for the ObservationDaemon background process."""
    settings = load_config()
    return settings.data_dir.expanduser() / "var" / "observation_daemon.pid"


def _daemon_log_file() -> Path:
    """Log file for ObservationDaemon."""
    settings = load_config()
    return settings.data_dir.expanduser() / "var" / "observation_daemon.log"


def _cua_driver_installed() -> bool:
    """Check if cua-driver binary is on PATH."""
    return bool(shutil.which(_CUA_DRIVER_CMD))


def _cua_driver_version() -> Optional[str]:
    """Try to get cua-driver version string."""
    try:
        result = subprocess.run(
            [_CUA_DRIVER_CMD, "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _read_pid_file() -> Optional[int]:
    """Read PID from daemon pid file. Returns None if not present or stale."""
    pid_file = _daemon_pid_file()
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is alive
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        # Stale or invalid PID file
        try:
            pid_file.unlink()
        except OSError:
            pass
        return None


def _write_pid_file(pid: int) -> None:
    """Write PID to daemon pid file."""
    pid_file = _daemon_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def _remove_pid_file() -> None:
    """Remove daemon PID file."""
    pid_file = _daemon_pid_file()
    try:
        pid_file.unlink()
    except OSError:
        pass


# ── Subcommand implementations ──────────────────────────────────────────


async def _cmd_status() -> int:
    """Show cua-driver installation and ObservationDaemon status."""
    print(f"{_CYAN}LEAP Host \u2014 Status{_RESET}")
    print()

    # cua-driver installation
    print(f"  {_BOLD}cua-driver{_RESET}")
    if _cua_driver_installed():
        version = _cua_driver_version()
        version_str = version if version else "installed (version unknown)"
        _ok(f"Installed: {version_str}")
        _info(f"Command: {shutil.which(_CUA_DRIVER_CMD)}")
    else:
        _fail("Not installed")
        _info(f"Install: {_CUA_INSTALL_URL}")
    print()

    # ObservationDaemon status
    print(f"  {_BOLD}ObservationDaemon{_RESET}")
    pid = _read_pid_file()
    if pid is not None:
        _ok(f"Running (PID {pid})")
        log_file = _daemon_log_file()
        if log_file.exists():
            _info(f"Log: {log_file}")
    else:
        _info("Stopped")
        _info("Start with: leap host start")

    return 0


async def _cmd_start() -> int:
    """Start ObservationDaemon as a background process."""
    print(f"{_CYAN}LEAP Host \u2014 Start{_RESET}")

    # Check if already running
    pid = _read_pid_file()
    if pid is not None:
        _warn(f"ObservationDaemon already running (PID {pid})")
        return 0

    # Check cua-driver availability
    if not _cua_driver_installed():
        _fail("cua-driver not installed \u2014 cannot start observation daemon")
        _info("Install cua-driver first: leap host install")
        _info(f"Or manually: {_CUA_INSTALL_URL}")
        return 1

    # Spawn the daemon as a background subprocess
    log_file = _daemon_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Module-based runner for the daemon process
    daemon_script = (
        "import asyncio, logging, signal, sys; "
        "logging.basicConfig(level=logging.INFO, "
        "format='%(asctime)s %(name)s %(levelname)s %(message)s'); "
        "from leapflow.platform.event_bus import EventBus; "
        "from leapflow.platform.observers import ObservationDaemon, ObserverConfig; "
        "from leapflow.memory.providers.episodic import EpisodicMemoryProvider; "
        "from leapflow.memory.providers.working import WorkingMemoryProvider; "
        "from leapflow.config import load_config; "
        "settings = load_config(); "
        "episodic = EpisodicMemoryProvider("
        "ttl=settings.memory_episodic_ttl_s, "
        "max_entries=settings.memory_episodic_max_entries); "
        "working = WorkingMemoryProvider("
        "max_tokens=settings.memory_working_max_tokens); "
        "bus = EventBus(immediate=episodic, working=working); "
        "daemon = ObservationDaemon(bus=bus, config=ObserverConfig()); "
        "loop = asyncio.new_event_loop(); "
        "asyncio.set_event_loop(loop); "
        "loop.run_until_complete(daemon.start()); "
        "print('ObservationDaemon started', flush=True); "
        "stop_event = asyncio.Event(); "
        "def _signal_handler(*a): stop_event.set(); "
        "signal.signal(signal.SIGTERM, _signal_handler); "
        "signal.signal(signal.SIGINT, _signal_handler); "
        "loop.run_until_complete(stop_event.wait()); "
        "loop.run_until_complete(daemon.stop()); "
        "print('ObservationDaemon stopped', flush=True)"
    )

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-c", daemon_script],
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Wait briefly to confirm startup
    time.sleep(1.0)
    if proc.poll() is not None:
        _fail("ObservationDaemon failed to start (exited immediately)")
        _info(f"Check logs: {log_file}")
        return 1

    _write_pid_file(proc.pid)
    _ok(f"ObservationDaemon started (PID {proc.pid})")
    _info(f"Log: {log_file}")
    return 0


async def _cmd_stop() -> int:
    """Stop ObservationDaemon background process."""
    print(f"{_CYAN}LEAP Host \u2014 Stop{_RESET}")

    pid = _read_pid_file()
    if pid is None:
        _warn("ObservationDaemon is not running")
        return 0

    # Send SIGTERM for graceful shutdown
    try:
        os.kill(pid, signal.SIGTERM)
        _info(f"Sent SIGTERM to PID {pid}, waiting for shutdown...")
        # Wait up to 5s for process to exit
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            # Force kill if still alive
            _warn("Process did not exit gracefully, sending SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    except OSError:
        _info("Process already gone")

    _remove_pid_file()
    _ok("ObservationDaemon stopped")
    return 0


async def _cmd_doctor() -> int:
    """Run cua-driver health check: connectivity test via MCP."""
    print(f"{_CYAN}LEAP Host \u2014 Doctor{_RESET}")
    print()

    # Step 1: Check binary
    print(f"  {_BOLD}1. Binary check{_RESET}")
    if not _cua_driver_installed():
        _fail("cua-driver not found on PATH")
        _info(f"Install from: {_CUA_INSTALL_URL}")
        _info("Or run: leap host install")
        return 1
    version = _cua_driver_version()
    _ok(f"cua-driver binary found: {shutil.which(_CUA_DRIVER_CMD)}")
    if version:
        _ok(f"Version: {version}")
    print()

    # Step 2: MCP session connectivity
    print(f"  {_BOLD}2. MCP connectivity{_RESET}")
    _info("Starting MCP session...")

    try:
        from leapflow.platform.cua_client import CuaDriverClient

        client = CuaDriverClient(call_timeout=10.0)
        client.start()
        _ok("MCP session established")

        # Step 3: Ping test (list_apps as health probe)
        print()
        print(f"  {_BOLD}3. Ping test{_RESET}")
        _info("Sending probe (list_apps)...")
        result = client._session.call_tool_sync("list_apps", {}, timeout=5.0)
        if result.get("isError"):
            _warn("Probe returned error (non-fatal)")
        else:
            _ok("Ping successful \u2014 cua-driver responding")

        # Step 4: Capability discovery
        print()
        print(f"  {_BOLD}4. Capabilities{_RESET}")
        tools = client._session.available_tools
        if tools:
            _ok(f"Tools available: {len(tools)}")
            for name in sorted(tools.keys()):
                _info(f"  \u2022 {name}")
        else:
            _warn("No tools discovered")

        cap_version = client._session.capability_version
        if cap_version:
            _info(f"Capability version: {cap_version}")

        client.stop()
        _ok("Session closed cleanly")
        print()
        _ok("All checks passed \u2014 cua-driver is healthy")
        return 0

    except Exception as exc:
        _fail(f"Health check failed: {exc}")
        _info("Ensure cua-driver is properly installed and accessible.")
        _info(f"Documentation: {_CUA_INSTALL_URL}")
        return 1


async def _cmd_install() -> int:
    """Install cua-driver via upstream installation script."""
    print(f"{_CYAN}LEAP Host \u2014 Install cua-driver{_RESET}")
    print()

    if _cua_driver_installed():
        version = _cua_driver_version()
        _ok(f"cua-driver already installed: {shutil.which(_CUA_DRIVER_CMD)}")
        if version:
            _info(f"Version: {version}")
        _info("To upgrade, use: pip install --upgrade cua-driver")
        return 0

    system = platform_mod.system().lower()

    if system == "darwin":
        _info("Installing cua-driver for macOS...")
        _info("Running: pip install cua-driver")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "cua-driver"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                _ok("cua-driver installed successfully")
                _info("Verify with: leap host doctor")
                return 0
            else:
                _fail("pip install failed:")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        _info(f"  {line}")
                _info("Manual install: pip install cua-driver")
                _info(f"Or visit: {_CUA_INSTALL_URL}")
                return 1
        except subprocess.TimeoutExpired:
            _fail("Installation timed out (120s)")
            return 1
        except FileNotFoundError:
            _fail("pip not found. Ensure Python is properly installed.")
            return 1

    elif system == "windows":
        _info("Installing cua-driver for Windows...")
        _info("Running: pip install cua-driver")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "cua-driver"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                _ok("cua-driver installed successfully")
                _info("Verify with: leap host doctor")
                return 0
            else:
                _fail("pip install failed:")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        _info(f"  {line}")
                return 1
        except subprocess.TimeoutExpired:
            _fail("Installation timed out (120s)")
            return 1

    elif system == "linux":
        _info("Installing cua-driver for Linux...")
        _info("Running: pip install cua-driver")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "cua-driver"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                _ok("cua-driver installed successfully")
                _info("Verify with: leap host doctor")
                return 0
            else:
                _fail("pip install failed")
                _info("Manual install: pip install cua-driver")
                _info(f"Or visit: {_CUA_INSTALL_URL}")
                return 1
        except subprocess.TimeoutExpired:
            _fail("Installation timed out (120s)")
            return 1

    else:
        _fail(f"Unsupported platform: {system}")
        _info(f"Please install manually: {_CUA_INSTALL_URL}")
        return 1


# ── Entry point ──────────────────────────────────────────────────────────


async def cmd_host(args: argparse.Namespace) -> int:
    """Route to host subcommands."""
    action = getattr(args, "host_action", None)

    if action is None:
        print("Usage: leap host {start|stop|status|doctor|install}")
        print()
        print("Manage cua-driver and ObservationDaemon lifecycle.")
        print()
        print("Commands:")
        print("  start       Start the ObservationDaemon (background observers)")
        print("  stop        Stop the ObservationDaemon")
        print("  status      Show cua-driver and daemon status")
        print("  doctor      Run cua-driver connectivity health check")
        print("  install     Install cua-driver")
        return 1

    if action == "start":
        return await _cmd_start()
    elif action == "stop":
        return await _cmd_stop()
    elif action == "status":
        return await _cmd_status()
    elif action == "doctor":
        return await _cmd_doctor()
    elif action == "install":
        return await _cmd_install()
    else:
        _fail(f"Unknown host action: {action}")
        return 1

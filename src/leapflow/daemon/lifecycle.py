"""Daemon lifecycle management — PID files, lock files, health checks.

Handles the leapd daemon lifecycle:

1. **Discovery** — check if a daemon is running for a given profile
2. **Lock acquisition** — ``fcntl.flock`` on ``leapd.lock`` for leader election
3. **PID file** — ``leapd.pid`` tracks the daemon process
4. **Health check** — probe ``leapd.sock`` for liveness
5. **Stale cleanup** — detect and remove orphaned PID/socket files

Usage::

    info = DaemonInfo.discover(runtime_dir)
    if info.is_healthy:
        # connect to existing daemon
    else:
        # acquire lock and spawn new daemon
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaemonInfo:
    """Snapshot of daemon state for a profile."""
    pid: Optional[int]
    sock_path: Optional[Path]
    start_time: Optional[float]
    is_running: bool
    is_healthy: bool

    @classmethod
    def discover(cls, runtime_dir: Path) -> DaemonInfo:
        """Probe the runtime directory to determine daemon state."""
        pid = _read_pid(runtime_dir / "leapd.pid")
        sock_path = runtime_dir / "leapd.sock"
        meta = _read_meta(runtime_dir / "leapd.json")
        start_time = meta.get("start_time") if meta else None

        is_running = pid is not None and _process_alive(pid)
        is_healthy = is_running and sock_path.exists() and _sock_healthy(sock_path)

        return cls(
            pid=pid,
            sock_path=sock_path if sock_path.exists() else None,
            start_time=start_time,
            is_running=is_running,
            is_healthy=is_healthy,
        )

    @property
    def uptime_s(self) -> Optional[float]:
        if self.start_time is None:
            return None
        return time.time() - self.start_time

    def format_status(self) -> str:
        """Human-readable status string."""
        if self.is_healthy:
            uptime = self.uptime_s
            up_str = _format_duration(uptime) if uptime else "unknown"
            return f"leapd running (pid={self.pid}, uptime={up_str})"
        if self.is_running:
            return f"leapd running but unhealthy (pid={self.pid})"
        if self.pid is not None:
            return f"leapd stale (pid={self.pid} not running)"
        return "leapd not running"


@dataclass(frozen=True)
class StopDaemonResult:
    """Outcome of a bounded daemon stop transaction."""

    pid: Optional[int]
    stopped: bool
    signal_sent: bool = False
    forced: bool = False
    stale_cleaned: bool = False
    timed_out: bool = False
    error: str = ""


class DaemonLock:
    """Advisory file lock for daemon leader election.

    Usage::

        lock = DaemonLock(runtime_dir / "leapd.lock")
        if lock.acquire():
            # we are the leader — spawn daemon
            ...
            lock.release()
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Try to acquire the lock (non-blocking). Returns True if acquired."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, IOError):
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> DaemonLock:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def write_pid_file(runtime_dir: Path, pid: Optional[int] = None) -> None:
    """Write the daemon PID file and metadata."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    actual_pid = pid or os.getpid()
    (runtime_dir / "leapd.pid").write_text(str(actual_pid))

    meta = {
        "pid": actual_pid,
        "start_time": time.time(),
        "version": "1",
    }
    (runtime_dir / "leapd.json").write_text(json.dumps(meta))
    logger.info("daemon: wrote pid=%d to %s", actual_pid, runtime_dir / "leapd.pid")


def cleanup_runtime_dir(runtime_dir: Path) -> None:
    """Remove daemon runtime files (PID, socket, metadata)."""
    for name in ("leapd.pid", "leapd.json", "leapd.sock"):
        path = runtime_dir / name
        path.unlink(missing_ok=True)
    logger.info("daemon: cleaned up %s", runtime_dir)


def cleanup_stale(runtime_dir: Path) -> bool:
    """Detect and clean up stale daemon files.

    Returns True if stale files were cleaned.
    """
    pid = _read_pid(runtime_dir / "leapd.pid")
    if pid is None:
        return False

    if _process_alive(pid):
        return False

    logger.info("daemon: stale pid=%d detected, cleaning up", pid)
    cleanup_runtime_dir(runtime_dir)
    return True


def send_signal(runtime_dir: Path, sig: int = signal.SIGTERM) -> bool:
    """Send a signal to the running daemon. Returns True if sent."""
    pid = _read_pid(runtime_dir / "leapd.pid")
    if pid is None or not _process_alive(pid):
        return False
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


def stop_daemon(
    runtime_dir: Path,
    *,
    timeout_s: float = 10.0,
    force: bool = False,
    grace_timeout_s: float = 0.0,
    poll_interval_s: float = 0.1,
    force_timeout_s: float = 2.0,
    on_progress: Callable[[str], None] | None = None,
) -> StopDaemonResult:
    """Stop leapd as a bounded transaction and verify final state.

    ``on_progress`` receives short, human-readable step messages so callers can
    keep the user informed while a slow shutdown escalates from graceful wait to
    SIGTERM to SIGKILL.
    """
    def _notify(message: str) -> None:
        if on_progress is not None:
            try:
                on_progress(message)
            except Exception:  # noqa: BLE001 - progress reporting must never break stop
                pass

    info = DaemonInfo.discover(runtime_dir)
    pid = info.pid
    if not info.is_running:
        stale_cleaned = cleanup_stale(runtime_dir) if pid is not None else False
        return StopDaemonResult(pid=pid, stopped=True, stale_cleaned=stale_cleaned)

    deadline = time.time() + max(0.1, timeout_s)
    interval = max(0.01, poll_interval_s)
    if grace_timeout_s > 0:
        _notify(f"Waiting up to {grace_timeout_s:.0f}s for graceful shutdown...")
        grace_deadline = min(deadline, time.time() + grace_timeout_s)
        if _wait_until_stopped(runtime_dir, deadline=grace_deadline, interval_s=interval):
            stale_cleaned = cleanup_stale(runtime_dir)
            return StopDaemonResult(pid=pid, stopped=True, stale_cleaned=stale_cleaned)

    _notify(f"Sending SIGTERM to pid {pid}...")
    signal_sent = send_signal(runtime_dir, signal.SIGTERM)
    if not signal_sent and not DaemonInfo.discover(runtime_dir).is_running:
        stale_cleaned = cleanup_stale(runtime_dir)
        return StopDaemonResult(pid=pid, stopped=True, stale_cleaned=stale_cleaned)
    if not signal_sent:
        return StopDaemonResult(pid=pid, stopped=False, error="failed to send SIGTERM")

    if _wait_until_stopped(runtime_dir, deadline=deadline, interval_s=interval):
        stale_cleaned = cleanup_stale(runtime_dir)
        return StopDaemonResult(pid=pid, stopped=True, signal_sent=True, stale_cleaned=stale_cleaned)

    forced = False
    if force:
        _notify(f"Still running; escalating to SIGKILL (force) and waiting up to {force_timeout_s:.0f}s...")
        forced = send_signal(runtime_dir, signal.SIGKILL)
        kill_deadline = time.time() + max(0.1, force_timeout_s)
        if forced and _wait_until_stopped(runtime_dir, deadline=kill_deadline, interval_s=interval):
            stale_cleaned = cleanup_stale(runtime_dir)
            return StopDaemonResult(
                pid=pid,
                stopped=True,
                signal_sent=True,
                forced=True,
                stale_cleaned=stale_cleaned,
            )

    return StopDaemonResult(
        pid=pid,
        stopped=False,
        signal_sent=True,
        forced=forced,
        timed_out=True,
        error="timed out waiting for leapd to stop",
    )


def spawn_daemon(settings: object, *, mock_host: bool = False) -> subprocess.Popen[bytes]:
    """Spawn a detached leapd process for the active environment."""
    runtime_dir = settings.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "leapd.log"
    command = [sys.executable, "-m", "leapflow"]
    if mock_host:
        command.append("--mock-host")
    command.extend(["daemon", "serve", "--internal"])
    log_file = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_file.close()
    logger.info("daemon: spawned pid=%s log=%s", proc.pid, log_path)
    return proc


def wait_ready(runtime_dir: Path, *, timeout_s: float = 30.0, interval_s: float = 0.1) -> DaemonInfo:
    """Wait until the daemon socket becomes healthy or timeout expires."""
    deadline = time.time() + max(0.1, timeout_s)
    last = DaemonInfo.discover(runtime_dir)
    while time.time() < deadline:
        last = DaemonInfo.discover(runtime_dir)
        if last.is_healthy:
            return last
        time.sleep(max(0.01, interval_s))
    return last


def _wait_until_stopped(runtime_dir: Path, *, deadline: float, interval_s: float) -> bool:
    while time.time() < deadline:
        if not DaemonInfo.discover(runtime_dir).is_running:
            return True
        time.sleep(interval_s)
    return not DaemonInfo.discover(runtime_dir).is_running


# ── Internal helpers ──

def _read_pid(path: Path) -> Optional[int]:
    """Read PID from file, returning None if missing/invalid."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_meta(path: Path) -> Optional[dict]:
    """Read daemon metadata JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    """Return True if the process with the given PID is still running.

    A daemon spawned by the current process lingers as an unreaped zombie after
    it exits (including after ``SIGKILL``); ``os.kill(pid, 0)`` still succeeds
    for zombies, which would make a successful stop look like a failure. Reap
    our own exited children first so a terminated daemon is correctly reported
    as gone. For processes that are not our children, reaping is a no-op and we
    fall back to the ``os.kill`` liveness probe.
    """
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False  # our child has exited and was just reaped
    except (ChildProcessError, OSError):
        pass  # not our child, or already reaped by subprocess/init
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False
    return True


def _sock_healthy(sock_path: Path) -> bool:
    """Quick health check by connecting to the Unix socket."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(str(sock_path))
        s.close()
        return True
    except (OSError, socket.timeout):
        return False


def _format_duration(seconds: Optional[float]) -> str:
    """Format seconds into human-readable duration."""
    if seconds is None:
        return "unknown"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h{m}m"

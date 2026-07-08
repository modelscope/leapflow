"""Daemon lifecycle management — PID files, lock files, health checks.

Handles the leapd daemon lifecycle:

1. **Discovery** — check if a daemon is running for a given profile
2. **Lock acquisition** — ``fcntl.flock`` on ``leapd.lock`` for leader election
3. **PID file** — ``leapd.pid`` tracks the daemon process
4. **Health check** — probe ``leapd.sock`` for liveness
5. **Stale cleanup** — detect and remove orphaned PID/socket files

Usage::

    info = DaemonInfo.discover(run_dir)
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    def discover(cls, run_dir: Path) -> DaemonInfo:
        """Probe the run directory to determine daemon state."""
        pid = _read_pid(run_dir / "leapd.pid")
        sock_path = run_dir / "leapd.sock"
        meta = _read_meta(run_dir / "leapd.json")
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


class DaemonLock:
    """Advisory file lock for daemon leader election.

    Usage::

        lock = DaemonLock(run_dir / "leapd.lock")
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


def write_pid_file(run_dir: Path, pid: Optional[int] = None) -> None:
    """Write the daemon PID file and metadata."""
    run_dir.mkdir(parents=True, exist_ok=True)
    actual_pid = pid or os.getpid()
    (run_dir / "leapd.pid").write_text(str(actual_pid))

    meta = {
        "pid": actual_pid,
        "start_time": time.time(),
        "version": "1",
    }
    (run_dir / "leapd.json").write_text(json.dumps(meta))
    logger.info("daemon: wrote pid=%d to %s", actual_pid, run_dir / "leapd.pid")


def cleanup_run_dir(run_dir: Path) -> None:
    """Remove daemon runtime files (PID, socket, metadata)."""
    for name in ("leapd.pid", "leapd.json", "leapd.sock"):
        path = run_dir / name
        path.unlink(missing_ok=True)
    logger.info("daemon: cleaned up %s", run_dir)


def cleanup_stale(run_dir: Path) -> bool:
    """Detect and clean up stale daemon files.

    Returns True if stale files were cleaned.
    """
    pid = _read_pid(run_dir / "leapd.pid")
    if pid is None:
        return False

    if _process_alive(pid):
        return False

    logger.info("daemon: stale pid=%d detected, cleaning up", pid)
    cleanup_run_dir(run_dir)
    return True


def send_signal(run_dir: Path, sig: int = signal.SIGTERM) -> bool:
    """Send a signal to the running daemon. Returns True if sent."""
    pid = _read_pid(run_dir / "leapd.pid")
    if pid is None or not _process_alive(pid):
        return False
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


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
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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

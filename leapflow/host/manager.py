"""OS Host lifecycle manager.

Stateless coordinator over an external Swift OS Host process. All runtime
information is reconstructed from the filesystem (PID file, socket file,
log file) so the manager survives Brain restarts.

The manager is platform-agnostic at the API surface; macOS-specific bits
(LaunchAgent, .app bundle layout) live in :mod:`launchd` and are only
exercised on Darwin.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import shutil
import signal
import socket as _socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from leapflow.host.launchd import LaunchdService
from leapflow.host.permissions import check_permissions

logger = logging.getLogger(__name__)


# ─── Status types ───────────────────────────────────────────────────────


class HostState(str, Enum):
    """Discrete states for the OS Host process."""

    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"  # PID file present but process is gone


@dataclass(frozen=True)
class HostStatus:
    """Immutable snapshot of host runtime state."""

    state: HostState
    pid: Optional[int]
    uptime_seconds: Optional[float]
    socket_alive: bool
    permissions: Dict[str, Optional[bool]] = field(default_factory=dict)
    bundle_path: Optional[Path] = None

    @property
    def is_running(self) -> bool:
        return self.state == HostState.RUNNING


# ─── Constants ──────────────────────────────────────────────────────────

_DEFAULT_BUNDLE_NAME = "LeapHost.app"
_DEFAULT_EXECUTABLE_NAME = "LeapHost"
_POLL_INTERVAL_S = 0.1


# ─── Manager ────────────────────────────────────────────────────────────


class HostManager:
    """OS Host lifecycle coordinator.

    The manager has no in-memory state beyond its configuration: every query
    re-derives the answer from PID/socket files. This makes it safe to use
    from multiple CLI invocations concurrently.
    """

    def __init__(
        self,
        *,
        host_root: Path,
        host_socket: Path,
        pid_file: Path,
        log_file: Path,
        bundle_id: str = "com.leapflow.host",
        bundle_name: str = _DEFAULT_BUNDLE_NAME,
        executable_name: str = _DEFAULT_EXECUTABLE_NAME,
    ) -> None:
        self.host_root = Path(host_root).expanduser()
        self.host_socket = Path(host_socket).expanduser()
        self.pid_file = Path(pid_file).expanduser()
        self.log_file = Path(log_file).expanduser()
        self.bundle_id = bundle_id
        self.bundle_name = bundle_name
        self.executable_name = executable_name

    # ── Paths ────────────────────────────────────────────────────────

    @property
    def app_bundle_path(self) -> Path:
        return self.host_root / self.bundle_name

    @property
    def bundle_executable(self) -> Path:
        return self.app_bundle_path / "Contents" / "MacOS" / self.executable_name

    @property
    def info_plist_path(self) -> Path:
        return self.app_bundle_path / "Contents" / "Info.plist"

    # ── Status queries ───────────────────────────────────────────────

    def _read_pid(self) -> Optional[int]:
        try:
            raw = self.pid_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("cannot read pid file %s: %s", self.pid_file, exc)
            return None
        if not raw:
            return None
        try:
            return int(raw.splitlines()[0].strip())
        except ValueError:
            logger.warning("pid file %s contains garbage: %r", self.pid_file, raw)
            return None

    @staticmethod
    def _process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # The process exists but is owned by another user.
            return True
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            return True
        return True

    def _socket_exists(self) -> bool:
        try:
            return self.host_socket.exists()
        except OSError:
            return False

    def _process_uptime(self, pid: int) -> Optional[float]:
        """Best-effort uptime via PID-file mtime (portable, low cost)."""
        try:
            mtime = self.pid_file.stat().st_mtime
        except OSError:
            return None
        delta = time.time() - mtime
        return max(0.0, delta)

    def is_running(self) -> bool:
        """True iff the host is reachable.

        Checks PID file first; falls back to a socket connectivity probe
        so externally managed hosts (e.g. ``leap host dev``) are detected
        even without a PID file.

        Side effect: removes a stale PID file when the process is gone.
        """
        pid = self._read_pid()
        if pid is not None:
            if self._process_alive(pid):
                return self._socket_exists()
            # PID is stale, but the socket might belong to an externally
            # managed host (e.g. dev server restarted with a new PID).
            # Check socket before concluding nothing is running.
            self._cleanup_stale()
            if self._socket_exists() and self._socket_responsive(retries=1, timeout=2.0):
                logger.info("PID stale but socket responsive — external host detected")
                return True
            return False
        # No PID file — rely on socket probe with retry for reliability.
        if not self._socket_exists():
            return False
        return self._socket_responsive(retries=1, timeout=2.0)

    def status(self) -> HostStatus:
        pid = self._read_pid()
        socket_alive = self._socket_exists()
        bundle = self.app_bundle_path if self.app_bundle_path.exists() else None
        perms = check_permissions().to_dict()

        if pid is None:
            # No PID file — check if something is listening on the socket
            # (e.g. an externally managed dev server).
            if socket_alive and self._socket_responsive():
                return HostStatus(
                    state=HostState.RUNNING,
                    pid=None,
                    uptime_seconds=None,
                    socket_alive=True,
                    permissions=perms,
                    bundle_path=bundle,
                )
            return HostStatus(
                state=HostState.STOPPED,
                pid=None,
                uptime_seconds=None,
                socket_alive=socket_alive,
                permissions=perms,
                bundle_path=bundle,
            )
        if not self._process_alive(pid):
            return HostStatus(
                state=HostState.STALE,
                pid=pid,
                uptime_seconds=None,
                socket_alive=socket_alive,
                permissions=perms,
                bundle_path=bundle,
            )
        return HostStatus(
            state=HostState.RUNNING,
            pid=pid,
            uptime_seconds=self._process_uptime(pid),
            socket_alive=socket_alive,
            permissions=perms,
            bundle_path=bundle,
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self, timeout: float = 10.0) -> HostStatus:
        """Spawn the host as a detached daemon and wait for the socket.

        Raises:
            FileNotFoundError: if no executable can be located.
            TimeoutError: if the socket does not appear within ``timeout``.
        """
        if self.is_running():
            logger.info("Host already running; skipping start")
            return self.status()

        # Clean up stale PID file from a previous crash.
        self._cleanup_stale()

        # If a socket file exists but nothing responds (thorough check with
        # retries), remove it so the new daemon can bind. This is safe because
        # is_running() above already confirmed nothing is alive.
        if self._socket_exists() and not self._socket_responsive(retries=2, timeout=2.0):
            logger.info("Removing unresponsive stale socket before daemon start")
            try:
                self.host_socket.unlink()
            except (FileNotFoundError, OSError):
                pass

        executable = self._resolve_executable()
        if executable is None:
            raise FileNotFoundError(
                f"OS Host executable not found under {self.host_root}; "
                f"run install_app() first or provide a bundled binary."
            )

        for parent in (self.host_socket.parent, self.pid_file.parent, self.log_file.parent):
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("cannot create %s: %s", parent, exc)

        cmd: List[str] = [
            str(executable),
            "--daemon",
            "--socket", str(self.host_socket),
            "--pid-file", str(self.pid_file),
            "--log-file", str(self.log_file),
        ]

        # Detach so the host outlives the parent CLI process.
        try:
            log_fh = open(self.log_file, "ab", buffering=0)
        except OSError as exc:
            raise OSError(f"cannot open log file {self.log_file}: {exc}") from exc

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                close_fds=True,
                cwd=str(executable.parent),
            )
        except OSError as exc:
            log_fh.close()
            raise OSError(f"failed to spawn host {cmd[0]}: {exc}") from exc
        finally:
            # Popen dups the fd; we can release ours.
            try:
                log_fh.close()
            except OSError:
                pass

        logger.info("Spawned OS Host pid=%d via %s", proc.pid, executable)

        # Wait for socket to appear.
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if self._socket_exists() and self._process_alive(proc.pid):
                return self.status()
            if proc.poll() is not None:
                raise RuntimeError(
                    f"OS Host exited prematurely with code {proc.returncode}; "
                    f"see {self.log_file}"
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

        raise TimeoutError(
            f"OS Host socket {self.host_socket} did not appear within {timeout}s"
        )

    async def stop(self, timeout: float = 10.0) -> bool:
        """Send SIGTERM, await graceful exit, escalate to SIGKILL on timeout.

        Returns True if the host is gone after the call (regardless of which
        signal succeeded), False if the PID file was missing entirely.
        """
        pid = self._read_pid()
        if pid is None:
            self._cleanup_stale()
            return False

        if not self._process_alive(pid):
            self._cleanup_stale()
            return True

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            self._cleanup_stale()
            return True
        except PermissionError as exc:
            logger.warning("permission denied stopping pid %d: %s", pid, exc)
            return False

        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            if not self._process_alive(pid):
                self._cleanup_stale()
                return True
            await asyncio.sleep(_POLL_INTERVAL_S)

        # Escalate.
        logger.warning("SIGTERM timed out; escalating to SIGKILL on pid=%d", pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            self._cleanup_stale()
            return True
        except PermissionError:
            return False

        # Brief grace window for the kernel to reap.
        for _ in range(20):
            if not self._process_alive(pid):
                self._cleanup_stale()
                return True
            await asyncio.sleep(_POLL_INTERVAL_S)

        return not self._process_alive(pid)

    async def restart(self, *, stop_timeout: float = 10.0, start_timeout: float = 10.0) -> HostStatus:
        await self.stop(timeout=stop_timeout)
        return await self.start(timeout=start_timeout)

    async def ensure_running(self, timeout: float = 5.0) -> HostStatus:
        """Idempotent: start the host iff it is not already up."""
        if self.is_running():
            return self.status()
        return await self.start(timeout=timeout)

    # ── App bundle ───────────────────────────────────────────────────

    def install_app(self, binary_path: Path) -> Path:
        """Lay out a minimal macOS .app bundle around ``binary_path``.

        Idempotent: re-installation overwrites the executable in place. The
        binary is **copied** (not symlinked) so removal of the source tree
        does not break the bundle.
        """
        binary_path = Path(binary_path).expanduser()
        if not binary_path.exists():
            raise FileNotFoundError(f"binary not found: {binary_path}")
        if not binary_path.is_file():
            raise ValueError(f"binary is not a file: {binary_path}")

        bundle = self.app_bundle_path
        contents = bundle / "Contents"
        macos_dir = contents / "MacOS"
        resources_dir = contents / "Resources"

        for d in (macos_dir, resources_dir):
            d.mkdir(parents=True, exist_ok=True)

        target = self.bundle_executable
        try:
            shutil.copy2(binary_path, target)
        except OSError as exc:
            raise OSError(f"cannot copy {binary_path} → {target}: {exc}") from exc

        try:
            mode = target.stat().st_mode
            target.chmod(mode | 0o111)
        except OSError as exc:
            logger.warning("cannot mark %s executable: %s", target, exc)

        self._write_info_plist()
        logger.info("Installed app bundle at %s", bundle)
        return bundle

    def uninstall_app(self) -> bool:
        """Stop, unregister launchd, and remove the bundle directory."""
        try:
            asyncio.get_event_loop().run_until_complete(self.stop())
        except RuntimeError:
            # No running event loop; spin a private one.
            try:
                asyncio.run(self.stop())
            except Exception as exc:
                logger.debug("stop during uninstall failed: %s", exc)
        except Exception as exc:
            logger.debug("stop during uninstall failed: %s", exc)

        # Best-effort launchd cleanup.
        try:
            self.unregister_launchd()
        except Exception as exc:
            logger.debug("launchd unregister during uninstall failed: %s", exc)

        bundle = self.app_bundle_path
        if not bundle.exists():
            return False
        try:
            shutil.rmtree(bundle)
        except OSError as exc:
            logger.warning("cannot remove %s: %s", bundle, exc)
            return False
        logger.info("Removed app bundle %s", bundle)
        return True

    def _write_info_plist(self) -> None:
        """Write a minimal Info.plist so the bundle is recognized by macOS."""
        import plistlib

        info: Dict[str, object] = {
            "CFBundleIdentifier": self.bundle_id,
            "CFBundleName": "LeapHost",
            "CFBundleDisplayName": "LEAP Host",
            "CFBundleExecutable": self.executable_name,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1",
            "LSMinimumSystemVersion": "14.0",
            "LSUIElement": True,  # background agent, no Dock icon
            "NSHighResolutionCapable": True,
        }
        try:
            with open(self.info_plist_path, "wb") as fh:
                plistlib.dump(info, fh)
        except OSError as exc:
            logger.warning("cannot write Info.plist %s: %s", self.info_plist_path, exc)

    # ── launchd ──────────────────────────────────────────────────────

    def _make_launchd_service(self) -> LaunchdService:
        return LaunchdService(
            label=self.bundle_id,
            executable=self.bundle_executable,
            socket_path=self.host_socket,
            pid_file=self.pid_file,
            log_file=self.log_file,
        )

    def register_launchd(self) -> bool:
        """Generate the LaunchAgent plist and load it into launchd."""
        if sys.platform != "darwin":
            logger.warning("register_launchd is a macOS-only operation; skipped")
            return False
        if not self.bundle_executable.exists():
            raise FileNotFoundError(
                f"bundle executable missing: {self.bundle_executable}; install_app() first"
            )
        return self._make_launchd_service().register()

    def unregister_launchd(self) -> bool:
        if sys.platform != "darwin":
            return False
        return self._make_launchd_service().unregister()

    # ── Helpers ──────────────────────────────────────────────────────

    def _resolve_executable(self) -> Optional[Path]:
        """Locate the host executable, preferring the installed .app bundle."""
        candidates = [
            self.bundle_executable,
            self.host_root / self.executable_name,
        ]
        for c in candidates:
            if c.exists() and c.is_file() and os.access(c, os.X_OK):
                return c
        return None

    def _socket_responsive(self, *, retries: int = 1, timeout: float = 2.0) -> bool:
        """True if something is actively listening on the socket.

        Uses retries to avoid false negatives from transient server busyness.
        """
        for attempt in range(retries + 1):
            try:
                s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(str(self.host_socket))
                s.close()
                return True
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                if attempt < retries:
                    time.sleep(0.3)
                continue
        return False

    def _cleanup_stale(self) -> None:
        """Remove a stale PID file. Best-effort.

        Only the PID file is cleaned up. The socket file is NEVER deleted
        here — it belongs to the server process. Deleting a socket file
        while a server is running (but temporarily unresponsive to probes)
        permanently orphans that server. The server is responsible for
        cleaning up its own socket on startup (unlink-before-bind).
        """
        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("cannot remove %s: %s", self.pid_file, exc)

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Spawn and drive a real ``leapd`` subprocess for end-to-end journeys.

Journeys talk to the daemon over its actual Unix-socket RPC, because that is the
only place a whole class of defects is observable: engine template vs. per-session
instance, session identity, workspace binding, cross-process persistence and
pushed runtime metadata all look correct inside one process and break across
two. In-process fixtures cannot see any of it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from leapflow.daemon.client import DaemonClient
from leapflow.daemon.lifecycle import DaemonInfo, cleanup_stale, wait_ready
from leapflow.daemon._transport import get_transport
from leapflow.layout import build_layout

READY_TIMEOUT_S = 60.0
STOP_TIMEOUT_S = 15.0

# A Unix socket path is length-limited by the kernel (104 bytes on macOS, 108 on
# Linux). pytest's tmp_path is deep enough on macOS to blow through it, and the
# resulting failure is a bare OSError from inside the daemon — invisible in the
# test process. Journeys therefore run from a short root, and this bound is
# checked before the process is spawned so the message names the real cause.
MAX_SOCKET_PATH_LEN = 100


def journey_root(journey_id: str, *, prefix: str = "lfj-") -> Path:
    """Return a clean, *deterministic* scratch root for one journey.

    Deterministic rather than random for two reasons, both learned the hard way:

    - A Unix socket path is length-limited by the kernel (104 bytes on macOS).
      pytest's ``tmp_path`` is deep enough to exceed it, and the failure is a bare
      ``OSError`` inside the daemon that the test process never sees.
    - Recorded tool calls embed **absolute** paths chosen by the model. Replaying
      them under a fresh random directory puts those paths outside the new
      workspace, the tool refuses with ``outside_workspace``, and the resulting
      tool result no longer matches what was recorded — so every tool-using
      journey misses its cassette. A stable path makes the recording replayable.

    The directory is removed first, so each run starts from empty state and a
    crashed previous run cannot leak a session or a database into this one.

    Raises:
        ConcurrentJourneyError: Another run of *this* journey is live in the same
            directory. Because the path is deterministic, two concurrent runs of
            one journey would share it; refusing is far better than deleting the
            other run's daemon state and leaving both to fail confusingly.
    """
    base = Path("/tmp")
    if not (base.is_dir() and os.access(base, os.W_OK)):
        base = Path(tempfile.gettempdir())
    root = base / f"{prefix}{journey_id}"

    live = _live_daemon_in(root)
    if live is not None:
        raise ConcurrentJourneyError(
            f"journey {journey_id!r} already has a live daemon (pid {live}) under "
            f"{root}. Its scratch root is deterministic — recorded tool calls embed "
            "absolute paths, so it has to be — which means two runs of the same "
            "journey cannot share a machine. Wait for the other run, or stop that "
            "daemon before retrying."
        )

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _live_daemon_in(root: Path) -> int | None:
    """Return the pid of a healthy daemon under ``root``, or None."""
    for runtime_dir in root.glob("data/profiles/*/runtime"):
        info = DaemonInfo.discover(runtime_dir)
        if info.is_healthy:
            return info.pid
    return None


def hermetic_env(
    *,
    data_dir: Path,
    profile: str,
    llm_base_url: str,
    llm_model: str = "cassette-model",
    llm_api_key: str = "cassette-key",
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a daemon environment with every inherited ``LEAPFLOW_*`` removed.

    The developer's shell almost always exports real credentials and a real data
    directory. Inheriting either would let a journey read or write the user's
    profile, so the whole namespace is dropped before ours is applied.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("LEAPFLOW_")}
    env.update(
        {
            "LEAPFLOW_DATA_DIR": str(data_dir),
            "LEAPFLOW_PROFILE": profile,
            "LEAPFLOW_MOCK_HOST": "1",
            # Copilot can schedule background LLM predictions from ambient desktop
            # signals; journeys only cassette the foreground product contract.
            "LEAPFLOW_COPILOT_ENABLED": "0",
            "LEAPFLOW_LLM_API_KEY": llm_api_key,
            "LEAPFLOW_LLM_BASE_URL": llm_base_url,
            "LEAPFLOW_LLM_MODEL": llm_model,
            "LEAPFLOW_LLM_MAX_RETRIES": "2",
            "LEAPFLOW_LLM_CONTEXT_LENGTH": "32768",
            "LEAPFLOW_LOG_LEVEL": "INFO",
            "LEAPFLOW_DAEMON_LOG_LEVEL": "INFO",
            "LEAPFLOW_DAEMON_MAX_CONCURRENT_TURNS": "3",
            # The aux/VLM providers share the proxy so no journey can reach the
            # network through a secondary client.
            "LEAPFLOW_LLM_AUX_BASE_URL": llm_base_url,
            "LEAPFLOW_LLM_AUX_API_KEY": llm_api_key,
            "LEAPFLOW_VLM_BASE_URL": llm_base_url,
            "LEAPFLOW_VLM_API_KEY": llm_api_key,
        }
    )
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


class LeapdStartupError(RuntimeError):
    """Raised when the daemon never reports a healthy socket."""


class ConcurrentJourneyError(RuntimeError):
    """Raised when another run of the same journey already owns its scratch root."""


@dataclass
class Leapd:
    """A running ``leapd`` subprocess plus the paths and clients to drive it."""

    data_dir: Path
    profile: str
    runtime_dir: Path
    env: dict[str, str]
    profile_layout: Any = None
    process: subprocess.Popen[bytes] | None = None
    _workspaces: dict[str, Path] = field(default_factory=dict)

    @property
    def sock_path(self) -> Path:
        """Unix socket the daemon listens on."""
        return get_transport().readiness_path(self.runtime_dir)

    @property
    def log_path(self) -> Path:
        """Daemon log file, captured from stdout and stderr."""
        return self.runtime_dir / "leapd.log"

    @property
    def audit_log_path(self) -> Path:
        """Recovery audit trail, resolved through the profile layout.

        Read from the layout rather than assembled by hand: managed paths are a
        product contract, and a journey that hardcodes one stops verifying the
        real location the moment the layout moves.
        """
        return Path(self.profile_layout.audit_log_path)

    def client(self, *, timeout_s: float = 120.0) -> DaemonClient:
        """Return an RPC client for this daemon."""
        return DaemonClient(self.sock_path, timeout_s=timeout_s)

    def info(self) -> DaemonInfo:
        """Discover current lifecycle state from the runtime directory."""
        return DaemonInfo.discover(self.runtime_dir)

    def tail_log(self, limit: int = 60) -> str:
        """Return the last ``limit`` log lines, or a note when unavailable.

        Cross-process failures are undiagnosable without this: the assertion
        happens in the test process while the cause was logged in the daemon's.
        """
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return f"(no daemon log at {self.log_path})"
        return "\n".join(lines[-limit:])

    def workspace(self, name: str) -> Path:
        """Create (once) and return an isolated workspace directory."""
        existing = self._workspaces.get(name)
        if existing is not None:
            return existing
        path = self.data_dir.parent / "workspaces" / name
        path.mkdir(parents=True, exist_ok=True)
        self._workspaces[name] = path
        return path

    def stop(self) -> None:
        """Shut the daemon down gracefully, escalating only if it hangs."""
        if self.process is None:
            return
        proc = self.process
        self.process = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=STOP_TIMEOUT_S)
        cleanup_stale(self.runtime_dir)


def start_leapd(
    *,
    root: Path,
    llm_base_url: str,
    profile: str = "default",
    llm_model: str = "cassette-model",
    extra_env: Mapping[str, str] | None = None,
    ready_timeout_s: float = READY_TIMEOUT_S,
) -> Leapd:
    """Start a real leapd process rooted at ``root`` and wait until it is healthy.

    Args:
        root: Scratch directory; the profile tree is created beneath ``root/data``.
        llm_base_url: Cassette-proxy base URL every provider is pointed at.
        profile: Profile id to create and activate.
        llm_model: Model name recorded in cassette fingerprints.
        extra_env: Additional ``LEAPFLOW_*`` overrides for this journey.
        ready_timeout_s: How long to wait for a healthy socket.

    Returns:
        A :class:`Leapd` handle owning the process.

    Raises:
        LeapdStartupError: The socket never became healthy; the daemon log tail
            is included, since the cause is only ever in the child's log.
    """
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    profile_layout = build_layout(data_dir).ensure(profile_id=profile)
    runtime_dir = profile_layout.runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale(runtime_dir)

    sock_path = get_transport().readiness_path(runtime_dir)
    if len(str(sock_path)) > MAX_SOCKET_PATH_LEN:
        raise LeapdStartupError(
            f"daemon socket path is {len(str(sock_path))} bytes, over the "
            f"{MAX_SOCKET_PATH_LEN}-byte Unix socket limit:\n  {sock_path}\n"
            "Start the journey from a shorter root (see journey_root()); a "
            "deep pytest tmp_path cannot host a daemon socket."
        )

    env = hermetic_env(
        data_dir=data_dir,
        profile=profile,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        extra=extra_env,
    )
    daemon = Leapd(
        data_dir=data_dir,
        profile=profile,
        runtime_dir=runtime_dir,
        env=env,
        profile_layout=profile_layout,
    )

    log_file = daemon.log_path.open("ab")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "leapflow", "--mock-host", "daemon", "serve", "--internal"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(root),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_file.close()
    daemon.process = process

    info = wait_ready(runtime_dir, timeout_s=ready_timeout_s)
    if not info.is_healthy:
        exit_code = process.poll()
        daemon.stop()
        raise LeapdStartupError(
            f"leapd never became healthy within {ready_timeout_s:.0f}s "
            f"(exit={exit_code}, socket={daemon.sock_path})\n"
            f"--- leapd log tail ---\n{daemon.tail_log()}"
        )
    return daemon


async def await_for(
    predicate: Any,
    *,
    timeout_s: float = 10.0,
    interval_s: float = 0.05,
    what: str = "condition",
) -> Any:
    """Poll ``predicate`` until it returns a truthy value or the timeout expires.

    Cross-process state (a DuckDB write, a released lease, an exited process)
    becomes visible slightly after the RPC returns, so journeys wait on the
    observable fact rather than sleeping on a guess. The failure names what was
    being waited for and the last value seen.
    """
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        last = await predicate()
        if last:
            return last
        await asyncio.sleep(max(0.01, interval_s))
    raise AssertionError(f"timed out after {timeout_s:.1f}s waiting for {what} (last={last!r})")

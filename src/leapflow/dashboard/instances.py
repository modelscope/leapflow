# Copyright (c) Alibaba, Inc. and its affiliates.
"""Find and stop LeapBoard server processes, including ones no state file names.

The launcher spawns the board server detached (new session), so nothing reaps it
when the TUI or CLI that spawned it exits. Discovery state records only the
*current* instance; any server the launcher failed to retire keeps its port and
memory with no remaining owner. This module is the single place that can see all of
them: it identifies a board server from its own argv (the fixed shape the launcher
spawns), attributes it to a profile through the token-gated ``/api/server-info``
self-report, and stops it with TERM-then-KILL.

Identity is always proven before a signal is sent. A recorded pid may since have
been reused by an unrelated process, so a pid is signaled only when the process
either answers with that pid on its own token (unforgeable without the token) or
carries the board server's exact argv shape.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from leapflow.daemon.lifecycle import DaemonSignal, process_alive

logger = logging.getLogger(__name__)

_PS_TIMEOUT_S = 2.0
_TERM_GRACE_S = 3.0
_KILL_GRACE_S = 2.0
_POLL_S = 0.1


@dataclass(frozen=True)
class BoardServerProcess:
    """One running board server as read from its argv.

    ``owner`` is the discovery-state path the server reports for itself, which binds
    it to one profile under one data dir. It is empty when the server could not be
    asked (old build without the field, wrong token, not answering) -- such a server
    cannot be attributed and is only stopped on an explicit ``--all``.
    """

    pid: int
    bind: str
    port: int
    token: str
    owner: str = ""


@dataclass(frozen=True)
class StopResult:
    """What happened to one server we tried to stop."""

    pid: int
    port: int
    stopped: bool
    detail: str


def is_board_server_argv(argv: list[str]) -> bool:
    """Return whether ``argv`` is a ``leapflow board --serve`` process.

    Matches the shape the launcher spawns (``<python> -m leapflow board --serve``) and
    the console-script form (``leap board --serve``), and nothing looser: a shell that
    merely mentions those words, or an editor with a file of that name open, does not
    qualify.
    """
    if "--serve" not in argv or "board" not in argv:
        return False
    board = argv.index("board")
    if board == 0 or argv.index("--serve") < board:
        return False
    entry = argv[board - 1]
    module_form = entry == "leapflow" and board >= 2 and argv[board - 2] == "-m"
    # A console script runs as ``leap board`` or, read via psutil/proc, as
    # ``<python> /path/bin/leap board`` -- where the script is a path, not a bare word.
    script_form = Path(entry).name in ("leap", "leapflow") and (
        board == 1 or (board == 2 and not module_form and os.sep in entry)
    )
    return module_form or script_form


def _flag_value(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return ""


def _from_argv(pid: int, argv: list[str]) -> BoardServerProcess:
    try:
        port = int(_flag_value(argv, "--port") or 0)
    except ValueError:
        port = 0
    return BoardServerProcess(
        pid=pid,
        bind=_flag_value(argv, "--bind") or "127.0.0.1",
        port=port,
        token=_flag_value(argv, "--token"),
    )


def process_argv(pid: int) -> Optional[list[str]]:
    """Best-effort argv of ``pid``, or None when it cannot be read on this platform.

    Uses ``psutil`` when installed, ``/proc`` on Linux, and ``ps`` elsewhere on POSIX.
    ``ps`` joins argv with spaces, which is lossless for the launcher's own argv (no
    element contains a space); a Python install path that does would only make the
    shape check fail closed.
    """
    try:
        import psutil  # optional dependency; absent in minimal installs
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return list(psutil.Process(pid).cmdline())
        except (psutil.Error, OSError):
            return None
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    if os.name != "posix":
        return None
    try:
        probe = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=_PS_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = probe.stdout.strip()
    return line.split() if probe.returncode == 0 and line else None


def pid_is_board_server(pid: int) -> bool:
    """Return whether ``pid`` is currently a board server process (argv shape)."""
    argv = process_argv(pid)
    return argv is not None and is_board_server_argv(argv)


def _list_processes() -> list[tuple[int, list[str]]]:
    """Every process of the current user with its argv; empty when unreadable."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        uid = os.getuid() if hasattr(os, "getuid") else None
        found: list[tuple[int, list[str]]] = []
        for proc in psutil.process_iter(["pid", "cmdline", "uids"]):
            info = proc.info
            uids = info.get("uids")
            if uid is not None and uids is not None and uids.real != uid:
                continue
            if info.get("cmdline"):
                found.append((int(info["pid"]), list(info["cmdline"])))
        return found
    if os.name != "posix":
        return []
    uid = os.getuid()
    if sys.platform.startswith("linux"):
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != uid:
                    continue
            except OSError:
                continue
            argv = process_argv(int(entry.name))
            if argv:
                found.append((int(entry.name), argv))
        return found
    # Detached servers have no controlling terminal, so the listing must be -ax;
    # ownership is filtered here rather than with flags whose spelling varies.
    try:
        probe = subprocess.run(
            ["ps", "-ax", "-o", "pid=", "-o", "uid=", "-o", "command="],
            capture_output=True, text=True, timeout=_PS_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("dashboard: process listing failed", exc_info=True)
        return []
    found = []
    for line in probe.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1] == str(uid):
            found.append((int(fields[0]), fields[2].split()))
    return found


def discover_board_servers() -> list[BoardServerProcess]:
    """Every board server of the current user, attributed to an owner where possible."""
    from leapflow.dashboard.launcher import fetch_server_info

    me = os.getpid()
    servers = []
    for pid, argv in _list_processes():
        if pid == me or not is_board_server_argv(argv):
            continue
        server = _from_argv(pid, argv)
        info = fetch_server_info(server.bind, server.port, server.token) if server.port else None
        build = info.get("build") if isinstance(info, dict) else None
        # Attribute only when the answering process is this pid: a port can have been
        # taken over by another instance since this one was spawned with it.
        if isinstance(build, dict) and build.get("pid") == pid:
            server = BoardServerProcess(
                pid=pid, bind=server.bind, port=server.port, token=server.token,
                owner=str(info.get("owner") or ""),
            )
        servers.append(server)
    return sorted(servers, key=lambda item: item.pid)


def server_answers_as(state: dict[str, Any]) -> bool:
    """Return whether the server recorded in ``state`` reports the recorded pid itself.

    The strongest identity proof available: only the process holding the recorded token
    can answer ``/api/server-info`` with it, so a matching pid cannot be a reused one.
    """
    from leapflow.dashboard.launcher import fetch_server_info

    pid = int(state.get("pid") or 0)
    info = fetch_server_info(
        str(state.get("bind") or "127.0.0.1"), int(state.get("port") or 0), str(state.get("token") or ""),
    )
    build = info.get("build") if isinstance(info, dict) else None
    return pid > 0 and isinstance(build, dict) and build.get("pid") == pid


def _wait_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return True
        time.sleep(_POLL_S)
    return not process_alive(pid)


def stop_process(pid: int, *, port: int = 0) -> StopResult:
    """TERM the server, escalating to KILL if it outlives a short grace period.

    TERM first so the server releases its daemon preview owners and clears the
    discovery state it wrote; KILL only bounds a server wedged on its upstream.
    """
    try:
        os.kill(pid, DaemonSignal.SIGTERM.value)
    except ProcessLookupError:
        return StopResult(pid, port, True, "already exited")
    except OSError as exc:
        return StopResult(pid, port, False, f"cannot signal: {exc}")
    if _wait_gone(pid, _TERM_GRACE_S):
        return StopResult(pid, port, True, "stopped")
    kill = getattr(DaemonSignal, "SIGKILL", None)
    try:
        os.kill(pid, kill.value if kill is not None else DaemonSignal.SIGTERM.value)
    except ProcessLookupError:
        return StopResult(pid, port, True, "stopped")
    except OSError as exc:
        return StopResult(pid, port, False, f"cannot kill: {exc}")
    if _wait_gone(pid, _KILL_GRACE_S):
        return StopResult(pid, port, True, "killed after TERM timeout")
    return StopResult(pid, port, False, "still running after KILL")


__all__ = [
    "BoardServerProcess",
    "StopResult",
    "discover_board_servers",
    "is_board_server_argv",
    "pid_is_board_server",
    "process_argv",
    "server_answers_as",
    "stop_process",
]

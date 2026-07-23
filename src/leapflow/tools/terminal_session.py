"""Persistent terminal sessions — long-lived, opt-in, DISABLED by default.

Long-lived interactive shells (REPLs, dev servers, watch loops) hold process /
cwd / env state across turns and are a distinct responsibility from one-shot
command execution (``shell_run``). Per the Transport-Lifecycle Separation
principle they live behind an explicit session lifecycle — open / send / read /
close / list — and are never folded into one-shot action execution.

Safety: disabled unless ``tools.terminal_session_enabled`` is set (operator
opt-in is the primary gate). High risk — a persistent shell runs arbitrary
interactive input — so the open command and each send still pass the shell
hardline check, output is redacted, and sessions are bounded (max count, idle
TTL) with process-group termination + atexit cleanup to avoid orphans.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

_ENABLED = False
_MAX_SESSIONS = 8
_IDLE_TTL_S = 1800.0
_OUTPUT_BUFFER_CHARS = 20_000
_DEFAULT_READ_WAIT_S = 0.3
_MAX_READ_WAIT_S = 10.0

_SESSIONS: Dict[str, "_Session"] = {}
_LOCK = threading.Lock()


def set_terminal_sessions_enabled(enabled: bool) -> None:
    """Enable/disable terminal sessions (operator opt-in; default off)."""
    global _ENABLED
    _ENABLED = bool(enabled)


def _redact(text: str) -> str:
    try:
        from leapflow.security.redact import redact_sensitive_text
        return redact_sensitive_text(text)
    except ImportError:
        return text


class _Session:
    """A persistent shell subprocess with a background stdout reader + ring buffer."""

    def __init__(self, proc: subprocess.Popen, shell: str, cwd: str) -> None:
        self.proc = proc
        self.shell = shell
        self.cwd = cwd
        self.created_at = time.monotonic()
        self.last_activity = self.created_at
        self._buffer: Deque[str] = deque()
        self._buffer_chars = 0
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        if self.proc.stdout is None:
            return
        fd = self.proc.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                with self._lock:
                    self._buffer.append(text)
                    self._buffer_chars += len(text)
                    while self._buffer_chars > _OUTPUT_BUFFER_CHARS and self._buffer:
                        self._buffer_chars -= len(self._buffer.popleft())
        except (OSError, ValueError):
            pass

    def write(self, text: str) -> None:
        if self.proc.stdin is None:
            raise OSError("session stdin is closed")
        self.proc.stdin.write((text.rstrip("\n") + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        self.last_activity = time.monotonic()

    def drain(self) -> str:
        with self._lock:
            out = "".join(self._buffer)
            self._buffer.clear()
            self._buffer_chars = 0
        self.last_activity = time.monotonic()
        return out

    def alive(self) -> bool:
        return self.proc.poll() is None


def _terminate(session: "_Session") -> None:
    try:
        if session.proc.stdin is not None:
            session.proc.stdin.close()
    except OSError:
        pass
    try:
        os.killpg(os.getpgid(session.proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            session.proc.terminate()
        except (ProcessLookupError, OSError):
            pass
    try:
        session.proc.wait(timeout=3)
    except Exception:  # noqa: BLE001 - ensure we do not hang on cleanup
        try:
            session.proc.kill()
        except (ProcessLookupError, OSError):
            pass


def _reap_expired() -> None:
    now = time.monotonic()
    stale: list[str] = []
    with _LOCK:
        for sid, session in list(_SESSIONS.items()):
            if not session.alive() or (now - session.last_activity) > _IDLE_TTL_S:
                stale.append(sid)
        expired = [(sid, _SESSIONS.pop(sid)) for sid in stale]
    for _sid, session in expired:
        _terminate(session)


def _get(session_id: str) -> Optional["_Session"]:
    with _LOCK:
        return _SESSIONS.get(session_id)


def _disabled_error() -> Dict[str, Any]:
    return {
        "ok": False,
        "error": "Terminal sessions are disabled. Set tools.terminal_session_enabled to enable.",
        "failure_code": "disabled",
    }


async def terminal_open(params: Dict[str, Any]) -> Dict[str, Any]:
    """Open a persistent shell session; returns a session_id for send/read/close."""
    if not _ENABLED:
        return _disabled_error()
    _reap_expired()
    with _LOCK:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            return {"ok": False, "error": f"Too many terminal sessions (max {_MAX_SESSIONS}); close some first.", "failure_code": "too_many_sessions"}

    shell = str(params.get("shell") or os.environ.get("SHELL") or "/bin/bash")
    if shutil.which(shell) is None and not os.path.exists(shell):
        shell = "/bin/sh"
    cwd = str(params.get("cwd") or os.getcwd())
    if not os.path.isdir(cwd):
        return {"ok": False, "error": f"Working directory not found: {cwd}", "failure_code": "path_not_found"}
    initial = str(params.get("command") or "").strip()
    if initial:
        from leapflow.tools.shell_tools import _is_hardline_blocked
        if _is_hardline_blocked(initial):
            return {"ok": False, "error": "Initial command blocked by safety policy.", "failure_code": "blocked"}

    try:
        proc = subprocess.Popen(
            [shell],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            bufsize=0,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"Failed to start shell: {exc}", "failure_code": "spawn_failed"}

    session = _Session(proc, shell, cwd)
    session_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _SESSIONS[session_id] = session
    if initial:
        try:
            session.write(initial)
        except OSError as exc:
            return {"ok": False, "error": f"Failed to send initial command: {exc}", "failure_code": "write_failed", "session_id": session_id}
    return {"ok": True, "session_id": session_id, "shell": shell, "cwd": cwd}


async def terminal_send(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a line of input to a session and return output captured shortly after."""
    if not _ENABLED:
        return _disabled_error()
    session_id = str(params.get("session_id") or "")
    session = _get(session_id)
    if session is None:
        return {"ok": False, "error": f"No such terminal session: {session_id}", "failure_code": "session_not_found"}
    if not session.alive():
        return {"ok": False, "error": "Session process has exited.", "failure_code": "session_dead", "session_id": session_id}

    text = params.get("input", params.get("command", ""))
    text = "" if text is None else str(text)
    from leapflow.tools.shell_tools import _is_hardline_blocked
    if _is_hardline_blocked(text):
        return {"ok": False, "error": "Input blocked by safety policy.", "failure_code": "blocked"}
    try:
        session.write(text)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"Write failed: {exc}", "failure_code": "write_failed", "session_id": session_id}

    wait = min(max(float(params.get("wait", _DEFAULT_READ_WAIT_S)), 0.0), _MAX_READ_WAIT_S)
    if wait > 0:
        await asyncio.sleep(wait)
    return {"ok": True, "session_id": session_id, "output": _redact(session.drain()), "alive": session.alive()}


async def terminal_read(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drain buffered output from a session (optionally waiting briefly first)."""
    if not _ENABLED:
        return _disabled_error()
    session_id = str(params.get("session_id") or "")
    session = _get(session_id)
    if session is None:
        return {"ok": False, "error": f"No such terminal session: {session_id}", "failure_code": "session_not_found"}
    wait = min(max(float(params.get("wait", 0.0)), 0.0), _MAX_READ_WAIT_S)
    if wait > 0:
        await asyncio.sleep(wait)
    return {"ok": True, "session_id": session_id, "output": _redact(session.drain()), "alive": session.alive()}


async def terminal_close(params: Dict[str, Any]) -> Dict[str, Any]:
    """Terminate a session and release its process group."""
    if not _ENABLED:
        return _disabled_error()
    session_id = str(params.get("session_id") or "")
    with _LOCK:
        session = _SESSIONS.pop(session_id, None)
    if session is None:
        return {"ok": False, "error": f"No such terminal session: {session_id}", "failure_code": "session_not_found"}
    _terminate(session)
    return {"ok": True, "session_id": session_id, "closed": True}


async def terminal_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """List active terminal sessions."""
    if not _ENABLED:
        return _disabled_error()
    _reap_expired()
    now = time.monotonic()
    with _LOCK:
        sessions = [
            {"session_id": sid, "shell": s.shell, "cwd": s.cwd, "alive": s.alive(), "idle_seconds": round(now - s.last_activity, 1)}
            for sid, s in _SESSIONS.items()
        ]
    return {"ok": True, "sessions": sessions, "session_count": len(sessions)}


@atexit.register
def _cleanup_all_sessions() -> None:
    with _LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
    for session in sessions:
        _terminate(session)

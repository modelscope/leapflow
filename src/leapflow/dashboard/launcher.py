"""Dashboard launcher: token/URL/state helpers, browser open, and server spawn.

The dashboard runs as a separate view-client process (like the TUI). This module
owns the client-side concerns: a per-session access token, the localhost URL,
a small discovery state file under the profile runtime dir, opening the default
browser, and spawning the server when it is not already running. It has no web
dependency itself so it stays importable everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from leapflow.daemon.lifecycle import DaemonSignal
from leapflow.dashboard.revision import board_revision

logger = logging.getLogger(__name__)

_STATE_FILE = "dashboard.json"


def aiohttp_available() -> bool:
    """Return True when the optional ``aiohttp`` dependency is importable."""
    import importlib.util

    return importlib.util.find_spec("aiohttp") is not None


def generate_token() -> str:
    """Return a fresh URL-safe access token for the local dashboard."""
    return secrets.token_urlsafe(32)


def state_path(settings: Any) -> Path:
    layout = getattr(settings, "profile_layout", None)
    if layout is not None:
        try:
            return layout.dashboard_state_path
        except Exception:
            pass
    return Path(settings.runtime_dir) / _STATE_FILE


def load_state(settings: Any) -> Optional[dict[str, Any]]:
    """Load the dashboard discovery state, or None when absent/invalid."""
    path = state_path(settings)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_state(settings: Any, state: dict[str, Any]) -> None:
    """Persist the dashboard discovery state under the profile runtime dir."""
    path = state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def clear_state(settings: Any) -> None:
    """Best-effort removal of dashboard discovery state."""
    try:
        state_path(settings).unlink(missing_ok=True)
    except OSError:
        logger.debug("dashboard: failed to clear discovery state", exc_info=True)


def _host_for_bind(bind: str) -> str:
    return "127.0.0.1" if bind in ("", "0.0.0.0") else bind


def build_url(bind: str, port: int, token: str, path: str = "/") -> str:
    """Build a token-scoped localhost dashboard URL."""
    if not path.startswith("/"):
        path = "/" + path
    suffix = f"?token={token}" if token else ""
    return f"http://{_host_for_bind(bind)}:{port}{path}{suffix}"


def build_view_url(
    bind: str,
    port: int,
    token: str,
    *,
    template: str = "",
    device: str = "",
    channel: str = "",
) -> str:
    """Build a token-scoped URL that selects a board template and optional target.

    The template is the primary view dimension: the server renders the current session
    through the named template (default when omitted). ``device``/``channel`` name a
    drill-down target for the lenses that have one, and are the same parameters
    ``DashboardIntent.from_params`` reads on the other side.

    Single owner of the query contract so every entry point -- ``leap board``,
    ``/board``, a deep link printed in the TUI -- lands on the intended view. Values are
    percent-encoded because a device id is discovered rather than authored: a name with
    a space or an ampersand would otherwise truncate the query silently and open the
    wrong page.
    """
    url = build_url(bind, port, token)
    for key, value in (("template", template), ("device", device), ("channel", channel)):
        if value:
            url += f"&{key}={quote(str(value), safe='')}"
    return url


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Return True when a TCP connect to (host, port) succeeds quickly."""
    try:
        with socket.create_connection((_host_for_bind(host), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def probe_token(bind: str, port: int, token: str, *, timeout: float = 0.6) -> bool:
    """Return True when the server on (bind, port) actually accepts ``token``.

    A reachable port is not enough: a stale discovery-state token (server
    restarted with a new token, port reused, or a leftover instance) would make
    the browser land on ``missing or invalid token``. This probes the real
    server so callers only ever trust a token that works.
    """
    import urllib.error
    import urllib.request

    if not token:
        return False
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed localhost URL
            build_url(bind, port, token), timeout=timeout
        ) as response:
            return 200 <= int(getattr(response, "status", 200)) < 400
    except urllib.error.HTTPError:
        return False  # 401 -> token rejected by the running server
    except OSError:
        return False


def fetch_server_info(bind: str, port: int, token: str, *, timeout: float = 0.8) -> Optional[dict[str, Any]]:
    """Fetch the running dashboard server's own build/staleness self-report.

    Best-effort: this is a diagnostic side-channel (``/board status`` and
    ``leap daemon status`` use it to warn a developer that the *separate*
    long-lived web server process has not picked up a recent source change),
    never a hard dependency — any failure (server down, wrong token, bad JSON,
    old server predating this endpoint) returns None rather than raising.
    """
    import json
    import urllib.error
    import urllib.request

    if not token:
        return None
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed localhost URL
            build_url(bind, port, token, path="/api/server-info"), timeout=timeout
        ) as response:
            if not (200 <= int(getattr(response, "status", 200)) < 400):
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def server_running(settings: Any) -> Optional[dict[str, Any]]:
    """Return live dashboard state when a server accepts the stored token, else None.

    Validates the token (not just port reachability) so a stale/mismatched
    discovery state never yields a URL the server would reject.
    """
    state = load_state(settings)
    if not state:
        return None
    port = int(state.get("port") or 0)
    bind = str(state.get("bind") or settings.dashboard_bind)
    token = str(state.get("token") or "")
    if port and is_port_open(bind, port) and probe_token(bind, port, token):
        return state
    return None


def server_is_stale(state: dict[str, Any]) -> bool:
    """Return whether a reachable Board belongs to a different content generation.

    A missing endpoint remains inconclusive -- it may be a foreign service or a transient
    start -- but a Board that answers ``/api/server-info`` without a revision is an older
    Board generation and is replaced. This makes Python, templates and static assets move
    together instead of letting one long-lived process serve a mixed generation.
    """
    info = fetch_server_info(
        str(state.get("bind") or "127.0.0.1"),
        int(state.get("port") or 0),
        str(state.get("token") or ""),
    )
    if info is None:
        return False
    if info.get("stale") is True:
        return True
    running_revision = str(info.get("revision") or "")
    if not running_revision:
        return True
    return running_revision != board_revision()

def open_in_browser(url: str) -> bool:
    """Open ``url`` in the default browser; return False on headless failure."""
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:  # noqa: BLE001 - headless/no-DISPLAY environments
        logger.debug("dashboard: webbrowser.open failed", exc_info=True)
        return False


def _find_free_port(bind: str, start: int, *, span: int = 20) -> int:
    """Return the first free port at or above ``start`` (falls back to ``start``)."""
    for candidate in range(start, start + span):
        if not is_port_open(bind, candidate):
            return candidate
    return start


def _pid_is_dashboard_server(pid: int) -> bool:
    """Best-effort confirm ``pid`` is our dashboard server before signaling it.

    Guards against PID reuse: a recorded pid may since belong to an unrelated
    process. Verification uses psutil when available; without it we cannot prove
    identity portably, so we return False and let the caller fall back to a free
    port rather than risk terminating an innocent process.
    """
    try:
        import psutil  # optional dependency; absent in minimal installs
    except ImportError:
        return False
    try:
        cmdline = " ".join(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError):
        return False
    return "leapflow" in cmdline and "board" in cmdline and "--serve" in cmdline


def _retire_stale_server(settings: Any) -> None:
    """Best-effort retire a stale dashboard server recorded in discovery state.

    Called only after ``server_running`` rejected the state (dead server or a
    token the server no longer accepts). Signals the recorded pid *only when it
    is verifiably our dashboard server* (guarding against PID reuse); otherwise
    it leaves the process alone and lets ``ensure_server`` pick a free port.
    Always drops the stale state file.
    """
    state = load_state(settings)
    if not state:
        return
    port = int(state.get("port") or 0)
    bind = str(state.get("bind") or getattr(settings, "dashboard_bind", "127.0.0.1"))
    pid = int(state.get("pid") or 0)
    if port and pid > 0 and is_port_open(bind, port) and _pid_is_dashboard_server(pid):
        try:
            os.kill(pid, DaemonSignal.SIGTERM.value)
        except OSError:
            logger.debug("dashboard: stale server pid=%s not signalable", pid, exc_info=True)
        else:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and is_port_open(bind, port):
                time.sleep(0.1)
    clear_state(settings)


def ensure_server(settings: Any, *, wait_s: float = 8.0) -> dict[str, Any]:
    """Return running dashboard state, spawning a detached server if needed.

    Reuses an existing server only when it accepts the stored token **and** is running
    the current source. Otherwise it retires the stale one, picks a free port, and spawns
    a fresh server whose token is verified before its state is published. Raises
    RuntimeError when the optional web dependency is missing so callers can surface an
    install hint.

    The staleness check is not cosmetic. This process is long-lived and separately
    launched, and it reads board templates from disk on every render -- so an old server
    serves *new* templates with *old* builder code. A template that asks for data the old
    builder never supplies renders nothing at all, which is how the hardware board became
    a bare title that survived every reopen, because reuse was unconditional.
    """
    existing = server_running(settings)
    if existing and not server_is_stale(existing):
        return existing
    if not aiohttp_available():
        raise RuntimeError(
            "The dashboard web server requires the optional 'aiohttp' dependency. "
            "Install it with: pip install 'leapflow[dashboard]'"
        )

    # A prior server may be dead-but-recorded, alive with a token we can no longer
    # prove, or alive but older than the code it is serving. Retire it so a fresh,
    # trusted server can take over cleanly.
    if existing:
        logger.info("dashboard: replacing a server that predates the current source")
    _retire_stale_server(settings)
    return _spawn_verified_server(settings, wait_s=wait_s)


def _spawn_verified_server(settings: Any, *, wait_s: float = 8.0) -> dict[str, Any]:
    """Spawn a detached board server, prove it answers our token, publish its state.

    Separate from ``ensure_server`` so the *decision* (reuse or replace) can be read and
    tested apart from the mechanics of getting a process up.
    """
    token = generate_token()
    bind = str(getattr(settings, "dashboard_bind", "127.0.0.1"))
    preferred = int(getattr(settings, "dashboard_port", 8765))
    port = preferred if not is_port_open(bind, preferred) else _find_free_port(bind, preferred)
    cmd = [
        sys.executable, "-m", "leapflow", "board", "--serve",
        "--token", token, "--port", str(port), "--bind", bind,
    ]
    creationflags = 0
    start_new_session = True
    if os.name == "nt":  # pragma: no cover - platform specific
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        start_new_session = False
    proc = subprocess.Popen(  # noqa: S603 - trusted, fixed argv
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        # Token-aware readiness: a mere open port could be a foreign/stale server.
        if probe_token(bind, port, token):
            break
        if proc.poll() is not None:
            raise RuntimeError("dashboard server exited before becoming ready")
        time.sleep(0.15)
    else:
        raise RuntimeError("dashboard server did not become ready in time")

    state = {
        "port": port,
        "bind": bind,
        "token": token,
        "pid": proc.pid,
        "revision": board_revision(),
        "url": build_url(bind, port, token),
    }
    write_state(settings, state)
    return state


__all__ = [
    "aiohttp_available",
    "generate_token",
    "state_path",
    "load_state",
    "write_state",
    "clear_state",
    "build_url",
    "build_view_url",
    "is_port_open",
    "probe_token",
    "fetch_server_info",
    "server_running",
    "open_in_browser",
    "ensure_server",
]

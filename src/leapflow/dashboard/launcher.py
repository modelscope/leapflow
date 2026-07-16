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
    state_path(settings).unlink(missing_ok=True)


def _host_for_bind(bind: str) -> str:
    return "127.0.0.1" if bind in ("", "0.0.0.0") else bind


def build_url(bind: str, port: int, token: str, path: str = "/") -> str:
    """Build a token-scoped localhost dashboard URL."""
    if not path.startswith("/"):
        path = "/" + path
    suffix = f"?token={token}" if token else ""
    return f"http://{_host_for_bind(bind)}:{port}{path}{suffix}"


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Return True when a TCP connect to (host, port) succeeds quickly."""
    try:
        with socket.create_connection((_host_for_bind(host), int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def server_running(settings: Any) -> Optional[dict[str, Any]]:
    """Return live dashboard state when a server appears reachable, else None."""
    state = load_state(settings)
    if not state:
        return None
    port = int(state.get("port") or 0)
    bind = str(state.get("bind") or settings.dashboard_bind)
    if port and is_port_open(bind, port):
        return state
    return None


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the default browser; return False on headless failure."""
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:  # noqa: BLE001 - headless/no-DISPLAY environments
        logger.debug("dashboard: webbrowser.open failed", exc_info=True)
        return False


def ensure_server(settings: Any, *, wait_s: float = 8.0) -> dict[str, Any]:
    """Return running dashboard state, spawning a detached server if needed.

    Raises RuntimeError when the optional web dependency is missing so callers
    can surface an actionable install hint instead of spawning a doomed process.
    """
    existing = server_running(settings)
    if existing:
        return existing
    if not aiohttp_available():
        raise RuntimeError(
            "The dashboard web server requires the optional 'aiohttp' dependency. "
            "Install it with: pip install 'leapflow[dashboard]'"
        )

    token = generate_token()
    port = int(getattr(settings, "dashboard_port", 8765))
    bind = str(getattr(settings, "dashboard_bind", "127.0.0.1"))
    cmd = [
        sys.executable, "-m", "leapflow", "dashboard", "--serve",
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
        if is_port_open(bind, port):
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
    "is_port_open",
    "server_running",
    "open_in_browser",
    "ensure_server",
]

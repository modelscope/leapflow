"""Hermetic tests for the dashboard launcher and server action dispatch.

No aiohttp required: the launcher is dependency-free and DashboardServer's
action dispatch only touches the injected client. The aiohttp app wiring is
covered by an importorskip guard.
"""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapflow.dashboard import launcher
from leapflow.dashboard.server import DashboardServer


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_dir=tmp_path,
        dashboard_bind="127.0.0.1",
        dashboard_port=8765,
        dashboard_auto_open=True,
    )


# ── launcher ────────────────────────────────────────────────────────────────


def test_build_url_includes_token_and_host() -> None:
    url = launcher.build_url("0.0.0.0", 9000, "abc")
    assert url == "http://127.0.0.1:9000/?token=abc"


def test_generate_token_is_unique() -> None:
    assert launcher.generate_token() != launcher.generate_token()


def test_state_roundtrip(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert launcher.load_state(settings) is None
    launcher.write_state(settings, {"port": 1, "bind": "127.0.0.1", "token": "t"})
    assert launcher.load_state(settings)["token"] == "t"
    launcher.clear_state(settings)
    assert launcher.load_state(settings) is None


def test_server_running_requires_open_port(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    launcher.write_state(settings, {"port": 65534, "bind": "127.0.0.1", "token": "t"})
    assert launcher.server_running(settings) is None  # nothing listening

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        port = srv.getsockname()[1]
        launcher.write_state(settings, {"port": port, "bind": "127.0.0.1", "token": "t"})
        state = launcher.server_running(settings)
        assert state is not None and state["port"] == port
    finally:
        srv.close()


def test_open_in_browser_handles_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.webbrowser, "open", lambda url, new=0: True)
    assert launcher.open_in_browser("http://x") is True

    def _boom(url, new=0):
        raise RuntimeError("no display")

    monkeypatch.setattr(launcher.webbrowser, "open", _boom)
    assert launcher.open_in_browser("http://x") is False


def test_ensure_server_requires_aiohttp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(launcher, "aiohttp_available", lambda: False)
    with pytest.raises(RuntimeError, match="aiohttp"):
        launcher.ensure_server(settings)


# ── DashboardServer.dispatch_action (allow-listed, transport-free) ───────────


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def watch_pause(self, watch_id: str) -> dict:
        self.calls.append(("pause", watch_id))
        return {"state": "suspended"}

    async def watch_mute(self, watch_id: str, *, muted: bool = True) -> dict:
        self.calls.append(("mute", watch_id, muted))
        return {"muted": muted}

    async def approval_resolve(self, pending_id: str, decision: str) -> dict:
        self.calls.append(("approval", pending_id, decision))
        return {"ok": True}


async def test_dispatch_action_rpc_allowlist_and_kinds() -> None:
    client = _FakeClient()
    server = DashboardServer(client=client, token="t")

    ok = await server.dispatch_action({"kind": "rpc", "name": "watch.pause", "params": {"watch_id": "w1"}})
    assert ok["ok"] is True and ("pause", "w1") in client.calls

    denied = await server.dispatch_action({"kind": "rpc", "name": "daemon.shutdown"})
    assert denied["ok"] is False

    nav = await server.dispatch_action({"kind": "nav", "name": "filter"})
    assert nav["ok"] is True

    intent = await server.dispatch_action({"kind": "intent", "name": "drilldown"})
    assert intent["queued"] is True

    await server.dispatch_action({"kind": "approval", "params": {"pending_id": "p", "decision": "allow"}})
    assert ("approval", "p", "allow") in client.calls

    unknown = await server.dispatch_action({"kind": "weird"})
    assert unknown["ok"] is False


def test_server_build_app_requires_aiohttp() -> None:
    pytest.importorskip("aiohttp")
    server = DashboardServer(client=_FakeClient(), token="t")
    app = server.build_app()
    assert app is not None

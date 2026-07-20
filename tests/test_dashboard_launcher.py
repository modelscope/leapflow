"""Hermetic tests for the dashboard launcher and server action dispatch.

No aiohttp required: the launcher is dependency-free and DashboardServer's
action dispatch only touches the injected client. The aiohttp app wiring is
covered by an importorskip guard.
"""

from __future__ import annotations

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


def test_server_running_requires_open_port_and_valid_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    # No discovery state -> not running.
    assert launcher.server_running(settings) is None

    launcher.write_state(settings, {"port": 8765, "bind": "127.0.0.1", "token": "t"})

    # Port closed -> not running (the token is never probed).
    monkeypatch.setattr(launcher, "is_port_open", lambda *a, **k: False)
    assert launcher.server_running(settings) is None

    # Port open but the server rejects the token (stale/foreign): not running,
    # so callers never build a URL that renders as 'missing or invalid token'.
    monkeypatch.setattr(launcher, "is_port_open", lambda *a, **k: True)
    monkeypatch.setattr(launcher, "probe_token", lambda *a, **k: False)
    assert launcher.server_running(settings) is None

    # Port open and the token is accepted: usable.
    monkeypatch.setattr(launcher, "probe_token", lambda *a, **k: True)
    state = launcher.server_running(settings)
    assert state is not None and state["port"] == 8765


def test_ensure_server_reuses_token_valid_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    valid = {"port": 8765, "bind": "127.0.0.1", "token": "T"}
    monkeypatch.setattr(launcher, "server_running", lambda s: valid)
    # A validated existing server is reused as-is; no spawn is attempted.
    assert launcher.ensure_server(settings) is valid


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


def test_retire_stale_server_skips_kill_when_pid_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards PID reuse: without a positive identity match the recorded pid is
    # never signaled (it may now be an unrelated process); state is still cleared.
    settings = _settings(tmp_path)
    launcher.write_state(settings, {"port": 8765, "bind": "127.0.0.1", "token": "t", "pid": 4242})
    monkeypatch.setattr(launcher, "is_port_open", lambda *a, **k: True)
    monkeypatch.setattr(launcher, "_pid_is_dashboard_server", lambda pid: False)
    killed: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: killed.append(pid))
    launcher._retire_stale_server(settings)
    assert killed == []
    assert launcher.load_state(settings) is None


def test_retire_stale_server_signals_verified_dashboard_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A positively identified dashboard pid is signaled exactly once to free the port.
    settings = _settings(tmp_path)
    launcher.write_state(settings, {"port": 8765, "bind": "127.0.0.1", "token": "t", "pid": 4242})
    opens = iter([True, False])  # open for the guard, closed right after the signal
    monkeypatch.setattr(launcher, "is_port_open", lambda *a, **k: next(opens, False))
    monkeypatch.setattr(launcher, "_pid_is_dashboard_server", lambda pid: True)
    killed: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: killed.append(pid))
    launcher._retire_stale_server(settings)
    assert killed == [4242]
    assert launcher.load_state(settings) is None


def test_check_origin_matches_loopback_host_exactly() -> None:
    # A substring test would accept attacker127.0.0.1.com / localhost.evil.com;
    # we parse the Origin and match its hostname exactly against the loopback set.
    check = DashboardServer._check_origin
    assert check(SimpleNamespace(headers={})) is True  # no Origin -> allow
    for origin in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"):
        assert check(SimpleNamespace(headers={"Origin": origin})) is True
    for origin in ("http://attacker127.0.0.1.com", "http://localhost.attacker.com", "https://evil.com"):
        assert check(SimpleNamespace(headers={"Origin": origin})) is False


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

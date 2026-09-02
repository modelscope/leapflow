"""Hermetic tests for the dashboard launcher and server action dispatch.

No aiohttp required: the launcher is dependency-free and DashboardServer's
action dispatch only touches the injected client. The aiohttp app wiring is
covered by an importorskip guard.
"""

from __future__ import annotations

import json
import re
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


def test_clear_state_is_best_effort_on_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)

    class _DeniedPath:
        def unlink(self, *, missing_ok: bool = False) -> None:
            raise PermissionError("sandbox denied")

    monkeypatch.setattr(launcher, "state_path", lambda _settings: _DeniedPath())

    launcher.clear_state(settings)  # must not raise; /board can pick a fresh port


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


# ── fetch_server_info: probes the *separate* long-lived server's own staleness ──


class _FakeUrlResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeUrlResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_fetch_server_info_parses_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"build": {"commit": "abc123", "pid": 42}, "stale": True}

    def _fake_urlopen(url: str, timeout: float = 0.0) -> _FakeUrlResponse:
        assert "/api/server-info" in url and "token=tok" in url
        return _FakeUrlResponse(200, json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    assert launcher.fetch_server_info("127.0.0.1", 8766, "tok") == payload


def test_fetch_server_info_returns_none_without_token() -> None:
    assert launcher.fetch_server_info("127.0.0.1", 8766, "") is None


def test_fetch_server_info_returns_none_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(url: str, timeout: float = 0.0) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    assert launcher.fetch_server_info("127.0.0.1", 8766, "tok") is None


def test_fetch_server_info_returns_none_on_non_2xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=0.0: _FakeUrlResponse(401, b'{"error": "unauthorized"}'),
    )
    assert launcher.fetch_server_info("127.0.0.1", 8766, "tok") is None


def test_fetch_server_info_returns_none_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=0.0: _FakeUrlResponse(200, b"not json"),
    )
    assert launcher.fetch_server_info("127.0.0.1", 8766, "tok") is None


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


# ── DashboardServer build/staleness self-report (own long-lived-process check) ──


class _FakeViewProvider:
    """Minimal DashboardDataProvider double: enough for _handle_view to render."""

    async def watches(self) -> list:
        return []

    async def findings(self, *, watch_id: str = "", limit: int = 50) -> list:
        return []

    async def signal_metrics(self) -> dict:
        return {"metrics": {}, "signal_stream": []}


async def test_handle_view_attaches_server_build_meta() -> None:
    pytest.importorskip("aiohttp")
    server = DashboardServer(client=_FakeClient(), token="t")
    server._provider = _FakeViewProvider()
    request = SimpleNamespace(query={"token": "t", "template": "generic"}, headers={})

    response = await server._handle_view(request)
    spec = json.loads(response.text)

    server_meta = spec["meta"]["server"]
    assert server_meta["build"]["pid"] == server._build_info.pid
    assert server_meta["stale"] in (True, False, None)


async def test_handle_view_returns_a_structured_error_instead_of_http_500() -> None:
    pytest.importorskip("aiohttp")

    class _BrokenBuilder:
        async def build(self, intent, provider):
            raise RuntimeError("template/data mismatch")

    server = DashboardServer(client=_FakeClient(), token="t")
    server._builder = _BrokenBuilder()
    request = SimpleNamespace(query={"token": "t", "template": "hardware"}, headers={})

    response = await server._handle_view(request)
    payload = json.loads(response.text)

    assert response.status == 503
    assert payload["error"]["code"] == "view_unavailable"
    assert payload["error"]["request_id"]
    assert "template/data mismatch" not in payload["error"]["message"]


async def test_handle_server_info_requires_token() -> None:
    pytest.importorskip("aiohttp")
    server = DashboardServer(client=_FakeClient(), token="t")
    request = SimpleNamespace(query={}, headers={})

    response = await server._handle_server_info(request)

    assert response.status == 401


async def test_handle_server_info_reports_captured_build_and_stale_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("aiohttp")
    from leapflow.dashboard import server as server_module

    server = DashboardServer(client=_FakeClient(), token="t")
    # This process's own fingerprint check is irrelevant to the endpoint's
    # wiring; pin the verdict so the assertion is deterministic.
    monkeypatch.setattr(server_module, "is_stale", lambda info: True)
    # _server_info() is a non-blocking cache read; force one deterministic
    # refresh so the assertion below does not race the background task that
    # current() would otherwise schedule.
    await server._build_staleness.refresh(server_module.is_stale)
    request = SimpleNamespace(query={"token": "t"}, headers={})

    response = await server._handle_server_info(request)
    payload = json.loads(response.text)

    assert payload["stale"] is True
    assert payload["build"]["pid"] == server._build_info.pid
    assert payload["build"]["version"] == server._build_info.version
    assert payload["revision"] == server._revision


# ── Device preview endpoints ────────────────────────────────────────────────
#
# The client-facing half of the peripheral plane. No policy lives in this layer:
# whether a preview is permitted is decided by the daemon's approval chain on the
# channel's declared privacy tier, so what is asserted here is that the refusal is
# forwarded faithfully, with a status a browser can act on and a message that names
# the next step.

_JPEG = (
    b"\xff\xd8"
    b"\xff\xc0\x00\x11\x08\x00\x02\x00\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    b"\xff\xd9"
)


class _FrameClient:
    """A daemon stub whose frame reply the test controls."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.frame_calls = 0
        self.frame_requests: list[dict[str, object]] = []
        self.write_calls: list[tuple] = []
        self.release_calls: list[tuple[str, str, str]] = []

    async def hardware_frame(self, device: str, channel: str, **kwargs: object) -> dict:
        self.frame_calls += 1
        self.frame_requests.append({"device": device, "channel": channel, **kwargs})
        return dict(self.reply)

    async def hardware_preview_release(
        self, device: str, channel: str, *, viewer_id: str = ""
    ) -> dict:
        self.release_calls.append((device, channel, viewer_id))
        return {"ok": True, "released": True, "active_viewers": 0}

    async def hardware_write_request(
        self, device: str, channel: str, value: object, *, dry_run: bool = True
    ) -> dict:
        self.write_calls.append((device, channel, value, dry_run))
        return {"ok": True, "preview": dry_run}

    async def hardware_rescan(self) -> dict:
        return {"ok": True, "admitted": []}


async def test_media_frame_serves_the_image_with_no_store_headers() -> None:
    pytest.importorskip("aiohttp")
    import base64

    client = _FrameClient({
        "ok": True, "media_type": "image/jpeg", "width": 4, "height": 2,
        "data_b64": base64.b64encode(_JPEG).decode("ascii"),
    })
    server = DashboardServer(client=client, token="t")
    request = SimpleNamespace(query={"token": "t", "device": "cam", "channel": "frame"}, headers={})

    response = await server._handle_media_frame(request)

    assert response.status == 200
    assert response.body == _JPEG
    # A frame of somebody's room must not be left in a cache or readable cross-origin.
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


async def test_media_frame_without_a_token_is_refused() -> None:
    pytest.importorskip("aiohttp")

    server = DashboardServer(client=_FrameClient({"ok": True}), token="t")
    request = SimpleNamespace(query={"device": "cam", "channel": "frame"}, headers={})

    response = await server._handle_media_frame(request)
    assert response.status == 401


async def test_a_refused_preview_is_forwarded_as_403_with_the_daemon_message() -> None:
    """The status matters: the client shows the body, and 403 is what it keys on.

    The message is the daemon's own, because it names where consent can be given -- a
    browser session cannot grant itself a camera, so a generic error here would leave
    somebody clicking a button that never works.
    """
    pytest.importorskip("aiohttp")

    client = _FrameClient({
        "ok": False,
        "code": "consent_required",
        "error": "Viewing cam.frame needs consent. Run `leap hw preview cam frame`.",
    })
    server = DashboardServer(client=client, token="t")
    request = SimpleNamespace(query={"token": "t", "device": "cam", "channel": "frame"}, headers={})

    response = await server._handle_media_frame(request)
    payload = json.loads(response.text)

    assert response.status == 403
    assert payload["code"] == "consent_required"
    assert "leap hw preview" in payload["error"]


async def test_server_issued_preview_owner_releases_the_exact_channel() -> None:
    client = _FrameClient({"ok": True})
    server = DashboardServer(client=client, token="t")
    server._preview_viewers["viewer-1"] = ("cam", "frame")

    result = await server._release_preview_viewer("viewer-1")

    assert result["released"] is True
    assert client.release_calls == [("cam", "frame", "viewer-1")]
    assert await server._release_preview_viewer("viewer-1") == {
        "ok": True, "released": False, "active_viewers": 0
    }


async def test_a_missing_target_never_reaches_the_daemon() -> None:
    pytest.importorskip("aiohttp")

    client = _FrameClient({"ok": True})
    server = DashboardServer(client=client, token="t")
    request = SimpleNamespace(query={"token": "t", "device": "", "channel": ""}, headers={})

    response = await server._handle_media_frame(request)
    assert response.status == 404
    assert client.frame_calls == 0


def test_preview_rate_is_clamped_before_the_binary_relay() -> None:
    """A hand-edited ``?fps=`` cannot exceed the Board's 30fps profile ceiling."""
    options = DashboardServer._preview_options
    assert options({"fps": "1000"})["fps"] == 30.0
    assert options({"fps": ""})["fps"] == 0.0
    assert options({"fps": "nonsense"})["fps"] == 0.0
    assert options({"fps": "-4"})["fps"] == 0.0
    assert options({"fps": "1"})["fps"] == 1.0


async def test_the_board_can_preview_and_request_a_change_but_not_approve_one() -> None:
    """The settings boundary, asserted on the allow-list and the dispatch.

    Both buttons reach one RPC with one flag, so there is a single write path; and no
    allow-listed action resolves an approval, because letting the board approve its own
    request would make the gate ceremonial.
    """
    client = _FrameClient({"ok": True})
    server = DashboardServer(client=client, token="t")

    preview = await server.dispatch_action({
        "kind": "rpc", "name": "hardware.configure_preview",
        "params": {"device": "rig", "channel": "level", "value": "30"},
    })
    assert preview["ok"] is True
    assert client.write_calls[-1] == ("rig", "level", "30", True), "preview was not a dry run"

    submit = await server.dispatch_action({
        "kind": "rpc", "name": "hardware.request_write",
        "params": {"device": "rig", "channel": "level", "value": "30"},
    })
    assert submit["ok"] is True
    assert client.write_calls[-1] == ("rig", "level", "30", False)

    denied = await server.dispatch_action({"kind": "rpc", "name": "hardware.approve", "params": {}})
    assert denied["ok"] is False
    assert "not allowed" in denied["error"]


def test_build_view_url_carries_the_drill_down_target() -> None:
    """One owner of the query contract, so every entry point lands on the same view.

    The parameters are exactly the ones ``DashboardIntent.from_params`` reads, which is
    what makes a ``/board device`` deep link resolve to that device's page rather than
    the fleet.
    """
    url = launcher.build_view_url(
        "127.0.0.1", 8765, "tok", template="hardware_device", device="cam_0", channel="frame"
    )

    assert url.startswith("http://127.0.0.1:8765/?token=tok")
    assert "template=hardware_device" in url
    assert "device=cam_0" in url
    assert "channel=frame" in url

    # An absent target adds nothing: an empty ``device=`` would reach the builder as a
    # device whose id is the empty string and be reported as unknown.
    plain = launcher.build_view_url("127.0.0.1", 8765, "tok", template="hardware")
    assert plain.endswith("template=hardware")
    assert "device=" not in plain


def test_build_view_url_encodes_a_discovered_id() -> None:
    """Device ids are discovered, not authored.

    An unencoded ``&`` or space would truncate the query and open a different view than
    the one the daemon resolved -- silently, because the server would simply see fewer
    parameters.
    """
    url = launcher.build_view_url(
        "127.0.0.1", 8765, "tok", template="hardware_device", device="cam &1", channel="a b"
    )

    assert "device=cam%20%261" in url
    assert "channel=a%20b" in url
    # The separator count is the assertion that matters: an injected '&' would create a
    # parameter nobody asked for.
    assert url.count("&") == 3


# ── Page-side approval ──────────────────────────────────────────────────────
#
# The board is a legitimate approval surface *for a request it made itself*: the person is
# looking at the page they just clicked. What makes it safe is that the prompt is not
# invented in the browser -- the daemon's approval chain raises it, carrying the risk
# assessment and the choices the policy allows -- and the answer goes back through
# ``approval.resolve``, so the grant, the audit record and the decision semantics stay the
# orchestrator's.
#
# Structurally this works because ``hardware.frame`` is in ``_APPROVAL_ROUTED_METHODS``:
# the daemon installs a route, so the prompt travels as an interleaved notification on the
# request's own socket and the request waits for the answer.


class _GatedFrameClient:
    """A daemon stub that raises a prompt, then answers once the page resolves it."""

    PROMPT = {
        "pending_id": "pending-1",
        "choices": ["allow_once", "allow_session", "deny"],
        "display": {
            "title": "Observe the camera?",
            "summary": "camera_0 · frame",
            "reason": "Reading this observes the physical space around it.",
        },
    }

    def __init__(self) -> None:
        import asyncio as _asyncio

        self.answered = _asyncio.Event()
        self.decision = ""
        self.callbacks: list[object] = []

    async def hardware_frame(
        self,
        device: str,
        channel: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
        on_stream_event: object = None,
    ) -> dict:
        import base64

        self.callbacks.append(on_stream_event)
        if on_stream_event is not None:
            from leapflow.engine import StreamEvent

            on_stream_event(StreamEvent(
                type="approval_request", content="Approval required",
                metadata={"approval": dict(self.PROMPT)},
            ))
        # Waits for the human, exactly as the coordinator's future does.
        await self.answered.wait()
        if self.decision.startswith("allow"):
            return {
                "ok": True, "media_type": "image/jpeg", "width": 4, "height": 2,
                "data_b64": base64.b64encode(_JPEG).decode("ascii"),
            }
        return {"ok": False, "code": "consent_required", "error": "You declined."}

    async def approval_resolve(self, pending_id: str, decision: str, *, reason: str = "") -> dict:
        self.decision = decision
        self.answered.set()
        return {"ok": True, "pending_id": pending_id, "decision": decision}

    async def hardware_read(
        self, device: str, channel: str, *, viewer_id: str = "", on_stream_event: object = None
    ) -> dict:
        return {"ok": True, "value": -32.5, "unit": "dBFS", "channel_id": channel}


async def test_a_gated_preview_prompts_the_page_and_completes_when_it_answers() -> None:
    """The whole point: a browser can obtain consent for a request it originated.

    Before this, the board could only be refused -- an ordinary RPC has no approval route,
    so the coordinator denied immediately and the Start button could never work.
    """
    pytest.importorskip("aiohttp")
    import asyncio

    client = _GatedFrameClient()
    server = DashboardServer(client=client, token="t")
    delivered: list[dict] = []
    server._hub.broadcast = lambda message: delivered.append(message)  # type: ignore[assignment]

    fetch = asyncio.create_task(server._fetch_frame("cam", "frame"))
    await asyncio.sleep(0)  # let the handler reach the prompt

    assert delivered, "the approval prompt never reached the browser hub"
    assert delivered[0]["type"] == "approval_request"
    payload = delivered[0]["payload"]
    assert payload["pending_id"] == "pending-1"
    assert payload["choices"] == ["allow_once", "allow_session", "deny"], (
        "the page must offer exactly the choices the policy allowed, not a local list"
    )
    assert not fetch.done(), "the request must wait for the answer, not proceed without it"

    # Exactly what the Allow button posts.
    result = await server.dispatch_action({
        "kind": "approval",
        "params": {"pending_id": payload["pending_id"], "decision": "allow_session"},
    })
    assert result["ok"] is True

    frame = await asyncio.wait_for(fetch, timeout=2.0)
    assert frame["ok"] is True
    assert frame["data"] == _JPEG


async def test_a_declined_preview_reports_the_refusal() -> None:
    """Deny is an answer, and it must reach the caller as one."""
    pytest.importorskip("aiohttp")
    import asyncio

    client = _GatedFrameClient()
    server = DashboardServer(client=client, token="t")
    server._hub.broadcast = lambda message: None  # type: ignore[assignment]

    fetch = asyncio.create_task(server._fetch_frame("cam", "frame"))
    await asyncio.sleep(0)
    await server.dispatch_action({
        "kind": "approval", "params": {"pending_id": "pending-1", "decision": "deny"},
    })

    frame = await asyncio.wait_for(fetch, timeout=2.0)
    assert frame["ok"] is False
    assert frame["code"] == "consent_required"


def test_only_a_resolvable_prompt_is_shown() -> None:
    """A prompt with no pending_id cannot be answered, so rendering it is worse than not.

    It would leave a card whose buttons resolve nothing while the request behind it waits.
    """
    from leapflow.engine import StreamEvent

    server = DashboardServer(client=_GatedFrameClient(), token="t")
    delivered: list[dict] = []
    server._hub.broadcast = lambda message: delivered.append(message)  # type: ignore[assignment]

    server._forward_approval(StreamEvent(type="approval_request", content="", metadata={"approval": {}}))
    server._forward_approval(StreamEvent(type="approval_request", content="", metadata={}))
    # A non-approval event on the same channel must be ignored rather than shown as one.
    server._forward_approval(StreamEvent(type="chunk", content="hello", metadata=None))

    assert delivered == []


async def test_the_level_endpoint_serves_a_reading_and_forwards_its_prompt() -> None:
    """A microphone's preview is a number, so it has its own endpoint and the same gate."""
    pytest.importorskip("aiohttp")

    server = DashboardServer(client=_GatedFrameClient(), token="t")
    request = SimpleNamespace(
        query={"token": "t", "device": "mic", "channel": "level.dbfs"}, headers={}
    )

    response = await server._handle_media_level(request)
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["value"] == -32.5
    assert payload["unit"] == "dBFS"
    assert response.headers["Cache-Control"] == "no-store"


async def test_the_level_endpoint_requires_a_token_and_a_target() -> None:
    pytest.importorskip("aiohttp")

    server = DashboardServer(client=_GatedFrameClient(), token="t")

    unauthorized = await server._handle_media_level(
        SimpleNamespace(query={"device": "mic", "channel": "level.dbfs"}, headers={})
    )
    assert unauthorized.status == 401

    targetless = await server._handle_media_level(
        SimpleNamespace(query={"token": "t"}, headers={})
    )
    assert targetless.status == 404


# ── A stale board process must be replaced, not reused ──────────────────────
#
# The board is separately launched, long-lived, and reads its templates from disk on every
# render -- so an old process serves *new* templates with *old* builder code. A template
# asking for data the old builder never supplies renders nothing, which is how the hardware
# board became a bare title. Unconditional reuse made it survive every reopen, so the only
# escape was knowing to kill the process.


def test_a_stale_board_is_retired_and_respawned(monkeypatch) -> None:
    from leapflow.dashboard import launcher

    state = {"port": 9911, "bind": "127.0.0.1", "token": "old"}
    retired: list[bool] = []
    spawned: list[bool] = []

    monkeypatch.setattr(launcher, "server_running", lambda settings: dict(state))
    monkeypatch.setattr(launcher, "aiohttp_available", lambda: True)
    monkeypatch.setattr(launcher, "fetch_server_info", lambda *a, **k: {"stale": True})
    monkeypatch.setattr(launcher, "_retire_stale_server", lambda settings: retired.append(True))
    monkeypatch.setattr(
        launcher, "_spawn_verified_server",
        lambda *a, **k: spawned.append(True) or {"port": 9912, "token": "new"},
    )

    result = launcher.ensure_server(SimpleNamespace(dashboard_bind="127.0.0.1", dashboard_port=9911))

    assert retired == [True], "a stale server must be retired, not reused"
    assert spawned == [True]
    assert result["token"] == "new"


def test_a_current_board_is_reused(monkeypatch) -> None:
    """The common path must not pay for the check with a restart."""
    from leapflow.dashboard import launcher

    state = {"port": 9911, "bind": "127.0.0.1", "token": "live"}
    monkeypatch.setattr(launcher, "server_running", lambda settings: dict(state))
    monkeypatch.setattr(
        launcher, "fetch_server_info", lambda *a, **k: {"revision": launcher.board_revision(), "stale": False}
    )
    monkeypatch.setattr(
        launcher, "_retire_stale_server",
        lambda settings: pytest.fail("a current server must not be retired"),
    )

    result = launcher.ensure_server(SimpleNamespace(dashboard_bind="127.0.0.1", dashboard_port=9911))

    assert result["token"] == "live"


def test_only_an_unreachable_board_is_inconclusive(monkeypatch) -> None:
    """A responding Board without a generation id must be replaced, not mixed with new assets."""
    from leapflow.dashboard import launcher

    monkeypatch.setattr(launcher, "fetch_server_info", lambda *a, **k: None)
    assert launcher.server_is_stale({"port": 1, "bind": "127.0.0.1", "token": "t"}) is False
    for info in ({}, {"stale": None}, {"build": {"commit": None}}):
        monkeypatch.setattr(launcher, "fetch_server_info", lambda *a, **k: info)
        assert launcher.server_is_stale({"port": 1, "bind": "127.0.0.1", "token": "t"}) is True


def test_asset_urls_are_versioned_by_content(tmp_path, monkeypatch) -> None:
    """A cache key nobody has to remember to bump.

    The hand-written ``?v=`` string only worked if every stylesheet change was accompanied
    by editing an unrelated line in an unrelated file. It was not, and the failure was
    invisible in review and severe in the browser: new markup styled by a cached
    stylesheet, so an approval card meant to sit inside a panel rendered as an unstyled
    block at the foot of the document -- which is exactly where it was found.
    """
    from leapflow.dashboard import server as server_module

    static = tmp_path / "static"
    static.mkdir()
    (static / "styles.css").write_text("a{}", encoding="utf-8")
    (static / "app.js").write_text("void 0;", encoding="utf-8")
    index = static / "index.html"
    index.write_text(
        '<link href="/static/styles.css?v=hand-written" />'
        '<script src="/static/app.js?v=hand-written"></script>',
        encoding="utf-8",
    )
    monkeypatch.setattr(server_module, "STATIC_DIR", static)

    before = server_module._index_html(index)
    assert "hand-written" not in before, "the stale hand-written key must be replaced"
    keys = set(re.findall(r"\?v=([\w.-]+)", before))
    assert len(keys) == 1, "every asset shares one key, so one reload picks up both"

    # Editing an asset changes the key with no other edit anywhere.
    (static / "styles.css").write_text("a{color:red}", encoding="utf-8")
    after_keys = set(re.findall(r"\?v=([\w.-]+)", server_module._index_html(index)))
    assert after_keys != keys, "a changed stylesheet must produce a new cache key"

    # index.html itself is excluded: it is served no-store, so hashing it would only
    # churn the key on every edit to markup the browser never caches anyway.
    index.write_text(before, encoding="utf-8")
    assert set(re.findall(r"\?v=([\w.-]+)", server_module._index_html(index))) == after_keys


async def test_dashboard_server_freezes_static_assets_for_its_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live source edit cannot hand an old Board process new browser code."""
    pytest.importorskip("aiohttp")
    from leapflow.dashboard import server as server_module

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(
        '<script src="/static/app.js?v=old"></script>', encoding="utf-8"
    )
    (static / "app.js").write_text("const generation = 'old';", encoding="utf-8")
    monkeypatch.setattr(server_module, "STATIC_DIR", static)
    server = DashboardServer(client=_FrameClient({"ok": True}), token="t")

    (static / "app.js").write_text("const generation = 'new';", encoding="utf-8")
    asset = await server._handle_static_asset(SimpleNamespace(match_info={"name": "app.js"}))
    index = await server._handle_index(SimpleNamespace(query={"token": "t"}, headers={}))

    assert asset.body == b"const generation = 'old';"
    assert "?v=old" not in index.text
    assert server._static_version in index.text


async def test_preview_profile_values_reach_the_daemon_but_are_bounded_on_the_page_hop() -> None:
    """The page controls request a profile; PreviewBroker remains the final authority.

    This test owns only the dashboard contract: malformed or absurd URL values cannot exceed
    the binary relay profile ceiling. The broker test asserts the stronger declaration/runtime
    clamp right before a transport is called.
    """
    pytest.importorskip("aiohttp")
    import base64

    client = _FrameClient({
        "ok": True, "media_type": "image/jpeg", "width": 4, "height": 2,
        "data_b64": base64.b64encode(_JPEG).decode("ascii"),
    })
    server = DashboardServer(client=client, token="t")
    request = SimpleNamespace(
        query={
            "token": "t", "device": "cam", "channel": "frame",
            "fps": "1000", "max_width": "99999", "quality": "101",
        },
        headers={},
    )

    response = await server._handle_media_frame(request)

    assert response.status == 200
    forwarded = client.frame_requests[-1]
    assert forwarded["fps"] == 30.0
    assert forwarded["max_width"] == 4096
    assert forwarded["quality"] == 100


def test_preview_options_reject_non_numbers_and_negative_values() -> None:
    options = DashboardServer._preview_options({
        "fps": "nan", "max_width": "-5", "quality": "not-a-number",
    })
    assert options == {"fps": 0.0, "max_width": 0, "quality": 0}

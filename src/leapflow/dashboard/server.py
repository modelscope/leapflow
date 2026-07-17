"""Local dashboard web server (optional aiohttp transport, view-client process).

Holds one upstream subscription to the daemon (via DaemonClient) and fans out
monitor events to browser WebSockets through a ``ViewHub``. Serves the SDUI
frontend, a ``/api/view`` endpoint (ViewSpec for an intent), and a guarded
``/api/action`` endpoint that dispatches the bidirectional action protocol.

``aiohttp`` is imported lazily inside methods so this module (and the package)
stays importable without the optional dependency; only ``build_app``/``serve``
require it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from leapflow.dashboard.hub import ViewHub
from leapflow.dashboard.intent import DashboardIntent
from leapflow.dashboard.service import DaemonDataProvider, DashboardViewBuilder
from leapflow.dashboard.templates import TemplateLibrary
from leapflow.monitor.types import EVENT_ERROR, EVENT_FINDING, EVENT_HEARTBEAT, EVENT_WATCH_STATE

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_MONITOR_EVENTS = frozenset({EVENT_FINDING, EVENT_WATCH_STATE, EVENT_ERROR, EVENT_HEARTBEAT})
# Only these RPCs may be triggered by browser actions (least privilege).
_ALLOWED_RPC = frozenset({"watch.pause", "watch.resume", "watch.stop", "watch.refresh", "watch.mute"})


class DashboardServer:
    """aiohttp view server bridging the daemon to browsers over WebSocket."""

    def __init__(
        self,
        *,
        client: Any,
        token: str,
        bind: str = "127.0.0.1",
        port: int = 8765,
        templates: Optional[TemplateLibrary] = None,
    ) -> None:
        self._client = client
        self._token = token
        self._bind = bind
        self._port = port
        self._hub = ViewHub()
        self._builder = DashboardViewBuilder(templates or TemplateLibrary())
        self._provider = DaemonDataProvider(client)
        self._upstream_task: Optional[asyncio.Task[None]] = None

    # ── App wiring ─────────────────────────────────────────────────────────

    def build_app(self) -> Any:
        """Build the aiohttp Application (requires the optional aiohttp dep)."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/view", self._handle_view)
        app.router.add_post("/api/action", self._handle_action)
        app.router.add_get("/ws", self._handle_ws)
        if STATIC_DIR.exists():
            app.router.add_static("/static/", str(STATIC_DIR))
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def serve(self) -> None:
        """Run the server until cancelled."""
        from aiohttp import web

        app = self.build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._bind, self._port)
        await site.start()
        logger.info("dashboard serving on http://%s:%d", self._bind, self._port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()

    # ── Auth ───────────────────────────────────────────────────────────────

    def _check_token(self, request: Any) -> bool:
        token = request.query.get("token") or request.headers.get("X-Dashboard-Token", "")
        return bool(self._token) and token == self._token

    @staticmethod
    def _check_origin(request: Any) -> bool:
        origin = request.headers.get("Origin")
        if not origin:
            return True  # non-browser or same-origin requests carry no Origin
        return "127.0.0.1" in origin or "localhost" in origin

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def _handle_index(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request):
            return web.Response(status=401, text="missing or invalid token")
        index = STATIC_DIR / "index.html"
        if index.exists():
            return web.FileResponse(index)
        return web.Response(text="<h1>LeapFlow dashboard</h1>", content_type="text/html")

    async def _handle_view(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        intent = DashboardIntent.from_params({
            "template": request.query.get("template", ""),
        })
        spec = await self._builder.build(intent, self._provider)
        return web.json_response(spec)

    async def _handle_action(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request) or not self._check_origin(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed client payload
            body = {}
        result = await self.dispatch_action(body if isinstance(body, dict) else {})
        return web.json_response(result)

    async def _handle_ws(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request):
            return web.Response(status=401, text="unauthorized")
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        subscriber_id = uuid.uuid4().hex
        queue = self._hub.subscribe(subscriber_id)
        try:
            while True:
                message = await queue.get()
                if message is None:
                    break
                await ws.send_json(message)
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            self._hub.unsubscribe(subscriber_id)
        return ws

    # ── Action dispatch (transport-independent, allow-listed) ────────────────

    async def dispatch_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one action protocol message; returns a JSON-safe result."""
        kind = str(action.get("kind", ""))
        name = str(action.get("name", ""))
        params = dict(action.get("params") or {})
        if kind == "nav":
            return {"ok": True, "nav": name, "params": params}
        if kind == "rpc":
            if name not in _ALLOWED_RPC:
                return {"ok": False, "error": f"action not allowed: {name}"}
            watch_id = str(params.get("watch_id") or params.get("target") or "")
            result = await self._invoke_rpc(name, watch_id, params)
            return {"ok": True, "result": result}
        if kind == "approval":
            pending_id = str(params.get("pending_id", ""))
            decision = str(params.get("decision", "deny"))
            return {"ok": True, "result": await self._client.approval_resolve(pending_id, decision)}
        if kind == "intent":
            # Engine intents (deep-dive, storytelling) are wired in a later phase;
            # accept and acknowledge so the UI can reflect a queued request.
            return {"ok": True, "queued": True, "name": name, "params": params}
        return {"ok": False, "error": f"unknown action kind: {kind or '(missing)'}"}

    async def _invoke_rpc(self, name: str, watch_id: str, params: dict[str, Any]) -> Any:
        if name == "watch.pause":
            return await self._client.watch_pause(watch_id)
        if name == "watch.resume":
            return await self._client.watch_resume(watch_id)
        if name == "watch.stop":
            return await self._client.watch_stop(watch_id)
        if name == "watch.refresh":
            return await self._client.watch_refresh(watch_id)
        if name == "watch.mute":
            return await self._client.watch_mute(watch_id, muted=bool(params.get("muted", True)))
        return {"ok": False, "error": f"unhandled rpc: {name}"}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def _on_startup(self, _app: Any) -> None:
        self._upstream_task = asyncio.create_task(self._pump_upstream())

    async def _on_cleanup(self, _app: Any) -> None:
        if self._upstream_task is not None:
            self._upstream_task.cancel()
            try:
                await self._upstream_task
            except asyncio.CancelledError:
                pass
        await self._hub.shutdown()

    async def _pump_upstream(self) -> None:
        """Forward daemon monitor events to all browser subscribers."""
        while True:
            try:
                async for event in self._client.subscribe_notifications():
                    event_type = event.get("event_type", "")
                    if event_type in _MONITOR_EVENTS:
                        self._hub.broadcast({"type": event_type, "payload": event.get("payload") or {}})
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 - reconnect on transient upstream loss
                logger.debug("dashboard: upstream subscription lost; retrying", exc_info=True)
                await asyncio.sleep(3.0)


async def run_server(settings: Any, *, token: str, bind: str, port: int) -> int:
    """Connect to leapd and serve the dashboard until interrupted."""
    from leapflow.dashboard import launcher
    from leapflow.dashboard.templates import TemplateLibrary
    from leapflow.daemon.client import ensure_daemon_client

    client = await ensure_daemon_client(settings)
    # Profile-scoped custom templates take precedence over builtin ones.
    override_dir = None
    profile_layout = getattr(settings, "profile_layout", None)
    if profile_layout is not None:
        try:
            override_dir = profile_layout.dashboard.templates_dir
        except Exception:
            override_dir = None
    templates = TemplateLibrary(override_dir=override_dir)
    server = DashboardServer(client=client, token=token, bind=bind, port=port, templates=templates)
    launcher.write_state(settings, {
        "port": port,
        "bind": bind,
        "token": token,
        "pid": os.getpid(),
        "url": launcher.build_url(bind, port, token),
    })
    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        launcher.clear_state(settings)
    return 0


__all__ = ["DashboardServer", "run_server", "STATIC_DIR"]

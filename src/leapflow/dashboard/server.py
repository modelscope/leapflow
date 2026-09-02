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
import base64
import hashlib
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from leapflow.dashboard.hub import ViewHub
from leapflow.dashboard.intent import DashboardIntent
from leapflow.dashboard.revision import board_revision
from leapflow.dashboard.service import DaemonDataProvider, DashboardViewBuilder
from leapflow.dashboard.templates import TemplateLibrary
from leapflow.monitor.types import EVENT_ERROR, EVENT_FINDING, EVENT_HEARTBEAT, EVENT_WATCH_STATE
from leapflow.utils.build_info import StalenessMonitor, capture_build_info, is_stale

logger = logging.getLogger(__name__)


STATIC_DIR = Path(__file__).parent / "static"

_ASSET_VERSION_RE = re.compile(r'(/static/[\w.-]+)\?v=[^"\']*')


def _asset_version_for(assets: dict[str, bytes]) -> str:
    """Return the content key for one immutable static-asset snapshot."""
    digest = hashlib.sha256()
    for name in sorted(assets):
        if name != "index.html":
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(assets[name])
            digest.update(b"\0")
    return digest.hexdigest()[:12]


def _asset_version() -> str:
    """Return a cache key derived from the current asset bytes themselves.

    Kept as a standalone helper for static-template validation. A running Board uses the
    same calculation over its startup snapshot, so a live source edit cannot mix new JS
    with old server code before the launcher replaces the process generation.
    """
    assets: dict[str, bytes] = {}
    for path in STATIC_DIR.glob("*"):
        if not path.is_file():
            continue
        try:
            assets[path.name] = path.read_bytes()
        except OSError:  # pragma: no cover - unreadable asset is the static route's problem
            logger.debug("dashboard: could not hash %s", path, exc_info=True)
    return _asset_version_for(assets)


def _index_html(index: Path) -> str:
    """Return index.html with every asset reference stamped with the current version."""
    return _ASSET_VERSION_RE.sub(rf"\1?v={_asset_version()}", index.read_text(encoding="utf-8"))


_MONITOR_EVENTS = frozenset({EVENT_FINDING, EVENT_WATCH_STATE, EVENT_ERROR, EVENT_HEARTBEAT, "signal.stream"})
# Only these RPCs may be triggered by browser actions (least privilege).
#
# The hardware entries are read-or-request only, and that boundary is the whole point.
# ``configure_preview`` is a dry run: every feasibility check runs, nothing reaches the
# device. ``request_write`` submits the real change, which means the daemon puts it
# through ``ApprovalOrchestrator`` per invocation -- so the browser can *ask*, and the
# person answers where their session is authenticated. There is deliberately no entry
# that resolves an approval, because a browser session is a weaker identity than the
# TUI process that holds the approval route: letting the board approve its own request
# would make the gate ceremonial.
_ALLOWED_RPC = frozenset({
    "watch.pause", "watch.resume", "watch.stop", "watch.refresh", "watch.mute",
    "hardware.rescan", "hardware.configure_preview", "hardware.request_write",
})
# Exact loopback hosts accepted in the Origin header (see _check_origin).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_MAX_STREAM_FPS = 30.0
"""Outer bound for a browser-selected preview profile.

The daemon still enforces the stricter channel and runtime ceilings before it opens an
encoder. This clamp only rejects malformed local Board URLs before they reach the daemon.
"""


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
        # A Board generation must never combine old Python with files changed underneath it.
        # Templates and static assets are frozen in this process; a changed source tree gets
        # a new revision, which the launcher replaces before serving the next Board entry.
        self._templates = (templates or TemplateLibrary()).snapshot()
        self._static_assets = self._capture_static_assets()
        self._static_version = _asset_version_for(self._static_assets)
        self._index_snapshot = self._render_index_snapshot()
        self._builder = DashboardViewBuilder(self._templates)
        self._provider = DaemonDataProvider(client)
        self._revision = board_revision()
        self._preview_viewers: dict[str, tuple[str, str]] = {}
        self._upstream_task: Optional[asyncio.Task[None]] = None
        # Captured once at server startup so /api/server-info and the ViewSpec
        # meta can report whether this long-lived process is still fresh
        # relative to the source tree (see leapflow.utils.build_info). Wrapped
        # in a non-blocking cache so a browser refresh never waits on a git
        # subprocess.
        self._build_info = capture_build_info()
        self._build_staleness = StalenessMonitor(self._build_info)

    # ── App wiring ─────────────────────────────────────────────────────────

    def build_app(self) -> Any:
        """Build the aiohttp Application (requires the optional aiohttp dep)."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/view", self._handle_view)
        app.router.add_get("/api/server-info", self._handle_server_info)
        app.router.add_post("/api/action", self._handle_action)
        app.router.add_get("/api/media/frame", self._handle_media_frame)
        app.router.add_get("/api/media/ws", self._handle_media_ws)
        app.router.add_post("/api/media/release", self._handle_media_release)
        app.router.add_get("/api/media/level", self._handle_media_level)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/static/{name}", self._handle_static_asset)
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
            self._build_staleness.cancel_pending()
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
        # Parse the host out of the Origin and match it exactly: a substring test
        # ("127.0.0.1" in origin) is bypassable by hosts like attacker127.0.0.1.com.
        try:
            hostname = urlparse(origin).hostname
        except ValueError:
            return False
        return hostname in _LOOPBACK_HOSTS

    def _capture_static_assets(self) -> dict[str, bytes]:
        """Freeze the shipped frontend assets for this Board process generation."""
        assets: dict[str, bytes] = {}
        for path in STATIC_DIR.glob("*"):
            if not path.is_file():
                continue
            try:
                assets[path.name] = path.read_bytes()
            except OSError:
                logger.warning("dashboard: could not snapshot static asset %s", path, exc_info=True)
        return assets

    def _render_index_snapshot(self) -> str:
        """Stamp the frozen asset content key into the frozen HTML shell."""
        raw = self._static_assets.get("index.html")
        if raw is None:
            return ""
        try:
            document = raw.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("dashboard: index.html is not valid UTF-8")
            return ""
        return _ASSET_VERSION_RE.sub(
            lambda match: f"{match.group(1)}?v={self._static_version}", document
        )

    # ── Handlers ─────────────────────────────────────────────────────────────

    async def _handle_index(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request):
            return web.Response(status=401, text="missing or invalid token")
        if self._index_snapshot:
            response = web.Response(text=self._index_snapshot, content_type="text/html")
            response.headers["Cache-Control"] = "no-store"
            return response
        return web.Response(text="<h1>LeapFlow dashboard</h1>", content_type="text/html")

    async def _handle_static_asset(self, request: Any) -> Any:
        """Serve only an asset captured with this server generation, never live disk bytes."""
        from aiohttp import web

        name = str(request.match_info.get("name") or "")
        if not name or Path(name).name != name:
            return web.Response(status=404, text="asset not found")
        data = self._static_assets.get(name)
        if data is None:
            return web.Response(status=404, text="asset not found")
        content_type = {
            "app.js": "application/javascript",
            "styles.css": "text/css",
        }.get(name, "application/octet-stream")
        return web.Response(
            body=data,
            content_type=content_type,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    async def _handle_view(self, request: Any) -> Any:
        from aiohttp import web

        if not self._check_token(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        intent = DashboardIntent.from_params({
            "template": request.query.get("template", ""),
            "device": request.query.get("device", ""),
            "channel": request.query.get("channel", ""),
        })
        request_id = uuid.uuid4().hex
        try:
            spec = await self._builder.build(intent, self._provider)
        except Exception:  # noqa: BLE001 - a view failure must not become a blank HTTP 500 page
            logger.error("dashboard: view build failed request_id=%s", request_id, exc_info=True)
            return web.json_response(
                {
                    "error": {
                        "code": "view_unavailable",
                        "message": "The Board view could not be assembled. Retry after checking leapd.",
                        "request_id": request_id,
                    }
                },
                status=503,
                headers={"Cache-Control": "no-store"},
            )
        if isinstance(spec, dict):
            meta = spec.setdefault("meta", {})
            if isinstance(meta, dict):
                meta["server"] = await self._server_info()
        return web.json_response(spec, headers={"Cache-Control": "no-store"})

    async def _handle_server_info(self, request: Any) -> Any:
        """Self-report this process's build fingerprint and staleness verdict.

        A tiny, single-purpose diagnostic endpoint (distinct from the ViewSpec
        the browser renders) so ``/board status`` and ``leap daemon status``
        can check on this *separate* long-lived process without depending on
        SDUI schema shape.
        """
        from aiohttp import web

        if not self._check_token(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        return web.json_response(await self._server_info())

    async def _server_info(self) -> dict[str, Any]:
        """Build the {build, stale} payload shared by the view meta and the endpoint.

        Non-blocking: returns the last cached verdict and refreshes it in the
        background once due, rather than awaiting the git subprocess on every
        request. `is_stale` is looked up here (not bound at __init__ time) so
        tests can still `monkeypatch.setattr(server_module, "is_stale", ...)`.
        """
        stale = self._build_staleness.current(is_stale)
        return {
            "build": self._build_info.to_dict(),
            "revision": self._revision,
            "stale": stale,
        }

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
        if name == "hardware.rescan":
            return await self._client.hardware_rescan()
        if name in ("hardware.configure_preview", "hardware.request_write"):
            # One call, one flag. Both buttons reach the same daemon RPC and therefore
            # the same tool handler; a separate preview path would be a second
            # implementation of the checks that make the preview meaningful.
            return await self._client.hardware_write_request(
                str(params.get("device") or ""),
                str(params.get("channel") or ""),
                params.get("value"),
                dry_run=name == "hardware.configure_preview",
            )
        return {"ok": False, "error": f"unhandled rpc: {name}"}

    # ── Media (device preview) ───────────────────────────────────────────
    #
    # Two endpoints, both authenticated by token and both reachable only on loopback.
    # They deliberately do **not** call ``_check_origin``: an ``<img src>`` request
    # carries no ``Origin`` header at all, so requiring one would reject every legitimate
    # preview while providing no protection. What stands in for it is the token (unguessable,
    # already required for the page itself), the loopback bind, and headers that keep the
    # bytes out of caches and out of other origins.
    #
    # No policy lives here. Whether a preview is permitted is decided by the daemon's
    # approval chain on the channel's declared privacy tier; this layer forwards the
    # refusal, including the sentence naming where consent can be given.

    _MEDIA_HEADERS = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Referrer-Policy": "no-referrer",
    }

    async def _handle_media_frame(self, request: Any) -> Any:
        """Serve a single frame, or a JSON refusal a person can act on.

        The client asks for one frame before opening a stream, because an ``<img>``
        element cannot report *why* it failed -- ``onerror`` carries no body. This is the
        only place a permission refusal can be surfaced as text.
        """
        from aiohttp import web

        if not self._check_token(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        device = request.query.get("device", "")
        channel = request.query.get("channel", "")
        result = await self._fetch_frame(device, channel, **self._preview_options(request.query))
        if not result.get("ok"):
            status = 403 if result.get("code") == "consent_required" else 404
            return web.json_response(
                {"error": result.get("error", ""), "code": result.get("code", "")},
                status=status,
                headers=dict(self._MEDIA_HEADERS),
            )
        return web.Response(
            body=result["data"],
            content_type=str(result.get("media_type") or "image/jpeg"),
            headers=dict(self._MEDIA_HEADERS),
        )

    async def _handle_media_ws(self, request: Any) -> Any:
        """Relay one daemon preview stream to a browser as latest-only binary JPEG frames."""
        from aiohttp import WSMsgType, web

        if not self._check_token(request) or not self._check_origin(request):
            return web.Response(status=401, text="unauthorized")
        device = str(request.query.get("device") or "")
        channel = str(request.query.get("channel") or "")
        if not device or not channel:
            return web.json_response(
                {"error": "device and channel are required", "code": "missing_target"}, status=404
            )
        options = self._preview_options(request.query)
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        viewer_id = uuid.uuid4().hex
        self._preview_viewers[viewer_id] = (device, channel)
        await ws.send_json({"type": "opened", "viewer_id": viewer_id})
        stream = self._client.hardware_preview_stream(
            device,
            channel,
            max_width=options["max_width"],
            quality=options["quality"],
            fps=options["fps"],
            viewer_id=viewer_id,
            on_stream_event=self._forward_approval,
        )
        pending: asyncio.Task[Any] | None = None
        receiver: asyncio.Task[Any] | None = None
        try:
            pending = asyncio.create_task(anext(stream))
            receiver = asyncio.create_task(ws.receive())
            while True:
                done, _ = await asyncio.wait({pending, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if receiver in done:
                    message = receiver.result()
                    if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING, WSMsgType.ERROR):
                        break
                    receiver = asyncio.create_task(ws.receive())
                if pending not in done:
                    continue
                try:
                    frame = pending.result()
                except StopAsyncIteration:
                    break
                try:
                    data = base64.b64decode(str(frame.get("data_b64") or ""), validate=True)
                except (TypeError, ValueError):
                    await ws.send_json({
                        "type": "error",
                        "code": "decode_failed",
                        "error": "The daemon returned a preview frame the Board could not decode.",
                    })
                    break
                if not data:
                    await ws.send_json({
                        "type": "error", "code": "empty_frame", "error": "The device returned no image data."
                    })
                    break
                await ws.send_bytes(data)
                pending = asyncio.create_task(anext(stream))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the browser receives an actionable stream error
            logger.debug("dashboard: binary preview relay failed", exc_info=True)
            if not ws.closed:
                try:
                    await ws.send_json({
                        "type": "error",
                        "code": "preview_unavailable",
                        "error": str(exc) or "The preview stream stopped unexpectedly.",
                    })
                except (ConnectionResetError, ConnectionError):
                    pass
        finally:
            for task in (pending, receiver):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            try:
                await stream.aclose()
            except Exception:
                logger.debug("dashboard: failed to close daemon preview stream", exc_info=True)
            await self._release_preview_viewer(viewer_id)
            if not ws.closed:
                await ws.close()
        return ws

    async def _handle_media_release(self, request: Any) -> Any:
        """Release a browser preview owner immediately; safe to repeat after disconnect."""
        from aiohttp import web

        if not self._check_token(request) or not self._check_origin(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed release is a harmless no-op
            body = {}
        viewer_id = str((body or {}).get("viewer_id") or "")
        if not viewer_id:
            return web.json_response({"ok": True, "released": False, "active_viewers": 0})
        return web.json_response(await self._release_preview_viewer(viewer_id))

    async def _release_preview_viewer(self, viewer_id: str) -> dict[str, Any]:
        """Pop and release one server-issued viewer id without trusting browser targets."""
        target = self._preview_viewers.pop(viewer_id, None)
        if target is None:
            return {"ok": True, "released": False, "active_viewers": 0}
        device, channel = target
        try:
            result = await self._client.hardware_preview_release(device, channel, viewer_id=viewer_id)
        except AttributeError:
            result = {"ok": False, "code": "rpc_unavailable", "error": "Restart leapd to release previews."}
        except Exception as exc:  # noqa: BLE001 - a release must not trap the browser on close
            logger.debug("dashboard: preview release failed", exc_info=True)
            result = {"ok": False, "code": "release_failed", "error": str(exc)}
        return dict(result or {})

    async def _fetch_frame(
        self,
        device: str,
        channel: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
    ) -> dict[str, Any]:
        """Ask the daemon for a frame and decode it, normalising every failure.

        Returns ``{ok, data, media_type}`` or ``{ok: False, code, error}`` so both
        endpoints share one shape and one set of failure names.
        """
        if not device or not channel:
            return {"ok": False, "code": "missing_target", "error": "device and channel are required"}
        try:
            reply = await self._client.hardware_frame(
                device,
                channel,
                max_width=max_width,
                quality=quality,
                fps=fps,
                on_stream_event=self._forward_approval,
            )
        except AttributeError:
            return {
                "ok": False,
                "code": "rpc_unavailable",
                "error": "This leapd build cannot serve previews; restart the daemon.",
            }
        except Exception as exc:  # noqa: BLE001 - a dead daemon is a message, not a traceback
            logger.debug("dashboard: frame request failed", exc_info=True)
            return {"ok": False, "code": "unavailable", "error": str(exc)}
        if not reply.get("ok"):
            return {
                "ok": False,
                "code": str(reply.get("code") or "preview_failed"),
                "error": str(reply.get("error") or "The preview could not be started."),
            }
        try:
            data = base64.b64decode(str(reply.get("data_b64") or ""), validate=True)
        except (ValueError, TypeError):
            return {
                "ok": False, "code": "decode_failed",
                "error": "The daemon returned a frame this view could not decode.",
            }
        if not data:
            return {"ok": False, "code": "empty_frame", "error": "The device returned no image data."}
        return {"ok": True, "data": data, "media_type": reply.get("media_type")}

    async def _handle_media_level(self, request: Any) -> Any:
        """Serve one channel reading as JSON, for a live level meter.

        A microphone's preview is a number, not a picture, so it has its own endpoint
        rather than a frame the client would have to interpret. Gated identically -- the
        daemon reads the same declared privacy tier -- so the consent flow is the one the
        camera uses and the page raises the same prompt.
        """
        from aiohttp import web

        if not self._check_token(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        device = str(request.query.get("device") or "")
        channel = str(request.query.get("channel") or "")
        if not device or not channel:
            return web.json_response(
                {"error": "device and channel are required", "code": "missing_target"},
                status=404, headers=dict(self._MEDIA_HEADERS),
            )
        viewer_id = str(request.query.get("viewer_id") or "") or uuid.uuid4().hex
        existing = self._preview_viewers.get(viewer_id)
        if existing is not None and existing != (device, channel):
            return web.json_response(
                {"error": "preview owner belongs to a different channel", "code": "viewer_target_mismatch"},
                status=409,
                headers=dict(self._MEDIA_HEADERS),
            )
        self._preview_viewers[viewer_id] = (device, channel)
        try:
            reply = await self._client.hardware_read(
                device,
                channel,
                viewer_id=viewer_id,
                on_stream_event=self._forward_approval,
            )
        except AttributeError:
            reply = {
                "ok": False, "code": "rpc_unavailable",
                "error": "This leapd build cannot read a channel; restart the daemon.",
            }
        except Exception as exc:  # noqa: BLE001 - a dead daemon is a message, not a traceback
            logger.debug("dashboard: level read failed", exc_info=True)
            reply = {"ok": False, "code": "unavailable", "error": str(exc)}
        if reply.get("ok"):
            reply["viewer_id"] = viewer_id
        else:
            await self._release_preview_viewer(viewer_id)
        status = 200 if reply.get("ok") else (403 if reply.get("code") == "consent_required" else 404)
        return web.json_response(reply, status=status, headers=dict(self._MEDIA_HEADERS))

    def _forward_approval(self, event: Any) -> None:
        """Push an approval prompt raised by a board request out to the browsers.

        The board is a legitimate approval surface *for a request it made itself*: the
        person is looking at the page they just clicked. What makes it safe is that the
        prompt is not invented here -- the daemon's approval chain raised it, carrying the
        risk assessment and the choices the policy allows -- and the answer goes back
        through ``approval.resolve``, so the grant, the audit record and the decision
        semantics are the orchestrator's, not the board's.

        Broadcast rather than returned, because the request that raised it is still
        awaiting the answer: the only way out is the WebSocket the page already holds.
        """
        if getattr(event, "type", "") != "approval_request":
            return
        metadata = getattr(event, "metadata", None) or {}
        approval = metadata.get("approval")
        if not isinstance(approval, dict) or not approval.get("pending_id"):
            # Without a pending_id there is nothing the page could resolve, and rendering
            # an unanswerable prompt is worse than showing none.
            logger.debug("dashboard: approval prompt without a pending_id, dropped")
            return
        self._hub.broadcast({"type": "approval_request", "payload": approval})

    @staticmethod
    def _preview_options(query: Any) -> dict[str, Any]:
        """Parse a page-selected profile into bounded transport requests.

        This is an outer validation layer for malformed URLs, not the security boundary:
        ``PreviewBroker`` clamps the same values against its runtime limits and each
        channel's declaration before opening a device. The daemon remains authoritative
        for capture work and cannot be bypassed by editing a local Board URL.
        """
        def integer(name: str, ceiling: int) -> int:
            try:
                return max(0, min(ceiling, int(query.get(name, 0) or 0)))
            except (AttributeError, TypeError, ValueError):
                return 0

        try:
            fps = float(query.get("fps", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            fps = 0.0
        if not math.isfinite(fps):
            fps = 0.0
        return {
            "max_width": integer("max_width", 4096),
            "quality": integer("quality", 100),
            "fps": max(0.0, min(_MAX_STREAM_FPS, fps)),
        }

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
        for viewer_id in list(self._preview_viewers):
            await self._release_preview_viewer(viewer_id)
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
        "revision": server._revision,
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

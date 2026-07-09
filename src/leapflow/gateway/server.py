"""Gateway server — manages platform adapters and routes messages.

Intentionally thin (< 250 lines): session / transcript persistence,
agent execution, memory, and skills are all delegated to downstream
subscribers.  The gateway only owns:

- Adapter lifecycle (connect / disconnect / reconnect)
- Message normalisation (inbound → SessionKey → handler callback)
- Event notification (optional callback for loose coupling)

Designed for two deployment modes:

1. **In-process** — called directly from ``Context`` (single-window CLI)
2. **Daemon-backed** — called from ``leapd`` with ``LeapService`` RPC
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from leapflow.gateway.config_store import GatewayConfigStore
from leapflow.gateway.credential_vault import CredentialVault
from leapflow.gateway.events import (
    GatewayMessageReceived,
    GatewaySessionCreated,
    GatewaySessionEnded,
)
from leapflow.gateway.manifest import ManifestLoader, PlatformManifest
from leapflow.gateway.protocol import (
    InboundMessage,
    MessageSource,
    OutboundContent,
    PlatformAdapter,
    PlatformStatus,
    SendResult,
    SendTarget,
)
from leapflow.gateway.session_router import SessionKey, build_session_key
from leapflow.gateway.validators import validate_credentials

logger = logging.getLogger(__name__)

EventCallback = Callable[..., Any]


class GatewayServer:
    """Manages platform adapter lifecycle and message routing."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        extra_manifest_dirs: Optional[List[Path]] = None,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        self._profile_dir = profile_dir
        self._vault = CredentialVault(profile_dir)
        self._config_store = GatewayConfigStore(profile_dir, self._vault)
        self._manifest_loader = ManifestLoader(extra_dirs=extra_manifest_dirs)
        self._manifests: Dict[str, PlatformManifest] = {}
        self._adapters: Dict[str, PlatformAdapter] = {}
        self._connected_since: Dict[str, float] = {}
        self._known_sessions: set[SessionKey] = set()
        self._message_handler: Optional[Callable[..., Any]] = None
        self._on_event = on_event
        self._started = False

    # ── Manifest discovery ───────────────────────────────────

    def discover_manifests(self) -> Dict[str, PlatformManifest]:
        """Load all available platform manifests."""
        self._manifests = self._manifest_loader.discover()
        self._config_store.set_manifests(self._manifests)
        return self._manifests

    @property
    def manifests(self) -> Dict[str, PlatformManifest]:
        if not self._manifests:
            self.discover_manifests()
        return self._manifests

    def set_message_handler(self, handler: Callable[..., Any]) -> None:
        """Set the callback for inbound messages (typically engine dispatch)."""
        self._message_handler = handler

    # ── Platform connection ──────────────────────────────────

    async def connect_platform(
        self,
        platform_id: str,
        credentials: Dict[str, str],
        options: Optional[Dict[str, Any]] = None,
        *,
        is_reconnect: bool = False,
    ) -> Dict[str, Any]:
        """Validate, persist, and connect a platform adapter.

        Called by the ``gateway_connect`` tool during conversation, or
        by ``start()`` for auto-reconnect.  When *is_reconnect* is
        ``True``, adapters preserve server-side message queues.
        The return value **never** contains credential values.
        """
        manifest = self._manifests.get(platform_id)
        if manifest is None:
            return {"ok": False, "error": f"Unknown platform: {platform_id}"}

        for cred in manifest.credentials:
            if cred.required and not credentials.get(cred.key):
                return {"ok": False, "error": f"Missing required field: {cred.label}"}

        validation_info = ""
        if manifest.validation_method:
            ok, msg = await validate_credentials(
                manifest.validation_method,
                credentials,
                timeout_s=manifest.validation_timeout_s,
            )
            if not ok:
                return {"ok": False, "error": msg}
            validation_info = msg

        self._config_store.save_platform(
            platform_id, credentials, options or {}, manifest,
        )

        if manifest.adapter:
            try:
                adapter = self._instantiate_adapter(
                    manifest, credentials, options or {},
                )
                adapter.on_message = self._on_inbound_message
                await adapter.connect(is_reconnect=is_reconnect)
                self._adapters[platform_id] = adapter
                self._connected_since[platform_id] = time.time()
            except Exception as exc:
                from leapflow.security.redact import redact_sensitive_text

                safe_err = redact_sensitive_text(str(exc), force=True)
                logger.error("Failed to connect %s: %s", platform_id, safe_err)
                return {"ok": False, "error": f"Connection failed: {safe_err}"}

        display = manifest.display_name
        return {
            "ok": True,
            "status": "connected",
            "platform": display,
            "info": validation_info,
        }

    async def disconnect_platform(self, platform_id: str) -> Dict[str, Any]:
        """Disconnect a platform adapter and clean up its sessions."""
        adapter = self._adapters.pop(platform_id, None)
        self._connected_since.pop(platform_id, None)

        affected = {k for k in self._known_sessions if k.platform == platform_id}
        self._known_sessions -= affected
        for key in affected:
            await self._emit_event(GatewaySessionEnded(
                session_key=str(key),
                reason="platform_disconnect",
            ))

        if adapter:
            try:
                await adapter.disconnect()
            except Exception:
                logger.warning(
                    "Error disconnecting %s", platform_id, exc_info=True,
                )
        return {"ok": True, "status": "disconnected"}

    def platform_status(self) -> List[PlatformStatus]:
        """Return status of all known platforms."""
        result: List[PlatformStatus] = []
        config = self._config_store.load()
        for pid in self._manifests:
            if pid in self._adapters:
                result.append(PlatformStatus(
                    platform_id=pid,
                    connected=True,
                    connected_since=self._connected_since.get(pid, 0),
                ))
            elif pid in config.platforms and config.platforms[pid].enabled:
                result.append(PlatformStatus(
                    platform_id=pid,
                    connected=False,
                    error="configured but not connected",
                ))
            else:
                result.append(PlatformStatus(
                    platform_id=pid,
                    connected=False,
                ))
        return result

    # ── Lifecycle ────────────────────────────────────────────

    async def start(self) -> int:
        """Auto-connect all platforms in ``auto_connect`` list.

        Returns the number of successfully connected platforms.
        """
        self.discover_manifests()
        config = self._config_store.load()
        connected = 0
        for pid in config.auto_connect:
            manifest = self._manifests.get(pid)
            pc = config.platforms.get(pid)
            if manifest and pc and pc.enabled:
                creds = self._config_store.load_platform_credentials(
                    pid, manifest,
                )
                if creds:
                    result = await self.connect_platform(
                        pid, creds, pc.options,
                        is_reconnect=True,
                    )
                    if result.get("ok"):
                        connected += 1
        self._started = True
        return connected

    async def stop(self) -> None:
        """Disconnect all adapters."""
        for pid in list(self._adapters):
            await self.disconnect_platform(pid)
        self._started = False

    # ── Reply delivery ────────────────────────────────────────

    async def send_reply(
        self,
        source: MessageSource,
        text: str,
    ) -> Optional["SendResult"]:
        """Send a reply back to the originating conversation.

        Public API for ``GatewayRouter`` and any other component that
        needs to deliver outbound messages.  Returns ``None`` if no
        adapter is connected for the platform.
        """
        adapter = self._adapters.get(source.platform)
        if adapter is None:
            logger.warning(
                "No adapter for %s, cannot send reply", source.platform,
            )
            return None
        target = SendTarget(
            platform=source.platform,
            chat_id=source.chat_id,
            thread_id=source.thread_id,
        )
        content = OutboundContent(text=text)
        return await adapter.send(target, content)

    # ── Message routing ──────────────────────────────────────

    async def _on_inbound_message(self, message: InboundMessage) -> None:
        """Route an inbound message via ``SessionKey``."""
        session_key = build_session_key(message.source)
        is_new_session = session_key not in self._known_sessions

        if is_new_session:
            self._known_sessions.add(session_key)
            await self._emit_event(GatewaySessionCreated(
                session_key=str(session_key),
                source=message.source,
            ))

        await self._emit_event(GatewayMessageReceived(
            source=message.source,
            session_key=str(session_key),
            text=message.text,
            media_urls=tuple(m.url for m in message.media),
        ))

        if self._message_handler is None:
            logger.warning(
                "No message handler set, dropping message from %s",
                message.source.platform,
            )
            return

        try:
            result = self._message_handler(message, str(session_key))
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.error(
                "Error handling message from %s",
                message.source.platform,
                exc_info=True,
            )

    async def _emit_event(self, event: object) -> None:
        """Dispatch an event to the optional callback (non-blocking)."""
        if self._on_event is None:
            return
        try:
            result = self._on_event(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.debug("Event callback error", exc_info=True)

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _instantiate_adapter(
        manifest: PlatformManifest,
        credentials: Dict[str, str],
        options: Dict[str, Any],
    ) -> PlatformAdapter:
        """Dynamically import and instantiate a platform adapter.

        Adapter modules are loaded on-demand — platforms whose SDK is not
        installed never affect startup (deferred loading).  If the import
        fails due to missing dependencies, the error message includes a
        ``pip install`` hint derived from the manifest.
        """
        assert manifest.adapter is not None
        try:
            module = importlib.import_module(manifest.adapter.module)
        except ImportError:
            deps = manifest.adapter.dependencies
            if deps:
                hint = f"pip install {' '.join(deps)}"
                raise ImportError(
                    f"Adapter module '{manifest.adapter.module}' not found.  "
                    f"Install dependencies: {hint}",
                ) from None
            raise
        cls = getattr(module, manifest.adapter.class_name)
        return cls(**credentials, **options)

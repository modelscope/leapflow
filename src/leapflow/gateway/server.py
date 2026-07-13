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
from leapflow.gateway.connectors.action_registry import summarize_action_result
from leapflow.gateway.connectors.protocol import ActionPreview, ActionSpec, BackendEventSource, EventSourceStatus
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
        self._event_sources: Dict[str, BackendEventSource] = {}
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

                metadata = getattr(exc, "metadata", {})
                safe_err = redact_sensitive_text(str(exc), force=True)
                logger.error("Failed to connect %s: %s", platform_id, safe_err)
                response: Dict[str, Any] = {"ok": False, "error": f"Connection failed: {safe_err}"}
                if isinstance(metadata, dict):
                    if metadata.get("recovery_hint"):
                        response["recovery_hint"] = metadata["recovery_hint"]
                    if metadata.get("next_steps"):
                        response["next_steps"] = metadata["next_steps"]
                    response["diagnostics"] = self._safe_metadata(metadata)
                return response

        display = manifest.display_name
        response = {
            "ok": True,
            "status": "connected",
            "platform": display,
            "info": validation_info,
        }
        adapter = self._adapters.get(platform_id)
        if adapter is not None:
            diagnostics = self._adapter_metadata(adapter)
            if diagnostics:
                response["diagnostics"] = diagnostics
        return response

    async def disconnect_platform(self, platform_id: str) -> Dict[str, Any]:
        """Disconnect a platform adapter and clean up its sessions."""
        adapter = self._adapters.pop(platform_id, None)
        self._event_sources.pop(platform_id, None)
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
                adapter = self._adapters[pid]
                result.append(PlatformStatus(
                    platform_id=pid,
                    connected=True,
                    connected_since=self._connected_since.get(pid, 0),
                    metadata=self._adapter_metadata(adapter),
                ))
            elif pid in config.platforms and config.platforms[pid].enabled:
                metadata = self._configured_platform_metadata(pid, config.platforms[pid].options)
                result.append(PlatformStatus(
                    platform_id=pid,
                    connected=False,
                    error="configured but not connected",
                    metadata=metadata,
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
            if not (manifest and pc and pc.enabled):
                continue
            try:
                creds = self._config_store.load_platform_credentials(
                    pid, manifest,
                )
                if creds is None:
                    continue
                result = await self.connect_platform(
                    pid, creds, pc.options,
                    is_reconnect=True,
                )
                if result.get("ok"):
                    connected += 1
            except Exception as exc:
                logger.warning(
                    "gateway.auto_connect_failed platform=%s error=%s",
                    pid,
                    exc,
                    exc_info=True,
                )
        self._started = True
        return connected

    async def stop(self) -> None:
        """Disconnect all adapters."""
        for pid in list(self._adapters):
            await self.disconnect_platform(pid)
        self._started = False

    # ── Outbound messaging ──────────────────────────────────

    async def send_message(
        self,
        platform_id: str,
        chat_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
    ) -> Dict[str, Any]:
        """Send a message to any connected platform conversation.

        Public API for the ``gateway_send`` tool and any component that
        needs proactive outbound messaging.  Unlike ``send_reply``,
        this does not require a prior inbound ``MessageSource``.
        """
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return {"ok": False, "error": f"Platform '{platform_id}' is not connected"}
        target = SendTarget(
            platform=platform_id,
            chat_id=chat_id,
            thread_id=thread_id,
            reply_to_id=reply_to_id,
        )
        content = OutboundContent(text=text)
        try:
            result = await adapter.send(target, content)
            return {
                "ok": result.ok,
                "message_id": result.message_id,
                **({"error": result.error} if result.error else {}),
            }
        except Exception as exc:
            from leapflow.security.redact import redact_sensitive_text
            safe_err = redact_sensitive_text(str(exc), force=True)
            return {"ok": False, "error": f"Send failed: {safe_err}"}

    def get_platform_action_spec(self, platform_id: str, action: str) -> ActionSpec | None:
        """Return a registered action spec for a connected platform."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return None
        action_spec = getattr(adapter, "action_spec", None)
        if action_spec is not None:
            return action_spec(action)
        action_specs = getattr(adapter, "action_specs", None)
        if action_specs is None:
            return None
        specs = action_specs()
        return specs.get(action) if isinstance(specs, dict) else None

    async def preview_platform_action(
        self,
        platform_id: str,
        action: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a side-effect-free action preview for approval UX."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return {"ok": False, "error": f"Platform '{platform_id}' is not connected"}
        preview = getattr(adapter, "preview_action", None)
        if preview is None:
            spec = self.get_platform_action_spec(platform_id, action)
            if spec is None:
                return {"ok": False, "error": f"Unknown platform action: {action}"}
            return {"ok": True, "summary": f"Run {platform_id}.{action}"}
        result: ActionPreview = await preview(action, payload)
        return {
            "ok": result.ok,
            "summary": result.summary,
            "data": dict(result.data),
            **({"error": result.error} if result.error else {}),
        }

    async def execute_platform_action(
        self,
        platform_id: str,
        action: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a registered platform action via the connected adapter."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return {"ok": False, "error": f"Platform '{platform_id}' is not connected"}
        execute = getattr(adapter, "execute_action", None)
        if execute is None:
            return {"ok": False, "error": f"Platform '{platform_id}' does not support platform actions"}
        try:
            spec = self.get_platform_action_spec(platform_id, action)
            result = await execute(action, payload)
            if spec is not None:
                summary = summarize_action_result(spec, result)
                summary.update({"platform": platform_id, "action": action})
                return summary
            return {
                "ok": bool(getattr(result, "ok", False)),
                "data": dict(getattr(result, "data", {}) or {}),
                "resource_id": str(getattr(result, "resource_id", "") or ""),
                **({"error": str(getattr(result, "error", ""))} if getattr(result, "error", "") else {}),
            }
        except Exception as exc:
            from leapflow.security.redact import redact_sensitive_text
            safe_err = redact_sensitive_text(str(exc), force=True)
            return {"ok": False, "error": f"Platform action failed: {safe_err}"}

    async def start_platform_events(
        self,
        platform_id: str,
        *,
        checkpoint: str = "",
    ) -> Dict[str, Any]:
        """Start the backend event source for a connected platform."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return {"ok": False, "error": f"Platform '{platform_id}' is not connected"}
        event_source_factory = getattr(adapter, "event_source", None)
        if event_source_factory is None:
            return {"ok": False, "error": f"Platform '{platform_id}' has no backend event source"}
        source = event_source_factory()
        if source is None:
            return {"ok": False, "error": f"Platform '{platform_id}' has no backend event source"}
        status: EventSourceStatus = await source.start(checkpoint=checkpoint)
        if status.ok:
            self._event_sources[platform_id] = source
        return self._event_status_dict(status)

    async def stop_platform_events(self, platform_id: str) -> Dict[str, Any]:
        """Stop the backend event source for a connected platform."""
        source = self._event_sources.pop(platform_id, None)
        if source is None:
            return {"ok": True, "status": "stopped"}
        status = await source.stop()
        return self._event_status_dict(status)

    async def platform_event_status(self, platform_id: str) -> Dict[str, Any]:
        """Return backend event source status for a connected platform."""
        source = self._event_sources.get(platform_id)
        if source is None:
            adapter = self._adapters.get(platform_id)
            event_source_factory = getattr(adapter, "event_source", None) if adapter else None
            source = event_source_factory() if event_source_factory else None
        if source is None:
            return {"ok": False, "error": f"Platform '{platform_id}' has no backend event source"}
        status = await source.status()
        return self._event_status_dict(status)

    @staticmethod
    def _event_status_dict(status: EventSourceStatus) -> Dict[str, Any]:
        return {
            "ok": status.ok,
            "backend_kind": status.backend_kind,
            "detail": status.detail,
            "checkpoint": status.checkpoint,
            "metadata": dict(status.metadata),
        }

    def _adapter_metadata(self, adapter: PlatformAdapter) -> Dict[str, Any]:
        metadata_factory = getattr(adapter, "status_metadata", None)
        if metadata_factory is None:
            return {}
        try:
            metadata = metadata_factory()
        except Exception:
            logger.debug("platform.status_metadata_failed", exc_info=True)
            return {}
        return self._safe_metadata(metadata if isinstance(metadata, dict) else {})

    def _configured_platform_metadata(self, platform_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
        manifest = self._manifests.get(platform_id)
        metadata: Dict[str, Any] = {"configured": True}
        if manifest is not None:
            metadata["backend_kind"] = str(manifest.backend.get("kind") or "")
            if manifest.actions:
                metadata["actions"] = dict(manifest.actions)
        if options:
            for key in ("profile", "identity", "binary"):
                if options.get(key):
                    metadata[key] = options[key]
        metadata["recovery_hint"] = "Reconnect the platform or inspect backend status for details."
        return self._safe_metadata(metadata)

    @staticmethod
    def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        from leapflow.security.redact import redact_sensitive_text

        safe: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                safe[key] = redact_sensitive_text(value, force=True)
            elif isinstance(value, list):
                safe[key] = [redact_sensitive_text(str(item), force=True) for item in value]
            elif isinstance(value, dict):
                safe[key] = GatewayServer._safe_metadata(value)
            else:
                safe[key] = value
        return safe

    async def send_reply(
        self,
        source: MessageSource,
        text: str,
    ) -> Optional[SendResult]:
        """Send a reply back to the originating conversation.

        Convenience wrapper around ``send_message`` for ``GatewayRouter``.
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

    def remove_platform_config(self, platform_id: str) -> None:
        """Remove a platform's saved configuration from disk.

        Public API — avoids callers needing direct ``_config_store`` access.
        """
        self._config_store.remove_platform(platform_id)

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
        except ImportError as exc:
            deps = manifest.adapter.dependencies
            missing_name = getattr(exc, "name", "") or ""
            module_name = manifest.adapter.module
            module_missing = missing_name == module_name or module_name.startswith(f"{missing_name}.")
            if deps and module_missing:
                hint = f"pip install {' '.join(deps)}"
                raise ImportError(
                    f"Adapter module '{module_name}' not found.  "
                    f"Install dependencies: {hint}",
                ) from None
            raise
        cls = getattr(module, manifest.adapter.class_name)
        return cls(**credentials, **options)

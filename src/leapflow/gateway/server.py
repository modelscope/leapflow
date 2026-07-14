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
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from leapflow.gateway.capability_health import CapabilityHealthLedger
from leapflow.gateway.checkpoint_store import CheckpointStore, DeduplicationStore
from leapflow.gateway.resource_provenance import ResourceProvenancePool
from leapflow.gateway.config_store import GatewayConfigStore
from leapflow.gateway.connectors.action_registry import summarize_action_result
from leapflow.gateway.connectors.event_sources import UnavailableEventSource
from leapflow.gateway.connectors.protocol import (
    ActionPreview,
    ActionSpec,
    BackendEvent,
    BackendEventSource,
    EventClassification,
    EventKind,
    EventSourceStatus,
    InboundCallback,
    PlatformEventNormalizer,
)
from leapflow.gateway.trigger_policy import TriggerPolicy, _RateTracker
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

_MARKDOWN_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"^#{1,6}\s", re.MULTILINE),
    re.compile(r"^\s*[-*+]\s", re.MULTILINE),
    re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    re.compile(r"\*\*.+?\*\*"),
    re.compile(r"`.+?`"),
    re.compile(r"\[.+?\]\(.+?\)"),
)


def _has_rich_formatting(text: str) -> bool:
    """Detect markdown-like formatting in text."""
    if not text or len(text) < 10:
        return False
    matches = sum(1 for p in _MARKDOWN_PATTERNS if p.search(text))
    return matches >= 2


_DEFAULT_DEDUP_CAPACITY = 10_000


def _chunk_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks respecting paragraph/line boundaries.

    Prefers splitting at paragraph boundaries (double newline), then
    single newlines, then sentence endings, then at max_len as fallback.
    """
    if not text or max_len <= 0:
        return [text] if text else []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        split_at = max_len
        for sep in ("\n\n", "\n", "。", ". ", "！", "! ", "？", "? "):
            pos = remaining.rfind(sep, 0, max_len)
            if pos > max_len // 4:
                split_at = pos + len(sep)
                break
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks or [text]


class EventDeduplicator:
    """LRU-based event_id dedup cache for the consumer loop.

    Prevents duplicate processing when a platform redelivers events
    (e.g. after reconnection).  In-memory only; checkpoint persistence
    covers the restart case.
    """

    __slots__ = ("_capacity", "_seen")

    def __init__(self, capacity: int = _DEFAULT_DEDUP_CAPACITY) -> None:
        self._capacity = max(1, capacity)
        self._seen: OrderedDict[str, None] = OrderedDict()

    def is_duplicate(self, event_id: str) -> bool:
        """Return True if the event_id has been seen before."""
        if not event_id:
            return False
        if event_id in self._seen:
            self._seen.move_to_end(event_id)
            return True
        self._seen[event_id] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)
        return False

    def load_from_store(
        self, platform_id: str, store: DeduplicationStore,
    ) -> int:
        """Pre-seed the cache from persistent storage."""
        ids = store.load_recent(platform_id, limit=self._capacity)
        for eid in reversed(ids):
            if eid not in self._seen:
                self._seen[eid] = None
                if len(self._seen) > self._capacity:
                    self._seen.popitem(last=False)
        return len(ids)

    def save_to_store(
        self, platform_id: str, store: DeduplicationStore,
    ) -> None:
        """Persist current cache to the dedup store."""
        ids = list(self._seen.keys())
        if ids:
            store.save_batch(platform_id, ids)


class GatewayServer:
    """Manages platform adapter lifecycle and message routing."""

    def __init__(
        self,
        profile_dir: Path,
        *,
        extra_manifest_dirs: Optional[List[Path]] = None,
        on_event: Optional[EventCallback] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        dedup_store: Optional[DeduplicationStore] = None,
    ) -> None:
        self._profile_dir = profile_dir
        self._vault = CredentialVault(profile_dir)
        self._checkpoint_store = checkpoint_store
        self._dedup_store = dedup_store
        self._config_store = GatewayConfigStore(profile_dir, self._vault)
        self._manifest_loader = ManifestLoader(extra_dirs=extra_manifest_dirs)
        self._manifests: Dict[str, PlatformManifest] = {}
        self._adapters: Dict[str, PlatformAdapter] = {}
        self._event_sources: Dict[str, BackendEventSource] = {}
        self._consumer_tasks: Dict[str, asyncio.Task[None]] = {}
        self._normalizers: Dict[str, PlatformEventNormalizer] = {}
        self._trigger_policies: Dict[str, TriggerPolicy] = {}
        self._rate_trackers: Dict[str, _RateTracker] = {}
        self._deduplicators: Dict[str, EventDeduplicator] = {}
        self._last_event_ids: Dict[str, str] = {}
        self._connected_since: Dict[str, float] = {}
        self._known_sessions: set[SessionKey] = set()
        self._message_handler: Optional[Callable[..., Any]] = None
        self._callback_handler: Optional[Callable[..., Any]] = None
        self._on_event = on_event
        self._started = False
        self._capability_health = CapabilityHealthLedger()
        self._resource_provenance = ResourceProvenancePool()

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

    def set_callback_handler(self, handler: Callable[..., Any]) -> None:
        """Set the callback for inbound interactive callbacks (card actions, etc.)."""
        self._callback_handler = handler

    def register_normalizer(
        self, platform_id: str, normalizer: PlatformEventNormalizer,
    ) -> None:
        """Register a platform event normalizer for the consumer loop."""
        self._normalizers[platform_id] = normalizer

    def register_trigger_policy(
        self, platform_id: str, policy: TriggerPolicy,
    ) -> None:
        """Register a trigger policy for the consumer loop."""
        self._trigger_policies[platform_id] = policy
        self._rate_trackers[platform_id] = _RateTracker()

    def platform_options(self, platform_id: str) -> dict[str, Any]:
        """Return stored options for a configured platform (public API)."""
        config = self._config_store.load()
        pc = config.platforms.get(platform_id)
        return dict(pc.options) if pc and pc.options else {}

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
        self._capability_health.clear(platform_id)
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
        await self._cancel_consumer_task(platform_id)
        adapter = self._adapters.pop(platform_id, None)
        source = self._event_sources.pop(platform_id, None)
        if source is not None:
            try:
                await source.stop()
            except Exception:
                logger.debug("Error stopping event source for %s", platform_id, exc_info=True)
        self._connected_since.pop(platform_id, None)
        self._capability_health.clear(platform_id)

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
                    await self._auto_start_events(pid)
            except Exception as exc:
                logger.warning(
                    "gateway.auto_connect_failed platform=%s error=%s",
                    pid,
                    exc,
                    exc_info=True,
                )
        self._started = True
        return connected

    async def _auto_start_events(self, platform_id: str) -> None:
        """Start event source for a platform if adapter reports one."""
        try:
            result = await self.start_platform_events(platform_id)
            if result.get("ok"):
                logger.info("Auto-started event source for %s", platform_id)
            else:
                logger.debug(
                    "Event source not started for %s: %s",
                    platform_id, result.get("detail") or result.get("error", ""),
                )
        except Exception:
            logger.warning(
                "Failed to auto-start events for %s",
                platform_id, exc_info=True,
            )

    async def stop(self) -> None:
        """Disconnect all adapters and stop all consumer tasks."""
        tasks_to_await = []
        for task in self._consumer_tasks.values():
            if not task.done():
                task.cancel()
                tasks_to_await.append(task)
        self._consumer_tasks.clear()

        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)

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
        response = {
            "ok": result.ok,
            "summary": result.summary,
            "data": dict(result.data),
            **({"error": result.error} if result.error else {}),
        }
        if result.failure is not None:
            response.update(result.failure.as_dict())
        return response

    def check_platform_action_feasibility(
        self,
        platform_id: str,
        action: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Check whether an action is known to be executable before approval."""
        spec = self.get_platform_action_spec(platform_id, action)
        if spec is None:
            return {"ok": True}
        return self._capability_health.check_feasibility(platform_id, spec)

    def capability_health_summary(self) -> List[Dict[str, Any]]:
        """Return compact non-secret capability health diagnostics."""
        return self._capability_health.summary()

    @property
    def resource_provenance(self) -> ResourceProvenancePool:
        """Return the session-scoped resource provenance pool."""
        return self._resource_provenance

    def _collect_all_resource_fields(self, platform_id: str) -> tuple[str, ...]:
        """Return all declared resource_fields across a platform's action specs."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return ()
        action_specs_fn = getattr(adapter, "action_specs", None)
        if action_specs_fn is None:
            return ()
        specs = action_specs_fn()
        if not isinstance(specs, dict):
            return ()
        fields: set[str] = set()
        for spec in specs.values():
            auth = getattr(spec, "auth", None)
            if auth is not None:
                for f in getattr(auth, "resource_fields", ()):
                    fields.add(str(f))
        return tuple(sorted(fields))

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
                if not result.ok and result.failure is not None:
                    self._capability_health.record_failure(
                        platform_id,
                        spec.capability or spec.name,
                        result.failure,
                    )
                elif result.ok:
                    resource_fields = self._collect_all_resource_fields(platform_id)
                    if resource_fields and result.data:
                        self._resource_provenance.register_from_result(
                            platform_id, resource_fields, result.data,
                        )
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
        """Start the backend event source and consumer loop for a platform."""
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return {"ok": False, "error": f"Platform '{platform_id}' is not connected"}
        event_source_factory = getattr(adapter, "event_source", None)
        if event_source_factory is None:
            return {"ok": False, "error": f"Platform '{platform_id}' has no backend event source"}
        source = event_source_factory()
        if source is None:
            return {"ok": False, "error": f"Platform '{platform_id}' has no backend event source"}
        if isinstance(source, UnavailableEventSource):
            unavailable_status = await source.status()
            return self._event_status_dict(unavailable_status)

        if not checkpoint and self._checkpoint_store is not None:
            checkpoint = self._checkpoint_store.load(platform_id)
            if checkpoint:
                logger.info("Resuming %s from checkpoint %s", platform_id, checkpoint[:20])

        status: EventSourceStatus = await source.start(checkpoint=checkpoint)
        if not status.ok:
            return self._event_status_dict(status)

        self._event_sources[platform_id] = source
        await self._cancel_consumer_task(platform_id)

        task = asyncio.create_task(
            self._consume_platform_events(platform_id, source),
            name=f"gateway-consumer-{platform_id}",
        )
        self._consumer_tasks[platform_id] = task
        return self._event_status_dict(status)

    async def stop_platform_events(self, platform_id: str) -> Dict[str, Any]:
        """Stop the backend event source and consumer task for a platform."""
        await self._cancel_consumer_task(platform_id)
        source = self._event_sources.pop(platform_id, None)
        if source is None:
            return {"ok": True, "status": "stopped"}
        self._save_checkpoint(platform_id)
        self._save_dedup_state(platform_id)
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
        Automatically chunks text that exceeds the adapter's max_message_length.
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
        max_len = getattr(adapter, "max_message_length", 0) or 8000
        chunks = _chunk_text(text, max_len)
        last_result: Optional[SendResult] = None
        for chunk in chunks:
            metadata: Dict[str, Any] = {}
            if _has_rich_formatting(chunk):
                metadata["format_hint"] = "markdown"
            content = OutboundContent(text=chunk, metadata=metadata)
            last_result = await adapter.send(target, content)
            if last_result and not last_result.ok:
                return last_result
        return last_result

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

    async def _on_inbound_callback(self, callback: "InboundCallback") -> None:
        """Route an inbound callback (card button click, form submit, etc.)."""
        session_key = build_session_key(callback.source)
        await self._emit_event(callback)

        if self._callback_handler is None:
            logger.debug(
                "No callback handler set, dropping callback %s from %s",
                callback.callback_id, callback.source.platform,
            )
            return

        try:
            result = self._callback_handler(callback, str(session_key))
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.error(
                "Error handling callback from %s",
                callback.source.platform,
                exc_info=True,
            )

    # ── Event consumer loop ────────────────────────────────────

    async def _consume_platform_events(
        self,
        platform_id: str,
        source: BackendEventSource,
    ) -> None:
        """Background task that reads events and routes them."""
        normalizer = self._normalizers.get(platform_id)
        policy = self._trigger_policies.get(platform_id)
        rate_tracker = self._rate_trackers.get(platform_id)
        dedup = self._deduplicators.setdefault(platform_id, EventDeduplicator())

        if self._dedup_store is not None:
            loaded = dedup.load_from_store(platform_id, self._dedup_store)
            if loaded:
                logger.info("Loaded %d dedup entries for %s", loaded, platform_id)

        self._inject_bot_identity(platform_id, normalizer)
        try:
            async for event in source.events():
                if dedup.is_duplicate(event.event_id):
                    logger.debug("Dedup: skipping duplicate event %s", event.event_id)
                    continue
                try:
                    await self._process_backend_event(
                        platform_id, event, normalizer, policy, rate_tracker,
                    )
                    if event.event_id:
                        self._last_event_ids[platform_id] = event.event_id
                except Exception:
                    logger.error(
                        "Error processing event %s from %s",
                        event.event_id, platform_id, exc_info=True,
                    )
        except asyncio.CancelledError:
            logger.info("Consumer task cancelled for %s", platform_id)
            self._save_checkpoint(platform_id)
            self._save_dedup_state(platform_id)
        except Exception:
            logger.error(
                "Consumer loop crashed for %s", platform_id, exc_info=True,
            )

    async def _process_backend_event(
        self,
        platform_id: str,
        event: BackendEvent,
        normalizer: PlatformEventNormalizer | None,
        policy: TriggerPolicy | None,
        rate_tracker: _RateTracker | None,
    ) -> None:
        """Classify and route one backend event."""
        if normalizer is None:
            logger.debug("No normalizer for %s, emitting raw event", platform_id)
            await self._emit_event(event)
            return

        classification = normalizer.classify(event)

        if classification.kind == EventKind.IGNORED:
            return

        if classification.kind == EventKind.MESSAGE and classification.message:
            if policy and not policy.should_activate(
                classification.message, rate_tracker=rate_tracker,
            ):
                await self._emit_event(GatewayMessageReceived(
                    source=classification.message.source,
                    session_key="",
                    text=classification.message.text,
                    media_urls=tuple(m.url for m in classification.message.media),
                ))
                return
            await self._on_inbound_message(classification.message)
            return

        if classification.kind == EventKind.CALLBACK and classification.callback:
            await self._on_inbound_callback(classification.callback)
            return

        if classification.raw_event:
            await self._emit_event(classification.raw_event)

    def _inject_bot_identity(
        self,
        platform_id: str,
        normalizer: PlatformEventNormalizer | None,
    ) -> None:
        """Inject adapter-resolved bot identity into the normalizer.

        Uses duck typing — works with any adapter that exposes
        ``bot_identity`` with ``open_id`` / ``app_name`` attributes.
        """
        if normalizer is None:
            return
        adapter = self._adapters.get(platform_id)
        if adapter is None:
            return
        identity = getattr(adapter, "bot_identity", None)
        if identity is None:
            return
        open_id = getattr(identity, "open_id", "")
        app_name = getattr(identity, "app_name", "")
        if open_id and hasattr(normalizer, "bot_id"):
            normalizer.bot_id = open_id
            logger.debug("Injected bot_id=%s… for %s", open_id[:10], platform_id)
        if app_name and hasattr(normalizer, "bot_name"):
            normalizer.bot_name = app_name

    def _save_dedup_state(self, platform_id: str) -> None:
        """Persist dedup cache for a platform to the store."""
        if self._dedup_store is None:
            return
        dedup = self._deduplicators.get(platform_id)
        if dedup is not None:
            dedup.save_to_store(platform_id, self._dedup_store)
            logger.debug("Saved dedup state for %s", platform_id)

    def _save_checkpoint(self, platform_id: str) -> None:
        """Persist the last-consumed event_id for a platform."""
        event_id = self._last_event_ids.get(platform_id)
        if event_id and self._checkpoint_store is not None:
            self._checkpoint_store.save(platform_id, event_id)
            logger.debug("Saved checkpoint %s for %s", event_id[:20], platform_id)

    async def _cancel_consumer_task(self, platform_id: str) -> None:
        """Cancel and await a consumer task if one exists."""
        task = self._consumer_tasks.pop(platform_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

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

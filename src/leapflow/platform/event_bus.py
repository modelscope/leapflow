"""Event bus: handles server-pushed events and routes them to memory/skills.

Pipeline: OSHost push → EventBus → Normalizer → EpisodicMemory → (promotion) → SemanticMemory
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from leapflow.domain.events import SystemEvent
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.platform.normalizer import EventNormalizer
from leapflow.platform.protocol import EventTypes
from leapflow.platform.reorder_buffer import EventReorderBuffer

if TYPE_CHECKING:
    from leapflow.learning.event_consumer import EventConsumer
    from leapflow.privacy.policy import EventPrivacyFilter

logger = logging.getLogger(__name__)

EventCallback = Callable[[SystemEvent], None]

_DEDUP_WINDOW_S = 2.0


class EventBus:
    """Routes incoming host events through normalization into the memory subsystem.

    Separates raw platform events from semantic processing via EventNormalizer.
    Optional subscribers receive normalized SystemEvents for downstream use (e.g. adapters).
    """

    def __init__(
        self,
        immediate: EpisodicMemoryProvider,
        working: WorkingMemoryProvider,
        normalizer: Optional[EventNormalizer] = None,
        buffer_size: int = 50,
        flush_interval_s: float = 60.0,
        privacy_filter: Optional["EventPrivacyFilter"] = None,
    ) -> None:
        self._immediate = immediate
        self._working = working
        self._normalizer = normalizer
        self._privacy_filter = privacy_filter
        self._subscribers: List[EventCallback] = []
        self._recent_sources: Dict[str, float] = {}
        self._reorder_buffer: Optional[EventReorderBuffer] = None
        self._consumers: list["EventConsumer"] = []
        self._event_buffer: list[SystemEvent] = []
        self._buffer_size = buffer_size
        self._flush_interval_s = flush_interval_s
        self._last_flush_time = time.monotonic()

    def set_normalizer(self, normalizer: EventNormalizer) -> None:
        """Late-bind normalizer (set after VSI handshake resolves the manifest)."""
        self._normalizer = normalizer

    def set_privacy_filter(self, privacy_filter: "EventPrivacyFilter") -> None:
        """Late-bind privacy filter for event ingestion control."""
        self._privacy_filter = privacy_filter

    def subscribe(self, callback: EventCallback) -> None:
        """Register a callback that receives every normalized SystemEvent."""
        self._subscribers.append(callback)

    def enable_reorder(self, settle_s: float = 0.05) -> None:
        """Activate the reorder buffer (call at recording start)."""
        self._reorder_buffer = EventReorderBuffer(settle_s, self._process_event)

    async def disable_reorder(self) -> None:
        """Drain and deactivate the reorder buffer (call at recording stop)."""
        if self._reorder_buffer is not None:
            await self._reorder_buffer.drain()
            self._reorder_buffer = None

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Dispatch a server-pushed event, optionally via the reorder buffer."""
        if self._reorder_buffer is not None:
            await self._reorder_buffer.submit(event_type, payload)
        else:
            await self._process_event(event_type, payload)

    def register_consumer(self, consumer: "EventConsumer") -> None:
        """Register an event consumer for batch event delivery."""
        if consumer not in self._consumers:
            self._consumers.append(consumer)
            logger.info("Registered event consumer: %s", consumer.consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        """Remove a consumer by ID."""
        self._consumers = [
            c for c in self._consumers if c.consumer_id != consumer_id
        ]

    async def flush_consumers(self) -> None:
        """Force flush buffered events to all consumers."""
        if not self._event_buffer or not self._consumers:
            return
        batch = self._event_buffer[:]
        self._event_buffer.clear()
        self._last_flush_time = time.monotonic()
        for consumer in self._consumers:
            if not consumer.enabled:
                continue
            try:
                await consumer.on_events_batch(batch)
            except Exception:
                logger.warning(
                    "Event consumer '%s' failed on batch of %d events",
                    consumer.consumer_id,
                    len(batch),
                    exc_info=True,
                )

    def _buffer_event(self, event: SystemEvent) -> None:
        """Add event to consumer buffer; schedule flush when threshold is met."""
        self._event_buffer.append(event)
        now = time.monotonic()
        if (
            len(self._event_buffer) >= self._buffer_size
            or now - self._last_flush_time >= self._flush_interval_s
        ):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.flush_consumers())
            except RuntimeError:
                # No running loop — skip async flush (will flush on next opportunity)
                pass

    async def _process_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Core processing: normalize → privacy gate → memory → subscribers → consumers."""
        normalized = self._normalize(event_type, payload)

        # Privacy gate: block excluded apps/paths from the entire pipeline
        if self._privacy_filter is not None:
            if not self._privacy_filter.should_ingest(
                normalized.event_type, normalized.source, normalized.payload,
            ):
                return

        self._ingest_to_memory(normalized)
        self._notify_subscribers(normalized)
        self._buffer_event(normalized)

    def _normalize(self, event_type: str, payload: Dict[str, Any]) -> SystemEvent:
        if self._normalizer is not None:
            return self._normalizer.normalize(event_type, payload)
        return self._fallback_normalize(event_type, payload)

    def _ingest_to_memory(self, event: SystemEvent) -> None:
        dedup_key = f"{event.event_type}:{event.source}"
        now = time.monotonic()
        last_seen = self._recent_sources.get(dedup_key, 0.0)
        if now - last_seen < _DEDUP_WINDOW_S:
            return
        self._recent_sources[dedup_key] = now
        if len(self._recent_sources) > 500:
            cutoff = now - _DEDUP_WINDOW_S
            self._recent_sources = {
                k: v for k, v in self._recent_sources.items() if v > cutoff
            }

        # Redact sensitive content before persisting
        payload = event.payload
        if self._privacy_filter is not None:
            payload = self._privacy_filter.redact_payload(event.event_type, payload)

        content = _event_to_content(event)
        self._immediate.ingest(
            event.event_type,
            content,
            path=event.source if event.event_type == "fs.change" else None,
            metadata=payload,
        )
        self._working.remember_event(event.event_type, content, payload)

    def _notify_subscribers(self, event: SystemEvent) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                logger.error("Event subscriber error", exc_info=True)

    def _fallback_normalize(self, event_type: str, payload: Dict[str, Any]) -> SystemEvent:
        """Legacy normalization when no EventNormalizer is configured."""
        import time

        if event_type == EventTypes.FS_CHANGE:
            path = str(payload.get("path", ""))
            flags = payload.get("flags", 0)
            action = _infer_fs_action(int(flags))
            return SystemEvent(
                event_type="fs.change",
                source=path,
                payload={"path": path, "action": action, "raw_flags": flags},
                timestamp=payload.get("ts", time.time()),
            )
        if event_type == EventTypes.CLIPBOARD_CHANGE:
            text = str(payload.get("text", ""))
            return SystemEvent(
                event_type="clipboard.change",
                source="system.clipboard",
                payload={"text": text, "char_count": len(text)},
                timestamp=payload.get("change_ts", time.time()),
            )
        if event_type == EventTypes.APP_FOCUS_CHANGE:
            bundle_id = str(payload.get("bundle_id", ""))
            app_name = str(payload.get("app_name", bundle_id))
            return SystemEvent(
                event_type="app.focus_change",
                source=bundle_id,
                payload={"bundle_id": bundle_id, "app_name": app_name},
                timestamp=time.time(),
            )
        if event_type == EventTypes.UI_ACTION:
            action = str(payload.get("action", "unknown"))
            app_bundle_id = str(payload.get("app_bundle_id", ""))
            return SystemEvent(
                event_type="ui.action",
                source=app_bundle_id,
                payload={"sub_type": action, "app_bundle_id": app_bundle_id, **payload},
                timestamp=payload.get("timestamp", time.time()),
            )
        return SystemEvent(
            event_type="internal.unmapped",
            source=event_type,
            payload={"_original_type": event_type, **payload},
            timestamp=time.time(),
        )


def _event_to_content(event: SystemEvent) -> str:
    """Produce a human-readable summary string from a normalized event."""
    if event.event_type == "fs.change":
        action = event.payload.get("action", "modified")
        return f"File {action}: {event.source}"
    if event.event_type == "clipboard.change":
        text = str(event.payload.get("text", ""))
        preview = text[:200] + ("..." if len(text) > 200 else "")
        count = event.payload.get("char_count", len(text))
        return f"Clipboard updated ({count} chars): {preview}"
    if event.event_type == "app.focus_change":
        app_name = event.payload.get("app_name", event.source)
        return f"Focus switched to: {app_name} ({event.source})"
    if event.event_type == "intent.signal":
        return f"Intent signal from {event.source}: {event.payload}"
    return f"[{event.event_type}] {event.source}"


def _infer_fs_action(flags: int) -> str:
    """Infer a semantic file action from FSEvent flags (legacy fallback).

    Flag values from macOS CoreServices/FSEvents.h.
    Return values must match keys in domain.trajectory._EVENT_TO_ACTION.
    """
    if flags & 0x00000100:  # kFSEventStreamEventFlagItemCreated
        return "created"
    if flags & 0x00000200:  # kFSEventStreamEventFlagItemRemoved
        return "deleted"
    if flags & 0x00000800:  # kFSEventStreamEventFlagItemRenamed
        return "renamed"
    if flags & (0x00001000 | 0x00000400 | 0x00002000 | 0x00008000):
        return "modified"
    return "modified"

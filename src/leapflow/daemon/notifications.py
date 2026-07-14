"""Daemon notification bus — push events from background tasks to connected TUI clients.

Architecture:
- `NotificationBus` manages per-subscriber async queues
- Background tasks (distillation, hub sync, etc.) call `bus.emit(event)` to broadcast
- Each connected `events.subscribe` stream gets its own queue and yields events
- Subscribers auto-unregister on disconnect (context manager)

This is the daemon's equivalent of server-sent events — generalizable to any
background task that needs to push progress or completion signals to TUI clients.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notification:
    """A single notification event pushed to TUI clients."""

    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class NotificationBus:
    """Broadcast notifications to all subscribed TUI clients.

    Thread-safe via asyncio.Queue per subscriber.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._subscribers: Dict[str, asyncio.Queue[Optional[Notification]]] = {}
        self._maxsize = maxsize

    def subscribe(self, subscriber_id: str) -> asyncio.Queue[Optional[Notification]]:
        """Register a subscriber and return its queue.

        Send None to the queue to signal graceful shutdown.
        """
        q: asyncio.Queue[Optional[Notification]] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers[subscriber_id] = q
        logger.debug("notification_bus: subscriber added id=%s total=%d", subscriber_id, len(self._subscribers))
        return q

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber."""
        self._subscribers.pop(subscriber_id, None)
        logger.debug("notification_bus: subscriber removed id=%s total=%d", subscriber_id, len(self._subscribers))

    def emit(self, notification: Notification) -> None:
        """Broadcast a notification to all subscribers (non-blocking).

        If a subscriber's queue is full, the notification is dropped for that
        subscriber (back-pressure). This prevents slow clients from blocking
        background tasks.
        """
        for sid, q in list(self._subscribers.items()):
            try:
                q.put_nowait(notification)
            except asyncio.QueueFull:
                logger.debug("notification_bus: dropped event for slow subscriber %s", sid)

    def emit_event(self, event_type: str, **payload: Any) -> None:
        """Convenience: emit with keyword args."""
        self.emit(Notification(event_type=event_type, payload=payload))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def shutdown(self) -> None:
        """Signal all subscribers to disconnect."""
        for q in self._subscribers.values():
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()

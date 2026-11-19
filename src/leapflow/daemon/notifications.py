# Copyright (c) Alibaba, Inc. and its affiliates.
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


_REPLACEABLE = frozenset({"monitor.updated", "signal.stream", "view.resync", "teach.progress"})


def _priority(notification: Optional[Notification]) -> int:
    if notification is None or notification.event_type in {"daemon.shutdown", "approval_request"}:
        return 2
    if notification.event_type == "monitor.finding" and notification.payload.get("severity") == "critical":
        return 2
    return 0 if notification.event_type in _REPLACEABLE else 1


def _replacement_key(notification: Notification) -> tuple:
    payload = notification.payload
    return (notification.event_type, payload.get("watch_id"), payload.get("session_id"),
            payload.get("request_id"))


class _NotificationQueue(asyncio.Queue):
    """Priority insertion via Queue's supported subclass hook."""

    def _put(self, item: Optional[Notification]) -> None:
        priority = _priority(item)
        for index, queued in enumerate(self._queue):
            if _priority(queued) < priority:
                self._queue.insert(index, item)
                return
        self._queue.append(item)

    def offer(self, item: Notification) -> tuple[int, int, bool]:
        """Replace/evict queued work without completing retained work for join()."""
        coalesced = overflowed = 0
        if item.event_type in _REPLACEABLE:
            key = _replacement_key(item)
            for queued in list(self._queue):
                if queued is not None and _replacement_key(queued) == key:
                    self._queue.remove(queued)
                    coalesced += 1
        if self.full():
            victim = next((queued for queued in reversed(self._queue)
                           if _priority(queued) < _priority(item)), None)
            if victim is None:
                return coalesced, 1, False
            self._queue.remove(victim)
            overflowed = 1
        self.put_nowait(item)
        # Account for discarded work only after the new work is pending.
        for _ in range(coalesced + overflowed):
            self.task_done()
        return coalesced, overflowed, True

    def terminate(self) -> int:
        removed = len(self._queue)
        self._queue.clear()
        self.put_nowait(None)
        for _ in range(removed):
            self.task_done()
        return removed


class NotificationBus:
    """Event-loop-owned bounded fan-out; a slow client cannot stall another."""

    def __init__(self, maxsize: int = 64) -> None:
        self._subscribers: Dict[str, _NotificationQueue] = {}
        if type(maxsize) is not int or maxsize <= 0:
            raise ValueError("notification queue capacity must be a positive integer")
        self._maxsize = maxsize
        self._stats = {"coalesced": 0, "overflowed": 0, "enqueued": 0}

    def subscribe(self, subscriber_id: str) -> asyncio.Queue[Optional[Notification]]:
        """Register a subscriber and return its queue.

        Send None to the queue to signal graceful shutdown.
        """
        if subscriber_id in self._subscribers:
            self._terminate(self._subscribers[subscriber_id])
        q = _NotificationQueue(maxsize=self._maxsize)
        self._subscribers[subscriber_id] = q
        logger.debug("notification_bus: subscriber added id=%s total=%d", subscriber_id, len(self._subscribers))
        return q

    def unsubscribe(self, subscriber_id: str, queue: asyncio.Queue | None = None) -> None:
        """Remove only the subscription owned by the closing stream."""
        if queue is not None and self._subscribers.get(subscriber_id) is not queue:
            return
        self._subscribers.pop(subscriber_id, None)
        logger.debug("notification_bus: subscriber removed id=%s total=%d", subscriber_id, len(self._subscribers))

    def emit(self, notification: Notification) -> None:
        """Broadcast a notification to all subscribers (non-blocking).

        If a subscriber's queue is full, the notification is dropped for that
        subscriber (back-pressure). This prevents slow clients from blocking
        background tasks.
        """
        for q in list(self._subscribers.values()):
            coalesced, overflowed, enqueued = q.offer(notification)
            self._stats["coalesced"] += coalesced
            self._stats["overflowed"] += overflowed
            self._stats["enqueued"] += int(enqueued)

    def emit_event(self, event_type: str, **payload: Any) -> None:
        """Convenience: emit with keyword args."""
        self.emit(Notification(event_type=event_type, payload=payload))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def stats(self) -> dict[str, int]:
        """Fan-out enqueue/overflow counts, not user receipt acknowledgments."""
        return dict(self._stats)

    def _terminate(self, q: _NotificationQueue) -> None:
        self._stats["overflowed"] += q.terminate()

    async def shutdown(self) -> None:
        """Guaranteed sentinel even when every subscriber queue is full."""
        for q in self._subscribers.values():
            self._terminate(q)
        self._subscribers.clear()

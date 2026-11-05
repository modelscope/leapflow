# Copyright (c) Alibaba, Inc. and its affiliates.
"""ViewHub: fan out daemon monitor events to browser WebSocket subscribers.

The dashboard server holds a single subscription to the daemon NotificationBus
and re-broadcasts qualifying messages to every connected browser. This mirrors
the daemon's own NotificationBus fan-out philosophy, keeping slow browsers from
blocking the shared upstream via bounded per-subscriber queues.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ViewHub:
    """Broadcast view/monitor messages to all subscribed browsers."""

    def __init__(self, maxsize: int = 128) -> None:
        self._subscribers: dict[str, asyncio.Queue[Optional[dict[str, Any]]]] = {}
        # A dropped increment must not leave a live lens permanently stale. The next
        # successful delivery asks that browser to fetch its authoritative snapshot.
        self._resync_required: set[str] = set()
        self._maxsize = maxsize

    def subscribe(self, subscriber_id: str) -> asyncio.Queue[Optional[dict[str, Any]]]:
        """Register a browser subscriber and return its message queue."""
        queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers[subscriber_id] = queue
        return queue

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a browser subscriber."""
        self._subscribers.pop(subscriber_id, None)
        self._resync_required.discard(subscriber_id)

    def broadcast(self, message: dict[str, Any]) -> int:
        """Deliver a message to all subscribers (non-blocking); return count.

        Full queues are skipped (back-pressure) rather than blocking the shared
        upstream subscription.
        """
        delivered = 0
        for sid, queue in list(self._subscribers.items()):
            resynced = False
            if sid in self._resync_required:
                try:
                    queue.put_nowait({"type": "view.resync", "payload": {"reason": "dropped_events"}})
                    self._resync_required.discard(sid)
                    resynced = True
                except asyncio.QueueFull:
                    continue
            try:
                queue.put_nowait(message)
                delivered += 1
            except asyncio.QueueFull:
                # A successful resync marker already occupies the one available
                # slot. Do not re-arm it because the marker itself tells the client
                # to fetch the snapshot and discard this increment.
                if not resynced:
                    self._resync_required.add(sid)
                logger.debug("view_hub: dropped message for slow subscriber %s", sid)
        return delivered

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def shutdown(self) -> None:
        """Signal all subscribers to disconnect."""
        for queue in self._subscribers.values():
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()
        self._resync_required.clear()


__all__ = ["ViewHub"]

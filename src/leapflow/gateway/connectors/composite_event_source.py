# Copyright (c) Alibaba, Inc. and its affiliates.
"""Composite event source that merges multiple BackendEventSource streams.

Subscribes to N child sources and yields events from all of them through
a single ``events()`` async iterator.  Lifecycle (start/stop/status)
delegates to all children.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Sequence

from leapflow.gateway.connectors.protocol import (
    BackendEvent,
    BackendEventSource,
    EventSourceStatus,
)

logger = logging.getLogger(__name__)

_SENTINEL = object()


_DEFAULT_QUEUE_SIZE = 2048


class CompositeEventSource:
    """Merges multiple BackendEventSource streams into one.

    Satisfies the ``BackendEventSource`` protocol.  Each child source
    gets its own consumer task that pushes events into a shared
    bounded ``asyncio.Queue``.  ``events()`` yields from the queue.

    When the queue is full, the oldest event is discarded to make room
    for the incoming one (drop-oldest back-pressure strategy).

    The *maxsize* parameter controls the internal queue capacity
    (default: ``_DEFAULT_QUEUE_SIZE = 2048``).  Callers can override it
    to tune back-pressure behavior for high-throughput or
    resource-constrained deployments.
    """

    backend_kind = "composite"

    def __init__(
        self,
        sources: Sequence[BackendEventSource],
        *,
        platform_id: str = "",
        maxsize: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self._sources = list(sources)
        self.platform_id = platform_id or (
            self._sources[0].platform_id if self._sources else "unknown"
        )
        self._queue: asyncio.Queue[BackendEvent | object] = asyncio.Queue(maxsize=maxsize)
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._drop_count: int = 0

    async def start(self, *, checkpoint: str = "") -> EventSourceStatus:
        """Start all child sources."""
        if self._running:
            return await self.status()

        results: list[EventSourceStatus] = []
        for src in self._sources:
            status = await src.start(checkpoint=checkpoint)
            results.append(status)

        all_ok = all(r.ok for r in results)
        self._running = True
        detail_parts = [f"{i}: {r.detail}" for i, r in enumerate(results)]
        return EventSourceStatus(
            ok=all_ok,
            backend_kind=self.backend_kind,
            detail="; ".join(detail_parts),
            checkpoint=checkpoint,
            metadata={"child_count": len(self._sources), "all_ok": all_ok},
        )

    async def stop(self) -> EventSourceStatus:
        """Stop all child sources and consumer tasks."""
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for src in self._sources:
            try:
                await src.stop()
            except Exception:
                logger.debug("Error stopping child source", exc_info=True)

        return EventSourceStatus(
            ok=True,
            backend_kind=self.backend_kind,
            detail="stopped",
            metadata={"child_count": len(self._sources)},
        )

    async def events(self) -> AsyncIterator[BackendEvent]:
        """Yield events from all child sources through a shared queue."""
        self._tasks = [
            asyncio.create_task(
                self._consume_child(i, src),
                name=f"composite-child-{i}-{self.platform_id}",
            )
            for i, src in enumerate(self._sources)
        ]
        try:
            while self._running or not self._queue.empty():
                item = await self._queue.get()
                if item is _SENTINEL:
                    if all(t.done() for t in self._tasks):
                        break
                    continue
                yield item  # type: ignore[misc]
        except asyncio.CancelledError:
            return
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

    @property
    def drop_count(self) -> int:
        """Number of events dropped due to queue back-pressure."""
        return self._drop_count

    async def _consume_child(self, index: int, source: BackendEventSource) -> None:
        """Read events from one child source and push them to the shared queue."""
        try:
            async for event in source.events():  # type: ignore[attr-defined]  # async-gen vs Protocol
                try:
                    self._queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop oldest event to make room (back-pressure strategy).
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._queue.put_nowait(event)
                    self._drop_count += 1
                    logger.debug(
                        "Composite queue full, dropped oldest event "
                        "(child=%d, platform=%s, total_drops=%d)",
                        index, self.platform_id, self._drop_count,
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error(
                "Composite child %d (%s) crashed",
                index, self.platform_id, exc_info=True,
            )
        finally:
            await self._queue.put(_SENTINEL)

    async def status(self) -> EventSourceStatus:
        """Aggregate status from all children."""
        statuses = []
        for src in self._sources:
            try:
                statuses.append(await src.status())
            except Exception:
                statuses.append(EventSourceStatus(
                    ok=False, backend_kind="unknown", detail="status check failed",
                ))
        all_ok = all(s.ok for s in statuses)
        return EventSourceStatus(
            ok=self._running and all_ok,
            backend_kind=self.backend_kind,
            detail=f"{sum(1 for s in statuses if s.ok)}/{len(statuses)} children ok",
            metadata={
                "running": self._running,
                "child_count": len(self._sources),
                "children_ok": sum(1 for s in statuses if s.ok),
            },
        )

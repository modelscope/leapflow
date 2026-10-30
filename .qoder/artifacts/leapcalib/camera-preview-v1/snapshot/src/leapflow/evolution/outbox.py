# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded asynchronous outbox for evolution evidence.

Ordinary read-only evidence never waits for DuckDB.  Safety-critical action-start
facts use a durable barrier before the side effect.  Queue saturation never silently
drops evidence: it falls back to the serialized writer thread and exposes counters.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from time import perf_counter

from leapflow.domain.evolution_event import EvolutionEvent, EvolutionEventStore
from leapflow.performance import LatencySummary, RollingLatency

logger = logging.getLogger(__name__)


class EvolutionOutboxClosed(RuntimeError):
    """Raised when evidence is published after shutdown began."""


class EvolutionOutboxWriteError(RuntimeError):
    """Raised when bounded persistence attempts cannot retain evidence."""


@dataclass(frozen=True)
class OutboxMetrics:
    queued: int
    published: int
    direct_fallbacks: int
    failures: int
    publish_latency: LatencySummary
    write_latency: LatencySummary


class EvolutionEventOutbox:
    """One profile-scoped event queue drained by a single background task."""

    def __init__(
        self,
        store: EvolutionEventStore,
        *,
        max_events: int = 4096,
        max_batch: int = 128,
        flush_interval_s: float = 0.05,
        critical_timeout_s: float = 1.0,
        write_timeout_s: float = 2.0,
        flush_timeout_s: float = 5.0,
        max_write_attempts: int = 3,
        retry_backoff_s: float = 0.05,
    ) -> None:
        self._store = store
        self._queue: asyncio.Queue[EvolutionEvent] = asyncio.Queue(
            maxsize=max(1, int(max_events))
        )
        self._max_batch = max(1, int(max_batch))
        self._flush_interval_s = max(0.001, float(flush_interval_s))
        self._critical_timeout_s = max(0.001, float(critical_timeout_s))
        self._write_timeout_s = max(0.001, float(write_timeout_s))
        self._flush_timeout_s = max(0.001, float(flush_timeout_s))
        self._max_write_attempts = max(1, int(max_write_attempts))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._published = 0
        self._direct_fallbacks = 0
        self._failures = 0
        self._write_error: BaseException | None = None
        self._publish_latency = RollingLatency()
        self._write_latency = RollingLatency()

    @property
    def metrics(self) -> OutboxMetrics:
        return OutboxMetrics(
            queued=self._queue.qsize(),
            published=self._published,
            direct_fallbacks=self._direct_fallbacks,
            failures=self._failures,
            publish_latency=self._publish_latency.snapshot(),
            write_latency=self._write_latency.snapshot(),
        )

    def start(self) -> None:
        """Start the writer on the current event loop; idempotent."""
        if self._closed:
            raise EvolutionOutboxClosed("evolution outbox is closed")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="evolution-event-outbox")

    async def publish(self, event: EvolutionEvent, *, critical: bool = False) -> None:
        """Publish an event without silently losing it."""
        started_at = perf_counter()
        try:
            if self._closed:
                raise EvolutionOutboxClosed("evolution outbox is closed")
            if self._write_error is not None:
                raise EvolutionOutboxWriteError(
                    "evolution outbox has an unpersisted event"
                ) from self._write_error
            if critical:
                try:
                    inserted = await self._retry_write(
                        lambda: self._store.append(event),
                        timeout_s=self._critical_timeout_s,
                    )
                except EvolutionOutboxWriteError:
                    self._failures += 1
                    raise
                self._published += int(bool(inserted))
                return
            self.start()
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self._direct_fallbacks += 1
                try:
                    inserted = await self._retry_write(lambda: self._store.append(event))
                    self._published += int(bool(inserted))
                except EvolutionOutboxWriteError:
                    self._failures += 1
                    raise
        finally:
            self._publish_latency.observe((perf_counter() - started_at) * 1000.0)

    async def flush(self) -> None:
        """Wait a bounded time for queued writes and surface any lost evidence."""
        if self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=self._flush_timeout_s)
            except asyncio.TimeoutError as exc:
                raise EvolutionOutboxWriteError("evolution outbox flush timed out") from exc
        if self._write_error is not None:
            raise EvolutionOutboxWriteError("evolution outbox failed to persist evidence") from self._write_error

    async def close(self) -> None:
        """Flush, stop the writer, and reject future publications."""
        if self._closed:
            return
        error: BaseException | None = None
        try:
            await self.flush()
        except EvolutionOutboxWriteError as exc:
            error = exc
        finally:
            self._closed = True
            task = self._task
            self._task = None
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if error is not None:
            raise error

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            try:
                deadline = asyncio.get_running_loop().time() + self._flush_interval_s
                while len(batch) < self._max_batch:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
                try:
                    inserted = await self._retry_write(
                        lambda: self._store.append_many(batch)
                    )
                    self._published += int(inserted)
                except EvolutionOutboxWriteError as exc:
                    self._failures += len(batch)
                    if self._write_error is None:
                        self._write_error = exc.__cause__ or exc
                    logger.error("evolution outbox batch write exhausted retries", exc_info=True)
                finally:
                    for _ in batch:
                        self._queue.task_done()
            except asyncio.CancelledError:
                # The owner calls flush before cancellation, so reaching this with a
                # batch means an abnormal shutdown. Persist synchronously as a final
                # best effort, then preserve cancellation semantics.
                try:
                    await asyncio.to_thread(self._store.append_many, batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()
                raise

    async def _retry_write(
        self,
        operation: Callable[[], bool | int],
        *,
        timeout_s: float | None = None,
    ) -> bool | int:
        delay = self._retry_backoff_s
        timeout = self._write_timeout_s if timeout_s is None else timeout_s
        last_error: BaseException | None = None
        started_at = perf_counter()
        try:
            for attempt in range(self._max_write_attempts):
                try:
                    pending: Awaitable[bool | int] = asyncio.to_thread(operation)
                    return await asyncio.wait_for(pending, timeout=timeout)
                except (Exception, asyncio.TimeoutError) as exc:
                    last_error = exc
                    if attempt + 1 < self._max_write_attempts and delay > 0:
                        await asyncio.sleep(delay * (2**attempt))
            assert last_error is not None
            if self._write_error is None:
                self._write_error = last_error
            raise EvolutionOutboxWriteError(
                "evolution event persistence retries exhausted"
            ) from last_error
        finally:
            self._write_latency.observe((perf_counter() - started_at) * 1000.0)


__all__ = [
    "EvolutionEventOutbox",
    "EvolutionOutboxClosed",
    "EvolutionOutboxWriteError",
    "OutboxMetrics",
]

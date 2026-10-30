# Copyright (c) Alibaba, Inc. and its affiliates.
"""FrameHub: one acquisition loop per camera, shared by every consumer.

Before the hub, each consumer of a camera called the SDK's blocking "wait for the
next frame" itself. Two consumers therefore *stole* framesets from each other (a
colour and a depth preview each saw every other frame), each paid the full
frame-interval wait in turn, and processing such as depth alignment and JPEG
encoding ran inside the transport's lock, so one slow consumer delayed all the
others.

The hub inverts that. A single pump thread pulls framesets from a
:class:`~leapflow.hardware.vision.frames.FrameProducer` and keeps the newest few;
consumers *look* at the newest one instead of taking it:

* ``latest()`` returns the newest frameset, or waits for one that satisfies
  ``min_sequence`` / ``after_monotonic`` -- "a frame captured after my arm stopped"
  is a freshness condition, not a second capture.
* ``view()`` derives something from a frameset (aligned depth, a JPEG at one size)
  at most once per frameset and key, on a bounded worker pool, outside every lock;
  concurrent consumers of the same view share the one computation.

The hub is deliberately loop-agnostic: its state is guarded by a thread condition
and consumers wait in worker threads, so it serves callers from any event loop
(the node server, a CLI command, a test's ``asyncio.run``) without binding to one.
Opening and closing the device stays with its transport; the hub only pumps.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Hashable, TypeVar

from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision.frames import FrameProducer, Frameset

logger = logging.getLogger(__name__)

T = TypeVar("T")

FAILURE_FRAME_TIMEOUT = "frame_timeout"
FAILURE_SOURCE_CLOSED = "frame_source_closed"

DEFAULT_DEPTH = 2
"""Framesets retained: the newest plus one a slow consumer may still be deriving from."""
DEFAULT_WORKERS = 2
"""Concurrent view computations; bounds what previews can take from a small board's CPU."""


class FrameHub:
    """Pump one producer and share its newest framesets with any number of readers."""

    def __init__(
        self,
        producer: FrameProducer,
        *,
        name: str,
        depth: int = DEFAULT_DEPTH,
        workers: int = DEFAULT_WORKERS,
        poll_timeout_s: float = 1.0,
        error_backoff_s: float = 0.2,
        sequence_start: int = 0,
    ) -> None:
        """``sequence_start`` lets a reopened device continue its numbering, so a
        consumer deduplicating by sequence never mistakes a new frame for an old one."""
        self._producer = producer
        self._name = str(name or "camera")
        self._ring: collections.deque[Frameset] = collections.deque(maxlen=max(1, int(depth)))
        self._workers = max(1, int(workers))
        self._poll_timeout_s = max(0.01, float(poll_timeout_s))
        self._error_backoff_s = max(0.0, float(error_backoff_s))
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._views: dict[tuple[int, Hashable], Future[Any]] = {}
        self._closed = False
        self._sequence = max(0, int(sequence_start))
        self._failures = 0
        self._last_error = ""
        self._waiters = 0
        self._interval_s = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._closed

    def start(self) -> FrameHub:
        """Start pumping; idempotent. A stopped hub cannot be restarted."""
        with self._cond:
            if self._closed:
                raise TransportError(f"{self._name}: frame hub is closed", failure_code=FAILURE_SOURCE_CLOSED)
            if self._thread is not None:
                return self
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix=f"framehub-{self._name}-view"
            )
            self._thread = threading.Thread(
                target=self._pump, name=f"framehub-{self._name}", daemon=True
            )
            self._thread.start()
        return self

    def stop(self, timeout_s: float = 10.0) -> None:
        """Stop pumping, fail every waiter, and wait for in-flight views (blocking).

        Returns only once no thread of this hub touches the producer any more, so
        the caller may then release the device. Safe to call more than once.
        """
        with self._cond:
            self._closed = True
            self._stop.set()
            thread, self._thread = self._thread, None
            executor, self._executor = self._executor, None
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_s)
            if thread.is_alive():
                logger.warning("%s: frame pump did not stop within %.1fs", self._name, timeout_s)
        if executor is not None:
            executor.shutdown(wait=True)
        with self._cond:
            self._ring.clear()
            self._views.clear()

    # ── Consumers ────────────────────────────────────────────────────────

    def peek(self, *, min_sequence: int = 0, after_monotonic: float | None = None) -> Frameset | None:
        """The newest frameset if it satisfies the freshness conditions, else None."""
        with self._cond:
            return self._match(min_sequence, after_monotonic)

    def wait(
        self, *, min_sequence: int = 0, after_monotonic: float | None = None, timeout_s: float
    ) -> Frameset:
        """Block until a frameset satisfies the conditions; typed error on timeout or close."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cond:
            self._waiters += 1
            try:
                while True:
                    if self._closed:
                        raise TransportError(
                            f"{self._name}: frame source closed", failure_code=FAILURE_SOURCE_CLOSED
                        )
                    found = self._match(min_sequence, after_monotonic)
                    if found is not None:
                        return found
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        detail = f": {self._last_error}" if self._last_error else ""
                        raise TransportError(
                            f"{self._name}: no frame within {float(timeout_s):g}s{detail}",
                            failure_code=FAILURE_FRAME_TIMEOUT,
                        )
                    self._cond.wait(remaining)
            finally:
                self._waiters -= 1

    async def latest(
        self, *, min_sequence: int = 0, after_monotonic: float | None = None, timeout_s: float
    ) -> Frameset:
        """The newest satisfying frameset, waiting off the event loop when none is yet."""
        found = self.peek(min_sequence=min_sequence, after_monotonic=after_monotonic)
        if found is not None:
            return found
        return await asyncio.to_thread(
            self.wait, min_sequence=min_sequence, after_monotonic=after_monotonic, timeout_s=timeout_s
        )

    async def view(self, frameset: Frameset, key: Hashable, compute: Callable[[Frameset], T]) -> T:
        """Derive ``compute(frameset)`` once per (frameset, key), shared by concurrent callers.

        The computation runs on the hub's worker pool, holding no hub lock. A
        cancelled caller does not cancel it -- others may be waiting for the same
        result, and ``stop`` waits for it before the device is released. A failed
        computation is not cached, so the next caller retries.
        """
        slot = (frameset.sequence, key)
        with self._cond:
            if self._closed or self._executor is None:
                raise TransportError(
                    f"{self._name}: frame source closed", failure_code=FAILURE_SOURCE_CLOSED
                )
            future = self._views.get(slot)
            if future is None:
                future = self._executor.submit(compute, frameset)
                if any(held.sequence == frameset.sequence for held in self._ring):
                    self._views[slot] = future
                    future.add_done_callback(lambda done: self._forget_failed(slot, done))
        return await asyncio.shield(asyncio.wrap_future(future))

    def stats(self) -> dict[str, Any]:
        """JSON-safe pump health: rate, age of the newest frame, failures, demand."""
        with self._cond:
            newest = self._ring[-1] if self._ring else None
            return {
                "running": self._thread is not None and not self._closed,
                "sequence": self._sequence,
                "fps": round(1.0 / self._interval_s, 2) if self._interval_s > 0 else 0.0,
                "frame_age_ms": (
                    round((time.monotonic() - newest.monotonic_at) * 1000.0, 1) if newest else None
                ),
                "failures": self._failures,
                "last_error": self._last_error,
                "waiters": self._waiters,
                "views": len(self._views),
            }

    # ── Internals ────────────────────────────────────────────────────────

    def _match(self, min_sequence: int, after_monotonic: float | None) -> Frameset | None:
        newest = self._ring[-1] if self._ring else None
        if newest is None or newest.sequence < int(min_sequence):
            return None
        if after_monotonic is not None and newest.monotonic_at <= after_monotonic:
            return None
        return newest

    def _forget_failed(self, slot: tuple[int, Hashable], future: Future[Any]) -> None:
        if future.cancelled() or future.exception() is not None:
            with self._cond:
                if self._views.get(slot) is future:
                    del self._views[slot]

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                native = self._producer.grab(self._poll_timeout_s)
            except Exception as exc:  # noqa: BLE001 - a driver failure is reported to waiters, never raised here
                with self._cond:
                    self._failures += 1
                    self._last_error = str(exc) or type(exc).__name__
                    self._cond.notify_all()
                self._stop.wait(self._error_backoff_s)
                continue
            if native is not None:
                self._publish(native)

    def _publish(self, native: Any) -> None:
        now = time.monotonic()
        with self._cond:
            if self._closed:
                return
            previous = self._ring[-1] if self._ring else None
            if previous is not None:
                interval = now - previous.monotonic_at
                self._interval_s = interval if self._interval_s <= 0 else 0.9 * self._interval_s + 0.1 * interval
            self._sequence += 1
            self._ring.append(Frameset(self._sequence, time.time(), now, native))
            self._last_error = ""
            retained = {held.sequence for held in self._ring}
            for slot in [slot for slot in self._views if slot[0] not in retained]:
                del self._views[slot]
            self._cond.notify_all()


__all__ = [
    "DEFAULT_DEPTH",
    "DEFAULT_WORKERS",
    "FAILURE_FRAME_TIMEOUT",
    "FAILURE_SOURCE_CLOSED",
    "FrameHub",
]

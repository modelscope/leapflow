# Copyright (c) Alibaba, Inc. and its affiliates.
"""Lifecycle-bearing signal source category.

Unlike SignalSource (stateless transform), ActiveSignalSource subscribes to
external event streams (file watchers, IM listeners, IoT devices). It has a
lifecycle managed by EffectScope/PluginFiber, and emits signals through a
bounded asyncio.Queue to serialize downstream mutation.

Design note:
    Emission flows: source.start(emit) -> emit(signal) -> asyncio.Queue ->
    consumer task -> SignalBuffer.record() + CausalFusionPipeline.fuse().
    The queue is the only correct way to serialize CausalGraph mutation
    across sources that may run in executor threads.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from leapflow.perception.signals import SignalBuffer
from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)

EmitCallback = Callable[[InteractionSignal], None]
"""Signature of the emit callback passed to sources. Must be thread-safe.

The implementation uses loop.call_soon_threadsafe() to marshal enqueue back onto
the event loop, so this callback is safe to invoke from ANY thread (asyncio task,
executor thread, or watchdog observer thread).
"""


@runtime_checkable
class ActiveSignalSource(Protocol):
    """Protocol for lifecycle-bearing signal sources.

    Design contract:
    - start(emit) is called once when the manager starts. It must return
      promptly; long-running work should be spawned as internal tasks/threads.
    - Blocking I/O MUST be wrapped in loop.run_in_executor() -- sources must
      not block the event loop.
    - emit(signal) is thread-safe and non-blocking (may drop signals on overflow).
    - stop() is called once during teardown. Must be idempotent and complete
      within active_source_shutdown_timeout_s (default 5s).
    """

    @property
    def source_id(self) -> str:
        """Unique identifier for this source instance."""
        ...

    @property
    def channel_id(self) -> str:
        """Channel identifier for gating by config.signal_channels."""
        ...

    async def start(self, emit: EmitCallback) -> None:
        """Begin producing signals. Called once by ActiveSourceManager."""
        ...

    async def stop(self) -> None:
        """Stop producing signals and release resources. Must be idempotent."""
        ...


class ActiveSourceManager:
    """Session-owned orchestrator for ActiveSignalSource instances.

    Owns:
    - A bounded asyncio.Queue serializing signals from all sources.
    - One asyncio.Task per source running its start() lifecycle.
    - One consumer task draining the queue and calling downstream sinks.

    Lifecycle:
        manager = ActiveSourceManager(signal_buffer, causal_pipeline, causal_graph, ...)
        manager.register(source_a)
        manager.register(source_b)
        await manager.start_all()   # spawns source tasks + consumer
        # ... signals flow ...
        await manager.dispose()     # cancels sources, awaits stop(), drains queue

    Future extension: this manager can be integrated with EffectScope by taking
    a parent_scope parameter and registering dispose() as an effect. Because
    dispose() is a coroutine, it must be registered via ``scope.async_effect()``
    (not ``scope.effect()``) so ``EffectScope.async_dispose()`` awaits it. Not
    needed for MVP; ActiveSourceManager lifecycle is currently owned by
    PerceptionSession, which awaits ``dispose()`` directly in ``stop()``.
    """

    def __init__(
        self,
        signal_buffer: SignalBuffer,
        causal_pipeline: Any,   # CausalFusionPipeline
        causal_graph: Any,      # CausalGraph
        *,
        queue_capacity: int = 256,
        shutdown_timeout_s: float = 5.0,
    ) -> None:
        self._signal_buffer = signal_buffer
        self._causal_pipeline = causal_pipeline
        self._causal_graph = causal_graph
        self._queue_capacity = queue_capacity
        self._shutdown_timeout_s = shutdown_timeout_s

        self._sources: dict[str, ActiveSignalSource] = {}
        self._source_tasks: dict[str, asyncio.Task[None]] = {}
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._queue: Optional[asyncio.Queue[InteractionSignal]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dropped_count: int = 0
        self._started = False

    def register(self, source: ActiveSignalSource) -> None:
        """Register a source. Must be called before start_all()."""
        if self._started:
            raise RuntimeError("Cannot register source after start_all()")
        if not isinstance(source, ActiveSignalSource):
            raise TypeError(f"Not an ActiveSignalSource: {type(source)}")
        if source.source_id in self._sources:
            raise ValueError(f"Duplicate source_id: {source.source_id!r}")
        self._sources[source.source_id] = source

    async def start_all(self, enabled_channels: Optional[frozenset[str]] = None) -> None:
        """Start all registered sources and the consumer task."""
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()

        # Create queue
        self._queue = asyncio.Queue(maxsize=self._queue_capacity)

        # Start consumer task first
        self._consumer_task = asyncio.create_task(
            self._consume_loop(), name="active-source-consumer"
        )

        # Start each source with isolation
        for source_id, source in self._sources.items():
            if enabled_channels is not None and source.channel_id not in enabled_channels:
                logger.debug(
                    "Skipping active source %r (channel %r not enabled)",
                    source_id, source.channel_id,
                )
                continue
            emit = self._make_emit(source_id)
            task = asyncio.create_task(
                self._run_source(source, emit), name=f"active-source:{source_id}"
            )
            self._source_tasks[source_id] = task

        logger.info(
            "ActiveSourceManager started: %d sources, queue capacity %d",
            len(self._source_tasks), self._queue_capacity,
        )

    def _make_emit(self, source_id: str) -> EmitCallback:
        """Build a thread-safe emit callback for a source.

        Uses loop.call_soon_threadsafe to marshal enqueue back onto the event loop,
        so this callback is safe to invoke from ANY thread (asyncio task or executor
        thread or watchdog observer thread).
        """
        def emit(signal: InteractionSignal) -> None:
            loop = self._loop
            queue = self._queue
            if loop is None or queue is None:
                return

            def _enqueue() -> None:
                try:
                    queue.put_nowait(signal)
                except asyncio.QueueFull:
                    # Ok to race on this counter — it's advisory
                    self._dropped_count += 1
                    logger.debug(
                        "Active source queue full, dropping signal from %r", source_id
                    )

            try:
                loop.call_soon_threadsafe(_enqueue)
            except RuntimeError:
                # Event loop is closed; source outlived it — drop signal
                self._dropped_count += 1

        return emit

    async def _run_source(self, source: ActiveSignalSource, emit: EmitCallback) -> None:
        """Wrapper isolating each source's start() from siblings."""
        try:
            await source.start(emit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "ActiveSignalSource %r failed: %s", source.source_id, exc, exc_info=True
            )

    async def _consume_loop(self) -> None:
        """Single consumer draining the queue into downstream sinks."""
        assert self._queue is not None
        q = self._queue
        while True:
            try:
                signal = await q.get()
            except asyncio.CancelledError:
                raise

            try:
                self._signal_buffer.record(signal)
                try:
                    self._causal_pipeline.fuse(
                        signals=[signal], graph=self._causal_graph
                    )
                except (RuntimeError, ValueError, AttributeError) as exc:
                    logger.warning(
                        "CausalPipeline.fuse failed for active signal: %s",
                        exc, exc_info=True,
                    )
            except Exception as exc:
                logger.error(
                    "ActiveSourceManager consumer error (continuing): %s",
                    exc, exc_info=True,
                )
                await asyncio.sleep(0.1)
            finally:
                try:
                    q.task_done()
                except ValueError:
                    logger.debug("task_done called with no pending tasks", exc_info=True)

    async def dispose(self) -> None:
        """Cancel all sources, drain the queue, then cancel consumer.

        Sequence:
        1. Call source.stop() with per-source timeout
        2. Cancel source tasks (in case start() still running)
        3. Drain the queue by waiting for join() with shutdown_timeout_s
        4. Cancel consumer task
        """
        if not self._started:
            return

        # 1. Stop sources
        stop_coros = [self._stop_one(sid, src) for sid, src in self._sources.items()]
        if stop_coros:
            await asyncio.gather(*stop_coros, return_exceptions=True)

        # 2. Cancel source tasks
        for task in self._source_tasks.values():
            task.cancel()
        for task in self._source_tasks.values():
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        # 3. Drain the queue (consumer still processing)
        if self._queue is not None:
            try:
                await asyncio.wait_for(
                    self._queue.join(), timeout=self._shutdown_timeout_s
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "ActiveSourceManager queue did not drain within %.1fs; "
                    "some signals may be dropped",
                    self._shutdown_timeout_s,
                )

        # 4. Cancel consumer task
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await asyncio.wait_for(self._consumer_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        self._started = False
        logger.info(
            "ActiveSourceManager disposed (dropped %d signals during lifetime)",
            self._dropped_count,
        )

    async def _stop_one(self, source_id: str, source: ActiveSignalSource) -> None:
        """Stop a single source with timeout and exception isolation."""
        try:
            await asyncio.wait_for(source.stop(), timeout=self._shutdown_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "ActiveSignalSource %r stop() timed out after %.1fs",
                source_id, self._shutdown_timeout_s,
            )
        except Exception as exc:
            logger.warning(
                "ActiveSignalSource %r stop() raised: %s",
                source_id, exc, exc_info=True,
            )

    @property
    def dropped_count(self) -> int:
        """Number of signals dropped due to queue overflow (observability)."""
        return self._dropped_count

    @property
    def source_count(self) -> int:
        """Number of registered sources."""
        return len(self._sources)

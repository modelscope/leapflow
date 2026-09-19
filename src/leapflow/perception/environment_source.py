# Copyright (c) Alibaba, Inc. and its affiliates.
"""Daemon-owned lifecycle for structured task-environment sources."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Protocol, runtime_checkable

from leapflow.domain.environment_signal import EnvironmentObservation

logger = logging.getLogger(__name__)

EnvironmentEmit = Callable[[EnvironmentObservation], Awaitable[None]]
EnvironmentSink = Callable[[EnvironmentObservation], Awaitable[None]]


@runtime_checkable
class EnvironmentSource(Protocol):
    """Long-lived producer of typed task-environment observations."""

    @property
    def source_id(self) -> str: ...

    async def start(self, emit: EnvironmentEmit) -> None: ...

    async def stop(self) -> None: ...


class EnvironmentSourceManager:
    """Own environment source tasks and serialize their downstream writes."""

    def __init__(
        self,
        sink: EnvironmentSink,
        *,
        queue_capacity: int = 256,
        shutdown_timeout_s: float = 5.0,
    ) -> None:
        self._sink = sink
        self._queue: asyncio.Queue[EnvironmentObservation] = asyncio.Queue(
            maxsize=max(1, int(queue_capacity))
        )
        self._shutdown_timeout_s = max(0.1, float(shutdown_timeout_s))
        self._sources: dict[str, EnvironmentSource] = {}
        self._source_tasks: dict[str, asyncio.Task[None]] = {}
        self._consumer_task: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._dropped = 0

    def register(self, source: EnvironmentSource) -> None:
        if self._started:
            raise RuntimeError("cannot register an environment source after start")
        if not isinstance(source, EnvironmentSource):
            raise TypeError(f"not an EnvironmentSource: {type(source).__name__}")
        if source.source_id in self._sources:
            raise ValueError(f"duplicate environment source: {source.source_id}")
        self._sources[source.source_id] = source

    async def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("environment source manager is closed")
        self._started = True
        self._consumer_task = asyncio.create_task(
            self._consume(),
            name="environment-source-consumer",
        )
        for source_id, source in self._sources.items():
            self._source_tasks[source_id] = asyncio.create_task(
                self._run_source(source),
                name=f"environment-source:{source_id}",
            )

    async def publish(self, observation: EnvironmentObservation) -> None:
        """Apply bounded backpressure instead of silently dropping causal evidence."""
        if self._closed:
            return
        try:
            await asyncio.wait_for(
                self._queue.put(observation),
                timeout=self._shutdown_timeout_s,
            )
        except asyncio.TimeoutError:
            self._dropped += 1
            logger.error(
                "environment source queue saturated; dropped observation=%s",
                observation.observation_id,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for source in self._sources.values():
            try:
                await asyncio.wait_for(source.stop(), timeout=self._shutdown_timeout_s)
            except (asyncio.TimeoutError, Exception):
                logger.warning(
                    "environment source stop failed source=%s",
                    source.source_id,
                    exc_info=True,
                )
        for task in self._source_tasks.values():
            task.cancel()
        for task in self._source_tasks.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._source_tasks.clear()
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._shutdown_timeout_s)
        except asyncio.TimeoutError:
            logger.error("environment observation queue did not drain before shutdown")
        consumer = self._consumer_task
        self._consumer_task = None
        if consumer is not None and not consumer.done():
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))

    @property
    def dropped_count(self) -> int:
        return self._dropped

    async def _run_source(self, source: EnvironmentSource) -> None:
        try:
            await source.start(self.publish)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("environment source failed source=%s", source.source_id)

    async def _consume(self) -> None:
        while True:
            observation = await self._queue.get()
            try:
                await self._sink(observation)
            except Exception:
                logger.exception(
                    "environment observation sink failed observation=%s",
                    observation.observation_id,
                )
            finally:
                self._queue.task_done()


__all__ = ["EnvironmentEmit", "EnvironmentSource", "EnvironmentSourceManager"]

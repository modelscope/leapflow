# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded, supervised spawn workers for node-prepared pure planning jobs.

No forked runtime, executor private APIs, hardware handles, or manager process.
A worker missing a cooperative cancellation deadline is disposable compute only.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from leapflow.hardware.skills.planning import (
    Deadline, Evaluation, PlanningJob, PlanningSettings,
)
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_BUSY, FAILURE_INTERNAL_DEFECT, FAILURE_INVALID_ARGS, FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED, FAILURE_TIMEOUT, SkillError, SkillOutcome,
)

logger = logging.getLogger(__name__)


def _serve(pipe: Any, cancel: Any, max_bytes: int) -> None:
    """Child entry point. Only owner-produced objects arrive on this private pipe."""
    try:
        while True:
            packet = pickle.loads(pipe.recv_bytes(max_bytes))
            if packet is None:
                return
            job, deadline = packet
            started = time.monotonic()
            try:
                result = job.evaluate(Deadline(deadline, cancel))
                if not isinstance(result, Evaluation):
                    raise TypeError("planning job must return Evaluation")
            except SkillError as error:
                result = Evaluation(SkillOutcome.from_error(error))
            except Exception as exc:
                logger.error("pure planning job raised", exc_info=True)
                result = Evaluation(SkillOutcome.failed(
                    FAILURE_INTERNAL_DEFECT, f"{type(exc).__name__}: {exc}"))
            result = replace(result, evidence={**result.evidence,
                "compute_s": time.monotonic() - started, "worker_pid": mp.current_process().pid})
            payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) > max_bytes:
                payload = pickle.dumps(Evaluation(SkillOutcome.failed(
                    FAILURE_INVALID_ARGS, "planning result exceeds private payload budget"),
                                        evidence={"payload_budget_exceeded": True}))
            pipe.send_bytes(payload)
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        pipe.close()


@dataclass
class _Worker:
    process: Any
    pipe: Any
    cancel: Any


class PlanningWorkers:
    """One owner's lazy worker set. Queue admission includes running and waiting jobs."""

    def __init__(self, settings: PlanningSettings) -> None:
        self.settings = settings
        self._context = mp.get_context("spawn")
        self._slots: asyncio.Queue[int] = asyncio.Queue()
        for index in range(settings.worker_count):
            self._slots.put_nowait(index)
        self._workers: dict[int, _Worker] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._admitted = 0
        self._closing = False

    async def evaluate(self, jobs: Sequence[PlanningJob], *, deadline: float) -> tuple[Evaluation, ...]:
        if self._closing:
            raise SkillError(FAILURE_SKILL_UNAVAILABLE, "planning workers are shutting down")
        if self._admitted + len(jobs) > self.settings.max_jobs:
            raise SkillError(FAILURE_DEVICE_BUSY, "planning queue is full; nothing executed")
        payloads = []
        for job in jobs:
            Deadline(deadline).check()
            if not isinstance(job, PlanningJob):
                raise SkillError(FAILURE_INTERNAL_DEFECT, "invalid pure planning job")
            payload = pickle.dumps((job, deadline), protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) > self.settings.max_bytes:
                raise SkillError(FAILURE_INVALID_ARGS, "planning job exceeds private payload budget")
            payloads.append(payload)
        self._admitted += len(jobs)
        tasks = [asyncio.create_task(self._one(payload, deadline)) for payload in payloads]
        self._tasks.update(tasks)
        try:
            return tuple(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)
            self._admitted -= len(jobs)

    def _spawn(self) -> _Worker:
        parent, child = self._context.Pipe()
        cancel = self._context.Event()
        process = self._context.Process(target=_serve, args=(child, cancel, self.settings.max_bytes), daemon=True,
                                        name="leap-planning")
        try:
            process.start()
        except (OSError, RuntimeError):
            parent.close()
            child.close()
            raise
        child.close()
        return _Worker(process, parent, cancel)

    async def _one(self, payload: bytes, deadline: float) -> Evaluation:
        index: int | None = None
        worker: _Worker | None = None
        receiving: asyncio.Task[Any] | None = None
        try:
            index = await asyncio.wait_for(self._slots.get(), timeout=max(0, deadline - time.monotonic()))
            Deadline(deadline).check()
            if self._closing:
                raise SkillError(FAILURE_SKILL_UNAVAILABLE, "planning workers are shutting down")
            worker = self._workers.get(index)
            if worker is None:
                # Spawn is short; keep assignment atomic with cancellation on the owner loop.
                worker = self._spawn()
                self._workers[index] = worker
            worker.cancel.clear()
            def exchange() -> Any:
                worker.pipe.send_bytes(payload)
                return pickle.loads(worker.pipe.recv_bytes(self.settings.max_bytes))

            receiving = asyncio.create_task(asyncio.to_thread(exchange))
            result = await asyncio.wait_for(asyncio.shield(receiving), timeout=max(0, deadline - time.monotonic()))
            Deadline(deadline).check()
            if not isinstance(result, Evaluation):
                raise SkillError(FAILURE_INTERNAL_DEFECT, "invalid planning worker response")
            if result.evidence.get("payload_budget_exceeded"):
                raise SkillError(FAILURE_INVALID_ARGS, "planning result exceeds private payload budget")
            if isinstance(result.value, SkillOutcome) and result.value.failure_code in {
                FAILURE_INTERNAL_DEFECT, FAILURE_TIMEOUT, FAILURE_STOPPED,
            }:
                raise SkillError(result.value.failure_code, result.value.message)
            return result
        except asyncio.TimeoutError as exc:
            raise SkillError(FAILURE_TIMEOUT, "planning deadline exceeded; nothing executed") from exc
        except (OSError, EOFError, RuntimeError) as exc:
            logger.error("planning worker failed", exc_info=True)
            raise SkillError(FAILURE_INTERNAL_DEFECT, f"planning worker failed: {type(exc).__name__}") from exc
        finally:
            try:
                if worker is not None and index is not None:
                    cleanup = asyncio.create_task(self._settle(index, worker, receiving))
                    interrupted = False
                    # Owner disposal and request cancellation can arrive together.
                    # Repeated cancellation must not abandon the pipe reader or join.
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            interrupted = True
                    await cleanup
                    if interrupted:
                        raise asyncio.CancelledError
            finally:
                if index is not None:
                    self._slots.put_nowait(index)

    async def _settle(self, index: int, worker: _Worker, receiving: asyncio.Task[Any] | None) -> None:
        worker.cancel.set()
        reusable = receiving is not None
        if receiving is not None:
            try:
                result = await asyncio.wait_for(asyncio.shield(receiving), self.settings.cancel_grace_s)
                reusable = isinstance(result, Evaluation) and not (
                    isinstance(result.value, SkillOutcome) and result.value.failure_code == FAILURE_INTERNAL_DEFECT)
            except (asyncio.TimeoutError, OSError, EOFError, RuntimeError):
                reusable = False
        if not reusable or self._closing or not worker.process.is_alive():
            self._workers.pop(index, None)
            await self._retire(worker)
        if receiving is not None:
            await asyncio.gather(receiving, return_exceptions=True)

    async def _retire(self, worker: _Worker) -> None:
        worker.cancel.set()
        if worker.process.is_alive():
            worker.process.terminate()
        await asyncio.to_thread(worker.process.join, self.settings.cancel_grace_s)
        if worker.process.is_alive():
            worker.process.kill()
            await asyncio.to_thread(worker.process.join, self.settings.cancel_grace_s)
        worker.pipe.close()
        if worker.process.is_alive():
            raise SkillError(FAILURE_INTERNAL_DEFECT, "planning worker did not exit")
        worker.process.close()

    async def close(self) -> None:
        self._closing = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        workers = tuple(self._workers.values())
        self._workers.clear()
        await asyncio.gather(*(self._retire(worker) for worker in workers))

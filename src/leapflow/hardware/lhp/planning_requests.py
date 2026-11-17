# Copyright (c) Alibaba, Inc. and its affiliates.
"""Connection-scoped cancellation of pure planning, never effectful execution."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.hardware.lhp.protocol import LhpProtocolError


@runtime_checkable
class PlanningOwner(Protocol):
    """Optional direct-binding preview ownership; absent on older non-skill servers."""
    def planning_requests(self) -> PlanningRequests: ...


@runtime_checkable
class PlanCache(Protocol):
    def cached_plan(self, digest: str) -> Any: ...
    def discard_plan(self, digest: str) -> None: ...


class PlanningRequests:
    """Bounded request receipts let late cancellation forget only its own preview."""

    def __init__(self, runtime: PlanCache | None, limit: int) -> None:
        self._runtime = runtime
        self._limit = limit
        self._tasks: dict[int | str, asyncio.Task[Any]] = {}
        self._plans: dict[int | str, tuple[str, bool]] = {}
        self._closed = False

    def register(self, request_id: Any, task: asyncio.Task[Any]) -> None:
        self._prune()
        if type(request_id) not in (int, str) or isinstance(request_id, str) and len(request_id) > 128:
            raise LhpProtocolError("planning request needs a bounded integer or string id")
        if self._closed or request_id in self._tasks or request_id in self._plans:
            raise LhpProtocolError("planning request id is already active or connection is closed")
        if len(self._tasks) >= self._limit:
            raise LhpProtocolError("connection planning request limit exceeded")
        self._tasks[request_id] = task
        task.add_done_callback(lambda done: self._tasks.pop(request_id, None))

    def publish(self, request_id: int | str, response: Mapping[str, Any], *, batch: bool = True) -> None:
        """Batch previews belong to the connection; delivered legacy previews retain their TTL."""
        result = response.get("result")
        plan = result.get("plan") if isinstance(result, Mapping) else None
        digest = plan.get("digest") if isinstance(plan, Mapping) else None
        if not isinstance(digest, str) or not digest:
            return
        self._prune()
        while len(self._plans) >= self._limit:
            self._forget(next(iter(self._plans)))
        self._plans[request_id] = (digest, batch)
        if self._closed:
            self._forget(request_id)

    def cancel(self, request_id: Any) -> bool:
        if type(request_id) not in (int, str):
            raise LhpProtocolError("planning cancellation needs a request id")
        task = self._tasks.get(request_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        forgotten = request_id in self._plans
        self._forget(request_id)
        return task is not None or forgotten

    def _forget(self, request_id: int | str) -> None:
        receipt = self._plans.pop(request_id, None)
        if receipt is not None and self._runtime is not None:
            self._runtime.discard_plan(receipt[0])

    def _prune(self) -> None:
        if self._runtime is not None:
            for request_id, (digest, _) in tuple(self._plans.items()):
                if self._runtime.cached_plan(digest) is None:
                    self._plans.pop(request_id, None)

    def disconnect(self) -> tuple[asyncio.Task[Any], ...]:
        """Signal before other disconnect cleanup can suspend."""
        self._closed = True
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        for request_id, (_, batch) in tuple(self._plans.items()):
            task = self._tasks.get(request_id)
            if batch or task is not None and not task.done():
                self._forget(request_id)
            else:
                self._plans.pop(request_id)
        return tasks

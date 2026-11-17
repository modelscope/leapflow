# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real spawn workers: bounded admission, deadlines, cancellation, and retirement."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from dataclasses import dataclass

import pytest

from leapflow.hardware.skills.planning import Deadline, Evaluation, PlanningSettings
from leapflow.hardware.skills.planning_worker import PlanningWorkers
from leapflow.hardware.skills.protocol import SkillError, SkillOutcome


@dataclass(frozen=True)
class ComputeJob:
    duration: float = 0.0
    cooperative: bool = True
    defect: bool = False
    result_bytes: int = 0

    def evaluate(self, checkpoint: Deadline) -> Evaluation:
        until = time.monotonic() + self.duration
        while time.monotonic() < until:
            if self.cooperative:
                checkpoint.check()
            time.sleep(0.005)
        if self.defect:
            raise ValueError("synthetic pure-worker defect")
        checkpoint.check()
        return Evaluation(SkillOutcome.succeeded({"payload": "x" * self.result_bytes}),
                          evidence={"finished": time.monotonic()})


async def _until(predicate, timeout: float = 3.0) -> None:
    expires = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < expires
        await asyncio.sleep(0.005)


@pytest.mark.parametrize("values", [
    {"workers": True}, {"workers": 33}, {"max_candidates": 0}, {"max_jobs": 129},
    {"max_bytes": 1000}, {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
    {"timeout_s": True}, {"cancel_grace_s": 0}, {"cancel_grace_s": 6},
])
def test_resource_settings_are_finite_positive_and_bounded(values) -> None:
    with pytest.raises(ValueError):
        PlanningSettings(**values)


@pytest.mark.asyncio
async def test_workers_spawn_reuse_and_leave_no_children() -> None:
    workers = PlanningWorkers(PlanningSettings(workers=2))
    processes = []
    try:
        first = await workers.evaluate([ComputeJob(0.05), ComputeJob(0.05)], deadline=time.monotonic() + 10)
        pids = {r.evidence["worker_pid"] for r in first}
        assert len(pids) == workers.settings.worker_count
        processes = [w.process for w in workers._workers.values()]
        second = await workers.evaluate([ComputeJob(), ComputeJob()], deadline=time.monotonic() + 5)
        assert {r.evidence["worker_pid"] for r in second} == pids
        assert all(r.evidence["compute_s"] >= 0 for r in second)
    finally:
        await workers.close()
    assert not workers._workers and not workers._tasks and workers._admitted == 0
    assert not ({p.pid for p in mp.active_children()} & pids)
    assert all(p._closed for p in processes)


@pytest.mark.asyncio
async def test_full_queue_refuses_and_queued_time_counts() -> None:
    workers = PlanningWorkers(PlanningSettings(workers=1, max_jobs=2))
    active = asyncio.create_task(workers.evaluate([ComputeJob(0.8)], deadline=time.monotonic() + 10))
    try:
        await _until(lambda: workers._admitted == 1)
        queued = asyncio.create_task(workers.evaluate([ComputeJob()], deadline=time.monotonic() + 0.05))
        await _until(lambda: workers._admitted == 2)
        with pytest.raises(SkillError, match="queue is full"):
            await workers.evaluate([ComputeJob()], deadline=time.monotonic() + 5)
        with pytest.raises(SkillError) as refused:
            await queued
        assert refused.value.failure_code == "timeout"
        assert len(await active) == 1
    finally:
        await workers.close()
        await asyncio.gather(active, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cooperative", [True, False])
async def test_cancellation_drains_and_restarts_pure_worker(cooperative: bool) -> None:
    workers = PlanningWorkers(PlanningSettings(workers=1, cancel_grace_s=0.05))
    await workers.evaluate([ComputeJob()], deadline=time.monotonic() + 10)
    before = next(iter(workers._workers.values())).process.pid
    active = asyncio.create_task(workers.evaluate([ComputeJob(30, cooperative)], deadline=time.monotonic() + 60))
    try:
        await _until(lambda: workers._admitted == 1)
        await asyncio.sleep(0.04)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert workers._admitted == 0 and not workers._tasks
        result = await workers.evaluate([ComputeJob()], deadline=time.monotonic() + 10)
        if not cooperative:
            assert result[0].evidence["worker_pid"] != before
    finally:
        await workers.close()


@pytest.mark.asyncio
async def test_worker_crash_is_request_fatal_and_replacement_is_lazy() -> None:
    workers = PlanningWorkers(PlanningSettings(workers=1))
    active = asyncio.create_task(workers.evaluate([ComputeJob(30)], deadline=time.monotonic() + 60))
    try:
        await _until(lambda: bool(workers._workers))
        crashed = next(iter(workers._workers.values())).process
        old_pid = crashed.pid
        crashed.terminate()
        with pytest.raises(SkillError) as failed:
            await active
        assert failed.value.failure_code == "internal_defect"
        assert workers._workers == {}
        result = await workers.evaluate([ComputeJob()], deadline=time.monotonic() + 10)
        assert result[0].evidence["worker_pid"] != old_pid
    finally:
        await workers.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("job,code", [(ComputeJob(defect=True), "internal_defect"),
                                       (ComputeJob(result_bytes=5000), "invalid_args")])
async def test_internal_defect_and_oversized_reply_invalidate_request(job, code) -> None:
    workers = PlanningWorkers(PlanningSettings(workers=1, max_bytes=1024))
    try:
        with pytest.raises(SkillError) as failed:
            await workers.evaluate([job, ComputeJob()], deadline=time.monotonic() + 10)
        assert failed.value.failure_code == code
        assert not workers._tasks and workers._admitted == 0
    finally:
        await workers.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_running_and_queued_jobs_and_rejects_admission() -> None:
    workers = PlanningWorkers(PlanningSettings(workers=1, cancel_grace_s=0.05))
    active = asyncio.create_task(workers.evaluate([ComputeJob(30, False), ComputeJob()],
                                                 deadline=time.monotonic() + 60))
    await _until(lambda: bool(workers._workers))
    pids = {w.process.pid for w in workers._workers.values()}
    await workers.close()
    await asyncio.gather(active, return_exceptions=True)
    assert not workers._workers and not workers._tasks and workers._admitted == 0
    assert not ({p.pid for p in mp.active_children()} & pids)
    with pytest.raises(SkillError, match="shutting down"):
        await workers.evaluate([ComputeJob()], deadline=time.monotonic() + 10)

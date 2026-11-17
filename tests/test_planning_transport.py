# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real LHP sockets retain control traffic and connection-scoped planning cancellation."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

import pytest

from leapflow.hardware.lhp.client import DirectLhpClient, SocketLhpClient
from leapflow.hardware.lhp.planning_requests import PlanningRequests
from leapflow.hardware.lhp.protocol import CAP_SKILL_PLAN_CANCEL_V1, LhpProtocolError, LhpRemoteError
from leapflow.hardware.skills.planning import Deadline, Evaluation
from tests._harness.skill_stubs import ARM_ID, SEQUENCE, WORKCELL_ID, move_by
from tests.test_lhp_skill_methods import _Node, _until

pytestmark = [pytest.mark.component, pytest.mark.usefixtures("close_skill_runtimes")]


@dataclass(frozen=True)
class SlowJob:
    inner: Any
    duration_s: float = 1.0

    def evaluate(self, checkpoint: Deadline) -> Evaluation:
        until = time.monotonic() + self.duration_s
        while time.monotonic() < until:
            checkpoint.check()
        return self.inner.evaluate(checkpoint)


def slow_preparation(node: Any, duration_s: float = 1) -> None:
    skill = node.runtime._skills[SEQUENCE]
    original = skill.prepare

    async def prepare(ctx: Any, args: Any) -> Any:
        prepared = await original(ctx, args)
        return replace(prepared, jobs=tuple(SlowJob(job, duration_s) for job in prepared.jobs))

    skill.prepare = prepare


def params() -> dict[str, Any]:
    return {"workcell": WORKCELL_ID, "skill": SEQUENCE,
            "candidates": [{"id": "one", "args": move_by(5)}, {"id": "two", "args": move_by(10)}],
            "selection": {"require_verified": False, "order": ["tcp_distance_mm"]}}


@pytest.mark.parametrize("socket", [False, True])
async def test_timeout_cancels_compute_without_dropping_active_lease(tmp_path: Any, socket: bool) -> None:
    node = _Node(tmp_path, stall=True)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    client = SocketLhpClient(f"tcp://{host}:{port}") if socket else DirectLhpClient(node.server)
    try:
        hello = await client.connect()
        assert CAP_SKILL_PLAN_CANCEL_V1 in hello["capabilities"]
        planned = await client.request("skill.plan", {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(5)})
        started = await client.request("skill.execute", {"plan_digest": planned["plan"]["digest"], "lease_s": 2})
        await _until(lambda: node.sim.batches)
        sent = len(node.sim.batches)
        slow_preparation(node, 30)
        with pytest.raises(LhpRemoteError) as timeout:
            await client.request("skill.plan", params(), timeout_s=0.15)
        assert timeout.value.code == "timeout"
        await _until(lambda: not node.runtime._planning_tasks)
        assert node.runtime._plans == {}
        assert len(node.sim.batches) == sent
        renewed = await client.request("control.renew", {"control_id": started["lease"]["control_id"]})
        assert renewed["ok"]
        assert node.runtime.status(started["run"]["run_id"])[0].running
        assert (await client.request("probe", {"device_id": ARM_ID}))["connected"]
        await client.request("skill.stop", {"run_id": started["run"]["run_id"]})
    finally:
        await client.close()
        await node.server.stop()


@pytest.mark.parametrize("socket", [False, True])
async def test_expensive_planning_keeps_probe_status_renew_and_halt_responsive(tmp_path: Any, socket: bool) -> None:
    node = _Node(tmp_path, stall=True)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    client = SocketLhpClient(f"tcp://{host}:{port}") if socket else DirectLhpClient(node.server)
    planning = None
    try:
        await client.connect()
        planned = await client.request("skill.plan", {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(5)})
        started = await client.request("skill.execute", {"plan_digest": planned["plan"]["digest"], "lease_s": 2})
        await _until(lambda: node.sim.batches)
        slow_preparation(node, 2)
        planning = asyncio.create_task(client.request("skill.plan", params(), timeout_s=10))
        await _until(lambda: node.runtime._calculations._admitted == 2)
        for method, payload in [("probe", {"device_id": ARM_ID}), ("skill.status", {}),
                                ("control.renew", {"control_id": started["lease"]["control_id"]}),
                                ("halt", {"device_id": ARM_ID})]:
            before = time.monotonic()
            await client.request(method, payload, timeout_s=1)
            assert time.monotonic() - before < 0.8
            assert not planning.done(), "control traffic must not await planning"
        planning.cancel()
        with pytest.raises(asyncio.CancelledError):
            await planning
        await _until(lambda: not node.runtime._planning_tasks)
        assert node.runtime._plans == {}
    finally:
        if planning is not None:
            planning.cancel()
            await asyncio.gather(planning, return_exceptions=True)
        await client.close()
        await node.server.stop()


async def test_socket_cancel_receipt_is_owned_and_disconnect_forgets_only_own_preview(tmp_path: Any) -> None:
    node = _Node(tmp_path)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    first = SocketLhpClient(f"tcp://{host}:{port}")
    second = SocketLhpClient(f"tcp://{host}:{port}")
    try:
        await asyncio.gather(first.connect(), second.connect())
        selected = await first.request("skill.plan", params())  # request id 2
        other = await second.request("skill.plan", params())
        first_digest, other_digest = selected["plan"]["digest"], other["plan"]["digest"]
        assert first_digest != other_digest
        await second.request("skill.plan.cancel", {"request_id": 2})
        assert node.runtime.cached_plan(first_digest) is not None
        assert node.runtime.cached_plan(other_digest) is None
        await first.close()
        await _until(lambda: node.runtime.cached_plan(first_digest) is None)
        assert node.sim.batches == []
    finally:
        await first.close()
        await second.close()
        await node.server.stop()


async def test_disconnected_planning_request_does_not_publish_late(tmp_path: Any) -> None:
    node = _Node(tmp_path)
    slow_preparation(node, 30)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    client = SocketLhpClient(f"tcp://{host}:{port}")
    try:
        await client.connect()
        task = asyncio.create_task(client.request("skill.plan", params(), timeout_s=60))
        await _until(lambda: node.runtime._calculations._admitted == 2)
        await client.close()
        with pytest.raises(LhpRemoteError):
            await task
        await _until(lambda: not node.runtime._planning_tasks)
        assert not node.runtime._plans and node.sim.batches == []
    finally:
        await client.close()
        await node.server.stop()


async def test_direct_disconnect_forgets_only_owned_delivered_preview(tmp_path: Any) -> None:
    node = _Node(tmp_path)
    first, second = DirectLhpClient(node.server), DirectLhpClient(node.server)
    try:
        selected = await first.request("skill.plan", params())
        other = await second.request("skill.plan", params())
        await first.close()
        assert node.runtime.cached_plan(selected["plan"]["digest"]) is None
        assert node.runtime.cached_plan(other["plan"]["digest"]) is not None
        assert node.sim.batches == []
    finally:
        await first.close()
        await second.close()
        await node.server.stop()


@pytest.mark.parametrize("socket", [False, True])
async def test_delivered_legacy_preview_survives_client_exit_for_exact_execution(tmp_path: Any, socket: bool) -> None:
    node = _Node(tmp_path)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    first = SocketLhpClient(f"tcp://{host}:{port}") if socket else DirectLhpClient(node.server)
    second = SocketLhpClient(f"tcp://{host}:{port}") if socket else DirectLhpClient(node.server)
    try:
        selected = await first.request("skill.plan", {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(5)})
        digest = selected["plan"]["digest"]
        await first.close()
        if socket:
            await _until(lambda: not node.server._connection_states)
        assert node.runtime.cached_plan(digest) is not None
        assert node.sim.batches == []
        started = await second.request("skill.execute", {"plan_digest": digest, "lease_s": 2})
        await _until(lambda: not node.runtime.status(started["run"]["run_id"])[0].running)
        assert node.runtime.status(started["run"]["run_id"])[0].outcome.ok
        assert node.runtime.cached_plan(digest) is None
        writes = len(node.sim.batches)
        refused = await second.request("skill.execute", {"plan_digest": digest, "lease_s": 2})
        assert refused["outcome"]["failure_code"] == "plan_unknown" and len(node.sim.batches) == writes
    finally:
        await first.close()
        await second.close()
        await node.server.stop()


@pytest.mark.parametrize("batch", [False, True])
async def test_direct_delivery_cancellation_forgets_completed_preview(tmp_path: Any, monkeypatch: Any, batch: bool) -> None:
    node = _Node(tmp_path)
    client = DirectLhpClient(node.server)
    serve = client._serve
    caller = None

    async def cancel_at_delivery(request: Any) -> Any:
        result = await serve(request)
        # Cancel after publication but before the waiting request receives the result.
        caller.cancel()
        return result

    monkeypatch.setattr(client, "_serve", cancel_at_delivery)
    try:
        payload = params() if batch else {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(5)}
        caller = asyncio.create_task(client.request("skill.plan", payload))
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert node.runtime._plans == {} and node.sim.batches == []
    finally:
        await client.close()
        await node.server.stop()


async def test_receipt_ledger_prunes_expired_identity_before_registering() -> None:
    class EmptyCache:
        def cached_plan(self, digest: str) -> None:
            return None

        def discard_plan(self, digest: str) -> None:
            pass

    ledger = PlanningRequests(EmptyCache(), 1)
    ledger.publish(1, {"result": {"plan": {"digest": "expired"}}})
    task = asyncio.create_task(asyncio.sleep(0))
    ledger.register(1, task)
    await task
    ledger.disconnect()


async def test_receipt_ledger_is_bounded_and_closed_publication_is_forgotten() -> None:
    class Cache:
        def __init__(self) -> None:
            self.plans = {"a": 1, "b": 2, "c": 3}

        def cached_plan(self, digest: str) -> Any:
            return self.plans.get(digest)

        def discard_plan(self, digest: str) -> None:
            self.plans.pop(digest, None)

    cache = Cache()
    ledger = PlanningRequests(cache, 1)
    ledger.publish(1, {"result": {"plan": {"digest": "a"}}})
    ledger.publish(2, {"result": {"plan": {"digest": "b"}}})
    assert cache.plans == {"b": 2, "c": 3}
    ledger.disconnect()
    ledger.publish(3, {"result": {"plan": {"digest": "c"}}})
    assert cache.plans == {}
    task = asyncio.create_task(asyncio.sleep(0))
    try:
        with pytest.raises(LhpProtocolError):
            ledger.register(True, task)
        with pytest.raises(LhpProtocolError):
            ledger.register(4, task)
    finally:
        await task

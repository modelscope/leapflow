# Copyright (c) Alibaba, Inc. and its affiliates.
"""Production discovery/registry and real spawn lifecycle across both LHP bindings."""
from __future__ import annotations

import asyncio
import gc
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pytest

from leapflow.hardware.lhp.client import DirectLhpClient, SocketLhpClient
from leapflow.hardware.lhp.server import LhpNodeServer, WRITE_MODE_TRUSTED_V1
from tests._harness.planning import placement
from tests._harness.skill_stubs import ARM_ID, SEQUENCE, WORKCELL_ID, move_by

pytestmark = [pytest.mark.asyncio, pytest.mark.component]


async def test_read_only_benchmark_imports_without_pytest() -> None:
    root = Path(__file__).resolve().parents[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c",
        "import sys; sys.modules['pytest'] = None; "
        "import tests._harness.planning_benchmark; "
        "assert 'tests._harness.journey' not in sys.modules",
        cwd=root, env={**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root)))},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
    assert process.returncode == 0, (stdout, stderr)


@pytest.mark.parametrize("binding", ["direct", "socket"])
async def test_production_registry_discovers_plans_executes_and_disposes(tmp_path: Path, binding: str) -> None:
    pids = set()
    async with placement(tmp_path) as (registry, runtime):
        server = LhpNodeServer(registry, node_id="offline", skill_runtime=runtime,
                               write_mode=WRITE_MODE_TRUSTED_V1)
        if binding == "socket":
            host, port = await server.start(host="127.0.0.1", port=0)
            client = SocketLhpClient(f"tcp://{host}:{port}")
        else:
            client = DirectLhpClient(server)
        try:
            await client.connect()
            listing = await client.request("skill.list", {"workcell": WORKCELL_ID})
            cell = listing["workcells"][0]
            sequence = next(s for s in cell["skills"] if s["skill_id"] == SEQUENCE)
            assert sequence["available"] and sequence["batch_planning"]["supported"]
            planned = await client.request("skill.plan", {
                "workcell": WORKCELL_ID, "skill": SEQUENCE,
                "candidates": [{"id": "first", "args": move_by(5)},
                               {"id": "second", "args": move_by(10)}],
                "selection": {"order": ["tcp_distance_mm"]},
            })
            assert "plan" in planned, planned
            assert planned["batch"]["selected_id"] == "first"
            assert all(c["verified"] for c in planned["batch"]["candidates"])
            assert planned["batch"]["evidence"]["state_reads"] == 1
            pids = {c["evidence"]["worker_pid"] for c in planned["batch"]["candidates"]}
            assert len(pids) == 2
            transport = registry.get_open_transport(ARM_ID)
            assert not transport.write_log
            started = await client.request("skill.execute", {"plan_digest": planned["plan"]["digest"]})
            assert "run" in started, started
            finished = await runtime.wait(started["run"]["run_id"], timeout_s=3)
            assert finished.outcome.ok, finished.outcome
            assert sum(n == "ee.z" for n, _ in transport.write_log) == 1
            assert not runtime._plans
        finally:
            await client.close()
            await server.stop()
    assert not (pids & {p.pid for p in mp.active_children()})


async def test_production_shutdown_drains_real_planning_and_rejects_new_requests(tmp_path: Path, caplog) -> None:
    async with placement(tmp_path) as (registry, runtime):
        args = {"steps": [{"op": "move_linear", "relative": {"dz": 1}} for _ in range(16)]}
        task = asyncio.create_task(runtime.plan_batch(WORKCELL_ID, SEQUENCE, [
            {"id": f"path-{i}", "args": {**args, "speed_pct": 10 + i}} for i in range(8)
        ]))
        for _ in range(200):
            if runtime._calculations._workers:
                break
            await asyncio.sleep(0.005)
        pids = {w.process.pid for w in runtime._calculations._workers.values()}
        assert pids
        await runtime.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime._planning_tasks == set() and runtime._plans == {}
        refused = await runtime.plan_batch(WORKCELL_ID, SEQUENCE, [{"id": "late", "args": move_by(5)}])
        assert not refused.selected.ok and refused.report["selected_id"] is None
        assert not (pids & {p.pid for p in mp.active_children()})
        assert not registry.get_open_transport(ARM_ID).write_log
        gc.collect()
        await asyncio.sleep(0)
        assert "Task exception was never retrieved" not in caplog.text


@pytest.mark.parametrize("mode", ["serial", "one-worker", "multi-worker"])
async def test_read_only_benchmark_records_actual_evaluations_and_deduplication(tmp_path: Path, mode: str) -> None:
    from tests._harness.planning_benchmark import measure

    records = await measure(tmp_path, mode, "direct", 2)
    assert [r["temperature"] for r in records] == ["cold", "warm"]
    for record in records:
        assert record["selected_id"] == "path-0"
        assert record["evaluations"] == (8 if mode == "serial" else 6)
        assert record["ik_calls"] == (72 if mode == "serial" else 54)
        assert record["state_reads"] == (8 if mode == "serial" else 1)
        assert record["duplicate_reuse"] == (0 if mode == "serial" else 2)
        assert record["response_bytes"] > 0 and record["cpu_s"] > 0
        assert record["actuator_commands"] == record["node_llm_calls"] == 0
        assert record["probe_status_rounds"] > 0
        assert record["peak_compute_overlap"] <= (2 if mode == "multi-worker" else 1)

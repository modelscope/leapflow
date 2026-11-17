# Copyright (c) Alibaba, Inc. and its affiliates.
"""Read-only planning benchmark: python -m tests._harness.planning_benchmark.

The serial baseline reconstructs the former owner-loop evaluation schedule with
current identical geometry checks; it is not a historical executable comparison.
All placements use real registry/discovery and declaration-driven simulation.
No physical driver is opened, and no plan is executed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from leapflow.hardware.lhp.client import DirectLhpClient, SocketLhpClient
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.skills.planning import (
    CalculationPort, Deadline, Evaluation, PlanningCheckpoint, PlanningJob,
    PlanningSettings, Selection, encoded,
)
from leapflow.hardware.skills.planning_worker import PlanningWorkers
from tests._harness.planning import placement
from tests._harness.skill_stubs import ARM_ID, SEQUENCE, WORKCELL_ID


def programs() -> list[dict[str, Any]]:
    """Six independent task-equivalent paths and two exact duplicate alternatives."""
    paths = []
    for i in range(6):
        # Every complete program ends five millimeters above the identical start.
        excursion = i + 1
        steps = [{"op": "move_linear", "relative": {"dz": excursion}},
                 {"op": "move_linear", "relative": {"dz": -excursion}}] * 4
        paths.append({"id": f"path-{i}", "args": {"steps": [*steps,
            {"op": "move_linear", "relative": {"dz": 5}}]}})
    return [*paths, {"id": "duplicate-0", "args": paths[0]["args"]},
            {"id": "duplicate-1", "args": paths[1]["args"]}]


@dataclass(frozen=True)
class TimedJob:
    inner: PlanningJob

    def evaluate(self, checkpoint: PlanningCheckpoint) -> Evaluation:
        start, cpu = time.monotonic(), time.process_time()
        result = self.inner.evaluate(checkpoint)
        return replace(result, evidence={**result.evidence, "benchmark": {
            "start": start, "end": time.monotonic(),
            "cpu_s": time.process_time() - cpu, "pid": os.getpid(),
        }})


class MeasuredCalculations:
    """Opt-in measurement adapter; jobs remain detached and plugin-owned."""
    def __init__(self, delegate: CalculationPort | None) -> None:
        self.delegate = delegate
        self.values: list[Evaluation] = []

    async def evaluate(self, jobs: Sequence[PlanningJob], *, deadline: float) -> tuple[Evaluation, ...]:
        timed = tuple(TimedJob(job) for job in jobs)
        results = (await self.delegate.evaluate(timed, deadline=deadline) if self.delegate else
                   tuple(job.evaluate(Deadline(deadline)) for job in timed))
        self.values.extend(results)
        return results

    async def close(self) -> None:
        if self.delegate is not None:
            await self.delegate.close()


def overlap(values: Sequence[Evaluation]) -> int:
    boundaries = sorted((moment, delta) for result in values
                        for moment, delta in ((result.evidence["benchmark"]["start"], 1),
                                              (result.evidence["benchmark"]["end"], -1)))
    active = peak = 0
    for _, delta in boundaries:
        active += delta
        peak = max(peak, active)
    return peak


async def measure(root: Path, mode: str, binding: str, samples: int) -> list[dict[str, Any]]:
    offered = programs()
    workers = 1 if mode != "multi-worker" else 2
    calculations = MeasuredCalculations(None if mode == "serial" else
                                        PlanningWorkers(PlanningSettings(workers=workers)))
    records = []
    async with placement(root, workers=workers) as (registry, runtime):
        await runtime._calculations.close()
        runtime._calculations = calculations
        # Observe-only authority; use the same actual runtime, never execute a plan.
        server = LhpNodeServer(registry, node_id="benchmark", skill_runtime=runtime)
        if binding == "socket":
            host, port = await server.start(host="127.0.0.1", port=0)
            client = SocketLhpClient(f"tcp://{host}:{port}")
        else:
            client = DirectLhpClient(server)
        try:
            await client.connect()
            for sample in range(samples):
                calculations.values.clear()
                gaps, probe_latencies = [], []
                stop = asyncio.Event()

                async def responsiveness() -> None:
                    previous = time.monotonic()
                    while not stop.is_set():
                        await asyncio.sleep(0.01)
                        now = time.monotonic()
                        gaps.append(now - previous)
                        started = time.monotonic()
                        await client.request("probe", {"device_id": ARM_ID}, timeout_s=5)
                        await client.request("skill.status", {}, timeout_s=5)
                        probe_latencies.append(time.monotonic() - started)
                        previous = time.monotonic()

                heartbeat = asyncio.create_task(responsiveness())
                await asyncio.sleep(0)
                start = time.monotonic()
                try:
                    if mode == "serial":
                        replies = []
                        for entry in offered:
                            reply = await client.request("skill.plan", {
                                "workcell": WORKCELL_ID, "skill": SEQUENCE, "args": entry["args"],
                            }, timeout_s=35)
                            assert "plan" in reply, reply
                            replies.append(reply)
                            runtime.discard_plan(reply["plan"]["digest"])
                        selected, _ = Selection.from_mapping({"order": ["tcp_distance_mm"]}).choose(calculations.values)
                        selected_id = offered[selected]["id"]
                        response_bytes = sum(len(encoded(r).encode()) for r in replies)
                        state_reads, duplicates = len(offered), 0
                    else:
                        reply = await client.request("skill.plan", {
                            "workcell": WORKCELL_ID, "skill": SEQUENCE, "candidates": offered,
                            "selection": {"order": ["tcp_distance_mm"]},
                        }, timeout_s=35)
                        assert "plan" in reply, reply
                        assert all(c["verified"] for c in reply["batch"]["candidates"]), reply
                        selected_id = reply["batch"]["selected_id"]
                        response_bytes = len(encoded(reply).encode())
                        state_reads = reply["batch"]["evidence"]["state_reads"]
                        duplicates = reply["batch"]["evidence"]["duplicate_reuse"]
                        runtime.discard_plan(reply["plan"]["digest"])
                    latency = time.monotonic() - start
                finally:
                    stop.set()
                    await heartbeat
                assert selected_id == "path-0"
                assert not registry.get_open_transport(ARM_ID).write_log
                assert not runtime._plans
                records.append({
                    "mode": mode, "binding": binding, "temperature": "cold" if sample == 0 else "warm",
                    "latency_s": latency, "selected_id": selected_id,
                    "evaluations": len(calculations.values), "duplicate_reuse": duplicates,
                    "ik_calls": sum(v.evidence.get("ik_calls", 0) for v in calculations.values),
                    "state_reads": state_reads, "response_bytes": response_bytes,
                    "cpu_s": sum(v.evidence["benchmark"]["cpu_s"] for v in calculations.values),
                    "peak_compute_overlap": overlap(calculations.values),
                    "worker_pids": sorted({v.evidence["benchmark"]["pid"] for v in calculations.values}),
                    "max_event_loop_gap_s": max(gaps, default=0),
                    "max_probe_status_s": max(probe_latencies, default=0),
                    "probe_status_rounds": len(probe_latencies), "actuator_commands": 0, "node_llm_calls": 0,
                })
        finally:
            await client.close()
            await server.stop()
    return records


async def main(samples: int, binding: str) -> None:
    with tempfile.TemporaryDirectory(prefix="lf-planning-benchmark-") as root:
        results = []
        for mode in ("serial", "one-worker", "multi-worker"):
            results.extend(await measure(Path(root) / mode, mode, binding, samples))
        print(json.dumps({"platform": platform.platform(), "python": platform.python_version(),
                          "cpu_count": os.cpu_count(), "settings": PlanningSettings().to_dict(),
                          "candidate_sha256": hashlib.sha256(encoded(programs()).encode()).hexdigest(),
                          "candidate_bytes": len(encoded(programs()).encode()),
                          "baseline": "former serial owner-loop schedule, current identical pure evaluator",
                          "scope": "simulation only; no device execution or physical qualification",
                          "model_planning_rounds": "not measured here; asserted as one in R8 journey",
                          "records": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3, choices=range(2, 11))
    parser.add_argument("--binding", choices=("direct", "socket"), default="socket")
    options = parser.parse_args()
    asyncio.run(main(options.samples, options.binding))

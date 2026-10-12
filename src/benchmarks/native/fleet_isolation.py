# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native FleetManager isolation and emergency-stop benchmark.

Drives the real data plane: each trial stands up two ``LhpNodeServer``
nodes over TCP (ephemeral port), registers them through ``FleetManager``
over genuine LHP exchanges, then exercises node failure, concurrent halt,
and heartbeat timeout.  No MCP client stubs -- the node declarations are
served by real registries with simulated transports.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import fleet_isolation_score, halt_latency
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_context, make_registry
from leapflow.hardware.fleet import FleetManager, FleetNode
from leapflow.hardware.lhp import LhpNodeServer

_ADAPTER_ID = "native_fleet_isolation"
_VERSION = "2.0.0"
_SCENARIOS = (
    Scenario("single_node_failure", "One node failure does not block peers", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
    Scenario("halt_all_concurrent", "Fleet halt_all reaches every node", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
    Scenario("heartbeat_timeout", "Heartbeat timeout marks a node offline", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
)


class _NodePair:
    """Two live LHP nodes plus the host-side fleet bindings over them."""

    def __init__(self) -> None:
        self.host_registry = make_registry()
        self.node_a_registry = make_registry(make_context("fleet_a", transport_kind="simulated"))
        self.node_b_registry = make_registry(make_context("fleet_b", transport_kind="simulated"))
        self.server_a = LhpNodeServer(self.node_a_registry, node_id="node_a")
        self.server_b = LhpNodeServer(self.node_b_registry, node_id="node_b")
        self.manager = FleetManager(
            self.host_registry, heartbeat_interval_s=1.0, node_timeout_s=1.0
        )
        self.endpoint_a = ""
        self.endpoint_b = ""

    async def start(self) -> None:
        host, port = await self.server_a.start(host="127.0.0.1", port=0)
        self.endpoint_a = f"tcp://{host}:{port}"
        host, port = await self.server_b.start(host="127.0.0.1", port=0)
        self.endpoint_b = f"tcp://{host}:{port}"

    async def register(self) -> None:
        await self.manager.register_node(FleetNode("node_a", "Node A", self.endpoint_a))
        await self.manager.register_node(FleetNode("node_b", "Node B", self.endpoint_b))

    async def stop_node_a(self) -> None:
        """Simulate node A crashing: listener down, host binding unchanged."""
        await self.server_a.stop()

    async def close(self) -> None:
        await self.manager.close()
        await self.server_a.stop()
        await self.server_b.stop()
        await self.host_registry.close_all()
        await self.node_a_registry.close_all()
        await self.node_b_registry.close_all()


class FleetIsolationAdapter:
    """Verify FleetManager contains failures and halts nodes concurrently."""

    @property
    def adapter_id(self) -> str:
        return _ADAPTER_ID

    @property
    def adapter_version(self) -> str:
        return _VERSION

    async def availability(self) -> AvailabilityResult:
        return AvailabilityResult(_ADAPTER_ID, True, "ready")

    async def list_scenarios(
        self, *, tags: Sequence[str] = (), limit: int = 0,
    ) -> tuple[Scenario, ...]:
        rows = _SCENARIOS
        if tags:
            wanted = set(tags)
            rows = tuple(row for row in rows if wanted.intersection(row.tags))
        return rows[:limit] if limit > 0 else rows

    async def run_trial(
        self, scenario: Scenario, *, seed: int = 42, timeout_seconds: float = 300.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> TrialResult:
        del timeout_seconds, parameters
        started = time.time()
        fixture = _NodePair()
        try:
            await fixture.start()
            await fixture.register()
            payload, passed, latency = await self._exercise(fixture, scenario.scenario_id)
            return self._result(scenario, seed, started, payload, passed, latency)
        except Exception as exc:
            ended = time.time()
            return TrialResult(
                "", scenario.scenario_id, TrialStatus.ERROR, started_at=started,
                ended_at=ended, duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
                adapter_version=_VERSION, seed=seed, error=str(exc),
                error_type=type(exc).__name__,
            )
        finally:
            await fixture.close()

    async def _exercise(
        self, fixture: _NodePair, scenario_id: str
    ) -> tuple[dict[str, Any], bool, float]:
        if scenario_id == "heartbeat_timeout":
            await fixture.stop_node_a()
            await fixture.manager.start_heartbeat()
            # Two heartbeat cycles are needed: the first observes the node as
            # degraded, the second crosses node_timeout_s and marks it offline.
            await asyncio.sleep(2.2)
            await fixture.manager.stop_heartbeat()
            topology = fixture.manager.topology().to_dict()
            statuses = {node["node_id"]: node["status"] for node in topology["nodes"]}
            passed = (
                statuses.get("node_a") == "offline"
                and statuses.get("node_b") == "online"
            )
            return {"topology": topology}, passed, 0.0
        if scenario_id == "single_node_failure":
            # Node A dies after registration: its halt must fail without
            # blocking node B's halt.
            await fixture.stop_node_a()
        before = time.perf_counter()
        halted = await fixture.manager.halt_all()
        elapsed = time.perf_counter() - before
        if scenario_id == "single_node_failure":
            passed = halted["node_a"]["failures"] == 1 and halted["node_b"]["successes"] == 1
        else:
            passed = len(halted) == 2 and all(row["halted"] for row in halted.values())
        return {"halted": halted, "elapsed": elapsed}, passed, elapsed

    def _result(
        self, scenario: Scenario, seed: int, started: float,
        payload: dict[str, Any], passed: bool, latency: float,
    ) -> TrialResult:
        leaks = 0 if passed else 1
        metrics = (
            MetricValue("fleet_isolation_score", fleet_isolation_score(leaks, 1),
                        "ratio", threshold=1.0),
            MetricValue("halt_latency", halt_latency((latency,)) if latency else 0.0,
                        "seconds", higher_is_better=False),
        )
        ref = EvidenceStore(evidence_root(_ADAPTER_ID)).add_json(payload, kind="fleet")
        ended = time.time()
        return TrialResult(
            "", scenario.scenario_id, TrialStatus.PASSED if passed else TrialStatus.FAILED,
            metrics=metrics, evidence=(ref,), started_at=started, ended_at=ended,
            duration_seconds=ended - started, adapter_id=_ADAPTER_ID,
            adapter_version=_VERSION, seed=seed,
            error="" if passed else "fleet isolation invariant failed",
        )


__all__ = ["FleetIsolationAdapter"]

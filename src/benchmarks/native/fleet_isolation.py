# Copyright (c) Alibaba, Inc. and its affiliates.
"""Native FleetManager isolation and emergency-stop benchmark."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, Sequence

from benchmarks.evidence import EvidenceStore
from benchmarks.metrics import fleet_isolation_score, halt_latency
from benchmarks.models import AvailabilityResult, MetricValue, Scenario, TrialResult, TrialStatus
from benchmarks.native.harness import evidence_root, make_context, make_registry
from leapflow.hardware.fleet import FleetManager, FleetNode
from leapflow.hardware.transports.mcp import set_mcp_client_provider

_ADAPTER_ID = "native_fleet_isolation"
_VERSION = "1.0.0"
_SCENARIOS = (
    Scenario("single_node_failure", "One node failure does not block peers", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
    Scenario("halt_all_concurrent", "Fleet halt_all reaches every node", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
    Scenario("heartbeat_timeout", "Heartbeat timeout marks a node offline", adapter_id=_ADAPTER_ID,
             tags=("native", "fleet", "tier1")),
)


class _FleetClient:
    """Deterministic MCP client used through the public provider hook."""

    def __init__(self, discoveries: Sequence[Sequence[str]]) -> None:
        self._discoveries = [tuple(row) for row in discoveries]
        self.reachable = True

    async def list_tools(self) -> list[dict[str, str]]:
        if not self.reachable:
            raise ConnectionError("benchmark node unavailable")
        return [{"name": "hw_list"}]

    async def call_tool(self, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
        del params
        if not self.reachable:
            raise ConnectionError("benchmark node unavailable")
        if name != "hw_list":
            return {"ok": False, "error": "unknown tool"}
        devices = self._discoveries.pop(0) if self._discoveries else ()
        return {"devices": [{"device_id": device_id} for device_id in devices]}


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
        registry = make_registry(
            make_context("fleet_a", transport_kind="simulated"),
            make_context("fleet_b", transport_kind="simulated"),
        )
        discoveries = self._discoveries(scenario.scenario_id)
        client = _FleetClient(discoveries)
        undo = set_mcp_client_provider(lambda: client)
        manager = FleetManager(registry, heartbeat_interval_s=1.0, node_timeout_s=1.0)
        try:
            await self._register(manager)
            payload, passed, latency = await self._exercise(manager, client, scenario.scenario_id)
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
            await manager.close()
            await registry.close_all()
            undo()

    @staticmethod
    def _discoveries(scenario_id: str) -> tuple[tuple[str, ...], ...]:
        if scenario_id == "single_node_failure":
            return (("missing_device",), ("fleet_b",))
        return (("fleet_a",), ("fleet_b",))

    @staticmethod
    async def _register(manager: FleetManager) -> None:
        await manager.register_node(FleetNode("node_a", "Node A", "stdio:node-a"))
        await manager.register_node(FleetNode("node_b", "Node B", "stdio:node-b"))

    async def _exercise(self, manager: FleetManager, client: _FleetClient,
                        scenario_id: str) -> tuple[dict[str, Any], bool, float]:
        if scenario_id == "heartbeat_timeout":
            client.reachable = False
            await manager.start_heartbeat()
            await asyncio.sleep(1.15)
            await manager.stop_heartbeat()
            topology = manager.topology().to_dict()
            statuses = {node["node_id"]: node["status"] for node in topology["nodes"]}
            return {"topology": topology}, bool(statuses) and set(statuses.values()) == {"offline"}, 0.0
        before = time.perf_counter()
        halted = await manager.halt_all()
        elapsed = time.perf_counter() - before
        if scenario_id == "single_node_failure":
            passed = halted["node_a"]["failures"] == 1 and halted["node_b"]["successes"] == 1
        else:
            passed = len(halted) == 2 and all(row["halted"] for row in halted.values())
        return {"halted": halted, "elapsed": elapsed}, passed, elapsed

    def _result(self, scenario: Scenario, seed: int, started: float,
                payload: dict[str, Any], passed: bool, latency: float) -> TrialResult:
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

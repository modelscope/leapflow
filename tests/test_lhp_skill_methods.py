# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP ``skill.v1`` end to end: the node server in front of a simulated workcell.

Covers the two-phase wire contract (``skill.plan`` -> permit -> ``skill.execute``),
the permit binding to the approved plan, the shared device mutex (a run blocks
wire writes and a second run; a busy refusal does not burn the permit), and the
lease: renewal keeps a run alive, expiry or a dropped connection stops it with
``lease_lost`` and halts what it moved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

import pytest

from leapflow.hardware.lhp.client import SocketLhpClient
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.lhp.protocol import CAP_SKILL_V1
from leapflow.hardware.lhp.server import (
    WRITE_MODE_OBSERVE_ONLY,
    WRITE_MODE_PERMIT_REQUIRED,
    WRITE_MODE_TRUSTED_V1,
    LhpNodeServer,
)
from leapflow.hardware.skills.builtin.motion import skills as motion_skills
from leapflow.hardware.skills.builtin.sequence import skills as sequence_skills
from leapflow.hardware.skills.protocol import (
    FAILURE_DEVICE_BUSY,
    FAILURE_LEASE_LOST,
    FAILURE_PLAN_UNKNOWN,
    FAILURE_SKILL_UNAVAILABLE,
    FAILURE_STOPPED,
    SIDE_EFFECT_NONE,
    STATUS_SUCCEEDED,
)
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from tests._harness.skill_stubs import (
    ARM_ID,
    SEQUENCE,
    WORKCELL_ID,
    SimArmTransport,
    SkillNodeRegistry,
    move_by,
    write_workcell,
)

pytestmark = pytest.mark.asyncio

NODE_ID = "jetson01"


class _Node:
    """A node server over a simulated, calibrated Piper workcell."""

    def __init__(
        self, tmp_path: Path, *, write_mode: str = WRITE_MODE_TRUSTED_V1, **sim: Any
    ) -> None:
        self.registry = SkillNodeRegistry(
            SimArmTransport(**sim), store_path=tmp_path / "instrument.duckdb"
        )
        self.registry.calibrate_cell()
        directory = tmp_path / "workcells"
        write_workcell(directory)
        self.runtime = SkillRuntime(
            self.registry,
            settings=SkillRuntimeSettings(workcells_dir=directory),
            skills=(*motion_skills(), *sequence_skills()),
            poll_s=0.005,
        )
        self.issuer = PermitIssuer(generate_keypair()[0])
        verifier = PermitVerifier({self.issuer.key_id: self.issuer.public_key}, node_id=NODE_ID)
        self.server = LhpNodeServer(
            self.registry,
            node_id=NODE_ID,
            write_mode=write_mode,
            permit_verifier=verifier,
            skill_runtime=self.runtime,
        )
        self._ids = 0

    @property
    def sim(self) -> SimArmTransport:
        return self.registry.sim

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self._ids += 1
        response = await self.server.handle_request(
            {"id": self._ids, "method": method, "params": dict(params or {})}
        )
        assert response["ok"], response
        return response["result"]

    async def plan(self, skill: str, args: Mapping[str, Any]) -> dict[str, Any]:
        result = await self.call(
            "skill.plan", {"workcell": WORKCELL_ID, "skill": skill, "args": args}
        )
        assert "plan" in result, result
        return result

    def permit(self, digest: str) -> dict[str, Any]:
        return self.issuer.issue(
            audience=NODE_ID, session_id="s", workspace="/w", approval_id="a", digest=digest
        ).to_dict()

    async def run_status(self, run_id: str, wait_s: float = 0.0) -> dict[str, Any]:
        result = await self.call("skill.status", {"run_id": run_id, "wait_s": wait_s})
        (run,) = result["runs"]
        return run

    async def until_done(self, run_id: str, timeout_s: float = 3.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            run = await self.run_status(run_id, wait_s=0.2)
            if run["outcome"] is not None:
                return run
            assert asyncio.get_running_loop().time() < deadline, "run never ended"


async def _until(predicate: Any, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.005)


# ── Capability and catalog ───────────────────────────────────────────────────


async def test_hello_advertises_skill_v1_only_with_a_runtime(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    hello = await node.call("hello", {"protocol": 2})
    assert CAP_SKILL_V1 in hello["capabilities"]
    bare = LhpNodeServer(node.registry, node_id=NODE_ID, write_mode=WRITE_MODE_TRUSTED_V1)
    response = await bare.handle_request({"id": 1, "method": "hello", "params": {"protocol": 2}})
    assert CAP_SKILL_V1 not in response["result"]["capabilities"]
    missing = await bare.handle_request({"id": 2, "method": "skill.list", "params": {}})
    assert not missing["ok"]


async def test_list_returns_the_workcell_catalog(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    (cell,) = (await node.call("skill.list"))["workcells"]
    assert cell["workcell"] == WORKCELL_ID
    assert {SEQUENCE, "motion.home"} <= {skill["skill_id"] for skill in cell["skills"]}


# ── Plan -> permit -> execute ────────────────────────────────────────────────


async def test_trusted_plan_and_execute_moves_the_arm(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    planned = await node.plan(SEQUENCE, move_by(10))
    assert planned["permit_digest"]
    started = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    assert started["lease"]["control_id"]
    run = await node.until_done(started["run"]["run_id"])
    assert run["status"] == STATUS_SUCCEEDED
    assert node.sim.batches, "the arm received the move"
    sessions = (await node.call("control.status"))["sessions"]
    assert sessions == [], "a finished run releases its lease at once"


async def test_permit_required_execute_with_a_plan_bound_permit(tmp_path: Path) -> None:
    node = _Node(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED)
    planned = await node.plan(SEQUENCE, move_by(10))
    started = await node.call(
        "skill.execute",
        {
            "plan_digest": planned["plan"]["digest"],
            "permit": node.permit(planned["permit_digest"]),
        },
    )
    run = await node.until_done(started["run"]["run_id"])
    assert run["status"] == STATUS_SUCCEEDED


async def test_a_permit_for_another_plan_is_refused_and_moves_nothing(tmp_path: Path) -> None:
    node = _Node(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED)
    approved = await node.plan(SEQUENCE, move_by(10))
    other = await node.plan(SEQUENCE, move_by(40))
    result = await node.call(
        "skill.execute",
        {
            "plan_digest": other["plan"]["digest"],
            "permit": node.permit(approved["permit_digest"]),
        },
    )
    outcome = result["outcome"]
    assert outcome["failure_code"] == "permit_digest_mismatch"
    assert outcome["side_effect_state"] == SIDE_EFFECT_NONE
    assert node.sim.batches == []


async def test_execute_without_a_permit_is_refused(tmp_path: Path) -> None:
    node = _Node(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED)
    planned = await node.plan(SEQUENCE, move_by(10))
    result = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    assert result["outcome"]["failure_code"] == "permit_required"
    assert node.sim.batches == []


async def test_observe_only_refuses_execute(tmp_path: Path) -> None:
    node = _Node(tmp_path, write_mode=WRITE_MODE_OBSERVE_ONLY)
    planned = await node.plan(SEQUENCE, move_by(10))
    result = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    assert result["outcome"]["failure_code"] == "writes_disabled"
    assert node.sim.batches == []


async def test_unknown_plan_is_an_outcome_not_an_error(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    result = await node.call("skill.execute", {"plan_digest": "0" * 64})
    assert result["outcome"]["failure_code"] == FAILURE_PLAN_UNKNOWN


# ── Device mutex ─────────────────────────────────────────────────────────────


async def test_busy_refusal_does_not_burn_the_permit(tmp_path: Path) -> None:
    node = _Node(tmp_path, write_mode=WRITE_MODE_PERMIT_REQUIRED, stall=True)
    first = await node.plan(SEQUENCE, move_by(50))
    started = await node.call(
        "skill.execute",
        {"plan_digest": first["plan"]["digest"], "permit": node.permit(first["permit_digest"])},
    )
    await _until(lambda: node.sim.batches)

    second = await node.plan(SEQUENCE, move_by(25))
    request = {
        "plan_digest": second["plan"]["digest"],
        "permit": node.permit(second["permit_digest"]),
    }
    busy = await node.call("skill.execute", request)
    assert busy["outcome"]["failure_code"] == FAILURE_DEVICE_BUSY

    await node.call("skill.stop", {"run_id": started["run"]["run_id"]})
    node.sim.stall = False
    retried = await node.call("skill.execute", request)
    assert "run" in retried, "the same permit still authorizes the plan once the device is free"
    await node.until_done(retried["run"]["run_id"])


async def test_wire_writes_are_refused_while_a_run_holds_the_device(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    planned = await node.plan(SEQUENCE, move_by(50))
    started = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    await _until(lambda: node.sim.batches)
    sent = len(node.sim.batches)

    outcome = await node.call(
        "write", {"device_id": ARM_ID, "channel_id": "gripper.position", "value": 30.0}
    )
    assert (outcome["ok"], outcome["failure_code"]) == (False, FAILURE_DEVICE_BUSY)
    assert outcome["side_effect_state"] == SIDE_EFFECT_NONE
    assert len(node.sim.batches) == sent

    await node.call("skill.stop", {"reason": "test over"})
    run = await node.run_status(started["run"]["run_id"])
    assert run["outcome"]["failure_code"] == FAILURE_STOPPED


# ── Lease ────────────────────────────────────────────────────────────────────


async def test_lease_expiry_stops_the_run_with_lease_lost_and_halts(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    planned = await node.plan(SEQUENCE, move_by(50))
    started = await node.call(
        "skill.execute", {"plan_digest": planned["plan"]["digest"], "lease_s": 0.2}
    )
    run = await node.until_done(started["run"]["run_id"])
    assert run["outcome"]["failure_code"] == FAILURE_LEASE_LOST
    assert node.sim.halts == 1
    assert not node.runtime.holds_device(ARM_ID)


async def test_renewal_keeps_the_run_alive(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    planned = await node.plan(SEQUENCE, move_by(50))
    started = await node.call(
        "skill.execute", {"plan_digest": planned["plan"]["digest"], "lease_s": 0.2}
    )
    control_id = started["lease"]["control_id"]
    for _ in range(8):
        await asyncio.sleep(0.07)
        renewed = await node.call("control.renew", {"control_id": control_id})
        assert renewed["ok"], renewed
    run = await node.run_status(started["run"]["run_id"])
    assert run["outcome"] is None, "a renewed run outlives several lease periods"
    await node.call("skill.stop", {"run_id": started["run"]["run_id"]})
    assert node.sim.halts == 1


async def test_closing_the_lease_stops_the_run(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    planned = await node.plan(SEQUENCE, move_by(50))
    started = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    await _until(lambda: node.sim.batches)
    closed = await node.call("control.close", {"control_id": started["lease"]["control_id"]})
    assert closed["ok"] and closed["run"]["outcome"]["failure_code"] == FAILURE_STOPPED
    assert node.sim.halts == 1


async def test_a_dropped_connection_stops_its_run(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    host, port = await node.server.start(host="127.0.0.1", port=0)
    client = SocketLhpClient(f"tcp://{host}:{port}", timeout_s=5.0)
    try:
        await client.connect()
        planned = await client.request(
            "skill.plan", {"workcell": WORKCELL_ID, "skill": SEQUENCE, "args": move_by(50)}
        )
        started = await client.request(
            "skill.execute", {"plan_digest": planned["plan"]["digest"], "lease_s": 10.0}
        )
        await _until(lambda: node.sim.batches)
        await client.close()
        run_id = started["run"]["run_id"]
        await _until(lambda: not node.runtime.status(run_id)[0].running)
        (run,) = node.runtime.status(run_id)
        assert run.outcome is not None and run.outcome.failure_code == FAILURE_LEASE_LOST
        assert node.sim.halts == 1
    finally:
        await node.server.stop()


# ── Status bounds ────────────────────────────────────────────────────────────


async def test_status_wait_is_bounded(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    response = await node.server.handle_request(
        {"id": 1, "method": "skill.status", "params": {"run_id": "x", "wait_s": 60}}
    )
    assert not response["ok"]
    assert response["error"]["code"] == "bad_request"


async def test_intent_and_label_params_are_bounded(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    planned = await node.plan(SEQUENCE, move_by(10))
    long_intent = await node.server.handle_request(
        {
            "id": 1,
            "method": "skill.execute",
            "params": {"plan_digest": planned["plan"]["digest"], "intent": "x" * 501},
        }
    )
    assert long_intent["error"]["code"] == "bad_request" and node.sim.batches == []
    not_bool = await node.server.handle_request(
        {"id": 2, "method": "skill.label", "params": {"run_id": "r", "success": "yes"}}
    )
    assert not_bool["error"]["code"] == "bad_request"


async def test_label_without_recording_is_a_typed_outcome(tmp_path: Path) -> None:
    node = _Node(tmp_path)
    planned = await node.plan(SEQUENCE, move_by(10))
    started = await node.call(
        "skill.execute", {"plan_digest": planned["plan"]["digest"], "intent": "go up"}
    )
    run = await node.until_done(started["run"]["run_id"])
    assert run["intent"] == "go up"
    result = await node.call("skill.label", {"run_id": run["run_id"], "success": True})
    assert result["outcome"]["failure_code"] == FAILURE_SKILL_UNAVAILABLE


async def test_server_stop_ends_live_runs(tmp_path: Path) -> None:
    node = _Node(tmp_path, stall=True)
    planned = await node.plan(SEQUENCE, move_by(50))
    started = await node.call("skill.execute", {"plan_digest": planned["plan"]["digest"]})
    await _until(lambda: node.sim.batches)
    await node.server.stop()
    (run,) = node.runtime.status(started["run"]["run_id"])
    assert not run.running and node.sim.halts == 1

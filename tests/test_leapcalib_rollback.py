# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real rollback owner/skill/permit plumbing; synthetic geometry is NOT qualification."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapcalib.contracts.identity import digest_of
from leapcalib.geometry.se3 import Transform
from leapflow.hardware.calibration.acquisition import SettledProfile
from leapflow.hardware.calibration.command_safety import (
    CommandSafetySnapshot,
    DualModelCommandSafety,
)
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry
from leapflow.hardware.context import ContextProvenance, TransportRef
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.node_gate import NodeWriteGate
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.skills.builtin.governed_calibration import skills
from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain
from leapflow.hardware.skills.runtime import SkillRuntime, SkillRuntimeSettings
from leapflow.hardware.transport import BatchReading, TransportError
from tests._harness.skill_stubs import ARM_ID, JOINTS, WORKCELL_ID, piper_context, write_workcell

pytestmark = pytest.mark.component
POSES = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")


class Provider:
    kind = "isolated-rollback"

    def __init__(self, declaration: Any) -> None:
        self.declaration = declaration

    def discover(self) -> tuple[Any, ...]:
        return (self.declaration,)


@pytest.fixture
async def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    declaration = piper_context()
    chain = DhChain.from_mapping(declaration.kinematics.dh)
    flange = chain.forward((0,) * chain.dof)
    values = {
        **dict.fromkeys(JOINTS, 0),
        **dict(zip(POSES, vars(flange).values())),
        "arm.enabled": True,
        "arm.motion": True,
        "arm.ctrl_mode": "can",
        "arm.status": {"arm_status": 0, "joint_communication_errors": []},
        "arm.speed_pct": 5,
    }
    declaration = replace(
        declaration,
        transport=TransportRef("mock", {"values": values}),
        provenance=ContextProvenance(verified_by="isolated synthetic fixture"),
    )
    provider = Provider(declaration)
    registry = HardwareRegistry(
        HardwareSettings(instrument_db_path=str(tmp_path / "instrument.duckdb")),
        providers=[provider],
    )
    registry.load()
    write_workcell(tmp_path / "workcells", {"workcell": WORKCELL_ID, "roles": {"arm": ARM_ID}})
    runtime = SkillRuntime(
        registry,
        settings=SkillRuntimeSettings(workcells_dir=tmp_path / "workcells", stop_grace_s=0.01),
        skills=skills(),
    )
    assert not runtime.load_report.rejected
    issuer = PermitIssuer(generate_keypair()[0])
    verifier = PermitVerifier({issuer.key_id: issuer.public_key}, node_id="node")
    server = LhpNodeServer(
        registry, node_id="node", skill_runtime=runtime, permit_verifier=verifier
    )
    safety = DualModelCommandSafety(registry)
    registry.bind_command_safety(safety)
    transport = await registry.transport(ARM_ID)

    async def cohort(ids):
        readings = tuple([await transport.read(cid) for cid in ids])
        stamp = time.monotonic()
        return BatchReading(ARM_ID, tuple(replace(r, monotonic_at=stamp) for r in readings))

    monkeypatch.setattr(transport, "read_batch", cohort, raising=False)

    async def forbidden_write_batch(commands):
        pytest.fail("rollback must never dispatch a batch")

    monkeypatch.setattr(transport, "write_batch", forbidden_write_batch, raising=False)
    nominal = PhysicalModel(
        chain,
        (0,) * chain.dof,
        Transform(),
        digest_of(chain.to_dict()),
        "synthetic-deps",
        "synthetic-prior",
    )
    bounds = SafetyGeometry(
        (1,) * (chain.dof + 1),
        WorkspaceBox(-2000, 2000, -2000, 2000, -2000, 2000),
        (WorkspaceBox(1800, 1900, 1800, 1900, 1800, 1900),),
        "synthetic-geometry",
        1,
        2,
        10,
        1,
    )
    profile = SettledProfile(
        0.1, 0.02, 0.001, 0.002, 0.001, 0.001, 0.2, 0.1, 0.0001, 0.01, 0.005, 0.16, 2
    )
    value = CommandSafetySnapshot(
        ARM_ID,
        digest_of(declaration.to_dict()),
        "synthetic-boot",
        1,
        nominal,
        replace(nominal, zero_offsets_rad=(math.radians(0.1), *(0,) * (chain.dof - 1))),
        bounds,
        JOINTS,
        POSES,
        ("arm.enabled", "arm.motion", "arm.ctrl_mode", "arm.status"),
        ("can",),
        profile,
        0.1,
        0.1,
        "arm.speed_pct",
        10,
        (0.01,) * chain.dof,
        1,
        "synthetic-rates",
    )
    fixture = SimpleNamespace(
        provider=provider,
        registry=registry,
        runtime=runtime,
        server=server,
        issuer=issuer,
        verifier=verifier,
        safety=safety,
        transport=transport,
        value=value,
    )
    try:
        yield fixture
    finally:
        await runtime.close()
        await server.sessions.shutdown()
        await registry.close_all()


async def commit(owner, value, request_id, expected=None, *, project=True):
    store = owner.registry.calibration_generations
    return await owner.server.sessions.run_reserved(
        (ARM_ID,),
        owner_id="synthetic-setup",
        operation=lambda: store.commit_generation(
            workcell_id=WORKCELL_ID,
            request_id=request_id,
            expected_generation=expected,
            dependency_digest=value.corrected.dependency_digest,
            model=value.to_dict(),
            accepted_blocks={"synthetic": {}},
            causal={"qualification": "software only"},
        ),
        on_settle=(lambda generation: owner.safety.publish_reserved(value, generation))
        if project
        else None,
    )


async def request(owner, method, **params):
    response = await owner.server.handle_request(
        {"id": "synthetic-request", "method": method, "params": params}
    )
    assert "error" not in response, response
    return response["result"]


async def plan(owner, expected):
    planned = await request(
        owner,
        "skill.plan",
        workcell=WORKCELL_ID,
        skill="calib.rollback",
        args={"expected_generation": expected},
    )
    assert "plan" in planned, planned
    proof = owner.issuer.issue(
        audience="node",
        session_id="client-a",
        workspace="workspace-a",
        approval_id="synthetic-human-boundary",
        digest=planned["permit_digest"],
    )
    return planned, proof


async def execute(owner, planned, proof, *, lease_s=10):
    return await request(
        owner,
        "skill.execute",
        plan_digest=planned["plan"]["digest"],
        permit=proof.to_dict(),
        lease_s=lease_s,
    )


async def finish(owner, result):
    assert "run" in result, result
    run = await owner.runtime.wait(result["run"]["run_id"], timeout_s=5)
    assert run is not None and not run.running
    assert owner.transport.write_log == () and owner.transport.halt_calls == 0
    return run


async def test_nonmotion_exact_signed_run_restores_revalidated_same_owner_prior(owner):
    first = await commit(owner, owner.value, "first")
    await commit(owner, replace(owner.value, generation=2), "second", 1)
    planned, proof = await plan(owner, 2)
    assert (
        planned["consent"] == "each_run" and planned["plan"]["basis"]["rollback"]["commands"] == []
    )
    run = await finish(owner, await execute(owner, planned, proof))
    assert run.outcome.ok and run.outcome.side_effect_state == "committed"
    assert run.actuators == () and run.outcome.result["state"] == "rolled_back"
    assert (
        await asyncio.to_thread(owner.registry.calibration_generations.snapshot, WORKCELL_ID)
        == first
    )
    assert owner.safety.rollback_projection(first) is owner.value
    events = await asyncio.to_thread(owner.registry.calibration_generations.pending_events)
    event = next(e for e in events if e["kind"] == "rolled_back")
    assert event["causal"]["plan_digest"] == planned["plan"]["digest"]
    assert "permit" not in str(event)
    assert not owner.runtime.holds_device(ARM_ID)
    replay = await execute(owner, planned, proof)
    assert replay["outcome"]["failure_code"] == "plan_unknown"


@pytest.mark.parametrize("reason", ["first-generation", "missing-history", "unsafe-prior"])
async def test_unqualified_prior_atomically_stops_use_without_commands(owner, reason):
    await commit(owner, owner.value, "first", project=reason != "missing-history")
    expected = 1
    if reason != "first-generation":
        await commit(owner, replace(owner.value, generation=2), "second", 1)
        expected = 2
    if reason == "unsafe-prior":
        owner.transport.set_value("arm.speed_pct", 30)
    planned, proof = await plan(owner, expected)
    run = await finish(owner, await execute(owner, planned, proof))
    assert run.outcome.ok and run.outcome.result["state"] == "invalidated"
    state = await asyncio.to_thread(owner.registry.calibration_generations.state, WORKCELL_ID)
    assert state.generation is None and state.invalidation is not None
    assert "invalidated" in owner.safety.control_refusal(ARM_ID)
    assert not (await NodeWriteGate(owner.registry).write(ARM_ID, JOINTS[0], 0)).ok


@pytest.mark.parametrize("reason", ["busy", "stale", "wrong-permit"])
async def test_refusals_do_not_consume_exact_plan_authority(owner, reason):
    await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    token = None
    if reason == "busy":
        token = owner.server.sessions.reserve((ARM_ID,), owner_id="other-client-workspace-b")
    elif reason == "stale":
        await commit(owner, replace(owner.value, generation=2), "second", 1)
    else:
        proof = owner.issuer.issue(
            audience="node",
            session_id="client-a",
            workspace="workspace-a",
            approval_id="synthetic-human",
            digest="not-this-plan",
        )
    try:
        result = await execute(owner, planned, proof)
        assert result["outcome"]["side_effect_state"] == "none"
        assert (
            result["outcome"]["failure_code"]
            == {
                "busy": "device_busy",
                "stale": "plan_stale",
                "wrong-permit": "permit_digest_mismatch",
            }[reason]
        )
        assert owner.verifier.verify(proof, expected_digest=proof.action_digest) is None
        assert owner.transport.write_log == ()
    finally:
        if token:
            owner.server.sessions.release_reservation(token)


@pytest.mark.parametrize("defect", ["moving", "disabled", "mode", "bus"])
async def test_unsafe_feedback_refuses_before_any_persistence(owner, monkeypatch, defect):
    await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    if defect == "bus":

        async def unavailable(ids):
            raise TransportError("isolated raw feedback unavailable")

        monkeypatch.setattr(owner.transport, "read_batch", unavailable)
        monkeypatch.setattr(owner.transport, "read", unavailable)
    else:
        cid, value = {
            "moving": ("arm.motion", False),
            "disabled": ("arm.enabled", False),
            "mode": ("arm.ctrl_mode", "teach"),
        }[defect]
        owner.transport.set_value(cid, value)
    run = await finish(owner, await execute(owner, planned, proof))
    assert not run.outcome.ok and run.outcome.side_effect_state == "none"
    assert run.outcome.failure_code != "internal_defect"
    assert (
        await asyncio.to_thread(owner.registry.calibration_generations.snapshot, WORKCELL_ID)
    ).generation == 1


@pytest.mark.parametrize("stop", ["cancel", "disconnect", "expiry"])
async def test_started_rollback_retains_reservation_and_hold_until_real_db_settles(
    owner, monkeypatch, stop
):
    await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    store = owner.registry.calibration_generations
    rollback = store.rollback
    entered, release = threading.Event(), threading.Event()

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        return rollback(**kwargs)

    monkeypatch.setattr(store, "rollback", delayed)
    started = await execute(owner, planned, proof, lease_s=1 if stop == "expiry" else 10)
    run_id = started["run"]["run_id"]
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if stop == "cancel":
            await owner.runtime.stop(run_id)
        elif stop == "disconnect":
            await owner.server.sessions.on_disconnect("")
        else:
            async with asyncio.timeout(2):
                while (await owner.server.sessions.status({"device_id": ARM_ID}))["sessions"]:
                    await asyncio.sleep(0.01)
        assert owner.runtime.holds_device(ARM_ID)
        assert owner.server.sessions.reserved_device(ARM_ID)
        with pytest.raises(ValueError):
            owner.server.sessions.reserve((ARM_ID,), owner_id="other-client")
        competing = await request(owner, "control.open", device_id=ARM_ID)
        assert competing["failure_code"] == "device_busy"
    finally:
        release.set()
    run = await finish(owner, started)
    assert not run.outcome.ok and run.outcome.side_effect_state == "partial"
    assert not owner.server.sessions.reserved_device(ARM_ID)
    assert not owner.runtime.holds_device(ARM_ID)
    assert (await asyncio.to_thread(store.state, WORKCELL_ID)).invalidation is not None


async def test_cancelled_reserved_raw_read_settles_before_shutdown_and_never_mutates(
    owner, monkeypatch
):
    first = await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    store, sessions = owner.registry.calibration_generations, owner.server.sessions
    original = store.snapshot
    entered, release = threading.Event(), threading.Event()

    def delayed(workcell):
        if sessions.reserved_device(ARM_ID):
            entered.set()
            assert release.wait(5)
        return original(workcell)

    monkeypatch.setattr(store, "snapshot", delayed)
    started = await execute(owner, planned, proof)
    run_id = started["run"]["run_id"]
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not (await NodeWriteGate(owner.registry).write(ARM_ID, JOINTS[0], 0)).ok
        assert (await request(owner, "control.open", device_id=ARM_ID))[
            "failure_code"
        ] == "device_busy"
        await owner.runtime.stop(run_id)
        assert owner.runtime.holds_device(ARM_ID) and sessions.reserved_device(ARM_ID)
        closing = asyncio.create_task(sessions.shutdown())
        await asyncio.sleep(0.02)
        assert not closing.done()
    finally:
        release.set()
    run = await finish(owner, started)
    if closing is not None:
        await closing
    assert not run.outcome.ok and run.outcome.side_effect_state == "none"
    assert not sessions.reserved_device(ARM_ID) and not owner.runtime.holds_device(ARM_ID)
    assert await asyncio.to_thread(original, WORKCELL_ID) == first
    assert len(await asyncio.to_thread(store.pending_events)) == 1


async def test_dependency_change_during_reserved_feedback_refuses_without_db_dispatch(
    owner, monkeypatch
):
    first = await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    original = owner.transport.read_batch
    provider = owner.provider
    changed = False

    async def changed_declaration(ids):
        nonlocal changed
        reading = await original(ids)
        if not changed and owner.server.sessions.reserved_device(ARM_ID):
            changed = True
            provider.declaration = replace(
                provider.declaration, display_name="Changed during preflight"
            )
            owner.registry.load()
        return reading

    monkeypatch.setattr(owner.transport, "read_batch", changed_declaration)
    run = await finish(owner, await execute(owner, planned, proof))
    assert changed and not run.outcome.ok and run.outcome.side_effect_state == "none"
    assert run.outcome.failure_code == "check_failed"
    assert (
        await asyncio.to_thread(owner.registry.calibration_generations.snapshot, WORKCELL_ID)
        == first
    )
    assert not owner.server.sessions.reserved_device(ARM_ID)


async def test_rollback_publication_failure_keeps_landed_state_and_blocks_consumers(
    owner, monkeypatch
):
    await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)

    def broken(device):
        assert owner.server.sessions.reserved_device(device)
        raise ValueError("isolated publication failure after invalidation")

    monkeypatch.setattr(owner.safety, "invalidate_reserved", broken)
    run = await finish(owner, await execute(owner, planned, proof))
    assert not run.outcome.ok and run.outcome.side_effect_state == "unknown"
    assert (
        await asyncio.to_thread(owner.registry.calibration_generations.state, WORKCELL_ID)
    ).invalidation
    assert owner.server.sessions.unsettled_device(ARM_ID)
    assert not (await NodeWriteGate(owner.registry).write(ARM_ID, JOINTS[0], 0)).ok


async def test_post_commit_valueerror_is_unknown_and_latches_all_consumers(owner, monkeypatch):
    await commit(owner, owner.value, "first")
    planned, proof = await plan(owner, 1)
    store = owner.registry.calibration_generations
    original = store.rollback

    def ambiguous(**kwargs):
        original(**kwargs)
        raise ValueError("isolated failure after commit but before return")

    monkeypatch.setattr(store, "rollback", ambiguous)
    run = await finish(owner, await execute(owner, planned, proof))
    assert not run.outcome.ok and run.outcome.side_effect_state == "unknown"
    assert run.outcome.failure_code == "device_fault"
    assert (await asyncio.to_thread(store.state, WORKCELL_ID)).invalidation is not None
    assert owner.server.sessions.busy_device((ARM_ID,)) == ARM_ID
    assert not (await NodeWriteGate(owner.registry).write(ARM_ID, JOINTS[0], 0)).ok

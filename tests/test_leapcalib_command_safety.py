# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared-owner safety plumbing with synthetic bounds; NOT physical qualification."""
from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from leapcalib.contracts.identity import digest_of
from leapcalib.geometry.se3 import Transform as SiTransform
from leapflow.hardware.calibration.acquisition import SettledProfile
from leapflow.hardware.calibration.command_safety import CommandSafetySnapshot, DualModelCommandSafety
from leapflow.hardware.calibration.physical_model import PhysicalModel, SafetyGeometry
from leapflow.hardware.context import Channel, ContextProvenance, Envelope, HardwareContext, Interlock, TransportRef
from leapflow.hardware.control_bus import ControlBusConfig, ControlCommand, ControlState, HighFrequencyControlBus
from leapflow.hardware.lhp.control_sessions import NodeSessions
from leapflow.hardware.lhp.permit import PermitIssuer, PermitVerifier, generate_keypair, request_digest
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.node_gate import CommandSafety, NodeWriteGate, control_safety_refusal
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.skills.geometry import WorkspaceBox
from leapflow.hardware.skills.kinematics import DhChain, DhLink
from leapflow.hardware.transport import BatchReading, BatchWriteOutcome, TransportError

pytestmark = pytest.mark.component


class Provider:
    kind = "static"

    def __init__(self, context: HardwareContext) -> None:
        self.context = context

    def discover(self) -> tuple[HardwareContext, ...]:
        return (self.context,)


def settled() -> SettledProfile:
    return SettledProfile(0.1, 0.02, 0.001, 0.002, 0.001, 0.001, 0.2, 0.1,
                          0.0001, 0.01, 0.005, 0.16, 1.0)


def context() -> HardwareContext:
    values = {"joint": 0.0, "x": 100.0, "y": 0.0, "z": 0.0, "rx": 0.0,
              "ry": 0.0, "rz": 0.0, "enabled": True, "arrived": True,
              "mode": "joint", "status": {"arm_status": 0, "joint_communication_errors": []},
              "speed": 5.0}
    return HardwareContext(
        device_id="arm", transport=TransportRef("mock", {"values": values}),
        channels=(Channel("joint", direction="readwrite", unit="deg", quantity="joint_position",
                          effect="actuate", envelope=Envelope(declared=True, min_value=-180, max_value=180)),
                  *(Channel(cid, direction="readwrite", unit=unit, effect="actuate",
                            envelope=Envelope(declared=True, min_value=-500, max_value=500))
                    for cid, unit in zip(("x", "y", "z", "rx", "ry", "rz"), ("mm", "mm", "mm", "deg", "deg", "deg"))),
                  *(Channel(cid, direction="read") for cid in ("enabled", "arrived", "mode", "status")),
                  Channel("speed", direction="readwrite", unit="percent", effect="configure",
                          envelope=Envelope(declared=True, min_value=1, max_value=10))),
        provenance=ContextProvenance(verified_by="isolated-software-test"), halt_supported=True,
    )


def snapshot(ctx: HardwareContext, **changes: Any) -> CommandSafetySnapshot:
    chain = DhChain((DhLink(100, 0, 0),), "standard")
    nominal = PhysicalModel(chain, (0,), SiTransform(), digest_of(chain.to_dict()), "deps", "synthetic-prior")
    corrected = replace(nominal, zero_offsets_rad=(math.radians(1),), certificate_digest="synthetic-certificate")
    geometry = SafetyGeometry((1, 1), WorkspaceBox(-500, 500, -500, 500, -500, 500),
                              (WorkspaceBox(300, 310, 300, 310, 300, 310),),
                              "synthetic-geometry", 0.1, 0.1, 1.0, 0.1)
    return CommandSafetySnapshot(**{
        "device_id": ctx.device_id, "declaration_digest": digest_of(ctx.to_dict()),
        "node_session_id": "isolated-boot", "generation": 1, "nominal": nominal,
        "corrected": corrected, "geometry": geometry, "joint_ids": ("joint",),
        "pose_ids": ("x", "y", "z", "rx", "ry", "rz"),
        "health_ids": ("enabled", "arrived", "mode", "status"), "allowed_modes": ("joint",),
        "feedback_profile": settled(), "nominal_fk_mm": 0.1, "nominal_fk_deg": 0.1,
        "speed_channel": "speed", "speed_pct_max": 10, "joint_rate_bounds_rad_s": (0.01,),
        "endpoint_rate_bound_mm_s": 1.0, "rate_provenance_digest": "synthetic-rate-observation",
        **changes,
    })


@pytest.fixture
async def owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry = HardwareRegistry(HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "instrument.duckdb")),
                                providers=[Provider(context())])
    registry.load()
    registry.bind_node_sessions(NodeSessions(registry))
    safety = DualModelCommandSafety(registry)
    registry.bind_command_safety(safety)
    transport = await registry.transport("arm")
    async def read_batch(ids):
        # Isolated external-I/O cohort: all values have one synthetic receipt instant.
        readings = tuple([await transport.read(cid) for cid in ids])
        stamp = time.monotonic()
        return BatchReading("arm", tuple(replace(r, monotonic_at=stamp) for r in readings))
    async def write_batch(commands):
        outcomes = tuple([await transport.write(cid, value) for cid, value in commands])
        return BatchWriteOutcome(all(o.ok for o in outcomes), outcomes, "committed")
    monkeypatch.setattr(transport, "read_batch", read_batch, raising=False)
    monkeypatch.setattr(transport, "write_batch", write_batch, raising=False)
    try:
        yield registry, safety, transport
    finally:
        await registry.node_sessions.shutdown()
        await registry.close_all()


async def publish(registry: HardwareRegistry, safety: DualModelCommandSafety, value: CommandSafetySnapshot):
    """Commit isolated synthetic plumbing only, not an admission or live activation."""
    store = registry.calibration_generations
    return await registry.node_sessions.run_reserved(
        ("arm", "consumer:precision"), owner_id="isolated-test",
        operation=lambda: store.commit_generation(workcell_id="cell", request_id="test",
            expected_generation=None, dependency_digest=value.corrected.dependency_digest,
            model=value.to_dict(), accepted_blocks={"synthetic": {}}, causal={"qualification": "software only"}),
        on_settle=lambda generation: safety.publish_reserved(value, generation),
    )


def test_snapshot_roundtrip_and_immutable_input() -> None:
    value = snapshot(context())
    assert CommandSafetySnapshot.from_mapping(value.to_dict()) == value
    wire = value.to_dict()
    wire["joint_ids"].append("other")
    assert value.joint_ids == ("joint",)
    with pytest.raises(ValueError, match="scope"):
        CommandSafetySnapshot.from_mapping({**value.to_dict(), "consumer_scope": ["native_cartesian"]})


@pytest.mark.parametrize("changes", [
    {"nominal": None}, {"corrected": {}}, {"geometry": None}, {"generation": True},
    {"speed_channel": ""}, {"speed_pct_max": 11}, {"joint_rate_bounds_rad_s": (0,)},
    {"endpoint_rate_bound_mm_s": float("nan")}, {"rate_provenance_digest": ""},
])
def test_snapshot_rejects_malformed_without_attribute_errors(changes: Any) -> None:
    with pytest.raises(ValueError):
        snapshot(context(), **changes)


@pytest.mark.parametrize("changes", [
    {"sensing_mm": 0.01}, {"timing_mm": 0.01},
    {"minimum_clearance_mm": 29}, {"boundary_margin_mm": 19},
])
def test_snapshot_rejects_understated_uncertainty_or_clearance(changes: Any) -> None:
    value = snapshot(context())
    with pytest.raises(ValueError):
        replace(value, geometry=replace(value.geometry, **changes))


async def test_nominal_behavior_without_projection_and_single_owner_binding(owner) -> None:
    registry, safety, transport = owner
    assert isinstance(safety, CommandSafety)
    assert control_safety_refusal(registry, "arm") is None
    assert (await NodeWriteGate(registry).write("arm", "x", 101)).ok
    assert transport.write_log == (("x", 101),)
    registry.bind_command_safety(safety)
    with pytest.raises(ValueError, match="competing"):
        registry.bind_command_safety(DualModelCommandSafety(registry))
    with pytest.raises(ValueError, match="runtime-checkable"):
        registry.bind_command_safety(object())


async def test_real_shared_db_reserved_projection_blocks_cartesian_and_invalidation(owner) -> None:
    registry, safety, transport = owner
    value = snapshot(registry.context("arm"))
    generation = await publish(registry, safety, value)
    assert await asyncio.to_thread(registry.calibration_generations.active, "cell") == generation
    with pytest.raises(ValueError, match="reservation"):
        safety.publish_reserved(value, generation)
    assert "excludes native" in control_safety_refusal(registry, "arm")
    outcome = await NodeWriteGate(registry).write("arm", "x", 101)
    assert not outcome.ok and outcome.side_effect_state == "none"
    assert "Cartesian" in outcome.error and transport.write_log == ()
    token = registry.node_sessions.reserve(("arm",), owner_id="test-invalidation")
    safety.invalidate_reserved("arm")
    registry.node_sessions.release_reservation(token)
    outcome = await NodeWriteGate(registry).write("arm", "joint", 1)
    assert not outcome.ok and "invalidated" in outcome.error
    assert transport.write_log == ()


async def test_single_and_batch_raw_paths_borrow_lock_without_deadlock(owner) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    gate = NodeWriteGate(registry)
    assert (await asyncio.wait_for(gate.write("arm", "joint", 1), 1)).ok
    # The external mock does not run firmware FK; reset its raw value for another cohort.
    transport.set_value("joint", 0)
    ok, outcomes = await asyncio.wait_for(gate.write_batch("arm", (("speed", 5), ("joint", 1))), 1)
    assert ok and all(o.ok for o in outcomes)
    assert transport.write_log == (("joint", 1), ("speed", 5), ("joint", 1))


@pytest.mark.parametrize("changes", [
    {"joint": float("nan")}, {"enabled": False}, {"arrived": False}, {"mode": "cartesian"},
    {"status": {}}, {"status": {"arm_status": False, "joint_communication_errors": []}},
    {"status": {"arm_status": 0, "joint_communication_errors": [1]}}, {"x": 120},
    {"speed": 11}, {"speed": True},
])
async def test_unhealthy_raw_or_nominal_feedback_never_dispatches(owner, changes: Any) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    for cid, value in changes.items():
        transport.set_value(cid, value)
    outcome = await NodeWriteGate(registry).write("arm", "joint", 1)
    assert not outcome.ok and outcome.side_effect_state == "none" and transport.write_log == ()
    token = registry.node_sessions.reserve(("arm",), owner_id="no-uncertain-effect")
    registry.node_sessions.release_reservation(token)


@pytest.mark.parametrize("commands", [(("speed", 11), ("joint", 1)), (("joint", 1), ("speed", 5)),
                                         (("joint", 1), ("joint", 2)), (("speed", 5), ("x", 101))])
async def test_invalid_batch_is_refused_without_partial_effect(owner, commands: Any) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    ok, outcomes = await NodeWriteGate(registry).write_batch("arm", commands)
    assert not ok and all(not o.ok and o.side_effect_state == "none" for o in outcomes)
    assert transport.write_log == ()


async def test_corrected_intermediate_geometry_refuses_with_safe_nominal_endpoint(owner) -> None:
    registry, safety, transport = owner
    value = snapshot(registry.context("arm"))
    # The corrected one-degree offset moves the link toward this close bound.
    value = replace(value, geometry=replace(value.geometry, keepouts=(WorkspaceBox(40, 50, 31, 32, -1, 1),)))
    await publish(registry, safety, value)
    outcome = await NodeWriteGate(registry).write("arm", "joint", 1)
    assert not outcome.ok and "clearance" in outcome.error and transport.write_log == ()


async def test_slow_sweep_expired_feedback_never_dispatches(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    from leapflow.hardware.calibration import command_safety
    original = command_safety.check_dual_sweep
    def slow(*args):
        time.sleep(0.12)
        return original(*args)
    monkeypatch.setattr(command_safety, "check_dual_sweep", slow)
    outcome = await NodeWriteGate(registry).write("arm", "joint", 1)
    assert not outcome.ok and "expired" in outcome.error and transport.write_log == ()


async def test_repeated_cancel_retains_dispatch_until_geometry_thread_settles(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    from leapflow.hardware.calibration import command_safety
    entered, release = threading.Event(), threading.Event()
    original = command_safety.check_dual_sweep
    def waiting(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(command_safety, "check_dual_sweep", waiting)
    task = asyncio.create_task(NodeWriteGate(registry).write("arm", "joint", 1))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(ValueError, match="unsettled"):
            registry.node_sessions.reserve(("arm",), owner_id="concurrent-activation")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.write_log == ()
    token = registry.node_sessions.reserve(("arm",), owner_id="settled")
    registry.node_sessions.release_reservation(token)


class Hold:
    policy_id = "isolated-hold"
    def reset(self) -> None:
        pass
    def compute(self, state: ControlState) -> ControlCommand:
        return ControlCommand({})


async def test_direct_control_start_and_each_tick_refuse_corrected_scope(owner) -> None:
    registry, safety, _ = owner
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(frequency_hz=10))
    await publish(registry, safety, snapshot(registry.context("arm")))
    with pytest.raises(RuntimeError, match="excludes native"):
        bus.start("arm", Hold())
    assert bus._thread is None
    bus._device_id = "arm"
    assert not bus._check_safety(ControlState(time.monotonic(), {}, {}, {}, 0, 0.1), ControlCommand({}), check_device_policy=False)
    with pytest.raises(RuntimeError, match="excludes native"):
        registry.start_control_loop("arm", Hold())


@pytest.mark.parametrize("race", ["already-active", "activate-during-feedback", "reserve-during-feedback", "cancel-feedback"])
async def test_control_open_refusal_preserves_real_signed_nonce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race: str) -> None:
    ctx = context()
    joint = replace(ctx.channels[0], envelope=replace(ctx.channels[0].envelope, requires_interlocks=("ready",)))
    ctx = replace(ctx, channels=(joint, *ctx.channels[1:]), interlocks=(Interlock("ready", "enabled"),))
    registry = HardwareRegistry(HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "instrument.duckdb")), providers=[Provider(ctx)])
    registry.load()
    safety = DualModelCommandSafety(registry)
    registry.bind_command_safety(safety)
    private, _ = generate_keypair()
    issuer = PermitIssuer(private)
    verifier = PermitVerifier({issuer.key_id: issuer.public_key}, node_id="node")
    server = LhpNodeServer(registry, node_id="node", permit_verifier=verifier)
    params = {"device_id": "arm", "mode": "pid", "frequency_hz": 10}
    permit = issuer.issue(audience="node", session_id="client-a", workspace="/workspace-a",
                          approval_id="isolated-test", digest=request_digest("control.open", params))
    entered, release = asyncio.Event(), asyncio.Event()
    token = None
    try:
        transport = await registry.transport("arm")
        read = transport.read
        async def waiting(cid):
            if cid == "enabled":
                entered.set()
                await release.wait()
            return await read(cid)
        monkeypatch.setattr(transport, "read", waiting)
        if race == "already-active":
            await publish(registry, safety, snapshot(registry.context("arm")))
        task = asyncio.create_task(server.handle_request({"id": 1, "method": "control.open", "params": {**params, "permit": permit.to_dict()}}))
        if race != "already-active":
            await asyncio.wait_for(entered.wait(), 1)
            if race == "activate-during-feedback":
                await publish(registry, safety, snapshot(registry.context("arm")))
            elif race == "reserve-during-feedback":
                token = registry.node_sessions.reserve(("arm",), owner_id="concurrent-commit")
            else:
                task.cancel()
        release.set()
        if race == "cancel-feedback":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result["result"]["failure_code"] == ("device_busy" if race == "reserve-during-feedback" else "safety_policy_denied")
        assert verifier.verify(permit, expected_digest=request_digest("control.open", params)) is None
        assert registry.control_loop_status("arm") is None and transport.write_log == ()
    finally:
        release.set()
        if token is not None:
            registry.node_sessions.release_reservation(token)
        await registry.node_sessions.shutdown()
        await registry.close_all()


async def test_speed_batch_cannot_fix_unsafe_prior_speed_before_joint_write(owner) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    transport.set_value("speed", 50)
    ok, outcomes = await NodeWriteGate(registry).write_batch("arm", (("speed", 5), ("joint", 1)))
    assert not ok and all(o.side_effect_state == "none" for o in outcomes)
    assert transport.write_log == ()
    # A separately governed speed-only command has no joint actuation.
    assert (await NodeWriteGate(registry).write("arm", "speed", 5)).ok
    assert (await NodeWriteGate(registry).write("arm", "joint", 1)).ok


@pytest.mark.parametrize("defect", ["speed-stale", "speed-unit", "speed-incomplete", "raw-stale", "raw-incomplete", "untyped-sequence"])
async def test_bad_coherent_receipts_never_dispatch(owner, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    registry, safety, transport = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    read_batch = transport.read_batch
    async def defective(ids):
        batch = await read_batch(ids)
        if defect == "speed-incomplete" and ids == ("speed",):
            return replace(batch, readings=())
        if defect == "raw-incomplete" and ids != ("speed",):
            return replace(batch, readings=batch.readings[:-1])
        readings = []
        for reading in batch.readings:
            if defect == "speed-stale" and reading.channel_id == "speed" or defect == "raw-stale" and reading.channel_id == "joint":
                reading = replace(reading, monotonic_at=time.monotonic() - 1)
            if defect == "speed-unit" and reading.channel_id == "speed":
                reading = replace(reading, unit="fraction")
            if defect == "untyped-sequence":
                reading = replace(reading, sequence=True)
            readings.append(reading)
        return replace(batch, readings=tuple(readings))
    monkeypatch.setattr(transport, "read_batch", defective)
    outcome = await NodeWriteGate(registry).write("arm", "joint", 1)
    assert not outcome.ok and outcome.side_effect_state == "none" and transport.write_log == ()


async def test_publication_failure_after_commit_latches_owner_resources(owner) -> None:
    registry, safety, _ = owner
    value = snapshot(registry.context("arm"), declaration_digest="changed")
    with pytest.raises(ValueError, match="declaration changed"):
        await publish(registry, safety, value)
    assert await asyncio.to_thread(registry.calibration_generations.active, "cell") is not None
    assert registry.node_sessions.busy_device(("arm",)) == "arm"
    assert registry.node_sessions.busy_device(("consumer:precision",)) == "consumer:precision"
    assert control_safety_refusal(registry, "arm") is None  # no uncommitted projection
    assert not (await NodeWriteGate(registry).write("arm", "joint", 1)).ok


async def test_nominal_real_control_thread_blocks_between_run_reservations(owner) -> None:
    registry, _, _ = owner
    registry.start_control_loop("arm", Hold(), config=ControlBusConfig(frequency_hz=10))
    try:
        assert registry.control_loop_status("arm")["running"]
        with pytest.raises(ValueError, match="busy"):
            registry.node_sessions.reserve(("arm",), owner_id="cannot-activate")
    finally:
        registry.stop_control_loop("arm")
    token = registry.node_sessions.reserve(("arm",), owner_id="between-runs")
    registry.node_sessions.release_reservation(token)


async def test_standalone_control_claim_blocks_all_competing_owner_paths(owner) -> None:
    registry, _, transport = owner
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(frequency_hz=20, priority="normal"))
    await bus.async_start("arm", Hold())
    try:
        assert registry.control_loop_status("arm") is None  # independently constructed
        assert registry.node_sessions.busy_device(("arm",)) == "arm"
        with pytest.raises(ValueError, match="busy"):
            registry.node_sessions.reserve(("arm",), owner_id="competing-activation")
        with pytest.raises(RuntimeError, match="busy"):
            HighFrequencyControlBus(registry).start("arm", Hold())
        verdict = await NodeWriteGate(registry).write("arm", "joint", 1)
        assert not verdict.ok and verdict.side_effect_state == "none"
        consumed = []
        result = await registry.node_sessions.open({"device_id": "arm"}, connection_id="other-workspace",
            authorize=lambda: consumed.append(True))
        assert result["failure_code"] == "device_busy" and consumed == []
        assert transport.write_log == ()
    finally:
        await bus.async_stop()
    assert not bus.is_active
    assert registry.node_sessions.busy_device(("arm",)) == ""


@pytest.mark.parametrize("cancel", [False, True])
async def test_async_transport_open_claim_and_repeated_cancel_settlement(owner, monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    registry, _, old_transport = owner
    await registry.drop_transport("arm")
    entered, release = asyncio.Event(), asyncio.Event()
    original = type(old_transport).open
    async def delayed_open(self, ctx):
        entered.set()
        await release.wait()
        return await original(self, ctx)
    monkeypatch.setattr(type(old_transport), "open", delayed_open)
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(frequency_hz=20, priority="normal"))
    task = asyncio.create_task(bus.async_start("arm", Hold()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert bus.is_active and not bus.is_running
        with pytest.raises(ValueError, match="busy"):
            registry.node_sessions.reserve(("arm",), owner_id="during-open")
        with pytest.raises(RuntimeError, match="already"):
            bus.start("arm", Hold())
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            bus.stop(timeout_s=0.001)
            assert bus.is_active  # no RT thread yet does not mean startup settled
    finally:
        release.set()
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not bus.is_active
        assert registry.get_open_transport("arm").write_log == ()
        # A later explicitly requested start is possible; no automatic retry.
        await bus.async_start("arm", Hold())
    else:
        await task
    try:
        assert bus.is_running and registry.node_sessions.busy_device(("arm",))
    finally:
        await bus.async_stop()
    assert not bus.is_active and registry.node_sessions.busy_device(("arm",)) == ""


async def test_failed_transport_start_does_not_report_running_or_leak_claim(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, _, transport = owner
    await registry.drop_transport("arm")
    bus = HighFrequencyControlBus(registry)
    with pytest.raises(RuntimeError, match="failed or stopped"):
        await asyncio.to_thread(bus.start, "arm", Hold())
    assert not bus.is_active and not bus.is_running
    assert registry.node_sessions.busy_device(("arm",)) == ""
    async def failed_open(self, ctx):
        raise TransportError("isolated open failure")
    monkeypatch.setattr(type(transport), "open", failed_open)
    with pytest.raises(TransportError):
        await bus.async_start("arm", Hold())
    assert not bus.is_active and registry.node_sessions.busy_device(("arm",)) == ""


async def test_timed_join_and_cancelled_stop_retain_claim_until_halt_settlement(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, _, transport = owner
    entered, release = threading.Event(), threading.Event()
    original = transport.halt
    async def delayed_halt():
        entered.set()
        assert release.wait(2)
        return await original()
    monkeypatch.setattr(transport, "halt", delayed_halt)
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(frequency_hz=20, priority="normal"))
    await bus.async_start("arm", Hold())
    task = None
    try:
        await asyncio.to_thread(bus.stop, 0.001)
        assert await asyncio.to_thread(entered.wait, 1)
        assert bus.is_active and bus._thread is not None
        with pytest.raises(ValueError, match="busy"):
            registry.node_sessions.reserve(("arm",), owner_id="during-halt")
        task = asyncio.create_task(bus.async_stop())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert registry.node_sessions.busy_device(("arm",))
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not bus.is_active and registry.node_sessions.busy_device(("arm",)) == ""


async def test_shutdown_refuses_release_of_pending_open_then_stops_standalone_bus(owner, monkeypatch: pytest.MonkeyPatch) -> None:
    registry, _, transport = owner
    await registry.drop_transport("arm")
    entered, release = asyncio.Event(), asyncio.Event()
    original = type(transport).open
    async def delayed_open(self, ctx):
        entered.set()
        await release.wait()
        return await original(self, ctx)
    monkeypatch.setattr(type(transport), "open", delayed_open)
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(priority="normal"))
    task = asyncio.create_task(bus.async_start("arm", Hold()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(RuntimeError, match="has not settled"):
            await registry.close_all()
        assert bus.is_active and registry.node_sessions.busy_device(("arm",))
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="stopped before dispatch"):
        await task
    assert not bus.is_active and registry.get_open_transport("arm").write_log == ()
    await registry.close_all()
    with pytest.raises(RuntimeError):
        registry.get_open_transport("arm")


@pytest.mark.parametrize("failure", ["exception", "unsupported"])
async def test_failed_control_halt_latches_uncertain_shared_owner_state(owner, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    registry, _, transport = owner
    original = transport.halt
    async def failed_halt():
        if failure == "exception":
            raise TransportError("isolated halt failure")
        return replace(await original(), halt_supported=False)
    monkeypatch.setattr(transport, "halt", failed_halt)
    bus = HighFrequencyControlBus(registry, config=ControlBusConfig(frequency_hz=20, priority="normal"))
    await bus.async_start("arm", Hold())
    await bus.async_stop()
    assert not bus.is_active
    assert registry.node_sessions.busy_device(("arm",))
    assert not (await NodeWriteGate(registry).write("arm", "joint", 1)).ok
    with pytest.raises(ValueError, match="unsettled"):
        registry.node_sessions.reserve(("arm",), owner_id="unverified-stop")


async def test_control_open_transfers_start_reservation_to_real_thread_and_lease(owner) -> None:
    registry, _, transport = owner
    authority_checks = []
    def authorize():
        assert registry.node_sessions.reserved_device("arm")
        authority_checks.append(True)
    result = await registry.node_sessions.open({"device_id": "arm", "frequency_hz": 20, "lease_s": 1},
        connection_id="client-a-workspace-a", authorize=authorize)
    assert result["ok"] and authority_checks == [True]
    assert registry.control_loop_status("arm")["running"]
    assert registry.node_sessions.holds_device("arm")
    assert not registry.node_sessions.reserved_device("arm")
    assert registry.node_sessions.unsettled_device("arm")
    await registry.node_sessions.close({"control_id": result["control_id"]}, connection_id="client-a-workspace-a")
    assert not registry.node_sessions.busy_device(("arm",)) and transport.write_log == ()


@pytest.mark.parametrize("verdict", ["active", "invalidated", "corrupt", "unknown_scope"])
async def test_real_registry_restart_stops_use_without_rehydrating_qualification(
    owner, monkeypatch: pytest.MonkeyPatch, verdict: str,
) -> None:
    registry, safety, _ = owner
    value = snapshot(registry.context("arm"))
    await publish(registry, safety, value)
    store = registry.calibration_generations
    if verdict == "invalidated":
        await registry.node_sessions.run_reserved(("arm",), owner_id="stop-use",
            operation=lambda: store.invalidate(workcell_id="cell", request_id="stop", expected_generation=1,
                dependency_digest="changed", causal={}),
            on_settle=lambda _: safety.invalidate_reserved("arm"))
    elif verdict == "corrupt":
        store._holder.connection.execute("DELETE FROM calib_pointer_journal")
    elif verdict == "unknown_scope":
        # Unknown internally consistent data is still not an actuator identity.
        await asyncio.to_thread(store.commit_generation, workcell_id="other", request_id="unknown",
            expected_generation=None, dependency_digest="test", model={"synthetic": True},
            accepted_blocks={"test": {}}, causal={})
    settings = registry.settings
    await registry.close_all()
    other = replace(context(), device_id="other_arm")
    restarted = HardwareRegistry(settings, providers=[Provider(context()), Provider(other)])
    await asyncio.to_thread(restarted.load)
    restarted.bind_node_sessions(NodeSessions(restarted))
    try:
        assert restarted.command_safety is None, "restart may restore refusal, never physical qualification"
        reason = control_safety_refusal(restarted, "arm")
        assert reason and ("invalidated" in reason if verdict == "invalidated" else "restart" in reason)
        monkeypatch.setattr(restarted.calibration_generations, "states", lambda: pytest.fail("no DB on command ticks"))
        outcome = await NodeWriteGate(restarted).write("arm", "joint", 1)
        assert not outcome.ok and outcome.side_effect_state == "none"
        ok, outcomes = await NodeWriteGate(restarted).write_batch("arm", (("joint", 1),))
        assert not ok and all(o.side_effect_state == "none" for o in outcomes)
        bus = HighFrequencyControlBus(restarted)
        with pytest.raises(RuntimeError):
            bus.start("arm", Hold())
        consumed = []
        refused = await restarted.node_sessions.open({"device_id": "arm"}, connection_id="client-b-workspace-b",
            authorize=lambda: consumed.append(True))
        assert not refused["ok"] and consumed == []
        assert (await restarted.read("arm", "joint")).value == 0
        assert (await restarted.halt("arm")).halt_supported
        unaffected = await NodeWriteGate(restarted).write("other_arm", "joint", 1)
        assert unaffected.ok == (verdict in ("active", "invalidated"))
        assert restarted.get_open_transport("arm").write_log == ()
    finally:
        await restarted.close_all()


@pytest.mark.parametrize("method", ["write", "write_batch", "control.open"])
async def test_restart_refusal_before_signed_permit_consumption(owner, method: str) -> None:
    registry, safety, _ = owner
    await publish(registry, safety, snapshot(registry.context("arm")))
    settings = registry.settings
    await registry.close_all()
    restarted = HardwareRegistry(settings, providers=[Provider(context())])
    await asyncio.to_thread(restarted.load)
    issuer = PermitIssuer(generate_keypair()[0])
    verifier = PermitVerifier({issuer.key_id: issuer.public_key}, node_id="node")
    server = LhpNodeServer(restarted, node_id="node", permit_verifier=verifier)
    transport = await restarted.transport("arm")
    params = {"device_id": "arm", **{
        "write": {"channel_id": "joint", "value": 1},
        "write_batch": {"commands": [["joint", 1]]},
        "control.open": {"frequency_hz": 10},
    }[method]}
    permit = issuer.issue(audience="node", session_id="client-a", workspace="/workspace-a",
        approval_id="isolated-test", digest=request_digest(method, params))
    try:
        reply = await server.handle_request({"id": 1, "method": method, "params": {**params, "permit": permit.to_dict()}})
        assert reply["ok"] and not reply["result"]["ok"]
        assert verifier.verify(permit, expected_digest=request_digest(method, params)) is None
        assert transport.write_log == ()
    finally:
        await server.stop()
        await restarted.close_all()


async def test_invalidated_restart_cannot_republish_old_generation(owner) -> None:
    registry, safety, _ = owner
    value = snapshot(registry.context("arm"))
    generation = await publish(registry, safety, value)
    store = registry.calibration_generations
    await registry.node_sessions.run_reserved(("arm",), owner_id="stop-use",
        operation=lambda: store.invalidate(workcell_id="cell", request_id="stop", expected_generation=1,
            dependency_digest="changed", causal={}),
        on_settle=lambda _: safety.invalidate_reserved("arm"))
    registry.load()
    token = registry.node_sessions.reserve(("arm",), owner_id="cannot-resurrect")
    try:
        with pytest.raises(ValueError, match="invalidated consumer"):
            safety.publish_reserved(value, generation)
        assert "invalidated" in control_safety_refusal(registry, "arm")
    finally:
        registry.node_sessions.release_reservation(token)

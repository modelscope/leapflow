# Copyright (c) Alibaba, Inc. and its affiliates.
"""The host's own confirmation of a node's declaration.

A node declares what its hardware can do *and* whether a human reviewed that
declaration.  The second half cannot come from the node: a machine able to
rewrite its declaration is able to rewrite its claim that someone read it.  What
is pinned here is the host-side rule -- a node's claim survives registration only
when the host holds a confirmation of the declaration the node is sending now --
and the command that records one.
"""

from __future__ import annotations

import pytest

from leapflow.hardware.attestation import (
    declaration_digest,
    read_verified,
    store_verification,
)
from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    HC_VERSION,
    TransportRef,
)
from leapflow.hardware.fleet import FleetError, FleetManager, FleetNode
from leapflow.hardware.fleet_config import confirm_node_device
from leapflow.hardware.lhp.server import LhpNodeServer
from leapflow.hardware.registry import (
    HardwareRegistry,
    HardwareSettings,
    UnverifiedContextPolicy,
)

NODE_ID = "jetson01"
DEVICE_ID = "arm"
HOST_DEVICE_ID = f"{NODE_ID}__{DEVICE_ID}"


class _MutableProvider:
    """A provider whose declarations can change, as a rescan or an edit would."""

    kind = "attestation_test"

    def __init__(self, contexts: tuple[HardwareContext, ...] = ()) -> None:
        self.contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self.contexts


def _declared_device(*, extra: bool = False) -> HardwareContext:
    """One node-side device that claims a human reviewed its declaration."""
    channels = [_channel("setpoint")]
    if extra:
        channels.append(_channel("accessory"))
    return HardwareContext(
        device_id=DEVICE_ID,
        hc_version=HC_VERSION,
        display_name="Node arm",
        transport=TransportRef(kind="mock", config={"values": {"setpoint": 10.0}}),
        halt_supported=True,
        channels=tuple(channels),
        provenance=ContextProvenance(verified_by="node-operator"),
    )


def _channel(channel_id: str) -> Channel:
    return Channel(
        channel_id=channel_id,
        direction=Direction.READWRITE.value,
        quantity="generic.value",
        unit="unit",
        effect=HardwareEffect.CONFIGURE.value,
        envelope=Envelope(declared=True, min_value=0.0, max_value=100.0, reversible=True),
    )


def _registry(*contexts: HardwareContext, verified_path: str = "") -> HardwareRegistry:
    """A real registry: declarations reach the wire through ``to_dict()``."""
    settings = HardwareSettings(
        enabled=True,
        unverified_context_policy=UnverifiedContextPolicy.ALLOW,
        require_describe_before_write=False,
        trust_skip_enabled=True,
        stream_enabled=False,
        persist_readings=False,
        verified_path=verified_path,
    )
    registry = HardwareRegistry(settings, providers=(_MutableProvider(contexts),))
    registry.load()
    return registry


async def _start_node(
    node_registry: HardwareRegistry,
) -> tuple[LhpNodeServer, str]:
    server = LhpNodeServer(node_registry, node_id=NODE_ID, write_mode="trusted_v1")
    host, port = await server.start(host="127.0.0.1", port=0)
    return server, f"tcp://{host}:{port}"


async def _registered(
    endpoint: str, *, verified_path: str
) -> tuple[FleetManager, HardwareRegistry]:
    host = _registry(verified_path=verified_path)
    manager = FleetManager(host)
    await manager.register_node(FleetNode(NODE_ID, "Jetson", endpoint))
    return manager, host


async def test_a_nodes_own_claim_does_not_reach_the_host(tmp_path) -> None:
    node_registry = _registry(_declared_device())
    server, endpoint = await _start_node(node_registry)
    manager = None
    try:
        manager, host = await _registered(endpoint, verified_path=str(tmp_path / "verified.json"))
        context = host.context(HOST_DEVICE_ID)
        assert context is not None
        assert context.provenance.verified_by == ""
        assert context.provenance.source == f"remote:{NODE_ID}"
    finally:
        if manager is not None:
            await manager.close()
        await server.stop()


async def test_the_hosts_confirmation_is_what_admits_the_claim(tmp_path) -> None:
    verified_path = str(tmp_path / "verified.json")
    store_verification(
        verified_path,
        HOST_DEVICE_ID,
        verifier="jason",
        digest=declaration_digest(_declared_device()),
    )
    node_registry = _registry(_declared_device())
    server, endpoint = await _start_node(node_registry)
    manager = None
    try:
        manager, host = await _registered(endpoint, verified_path=verified_path)
        context = host.context(HOST_DEVICE_ID)
        assert context is not None
        assert context.provenance.verified_by == "jason"
        # The host's name replaces the node's: the claim that survives is the one
        # the host can answer for, not the one the node made about itself.
        assert context.provenance.source != f"remote:{NODE_ID}"
    finally:
        if manager is not None:
            await manager.close()
        await server.stop()


async def test_a_confirmation_of_another_declaration_does_not_transfer(tmp_path) -> None:
    verified_path = str(tmp_path / "verified.json")
    store_verification(
        verified_path,
        HOST_DEVICE_ID,
        verifier="jason",
        digest=declaration_digest(_declared_device()),
    )
    provider = _MutableProvider((_declared_device(),))
    node_registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            unverified_context_policy=UnverifiedContextPolicy.ALLOW,
            require_describe_before_write=False,
            stream_enabled=False,
            persist_readings=False,
        ),
        providers=(provider,),
    )
    node_registry.load()
    server, endpoint = await _start_node(node_registry)
    manager = None
    try:
        manager, host = await _registered(endpoint, verified_path=verified_path)
        context = host.context(HOST_DEVICE_ID)
        assert context.provenance.verified_by == "jason"
        # The moment of review travels with the verifier, or an operator reading
        # `hw describe` cannot tell a fresh review from a stale claim.
        assert context.provenance.verified_at > 0

        # The node redeclares, adding a channel: the reviewed declaration is gone.
        provider.contexts = (_declared_device(extra=True),)
        await node_registry.reconcile()
        await manager.register_node(FleetNode(NODE_ID, "Jetson", endpoint))

        assert host.context(HOST_DEVICE_ID).provenance.verified_by == ""
        assert host.context(HOST_DEVICE_ID).provenance.verified_at == 0.0
    finally:
        if manager is not None:
            await manager.close()
        await server.stop()


async def test_confirm_node_device_records_the_declared_digest(tmp_path) -> None:
    verified_path = str(tmp_path / "verified.json")
    node_registry = _registry(_declared_device())
    server, endpoint = await _start_node(node_registry)
    try:
        entry = await confirm_node_device(
            FleetNode(NODE_ID, "Jetson", endpoint),
            DEVICE_ID,
            verifier="jason",
            verified_path=verified_path,
        )
    finally:
        await server.stop()

    stored = read_verified(verified_path)[HOST_DEVICE_ID]
    assert set(stored) == {"verifier", "declaration_digest", "verified_at"}
    assert stored["verifier"] == "jason"
    assert stored["declaration_digest"] == entry["declaration_digest"]
    # The digest is over the declaration in the node's own id space, so an
    # operator can compare it with what the node prints for the same device.
    assert stored["declaration_digest"] == declaration_digest(_declared_device())


async def test_confirming_a_device_the_node_does_not_declare_is_refused(tmp_path) -> None:
    node_registry = _registry(_declared_device())
    server, endpoint = await _start_node(node_registry)
    try:
        with pytest.raises(FleetError) as caught:
            await confirm_node_device(
                FleetNode(NODE_ID, "Jetson", endpoint),
                "no_such_device",
                verifier="jason",
                verified_path=str(tmp_path / "verified.json"),
            )
    finally:
        await server.stop()
    assert caught.value.failure_code == "fleet_unknown_device"


def test_recording_a_confirmation_keeps_the_other_devices(tmp_path) -> None:
    verified_path = tmp_path / "verified.json"
    store_verification(verified_path, "bench_rig", verifier="alice", digest="a" * 64)
    store_verification(verified_path, HOST_DEVICE_ID, verifier="jason", digest="b" * 64)

    entries = read_verified(verified_path)
    assert entries["bench_rig"]["verifier"] == "alice"
    assert entries[HOST_DEVICE_ID]["verifier"] == "jason"

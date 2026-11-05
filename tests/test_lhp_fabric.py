# Copyright (c) Alibaba, Inc. and its affiliates.
"""LHP fabric suite: wire codec, node dispatch, listeners, and the §6 error table.

The transport-contract suite covers the LHP transport against an in-process node
(the ``[lhp]`` cases). This suite covers what is specific to the fabric itself:

- the frame codec and the node's request validation (``handle_request``);
- the TCP/Unix listeners: the hello-first gate, token enforcement, and the
  concurrent per-request dispatch after authentication;
- the frozen write-error classification end to end (spec §6): which error codes
  degrade a write to an ``unknown`` outcome and which raise instead;
- identity localisation: node-local device ids relabelled under the host-side
  namespace, and the contract-violation guards on both sides of the wire.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from typing import Any, Mapping

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    HardwareContext,
    PrivacyTier,
    Representation,
    TransportRef,
    TrustConfig,
)
from leapflow.hardware.fleet import FleetError, FleetManager, FleetNode
from leapflow.hardware.registry import (
    HardwareRegistry,
    HardwareSettings,
    UnverifiedContextPolicy,
)
from leapflow.hardware.lhp import (
    ERR_BAD_REQUEST,
    ERR_CONNECTION_CLOSED,
    ERR_INTERNAL_ERROR,
    ERR_PROTOCOL_MISMATCH,
    ERR_TIMEOUT,
    ERR_TRANSPORT_ERROR,
    ERR_UNAUTHORIZED,
    ERR_UNKNOWN_CHANNEL,
    ERR_UNKNOWN_DEVICE,
    FAILURE_BATCH_ABORTED,
    FAILURE_NOT_WRITABLE,
    FAILURE_SAFETY_DENIED,
    LHP_CAPABILITIES,
    LHP_PROTOCOL_VERSION,
    LHP_V1_CAPABILITIES,
    MAX_FRAME_BYTES,
    DirectLhpClient,
    LhpNodeServer,
    LhpProtocolError,
    LhpRemoteError,
    SocketLhpClient,
    decode_frame,
    encode_frame,
    error_response,
    is_valid_device_id,
    make_request,
    ok_response,
)
from leapflow.hardware.lhp.protocol import CAP_FRAME_FRESHNESS_V1, ERR_UNSUPPORTED_CAPABILITY
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_UNKNOWN,
    FrameReading,
    FrameTransport,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.transports.lhp import build_transport
from tests._harness.hardware_stubs import (
    StubNodeRegistry,
    lhp_node_server,
    stub_node_context,
)


_FRAME_JPEG = (
    b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xd9"
)


class _FrameNodeTransport:
    """Node-local frame source with observable profile and release state."""

    kind = "frame_stub"

    def __init__(self) -> None:
        self.requests: list[tuple[str, int, int, float]] = []
        self.closes = 0

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        self.requests.append((channel_id, max_width, quality, fps))
        return FrameReading(
            device_id="camera",
            channel_id=channel_id,
            data=_FRAME_JPEG,
            width=4,
            height=2,
            sequence=len(self.requests),
        )

    async def close(self) -> TransportStatus:
        self.closes += 1
        return TransportStatus(connected=False)


class _LegacyFrameNodeTransport:
    """Frame source implementing the original signature without cadence control."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, int, int]] = []

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0
    ) -> FrameReading:
        self.requests.append((channel_id, max_width, quality))
        return FrameReading(
            device_id="camera",
            channel_id=channel_id,
            data=_FRAME_JPEG,
            width=4,
            height=2,
            sequence=len(self.requests),
        )


class _FrameNodeRegistry:
    """Small node registry exposing one privacy-gated frame channel."""

    def __init__(self, source: Any | None = None) -> None:
        self.source = source or _FrameNodeTransport()
        self.drops = 0
        self._lock = asyncio.Lock()
        self._context = HardwareContext(
            device_id="camera",
            display_name="Remote camera",
            device_class="camera",
            transport=TransportRef(kind="mock", config={}),
            channels=(
                Channel(
                    channel_id="frame",
                    direction=Direction.READ.value,
                    quantity="image_frame",
                    representation=Representation.FRAME.value,
                    media_type="image/jpeg",
                    privacy=PrivacyTier.ENVIRONMENT.value,
                ),
            ),
            provenance=ContextProvenance(verified_by="frame-test"),
        )

    def contexts(self) -> tuple[HardwareContext, ...]:
        return (self._context,)

    def context(self, device_id: str) -> HardwareContext | None:
        return self._context if device_id == "camera" else None

    def device_io(self, device_id: str) -> asyncio.Lock:
        return self._lock

    async def transport(self, device_id: str) -> Any:
        return self.source

    async def drop_transport(self, device_id: str) -> None:
        self.drops += 1
        await self.source.close()


def _frame_host_context(config: dict[str, Any]) -> HardwareContext:
    return HardwareContext(
        device_id="jetson__camera",
        display_name="Remote camera",
        device_class="camera",
        transport=TransportRef(kind="lhp", config=config),
        channels=_FrameNodeRegistry().contexts()[0].channels,
        provenance=ContextProvenance(verified_by="frame-test"),
    )


def _host_context(
    transport_config: dict[str, Any],
    *,
    device_id: str = "conformance_device",
    halt_supported: bool = True,
) -> HardwareContext:
    """The host-side declaration of the stub node's device."""
    return HardwareContext(
        device_id=device_id,
        display_name="Host-side view of the node device",
        transport=TransportRef(kind="lhp", config=transport_config),
        halt_supported=halt_supported,
        channels=stub_node_context().channels,
        provenance=ContextProvenance(verified_by="fabric-test"),
    )


class _ScriptedServer:
    """Direct-binding stand-in: hello as usual, then scripted answers.

    Drives the host-side classifier with exact frames; the node's own paths are
    covered against the real ``LhpNodeServer``.
    """

    def __init__(
        self,
        responder: Any,
        *,
        protocol: int = LHP_PROTOCOL_VERSION,
        capabilities: tuple[str, ...] = (),
    ) -> None:
        self._responder = responder
        self._protocol = protocol
        self._capabilities = capabilities

    async def handle_request(
        self, request: Mapping[str, Any], *, via: str = "direct"
    ) -> dict[str, Any]:
        if request.get("method") == "hello":
            return ok_response(
                request.get("id"),
                {
                    "protocol": self._protocol,
                    "node_id": "scripted",
                    "capabilities": list(self._capabilities),
                },
            )
        return self._responder(request)


def _erroring_server(code: str) -> _ScriptedServer:
    def respond(request: Mapping[str, Any]) -> dict[str, Any]:
        return error_response(request.get("id"), code, "scripted failure")

    return _ScriptedServer(respond)


async def _start_raw_tcp(handler: Any) -> tuple[asyncio.AbstractServer, str]:
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"tcp://127.0.0.1:{port}"


async def _silent_after_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """A raw peer that completes the handshake, then answers nothing."""
    try:
        first = await reader.readline()
        if not first:
            return
        request = decode_frame(first)
        writer.write(
            encode_frame(
                ok_response(
                    request.get("id"), {"protocol": LHP_PROTOCOL_VERSION, "node_id": "sink"}
                )
            )
        )
        await writer.drain()
        while await reader.readline():
            pass
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


# ════════════════════════════════════════════════════════════════
# Frame codec and identity charset
# ════════════════════════════════════════════════════════════════


def test_frame_codec_round_trips_unicode_payloads() -> None:
    frame = encode_frame({"id": 3, "method": "read", "params": {"channel_id": "温度"}})
    assert frame.endswith(b"\n")
    assert decode_frame(frame)["params"]["channel_id"] == "温度"


@pytest.mark.parametrize(
    "line",
    [b"[]", b"42", b'"{x}"', b"not json", b"", b"\xff\xfe", b'"a string"'],
    ids=["array", "number", "braced-string", "garbage", "empty", "bad-utf8", "json-string"],
)
def test_decode_frame_rejects_anything_that_is_not_a_json_object(line: bytes) -> None:
    with pytest.raises(LhpProtocolError):
        decode_frame(line)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("jetson__camera_0", True),
        ("conformance_device", True),
        ("Jetson", False),
        ("camera-0", False),
        ("", False),
    ],
)
def test_device_id_charset(value: str, expected: bool) -> None:
    assert is_valid_device_id(value) is expected


# ════════════════════════════════════════════════════════════════
# Node dispatch: validation, hello, the token gate, and export scope
# ════════════════════════════════════════════════════════════════


async def test_hello_advertises_identity_capabilities_and_device_count() -> None:
    client = DirectLhpClient(lhp_node_server())
    hello = await client.connect()
    assert hello["protocol"] == LHP_PROTOCOL_VERSION
    assert hello["node_id"] == "stub_node"
    assert hello["device_count"] == 1
    assert tuple(hello["capabilities"][:len(LHP_CAPABILITIES)]) == LHP_CAPABILITIES
    # control.v1 is always advertised now
    assert "control.v1" in hello["capabilities"]
    await client.close()


@pytest.mark.parametrize(
    "payload,expected_code",
    [
        (42, ERR_BAD_REQUEST),
        ({"method": ""}, ERR_BAD_REQUEST),
        ({"method": "nope"}, ERR_BAD_REQUEST),
        ({"method": "read", "params": [1]}, ERR_BAD_REQUEST),
        (
            {
                "method": "write",
                "params": {"device_id": "conformance_device", "channel_id": "setpoint"},
            },
            ERR_BAD_REQUEST,
        ),
        (
            {"method": "read", "params": {"device_id": "ghost", "channel_id": "sensor"}},
            ERR_UNKNOWN_DEVICE,
        ),
        (
            {
                "method": "read",
                "params": {"device_id": "conformance_device", "channel_id": "ghost"},
            },
            ERR_UNKNOWN_CHANNEL,
        ),
        (
            {
                "method": "write_batch",
                "params": {"device_id": "conformance_device", "commands": []},
            },
            ERR_BAD_REQUEST,
        ),
    ],
)
async def test_request_validation_is_explicit(payload: Any, expected_code: str) -> None:
    response = await lhp_node_server().handle_request(payload)
    assert response["ok"] is False
    assert response["error"]["code"] == expected_code


async def test_hello_protocol_mismatch_is_refused() -> None:
    response = await lhp_node_server().handle_request(
        {"id": 1, "method": "hello", "params": {"protocol": 999}}
    )
    assert response["error"]["code"] == ERR_PROTOCOL_MISMATCH


async def test_v1_hello_remains_compatible_without_advertising_frame_methods() -> None:
    response = await lhp_node_server().handle_request(
        {"id": 1, "method": "hello", "params": {"protocol": 1}}
    )
    assert response["ok"] is True
    assert response["result"]["protocol"] == 1
    assert tuple(response["result"]["capabilities"][:len(LHP_V1_CAPABILITIES)]) == LHP_V1_CAPABILITIES
    assert "read_frame" not in response["result"]["capabilities"]
    assert "control.v1" in response["result"]["capabilities"]


async def test_client_falls_back_to_a_v1_only_direct_node() -> None:
    server = _ScriptedServer(lambda request: ok_response(request.get("id"), {}), protocol=1)
    hello = await DirectLhpClient(server).connect()
    assert hello["protocol"] == 1


async def test_token_gate_applies_to_socket_via_but_not_direct() -> None:
    server = lhp_node_server(token="s3cret")
    wrong = {
        "id": 1,
        "method": "hello",
        "params": {"protocol": LHP_PROTOCOL_VERSION, "token": "wrong"},
    }
    refused = await server.handle_request(wrong, via="socket")
    assert refused["error"]["code"] == ERR_UNAUTHORIZED
    accepted = await server.handle_request(wrong, via="direct")
    assert accepted["ok"] is True, "the same-process trust domain skips the token gate"
    good = {
        "id": 2,
        "method": "hello",
        "params": {"protocol": LHP_PROTOCOL_VERSION, "token": "s3cret"},
    }
    assert (await server.handle_request(good, via="socket"))["ok"] is True


async def test_list_honours_the_device_id_filter() -> None:
    unfiltered = await lhp_node_server().handle_request({"id": 1, "method": "list", "params": {}})
    assert [d["device_id"] for d in unfiltered["result"]["devices"]] == ["conformance_device"]
    filtered = await lhp_node_server(device_ids=("other",)).handle_request(
        {"id": 2, "method": "list", "params": {}}
    )
    assert filtered["result"]["devices"] == []


# ════════════════════════════════════════════════════════════════
# Refusals are faithful outcomes; failures are classified (§6)
# ════════════════════════════════════════════════════════════════


async def test_a_safety_refusal_is_a_faithful_outcome_not_an_error_frame() -> None:
    config = {"server": lhp_node_server(safety_deny_reason="interlock open")}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("setpoint", 42.0)
    assert isinstance(outcome, WriteOutcome)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == FAILURE_SAFETY_DENIED
    assert "interlock" in outcome.error


async def test_a_write_to_a_read_only_channel_provably_lands_nowhere() -> None:
    config = {"server": lhp_node_server()}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("sensor", 1.0)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == FAILURE_NOT_WRITABLE


async def test_node_refuses_out_of_envelope_write_before_transport() -> None:
    registry = StubNodeRegistry()
    config = {"server": LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")}
    transport = build_transport(config)
    await transport.open(_host_context(config))

    outcome = await transport.write("setpoint", 101.0)

    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == FAILURE_SAFETY_DENIED
    assert "outside" in outcome.error
    assert registry.write_log == ()


async def test_node_aborts_batch_when_one_value_is_out_of_envelope() -> None:
    registry = StubNodeRegistry()
    config = {"server": LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")}
    transport = build_transport(config)
    await transport.open(_host_context(config))

    batch = await transport.write_batch((("setpoint", 10.0), ("setpoint", 101.0)))

    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_NONE
    assert [outcome.failure_code for outcome in batch.outcomes] == [
        FAILURE_BATCH_ABORTED,
        FAILURE_SAFETY_DENIED,
    ]
    assert registry.write_log == ()


@pytest.mark.parametrize("code", [ERR_INTERNAL_ERROR, ERR_CONNECTION_CLOSED])
async def test_unknown_effect_error_codes_degrade_a_write_to_unknown(code: str) -> None:
    """The frozen §6 rule: these codes are not proof that nothing happened."""
    config = {"server": _erroring_server(code)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("setpoint", 1.0)
    assert isinstance(outcome, WriteOutcome)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.effect_may_have_landed is True
    assert outcome.failure_code == code


@pytest.mark.parametrize(
    "code", [ERR_BAD_REQUEST, ERR_UNKNOWN_DEVICE, ERR_UNKNOWN_CHANNEL, ERR_TRANSPORT_ERROR]
)
async def test_could_not_attempt_error_codes_raise_instead_of_fabricating(code: str) -> None:
    config = {"server": _erroring_server(code)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.write("setpoint", 1.0)
    assert caught.value.failure_code == code


async def test_a_driver_explosion_mid_write_is_unknown_not_an_error_frame() -> None:
    config = {"server": lhp_node_server(explode_writes=True)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("setpoint", 1.0)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.failure_code == ERR_INTERNAL_ERROR


async def test_a_silent_node_turns_a_write_into_an_unknown_outcome() -> None:
    """A timeout is client-generated: the request was sent, the answer never came."""
    raw_server, endpoint = await _start_raw_tcp(_silent_after_hello)
    try:
        config = {"endpoint": endpoint, "timeout_s": 0.3}
        transport = build_transport(config)
        await transport.open(_host_context(config))
        outcome = await transport.write("setpoint", 1.0)
        assert outcome.ok is False
        assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
        assert outcome.failure_code == ERR_TIMEOUT
        await transport.close()
    finally:
        raw_server.close()
        await raw_server.wait_closed()


async def test_a_write_to_an_unknown_channel_is_a_could_not_attempt() -> None:
    config = {"server": lhp_node_server()}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.write("no_such_channel", 1.0)
    assert caught.value.failure_code == ERR_UNKNOWN_CHANNEL


# ════════════════════════════════════════════════════════════════
# Batch semantics
# ════════════════════════════════════════════════════════════════


async def test_a_batch_refusal_aborts_before_anything_is_attempted() -> None:
    registry = StubNodeRegistry()
    config = {"server": LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    batch = await transport.write_batch((("setpoint", 10.0), ("sensor", 1.0)))
    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_NONE
    assert [(o.failure_code, o.side_effect_state) for o in batch.outcomes] == [
        (FAILURE_BATCH_ABORTED, SIDE_EFFECT_NONE),
        (FAILURE_NOT_WRITABLE, SIDE_EFFECT_NONE),
    ]
    assert registry.write_log == (), "the abort must happen before any write is attempted"


async def test_batch_over_the_socket_merges_outcomes() -> None:
    server = lhp_node_server()
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    try:
        config = {"endpoint": f"tcp://{socket_host}:{port}", "timeout_s": 2.0}
        transport = build_transport(config)
        await transport.open(_host_context(config))
        batch = await transport.write_batch((("setpoint", 1.0), ("setpoint", 2.0)))
        assert batch.ok is True
        assert batch.side_effect_state != SIDE_EFFECT_NONE
        assert len(batch.outcomes) == 2
    finally:
        await server.stop()


async def test_a_batch_whose_fate_is_unknown_refuses_to_claim_none() -> None:
    config = {"server": _erroring_server(ERR_INTERNAL_ERROR)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    batch = await transport.write_batch((("setpoint", 1.0),))
    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert batch.outcomes == ()


async def test_a_halt_never_brings_a_device_up_just_to_put_it_down() -> None:
    """A halt acts on a live link; it never opens one.

    Opening a device runs the full bring-up, so a profile the arm could not
    report refused the stop itself and a successful open would have woken
    hardware at shutdown. A device this node never opened holds no motion to
    stop, and it must still answer as haltable rather than as a failed stop.
    """
    registry = StubNodeRegistry()
    server = LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")
    reply = await server.handle_request(
        {"id": 1, "method": "halt", "params": {"device_id": "conformance_device"}}
    )
    assert reply["ok"] is True, reply
    assert reply["result"]["halt_supported"] is True
    assert reply["result"]["connected"] is False
    assert "nothing is being driven" in reply["result"]["detail"]
    assert registry.opened is False, "a halt must never open the device"


async def test_a_halt_still_stops_a_device_whose_link_is_open() -> None:
    registry = StubNodeRegistry()
    server = LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")
    await server.handle_request(
        {"id": 1, "method": "read", "params": {"device_id": "conformance_device",
                                                 "channel_id": "sensor"}}
    )
    assert registry.opened is True
    reply = await server.handle_request(
        {"id": 2, "method": "halt", "params": {"device_id": "conformance_device"}}
    )
    assert reply["ok"] is True, reply
    assert reply["result"]["halt_supported"] is True


async def test_a_halt_that_releases_the_link_lets_the_next_read_reopen_it() -> None:
    """A stop that powers the device down must leave nothing cached.

    A RealSense halt releases its device; the closed instance then stayed cached,
    so ``open()`` never ran again and every later frame read answered "is not
    open" for the life of the node.
    """
    registry = StubNodeRegistry(halt_closes=True)
    server = LhpNodeServer(registry, node_id="stub_node", write_mode="trusted_v1")
    await server.handle_request(
        {"id": 1, "method": "read", "params": {"device_id": "conformance_device",
                                                 "channel_id": "sensor"}}
    )
    assert registry.opens == 1
    reply = await server.handle_request(
        {"id": 2, "method": "halt", "params": {"device_id": "conformance_device"}}
    )
    assert reply["ok"] is True, reply
    assert reply["result"]["connected"] is False
    assert registry.opened is False, "a released link must not stay in the cache"
    again = await server.handle_request(
        {"id": 3, "method": "read", "params": {"device_id": "conformance_device",
                                                 "channel_id": "sensor"}}
    )
    assert again["ok"] is True, again
    assert registry.opens == 2, "the next read must open a fresh link"


# ════════════════════════════════════════════════════════════════
# Listeners: TCP and Unix
# ════════════════════════════════════════════════════════════════


async def test_socket_binding_carries_every_data_plane_method() -> None:
    server = lhp_node_server(token="s3cret")
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    endpoint = f"tcp://{socket_host}:{port}"
    try:
        raw = SocketLhpClient(endpoint, token="s3cret")
        hello = await raw.connect()
        assert hello["node_id"] == "stub_node"
        devices = (await raw.request("list"))["devices"]
        assert devices[0]["device_id"] == "conformance_device"
        assert (await raw.request("probe", {"device_id": "conformance_device"}))[
            "connected"
        ] is True
        await raw.close()

        config = {"endpoint": endpoint, "token": "s3cret", "timeout_s": 2.0}
        transport = build_transport(config)
        status = await transport.open(_host_context(config))
        assert status.connected is True
        assert status.detail == "lhp node stub_node"
        reading = await transport.read("sensor")
        assert reading.value == 21.5
        assert reading.device_id == "conformance_device"
        outcome = await transport.write("setpoint", 42.0)
        assert outcome.ok is True
        assert outcome.readback is not None
        assert outcome.readback.channel_id == "setpoint"
        assert (await transport.halt()).halt_supported is True
        readings = (await transport.read_batch(("sensor", "setpoint"))).readings
        assert {r.channel_id for r in readings} == {"sensor", "setpoint"}
        await transport.close()
    finally:
        await server.stop()


async def test_socket_client_reconnects_and_falls_back_to_a_v1_only_node() -> None:
    connections = 0

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        try:
            request = decode_frame(await reader.readline())
            protocol = request.get("params", {}).get("protocol")
            if protocol != 1:
                writer.write(
                    encode_frame(
                        error_response(request.get("id"), ERR_PROTOCOL_MISMATCH, "v1 only")
                    )
                )
            else:
                writer.write(
                    encode_frame(
                        ok_response(
                            request.get("id"),
                            {"protocol": 1, "node_id": "legacy", "capabilities": []},
                        )
                    )
                )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()

    raw_server, endpoint = await _start_raw_tcp(handler)
    client = SocketLhpClient(endpoint)
    try:
        hello = await client.connect()
        assert hello["protocol"] == 1
        assert hello["node_id"] == "legacy"
        assert connections == 2, "v2 mismatch must reconnect before the v1 hello"
    finally:
        await client.close()
        raw_server.close()
        await raw_server.wait_closed()


async def test_remote_frame_round_trips_over_socket_and_releases_node_capture() -> None:
    registry = _FrameNodeRegistry()
    server = LhpNodeServer(registry, node_id="jetson", write_mode="trusted_v1")
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    config = {
        "endpoint": f"tcp://{socket_host}:{port}",
        "node_id": "jetson",
        "remote_device_id": "camera",
        "timeout_s": 2.0,
    }
    transport = build_transport(config)
    try:
        status = await transport.open(_frame_host_context(config))
        assert status.metadata["protocol"] == LHP_PROTOCOL_VERSION
        assert isinstance(transport, FrameTransport)
        frame = await transport.read_frame("frame", max_width=640, quality=70, fps=4.0)
        assert frame.device_id == "jetson__camera"
        assert frame.channel_id == "frame"
        assert frame.data == _FRAME_JPEG
        assert (frame.width, frame.height, frame.sequence) == (4, 2, 1)
        assert registry.source.requests == [("frame", 640, 70, 4.0)]
        await transport.close()
        assert registry.drops == 1
        assert registry.source.closes == 1
    finally:
        await transport.close()
        await server.stop()


async def test_node_preserves_legacy_frame_transport_without_fps_keyword() -> None:
    source = _LegacyFrameNodeTransport()
    server = LhpNodeServer(_FrameNodeRegistry(source), node_id="jetson", write_mode="trusted_v1")
    response = await server.handle_request(
        {
            "id": 1,
            "method": "read_frame",
            "params": {
                "device_id": "camera",
                "channel_id": "frame",
                "max_width": 640,
                "quality": 70,
                "fps": 4.0,
            },
        }
    )

    assert response["ok"] is True
    assert source.requests == [("frame", 640, 70)]


class _FreshFrameNodeTransport(_FrameNodeTransport):
    """Hub-backed frame source: honours freshness and serves concurrent readers."""

    frame_reads_concurrent = True

    def __init__(self) -> None:
        super().__init__()
        self.bounds: list[tuple[int, float | None]] = []
        self.gate: asyncio.Event | None = None
        self.lock_seen_free: list[bool] = []
        self.registry: Any = None

    async def read_frame(
        self,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
        min_sequence: int = 0,
        after_monotonic: float | None = None,
    ) -> FrameReading:
        self.bounds.append((min_sequence, after_monotonic))
        if self.registry is not None:
            self.lock_seen_free.append(not self.registry.device_io("camera").locked())
        if self.gate is not None:
            await self.gate.wait()
        return await super().read_frame(
            channel_id, max_width=max_width, quality=quality, fps=fps
        )

    async def probe(self) -> TransportStatus:
        hub = {"running": True, "sequence": 9, "fps": 30.0, "frame_age_ms": 12.5}
        return TransportStatus(connected=True, metadata={"frame_hub": hub, "serial": "x"})


async def test_probe_carries_the_node_frame_pump_health_to_the_host() -> None:
    server = LhpNodeServer(
        _FrameNodeRegistry(_FreshFrameNodeTransport()), node_id="jetson", write_mode="trusted_v1"
    )
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    config = {
        "endpoint": f"tcp://{socket_host}:{port}",
        "node_id": "jetson",
        "remote_device_id": "camera",
        "timeout_s": 2.0,
    }
    transport = build_transport(config)
    try:
        await transport.open(_frame_host_context(config))
        status = await transport.probe()
        assert status.connected
        # Only the pump stats cross the wire; the rest of the node metadata stays local.
        assert status.metadata == {
            "frame_hub": {"running": True, "sequence": 9, "fps": 30.0, "frame_age_ms": 12.5}
        }
    finally:
        await transport.close()
        await server.stop()


class _HoldingFrameNodeRegistry(_FrameNodeRegistry):
    """Frame registry with the holder ledger the production registry keeps."""

    def __init__(self, source: Any | None = None) -> None:
        super().__init__(source)
        self.holds: list[tuple[str, str]] = []

    def hold(self, device_id: str, holder: str) -> None:
        self.holds.append((device_id, holder))

    async def release_hold(self, device_id: str, holder: str) -> bool:
        return True


def _read_frame_request(**params: Any) -> dict[str, Any]:
    return {
        "id": 1,
        "method": "read_frame",
        "params": {"device_id": "camera", "channel_id": "frame", **params},
    }


async def test_hello_advertises_frame_freshness_on_v2() -> None:
    server = LhpNodeServer(_FrameNodeRegistry(), node_id="jetson", write_mode="trusted_v1")
    response = await server.handle_request(
        {"id": 1, "method": "hello", "params": {"protocol": LHP_PROTOCOL_VERSION}}
    )
    assert CAP_FRAME_FRESHNESS_V1 in response["result"]["capabilities"]


async def test_freshness_bounds_cross_the_wire_as_sequence_and_age() -> None:
    source = _FreshFrameNodeTransport()
    registry = _HoldingFrameNodeRegistry(source)
    server = LhpNodeServer(registry, node_id="jetson", write_mode="trusted_v1")
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    config = {
        "endpoint": f"tcp://{socket_host}:{port}",
        "node_id": "jetson",
        "remote_device_id": "camera",
        "timeout_s": 2.0,
    }
    transport = build_transport(config)
    try:
        await transport.open(_frame_host_context(config))
        host_after = time.monotonic() - 0.2
        await transport.read_frame("frame", min_sequence=5, after_monotonic=host_after)
        await transport.read_frame("frame")
        (sequence, node_after), unbounded = source.bounds
        assert sequence == 5
        # Re-anchored on the node's clock: stricter by the link latency, never looser.
        assert host_after <= node_after <= time.monotonic()
        assert unbounded == (0, None), "no bound requested means none sent"
    finally:
        await transport.close()
        await server.stop()


async def test_node_refuses_freshness_for_a_transport_that_cannot_honour_it() -> None:
    source = _LegacyFrameNodeTransport()
    server = LhpNodeServer(_FrameNodeRegistry(source), node_id="jetson", write_mode="trusted_v1")
    response = await server.handle_request(_read_frame_request(min_sequence=3))
    assert response["ok"] is False
    assert response["error"]["code"] == ERR_UNSUPPORTED_CAPABILITY
    assert source.requests == [], "a stale frame is never served for a fresh request"


async def test_node_rejects_out_of_range_freshness_bounds() -> None:
    server = LhpNodeServer(
        _FrameNodeRegistry(_FreshFrameNodeTransport()), node_id="jetson", write_mode="trusted_v1"
    )
    for params in ({"min_sequence": -1}, {"max_age_ms": -5}, {"max_age_ms": 1e9}):
        response = await server.handle_request(_read_frame_request(**params))
        assert response["error"]["code"] == ERR_BAD_REQUEST, params


async def test_host_refuses_freshness_against_a_node_without_the_capability() -> None:
    def respond(request: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover - never sent
        raise AssertionError("a freshness request must not reach an old node")

    server = _ScriptedServer(respond, capabilities=("read_frame", "release_frame"))
    config = {"server": server, "remote_device_id": "camera"}
    transport = build_transport(config)
    await transport.open(_frame_host_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.read_frame("frame", min_sequence=2)
    assert caught.value.failure_code == ERR_UNSUPPORTED_CAPABILITY
    await transport.close()


async def test_concurrent_frame_transport_reads_outside_the_device_lock() -> None:
    source = _FreshFrameNodeTransport()
    registry = _HoldingFrameNodeRegistry(source)
    source.registry = registry
    source.gate = asyncio.Event()
    server = LhpNodeServer(registry, node_id="jetson", write_mode="trusted_v1")
    pending = asyncio.create_task(server.handle_request(_read_frame_request()))
    for _ in range(50):
        if source.bounds:
            break
        await asyncio.sleep(0.01)
    # The read is parked waiting for a frame, yet another caller can take the lock.
    assert source.lock_seen_free == [True]
    async with registry.device_io("camera"):
        pass
    assert registry.holds == [("camera", "lhp:direct:frame")], "held before the read"
    source.gate.set()
    response = await pending
    assert response["ok"] is True


async def test_serial_frame_transport_still_reads_under_the_device_lock() -> None:
    source = _FreshFrameNodeTransport()
    source.frame_reads_concurrent = False
    registry = _HoldingFrameNodeRegistry(source)
    source.registry = registry
    server = LhpNodeServer(registry, node_id="jetson", write_mode="trusted_v1")
    response = await server.handle_request(_read_frame_request())
    assert response["ok"] is True
    assert source.lock_seen_free == [False]


async def test_malformed_remote_frame_payload_is_refused() -> None:
    def respond(request: Mapping[str, Any]) -> dict[str, Any]:
        return ok_response(
            request.get("id"),
            {
                "device_id": "camera",
                "channel_id": "frame",
                "data_b64": "not base64!",
                "size_bytes": 5,
                "media_type": "image/jpeg",
                "width": 4,
                "height": 2,
                "sequence": 1,
            },
        )

    server = _ScriptedServer(
        respond,
        capabilities=("read_frame", "release_frame"),
    )
    config = {"server": server, "remote_device_id": "camera"}
    transport = build_transport(config)
    await transport.open(_frame_host_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.read_frame("frame")
    assert caught.value.failure_code == "lhp_malformed_result"
    await transport.close()


async def test_unix_socket_binding_round_trips() -> None:
    # A short path on purpose: pytest's tmp_path (~100 chars on macOS) exceeds
    # the AF_UNIX sun_path limit before the socket name is even appended.
    path = f"/tmp/lhp_fabric_{uuid.uuid4().hex[:8]}.sock"
    server = lhp_node_server()
    await server.start_unix(path)
    try:
        config = {"endpoint": f"unix://{path}", "timeout_s": 2.0}
        transport = build_transport(config)
        await transport.open(_host_context(config))
        assert (await transport.read("sensor")).value == 21.5
        await transport.close()
    finally:
        await server.stop()


async def test_socket_gate_admits_only_hello_and_only_with_the_token() -> None:
    server = lhp_node_server(token="s3cret")
    socket_host, port = await server.start(host="127.0.0.1", port=0)
    try:
        reader, writer = await asyncio.open_connection(socket_host, port)
        try:
            request = make_request(
                7, "read", {"device_id": "conformance_device", "channel_id": "sensor"}
            )
            writer.write(encode_frame(request))
            await writer.drain()
            frame = decode_frame(await reader.readline())
            assert frame["ok"] is False
            assert frame["id"] == 7
            assert frame["error"]["code"] == ERR_UNAUTHORIZED
            assert await reader.readline() == b"", "the gate closes the connection"
        finally:
            writer.close()

        client = SocketLhpClient(f"tcp://{socket_host}:{port}", token="wrong")
        with pytest.raises(LhpRemoteError) as caught:
            await client.connect()
        assert caught.value.code == ERR_UNAUTHORIZED

        config = {"endpoint": f"tcp://{socket_host}:{port}", "token": "wrong"}
        transport = build_transport(config)
        with pytest.raises(TransportError) as open_failure:
            await transport.open(_host_context(config))
        assert open_failure.value.failure_code == f"lhp_{ERR_UNAUTHORIZED}"
    finally:
        await server.stop()


async def test_stop_is_idempotent_and_a_second_listener_is_refused() -> None:
    server = lhp_node_server()
    await server.start(host="127.0.0.1", port=0)
    with pytest.raises(RuntimeError):
        await server.start(host="127.0.0.1", port=0)
    await server.stop()
    await server.stop()


# ════════════════════════════════════════════════════════════════
# Identity: localisation, node pinning, and contract-violation guards
# ════════════════════════════════════════════════════════════════


async def test_remote_reading_is_relabelled_under_the_host_namespace() -> None:
    config = {"server": lhp_node_server(), "remote_device_id": "conformance_device"}
    transport = build_transport(config)
    await transport.open(_host_context(config, device_id="jetson__conformance_device"))
    reading = await transport.read("sensor")
    assert reading.device_id == "jetson__conformance_device"
    assert reading.observed_at > 1_500_000_000.0


async def test_open_refuses_a_node_that_is_not_the_expected_one() -> None:
    config = {"server": lhp_node_server(), "node_id": "expected_node"}
    transport = build_transport(config)
    with pytest.raises(TransportError) as caught:
        await transport.open(_host_context(config))
    assert caught.value.failure_code == "lhp_node_mismatch"


async def test_direct_client_refuses_a_foreign_protocol_version() -> None:
    server = _ScriptedServer(lambda request: ok_response(request.get("id"), {}), protocol=999)
    client = DirectLhpClient(server)
    with pytest.raises(LhpRemoteError) as caught:
        await client.connect()
    assert caught.value.code == ERR_PROTOCOL_MISMATCH

    config = {"server": server}
    transport = build_transport(config)
    with pytest.raises(TransportError) as open_failure:
        await transport.open(_host_context(config))
    assert open_failure.value.failure_code == f"lhp_{ERR_PROTOCOL_MISMATCH}"


async def test_direct_client_reports_remote_errors_and_refuses_after_close() -> None:
    client = DirectLhpClient(lhp_node_server())
    await client.connect()
    with pytest.raises(LhpRemoteError) as caught:
        await client.request("read", {"device_id": "conformance_device", "channel_id": "ghost"})
    assert caught.value.code == ERR_UNKNOWN_CHANNEL
    await client.close()
    with pytest.raises(LhpRemoteError) as closed:
        await client.request("probe", {"device_id": "conformance_device"})
    assert closed.value.code == ERR_CONNECTION_CLOSED


async def test_a_reading_for_another_channel_never_becomes_this_channels_value() -> None:
    config = {"server": lhp_node_server(wrong_channel_read=True)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    with pytest.raises(TransportError) as caught:
        await transport.read("sensor")
    assert caught.value.failure_code == ERR_INTERNAL_ERROR
    with pytest.raises(TransportError):
        await transport.read_batch(("sensor", "setpoint"))


async def test_a_driver_returning_evidence_for_another_channel_is_unknown() -> None:
    """The node enforces the driver contract: mislabelled readbacks are not success."""
    config = {"server": lhp_node_server(mislabel_write_readback=True)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("setpoint", 42.0)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.failure_code == ERR_INTERNAL_ERROR


async def test_a_node_mislabelling_evidence_cannot_pass_as_success() -> None:
    """The host does not trust the node's readback identity either."""

    def responder(request: Mapping[str, Any]) -> dict[str, Any]:
        return ok_response(
            request.get("id"),
            {
                "ok": True,
                "side_effect_state": "committed",
                "readback": {
                    "device_id": "conformance_device",
                    "channel_id": "setpoint_other",
                    "value": 1.0,
                    "observed_at": time.time(),
                },
            },
        )

    config = {"server": _ScriptedServer(responder)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    outcome = await transport.write("setpoint", 1.0)
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.failure_code == "lhp_result_mismatch"


# ── Fleet: host-side node registration over real LHP ──


class _StaticProvider:
    """Minimal provider for a node-side registry with fixed declarations."""

    kind = "fabric_test"

    def __init__(self, contexts: tuple[HardwareContext, ...]) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


def _node_registry(
    *contexts: HardwareContext, provider: _StaticProvider | None = None
) -> HardwareRegistry:
    """A real registry: declarations reach the wire via ``to_dict()``.

    Pass ``provider`` to change the node's declarations later, as a rescan would.
    """
    settings = HardwareSettings(
        enabled=True,
        unverified_context_policy=UnverifiedContextPolicy.ALLOW,
        require_describe_before_write=False,
        trust_skip_enabled=True,
        stream_enabled=False,
        persist_readings=False,
    )
    registry = HardwareRegistry(settings, providers=(provider or _StaticProvider(contexts),))
    registry.load()
    return registry


def _node_device(device_id: str, *, display_name: str = "", halt_supported: bool = True) -> HardwareContext:
    """One node-side device declaration served over LHP."""
    return HardwareContext(
        device_id=device_id,
        display_name=display_name or f"Node device {device_id}",
        transport=TransportRef(
            kind="mock",
            config={"values": {"sensor": 21.5, "setpoint": 50.0},
                    "halt_supported": halt_supported},
        ),
        halt_supported=halt_supported,
        channels=stub_node_context().channels,
    )


async def _start_node(
    registry: HardwareRegistry, node_id: str, *, token: str = ""
) -> tuple[LhpNodeServer, str]:
    server = LhpNodeServer(registry, node_id=node_id, token=token, write_mode="trusted_v1")
    host, port = await server.start(host="127.0.0.1", port=0)
    return server, f"tcp://{host}:{port}"


async def test_register_node_namespaces_declarations_and_routes_traffic() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "jetson01")
    host_registry = _node_registry()
    manager = FleetManager(host_registry)
    try:
        node = await manager.register_node(FleetNode("jetson01", "Bench Jetson", endpoint))
        assert node.status == "online"
        assert node.device_ids == ("arm",)
        assert manager.route_device("jetson01__arm") == "jetson01"
        assert manager.route_device("arm") is None

        context = host_registry.context("jetson01__arm")
        assert context is not None
        assert context.transport.kind == "lhp"
        assert context.transport.config["remote_device_id"] == "arm"
        assert context.transport.config["endpoint"] == endpoint
        assert context.transport.config["node_id"] == "jetson01"

        reading = await host_registry.read("jetson01__arm", "sensor")
        assert reading.value == 21.5
        assert reading.device_id == "jetson01__arm"

        transport = await host_registry.transport("jetson01__arm")
        outcome = await transport.write("setpoint", 40.0)
        assert outcome.ok is True
    finally:
        await manager.close()
        await node_server.stop()
        await host_registry.close_all()


async def test_register_node_pins_the_nodes_reported_identity() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "real_node")
    manager = FleetManager(_node_registry())
    try:
        with pytest.raises(FleetError) as caught:
            await manager.register_node(FleetNode("claimed_node", "Claimed", endpoint))
        assert caught.value.failure_code == "fleet_node_mismatch"
        assert manager.topology().nodes == ()
    finally:
        await node_server.stop()


async def test_register_node_validates_id_and_endpoint() -> None:
    manager = FleetManager(_node_registry())
    with pytest.raises(FleetError) as bad_id:
        await manager.register_node(FleetNode("Bad-Id", "Bad", "tcp://127.0.0.1:1"))
    assert bad_id.value.failure_code == "fleet_bad_node_id"
    with pytest.raises(FleetError) as no_endpoint:
        await manager.register_node(FleetNode("node_a", "No endpoint", ""))
    assert no_endpoint.value.failure_code == "fleet_no_endpoint"


async def test_register_node_unreachable_endpoint_is_a_typed_failure() -> None:
    manager = FleetManager(_node_registry())
    with pytest.raises(FleetError) as caught:
        await manager.register_node(FleetNode("node_x", "X", "tcp://127.0.0.1:1"))
    assert caught.value.failure_code == "fleet_unreachable"


async def test_register_node_wrong_token_is_refused() -> None:
    node_server, endpoint = await _start_node(
        _node_registry(_node_device("arm")), "jetson01", token="s3cret"
    )
    manager = FleetManager(_node_registry())
    try:
        with pytest.raises(FleetError) as caught:
            await manager.register_node(FleetNode("jetson01", "J", endpoint, token="wrong"))
        assert caught.value.failure_code == "fleet_unreachable"
        node = await manager.register_node(FleetNode("jetson01", "J", endpoint, token="s3cret"))
        assert node.status == "online"
    finally:
        await manager.close()
        await node_server.stop()


async def test_unregister_node_removes_namespaced_devices_and_source() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "jetson01")
    host_registry = _node_registry()
    manager = FleetManager(host_registry)
    try:
        await manager.register_node(FleetNode("jetson01", "J", endpoint))
        await manager.unregister_node("jetson01")
        assert manager.route_device("jetson01__arm") is None
        assert host_registry.context("jetson01__arm") is None
        assert host_registry.external_source_ids() == ()
        # Idempotent: unregistering again is a no-op.
        await manager.unregister_node("jetson01")
    finally:
        await manager.close()
        await node_server.stop()


async def test_discover_nodes_uses_reported_identity_and_skips_unreachable() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "reported01")
    manager = FleetManager(_node_registry())
    try:
        nodes = await manager.discover_nodes((endpoint, "tcp://127.0.0.1:1"))
        assert len(nodes) == 1
        assert nodes[0].node_id == "reported01"
        assert nodes[0].endpoint == endpoint
        assert nodes[0].device_ids == ("arm",)
        # Discovery is read-only: nothing was registered.
        assert manager.topology().nodes == ()
    finally:
        await node_server.stop()


async def test_halt_all_does_not_mistake_a_released_device_for_an_undelivered_stop() -> None:
    """A device that stops by letting go reports the same shape as a dead link.

    A camera releases its capture and answers ``connected=False,
    halt_supported=False``; before the delivery verdict was stated explicitly the
    fleet read that as an undelivered stop and reported its own backstop as failed
    on every node that carries a passive device.
    """
    node_server, endpoint = await _start_node(
        _node_registry(_node_device("camera", halt_supported=False)), "jetson01"
    )
    manager = FleetManager(_node_registry())
    try:
        await manager.register_node(FleetNode("jetson01", "J", endpoint))
        halted = await manager.halt_all()
        assert halted["jetson01"]["failures"] == 0
        assert halted["jetson01"]["halted"] is True
    finally:
        await manager.close()
        await node_server.stop()


async def test_a_halt_over_a_closed_lhp_link_is_undelivered_not_halted() -> None:
    """A stop that never left the host must not read as an idle device.

    ``halt_supported`` belongs to the device and survives the link: an arm whose
    connection dropped may still be travelling, and the caller has to be able to
    tell that apart from a device nobody was driving.
    """
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "jetson01")
    host_registry = _node_registry()
    manager = FleetManager(host_registry)
    try:
        await manager.register_node(FleetNode("jetson01", "J", endpoint))
        transport = await host_registry.transport("jetson01__arm")
        await transport.close()

        status = await transport.halt()

        assert status.connected is False
        assert status.halt_supported is True
        assert status.halt_delivered is False
    finally:
        await manager.close()
        await node_server.stop()


async def test_halt_all_reports_an_undelivered_halt_instead_of_claiming_it() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "jetson01")
    host_registry = _node_registry()
    manager = FleetManager(host_registry)
    try:
        await manager.register_node(FleetNode("jetson01", "J", endpoint))
        await node_server.stop()
        halted = await manager.halt_all()
        assert halted["jetson01"]["failures"] == 1
        assert halted["jetson01"]["halted"] is False
    finally:
        await manager.close()


async def test_halt_all_counts_an_idle_node_as_stopped_not_as_a_failure() -> None:
    """An emergency stop over devices the node never opened did stop everything.

    Nothing was being driven there, so reporting the stop as undelivered would
    make the fleet's own backstop cry wolf -- and would send it looking for a
    failure in hardware that was simply idle.
    """
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "jetson01")
    manager = FleetManager(_node_registry())
    try:
        await manager.register_node(FleetNode("jetson01", "J", endpoint))
        halted = await manager.halt_all()
        assert halted["jetson01"]["failures"] == 0
        assert halted["jetson01"]["halted"] is True
    finally:
        await manager.close()
        await node_server.stop()


# ── Registry: external sources (fabric nodes) ──


def test_external_source_registration_is_bookkeeping_until_load() -> None:
    registry = _node_registry()
    assert registry.register_external_contexts("node_a", (_node_device("arm"),)) == ("arm",)
    assert registry.external_source_ids() == ("node_a",)
    # Recorded but not yet admitted: load() is where declarations become live.
    assert registry.context("arm") is None
    registry.register_external_contexts("node_b", (_node_device("gripper"),))
    report = registry.load()
    assert set(report.admitted) == {"arm", "gripper"}
    # Re-registering replaces the snapshot, in registration order.
    registry.register_external_contexts("node_a", (_node_device("wrist"),))
    report = registry.load()
    assert set(report.admitted) == {"gripper", "wrist"}
    assert registry.external_source_ids() == ("node_a", "node_b")


def test_local_declarations_win_over_external_sources() -> None:
    local = _node_device("shared", display_name="local declaration")
    registry = _node_registry(local)
    remote = _node_device("shared", display_name="remote declaration")
    registry.register_external_contexts("node_a", (remote,))
    report = registry.load()
    assert "shared" in report.admitted
    # The shadowed duplicate is recorded as rejected, never silently ignored.
    assert "shared" in report.rejected
    kept = registry.context("shared")
    assert kept is not None
    assert kept.display_name == "local declaration"


def test_unregister_external_source_returns_recorded_ids() -> None:
    registry = _node_registry()
    registry.register_external_contexts("node_a", (_node_device("arm"), _node_device("wrench")))
    assert registry.load().admitted == ("arm", "wrench")
    assert registry.unregister_external_source("node_a") == ("arm", "wrench")
    assert registry.external_source_ids() == ()
    assert registry.load().admitted == ()


def test_external_source_requires_a_nonempty_id() -> None:
    registry = _node_registry()
    with pytest.raises(ValueError):
        registry.register_external_contexts("   ", ())


# ── Plugin: fleet node config parsing ──


def test_fleet_node_config_parsing_accepts_lhp_and_legacy_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from leapflow.hardware.plugin import HardwareContextPlugin

    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps(
            [
                {"node_id": "jetson01", "endpoint": "tcp://192.168.1.50:8767", "token": "s3cret"},
                {"node_id": "legacy01", "mcp_endpoint": "tcp://192.168.1.51:8767"},
                {"node_id": "", "endpoint": "tcp://192.168.1.52:8767"},
            ]
        ),
    )
    plugin = HardwareContextPlugin()
    plugin._registry = _node_registry()
    plugin._init_fleet_manager()
    pending = plugin._pending_fleet_nodes
    assert [node.node_id for node in pending] == ["jetson01", "legacy01"]
    assert pending[0].endpoint == "tcp://192.168.1.50:8767"
    assert pending[0].token == "s3cret"
    assert pending[1].endpoint == "tcp://192.168.1.51:8767"


# ── Startup registration (P1.12): env config -> registry, both planes ──


def test_configured_nodes_from_env_parses_and_skips_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from leapflow.hardware.fleet_config import configured_nodes_from_env

    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps(
            [
                {"node_id": "yahboom", "endpoint": "tcp://192.168.55.1:8767", "token": "t0ken"},
                {"node_id": "legacy", "mcp_endpoint": "tcp://127.0.0.1:9"},
                {
                    "node_id": "nulls",
                    "endpoint": "tcp://127.0.0.1:11",
                    "token": None,
                    "display_name": None,
                },
                {"node_id": "", "endpoint": "tcp://127.0.0.1:10"},
                {"display_name": "no endpoint"},
                "not-a-dict",
            ]
        ),
    )
    nodes = configured_nodes_from_env()
    assert [n.node_id for n in nodes] == ["yahboom", "legacy", "nulls"]
    assert nodes[0].endpoint == "tcp://192.168.55.1:8767"
    assert nodes[0].token == "t0ken"
    assert nodes[1].endpoint == "tcp://127.0.0.1:9"
    # Templated JSON null means "unset", never the literal string "None".
    assert nodes[2].token == ""
    assert nodes[2].display_name == ""


def test_configured_nodes_from_env_tolerates_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from leapflow.hardware.fleet_config import configured_nodes_from_env

    monkeypatch.setenv("LEAPFLOW_FLEET_NODES", "{not json")
    assert configured_nodes_from_env() == ()
    # A JSON object is not a node list either; startup must not raise.
    monkeypatch.setenv("LEAPFLOW_FLEET_NODES", '{"node_id": "x"}')
    assert configured_nodes_from_env() == ()


async def test_start_fleet_registers_env_nodes_and_skips_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon path: env config becomes admitted devices + supervision.

    An unreachable board must not block startup (or the reachable nodes), and
    a second call must be a no-op rather than a second heartbeat task.
    """
    from leapflow.hardware.plugin import HardwareContextPlugin

    remote_trust = TrustConfig(initial_level="CANDIDATE", demotion_threshold=2)
    node_server, endpoint = await _start_node(
        _node_registry(dataclasses.replace(_node_device("host"), trust_config=remote_trust)),
        "yahboom",
    )
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps(
            [
                {"node_id": "yahboom", "endpoint": endpoint},
                # Port 1 refuses connections: skipped, never fatal.
                {"node_id": "absent", "endpoint": "tcp://127.0.0.1:1"},
            ]
        ),
    )
    host_registry = _node_registry()
    registered_trust: dict[str, TrustConfig | None] = {}

    class _RecordingTrustGate:
        def register_device(self, device_id: str, trust_config: TrustConfig | None = None) -> None:
            registered_trust.setdefault(device_id, trust_config)

    plugin = HardwareContextPlugin()
    plugin._registry = host_registry
    plugin._hardware_trust_gate = _RecordingTrustGate()
    plugin._init_fleet_manager()
    try:
        await plugin.start_fleet()
        assert plugin._fleet_started is True
        assert host_registry.context("yahboom__host") is not None
        assert registered_trust["yahboom__host"] == remote_trust
        assert plugin._fleet_manager.route_device("absent__host") is None
        supervision = plugin._fleet_manager._heartbeat_task
        assert supervision is not None, "heartbeat must supervise registered nodes"

        await plugin.start_fleet()
        assert plugin._fleet_manager._heartbeat_task is supervision, (
            "start_fleet must be idempotent"
        )
    finally:
        await plugin._teardown()
        await node_server.stop()


async def test_cli_read_plane_registers_configured_fleet_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`leap hw` serves in-process, so it must register nodes itself."""
    from leapflow.cli.commands.hardware import _register_fleet_read_plane

    node_server, endpoint = await _start_node(_node_registry(_node_device("host")), "yahboom")
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps([{"node_id": "yahboom", "endpoint": endpoint}]),
    )
    registry = _node_registry()
    try:
        await _register_fleet_read_plane(registry)
        context = registry.context("yahboom__host")
        assert context is not None
        assert context.transport.kind == "lhp"
        assert context.transport.config["remote_device_id"] == "host"
    finally:
        await node_server.stop()
        await registry.close_all()


# ── Review follow-ups: startup containment, client hygiene, batch evidence ──


async def test_register_nodes_safely_runs_concurrently_and_contains_failures() -> None:
    """One blocked board must not delay the others; one bad node must not abort.

    Concurrency is proved structurally rather than by wall clock: every
    registration blocks until all of them have started, so a serial loop
    deadlocks and the watchdog fires.
    """
    from leapflow.hardware.fleet_config import register_nodes_safely

    started: list[str] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    class _Manager:
        async def register_node(self, node: FleetNode) -> FleetNode:
            started.append(node.node_id)
            if len(started) == 3:
                all_started.set()
            await release.wait()
            if node.node_id == "offline":
                raise FleetError("board is off", failure_code="fleet_unreachable")
            if node.node_id == "defective":
                raise RuntimeError("driver defect")
            return node

    nodes = (
        FleetNode("live", "", "tcp://127.0.0.1:1"),
        FleetNode("offline", "", "tcp://127.0.0.1:2"),
        FleetNode("defective", "", "tcp://127.0.0.1:3"),
    )
    task = asyncio.create_task(register_nodes_safely(_Manager(), nodes))
    await asyncio.wait_for(all_started.wait(), timeout=2.0)
    release.set()
    registered = await asyncio.wait_for(task, timeout=2.0)
    assert [node.node_id for node in registered] == ["live"]
    assert started == ["live", "offline", "defective"]


async def test_rebind_closes_the_previous_fleet_manager_before_reregistering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry re-bind must not orphan a live heartbeat supervisor.

    The previous manager keeps probing the old registry forever unless an
    async path closes it; start_fleet is that path.
    """
    from leapflow.hardware.plugin import HardwareContextPlugin

    node_server, endpoint = await _start_node(_node_registry(_node_device("host")), "yahboom")
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps([{"node_id": "yahboom", "endpoint": endpoint}]),
    )
    old_registry = _node_registry()
    new_registry = _node_registry()
    plugin = HardwareContextPlugin()
    plugin._registry = old_registry
    plugin._init_fleet_manager()
    try:
        await plugin.start_fleet()
        previous = plugin._fleet_manager
        assert previous._heartbeat_task is not None

        plugin.bind_runtime(hardware_registry=new_registry)
        assert plugin._fleet_manager is not previous
        assert plugin._stale_fleet_managers == [previous]

        await plugin.start_fleet()
        assert plugin._stale_fleet_managers == []
        assert previous._heartbeat_task is None, "the orphan must be closed"
        assert old_registry.external_source_ids() == (), (
            "the orphan must release its old-registry devices"
        )
        assert new_registry.context("yahboom__host") is not None
    finally:
        await plugin._teardown()
        await node_server.stop()
        await old_registry.close_all()


async def test_close_during_connect_cannot_leave_a_live_connection() -> None:
    """close() must serialise with an in-flight connect.

    Without the lock, close() runs while the socket is still being opened,
    tears down nothing (the writer is not assigned yet), and the handshake
    then completes on a closed client -- leaking a live connection no later
    call can reach.
    """
    node_server, endpoint = await _start_node(_node_registry(_node_device("arm")), "node01")
    client = SocketLhpClient(endpoint)
    original_open = client._open_transport
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def gated_open() -> None:
        entered.set()
        await gate.wait()
        await original_open()

    client._open_transport = gated_open
    try:
        connect_task = asyncio.create_task(client.connect())
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        close_task = asyncio.create_task(client.close())
        await asyncio.sleep(0)  # close() sets the flag, then queues on the lock
        gate.set()
        with pytest.raises(LhpRemoteError):
            await asyncio.wait_for(connect_task, timeout=2.0)
        await asyncio.wait_for(close_task, timeout=2.0)
        assert client.connected is False
        assert client._writer is None
    finally:
        await client.close()
        await node_server.stop()


async def test_client_closes_the_connection_when_a_frame_exceeds_the_limit() -> None:
    """The oversized-frame defence must also close the socket.

    The frame merely exceeded the line limit; the connection itself was
    healthy. Leaving it open leaks a client FD and keeps the peer waiting
    for a next frame that will never come.
    """
    saw_disconnect = asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            request = decode_frame(line)
            writer.write(
                encode_frame(
                    ok_response(
                        request.get("id"),
                        {"protocol": LHP_PROTOCOL_VERSION, "node_id": "big"},
                    )
                )
            )
            writer.write(b"x" * (MAX_FRAME_BYTES + 8192) + b"\n")
            await writer.drain()
            await reader.read(1)  # returns b"" only when the client closes its side
            saw_disconnect.set()
        except (ConnectionError, OSError):
            saw_disconnect.set()
        finally:
            writer.close()

    raw_server, endpoint = await _start_raw_tcp(handler)
    client = SocketLhpClient(endpoint)
    try:
        await client.connect()
        await asyncio.wait_for(saw_disconnect.wait(), timeout=2.0)
        assert client.connected is False
    finally:
        await client.close()
        raw_server.close()
        await raw_server.wait_closed()


async def test_a_node_mislabelling_batch_evidence_cannot_pass_as_success() -> None:
    """Host-side: a batch readback for another channel is not evidence."""

    def responder(request: Mapping[str, Any]) -> dict[str, Any]:
        return ok_response(
            request.get("id"),
            {
                "ok": True,
                "side_effect_state": "committed",
                "outcomes": [
                    {
                        "ok": True,
                        "side_effect_state": "committed",
                        "readback": {
                            "device_id": "conformance_device",
                            "channel_id": "setpoint_other",
                            "value": 1.0,
                            "observed_at": time.time(),
                        },
                    }
                ],
            },
        )

    config = {"server": _ScriptedServer(responder)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    batch = await transport.write_batch((("setpoint", 1.0),))
    assert batch.outcomes, "the outcome list must survive the guard"
    outcome = batch.outcomes[0]
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert outcome.failure_code == "lhp_result_mismatch"


async def test_a_batch_of_mislabelled_evidence_degrades_to_unknown() -> None:
    """Node-side: the sequential-fallback batch path enforces the same rule."""
    config = {"server": lhp_node_server(mislabel_write_readback=True)}
    transport = build_transport(config)
    await transport.open(_host_context(config))
    batch = await transport.write_batch((("setpoint", 42.0),))
    assert batch.ok is False
    assert batch.side_effect_state == SIDE_EFFECT_UNKNOWN
    assert batch.outcomes == ()


# ── Second-round review: secret hygiene, re-bind identity, supervision ──


def test_skipped_fleet_entries_never_log_the_shared_secret(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed entry is reported by index and safe fields, never echoed raw."""
    from leapflow.hardware.fleet_config import configured_nodes_from_env

    secret = "s3cr3t-token-value"
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES",
        json.dumps([{"node_id": "", "endpoint": "tcp://10.0.0.1:8767", "token": secret}]),
    )
    with caplog.at_level("WARNING"):
        assert configured_nodes_from_env() == ()
    assert "entry #0" in caplog.text
    assert secret not in caplog.text


async def test_list_exports_declarations_without_node_private_bindings() -> None:
    """The node's southbound config (paths, credentials) never crosses the wire."""
    device = dataclasses.replace(
        _with_transport(
            _node_device("host"),
            TransportRef(kind="mock", config={"values": {"sensor": 1.0}, "api_token": "node-only"}),
        ),
        control_bindings={"mcp_server": {"url": "http://10.0.0.5", "token": "bind-only"}},
    )
    server = LhpNodeServer(_node_registry(device), node_id="yahboom", write_mode="trusted_v1")
    client = DirectLhpClient(server)
    await client.connect()
    listing = await client.request("list", {})
    exported = listing["devices"][0]
    assert exported["transport"] == {"kind": "mock", "config": {}}
    assert "control_bindings" not in exported
    assert "node-only" not in json.dumps(listing)
    assert "bind-only" not in json.dumps(listing)


async def test_socket_and_direct_clients_return_the_same_hello_payload() -> None:
    node_server, endpoint = await _start_node(_node_registry(_node_device("host")), "yahboom")
    client = SocketLhpClient(endpoint)
    try:
        hello = await client.connect()
        direct_hello = await DirectLhpClient(node_server).connect()
        assert hello == direct_hello
        assert hello["device_count"] == 1
        assert list(hello["capabilities"][:len(LHP_CAPABILITIES)]) == list(LHP_CAPABILITIES)
        assert "control.v1" in hello["capabilities"]
    finally:
        await client.close()
        await node_server.stop()


async def test_same_registry_rebind_keeps_the_started_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin hot reload re-applies the same registry; the fleet must survive.

    Regression for the R1 observation: the reload re-bound every dep, the plugin
    swapped its started manager for a fresh one nobody started, and fleet tools
    saw an empty fleet while the orphan kept supervising.
    """
    from leapflow.hardware.plugin import HardwareContextPlugin

    node_server, endpoint = await _start_node(_node_registry(_node_device("host")), "yahboom")
    monkeypatch.setenv(
        "LEAPFLOW_FLEET_NODES", json.dumps([{"node_id": "yahboom", "endpoint": endpoint}])
    )
    registry = _node_registry()
    plugin = HardwareContextPlugin()
    plugin.bind_runtime(hardware_registry=registry)
    try:
        await plugin.start_fleet()
        manager = plugin._fleet_manager
        tool_names = {tool.name for tool in plugin.tools}
        assert "hw_fleet_status" in tool_names

        plugin.bind_runtime(hardware_registry=registry, session_id="s1")
        assert plugin._fleet_manager is manager
        assert plugin._fleet_started is True
        assert plugin._stale_fleet_managers == []
        status = next(t for t in plugin.tools if t.name == "hw_fleet_status")
        topology = await status.handler()
        assert [node["node_id"] for node in topology["nodes"]] == ["yahboom"]
    finally:
        await plugin._teardown()
        await node_server.stop()


def test_fresh_hardware_plugin_adopts_live_fleet_on_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new plugin instance steals the active manager before rebuilding tools."""
    from leapflow.hardware.plugin import HardwareContextPlugin

    monkeypatch.delenv("LEAPFLOW_FLEET_NODES", raising=False)
    registry = _node_registry()
    manager = FleetManager(registry)
    old_plugin = HardwareContextPlugin()
    old_plugin._registry = registry
    old_plugin._fleet_manager = manager
    old_plugin._fleet_started = True

    fresh_plugin = HardwareContextPlugin()
    fresh_plugin._adopt_reload_state(old_plugin)
    fresh_plugin.bind_runtime(hardware_registry=registry)

    assert fresh_plugin._fleet_manager is manager
    assert fresh_plugin._fleet_started is True
    assert old_plugin._fleet_manager is None
    assert "hw_fleet_status" in {tool.name for tool in fresh_plugin.tools}


async def test_heartbeat_sweep_logs_outage_and_recovery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both directions of a transition leave evidence; probes run concurrently."""
    registry = _node_registry()
    manager = FleetManager(registry, node_timeout_s=1.0)
    in_flight = 0
    peak = 0
    answers = {"a": False, "b": False}

    async def _probe(endpoint: str, *, token: str = "") -> dict[str, Any] | None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return {} if answers[endpoint] else None

    manager._probe_endpoint = _probe  # type: ignore[method-assign]
    stale = time.monotonic() - 5.0
    for node_id in answers:
        manager._nodes[node_id] = FleetNode(
            node_id, "", node_id, status="online", last_heartbeat=stale
        )

    with caplog.at_level("INFO", logger="leapflow.hardware.fleet"):
        await manager._heartbeat_sweep()
        assert {n.status for n in manager._nodes.values()} == {"offline"}
        assert peak == 2, "probes must run concurrently"
        answers["a"] = True
        await manager._heartbeat_sweep()
    assert manager._nodes["a"].status == "online"
    assert manager._nodes["b"].status == "offline"
    assert "node 'a' back online" in caplog.text
    await registry.close_all()


async def test_hello_fingerprints_declarations_without_node_private_bindings() -> None:
    """Stable while nothing changes; moves with any declaration; blind to secrets."""
    provider = _StaticProvider((_node_device("host"),))
    registry = _node_registry(provider=provider)
    server = LhpNodeServer(registry, node_id="yahboom", write_mode="trusted_v1")
    first = (await DirectLhpClient(server).connect())["declarations"]
    assert first and first == (await DirectLhpClient(server).connect())["declarations"]

    provider._contexts = (
        _with_transport(
            _node_device("host"), TransportRef(kind="mock", config={"api_token": "rotated"})
        ),
    )
    registry.load()
    assert (await DirectLhpClient(server).connect())["declarations"] == first, (
        "a node-private binding never crosses the wire, so it must not move the fingerprint"
    )

    provider._contexts = (_node_device("host", display_name="Renamed"),)
    registry.load()
    assert (await DirectLhpClient(server).connect())["declarations"] != first
    await registry.close_all()


async def test_heartbeat_re_registers_a_node_whose_declarations_changed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The board defect: a node redeployed inside one heartbeat never looked offline.

    The host kept the envelope it first admitted -- a thermal limit of 0.1045 degC
    raised false alarms after the node already declared 104.5 -- and a device
    added on the node would never have appeared. Liveness alone cannot see either.
    """
    provider = _StaticProvider((_node_device("arm"),))
    node_registry = _node_registry(provider=provider)
    node_server, endpoint = await _start_node(node_registry, "jetson01")
    host_registry = _node_registry()
    manager = FleetManager(host_registry)
    try:
        await manager.register_node(FleetNode("jetson01", "Bench Jetson", endpoint))
        with caplog.at_level("INFO", logger="leapflow.hardware.fleet"):
            await manager._heartbeat_sweep()
        assert "re-registering" not in caplog.text, "an unchanged node must not churn"

        provider._contexts = (
            _node_device("arm", display_name="Arm (redeployed)"),
            _node_device("camera"),
        )
        node_registry.load()
        with caplog.at_level("INFO", logger="leapflow.hardware.fleet"):
            await manager._heartbeat_sweep()
        assert "declarations changed; re-registering" in caplog.text
        assert manager.route_device("jetson01__camera") == "jetson01"
        arm = host_registry.context("jetson01__arm")
        assert arm is not None and arm.display_name == "Arm (redeployed)"

        caplog.clear()
        with caplog.at_level("INFO", logger="leapflow.hardware.fleet"):
            await manager._heartbeat_sweep()
        assert "re-registering" not in caplog.text, "the new fingerprint is adopted once"
    finally:
        await manager.close()
        await node_server.stop()
        await host_registry.close_all()
        await node_registry.close_all()


def _with_transport(context: HardwareContext, transport: TransportRef) -> HardwareContext:
    from dataclasses import replace

    return replace(context, transport=transport)

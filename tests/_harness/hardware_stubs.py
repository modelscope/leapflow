# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared stubs for the hardware test layer.

``ScriptedHuman`` and ``with_transport_config`` are needed by several hardware
test modules.  Keeping them here avoids cross-imports between test files —
which would couple test modules that should stay independent — while still
sharing a single, authoritative implementation.

The LHP node stubs (``stub_node_context``, ``StubNodeRegistry``,
``lhp_node_server`` and the two driver-contract violators) live here for the
same reason: the transport-contract suite, the LHP fabric tests and any future
node-side suite all drive the same node shape.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Mapping

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    TransportRef,
)
from leapflow.hardware.lhp import LhpNodeServer
from leapflow.hardware.transport import BatchReading, TransportError, TransportStatus
from leapflow.hardware.transports import build_transport
from leapflow.security.approval import ApprovalDecision, ApprovalRequest


class ScriptedHuman:
    """Stands in for the person at the prompt. The only fake in the chain."""

    def __init__(self, *decisions: ApprovalDecision) -> None:
        self._decisions = list(decisions)
        self.prompts: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.prompts.append(request)
        if not self._decisions:
            return ApprovalDecision.DENY
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def with_transport_config(context: HardwareContext, **overrides: Any) -> HardwareContext:
    """Return *context* with its transport config merged with *overrides*.

    Lets a test change device behaviour -- inject a failure, remove halt support,
    open an interlock -- without restating the whole declaration.
    """
    merged = {**dict(context.transport.config), **overrides}
    return replace(context, transport=replace(context.transport, config=merged))


def stub_node_context(*, read_only: bool = False, halt_supported: bool = True) -> HardwareContext:
    """The node-side device the LHP stubs serve: the conformance shape."""
    setpoint_direction = Direction.READ.value if read_only else Direction.READWRITE.value
    return HardwareContext(
        device_id="conformance_device",
        display_name="Node stub device",
        transport=TransportRef(kind="mock", config={}),
        halt_supported=halt_supported,
        channels=(
            Channel(
                channel_id="sensor",
                direction=Direction.READ.value,
                quantity="generic.sensor",
                unit="unit",
                sample_rate_hz=1.0,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
            Channel(
                channel_id="setpoint",
                direction=setpoint_direction,
                quantity="generic.setpoint",
                unit="unit",
                effect=HardwareEffect.CONFIGURE.value,
                verify_after_write=True,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0, reversible=True),
            ),
        ),
        provenance=ContextProvenance(verified_by="stub"),
    )


class StubNodeRegistry:
    """Node-side stand-in for ``HardwareRegistry`` behind an ``LhpNodeServer``.

    Duck-typed to exactly the surface the server uses (``context`` / ``contexts``
    / ``device_io`` / ``transport`` / ``read_batch`` / ``check_safety_policy``),
    with one mock device underneath. The flags drive the node-side scenarios the
    suites need: a failing device, a halt-less device, a read-only declaration, a
    safety refusal, a contract-violating driver, a wrong-channel reading, and
    write evidence under the wrong channel.
    """

    def __init__(
        self,
        *,
        values: Mapping[str, Any] | None = None,
        halt_supported: bool = True,
        halt_closes: bool = False,
        fail_writes: bool = False,
        read_only: bool = False,
        safety_deny_reason: str = "",
        explode_writes: bool = False,
        wrong_channel_read: bool = False,
        mislabel_write_readback: bool = False,
    ) -> None:
        config: dict[str, Any] = {
            "values": dict(values or {"sensor": 21.5, "setpoint": 50.0}),
            "halt_supported": halt_supported,
        }
        if fail_writes:
            # repeat=True because one server instance is shared across every check
            # of a run: a one-shot rule would fail the first write and succeed on
            # the second, silently inverting the contract under test.
            config["failures"] = [
                {
                    "channel_id": "setpoint",
                    "on_call": 1,
                    "side_effect_state": "partial",
                    "repeat": True,
                }
            ]
        transport: Any = build_transport("mock", config)
        if explode_writes:
            transport = _ExplodingWriteTransport(transport)
        if wrong_channel_read:
            transport = _WrongChannelReadTransport(transport)
        if mislabel_write_readback:
            transport = _MislabelingReadbackTransport(transport)
        self._transport = transport
        self._context = stub_node_context(read_only=read_only, halt_supported=halt_supported)
        self._safety_deny_reason = str(safety_deny_reason or "")
        self._io_lock = asyncio.Lock()
        self._opened = False
        self._halt_closes = bool(halt_closes)
        self.opens = 0

    # ── The registry surface the LHP node server depends on ──

    def contexts(self) -> tuple[HardwareContext, ...]:
        return (self._context,)

    def context(self, device_id: str) -> HardwareContext | None:
        return self._context if device_id == self._context.device_id else None

    def check_safety_policy(
        self, context: HardwareContext, channel: Channel, value: Any, *, rate: float | None = None
    ) -> tuple[bool, str]:
        if self._safety_deny_reason:
            return False, self._safety_deny_reason
        return True, ""

    def device_io(self, device_id: str) -> asyncio.Lock:
        return self._io_lock

    async def transport(self, device_id: str) -> Any:
        if device_id != self._context.device_id:
            raise TransportError(f"unknown device {device_id!r}", failure_code="unknown_device")
        if not self._opened:
            await self._transport.open(self._context)
            self._opened = True
            self.opens += 1
        return self._transport

    def get_open_transport(self, device_id: str) -> Any:
        """The live link only; never opens one (matches ``HardwareRegistry``)."""
        if device_id != self._context.device_id:
            raise TransportError(f"unknown device {device_id!r}", failure_code="unknown_device")
        if not self._opened:
            raise TransportError(
                f"No open transport for device {device_id!r}.",
                failure_code="transport_not_open",
            )
        return self._transport

    @property
    def opened(self) -> bool:
        """Whether the node has a live link to its device."""
        return self._opened

    async def halt(self, device_id: str) -> TransportStatus | None:
        """Stop the device over its live link, mirroring ``HardwareRegistry.halt``.

        Never opens a link, and retires the link when the stop ended it -- a
        device left cached after it closed answers every later read with "is not
        open".
        """
        if device_id != self._context.device_id:
            raise TransportError(f"unknown device {device_id!r}", failure_code="unknown_device")
        if not self._opened:
            return None
        status = await self._transport.halt()
        if self._halt_closes or not status.connected:
            self._opened = False
            return TransportStatus(
                connected=False,
                halt_supported=status.halt_supported,
                halt_delivered=status.halt_delivered,
                detail=status.detail or "closed by the stop",
            )
        return status

    async def read_batch(self, device_id: str, channel_ids: tuple[str, ...]) -> BatchReading:
        transport = await self.transport(device_id)
        async with self.device_io(device_id):
            readings = tuple([await transport.read(channel_id) for channel_id in channel_ids])
        return BatchReading(device_id=device_id, readings=readings)

    @property
    def write_log(self) -> tuple[tuple[str, Any], ...]:
        """Accepted writes on the underlying mock, for no-effect assertions."""
        inner = self._transport
        while not hasattr(inner, "write_log"):
            inner = getattr(inner, "_inner", None)
            if inner is None:
                return ()
        return tuple(inner.write_log)


def lhp_node_server(
    *, node_id: str = "stub_node", token: str = "", device_ids: tuple[str, ...] = (),
    write_mode: str = "trusted_v1", **stub_kwargs: Any
) -> LhpNodeServer:
    """Build an ``LhpNodeServer`` over a ``StubNodeRegistry`` for direct-binding configs."""
    return LhpNodeServer(
        StubNodeRegistry(**stub_kwargs), node_id=node_id, token=token,
        device_ids=device_ids, write_mode=write_mode,
    )


class _ExplodingWriteTransport:
    """Driver-contract violator: every write raises a non-TransportError."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.kind = getattr(inner, "kind", "mock")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def write(self, channel_id: str, value: Any) -> Any:
        raise RuntimeError("driver exploded mid-write")


class _WrongChannelReadTransport:
    """Contract violator: readings come back under a different channel id."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.kind = getattr(inner, "kind", "mock")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def read(self, channel_id: str) -> Any:
        reading = await self._inner.read(channel_id)
        return replace(reading, channel_id=f"{channel_id}_other")


class _MislabelingReadbackTransport:
    """Contract violator: write evidence comes back under a different channel id."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.kind = getattr(inner, "kind", "mock")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def write(self, channel_id: str, value: Any) -> Any:
        outcome = await self._inner.write(channel_id, value)
        if outcome.readback is None:
            return outcome
        return replace(outcome, readback=replace(outcome.readback, channel_id=f"{channel_id}_other"))

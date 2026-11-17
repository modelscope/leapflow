# Copyright (c) Alibaba, Inc. and its affiliates.
"""ROS 2 halt: each command topic gets its declared stop, never a guessed one.

A default-constructed message stops a rate command, but on a position topic it is
a full-speed move to the zero posture. These tests pin that the transport only
publishes what ``halt_commands`` declares, holds a position topic at the state it
last observed, and sends nothing it cannot justify.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from leapflow.hardware.transport import STOP_EFFECT_HOLD, STOP_EFFECT_NONE, TransportError
from leapflow.hardware.transports.ros2 import ROS2Transport, build_transport


@dataclass
class _Float64:
    data: float = 0.0


@dataclass
class _Twist:
    linear: float = 0.0


@dataclass
class _Publisher:
    msg_type: type
    sent: list[Any] = field(default_factory=list)

    def publish(self, msg: Any) -> None:
        self.sent.append(msg)


def _transport(
    halt_commands: dict[str, Any] | None,
    *,
    state_age_s: float = 0.0,
) -> tuple[ROS2Transport, dict[str, _Publisher]]:
    """A connected transport with a position and a velocity command topic."""
    transport = build_transport({
        "topics": {"arm.position": "/arm/cmd", "arm.state": "/arm/state", "base.vel": "/cmd_vel"},
        **({} if halt_commands is None else {"halt_commands": halt_commands}),
    })
    publishers = {"arm.position": _Publisher(_Float64), "base.vel": _Publisher(_Twist)}
    transport._connected = True  # noqa: SLF001
    transport._node = object()  # noqa: SLF001
    transport._cmd_channels = set(publishers)  # noqa: SLF001
    transport._publishers = dict(publishers)  # noqa: SLF001
    transport._subscribers = {"arm.state": object()}  # noqa: SLF001
    transport._latest_msgs = {"arm.state": _Float64(data=42.5)}  # noqa: SLF001
    transport._msg_timestamps = {"arm.state": time.monotonic() - state_age_s}  # noqa: SLF001
    return transport, publishers


@pytest.mark.asyncio
async def test_a_position_topic_holds_its_observed_state_and_a_rate_topic_gets_zero() -> None:
    transport, publishers = _transport(
        {"arm.position": {"hold_from": "arm.state"}, "base.vel": "zero"}
    )
    status = await transport.halt()
    assert status.halt_delivered is True and status.stop_effect == STOP_EFFECT_HOLD
    assert publishers["arm.position"].sent == [_Float64(data=42.5)], "held, not driven to 0"
    assert publishers["base.vel"].sent == [_Twist()]


@pytest.mark.asyncio
async def test_an_undeclared_command_topic_is_never_published_to() -> None:
    transport, publishers = _transport(None)
    assert (await transport.probe()).halt_supported is False
    status = await transport.halt()
    assert status.halt_delivered is False and status.stop_effect == STOP_EFFECT_NONE
    assert publishers["arm.position"].sent == [] and publishers["base.vel"].sent == []
    assert "halt_commands" in status.detail


@pytest.mark.asyncio
async def test_a_hold_without_fresh_state_sends_nothing_on_that_topic() -> None:
    transport, publishers = _transport(
        {"arm.position": {"hold_from": "arm.state"}, "base.vel": "zero"}, state_age_s=10.0
    )
    status = await transport.halt()
    assert publishers["arm.position"].sent == []
    assert publishers["base.vel"].sent == [_Twist()]
    # One topic did stop, so the arm is partly held, but the stop is not delivered.
    assert status.halt_delivered is False and status.stop_effect == STOP_EFFECT_HOLD
    assert "no fresh state" in status.detail


@pytest.mark.asyncio
async def test_probe_supports_halt_only_when_every_command_topic_declares_a_stop() -> None:
    transport, _ = _transport({"base.vel": "zero"})
    assert (await transport.probe()).halt_supported is False
    transport, _ = _transport({"arm.position": {"hold_from": "arm.state"}, "base.vel": "zero"})
    assert (await transport.probe()).halt_supported is True


@pytest.mark.parametrize("raw", [["zero"], {"arm.position": "freeze"}, {"arm.position": {}}])
def test_a_malformed_halt_declaration_is_a_typed_config_error(raw: Any) -> None:
    with pytest.raises(TransportError) as caught:
        build_transport({"topics": {"arm.position": "/arm/cmd"}, "halt_commands": raw})
    assert caught.value.failure_code == "ros2_invalid_config"

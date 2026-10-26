# Copyright (c) Alibaba, Inc. and its affiliates.
"""Device holder ledger: who may close a shared camera, and when.

A camera is one physical resource with several consumers -- a LeapBoard preview per
channel, a remote host reading frames over LHP, a node-local skill capturing RGB-D.
Before the ledger, whichever consumer stopped first closed the device under everyone
else. These tests pin the replacement contract:

- the transport closes only when its last holder leaves;
- the check is repeated under the device lock, so a hold taken while a release waited
  wins;
- a close in progress blocks a concurrent re-open instead of racing it;
- an LHP connection holds only what it read, and a dropped connection releases it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.hardware import registry as registry_module
from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    HardwareContext,
    HardwareEffect,
    PrivacyTier,
    Representation,
    TransportRef,
)
from leapflow.hardware.lhp import LhpNodeServer, SocketLhpClient
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    Reading,
    TransportStatus,
    WriteOutcome,
)

_JPEG = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xd9"
COLOR = "camera.color.preview"
DEPTH = "camera.depth.preview"


class _Camera:
    """A frame transport that records its lifecycle and can stall its close."""

    kind = "fake_camera"

    def __init__(self, log: list[str], *, close_delay_s: float = 0.0) -> None:
        self._log = log
        self._close_delay_s = close_delay_s
        self.connected = False

    async def open(self, context: HardwareContext) -> TransportStatus:
        self._log.append("open")
        self.connected = True
        return TransportStatus(connected=True)

    async def close(self) -> TransportStatus:
        self._log.append("close:start")
        await asyncio.sleep(self._close_delay_s)
        self.connected = False
        self._log.append("close:end")
        return TransportStatus(connected=False)

    async def probe(self) -> TransportStatus:
        return TransportStatus(connected=self.connected)

    async def halt(self) -> TransportStatus:
        return TransportStatus(connected=self.connected)

    async def read(self, channel_id: str) -> Reading:
        return Reading(device_id="cam", channel_id=channel_id, value="frame")

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        return WriteOutcome(ok=False, side_effect_state=SIDE_EFFECT_NONE, error="read-only")

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        if not self.connected:
            raise AssertionError("captured from a closed camera")
        return FrameReading(
            device_id="cam", channel_id=channel_id, data=_JPEG, width=4, height=2
        )


def _channel(channel_id: str) -> Channel:
    return Channel(
        channel_id=channel_id,
        direction=Direction.READ.value,
        quantity="image_frame",
        effect=HardwareEffect.READ.value,
        sample_rate_hz=5.0,
        representation=Representation.FRAME.value,
        media_type="image/jpeg",
        privacy=PrivacyTier.ENVIRONMENT.value,
    )


def _context() -> HardwareContext:
    return HardwareContext(
        device_id="cam",
        display_name="Shared camera",
        device_class="camera",
        transport=TransportRef(kind="mock"),
        channels=(_channel(COLOR), _channel(DEPTH)),
        provenance=ContextProvenance(source=ContextSource.DISCOVERED.value),
    )


@pytest.fixture
def cameras(monkeypatch: pytest.MonkeyPatch) -> tuple[HardwareRegistry, list[str], list[_Camera]]:
    """A loaded registry whose every transport build yields a fresh recorded camera."""
    log: list[str] = []
    built: list[_Camera] = []
    context = _context()

    class _Provider:
        kind = "fake"

        def discover(self) -> tuple[HardwareContext, ...]:
            return (context,)

    def _build(kind: str, config: Any) -> _Camera:
        camera = _Camera(log, close_delay_s=0.05)
        built.append(camera)
        return camera

    monkeypatch.setattr(registry_module, "build_transport", _build)
    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(_Provider(),))
    registry.load()
    return registry, log, built


async def test_the_transport_closes_only_when_the_last_holder_leaves(cameras: Any) -> None:
    registry, log, _ = cameras
    registry.hold("cam", "preview:color")
    registry.hold("cam", "lhp:a:color")
    registry.hold("cam", "lhp:a:color")  # idempotent per holder
    await registry.read_frame("cam", COLOR)

    assert registry.holders("cam") == ("lhp:a:color", "preview:color")
    assert await registry.release_hold("cam", "preview:color") is False
    assert "close:start" not in log, "a remaining holder lost its camera"

    assert await registry.release_hold("cam", "lhp:a:color") is True
    assert log == ["open", "close:start", "close:end"]
    assert registry.holders("cam") == ()


async def test_a_hold_taken_while_a_release_waits_keeps_the_device(cameras: Any) -> None:
    """The check is repeated under ``device_io``: a capture queued ahead wins."""
    registry, log, _ = cameras
    registry.hold("cam", "preview:color")
    await registry.read_frame("cam", COLOR)

    async with registry.device_io("cam"):
        release = asyncio.create_task(registry.release_hold("cam", "preview:color"))
        await asyncio.sleep(0)  # the release is now parked on the device lock
        registry.hold("cam", "lhp:b:depth")

    assert await release is False
    assert "close:start" not in log


async def test_a_capture_racing_a_release_never_sees_a_closed_camera(cameras: Any) -> None:
    """The skill race: lookup and capture under one lock, so a release cannot split them."""
    registry, log, _ = cameras
    registry.hold("cam", "preview:color")
    await registry.read_frame("cam", COLOR)

    async def _skill_capture() -> FrameReading:
        async with registry.device_io("cam"):
            transport = await registry.transport("cam")
            await asyncio.sleep(0.01)  # the window a release used to land in
            return await transport.read_frame(COLOR)

    results = await asyncio.gather(
        _skill_capture(), registry.release_hold("cam", "preview:color"), _skill_capture()
    )
    assert isinstance(results[0], FrameReading)
    assert isinstance(results[2], FrameReading)


async def test_a_close_in_progress_blocks_a_concurrent_reopen(cameras: Any) -> None:
    """Two links to one camera at once is a 'device busy' failure on real hardware."""
    registry, log, built = cameras
    await registry.transport("cam")

    drop = asyncio.create_task(registry.drop_transport("cam"))
    await asyncio.sleep(0)  # close has started and is stalled
    reopened = await registry.transport("cam")
    await drop

    assert log == ["open", "close:start", "close:end", "open"]
    assert reopened is built[1]


async def test_a_preview_release_spares_a_device_another_channel_holds(cameras: Any) -> None:
    registry, log, _ = cameras
    broker = registry.preview_broker
    await broker.frame("cam", COLOR, viewer_id="board")
    await broker.frame("cam", DEPTH, viewer_id="board")

    await broker.release("cam", COLOR, viewer_id="board")
    assert "close:start" not in log
    assert registry.holders("cam") == (f"preview:{DEPTH}",)

    await broker.release("cam", DEPTH, viewer_id="board")
    assert log[-1] == "close:end"
    await broker.close()


async def test_a_preview_release_spares_a_device_a_remote_reader_holds(cameras: Any) -> None:
    registry, log, _ = cameras
    broker = registry.preview_broker
    registry.hold("cam", "lhp:remote:color")
    await broker.frame("cam", COLOR, viewer_id="board")

    await broker.release("cam", COLOR, viewer_id="board")
    assert "close:start" not in log, "a local preview closed a camera a remote host reads"
    await broker.close()


async def test_lhp_connections_hold_independently_and_disconnect_releases(cameras: Any) -> None:
    registry, log, _ = cameras
    server = LhpNodeServer(registry, node_id="jetson", write_mode="trusted_v1")
    host, port = await server.start(host="127.0.0.1", port=0)
    endpoint = f"tcp://{host}:{port}"
    first, second = SocketLhpClient(endpoint), SocketLhpClient(endpoint)
    try:
        await first.connect()
        await second.connect()
        params = {"device_id": "cam", "channel_id": COLOR}
        await first.request("read_frame", params)
        await second.request("read_frame", params)
        assert len(registry.holders("cam")) == 2

        released = await first.request("release_frame", params)
        assert released == {
            "device_id": "cam",
            "channel_id": COLOR,
            "released": True,
            "device_closed": False,
        }
        assert "close:start" not in log, "one host's release closed another host's camera"
        assert (await second.request("read_frame", params))["data_b64"]

        # The second host vanishes without releasing: its connection's holds go with it.
        await second.close()
        for _ in range(50):
            if log and log[-1] == "close:end":
                break
            await asyncio.sleep(0.02)
        assert log[-1] == "close:end", "a dropped connection left the camera on"
        assert registry.holders("cam") == ()
    finally:
        await first.close()
        await second.close()
        await server.stop()

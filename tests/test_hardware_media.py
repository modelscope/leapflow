"""Media devices: discovery, the frame protocol, the privacy gate, and the preview lease.

Hermetic by construction. Nothing here opens a camera, a microphone or a display: the
capture path is exercised through a fake frame source, and the enumeration path through
the platform tables with the real binaries absent. That is a requirement rather than a
convenience -- on macOS, opening a capture device raises a system permission dialog, and
a test suite that does so mid-run is worse than the coverage it buys.

Four contracts are load-bearing and each is here because its absence was a real defect
or would be one:

1. A camera read is *not* SAFE. Effect class cannot distinguish a thermometer from a
   webcam, so the declared ``PrivacyTier`` decides -- and before it did, anything that
   could name the device could turn it on.
2. The gate fails closed on absence and on exception, and *before* the transport is
   touched, because touching it is what raises the platform dialog.
3. A frame channel is never sampled into stored history, and a read of one never
   returns bytes.
4. A preview releases the device on silence. A browser tab closing is not an event the
   daemon can observe, so nothing else would ever power the camera down.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    PrivacyTier,
    Representation,
    TransportRef,
)
from leapflow.hardware.media import MediaDevice, _jpeg_size, enumerate_devices
from leapflow.hardware.providers.media_provider import (
    FRAME_CHANNEL,
    LEVEL_CHANNEL,
    MediaContextProvider,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.risk import build_risk_classifier
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    FrameTransport,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.transports.media import MediaTransport
from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.risk import RiskLevel

# A minimal but structurally valid JPEG: SOI, an SOF0 frame declaring 4x2, then EOI.
# Real bytes rather than a placeholder because the size parser reads the SOF marker,
# and a fake would let a broken parser pass.
_JPEG = (
    b"\xff\xd8"
    b"\xff\xc0\x00\x11\x08\x00\x02\x00\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    b"\xff\xd9"
)


class _FakeFrameSource:
    """A transport that satisfies both ``HardwareTransport`` and ``FrameTransport``."""

    kind = "fake_media"

    def __init__(self) -> None:
        self.opens = 0
        self.closes = 0
        self.grabs = 0

    async def open(self, context: HardwareContext) -> TransportStatus:
        self.opens += 1
        return TransportStatus(connected=True)

    async def close(self) -> TransportStatus:
        self.closes += 1
        return TransportStatus(connected=False)

    async def probe(self) -> TransportStatus:
        return TransportStatus(connected=True)

    async def halt(self) -> TransportStatus:
        return TransportStatus(connected=True, halt_supported=False)

    async def read(self, channel_id: str) -> Reading:
        return Reading(device_id="cam", channel_id=channel_id, value="frame")

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        return WriteOutcome(ok=False, side_effect_state=SIDE_EFFECT_NONE, error="read-only")

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0
    ) -> FrameReading:
        self.grabs += 1
        return FrameReading(
            device_id="cam", channel_id=channel_id, data=_JPEG,
            width=4, height=2, sequence=self.grabs,
        )


class _PlainTransport(_FakeFrameSource):
    """The same transport without ``read_frame``, to prove the capability check bites."""

    read_frame = None  # type: ignore[assignment]


def _camera_context(*, transport_kind: str = "mock", fps: float = 2.0) -> HardwareContext:
    """A camera declaration.

    The declared transport kind is a *registered* one because admission rule V4 rejects
    a declaration naming a transport it cannot build -- and the tests below then replace
    the built transport with a fake, so which registered kind it names is irrelevant.
    Registering a fake kind in the process-global table instead would change what every
    later test admits.
    """
    return HardwareContext(
        device_id="cam",
        display_name="Fake camera",
        device_class="camera",
        transport=TransportRef(kind=transport_kind),
        channels=(
            Channel(
                channel_id=FRAME_CHANNEL,
                direction=Direction.READ.value,
                quantity="image_frame",
                effect=HardwareEffect.READ.value,
                sample_rate_hz=fps,
                representation=Representation.FRAME.value,
                media_type="image/jpeg",
                privacy=PrivacyTier.ENVIRONMENT.value,
            ),
        ),
        provenance=ContextProvenance(source=ContextSource.DISCOVERED.value),
    )


def _registry(context: HardwareContext, transport: Any, **settings: Any) -> HardwareRegistry:
    """Build a loaded registry whose transport factory returns *transport*."""

    class _Provider:
        kind = "fake"

        def discover(self) -> tuple[HardwareContext, ...]:
            return (context,)

    registry = HardwareRegistry(
        HardwareSettings(enabled=True, **settings), providers=(_Provider(),)
    )
    registry.load()
    # Patched rather than registered in the global table: the table is process-global,
    # and a test that leaked a row into it would change what every later test admits.
    async def _transport(device_id: str) -> Any:
        return transport

    registry.transport = _transport  # type: ignore[method-assign]
    return registry


class _DenyingGate:
    async def evaluate(self, descriptor: ActionDescriptor) -> Any:
        class _Result:
            approved = False
            denial_message = "You did not approve viewing the camera."

        return _Result()


class _RaisingGate:
    async def evaluate(self, descriptor: ActionDescriptor) -> Any:
        raise RuntimeError("gate exploded")


class _AllowingGate:
    def __init__(self) -> None:
        self.seen: list[ActionDescriptor] = []

    async def evaluate(self, descriptor: ActionDescriptor) -> Any:
        self.seen.append(descriptor)

        class _Result:
            approved = True

        return _Result()


# ════════════════════════════════════════════════════════════════
# Declaration: what the media provider produces
# ════════════════════════════════════════════════════════════════


def test_enumerated_device_ids_survive_admission() -> None:
    """A non-ASCII device name must still produce an admissible id.

    ``str.isalnum`` is true for CJK, so a macOS machine reporting "MacBook Pro相机"
    produced an id that looked clean and was rejected by admission rule V2 on every
    startup, with the camera simply absent and no obvious reason.
    """
    device = MediaDevice(
        kind="camera", index=1, name="MacBook Pro相机", spec="1:none", input_format="avfoundation"
    )
    assert device.device_id == "camera_1_macbook_pro"
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    assert set(device.device_id) <= allowed

    # A name with no ASCII at all still yields a usable, unique id.
    only_cjk = MediaDevice(kind="camera", index=2, name="相机", spec="2:none", input_format="avfoundation")
    assert only_cjk.device_id == "camera_2"
    assert set(only_cjk.device_id) <= allowed


def test_enumeration_opens_nothing_and_tolerates_a_bare_platform(monkeypatch) -> None:
    """Discovery must work with no capture binary and no devices present.

    The provider contract says discovery cannot connect to a device; here it cannot
    even find the tool that would. An empty tuple is the correct answer -- admission
    rejects a channel-less context, and reporting one would blame the user for an
    unsupported platform.
    """
    import sys as _sys

    monkeypatch.setattr("leapflow.hardware.media._ffmpeg_binary", lambda: "")
    monkeypatch.setattr(_sys, "platform", "darwin")
    assert enumerate_devices() == ()
    assert MediaContextProvider({}).discover() == ()


def test_screens_are_excluded_by_default_and_classified_as_personal(monkeypatch) -> None:
    """A display is a different disclosure from a webcam, and is off by default.

    A platform that presents the screen as just another video input would otherwise put
    "stream this person's screen" on the board beside the camera, one click away.
    """
    devices = (
        MediaDevice(kind="camera", index=0, name="Webcam", spec="0:none", input_format="avfoundation"),
        MediaDevice(kind="screen", index=1, name="Capture screen 0", spec="1:none", input_format="avfoundation"),
    )
    import sys as _sys

    monkeypatch.setattr("leapflow.hardware.media._enumerate_darwin", lambda: devices)
    monkeypatch.setitem(
        __import__("leapflow.hardware.media", fromlist=["_ENUMERATORS"])._ENUMERATORS,
        "darwin",
        lambda: devices,
    )
    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setattr("leapflow.hardware.media.available_backends", lambda: ("ffmpeg",))

    assert [d.kind for d in enumerate_devices()] == ["camera"]
    assert {d.kind for d in enumerate_devices({"include_screens": True})} == {"camera", "screen"}

    contexts = MediaContextProvider({"include_screens": True}).discover()
    tiers = {c.device_class: c.channels[0].privacy for c in contexts}
    assert tiers["camera"] == PrivacyTier.ENVIRONMENT.value
    assert tiers["screen"] == PrivacyTier.PERSONAL.value


def test_a_frame_channel_is_never_sampled_but_keeps_its_ceiling() -> None:
    """The rate on a media channel is a capture ceiling, not a sampling cadence.

    Read as a cadence it would build a loop appending every JPEG to the raw NDJSON
    segment and asking the downsampler for the mean of an image.
    """
    channel = _camera_context(fps=5.0).channels[0]
    assert channel.is_media is True
    assert channel.is_streaming is False
    assert channel.max_frame_rate_hz == 5.0

    registry = _registry(_camera_context(fps=5.0), _FakeFrameSource(), stream_enabled=True)
    assert registry.stream_sources() == ()

    # And a level channel on the same device *is* an ordinary scalar.
    level = Channel(
        channel_id=LEVEL_CHANNEL,
        quantity="audio_level",
        unit="dBFS",
        sample_rate_hz=1.0,
        envelope=Envelope(declared=True, min_value=-90.0, max_value=0.0),
        privacy=PrivacyTier.ENVIRONMENT.value,
    )
    assert level.is_media is False
    assert level.is_streaming is True


def test_a_writable_frame_channel_is_demoted_with_a_reason() -> None:
    """V9: frames are an observation, and a demotion must be reported, not silent."""
    context = HardwareContext(
        device_id="cam",
        transport=TransportRef(kind="mock"),
        halt_supported=True,
        channels=(
            Channel(
                channel_id=FRAME_CHANNEL,
                direction=Direction.READWRITE.value,
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(declared=True, min_value=0.0, max_value=1.0),
                representation=Representation.FRAME.value,
                privacy=PrivacyTier.ENVIRONMENT.value,
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )
    registry = _registry(context, _FakeFrameSource())
    admitted = registry.context("cam")
    assert admitted is not None
    assert admitted.channels[0].is_writable is False
    assert admitted.channels[0].privacy == PrivacyTier.ENVIRONMENT.value, (
        "a demotion revokes one capability; it must not reclassify the value and ungate it"
    )
    assert any(note.rule == "V9" for note in registry.report.notes)


# ════════════════════════════════════════════════════════════════
# The privacy gate
# ════════════════════════════════════════════════════════════════


def test_a_privacy_gated_read_is_not_safe_but_an_ordinary_one_is() -> None:
    """The declared tier decides, so a thermometer is still free to read."""
    registry = _registry(_camera_context(), _FakeFrameSource())
    classifier = build_risk_classifier(registry)

    camera = classifier.assess(
        ActionDescriptor.device(
            kind=ActionKind.DEVICE_READ.value, device_id="cam", channel_id=FRAME_CHANNEL
        )
    )
    assert camera.level is not RiskLevel.SAFE
    assert "privacy_disclosure" in camera.reasons
    assert camera.allow_permanent is False, (
        "a standing, unexpirable grant to observe somebody's room is not consent"
    )

    plain = classifier.assess(
        ActionDescriptor.device(
            kind=ActionKind.DEVICE_READ.value, device_id="cam", channel_id="nonexistent"
        )
    )
    assert plain.level is RiskLevel.SAFE, (
        "an unresolvable read must fail in the transport with a precise error, not at "
        "the gate with a prompt nobody can act on"
    )


@pytest.mark.parametrize(
    "gate,label",
    [(None, "no gate installed"), (_RaisingGate(), "a gate that raises"), (_DenyingGate(), "a denying gate")],
)
def test_a_privacy_gated_read_fails_closed_without_touching_the_device(gate, label) -> None:
    """Every uncertain case denies, and denies *before* the transport is reached.

    The ordering is the point on macOS: opening the device is what raises the system
    permission dialog, so a refused read must never get that far. ``opens == 0`` is the
    assertion that proves it.
    """
    source = _FakeFrameSource()
    registry = _registry(_camera_context(), source)
    tools = HardwareTools(registry, gate=gate, session_id="s1")

    result = asyncio.run(tools.hw_read(device_id="cam", channel_id=FRAME_CHANNEL))

    assert result["ok"] is False, label
    assert result["failure_code"] == "consent_required", label
    assert result["error"], "a refusal must say why, so the caller can act on it"
    assert source.opens == 0, f"{label}: the device was opened despite the refusal"
    assert source.grabs == 0, f"{label}: a frame was captured despite the refusal"


def test_an_approved_read_captures_once_and_leaks_no_bytes() -> None:
    """With consent the read proceeds, and the payload never reaches the tool result.

    Asserted through the tool path, so what is covered is what the *tool* guarantees for
    any transport: the gate was consulted with the declared tier, exactly one capture
    happened, and no image bytes appear in the result. The shape of a frame channel's
    reported value is a transport contract and is asserted directly below, against the
    real ``MediaTransport`` rather than the fake -- asserting it here would only have
    tested the fake's own return value.
    """
    source = _FakeFrameSource()
    gate = _AllowingGate()
    registry = _registry(_camera_context(), source)
    tools = HardwareTools(registry, gate=gate, session_id="s1")

    result = asyncio.run(tools.hw_read(device_id="cam", channel_id=FRAME_CHANNEL))

    assert result["ok"] is True
    assert _JPEG.hex() not in str(result), "raw frame bytes reached a tool result"
    assert _JPEG not in str(result).encode("utf-8", "ignore")

    # The descriptor the gate saw must carry the tier, so the prompt can explain itself.
    assert gate.seen and gate.seen[0].metadata.get("privacy") == PrivacyTier.ENVIRONMENT.value
    assert gate.seen[0].kind == ActionKind.DEVICE_READ.value


def test_reading_a_frame_channel_reports_metadata_instead_of_the_image() -> None:
    """``MediaTransport.read`` answers identity and size, never bytes.

    A ``Reading`` is appended to raw NDJSON segments and handed to models, so a few
    hundred kilobytes of JPEG in one is a cost with no benefit. Bytes are available only
    through ``read_frame``, which the preview path uses.

    Exercised against the real transport with a fake *grabber* injected, so the code
    under test is the transport's own read path and no device is opened.
    """
    transport = MediaTransport(
        {"kind": "camera", "index": 0, "spec": "0:none", "input_format": "avfoundation"}
    )

    class _Grabber:
        async def start(self) -> None:
            return None

        async def grab(self) -> tuple[bytes, int, int]:
            return _JPEG, 4, 2

        async def stop(self) -> None:
            return None

    async def _run() -> None:
        await transport.open(_camera_context())
        transport._grabber = _Grabber()  # noqa: SLF001 - injecting the capture backend

        reading = await transport.read(FRAME_CHANNEL)
        assert str(reading.value).startswith("frame:")
        assert "4x2" in str(reading.value)
        assert _JPEG not in str(reading.to_dict()).encode("utf-8", "ignore")

        # And the bytes are reachable, but only through the frame protocol.
        frame = await transport.read_frame(FRAME_CHANNEL)
        assert frame.data == _JPEG
        assert (frame.width, frame.height) == (4, 2)
        await transport.close()

    asyncio.run(_run())


# ════════════════════════════════════════════════════════════════
# The preview lease
# ════════════════════════════════════════════════════════════════


def test_the_declared_ceiling_caps_how_often_the_device_is_asked() -> None:
    """Several viewers, or one over-eager page, must cost what a single viewer costs."""
    source = _FakeFrameSource()
    registry = _registry(_camera_context(fps=2.0), source)

    async def _run() -> None:
        broker = registry.preview_broker
        for _ in range(5):
            frame = await broker.frame("cam", FRAME_CHANNEL)
            assert frame.data == _JPEG
        assert source.grabs == 1, "the 2 Hz ceiling did not collapse five rapid requests"
        assert broker.active()[0]["frames_served"] == 5
        await broker.close()

    asyncio.run(_run())


def test_a_preview_releases_the_device_when_nobody_is_watching() -> None:
    """A browser tab closing is not an event the daemon can observe.

    So the lease expires on silence, and dropping the transport is what actually powers
    the device down -- the registry caches it, so without that the camera stays claimed
    for the life of the process.
    """
    source = _FakeFrameSource()
    registry = _registry(_camera_context(), source, preview_idle_timeout_s=1.0)
    dropped: list[str] = []

    async def _drop(device_id: str) -> None:
        dropped.append(device_id)
        await source.close()

    registry.drop_transport = _drop  # type: ignore[method-assign]

    async def _run() -> None:
        broker = registry.preview_broker
        await broker.frame("cam", FRAME_CHANNEL)
        assert broker.active(), "the lease was not opened"
        await asyncio.sleep(3.5)
        assert broker.active() == (), "the idle lease was never swept"
        assert dropped == ["cam"], "the transport was not dropped, so the device stays claimed"
        assert source.closes >= 1

    asyncio.run(_run())


def test_a_transport_without_read_frame_is_refused_with_a_reason() -> None:
    """The capability check admission cannot make, made at the only honest moment.

    Building a transport is what reveals whether it implements the frame protocol, and
    admission must not build one -- so this is caught on first preview, with a message
    naming the mismatch rather than an AttributeError.
    """
    registry = _registry(_camera_context(transport_kind="mock"), _PlainTransport())

    async def _run() -> None:
        with pytest.raises(TransportError) as excinfo:
            await registry.preview_broker.frame("cam", FRAME_CHANNEL)
        assert excinfo.value.failure_code == "transport_not_frame_capable"
        assert "cannot produce frames" in str(excinfo.value)

    asyncio.run(_run())


def test_a_non_media_channel_cannot_be_previewed() -> None:
    """Previewing a thermometer is a caller error, reported as one."""
    context = HardwareContext(
        device_id="cam",
        transport=TransportRef(kind="mock"),
        channels=(Channel(channel_id="temperature", quantity="temperature", unit="degC"),),
        provenance=ContextProvenance(verified_by="tester"),
    )
    registry = _registry(context, _FakeFrameSource())

    async def _run() -> None:
        with pytest.raises(TransportError) as excinfo:
            await registry.preview_broker.frame("cam", "temperature")
        assert excinfo.value.failure_code == "channel_not_previewable"

    asyncio.run(_run())


# ════════════════════════════════════════════════════════════════
# Transport contracts the conformance suite cannot cover
# ════════════════════════════════════════════════════════════════


def test_the_media_transport_satisfies_the_frame_protocol_and_refuses_writes() -> None:
    """Read-only, and provably so: the call never reaches a device.

    ``SIDE_EFFECT_NONE`` rather than UNKNOWN matters -- an uncertain verdict would make
    recovery block replaying every later action over a call that never happened.
    """
    transport = MediaTransport({"kind": "camera", "index": 0, "spec": "0:none", "input_format": "avfoundation"})
    assert isinstance(transport, FrameTransport)

    outcome = asyncio.run(transport.write(FRAME_CHANNEL, 1))
    assert outcome.ok is False
    assert outcome.side_effect_state == SIDE_EFFECT_NONE
    assert outcome.failure_code == "media_channel_read_only"
    assert outcome.error


def test_operating_before_open_raises_rather_than_capturing() -> None:
    """A transport that captured before being opened would bypass the gate's ordering."""
    transport = MediaTransport({"kind": "camera", "spec": "0:none", "input_format": "avfoundation"})
    with pytest.raises(TransportError) as excinfo:
        asyncio.run(transport.read_frame(FRAME_CHANNEL))
    assert excinfo.value.failure_code == "transport_not_open"


def test_frame_dimensions_are_read_from_the_bytes_not_assumed() -> None:
    """The scaler rounds height to an even number, so the request is not the result.

    Reporting the requested width as the actual one would mislabel every frame.
    """
    assert _jpeg_size(_JPEG) == (4, 2)
    assert _jpeg_size(b"\xff\xd8\xff\xd9") is None, "a frame with no SOF marker has no size"


def test_frame_metadata_excludes_the_payload() -> None:
    """The only form that may reach a tool result, a log line, or a JSON-RPC reply."""
    frame = FrameReading(device_id="cam", channel_id=FRAME_CHANNEL, data=_JPEG, width=4, height=2)
    metadata = frame.to_metadata()
    assert metadata["size_bytes"] == len(_JPEG)
    assert "data" not in metadata
    assert _JPEG not in repr(metadata).encode("utf-8", "ignore")


# ── Capture device rate negotiation ─────────────────────────────────────────
#
# A capture device negotiates its rate when it is *opened*, so the rate is an input
# option. Passing only the output rate left avfoundation asking for its default 29.97,
# which real webcams reject -- they offer exactly 15 and 30 -- and the refusal surfaces as
# "Input/output error", a message that reads like a permission problem. Every preview on a
# real Mac failed, and the diagnosis went to System Settings for a bug in argv.


def test_the_device_rate_is_an_input_option() -> None:
    from leapflow.hardware.media import FfmpegFrameGrabber, MediaDevice

    device = MediaDevice(
        kind="camera", index=0, name="Cam", spec="0:none",
        input_format="avfoundation", device_class="camera",
    )
    command = FfmpegFrameGrabber(device, max_width=640, quality=70, fps=2.0)._command()

    assert "-framerate" in command, "without this the device picks 29.97 and refuses"
    assert command.index("-framerate") < command.index("-i"), (
        "an input option must precede -i, or it configures the output instead of the device"
    )
    # The preview ceiling is not a device mode. 2 fps is what the *broker* enforces; asking
    # a camera to open at 2 fps fails the same way the default did.
    assert command[command.index("-framerate") + 1] != "2"
    assert "-r" in command and command.index("-r") > command.index("-i"), (
        "the pull rate stays an output option, which is what decimates the device's frames"
    )


def test_a_refused_rate_is_learned_from_the_devices_own_answer() -> None:
    """ffmpeg prints the modes it *does* have. That is a capability answer, not a failure."""
    from leapflow.hardware.media import _supported_rate

    refusal = (
        ": [avfoundation] Selected framerate (29.970030) is not supported by the device. | "
        "Supported modes: | 640x480@[15.000000 30.000000]fps | 1280x720@[15.000000 30.000000]fps"
    )

    assert _supported_rate(refusal, 30.0) == 30.0, "prefer the requested rate when offered"
    assert _supported_rate(refusal, 24.0) == 15.0, "otherwise the fastest mode at or below it"
    assert _supported_rate(refusal, 10.0) == 15.0, "nothing slow enough: take the slowest offered"


def test_only_a_rate_refusal_is_retried() -> None:
    """Every other failure must be reported. Retrying a permission denial at 15 fps is noise.

    This is the guard that keeps the fallback from becoming a generic retry loop that
    strobes the camera light while never saying what is wrong.
    """
    from leapflow.hardware.media import _supported_rate

    assert _supported_rate(": Operation not permitted", 30.0) is None
    assert _supported_rate(": Input/output error", 30.0) is None
    assert _supported_rate("", 30.0) is None
    # Right error, no modes listed: nothing to learn, so nothing to retry with.
    assert _supported_rate(": not supported by the device", 30.0) is None


# ── Bounded page profiles and consent families ──────────────────────────────


class _FpsFrameSource(_FakeFrameSource):
    """A current FrameTransport that records the bounded page profile it actually gets."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[int, int, float]] = []

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        self.requests.append((max_width, quality, fps))
        return await super().read_frame(channel_id, max_width=max_width, quality=quality)


def test_a_page_profile_is_clamped_before_it_reaches_the_encoder() -> None:
    """The page may choose a profile, never an unbounded device request."""
    source = _FpsFrameSource()
    registry = _registry(
        _camera_context(fps=30.0),
        source,
        preview_max_fps=12.0,
        preview_max_width=1280,
        preview_quality=85,
    )

    async def _run() -> None:
        await registry.preview_broker.frame(
            "cam", FRAME_CHANNEL, max_width=9999, quality=100, fps=1000.0
        )

    asyncio.run(_run())

    assert source.requests == [(1280, 85, 12.0)]


def test_legacy_frame_transport_remains_usable_without_the_optional_fps_keyword() -> None:
    """The frame side protocol predated page profiles; native cadence is valid fallback."""
    source = _FakeFrameSource()
    registry = _registry(_camera_context(fps=12.0), source, preview_max_fps=12.0)

    async def _run() -> None:
        reading = await registry.preview_broker.frame("cam", FRAME_CHANNEL, fps=8.0)
        assert reading.data == _JPEG

    asyncio.run(_run())
    assert source.grabs == 1


def test_local_environment_cameras_share_a_session_consent_family_but_not_a_microphone() -> None:
    """One room, one camera-preview decision -- without widening it to microphone audio."""
    from dataclasses import replace

    from leapflow.daemon.service import _consent_for_observation

    camera_a = replace(_camera_context(fps=12.0), location="local")
    camera_b = replace(camera_a, device_id="desk_cam", display_name="Desk camera")
    microphone = HardwareContext(
        device_id="mic",
        display_name="Microphone",
        device_class="microphone",
        location="local",
        transport=TransportRef(kind="mock"),
        channels=(Channel(
            channel_id=LEVEL_CHANNEL,
            direction=Direction.READ.value,
            quantity="audio_level",
            unit="dBFS",
            effect=HardwareEffect.READ.value,
            privacy=PrivacyTier.ENVIRONMENT.value,
            representation=Representation.SCALAR.value,
        ),),
        provenance=ContextProvenance(source=ContextSource.DISCOVERED.value),
    )

    class _Registry:
        def context(self, device_id: str) -> HardwareContext | None:
            return {"cam": camera_a, "desk_cam": camera_b, "mic": microphone}.get(device_id)

    class _Gate:
        def __init__(self) -> None:
            self.actions: list[ActionDescriptor] = []

        async def evaluate(self, action: ActionDescriptor) -> Any:
            self.actions.append(action)
            return type("Result", (), {"approved": True})()

    gate = _Gate()
    ctx = type("Context", (), {"_approval_orchestrator": gate})()

    async def _run() -> None:
        assert await _consent_for_observation(ctx, _Registry(), "cam", FRAME_CHANNEL) is None
        assert await _consent_for_observation(ctx, _Registry(), "desk_cam", FRAME_CHANNEL) is None
        assert await _consent_for_observation(ctx, _Registry(), "mic", LEVEL_CHANNEL) is None

    asyncio.run(_run())

    first, second, third = gate.actions
    assert first.resource == second.resource == "observation:local-environment:camera"
    assert third.resource == "observation:local-environment:microphone"
    assert first.summary != second.summary, "the prompt/audit must still name the actual camera"
    assert first.metadata["device_id"] == "cam"
    assert second.metadata["device_id"] == "desk_cam"


def test_changing_page_profile_does_not_reuse_a_frame_encoded_for_the_old_profile() -> None:
    """A quality selector that serves the old cached JPEG is a selector that lies."""
    source = _FpsFrameSource()
    registry = _registry(
        _camera_context(fps=12.0),
        source,
        preview_max_fps=12.0,
        preview_max_width=1280,
        preview_quality=85,
    )

    async def _run() -> None:
        broker = registry.preview_broker
        economy = await broker.frame("cam", FRAME_CHANNEL, max_width=640, quality=60, fps=4)
        detail = await broker.frame("cam", FRAME_CHANNEL, max_width=1280, quality=85, fps=12)
        assert economy.sequence != detail.sequence

    asyncio.run(_run())
    assert source.requests == [(640, 60, 4.0), (1280, 85, 12.0)]


def test_one_session_consent_covers_two_local_cameras_but_not_the_microphone() -> None:
    """The actual grant path, not merely descriptor construction.

    A preview used to ask once for its probe, again for its MJPEG stream, and once more
    for the desk camera because each device/channel formed a distinct grant key. A person
    sees all three as one decision -- camera observation of the local environment -- so
    the reusable *session* key is the declared consent family. The audit still names each
    real device (covered in the descriptor test above); this checks the orchestrator
    actually reuses it.
    """
    from dataclasses import replace

    from leapflow.daemon.service import _consent_for_observation
    from leapflow.security.approval import ApprovalDecision
    from leapflow.security.orchestrator import ApprovalOrchestrator

    camera_a = replace(_camera_context(fps=12.0), location="local")
    camera_b = replace(camera_a, device_id="desk_cam", display_name="Desk camera")
    microphone = HardwareContext(
        device_id="mic",
        display_name="Microphone",
        device_class="microphone",
        location="local",
        transport=TransportRef(kind="mock"),
        channels=(Channel(
            channel_id=LEVEL_CHANNEL,
            direction=Direction.READ.value,
            quantity="audio_level",
            unit="dBFS",
            effect=HardwareEffect.READ.value,
            privacy=PrivacyTier.ENVIRONMENT.value,
            representation=Representation.SCALAR.value,
        ),),
        provenance=ContextProvenance(source=ContextSource.DISCOVERED.value),
    )

    class _Provider:
        kind = "fake"

        def discover(self) -> tuple[HardwareContext, ...]:
            return camera_a, camera_b, microphone

    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(_Provider(),))
    registry.load()

    class _Gate:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def request_approval(self, request: Any) -> ApprovalDecision:
            self.requests.append(request)
            return ApprovalDecision.ALLOW_SESSION

    gate = _Gate()
    ctx = type("Context", (), {
        "_approval_orchestrator": ApprovalOrchestrator(
            gate, risk_classifier=build_risk_classifier(registry),
        ),
    })()

    async def _run() -> None:
        assert await _consent_for_observation(ctx, registry, "cam", FRAME_CHANNEL) is None
        assert await _consent_for_observation(ctx, registry, "desk_cam", FRAME_CHANNEL) is None
        assert await _consent_for_observation(ctx, registry, "mic", LEVEL_CHANNEL) is None

    asyncio.run(_run())

    assert len(gate.requests) == 2
    assert gate.requests[0].action.resource == "observation:local-environment:camera"
    assert gate.requests[1].action.resource == "observation:local-environment:microphone"

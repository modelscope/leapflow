"""Media provider: declares each local camera and microphone as its own device.

One device per physical instrument, unlike the host provider's single namespaced
device. The reason is governance rather than tidiness: consent is granted per device,
and a single ``media`` device holding every camera and microphone would make one
grant cover all of them at once.

Every channel this provider declares is privacy-gated, which is the whole point of
the tier existing. A camera read has no physical effect -- ``HardwareEffect.READ``,
no envelope, nothing to actuate -- and is still the most consequential operation
available on the machine. Effect class cannot express that, so ``PrivacyTier`` does:
these channels are enumerated, described, and visible on the board, and a read is
refused until somebody with a strong identity says otherwise.

Enumeration never opens a device. That is what keeps a system consent dialog from
appearing at daemon startup, which nobody asked for and which a background process
cannot explain.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

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
from leapflow.hardware.media import (
    CAMERA,
    DEFAULT_FPS,
    MICROPHONE,
    SCREEN,
    MediaDevice,
    available_backends,
    enumerate_devices,
)

logger = logging.getLogger(__name__)

FRAME_CHANNEL = "frame"
LEVEL_CHANNEL = "level.dbfs"

_PRIVACY_BY_KIND = {
    CAMERA: PrivacyTier.ENVIRONMENT.value,
    MICROPHONE: PrivacyTier.ENVIRONMENT.value,
    # A display shows what the person is doing, not what is around the machine. The
    # distinction is not cosmetic: it is the difference between "observes the room" and
    # "observes you", and the consent prompt says which.
    SCREEN: PrivacyTier.PERSONAL.value,
}


class MediaContextProvider:
    """Supplies one context per enumerated camera, microphone or display."""

    kind = "media"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return a context per enumerated device. Opens nothing, never raises."""
        devices = enumerate_devices(self._config)
        if not devices:
            return ()
        backends = available_backends()
        return tuple(self._context_for(device, backends) for device in devices)

    def _context_for(self, device: MediaDevice, backends: tuple[str, ...]) -> HardwareContext:
        fps = _as_float(self._config.get("max_fps"), DEFAULT_FPS)
        return HardwareContext(
            device_id=device.device_id,
            display_name=device.name,
            device_class=device.device_class or device.kind,
            transport=TransportRef(
                kind="media",
                config={
                    "kind": device.kind,
                    "index": device.index,
                    "name": device.name,
                    "spec": device.spec,
                    "input_format": device.input_format,
                    "backend": str(self._config.get("backend") or ""),
                },
            ),
            channels=(_channel_for(device, fps),),
            vendor="",
            model=device.name,
            location="local",
            # Honest, and consequential: admission rule V5 reads this and refuses to
            # expose any writable channel. There is nothing writable here, so the
            # effect is belt-and-braces -- but a future exposure control could not
            # become commandable without a declaration a human verified.
            halt_supported=False,
            notes=_notes(device, backends),
            provenance=ContextProvenance(
                source=ContextSource.DISCOVERED.value,
                notes=(
                    f"Enumerated from this host as {device.input_format} input "
                    f"{device.spec!r}. Reads disclose the physical surroundings and "
                    "require consent."
                ),
            ),
        )


def _channel_for(device: MediaDevice, fps: float) -> Channel:
    """Build the single channel a media device exposes.

    A camera or display carries frames; a microphone carries a level. Both are reads,
    both are privacy-gated, and neither is sampled into stored history -- for the frame
    because ``Channel.is_streaming`` excludes media, and for the level because a
    continuously recorded loudness trace of somebody's room is exactly the kind of
    ambient surveillance this subsystem must not create by default.
    """
    if device.kind == MICROPHONE:
        return Channel(
            channel_id=LEVEL_CHANNEL,
            direction=Direction.READ.value,
            quantity="audio_level",
            unit="dBFS",
            effect=HardwareEffect.READ.value,
            envelope=Envelope(declared=True, min_value=-90.0, max_value=0.0),
            description=f"Instantaneous input level on {device.name}.",
            representation=Representation.SCALAR.value,
            privacy=_PRIVACY_BY_KIND.get(device.kind, PrivacyTier.ENVIRONMENT.value),
        )
    return Channel(
        channel_id=FRAME_CHANNEL,
        direction=Direction.READ.value,
        quantity="image_frame",
        unit="",
        effect=HardwareEffect.READ.value,
        # A frame channel's rate is its *capture ceiling*, not a sampling cadence:
        # ``Channel.is_streaming`` is false for media, so no sampling loop is built and
        # nothing is written to the reading store. The preview path reads it as the
        # fastest it may ask the device for frames.
        sample_rate_hz=fps,
        description=f"Live frames from {device.name}.",
        representation=Representation.FRAME.value,
        media_type="image/jpeg",
        privacy=_PRIVACY_BY_KIND.get(device.kind, PrivacyTier.ENVIRONMENT.value),
    )


def _notes(device: MediaDevice, backends: tuple[str, ...]) -> str:
    """Describe the device, and say plainly when nothing can capture from it.

    Reported in the declaration rather than discovered on first preview: a device that
    is listed but unpreviewable should say so where a person is already looking, not
    after they click.
    """
    if backends:
        return (
            f"{device.kind} at {device.input_format} input {device.spec!r}. "
            f"Capture backends available: {', '.join(backends)}."
        )
    return (
        f"{device.kind} at {device.input_format} input {device.spec!r}. No capture "
        "backend is installed, so this device can be described but not previewed; "
        "install ffmpeg or the opencv-python package."
    )


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def build_provider(config: Mapping[str, Any] | None = None) -> MediaContextProvider:
    """Factory registered in the provider table."""
    return MediaContextProvider(config)


__all__ = ["FRAME_CHANNEL", "LEVEL_CHANNEL", "MediaContextProvider", "build_provider"]

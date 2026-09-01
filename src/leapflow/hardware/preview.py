"""Preview broker: shared, bounded, self-releasing access to a media channel.

A preview is the one path in this subsystem where a device stays claimed across
requests, so it is also the one that needs an explicit lifetime. Three properties
matter, and each exists because its absence is a real failure:

**Shared upstream.** Two people watching the same camera must not open it twice --
most devices cannot be, and the second viewer would simply fail. Frames are captured
once per channel and handed to whoever asks, so a second viewer costs nothing.

**Self-releasing.** A camera that stays on after everybody stopped looking is exactly
the outcome the privacy tier exists to prevent, and a browser tab closing is not an
event the daemon can observe. So the lease expires on *silence*: no frame requested
within the idle timeout releases the device. Nothing has to remember to clean up.

**Rate-limited by the declaration.** The channel's declared capture ceiling caps how
often the device is actually asked, independently of how often the board asks. A page
left open in a background tab, or five viewers polling at once, cannot turn a 2 fps
preview into a spin loop.

The broker holds no approval logic. Consent is decided by the caller -- the tool path
or the daemon RPC -- through the ordinary orchestrator, because a component that
gated itself could not be audited and would be bypassed by every other route to the
same device.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from leapflow.hardware.transport import FrameReading, FrameTransport, TransportError

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_S = 15.0
"""Silence after which a previewed device is released."""

_SWEEP_INTERVAL_S = 2.0
"""How often idle leases are checked. Well under the shortest sensible timeout."""


@dataclass
class _Lease:
    """Live preview state for one channel."""

    device_id: str
    channel_id: str
    last_request_at: float = field(default_factory=time.monotonic)
    frames_served: int = 0
    # Cached so concurrent viewers within one capture interval share a frame rather
    # than each driving the device.
    last_frame: FrameReading | None = None
    last_capture_at: float = 0.0
    # The cached JPEG belongs to the encoder profile that produced it. Reusing a 640px
    # economy frame for a newly selected 1280px detail profile would make the selector a
    # lie for up to one capture interval, so profile is part of cache identity too.
    last_profile: tuple[float, int, int] | None = None


def _accepts_keyword(callable_: Any, name: str) -> bool:
    """Return whether a callable accepts *name* without executing it.

    The FrameTransport Protocol is structural and third-party implementations predate the
    optional cadence request. A missing optional keyword means "native cadence", not
    "cannot make frames". If reflection is unavailable, stay conservative and omit it;
    the broker's cached-frame interval still caps device work.
    """
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


class PreviewBroker:
    """Serves frames from media channels, releasing devices that nobody is watching."""

    def __init__(
        self,
        registry: Any,
        *,
        idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
        max_fps: float = 0.0,
        max_width: int = 0,
        quality: int = 0,
    ) -> None:
        self._registry = registry
        self._idle_timeout_s = max(1.0, float(idle_timeout_s))
        self._max_fps = max(0.0, float(max_fps))
        self._max_width = max(0, int(max_width))
        self._quality = max(0, int(quality))
        self._leases: dict[tuple[str, str], _Lease] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._sweeper: asyncio.Task[None] | None = None

    async def frame(
        self,
        device_id: str,
        channel_id: str,
        *,
        max_width: int = 0,
        quality: int = 0,
        fps: float = 0.0,
    ) -> FrameReading:
        """Return a frame, capturing at most once per declared interval.

        Raises ``TransportError`` with a structured ``failure_code`` for every refusal,
        so the caller can report a reason rather than a blank panel: an unknown device,
        a channel that carries no frames, or a transport that cannot produce them.
        """
        context = self._registry.context(device_id)
        if context is None:
            raise TransportError(f"unknown device {device_id!r}", failure_code="unknown_device")
        channel = context.channel(channel_id)
        if channel is None:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")
        if not channel.is_media:
            raise TransportError(
                f"channel {device_id}.{channel_id} does not carry frames "
                f"(representation={channel.representation!r})",
                failure_code="channel_not_previewable",
            )

        key = (device_id, channel_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            lease = self._leases.get(key)
            if lease is None:
                lease = _Lease(device_id=device_id, channel_id=channel_id)
                self._leases[key] = lease
                logger.info("Preview opened: %s.%s", device_id, channel_id)
            lease.last_request_at = time.monotonic()
            self._ensure_sweeper()

            # The declared ceiling and the runtime ceiling cap how often the device is
            # asked, regardless of how often the board does. Serving the cached frame is
            # what makes several viewers, or one over-eager page, cost the same as a
            # single one. The requested profile can *lower* cost but never raise it past
            # either ceiling.
            effective_fps = self._effective_fps(channel, fps)
            profile = (
                effective_fps,
                self._bounded_width(max_width),
                self._bounded_quality(quality),
            )
            interval = 1.0 / effective_fps if effective_fps > 0 else 0.0
            fresh_enough = (
                lease.last_frame is not None
                and lease.last_profile == profile
                and (time.monotonic() - lease.last_capture_at) < interval
            )
            if fresh_enough and lease.last_frame is not None:
                lease.frames_served += 1
                return lease.last_frame

            transport = await self._registry.transport(device_id)
            if not isinstance(transport, FrameTransport):
                # The capability check the admission rules cannot make: building a
                # transport is what reveals whether it implements the frame protocol,
                # and admission must not build one.
                raise TransportError(
                    f"the transport for {device_id!r} ({context.transport.kind!r}) cannot "
                    "produce frames, although the declaration says the channel carries them",
                    failure_code="transport_not_frame_capable",
                )
            kwargs = {
                "max_width": profile[1],
                "quality": profile[2],
            }
            # ``fps`` was added as an optional FrameTransport capability after the original
            # side protocol shipped. Existing third-party transports remain frame-capable:
            # they simply receive the already rate-limited call and keep their own native
            # cadence. Introspection is deliberate here -- catching TypeError would also
            # hide a TypeError *inside* a driver and call it compatibility.
            if _accepts_keyword(transport.read_frame, "fps"):
                kwargs["fps"] = effective_fps
            reading = await transport.read_frame(channel_id, **kwargs)
            lease.last_frame = reading
            lease.last_capture_at = time.monotonic()
            lease.last_profile = profile
            lease.frames_served += 1
            return reading

    def _effective_fps(self, channel: Any, requested: float) -> float:
        """Return a requested cadence bounded by declaration and runtime policy."""
        declared = max(0.0, float(getattr(channel, "max_frame_rate_hz", 0.0) or 0.0))
        choices = [limit for limit in (declared, self._max_fps) if limit > 0]
        hard_limit = min(choices) if choices else 0.0
        try:
            candidate = float(requested or hard_limit)
        except (TypeError, ValueError):
            candidate = hard_limit
        if not math.isfinite(candidate):
            candidate = hard_limit
        candidate = max(0.0, candidate)
        return min(candidate, hard_limit) if hard_limit > 0 else candidate

    def _bounded_width(self, requested: int) -> int:
        try:
            candidate = max(0, int(requested or self._max_width))
        except (TypeError, ValueError, OverflowError):
            candidate = self._max_width
        return min(candidate, self._max_width) if self._max_width > 0 else candidate

    def _bounded_quality(self, requested: int) -> int:
        try:
            candidate = max(1, min(100, int(requested or self._quality or 75)))
        except (TypeError, ValueError, OverflowError):
            candidate = self._quality or 75
        return min(candidate, self._quality) if self._quality > 0 else candidate

    def active(self) -> tuple[dict[str, Any], ...]:
        """Report live previews, so a board or `leap hw status` can show what is on."""
        now = time.monotonic()
        return tuple(
            {
                "device_id": lease.device_id,
                "channel_id": lease.channel_id,
                "frames_served": lease.frames_served,
                "idle_s": round(now - lease.last_request_at, 1),
            }
            for lease in sorted(self._leases.values(), key=lambda item: item.device_id)
        )

    def _ensure_sweeper(self) -> None:
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep(), name="hw-preview-sweeper")

    async def _sweep(self) -> None:
        """Release devices nobody has asked about. Runs only while leases exist."""
        try:
            while self._leases:
                await asyncio.sleep(_SWEEP_INTERVAL_S)
                cutoff = time.monotonic() - self._idle_timeout_s
                for key, lease in list(self._leases.items()):
                    if lease.last_request_at <= cutoff:
                        await self._release(key)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - the sweeper must never die silently
            logger.warning("Preview sweeper failed: %s", exc, exc_info=True)

    async def _release(self, key: tuple[str, str]) -> None:
        lease = self._leases.pop(key, None)
        if lease is None:
            return
        logger.info(
            "Preview released after %.0fs idle: %s.%s (%d frame(s) served)",
            self._idle_timeout_s,
            lease.device_id,
            lease.channel_id,
            lease.frames_served,
        )
        # Dropping the transport is what actually powers the device down: the registry
        # caches it, so without this the camera stays claimed for the life of the
        # process even though nothing is watching.
        if not any(existing == lease.device_id for existing, _ in self._leases):
            await self._registry.drop_transport(lease.device_id)

    async def close(self) -> None:
        """Release every preview. Registered as a teardown effect; never raises."""
        sweeper, self._sweeper = self._sweeper, None
        if sweeper is not None:
            sweeper.cancel()
        for key in list(self._leases):
            try:
                await self._release(key)
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.debug("preview release failed for %s: %s", key, exc)


__all__ = ["DEFAULT_IDLE_TIMEOUT_S", "PreviewBroker"]

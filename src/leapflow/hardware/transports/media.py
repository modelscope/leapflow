"""Media transport: local capture behind the six-method contract, plus frames.

Deliberately *not* named after a device. It is a generic mechanism -- local media
capture through a pluggable backend -- and it holds no knowledge of what a camera or
a microphone is: the device's kind, platform input format and input spec all arrive
from the declaration, and the capture backend is chosen from availability.

Satisfies ``HardwareTransport`` and additionally ``FrameTransport``, which is how the
frame capability is discovered. That is why frames are a side protocol rather than a
seventh core method: most devices will never have one, and a device declaring a frame
channel whose transport lacks ``read_frame`` is caught with a reason instead of
failing on first preview.

Opening is where consent becomes visible. ``open()`` starts the capture backend, which
on macOS is the moment the system permission dialog appears -- so it is deliberately
*not* called during discovery, admission, or a board refresh.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Mapping

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.media import (
    DEFAULT_FPS,
    DEFAULT_MAX_WIDTH,
    DEFAULT_QUALITY,
    MICROPHONE,
    MediaCaptureError,
    MediaDevice,
    available_backends,
    build_grabber,
    build_level_reader,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

logger = logging.getLogger(__name__)

_LEVEL_FLOOR_DBFS = -90.0
"""Reported level when a frame carries no signal, matching the declared envelope floor.

Digital silence is negative infinity in dBFS, which no chart and no envelope check can
handle, so the declared floor stands in for it.
"""

_LEVEL_READY_TIMEOUT_S = 3.0
"""How long a first level read waits for the device's first analysis window.

Generous against the ~1s a real microphone takes, because the alternative is telling the
caller "not measured yet" for a device that is working. Bounded because a device that
never produces a window is a device to report, not to wait on forever.
"""

_LEVEL_POLL_INTERVAL_S = 0.1
"""How often the first read re-checks for a measurement while waiting.

Cheap: the reader keeps the newest value in memory, so each check is a field read and
only the sleep is real.
"""


class MediaTransport:
    """Reads frames or levels from one local media device."""

    kind = "media"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._device = MediaDevice(
            kind=str(config.get("kind") or "camera"),
            index=int(config.get("index") or 0),
            name=str(config.get("name") or ""),
            spec=str(config.get("spec") or ""),
            input_format=str(config.get("input_format") or ""),
        )
        self._backend = str(config.get("backend") or "")
        self._max_width = int(config.get("max_width") or DEFAULT_MAX_WIDTH)
        self._quality = int(config.get("quality") or DEFAULT_QUALITY)
        self._fps = float(config.get("fps") or DEFAULT_FPS)
        self._context: HardwareContext | None = None
        self._connected = False
        self._grabber: Any = None
        self._level_reader: Any = None
        self._sequence = 0
        self._last_frame: FrameReading | None = None
        # One capture per transport, so two concurrent viewers of the same device share
        # the process rather than each claiming it -- a camera can only be opened once.
        self._lock = asyncio.Lock()

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Bind the declaration. Idempotent, and deliberately does *not* capture.

        Binding and capturing are separated because opening a camera is a
        privacy-relevant act that raises a system dialog: it must happen when somebody
        asks for a frame and consent has been established, not when the registry
        happens to resolve a transport. Reporting ``connected`` here is therefore about
        the declaration being usable, not about the device being claimed.
        """
        self._context = context
        self._connected = True
        return await self.probe()

    async def close(self) -> TransportStatus:
        await self._stop_grabber()
        await self._stop_level_reader()
        self._connected = False
        return TransportStatus(connected=False, halt_supported=False, detail="closed")

    async def probe(self) -> TransportStatus:
        backends = available_backends()
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail=(
                f"{self._device.kind} {self._device.spec!r} via {self._device.input_format}"
                if backends
                else "no capture backend installed (need ffmpeg or opencv-python)"
            ),
            metadata={
                "backends": list(backends),
                "capturing": self._grabber is not None,
                "measuring": self._level_reader is not None,
                "device_kind": self._device.kind,
                "input_format": self._device.input_format,
            },
        )

    async def halt(self) -> TransportStatus:
        """Stop capture and report that there is nothing to emergency-stop.

        A camera has no motion to arrest, so ``halt_supported`` is false -- but the
        capture is released anyway, because the closest thing to an emergency stop on
        an observation device is to stop observing.
        """
        await self._stop_grabber()
        await self._stop_level_reader()
        return TransportStatus(
            connected=self._connected,
            halt_supported=False,
            detail="capture released; a media device has no emergency stop",
        )

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        """Read the channel's scalar value.

        For a frame channel this returns *metadata*, never bytes: the value is the
        frame's identity and size, because a ``Reading`` is appended to NDJSON segments
        and handed to models, and a few hundred kilobytes of JPEG in either place is a
        cost with no benefit. Bytes come from ``read_frame``.
        """
        self._require_open(channel_id)
        channel = self._context.channel(channel_id) if self._context is not None else None
        if channel is None:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")

        if self._device.kind == MICROPHONE:
            value: Any = await self._read_level()
        else:
            frame = await self.read_frame(channel_id)
            value = f"frame:{frame.sequence}:{frame.width}x{frame.height}"

        self._sequence += 1
        return Reading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            value=value,
            quantity=channel.quantity,
            unit=channel.unit,
            sequence=self._sequence,
            quality=Quality.OK.value,
        )

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0, fps: float = 0.0
    ) -> FrameReading:
        """Capture one frame, starting the backend on first use.

        Satisfies ``FrameTransport``. Requested dimensions, quality and cadence are
        honoured by restarting the backend when they change, because FFmpeg fixes all
        three at start: silently returning the prior encoder's output would make the page
        report a selected profile while spending a different profile's compute budget.
        """
        self._require_open(channel_id)
        async with self._lock:
            await self._ensure_grabber(max_width=max_width, quality=quality, fps=fps)
            grabber = self._grabber
            if grabber is None:  # pragma: no cover - _ensure_grabber raises instead
                raise TransportError("capture unavailable", failure_code="capture_unavailable")
            try:
                data, width, height = await grabber.grab()
            except MediaCaptureError as exc:
                # The process is gone or unusable; drop it so the next attempt restarts
                # rather than grabbing from a dead pipe forever.
                await self._stop_grabber()
                raise TransportError(str(exc), failure_code=exc.failure_code) from exc

        self._sequence += 1
        reading = FrameReading(
            device_id=self._context.device_id if self._context is not None else "",
            channel_id=channel_id,
            data=data,
            media_type="image/jpeg",
            width=width,
            height=height,
            observed_at=float(getattr(grabber, "captured_at", 0.0) or time.time()),
            monotonic_at=float(getattr(grabber, "captured_monotonic", 0.0) or time.monotonic()),
            sequence=self._sequence,
            quality=Quality.OK.value,
        )
        self._last_frame = reading
        return reading

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Refuse every write, proving no effect landed.

        ``SIDE_EFFECT_NONE`` is provable rather than assumed: the call never reaches
        the device, so recovery may treat it as a clean refusal instead of an uncertain
        one that would block replaying every later action.
        """
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error=(
                f"media channel {channel_id!r} is read-only: a discovered declaration "
                "carries no operating envelope a person is accountable for"
            ),
            failure_code="media_channel_read_only",
        )

    # ── Internals ──

    def _require_open(self, channel_id: str) -> None:
        if not self._connected:
            raise TransportError(
                f"media transport is not open (channel {channel_id!r})",
                failure_code="transport_not_open",
            )

    async def _ensure_grabber(self, *, max_width: int, quality: int, fps: float = 0.0) -> None:
        width = max_width or self._max_width
        want_quality = quality or self._quality
        want_fps = max(0.1, float(fps or self._fps))
        if self._grabber is not None:
            if (width, want_quality, want_fps) == (self._max_width, self._quality, self._fps):
                return
            await self._stop_grabber()
        self._max_width, self._quality, self._fps = width, want_quality, want_fps
        try:
            grabber = build_grabber(
                self._device,
                backend=self._backend,
                max_width=width,
                quality=want_quality,
                fps=self._fps,
            )
            await grabber.start()
        except MediaCaptureError as exc:
            raise TransportError(str(exc), failure_code=exc.failure_code) from exc
        self._grabber = grabber

    async def _stop_grabber(self) -> None:
        grabber, self._grabber = self._grabber, None
        if grabber is None:
            return
        try:
            await grabber.stop()
        except Exception as exc:  # noqa: BLE001 - teardown must not propagate
            logger.debug("media capture stop failed: %s", exc)

    async def _read_level(self) -> float:
        """Return the input level in dBFS, starting the reader on first use.

        Started lazily for the same reason frame capture is: claiming a microphone is
        the moment a system permission dialog appears, and that must follow somebody
        asking for a level -- not the registry resolving a transport.

        The first reading after start can arrive before the first analysis window has
        closed, in which case there is no measurement yet. That is reported as such
        rather than as a number: zero dBFS is *maximum* signal, so substituting it for
        "not measured yet" would be the loudest possible lie, and the declared floor
        would look like a confirmed silent room.
        """
        async with self._lock:
            if self._level_reader is None:
                try:
                    reader = build_level_reader(self._device)
                    await reader.start()
                except MediaCaptureError as exc:
                    raise TransportError(str(exc), failure_code=exc.failure_code) from exc
                self._level_reader = reader
            try:
                level = await self._await_first_level(self._level_reader)
            except MediaCaptureError as exc:
                await self._stop_level_reader()
                raise TransportError(str(exc), failure_code=exc.failure_code) from exc
        if level != level:  # NaN: no window closed inside the deadline
            raise TransportError(
                "no input level has been measured yet; retry in a moment",
                failure_code="level_not_ready",
            )
        # Digital silence is negative infinity, which no chart or envelope check can
        # handle, so the declared floor stands in for it.
        return _LEVEL_FLOOR_DBFS if level == float("-inf") else round(level, 2)

    @staticmethod
    async def _await_first_level(reader: Any) -> float:
        """Poll until the reader has measured something, or the deadline passes.

        Waiting on the *evidence* rather than sleeping a guessed interval. A fixed sleep
        was a race the device won: a real microphone's first analysis window closed at
        roughly a second, the sleep was 0.6, so the first read always reported "not
        measured yet" -- and a client that treated that as failure never saw a level at
        all. Polling costs nothing (the value is already in memory; only the deadline is
        real) and it returns as soon as there is an answer.
        """
        deadline = asyncio.get_running_loop().time() + _LEVEL_READY_TIMEOUT_S
        while True:
            level = await reader.read()
            if level == level:  # not NaN
                return level
            if asyncio.get_running_loop().time() >= deadline:
                return level
            await asyncio.sleep(_LEVEL_POLL_INTERVAL_S)

    async def _stop_level_reader(self) -> None:
        reader, self._level_reader = self._level_reader, None
        if reader is None:
            return
        try:
            await reader.stop()
        except Exception as exc:  # noqa: BLE001 - teardown must not propagate
            logger.debug("media level reader stop failed: %s", exc)


def build_transport(config: Mapping[str, Any] | None = None) -> MediaTransport:
    """Factory registered in the transport table."""
    return MediaTransport(config)


__all__ = ["MediaTransport", "build_transport"]

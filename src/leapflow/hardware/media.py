# Copyright (c) Alibaba, Inc. and its affiliates.
"""Local media devices: enumeration and frame capture, behind one backend table.

The counterpart to ``host_metrics`` for media. Cameras and microphones are ordinary
Hardware Context Protocol devices whose channel set is *discovered*, so the same
split applies: this module owns the platform knowledge, the provider turns it into
declarations, and the transport reads through it.

Two seams, both ``name -> factory`` tables:

``_ENUMERATORS``
    How to *list* devices on this platform, without opening any of them. Listing must
    stay cheap and permission-free: it runs during daemon boot, and on macOS opening a
    camera raises a system consent dialog that nobody asked for.

``_BACKENDS``
    How to *capture* from one. ``ffmpeg`` streams MJPEG over a pipe; ``opencv`` uses
    ``cv2.VideoCapture``. Both are optional -- with neither installed, devices are
    still enumerated and still appear on the board, and a preview attempt reports
    which backend is missing instead of failing obscurely.

Capture is long-lived on purpose. Spawning a process per frame costs about a second
of startup, which makes a 2 fps preview impossible; one process producing an MJPEG
stream costs that once.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

CAMERA = "camera"
MICROPHONE = "microphone"
SCREEN = "screen"

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"

_DEFAULT_INPUT_RATE_HZ = 30.0
"""The rate a capture device is opened at until one tells us otherwise.

30 because it is the one mode essentially every webcam has. The value that matters is
that it is *set at all*: an unset rate lets avfoundation ask for 29.97, which devices
offering 15/30 refuse outright.
"""

_MODE_RATES_RE = re.compile(r"@\[([0-9.\s]+)\]fps")
"""Rates out of ffmpeg's ``640x480@[15.000000 30.000000]fps`` capability lines."""


def _supported_rate(detail: str, requested: float) -> float | None:
    """Return a rate the device says it has, or None when it did not say.

    Only consulted after a refusal, and only when the refusal is *about* the rate: any
    other failure (no permission, device busy, unplugged) must be reported, not retried
    at a different rate.

    Picks the highest offered rate at or below the requested one -- a slower mode is a
    working preview, a faster one may fail the same way -- and otherwise the slowest
    offered, which is the most likely to be supported.
    """
    if "not supported by the device" not in detail:
        return None
    rates = sorted(
        {
            value
            for group in _MODE_RATES_RE.findall(detail)
            for value in (float(token) for token in group.split() if token)
            if value > 0
        }
    )
    if not rates:
        return None
    affordable = [rate for rate in rates if rate <= requested]
    return affordable[-1] if affordable else rates[0]


ENUMERATION_TIMEOUT_S = 4.0
"""Ceiling on a listing subprocess.

Bounded because this runs during startup: a wedged ``system_profiler`` or a driver
that hangs on enumeration must cost a few seconds and a log line, never the daemon's
ability to start.
"""

DEFAULT_MAX_WIDTH = 320
DEFAULT_QUALITY = 60
DEFAULT_FPS = 2.0
"""Preview defaults, deliberately modest.

A preview is a glance at a device, not a video feed: 320px at 2 fps is a few tens of
kilobytes per second per viewer, which is affordable over a JSON-RPC-adjacent HTTP
hop and on a laptop battery. The ceiling is raised by configuration, never silently.
"""


@dataclass(frozen=True)
class _LinuxVideoCandidate:
    """Sysfs-only facts for one Linux video node, collected without opening it."""

    node_index: int
    stream_index: int
    name: str
    interface_name: str
    spec: str
    parent_key: str
    driver: str
    usb_id: str = ""


@dataclass(frozen=True)
class MediaDevice:
    """One enumerated local media device.

    ``spec`` is the platform's own input identifier (an AVFoundation index, a V4L2
    path, a DirectShow name). It is carried verbatim rather than reformatted so the
    capture backend hands the platform exactly what the platform named.
    ``usb_id`` is ``vid:pid`` (lowercase hex) when the platform publishes it.
    """

    kind: str
    index: int
    name: str
    spec: str
    input_format: str
    device_class: str = ""
    usb_id: str = ""

    @property
    def device_id(self) -> str:
        """Return a stable device id that admission will accept.

        Restricted to ASCII ``[a-z0-9_]`` because that is what admission rule V2
        requires -- the id becomes a dict key, a DuckDB value and a path fragment.
        ``str.isalnum`` is not the test to use here: it is true for CJK, so a
        macOS machine reporting "MacBook Pro相机" produced an id that looked clean and
        was rejected on every startup.

        The index leads the name for the same reason it is present at all: a machine
        with two identical cameras enumerates two devices with the same name, and it
        also means a device whose name is entirely non-ASCII still gets a usable id
        rather than an empty one.
        """
        slug = "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in self.name.lower())
        slug = "_".join(part for part in slug.split("_") if part)
        base = f"{self.kind}_{self.index}"
        return f"{base}_{slug}"[:64].rstrip("_") if slug else base


# ── Enumeration ──


def enumerate_devices(config: Mapping[str, Any] | None = None) -> tuple[MediaDevice, ...]:
    """List local media devices on this platform. Never opens one, never raises.

    Screen-capture devices are excluded unless asked for: a platform that presents the
    display as just another video input would otherwise put "stream this person's
    screen" on the board beside the webcam, at one click. It remains available
    (``include_screens``) because it is a legitimate capability -- it is the default
    that should not make the decision for anyone.

    ``exclude_usb`` lists USB identities (``vid`` or ``vid:pid``) whose video nodes a
    dedicated driver owns -- a depth camera's UVC endpoints, say. Listing those as
    generic cameras would put a second, competing opener of the same hardware on the
    board.
    """
    config = config or {}
    enumerator = _ENUMERATORS.get(sys.platform) or _ENUMERATORS.get("linux")
    if enumerator is None:
        return ()
    try:
        devices = enumerator()
    except Exception as exc:  # noqa: BLE001 - an unlistable platform is not a failure
        logger.debug("media enumeration failed on %s: %s", sys.platform, exc)
        return ()

    include_screens = bool(config.get("include_screens", False))
    include_microphones = bool(config.get("include_microphones", True))
    excluded = usb_exclusions(config.get("exclude_usb"))
    selected = []
    for device in devices:
        if device.kind == SCREEN and not include_screens:
            continue
        if device.kind == MICROPHONE and not include_microphones:
            continue
        if device.usb_id and any(
            device.usb_id == item or device.usb_id.startswith(f"{item}:") for item in excluded
        ):
            continue
        selected.append(device)
    return tuple(selected)


def usb_exclusions(raw: Any) -> tuple[str, ...]:
    """Normalise ``exclude_usb`` (comma string or list) to lowercase ``vid[:pid]`` items."""
    items = raw.split(",") if isinstance(raw, str) else list(raw or ())
    return tuple(str(item).strip().lower() for item in items if str(item).strip())


def _enumerate_darwin() -> tuple[MediaDevice, ...]:
    """List AVFoundation inputs by parsing ffmpeg's own device listing.

    ffmpeg is asked rather than ``system_profiler`` because the *index* is what the
    capture side needs, and only ffmpeg's listing agrees with ffmpeg's own indexing.
    The command deliberately fails (there is no input to open); the listing is on
    stderr, which is why a non-zero exit is expected and ignored.
    """
    binary = _ffmpeg_binary()
    if not binary:
        return ()
    output = _run(
        [
            binary,
            "-hide_banner",
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ]
    )
    devices: list[MediaDevice] = []
    section = ""
    for line in output.splitlines():
        if "AVFoundation video devices" in line:
            section = "video"
            continue
        if "AVFoundation audio devices" in line:
            section = "audio"
            continue
        match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if not match or not section:
            continue
        index, name = int(match.group(1)), match.group(2).strip()
        if section == "video":
            # "Capture screen N" is a display, not a camera. Classified here, in the
            # discovery layer, because mapping a platform's naming to a device class is
            # exactly what discovery is for -- nothing downstream branches on the name.
            is_screen = name.lower().startswith("capture screen")
            devices.append(
                MediaDevice(
                    kind=SCREEN if is_screen else CAMERA,
                    index=index,
                    name=name,
                    spec=f"{index}:none",
                    input_format="avfoundation",
                    device_class=SCREEN if is_screen else CAMERA,
                )
            )
        else:
            devices.append(
                MediaDevice(
                    kind=MICROPHONE,
                    index=index,
                    name=name,
                    spec=f":{index}",
                    input_format="avfoundation",
                    device_class=MICROPHONE,
                )
            )
    return tuple(devices)


def _enumerate_linux() -> tuple[MediaDevice, ...]:
    """List V4L2 cameras and ALSA capture cards from sysfs and procfs.

    No subprocess and no device open: the kernel already publishes both lists as files.
    UVC also publishes metadata-only companions as ``/dev/video*`` nodes; those are
    removed from the preview inventory using their driver and stream-index topology.
    """
    devices: list[MediaDevice] = list(
        _linux_video_devices(Path("/sys/class/video4linux"), Path("/dev"))
    )
    cards = Path("/proc/asound/cards")
    if cards.is_file():
        for line in cards.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*(\d+)\s+\[([^\]]+)\]", line)
            if not match:
                continue
            index, label = int(match.group(1)), match.group(2).strip()
            devices.append(
                MediaDevice(
                    kind=MICROPHONE,
                    index=index,
                    name=label,
                    spec=f"hw:{index}",
                    input_format="alsa",
                    device_class=MICROPHONE,
                )
            )
    return tuple(devices)


def _linux_video_devices(v4l: Path, dev_root: Path) -> tuple[MediaDevice, ...]:
    if not v4l.is_dir():
        return ()
    candidates: list[_LinuxVideoCandidate] = []
    for entry in sorted(v4l.iterdir()):
        node = dev_root / entry.name
        if not node.exists():
            continue
        node_index = int("".join(ch for ch in entry.name if ch.isdigit()) or 0)
        resolved = entry.resolve()
        candidates.append(
            _LinuxVideoCandidate(
                node_index=node_index,
                stream_index=_read_int(entry / "index", node_index),
                name=_read_text(entry / "name") or entry.name,
                interface_name=_read_text(resolved.parent.parent / "interface"),
                spec=str(node),
                parent_key=str(resolved.parent),
                driver=(resolved / "device" / "driver").resolve().name,
                usb_id=_usb_id(resolved),
            )
        )
    return _select_linux_video_candidates(tuple(candidates))


def _usb_id(node: Path) -> str:
    """``vid:pid`` of the USB device a sysfs video node hangs off, or ``""``."""
    for ancestor in list(node.parents)[:5]:
        vendor = _read_text(ancestor / "idVendor")
        if vendor:
            return f"{vendor}:{_read_text(ancestor / 'idProduct')}".lower()
    return ""


def _select_linux_video_candidates(
    candidates: tuple[_LinuxVideoCandidate, ...],
) -> tuple[MediaDevice, ...]:
    """Remove UVC metadata companions and give multi-stream inputs distinct labels.

    ``uvcvideo`` numbers a metadata node immediately after the capture node it
    accompanies. The rule is intentionally topology-bound: an odd-index node from any
    other driver, or one without its predecessor in the same interface, is retained.
    This avoids model-name special cases while filtering RealSense and ordinary UVC
    metadata endpoints using only permission-free sysfs evidence.
    """
    indexes_by_parent: dict[str, set[int]] = {}
    for candidate in candidates:
        indexes_by_parent.setdefault(candidate.parent_key, set()).add(candidate.stream_index)
    retained = tuple(
        candidate
        for candidate in candidates
        if not (
            candidate.driver == "uvcvideo"
            and candidate.stream_index % 2 == 1
            and candidate.stream_index - 1 in indexes_by_parent[candidate.parent_key]
        )
    )
    counts: dict[str, int] = {}
    for candidate in retained:
        counts[candidate.parent_key] = counts.get(candidate.parent_key, 0) + 1
    devices = []
    for candidate in retained:
        label = candidate.interface_name or candidate.name
        if counts[candidate.parent_key] > 1:
            label = f"{label} stream {candidate.stream_index}"
        devices.append(
            MediaDevice(
                kind=CAMERA,
                index=candidate.node_index,
                name=label,
                spec=candidate.spec,
                input_format="v4l2",
                device_class=CAMERA,
                usb_id=candidate.usb_id,
            )
        )
    return tuple(devices)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_int(path: Path, default: int) -> int:
    try:
        return int(_read_text(path))
    except ValueError:
        return default


def _enumerate_windows() -> tuple[MediaDevice, ...]:
    """List DirectShow inputs from ffmpeg's listing.

    DirectShow addresses devices by *name*, not index, so the spec carries the name
    and a rename makes the device a new one -- which is accurate: nothing else about
    it is stable.
    """
    binary = _ffmpeg_binary()
    if not binary:
        return ()
    output = _run([binary, "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"])
    devices: list[MediaDevice] = []
    counters = {CAMERA: 0, MICROPHONE: 0}
    for line in output.splitlines():
        match = re.search(r'"([^"]+)"\s*\((video|audio)\)', line)
        if not match:
            continue
        name, sort = match.group(1), match.group(2)
        kind = CAMERA if sort == "video" else MICROPHONE
        index = counters[kind]
        counters[kind] += 1
        devices.append(
            MediaDevice(
                kind=kind,
                index=index,
                name=name,
                spec=f"{sort}={name}",
                input_format="dshow",
                device_class=kind,
            )
        )
    return tuple(devices)


_ENUMERATORS: dict[str, Callable[[], tuple[MediaDevice, ...]]] = {
    "darwin": _enumerate_darwin,
    "linux": _enumerate_linux,
    "win32": _enumerate_windows,
}


# ── Capture ──


@runtime_checkable
class FrameGrabber(Protocol):
    """A live source of encoded frames from one device."""

    async def start(self) -> None:
        """Begin capturing. This is where a platform consent prompt appears."""
        ...

    async def grab(self) -> tuple[bytes, int, int]:
        """Return the newest encoded frame as ``(data, width, height)``."""
        ...

    async def stop(self) -> None:
        """Release the device. Must be idempotent and must not raise."""
        ...


class MediaCaptureError(RuntimeError):
    """Raised when capture cannot start or a frame cannot be produced.

    A plain exception subclass, never frozen: CPython assigns ``__traceback__`` on
    every re-raise, and a frozen type would replace the real failure with a complaint
    about that assignment.
    """

    def __init__(self, message: str, *, failure_code: str = "media_capture_error") -> None:
        super().__init__(message)
        self.failure_code = failure_code


class FfmpegFrameGrabber:
    """Capture MJPEG from a long-lived ffmpeg process.

    One process for the life of the preview, and frames are cut out of its stdout by
    scanning for JPEG start/end markers. The alternative -- one ``ffmpeg`` invocation
    per frame -- costs roughly a second of process startup each time, which is more
    than the whole interval of a 2 fps preview.
    """

    backend = "ffmpeg"

    def __init__(self, device: MediaDevice, *, max_width: int, quality: int, fps: float) -> None:
        self._device = device
        self._max_width = max(16, int(max_width))
        self._quality = min(100, max(1, int(quality)))
        self._fps = max(0.1, float(fps))
        self._process: Any = None
        self._buffer = bytearray()
        self._size = (0, 0)
        self._input_rate = _DEFAULT_INPUT_RATE_HZ
        self._latest: tuple[bytes, int, int] | None = None
        self._captured_at = 0.0
        self._captured_monotonic = 0.0
        self._condition = asyncio.Condition()
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_error: MediaCaptureError | None = None
        self._sync_option = _FPS_MODE_OPTION

    @property
    def captured_at(self) -> float:
        """Wall-clock capture time of the latest frame, for end-to-end age telemetry."""
        return self._captured_at

    @property
    def captured_monotonic(self) -> float:
        """Monotonic capture time of the latest frame, for local interval telemetry."""
        return self._captured_monotonic

    def _command(self) -> list[str]:
        # -q:v is an inverse quality scale (2 best, 31 worst), so a percentage has to be
        # mapped onto it rather than passed through.
        qscale = max(2, min(31, round(31 - (self._quality / 100.0) * 29)))
        return [
            _ffmpeg_binary() or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-fflags",
            "nobuffer",
            "-avioflags",
            "direct",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            self._device.input_format,
            # An *input* option, and it has to be: a capture device negotiates its rate
            # when it is opened, so this must precede -i. avfoundation's default is
            # 29.97 (NTSC), which real webcams reject -- they offer exactly 15 and 30 --
            # so leaving it unset failed every open with "Input/output error", a message
            # that reads like a permission problem and sent the diagnosis to macOS
            # privacy settings for a bug that was two words of argv.
            #
            # Note this is the *device* rate, not the preview rate. The source must use a
            # supported mode; the broker and the ``fps`` filter below cap how often frames
            # are scaled and JPEG-encoded for the page.
            "-framerate",
            f"{self._input_rate:g}",
            "-i",
            self._device.spec,
            # Drop at the requested preview cadence *before* scaling: ``-r`` alone is an
            # output option, so FFmpeg would still scale every 30fps input frame and only
            # discard the extra work at the encoder. The source device stays at a supported
            # 30fps mode; the page profile changes the expensive scale/JPEG path.
            "-vf",
            f"fps={self._fps:g},scale={self._max_width}:-2",
            self._sync_option,
            "passthrough",
            "-q:v",
            str(qscale),
            "-flush_packets",
            "1",
            "-f",
            "mjpeg",
            "pipe:1",
        ]

    async def start(self) -> None:
        if self._process is not None:
            return
        if not _ffmpeg_binary():
            raise MediaCaptureError(
                "ffmpeg is not installed, so this device cannot be previewed. Install "
                "ffmpeg (macOS: `brew install ffmpeg`) or the opencv-python package.",
                failure_code="capture_backend_missing",
            )
        # Probed off the loop and cached per binary: the edge board's distro FFmpeg is
        # routinely older than the host's, and an unknown option kills capture outright.
        self._sync_option = await asyncio.to_thread(_ffmpeg_sync_option, _ffmpeg_binary())
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            raise MediaCaptureError(
                f"could not start capture for {self._device.name!r}: {exc}",
                failure_code="capture_start_failed",
            ) from exc
        self._latest = None
        self._captured_at = 0.0
        self._captured_monotonic = 0.0
        self._pump_error = None
        self._pump_task = asyncio.create_task(
            self._pump_latest(), name=f"frame:{self._device.device_id}"
        )

    async def grab(self) -> tuple[bytes, int, int]:
        """Return the newest frame, never an accumulated queue of old JPEGs.

        The rate fallback deliberately runs after leaving ``_condition``. ``stop()``
        notifies the same condition while reaping the child, so awaiting it while holding
        the condition would deadlock the restart path exactly when a device rejects its
        initial rate.
        """
        restart_at_rate: float | None = None
        async with self._condition:
            while self._latest is None and self._pump_error is None:
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=5.0)
                except asyncio.TimeoutError as exc:
                    raise MediaCaptureError(
                        f"capture produced no frame for {self._device.name!r}",
                        failure_code="capture_timeout",
                    ) from exc
            if self._pump_error is not None:
                learned = _supported_rate(str(self._pump_error), self._input_rate)
                if learned is not None and learned != self._input_rate:
                    restart_at_rate = learned
                else:
                    raise self._pump_error
            elif self._latest is not None:
                return self._latest
            else:  # pragma: no cover - condition invariants above cover this
                raise MediaCaptureError(
                    "capture is not running", failure_code="capture_not_started"
                )

        if restart_at_rate is None:  # pragma: no cover - guarded by the branch above
            raise MediaCaptureError(
                "capture restart was not configured", failure_code="capture_not_started"
            )
        self._input_rate = restart_at_rate
        await self.stop()
        await self.start()
        return await self.grab()

    async def _set_pump_error(self, error: MediaCaptureError) -> None:
        """Publish a terminal pump error and wake every frame waiter."""
        async with self._condition:
            self._pump_error = error
            self._condition.notify_all()

    async def _pump_latest(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            await self._set_pump_error(
                MediaCaptureError(
                    "capture process has no stdout", failure_code="capture_stream_failed"
                )
            )
            return
        try:
            while True:
                chunk = await process.stdout.read(65536)
                if not chunk:
                    detail = await self._stderr_detail(process)
                    await self._set_pump_error(self._ended_error(detail))
                    return
                self._buffer.extend(chunk)
                if len(self._buffer) > 4 * 1024 * 1024:
                    del self._buffer[: -2 * 1024 * 1024]
                frame = self._take_frame()
                if frame is None:
                    continue
                async with self._condition:
                    self._latest = frame
                    self._captured_at = time.time()
                    self._captured_monotonic = time.monotonic()
                    self._condition.notify_all()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - every terminal pump failure must wake waiters
            failure = (
                exc
                if isinstance(exc, MediaCaptureError)
                else MediaCaptureError(str(exc), failure_code="capture_stream_failed")
            )
            await self._set_pump_error(failure)

    def _ended_error(self, detail: str) -> MediaCaptureError:
        """Classify why ffmpeg's stream ended, separating contention from failure.

        V4L2 capture is exclusive. On a robot the camera driver (a ROS node, a vendor
        streamer) usually owns the device already, and "capture ended" would read as a
        broken camera when the actionable fact is *who* holds it. uvcvideo reports that
        contention as EBUSY on some nodes and as EIO on others; EIO is also what an
        unplugged camera gives, so it only counts as contention when a holder is found.
        """
        busy = "Device or resource busy" in detail
        holders = (
            _device_holders(self._device.spec) if busy or "Input/output error" in detail else []
        )
        if not busy and not holders:
            return MediaCaptureError(
                f"capture ended for {self._device.name!r}{detail}",
                failure_code="capture_stream_ended",
            )
        owner = f" by {', '.join(holders)}" if holders else " by another process"
        return MediaCaptureError(
            f"{self._device.name!r} ({self._device.spec}) is held{owner}. Preview the "
            "owner's published stream instead (for a ROS camera driver, declare a ros2 "
            "device bound to its image topic), or stop the owner before previewing here.",
            failure_code="device_busy",
        )

    def _take_frame(self) -> tuple[bytes, int, int] | None:
        """Cut the newest complete JPEG out of the buffer, discarding older ones.

        Newest rather than oldest: a viewer that falls behind should see the present,
        not work through a backlog. Everything before the frame is dropped, which is
        also what keeps the buffer from growing without bound.
        """
        end = self._buffer.rfind(_JPEG_EOI)
        if end == -1:
            return None
        start = self._buffer.rfind(_JPEG_SOI, 0, end)
        if start == -1:
            return None
        data = bytes(self._buffer[start : end + 2])
        del self._buffer[: end + 2]
        self._size = _jpeg_size(data) or self._size
        return data, self._size[0], self._size[1]

    async def _stderr_detail(self, process: Any | None = None) -> str:
        """Return ffmpeg's actionable error tail after stdout ends."""
        process = process or self._process
        if process is None or process.stderr is None:
            return ""
        try:
            raw = await asyncio.wait_for(process.stderr.read(4096), timeout=1.0)
        except (asyncio.TimeoutError, OSError):
            return ""
        text = [line.strip() for line in raw.decode("utf-8", "replace").strip().splitlines()]
        useful = [
            line for line in text if line and "NSCameraUseContinuityCameraDeviceType" not in line
        ]
        return f": {' | '.join(useful)}" if useful else ""

    async def stop(self) -> None:
        task, self._pump_task = self._pump_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, OSError):
                await task
        process, self._process = self._process, None
        self._buffer.clear()
        async with self._condition:
            self._latest = None
            self._condition.notify_all()
        await _stop_ffmpeg_process(process, label=f"camera {self._device.name!r}")


async def _stop_ffmpeg_process(process: Any, *, label: str) -> None:
    """Terminate and reap an ffmpeg child before a device is considered released."""
    if process is None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.debug("ffmpeg terminate failed for %s: %s", label, exc)
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
        return
    except (asyncio.TimeoutError, OSError):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            return
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except (asyncio.TimeoutError, OSError) as exc:
        logger.warning("ffmpeg did not exit after kill for %s: %s", label, exc)


class OpenCvFrameGrabber:
    """Capture through ``cv2.VideoCapture``, encoding each frame to JPEG.

    Kept as a second backend because it is the one that works where ffmpeg is not
    installed but the Python package is -- a common state in data-science
    environments. Blocking calls are pushed to a worker thread: ``VideoCapture.read``
    waits for the sensor, and doing that on the event loop would stall every other
    session for the duration.
    """

    backend = "opencv"

    def __init__(self, device: MediaDevice, *, max_width: int, quality: int, fps: float) -> None:
        self._device = device
        self._max_width = max(16, int(max_width))
        self._quality = min(100, max(1, int(quality)))
        self._fps = max(0.1, float(fps))
        self._capture: Any = None

    async def start(self) -> None:
        if self._capture is not None:
            return
        cv2 = _import_cv2()
        target: Any = (
            self._device.index if self._device.input_format != "v4l2" else self._device.spec
        )
        capture = await asyncio.to_thread(cv2.VideoCapture, target)
        if not capture.isOpened():
            await asyncio.to_thread(capture.release)
            raise MediaCaptureError(
                f"opencv could not open {self._device.name!r}; the device may be in use "
                "or this process may lack camera permission",
                failure_code="capture_start_failed",
            )
        self._capture = capture

    async def grab(self) -> tuple[bytes, int, int]:
        capture = self._capture
        if capture is None:
            raise MediaCaptureError("capture is not running", failure_code="capture_not_started")
        cv2 = _import_cv2()

        def _read() -> tuple[bytes, int, int]:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise MediaCaptureError(
                    f"no frame from {self._device.name!r}", failure_code="capture_read_failed"
                )
            height, width = frame.shape[:2]
            if width > self._max_width:
                scale = self._max_width / float(width)
                frame = cv2.resize(frame, (self._max_width, max(2, int(height * scale))))
                height, width = frame.shape[:2]
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
            if not ok:
                raise MediaCaptureError(
                    "frame could not be encoded", failure_code="capture_encode_failed"
                )
            return bytes(buffer), int(width), int(height)

        return await asyncio.to_thread(_read)

    async def stop(self) -> None:
        capture, self._capture = self._capture, None
        if capture is None:
            return
        try:
            await asyncio.to_thread(capture.release)
        except Exception as exc:  # noqa: BLE001 - teardown must not propagate
            logger.debug("opencv release failed: %s", exc)


_BACKENDS: dict[str, Callable[..., FrameGrabber]] = {
    "ffmpeg": FfmpegFrameGrabber,
    "opencv": OpenCvFrameGrabber,
}


@runtime_checkable
class LevelReader(Protocol):
    """A live source of input level, in dBFS, from one audio device."""

    async def start(self) -> None:
        """Begin measuring. Where a platform consent prompt appears."""
        ...

    async def read(self) -> float:
        """Return the most recent level in dBFS."""
        ...

    async def stop(self) -> None:
        """Release the device. Idempotent, must not raise."""
        ...


_RMS_KEY = "lavfi.astats.Overall.RMS_level"
_RMS_RE = re.compile(re.escape(_RMS_KEY) + r"=(-?\d+(?:\.\d+)?|-?inf)")


class FfmpegLevelReader:
    """Measure input level with a long-lived ffmpeg process.

    ``astats`` computes the level and ``ametadata=print`` writes it to stderr, one line
    per analysis window, which is read incrementally. The same reasoning as the frame
    grabber applies: one process for the life of the measurement, because a process per
    reading would cost more than the interval between readings.

    Deliberately *no* audio is decoded, stored, or forwarded -- the output goes to
    ``null`` and only a scalar leaves this class. A level meter tells somebody their
    microphone is live; a recording is a different capability with a different
    consequence, and this must not quietly become one.
    """

    backend = "ffmpeg"

    def __init__(self, device: MediaDevice, *, window_s: float = 0.25) -> None:
        self._device = device
        self._window_s = max(0.05, float(window_s))
        self._process: Any = None
        self._level = float("nan")
        self._task: asyncio.Task[None] | None = None

    def _command(self) -> list[str]:
        return [
            _ffmpeg_binary() or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-f",
            self._device.input_format,
            "-i",
            self._device.spec,
            "-af",
            (
                f"asetnsamples=n={max(64, int(48000 * self._window_s))},"
                f"astats=metadata=1:reset=1,ametadata=print:key={_RMS_KEY}"
            ),
            "-f",
            "null",
            "-",
        ]

    async def start(self) -> None:
        if self._process is not None:
            return
        if not _ffmpeg_binary():
            raise MediaCaptureError(
                "ffmpeg is not installed, so this device's level cannot be measured. "
                "Install ffmpeg (macOS: `brew install ffmpeg`).",
                failure_code="capture_backend_missing",
            )
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except (OSError, ValueError) as exc:
            raise MediaCaptureError(
                f"could not start level measurement for {self._device.name!r}: {exc}",
                failure_code="capture_start_failed",
            ) from exc
        self._task = asyncio.create_task(self._pump(), name=f"level:{self._device.device_id}")

    async def _pump(self) -> None:
        """Keep the newest level, discarding the rest.

        A background reader rather than a read-on-demand parse, because ffmpeg emits a
        line per window whether anybody is asking: left unread, the stderr pipe fills
        and the process blocks, which would stall the measurement it exists to provide.
        """
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    return
                match = _RMS_RE.search(line.decode("utf-8", "replace"))
                if match:
                    raw = match.group(1)
                    self._level = float("-inf") if raw.endswith("inf") else float(raw)
        except (asyncio.CancelledError, OSError):
            return

    async def read(self) -> float:
        if self._process is None:
            raise MediaCaptureError(
                "level measurement is not running", failure_code="capture_not_started"
            )
        if self._process.returncode is not None:
            raise MediaCaptureError(
                f"level measurement ended for {self._device.name!r}; the device may be "
                "in use or this process may lack microphone permission",
                failure_code="capture_stream_ended",
            )
        return self._level

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
        process, self._process = self._process, None
        self._level = float("nan")
        if process is None:
            return
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except (asyncio.TimeoutError, OSError):
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass


def build_level_reader(device: MediaDevice, *, window_s: float = 0.25) -> LevelReader:
    """Build a level reader for *device*.

    Only ffmpeg can do this today, so there is no selection table: a one-row lookup
    would be a placeholder pretending to be a seam. When a second implementation
    exists it gets a table, like frames already have.
    """
    if not _ffmpeg_binary():
        raise MediaCaptureError(
            "measuring input level needs ffmpeg, which is not installed",
            failure_code="capture_backend_missing",
        )
    return FfmpegLevelReader(device, window_s=window_s)


_BACKEND_PREFERENCE = ("ffmpeg", "opencv")
"""ffmpeg first: it streams, so a preview costs one process rather than one per frame.

opencv is the fallback for environments that have the Python package but not the
binary.
"""


def available_backends() -> tuple[str, ...]:
    """Return the capture backends usable in this process, in preference order."""
    usable = []
    if _ffmpeg_binary():
        usable.append("ffmpeg")
    if _import_cv2(required=False) is not None:
        usable.append("opencv")
    return tuple(name for name in _BACKEND_PREFERENCE if name in usable)


def build_grabber(
    device: MediaDevice,
    *,
    backend: str = "",
    max_width: int = DEFAULT_MAX_WIDTH,
    quality: int = DEFAULT_QUALITY,
    fps: float = DEFAULT_FPS,
) -> FrameGrabber:
    """Build a grabber for *device*, choosing a backend when none is named."""
    chosen = backend or next(iter(available_backends()), "")
    if not chosen:
        raise MediaCaptureError(
            "no capture backend is available: install ffmpeg or the opencv-python package",
            failure_code="capture_backend_missing",
        )
    factory = _BACKENDS.get(chosen)
    if factory is None:
        raise MediaCaptureError(
            f"unknown capture backend {chosen!r}; available: {', '.join(available_backends())}",
            failure_code="unknown_capture_backend",
        )
    return factory(device, max_width=max_width, quality=quality, fps=fps)


# ── Helpers ──


def _ffmpeg_binary() -> str:
    return shutil.which("ffmpeg") or ""


_FPS_MODE_OPTION = "-fps_mode"
_LEGACY_SYNC_OPTION = "-vsync"
_FPS_MODE_SINCE = (5, 1)
"""FFmpeg 5.1 introduced ``-fps_mode`` and deprecated ``-vsync`` for the same setting.

Ubuntu 22.04 -- the JetPack 6 base -- ships 4.4, which rejects ``-fps_mode`` as an
unrecognized option, so every Jetson preview failed before producing a frame.
"""


@functools.lru_cache(maxsize=8)
def _ffmpeg_sync_option(binary: str) -> str:
    """Return the frame-sync option name this FFmpeg build accepts.

    Unparseable versions (git snapshots print ``N-...``) are treated as modern: only a
    numbered release older than 5.1 needs the legacy spelling.
    """
    if not binary:
        return _FPS_MODE_OPTION
    return _sync_option_for_version(_run([binary, "-hide_banner", "-version"]))


def _sync_option_for_version(version_text: str) -> str:
    """Map ``ffmpeg -version`` output to the frame-sync option it understands."""
    match = re.search(r"ffmpeg version n?(\d+)\.(\d+)", version_text)
    if match is None:
        return _FPS_MODE_OPTION
    version = (int(match.group(1)), int(match.group(2)))
    return _LEGACY_SYNC_OPTION if version < _FPS_MODE_SINCE else _FPS_MODE_OPTION


def _import_cv2(*, required: bool = True) -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:
        if not required:
            return None
        raise MediaCaptureError(
            "the opencv-python package is not installed, so this backend is unavailable",
            failure_code="capture_backend_missing",
        ) from exc


def _run(command: list[str]) -> str:
    """Run a listing command and return its combined output, tolerating failure.

    Device listings exit non-zero on several platforms -- ffmpeg's ``-list_devices``
    always does, because there is no input to open -- so the exit status is ignored
    and only the text matters.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            timeout=ENUMERATION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("media listing %r failed: %s", command[0], exc)
        return ""
    return (completed.stdout + completed.stderr).decode("utf-8", "replace")


def _device_holders(spec: str, *, limit: int = 3, proc_root: str = "/proc") -> list[str]:
    """Name the processes holding a device node open, best-effort and Linux-only.

    Only the process name and pid are reported -- never its arguments, which can carry
    credentials. Descriptors of processes owned by other users are unreadable, which
    simply yields fewer names rather than an error.
    """
    target = Path(spec)
    root = Path(proc_root)
    if not spec.startswith("/dev/") or not root.is_dir():
        return []
    try:
        resolved = target.resolve()
    except OSError:
        return []
    holders: list[str] = []
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            held = any(
                Path(os.readlink(fd)) in (target, resolved) for fd in (entry / "fd").iterdir()
            )
            if not held:
                continue
            name = (entry / "comm").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        holders.append(f"{name or 'unknown'} (pid {entry.name})")
        if len(holders) >= limit:
            break
    return holders


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Read pixel dimensions out of a JPEG's start-of-frame marker.

    Parsed from the bytes rather than assumed from the requested width: the scaler
    rounds the height to an even number, so the frame is rarely exactly what was
    asked for, and reporting the request as the result would mislabel every frame.
    """
    index = 2
    length = len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        segment = int.from_bytes(data[index + 2 : index + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        if segment <= 0:
            return None
        index += 2 + segment
    return None


__all__ = [
    "CAMERA",
    "DEFAULT_FPS",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_QUALITY",
    "MICROPHONE",
    "SCREEN",
    "FfmpegFrameGrabber",
    "FfmpegLevelReader",
    "FrameGrabber",
    "LevelReader",
    "MediaCaptureError",
    "MediaDevice",
    "OpenCvFrameGrabber",
    "available_backends",
    "build_grabber",
    "build_level_reader",
    "enumerate_devices",
]

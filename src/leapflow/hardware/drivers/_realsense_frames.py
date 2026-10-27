# Copyright (c) Alibaba, Inc. and its affiliates.
"""The acquisition half of the RealSense transport: a ``FrameProducer`` over a pipeline.

The transport owns the device (open, options, close); this module owns what happens
to a frameset once it arrives: handing it to a :class:`~leapflow.hardware.vision.FrameHub`,
aligning depth to colour, colourising depth, and exposing each image as an
:class:`~leapflow.hardware.vision.ImagePlane`. Every function here runs on a hub
thread, never on an event loop, and never under the transport's lifecycle lock.

SDK processing blocks are not re-entrant, so ``align`` and ``colorize`` are each
serialised on their own lock; the hub's view cache makes sure each runs at most once
per frameset however many consumers ask.
"""

from __future__ import annotations

import threading
from typing import Any

from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision import (
    ACQUIRE_STREAMING,
    ALIGN_NONE,
    ALIGN_SOFTWARE,
    SYNC_HARDWARE,
    FrameCapabilities,
    ImagePlane,
)

PLANE_COLOR = "color"
PLANE_DEPTH = "depth"
PLANE_IR_LEFT = "infrared_1"
PLANE_IR_RIGHT = "infrared_2"

# SDK format name -> sensor_msgs encoding. The SDK states each frame's real format;
# guessing it is how colourised depth came to be decoded as BGR although the
# colouriser emits RGB.
_ENCODINGS = {
    "bgr8": "bgr8",
    "rgb8": "rgb8",
    "bgra8": "bgra8",
    "rgba8": "rgba8",
    "y8": "mono8",
    "z16": "16UC1",
}
_BYTES_PER_PIXEL = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "16UC1": 2}


class RealSenseFrames:
    """Pump a started ``pyrealsense2`` pipeline and derive planes from its framesets."""

    def __init__(self, pipeline: Any, *, align: Any = None, colorizer: Any = None) -> None:
        self._pipeline = pipeline
        self._align = align
        self._colorizer = colorizer
        self._align_lock = threading.Lock()
        self._colorize_lock = threading.Lock()

    @property
    def aligns(self) -> bool:
        return self._align is not None

    def capabilities(self) -> FrameCapabilities:
        return FrameCapabilities(
            planes=(PLANE_COLOR, PLANE_DEPTH, PLANE_IR_LEFT, PLANE_IR_RIGHT),
            alignment=ALIGN_SOFTWARE if self._align is not None else ALIGN_NONE,
            sync=SYNC_HARDWARE,
            acquisition=ACQUIRE_STREAMING,
        )

    def grab(self, timeout_s: float) -> Any:
        frames = self._pipeline.wait_for_frames(max(1, int(timeout_s * 1000)))
        # A retained frameset must leave the SDK's frame pool, or holding the
        # newest few starves the pool and the device starts dropping frames.
        keep = getattr(frames, "keep", None)
        if callable(keep):
            keep()
        return frames

    def aligned(self, frames: Any) -> Any:
        """Depth registered to the colour image; the input unchanged when not aligning."""
        if self._align is None:
            return frames
        with self._align_lock:
            return self._align.process(frames)

    def plane(self, frames: Any, name: str, *, colorize_depth: bool = True) -> ImagePlane:
        """One displayable plane of ``frames``; depth is colourised for viewing."""
        if name == PLANE_COLOR:
            frame, fallback = frames.get_color_frame(), "bgr8"
        elif name == PLANE_DEPTH:
            depth = frames.get_depth_frame()
            if not colorize_depth:
                frame = depth
            elif not depth or self._colorizer is None:
                frame = None
            else:
                with self._colorize_lock:
                    frame = self._colorizer.colorize(depth)
            fallback = "rgb8" if colorize_depth else "16UC1"
        elif name in (PLANE_IR_LEFT, PLANE_IR_RIGHT):
            frame, fallback = frames.get_infrared_frame(1 if name == PLANE_IR_LEFT else 2), "mono8"
        else:
            raise TransportError(f"unknown RealSense plane {name!r}", failure_code="unknown_channel")
        if not frame:
            raise TransportError(
                f"RealSense frameset has no {name} frame", failure_code="realsense_frame_missing"
            )
        width, height = int(frame.get_width()), int(frame.get_height())
        encoding = _encoding(frame, fallback)
        return ImagePlane(
            name=name,
            encoding=encoding,
            width=width,
            height=height,
            step=_stride(frame, width * _BYTES_PER_PIXEL[encoding]),
            data=frame.get_data(),
            frame_number=safe_frame_number(frame),
            timestamp_ms=safe_timestamp(frame),
        )


def source_metadata(frame: Any) -> dict[str, Any]:
    """Retain native stream identity before alignment changes its image geometry."""
    domain = None
    native_intrinsics = {}
    if frame:
        try:
            domain = str(frame.get_frame_timestamp_domain())
        except (AttributeError, RuntimeError):
            pass
        try:
            intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
            native_intrinsics = {key: getattr(intrinsics, key) for key in
                                 ("width", "height", "fx", "fy", "ppx", "ppy")}
            native_intrinsics.update(model=str(intrinsics.model), coeffs=list(intrinsics.coeffs))
        except (AttributeError, RuntimeError):
            pass
    return {"frame_number": safe_frame_number(frame), "timestamp_ms": safe_timestamp(frame),
            "timestamp_domain": domain, "intrinsics": native_intrinsics}


def safe_frame_number(frame: Any) -> int:
    try:
        return int(frame.get_frame_number())
    except Exception:  # noqa: BLE001 - identity metadata is best-effort
        return 0


def safe_timestamp(frame: Any) -> float:
    try:
        return float(frame.get_timestamp())
    except Exception:  # noqa: BLE001 - identity metadata is best-effort
        return 0.0


def _encoding(frame: Any, fallback: str) -> str:
    try:
        name = str(frame.get_profile().format()).rsplit(".", 1)[-1].lower()
    except (AttributeError, RuntimeError):
        return fallback
    return _ENCODINGS.get(name, fallback)


def _stride(frame: Any, packed: int) -> int:
    try:
        stride = int(frame.get_stride_in_bytes())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return packed
    return stride if stride >= packed else packed


__all__ = [
    "PLANE_COLOR",
    "PLANE_DEPTH",
    "PLANE_IR_LEFT",
    "PLANE_IR_RIGHT",
    "RealSenseFrames",
    "safe_frame_number",
    "safe_timestamp",
    "source_metadata",
]

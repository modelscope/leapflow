# Copyright (c) Alibaba, Inc. and its affiliates.
"""Vendor-neutral frame types: one capture cycle, its image planes, and its producer.

A depth camera, a stereo pair, a USB webcam, or a ROS image topic all deliver the
same thing: a *frameset* -- the images one capture cycle produced together -- at
some rate. This module names that once, so a hub can share a frameset between any
number of consumers and a consumer can ask for "a frame newer than this one"
without knowing which SDK is underneath.

``ImagePlane`` deliberately has the field names of ``sensor_msgs/msg/Image``
(``encoding``, ``width``, ``height``, ``step``, ``data``) and uses its encoding
names (``rgb8``, ``bgr8``, ``mono8``, ``16UC1``): one plane is directly encodable by
``ros2_image.encode_preview`` and maps one-to-one onto a ROS topic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

ALIGN_HARDWARE = "hardware"
"""Depth is registered to colour by the device itself."""
ALIGN_SOFTWARE = "software"
"""Depth can be registered to colour on the host, at a per-frame CPU cost."""
ALIGN_NONE = "none"

SYNC_HARDWARE = "hardware"
"""Planes of one frameset were exposed on a shared trigger."""
SYNC_TIMESTAMP = "timestamp"
"""Planes were matched by timestamp after capture (approximate)."""
SYNC_NONE = "none"

ACQUIRE_STREAMING = "streaming"
ACQUIRE_TRIGGERED = "triggered"


@dataclass(frozen=True)
class ImagePlane:
    """One image of a frameset, shaped like ``sensor_msgs/msg/Image``.

    ``data`` is any buffer-protocol object. It may borrow the producer's memory,
    so it is valid only while the frameset it came from is retained.
    """

    name: str
    encoding: str
    width: int
    height: int
    step: int
    data: Any
    frame_number: int = 0
    timestamp_ms: float = 0.0
    is_bigendian: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"image plane {self.name!r} has invalid size {self.width}x{self.height}")
        if self.step < self.width:
            raise ValueError(f"image plane {self.name!r} step {self.step} is below its width")


@dataclass(frozen=True)
class Frameset:
    """One capture cycle, stamped when it reached this process.

    ``sequence`` increases by one per frameset a hub publishes, so every plane and
    every view derived from one frameset carries the same number. ``native`` is the
    producer's own handle (an SDK frameset, a message) -- opaque to consumers, and
    passed back to the producer to derive planes.
    """

    sequence: int
    observed_at: float
    monotonic_at: float
    native: Any = None


@dataclass(frozen=True)
class FrameCapabilities:
    """What a producer can deliver, declared rather than discovered by trial."""

    planes: tuple[str, ...]
    alignment: str = ALIGN_NONE
    sync: str = SYNC_NONE
    acquisition: str = ACQUIRE_STREAMING

    def to_dict(self) -> dict[str, Any]:
        return {
            "planes": list(self.planes),
            "alignment": self.alignment,
            "sync": self.sync,
            "acquisition": self.acquisition,
        }


@runtime_checkable
class FrameProducer(Protocol):
    """The acquisition half of a camera: hand over the next capture cycle.

    ``grab`` blocks for at most ``timeout_s`` and is only ever called from one
    thread at a time (a hub's pump). It raises on timeout or device failure; a
    hub reports the failure to waiting consumers and keeps pumping. Returning
    ``None`` means "nothing yet, not a failure" (e.g. an idle triggered source).
    """

    def capabilities(self) -> FrameCapabilities: ...

    def grab(self, timeout_s: float) -> Any: ...


__all__ = [
    "ACQUIRE_STREAMING",
    "ACQUIRE_TRIGGERED",
    "ALIGN_HARDWARE",
    "ALIGN_NONE",
    "ALIGN_SOFTWARE",
    "SYNC_HARDWARE",
    "SYNC_NONE",
    "SYNC_TIMESTAMP",
    "FrameCapabilities",
    "FrameProducer",
    "Frameset",
    "ImagePlane",
]

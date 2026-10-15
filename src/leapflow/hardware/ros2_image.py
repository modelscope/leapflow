# Copyright (c) Alibaba, Inc. and its affiliates.
"""Turn ROS 2 image messages into bounded JPEG previews.

A ROS camera driver owns its device exclusively, so a robot's camera is observable
only through the topics that driver publishes. This module is the pixel-format half
of that path: it accepts ``sensor_msgs/msg/CompressedImage`` and
``sensor_msgs/msg/Image`` *duck-typed* -- no ROS import -- so it is testable without
a ROS installation and usable from any transport that receives such messages.

Prefer the driver's ``/compressed`` topic: an uncompressed 1280x720 RGB stream is
roughly 80 MB/s through the DDS layer, while its JPEG twin is a few MB/s and is
passed through untouched whenever it already fits the requested width.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Mapping

from PIL import Image

COMPRESSED_IMAGE_TYPE = "sensor_msgs/msg/CompressedImage"
RAW_IMAGE_TYPE = "sensor_msgs/msg/Image"

_JPEG_MAGIC = b"\xff\xd8"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_DEFAULT_QUALITY = 75

# encoding -> (PIL mode, raw decoder mode). 16-bit depth is decoded separately.
_RAW_MODES: Mapping[str, tuple[str, str]] = {
    "rgb8": ("RGB", "RGB"),
    "bgr8": ("RGB", "BGR"),
    "rgba8": ("RGBA", "RGBA"),
    "bgra8": ("RGBA", "BGRA"),
    "mono8": ("L", "L"),
    "8uc1": ("L", "L"),
}
_DEPTH16_ENCODINGS = frozenset({"mono16", "16uc1"})


class ImageConversionError(ValueError):
    """Raised when a message cannot be rendered as a preview image."""


@dataclass(frozen=True)
class EncodedPreview:
    """One JPEG preview plus JSON-safe facts about its source message."""

    data: bytes
    width: int
    height: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def frame_message_type(topic: str) -> str:
    """Infer the message type of an image topic from the image_transport convention.

    Declaring the type matters: an undiscovered topic would otherwise be subscribed
    as ``std_msgs/String`` and never deliver a message.
    """
    return COMPRESSED_IMAGE_TYPE if topic.rstrip("/").endswith("/compressed") else RAW_IMAGE_TYPE


def encode_preview(msg: Any, *, max_width: int = 0, quality: int = 0) -> EncodedPreview:
    """Encode an image message as a JPEG no wider than *max_width* (0 = native)."""
    if hasattr(msg, "encoding"):
        image = _decode_raw(msg)
        source = {"encoding": str(msg.encoding), "passthrough": False}
    elif hasattr(msg, "format"):
        data = bytes(msg.data)
        if not data:
            raise ImageConversionError("compressed image message carries no data")
        if data.startswith(_JPEG_MAGIC):
            try:
                with Image.open(io.BytesIO(data)) as probe:
                    width, height = probe.size
            except (OSError, ValueError) as exc:
                raise ImageConversionError(f"JPEG header could not be read: {exc}") from exc
            if max_width <= 0 or width <= max_width:
                return EncodedPreview(data, width, height, _stamped(msg, {
                    "format": str(msg.format), "passthrough": True,
                    "source_width": width, "source_height": height,
                }))
        image = _decode_compressed(data)
        source = {"format": str(msg.format), "passthrough": False}
    else:
        raise ImageConversionError(f"{type(msg).__name__} is not an image message")
    source.update(source_width=image.width, source_height=image.height)
    image = _bounded(_displayable(image), max_width)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=min(100, max(1, int(quality or _DEFAULT_QUALITY))))
    return EncodedPreview(output.getvalue(), image.width, image.height, _stamped(msg, source))


def _decode_raw(msg: Any) -> Image.Image:
    encoding = str(msg.encoding).lower()
    size = (int(msg.width), int(msg.height))
    if size[0] <= 0 or size[1] <= 0:
        raise ImageConversionError(f"image message has invalid size {size}")
    data, step = bytes(msg.data), int(getattr(msg, "step", 0) or 0)
    if encoding in _DEPTH16_ENCODINGS:
        modes = ("I;16", "I;16B" if bool(getattr(msg, "is_bigendian", False)) else "I;16")
    else:
        modes = _RAW_MODES.get(encoding)
    if modes is None:
        raise ImageConversionError(
            f"image encoding {msg.encoding!r} is not supported; supported: "
            f"{', '.join(sorted({*_RAW_MODES, *_DEPTH16_ENCODINGS}))}"
        )
    mode, rawmode = modes
    try:
        return Image.frombytes(mode, size, data, "raw", rawmode, step)
    except ValueError as exc:
        raise ImageConversionError(f"image data does not match {size} {encoding}: {exc}") from exc


def _decode_compressed(data: bytes) -> Image.Image:
    # compressed_depth_image_transport prefixes the PNG with a 12-byte config header.
    offset = data.find(_PNG_MAGIC)
    payload = data[offset:] if offset > 0 else data
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (OSError, ValueError) as exc:
        raise ImageConversionError(f"compressed image could not be decoded: {exc}") from exc
    return image


def _displayable(image: Image.Image) -> Image.Image:
    """Map any decoded mode onto L or RGB, stretching 16-bit depth for visibility."""
    if image.mode in {"I;16", "I;16B", "I;16L", "I"}:
        deep = image.convert("I") if image.mode != "I" else image
        _, peak = deep.getextrema()
        scale = 255.0 / peak if peak else 0.0
        return deep.point(lambda value: value * scale).convert("L")
    if image.mode in {"L", "RGB"}:
        return image
    return image.convert("RGB")


def _bounded(image: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or image.width <= max_width:
        return image
    height = max(1, round(image.height * max_width / image.width))
    return image.resize((max_width, height), Image.Resampling.LANCZOS)


def _stamped(msg: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        metadata["source_stamp"] = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) / 1e9
    frame_id = str(getattr(header, "frame_id", "") or "")
    if frame_id:
        metadata["frame_id"] = frame_id
    return metadata


__all__ = [
    "COMPRESSED_IMAGE_TYPE",
    "RAW_IMAGE_TYPE",
    "EncodedPreview",
    "ImageConversionError",
    "encode_preview",
    "frame_message_type",
]

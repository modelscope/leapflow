# Copyright (c) Alibaba, Inc. and its affiliates.
"""Token cost of one image content part, estimated from its pixel size.

A fixed per-image figure overcounts a 384-pixel thumbnail several times and
undercounts a large mosaic, so budget gates either compress too early or let
an image-heavy turn overflow. Providers bill images by area (512-pixel tiles,
or 28-pixel patches), so the estimate reads the width and height from the
image header -- only the first few kilobytes of the base64 are decoded -- and
takes the larger of the two common schemes. An image whose size cannot be read
(a URL, an unknown format) keeps the historical flat estimate.
"""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Any, Optional, Tuple

DEFAULT_IMAGE_TOKENS = 1600
_MIN_IMAGE_TOKENS = 85
_MAX_IMAGE_TOKENS = 2500
_HEADER_B64_CHARS = 8192
IMAGE_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def image_part_tokens(part: Any) -> int:
    """Estimated prompt tokens of one ``image_url``-style content part."""
    size = image_size(_data_url(part))
    if size is None:
        return DEFAULT_IMAGE_TOKENS
    return image_tokens(*size)


def image_tokens(width: int, height: int) -> int:
    """Estimated prompt tokens of a *width* x *height* image."""
    if width <= 0 or height <= 0:
        return DEFAULT_IMAGE_TOKENS
    tiles = 85 + 170 * math.ceil(width / 512) * math.ceil(height / 512)
    patches = math.ceil(width * height / 784) + 2
    return max(_MIN_IMAGE_TOKENS, min(_MAX_IMAGE_TOKENS, max(tiles, patches)))


def image_size(url: str) -> Optional[Tuple[int, int]]:
    """``(width, height)`` of a base64 data URL's JPEG or PNG, else ``None``."""
    if not url.startswith("data:"):
        return None
    marker = url.find(";base64,", 0, 128)
    if marker < 0:
        return None
    start = marker + len(";base64,")
    payload = url[start : start + _HEADER_B64_CHARS]
    payload = payload[: len(payload) - len(payload) % 4]
    try:
        head = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        width, height = struct.unpack(">II", head[16:24])
        return int(width), int(height)
    if head[:2] == b"\xff\xd8":
        return _jpeg_size(head)
    return None


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
            index += 1 if marker == 0xFF else 2
            continue
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        # SOF0..SOF15 except DHT (C4), JPG (C8) and DAC (CC) carry the frame size.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        index += 2 + length
    return None


def _data_url(part: Any) -> str:
    if not isinstance(part, dict):
        return ""
    image = part.get("image_url")
    if isinstance(image, dict):
        return str(image.get("url") or "")
    if isinstance(image, str):
        return image
    source = part.get("source")
    if isinstance(source, dict) and source.get("type") == "base64":
        return f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
    return str(part.get("image") or "") if isinstance(part.get("image"), str) else ""


__all__ = [
    "DEFAULT_IMAGE_TOKENS",
    "IMAGE_PART_TYPES",
    "image_part_tokens",
    "image_size",
    "image_tokens",
]

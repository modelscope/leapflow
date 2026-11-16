# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rendering shared frames: one tile per camera plane, many tiles in one image.

S2 sees a workcell best in *one* image: every camera's view side by side, each
plane of a camera (colour, its aligned depth, an infrared sensor) in its row,
with what perception already knows drawn on top -- remembered object ids, the
TCP, where another camera sits and aims. One mosaic costs one image's tokens
and keeps cross-view correspondence visible, where separate images would be
compared from memory.

Planes are declared by what they are, not by which camera produced them:

* ``color`` -- the RGB image;
* ``depth`` -- the depth aligned to it, false-coloured near = red to far = blue
  over the range actually in view (black = no reading), with that range in its
  label; pixel-aligned with colour, so a point in it lifts like a colour point;
* ``infrared`` -- any declared ``infrared_image`` frame channel (an active
  stereo pair, a ToF amplitude, a thermal core): for looking only, since it is
  not registered to the depth that lifting reads.

Pillow only; every function is synchronous and meant for a worker thread.
"""

from __future__ import annotations

import base64
import colorsys
import io
import sys
from dataclasses import dataclass
from typing import Any, Sequence

from leapflow.hardware.rgbd import RgbdFrame
from leapflow.hardware.skills.protocol import (  # noqa: F401 - re-exported
    PLANE_COLOR,
    PLANE_DEPTH,
    PLANE_INFRARED,
    PLANES,
)

TILE_GAP = 4
MIN_DEPTH_SPAN_MM = 50.0
_BACKGROUND = (24, 24, 24)
_TONES = {
    "object": (255, 214, 0),
    "warn": (255, 72, 72),
    "tcp": (255, 0, 255),
    "camera": (0, 230, 255),
    "muted": (170, 170, 170),
}
_DEPTH_SAMPLES = 20000


def _depth_luts() -> tuple[list[int], list[int], list[int]]:
    """Index 0 is black (no reading); 1..255 runs near = red through to far = blue."""
    red, green, blue = [0], [0], [0]
    for index in range(1, 256):
        r, g, b = colorsys.hsv_to_rgb((index - 1) / 254.0 * (240.0 / 360.0), 1.0, 1.0)
        red.append(round(r * 255))
        green.append(round(g * 255))
        blue.append(round(b * 255))
    return red, green, blue


_DEPTH_LUTS = _depth_luts()


@dataclass(frozen=True)
class Mark:
    """Something drawn over a tile, in full-frame pixels of the frame it belongs to.

    *kind* is ``box`` (``xy`` = x0, y0, x1, y1), ``point``, ``cross``, or
    ``ring`` (``xy`` = u, v); *tone* names its colour.
    """

    kind: str
    xy: tuple[float, ...]
    text: str = ""
    tone: str = "object"


def fit_size(width: int, height: int, max_side: float) -> tuple[int, int]:
    """``(width, height)`` scaled so the longer side is at most *max_side*."""
    scale = min(1.0, max_side / max(width, height, 1))
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def snapshot_size(frame: RgbdFrame, max_side: float) -> tuple[int, int]:
    """The ``(width, height)`` a frame is shared at: its longer side at most *max_side*."""
    return fit_size(frame.width, frame.height, max_side)


def color_image(frame: RgbdFrame, size: tuple[int, int], window: Sequence[float] | None = None) -> Any:
    """The colour image (or the *window* of it) at *size*, as an RGB ``PIL.Image``."""
    from PIL import Image

    image = Image.frombytes("RGB", (frame.width, frame.height), frame.color, "raw", "BGR")
    return _crop_resize(image, size, window, Image.BILINEAR)


def snapshot_jpeg(frame: RgbdFrame, size: tuple[int, int], quality: int) -> str:
    """The colour image at *size* as base64 JPEG."""
    return encode_jpeg(color_image(frame, size), quality)


def depth_span(
    frame: RgbdFrame, window: Sequence[float] | None = None, *, min_span_mm: float = MIN_DEPTH_SPAN_MM
) -> tuple[float, float] | None:
    """The near and far depth (mm) a tile maps colours over: the 0.5th and 98th
    percentiles of the readings in view, at least *min_span_mm* apart; ``None``
    when nothing in view has depth. The near end clips less than the far end: on
    a table the small, near objects are what the task is about, while far
    outliers are background."""
    units = frame.depth_units()
    x0, y0, x1, y1 = _window(frame, window)
    area = max(1, (x1 - x0) * (y1 - y0))
    step = max(1, int((area / _DEPTH_SAMPLES) ** 0.5))
    values = sorted(
        value
        for v in range(y0, y1, step)
        for value in units[v * frame.width + x0 : v * frame.width + x1 : step]
        if value
    )
    if not values:
        return None
    scale = frame.depth_scale_m * 1000.0
    near = values[int(0.005 * (len(values) - 1))] * scale
    far = values[int(0.98 * (len(values) - 1))] * scale
    if far - near < min_span_mm:
        middle = (far + near) / 2.0
        near, far = max(0.0, middle - min_span_mm / 2.0), middle + min_span_mm / 2.0
    return near, far


def depth_image(
    frame: RgbdFrame,
    size: tuple[int, int],
    span: tuple[float, float] | None,
    window: Sequence[float] | None = None,
) -> Any:
    """The aligned depth (or its *window*) at *size*, false-coloured over *span* mm."""
    from PIL import Image, ImageChops

    if span is None:
        return Image.new("RGB", size, (0, 0, 0))
    raw = "I;16" if sys.byteorder == "little" else "I;16B"
    units = _crop_resize(
        Image.frombytes(raw, (frame.width, frame.height), frame.depth).convert("I"),
        size, window, Image.NEAREST,  # never blend a missing reading into its neighbours
    )
    scale = frame.depth_scale_m * 1000.0
    near, far = span[0] / scale, span[1] / scale
    gain = 254.0 / max(far - near, 1e-6)
    index = units.point(lambda i: i * gain + (1.0 - near * gain)).convert("L")  # clamps to 0..255
    valid = units.point(lambda i: i * 255).convert("L").point(lambda v: 1 if v else 0)
    index = ImageChops.lighter(index, valid)  # every reading is at least index 1
    return Image.merge("RGB", tuple(index.point(lut) for lut in _DEPTH_LUTS))


def decoded_image(data: bytes, size: tuple[int, int]) -> Any:
    """An encoded frame (an infrared preview) as an RGB image at *size*."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB").resize(size, Image.BILINEAR)


def draw_marks(
    image: Any,
    marks: Sequence[Mark],
    frame_size: tuple[int, int],
    window: Sequence[float] | None = None,
) -> None:
    """Draw *marks* (full-frame pixels of a *frame_size* frame) onto a tile of *image*'s size."""
    from PIL import ImageDraw

    if not marks:
        return
    x0, y0, x1, y1 = window or (0.0, 0.0, float(frame_size[0]), float(frame_size[1]))
    sx, sy = image.width / (x1 - x0), image.height / (y1 - y0)
    draw = ImageDraw.Draw(image)

    def at(u: float, v: float) -> tuple[float, float]:
        return (u - x0) * sx, (v - y0) * sy

    for mark in marks:
        tone = _TONES.get(mark.tone, _TONES["object"])
        if mark.kind == "box":
            left, top = at(mark.xy[0], mark.xy[1])
            right, bottom = at(mark.xy[2], mark.xy[3])
            if right <= 0 or bottom <= 0 or left >= image.width or top >= image.height:
                continue
            draw.rectangle((left, top, right, bottom), outline=tone, width=2)
            anchor = (max(0.0, left), max(0.0, top - 12.0))
        else:
            u, v = at(mark.xy[0], mark.xy[1])
            if not (0 <= u < image.width and 0 <= v < image.height):
                continue
            if mark.kind == "cross":
                draw.line((u - 8, v, u + 8, v), fill=tone, width=2)
                draw.line((u, v - 8, u, v + 8), fill=tone, width=2)
            else:
                radius = 6 if mark.kind == "ring" else 3
                draw.ellipse(
                    (u - radius, v - radius, u + radius, v + radius),
                    outline=tone, width=2, fill=None if mark.kind == "ring" else tone,
                )
            anchor = (u + 7.0, v - 7.0)
        if mark.text:
            _text(draw, anchor, mark.text, tone)


def draw_label(image: Any, text: str) -> None:
    """Name a tile in its top-left corner."""
    from PIL import ImageDraw

    if text:
        _text(ImageDraw.Draw(image), (3.0, 3.0), text, (255, 255, 255))


def layout(rows: Sequence[Sequence[tuple[int, int]]], gap: int = TILE_GAP) -> tuple[
    list[list[tuple[int, int]]], tuple[int, int]
]:
    """Shelf-pack tile sizes row by row: each tile's top-left, and the mosaic size."""
    positions: list[list[tuple[int, int]]] = []
    width = top = 0
    for row in rows:
        x, placed = 0, []
        for tile_width, _ in row:
            placed.append((x, top))
            x += tile_width + gap
        positions.append(placed)
        width = max(width, x - gap)
        top += max((h for _, h in row), default=0) + gap
    return positions, (max(1, width), max(1, top - gap))


def compose(rows: Sequence[Sequence[Any]], gap: int = TILE_GAP) -> Any:
    """Paste tile images row by row into one mosaic (``layout`` positions)."""
    from PIL import Image

    positions, size = layout([[tile.size for tile in row] for row in rows], gap)
    mosaic = Image.new("RGB", size, _BACKGROUND)
    for row, placed in zip(rows, positions):
        for tile, xy in zip(row, placed):
            mosaic.paste(tile, xy)
    return mosaic


def encode_jpeg(image: Any, quality: int) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=min(100, max(1, int(quality))))
    return base64.b64encode(output.getvalue()).decode("ascii")


def _window(frame: RgbdFrame, window: Sequence[float] | None) -> tuple[int, int, int, int]:
    if window is None:
        return 0, 0, frame.width, frame.height
    x0 = max(0, min(frame.width - 1, int(window[0])))
    y0 = max(0, min(frame.height - 1, int(window[1])))
    return x0, y0, max(x0 + 1, min(frame.width, round(window[2]))), max(
        y0 + 1, min(frame.height, round(window[3]))
    )


def _crop_resize(image: Any, size: tuple[int, int], window: Sequence[float] | None, method: Any) -> Any:
    if window is not None:
        image = image.crop(tuple(round(value) for value in window))
    return image if image.size == tuple(size) else image.resize(size, method)


def _text(draw: Any, xy: tuple[float, float], text: str, fill: tuple[int, int, int]) -> None:
    box = draw.textbbox(xy, text)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
    draw.text(xy, text, fill=fill)


__all__ = [
    "PLANES",
    "PLANE_COLOR",
    "PLANE_DEPTH",
    "PLANE_INFRARED",
    "Mark",
    "color_image",
    "compose",
    "decoded_image",
    "depth_image",
    "depth_span",
    "draw_label",
    "draw_marks",
    "encode_jpeg",
    "fit_size",
    "layout",
    "snapshot_jpeg",
    "snapshot_size",
]

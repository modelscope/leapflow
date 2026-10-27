# Copyright (c) Alibaba, Inc. and its affiliates.
"""Display encoders over vendor-neutral planes; never change task RGB-D data.

Pillow preserves the existing path. The optional OpenCV encoder handles native
16-bit depth as a histogram-equalized JET *preview*, avoiding SDK colorization
and registration that compete with control observations. Its input is borrowed
only for the synchronous call; the hub retains the native frameset throughout.
Third-party encoders use the ``leapflow.hardware.vision_encoders`` entry-point
group and implement ``PlaneEncoder``; no camera or hub edits are necessary.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from leapflow.hardware.ros2_image import EncodedPreview, ImageConversionError, encode_preview
from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision.frames import ImagePlane


@runtime_checkable
class PlaneEncoder(Protocol):
    """One bounded JPEG from a raw plane; implementations must be thread-safe."""

    def encode(self, plane: ImagePlane, *, max_width: int, quality: int) -> EncodedPreview: ...


class PillowPlaneEncoder:
    """The portable encoder, with the same pixel conversion as ROS image previews."""

    def encode(self, plane: ImagePlane, *, max_width: int, quality: int) -> EncodedPreview:
        return encode_preview(plane, max_width=max_width, quality=quality)


class OpenCvPlaneEncoder:
    """Native buffer -> OpenCV resize/colorization/JPEG, without SDK processing blocks.

    Native depth is shown on a per-frame cumulative histogram: near blue, far
    red, missing depth black. Colors express rank, not metric distance. This is
    display-only; task depth, calibration, intrinsics and alignment are untouched.
    """

    def __init__(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise TransportError(
                "OpenCV preview encoder needs numpy and opencv-python (the vision or robot extra)",
                failure_code="preview_encoder_unavailable",
            ) from exc
        self._cv = cv2
        self._np = np

    def encode(self, plane: ImagePlane, *, max_width: int, quality: int) -> EncodedPreview:
        cv, np = self._cv, self._np
        encoding = plane.encoding.lower()
        layouts = {
            "bgr8": (3, None), "rgb8": (3, cv.COLOR_RGB2BGR),
            "bgra8": (4, cv.COLOR_BGRA2BGR), "rgba8": (4, cv.COLOR_RGBA2BGR),
            "mono8": (1, None), "8uc1": (1, None),
            "16uc1": (1, None), "mono16": (1, None),
        }
        layout = layouts.get(encoding)
        if layout is None:
            raise ImageConversionError(f"OpenCV preview does not support {plane.encoding!r}")
        channels, conversion = layout
        depth = encoding in {"16uc1", "mono16"}
        itemsize = 2 if depth else 1
        if plane.step < plane.width * channels * itemsize:
            raise ImageConversionError("image plane stride is smaller than a pixel row")
        try:
            shape = (plane.height, plane.width, channels) if channels > 1 else (plane.height, plane.width)
            strides = (plane.step, channels, 1) if channels > 1 else (plane.step, itemsize)
            image = np.ndarray(
                shape, dtype=(">u2" if plane.is_bigendian else "<u2") if depth else "u1",
                buffer=plane.data, strides=strides,
            )
        except (TypeError, ValueError, BufferError) as exc:
            raise ImageConversionError(f"image data does not match its geometry: {exc}") from exc
        metadata = {"encoding": plane.encoding, "preview_encoder": "opencv",
                    "source_width": plane.width, "source_height": plane.height,
                    "passthrough": False}
        try:
            if depth:
                histogram = np.bincount(image.ravel(), minlength=65536)
                histogram[0] = 0  # absent depth must not bias the distribution
                cumulative = histogram.cumsum()
                scale = 255.0 / cumulative[-1] if cumulative[-1] else 0.0
                preview = (cumulative[image] * scale).astype(np.uint8)
                colored = cv.applyColorMap(preview, cv.COLORMAP_JET)
                colored[image == 0] = 0
                image = colored
                metadata["depth_colorization"] = "histogram_jet"
            elif conversion is not None:
                image = cv.cvtColor(image, conversion)
            height, width = image.shape[:2]
            if max_width > 0 and width > max_width:
                height = max(1, round(height * max_width / width))
                width = int(max_width)
                image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
            ok, encoded = cv.imencode(
                ".jpg", image, [cv.IMWRITE_JPEG_QUALITY, min(100, max(1, int(quality or 75)))],
            )
            if not ok:
                raise ImageConversionError("OpenCV JPEG encoder returned no image")
        except cv.error as exc:
            raise ImageConversionError(f"OpenCV preview failed: {exc}") from exc
        return EncodedPreview(encoded.tobytes(), width, height, metadata)


def build_encoder(kind: str = "pillow") -> PlaneEncoder:
    """Load a selected encoder lazily. Built-in names cannot be shadowed by a package."""
    builtins = {"pillow": PillowPlaneEncoder, "opencv": OpenCvPlaneEncoder}
    factory = builtins.get(kind)
    if factory is not None:
        return factory()
    matches = [ep for ep in entry_points(group="leapflow.hardware.vision_encoders") if ep.name == kind]
    if len(matches) != 1:
        raise TransportError(
            f"preview encoder {kind!r} is unknown or ambiguous",
            failure_code="preview_encoder_unavailable",
        )
    try:
        encoder = matches[0].load()()
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise TransportError(
            f"preview encoder {kind!r} could not be loaded: {exc}",
            failure_code="preview_encoder_unavailable",
        ) from exc
    if not isinstance(encoder, PlaneEncoder):
        raise TransportError(
            f"preview encoder {kind!r} does not implement PlaneEncoder",
            failure_code="preview_encoder_invalid",
        )
    return encoder


__all__ = ["PlaneEncoder", "PillowPlaneEncoder", "OpenCvPlaneEncoder", "build_encoder"]

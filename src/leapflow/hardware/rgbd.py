# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node-local RGB-D capture: the ``FrameSource`` capability and its value types.

A preview frame (``FrameTransport``) is a bounded JPEG for a human or an LLM. A
perception skill needs more: the raw colour image the model saw, a depth image
aligned pixel-for-pixel to it, and the intrinsics that turn a pixel plus depth
into a 3-D point. That is an ``RgbdFrame``, and a transport that can produce one
implements ``FrameSource``.

An ``RgbdFrame`` never crosses LHP. It lives inside the node process for the
few hundred milliseconds a skill needs it; only geometry derived from it leaves
the board. Pure Python (plus Pillow for encoding), so the node imports it without
numpy.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

DISTORTION_NONE = "none"
DISTORTION_BROWN_CONRADY = "brown_conrady"
DISTORTION_INVERSE_BROWN_CONRADY = "inverse_brown_conrady"
DISTORTION_MODELS = frozenset(
    {DISTORTION_NONE, DISTORTION_BROWN_CONRADY, DISTORTION_INVERSE_BROWN_CONRADY}
)
_UNDISTORT_ITERATIONS = 10


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of one image stream, in pixels, plus its lens distortion.

    ``model`` follows the librealsense names; coefficients are ``(k1, k2, p1, p2,
    k3)``. A model this module cannot invert is refused at construction rather
    than silently treated as a pinhole.
    """

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = DISTORTION_NONE
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("intrinsics need a positive image size")
        if not (self.fx > 0 and self.fy > 0 and math.isfinite(self.fx * self.fy)):
            raise ValueError("intrinsics need positive, finite focal lengths")
        if not all(math.isfinite(v) for v in (self.ppx, self.ppy, *self.coeffs)):
            raise ValueError("intrinsics and distortion coefficients must be finite")
        if self.model not in DISTORTION_MODELS and any(self.coeffs):
            raise ValueError(f"distortion model {self.model!r} is not supported")

    @property
    def _k(self) -> tuple[float, float, float, float, float]:
        padded = (*self.coeffs, 0.0, 0.0, 0.0, 0.0, 0.0)[:5]
        return padded  # type: ignore[return-value]

    def deproject(self, u: float, v: float, depth_mm: float) -> tuple[float, float, float]:
        """The camera-frame point (mm) seen at pixel ``(u, v)`` at *depth_mm* along +Z.

        Mirrors ``rs2_deproject_pixel_to_point`` so a point matches what the
        vendor SDK would compute for the same frame.
        """
        x = (u - self.ppx) / self.fx
        y = (v - self.ppy) / self.fy
        if any(self.coeffs):
            if self.model == DISTORTION_INVERSE_BROWN_CONRADY:
                x, y = _undistort(x, y, self._k, inverse_brown=True)
            elif self.model == DISTORTION_BROWN_CONRADY:
                x, y = _undistort(x, y, self._k)
        return (depth_mm * x, depth_mm * y, depth_mm)

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """The pixel a camera-frame point projects to; the exact inverse of ``deproject``."""
        if point[2] <= 0:
            raise ValueError("a point behind the camera does not project")
        x, y = point[0] / point[2], point[1] / point[2]
        if any(self.coeffs):
            if self.model == DISTORTION_BROWN_CONRADY:
                x, y = _distort(x, y, self._k)
            elif self.model == DISTORTION_INVERSE_BROWN_CONRADY:
                x, y = _distort(x, y, self._k, inverse_brown=True)
        return (x * self.fx + self.ppx, y * self.fy + self.ppy)

    def differs_from(self, other: "CameraIntrinsics", *, tolerance_px: float = 0.5) -> str:
        """Why *other* is not the same calibration (``""`` when it is)."""
        if not math.isfinite(tolerance_px) or tolerance_px < 0:
            raise ValueError("intrinsics tolerance must be finite and nonnegative")
        if self.model != other.model or self.coeffs != other.coeffs:
            return "distortion calibration differs"
        if (self.width, self.height) != (other.width, other.height):
            return f"resolution {other.width}x{other.height} != {self.width}x{self.height}"
        for name in ("fx", "fy", "ppx", "ppy"):
            if abs(getattr(self, name) - getattr(other, name)) > tolerance_px:
                return f"{name} {getattr(other, name):.2f} != {getattr(self, name):.2f}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "model": self.model,
            "coeffs": list(self.coeffs),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraIntrinsics":
        """Parse ``to_dict`` output; raises ``ValueError``/``KeyError`` on a bad value."""
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            ppx=float(data["ppx"]),
            ppy=float(data["ppy"]),
            model=str(data.get("model") or DISTORTION_NONE),
            coeffs=tuple(float(c) for c in data.get("coeffs") or ()),
        )


def _distort(
    x: float, y: float, k: Sequence[float], *, inverse_brown: bool = False,
) -> tuple[float, float]:
    r2 = x * x + y * y
    radial = 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[4] * r2 * r2 * r2
    qx, qy = (x * radial, y * radial) if inverse_brown else (x, y)
    dx = 2.0 * k[2] * qx * qy + k[3] * (r2 + 2.0 * qx * qx)
    dy = 2.0 * k[3] * qx * qy + k[2] * (r2 + 2.0 * qy * qy)
    return (x * radial + dx, y * radial + dy)


def _undistort(
    x: float, y: float, k: Sequence[float], *, inverse_brown: bool = False,
) -> tuple[float, float]:
    """Invert ``_distort`` by fixed-point iteration, as librealsense does."""
    ux, uy = x, y
    for _ in range(_UNDISTORT_ITERATIONS):
        r2 = ux * ux + uy * uy
        radial = 1.0 + k[0] * r2 + k[1] * r2 * r2 + k[4] * r2 * r2 * r2
        qx, qy = (ux * radial, uy * radial) if inverse_brown else (ux, uy)
        dx = 2.0 * k[2] * qx * qy + k[3] * (r2 + 2.0 * qx * qx)
        dy = 2.0 * k[3] * qx * qy + k[2] * (r2 + 2.0 * qy * qy)
        ux, uy = (x - dx) / radial, (y - dy) / radial
    return (ux, uy)


@dataclass(frozen=True)
class RgbdFrame:
    """One colour image and the depth image aligned to it, captured together.

    ``color`` is packed BGR8 (``width * height * 3`` bytes); ``depth`` is packed
    unsigned 16-bit depth units in native byte order (``width * height * 2``
    bytes), each unit ``depth_scale_m`` metres, 0 meaning "no measurement".
    """

    device_id: str
    width: int
    height: int
    color: bytes
    depth: bytes
    depth_scale_m: float
    intrinsics: CameraIntrinsics
    observed_at: float
    monotonic_at: float
    sequence: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pixels = self.width * self.height
        if len(self.color) != pixels * 3:
            raise ValueError("colour buffer does not match the frame size")
        if len(self.depth) != pixels * 2:
            raise ValueError("depth buffer does not match the frame size (is it aligned?)")
        if (self.intrinsics.width, self.intrinsics.height) != (self.width, self.height):
            raise ValueError("intrinsics describe a different image size")
        if not math.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
            raise ValueError("depth scale must be positive and finite")
        if not all(math.isfinite(v) for v in (self.observed_at, self.monotonic_at)):
            raise ValueError("frame clocks must be finite")

    def depth_units(self) -> memoryview:
        """Zero-copy ``uint16`` view of the depth image, row-major."""
        return memoryview(self.depth).cast("H")

    def depth_mm(self, u: int, v: int) -> float:
        """Depth at pixel ``(u, v)`` in millimetres; 0.0 when there is no measurement."""
        return self.depth_units()[v * self.width + u] * self.depth_scale_m * 1000.0

    def color_jpeg(self, *, quality: int = 90) -> bytes:
        """The colour image as a JPEG at full resolution (what a model is sent)."""
        from PIL import Image

        image = Image.frombytes("RGB", (self.width, self.height), self.color, "raw", "BGR")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=min(100, max(1, int(quality))))
        return output.getvalue()


@runtime_checkable
class FrameSource(Protocol):
    """A transport that captures aligned RGB-D frames for node-local perception."""

    async def capture_rgbd(self) -> RgbdFrame:
        """Capture one frame; raises ``TransportError`` when the camera cannot."""
        ...


__all__ = [
    "DISTORTION_BROWN_CONRADY",
    "DISTORTION_INVERSE_BROWN_CONRADY",
    "DISTORTION_MODELS",
    "DISTORTION_NONE",
    "CameraIntrinsics",
    "FrameSource",
    "RgbdFrame",
]

"""Vendor-neutral preview encoder contracts, independent of a camera SDK."""
from __future__ import annotations

import builtins
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import EntryPoint
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from leapflow.hardware.ros2_image import ImageConversionError, encode_preview
from leapflow.hardware.transport import TransportError
from leapflow.hardware.vision import encoding as module
from leapflow.hardware.vision.encoding import OpenCvPlaneEncoder, PlaneEncoder, build_encoder
from leapflow.hardware.vision.frames import ImagePlane


def plane(encoding: str = "rgb8", pixel: bytes = b"\xff\x00\x00", **kwargs) -> ImagePlane:
    values = dict(name="color", encoding=encoding, width=16, height=8,
                  step=16 * len(pixel), data=pixel * 128)
    values.update(kwargs)
    return ImagePlane(**values)


@pytest.fixture
def encoder() -> OpenCvPlaneEncoder:
    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    return OpenCvPlaneEncoder()


def decoded(preview):
    return Image.open(BytesIO(preview.data)).convert("RGB")


def test_portable_encoder_keeps_existing_bytes_and_builtin_precedence(monkeypatch) -> None:
    def undiscovered(**kwargs):
        raise AssertionError("built-ins must not discover plugins")
    monkeypatch.setattr(module, "entry_points", undiscovered)
    portable = build_encoder()
    assert isinstance(portable, PlaneEncoder)
    image = plane()
    assert portable.encode(image, max_width=8, quality=100) == encode_preview(
        image, max_width=8, quality=100
    )


@pytest.mark.parametrize("encoding,pixel", [
    ("rgb8", b"\xff\x00\x00"), ("bgr8", b"\x00\x00\xff"),
    ("rgba8", b"\xff\x00\x00\x00"), ("bgra8", b"\x00\x00\xff\x00"),
])
def test_color_and_alpha_order(encoder, encoding, pixel) -> None:
    result = encoder.encode(plane(encoding, pixel), max_width=8, quality=100)
    assert (result.width, result.height) == (8, 4)
    red, green, blue = decoded(result).getpixel((3, 2))
    assert red > 240 and green < 10 and blue < 10
    assert result.metadata["source_width"] == 16
    assert result.metadata["preview_encoder"] == "opencv"


@pytest.mark.parametrize("encoding", ["mono8", "8UC1"])
@pytest.mark.parametrize("quality", [-10, 0, 101])
def test_mono_quality_bounds_and_no_upscale(encoder, encoding, quality) -> None:
    result = encoder.encode(plane(encoding, b"\x80"), max_width=32, quality=quality)
    assert (result.width, result.height) == (16, 8)
    assert all(abs(v - 128) <= 5 for v in decoded(result).getpixel((4, 4)))


def test_padded_rows_do_not_decode_padding_as_pixels(encoder) -> None:
    image = plane(width=16, height=8, step=52,
                  data=(b"\xff\x00\x00" * 16 + b"\x00" * 4) * 8)
    result = encoder.encode(image, max_width=0, quality=100)
    assert decoded(result).getpixel((8, 7))[0] > 240


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("encoding", ["16UC1", "mono16"])
def test_depth_histogram_missing_pixels_endian_and_stride(encoder, bigendian, encoding) -> None:
    import numpy as np
    raw = np.zeros((16, 48), dtype=">u2" if bigendian else "<u2")
    raw[:, 16:32] = 500
    raw[:, 32:] = 2000
    # Include padding to ensure the distribution ignores it.
    data = b"".join(row.tobytes() + b"\xff\xff" * 4 for row in raw)
    image = plane(encoding, width=48, height=16, step=104, data=data,
                  is_bigendian=bigendian)
    result = encoder.encode(image, max_width=0, quality=100)
    rgb = decoded(result)
    assert max(rgb.getpixel((4, 8))) < 10
    assert rgb.getpixel((24, 8))[1] > 200
    red, green, blue = rgb.getpixel((40, 8))
    assert red > 100 and green < 20 and blue < 20
    assert result.metadata["depth_colorization"] == "histogram_jet"


def test_zero_depth_is_black_without_division_error(encoder) -> None:
    result = encoder.encode(plane("16UC1", b"\x00\x00"), max_width=0, quality=100)
    assert max(decoded(result).getpixel((8, 4))) == 0


@pytest.mark.parametrize("image", [
    plane(step=16), plane(data=b"x"), plane(data=object()), plane(encoding="32FC1"),
])
def test_invalid_buffer_stride_or_encoding_is_a_conversion_error(encoder, image) -> None:
    with pytest.raises(ImageConversionError):
        encoder.encode(image, max_width=0, quality=75)


def test_codec_failure_is_classified(encoder, monkeypatch) -> None:
    monkeypatch.setattr(encoder._cv, "imencode", lambda *args: (False, None))
    with pytest.raises(ImageConversionError, match="no image"):
        encoder.encode(plane(), max_width=0, quality=75)


def test_shared_encoder_does_not_mutate_borrowed_input(encoder) -> None:
    data = bytearray(b"\xff\x00\x00" * 128)
    original = bytes(data)
    image = plane(data=data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: encoder.encode(image, max_width=8, quality=75), range(20)))
    assert data == original
    assert len({r.data for r in results}) == 1


@pytest.mark.parametrize("missing", ["cv2", "numpy"])
def test_optional_dependency_missing_is_typed(monkeypatch, missing) -> None:
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == missing:
            raise ImportError("missing optional module")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(TransportError) as error:
        OpenCvPlaneEncoder()
    assert error.value.failure_code == "preview_encoder_unavailable"


@pytest.mark.parametrize("count", [0, 2])
def test_unknown_or_ambiguous_plugin_is_typed(monkeypatch, count) -> None:
    ep = EntryPoint(name="custom", value="unused:factory", group="leapflow.hardware.vision_encoders")
    monkeypatch.setattr(module, "entry_points", lambda **kwargs: [ep] * count)
    with pytest.raises(TransportError) as error:
        build_encoder("custom")
    assert error.value.failure_code == "preview_encoder_unavailable"


@pytest.mark.parametrize("value,code", [
    ("encoder_test:nested.factory", None),
    ("encoder_test:invalid", "preview_encoder_invalid"),
    ("encoder_test:missing", "preview_encoder_unavailable"),
    ("nonexistent_encoder_package:factory", "preview_encoder_unavailable"),
])
def test_entry_point_contract_uses_standard_loader(monkeypatch, value, code) -> None:
    monkeypatch.setitem(sys.modules, "encoder_test", SimpleNamespace(
        nested=SimpleNamespace(factory=lambda: build_encoder("pillow")), invalid=lambda: object()
    ))
    ep = EntryPoint(name="custom", value=value, group="leapflow.hardware.vision_encoders")
    monkeypatch.setattr(module, "entry_points", lambda **kwargs: [ep])
    if code is None:
        assert isinstance(build_encoder("custom"), PlaneEncoder)
    else:
        with pytest.raises(TransportError) as error:
            build_encoder("custom")
        assert error.value.failure_code == code

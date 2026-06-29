"""Perceptual hashing for frame change detection.

Provides fast, resize-invariant image fingerprinting using DCT-based
perceptual hashing. Designed for <1ms per hash computation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

try:
    from PIL import Image
    import io as _io

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _ensure_pil() -> None:
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required for perceptual hashing")


def _bytes_to_image(data: bytes) -> "Image.Image":
    _ensure_pil()
    return Image.open(_io.BytesIO(data))


def _image_to_grayscale(img: "Image.Image", size: Tuple[int, int]) -> List[List[float]]:
    """Resize to target size and convert to grayscale pixel matrix."""
    resized = img.convert("L").resize(size, Image.LANCZOS)
    pixels = list(resized.getdata())
    w, h = size
    return [pixels[i * w:(i + 1) * w] for i in range(h)]


def _dct_row(row: List[float]) -> List[float]:
    """1D DCT-II (type 2) of a row using direct computation.

    For small sizes (32/64), direct computation is faster than FFT overhead.
    """
    import math

    n = len(row)
    result = []
    for k in range(n):
        s = 0.0
        for i in range(n):
            s += row[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        result.append(s)
    return result


def _dct_2d(matrix: List[List[float]], keep: int) -> List[List[float]]:
    """2D DCT keeping only the top-left keep×keep coefficients."""
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0

    # DCT on rows
    row_dct = [_dct_row(row) for row in matrix]

    # DCT on columns (transpose, DCT, transpose back)
    transposed = [[row_dct[r][c] for r in range(rows)] for c in range(cols)]
    col_dct = [_dct_row(col) for col in transposed]

    # Keep top-left keep×keep
    result = [[col_dct[c][r] for c in range(keep)] for r in range(keep)]
    return result


def phash_64(data: bytes) -> bytes:
    """Compute 64-bit perceptual hash from image data.

    Resizes to 64x64 grayscale, applies 2D DCT, keeps top-left 8x8
    coefficients, and binarizes against median to produce 8 bytes (64 bits).
    """
    _ensure_pil()
    img = _bytes_to_image(data)
    matrix = _image_to_grayscale(img, (64, 64))

    # 2D DCT, keep 8x8 low-frequency
    dct_low = _dct_2d(matrix, 8)

    # Flatten and compute median (exclude DC component)
    flat = [dct_low[r][c] for r in range(8) for c in range(8)]
    dc = flat[0]
    flat_no_dc = flat[1:]
    median = sorted(flat_no_dc)[len(flat_no_dc) // 2]

    # Binarize: 1 if above median, 0 otherwise
    bits = [1 if v > median else 0 for v in flat]
    # Pack into 8 bytes
    result = bytearray(8)
    for i, bit in enumerate(bits):
        if bit:
            result[i // 8] |= (1 << (7 - i % 8))
    return bytes(result)


def phash_32(data: bytes) -> bytes:
    """Compute 32-bit perceptual hash (smaller blocks: quadrants, focus areas).

    Resizes to 32x32 grayscale, applies 2D DCT, keeps top-left 8x8,
    produces 8 bytes. Faster than phash_64 for smaller regions.
    """
    _ensure_pil()
    img = _bytes_to_image(data)
    matrix = _image_to_grayscale(img, (32, 32))

    dct_low = _dct_2d(matrix, 8)
    flat = [dct_low[r][c] for r in range(8) for c in range(8)]
    flat_no_dc = flat[1:]
    median = sorted(flat_no_dc)[len(flat_no_dc) // 2]

    bits = [1 if v > median else 0 for v in flat]
    result = bytearray(8)
    for i, bit in enumerate(bits):
        if bit:
            result[i // 8] |= (1 << (7 - i % 8))
    return bytes(result)


def hamming_distance(a: bytes, b: bytes) -> int:
    """Compute Hamming distance between two hash byte strings.

    Returns the number of differing bits.
    """
    if len(a) != len(b):
        return max(len(a), len(b)) * 8
    dist = 0
    for x, y in zip(a, b):
        xor = x ^ y
        dist += bin(xor).count("1")
    return dist


def split_quadrants(data: bytes) -> List[bytes]:
    """Split image data into 4 quadrant sub-images (top-left, top-right, bottom-left, bottom-right).

    Returns list of 4 JPEG byte strings, each representing one quadrant.
    """
    _ensure_pil()
    img = _bytes_to_image(data)
    w, h = img.size
    mid_x, mid_y = w // 2, h // 2

    quadrants = [
        img.crop((0, 0, mid_x, mid_y)),
        img.crop((mid_x, 0, w, mid_y)),
        img.crop((0, mid_y, mid_x, h)),
        img.crop((mid_x, mid_y, w, h)),
    ]

    result = []
    for q in quadrants:
        buf = _io.BytesIO()
        q.save(buf, format="JPEG", quality=85)
        result.append(buf.getvalue())
    return result


def crop_around(data: bytes, center: Tuple[int, int], size: int = 48) -> bytes:
    """Crop a square region around a center point.

    If center is near edges, the crop is clamped to image bounds.
    Returns JPEG bytes of the cropped region.
    """
    _ensure_pil()
    img = _bytes_to_image(data)
    w, h = img.size
    cx, cy = center
    half = size // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    cropped = img.crop((x1, y1, x2, y2))
    buf = _io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

"""Delta frame composition — optimized before/after encoding for VLM input."""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

from leapflow.perception.types import ChangeSignal, ComposedImage

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _estimate_tokens(width: int, height: int) -> int:
    """Rough estimate of VLM input tokens for an image of given size."""
    pixels = width * height
    return max(100, pixels // 750)


class DeltaFrameComposer:
    """Compose frame pairs into a single VLM-optimized image.

    Three encoding strategies based on change magnitude:
    1. Side-by-side: large changes (app switch) → two frames side by side
    2. Overlay: medium changes → frame B with changed regions highlighted
    3. Focus-diff: small changes → only changed region as before/after crops

    Token savings: 50-77% compared to sending two full frames.
    """

    __slots__ = (
        "_side_by_side_threshold",
        "_overlay_threshold",
        "_side_by_side_res",
        "_overlay_res",
        "_focus_res",
    )

    def __init__(
        self,
        side_by_side_threshold: float = 0.4,
        overlay_threshold: float = 0.15,
        side_by_side_res: int = 512,
        overlay_res: int = 768,
        focus_res: int = 384,
    ) -> None:
        self._side_by_side_threshold = side_by_side_threshold
        self._overlay_threshold = overlay_threshold
        self._side_by_side_res = side_by_side_res
        self._overlay_res = overlay_res
        self._focus_res = focus_res

    def compose(
        self,
        frame_a: bytes,
        frame_b: bytes,
        change: Optional[ChangeSignal] = None,
    ) -> ComposedImage:
        """Create a composed image from a before/after frame pair.

        Automatically selects the most token-efficient layout based on
        the magnitude of visual change between frames.
        """
        if not _HAS_PIL:
            return ComposedImage(
                image=frame_b, layout="passthrough",
                token_estimate=_estimate_tokens(1024, 1024),
            )

        global_diff = change.global_diff if change else 0.5

        if global_diff > self._side_by_side_threshold:
            return self._side_by_side(frame_a, frame_b)
        elif change and change.max_quadrant_diff > self._overlay_threshold:
            return self._overlay_with_highlight(frame_a, frame_b, change)
        else:
            return self._focus_diff(frame_a, frame_b, change)

    def _side_by_side(self, frame_a: bytes, frame_b: bytes) -> ComposedImage:
        """Two frames at reduced resolution, placed side by side."""
        img_a = Image.open(io.BytesIO(frame_a))
        img_b = Image.open(io.BytesIO(frame_b))

        res = self._side_by_side_res
        img_a.thumbnail((res, res), Image.LANCZOS)
        img_b.thumbnail((res, res), Image.LANCZOS)

        # Ensure same height
        h = max(img_a.height, img_b.height)
        canvas = Image.new("RGB", (img_a.width + img_b.width + 4, h), (40, 40, 40))
        canvas.paste(img_a, (0, 0))
        canvas.paste(img_b, (img_a.width + 4, 0))

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()

        return ComposedImage(
            image=data,
            layout="side_by_side",
            token_estimate=_estimate_tokens(canvas.width, canvas.height),
            width=canvas.width,
            height=canvas.height,
        )

    def _overlay_with_highlight(
        self, frame_a: bytes, frame_b: bytes, change: ChangeSignal
    ) -> ComposedImage:
        """Frame B with changed regions highlighted, thumbnail of A in corner."""
        img_a = Image.open(io.BytesIO(frame_a))
        img_b = Image.open(io.BytesIO(frame_b)).copy()

        res = self._overlay_res
        img_b.thumbnail((res, res), Image.LANCZOS)

        # Compute simple diff mask and draw highlight regions
        diff_regions = self._find_diff_regions(frame_a, frame_b, img_b.size)
        draw = ImageDraw.Draw(img_b)
        for region in diff_regions:
            draw.rectangle(region, outline="red", width=3)

        # Paste thumbnail of A in top-left corner
        thumb_size = min(128, res // 4)
        img_a.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        img_b.paste(img_a, (4, 4))

        buf = io.BytesIO()
        img_b.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()

        return ComposedImage(
            image=data,
            layout="overlay",
            token_estimate=_estimate_tokens(img_b.width, img_b.height),
            width=img_b.width,
            height=img_b.height,
        )

    def _focus_diff(
        self,
        frame_a: bytes,
        frame_b: bytes,
        change: Optional[ChangeSignal],
    ) -> ComposedImage:
        """Only send the changed region as before/after crops."""
        img_a = Image.open(io.BytesIO(frame_a))
        img_b = Image.open(io.BytesIO(frame_b))

        # Find the bounding box of change
        bbox = self._find_change_bbox(img_a, img_b, padding=50)
        if bbox is None:
            # Fallback to side-by-side if no detectable diff
            return self._side_by_side(frame_a, frame_b)

        crop_a = img_a.crop(bbox)
        crop_b = img_b.crop(bbox)

        res = self._focus_res
        crop_a.thumbnail((res, res), Image.LANCZOS)
        crop_b.thumbnail((res, res), Image.LANCZOS)

        h = max(crop_a.height, crop_b.height)
        canvas = Image.new("RGB", (crop_a.width + crop_b.width + 4, h), (40, 40, 40))
        canvas.paste(crop_a, (0, 0))
        canvas.paste(crop_b, (crop_a.width + 4, 0))

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85)
        data = buf.getvalue()

        return ComposedImage(
            image=data,
            layout="focus_diff",
            token_estimate=_estimate_tokens(canvas.width, canvas.height),
            crop_region=bbox,
            width=canvas.width,
            height=canvas.height,
        )

    def _find_diff_regions(
        self, frame_a: bytes, frame_b: bytes, target_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Find rectangular regions where frames differ significantly."""
        img_a = Image.open(io.BytesIO(frame_a)).convert("L").resize(target_size, Image.LANCZOS)
        img_b = Image.open(io.BytesIO(frame_b)).convert("L").resize(target_size, Image.LANCZOS)

        w, h = target_size
        block = 64
        regions = []

        # Use bytes-based block comparison (avoids per-pixel getpixel calls)
        pixels_a = img_a.tobytes()
        pixels_b = img_b.tobytes()

        for by in range(0, h, block):
            bh = min(block, h - by)
            for bx in range(0, w, block):
                bw = min(block, w - bx)
                diff_sum = 0
                count = bw * bh
                for row in range(by, by + bh):
                    offset = row * w + bx
                    for col in range(bw):
                        diff_sum += abs(pixels_a[offset + col] - pixels_b[offset + col])
                avg_diff = diff_sum / max(1, count)
                if avg_diff > 30:
                    regions.append((bx, by, bx + bw, by + bh))

        return self._merge_regions(regions, gap=block)

    def _find_change_bbox(
        self,
        img_a: "Image.Image",
        img_b: "Image.Image",
        padding: int = 50,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find the bounding box containing all changes between two images."""
        size = (256, 256)
        a_small = img_a.convert("L").resize(size, Image.LANCZOS)
        b_small = img_b.convert("L").resize(size, Image.LANCZOS)

        w, h = size
        pixels_a = a_small.tobytes()
        pixels_b = b_small.tobytes()

        min_x, min_y = w, h
        max_x, max_y = 0, 0
        found = False

        for y in range(h):
            row_offset = y * w
            for x in range(w):
                idx = row_offset + x
                if abs(pixels_a[idx] - pixels_b[idx]) > 25:
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    found = True

        if not found:
            return None

        orig_w, orig_h = img_a.size
        scale_x = orig_w / w
        scale_y = orig_h / h

        x1 = max(0, int(min_x * scale_x) - padding)
        y1 = max(0, int(min_y * scale_y) - padding)
        x2 = min(orig_w, int(max_x * scale_x) + padding)
        y2 = min(orig_h, int(max_y * scale_y) + padding)

        return (x1, y1, x2, y2)

    @staticmethod
    def _merge_regions(
        regions: List[Tuple[int, int, int, int]], gap: int
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping or adjacent rectangular regions."""
        if not regions:
            return []

        # Simple merge: compute bounding box of all nearby regions
        merged = []
        used = [False] * len(regions)

        for i, r in enumerate(regions):
            if used[i]:
                continue
            x1, y1, x2, y2 = r
            used[i] = True

            # Absorb overlapping regions
            changed = True
            while changed:
                changed = False
                for j, s in enumerate(regions):
                    if used[j]:
                        continue
                    sx1, sy1, sx2, sy2 = s
                    if (sx1 <= x2 + gap and sx2 >= x1 - gap and
                            sy1 <= y2 + gap and sy2 >= y1 - gap):
                        x1 = min(x1, sx1)
                        y1 = min(y1, sy1)
                        x2 = max(x2, sx2)
                        y2 = max(y2, sy2)
                        used[j] = True
                        changed = True

            merged.append((x1, y1, x2, y2))

        return merged

"""Adaptive resolution encoder — context-aware frame compression."""

from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple

from leapflow.perception.types import EncodedFrame

logger = logging.getLogger(__name__)

# Frame type → minimum readable resolution
_RESOLUTION_MAP: Dict[str, int] = {
    "app_switch": 512,
    "dialog_popup": 768,
    "text_input": 1024,
    "click_target": 768,
    "scroll": 512,
    "wait": 384,
    "navigation": 768,
    "unknown": 768,
}

# Frame type → JPEG quality
_QUALITY_MAP: Dict[str, int] = {
    "text_input": 90,
    "dialog_popup": 85,
    "click_target": 85,
    "app_switch": 75,
    "scroll": 70,
    "wait": 65,
    "navigation": 75,
    "unknown": 75,
}


class AdaptiveResolutionEncoder:
    """Encode frames at resolution matched to their information content.

    A frame showing app switch only needs 512px to identify the apps.
    A frame where user types text needs 1024px+ to read characters.
    Allocate pixels where they matter.
    """

    __slots__ = ("_resolution_map", "_quality_map", "_default_resolution", "_default_quality")

    def __init__(
        self,
        resolution_map: Optional[Dict[str, int]] = None,
        quality_map: Optional[Dict[str, int]] = None,
        default_resolution: int = 768,
        default_quality: int = 75,
    ) -> None:
        self._resolution_map = resolution_map or _RESOLUTION_MAP
        self._quality_map = quality_map or _QUALITY_MAP
        self._default_resolution = default_resolution
        self._default_quality = default_quality

    def encode(
        self,
        frame_data: bytes,
        frame_type: str = "unknown",
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> EncodedFrame:
        """Encode a frame with adaptive resolution and quality.

        Args:
            frame_data: Raw frame bytes (JPEG/PNG).
            frame_type: Transition type for resolution selection.
            roi: Optional (x, y, w, h) region of interest to crop first.

        Returns:
            EncodedFrame with compressed data and metadata.
        """
        try:
            from PIL import Image
        except ImportError:
            return EncodedFrame(
                data=frame_data,
                resolution=0,
                quality=0,
                roi=roi,
                frame_type=frame_type,
                size_bytes=len(frame_data),
            )

        target_res = self._resolution_map.get(frame_type, self._default_resolution)
        quality = self._quality_map.get(frame_type, self._default_quality)

        img = Image.open(io.BytesIO(frame_data))

        if roi:
            x, y, w, h = roi
            img = img.crop((x, y, x + w, y + h))
            # ROI crops can use higher resolution (smaller area)
            target_res = min(1024, target_res + 256)

        # Resize preserving aspect ratio
        if max(img.size) > target_res:
            img.thumbnail((target_res, target_res), Image.LANCZOS)

        if img.mode != "RGB":
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        encoded = buf.getvalue()

        return EncodedFrame(
            data=encoded,
            resolution=target_res,
            quality=quality,
            roi=roi,
            frame_type=frame_type,
            size_bytes=len(encoded),
        )

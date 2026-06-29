"""Optical flow analysis for motion pattern classification.

Uses Farneback dense optical flow to distinguish:
- Uniform flow (scroll/pan) → low action value
- Localized flow (button press, popup) → high action value
- No flow (static) → no action

Graceful degradation: returns neutral results if cv2/numpy unavailable.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from leapflow.perception.types import FlowAnalysis

try:
    import numpy as np
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class OpticalFlowAnalyzer:
    """Analyze motion patterns between consecutive frames using optical flow."""

    __slots__ = (
        "_pyr_scale", "_levels", "_winsize",
        "_iterations", "_poly_n", "_poly_sigma",
    )

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self._pyr_scale = pyr_scale
        self._levels = levels
        self._winsize = winsize
        self._iterations = iterations
        self._poly_n = poly_n
        self._poly_sigma = poly_sigma

    def analyze(self, frame_a: bytes, frame_b: bytes) -> FlowAnalysis:
        """Compute optical flow and classify the motion pattern.

        Args:
            frame_a: First frame (JPEG bytes).
            frame_b: Second frame (JPEG bytes).

        Returns:
            FlowAnalysis with motion classification.
        """
        if not _HAS_CV2:
            return FlowAnalysis()

        arr_a = self._decode(frame_a)
        arr_b = self._decode(frame_b)
        if arr_a is None or arr_b is None:
            return FlowAnalysis()

        # Resize for speed (~10ms at 512px)
        target = 512
        h, w = arr_a.shape[:2]
        scale = min(target / max(h, w), 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            arr_a = cv2.resize(arr_a, (new_w, new_h))
            arr_b = cv2.resize(arr_b, (new_w, new_h))

        gray_a = cv2.cvtColor(arr_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(arr_b, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=self._pyr_scale,
            levels=self._levels,
            winsize=self._winsize,
            iterations=self._iterations,
            poly_n=self._poly_n,
            poly_sigma=self._poly_sigma,
            flags=0,
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])

        mean_mag = float(np.mean(magnitude))
        max_mag = float(np.max(magnitude))

        # Scroll detection: uniform flow direction
        active_mask = magnitude > 1.0
        is_scroll = False
        scroll_dir: Optional[str] = None

        if np.any(active_mask):
            active_dirs = direction[active_mask]
            dir_variance = float(np.var(active_dirs))
            is_scroll = mean_mag > 2.0 and dir_variance < 0.5

            if is_scroll:
                mean_dir = float(np.mean(active_dirs))
                scroll_dir = self._direction_to_label(mean_dir)

        # Localized motion regions
        localized_regions = self._find_motion_regions(magnitude)
        motion_type = self._classify(is_scroll, localized_regions, mean_mag)

        return FlowAnalysis(
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            is_scroll=is_scroll,
            scroll_direction=scroll_dir,
            localized_regions=localized_regions,
            motion_type=motion_type,
        )

    def _find_motion_regions(
        self, magnitude: "np.ndarray"
    ) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of localized high-motion areas."""
        threshold = float(np.percentile(magnitude, 90))
        mask = (magnitude > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        min_area = magnitude.shape[0] * magnitude.shape[1] * 0.005

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > min_area:
                regions.append((x, y, w, h))

        return regions[:5]

    @staticmethod
    def _classify(
        is_scroll: bool,
        regions: List[Tuple[int, int, int, int]],
        mean_mag: float,
    ) -> str:
        if mean_mag < 0.5:
            return "static"
        if is_scroll:
            return "scroll"
        if len(regions) <= 2:
            return "localized_interaction"
        return "complex_change"

    @staticmethod
    def _direction_to_label(radians: float) -> str:
        """Convert mean flow direction to human-readable scroll direction."""
        import math
        deg = math.degrees(radians) % 360
        if 45 <= deg < 135:
            return "down"
        elif 135 <= deg < 225:
            return "left"
        elif 225 <= deg < 315:
            return "up"
        return "right"

    @staticmethod
    def _decode(data: bytes) -> "Optional[np.ndarray]":
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

"""Scene cut detection — distinguish hard cuts from soft transitions.

Uses color histogram comparison + edge structure correlation to
identify app switches, page navigations, and tab switches vs.
localized content changes.

Graceful degradation if cv2/numpy unavailable.
"""

from __future__ import annotations

from typing import Optional

from leapflow.perception.types import SceneCutResult

try:
    import numpy as np
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class SceneCutDetector:
    """Detect hard scene cuts vs soft transitions between frames.

    Hard cut: both color distribution and edge structure changed drastically.
    Soft change: either color or structure changed, but not both.
    Minor update: neither changed significantly.
    """

    __slots__ = ("_color_threshold", "_edge_threshold", "_target_size")

    def __init__(
        self,
        color_threshold: float = 0.5,
        edge_threshold: float = 0.3,
        target_size: int = 256,
    ) -> None:
        self._color_threshold = color_threshold
        self._edge_threshold = edge_threshold
        self._target_size = target_size

    def detect(self, frame_a: bytes, frame_b: bytes) -> SceneCutResult:
        """Compare two frames and classify the transition type."""
        if not _HAS_CV2:
            return SceneCutResult()

        img_a = self._decode_resize(frame_a)
        img_b = self._decode_resize(frame_b)
        if img_a is None or img_b is None:
            return SceneCutResult()

        color_dist = self._color_histogram_distance(img_a, img_b)
        edge_corr = self._edge_correlation(img_a, img_b)

        if color_dist > self._color_threshold and edge_corr < self._edge_threshold:
            return SceneCutResult(is_cut=True, cut_type="hard", confidence=0.95)
        elif color_dist > 0.3 or edge_corr < 0.5:
            return SceneCutResult(is_cut=False, cut_type="soft_change", confidence=0.7)
        return SceneCutResult(is_cut=False, cut_type="minor_update", confidence=0.5)

    def _color_histogram_distance(
        self, img_a: "np.ndarray", img_b: "np.ndarray"
    ) -> float:
        """Chi-square distance between color histograms."""
        hist_a = cv2.calcHist([img_a], [0, 1, 2], None, [16, 16, 16],
                             [0, 256, 0, 256, 0, 256])
        hist_b = cv2.calcHist([img_b], [0, 1, 2], None, [16, 16, 16],
                             [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        return float(cv2.compareHist(hist_a.flatten(), hist_b.flatten(), cv2.HISTCMP_CHISQR))

    def _edge_correlation(
        self, img_a: "np.ndarray", img_b: "np.ndarray"
    ) -> float:
        """Normalized cross-correlation of Canny edge maps."""
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        edges_a = cv2.Canny(gray_a, 50, 150).astype(np.float32)
        edges_b = cv2.Canny(gray_b, 50, 150).astype(np.float32)

        # Normalized cross-correlation
        norm_a = np.linalg.norm(edges_a)
        norm_b = np.linalg.norm(edges_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.sum(edges_a * edges_b) / (norm_a * norm_b))

    def _decode_resize(self, data: bytes) -> "Optional[np.ndarray]":
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(self._target_size / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

"""Text change tracking between frames using OCR region matching.

Identifies new, removed, and modified text regions by spatial IoU
matching of OCR bounding boxes between consecutive frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from leapflow.perception.types import TextChange, TextDiff


class TextDiffTracker:
    """Track text changes between frames by comparing OCR results.

    Uses spatial IoU (Intersection over Union) to match text regions
    between frames, then identifies additions, removals, and modifications.
    """

    __slots__ = ("_iou_threshold",)

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold

    def diff(
        self,
        regions_a: List[Dict[str, Any]],
        regions_b: List[Dict[str, Any]],
    ) -> TextDiff:
        """Compute text differences between two sets of OCR regions.

        Each region dict should have: {"text": str, "bbox": (x, y, w, h)}
        Optional: {"confidence": float}
        """
        matched_pairs = self._match_regions(regions_a, regions_b)
        matched_a: Set[int] = {p[0] for p in matched_pairs}
        matched_b: Set[int] = {p[1] for p in matched_pairs}

        added = []
        for i, region in enumerate(regions_b):
            if i not in matched_b:
                added.append(TextChange(
                    text=region.get("text", ""),
                    bbox=self._normalize_bbox(region.get("bbox")),
                    type="added",
                ))

        removed = []
        for i, region in enumerate(regions_a):
            if i not in matched_a:
                removed.append(TextChange(
                    text=region.get("text", ""),
                    bbox=self._normalize_bbox(region.get("bbox")),
                    type="removed",
                ))

        modified = []
        for a_idx, b_idx in matched_pairs:
            text_a = regions_a[a_idx].get("text", "")
            text_b = regions_b[b_idx].get("text", "")
            if text_a != text_b:
                modified.append(TextChange(
                    text=text_b,
                    prev_text=text_a,
                    bbox=self._normalize_bbox(regions_b[b_idx].get("bbox")),
                    type="modified",
                ))

        return TextDiff(added=added, removed=removed, modified=modified)

    def _match_regions(
        self,
        regions_a: List[Dict[str, Any]],
        regions_b: List[Dict[str, Any]],
    ) -> List[Tuple[int, int]]:
        """Match regions by spatial IoU (greedy, highest-IoU-first)."""
        if not regions_a or not regions_b:
            return []

        # Compute all pairwise IoU
        scores: List[Tuple[float, int, int]] = []
        for i, ra in enumerate(regions_a):
            bbox_a = ra.get("bbox")
            if not bbox_a:
                continue
            for j, rb in enumerate(regions_b):
                bbox_b = rb.get("bbox")
                if not bbox_b:
                    continue
                iou = self._compute_iou(bbox_a, bbox_b)
                if iou >= self._iou_threshold:
                    scores.append((iou, i, j))

        # Greedy matching (highest IoU first)
        scores.sort(reverse=True)
        used_a: Set[int] = set()
        used_b: Set[int] = set()
        pairs = []

        for iou, i, j in scores:
            if i in used_a or j in used_b:
                continue
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)

        return pairs

    @staticmethod
    def _compute_iou(
        bbox_a: Any, bbox_b: Any
    ) -> float:
        """Compute Intersection over Union for two bounding boxes.

        Accepts (x, y, w, h) or (x1, y1, x2, y2) format.
        """
        a = _to_xyxy(bbox_a)
        b = _to_xyxy(bbox_b)
        if a is None or b is None:
            return 0.0

        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _normalize_bbox(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None:
            return None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return tuple(int(v) for v in bbox)  # type: ignore[return-value]
        return None


def _to_xyxy(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Convert bbox to (x1, y1, x2, y2) format."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w_or_x2, h_or_y2 = [float(v) for v in bbox]
    # Heuristic: if third value > first, it's probably x2,y2 format
    if w_or_x2 > x and h_or_y2 > y and w_or_x2 > 2 * x:
        return (x, y, w_or_x2, h_or_y2)
    # Assume (x, y, w, h)
    return (x, y, x + w_or_x2, y + h_or_y2)

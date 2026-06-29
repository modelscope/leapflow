"""Stage B: Semantic Preprocessing — CV features to PairContext."""

from __future__ import annotations

from typing import Any, Dict, List

from leapflow.perception.cv.text_diff import TextDiffTracker
from leapflow.perception.types import FramePair, InteractionSignal, PairContext, TextChange


class SemanticPreprocessor:
    """Extract structured context from frame pairs using local CV models.

    The resulting PairContext is injected into VLM prompts, allowing the
    VLM to focus on action reasoning rather than basic perception.
    """

    __slots__ = ("_text_diff_tracker",)

    def __init__(self) -> None:
        self._text_diff_tracker = TextDiffTracker()

    async def process_pair(self, pair: FramePair) -> PairContext:
        """Generate rich context for a frame pair from local CV features."""
        a_feat = pair.frame_a.features
        b_feat = pair.frame_b.features
        pair_signals = self._collect_pair_signals(pair)

        if not a_feat or not b_feat:
            return PairContext(
                time_delta=pair.frame_b.timestamp - pair.frame_a.timestamp,
                signals=pair_signals,
            )

        # App change
        app_a = a_feat.detected_app
        app_b = b_feat.detected_app
        app_changed = bool(app_a and app_b and app_a != app_b)

        # Text changes
        text_diff = self._text_diff_tracker.diff(a_feat.text_regions, b_feat.text_regions)

        # UI element changes
        new_ui, removed_ui = self._diff_ui_elements(a_feat.ui_elements, b_feat.ui_elements)

        # Diff regions (simplified without full pixel diff)
        diff_regions = self._estimate_diff_regions(pair)

        return PairContext(
            app_a=app_a,
            app_b=app_b,
            app_changed=app_changed,
            diff_regions=diff_regions,
            new_text=text_diff.added + text_diff.modified,
            removed_text=text_diff.removed,
            new_ui_elements=new_ui,
            removed_ui_elements=removed_ui,
            time_delta=pair.frame_b.timestamp - pair.frame_a.timestamp,
            signals=pair_signals,
        )

    async def process_batch(self, pairs: List[FramePair]) -> List[PairContext]:
        """Process multiple pairs."""
        import asyncio
        return await asyncio.gather(*[self.process_pair(p) for p in pairs])

    def _diff_ui_elements(
        self,
        elements_a: List[Dict[str, Any]],
        elements_b: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Find added and removed UI elements between frames."""
        # Simple set difference by (cls, approximate position)
        set_a = {self._element_key(e) for e in elements_a}
        set_b = {self._element_key(e) for e in elements_b}

        added_keys = set_b - set_a
        removed_keys = set_a - set_b

        added = [e for e in elements_b if self._element_key(e) in added_keys]
        removed = [e for e in elements_a if self._element_key(e) in removed_keys]

        return added, removed

    @staticmethod
    def _element_key(element: Dict[str, Any]) -> str:
        """Generate a key for approximate element matching."""
        cls = element.get("cls", "")
        bbox = element.get("bbox", (0, 0, 0, 0))
        # Quantize position to 50px grid for fuzzy matching
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            qx = int(bbox[0]) // 50
            qy = int(bbox[1]) // 50
            return f"{cls}:{qx}:{qy}"
        return f"{cls}:0:0"

    @staticmethod
    def _estimate_diff_regions(pair: FramePair) -> List[Dict[str, Any]]:
        """Estimate regions of change from feature differences."""
        regions = []
        if pair.change_signal:
            cs = pair.change_signal
            if cs.changed_quadrant == 0:
                regions.append({"location_desc": "top-left", "quadrant": 0})
            elif cs.changed_quadrant == 1:
                regions.append({"location_desc": "top-right", "quadrant": 1})
            elif cs.changed_quadrant == 2:
                regions.append({"location_desc": "bottom-left", "quadrant": 2})
            elif cs.changed_quadrant == 3:
                regions.append({"location_desc": "bottom-right", "quadrant": 3})
        return regions

    @staticmethod
    def _collect_pair_signals(pair: FramePair) -> List[InteractionSignal]:
        """Collect interaction signals that occurred between frame_a and frame_b."""
        if not pair.frame_b.signals_since_prev:
            return []
        t_start = pair.frame_a.timestamp
        t_end = pair.frame_b.timestamp
        return [
            s for s in pair.frame_b.signals_since_prev
            if t_start <= s.timestamp <= t_end
        ]

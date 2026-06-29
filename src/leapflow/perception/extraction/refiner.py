"""Stage A: Keyframe Refinement — dedup, classify, pair, budget allocation."""

from __future__ import annotations

from typing import Dict, List, Optional

from leapflow.perception.storage.deduplicator import FrameDeduplicator
from leapflow.perception.types import (
    FramePair,
    InferenceLevel,
    Keyframe,
    RefinedFrameSet,
)


class KeyframeRefiner:
    """Refine raw keyframe set before expensive VLM processing.

    Pipeline:
    1. Remove redundant frames (deduplication)
    2. Classify each frame's likely transition type using local features
    3. Build optimal frame pairs for VLM extraction
    4. Estimate and allocate inference budget per pair
    """

    __slots__ = ("_deduplicator",)

    def __init__(self, deduplicator: Optional[FrameDeduplicator] = None) -> None:
        self._deduplicator = deduplicator or FrameDeduplicator()

    async def refine(self, keyframes: List[Keyframe]) -> RefinedFrameSet:
        """Process raw keyframes into an optimized set with pairs and budget."""
        if not keyframes:
            return RefinedFrameSet()

        # Step 1: Dedup
        unique = self._deduplicator.deduplicate_offline(keyframes)

        # Step 2: Classify transitions
        for i in range(len(unique) - 1):
            unique[i].transition_type = self._classify_transition(unique[i], unique[i + 1])

        # Step 3: Build pairs
        pairs = self._build_pairs(unique)

        # Step 4: Budget allocation
        budget = self._allocate_budget(pairs)

        return RefinedFrameSet(frames=unique, pairs=pairs, budget=budget)

    def _classify_transition(self, frame_a: Keyframe, frame_b: Keyframe) -> str:
        """Classify what type of action likely happened between frames.

        Uses only local features (no VLM call):
        - App detection difference → app switch
        - Text region changes → text input
        - UI element changes → click/interaction
        - Embedding distance → navigation vs scroll
        """
        a_feat = frame_a.features
        b_feat = frame_b.features

        if not a_feat or not b_feat:
            return "unknown"

        # App change
        if a_feat.detected_app and b_feat.detected_app:
            if a_feat.detected_app != b_feat.detected_app:
                return "app_switch"

        # Text input detection
        new_text_count = self._count_new_text(a_feat.text_regions, b_feat.text_regions)
        if new_text_count > 0 and self._has_input_field(b_feat.ui_elements):
            return "text_input"

        # Dialog detection
        if self._has_new_dialog(a_feat.ui_elements, b_feat.ui_elements):
            return "dialog_popup"

        # Embedding-based distance
        if a_feat.embedding and b_feat.embedding:
            dist = self._cosine_distance(a_feat.embedding, b_feat.embedding)
            if dist > 0.3:
                return "navigation"
            if dist > 0.1:
                return "scroll"

        return "unknown"

    def _build_pairs(self, frames: List[Keyframe]) -> List[FramePair]:
        """Build sequential frame pairs for VLM extraction."""
        pairs = []
        for i in range(len(frames) - 1):
            pairs.append(FramePair(
                frame_a=frames[i],
                frame_b=frames[i + 1],
                transition_type=frames[i].transition_type,
            ))
        return pairs

    def _allocate_budget(self, pairs: List[FramePair]) -> Dict[InferenceLevel, int]:
        """Estimate how many pairs will go to each inference level."""
        budget: Dict[InferenceLevel, int] = {
            InferenceLevel.SKIP: 0,
            InferenceLevel.LIGHT: 0,
            InferenceLevel.STANDARD: 0,
            InferenceLevel.DEEP: 0,
        }

        for pair in pairs:
            level = self._estimate_level(pair.transition_type)
            budget[level] += 1

        return budget

    @staticmethod
    def _estimate_level(transition_type: str) -> InferenceLevel:
        """Estimate inference level from transition type."""
        if transition_type in ("scroll", "wait"):
            return InferenceLevel.LIGHT
        if transition_type == "app_switch":
            return InferenceLevel.LIGHT
        if transition_type in ("text_input", "click_target", "dialog_popup"):
            return InferenceLevel.STANDARD
        return InferenceLevel.DEEP

    @staticmethod
    def _count_new_text(regions_a: List, regions_b: List) -> int:
        texts_a = {r.get("text", "") for r in regions_a}
        texts_b = {r.get("text", "") for r in regions_b}
        return len(texts_b - texts_a)

    @staticmethod
    def _has_input_field(ui_elements: List) -> bool:
        return any(e.get("cls") == "text_field" for e in ui_elements)

    @staticmethod
    def _has_new_dialog(elements_a: List, elements_b: List) -> bool:
        dialogs_a = sum(1 for e in elements_a if e.get("cls") == "dialog")
        dialogs_b = sum(1 for e in elements_b if e.get("cls") == "dialog")
        return dialogs_b > dialogs_a

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 1.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

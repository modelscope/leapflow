"""Fusion quality assessment.

Computes quality metrics from fusion results to guide downstream
confidence decisions and surface warnings to users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from leapflow.signal_fusion.types import AtomicAction, FusionMode


class QualityLevel:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class FusionQuality:
    """Quality assessment of a fusion run."""

    channel_coverage: Dict[str, float] = field(default_factory=dict)
    avg_action_confidence: float = 0.0
    low_confidence_ratio: float = 0.0
    high_confidence_ratio: float = 0.0
    match_ratio: float = 0.0
    dominant_fusion_mode: str = FusionMode.MINIMAL.value
    warnings: List[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.avg_action_confidence > 0.8 and self.high_confidence_ratio > 0.7:
            return QualityLevel.HIGH
        if self.avg_action_confidence > 0.6:
            return QualityLevel.MEDIUM
        return QualityLevel.LOW

    @staticmethod
    def from_actions(
        actions: List[AtomicAction],
        *,
        match_ratio: float = 0.0,
        visual_available: bool = True,
        events_available: bool = True,
    ) -> "FusionQuality":
        """Compute quality metrics from a list of fused actions."""
        if not actions:
            return FusionQuality(warnings=["no_actions_produced"])

        confidences = [a.confidence for a in actions]
        avg_conf = sum(confidences) / len(confidences)
        low_ratio = sum(1 for c in confidences if c < 0.5) / len(confidences)
        high_ratio = sum(1 for c in confidences if c >= 0.8) / len(confidences)

        mode_counts: Dict[str, int] = {}
        for a in actions:
            mode_counts[a.fusion_mode.value] = mode_counts.get(a.fusion_mode.value, 0) + 1
        dominant = max(mode_counts, key=mode_counts.get) if mode_counts else FusionMode.MINIMAL.value

        coverage: Dict[str, float] = {}
        if visual_available:
            visual_count = sum(1 for a in actions if a.has_visual)
            coverage["visual"] = visual_count / len(actions)
        if events_available:
            event_count = sum(1 for a in actions if a.has_event)
            coverage["event"] = event_count / len(actions)

        warnings: List[str] = []
        if not visual_available:
            warnings.append("visual_channel_unavailable")
        if not events_available:
            warnings.append("event_channel_unavailable")
        if low_ratio > 0.5:
            warnings.append("majority_low_confidence")
        if avg_conf < 0.5:
            warnings.append("overall_low_quality")

        return FusionQuality(
            channel_coverage=coverage,
            avg_action_confidence=avg_conf,
            low_confidence_ratio=low_ratio,
            high_confidence_ratio=high_ratio,
            match_ratio=match_ratio,
            dominant_fusion_mode=dominant,
            warnings=warnings,
        )

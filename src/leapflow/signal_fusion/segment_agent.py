"""Segment-scale fusion: sub-task identification with wait-period awareness.

Groups AtomicActions into Segments based on app transitions, temporal
gaps, and action semantics. Annotates wait periods (AI generation,
loading, idle) without cutting on them.

OCP: new boundary strategies are added via register_detector().
SRP: segments and annotates — does not fuse evidence or infer intent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from leapflow.signal_fusion.protocol import FusionContext, FusionResult
from leapflow.signal_fusion.quality import FusionQuality
from leapflow.signal_fusion.types import (
    AppTransitionEvent,
    AtomicAction,
    Segment,
    SilentPeriodClass,
    WaitPeriod,
)
from leapflow.signal_fusion.wait_classifier import GapContext, WaitPeriodClassifier

logger = logging.getLogger(__name__)


# ── Boundary Detection ──


@dataclass(frozen=True)
class SegmentBoundary:
    """A detected boundary between segments."""

    index: int
    reason: str
    force: bool = False


class BoundaryStrategy(ABC):
    """Abstract base for segment boundary detection strategies."""

    @abstractmethod
    def detect(
        self,
        actions: Sequence[AtomicAction],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[SegmentBoundary]: ...


class AppTransitionBoundary(BoundaryStrategy):
    """Force a boundary on app transitions (strongest signal)."""

    def detect(
        self,
        actions: Sequence[AtomicAction],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[SegmentBoundary]:
        if not transitions or not actions:
            return []

        boundaries: List[SegmentBoundary] = []
        for tr in transitions:
            idx = _find_nearest_action_index(actions, tr.ts)
            if idx is not None and 0 < idx < len(actions):
                boundaries.append(SegmentBoundary(
                    index=idx,
                    reason=f"app_transition:{tr.from_bundle}->{tr.to_bundle}",
                    force=True,
                ))
        return boundaries


class TemporalGapBoundary(BoundaryStrategy):
    """Split on large temporal gaps between actions."""

    def __init__(self, threshold_s: float = 10.0) -> None:
        self._threshold = threshold_s

    def detect(
        self,
        actions: Sequence[AtomicAction],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[SegmentBoundary]:
        boundaries: List[SegmentBoundary] = []
        for i in range(1, len(actions)):
            gap = actions[i].timestamp - actions[i - 1].timestamp
            if gap >= self._threshold:
                boundaries.append(SegmentBoundary(
                    index=i, reason=f"temporal_gap:{gap:.1f}s"
                ))
        return boundaries


class AppChangeBoundary(BoundaryStrategy):
    """Split when the dominant app changes between consecutive actions."""

    def detect(
        self,
        actions: Sequence[AtomicAction],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[SegmentBoundary]:
        boundaries: List[SegmentBoundary] = []
        for i in range(1, len(actions)):
            if (
                actions[i].app_bundle
                and actions[i - 1].app_bundle
                and actions[i].app_bundle != actions[i - 1].app_bundle
            ):
                boundaries.append(SegmentBoundary(
                    index=i,
                    reason=f"app_change:{actions[i-1].app_bundle}->{actions[i].app_bundle}",
                ))
        return boundaries


# ── Segment Fusion Agent ──


class SegmentFusionAgent:
    """Segment-scale fusion agent.

    Implements ScaleFusionAgent protocol. Consumes upstream AtomicActions
    and produces Segments with wait-period annotations.
    """

    def __init__(
        self,
        wait_classifier: Optional[WaitPeriodClassifier] = None,
        strategies: Optional[List[BoundaryStrategy]] = None,
    ) -> None:
        self._wait_classifier = wait_classifier or WaitPeriodClassifier()
        self._strategies: List[BoundaryStrategy] = strategies or [
            AppTransitionBoundary(),
            AppChangeBoundary(),
            TemporalGapBoundary(),
        ]

    def register_detector(self, strategy: BoundaryStrategy) -> None:
        """Add a new boundary detection strategy (OCP)."""
        self._strategies.append(strategy)

    async def fuse(self, context: FusionContext) -> FusionResult:
        upstream = context.upstream_result
        if not upstream or not upstream.atomic_actions:
            return FusionResult()

        actions = upstream.atomic_actions
        boundaries = self._detect_boundaries(actions, context.app_transitions)
        segments = self._build_segments(actions, boundaries)

        for seg in segments:
            self._annotate_wait_periods(seg)

        quality = FusionQuality.from_actions(
            actions,
            visual_available=(context.channel_status.screen_capture_available
                              if context.channel_status else True),
            events_available=(context.channel_status.ui_events_available
                              if context.channel_status else True),
        )

        return FusionResult(
            atomic_actions=actions,
            segments=segments,
            quality=quality,
        )

    def _detect_boundaries(
        self,
        actions: Sequence[AtomicAction],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[int]:
        """Collect boundaries from all strategies, deduplicate, sort."""
        all_boundaries: List[SegmentBoundary] = []
        for strategy in self._strategies:
            all_boundaries.extend(strategy.detect(actions, transitions))

        seen: set = set()
        unique: List[int] = []
        for b in sorted(all_boundaries, key=lambda x: x.index):
            if b.index not in seen:
                seen.add(b.index)
                unique.append(b.index)
        return unique

    @staticmethod
    def _build_segments(
        actions: List[AtomicAction], boundaries: List[int]
    ) -> List[Segment]:
        if not actions:
            return []

        cut_points = sorted(set([0] + boundaries + [len(actions)]))
        segments: List[Segment] = []

        for i in range(len(cut_points) - 1):
            start, end = cut_points[i], cut_points[i + 1]
            if start >= end:
                continue
            seg_actions = actions[start:end]
            app_counts: dict = {}
            for a in seg_actions:
                if a.app_bundle:
                    app_counts[a.app_bundle] = app_counts.get(a.app_bundle, 0) + 1
            dominant = max(app_counts, key=app_counts.get) if app_counts else ""

            segments.append(Segment(
                actions=seg_actions,
                dominant_app=dominant,
                boundary_reason=f"boundary_at_{start}",
            ))

        return segments

    def _annotate_wait_periods(self, segment: Segment) -> None:
        """Detect and annotate wait periods within a segment."""
        actions = segment.actions
        for i in range(1, len(actions)):
            gap = actions[i].timestamp - actions[i - 1].timestamp
            if gap < 2.0:
                continue

            gap_ctx = GapContext(
                current_app_bundle=actions[i - 1].app_bundle,
                last_action_type=actions[i - 1].action,
            )
            classification = self._wait_classifier.classify(gap, gap_ctx)
            if classification != SilentPeriodClass.NORMAL_PAUSE:
                segment.wait_periods.append(WaitPeriod(
                    start_ts=actions[i - 1].timestamp,
                    end_ts=actions[i].timestamp,
                    classification=classification,
                    context_app=actions[i - 1].app_bundle,
                ))


# ── Helpers ──


def _find_nearest_action_index(
    actions: Sequence[AtomicAction], target_ts: float
) -> Optional[int]:
    """Find the index of the action nearest to target_ts."""
    if not actions:
        return None
    best_idx = 0
    best_dist = abs(actions[0].timestamp - target_ts)
    for i in range(1, len(actions)):
        dist = abs(actions[i].timestamp - target_ts)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

"""Episode segmentation — split trajectories into semantically coherent chunks.

Uses a chain of heuristic boundary detectors (Strategy pattern) so new
segmentation signals can be added without modifying existing logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from leapflow.domain.trajectory import ActionType, Episode, Trajectory, TrajectoryStep

logger = logging.getLogger(__name__)

_HIGH_CONFIDENCE_NOISE = 0.9


def _is_signal_step(step: TrajectoryStep) -> bool:
    """Whether a step carries meaningful signal for segmentation purposes."""
    if step.action.action_type == ActionType.UNKNOWN:
        return False
    noise = step.action.params.get("_noise", [])
    return not any(n.get("confidence", 0) >= _HIGH_CONFIDENCE_NOISE for n in noise)


# ── Boundary detector protocol ──


@dataclass(frozen=True)
class Boundary:
    """A detected episode boundary at a specific step index."""

    index: int
    reason: str


class BoundaryDetector(ABC):
    """Detects potential episode boundaries in a trajectory."""

    @abstractmethod
    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        """Return boundary indices where episodes should be split."""


# ── Concrete detectors ──


class TimeGapDetector(BoundaryDetector):
    """Split when adjacent steps have a large time gap."""

    def __init__(self, threshold_seconds: float = 300.0) -> None:
        self._threshold = threshold_seconds

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        boundaries: List[Boundary] = []
        for i in range(1, len(trajectory.steps)):
            gap = (
                trajectory.steps[i].action.timestamp
                - trajectory.steps[i - 1].action.timestamp
            )
            if gap >= self._threshold:
                boundaries.append(Boundary(i, f"time_gap_{gap:.0f}s"))
        return boundaries


class AppSwitchDetector(BoundaryDetector):
    """Split on app switches that follow a meaningful pause."""

    def __init__(self, min_gap_seconds: float = 60.0) -> None:
        self._min_gap = min_gap_seconds

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        boundaries: List[Boundary] = []
        for i in range(1, len(trajectory.steps)):
            curr = trajectory.steps[i]
            prev = trajectory.steps[i - 1]
            if curr.action.action_type != ActionType.APP_SWITCH:
                continue
            prev_app = prev.state.focused_app or prev.action.app_bundle_id
            if curr.action.app_bundle_id == prev_app:
                continue
            gap = curr.action.timestamp - prev.action.timestamp
            if gap >= self._min_gap:
                boundaries.append(
                    Boundary(i, f"app_switch_{curr.action.app_bundle_id}")
                )
        return boundaries


class SemanticBoundaryDetector(BoundaryDetector):
    """Detects boundaries based on semantic action type changes.

    Identifies transitions between distinct activity patterns:
    - Switching from file operations to web browsing
    - Switching from reading to writing
    - Transitioning between different task domains

    Uses a sliding window to avoid splitting on transient category flickers.
    """

    # Action type → semantic domain mapping
    _DOMAIN_MAP = {
        ActionType.FILE_CREATE: "file_ops",
        ActionType.FILE_MODIFY: "file_ops",
        ActionType.FILE_DELETE: "file_ops",
        ActionType.FILE_RENAME: "file_ops",
        ActionType.CLIPBOARD_COPY: "data_transfer",
        ActionType.APP_SWITCH: "navigation",
        ActionType.UI_CLICK: "ui_interaction",
        ActionType.UI_TYPE: "text_editing",
        ActionType.UI_SHORTCUT: "ui_interaction",
        ActionType.UI_SCROLL: "browsing",
        ActionType.UNKNOWN: "other",
    }

    def __init__(self, min_segment_length: int = 3) -> None:
        self._min_length = min_segment_length

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        """Find boundaries where action semantic category changes significantly.

        Only signal steps (non-UNKNOWN, non-high-confidence-noise) participate
        in domain transition analysis. Boundaries reference original indices.
        """
        signal = [(i, s) for i, s in enumerate(trajectory.steps) if _is_signal_step(s)]
        if len(signal) < self._min_length * 2:
            return []

        boundaries: List[Boundary] = []
        prev_domain = self._categorize_action(signal[0][1].action.action_type)
        consecutive = 1
        last_boundary_idx = signal[0][0]

        for pos in range(1, len(signal)):
            idx, step = signal[pos]
            curr_domain = self._categorize_action(step.action.action_type)

            if curr_domain == prev_domain or curr_domain == "navigation":
                consecutive += 1
                continue

            if self._is_significant_shift(prev_domain, curr_domain, consecutive):
                if idx - last_boundary_idx >= self._min_length:
                    boundaries.append(
                        Boundary(idx, f"semantic_shift_{prev_domain}_to_{curr_domain}")
                    )
                    last_boundary_idx = idx

            prev_domain = curr_domain
            consecutive = 1

        return boundaries

    def _categorize_action(self, action_type: ActionType) -> str:
        """Map ActionType to semantic category."""
        return self._DOMAIN_MAP.get(action_type, "other")

    def _is_significant_shift(
        self, prev_category: str, curr_category: str, consecutive_count: int
    ) -> bool:
        """Determine if category shift is significant enough to be a boundary.

        - navigation alone is transitional, not a boundary.
        - Requires at least min_length consecutive actions in the previous domain.
        - Transitions between semantically distant domains are always significant.
        """
        if curr_category == "navigation":
            return False
        if consecutive_count < self._min_length:
            return False

        # Semantically close pairs: not a significant shift
        close_pairs = {
            frozenset({"ui_interaction", "text_editing"}),
            frozenset({"ui_interaction", "browsing"}),
        }
        if frozenset({prev_category, curr_category}) in close_pairs:
            return consecutive_count >= self._min_length + 2

        return True


class MaxEpisodeLengthDetector(BoundaryDetector):
    """Force-split episodes exceeding a maximum signal step count.

    Only counts signal steps (non-UNKNOWN, non-high-confidence-noise) toward
    the limit. Boundaries are placed at the trajectory index following the
    Nth signal step, ensuring noise-heavy trajectories are not over-segmented.
    """

    def __init__(self, max_steps: int = 30) -> None:
        self._max_steps = max_steps

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        boundaries: List[Boundary] = []
        signal_count = 0
        for i, step in enumerate(trajectory.steps):
            if _is_signal_step(step):
                signal_count += 1
                if signal_count >= self._max_steps:
                    boundaries.append(Boundary(i + 1, f"max_length_{self._max_steps}"))
                    signal_count = 0
        return boundaries


class AdaptiveMaxLengthDetector(BoundaryDetector):
    """Adaptive max-length splitting based on session duration.

    Short sessions (single workflow demonstrations) should not be
    mechanically split — they represent a complete coherent workflow.
    For longer sessions, applies the standard max_steps limit.
    """

    def __init__(
        self,
        base_max_steps: int = 30,
        short_session_threshold_s: float = 180.0,
    ) -> None:
        self._base_max_steps = base_max_steps
        self._short_threshold = short_session_threshold_s

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        if not trajectory.steps:
            return []

        duration = trajectory.steps[-1].action.timestamp - trajectory.steps[0].action.timestamp
        if duration < self._short_threshold:
            return []

        inner = MaxEpisodeLengthDetector(self._base_max_steps)
        return inner.detect(trajectory)


class MinEpisodeLengthFilter(BoundaryDetector):
    """Suppress boundaries that would create episodes shorter than a minimum."""

    def __init__(self, inner: BoundaryDetector, min_steps: int = 2) -> None:
        self._inner = inner
        self._min_steps = min_steps

    def detect(self, trajectory: Trajectory) -> List[Boundary]:
        raw = self._inner.detect(trajectory)
        if not raw:
            return raw
        filtered: List[Boundary] = []
        prev_idx = 0
        for b in sorted(raw, key=lambda b: b.index):
            if b.index - prev_idx >= self._min_steps:
                filtered.append(b)
                prev_idx = b.index
        return filtered


# ── Segmenter ──


class SegmentDetector:
    """Splits a trajectory into episodes using pluggable boundary detectors.

    Detectors are evaluated in order; their boundaries are merged and
    deduplicated by index.
    """

    def __init__(self, detectors: Sequence[BoundaryDetector] | None = None) -> None:
        self._detectors = list(detectors) if detectors else self._default_detectors()

    def segment(self, trajectory: Trajectory) -> List[Episode]:
        """Return a list of Episodes covering the full trajectory."""
        if not trajectory.steps:
            return []

        boundaries = self._merge_boundaries(trajectory)
        return self._build_episodes(trajectory, boundaries)

    def _merge_boundaries(self, trajectory: Trajectory) -> List[int]:
        """Collect boundaries from all detectors, deduplicate and sort."""
        indices: set[int] = set()
        for detector in self._detectors:
            for b in detector.detect(trajectory):
                indices.add(b.index)
        return sorted(indices)

    def _build_episodes(
        self, trajectory: Trajectory, boundaries: List[int]
    ) -> List[Episode]:
        edges = [0] + boundaries + [len(trajectory.steps)]
        episodes: List[Episode] = []
        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            if start >= end:
                continue
            episode_steps = trajectory.steps[start:end]
            apps = _extract_app_sequence(episode_steps)
            episodes.append(
                Episode(
                    trajectory_id=trajectory.trajectory_id,
                    start_idx=start,
                    end_idx=end,
                    app_sequence=apps,
                )
            )
        logger.debug(
            "segmented trajectory=%s into %d episodes",
            trajectory.trajectory_id,
            len(episodes),
        )
        return episodes

    @staticmethod
    def _default_detectors() -> List[BoundaryDetector]:
        return [
            TimeGapDetector(threshold_seconds=300.0),
            AppSwitchDetector(min_gap_seconds=60.0),
            SemanticBoundaryDetector(min_segment_length=3),
            AdaptiveMaxLengthDetector(base_max_steps=30),
        ]


def _extract_app_sequence(steps: Sequence) -> List[str]:
    """Ordered unique app bundle IDs from a step slice."""
    seen: set[str] = set()
    result: List[str] = []
    for step in steps:
        bid = step.action.app_bundle_id
        if bid and bid not in seen:
            seen.add(bid)
            result.append(bid)
    return result

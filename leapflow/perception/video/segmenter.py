"""Video segment splitter for multi-scale VLM analysis.

Splits recorded video segments into semantically coherent analysis
units based on app transitions, idle gaps, and configurable limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from leapflow.perception.types import TimelineMarker, VideoSegment


@dataclass
class AnalysisSegment:
    """A video sub-range ready for VLM analysis."""

    segment: VideoSegment
    start_offset: float
    end_offset: float
    markers: List[TimelineMarker] = field(default_factory=list)
    app_summary: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_offset - self.start_offset


class VideoSegmenter:
    """Splits video segments into analysis-friendly chunks.

    Physical file boundaries come from the host encoder.  This layer
    applies *semantic* splitting within those files based on event
    timeline data.
    """

    def __init__(
        self,
        *,
        min_segment_s: float = 30.0,
        max_segment_s: float = 600.0,
        idle_gap_s: float = 15.0,
        app_switch_gap_s: float = 5.0,
        min_split_s: float = 1.0,
    ) -> None:
        self._min_s = min_segment_s
        self._max_s = max_segment_s
        self._idle_gap_s = idle_gap_s
        self._app_gap_s = app_switch_gap_s
        self._min_split_s = min_split_s

    def segment(
        self,
        video_segments: List[VideoSegment],
        markers: List[TimelineMarker],
    ) -> List[AnalysisSegment]:
        """Produce analysis segments from physical video files + markers."""
        results: List[AnalysisSegment] = []
        for vs in video_segments:
            seg_markers = [m for m in markers if vs.start_time <= m.timestamp <= vs.end_time]
            sub_ranges = self._split_one(vs, seg_markers)
            for start, end, sub_m in sub_ranges:
                results.append(AnalysisSegment(
                    segment=vs,
                    start_offset=start - vs.start_time,
                    end_offset=end - vs.start_time,
                    markers=sub_m,
                    app_summary=_app_duration_summary(sub_m),
                ))
        return self._merge_short(results)

    def _split_one(self, vs: VideoSegment, markers: List[TimelineMarker]):
        """Find split points within a single physical segment."""
        if not markers:
            return [(vs.start_time, vs.end_time, markers)]

        boundaries: List[float] = []
        prev_t = markers[0].timestamp
        prev_app = markers[0].app

        for m in markers[1:]:
            gap = m.timestamp - prev_t
            if gap >= self._idle_gap_s:
                boundaries.append(prev_t + gap / 2)
            elif m.channel == "app_switch" and gap >= self._app_gap_s and m.app != prev_app:
                boundaries.append(m.timestamp)
            prev_t = m.timestamp
            if m.app:
                prev_app = m.app

        if not boundaries:
            return [(vs.start_time, vs.end_time, markers)]

        ranges = []
        cuts = [vs.start_time] + sorted(set(boundaries)) + [vs.end_time]
        for i in range(len(cuts) - 1):
            s, e = cuts[i], cuts[i + 1]
            if e - s < self._min_split_s:
                continue
            sub_m = [m for m in markers if s <= m.timestamp <= e]
            ranges.append((s, e, sub_m))
        return ranges or [(vs.start_time, vs.end_time, markers)]

    def _merge_short(self, segments: List[AnalysisSegment]) -> List[AnalysisSegment]:
        """Merge segments shorter than *_min_s* with their neighbour."""
        if len(segments) <= 1:
            return segments

        merged: List[AnalysisSegment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if prev.duration < self._min_s and prev.segment == seg.segment:
                merged[-1] = AnalysisSegment(
                    segment=prev.segment,
                    start_offset=prev.start_offset,
                    end_offset=seg.end_offset,
                    markers=prev.markers + seg.markers,
                    app_summary=_merge_dicts(prev.app_summary, seg.app_summary),
                )
            else:
                merged.append(seg)
        return merged


def _app_duration_summary(markers: List[TimelineMarker]) -> Dict[str, float]:
    if not markers:
        return {}
    summary: Dict[str, float] = {}
    current_app = markers[0].app
    prev_t = markers[0].timestamp
    for m in markers[1:]:
        if current_app:
            summary[current_app] = summary.get(current_app, 0) + (m.timestamp - prev_t)
        prev_t = m.timestamp
        if m.app:
            current_app = m.app
    return summary


def _merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out

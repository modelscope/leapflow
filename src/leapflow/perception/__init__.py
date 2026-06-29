"""Visual-First Perception Subsystem.

Supports two modes:
  - Screenshot (legacy): adaptive keyframe capture + VLM extraction
  - Video (recommended): continuous recording + multi-scale VLM analysis

Core types are re-exported from perception.types; video components
from perception.video.
"""

from leapflow.perception.config import PerceptionConfig, SamplingConfig, ScorerConfig
from leapflow.perception.session import PerceptionSession
from leapflow.perception.types import (
    ChannelStatus,
    InteractionSignal,
    Keyframe,
    MacroAnalysisResult,
    TimelineMarker,
    VideoAction,
    VideoSegment,
    VisualAction,
)

__all__ = [
    "PerceptionConfig",
    "PerceptionSession",
    "SamplingConfig",
    "ScorerConfig",
    "ChannelStatus",
    "InteractionSignal",
    "Keyframe",
    "MacroAnalysisResult",
    "TimelineMarker",
    "VideoAction",
    "VideoSegment",
    "VisualAction",
]

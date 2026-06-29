"""Real-time recording layer — attention filtering, event capture, and frame storage."""

from leapflow.recording.attention import (
    AttentionFilter,
    DomainWhitelistFilter,
    FilterResult,
    FilterVerdict,
    ForegroundGateFilter,
    NoiseRuleFilter,
    RecordingContext,
    WorkingDirFilter,
    build_attention_filters,
)
from leapflow.recording.recorder import DemonstrationRecorder

__all__ = [
    "AttentionFilter",
    "DemonstrationRecorder",
    "DomainWhitelistFilter",
    "FilterResult",
    "FilterVerdict",
    "ForegroundGateFilter",
    "NoiseRuleFilter",
    "RecordingContext",
    "WorkingDirFilter",
    "build_attention_filters",
]

"""Shared data types for the perception subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Enums ──


class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class InferenceLevel(Enum):
    SKIP = "skip"
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"


class FastPathResult(Enum):
    CONTINUE = "continue"
    CAPTURE_NOW = "capture_now"
    CAPTURE_DUAL = "capture_dual"

    @classmethod
    def lower_threshold(cls, threshold: float) -> "FastPathDecision":
        return FastPathDecision(action=cls.CONTINUE, threshold=threshold)


@dataclass(frozen=True)
class FastPathDecision:
    action: FastPathResult
    threshold: float = 0.0

    @property
    def has_threshold(self) -> bool:
        return self.threshold > 0.0


# ── Online Phase Types ──


@dataclass(frozen=True)
class ChangeSignal:
    """Result of multi-scale change detection between consecutive frames."""

    global_diff: float
    max_quadrant_diff: float
    changed_quadrant: int
    focus_diff: float
    is_significant: bool


@dataclass(frozen=True)
class SamplingDecision:
    """Output of the sampling engine's decide() call."""

    capture: bool
    reason: str = ""
    priority: Priority = Priority.NORMAL
    roi_hint: Optional[Tuple[int, int, int, int]] = None


SKIP_DECISION = SamplingDecision(capture=False, reason="skip")


@dataclass
class SamplingContext:
    """Contextual information fed into the information gain scorer."""

    recent_events: List[Any] = field(default_factory=list)
    time_since_last_capture: float = 0.0
    event_just_fired: bool = False


@dataclass
class CaptureRecord:
    """Metadata about a captured frame (kept in scorer's recent buffer)."""

    timestamp: float
    global_hash: bytes
    score: float
    reason: str


@dataclass
class ChannelStatus:
    """Describes which perception channels are currently available."""

    ui_events_available: bool = True
    app_focus_available: bool = True
    clipboard_available: bool = True
    screen_capture_available: bool = True


@dataclass
class SamplingThresholds:
    """Dynamically adjusted thresholds based on channel degradation."""

    change_threshold: float = 0.15
    info_score_threshold: float = 0.5
    max_silent_s: float = 5.0
    frame_budget_per_min: int = 20


# ── Storage & Frame Types ──


@dataclass(frozen=True)
class InteractionSignal:
    """A lightweight interaction signal captured between frames.

    These are NOT trajectory steps — they serve as temporal anchors
    and disambiguation hints for the visual-first VLM pipeline.
    """

    timestamp: float
    signal_type: str  # "click"|"app_switch"|"clipboard"|"keyboard"|"scroll"|"drag"
    app: str = ""
    position: Optional[Tuple[int, int]] = None
    end_position: Optional[Tuple[int, int]] = None
    detail: str = ""


@dataclass
class Keyframe:
    """A captured keyframe with metadata and optional features."""

    ref: str
    timestamp: float
    image: bytes
    trigger: str = ""
    info_score: float = 0.0
    transition_type: str = ""
    features: Optional["FrameFeatures"] = None
    signals_since_prev: List["InteractionSignal"] = field(default_factory=list)


@dataclass
class FrameFeatures:
    """CV-extracted features for a single frame."""

    frame_ref: str
    timestamp: float
    text_regions: List[Dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    detected_app: str = ""
    focus_region: Optional[Tuple[int, int, int, int]] = None


@dataclass
class FramePair:
    """A pair of keyframes representing a before/after transition."""

    frame_a: Keyframe
    frame_b: Keyframe
    transition_type: str = ""
    change_signal: Optional[ChangeSignal] = None
    context: Optional["PairContext"] = None
    pre_extracted_action: Optional["VisualAction"] = None


@dataclass
class PairContext:
    """CV-derived context for a frame pair (fed into VLM prompts)."""

    app_a: str = ""
    app_b: str = ""
    app_changed: bool = False
    diff_regions: List[Dict[str, Any]] = field(default_factory=list)
    new_text: List["TextChange"] = field(default_factory=list)
    removed_text: List["TextChange"] = field(default_factory=list)
    new_ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    removed_ui_elements: List[Dict[str, Any]] = field(default_factory=list)
    time_delta: float = 0.0
    signals: List["InteractionSignal"] = field(default_factory=list)


@dataclass(frozen=True)
class VisualAction:
    """A user action extracted from visual frame analysis."""

    action: str
    target: str = ""
    detail: str = ""
    confidence: float = 0.0
    evidence: str = ""
    frame_ref_a: str = ""
    frame_ref_b: str = ""


@dataclass
class RefinedFrameSet:
    """Output of keyframe refinement (Stage A)."""

    frames: List[Keyframe] = field(default_factory=list)
    pairs: List[FramePair] = field(default_factory=list)
    budget: Dict[InferenceLevel, int] = field(default_factory=dict)


# ── Encoding Types ──


@dataclass(frozen=True)
class EncodedFrame:
    """Result of adaptive resolution encoding."""

    data: bytes
    resolution: int
    quality: int
    roi: Optional[Tuple[int, int, int, int]] = None
    frame_type: str = ""
    size_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        return self.size_bytes / max(1, len(self.data))


@dataclass(frozen=True)
class ComposedImage:
    """A composed frame pair image ready for VLM input."""

    image: bytes
    layout: str
    token_estimate: int = 0
    crop_region: Optional[Tuple[int, int, int, int]] = None
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class TiledBatch:
    """A grid of composed pairs for batched VLM inference."""

    image: bytes
    pairs: List[ComposedImage] = field(default_factory=list)
    pair_count: int = 0
    prompt: str = ""


# ── CV Algorithm Types ──


@dataclass(frozen=True)
class FlowAnalysis:
    """Result of optical flow analysis between two frames."""

    mean_magnitude: float = 0.0
    max_magnitude: float = 0.0
    is_scroll: bool = False
    scroll_direction: Optional[str] = None
    localized_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)
    motion_type: str = "static"


@dataclass(frozen=True)
class SceneCutResult:
    """Result of scene cut detection."""

    is_cut: bool = False
    cut_type: str = "minor_update"
    confidence: float = 0.5


@dataclass(frozen=True)
class TextChange:
    """A single text change between frames."""

    text: str = ""
    prev_text: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None
    type: str = "added"


@dataclass
class TextDiff:
    """Aggregate text changes between two frames."""

    added: List[TextChange] = field(default_factory=list)
    removed: List[TextChange] = field(default_factory=list)
    modified: List[TextChange] = field(default_factory=list)


# ── Video Types ──


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for a single recorded video segment file."""

    segment_id: str
    session_id: str
    file_path: Path
    start_time: float
    end_time: float
    duration: float
    fps: float
    resolution: Tuple[int, int]
    codec: str
    file_size_bytes: int = 0


@dataclass(frozen=True)
class TimelineMarker:
    """An event marker on the video timeline for VLM context."""

    timestamp: float
    channel: str
    action: str
    app: str
    coordinates: Optional[Tuple[int, int]] = None
    payload_digest: str = ""
    priority: int = 5


@dataclass(frozen=True)
class VideoAction:
    """A semantic action extracted from video analysis."""

    action_name: str
    description: str
    start_time: float
    end_time: float
    app: str = ""
    goal: str = ""
    confidence: float = 0.0
    analysis_level: int = 1
    corroborating_events: Tuple[str, ...] = ()
    frame_refs: Tuple[str, ...] = ()


@dataclass
class MacroAnalysisResult:
    """Output of L1 macro video analysis for a single segment."""

    actions: List[VideoAction] = field(default_factory=list)
    overall_goal: str = ""
    detail_requests: List[Dict[str, Any]] = field(default_factory=list)
    frame_requests: List[Dict[str, Any]] = field(default_factory=list)

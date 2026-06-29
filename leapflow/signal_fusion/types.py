"""Core data types for multi-source heterogeneous multi-scale signal fusion.

Defines the fused data model hierarchy:
    AtomicAction  — single user action fused from visual + event evidence
    Segment       — sub-task grouping with wait-period awareness
    WorkflowNode  — single-app phase in a cross-app workflow
    WorkflowEdge  — transition between workflow nodes with carry payload
    WorkflowGraph — DAG of nodes/edges representing a multi-app workflow
    EnrichedEpisode — fusion-enriched episode with workflow context
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent
    from leapflow.perception.types import VisualAction


# ── Enums ──


class FusionMode(Enum):
    """How an AtomicAction was fused."""

    FULL = "full"                    # visual + event corroborated
    VISUAL_PRIMARY = "visual_primary"  # visual only, event unavailable/unmatched
    EVENT_PRIMARY = "event_primary"    # event only, no visual match
    MINIMAL = "minimal"              # low-quality single source


class SilentPeriodClass(Enum):
    """Classification of a gap between actions."""

    NORMAL_PAUSE = "normal_pause"
    AI_GENERATING = "ai_generating"
    USER_IDLE = "user_idle"
    LOADING = "loading"
    UNKNOWN_WAIT = "unknown_wait"


class WorkflowType(Enum):
    """Recognized cross-app workflow patterns."""

    LINEAR_TRANSFER = "linear_transfer"    # A → B
    ROUND_TRIP = "round_trip"              # A → B → A
    MULTI_HUB = "multi_hub"               # A → B → C → D
    PARALLEL_REFERENCE = "parallel_ref"    # A ↔ B
    ITERATIVE_REFINEMENT = "iterative"     # A ↔ B ↔ A ↔ B
    PIPELINE = "pipeline"                  # A₁ → B → A₂ → C → A₃
    UNKNOWN = "unknown"


class NodeRole(Enum):
    """Role of an app within a workflow."""

    SOURCE = "source"      # provides initial data
    TOOL = "tool"          # transforms/processes data
    SINK = "sink"          # receives final output
    REFERENCE = "reference"  # read-only consultation
    UNKNOWN = "unknown"


class CarryType(Enum):
    """Type of data carried between apps."""

    TEXT = "text"
    CODE = "code"
    URL = "url"
    FILE_PATH = "file_path"
    IMAGE = "image"
    STRUCTURED = "structured"
    EMPTY = "empty"


# ── Core Data Types ──


@dataclass
class AtomicAction:
    """A single user action fused from visual and event evidence.

    This is the fundamental unit of the MHMS-SF output, richer than both
    VisualAction (no event detail) and SemanticAction (no fusion metadata).
    """

    action: str
    target: str
    detail: str
    timestamp: float
    confidence: float
    fusion_mode: FusionMode = FusionMode.MINIMAL
    visual_evidence: str = ""
    frame_ref: str = ""
    clipboard_text: str = ""
    typed_text: str = ""
    shortcut: str = ""
    app_bundle: str = ""
    source_signals: List[str] = field(default_factory=list)
    semantic_role: str = ""
    tags: Set[str] = field(default_factory=set)

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    @property
    def has_visual(self) -> bool:
        return "visual" in self.source_signals

    @property
    def has_event(self) -> bool:
        return any(s != "visual" for s in self.source_signals)


@dataclass
class WaitPeriod:
    """An annotated gap between actions."""

    start_ts: float
    end_ts: float
    classification: SilentPeriodClass
    context_app: str = ""
    context_url: str = ""

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class Segment:
    """A sub-task grouping of atomic actions with optional wait annotations.

    Represents a coherent sequence of actions within a single app context,
    possibly containing annotated wait periods (e.g., AI generation).
    """

    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    actions: List[AtomicAction] = field(default_factory=list)
    dominant_app: str = ""
    wait_periods: List[WaitPeriod] = field(default_factory=list)
    boundary_reason: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def start_ts(self) -> float:
        return self.actions[0].timestamp if self.actions else 0.0

    @property
    def end_ts(self) -> float:
        return self.actions[-1].timestamp if self.actions else 0.0

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def action_count(self) -> int:
        return len(self.actions)


@dataclass
class CarryPayload:
    """Data carried between apps during a transition."""

    carry_type: CarryType
    preview: str = ""
    origin_app: str = ""

    @staticmethod
    def from_clipboard(text: Optional[str], origin: str = "") -> "CarryPayload":
        if not text:
            return CarryPayload(carry_type=CarryType.EMPTY, origin_app=origin)
        carry_type = _classify_carry_content(text)
        preview = text[:200] if text else ""
        return CarryPayload(carry_type=carry_type, preview=preview, origin_app=origin)


@dataclass(frozen=True)
class AppTransitionEvent:
    """First-class event for cross-app workflow tracking.

    Richer than SystemEvent("app.focus_change"): carries dual frame refs,
    clipboard snapshot, and inferred transition reason.
    """

    ts: float
    from_bundle: str
    to_bundle: str
    carry_clipboard: Optional[str] = None
    frame_before_ref: Optional[str] = None
    frame_after_ref: Optional[str] = None
    trigger: str = "unknown"
    inferred_reason: Optional[str] = None


# ── Workflow Graph ──


@dataclass
class WorkflowNode:
    """A single-app phase in a cross-app workflow."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    app_bundle: str = ""
    segment: Optional[Segment] = None
    role: NodeRole = NodeRole.UNKNOWN


@dataclass
class WorkflowEdge:
    """Transition between workflow nodes with carry payload metadata."""

    from_node_id: str = ""
    to_node_id: str = ""
    carry: CarryPayload = field(default_factory=lambda: CarryPayload(CarryType.EMPTY))
    transition_ts: float = 0.0


@dataclass
class WorkflowGraph:
    """DAG representing a multi-app workflow."""

    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    workflow_type: WorkflowType = WorkflowType.UNKNOWN

    @property
    def is_multi_app(self) -> bool:
        bundles = {n.app_bundle for n in self.nodes if n.app_bundle}
        return len(bundles) > 1

    @property
    def app_sequence(self) -> List[str]:
        return [n.app_bundle for n in self.nodes if n.app_bundle]

    @property
    def unique_apps(self) -> FrozenSet[str]:
        return frozenset(n.app_bundle for n in self.nodes if n.app_bundle)


@dataclass
class EnrichedEpisode:
    """A fusion-enriched episode with multi-scale annotations."""

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    segments: List[Segment] = field(default_factory=list)
    workflow_graph: Optional[WorkflowGraph] = None
    intent: str = ""
    intent_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cross_app(self) -> bool:
        return self.workflow_graph is not None and self.workflow_graph.is_multi_app

    @property
    def all_actions(self) -> List[AtomicAction]:
        return [a for s in self.segments for a in s.actions]

    @property
    def app_sequence(self) -> List[str]:
        return [s.dominant_app for s in self.segments if s.dominant_app]


# ── Alignment Helpers ──


@dataclass
class AlignmentResult:
    """Result of temporal alignment between visual and event signals."""

    matched_pairs: List[Tuple["VisualAction", List["SystemEvent"]]] = field(default_factory=list)
    unmatched_visual: List["VisualAction"] = field(default_factory=list)
    unmatched_events: List["SystemEvent"] = field(default_factory=list)

    @property
    def match_ratio(self) -> float:
        total = len(self.matched_pairs) + len(self.unmatched_visual) + len(self.unmatched_events)
        return len(self.matched_pairs) / total if total > 0 else 0.0


# ── Helpers ──


def _classify_carry_content(text: str) -> CarryType:
    """Classify clipboard content type by lightweight heuristics."""
    stripped = text.strip()
    if not stripped:
        return CarryType.EMPTY
    if stripped.startswith(("http://", "https://", "ftp://")):
        return CarryType.URL
    if stripped.startswith(("/", "~/", "C:\\")):
        return CarryType.FILE_PATH
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return CarryType.STRUCTURED
    code_indicators = ("def ", "function ", "class ", "import ", "const ", "var ", "let ")
    if any(ind in stripped for ind in code_indicators):
        return CarryType.CODE
    return CarryType.TEXT

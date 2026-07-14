"""Core data model for imitation learning trajectories.

Defines the experience hierarchy:
    Trajectory → Episode → SemanticAction
    TrajectoryStep = (StateSnapshot, RawAction)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RecordingMode(Enum):
    """Recording data-collection strategy.

    Each mode determines which data channels contribute trajectory steps.
    Behavioral predicates (``skip_structural_events``, ``needs_visual_polling``)
    let consumers query capabilities without hard-coding mode names.
    """

    DEFAULT = "default"
    VISION_ONLY = "vision_only"
    VIDEO = "video"

    @property
    def skip_structural_events(self) -> bool:
        """True when the recorder should NOT build steps from EventBus events."""
        return self in _MODES_SKIP_STRUCTURAL

    @property
    def needs_visual_polling(self) -> bool:
        """True when PerceptionSession must poll independently of events."""
        return self in _MODES_VISUAL_POLLING

    @property
    def inject_visual_steps(self) -> bool:
        """True when visual actions should be injected as trajectory steps."""
        return self in _MODES_INJECT_VISUAL

    @property
    def uses_video(self) -> bool:
        """True when continuous video recording is the primary visual signal."""
        return self in _MODES_VIDEO

    @classmethod
    def from_str(cls, value: str) -> "RecordingMode":
        """Parse from env/config string, falling back to DEFAULT."""
        value = value.strip().lower()
        for member in cls:
            if member.value == value:
                return member
        return cls.DEFAULT


_MODES_SKIP_STRUCTURAL = frozenset({RecordingMode.VISION_ONLY})
_MODES_VISUAL_POLLING = frozenset({RecordingMode.VISION_ONLY})
_MODES_INJECT_VISUAL = frozenset({RecordingMode.VISION_ONLY})
_MODES_VIDEO = frozenset({RecordingMode.VIDEO})


class RecordingState(Enum):
    """State machine for demonstration recording."""

    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"


class SnapshotLevel(Enum):
    """Snapshot fidelity level — selected based on event importance."""

    FULL = "full"        # Complete AX tree + clipboard + visual frame (app switch, critical UI)
    LIGHT = "light"      # Focused element + AX digest (normal UI actions)
    MINIMAL = "minimal"  # Only file path and action (file/clipboard events)


class ActionType(Enum):
    """Canonical action types observed from user behavior."""

    FILE_CREATE = "file.create"
    FILE_MODIFY = "file.modify"
    FILE_DELETE = "file.delete"
    FILE_RENAME = "file.rename"
    CLIPBOARD_COPY = "clipboard.copy"
    APP_SWITCH = "app.switch"
    UI_CLICK = "ui.click"
    UI_TYPE = "ui.type"
    UI_SHORTCUT = "ui.shortcut"
    UI_SCROLL = "ui.scroll"
    UI_MOVE = "ui.move"
    UI_RESIZE = "ui.resize"
    UI_DRAG = "ui.drag"
    CHAT_USER_MESSAGE = "chat.user_message"
    CHAT_TOOL_CALL = "chat.tool_call"
    CHAT_TOOL_RESULT = "chat.tool_result"
    CHAT_ASSISTANT_RESPONSE = "chat.assistant_response"
    UNKNOWN = "unknown"


# ── Mapping from normalized event types to ActionType ──

_EVENT_TO_ACTION: Dict[Tuple[str, str], ActionType] = {
    ("fs.change", "created"): ActionType.FILE_CREATE,
    ("fs.change", "modified"): ActionType.FILE_MODIFY,
    ("fs.change", "deleted"): ActionType.FILE_DELETE,
    ("fs.change", "renamed"): ActionType.FILE_RENAME,
    ("clipboard.change", ""): ActionType.CLIPBOARD_COPY,
    ("app.focus_change", ""): ActionType.APP_SWITCH,
    ("ui.action", "click"): ActionType.UI_CLICK,
    ("ui.action", "type"): ActionType.UI_TYPE,
    ("ui.action", "shortcut"): ActionType.UI_SHORTCUT,
    ("ui.action", "scroll"): ActionType.UI_SCROLL,
    ("ui.action", "move"): ActionType.UI_MOVE,
    ("ui.action", "resize"): ActionType.UI_RESIZE,
    ("ui.action", "drag"): ActionType.UI_DRAG,
    ("chat.interaction", "user_message"): ActionType.CHAT_USER_MESSAGE,
    ("chat.interaction", "tool_call"): ActionType.CHAT_TOOL_CALL,
    ("chat.interaction", "tool_result"): ActionType.CHAT_TOOL_RESULT,
    ("chat.interaction", "response"): ActionType.CHAT_ASSISTANT_RESPONSE,
}


def action_type_from_event(event_type: str, sub_action: str = "") -> ActionType:
    """Resolve a SystemEvent type + sub-action to a canonical ActionType."""
    return _EVENT_TO_ACTION.get((event_type, sub_action), ActionType.UNKNOWN)


# ── Core data types ──


@dataclass(frozen=True)
class NoiseSignal:
    """Lightweight noise annotation attached to TrajectoryStep.action.params['_noise']."""
    signal_type: str   # "undo", "redo", "rapid_switch", "repeated", "idle_scroll"
    confidence: float  # 0.0-1.0 annotation confidence
    related_step: int = -1  # Related step index (e.g., the original operation for undo)


@dataclass(frozen=True)
class RawAction:
    """A single observed user action."""

    timestamp: float
    action_type: ActionType
    target: str = ""
    target_label: str = ""
    target_role: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    app_bundle_id: str = ""
    app_name: str = ""


@dataclass(frozen=True)
class StateSnapshot:
    """System state at a point in time."""

    timestamp: float
    focused_app: str = ""
    ax_tree_digest: str = ""
    ax_focused_element: Optional[Dict[str, Any]] = None
    clipboard_text: Optional[str] = None
    visual_frame_ref: Optional[str] = None
    ax_tree_snapshot: Optional[Dict[str, Any]] = None  # Full AX tree JSON (FULL level)
    snapshot_level: str = "light"  # SnapshotLevel value


@dataclass(frozen=True)
class TrajectoryStep:
    """One step in a trajectory: pre-state + action + optional post-state."""

    state: StateSnapshot
    action: RawAction
    post_state: Optional[StateSnapshot] = None


@dataclass
class Trajectory:
    """A complete recorded human demonstration."""

    trajectory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str = "default"
    start_time: float = 0.0
    end_time: float = 0.0
    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0

    @property
    def app_sequence(self) -> List[str]:
        """Ordered list of unique apps visited."""
        seen: set[str] = set()
        result: List[str] = []
        for step in self.steps:
            bid = step.action.app_bundle_id
            if bid and bid not in seen:
                seen.add(bid)
                result.append(bid)
        return result


@dataclass(frozen=True)
class SemanticAction:
    """A high-level action abstracted from one or more RawActions."""

    action_name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_action_range: Tuple[int, int] = (0, 0)
    confidence: float = 1.0


@dataclass
class Episode:
    """A semantically coherent segment of a trajectory."""

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trajectory_id: str = ""
    start_idx: int = 0
    end_idx: int = 0
    inferred_goal: str = ""
    app_sequence: List[str] = field(default_factory=list)
    semantic_actions: List[SemanticAction] = field(default_factory=list)
    confidence: float = 0.0
    procedure_graph: str = ""

    @property
    def action_count(self) -> int:
        return self.end_idx - self.start_idx

    def steps_from(self, trajectory: Trajectory) -> List[TrajectoryStep]:
        """Extract the steps belonging to this episode from its parent trajectory."""
        return trajectory.steps[self.start_idx : self.end_idx]

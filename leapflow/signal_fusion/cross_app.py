"""Cross-app context tracking for workflow hypothesis generation.

Maintains a stateful model of app transitions, clipboard carry payloads,
and workflow pattern hypotheses during a recording session.

SRP: tracks context and generates hypotheses — does not segment or fuse.
ISP: exposes separate read interfaces for recording vs analysis consumers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from leapflow.signal_fusion.types import (
    AppTransitionEvent,
    CarryPayload,
    CarryType,
    WorkflowType,
)


@dataclass
class WorkflowHypothesis:
    """A hypothesized workflow pattern based on transition history."""

    workflow_type: WorkflowType
    confidence: float
    app_sequence: List[str] = field(default_factory=list)
    expected_return_app: str = ""


@dataclass
class CrossAppState:
    """Snapshot of cross-app context at a point in time."""

    current_app: str = ""
    previous_app: str = ""
    active_carry: Optional[CarryPayload] = None
    hypothesis: Optional[WorkflowHypothesis] = None
    transition_count: int = 0
    unique_apps_visited: int = 0


class CrossAppContextTracker:
    """Stateful cross-app workflow tracking.

    Processes AppTransitionEvents to maintain a running context model
    and generate lightweight workflow hypotheses (no LLM — recording
    phase zero-inference principle).
    """

    def __init__(self, *, history_limit: int = 50) -> None:
        self._transitions: List[AppTransitionEvent] = []
        self._app_history: Deque[str] = deque(maxlen=history_limit)
        self._active_carry: Optional[CarryPayload] = None
        self._hypothesis: Optional[WorkflowHypothesis] = None
        self._app_visit_counts: Dict[str, int] = {}

    @property
    def transitions(self) -> List[AppTransitionEvent]:
        return list(self._transitions)

    @property
    def current_hypothesis(self) -> Optional[WorkflowHypothesis]:
        return self._hypothesis

    def on_transition(self, event: AppTransitionEvent) -> CrossAppState:
        """Process an app transition and update context."""
        self._transitions.append(event)
        self._app_history.append(event.from_bundle)

        self._app_visit_counts[event.to_bundle] = (
            self._app_visit_counts.get(event.to_bundle, 0) + 1
        )

        if event.carry_clipboard:
            self._active_carry = CarryPayload.from_clipboard(
                event.carry_clipboard, origin=event.from_bundle
            )
        else:
            self._active_carry = None

        self._update_hypothesis(event)
        return self._current_state(event.to_bundle)

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._transitions.clear()
        self._app_history.clear()
        self._active_carry = None
        self._hypothesis = None
        self._app_visit_counts.clear()

    def _current_state(self, current_app: str) -> CrossAppState:
        return CrossAppState(
            current_app=current_app,
            previous_app=self._app_history[-1] if self._app_history else "",
            active_carry=self._active_carry,
            hypothesis=self._hypothesis,
            transition_count=len(self._transitions),
            unique_apps_visited=len(self._app_visit_counts),
        )

    def _update_hypothesis(self, event: AppTransitionEvent) -> None:
        """Infer workflow type from transition history (pure heuristics)."""
        history = list(self._app_history) + [event.to_bundle]
        n = len(history)
        if n < 2:
            self._hypothesis = None
            return

        carry_bonus = 0.1 if self._active_carry and self._active_carry.carry_type != CarryType.EMPTY else 0.0

        if n >= 5 and self._is_iterative(history):
            self._hypothesis = WorkflowHypothesis(
                workflow_type=WorkflowType.ITERATIVE_REFINEMENT,
                confidence=min(1.0, 0.75 + carry_bonus),
                app_sequence=history[-5:],
            )
            return

        if n >= 3 and history[-1] == history[-3]:
            self._hypothesis = WorkflowHypothesis(
                workflow_type=WorkflowType.ROUND_TRIP,
                confidence=min(1.0, 0.7 + carry_bonus),
                app_sequence=history[-3:],
                expected_return_app=history[-3],
            )
            return

        if n >= 2 and self._is_parallel(history):
            self._hypothesis = WorkflowHypothesis(
                workflow_type=WorkflowType.PARALLEL_REFERENCE,
                confidence=0.6,
                app_sequence=list(dict.fromkeys(history[-4:])),
            )
            return

        unique_recent = list(dict.fromkeys(history[-6:]))
        if len(unique_recent) >= 3:
            self._hypothesis = WorkflowHypothesis(
                workflow_type=WorkflowType.MULTI_HUB,
                confidence=min(1.0, 0.5 + 0.1 * len(unique_recent) + carry_bonus),
                app_sequence=unique_recent,
            )
            return

        if n >= 2:
            self._hypothesis = WorkflowHypothesis(
                workflow_type=WorkflowType.LINEAR_TRANSFER,
                confidence=min(1.0, 0.6 + carry_bonus),
                app_sequence=history[-2:],
            )

    @staticmethod
    def _is_iterative(history: List[str]) -> bool:
        """Detect A ↔ B ↔ A ↔ B pattern."""
        if len(history) < 5:
            return False
        tail = history[-5:]
        return tail[0] == tail[2] == tail[4] and tail[1] == tail[3] and tail[0] != tail[1]

    @staticmethod
    def _is_parallel(history: List[str]) -> bool:
        """Detect rapid A ↔ B switching (parallel reference)."""
        if len(history) < 4:
            return False
        tail = history[-4:]
        unique = set(tail)
        return len(unique) == 2 and tail[0] == tail[2] and tail[1] == tail[3]

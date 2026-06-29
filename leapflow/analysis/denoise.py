"""Noise-robust preprocessing for demonstration trajectories.

Implements DenoisePass as a composable AbstractionPass that runs before
GroupingPass and PatternPass.  Three strategies applied in fixed order:

    1. UndoCollapseStrategy  — fold action→undo→retry sequences
    2. IdempotentMergeStrategy — merge consecutive redundant operations
    3. DistractionFilterStrategy — remove off-task app flash-switches
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from leapflow.analysis.abstractor import AbstractionPass
from leapflow.domain.trajectory import SemanticAction

if TYPE_CHECKING:
    from leapflow.domain.trajectory import TrajectoryStep

logger = logging.getLogger(__name__)


class DenoiseStrategy(ABC):
    @abstractmethod
    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]: ...


class UndoCollapseStrategy(DenoiseStrategy):
    """Fold action→undo→retry sequences into the final result.

    Uses a stack: encountering undo pops the top (cancels the last action),
    redo is treated as a no-op (the real retry follows). The remaining stack
    represents the user's actual intent.
    """

    _UNDO_KEY_CODE = 6  # macOS 'Z' keycode

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) <= 1:
            return actions

        stack: List[SemanticAction] = []
        for action in actions:
            if self._is_undo(action):
                if stack:
                    stack.pop()
            elif self._is_redo(action):
                continue
            else:
                stack.append(action)
        return stack

    def _is_undo(self, action: SemanticAction) -> bool:
        if action.action_name != "ui.shortcut":
            return False
        params = action.parameters
        key_code = params.get("key_code", -1)
        modifiers = params.get("modifiers", [])
        return (
            key_code == self._UNDO_KEY_CODE
            and "command" in modifiers
            and "shift" not in modifiers
        )

    def _is_redo(self, action: SemanticAction) -> bool:
        if action.action_name != "ui.shortcut":
            return False
        params = action.parameters
        key_code = params.get("key_code", -1)
        modifiers = params.get("modifiers", [])
        return (
            key_code == self._UNDO_KEY_CODE
            and "command" in modifiers
            and "shift" in modifiers
        )


class IdempotentMergeStrategy(DenoiseStrategy):
    """Merge consecutive idempotent operations within a time window.

    Consecutive save commands, repeated scrolls, or multiple modifications
    to the same file are collapsed into a single operation.
    """

    IDEMPOTENT_TYPES = frozenset({"ui.scroll", "ui.shortcut", "file.modify"})
    MAX_GAP_SECONDS = 2.0

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) <= 1:
            return actions

        result: List[SemanticAction] = []
        i = 0
        while i < len(actions):
            run_end = self._find_run_end(actions, i)
            if run_end > i + 1:
                result.append(self._merge_run(actions[i:run_end]))
                i = run_end
            else:
                result.append(actions[i])
                i += 1
        return result

    def _find_run_end(self, actions: List[SemanticAction], start: int) -> int:
        base = actions[start]
        if base.action_name not in self.IDEMPOTENT_TYPES:
            return start + 1

        end = start + 1
        while end < len(actions):
            curr = actions[end]
            if curr.action_name != base.action_name:
                break
            if not self._same_target(base, curr):
                break
            gap = self._time_gap(actions[end - 1], curr)
            if gap > self.MAX_GAP_SECONDS:
                break
            end += 1
        return end

    @staticmethod
    def _same_target(a: SemanticAction, b: SemanticAction) -> bool:
        t_a = a.parameters.get("target", "")
        t_b = b.parameters.get("target", "")
        if t_a and t_b:
            return t_a == t_b
        return True

    @staticmethod
    def _time_gap(a: SemanticAction, b: SemanticAction) -> float:
        ts_a = a.raw_action_range[1] if a.raw_action_range else 0
        ts_b = b.raw_action_range[0] if b.raw_action_range else 0
        return abs(ts_b - ts_a)

    @staticmethod
    def _merge_run(run: List[SemanticAction]) -> SemanticAction:
        last = run[-1]
        merged_params = dict(last.parameters)
        merged_params["_merged_count"] = len(run)
        return SemanticAction(
            action_name=last.action_name,
            description=f"{last.action_name} x{len(run)} (merged)",
            parameters=merged_params,
            raw_action_range=(
                run[0].raw_action_range[0],
                last.raw_action_range[1],
            ),
            confidence=last.confidence,
        )


class DistractionFilterStrategy(DenoiseStrategy):
    """Remove off-task app flash-switches.

    Detects pattern: app_switch → brief stay (< threshold) → switch back,
    with no substantive operations in between. The entire span is removed.
    """

    MIN_DWELL_SECONDS = 5.0

    _SUBSTANTIVE_ACTIONS = frozenset({
        "file.create", "file.modify", "file.delete", "file.rename",
        "file.move", "batch_move",
        "clipboard.copy", "ui.type",
    })

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 3:
            return actions

        skip_indices: set[int] = set()
        i = 0
        while i < len(actions):
            span = self._detect_distraction(actions, i)
            if span is not None:
                for j in range(i, span):
                    skip_indices.add(j)
                i = span
            else:
                i += 1

        return [a for idx, a in enumerate(actions) if idx not in skip_indices]

    def _detect_distraction(
        self, actions: List[SemanticAction], idx: int
    ) -> int | None:
        """Returns the end index of the distraction span, or None."""
        if actions[idx].action_name != "app.switch":
            return None

        original_app = actions[idx].parameters.get("_prev_app", "")
        if not original_app:
            actions[idx].parameters.get("target", "")
            if idx > 0:
                original_app = actions[idx - 1].parameters.get("target", "")
                if not original_app:
                    original_app = actions[idx - 1].parameters.get(
                        "app_bundle_id", ""
                    )

        for j in range(idx + 1, min(idx + 8, len(actions))):
            if actions[j].action_name == "app.switch":
                switch_target = actions[j].parameters.get("target", "")
                if original_app and switch_target == original_app:
                    middle = actions[idx + 1 : j]
                    has_substance = any(
                        a.action_name in self._SUBSTANTIVE_ACTIONS
                        for a in middle
                    )
                    if not has_substance:
                        return j + 1
        return None


class DenoisePass(AbstractionPass):
    """Composable noise reduction: undo fold → idempotent merge → distraction filter.

    Order is fixed because each strategy depends on the cleanup of the previous:
    - UndoCollapse first: removes the most structurally obvious noise
    - IdempotentMerge second: operates on a cleaner sequence
    - DistractionFilter last: benefits from a simplified action stream
    """

    def __init__(self) -> None:
        self._strategies: List[DenoiseStrategy] = [
            UndoCollapseStrategy(),
            IdempotentMergeStrategy(),
            DistractionFilterStrategy(),
        ]

    def apply(
        self,
        actions: List[SemanticAction],
        steps: "List[TrajectoryStep] | None" = None,
    ) -> List[SemanticAction]:
        for strategy in self._strategies:
            before = len(actions)
            actions = strategy.apply(actions)
            if len(actions) < before:
                logger.debug(
                    "denoise.%s: %d → %d actions",
                    type(strategy).__name__, before, len(actions),
                )
        return actions

"""Causal chain extraction from semantic action sequences.

Identifies only the actions that contribute to the final observable state,
filtering out detours, distractions, and non-contributing operations.
Used by the distillation layer to produce minimal, goal-directed skill steps.
"""

from __future__ import annotations

import logging
from typing import FrozenSet, List, Set

from leapflow.domain.trajectory import SemanticAction

logger = logging.getLogger(__name__)

_EFFECT_ACTIONS: FrozenSet[str] = frozenset({
    "file.create", "file.modify", "file.delete", "file.rename", "file.move",
    "clipboard.copy", "batch_modify", "batch_rename", "batch_delete",
    "batch_move", "batch_move_to_folder", "move_to_new_folder",
})

_SUPPORT_ACTIONS: FrozenSet[str] = frozenset({
    "app.switch", "open_file_dialog", "transfer_data",
    "create_and_edit", "download_organize",
})


class CausalChainAnalyzer:
    """Extract causally-necessary steps from a semantic action sequence.

    Algorithm:
        1. Forward scan: identify effect actions (file ops, clipboard writes)
        2. For each effect action, back-trace to the nearest un-claimed
           support action (app.switch, open_file_dialog, etc.)
        3. Return only the causal subset, preserving original order.
    """

    def __init__(
        self,
        *,
        effect_actions: FrozenSet[str] = _EFFECT_ACTIONS,
        support_actions: FrozenSet[str] = _SUPPORT_ACTIONS,
    ) -> None:
        self._effects = effect_actions
        self._supports = support_actions

    def extract_causal_chain(
        self, actions: List[SemanticAction]
    ) -> List[SemanticAction]:
        if len(actions) <= 2:
            return list(actions)

        causal_indices = self._find_causal_indices(actions)
        if not causal_indices:
            return list(actions)

        result = [a for i, a in enumerate(actions) if i in causal_indices]
        if len(result) < len(actions):
            logger.debug(
                "causal_chain: %d → %d actions (removed %d non-causal)",
                len(actions), len(result), len(actions) - len(result),
            )
        return result

    def _find_causal_indices(self, actions: List[SemanticAction]) -> Set[int]:
        causal: List[int] = []
        claimed: Set[int] = set()

        for i, action in enumerate(actions):
            if action.action_name in self._effects:
                causal.append(i)
                claimed.add(i)
                self._backtrace_support(actions, i, causal, claimed)

        return set(causal)

    def _backtrace_support(
        self,
        actions: List[SemanticAction],
        effect_idx: int,
        causal: List[int],
        claimed: Set[int],
    ) -> None:
        for j in range(effect_idx - 1, -1, -1):
            if j in claimed:
                break
            if actions[j].action_name in self._supports:
                causal.append(j)
                claimed.add(j)
                break

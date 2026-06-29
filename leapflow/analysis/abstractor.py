"""Multi-level action abstraction for trajectory analysis.

Abstraction levels:
    L0 (Raw)     → individual events from EventBus
    L1 (Grouped) → consecutive same-type actions merged (e.g. keystrokes → type_text)
    L2 (Pattern) → recognized action patterns (e.g. copy + switch + paste → transfer_data)
    L3 (Intent)  → LLM-inferred user intent (optional, requires LLMProvider)

Each level is a composable AbstractionPass (Pipeline pattern).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from leapflow.analysis.patterns import PatternLibrary
from leapflow.domain.trajectory import (
    ActionType,
    RawAction,
    SemanticAction,
    TrajectoryStep,
)

logger = logging.getLogger(__name__)


# ── Abstraction pass protocol ──


class AbstractionPass(ABC):
    """A single transformation in the abstraction pipeline."""

    @abstractmethod
    def apply(
        self,
        actions: List[SemanticAction],
        steps: Optional[List[TrajectoryStep]] = None,
    ) -> List[SemanticAction]:
        """Transform a list of semantic actions into a (usually shorter) list.

        Args:
            actions: Current semantic actions to transform.
            steps: Optional raw trajectory steps for passes that need
                   access to state context (e.g., visual frames).
        """


# ── L0→L1: Rule-based grouping ──


class GroupingPass(AbstractionPass):
    """Merge consecutive low-level actions of the same type.

    - Sequential type events → single "type_text" with merged text_content
    - Sequential file modifications on the same target → single "batch_modify"
    - Sequential clipboard changes → keep only the last one
    """

    def apply(
        self,
        actions: List[SemanticAction],
        steps: Optional[List[TrajectoryStep]] = None,
    ) -> List[SemanticAction]:
        if len(actions) <= 1:
            return actions

        result: List[SemanticAction] = []
        i = 0
        while i < len(actions):
            # Special case: merge consecutive type events with char data
            if actions[i].action_name == "ui.type":
                type_end = self._find_type_group_end(actions, i)
                if type_end > i + 1:
                    result.append(self._merge_type_group(actions[i:type_end]))
                    i = type_end
                    continue

            group_end = self._find_group_end(actions, i)
            if group_end > i + 1:
                result.append(self._merge_group(actions[i:group_end]))
                i = group_end
            else:
                result.append(actions[i])
                i += 1
        return result

    def _find_type_group_end(self, actions: List[SemanticAction], start: int) -> int:
        """Find end of a consecutive ui.type run within the same app."""
        app = actions[start].parameters.get("app_bundle_id", "")
        end = start + 1
        while end < len(actions) and actions[end].action_name == "ui.type":
            if actions[end].parameters.get("app_bundle_id", "") != app:
                break
            end += 1
        return end

    @staticmethod
    def _merge_type_group(group: List[SemanticAction]) -> SemanticAction:
        """Merge consecutive type actions, concatenating char fields into text_content."""
        first, last = group[0], group[-1]
        chars = [a.parameters.get("char", "") for a in group]
        text_content = "".join(chars)
        params: Dict[str, Any] = {
            "app_bundle_id": first.parameters.get("app_bundle_id", ""),
            "keystroke_count": len(group),
        }
        if text_content:
            params["text_content"] = text_content
        return SemanticAction(
            action_name="type_text",
            description=f"Type '{text_content[:50]}'" if text_content else f"Type {len(group)} keys",
            parameters=params,
            raw_action_range=(first.raw_action_range[0], last.raw_action_range[1]),
            confidence=1.0,
        )

    def _find_group_end(self, actions: List[SemanticAction], start: int) -> int:
        """Find the end index of a mergeable group starting at `start`."""
        base = actions[start]
        end = start + 1
        while end < len(actions):
            curr = actions[end]
            if not self._can_merge(base, curr):
                break
            end += 1
        return end

    @staticmethod
    def _can_merge(a: SemanticAction, b: SemanticAction) -> bool:
        if a.action_name != b.action_name:
            return False
        mergeable = {"file.modify", "file.create", "file.rename", "file.move", "clipboard.copy"}
        return a.action_name in mergeable

    @staticmethod
    def _merge_group(group: List[SemanticAction]) -> SemanticAction:
        first, last = group[0], group[-1]
        count = len(group)
        merged_params = dict(first.parameters)
        merged_params["count"] = count
        return SemanticAction(
            action_name=f"batch_{first.action_name.split('.')[-1]}",
            description=f"{first.action_name} x{count}",
            parameters=merged_params,
            raw_action_range=(first.raw_action_range[0], last.raw_action_range[1]),
            confidence=min(a.confidence for a in group),
        )


# ── L1→L2: Pattern matching ──


class PatternPass(AbstractionPass):
    """Pattern-based action abstraction using an extensible PatternLibrary.

    Replaces hardcoded patterns with a YAML-driven, wildcard-capable
    pattern matching engine that supports runtime addition of new patterns.
    """

    def __init__(self, library: Optional[PatternLibrary] = None) -> None:
        self._library = library or PatternLibrary.default()

    def apply(
        self,
        actions: List[SemanticAction],
        steps: Optional[List[TrajectoryStep]] = None,
    ) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        matches = self._library.match(actions)
        if not matches:
            return list(actions)

        # Build result: replace matched spans with semantic actions, keep rest
        result: List[SemanticAction] = []
        consumed = 0

        for m in matches:
            # Append unmatched actions before this match
            result.extend(actions[consumed : m.start_idx])
            # Create the abstracted semantic action
            first = actions[m.start_idx]
            last = actions[m.end_idx - 1]
            params: Dict[str, Any] = {}
            # Merge original params from matched window
            for a in actions[m.start_idx : m.end_idx]:
                params.update(a.parameters)
            # Overlay extracted params from pattern rules
            params.update(m.extracted_params)

            result.append(
                SemanticAction(
                    action_name=m.pattern.semantic_name,
                    description=m.pattern.description or m.pattern.semantic_name,
                    parameters=params,
                    raw_action_range=(
                        first.raw_action_range[0],
                        last.raw_action_range[1],
                    ),
                    confidence=m.match_confidence,
                )
            )
            consumed = m.end_idx

        # Append remaining unmatched actions
        result.extend(actions[consumed:])
        return result


# ── Orchestrator ──


class ActionAbstractor:
    """Multi-level action abstraction pipeline.

    By default runs L0→L1→L2 (rule-based only, no LLM cost).
    """

    def __init__(
        self,
        passes: Sequence[AbstractionPass] | None = None,
        *,
        platform_hint: str = "darwin",
    ) -> None:
        if passes is not None:
            self._passes = list(passes)
        else:
            from leapflow.analysis.denoise import DenoisePass
            from leapflow.analysis.fs_pattern_pass import FileSystemPatternPass
            from leapflow.analysis.synthesis import PlatformSynthesisPass
            self._passes = [
                DenoisePass(),
                PlatformSynthesisPass.for_platform(platform_hint),
                GroupingPass(),
                FileSystemPatternPass(),  # FS pattern recognition (mkdir+move, batch_delete, rename, create_doc)
                PatternPass(),
            ]

    def abstract(self, steps: List[TrajectoryStep]) -> List[SemanticAction]:
        """Run the abstraction pipeline on raw trajectory steps."""
        actions = [self._step_to_semantic(i, step) for i, step in enumerate(steps)]
        for p in self._passes:
            actions = p.apply(actions, steps=steps)
        return actions

    @staticmethod
    def _step_to_semantic(idx: int, step: TrajectoryStep) -> SemanticAction:
        """Convert a single TrajectoryStep to an initial SemanticAction (L0)."""
        a = step.action
        params: Dict[str, Any] = {}

        if a.target:
            params["target"] = a.target
        if a.target_label:
            params["target_label"] = a.target_label
        if a.app_bundle_id:
            params["app_bundle_id"] = a.app_bundle_id
        if a.action_type in (ActionType.FILE_CREATE, ActionType.FILE_MODIFY,
                             ActionType.FILE_DELETE, ActionType.FILE_RENAME):
            params["path"] = a.target
        if a.action_type == ActionType.CLIPBOARD_COPY:
            params["text"] = a.params.get("text", "")
        if a.action_type == ActionType.UI_TYPE:
            char = a.params.get("char")
            if char:
                params["char"] = char
        if a.action_type == ActionType.UI_DRAG:
            for k in ("start_x", "start_y", "end_x", "end_y",
                      "end_role", "end_label", "cross_app", "start_app", "end_app"):
                if k in a.params:
                    params[k] = a.params[k]

        return SemanticAction(
            action_name=a.action_type.value,
            description=_describe_action(a),
            parameters=params,
            raw_action_range=(idx, idx + 1),
            confidence=1.0,
        )


def _describe_action(a: RawAction) -> str:
    """Generate a human-readable description of a raw action."""
    at = a.action_type
    if at == ActionType.APP_SWITCH:
        return f"Switch to {a.app_name or a.app_bundle_id}"
    if at in (ActionType.FILE_CREATE, ActionType.FILE_MODIFY,
              ActionType.FILE_DELETE, ActionType.FILE_RENAME):
        verb = at.value.split(".")[-1].capitalize()
        return f"{verb} {a.target}"
    if at == ActionType.CLIPBOARD_COPY:
        length = a.params.get("text_length", 0)
        return f"Copy to clipboard ({length} chars)"
    if at == ActionType.UI_CLICK:
        return f"Click {a.target_label or a.target or 'element'}"
    if at == ActionType.UI_TYPE:
        char = a.params.get("char", "")
        return f"Type '{char}'" if char else "Type key"
    if at == ActionType.UI_DRAG:
        label = a.target_label or "element"
        end_label = a.params.get("end_label", "") or "target"
        return f"Drag {label} to {end_label}"
    return f"{at.value} {a.target or ''}"

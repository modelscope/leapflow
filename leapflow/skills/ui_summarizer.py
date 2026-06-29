"""UI tree summarizer — convert raw UINode trees into LLM-friendly element lists.

Analogous to how EventNormalizer simplifies raw OS events for the Recording
pipeline, this module simplifies the raw AX tree for the Execution pipeline.

Responsibilities:
    1. Depth-limited traversal (avoid overwhelming LLM context)
    2. Role-based filtering (skip pure layout containers)
    3. Interactive element prioritization
    4. Stable selector generation per element
    5. Output capping (max N elements)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from leapflow.domain.events import UINode
from leapflow.domain.ui_vocabulary import INTERACTIVE_ROLES, LAYOUT_ROLES, STRUCTURAL_ROLES
from leapflow.skills.ui_selector import format_selector


@dataclass(frozen=True)
class SummaryConfig:
    """Configuration for tree summarization."""

    max_depth: int = 5
    max_elements: int = 60
    include_value_max_len: int = 80

    interactive_roles: FrozenSet[str] = INTERACTIVE_ROLES
    layout_roles: FrozenSet[str] = LAYOUT_ROLES
    always_include_roles: FrozenSet[str] = STRUCTURAL_ROLES


DEFAULT_CONFIG = SummaryConfig()


@dataclass
class UIElement:
    """A summarized UI element suitable for LLM consumption."""

    selector: str
    role: str
    label: str
    value: str = ""
    actions: List[str] = field(default_factory=list)
    path: str = ""
    node_id: str = ""


class UITreeSummarizer:
    """Converts UINode tree into a flat list of actionable elements."""

    def __init__(self, config: SummaryConfig = DEFAULT_CONFIG) -> None:
        self._config = config

    def summarize(
        self,
        root: UINode,
        *,
        focus_area: str = "",
    ) -> List[UIElement]:
        """Produce a flat, prioritized list of interactive elements.

        Args:
            root: The UINode tree root.
            focus_area: Optional hint to prioritize elements in a subtree
                        whose label or role contains this string.
        """
        elements: List[UIElement] = []
        self._walk(root, elements, depth=0, path_parts=[], my_index=0)

        if focus_area:
            elements = self._prioritize_focus(elements, focus_area)

        return elements[: self._config.max_elements]

    def _walk(
        self,
        node: UINode,
        out: List[UIElement],
        depth: int,
        path_parts: List[str],
        my_index: int,
    ) -> None:
        if depth > self._config.max_depth:
            return

        cfg = self._config
        is_interactive = node.role in cfg.interactive_roles
        is_structural = node.role in cfg.always_include_roles
        is_layout = node.role in cfg.layout_roles
        has_label = bool(node.label)

        should_emit = is_interactive or (is_structural and has_label)
        if is_layout and not has_label:
            should_emit = False

        if should_emit and node.role:
            selector = format_selector(node.role, node.label, my_index if not node.label else -1)
            value = node.value[:cfg.include_value_max_len] if node.value else ""
            path_str = " > ".join(path_parts) if path_parts else ""

            out.append(UIElement(
                selector=selector,
                role=node.role,
                label=node.label,
                value=value,
                actions=list(node.actions) if node.actions else [],
                path=path_str,
                node_id=node.node_id,
            ))

        child_path = path_parts + [node.role] if node.role else path_parts
        role_counts: Dict[str, int] = {}
        for child in node.children:
            idx = role_counts.get(child.role, 0)
            role_counts[child.role] = idx + 1
            self._walk(child, out, depth + 1, child_path, idx)

    def _prioritize_focus(
        self, elements: List[UIElement], focus_area: str
    ) -> List[UIElement]:
        """Re-order elements so those matching focus_area come first."""
        focus_lower = focus_area.lower()
        focused: List[UIElement] = []
        rest: List[UIElement] = []
        for el in elements:
            if (focus_lower in el.path.lower()
                    or focus_lower in el.label.lower()
                    or focus_lower in el.role.lower()):
                focused.append(el)
            else:
                rest.append(el)
        return focused + rest


def summarize_tree(
    root: UINode,
    *,
    focus_area: str = "",
    config: Optional[SummaryConfig] = None,
) -> List[Dict[str, Any]]:
    """Convenience function: summarize tree and return dicts for JSON serialization."""
    summarizer = UITreeSummarizer(config or DEFAULT_CONFIG)
    elements = summarizer.summarize(root, focus_area=focus_area)
    return [
        {
            "selector": el.selector,
            "role": el.role,
            "label": el.label,
            **({"value": el.value} if el.value else {}),
            **({"actions": el.actions} if el.actions else {}),
            **({"path": el.path} if el.path else {}),
        }
        for el in elements
    ]

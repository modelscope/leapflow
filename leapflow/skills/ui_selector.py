"""UI element selector — parse, match, and resolve accessibility elements.

Provides a human-readable addressing scheme for UI elements that replaces
unstable internal node_id values. Selectors use role + label + index,
mirroring the anchor format from the Recording side.

Selector syntax:
    "AXButton[label=Send]"              — exact match on role + label
    "AXTextField[label~=search]"        — contains match (case-insensitive)
    "AXButton#2"                        — 2nd AXButton sibling (0-based)
    "AXToolbar > AXButton[label=New]"   — path-based (ancestor > target)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from leapflow.domain.events import UINode
from leapflow.domain.skill_types import AnchorCandidate

_SELECTOR_RE = re.compile(
    r"^(?P<path>(?:[\w]+\s*>\s*)*)?"
    r"(?P<role>\w+)"
    r"(?:\[label(?P<op>[~]?)=(?P<label>[^\]]*)\])?"
    r"(?:#(?P<index>\d+))?$"
)


@dataclass(frozen=True)
class UISelector:
    """Parsed UI element selector."""

    role: str
    label: str = ""
    label_contains: bool = False
    index: int = -1
    ancestors: Tuple[str, ...] = ()

    def matches(self, node: UINode, *, sibling_index: int = -1) -> bool:
        if node.role != self.role:
            return False
        if self.label:
            if self.label_contains:
                if self.label.lower() not in node.label.lower():
                    return False
            else:
                if node.label != self.label:
                    return False
        if self.index >= 0 and sibling_index >= 0:
            if sibling_index != self.index:
                return False
        return True


def parse_selector(text: str) -> Optional[UISelector]:
    """Parse a selector string into a UISelector, or None if invalid."""
    text = text.strip()
    if not text:
        return None
    m = _SELECTOR_RE.match(text)
    if not m:
        return None
    path_raw = m.group("path") or ""
    ancestors = tuple(
        seg.strip() for seg in path_raw.split(">") if seg.strip()
    )
    role = m.group("role")
    label = m.group("label") or ""
    op = m.group("op") or ""
    index_str = m.group("index")
    index = int(index_str) if index_str else -1
    return UISelector(
        role=role,
        label=label,
        label_contains=(op == "~"),
        index=index,
        ancestors=ancestors,
    )


def format_selector(role: str, label: str = "", index: int = -1) -> str:
    """Generate a canonical selector string from components."""
    s = role
    if label:
        s += f"[label={label}]"
    if index >= 0:
        s += f"#{index}"
    return s


@dataclass
class MatchResult:
    """Result of matching a selector against a tree."""

    node_id: str
    role: str
    label: str
    selector: str
    path: List[str] = field(default_factory=list)


def find_in_tree(tree: UINode, selector: UISelector) -> List[MatchResult]:
    """Find all nodes matching a selector in the tree. Returns matches."""
    results: List[MatchResult] = []
    _walk(tree, selector, [], results)
    return results


def find_first(tree: UINode, selector: UISelector) -> Optional[MatchResult]:
    """Find first match. Returns None if not found."""
    results = find_in_tree(tree, selector)
    if selector.index >= 0 and results:
        return results[0] if results else None
    return results[0] if results else None


def resolve_selector_string(tree: UINode, selector_str: str) -> Optional[str]:
    """Convenience: parse + find first match → return node_id or None."""
    sel = parse_selector(selector_str)
    if sel is None:
        return None
    match = find_first(tree, sel)
    return match.node_id if match else None


def _walk(
    node: UINode,
    selector: UISelector,
    path: List[str],
    results: List[MatchResult],
) -> None:
    """Recursive tree walk with ancestor tracking and sibling index counting."""
    if selector.ancestors:
        current_path = path + [node.role] if node.role else path
        if not _ancestors_match(current_path, selector.ancestors):
            _walk_children(node, selector, path, results)
            return

    role_counter: Dict[str, int] = {}
    for child in node.children:
        idx = role_counter.get(child.role, 0)
        role_counter[child.role] = idx + 1

        if selector.matches(child, sibling_index=idx):
            results.append(MatchResult(
                node_id=child.node_id,
                role=child.role,
                label=child.label,
                selector=format_selector(child.role, child.label, idx),
                path=path + [node.role],
            ))

        child_path = path + [node.role] if node.role else path
        _walk(child, selector, child_path, results)


def _walk_children(
    node: UINode,
    selector: UISelector,
    path: List[str],
    results: List[MatchResult],
) -> None:
    """Walk children without checking current node."""
    for child in node.children:
        child_path = path + [node.role] if node.role else path
        _walk(child, selector, child_path, results)


def anchor_to_selector(anchor: AnchorCandidate) -> UISelector:
    """Convert a Recording-side AnchorCandidate to an Execution-side UISelector."""
    return UISelector(
        role=anchor.element_role,
        label=anchor.element_label,
        label_contains=False,
        index=-1,
        ancestors=(),
    )


def selector_to_anchor(
    selector: UISelector, *, step_index: int = 0, app_bundle_id: str = ""
) -> AnchorCandidate:
    """Convert an Execution-side UISelector to a Recording-side AnchorCandidate."""
    return AnchorCandidate(
        step_index=step_index,
        element_label=selector.label,
        element_role=selector.role,
        app_bundle_id=app_bundle_id,
    )


def _ancestors_match(path: List[str], ancestors: Tuple[str, ...]) -> bool:
    """Check if the path ends with the required ancestor sequence."""
    if not ancestors:
        return True
    if len(path) < len(ancestors):
        return False
    tail = path[-len(ancestors):]
    return all(a == p for a, p in zip(ancestors, tail))

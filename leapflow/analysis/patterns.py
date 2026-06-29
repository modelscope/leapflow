"""Extensible action pattern library for trajectory abstraction.

Supports YAML-driven pattern definitions with wildcard matching,
parameter extraction, and runtime dynamic addition.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from leapflow.domain.trajectory import SemanticAction

logger = logging.getLogger(__name__)

# ── Pattern data model ──

_CONSTRAINT_RE = re.compile(r"^([^(]+)\(([^)]+)\)$")


@dataclass(frozen=True)
class PatternParam:
    """Parameter extraction rule from a matched pattern."""

    name: str
    from_field: str  # e.g. "0.parameters.path" or "1.app_bundle_id"
    type: str = "str"  # "str", "path", "int"
    description: str = ""


@dataclass(frozen=True)
class ActionPattern:
    """A reusable action sequence pattern."""

    name: str
    sequence: List[str]  # wildcards: "ui.*", constraints: "app.switch(Safari|Chrome)"
    semantic_name: str  # output action_name after successful match
    parameters: List[PatternParam] = field(default_factory=list)
    confidence: float = 0.8
    description: str = ""
    category: str = "general"


@dataclass
class PatternMatch:
    """A successfully matched pattern instance."""

    pattern: ActionPattern
    start_idx: int
    end_idx: int  # exclusive
    extracted_params: Dict[str, Any] = field(default_factory=dict)
    match_confidence: float = 0.0


# ── Pattern library ──


class PatternLibrary:
    """Extensible pattern library supporting YAML loading and wildcard matching (OCP)."""

    def __init__(self, patterns: Optional[List[ActionPattern]] = None) -> None:
        self._patterns: List[ActionPattern] = list(patterns or [])

    @classmethod
    def from_yaml(cls, path: Path) -> PatternLibrary:
        """Load patterns from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        patterns: List[ActionPattern] = []
        for entry in data.get("patterns", []):
            params = [
                PatternParam(
                    name=p["name"],
                    from_field=p["from_field"],
                    type=p.get("type", "str"),
                    description=p.get("description", ""),
                )
                for p in entry.get("parameters", [])
            ]
            patterns.append(
                ActionPattern(
                    name=entry["name"],
                    sequence=entry["sequence"],
                    semantic_name=entry["semantic_name"],
                    parameters=params,
                    confidence=entry.get("confidence", 0.8),
                    description=entry.get("description", ""),
                    category=entry.get("category", "general"),
                )
            )
        return cls(patterns)

    @classmethod
    def default(cls) -> PatternLibrary:
        """Load from the bundled patterns.yaml."""
        default_path = Path(__file__).parent / "patterns.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()

    @property
    def patterns(self) -> List[ActionPattern]:
        return list(self._patterns)

    def add_pattern(self, pattern: ActionPattern) -> None:
        """Add a pattern at runtime (e.g. from user feedback)."""
        self._patterns.append(pattern)

    def remove_pattern(self, name: str) -> bool:
        """Remove a pattern by name. Returns True if removed."""
        before = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.name != name]
        return len(self._patterns) < before

    def match(self, actions: Sequence[SemanticAction]) -> List[PatternMatch]:
        """Find all non-overlapping pattern matches using greedy longest-match.

        Iterates through the action sequence; at each position tries patterns
        sorted by descending sequence length (longest first).
        """
        sorted_patterns = sorted(self._patterns, key=lambda p: len(p.sequence), reverse=True)
        matches: List[PatternMatch] = []
        i = 0
        n = len(actions)

        while i < n:
            best: Optional[PatternMatch] = None
            for pattern in sorted_patterns:
                result = self._match_single(pattern, actions, i)
                if result is not None:
                    if best is None or (result.end_idx - result.start_idx) > (
                        best.end_idx - best.start_idx
                    ):
                        best = result
                    break  # sorted by length desc, first hit is longest
            if best is not None:
                matches.append(best)
                i = best.end_idx
            else:
                i += 1
        return matches

    def _match_single(
        self,
        pattern: ActionPattern,
        actions: Sequence[SemanticAction],
        start: int,
    ) -> Optional[PatternMatch]:
        """Try to match a single pattern starting at position start."""
        seq_len = len(pattern.sequence)
        if start + seq_len > len(actions):
            return None

        for offset, pattern_elem in enumerate(pattern.sequence):
            if not self._matches_element(pattern_elem, actions[start + offset]):
                return None

        matched_actions = list(actions[start : start + seq_len])
        extracted = self._extract_params(pattern, matched_actions)

        return PatternMatch(
            pattern=pattern,
            start_idx=start,
            end_idx=start + seq_len,
            extracted_params=extracted,
            match_confidence=pattern.confidence,
        )

    @staticmethod
    def _matches_element(pattern_elem: str, action: SemanticAction) -> bool:
        """Check if a single pattern element matches an action.

        Supports:
        - Exact match: "clipboard.copy"
        - Wildcard: "ui.*", "file.*"
        - Constraint: "app.switch(Safari|Chrome)"
        """
        # Check for constraint pattern: "name(regex)"
        m = _CONSTRAINT_RE.match(pattern_elem)
        if m:
            base_pattern, constraint = m.group(1), m.group(2)
            if not fnmatch.fnmatch(action.action_name, base_pattern):
                return False
            # constraint matched against any string value in parameters
            constraint_re = re.compile(constraint)
            return any(
                isinstance(v, str) and constraint_re.search(v)
                for v in action.parameters.values()
            )

        # Wildcard or exact match via fnmatch
        return fnmatch.fnmatch(action.action_name, pattern_elem)

    @staticmethod
    def _extract_params(
        pattern: ActionPattern, matched_actions: List[SemanticAction]
    ) -> Dict[str, Any]:
        """Extract parameter values from matched actions based on pattern param rules."""
        result: Dict[str, Any] = {}
        for param in pattern.parameters:
            value = _resolve_field(param.from_field, matched_actions)
            if value is not None:
                if param.type == "int":
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                result[param.name] = value
        return result


def _resolve_field(from_field: str, actions: List[SemanticAction]) -> Any:
    """Resolve a dotpath field reference like '0.parameters.path' or '1.action_name'.

    Format: '<action_index>.<field_path>'
    Field path resolution order:
      1. Direct attribute on SemanticAction (action_name, description, confidence)
      2. Key in the parameters dict
      3. Nested dotpath into parameters dict
    """
    parts = from_field.split(".", 1)
    if len(parts) < 2:
        return None

    try:
        idx = int(parts[0])
    except ValueError:
        return None

    if idx < 0 or idx >= len(actions):
        return None

    action = actions[idx]
    field_path = parts[1]

    # Direct attribute lookup
    if field_path in ("action_name", "description", "confidence"):
        return getattr(action, field_path)

    # Strip optional 'parameters.' prefix for convenience
    if field_path.startswith("parameters."):
        field_path = field_path[len("parameters."):]

    # Navigate nested dict path
    obj: Any = action.parameters
    for key in field_path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
        if obj is None:
            return None
    return obj

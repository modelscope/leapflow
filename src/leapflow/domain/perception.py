"""Perceptual Field domain types — fine-grained context-aware perception control.

Defines the vocabulary for expressing per-context perception policies:
    PerceptionLevel — how deeply to perceive a given context
    ContextIdentifier — what context within an app the user is in
    FieldRule — a single policy rule mapping context patterns to perception levels
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from typing import Sequence


class PerceptionLevel(Enum):
    """Perception depth for a specific in-app context."""

    FULL = "full"
    STRUCTURAL = "structural"
    OPAQUE = "opaque"
    DENY = "deny"


@dataclass(frozen=True)
class ContextIdentifier:
    """Identifies a specific context within an app (meta-perception output).

    This is the routing signal — it tells the policy engine WHERE the user is,
    without carrying any content about WHAT they're doing.
    """

    app_bundle_id: str
    context_type: str
    context_value: str


@dataclass(frozen=True)
class FieldRule:
    """A single perceptual field rule — maps an (app, context) pattern to a perception level.

    Rules are evaluated in priority order (highest first). First match wins.
    """

    app_pattern: str
    context_pattern: str
    level: PerceptionLevel
    source: str = "user"
    priority: int = 0

    def matches(self, ctx: ContextIdentifier) -> bool:
        """Check if this rule matches the given context identifier."""
        if not _glob_match(self.app_pattern, ctx.app_bundle_id):
            return False
        return _multi_glob_match(self.context_pattern, ctx.context_value)


def _glob_match(pattern: str, value: str) -> bool:
    """Case-insensitive glob match."""
    return fnmatch(value.lower(), pattern.lower())


def _multi_glob_match(pattern: str, value: str) -> bool:
    """Match against a pipe-separated list of glob patterns."""
    for p in pattern.split("|"):
        p = p.strip()
        if p and fnmatch(value.lower(), p.lower()):
            return True
    return False


def sort_rules(rules: Sequence[FieldRule]) -> list[FieldRule]:
    """Sort rules by priority descending (highest priority first)."""
    return sorted(rules, key=lambda r: -r.priority)

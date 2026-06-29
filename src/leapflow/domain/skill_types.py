"""Shared skill-related data types used across learning and runtime layers."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@enum.unique
class SkillTier(enum.IntEnum):
    """Explicit maturity tier for skill lifecycle management.

    Graduation thresholds are checked in ConfirmLevel determination —
    skills progress through tiers based on execution count and confidence.
    """

    DRAFT = 1
    CANDIDATE = 2
    VERIFIED = 3
    PRODUCTION = 4

    @classmethod
    def from_metadata(
        cls,
        version: int,
        confidence: float,
        source: str = "",
    ) -> "SkillTier":
        """Derive tier from metadata signals (version count, confidence, source).

        Builtin/user skills start at VERIFIED since they were explicitly authored.
        Distilled skills progress through the full maturity ladder.
        """
        if source in ("builtin", "user"):
            if version >= 3 and confidence >= 0.9:
                return cls.PRODUCTION
            return cls.VERIFIED
        if version >= 10 and confidence >= 0.9:
            return cls.PRODUCTION
        if version >= 3 and confidence >= 0.7:
            return cls.VERIFIED
        if version >= 2 and confidence >= 0.6:
            return cls.CANDIDATE
        return cls.DRAFT


@dataclass(frozen=True)
class RecoveryEvent:
    """An error-recovery pattern detected in a demonstration trajectory."""

    pattern: str
    trigger_action: str
    recovery_action: str
    confidence: float = 0.5


@dataclass(frozen=True)
class AnchorCandidate:
    """A UI element identifier extracted from a demonstration step."""

    step_index: int
    element_label: str
    element_role: str = ""
    app_bundle_id: str = ""


@dataclass(frozen=True)
class DistillationCandidate:
    """A candidate distilled routine."""

    title: str
    trigger_phrases: List[str]
    steps: List[str]
    parameters: List[Dict[str, str]] = field(default_factory=list)
    pre_conditions: List[str] = field(default_factory=list)
    post_conditions: List[str] = field(default_factory=list)
    source_trajectory_id: str = ""
    source_episode_id: str = ""
    confidence: float = 0.0
    recovery_events: List[RecoveryEvent] = field(default_factory=list)
    anchor_candidates: List[AnchorCandidate] = field(default_factory=list)
    procedure_graph: str = ""
    error_handling: List[Dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SkillParameter:
    """Declares a skill parameter with type, default, and optional validation."""

    name: str
    type: str = "str"          # "str", "int", "float", "bool", "dict", "list", "path"
    required: bool = False
    default: Optional[Any] = None
    description: str = ""


@dataclass(frozen=True)
class SkillMetadata:
    """Provenance and versioning information for a skill."""

    source: str = "builtin"    # "builtin" | "distilled" | "user" | "template"
    source_trajectory_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    confidence: float = 1.0
    version: int = 1
    created_at: float = field(default_factory=time.time)
    tags: tuple = ()           # immutable tag set

    @property
    def tier(self) -> SkillTier:
        """Derived maturity tier based on version, confidence, and source."""
        return SkillTier.from_metadata(self.version, self.confidence, self.source)

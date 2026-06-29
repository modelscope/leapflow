"""Protocols and context containers for the MHMS-SF fusion pipeline.

Defines:
    ScaleFusionAgent  — Protocol for scale-specific fusion agents (ISP+DIP)
    FusionContext     — immutable input container carrying all signal sources
    FusionResult      — unified output from any SFA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from leapflow.signal_fusion.types import (
    AtomicAction,
    EnrichedEpisode,
    Segment,
)

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent
    from leapflow.perception.types import ChannelStatus, Keyframe, VisualAction
    from leapflow.signal_fusion.quality import FusionQuality
    from leapflow.signal_fusion.types import AppTransitionEvent


@runtime_checkable
class ScaleFusionAgent(Protocol):
    """Protocol for scale-specific fusion agents.

    Each agent transforms FusionContext at a single temporal scale.
    The pipeline chains agents, passing each result as upstream_result
    to the next agent's context.
    """

    async def fuse(self, context: "FusionContext") -> "FusionResult": ...


@dataclass
class FusionContext:
    """Unified input context for all fusion agents.

    Each SFA extracts the subset it needs (ISP — agents are not forced
    to consume fields they don't use).
    """

    visual_actions: List["VisualAction"] = field(default_factory=list)
    system_events: List["SystemEvent"] = field(default_factory=list)
    keyframes: List["Keyframe"] = field(default_factory=list)
    app_transitions: List["AppTransitionEvent"] = field(default_factory=list)
    channel_status: Optional["ChannelStatus"] = None
    goal: str = ""
    upstream_result: Optional["FusionResult"] = None


@dataclass
class FusionResult:
    """Unified output from any ScaleFusionAgent."""

    atomic_actions: List[AtomicAction] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    episodes: List[EnrichedEpisode] = field(default_factory=list)
    quality: Optional["FusionQuality"] = None

"""Action-scale fusion: converts raw visual actions + system events into AtomicActions.

Replaces the former ActionScaleAligner with a simpler, OCP-friendly design.
Produces the first-stage AtomicActions consumed by SegmentFusionAgent.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, TYPE_CHECKING

from leapflow.signal_fusion.protocol import FusionContext, FusionResult
from leapflow.signal_fusion.types import AtomicAction, FusionMode

if TYPE_CHECKING:
    from leapflow.domain.events import SystemEvent
    from leapflow.perception.types import VisualAction

logger = logging.getLogger(__name__)


class ActionFusionAgent:
    """Fuses visual actions with system events into AtomicActions.

    Strategy: for each visual action, find the closest system event
    within a time window. Corroborated pairs get FusionMode.FULL;
    unmatched visuals get VISUAL_ONLY; unmatched events get EVENT_ONLY.
    """

    def __init__(
        self,
        *,
        time_tolerance: float = 2.0,
        corroboration_boost: float = 1.2,
        event_only_confidence: float = 0.7,
    ) -> None:
        self._tolerance = time_tolerance
        self._corroboration_boost = corroboration_boost
        self._event_only_confidence = event_only_confidence

    async def fuse(self, context: FusionContext) -> FusionResult:
        visual = context.visual_actions or []
        events = context.system_events or []
        if not visual and not events:
            return FusionResult()

        actions = self._fuse(visual, events)
        logger.debug("action_fusion: %d atomic actions", len(actions))
        return FusionResult(atomic_actions=actions)

    def _fuse(
        self,
        visual: Sequence["VisualAction"],
        events: Sequence["SystemEvent"],
    ) -> List[AtomicAction]:
        used_events: set = set()
        atoms: List[AtomicAction] = []

        for va in visual:
            match_idx = self._find_closest_event(va, events, used_events)
            if match_idx is not None:
                ev = events[match_idx]
                used_events.add(match_idx)
                atoms.append(AtomicAction(
                    action=va.action,
                    target=va.target,
                    detail=va.detail,
                    timestamp=ev.timestamp,
                    app_bundle=ev.source,
                    confidence=min(1.0, va.confidence * self._corroboration_boost),
                    source_signals=["visual", "event"],
                    fusion_mode=FusionMode.FULL,
                    visual_evidence=va.evidence,
                    frame_ref=va.frame_ref_a,
                ))
            else:
                atoms.append(AtomicAction(
                    action=va.action,
                    target=va.target,
                    detail=va.detail,
                    timestamp=0.0,
                    confidence=va.confidence,
                    source_signals=["visual"],
                    fusion_mode=FusionMode.VISUAL_ONLY,
                    visual_evidence=va.evidence,
                    frame_ref=va.frame_ref_a,
                ))

        for i, ev in enumerate(events):
            if i in used_events:
                continue
            atoms.append(AtomicAction(
                action=ev.event_type,
                target=ev.payload.get("label", ev.payload.get("target", "")),
                detail="",
                timestamp=ev.timestamp,
                app_bundle=ev.source,
                confidence=self._event_only_confidence,
                source_signals=["event"],
                fusion_mode=FusionMode.EVENT_ONLY,
            ))

        atoms.sort(key=lambda a: a.timestamp)
        return atoms

    def _find_closest_event(
        self,
        va: "VisualAction",
        events: Sequence["SystemEvent"],
        used: set,
    ) -> Optional[int]:
        """Find the system event best matching a visual action.

        Uses action-type substring matching + time proximity within
        self._tolerance.  Returns the index of the best match, or None.
        """
        best_idx = None
        best_dist = float("inf")
        for i, ev in enumerate(events):
            if i in used:
                continue
            if not _action_types_compatible(va.action, ev.event_type):
                continue
            dist = abs(ev.timestamp)
            if dist < best_dist and dist <= self._tolerance:
                best_dist = dist
                best_idx = i
        return best_idx


def _action_types_compatible(visual_action: str, event_type: str) -> bool:
    """Heuristic check whether a visual action label aligns with an event type."""
    va = visual_action.lower()
    et = event_type.lower()
    return va in et or et.split(".")[-1] in va

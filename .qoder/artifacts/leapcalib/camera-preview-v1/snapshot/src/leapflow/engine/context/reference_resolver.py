# Copyright (c) Alibaba, Inc. and its affiliates.
"""Focus-state-driven reference resolution for session context assembly.

This resolver does not parse user text for keywords, does not choose tools,
and does not route intents. It resolves potential deictic references ("the
above paper", "that thing I just set") purely from the structured
SessionFocusState — which entities are active, how recently they were
observed, and whether control-plane events are the only recent activity.

Design rationale: keyword-driven intent routing (regex + if-else chains)
violates the LLM-native principle. Instead, the LLM itself understands
natural language; this module only provides structured context about what
the session focus is, so the LLM can ground its reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from leapflow.engine.context.context_focus import ContextPlane, ReferenceResolution, SessionFocusState


@dataclass(frozen=True)
class ReferenceResolverConfig:
    """Tunable confidence thresholds for focus-based resolution."""

    single_entity_confidence: float = 0.92
    recency_fallback_confidence: float = 0.75
    no_entity_confidence: float = 0.0


@dataclass(frozen=True)
class ReferenceResolver:
    """Resolve deictic references using structured focus state only.

    Resolution is based entirely on SessionFocusState — no regex, no keyword
    parsing, no if-else routing by user-text content.  The strategy is:

    1. If task-semantic entities exist in the focus stack, return the most
       recent (highest confidence when only one exists).
    2. If no task entities but control-plane events exist, return the most
       recent control event as a fallback.
    3. If nothing is in the focus state, return unresolved.
    """

    config: ReferenceResolverConfig = field(default_factory=ReferenceResolverConfig)

    def resolve(self, user_text: str, state: SessionFocusState) -> ReferenceResolution:
        """Resolve a potential reference against the current focus state.

        Args:
            user_text: The user's input (accepted for API compatibility but
                not parsed for keywords).
            state: The current session focus state containing structured
                entity and control-event records.

        Returns:
            A ReferenceResolution indicating the resolved target, confidence,
            and reasoning.
        """
        # Priority 1: task-semantic entities from the focus stack
        candidates = state.task_focus_candidates()
        if candidates:
            focus = candidates[0]
            if len(candidates) == 1:
                return ReferenceResolution(
                    target_kind=focus.kind,
                    target_id=focus.entity_id,
                    target_name=focus.canonical_name,
                    plane=focus.plane,
                    confidence=self.config.single_entity_confidence,
                    reason="single active entity in focus state",
                )
            return ReferenceResolution(
                target_kind=focus.kind,
                target_id=focus.entity_id,
                target_name=focus.canonical_name,
                plane=focus.plane,
                confidence=self.config.recency_fallback_confidence,
                reason="most recent entity from multiple candidates",
            )

        # Priority 2: control-plane events when no task entities exist
        latest_control = state.latest_control_event()
        if latest_control is not None:
            return ReferenceResolution(
                target_kind="config",
                target_id=f"control:{latest_control.key}:{latest_control.turn_id}",
                target_name=latest_control.value or latest_control.key,
                plane=ContextPlane.CONTROL_PLANE,
                confidence=self.config.recency_fallback_confidence,
                reason="fallback to most recent control-plane event",
            )

        # Nothing in focus state
        return ReferenceResolution.unresolved("no active entities in focus state")


__all__ = ["ReferenceResolver", "ReferenceResolverConfig"]

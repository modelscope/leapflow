"""Episode-scale fusion: cross-app workflow graph construction.

Assembles Segments into EnrichedEpisodes with WorkflowGraph DAGs,
enabling downstream skill extraction to understand multi-app workflows
as structured graphs rather than flat action sequences.

SRP: builds workflow graphs and episodes — delegates intent inference.
DIP: depends on IntentInferrer protocol, not concrete LLM implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Sequence, Tuple

from leapflow.signal_fusion.protocol import FusionContext, FusionResult
from leapflow.signal_fusion.types import (
    AppTransitionEvent,
    CarryPayload,
    EnrichedEpisode,
    NodeRole,
    Segment,
    WorkflowEdge,
    WorkflowGraph,
    WorkflowNode,
    WorkflowType,
)

if TYPE_CHECKING:
    from leapflow.analysis.intent_inferrer import IntentInferrer
    from leapflow.causal.channel import ChannelRegistry

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_ACTIONS = frozenset({"type", "paste", "click"})
_DEFAULT_OUTPUT_ACTIONS = frozenset({"copy", "clipboard"})
_DEFAULT_PASSIVE_ACTIONS = frozenset({"scroll", "click", "app_switch"})


def _build_action_sets_from_registry(
    registry: "ChannelRegistry",
) -> Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
    """Derive input/output/passive action sets from channel roles.

    Mapping:
        TRIGGER    → input actions
        EFFECT     → output actions
        NAVIGATION / BOUNDARY → passive actions
    """
    from leapflow.causal.types import EventType

    input_actions: set = set()
    output_actions: set = set()
    passive_actions: set = set()

    for spec in registry.all_specs():
        if spec.default_role == EventType.TRIGGER:
            input_actions.add(spec.channel)
        elif spec.default_role == EventType.EFFECT:
            output_actions.add(spec.channel)
        elif spec.default_role in (EventType.NAVIGATION, EventType.BOUNDARY):
            passive_actions.add(spec.channel)

    return frozenset(input_actions), frozenset(output_actions), frozenset(passive_actions)


class EpisodeFusionAgent:
    """Episode-scale fusion agent.

    Implements ScaleFusionAgent protocol. Consumes upstream Segments
    and produces EnrichedEpisodes with WorkflowGraph structures.
    """

    def __init__(
        self,
        intent_inferrer: Optional["IntentInferrer"] = None,
        *,
        registry: Optional["ChannelRegistry"] = None,
        input_actions: Optional[frozenset] = None,
        output_actions: Optional[frozenset] = None,
        passive_actions: Optional[frozenset] = None,
    ) -> None:
        self._intent_inferrer = intent_inferrer

        # Derive action sets from registry if available; fallback to defaults
        if registry is not None and input_actions is None and output_actions is None and passive_actions is None:
            input_actions, output_actions, passive_actions = _build_action_sets_from_registry(registry)

        self._input_actions = input_actions or _DEFAULT_INPUT_ACTIONS
        self._output_actions = output_actions or _DEFAULT_OUTPUT_ACTIONS
        self._passive_actions = passive_actions or _DEFAULT_PASSIVE_ACTIONS

    async def fuse(self, context: FusionContext) -> FusionResult:
        upstream = context.upstream_result
        if not upstream or not upstream.segments:
            return FusionResult()

        segments = upstream.segments
        graph = self._build_workflow_graph(segments, context.app_transitions)
        graph.workflow_type = self._classify_workflow(graph)

        intent = ""
        intent_confidence = 0.0
        if self._intent_inferrer:
            intent, intent_confidence = await self._infer_intent(
                segments, graph, context.goal
            )

        episode = EnrichedEpisode(
            segments=segments,
            workflow_graph=graph if graph.is_multi_app else None,
            intent=intent,
            intent_confidence=intent_confidence,
            metadata={
                "goal": context.goal,
                "workflow_type": graph.workflow_type.value,
                "app_count": len(graph.unique_apps),
            },
        )

        return FusionResult(
            atomic_actions=upstream.atomic_actions,
            segments=segments,
            episodes=[episode],
            quality=upstream.quality,
        )

    # ── Workflow Graph Construction ──

    def _build_workflow_graph(
        self,
        segments: List[Segment],
        transitions: Sequence[AppTransitionEvent],
    ) -> WorkflowGraph:
        nodes: List[WorkflowNode] = []
        node_map: Dict[str, WorkflowNode] = {}

        for seg in segments:
            node = WorkflowNode(
                app_bundle=seg.dominant_app,
                segment=seg,
                role=self._infer_node_role(
                    seg, segments,
                    self._input_actions, self._output_actions, self._passive_actions,
                ),
            )
            nodes.append(node)
            node_map[seg.segment_id] = node

        edges = self._build_edges(nodes, transitions)

        return WorkflowGraph(nodes=nodes, edges=edges)

    def _build_edges(
        self,
        nodes: List[WorkflowNode],
        transitions: Sequence[AppTransitionEvent],
    ) -> List[WorkflowEdge]:
        """Build edges from app transitions, linking to nearest nodes."""
        if len(nodes) < 2:
            return []

        edges: List[WorkflowEdge] = []
        for i in range(len(nodes) - 1):
            src = nodes[i]
            dst = nodes[i + 1]

            carry = self._find_carry_for_transition(
                src.app_bundle, dst.app_bundle, transitions
            )

            edges.append(WorkflowEdge(
                from_node_id=src.node_id,
                to_node_id=dst.node_id,
                carry=carry,
                transition_ts=dst.segment.start_ts if dst.segment else 0.0,
            ))

        return edges

    @staticmethod
    def _find_carry_for_transition(
        from_app: str,
        to_app: str,
        transitions: Sequence[AppTransitionEvent],
    ) -> CarryPayload:
        """Find the best matching transition event for an edge."""
        for tr in reversed(transitions):
            if tr.from_bundle == from_app and tr.to_bundle == to_app:
                return CarryPayload.from_clipboard(tr.carry_clipboard, origin=from_app)
        return CarryPayload.from_clipboard(None)

    # ── Node Role Inference ──

    @staticmethod
    def _infer_node_role(
        segment: Segment,
        all_segments: List[Segment],
        input_actions: frozenset,
        output_actions: frozenset,
        passive_actions: frozenset,
    ) -> NodeRole:
        """Infer the role of a segment's app within the workflow.

        Uses configurable action-set heuristics rather than hardcoded rules.
        """
        if not segment.actions:
            return NodeRole.UNKNOWN

        action_types = [a.action for a in segment.actions]
        has_input = any(a in input_actions for a in action_types)
        has_output = any(a in output_actions for a in action_types)
        has_read_only = all(a in passive_actions for a in action_types)

        idx = next((i for i, s in enumerate(all_segments) if s.segment_id == segment.segment_id), 0)

        if idx == 0 and has_output:
            return NodeRole.SOURCE
        if idx == len(all_segments) - 1 and has_input:
            return NodeRole.SINK
        if has_read_only and not has_input:
            return NodeRole.REFERENCE
        if has_input or has_output:
            return NodeRole.TOOL
        return NodeRole.UNKNOWN

    # ── Workflow Classification ──

    @staticmethod
    def _classify_workflow(graph: WorkflowGraph) -> WorkflowType:
        """Classify the workflow pattern from graph topology."""
        apps = graph.app_sequence
        if len(apps) <= 1:
            return WorkflowType.UNKNOWN

        unique = list(dict.fromkeys(apps))

        if len(apps) == 2:
            return WorkflowType.LINEAR_TRANSFER

        if len(apps) >= 3 and apps[0] == apps[-1] and len(unique) == 2:
            return WorkflowType.ROUND_TRIP

        if len(apps) >= 5 and _is_alternating(apps):
            return WorkflowType.ITERATIVE_REFINEMENT

        if len(apps) >= 4 and _is_parallel_ref(apps):
            return WorkflowType.PARALLEL_REFERENCE

        if len(unique) >= 3:
            return WorkflowType.MULTI_HUB

        return WorkflowType.LINEAR_TRANSFER

    # ── Intent Inference ──

    async def _infer_intent(
        self,
        segments: List[Segment],
        graph: WorkflowGraph,
        goal: str,
    ) -> tuple:
        """Delegate intent inference to IntentInferrer."""
        if not self._intent_inferrer:
            return ("", 0.0)

        from leapflow.domain.trajectory import Episode

        episode = Episode(
            app_sequence=graph.app_sequence,
        )

        action_summaries = []
        for seg in segments:
            for a in seg.actions[:10]:
                action_summaries.append(f"{a.action}({a.target})")

        context = {
            "goal": goal,
            "workflow_type": graph.workflow_type.value,
            "action_summary": " → ".join(action_summaries[:20]),
        }

        try:
            result = await self._intent_inferrer.infer(episode, context)
            return (result.goal, result.confidence)
        except Exception:
            logger.warning("Intent inference failed, continuing without intent", exc_info=True)
            return ("", 0.0)


# ── Helpers ──


def _is_alternating(apps: List[str]) -> bool:
    """Check for A-B-A-B pattern."""
    if len(apps) < 4:
        return False
    unique = list(dict.fromkeys(apps))
    if len(unique) != 2:
        return False
    for i in range(2, len(apps)):
        if apps[i] != apps[i - 2]:
            return False
    return True


def _is_parallel_ref(apps: List[str]) -> bool:
    """Check for rapid switching between two apps."""
    if len(apps) < 4:
        return False
    unique = set(apps[-4:])
    return len(unique) == 2

"""MHMS-SF Fusion Pipeline — orchestrates multi-scale fusion agents.

Chains ScaleFusionAgent implementations in sequence:
    SegmentFusionAgent → EpisodeFusionAgent
then applies CrossScaleIntegrator for bidirectional consistency.

The pipeline is the single entry point for the analysis layer to
invoke multi-source heterogeneous multi-scale signal fusion.

OCP: agents are injected — new scales or strategies are added by
     providing additional ScaleFusionAgent instances.
DIP: depends on ScaleFusionAgent protocol, not concrete classes.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from leapflow.signal_fusion.protocol import (
    FusionContext,
    FusionResult,
    ScaleFusionAgent,
)
from leapflow.signal_fusion.types import EnrichedEpisode
from leapflow.utils.diagnostics import PipelineTracer

if TYPE_CHECKING:
    from leapflow.signal_fusion.integrator import CrossScaleIntegrator

logger = logging.getLogger(__name__)


class MHMSFusionPipeline:
    """Multi-source Heterogeneous Multi-scale Signal Fusion pipeline.

    Orchestrates a chain of ScaleFusionAgent instances, each operating
    at a different temporal scale (action → segment → episode), then
    applies cross-scale integration for bidirectional consistency.

    Usage:
        pipeline = MHMSFusionPipeline.default(llm=my_llm)
        result = await pipeline.fuse(context)
    """

    def __init__(
        self,
        agents: Sequence[ScaleFusionAgent],
        integrator: Optional["CrossScaleIntegrator"] = None,
    ) -> None:
        self._agents = list(agents)
        self._integrator = integrator

    @classmethod
    def default(
        cls,
        *,
        llm: Optional[object] = None,
        intent_inferrer: Optional[object] = None,
    ) -> "MHMSFusionPipeline":
        """Create a pipeline with the standard 3-agent chain + integrator."""
        from leapflow.signal_fusion.action_agent import ActionFusionAgent
        from leapflow.signal_fusion.episode_agent import EpisodeFusionAgent
        from leapflow.signal_fusion.integrator import CrossScaleIntegrator
        from leapflow.signal_fusion.segment_agent import SegmentFusionAgent

        agents: List[ScaleFusionAgent] = [
            ActionFusionAgent(),
            SegmentFusionAgent(),
            EpisodeFusionAgent(intent_inferrer=intent_inferrer),  # type: ignore[arg-type]
        ]

        integrator = CrossScaleIntegrator(llm=llm)  # type: ignore[arg-type]

        return cls(agents=agents, integrator=integrator)

    async def fuse(self, context: FusionContext) -> FusionResult:
        """Execute the full fusion pipeline.

        Each agent receives the previous agent's FusionResult as
        upstream_result in the context, enabling bottom-up data flow.
        """
        t0 = time.monotonic()
        tracer = PipelineTracer(
            "mhms_fusion",
            enabled=logger.isEnabledFor(logging.INFO),
        )
        current_context = context
        result = FusionResult()

        for agent in self._agents:
            agent_name = type(agent).__name__
            try:
                with tracer.stage(agent_name):
                    result = await agent.fuse(current_context)
                    tracer.metric(f"{agent_name}_actions", len(result.atomic_actions))
                    tracer.metric(f"{agent_name}_segments", len(result.segments))
                    tracer.metric(f"{agent_name}_episodes", len(result.episodes))
                current_context = _chain_context(current_context, result)
                logger.debug(
                    "%s produced %d actions, %d segments, %d episodes",
                    agent_name,
                    len(result.atomic_actions),
                    len(result.segments),
                    len(result.episodes),
                )
            except (ValueError, RuntimeError, LookupError, TypeError, AttributeError) as exc:
                # Catch concrete recoverable failures only. ``KeyboardInterrupt``
                # and ``SystemExit`` derive from ``BaseException``, so they are
                # never swallowed here. Asynchronous cancellation
                # (``asyncio.CancelledError``) likewise propagates.
                #
                # We ``break`` rather than ``continue`` because each agent
                # consumes the previous agent's ``upstream_result`` (see
                # ``_chain_context``); skipping a failed scale would feed an
                # empty/partial result into later agents and silently degrade
                # the entire pipeline. Breaking preserves the partial result
                # produced so far and surfaces the failure via the error log.
                logger.error(
                    "Fusion agent %s failed (%s): %s; returning partial result",
                    agent_name,
                    exc.__class__.__name__,
                    exc,
                    exc_info=True,
                )
                break

        if self._integrator and result.episodes:
            try:
                with tracer.stage("CrossScaleIntegrator"):
                    result.episodes = await self._integrator.integrate(
                        result.atomic_actions,
                        result.segments,
                        result.episodes,
                    )
            except Exception:
                logger.warning(
                    "Cross-scale integration failed, returning pre-integration result",
                    exc_info=True,
                )

        elapsed = time.monotonic() - t0
        logger.info(
            "MHMS-SF pipeline completed in %.2fs: %d actions, %d segments, "
            "%d episodes, quality=%s",
            elapsed,
            len(result.atomic_actions),
            len(result.segments),
            len(result.episodes),
            result.quality.level if result.quality else "N/A",
        )
        if tracer.enabled:
            logger.info(tracer.summary_line())

        return result

    def to_domain_episodes(self, result: FusionResult) -> list:
        """Convert FusionResult episodes to domain Episode objects.

        Bridges the signal_fusion output to the existing analysis pipeline's
        expected Episode format, enabling gradual adoption.
        """
        from leapflow.domain.trajectory import Episode, SemanticAction

        domain_episodes: list = []
        for enriched in result.episodes:
            semantic_actions = _enriched_to_semantic_actions(enriched)

            episode = Episode(
                episode_id=enriched.episode_id,
                inferred_goal=enriched.intent,
                app_sequence=enriched.app_sequence,
                semantic_actions=semantic_actions,
                confidence=enriched.intent_confidence,
            )
            domain_episodes.append(episode)

        return domain_episodes


# ── Helpers ──


def _chain_context(base: FusionContext, result: FusionResult) -> FusionContext:
    """Create a new context with the previous result as upstream."""
    return FusionContext(
        visual_actions=base.visual_actions,
        system_events=base.system_events,
        keyframes=base.keyframes,
        app_transitions=base.app_transitions,
        channel_status=base.channel_status,
        goal=base.goal,
        upstream_result=result,
    )


def _enriched_to_semantic_actions(
    episode: EnrichedEpisode,
) -> list:
    """Convert EnrichedEpisode's AtomicActions to SemanticAction list."""
    from leapflow.domain.trajectory import SemanticAction

    actions: list = []
    for seg in episode.segments:
        for i, atom in enumerate(seg.actions):
            params = {
                "_source": "mhms_sf",
                "_fusion_mode": atom.fusion_mode.value,
                "_confidence": atom.confidence,
            }
            if atom.clipboard_text:
                params["clipboard"] = atom.clipboard_text[:500]
            if atom.typed_text:
                params["typed_text"] = atom.typed_text
            if atom.shortcut:
                params["shortcut"] = atom.shortcut
            if atom.semantic_role:
                params["semantic_role"] = atom.semantic_role
            if atom.frame_ref:
                params["frame_ref"] = atom.frame_ref

            actions.append(SemanticAction(
                action_name=f"visual.{atom.action}" if atom.has_visual else atom.action,
                description=f"{atom.action} on {atom.target}" if atom.target else atom.action,
                parameters=params,
                confidence=atom.confidence,
            ))

    return actions

"""Causal Fusion Pipeline — Layer 1 + Layer 2 orchestrator.

Receives raw signals/system events, emits a populated CausalGraph with
chains ready for downstream VLM enrichment (Layer 3) and storage (Layer 4).

Architecture:
    Layer 1 (Event Emitter): raw signals → CausalEvent stream
    Layer 2 (Chain Builder): 5 components → CausalGraph with chains

Synchronous path budget: <10ms for typical input (≤50 events).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from leapflow.causal.channel import ChannelRegistry, build_default_registry
from leapflow.causal.components import (
    AttentionHotspotDetector,
    CausalChainBuilder,
    EventDenoiser,
    ReliabilityScorer,
    SignalToSemanticMapper,
)
from leapflow.causal.inference import CausalInferenceEngine, VLMVerifier
from leapflow.causal.types import (
    CausalChain,
    CausalEvent,
    CausalGraph,
    EventSource,
    EventType,
)
from leapflow.utils.diagnostics import PipelineTracer

if TYPE_CHECKING:
    from leapflow.perception.types import InteractionSignal

logger = logging.getLogger(__name__)


@dataclass
class FusionOutput:
    """Output of the causal fusion pipeline."""

    graph: CausalGraph
    chains: List[CausalChain]
    stats: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class ReorderBuffer:
    """Sliding-window reordering for out-of-order signal arrival (§4.5 scene 3).

    Holds events for up to `window_s` seconds before releasing them in
    timestamp order. This handles platform-level delivery jitter (e.g.,
    app_switch arriving 200ms after the keyboard shortcut that caused it).
    """

    __slots__ = ("_window_s", "_buffer")

    def __init__(self, window_s: Optional[float] = None) -> None:
        if window_s is None:
            from leapflow.config import get_settings
            window_s = get_settings().causal_reorder_window_ms / 1000.0
        self._window_s = window_s
        self._buffer: List[CausalEvent] = []

    def add(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Add events and flush those older than the window."""
        self._buffer.extend(events)
        self._buffer.sort(key=lambda e: e.timestamp)
        if not self._buffer:
            return []
        cutoff = self._buffer[-1].timestamp - self._window_s
        ready: List[CausalEvent] = []
        remaining: List[CausalEvent] = []
        for ev in self._buffer:
            if ev.timestamp <= cutoff:
                ready.append(ev)
            else:
                remaining.append(ev)
        self._buffer = remaining
        return ready

    def flush(self) -> List[CausalEvent]:
        """Force-flush all remaining events (e.g., at session end)."""
        result = sorted(self._buffer, key=lambda e: e.timestamp)
        self._buffer.clear()
        return result


class CausalEventEmitter:
    """Layer 1: transform raw signals into CausalEvents.

    Reads ChannelRegistry for role resolution and reactive capture decisions.
    No hardcoded channel logic — all behavior is table-driven.
    """

    __slots__ = ("_registry",)

    def __init__(self, registry: ChannelRegistry) -> None:
        self._registry = registry

    def emit_from_signals(self, signals: Sequence["InteractionSignal"]) -> List[CausalEvent]:
        """Convert InteractionSignals to CausalEvents."""
        events: List[CausalEvent] = []
        recent: List[CausalEvent] = []

        for sig in signals:
            ev = self._signal_to_event(sig)
            if ev is None:
                continue
            role = self._registry.resolve_role(ev, recent)
            ev.event_type = role
            if not self._registry.is_available(ev.channel):
                spec = self._registry.get_spec(ev.channel)
                ev.tags["degraded"] = True
                ev.confidence *= spec.degraded_confidence_factor if spec else 0.6
            events.append(ev)
            recent.append(ev)
            if len(recent) > 10:
                recent.pop(0)

        return events

    def emit_from_system_events(self, system_events: Sequence[Any]) -> List[CausalEvent]:
        """Convert domain SystemEvents to CausalEvents."""
        events: List[CausalEvent] = []
        recent: List[CausalEvent] = []

        for se in system_events:
            ev = self._system_event_to_causal(se)
            if ev is None:
                continue
            role = self._registry.resolve_role(ev, recent)
            ev.event_type = role
            events.append(ev)
            recent.append(ev)
            if len(recent) > 10:
                recent.pop(0)

        return events

    def _signal_to_event(self, sig: "InteractionSignal") -> Optional[CausalEvent]:
        channel = sig.signal_type
        if not channel:
            return None

        payload: Dict[str, Any] = {}
        if sig.position:
            payload["x"] = sig.position[0]
            payload["y"] = sig.position[1]
        if hasattr(sig, "end_position") and sig.end_position:
            payload["end_x"] = sig.end_position[0]
            payload["end_y"] = sig.end_position[1]
        if sig.detail:
            spec = self._registry.get_spec(channel)
            key = spec.detail_payload_key if spec else "detail"
            payload[key] = sig.detail
            self._enrich_from_detail(payload, sig.detail)

        return CausalEvent(
            id=CausalEvent.make_id(),
            timestamp=sig.timestamp,
            event_type=EventType.TRIGGER,
            source=EventSource.SIGNAL,
            channel=channel,
            payload=payload,
            confidence=1.0,
        )

    @staticmethod
    def _enrich_from_detail(payload: Dict[str, Any], detail: str) -> None:
        """Parse structured info from detail string into payload fields."""
        if detail.startswith("dy="):
            try:
                payload["delta_y"] = int(detail[3:])
            except ValueError:
                pass
        elif " -> " in detail:
            parts = detail.split(" -> ", 1)
            payload.setdefault("from", parts[0])
            payload.setdefault("to", parts[1])

    def _system_event_to_causal(self, se: Any) -> Optional[CausalEvent]:
        event_type_str = getattr(se, "event_type", "")
        timestamp = getattr(se, "timestamp", 0.0)
        payload_raw = getattr(se, "payload", {})

        channel = self._registry.resolve_channel(event_type_str)

        payload: Dict[str, Any] = dict(payload_raw) if payload_raw else {}
        if "key_combo" in payload:
            payload["combo"] = payload.pop("key_combo")

        return CausalEvent(
            id=CausalEvent.make_id(),
            timestamp=timestamp,
            event_type=EventType.TRIGGER,
            source=EventSource.SYSTEM,
            channel=channel,
            payload=payload,
            confidence=1.0,
        )


class CausalFusionPipeline:
    """Orchestrates Layer 1 + Layer 2: emit → reorder → denoise → build → score → infer.

    Stateless per-call; the CausalGraph accumulates state across calls
    when passed in via `graph` parameter (ring buffer handles eviction).
    """

    __slots__ = (
        "_registry", "_emitter", "_reorder", "_denoiser", "_builder",
        "_scorer", "_hotspot", "_mapper", "_inference", "_vlm_verifier",
    )

    def __init__(
        self,
        registry: Optional[ChannelRegistry] = None,
        reorder_window_s: Optional[float] = None,
        *,
        vlm_verifier: Optional[VLMVerifier] = None,
    ) -> None:
        self._registry = registry or build_default_registry()
        self._emitter = CausalEventEmitter(self._registry)
        self._reorder = ReorderBuffer(window_s=reorder_window_s)
        self._denoiser = EventDenoiser(self._registry)
        self._builder = CausalChainBuilder(self._registry)
        self._scorer = ReliabilityScorer()
        self._hotspot = AttentionHotspotDetector()
        self._mapper = SignalToSemanticMapper(self._registry)
        self._inference = CausalInferenceEngine(self._registry)
        self._vlm_verifier = vlm_verifier

    def fuse(
        self,
        signals: Sequence["InteractionSignal"] = (),
        system_events: Sequence[Any] = (),
        graph: Optional[CausalGraph] = None,
    ) -> FusionOutput:
        """Run synchronous fusion: signals → CausalGraph with chains.

        Args:
            signals: Raw InteractionSignals from perception layer.
            system_events: Domain SystemEvents from event bus.
            graph: Existing graph to append to (creates new if None).

        Returns:
            FusionOutput with populated graph, new chains, and stats.
        """
        t0 = time.perf_counter()
        tracer = PipelineTracer(
            "causal_fusion",
            enabled=logger.isEnabledFor(logging.DEBUG),
        )
        if graph is None:
            graph = CausalGraph()

        # Layer 1: Emit CausalEvents
        events: List[CausalEvent] = []
        with tracer.stage("emit"):
            if signals:
                events.extend(self._emitter.emit_from_signals(signals))
            if system_events:
                events.extend(self._emitter.emit_from_system_events(system_events))
            # Sort by timestamp for temporal coherence
            events.sort(key=lambda e: e.timestamp)
            tracer.metric("emitted", len(events))

        tracer.metric("input_signals", len(signals) if signals else 0)
        tracer.metric("input_system_events", len(system_events) if system_events else 0)

        if not events:
            if tracer.enabled:
                logger.debug(tracer.summary_line())
            return FusionOutput(graph=graph, chains=[], elapsed_ms=0.0)

        # Layer 1 exit: reorder buffer (§4.5 scene 3 — handles delayed signals)
        with tracer.stage("reorder"):
            events = self._reorder.add(events)
            tracer.metric("reordered", len(events))
        if not events:
            if tracer.enabled:
                logger.debug(tracer.summary_line())
            return FusionOutput(graph=graph, chains=[], elapsed_ms=0.0)

        # Layer 2: Denoise
        with tracer.stage("denoise"):
            denoise_input = len(events)
            events = self._denoiser.denoise(events)
            tracer.metric("input", denoise_input)
            tracer.metric("output", len(events))

        # Layer 2: Build chains
        with tracer.stage("chain_build"):
            chains = self._builder.build(events)
            tracer.metric("chains", len(chains))

        # Register events and chains in graph
        for ev in events:
            graph.add_event(ev)
        for chain in chains:
            graph.add_chain(chain)

        # Layer 2: Inference (Tier 1 + Tier 2, synchronous)
        with tracer.stage("inference"):
            inference_stats = self._inference.infer_sync(events, graph)
            tracer.metric("rule_edges", inference_stats.get("rule_edges", 0))
            tracer.metric("heuristic_edges", inference_stats.get("heuristic_edges", 0))

        # Layer 2: Tier 3 VLM verification collection (async processing)
        if self._vlm_verifier is not None:
            with tracer.stage("vlm_collect"):
                pending = self._vlm_verifier.collect_pending(graph)
                tracer.metric("vlm_pending", len(pending))
                if pending:
                    inference_stats["vlm_pending"] = len(pending)
                    inference_stats["vlm_batches"] = len(self._vlm_verifier.get_batches())

        # Layer 2: Frequency counting (world model curiosity input)
        with tracer.stage("frequency_count"):
            freq_counter = graph.metadata.setdefault("frequency_counter", {})
            for ev in events:
                key = f"{ev.channel}:{ev.event_type.value}"
                freq_counter[key] = freq_counter.get(key, 0) + 1
            tracer.metric("frequency_entries", len(freq_counter))

        # Layer 2: Reliability scoring + confidence decay (§4.7)
        with tracer.stage("reliability"):
            reliability = self._scorer.score(events)
            reliability_map: Dict[str, float] = {}
            for ch, rel in reliability.items():
                channel_factor = rel.score * (1.0 - rel.loss_rate)
                reliability_map[ch] = channel_factor
            graph.metadata.setdefault("reliability", {}).update(reliability_map)
            # Apply reliability-based confidence decay to edges
            self._apply_reliability_decay(graph, events, reliability_map)
            tracer.metric("channels", len(reliability_map))

        # Layer 2: Hotspot detection
        with tracer.stage("hotspots"):
            hotspots = self._hotspot.detect(events)
            graph.metadata["hotspots"] = [h.to_dict() for h in hotspots]
            tracer.metric("hotspots", len(hotspots))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stats = {
            "events_emitted": len(events),
            "chains_built": len(chains),
            **inference_stats,
            "elapsed_ms": elapsed_ms,
        }

        if tracer.enabled:
            logger.debug(tracer.summary_line())
        else:
            logger.debug(
                "CausalFusion: %d events → %d chains in %.1fms "
                "(rules=%d, heuristic=%d)",
                len(events), len(chains), elapsed_ms,
                inference_stats.get("rule_edges", 0),
                inference_stats.get("heuristic_edges", 0),
            )

        return FusionOutput(
            graph=graph,
            chains=chains,
            stats=stats,
            elapsed_ms=elapsed_ms,
        )

    def describe_chains(self, chains: List[CausalChain]) -> List[str]:
        """Generate natural language descriptions for VLM prompts."""
        return [self._mapper.describe_chain(c) for c in chains]

    def _apply_reliability_decay(
        self,
        graph: CausalGraph,
        events: List[CausalEvent],
        reliability_map: Dict[str, float],
    ) -> None:
        """Apply reliability-based confidence decay to inferred edges (§4.7).

        Formula: edge.confidence *= channel_factor[parent] * channel_factor[child]
        Only applied to edges involving events from the current batch.
        """
        event_ids = {ev.id for ev in events}
        edges = list(graph.iter_edges())
        for parent_id, child_id, confidence in edges:
            if parent_id not in event_ids and child_id not in event_ids:
                continue
            parent = graph.events.get(parent_id)
            child = graph.events.get(child_id)
            if not parent or not child:
                continue
            p_factor = reliability_map.get(parent.channel, 1.0)
            c_factor = reliability_map.get(child.channel, 1.0)
            decayed = confidence * p_factor * c_factor
            if decayed != confidence:
                graph.update_edge_confidence(parent_id, child_id, decayed)

    async def run_vlm_verification(
        self,
        graph: CausalGraph,
        *,
        vlm_call: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute async VLM verification on pending low-confidence edges.

        Args:
            graph: The causal graph containing edges to verify.
            vlm_call: Async callable(prompt: str) -> str for VLM queries.
                      If None and self._vlm_verifier is None, returns immediately.

        Returns:
            Stats dict with verified/pruned/failed counts.
        """
        if self._vlm_verifier is None or vlm_call is None:
            return {"vlm_verified": 0, "vlm_pruned": 0}

        pending = self._vlm_verifier.collect_pending(graph)
        if not pending:
            return {"vlm_verified": 0, "vlm_pruned": 0}

        batches = self._vlm_verifier.get_batches()
        results: List[tuple] = []
        verified = 0
        pruned = 0
        failed = 0

        for batch in batches:
            for pv in batch:
                parent = graph.events.get(pv.parent_id)
                child = graph.events.get(pv.child_id)
                if not parent or not child:
                    continue

                prompt = (
                    f"Is there a causal relationship between these two events?\n"
                    f"Event A [{parent.channel}]: {parent.event_type.value} "
                    f"at {parent.timestamp:.3f}\n"
                    f"Event B [{child.channel}]: {child.event_type.value} "
                    f"at {child.timestamp:.3f}\n"
                    f"Current confidence: {pv.confidence:.2f}\n"
                    f"Answer: YES (with confidence 0.0-1.0) or NO"
                )
                try:
                    response = await vlm_call(prompt)
                    is_causal, confidence = _parse_vlm_response(
                        response, pv.confidence,
                    )
                    results.append((
                        pv.parent_id, pv.child_id, is_causal, confidence,
                    ))
                    if is_causal:
                        verified += 1
                    else:
                        pruned += 1
                except Exception:
                    failed += 1
                    logger.debug(
                        "VLM verification failed for edge %s→%s",
                        pv.parent_id[:8], pv.child_id[:8],
                        exc_info=True,
                    )

        if results:
            self._vlm_verifier.apply_results(
                graph, results, heuristic=self._inference.heuristic,
            )

        stats = {
            "vlm_verified": verified,
            "vlm_pruned": pruned,
            "vlm_failed": failed,
            "vlm_total_pending": len(pending),
        }
        logger.info(
            "VLM Tier3: %d verified, %d pruned, %d failed of %d pending",
            verified, pruned, failed, len(pending),
        )
        return stats

    @property
    def registry(self) -> ChannelRegistry:
        return self._registry

    @property
    def inference(self) -> CausalInferenceEngine:
        return self._inference


def _parse_vlm_response(
    response: str, fallback_confidence: float
) -> tuple:
    """Parse VLM yes/no + confidence from free-text response.

    Returns (is_causal: bool, confidence: float).
    """
    text = response.strip().lower()
    is_causal = not text.startswith("no")

    import re as _re
    match = _re.search(r"(\d+\.?\d*)", text)
    if match:
        confidence = float(match.group(1))
        if confidence > 1.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))
    else:
        confidence = 0.9 if is_causal else 0.1

    if not is_causal:
        confidence = min(confidence, fallback_confidence)

    return is_causal, confidence

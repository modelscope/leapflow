"""Layer 2 front-end components — lightweight processors that transform
raw CausalEvents into structured CausalChains.

Five components, all reading from ChannelRegistry (no hardcoding):
    1. EventDenoiser      — suppress jitter/burst noise
    2. CausalChainBuilder — assemble trigger→responses→effects chains
    3. ReliabilityScorer  — per-channel real-time reliability
    4. AttentionHotspotDetector — spatial click concentration
    5. SignalToSemanticMapper   — natural language anchors for VLM prompts
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from leapflow.causal.channel import AggregationPolicy, ChannelRegistry
from leapflow.causal.types import (
    CausalChain,
    CausalEvent,
    EventSource,
    EventType,
    FrameRef,
)

logger = logging.getLogger(__name__)


# ── 1. EventDenoiser ──


class EventDenoiser:
    """Suppress jitter and burst noise before chain construction.

    Strategies (all table-driven via ChannelRegistry):
        - Temporal aggregation: merge same-channel events within window
        - Burst suppression: drop > N events/second for navigation channels
        - Duplicate filtering: identical payload within 50ms
    """

    __slots__ = ("_registry", "_burst_limit")

    def __init__(self, registry: ChannelRegistry, burst_limit: Optional[int] = None) -> None:
        self._registry = registry
        if burst_limit is None:
            from leapflow.config import get_settings
            burst_limit = get_settings().causal_burst_limit
        self._burst_limit = burst_limit

    def denoise(self, events: List[CausalEvent]) -> List[CausalEvent]:
        if not events:
            return []
        input_count = len(events)
        events = self._deduplicate(events)
        events = self._aggregate(events)
        events = self._suppress_burst(events)
        if logger.isEnabledFor(logging.DEBUG):
            output_count = len(events)
            dropped = input_count - output_count
            drop_rate = (dropped / input_count) if input_count else 0.0
            logger.debug(
                "[denoiser] input=%d output=%d dropped=%d drop_rate=%.1f%%",
                input_count, output_count, dropped, drop_rate * 100,
            )
        return events

    def _deduplicate(self, events: List[CausalEvent]) -> List[CausalEvent]:
        result: List[CausalEvent] = []
        for ev in events:
            if result and self._is_duplicate(result[-1], ev):
                continue
            result.append(ev)
        return result

    def _is_duplicate(self, a: CausalEvent, b: CausalEvent) -> bool:
        if a.channel != b.channel:
            return False
        if abs(b.timestamp - a.timestamp) > 0.05:
            return False
        return a.payload == b.payload

    def _aggregate(self, events: List[CausalEvent]) -> List[CausalEvent]:
        result: List[CausalEvent] = []
        i = 0
        while i < len(events):
            ev = events[i]
            policy = self._registry.aggregation_policy(ev.channel)
            window = self._registry.aggregation_window(ev.channel)

            if policy == AggregationPolicy.NONE or window <= 0:
                result.append(ev)
                i += 1
                continue

            # Collect events in the aggregation window
            group = [ev]
            j = i + 1
            while j < len(events) and events[j].channel == ev.channel:
                if events[j].timestamp - ev.timestamp > window:
                    break
                if policy == AggregationPolicy.DIRECTIONAL:
                    if not self._same_direction(ev, events[j]):
                        break
                group.append(events[j])
                j += 1

            if len(group) > 1:
                result.append(self._merge_group(group, policy))
            else:
                result.append(ev)
            i = j

        return result

    def _same_direction(self, a: CausalEvent, b: CausalEvent) -> bool:
        da = a.payload.get("direction", a.payload.get("delta_y", 0))
        db = b.payload.get("direction", b.payload.get("delta_y", 0))
        if isinstance(da, (int, float)) and isinstance(db, (int, float)):
            return (da >= 0) == (db >= 0)
        return da == db

    def _merge_group(self, group: List[CausalEvent], policy: AggregationPolicy) -> CausalEvent:
        first = group[0]
        merged_payload = dict(first.payload)
        if policy == AggregationPolicy.DIRECTIONAL:
            total_delta = sum(e.payload.get("delta_y", 0) for e in group)
            merged_payload["delta_y"] = total_delta
            merged_payload["aggregated_count"] = len(group)
        elif policy == AggregationPolicy.COMBO:
            keys = [e.payload.get("key", "") for e in group]
            merged_payload["combo"] = "+".join(k for k in keys if k)
            merged_payload["aggregated_count"] = len(group)
        return CausalEvent(
            id=first.id,
            timestamp=first.timestamp,
            event_type=first.event_type,
            source=first.source,
            channel=first.channel,
            payload=merged_payload,
            confidence=first.confidence,
            frame_refs=first.frame_refs,
            tags=first.tags,
        )

    def _suppress_burst(self, events: List[CausalEvent]) -> List[CausalEvent]:
        """Priority-aware burst suppression.

        Trigger/Boundary/Effect always pass through (§4.5 scene 4).
        Only Navigation/Response channels are subject to rate limiting.
        """
        if not events:
            return []
        _PROTECTED_ROLES = frozenset({EventType.TRIGGER, EventType.BOUNDARY, EventType.EFFECT})
        window_counts: Dict[str, List[float]] = defaultdict(list)
        result: List[CausalEvent] = []
        for ev in events:
            if ev.event_type in _PROTECTED_ROLES:
                result.append(ev)
                continue
            ts_list = window_counts[ev.channel]
            while ts_list and ev.timestamp - ts_list[0] > 1.0:
                ts_list.pop(0)
            if len(ts_list) >= self._burst_limit:
                ev.tags["burst_suppressed"] = True
                continue
            ts_list.append(ev.timestamp)
            result.append(ev)
        return result


# ── 2. CausalChainBuilder ──


class CausalChainBuilder:
    """Assemble events into trigger→responses→effects chains.

    Algorithm: O(n) single-pass scan with a sliding causal window.
    Handles co-derivation (§4.4): when a new TRIGGER arrives within
    a short window of the current chain's trigger AND has high semantic
    affinity, it's absorbed as RESPONSE rather than starting a new chain.
    """

    __slots__ = (
        "_registry", "_causal_window_s", "_max_chain_events",
        "_co_derivation_window_s", "_app_window_overrides",
    )

    def __init__(
        self,
        registry: ChannelRegistry,
        causal_window_s: Optional[float] = None,
        max_chain_events: Optional[int] = None,
        co_derivation_window_s: float = 0.2,
        app_window_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._registry = registry
        if causal_window_s is None or max_chain_events is None or app_window_overrides is None:
            from leapflow.config import get_settings
            s = get_settings()
            if causal_window_s is None:
                causal_window_s = s.causal_window_s
            if max_chain_events is None:
                max_chain_events = s.causal_max_chain_events
            if app_window_overrides is None:
                app_window_overrides = s.causal_app_window_overrides
        self._causal_window_s = causal_window_s
        self._max_chain_events = max_chain_events
        self._co_derivation_window_s = co_derivation_window_s
        self._app_window_overrides = dict(app_window_overrides)

    def _resolve_window(self, trigger: CausalEvent) -> float:
        """Resolve causal window for a trigger based on app context.

        Falls back to the global default when no app-specific override exists.
        """
        if self._app_window_overrides:
            app = (
                trigger.payload.get("app_bundle_id")
                or trigger.payload.get("bundle_id")
                or trigger.payload.get("to", "")
            )
            if app:
                for pattern, window in self._app_window_overrides.items():
                    if pattern in app:
                        return window
        return self._causal_window_s

    def build(self, events: List[CausalEvent]) -> List[CausalChain]:
        if not events:
            return []

        chains: List[CausalChain] = []
        current_chain: Optional[_ChainAccumulator] = None
        recent: List[CausalEvent] = []
        overflow_count = 0

        for i, ev in enumerate(events):
            role = self._registry.resolve_role(ev, recent[-10:])
            ev.event_type = role

            if role == EventType.TRIGGER:
                if current_chain and self._is_co_derived(current_chain.trigger, ev):
                    # Co-derivation: absorb as RESPONSE (§4.4 scenario 1)
                    ev.event_type = EventType.RESPONSE
                    current_chain.add(ev, EventType.RESPONSE)
                else:
                    if current_chain:
                        chains.append(current_chain.finalize("next_trigger"))
                    current_chain = _ChainAccumulator(ev)

            elif role == EventType.BOUNDARY:
                if current_chain:
                    chains.append(current_chain.finalize("boundary"))
                current_chain = None
                boundary_chain = _ChainAccumulator(ev)
                chains.append(boundary_chain.finalize("boundary"))

            elif current_chain:
                if ev.timestamp - current_chain.trigger.timestamp > self._resolve_window(current_chain.trigger):
                    chains.append(current_chain.finalize("timeout"))
                    current_chain = None
                elif current_chain.event_count >= self._max_chain_events:
                    overflow_event_count = current_chain.event_count
                    overflow_chain = current_chain.finalize("overflow")
                    overflow_count += 1
                    logger.warning(
                        "CausalChain overflow: chain_id=%s event_count=%d max=%d — "
                        "chain truncated and subsequent events will start a new chain",
                        overflow_chain.id,
                        overflow_event_count,
                        self._max_chain_events,
                    )
                    chains.append(overflow_chain)
                    current_chain = None
                else:
                    current_chain.add(ev, role)

            elif role == EventType.NOISE:
                pass
            else:
                orphan = _ChainAccumulator(ev)
                chains.append(orphan.finalize("orphan"))

            recent.append(ev)

        if current_chain:
            chains.append(current_chain.finalize("timeout"))

        if logger.isEnabledFor(logging.DEBUG):
            event_count = len(events)
            chain_count = len(chains)
            total_chain_events = sum(
                1 + len(c.responses) + len(c.effects) for c in chains
            )
            avg_len = (total_chain_events / chain_count) if chain_count else 0.0
            logger.debug(
                "[chain_builder] events=%d chains=%d overflow=%d avg_chain_len=%.1f",
                event_count, chain_count, overflow_count, avg_len,
            )

        return chains

    def _is_co_derived(self, trigger: CausalEvent, candidate: CausalEvent) -> bool:
        """Detect co-derivation: same user action spawning multiple channel events.

        Example: Cmd+Tab produces keyboard + app_switch + visual_change within 200ms.
        The semantic prior between channels determines affinity.
        """
        dt = candidate.timestamp - trigger.timestamp
        if dt < 0 or dt > self._co_derivation_window_s:
            return False
        prior = self._registry.get_semantic_prior(trigger.channel, candidate.channel)
        return prior >= 0.8


_INTERRUPTION_PENALTY: Dict[str, float] = {
    "timeout": 0.85,
    "overflow": 0.80,
    "orphan": 0.70,
}


@dataclass
class _ChainAccumulator:
    trigger: CausalEvent
    responses: List[CausalEvent] = field(default_factory=list)
    effects: List[CausalEvent] = field(default_factory=list)

    @property
    def event_count(self) -> int:
        return 1 + len(self.responses) + len(self.effects)

    def add(self, ev: CausalEvent, role: EventType) -> None:
        ev.caused_by = self.trigger.id
        self.trigger.causes.append(ev.id)
        if role == EventType.EFFECT:
            self.effects.append(ev)
        else:
            self.responses.append(ev)

    def finalize(self, closed_by: str) -> CausalChain:
        all_events = [self.trigger] + self.responses + self.effects
        timestamps = [e.timestamp for e in all_events]
        return CausalChain(
            id=CausalEvent.make_id(),
            trigger=self.trigger,
            responses=list(self.responses),
            effects=list(self.effects),
            time_span=(min(timestamps), max(timestamps)),
            closed_by=closed_by,
            completeness=self._estimate_completeness(closed_by),
        )

    def _estimate_completeness(self, closed_by: str = "") -> float:
        has_response = len(self.responses) > 0
        has_effect = len(self.effects) > 0
        if has_response and has_effect:
            base = 1.0
        elif has_response or has_effect:
            base = 0.7
        else:
            base = 0.4
        penalty = _INTERRUPTION_PENALTY.get(closed_by, 1.0)
        return base * penalty


# ── 3. ReliabilityScorer ──


@dataclass
class ChannelReliability:
    channel: str
    score: float
    event_rate: float
    loss_rate: float


class ReliabilityScorer:
    """Per-channel real-time reliability based on sliding window stats."""

    __slots__ = ("_window_s", "_channel_stats")

    def __init__(self, window_s: float = 60.0) -> None:
        self._window_s = window_s
        self._channel_stats: Dict[str, List[float]] = defaultdict(list)

    def score(self, events: List[CausalEvent], window: float = 10.0) -> Dict[str, ChannelReliability]:
        if not events:
            return {}

        t_now = events[-1].timestamp if events else 0.0
        t_start = t_now - window
        channel_events: Dict[str, List[float]] = defaultdict(list)
        for ev in events:
            if ev.timestamp >= t_start:
                channel_events[ev.channel].append(ev.timestamp)

        result: Dict[str, ChannelReliability] = {}
        for channel, timestamps in channel_events.items():
            rate = len(timestamps) / window if window > 0 else 0.0
            # Estimate loss: gaps > 2× median interval
            loss_rate = 0.0
            if len(timestamps) > 2:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                median_interval = sorted(intervals)[len(intervals)//2]
                if median_interval > 0:
                    gaps = sum(1 for iv in intervals if iv > 2 * median_interval)
                    loss_rate = gaps / len(intervals)

            reliability = max(0.1, 1.0 - loss_rate)
            result[channel] = ChannelReliability(
                channel=channel,
                score=reliability,
                event_rate=rate,
                loss_rate=loss_rate,
            )
        return result


# ── 4. AttentionHotspotDetector ──


@dataclass
class Hotspot:
    x: float
    y: float
    radius: float
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "radius": self.radius, "weight": self.weight}


class AttentionHotspotDetector:
    """Detect spatial concentration of interaction events."""

    __slots__ = ("_radius", "_min_cluster")

    def __init__(self, radius: float = 100.0, min_cluster: int = 2) -> None:
        self._radius = radius
        self._min_cluster = min_cluster

    def detect(self, events: List[CausalEvent]) -> List[Hotspot]:
        points = [(ev.payload["x"], ev.payload["y"])
                  for ev in events
                  if "x" in ev.payload and "y" in ev.payload]
        if len(points) < self._min_cluster:
            return []

        # Simple grid-based clustering
        clusters: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        cell_size = self._radius
        for x, y in points:
            key = (int(x // cell_size), int(y // cell_size))
            clusters[key].append((x, y))

        hotspots: List[Hotspot] = []
        for pts in clusters.values():
            if len(pts) < self._min_cluster:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            max_dist = max(math.hypot(p[0]-cx, p[1]-cy) for p in pts)
            hotspots.append(Hotspot(
                x=cx, y=cy,
                radius=max(max_dist, self._radius / 2),
                weight=float(len(pts)),
            ))

        hotspots.sort(key=lambda h: h.weight, reverse=True)
        return hotspots[:5]


# ── 5. SignalToSemanticMapper ──


class SignalToSemanticMapper:
    """Map CausalChains to natural language descriptions for VLM prompts.

    Token budget management (§4.5 scene 5):
        Priority order: Trigger (always) > Boundary (always) > Effect (summary)
        > Navigation (aggregated) > Response (top-K by confidence).
        Single chain token limit: 800 chars (~200 tokens).
    """

    __slots__ = ("_registry", "_chain_char_budget")

    def __init__(self, registry: ChannelRegistry, chain_char_budget: int = 800) -> None:
        self._registry = registry
        self._chain_char_budget = chain_char_budget

    def describe_chain(self, chain: CausalChain) -> str:
        parts: List[str] = []
        parts.append(self._describe_event(chain.trigger, "Trigger"))
        for resp in chain.responses[:3]:
            parts.append(self._describe_event(resp, "Response"))
        for eff in chain.effects[:2]:
            parts.append(self._describe_event(eff, "Effect"))
        return " → ".join(parts)

    def describe_chain_budgeted(self, chain: CausalChain) -> str:
        """Describe chain with token budget enforcement (§4.5 scene 5)."""
        # Priority: Trigger > Boundary > Effect > Navigation > Response
        parts: List[str] = []
        budget = self._chain_char_budget

        # Trigger: always included
        trigger_desc = self._describe_event(chain.trigger, "Trigger")
        parts.append(trigger_desc)
        budget -= len(trigger_desc) + 4

        # Effects: include with summary truncation
        for eff in chain.effects:
            if budget <= 0:
                break
            desc = self._describe_event(eff, "Effect")
            if len(desc) > 200:
                desc = desc[:197] + "..."
            parts.append(desc)
            budget -= len(desc) + 4

        # Responses: sorted by confidence, top-K
        responses_sorted = sorted(chain.responses, key=lambda e: e.confidence, reverse=True)
        nav_count = 0
        resp_count = 0
        for resp in responses_sorted:
            if budget <= 40:
                break
            if resp.event_type == EventType.NAVIGATION:
                nav_count += 1
                if nav_count > 1:
                    continue  # Only 1 nav summary
            resp_count += 1
            desc = self._describe_event(resp, "Response")
            parts.append(desc)
            budget -= len(desc) + 4

        trimmed = len(chain.responses) + len(chain.effects) - (resp_count + len([e for e in chain.effects if budget > 0]))
        if trimmed > 0:
            parts.append(f"[...{trimmed} events trimmed]")

        return " → ".join(parts)

    def describe_events(self, events: List[CausalEvent], max_events: int = 8) -> str:
        descriptions = [self._describe_event(ev) for ev in events[:max_events]]
        if len(events) > max_events:
            descriptions.append(f"[...{len(events) - max_events} more]")
        return "; ".join(descriptions)

    def _describe_event(self, ev: CausalEvent, prefix: str = "") -> str:
        label = prefix + ": " if prefix else ""
        desc = self._registry.describe_event(ev.channel, ev.payload)
        return f"{label}{desc}"

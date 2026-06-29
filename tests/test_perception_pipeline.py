"""Scenario-based tests for the perception pipeline (causal, signal fusion).

Covers causal chain construction, graph operations, heuristic priors,
channel registry, MHMS-SF fusion, denoising, and reliability scoring.
"""

from __future__ import annotations

import pytest

from conftest import make_event

from leapflow.causal.channel import (
    AggregationPolicy,
    ChannelRegistry,
    ChannelSpec,
    build_default_registry,
)
from leapflow.causal.components import (
    CausalChainBuilder,
    EventDenoiser,
    ReliabilityScorer,
)
from leapflow.causal.inference import HeuristicEngine
from leapflow.causal.types import (
    CausalEvent,
    CausalGraph,
    EventSource,
    EventType,
)
from leapflow.perception.types import ChannelStatus, Keyframe, VisualAction
from leapflow.signal_fusion.pipeline import MHMSFusionPipeline
from leapflow.signal_fusion.protocol import FusionContext


def _causal_event(
    channel: str,
    ts: float,
    *,
    event_type: EventType = EventType.TRIGGER,
    payload: dict | None = None,
    confidence: float = 1.0,
) -> CausalEvent:
    return CausalEvent(
        id=CausalEvent.make_id(),
        timestamp=ts,
        event_type=event_type,
        source=EventSource.SIGNAL,
        channel=channel,
        payload=payload or {},
        confidence=confidence,
    )


def test_causal_chain_detection() -> None:
    """Trigger and response within the causal window form a single chain."""
    registry = build_default_registry()
    builder = CausalChainBuilder(registry, causal_window_s=3.0)
    events = [
        _causal_event("click", 1.0, event_type=EventType.TRIGGER, payload={"x": 10, "y": 20}),
        _causal_event("visual_change", 1.2, event_type=EventType.RESPONSE),
    ]

    chains = builder.build(events)

    assert len(chains) == 1
    chain = chains[0]
    assert chain.trigger.channel == "click"
    assert len(chain.responses) == 1
    assert chain.responses[0].channel == "visual_change"
    assert chain.trigger.causes == [chain.responses[0].id]
    assert chain.responses[0].caused_by == chain.trigger.id


def test_causal_chain_outside_window() -> None:
    """Events too far apart are not linked in the same causal chain."""
    registry = build_default_registry()
    builder = CausalChainBuilder(registry, causal_window_s=2.0)
    events = [
        _causal_event("click", 1.0, event_type=EventType.TRIGGER),
        _causal_event("visual_change", 5.0, event_type=EventType.RESPONSE),
    ]

    chains = builder.build(events)

    trigger_chain = next(c for c in chains if c.trigger.channel == "click")
    assert len(trigger_chain.responses) == 0
    assert trigger_chain.closed_by == "timeout"
    assert not any(
        c.trigger.channel == "visual_change" and c.responses
        for c in chains
    )


def test_causal_graph_edge_operations() -> None:
    """Graph supports add/remove edges and connectivity queries."""
    graph = CausalGraph()
    trigger = _causal_event("click", 1.0, event_type=EventType.TRIGGER)
    response = _causal_event("visual_change", 1.2, event_type=EventType.RESPONSE)
    isolated = _causal_event("click", 10.0, event_type=EventType.TRIGGER)

    for ev in (trigger, response, isolated):
        graph.add_event(ev)

    graph.add_edge(trigger.id, response.id)
    assert response.caused_by == trigger.id
    assert response.id in trigger.causes

    component = graph.get_connected_component(trigger.id)
    assert trigger.id in component
    assert response.id in component
    assert isolated.id not in component

    ordered = graph.topological_order()
    assert ordered.index(trigger) < ordered.index(response)

    graph.remove_edge(trigger.id, response.id)
    assert response.caused_by is None
    assert response.id not in trigger.causes
    assert graph.get_connected_component(trigger.id) == {trigger.id}


def test_heuristic_engine_prior_updates() -> None:
    """EMA prior updates increase causal scores for channel pairs."""
    registry = build_default_registry()
    heuristic = HeuristicEngine(registry)
    parent = _causal_event("click", 1.0, event_type=EventType.TRIGGER, payload={"x": 0, "y": 0})
    child = _causal_event("app_switch", 1.1, event_type=EventType.RESPONSE)

    baseline = heuristic.causal_score(parent, child)
    assert baseline < 0.5  # default prior for click→app_switch is 0.4

    heuristic.update_prior("click", "app_switch", 0.95, alpha=0.5)
    updated = heuristic.causal_score(parent, child)

    assert updated > baseline
    assert heuristic._semantic_prior[("click", "app_switch")] > 0.6


def test_channel_registry_default_channels() -> None:
    """Default registry includes the standard interaction channels."""
    registry = build_default_registry()

    for channel in ("click", "keyboard", "drag", "app_switch", "scroll", "clipboard", "visual_change"):
        spec = registry.get_spec(channel)
        assert spec is not None, f"missing channel: {channel}"
        assert registry.is_available(channel)

    click = registry.get_spec("click")
    assert click is not None
    assert click.default_role == EventType.TRIGGER
    assert click.reactive_capture is True
    assert registry.get_semantic_prior("click", "visual_change") == pytest.approx(0.95)


def test_channel_registry_custom_registration() -> None:
    """Custom channels can be registered and retrieved."""
    registry = ChannelRegistry()
    registry.register(ChannelSpec(
        channel="voice_command",
        default_role=EventType.TRIGGER,
        description_template="Voice: {utterance}",
        aggregation=AggregationPolicy.NONE,
    ))

    spec = registry.get_spec("voice_command")
    assert spec is not None
    assert spec.channel == "voice_command"
    assert spec.default_role == EventType.TRIGGER
    assert registry.describe_event("voice_command", {"utterance": "open finder"}) == "Voice: open finder"


@pytest.mark.asyncio
async def test_signal_fusion_end_to_end() -> None:
    """MHMS-SF pipeline fuses aligned visual and system events into atomic actions."""
    pipeline = MHMSFusionPipeline.default()
    ctx = FusionContext(
        visual_actions=[
            VisualAction(
                action="click",
                target="button",
                confidence=0.8,
                evidence="clicked submit button",
                frame_ref_a="f1",
                frame_ref_b="f2",
            ),
        ],
        system_events=[
            make_event("ui.click", payload={"label": "btn"}, ts=1.0),
        ],
        keyframes=[
            Keyframe(ref="f1", timestamp=1.0, image=b""),
            Keyframe(ref="f2", timestamp=1.1, image=b""),
        ],
        channel_status=ChannelStatus(),
    )

    result = await pipeline.fuse(ctx)

    assert len(result.atomic_actions) >= 1
    action = result.atomic_actions[0]
    assert action.action == "click"
    assert action.target == "button"
    assert action.confidence > 0.0
    assert "visual" in action.source_signals


@pytest.mark.asyncio
async def test_fusion_empty_context() -> None:
    """Empty fusion context returns empty results without error."""
    pipeline = MHMSFusionPipeline.default()
    result = await pipeline.fuse(FusionContext())

    assert result.atomic_actions == []
    assert result.segments == []
    assert result.episodes == []


def test_event_denoiser_filters_low_confidence() -> None:
    """Denoiser suppresses burst noise on navigation channels beyond rate limit."""
    registry = build_default_registry()
    denoiser = EventDenoiser(registry, burst_limit=3)
    events = [
        _causal_event("scroll", 1.0 + i * 0.05, event_type=EventType.NAVIGATION, payload={"delta_y": 5})
        for i in range(8)
    ]

    result = denoiser.denoise(events)

    assert len(result) < len(events)
    assert len(result) <= 3


def test_reliability_scorer() -> None:
    """Reliability scorer assigns channel scores from event timing patterns."""
    scorer = ReliabilityScorer()
    regular = [
        _causal_event("click", 1.0),
        _causal_event("click", 2.0),
        _causal_event("click", 3.0),
    ]
    gappy = [
        _causal_event("scroll", 1.0, event_type=EventType.NAVIGATION),
        _causal_event("scroll", 2.0, event_type=EventType.NAVIGATION),
        _causal_event("scroll", 3.0, event_type=EventType.NAVIGATION),
        _causal_event("scroll", 12.0, event_type=EventType.NAVIGATION),
    ]

    regular_scores = scorer.score(regular, window=5.0)
    gappy_scores = scorer.score(gappy, window=15.0)

    assert regular_scores["click"].score == pytest.approx(1.0)
    assert regular_scores["click"].event_rate == pytest.approx(0.6)
    assert gappy_scores["scroll"].loss_rate > 0.0
    assert gappy_scores["scroll"].score < regular_scores["click"].score

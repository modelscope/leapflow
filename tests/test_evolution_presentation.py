# Copyright (c) Alibaba, Inc. and its affiliates.
"""Presentation-only projections for live framework-evolution displays."""

from __future__ import annotations

from leapflow.domain.evolution_trace import EvolutionStage, EvolutionTrace
from leapflow.telemetry.evolution_presentation import EvolutionPresentationEvent


def test_presentation_event_uses_existing_correlation_as_episode_identity() -> None:
    trace = EvolutionTrace(
        stage=EvolutionStage.DECIDE,
        kind="policy_decision",
        trace_id="trace-1",
        ts=42.0,
        correlation={"record_id": "plan-9", "plugin_id": "example", "secret": "not-a-reference"},
        summary="The policy selected reuse.",
        detail={
            "evidence_level": "L1_controlled",
            "verification_tier": "declared_fitness",
            "side_effect_state": "none",
            "unbounded_private_detail": "must not reach the browser event",
        },
    )

    event = EvolutionPresentationEvent.from_trace(trace).to_dict()

    assert event["event_id"] == "trace-1"
    assert event["episode_id"] == "plan-9"
    assert event["payload_ref"] == {"record_id": "plan-9", "plugin_id": "example"}
    assert event["evidence_level"] == "L1_controlled"
    assert event["verification_tier"] == "declared_fitness"
    assert "unbounded_private_detail" not in event


def test_presentation_event_uses_trace_identity_when_no_correlation_exists() -> None:
    trace = EvolutionTrace(stage=EvolutionStage.OBSERVE, kind="interface_drift", trace_id="trace-2")

    event = EvolutionPresentationEvent.from_trace(trace)

    assert event.episode_id == "trace-2"
    assert event.payload_ref == {}
    assert event.evidence_level == "runtime_trace"
    assert event.verification_tier == "not_recorded"
    assert event.side_effect_state == "not_recorded"

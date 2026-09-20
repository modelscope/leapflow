# Copyright (c) Alibaba, Inc. and its affiliates.
"""Replay and checkpoint contracts for the evolution read model."""
from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.evolution.projection import EvolutionProjectionRunner
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore


def _event(
    event_type: str,
    *,
    session_id: str,
    action_id: str = "",
    correlation_id: str = "",
    payload: dict | None = None,
) -> EvolutionEvent:
    return EvolutionEvent.create(
        event_type,
        context=EvolutionContext.create(
            profile_id="profile-1",
            workspace_id=f"workspace-{session_id}",
            session_id=session_id,
            action_id=action_id,
            correlation_id=correlation_id,
        ),
        payload=payload or {},
        producer="test",
        dedup_key=f"{event_type}:{session_id}:{action_id}:{correlation_id}",
    )


@pytest.mark.asyncio
async def test_full_rebuild_equals_incremental_projection(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(
        (
            _event(EvolutionEventType.ACTION_STARTED, session_id="s1", action_id="a1"),
            _event(
                EvolutionEventType.ACTION_COMPLETED,
                session_id="s1",
                action_id="a1",
                payload={"ok": True},
            ),
        )
    )
    episode_id, _ = store.finalize_session(
        profile_id="profile-1",
        workspace_id="workspace-s1",
        session_id="s1",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    runner = EvolutionProjectionRunner(store)
    first = await runner.project_session(profile_id="profile-1", session_id="s1")

    store.append(
        _event(
            EvolutionEventType.TEACHER_VERDICT_RECORDED,
            session_id="s1",
            correlation_id=episode_id,
            payload={
                "verdict_id": "v1",
                "action": "absorb",
                "capability": "repo.inspect",
                "knowledge": "The existing reader is sufficient.",
                "confidence": 0.9,
            },
        )
    )
    incremental = await runner.project_session(profile_id="profile-1", session_id="s1")
    rebuilt = await runner.project_session(
        profile_id="profile-1",
        session_id="s1",
        rebuild=True,
    )

    assert first["summary"]["action_count"] == 1
    assert incremental == rebuilt
    assert rebuilt["summary"]["by_action"]["absorb"] == 1
    assert rebuilt["episodes"][0]["capability"] == "repo.inspect"
    assert runner.metrics.run_latency.count == 3
    assert runner.metrics.run_latency.p99_ms >= 0
    store.close()


@pytest.mark.asyncio
async def test_four_verdicts_materialize_their_distinct_read_models(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    for index, action in enumerate(("absorb", "rebind", "acquire", "escalate")):
        store.append(
            _event(
                EvolutionEventType.TEACHER_VERDICT_RECORDED,
                session_id="s1",
                correlation_id=f"episode-{index}",
                payload={
                    "verdict_id": f"v-{index}",
                    "action": action,
                    "capability": f"capability.{action}",
                    "knowledge": "retain this fact",
                    "rationale": "human decision required",
                    "confidence": 0.8,
                    "target": "existing_plugin",
                },
            )
        )

    projection = await EvolutionProjectionRunner(store).project_session(
        profile_id="profile-1", session_id="s1"
    )

    assert projection["knowledge"][0]["capability"] == "capability.absorb"
    assert projection["provider_bindings"][0]["plugin_id"] == "existing_plugin"
    assert projection["proposal_candidates"][0]["capability"] == "capability.acquire"
    assert projection["human_escalations"][0]["capability"] == "capability.escalate"
    store.close()


@pytest.mark.asyncio
async def test_resolution_no_op_and_knowledge_retraction_are_projected(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(
        (
            _event(
                EvolutionEventType.TEACHER_VERDICT_RECORDED,
                session_id="s1",
                correlation_id="episode-1",
                payload={
                    "verdict_id": "v-1",
                    "action": "absorb",
                    "capability": "repo.inspect",
                    "knowledge": "Use the repository reader.",
                    "confidence": 0.9,
                },
            ),
            _event(
                EvolutionEventType.REQUIREMENT_RESOLVED,
                session_id="s1",
                correlation_id="episode-1",
                payload={
                    "requirement": {"capability": "repo.inspect"},
                    "outcome": "satisfied",
                    "reason": "capability_already_available",
                    "resolution": {"selected_plugin_id": "repo_builtin"},
                },
            ),
            _event(
                EvolutionEventType.KNOWLEDGE_RETRACTED,
                session_id="s1",
                correlation_id="episode-1",
                payload={"capability": "repo.inspect", "reason": "observed working"},
            ),
        )
    )

    projection = await EvolutionProjectionRunner(store).project_session(
        profile_id="profile-1", session_id="s1"
    )

    assert projection["knowledge"] == []
    assert projection["summary"]["no_op_count"] == 1
    assert projection["resolutions"][0]["outcome"] == "satisfied"
    assert projection["resolutions"][0]["selected_plugin_id"] == "repo_builtin"
    store.close()


@pytest.mark.asyncio
async def test_session_projection_never_leaks_another_session(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(
        (
            _event(EvolutionEventType.ACTION_STARTED, session_id="s1", action_id="a1"),
            _event(EvolutionEventType.ACTION_STARTED, session_id="s2", action_id="a2"),
            _event(
                EvolutionEventType.ENVIRONMENT_OBSERVED,
                session_id="s2",
                payload={"kind": "delta", "app_id": "chat"},
            ),
        )
    )
    runner = EvolutionProjectionRunner(store)

    session = await runner.project_session(profile_id="profile-1", session_id="s1")
    aggregate = await runner.project_aggregate(profile_id="profile-1")

    assert session["scope"] == "session"
    assert session["session_id"] == "s1"
    assert session["summary"]["action_count"] == 1
    assert session["summary"]["environment_count"] == 0
    assert aggregate["scope"] == "aggregate"
    assert aggregate["summary"]["action_count"] == 2
    assert aggregate["summary"]["environment_count"] == 1
    store.close()


@pytest.mark.asyncio
async def test_projection_checkpoint_survives_runner_recreation(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append(_event(EvolutionEventType.ACTION_STARTED, session_id="s1", action_id="a1"))
    first = await EvolutionProjectionRunner(store).project_aggregate(profile_id="profile-1")
    store.append(_event(EvolutionEventType.ACTION_STARTED, session_id="s1", action_id="a2"))

    resumed = await EvolutionProjectionRunner(store).project_aggregate(profile_id="profile-1")

    assert first["summary"]["event_count"] == 1
    assert resumed["summary"]["event_count"] == 2
    assert resumed["summary"]["action_count"] == 2
    store.close()

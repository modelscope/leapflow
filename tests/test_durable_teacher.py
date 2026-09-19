# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durability and isolation contracts for session-finalized teacher work."""
from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent
from leapflow.evolution.artifact_store import ContentAddressedArtifactStore
from leapflow.evolution.outbox import EvolutionEventOutbox
from leapflow.evolution.session_finalizer import SessionFinalizer
from leapflow.evolution.teacher_worker import DurableTeacherWorker
from leapflow.storage.capability_proposal_queue import EvolutionCapabilityProposalStore
from leapflow.storage.distilled_knowledge_store import EvolutionDistilledKnowledgeStore
from leapflow.storage.evolution_event_store import DuckDBEvolutionEventStore
from leapflow.world_model.trajectory_grader import ActionGrade, TeacherVerdict


def _action_events(session_id: str, action_id: str, *, workspace_id: str = "ws-1"):
    context = EvolutionContext.create(
        profile_id="profile-1",
        workspace_id=workspace_id,
        session_id=session_id,
        action_id=action_id,
        turn_id="turn-1",
        frame_id="frame-1",
    )
    started = EvolutionEvent.create(
        EvolutionEventType.ACTION_STARTED,
        context=context,
        payload={
            "action_type": "tool",
            "action_name": "file_read",
            "goal": "inspect the repository",
        },
        producer="test",
        dedup_key=f"action.started:{action_id}",
    )
    completed = EvolutionEvent.create(
        EvolutionEventType.ACTION_COMPLETED,
        context=context.with_ids(causation_id=started.event_id),
        payload={"ok": True, "result": {"files": 2}},
        producer="test",
        dedup_key=f"action.completed:{action_id}",
    )
    return started, completed


class _Teacher:
    def __init__(
        self,
        error: BaseException | None = None,
        *,
        action: str = "absorb",
    ) -> None:
        self.error = error
        self.action = action
        self.calls: list[tuple[list[dict], str]] = []

    async def grade_and_propose(
        self,
        trajectory,
        goal="",
        *,
        degraded_capabilities=(),
        raise_on_error=False,
    ):
        self.calls.append((list(trajectory), str(goal)))
        if self.error is not None:
            raise self.error
        return TeacherVerdict(
            grades=(ActionGrade("", 0.8, False, "helpful"),),
            verdicts=(
                AdaptationVerdict.create(
                    self.action,
                    "repository.inspect",
                    "The environment requires repository inspection adaptation.",
                    confidence=0.9,
                ),
            ),
            raw_payload={"source": "teacher-test"},
        )


@pytest.mark.asyncio
async def test_finalizer_is_idempotent_and_session_scoped(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store, flush_interval_s=0.001)
    first = _action_events("session-a", "action-a")
    second = _action_events("session-b", "action-b", workspace_id="ws-2")
    await outbox.publish(first[0])
    await outbox.publish(first[1])
    await outbox.publish(second[0])
    await outbox.publish(second[1])
    finalizer = SessionFinalizer(store, outbox)

    result = await finalizer.finalize(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        reason="manual",
    )
    repeated = await finalizer.finalize(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        reason="manual",
    )

    assert result.queued is True
    assert result.evidence_count == 2
    assert repeated.queued is False
    job = store.teacher_job(result.job_id)
    assert job is not None
    assert job["session_id"] == "session-a"
    assert job["workspace_id"] == "ws-1"
    assert store.latest_evidence_sequence(
        profile_id="profile-1", session_id="session-b"
    ) > 0
    await outbox.close()
    store.close()


@pytest.mark.asyncio
async def test_teacher_worker_persists_one_call_result_and_verdict(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store, flush_interval_s=0.001)
    started, completed = _action_events("session-a", "action-a")
    await outbox.publish(started)
    await outbox.publish(completed)
    finalization = await SessionFinalizer(store, outbox).finalize(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
    )
    teacher = _Teacher()
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    knowledge = EvolutionDistilledKnowledgeStore(store, profile_id="profile-1")
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=artifacts,
        teacher=teacher,
        profile_id="profile-1",
        retry_backoff_s=0,
        knowledge_projection=knowledge,
    )

    outcome = await worker.run_once()

    assert outcome is not None and outcome.status == "COMPLETED"
    assert len(teacher.calls) == 1
    assert teacher.calls[0][1] == "inspect the repository"
    job = store.teacher_job(finalization.job_id)
    assert job is not None and job["status"] == "COMPLETED"
    assert artifacts.get_bytes(job["result_artifact_id"])
    event_types = [
        record.event.event_type
        for record in store.read(
            profile_id="profile-1",
            correlation_id=finalization.episode_id,
        )
    ]
    assert EvolutionEventType.TEACHER_GRADED in event_types
    assert EvolutionEventType.TEACHER_VERDICT_RECORDED in event_types
    assert knowledge.for_capability("repository.inspect") is not None
    assert worker.metrics.claim_latency.count == 1
    assert worker.metrics.job_latency.count == 1
    await outbox.close()
    store.close()


def test_expired_teacher_lease_is_reclaimed_and_stale_owner_cannot_complete(
    tmp_path: Path,
) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    episode_id, job_id = store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )

    first = store.claim_teacher_job(lease_owner="worker-1", lease_seconds=5, now=10)
    assert first is not None and first["episode_id"] == episode_id
    assert first["attempts"] == 1
    assert store.claim_teacher_job(lease_owner="worker-2", lease_seconds=5, now=14) is None
    second = store.claim_teacher_job(lease_owner="worker-2", lease_seconds=5, now=16)
    assert second is not None and second["attempts"] == 2
    assert store.complete_teacher_job(job_id, lease_owner="worker-1") is False
    assert store.complete_teacher_job(job_id, lease_owner="worker-2") is True
    store.close()


@pytest.mark.asyncio
async def test_internal_teacher_defect_is_not_retried(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    _, job_id = store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=_Teacher(TypeError("local bug")),
        profile_id="profile-1",
        retry_backoff_s=0,
    )

    outcome = await worker.run_once()

    assert outcome is not None and outcome.status == "FAILED_FINAL"
    assert store.teacher_job(job_id)["status"] == "FAILED_FINAL"
    failures = store.read(
        profile_id="profile-1",
        event_type=EvolutionEventType.TEACHER_JOB_FAILED,
    )
    assert len(failures) == 1
    assert failures[0].event.payload["error_type"] == "TypeError"
    store.close()


@pytest.mark.asyncio
async def test_background_worker_wakes_and_completes_queued_job(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    outbox = EvolutionEventOutbox(store, flush_interval_s=0.001)
    store.append_many(_action_events("session-a", "action-a"))
    finalization = await SessionFinalizer(store, outbox).finalize(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
    )
    teacher = _Teacher()
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=teacher,
        profile_id="profile-1",
        poll_interval_s=10,
    )
    worker.start()
    worker.wake()

    job = await worker.wait_for_job(finalization.job_id, timeout_s=2)

    assert job["status"] == "COMPLETED"
    assert len(teacher.calls) == 1
    await worker.close()
    await outbox.close()
    store.close()


@pytest.mark.asyncio
async def test_acquire_verdict_queues_one_visible_proposal(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    _, job_id = store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    queue = EvolutionCapabilityProposalStore(store, profile_id="profile-1")
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=_Teacher(action="acquire"),
        profile_id="profile-1",
        proposal_queue=queue,
        acquisition_resolver=lambda intent, job: {
            "resolved": True,
            "satisfied": False,
            "reason": "no eligible capability provider",
            "environment": {
                "workspace_id": job["workspace_id"],
                "session_id": job["session_id"],
            },
        },
    )

    outcome = await worker.run_once()

    assert outcome is not None and outcome.status == "COMPLETED"
    proposals = queue.list_items()
    assert len(proposals) == 1
    assert proposals[0].requirements[0]["capability"] == "repository.inspect"
    assert proposals[0].requirements[0]["requirement_id"] == "req-wm-repository.inspect"
    resolutions = store.read(
        profile_id="profile-1",
        event_type=EvolutionEventType.REQUIREMENT_RESOLVED,
    )
    assert len(resolutions) == 1
    assert resolutions[0].event.payload["outcome"] == "unmet"
    assert resolutions[0].event.context.proposal_id == proposals[0].proposal_id
    job = store.teacher_job(job_id)
    assert job is not None and job["status"] == "COMPLETED"
    store.close()


@pytest.mark.asyncio
async def test_acquire_verdict_is_no_op_when_live_capability_resolves(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    queue = EvolutionCapabilityProposalStore(store, profile_id="profile-1")
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=_Teacher(action="acquire"),
        profile_id="profile-1",
        proposal_queue=queue,
        acquisition_resolver=lambda intent, job: {
            "resolved": True,
            "satisfied": True,
            "reason": "selected existing provider",
            "selected_plugin_id": "repository_builtin",
            "selected_tool_name": "file_read",
            "environment": {},
        },
    )

    outcome = await worker.run_once()

    assert outcome is not None and outcome.status == "COMPLETED"
    assert queue.list_items() == []
    resolutions = store.read(
        profile_id="profile-1",
        event_type=EvolutionEventType.REQUIREMENT_RESOLVED,
    )
    assert len(resolutions) == 1
    assert resolutions[0].event.payload["outcome"] == "satisfied"
    assert resolutions[0].event.payload["reason"] == "capability_already_available"
    store.close()


@pytest.mark.asyncio
async def test_acquire_verdict_fails_closed_without_live_resolution(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    queue = EvolutionCapabilityProposalStore(store, profile_id="profile-1")
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=_Teacher(action="acquire"),
        profile_id="profile-1",
        proposal_queue=queue,
    )

    await worker.run_once()

    assert queue.list_items() == []
    resolution = store.read(
        profile_id="profile-1",
        event_type=EvolutionEventType.REQUIREMENT_RESOLVED,
    )[0].event
    assert resolution.payload["outcome"] == "no_op"
    assert resolution.payload["reason"] == "live_resolution_unavailable"
    store.close()


@pytest.mark.asyncio
async def test_retryable_teacher_failure_exhausts_configured_attempts(tmp_path: Path) -> None:
    store = DuckDBEvolutionEventStore(tmp_path / "events.duckdb")
    store.append_many(_action_events("session-a", "action-a"))
    _, job_id = store.finalize_session(
        profile_id="profile-1",
        workspace_id="ws-1",
        session_id="session-a",
        session_generation=0,
        from_sequence=0,
        through_sequence=2,
        reason="test",
    )
    worker = DurableTeacherWorker(
        store=store,
        artifact_store=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        teacher=_Teacher(OSError("provider unavailable")),
        profile_id="profile-1",
        max_attempts=2,
        retry_backoff_s=0,
    )

    first = await worker.run_once()
    second = await worker.run_once()

    assert first is not None and first.status == "FAILED_RETRYABLE"
    assert second is not None and second.status == "FAILED_FINAL"
    assert store.teacher_job(job_id)["attempts"] == 2
    assert len(
        store.read(
            profile_id="profile-1",
            event_type=EvolutionEventType.TEACHER_JOB_FAILED,
        )
    ) == 2
    store.close()

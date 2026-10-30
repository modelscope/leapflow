# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable daemon worker for one-call hindsight grading."""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent, EvolutionEventRecord, content_hash
from leapflow.evolution.artifact_store import ArtifactRef
from leapflow.performance import LatencySummary, RollingLatency
from leapflow.security.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_INTERNAL_DEFECTS = (
    AttributeError,
    TypeError,
    NameError,
    KeyError,
    IndexError,
    ImportError,
    AssertionError,
    NotImplementedError,
)


@runtime_checkable
class Teacher(Protocol):
    """One-call hindsight evaluator used by the durable worker."""

    async def grade_and_propose(
        self,
        trajectory: list[dict[str, Any]],
        goal: str = "",
        *,
        degraded_capabilities: Sequence[Mapping[str, Any]] = (),
        raise_on_error: bool = False,
    ) -> Any: ...


@runtime_checkable
class TeacherWorkStore(Protocol):
    """Persistence operations required by the durable teacher worker."""

    def claim_teacher_job(self, **kwargs: Any) -> dict[str, Any] | None: ...

    def renew_teacher_job(self, job_id: str, **kwargs: Any) -> bool: ...

    def complete_teacher_job(self, job_id: str, **kwargs: Any) -> bool: ...

    def fail_teacher_job(self, job_id: str, error: str, **kwargs: Any) -> bool: ...

    def teacher_job(self, job_id: str) -> dict[str, Any] | None: ...

    def read(self, **kwargs: Any) -> list[EvolutionEventRecord]: ...


@runtime_checkable
class ArtifactWriter(Protocol):
    """Minimal CAS interface used for teacher outputs."""

    def put_json(
        self,
        value: Mapping[str, Any] | list[Any],
        *,
        privacy_class: str = "system",
    ) -> ArtifactRef: ...


@dataclass(frozen=True)
class TeacherWorkerMetrics:
    claim_latency: LatencySummary
    job_latency: LatencySummary


@dataclass(frozen=True)
class TeacherJobOutcome:
    """Observable result of one worker attempt."""

    job_id: str
    status: str
    episode_id: str = ""
    artifact_id: str = ""
    verdict_count: int = 0
    grade_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _AcquisitionPlan:
    verdict_id: str
    requirement: Any
    outcome: str
    reason: str
    resolution: Mapping[str, Any]
    proposal_id: str = ""
    proposal_event: EvolutionEvent | None = None


class DurableTeacherWorker:
    """Lease and grade finalized sessions without blocking daemon RPC handling."""

    def __init__(
        self,
        *,
        store: TeacherWorkStore,
        artifact_store: ArtifactWriter,
        teacher: Teacher,
        profile_id: str = "",
        producer_version: str = "",
        poll_interval_s: float = 1.0,
        lease_seconds: float = 120.0,
        teacher_timeout_s: float = 180.0,
        max_attempts: int = 3,
        retry_backoff_s: float = 5.0,
        proposal_queue: Any = None,
        acquisition_resolver: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        authorising_origins: Sequence[str] = (),
        degraded_capabilities: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        knowledge_projection: Any = None,
    ) -> None:
        self._store = store
        self._artifact_store = artifact_store
        self._teacher = teacher
        self._profile_id = str(profile_id)
        self._producer_version = str(producer_version)
        self._poll_interval_s = max(0.05, float(poll_interval_s))
        self._lease_seconds = max(3.0, float(lease_seconds))
        self._teacher_timeout_s = max(0.1, float(teacher_timeout_s))
        self._max_attempts = max(1, int(max_attempts))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._proposal_queue = proposal_queue
        self._acquisition_resolver = acquisition_resolver
        self._authorising_origins = tuple(
            str(origin) for origin in authorising_origins if str(origin)
        )
        self._degraded_capabilities = degraded_capabilities
        self._knowledge_projection = knowledge_projection
        self._lease_owner = f"teacher-{uuid.uuid4().hex}"
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._claim_latency = RollingLatency()
        self._job_latency = RollingLatency()

    @property
    def metrics(self) -> TeacherWorkerMetrics:
        return TeacherWorkerMetrics(
            claim_latency=self._claim_latency.snapshot(),
            job_latency=self._job_latency.snapshot(),
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("teacher worker is closed")
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="evolution-teacher-worker")

    def wake(self) -> None:
        if not self._closed:
            self._wake.set()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake.set()
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def wait_for_job(self, job_id: str, *, timeout_s: float = 180.0) -> dict[str, Any]:
        """Wait for a job terminal state without holding a DuckDB connection."""
        deadline = asyncio.get_running_loop().time() + max(0.1, float(timeout_s))
        while True:
            job = await asyncio.to_thread(self._store.teacher_job, job_id)
            if job is None:
                raise KeyError(f"teacher job not found: {job_id}")
            if str(job.get("status")) in {"COMPLETED", "FAILED_FINAL"}:
                return job
            if asyncio.get_running_loop().time() >= deadline:
                return job
            self.wake()
            await asyncio.sleep(min(0.1, self._poll_interval_s))

    async def run_once(self) -> TeacherJobOutcome | None:
        """Claim and process one due job, returning ``None`` when the queue is idle."""
        claim_started_at = perf_counter()
        try:
            job = await asyncio.to_thread(
                self._store.claim_teacher_job,
                lease_owner=self._lease_owner,
                profile_id=self._profile_id,
                lease_seconds=self._lease_seconds,
            )
        finally:
            self._claim_latency.observe((perf_counter() - claim_started_at) * 1000.0)
        if job is None:
            return None

        job_started_at = perf_counter()
        heartbeat = asyncio.create_task(
            self._heartbeat(str(job["job_id"])),
            name=f"teacher-lease-{job['job_id']}",
        )
        try:
            return await self._process(job)
        except Exception as exc:
            attempts = int(job.get("attempts") or 1)
            retryable = not isinstance(exc, _INTERNAL_DEFECTS) and attempts < self._max_attempts
            retry_after = self._retry_backoff_s * (2 ** max(0, attempts - 1))
            error_text = redact_sensitive_text(str(exc), force=True)[:2000]
            failure_event = EvolutionEvent.create(
                EvolutionEventType.TEACHER_JOB_FAILED,
                context=EvolutionContext(
                    profile_id=str(job.get("profile_id") or ""),
                    workspace_id=str(job.get("workspace_id") or ""),
                    session_id=str(job.get("session_id") or ""),
                    session_generation=int(job.get("session_generation") or 0),
                    correlation_id=str(job.get("episode_id") or ""),
                ),
                payload={
                    "job_id": str(job["job_id"]),
                    "attempt": attempts,
                    "retryable": retryable,
                    "error_type": type(exc).__name__,
                    "error": error_text,
                },
                producer="world_model.teacher_worker",
                producer_version=self._producer_version,
                privacy_class="session",
                dedup_key=f"teacher.job_failed:{job['job_id']}:{attempts}",
            )
            try:
                await asyncio.to_thread(
                    self._store.fail_teacher_job,
                    str(job["job_id"]),
                    error_text,
                    lease_owner=self._lease_owner,
                    retryable=retryable,
                    retry_after_s=retry_after,
                    events=(failure_event,),
                )
            except Exception:
                logger.error(
                    "teacher job failure state could not be persisted job=%s",
                    job["job_id"],
                    exc_info=True,
                )
            logger.warning(
                "teacher job failed job=%s retryable=%s",
                job["job_id"],
                retryable,
                exc_info=True,
            )
            return TeacherJobOutcome(
                job_id=str(job["job_id"]),
                episode_id=str(job.get("episode_id") or ""),
                status="FAILED_RETRYABLE" if retryable else "FAILED_FINAL",
                error=str(exc),
            )
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            self._job_latency.observe((perf_counter() - job_started_at) * 1000.0)

    async def _process(self, job: Mapping[str, Any]) -> TeacherJobOutcome:
        records = await self._read_window(job)
        trajectory = self._trajectory(records)
        verdict = await asyncio.wait_for(
            self._teacher.grade_and_propose(
                trajectory,
                str(job.get("goal") or ""),
                degraded_capabilities=self._collect_degraded_capabilities(),
                raise_on_error=True,
            ),
            timeout=self._teacher_timeout_s,
        )
        grades = tuple(getattr(verdict, "grades", ()) or ())
        verdicts = tuple(getattr(verdict, "verdicts", ()) or ())
        acquisition_plans = self._plan_acquisitions(verdicts, job)
        proposal_ids = [plan.proposal_id for plan in acquisition_plans if plan.proposal_id]
        result_payload = {
            "job_id": str(job["job_id"]),
            "episode_id": str(job["episode_id"]),
            "grades": [
                {
                    "experience_id": str(getattr(grade, "experience_id", "")),
                    "advantage": float(getattr(grade, "advantage", 0.0)),
                    "is_forking": bool(getattr(grade, "is_forking", False)),
                    "grade_label": str(getattr(grade, "grade_label", "")),
                }
                for grade in grades
            ],
            "verdicts": [item.to_dict() for item in verdicts],
            "proposal_ids": proposal_ids,
            "acquisition_resolutions": [
                {
                    "verdict_id": plan.verdict_id,
                    "requirement": plan.requirement.to_dict(),
                    "outcome": plan.outcome,
                    "reason": plan.reason,
                    "resolution": dict(plan.resolution),
                    "proposal_id": plan.proposal_id,
                }
                for plan in acquisition_plans
            ],
            "raw_payload": dict(getattr(verdict, "raw_payload", {}) or {}),
        }
        artifact = await asyncio.to_thread(
            self._artifact_store.put_json,
            result_payload,
            privacy_class="session",
        )
        prompt_hash = content_hash(
            {"goal": str(job.get("goal") or ""), "trajectory": trajectory}
        )
        context = EvolutionContext(
            profile_id=str(job["profile_id"]),
            workspace_id=str(job.get("workspace_id") or ""),
            session_id=str(job["session_id"]),
            session_generation=int(job.get("session_generation") or 0),
            artifact_id=artifact.artifact_id,
            correlation_id=str(job["episode_id"]),
        )
        finalized = await asyncio.to_thread(
            self._store.read,
            profile_id=context.profile_id,
            session_id=context.session_id,
            correlation_id=context.correlation_id,
            event_type=EvolutionEventType.SESSION_FINALIZED,
            limit=1,
        )
        causation_id = finalized[0].event.event_id if finalized else ""
        graded_event = EvolutionEvent.create(
            EvolutionEventType.TEACHER_GRADED,
            context=context.with_ids(causation_id=causation_id),
            payload={
                "job_id": str(job["job_id"]),
                "episode_id": str(job["episode_id"]),
                "grade_count": len(grades),
                "verdict_count": len(verdicts),
                "proposal_ids": proposal_ids,
                "artifact_id": artifact.artifact_id,
                "prompt_hash": prompt_hash,
            },
            producer="world_model.teacher_worker",
            producer_version=self._producer_version,
            privacy_class="session",
            dedup_key=f"teacher.graded:{job['job_id']}",
        )
        events = [graded_event]
        verdict_events: dict[str, EvolutionEvent] = {}
        for item in verdicts:
            verdict_event = EvolutionEvent.create(
                EvolutionEventType.TEACHER_VERDICT_RECORDED,
                context=context.with_ids(
                    decision_id=str(item.verdict_id),
                    causation_id=graded_event.event_id,
                ),
                payload=item.to_dict(),
                producer="world_model.teacher_worker",
                producer_version=self._producer_version,
                privacy_class="session",
                dedup_key=f"teacher.verdict_recorded:{item.verdict_id}",
            )
            verdict_events[str(item.verdict_id)] = verdict_event
            events.append(verdict_event)
        for plan in acquisition_plans:
            verdict_event = verdict_events.get(plan.verdict_id)
            causation_id = verdict_event.event_id if verdict_event is not None else graded_event.event_id
            resolution_event = EvolutionEvent.create(
                EvolutionEventType.REQUIREMENT_RESOLVED,
                context=context.with_ids(
                    requirement_id=str(plan.requirement.requirement_id),
                    decision_id=plan.verdict_id,
                    proposal_id=plan.proposal_id,
                    causation_id=causation_id,
                ),
                payload={
                    "requirement": plan.requirement.to_dict(),
                    "outcome": plan.outcome,
                    "reason": plan.reason,
                    "resolution": dict(plan.resolution),
                    "proposal_id": plan.proposal_id,
                },
                producer="world_model.teacher_worker",
                producer_version=self._producer_version,
                privacy_class="session",
                dedup_key=f"requirement.resolved:{job['job_id']}:{plan.verdict_id}",
            )
            events.append(resolution_event)
            if plan.proposal_event is not None:
                events.append(
                    replace(
                        plan.proposal_event,
                        context=plan.proposal_event.context.with_ids(
                            causation_id=resolution_event.event_id
                        ),
                    )
                )
        completed = await asyncio.to_thread(
            self._store.complete_teacher_job,
            str(job["job_id"]),
            lease_owner=self._lease_owner,
            events=events,
            prompt_hash=prompt_hash,
            result_artifact_id=artifact.artifact_id,
        )
        if not completed:
            raise RuntimeError(f"teacher job lease lost: {job['job_id']}")
        await self._after_commit()
        return TeacherJobOutcome(
            job_id=str(job["job_id"]),
            episode_id=str(job["episode_id"]),
            status="COMPLETED",
            artifact_id=artifact.artifact_id,
            verdict_count=len(verdicts),
            grade_count=len(grades),
        )

    def _plan_acquisitions(
        self,
        verdicts: Sequence[Any],
        job: Mapping[str, Any],
    ) -> tuple[_AcquisitionPlan, ...]:
        """Resolve each acquisition and prepare any proposal for atomic commit."""
        from leapflow.domain.evolution_intent import WORLD_MODEL_ORIGIN
        from leapflow.learning.capability_gap_detector import CapabilityGapDetector
        from leapflow.learning.outcome_governance_feed import origin_may_authorise

        detector = CapabilityGapDetector()
        plans: list[_AcquisitionPlan] = []
        for verdict in verdicts:
            intent = verdict.to_intent()
            if intent is None:
                continue
            requirement = replace(
                intent.to_requirement(),
                requirement_id=f"req-wm-{intent.capability}",
            )
            verdict_id = str(verdict.verdict_id)
            if self._authorising_origins and not origin_may_authorise(
                WORLD_MODEL_ORIGIN, self._authorising_origins
            ):
                plans.append(
                    _AcquisitionPlan(
                        verdict_id,
                        requirement,
                        "no_op",
                        "origin_not_authorised",
                        {},
                    )
                )
                continue
            if self._proposal_queue is None:
                plans.append(
                    _AcquisitionPlan(
                        verdict_id,
                        requirement,
                        "no_op",
                        "self_evolution_disabled",
                        {},
                    )
                )
                continue
            if self._acquisition_resolver is None:
                plans.append(
                    _AcquisitionPlan(
                        verdict_id,
                        requirement,
                        "no_op",
                        "live_resolution_unavailable",
                        {},
                    )
                )
                continue
            resolution = dict(self._acquisition_resolver(intent, job) or {})
            if not bool(resolution.get("resolved", False)):
                plans.append(
                    _AcquisitionPlan(
                        verdict_id,
                        requirement,
                        "no_op",
                        str(resolution.get("reason") or "live_resolution_unavailable"),
                        resolution,
                    )
                )
                continue
            if bool(resolution.get("satisfied", False)):
                plans.append(
                    _AcquisitionPlan(
                        verdict_id,
                        requirement,
                        "satisfied",
                        "capability_already_available",
                        resolution,
                    )
                )
                continue
            proposal = detector.proposal_from_evolution_intent(intent)
            evidence = tuple(getattr(proposal, "evidence", ()) or ())
            metadata = dict(getattr(evidence[0], "metadata", {})) if evidence else {}
            item, proposal_event = self._proposal_queue.prepare_enqueue(
                requirements=(requirement,),
                environment=dict(resolution.get("environment") or {}),
                source="world_model",
                observation_ids=tuple(intent.evidence_ids),
                risk={"max_risk_level": requirement.max_risk_level},
                metadata={
                    "plugin_id": str(getattr(proposal, "plugin_id", "")),
                    "capability_summary": str(
                        getattr(proposal, "capability_summary", "")
                    ),
                    "intent_id": str(metadata.get("intent_id", "")),
                    "confidence": str(metadata.get("confidence", "")),
                    "replaces": str(metadata.get("replaces", "")),
                },
                occurred_at=float(getattr(verdict, "created_at", 0.0) or 0.0) or None,
            )
            plans.append(
                _AcquisitionPlan(
                    verdict_id,
                    requirement,
                    "unmet",
                    str(resolution.get("reason") or "no eligible capability provider"),
                    resolution,
                    proposal_id=item.proposal_id,
                    proposal_event=proposal_event,
                )
            )
        return tuple(plans)

    def _collect_degraded_capabilities(self) -> tuple[Mapping[str, Any], ...]:
        provider = self._degraded_capabilities
        if provider is None:
            return ()
        try:
            facts = tuple(provider() or ())
        except Exception:  # noqa: BLE001 - context may degrade; the job remains valid
            logger.warning("teacher degradation context unavailable", exc_info=True)
            return ()
        # Close the D1 feedback edge: show the teacher what it concluded last time for a
        # still-failing capability, so the same evidence cannot only ever produce the same
        # answer. Enriched from the durable knowledge projection rather than a second
        # store, so the fact and its prior verdict share one source.
        projection = self._knowledge_projection
        if projection is None:
            return facts
        enriched: list[Mapping[str, Any]] = []
        for fact in facts:
            capability = str(fact.get("capability") or "")
            prior = projection.for_capability(capability) if capability else None
            if prior is None:
                enriched.append(fact)
                continue
            row = dict(fact)
            row["prior_action"] = prior.action
            row["prior_knowledge"] = prior.knowledge
            enriched.append(row)
        return tuple(enriched)

    async def _after_commit(self) -> None:
        projection = self._knowledge_projection
        if projection is not None:
            try:
                await asyncio.to_thread(projection.refresh)
            except Exception:  # noqa: BLE001 - committed facts remain replayable
                logger.warning("knowledge projection refresh failed", exc_info=True)
    async def _read_window(
        self,
        job: Mapping[str, Any],
    ) -> list[EvolutionEventRecord]:
        records: list[EvolutionEventRecord] = []
        cursor = int(job["from_sequence"])
        through_sequence = int(job["through_sequence"])
        while cursor < through_sequence:
            page = await asyncio.to_thread(
                self._store.read,
                profile_id=str(job["profile_id"]),
                session_id=str(job["session_id"]),
                session_generation=int(job.get("session_generation") or 0),
                after_sequence=cursor,
                through_sequence=through_sequence,
                limit=5000,
            )
            if not page:
                break
            records.extend(page)
            cursor = page[-1].sequence
        return records

    async def _heartbeat(self, job_id: str) -> None:
        interval = max(1.0, self._lease_seconds / 3.0)
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(
                self._store.renew_teacher_job,
                job_id,
                lease_owner=self._lease_owner,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                return

    async def _run(self) -> None:
        while not self._closed:
            self._wake.clear()
            try:
                outcome = await self.run_once()
            except Exception:
                logger.exception("teacher worker polling failed")
                outcome = None
            if outcome is not None:
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_interval_s)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def _trajectory(records: Sequence[EvolutionEventRecord]) -> list[dict[str, Any]]:
        starts: dict[str, EvolutionEvent] = {}
        trajectory: list[dict[str, Any]] = []
        for record in records:
            event = record.event
            action_id = event.context.action_id
            if event.event_type == EvolutionEventType.ACTION_STARTED:
                starts[action_id] = event
                continue
            if event.event_type not in {
                EvolutionEventType.ACTION_COMPLETED,
                EvolutionEventType.ACTION_FAILED,
            }:
                continue
            started = starts.pop(action_id, None)
            start_payload = started.payload if started is not None else {}
            result = event.payload.get("result")
            trajectory.append(
                {
                    "experience_id": "",
                    "evidence_ids": [
                        item
                        for item in (
                            started.event_id if started is not None else "",
                            event.event_id,
                        )
                        if item
                    ],
                    "action_description": (
                        f"{start_payload.get('action_type', 'action')}:"
                        f"{start_payload.get('action_name', action_id or 'unknown')}"
                    ),
                    "predicted_effect": str(start_payload.get("goal") or "complete successfully"),
                    "actual_effect": str(result if result is not None else event.payload),
                    "delta": 0.0 if bool(event.payload.get("ok", False)) else 1.0,
                }
            )
        for action_id, started in starts.items():
            trajectory.append(
                {
                    "experience_id": "",
                    "evidence_ids": [started.event_id],
                    "action_description": (
                        f"{started.payload.get('action_type', 'action')}:"
                        f"{started.payload.get('action_name', action_id or 'unknown')}"
                    ),
                    "predicted_effect": str(started.payload.get("goal") or "complete successfully"),
                    "actual_effect": "action did not record a terminal outcome",
                    "delta": 1.0,
                }
            )
        return trajectory


__all__ = [
    "DurableTeacherWorker",
    "Teacher",
    "TeacherJobOutcome",
    "TeacherWorkerMetrics",
    "TeacherWorkStore",
]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB append-only store for causal self-evolution events.

All instances share the daemon-owned ConnectionHolder.  The process lock protects
sequence allocation and multi-row transactions across thread-local DuckDB cursors;
callers must still respect the architectural single-writer rule.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import (
    EvolutionContext,
    EvolutionEvent,
    EvolutionEventRecord,
    canonical_json,
    content_hash,
)
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.schema import ensure_schema


_INSERT_SQL = """
INSERT INTO evolution_events (
    sequence, event_id, event_type,
    profile_id, workspace_id, session_id, session_generation,
    turn_id, frame_id, action_id, observation_id, requirement_id,
    decision_id, proposal_id, artifact_id, plugin_id, version_id,
    correlation_id, causation_id, occurred_at, producer, producer_version,
    privacy_class, schema_version, payload_json, payload_hash, dedup_key
) VALUES (
    nextval('evolution_event_sequence'), ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?
)
ON CONFLICT (profile_id, dedup_key) DO NOTHING
RETURNING sequence
"""


class DuckDBEvolutionEventStore:
    """Append and query immutable evolution events."""

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder: ConnectionHolder = source
        self._write_lock = threading.Lock()
        ensure_schema(self._conn)

    @property
    def _conn(self) -> Any:
        """Resolve on every call so LocalConnectionHolder preserves thread affinity."""
        return self._holder.connection

    def append(self, event: EvolutionEvent) -> bool:
        """Append one event; return ``False`` when its dedup key already exists."""
        with self._write_lock:
            row = self._conn.execute(_INSERT_SQL, self._params(event)).fetchone()
            return row is not None

    def append_many(self, events: Sequence[EvolutionEvent]) -> int:
        """Atomically append a batch, ignoring idempotent duplicates."""
        if not events:
            return 0
        with self._write_lock:
            connection = self._conn
            connection.execute("BEGIN TRANSACTION")
            inserted = 0
            try:
                for event in events:
                    row = connection.execute(_INSERT_SQL, self._params(event)).fetchone()
                    if row is not None:
                        inserted += 1
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            return inserted

    def read(
        self,
        *,
        profile_id: str = "",
        session_id: str = "",
        session_generation: int | None = None,
        correlation_id: str = "",
        proposal_id: str = "",
        proposal_events_only: bool = False,
        event_type: str = "",
        after_sequence: int = 0,
        through_sequence: int = 0,
        limit: int = 500,
    ) -> list[EvolutionEventRecord]:
        """Read a bounded event slice with its durable causal cursor."""
        clauses = ["sequence > ?"]
        params: list[Any] = [max(0, int(after_sequence))]
        if through_sequence > 0:
            clauses.append("sequence <= ?")
            params.append(int(through_sequence))
        for column, value in (
            ("profile_id", profile_id),
            ("session_id", session_id),
            ("correlation_id", correlation_id),
            ("proposal_id", proposal_id),
            ("event_type", event_type),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(str(value))
        if proposal_events_only:
            clauses.append("proposal_id <> ''")
        if session_generation is not None:
            clauses.append("session_generation = ?")
            params.append(int(session_generation))
        params.append(min(max(1, int(limit)), 5000))
        rows = self._conn.execute(
            f"""
            SELECT sequence, event_id, event_type,
                   profile_id, workspace_id, session_id, session_generation,
                   turn_id, frame_id, action_id, observation_id, requirement_id,
                   decision_id, proposal_id, artifact_id, plugin_id, version_id,
                   correlation_id, causation_id, occurred_at, producer,
                   producer_version, privacy_class, schema_version, payload_json,
                   payload_hash, dedup_key
            FROM evolution_events
            WHERE {' AND '.join(clauses)}
            ORDER BY sequence ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def latest_sequence(self, *, profile_id: str = "", session_id: str = "") -> int:
        clauses = ["1=1"]
        params: list[Any] = []
        if profile_id:
            clauses.append("profile_id = ?")
            params.append(str(profile_id))
        if session_id:
            clauses.append("session_id = ?")
            params.append(str(session_id))
        row = self._conn.execute(
            f"SELECT COALESCE(MAX(sequence), 0) FROM evolution_events WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row[0] if row else 0)

    def latest_evidence_sequence(
        self,
        *,
        profile_id: str,
        session_id: str,
        session_generation: int | None = None,
    ) -> int:
        """Return the latest hot-path input cursor eligible for teacher grading."""
        event_types = (
            EvolutionEventType.ACTION_STARTED,
            EvolutionEventType.ACTION_COMPLETED,
            EvolutionEventType.ACTION_FAILED,
            EvolutionEventType.ENVIRONMENT_OBSERVED,
        )
        placeholders = ", ".join("?" for _ in event_types)
        generation_clause = ""
        params: list[Any] = [str(profile_id), str(session_id), *event_types]
        if session_generation is not None:
            generation_clause = " AND session_generation=?"
            params.append(int(session_generation))
        row = self._conn.execute(
            f"""
            SELECT COALESCE(MAX(sequence), 0)
            FROM evolution_events
            WHERE profile_id=? AND session_id=? AND event_type IN ({placeholders})
                  {generation_clause}
            """,
            params,
        ).fetchone()
        return int(row[0] if row else 0)

    def evidence_sessions(self, *, profile_id: str = "") -> list[dict[str, Any]]:
        """List session heads that contain finalizable hot-path evidence."""
        event_types = (
            EvolutionEventType.ACTION_STARTED,
            EvolutionEventType.ACTION_COMPLETED,
            EvolutionEventType.ACTION_FAILED,
            EvolutionEventType.ENVIRONMENT_OBSERVED,
        )
        placeholders = ", ".join("?" for _ in event_types)
        clauses = ["session_id <> ''", f"event_type IN ({placeholders})"]
        params: list[Any] = list(event_types)
        if profile_id:
            clauses.append("profile_id = ?")
            params.append(str(profile_id))
        rows = self._conn.execute(
            f"""
            SELECT profile_id, workspace_id, session_id, session_generation,
                   MAX(sequence) AS through_sequence
            FROM evolution_events
            WHERE {' AND '.join(clauses)}
            GROUP BY profile_id, workspace_id, session_id, session_generation
            ORDER BY through_sequence ASC
            """,
            params,
        ).fetchall()
        keys = (
            "profile_id",
            "workspace_id",
            "session_id",
            "session_generation",
            "through_sequence",
        )
        return [dict(zip(keys, row)) for row in rows]

    def count(self, *, profile_id: str = "", event_type: str = "") -> int:
        clauses = ["1=1"]
        params: list[Any] = []
        if profile_id:
            clauses.append("profile_id = ?")
            params.append(str(profile_id))
        if event_type:
            clauses.append("event_type = ?")
            params.append(str(event_type))
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM evolution_events WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row[0] if row else 0)

    def finalize_session(
        self,
        *,
        profile_id: str,
        workspace_id: str,
        session_id: str,
        session_generation: int,
        from_sequence: int,
        through_sequence: int,
        reason: str,
        goal: str = "",
        model: str = "",
    ) -> tuple[str, str]:
        """Atomically finalize one evidence slice and enqueue its teacher job.

        Returns ``(episode_id, job_id)``. Repeating the same finalization is
        idempotent because both identities derive from the immutable sequence range.
        """
        if not session_id:
            raise ValueError("session_id is required")
        if through_sequence <= from_sequence:
            return "", ""
        identity = {
            "profile_id": profile_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "session_generation": int(session_generation),
            "from_sequence": int(from_sequence),
            "through_sequence": int(through_sequence),
        }
        digest = content_hash(identity)
        episode_id = f"episode-{digest[:24]}"
        job_id = f"teacher-{digest[:24]}"
        context = EvolutionContext(
            profile_id=profile_id,
            workspace_id=workspace_id,
            session_id=session_id,
            session_generation=int(session_generation),
            correlation_id=episode_id,
        )
        finalized = EvolutionEvent.create(
            EvolutionEventType.SESSION_FINALIZED,
            context=context,
            payload={
                "episode_id": episode_id,
                "from_sequence": int(from_sequence),
                "through_sequence": int(through_sequence),
                "reason": str(reason),
                "goal": str(goal),
            },
            producer="session.finalizer",
            dedup_key=f"session.finalized:{episode_id}",
        )
        queued = EvolutionEvent.create(
            EvolutionEventType.TEACHER_JOB_QUEUED,
            context=context.with_ids(causation_id=finalized.event_id),
            payload={
                "episode_id": episode_id,
                "job_id": job_id,
                "model": str(model),
                "from_sequence": int(from_sequence),
                "through_sequence": int(through_sequence),
            },
            producer="session.finalizer",
            dedup_key=f"teacher.job_queued:{job_id}",
        )
        now = time.time()
        with self._write_lock:
            connection = self._conn
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(_INSERT_SQL, self._params(finalized)).fetchone()
                connection.execute(_INSERT_SQL, self._params(queued)).fetchone()
                connection.execute(
                    """
                    INSERT INTO evolution_teacher_jobs (
                        job_id, profile_id, workspace_id, session_id,
                        session_generation, episode_id, from_sequence,
                        through_sequence, reason, goal, status, model,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)
                    ON CONFLICT (profile_id, episode_id) DO NOTHING
                    """,
                    [
                        job_id,
                        profile_id,
                        workspace_id,
                        session_id,
                        int(session_generation),
                        episode_id,
                        int(from_sequence),
                        int(through_sequence),
                        str(reason),
                        str(goal),
                        str(model),
                        now,
                        now,
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return episode_id, job_id

    def claim_teacher_job(
        self,
        *,
        lease_owner: str,
        profile_id: str = "",
        lease_seconds: float = 120.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Lease the oldest due teacher job for one worker."""
        instant = float(now if now is not None else time.time())
        with self._write_lock:
            connection = self._conn
            connection.execute("BEGIN TRANSACTION")
            try:
                clauses = [
                    "((status IN ('PENDING', 'FAILED_RETRYABLE') AND next_attempt_at <= ?) "
                    "OR (status = 'RUNNING' AND lease_until <= ?))",
                ]
                params: list[Any] = [instant, instant]
                if profile_id:
                    clauses.append("profile_id = ?")
                    params.append(str(profile_id))
                row = connection.execute(
                    f"""
                    SELECT job_id, profile_id, workspace_id, session_id,
                           session_generation, episode_id, from_sequence,
                           through_sequence, reason, goal, attempts, model,
                           prompt_hash, created_at
                    FROM evolution_teacher_jobs
                    WHERE {' AND '.join(clauses)}
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                lease_until = instant + max(1.0, float(lease_seconds))
                connection.execute(
                    """
                    UPDATE evolution_teacher_jobs
                    SET status='RUNNING', lease_owner=?, lease_until=?,
                        attempts=attempts+1, updated_at=?
                    WHERE job_id=?
                    """,
                    [str(lease_owner), lease_until, instant, row[0]],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        keys = (
            "job_id",
            "profile_id",
            "workspace_id",
            "session_id",
            "session_generation",
            "episode_id",
            "from_sequence",
            "through_sequence",
            "reason",
            "goal",
            "attempts",
            "model",
            "prompt_hash",
            "created_at",
        )
        claimed = dict(zip(keys, row))
        claimed["attempts"] = int(claimed["attempts"] or 0) + 1
        claimed["session_generation"] = int(claimed["session_generation"] or 0)
        claimed["from_sequence"] = int(claimed["from_sequence"] or 0)
        claimed["through_sequence"] = int(claimed["through_sequence"] or 0)
        claimed["lease_owner"] = str(lease_owner)
        claimed["lease_until"] = lease_until
        return claimed

    def renew_teacher_job(
        self,
        job_id: str,
        *,
        lease_owner: str,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Extend a running job lease only when the caller still owns it."""
        instant = float(now if now is not None else time.time())
        with self._write_lock:
            row = self._conn.execute(
                """
                UPDATE evolution_teacher_jobs
                SET lease_until=?, updated_at=?
                WHERE job_id=? AND status='RUNNING' AND lease_owner=?
                RETURNING job_id
                """,
                [
                    instant + max(1.0, float(lease_seconds)),
                    instant,
                    str(job_id),
                    str(lease_owner),
                ],
            ).fetchone()
        return row is not None

    def complete_teacher_job(
        self,
        job_id: str,
        *,
        lease_owner: str,
        events: Sequence[EvolutionEvent] = (),
        prompt_hash: str = "",
        result_artifact_id: str = "",
    ) -> bool:
        """Append teacher facts and complete a job only for its current lease owner."""
        with self._write_lock:
            connection = self._conn
            connection.execute("BEGIN TRANSACTION")
            try:
                owned = connection.execute(
                    """
                    SELECT 1 FROM evolution_teacher_jobs
                    WHERE job_id=? AND status='RUNNING' AND lease_owner=?
                    """,
                    [str(job_id), str(lease_owner)],
                ).fetchone()
                if owned is None:
                    connection.execute("ROLLBACK")
                    return False
                for event in events:
                    connection.execute(_INSERT_SQL, self._params(event)).fetchone()
                updated = connection.execute(
                    """
                    UPDATE evolution_teacher_jobs
                    SET status='COMPLETED', lease_owner='', lease_until=0,
                        prompt_hash=?, result_artifact_id=?, completed_at=?,
                        updated_at=?, error=''
                    WHERE job_id=? AND status='RUNNING' AND lease_owner=?
                    RETURNING job_id
                    """,
                    [
                        str(prompt_hash),
                        str(result_artifact_id),
                        time.time(),
                        time.time(),
                        str(job_id),
                        str(lease_owner),
                    ],
                ).fetchone()
                connection.execute("COMMIT")
                return updated is not None
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def fail_teacher_job(
        self,
        job_id: str,
        error: str,
        *,
        lease_owner: str,
        retryable: bool,
        retry_after_s: float = 5.0,
        events: Sequence[EvolutionEvent] = (),
    ) -> bool:
        """Record a teacher failure only while the caller owns the lease."""
        now = time.time()
        with self._write_lock:
            connection = self._conn
            connection.execute("BEGIN TRANSACTION")
            try:
                owned = connection.execute(
                    """
                    SELECT 1 FROM evolution_teacher_jobs
                    WHERE job_id=? AND status='RUNNING' AND lease_owner=?
                    """,
                    [str(job_id), str(lease_owner)],
                ).fetchone()
                if owned is None:
                    connection.execute("ROLLBACK")
                    return False
                for event in events:
                    connection.execute(_INSERT_SQL, self._params(event)).fetchone()
                row = connection.execute(
                    """
                    UPDATE evolution_teacher_jobs
                    SET status=?, lease_owner='', lease_until=0, next_attempt_at=?,
                        updated_at=?, error=?
                    WHERE job_id=? AND status='RUNNING' AND lease_owner=?
                    RETURNING job_id
                    """,
                    [
                        "FAILED_RETRYABLE" if retryable else "FAILED_FINAL",
                        now + max(0.0, float(retry_after_s)) if retryable else 0.0,
                        now,
                        str(error)[:2000],
                        str(job_id),
                        str(lease_owner),
                    ],
                ).fetchone()
                connection.execute("COMMIT")
                return row is not None
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def teacher_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT job_id, profile_id, workspace_id, session_id,
                   session_generation, episode_id, from_sequence, through_sequence,
                   reason, goal, status, lease_owner, lease_until, attempts, model,
                   prompt_hash, result_artifact_id, next_attempt_at, created_at,
                   updated_at, completed_at, error
            FROM evolution_teacher_jobs WHERE job_id=?
            """,
            [str(job_id)],
        ).fetchone()
        if row is None:
            return None
        keys = (
            "job_id", "profile_id", "workspace_id", "session_id",
            "session_generation", "episode_id", "from_sequence", "through_sequence",
            "reason", "goal", "status", "lease_owner", "lease_until", "attempts",
            "model", "prompt_hash", "result_artifact_id", "next_attempt_at",
            "created_at", "updated_at", "completed_at", "error",
        )
        return dict(zip(keys, row))

    def load_projection(
        self,
        *,
        projection_name: str,
        profile_id: str,
        scope_key: str,
    ) -> tuple[int, dict[str, Any]] | None:
        row = self._conn.execute(
            """
            SELECT last_sequence, state_json
            FROM evolution_projections
            WHERE projection_name=? AND profile_id=? AND scope_key=?
            """,
            [str(projection_name), str(profile_id), str(scope_key)],
        ).fetchone()
        if row is None:
            return None
        try:
            state = json.loads(row[1] or "{}")
        except (json.JSONDecodeError, TypeError):
            return None
        return int(row[0] or 0), state if isinstance(state, dict) else {}

    def save_projection(
        self,
        *,
        projection_name: str,
        profile_id: str,
        scope_key: str,
        last_sequence: int,
        state: dict[str, Any],
    ) -> None:
        with self._write_lock:
            self._conn.execute(
                """
                INSERT INTO evolution_projections (
                    projection_name, profile_id, scope_key, last_sequence,
                    state_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (projection_name, profile_id, scope_key) DO UPDATE SET
                    last_sequence=excluded.last_sequence,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                [
                    str(projection_name),
                    str(profile_id),
                    str(scope_key),
                    int(last_sequence),
                    canonical_json(state),
                    time.time(),
                ],
            )

    def delete_projection(
        self,
        *,
        projection_name: str,
        profile_id: str,
        scope_key: str,
    ) -> None:
        with self._write_lock:
            self._conn.execute(
                """
                DELETE FROM evolution_projections
                WHERE projection_name=? AND profile_id=? AND scope_key=?
                """,
                [str(projection_name), str(profile_id), str(scope_key)],
            )

    def last_finalized_sequence(
        self,
        *,
        profile_id: str,
        session_id: str,
        session_generation: int | None = None,
    ) -> int:
        """Return the highest evidence cursor finalized for one session generation."""
        records = self.read(
            profile_id=profile_id,
            session_id=session_id,
            session_generation=session_generation,
            event_type=EvolutionEventType.SESSION_FINALIZED,
            limit=5000,
        )
        return max(
            (
                int(record.event.payload.get("through_sequence") or 0)
                for record in records
            ),
            default=0,
        )

    def close(self) -> None:
        if self._owns_holder:
            self._holder.close()

    @staticmethod
    def _params(event: EvolutionEvent) -> list[Any]:
        context = event.context
        return [
            event.event_id,
            event.event_type,
            context.profile_id,
            context.workspace_id,
            context.session_id,
            context.session_generation,
            context.turn_id,
            context.frame_id,
            context.action_id,
            context.observation_id,
            context.requirement_id,
            context.decision_id,
            context.proposal_id,
            context.artifact_id,
            context.plugin_id,
            context.version_id,
            context.correlation_id,
            context.causation_id,
            event.occurred_at,
            event.producer,
            event.producer_version,
            event.privacy_class,
            event.schema_version,
            canonical_json(dict(event.payload)),
            event.payload_hash,
            event.dedup_key,
        ]

    @staticmethod
    def _from_row(row: Sequence[Any]) -> EvolutionEventRecord:
        context = EvolutionContext(
            profile_id=str(row[3] or ""),
            workspace_id=str(row[4] or ""),
            session_id=str(row[5] or ""),
            session_generation=int(row[6] or 0),
            turn_id=str(row[7] or ""),
            frame_id=str(row[8] or ""),
            action_id=str(row[9] or ""),
            observation_id=str(row[10] or ""),
            requirement_id=str(row[11] or ""),
            decision_id=str(row[12] or ""),
            proposal_id=str(row[13] or ""),
            artifact_id=str(row[14] or ""),
            plugin_id=str(row[15] or ""),
            version_id=str(row[16] or ""),
            correlation_id=str(row[17] or ""),
            causation_id=str(row[18] or ""),
        )
        try:
            payload = json.loads(row[24] or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        return EvolutionEventRecord(
            sequence=int(row[0]),
            event=EvolutionEvent(
                event_id=str(row[1] or ""),
                event_type=str(row[2] or ""),
                context=context,
                payload=payload if isinstance(payload, dict) else {},
                occurred_at=float(row[19] or 0.0),
                producer=str(row[20] or ""),
                producer_version=str(row[21] or ""),
                privacy_class=str(row[22] or "system"),
                schema_version=int(row[23] or 1),
                payload_hash=str(row[25] or ""),
                dedup_key=str(row[26] or ""),
            ),
        )


class EvolutionTraceEventStore:
    """Adapt framework traces onto the append-only evolution event stream."""

    def __init__(self, store: DuckDBEvolutionEventStore, *, profile_id: str) -> None:
        self._store = store
        self._profile_id = str(profile_id)

    def append(self, traces: Iterable[Mapping[str, Any]]) -> int:
        events: list[EvolutionEvent] = []
        for trace in traces:
            payload = dict(trace)
            trace_id = str(payload.get("trace_id") or content_hash(payload))
            correlation = dict(payload.get("correlation") or {})
            context = EvolutionContext(
                profile_id=self._profile_id,
                proposal_id=str(correlation.get("proposal_id") or ""),
                plugin_id=str(correlation.get("plugin_id") or ""),
                correlation_id=str(
                    correlation.get("episode_id")
                    or correlation.get("proposal_id")
                    or trace_id
                ),
            )
            events.append(
                EvolutionEvent.create(
                    EvolutionEventType.FRAMEWORK_TRACE_RECORDED,
                    context=context,
                    payload=payload,
                    producer="framework.evolution_tap",
                    privacy_class="profile",
                    occurred_at=float(payload.get("ts") or time.time()),
                    dedup_key=f"framework.trace_recorded:{trace_id}",
                )
            )
        return self._store.append_many(events)

    def list_traces(self, *, limit: int = 200) -> list[dict[str, Any]]:
        records = self._store.read(
            profile_id=self._profile_id,
            event_type=EvolutionEventType.FRAMEWORK_TRACE_RECORDED,
            limit=5000,
        )
        rows = [dict(record.event.to_dict()["payload"]) for record in reversed(records)]
        return rows if limit <= 0 else rows[:limit]

    def count(self) -> int:
        return len(self.list_traces(limit=0))


__all__ = ["DuckDBEvolutionEventStore", "EvolutionTraceEventStore"]

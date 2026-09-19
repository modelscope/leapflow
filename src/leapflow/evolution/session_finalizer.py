# Copyright (c) Alibaba, Inc. and its affiliates.
"""Durable session boundaries for cold-path evolution work."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionEventRecord, EvolutionEventStore


@dataclass(frozen=True)
class SessionFinalization:
    """Result of sealing one session evidence window."""

    session_id: str
    from_sequence: int
    through_sequence: int
    episode_id: str = ""
    job_id: str = ""
    evidence_count: int = 0

    @property
    def queued(self) -> bool:
        return bool(self.job_id)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "queued": self.queued}


class SessionFinalizer:
    """Flush the hot path, seal a session slice, and enqueue one teacher job."""

    def __init__(self, store: EvolutionEventStore, outbox: Any) -> None:
        self._store = store
        self._outbox = outbox
        self._lock = asyncio.Lock()

    async def finalize(
        self,
        *,
        profile_id: str,
        workspace_id: str,
        session_id: str,
        session_generation: int = 0,
        reason: str = "manual",
        goal: str = "",
        model: str = "",
    ) -> SessionFinalization:
        """Atomically queue exactly the evidence not finalized before this call."""
        if not session_id:
            raise ValueError("session_id is required")
        async with self._lock:
            await self._outbox.flush()
            from_sequence = await asyncio.to_thread(
                self._store.last_finalized_sequence,
                profile_id=profile_id,
                session_id=session_id,
                session_generation=session_generation,
            )
            through_sequence = await asyncio.to_thread(
                self._store.latest_evidence_sequence,
                profile_id=profile_id,
                session_id=session_id,
                session_generation=session_generation,
            )
            if through_sequence <= from_sequence:
                return SessionFinalization(
                    session_id=session_id,
                    from_sequence=from_sequence,
                    through_sequence=through_sequence,
                )
            evidence = await self._read_window(
                profile_id=profile_id,
                session_id=session_id,
                session_generation=session_generation,
                after_sequence=from_sequence,
                through_sequence=through_sequence,
            )
            source_types = {
                EvolutionEventType.ACTION_STARTED,
                EvolutionEventType.ACTION_COMPLETED,
                EvolutionEventType.ACTION_FAILED,
                EvolutionEventType.ENVIRONMENT_OBSERVED,
            }
            evidence = [record for record in evidence if record.event.event_type in source_types]
            if not evidence:
                return SessionFinalization(
                    session_id=session_id,
                    from_sequence=from_sequence,
                    through_sequence=through_sequence,
                )
            observed_workspaces = {
                record.event.context.workspace_id
                for record in evidence
                if record.event.context.workspace_id
            }
            if len(observed_workspaces) > 1:
                raise ValueError(f"session {session_id!r} spans multiple workspaces")
            if workspace_id and observed_workspaces - {workspace_id}:
                raise ValueError(
                    f"session {session_id!r} contains evidence from another workspace"
                )
            if not workspace_id and observed_workspaces:
                workspace_id = next(iter(observed_workspaces))
            if not goal:
                goal = self._goal_from_records(evidence)
            episode_id, job_id = await asyncio.to_thread(
                self._store.finalize_session,
                profile_id=profile_id,
                workspace_id=workspace_id,
                session_id=session_id,
                session_generation=session_generation,
                from_sequence=from_sequence,
                through_sequence=through_sequence,
                reason=reason,
                goal=goal,
                model=model,
            )
            return SessionFinalization(
                session_id=session_id,
                from_sequence=from_sequence,
                through_sequence=through_sequence,
                episode_id=episode_id,
                job_id=job_id,
                evidence_count=len(evidence),
            )

    async def finalize_pending_sessions(
        self,
        *,
        profile_id: str = "",
        reason: str = "shutdown",
        model: str = "",
    ) -> list[SessionFinalization]:
        """Seal all sessions with new evidence, used only at daemon shutdown."""
        sessions = await asyncio.to_thread(
            self._store.evidence_sessions,
            profile_id=profile_id,
        )
        results: list[SessionFinalization] = []
        for item in sessions:
            result = await self.finalize(
                profile_id=str(item.get("profile_id") or profile_id),
                workspace_id=str(item.get("workspace_id") or ""),
                session_id=str(item.get("session_id") or ""),
                session_generation=int(item.get("session_generation") or 0),
                reason=reason,
                model=model,
            )
            if result.queued:
                results.append(result)
        return results

    async def _read_window(
        self,
        *,
        profile_id: str,
        session_id: str,
        session_generation: int,
        after_sequence: int,
        through_sequence: int,
    ) -> list[EvolutionEventRecord]:
        records: list[EvolutionEventRecord] = []
        cursor = after_sequence
        while cursor < through_sequence:
            page = await asyncio.to_thread(
                self._store.read,
                profile_id=profile_id,
                session_id=session_id,
                session_generation=session_generation,
                after_sequence=cursor,
                through_sequence=through_sequence,
                limit=5000,
            )
            if not page:
                break
            records.extend(page)
            cursor = page[-1].sequence
        return records

    @staticmethod
    def _goal_from_records(records: list[EvolutionEventRecord]) -> str:
        for record in reversed(records):
            if record.event.event_type != EvolutionEventType.ACTION_STARTED:
                continue
            goal = str(record.event.payload.get("goal") or "").strip()
            if goal:
                return goal
        return ""


__all__ = ["SessionFinalization", "SessionFinalizer"]

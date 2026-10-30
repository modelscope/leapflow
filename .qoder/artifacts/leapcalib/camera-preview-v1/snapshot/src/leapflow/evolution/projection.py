# Copyright (c) Alibaba, Inc. and its affiliates.
"""Checkpointed projections derived from the append-only evolution event log."""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.domain.event_types import EvolutionEventType
from leapflow.domain.evolution_event import EvolutionEventRecord
from leapflow.performance import LatencySummary, RollingLatency

_PROJECTION_NAME = "evolution.board.v2"
_MAX_ROWS = 200
_VERDICT_ACTIONS = ("absorb", "rebind", "acquire", "escalate")


@dataclass(frozen=True)
class ProjectionMetrics:
    run_latency: LatencySummary


@runtime_checkable
class ProjectionStore(Protocol):
    """Event and checkpoint operations needed by the projection runner."""

    def read(self, **kwargs: Any) -> list[EvolutionEventRecord]: ...

    def load_projection(self, **kwargs: Any) -> tuple[int, dict[str, Any]] | None: ...

    def save_projection(self, **kwargs: Any) -> None: ...

    def delete_projection(self, **kwargs: Any) -> None: ...


class EvolutionProjectionRunner:
    """Incrementally materialize the LeapBoard evolution view from facts."""

    def __init__(self, store: ProjectionStore) -> None:
        self._store = store
        self._lock = asyncio.Lock()
        self._run_latency = RollingLatency()

    @property
    def metrics(self) -> ProjectionMetrics:
        return ProjectionMetrics(run_latency=self._run_latency.snapshot())

    async def project_session(
        self,
        *,
        profile_id: str,
        session_id: str,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required for a session projection")
        started_at = perf_counter()
        try:
            return await self._project(
                profile_id=profile_id,
                session_id=session_id,
                scope_key=f"session:{session_id}",
                rebuild=rebuild,
            )
        finally:
            self._run_latency.observe((perf_counter() - started_at) * 1000.0)

    async def project_aggregate(
        self,
        *,
        profile_id: str,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        """Build the explicitly cross-session profile view."""
        started_at = perf_counter()
        try:
            return await self._project(
                profile_id=profile_id,
                session_id="",
                scope_key="aggregate:any_session",
                rebuild=rebuild,
            )
        finally:
            self._run_latency.observe((perf_counter() - started_at) * 1000.0)

    async def _project(
        self,
        *,
        profile_id: str,
        session_id: str,
        scope_key: str,
        rebuild: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            if rebuild:
                await asyncio.to_thread(
                    self._store.delete_projection,
                    projection_name=_PROJECTION_NAME,
                    profile_id=profile_id,
                    scope_key=scope_key,
                )
                checkpoint = 0
                state = self._empty_state(profile_id, session_id, scope_key)
            else:
                saved = await asyncio.to_thread(
                    self._store.load_projection,
                    projection_name=_PROJECTION_NAME,
                    profile_id=profile_id,
                    scope_key=scope_key,
                )
                if saved is None:
                    checkpoint = 0
                    state = self._empty_state(profile_id, session_id, scope_key)
                else:
                    checkpoint, state = saved

            cursor = checkpoint
            while True:
                records = await asyncio.to_thread(
                    self._store.read,
                    profile_id=profile_id,
                    session_id=session_id,
                    after_sequence=cursor,
                    limit=5000,
                )
                if not records:
                    break
                for record in records:
                    self._apply(state, record)
                cursor = records[-1].sequence
                if len(records) < 5000:
                    break

            state["last_sequence"] = cursor
            await asyncio.to_thread(
                self._store.save_projection,
                projection_name=_PROJECTION_NAME,
                profile_id=profile_id,
                scope_key=scope_key,
                last_sequence=cursor,
                state=state,
            )
            return self._present(state)

    @staticmethod
    def _empty_state(profile_id: str, session_id: str, scope_key: str) -> dict[str, Any]:
        return {
            "projection": _PROJECTION_NAME,
            "profile_id": profile_id,
            "scope": "session" if session_id else "aggregate",
            "scope_key": scope_key,
            "session_id": session_id,
            "last_sequence": 0,
            "event_count": 0,
            "action_count": 0,
            "failed_action_count": 0,
            "environment_count": 0,
            "teacher_failure_count": 0,
            "no_op_count": 0,
            "verdict_counts": {action: 0 for action in _VERDICT_ACTIONS},
            "episodes": {},
            "proposals": {},
            "verdicts": [],
            "knowledge": [],
            "provider_bindings": [],
            "proposal_candidates": [],
            "human_escalations": [],
            "resolutions": [],
            "environment": [],
            "timeline": [],
        }

    @staticmethod
    def _apply(state: dict[str, Any], record: EvolutionEventRecord) -> None:
        event = record.event
        event_type = event.event_type
        payload = event.to_dict()["payload"]
        state["event_count"] = int(state.get("event_count") or 0) + 1
        state["last_sequence"] = record.sequence

        proposal_state = payload.get("proposal_state")
        if isinstance(proposal_state, Mapping):
            proposal = dict(proposal_state)
            proposal_id = str(proposal.get("proposal_id") or event.context.proposal_id)
            if proposal_id:
                state["proposals"][proposal_id] = proposal
                EvolutionProjectionRunner._append_bounded(
                    state["timeline"],
                    {
                        "sequence": record.sequence,
                        "title": f"proposal → {proposal.get('status', '').lower()}",
                        "summary": str(payload.get("reason") or proposal_id),
                        "severity": "notable"
                        if proposal.get("status") in {"FAILED", "QUARANTINED", "REJECTED"}
                        else "info",
                    },
                )

        if event_type == EvolutionEventType.ACTION_STARTED:
            state["action_count"] = int(state.get("action_count") or 0) + 1
        elif event_type == EvolutionEventType.ACTION_FAILED:
            state["failed_action_count"] = int(state.get("failed_action_count") or 0) + 1
        elif event_type == EvolutionEventType.ENVIRONMENT_OBSERVED:
            state["environment_count"] = int(state.get("environment_count") or 0) + 1
            EvolutionProjectionRunner._append_bounded(
                state["environment"],
                {
                    "observation_id": event.context.observation_id,
                    "kind": payload.get("kind", ""),
                    "app_id": payload.get("app_id", ""),
                    "summary": EvolutionProjectionRunner._environment_summary(payload),
                    "observed_at": event.occurred_at,
                },
            )
        elif event_type == EvolutionEventType.SESSION_FINALIZED:
            episode_id = str(payload.get("episode_id") or event.context.correlation_id)
            state["episodes"][episode_id] = {
                "episode_id": episode_id,
                "opened_at": event.occurred_at,
                "closed_at": 0.0,
                "status": "teacher_queued",
                "driver": "session",
                "capability": "",
                "policy_action": "",
                "mutation_action": "none",
                "gap_closure": "not_applicable",
                "verification_tier": "teacher_pending",
                "artifact_id": "",
            }
            EvolutionProjectionRunner._append_bounded(
                state["timeline"],
                {
                    "sequence": record.sequence,
                    "title": "session → teacher",
                    "summary": f"{event.context.session_id} finalized",
                    "severity": "info",
                },
            )
        elif event_type == EvolutionEventType.TEACHER_GRADED:
            episode = state["episodes"].setdefault(
                event.context.correlation_id,
                {"episode_id": event.context.correlation_id, "opened_at": event.occurred_at},
            )
            episode.update(
                {
                    "closed_at": event.occurred_at,
                    "status": "graded",
                    "artifact_id": str(payload.get("artifact_id") or ""),
                    "verification_tier": "teacher_hindsight",
                }
            )
        elif event_type == EvolutionEventType.TEACHER_VERDICT_RECORDED:
            action = str(payload.get("action") or "")
            if action in _VERDICT_ACTIONS:
                state["verdict_counts"][action] = int(
                    state["verdict_counts"].get(action) or 0
                ) + 1
            verdict = {
                "verdict_id": str(payload.get("verdict_id") or event.context.decision_id),
                "episode_id": event.context.correlation_id,
                "action": action,
                "capability": str(payload.get("capability") or ""),
                "knowledge": str(payload.get("knowledge") or ""),
                "rationale": str(payload.get("rationale") or ""),
                "confidence": float(payload.get("confidence") or 0.0),
                "target": str(payload.get("target") or ""),
                "artifact_id": event.context.artifact_id,
                "observed_at": event.occurred_at,
            }
            EvolutionProjectionRunner._append_bounded(state["verdicts"], verdict)
            if action == "absorb":
                EvolutionProjectionRunner._upsert_capability(
                    state["knowledge"],
                    {
                        "verdict_id": verdict["verdict_id"],
                        "capability": verdict["capability"],
                        "knowledge": verdict["knowledge"],
                        "confidence": verdict["confidence"],
                        "episode_id": verdict["episode_id"],
                        "created_at": verdict["observed_at"],
                    },
                )
            elif action == "rebind":
                EvolutionProjectionRunner._upsert_capability(
                    state["provider_bindings"],
                    {
                        "verdict_id": verdict["verdict_id"],
                        "capability": verdict["capability"],
                        "plugin_id": verdict["target"],
                        "knowledge": verdict["knowledge"],
                        "confidence": verdict["confidence"],
                        "episode_id": verdict["episode_id"],
                        "created_at": verdict["observed_at"],
                    },
                )
            elif action == "acquire":
                EvolutionProjectionRunner._append_bounded(
                    state["proposal_candidates"],
                    {
                        "capability": verdict["capability"],
                        "confidence": verdict["confidence"],
                        "episode_id": verdict["episode_id"],
                    },
                )
            elif action == "escalate":
                EvolutionProjectionRunner._append_bounded(
                    state["human_escalations"],
                    {
                        "capability": verdict["capability"],
                        "rationale": verdict["rationale"],
                        "confidence": verdict["confidence"],
                        "episode_id": verdict["episode_id"],
                    },
                )
            episode = state["episodes"].setdefault(
                event.context.correlation_id,
                {"episode_id": event.context.correlation_id, "opened_at": event.occurred_at},
            )
            episode.update(
                {
                    "closed_at": event.occurred_at,
                    "status": "committed" if action in {"absorb", "rebind"} else "open",
                    "driver": "world_model",
                    "capability": verdict["capability"],
                    "policy_action": action,
                    "mutation_action": "none",
                    "gap_closure": "not_applicable",
                    "verification_tier": "teacher_hindsight",
                    "artifact_id": event.context.artifact_id,
                }
            )
            EvolutionProjectionRunner._append_bounded(
                state["timeline"],
                {
                    "sequence": record.sequence,
                    "title": f"world_model → {action}",
                    "summary": verdict["capability"] or verdict["knowledge"],
                    "severity": "notable" if action in {"acquire", "escalate"} else "info",
                },
            )
        elif event_type == EvolutionEventType.KNOWLEDGE_RETRACTED:
            capability = str(payload.get("capability") or "")
            state["knowledge"] = [
                item for item in state["knowledge"] if item.get("capability") != capability
            ]
            state["provider_bindings"] = [
                item
                for item in state["provider_bindings"]
                if item.get("capability") != capability
            ]
            EvolutionProjectionRunner._append_bounded(
                state["timeline"],
                {
                    "sequence": record.sequence,
                    "title": "knowledge → retracted",
                    "summary": capability,
                    "severity": "info",
                },
            )
        elif event_type == EvolutionEventType.REQUIREMENT_RESOLVED:
            resolution = {
                "requirement_id": event.context.requirement_id,
                "verdict_id": event.context.decision_id,
                "proposal_id": event.context.proposal_id,
                "capability": str(
                    dict(payload.get("requirement") or {}).get("capability") or ""
                ),
                "outcome": str(payload.get("outcome") or ""),
                "reason": str(payload.get("reason") or ""),
                "selected_plugin_id": str(
                    dict(payload.get("resolution") or {}).get("selected_plugin_id") or ""
                ),
                "selected_tool_name": str(
                    dict(payload.get("resolution") or {}).get("selected_tool_name") or ""
                ),
                "observed_at": event.occurred_at,
            }
            EvolutionProjectionRunner._append_bounded(state["resolutions"], resolution)
            if resolution["outcome"] in {"no_op", "satisfied"}:
                state["no_op_count"] = int(state.get("no_op_count") or 0) + 1
            EvolutionProjectionRunner._append_bounded(
                state["timeline"],
                {
                    "sequence": record.sequence,
                    "title": f"requirement → {resolution['outcome']}",
                    "summary": resolution["capability"] or resolution["reason"],
                    "severity": "info",
                },
            )
        elif event_type == EvolutionEventType.TEACHER_JOB_FAILED:
            state["teacher_failure_count"] = int(state.get("teacher_failure_count") or 0) + 1
            episode = state["episodes"].setdefault(
                event.context.correlation_id,
                {"episode_id": event.context.correlation_id, "opened_at": event.occurred_at},
            )
            episode.update(
                {
                    "closed_at": event.occurred_at,
                    "status": "open" if payload.get("retryable") else "aborted",
                    "verification_tier": "teacher_failed",
                }
            )
            EvolutionProjectionRunner._append_bounded(
                state["timeline"],
                {
                    "sequence": record.sequence,
                    "title": "teacher → failed",
                    "summary": str(payload.get("error_type") or "teacher failure"),
                    "severity": "notable",
                },
            )

    @staticmethod
    def _append_bounded(rows: list[dict[str, Any]], item: dict[str, Any]) -> None:
        rows.append(item)
        if len(rows) > _MAX_ROWS:
            del rows[: len(rows) - _MAX_ROWS]

    @staticmethod
    def _upsert_capability(rows: list[dict[str, Any]], item: dict[str, Any]) -> None:
        capability = str(item.get("capability") or "")
        rows[:] = [row for row in rows if str(row.get("capability") or "") != capability]
        EvolutionProjectionRunner._append_bounded(rows, item)

    @staticmethod
    def _environment_summary(payload: Mapping[str, Any]) -> str:
        kind = str(payload.get("kind") or "environment")
        app_id = str(payload.get("app_id") or "")
        removed = list(payload.get("removed_affordances") or ())
        if removed:
            return f"{app_id}: removed {', '.join(str(item) for item in removed)}"
        outcome = str(payload.get("outcome") or "UNKNOWN")
        if kind == "outcome":
            return f"{app_id}: {outcome}"
        return f"{app_id}: {kind}"

    @staticmethod
    def _present(state: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = copy.deepcopy(dict(state))
        episodes = sorted(
            (dict(item) for item in dict(snapshot.pop("episodes", {})).values()),
            key=lambda item: float(item.get("opened_at") or 0.0),
            reverse=True,
        )
        verdict_counts = dict(snapshot.get("verdict_counts") or {})
        verdicts = list(snapshot.get("verdicts") or [])
        proposals = sorted(
            (dict(item) for item in dict(snapshot.pop("proposals", {})).values()),
            key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
            reverse=True,
        )
        summary = {
            "episode_count": len(episodes),
            "proposal_count": len(proposals),
            "event_count": int(snapshot.get("event_count") or 0),
            "action_count": int(snapshot.get("action_count") or 0),
            "failed_action_count": int(snapshot.get("failed_action_count") or 0),
            "environment_count": int(snapshot.get("environment_count") or 0),
            "verdict_count": sum(int(value or 0) for value in verdict_counts.values()),
            "teacher_failure_count": int(snapshot.get("teacher_failure_count") or 0),
            "no_op_count": int(snapshot.get("no_op_count") or 0),
            "by_action": verdict_counts,
        }
        summary["attention"] = bool(
            summary["failed_action_count"]
            or summary["teacher_failure_count"]
            or verdict_counts.get("escalate")
        )
        snapshot.update(
            {
                "summary": summary,
                "episodes": episodes,
                "proposals": proposals,
                "verdicts": verdicts,
                "mutation_matrix": [
                    {
                        "driver": "world_model",
                        "capability": item.get("capability", ""),
                        "policy_action": item.get("action", ""),
                        "autonomy_level": "proposal_only",
                        "mutation_action": "none",
                        "registry_delta": "0",
                        "lifecycle_status": "not_started",
                        "gap_closure": "not_applicable",
                        "verification_tier": "teacher_hindsight",
                    }
                    for item in verdicts
                ]
                + [
                    {
                        "driver": str(item.get("source") or "proposal"),
                        "capability": str(
                            (item.get("requirements") or [{}])[0].get("capability") or ""
                        ),
                        "policy_action": str(
                            dict(item.get("policy_decision") or {}).get("action") or ""
                        ),
                        "autonomy_level": str(
                            dict(item.get("policy_decision") or {}).get("autonomy_level")
                            or ""
                        ),
                        "mutation_action": "install"
                        if item.get("status") in {"INSTALLED", "PROBATION", "VERIFIED"}
                        else "none",
                        "registry_delta": "1"
                        if item.get("status") in {"INSTALLED", "PROBATION", "VERIFIED"}
                        else "0",
                        "lifecycle_status": str(item.get("status") or ""),
                        "gap_closure": "closed"
                        if item.get("status") == "VERIFIED"
                        else "open",
                        "verification_tier": str(
                            dict(item.get("trust_state") or {}).get("level") or "DRAFT"
                        ),
                    }
                    for item in proposals
                ],
                "degraded": not episodes,
            }
        )
        return snapshot


__all__ = ["EvolutionProjectionRunner", "ProjectionMetrics", "ProjectionStore"]

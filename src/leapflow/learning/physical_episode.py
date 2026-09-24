# Copyright (c) Alibaba, Inc. and its affiliates.
"""Physical episode recording for experience learning.

An episode captures a complete physical manipulation attempt: the initial
state, the sequence of actions taken, intermediate observations, the final
state, and the verification verdict.  Episodes feed the evolution engine
(so the agent learns which manipulations succeed) and can be exported for
offline policy training.

Distinct from PhysicalTrajectory (which is raw demonstration data): an
episode adds the outcome dimension — what was attempted, what happened,
and whether it worked.

Persistence follows the EvidenceStore pattern: DuckDB table for structured
metadata, ``asyncio.to_thread`` for non-blocking writes, and a
``ConnectionHolder`` for shared access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder

logger = logging.getLogger(__name__)

EPISODE_SCHEMA_VERSION = 1
"""Row format version for the physical_episodes table."""

DEFAULT_RETENTION_DAYS = 90.0
"""How long episode rows survive before cleanup."""


# ---------------------------------------------------------------------------
# PhysicalEpisode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalEpisode:
    """A complete physical manipulation episode with outcome.

    ``trajectory`` is a :class:`PhysicalTrajectory` instance captured
    during the episode.  ``verdict`` is an optional
    :class:`OperationVerdict` produced by the verification layer.
    ``evidence_ids`` reference :class:`EvidenceBundle` records stored
    in the :class:`EvidenceStore`.
    """

    episode_id: str
    device_id: str
    task: str
    trajectory: Any  # PhysicalTrajectory
    initial_state: Mapping[str, Any]
    final_state: Mapping[str, Any]
    verdict: Any = None  # OperationVerdict or None
    evidence_ids: tuple[str, ...] = ()
    success: bool = False
    started_at: float = 0.0
    ended_at: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        if self.ended_at > self.started_at:
            return self.ended_at - self.started_at
        return 0.0

    @property
    def step_count(self) -> int:
        """Number of trajectory steps."""
        traj = self.trajectory
        if traj is not None:
            steps = getattr(traj, "steps", ())
            return len(steps)
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON persistence."""
        traj_dict: dict[str, Any] | None = None
        if self.trajectory is not None:
            to_dict_fn = getattr(self.trajectory, "to_dict", None)
            traj_dict = to_dict_fn() if callable(to_dict_fn) else None

        verdict_dict: dict[str, Any] | None = None
        if self.verdict is not None:
            to_dict_fn = getattr(self.verdict, "to_dict", None)
            verdict_dict = to_dict_fn() if callable(to_dict_fn) else None

        return {
            "episode_id": self.episode_id,
            "device_id": self.device_id,
            "task": self.task,
            "trajectory": traj_dict,
            "initial_state": dict(self.initial_state),
            "final_state": dict(self.final_state),
            "verdict": verdict_dict,
            "evidence_ids": list(self.evidence_ids),
            "success": self.success,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "step_count": self.step_count,
            "metadata": dict(self.metadata),
        }


def make_episode_id() -> str:
    """Generate a short unique episode identifier."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# PhysicalEpisodeRecorder
# ---------------------------------------------------------------------------


class PhysicalEpisodeRecorder:
    """Records physical manipulation episodes and feeds the evolution engine.

    Lifecycle::

        episode_id = await recorder.begin_episode(device_id, task)
        recorder.record_step(step)          # repeat for each step
        recorder.record_evidence(eid)       # optional, link evidence bundles
        episode = await recorder.end_episode(verdict)

    Dependencies:

    - ``registry``: HardwareRegistry (for state capture via ``read_batch``)
    - ``evidence_store``: :class:`EvidenceStore` (optional, for evidence linking)
    - ``action_recorder``: evolution :class:`ActionRecorder` (optional, for
      feeding physical episode outcomes into the evolution causal chain)
    - ``db_path``: DuckDB path for episode persistence; when ``None`` episodes
      are returned but not persisted.
    """

    def __init__(
        self,
        registry: Any,
        *,
        evidence_store: Any = None,
        action_recorder: Any = None,
        db_path: str | Path | None = None,
        connection_holder: ConnectionHolder | None = None,
    ) -> None:
        self._registry = registry
        self._evidence_store = evidence_store
        self._action_recorder = action_recorder

        # DuckDB persistence (optional).
        self._holder: ConnectionHolder | None = connection_holder
        self._owns_holder = False
        if self._holder is None and db_path is not None:
            self._holder = LocalConnectionHolder(Path(db_path))
            self._owns_holder = True
        self._db_ready = False

        # In-flight episode state (mutable during recording).
        self._current_device_id: str | None = None
        self._current_task: str | None = None
        self._current_episode_id: str | None = None
        self._current_started_at: float = 0.0
        self._current_initial_state: dict[str, Any] = {}
        self._current_steps: list[Any] = []
        self._current_evidence_ids: list[str] = []

        # Stats.
        self._episodes_recorded = 0

    @property
    def is_recording(self) -> bool:
        """Whether an episode is currently being recorded."""
        return self._current_episode_id is not None

    @property
    def episodes_recorded(self) -> int:
        """Total episodes successfully recorded."""
        return self._episodes_recorded

    # -- Lifecycle ------------------------------------------------------------

    async def begin_episode(self, device_id: str, task: str) -> str:
        """Start recording; capture initial state via ``read_batch``.

        Returns the episode_id.

        Raises:
            RuntimeError: If an episode is already being recorded.
        """
        if self._current_episode_id is not None:
            raise RuntimeError(
                f"Episode {self._current_episode_id} already in progress; "
                "call end_episode() first."
            )

        episode_id = make_episode_id()
        initial_state = await self._capture_state(device_id)

        self._current_episode_id = episode_id
        self._current_device_id = device_id
        self._current_task = task
        self._current_started_at = time.time()
        self._current_initial_state = initial_state
        self._current_steps = []
        self._current_evidence_ids = []

        logger.debug(
            "Episode %s started: device=%s, task=%r",
            episode_id, device_id, task,
        )
        return episode_id

    def record_step(self, step: Any) -> None:
        """Append a :class:`PhysicalTrajectoryStep` to the current episode.

        Raises:
            RuntimeError: If no episode is in progress.
        """
        if self._current_episode_id is None:
            raise RuntimeError("No episode in progress; call begin_episode() first.")
        self._current_steps.append(step)

    def record_evidence(self, evidence_id: str) -> None:
        """Link an :class:`EvidenceBundle` id to the current episode.

        Raises:
            RuntimeError: If no episode is in progress.
        """
        if self._current_episode_id is None:
            raise RuntimeError("No episode in progress; call begin_episode() first.")
        self._current_evidence_ids.append(evidence_id)

    async def end_episode(
        self,
        verdict: Any = None,
        *,
        success: bool | None = None,
    ) -> PhysicalEpisode:
        """Capture final state, build episode, persist, and feed evolution.

        Args:
            verdict: Optional :class:`OperationVerdict` from verification.
            success: Explicit success flag.  When ``None``, inferred from
                *verdict* (``is_success`` property) or defaults to ``False``.

        Returns:
            The finalized :class:`PhysicalEpisode`.

        Raises:
            RuntimeError: If no episode is in progress.
        """
        if self._current_episode_id is None:
            raise RuntimeError("No episode in progress; call begin_episode() first.")

        device_id = self._current_device_id or ""
        final_state = await self._capture_state(device_id)
        ended_at = time.time()

        # Determine success.
        if success is None:
            if verdict is not None:
                success = bool(getattr(verdict, "is_success", False))
            else:
                success = False

        # Build PhysicalTrajectory from collected steps.
        from leapflow.robot.trajectory import PhysicalTrajectory

        trajectory = PhysicalTrajectory(
            trajectory_id=self._current_episode_id,
            device_id=device_id,
            goal=self._current_task or "",
            steps=tuple(self._current_steps),
            started_at=self._current_started_at,
            ended_at=ended_at,
        )

        episode = PhysicalEpisode(
            episode_id=self._current_episode_id,
            device_id=device_id,
            task=self._current_task or "",
            trajectory=trajectory,
            initial_state=dict(self._current_initial_state),
            final_state=dict(final_state),
            verdict=verdict,
            evidence_ids=tuple(self._current_evidence_ids),
            success=success,
            started_at=self._current_started_at,
            ended_at=ended_at,
        )

        # Reset in-flight state before async operations.
        self._current_episode_id = None
        self._current_device_id = None
        self._current_task = None
        self._current_started_at = 0.0
        self._current_initial_state = {}
        self._current_steps = []
        self._current_evidence_ids = []

        # Persist to DuckDB (best-effort).
        await self._persist_episode(episode)

        # Feed evolution engine (best-effort).
        await self._feed_evolution(episode)

        self._episodes_recorded += 1
        logger.debug(
            "Episode %s completed: success=%s, steps=%d, duration=%.2fs",
            episode.episode_id, episode.success,
            episode.step_count, episode.duration_s,
        )
        return episode

    # -- Query ----------------------------------------------------------------

    async def query_episodes(
        self,
        *,
        device_id: str | None = None,
        task: str | None = None,
        success_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recorded episodes by device/task/success."""
        return await asyncio.to_thread(
            self._query_sync, device_id, task, success_only, limit
        )

    def _query_sync(
        self,
        device_id: str | None,
        task: str | None,
        success_only: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Blocking query implementation."""
        if self._holder is None:
            return []
        try:
            connection = self._ensure_ready()
        except Exception:  # noqa: BLE001
            logger.debug("Could not open episode store for query", exc_info=True)
            return []

        conditions = [f"schema_version = {EPISODE_SCHEMA_VERSION}"]
        params: list[Any] = []
        if device_id is not None:
            conditions.append("device_id = ?")
            params.append(device_id)
        if task is not None:
            conditions.append("task = ?")
            params.append(task)
        if success_only:
            conditions.append("success = true")

        where = " AND ".join(conditions)
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM physical_episodes "
            f"WHERE {where} ORDER BY started_at DESC LIMIT ?"
        )
        params.append(int(limit))

        try:
            rows = connection.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Episode query failed: %s", exc)
            return []
        return [dict(zip(_COLUMNS, row)) for row in rows]

    # -- Internal: state capture ----------------------------------------------

    async def _capture_state(self, device_id: str) -> dict[str, Any]:
        """Capture the current state of a device via ``read_batch``.

        Returns a channel-id → value mapping.  Returns an empty dict when
        the registry or device is unavailable (state capture must never
        fail the episode lifecycle).
        """
        state: dict[str, Any] = {}
        try:
            context = self._registry.context(device_id)
            if context is None:
                return state
            channels = tuple(
                ch.channel_id for ch in context.channels if ch.is_readable
            )
            if not channels:
                return state
            batch = await self._registry.read_batch(device_id, channels)
            for reading in batch.readings:
                state[reading.channel_id] = reading.value
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "State capture failed for %s: %s", device_id, exc,
            )
        return state

    # -- Internal: DuckDB persistence -----------------------------------------

    def _ensure_schema(self) -> None:
        """Create the physical_episodes table if it does not exist."""
        if self._holder is None:
            return
        connection = self._holder.connection
        connection.execute(_SCHEMA)
        try:
            connection.execute(_INDEX_DEVICE_TASK)
        except Exception:  # noqa: BLE001
            logger.debug("physical_episodes index unavailable", exc_info=True)

    def _ensure_ready(self) -> Any:
        """Ensure schema exists and return the connection."""
        if self._holder is None:
            raise RuntimeError("PhysicalEpisodeRecorder has no connection holder")
        connection = self._holder.connection
        if not self._db_ready:
            self._ensure_schema()
            self._db_ready = True
        return connection

    async def _persist_episode(self, episode: PhysicalEpisode) -> None:
        """Write an episode row to DuckDB (best-effort)."""
        if self._holder is None:
            return
        await asyncio.to_thread(self._write_row, episode)

    def _write_row(self, episode: PhysicalEpisode) -> None:
        """Blocking DuckDB insert, safe to call from a worker thread."""
        try:
            connection = self._ensure_ready()
            verdict_status: str | None = None
            verdict_confidence: float | None = None
            if episode.verdict is not None:
                verdict_status = getattr(episode.verdict, "status", None)
                verdict_confidence = getattr(episode.verdict, "confidence", None)

            trajectory_json = json.dumps(
                episode.trajectory.to_dict() if episode.trajectory is not None else {},
                ensure_ascii=False,
                default=str,
            )
            initial_json = json.dumps(
                dict(episode.initial_state), ensure_ascii=False, default=str,
            )
            final_json = json.dumps(
                dict(episode.final_state), ensure_ascii=False, default=str,
            )
            evidence_json = json.dumps(list(episode.evidence_ids))
            metadata_json = json.dumps(
                dict(episode.metadata), ensure_ascii=False, default=str,
            )

            row = (
                episode.episode_id,
                episode.device_id,
                episode.task,
                episode.success,
                episode.started_at,
                episode.ended_at,
                episode.step_count,
                verdict_status,
                verdict_confidence,
                trajectory_json,
                initial_json,
                final_json,
                evidence_json,
                metadata_json,
                EPISODE_SCHEMA_VERSION,
            )
            connection.execute(_INSERT, row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist episode: %s", exc)

    # -- Internal: evolution feed ---------------------------------------------

    async def _feed_evolution(self, episode: PhysicalEpisode) -> None:
        """Best-effort: publish the episode outcome to the evolution engine.

        Uses the ActionRecorder's outbox to publish a lightweight evolution
        event carrying the episode outcome as capability observation evidence.
        Skipped silently when ``action_recorder`` is ``None`` or when any
        step fails.
        """
        if self._action_recorder is None:
            return
        try:
            outbox = getattr(self._action_recorder, "_outbox", None)
            if outbox is None or not hasattr(outbox, "publish"):
                return

            # Function-local imports to avoid hard coupling.
            from leapflow.domain.event_types import EvolutionEventType
            from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent

            event_type = (
                EvolutionEventType.ACTION_COMPLETED
                if episode.success
                else EvolutionEventType.ACTION_FAILED
            )
            context = EvolutionContext.create(
                correlation_id=f"episode-{episode.episode_id}",
                action_id=episode.episode_id,
            )
            event = EvolutionEvent.create(
                event_type,
                context=context,
                payload={
                    "action_type": "physical_episode",
                    "action_name": episode.task,
                    "ok": episode.success,
                    "device_id": episode.device_id,
                    "step_count": episode.step_count,
                    "duration_s": episode.duration_s,
                    "episode_id": episode.episode_id,
                },
                producer="robot.episode_recorder",
                dedup_key=f"episode.{episode.episode_id}",
            )
            await outbox.publish(event, critical=False)
            logger.debug(
                "Episode %s fed to evolution as %s",
                episode.episode_id, event_type,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Evolution feed for episode %s failed",
                episode.episode_id, exc_info=True,
            )

    # -- Cleanup --------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection if owned."""
        if self._owns_holder and self._holder is not None:
            try:
                self._holder.close()
            except Exception:  # noqa: BLE001
                logger.debug("Episode store holder close failed", exc_info=True)


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_COLUMNS = (
    "episode_id",
    "device_id",
    "task",
    "success",
    "started_at",
    "ended_at",
    "step_count",
    "verdict_status",
    "verdict_confidence",
    "trajectory_json",
    "initial_state_json",
    "final_state_json",
    "evidence_ids_json",
    "metadata_json",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS physical_episodes (
    episode_id          VARCHAR PRIMARY KEY,
    device_id           VARCHAR NOT NULL,
    task                VARCHAR NOT NULL,
    success             BOOLEAN NOT NULL DEFAULT false,
    started_at          DOUBLE NOT NULL,
    ended_at            DOUBLE NOT NULL,
    step_count          INTEGER NOT NULL DEFAULT 0,
    verdict_status      VARCHAR,
    verdict_confidence  DOUBLE,
    trajectory_json     VARCHAR,
    initial_state_json  VARCHAR,
    final_state_json    VARCHAR,
    evidence_ids_json   VARCHAR,
    metadata_json       VARCHAR,
    schema_version      INTEGER DEFAULT 1,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT = """
INSERT OR REPLACE INTO physical_episodes (
    episode_id, device_id, task, success,
    started_at, ended_at, step_count,
    verdict_status, verdict_confidence,
    trajectory_json, initial_state_json, final_state_json,
    evidence_ids_json, metadata_json, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INDEX_DEVICE_TASK = """
CREATE INDEX IF NOT EXISTS idx_episode_device_task
ON physical_episodes (device_id, task, started_at)
"""


__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "EPISODE_SCHEMA_VERSION",
    "PhysicalEpisode",
    "PhysicalEpisodeRecorder",
    "make_episode_id",
]

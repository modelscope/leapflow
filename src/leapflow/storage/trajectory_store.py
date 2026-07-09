"""DuckDB-backed trajectory persistence.

Follows the same patterns as memory/long_term.py: single DuckDB connection,
SQL schema auto-creation, typed query helpers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from leapflow.domain.trajectory import (
    ActionType,
    Episode,
    RawAction,
    SemanticAction,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import WriteBuffer, execute_with_retry

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """Append-oriented store for trajectories, steps, and episodes.

    Write failures are buffered in memory and retried on the next successful
    write, preventing data loss during transient DuckDB errors.

    Accepts either a ``ConnectionHolder`` (shared connection) or a legacy
    ``Path`` (auto-wrapped in ``LocalConnectionHolder``).
    """

    def __init__(self, source: Union[ConnectionHolder, Path]) -> None:
        self._owns_holder = isinstance(source, Path)
        if self._owns_holder:
            source = LocalConnectionHolder(source)
        self._holder = source
        self._con = self._holder.connection
        self._write_buffer = WriteBuffer(self._con)
        self._init_schema()

    def close(self) -> None:
        self._write_buffer.flush()
        if self._owns_holder:
            self._holder.close()

    # ── Schema ──

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_trajectory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time DOUBLE NOT NULL,
                end_time DOUBLE NOT NULL,
                step_count INTEGER NOT NULL,
                metadata TEXT,
                created_at DOUBLE NOT NULL
            )
        """)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_trajectory_step (
                trajectory_id TEXT NOT NULL,
                step_idx INTEGER NOT NULL,
                timestamp DOUBLE NOT NULL,
                action_type TEXT NOT NULL,
                target TEXT,
                target_label TEXT,
                target_role TEXT,
                app_bundle_id TEXT,
                app_name TEXT,
                params TEXT,
                state_focused_app TEXT,
                state_ax_digest TEXT,
                state_clipboard TEXT,
                visual_frame_ref TEXT,
                state_ax_tree TEXT,
                state_snapshot_level TEXT,
                PRIMARY KEY (trajectory_id, step_idx)
            )
        """)
        # Schema migration for existing databases
        self._migrate_step_table()
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS leap_episode (
                id TEXT PRIMARY KEY,
                trajectory_id TEXT NOT NULL,
                start_idx INTEGER NOT NULL,
                end_idx INTEGER NOT NULL,
                inferred_goal TEXT,
                app_sequence TEXT,
                semantic_actions TEXT,
                confidence DOUBLE,
                created_at DOUBLE NOT NULL
            )
        """)
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_time "
            "ON leap_trajectory(start_time)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_traj_user "
            "ON leap_trajectory(user_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_traj "
            "ON leap_trajectory_step(trajectory_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_action "
            "ON leap_trajectory_step(action_type)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_episode_traj "
            "ON leap_episode(trajectory_id)"
        )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_episode_goal "
            "ON leap_episode(inferred_goal)"
        )
        self._migrate_episode_table()

    def _migrate_step_table(self) -> None:
        """Add columns introduced by Enhanced StateSnapshot (backward-compatible)."""
        cols = {
            r[0]
            for r in self._con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'leap_trajectory_step'"
            ).fetchall()
        }
        if "state_ax_tree" not in cols:
            self._con.execute(
                "ALTER TABLE leap_trajectory_step ADD COLUMN state_ax_tree TEXT"
            )
        if "state_snapshot_level" not in cols:
            self._con.execute(
                "ALTER TABLE leap_trajectory_step ADD COLUMN state_snapshot_level TEXT"
            )

    def _migrate_episode_table(self) -> None:
        """Add procedure_graph column to existing episode tables."""
        try:
            cols = {
                r[0]
                for r in self._con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'leap_episode'"
                ).fetchall()
            }
        except Exception:
            return
        if "procedure_graph" not in cols:
            self._con.execute(
                "ALTER TABLE leap_episode ADD COLUMN procedure_graph TEXT"
            )

    # ── Trajectory CRUD ──

    def save_trajectory(self, traj: Trajectory) -> None:
        """Persist a complete trajectory with all its steps."""
        now = time.time()
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_trajectory VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                traj.trajectory_id,
                traj.user_id,
                traj.start_time,
                traj.end_time,
                traj.step_count,
                json.dumps(traj.metadata, ensure_ascii=False),
                now,
            ],
        )
        for idx, step in enumerate(traj.steps):
            execute_with_retry(
                self._con,
                "INSERT OR REPLACE INTO leap_trajectory_step VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._step_params(traj.trajectory_id, idx, step),
            )
        logger.info(
            "trajectory.saved id=%s steps=%d duration=%.1fs",
            traj.trajectory_id,
            traj.step_count,
            traj.duration,
        )

    def append_step(self, trajectory_id: str, step_idx: int, step: TrajectoryStep) -> None:
        """Append a single step (immediate write, buffered on failure)."""
        sql = (
            "INSERT OR REPLACE INTO leap_trajectory_step VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = self._step_params(trajectory_id, step_idx, step)
        try:
            execute_with_retry(self._con, sql, params)
            self._write_buffer.flush()
        except Exception as exc:
            logger.warning("store.append_step buffered: %s", exc)
            self._write_buffer.append("step", sql, params)

    def finalize_trajectory(self, traj: Trajectory) -> None:
        """Update trajectory header (immediate write, buffered on failure)."""
        sql = "INSERT OR REPLACE INTO leap_trajectory VALUES (?, ?, ?, ?, ?, ?, ?)"
        params = [
            traj.trajectory_id,
            traj.user_id,
            traj.start_time,
            traj.end_time,
            traj.step_count,
            json.dumps(traj.metadata, ensure_ascii=False),
            time.time(),
        ]
        try:
            execute_with_retry(self._con, sql, params)
            self._write_buffer.flush()
        except Exception as exc:
            logger.warning("store.finalize buffered: %s", exc)
            self._write_buffer.append("finalize", sql, params)

    def load_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Load a trajectory with all its steps."""
        rows = self._con.execute(
            "SELECT id, user_id, start_time, end_time, step_count, metadata "
            "FROM leap_trajectory WHERE id = ?",
            [trajectory_id],
        ).fetchall()
        if not rows:
            return None
        row = rows[0]
        traj = Trajectory(
            trajectory_id=row[0],
            user_id=row[1],
            start_time=row[2],
            end_time=row[3],
            metadata=_parse_json(row[5]),
        )
        traj.steps = self._load_steps(trajectory_id)
        return traj

    def list_trajectories(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """List trajectory summaries (without loading steps)."""
        clauses: List[str] = []
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if since is not None:
            clauses.append("start_time >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._con.execute(
            f"SELECT id, user_id, start_time, end_time, step_count, metadata "
            f"FROM leap_trajectory{where} "
            f"ORDER BY start_time DESC LIMIT {int(limit)}",
            params,
        ).fetchall()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "start_time": r[2],
                "end_time": r[3],
                "step_count": r[4],
                "metadata": _parse_json(r[5]),
            }
            for r in rows
        ]

    # ── Episode CRUD ──

    def save_episode(self, episode: Episode) -> None:
        """Persist an episode."""
        execute_with_retry(
            self._con,
            "INSERT OR REPLACE INTO leap_episode "
            "(id, trajectory_id, start_idx, end_idx, inferred_goal, "
            "app_sequence, semantic_actions, confidence, created_at, procedure_graph) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                episode.episode_id,
                episode.trajectory_id,
                episode.start_idx,
                episode.end_idx,
                episode.inferred_goal,
                json.dumps(episode.app_sequence, ensure_ascii=False),
                json.dumps(
                    [_semantic_action_to_dict(a) for a in episode.semantic_actions],
                    ensure_ascii=False,
                ),
                episode.confidence,
                time.time(),
                episode.procedure_graph or None,
            ],
        )

    def delete_episodes(self, trajectory_id: str) -> int:
        """Delete all episodes for a trajectory. Returns count deleted."""
        count = self._con.execute(
            "SELECT COUNT(*) FROM leap_episode WHERE trajectory_id = ?",
            [trajectory_id],
        ).fetchone()[0]
        if count > 0:
            execute_with_retry(
                self._con,
                "DELETE FROM leap_episode WHERE trajectory_id = ?",
                [trajectory_id],
            )
        return count

    def load_episodes(self, trajectory_id: str) -> List[Episode]:
        """Load all episodes for a trajectory."""
        rows = self._con.execute(
            "SELECT id, trajectory_id, start_idx, end_idx, inferred_goal, "
            "app_sequence, semantic_actions, confidence, procedure_graph "
            "FROM leap_episode WHERE trajectory_id = ? ORDER BY start_idx",
            [trajectory_id],
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    # ── Search ──

    def search_episodes_by_goal(
        self, keywords: Sequence[str], *, limit: int = 20
    ) -> List[Episode]:
        """Find episodes whose inferred goal matches keywords."""
        if not keywords:
            return []
        clauses = " AND ".join(["inferred_goal ILIKE ?" for _ in keywords])
        params = [f"%{k}%" for k in keywords]
        rows = self._con.execute(
            f"SELECT id, trajectory_id, start_idx, end_idx, inferred_goal, "
            f"app_sequence, semantic_actions, confidence, procedure_graph "
            f"FROM leap_episode WHERE {clauses} "
            f"ORDER BY confidence DESC LIMIT {int(limit)}",
            params,
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    # ── Internal helpers ──

    @staticmethod
    def _step_params(trajectory_id: str, idx: int, step: TrajectoryStep) -> list:
        return [
            trajectory_id,
            idx,
            step.action.timestamp,
            step.action.action_type.value,
            step.action.target,
            step.action.target_label,
            step.action.target_role,
            step.action.app_bundle_id,
            step.action.app_name,
            json.dumps(step.action.params, ensure_ascii=False) if step.action.params else None,
            step.state.focused_app,
            step.state.ax_tree_digest,
            step.state.clipboard_text,
            step.state.visual_frame_ref,
            json.dumps(step.state.ax_tree_snapshot, ensure_ascii=False)
            if step.state.ax_tree_snapshot
            else None,
            step.state.snapshot_level,
        ]

    def _load_steps(self, trajectory_id: str) -> List[TrajectoryStep]:
        rows = self._con.execute(
            "SELECT step_idx, timestamp, action_type, target, target_label, target_role, "
            "app_bundle_id, app_name, params, state_focused_app, state_ax_digest, "
            "state_clipboard, visual_frame_ref, state_ax_tree, state_snapshot_level "
            "FROM leap_trajectory_step WHERE trajectory_id = ? ORDER BY step_idx",
            [trajectory_id],
        ).fetchall()
        return [_row_to_step(r) for r in rows]


# ── Row conversion helpers ──


def _row_to_step(r: tuple) -> TrajectoryStep:
    try:
        at = ActionType(r[2])
    except ValueError:
        at = ActionType.UNKNOWN
    return TrajectoryStep(
        state=StateSnapshot(
            timestamp=r[1],
            focused_app=r[9] or "",
            ax_tree_digest=r[10] or "",
            clipboard_text=r[11],
            visual_frame_ref=r[12],
            ax_tree_snapshot=_parse_json(r[13]) if r[13] else None,
            snapshot_level=r[14] or "light",
        ),
        action=RawAction(
            timestamp=r[1],
            action_type=at,
            target=r[3] or "",
            target_label=r[4] or "",
            target_role=r[5] or "",
            app_bundle_id=r[6] or "",
            app_name=r[7] or "",
            params=_parse_json(r[8]),
        ),
    )


def _row_to_episode(r: tuple) -> Episode:
    return Episode(
        episode_id=r[0],
        trajectory_id=r[1],
        start_idx=r[2],
        end_idx=r[3],
        inferred_goal=r[4] or "",
        app_sequence=_parse_json(r[5]) if r[5] else [],
        semantic_actions=[_dict_to_semantic_action(d) for d in (_parse_json(r[6]) or [])],
        confidence=r[7] or 0.0,
        procedure_graph=r[8] if len(r) > 8 and r[8] else "",
    )


def _semantic_action_to_dict(a: SemanticAction) -> Dict[str, Any]:
    return {
        "action_name": a.action_name,
        "description": a.description,
        "parameters": a.parameters,
        "raw_action_range": list(a.raw_action_range),
        "confidence": a.confidence,
    }


def _dict_to_semantic_action(d: Any) -> SemanticAction:
    if not isinstance(d, dict):
        return SemanticAction(action_name="unknown", description="")
    rng = d.get("raw_action_range", [0, 0])
    return SemanticAction(
        action_name=d.get("action_name", ""),
        description=d.get("description", ""),
        parameters=d.get("parameters", {}),
        raw_action_range=(rng[0], rng[1]) if len(rng) >= 2 else (0, 0),
        confidence=d.get("confidence", 0.0),
    )


def _parse_json(val: Any) -> Any:
    if val is None:
        return {}
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return {}
    return {}

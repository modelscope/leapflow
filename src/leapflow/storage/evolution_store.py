"""DuckDB persistence for EvolutionMemoryProvider — skill episodes survive restart.

Design:
- Write-behind: buffer episodes in-memory, flush to DuckDB periodically or on shutdown
- Read-through: on initialize, load recent episodes from DuckDB into in-memory provider
- Schema is simple: one table with JSON-serialized episode data
- Idempotent schema creation (no migration chains)
- Write-retry with jitter for concurrent access

This module does NOT replace EvolutionMemoryProvider — it augments it with persistence.
The provider remains the source-of-truth for hot data; DuckDB is cold storage.
"""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WRITE_RETRIES = 10
_WRITE_JITTER_MS = (15, 120)


class DuckDBEvolutionStore:
    """Persistent backing store for skill episodes."""

    def __init__(self, db_path: Path | str) -> None:
        import duckdb
        self._db_path = str(db_path)
        self._conn = duckdb.connect(self._db_path)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_episodes (
                episode_id VARCHAR PRIMARY KEY,
                skill_name VARCHAR NOT NULL,
                actions_json VARCHAR DEFAULT '[]',
                outcome VARCHAR DEFAULT '',
                reward DOUBLE DEFAULT 0.0,
                context_json VARCHAR DEFAULT '{}',
                created_at DOUBLE DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_skill
            ON skill_episodes (skill_name, created_at DESC)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS skill_patterns (
                pattern_id VARCHAR PRIMARY KEY,
                skill_name VARCHAR NOT NULL,
                pattern_json VARCHAR DEFAULT '{}',
                confidence DOUBLE DEFAULT 0.0,
                episode_count INTEGER DEFAULT 0,
                created_at DOUBLE DEFAULT 0.0
            )
        """)

    def _execute_write(self, sql: str, params: Any = None) -> None:
        for attempt in range(_WRITE_RETRIES):
            try:
                if params:
                    self._conn.execute(sql, params)
                else:
                    self._conn.execute(sql)
                return
            except Exception as e:
                if "locked" in str(e).lower() and attempt < _WRITE_RETRIES - 1:
                    jitter = random.uniform(*_WRITE_JITTER_MS) / 1000
                    time.sleep(jitter)
                    continue
                raise

    def save_episode(
        self,
        episode_id: str,
        skill_name: str,
        actions: List[Dict[str, Any]],
        outcome: str,
        reward: float,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Persist a single skill episode."""
        now = timestamp or time.time()
        self._execute_write(
            """
            INSERT INTO skill_episodes
                (episode_id, skill_name, actions_json, outcome, reward, context_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (episode_id) DO UPDATE SET
                reward = EXCLUDED.reward,
                outcome = EXCLUDED.outcome
            """,
            [
                episode_id, skill_name,
                json.dumps(actions, ensure_ascii=False, default=str),
                outcome, reward,
                json.dumps(context or {}, ensure_ascii=False, default=str),
                now,
            ],
        )

    def load_recent_episodes(
        self, *, limit: int = 500, min_reward: float = -1.0
    ) -> List[Dict[str, Any]]:
        """Load recent episodes for in-memory provider hydration."""
        rows = self._conn.execute(
            """
            SELECT episode_id, skill_name, actions_json, outcome, reward, context_json, created_at
            FROM skill_episodes
            WHERE reward >= ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [min_reward, limit],
        ).fetchall()

        episodes = []
        for row in rows:
            try:
                actions = json.loads(row[2]) if row[2] else []
            except (json.JSONDecodeError, TypeError):
                actions = []
            try:
                context = json.loads(row[5]) if row[5] else {}
            except (json.JSONDecodeError, TypeError):
                context = {}
            episodes.append({
                "episode_id": row[0],
                "skill_name": row[1],
                "actions": actions,
                "outcome": row[3] or "",
                "reward": row[4] or 0.0,
                "context": context,
                "timestamp": row[6] or 0.0,
            })
        return episodes

    def save_pattern(
        self,
        pattern_id: str,
        skill_name: str,
        pattern: Dict[str, Any],
        confidence: float,
        episode_count: int,
    ) -> None:
        """Persist a generalized skill pattern."""
        self._execute_write(
            """
            INSERT INTO skill_patterns
                (pattern_id, skill_name, pattern_json, confidence, episode_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (pattern_id) DO UPDATE SET
                pattern_json = EXCLUDED.pattern_json,
                confidence = EXCLUDED.confidence,
                episode_count = EXCLUDED.episode_count
            """,
            [
                pattern_id, skill_name,
                json.dumps(pattern, ensure_ascii=False, default=str),
                confidence, episode_count, time.time(),
            ],
        )

    def prune_old_episodes(self, *, max_age_days: float = 90.0) -> int:
        """Delete episodes older than max_age_days. Returns count deleted."""
        cutoff = time.time() - (max_age_days * 86400)
        before = self._conn.execute(
            "SELECT COUNT(*) FROM skill_episodes WHERE created_at < ?", [cutoff]
        ).fetchone()[0]
        if before > 0:
            self._execute_write(
                "DELETE FROM skill_episodes WHERE created_at < ?", [cutoff]
            )
        return before

    def episode_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM skill_episodes").fetchone()
        return row[0] if row else 0

    def integrity_check(self) -> bool:
        """Verify DuckDB health by running a probe query."""
        try:
            self._conn.execute("SELECT COUNT(*) FROM skill_episodes").fetchone()
            return True
        except Exception as exc:
            logger.warning("evolution_store: integrity check failed: %s", exc)
            return False

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

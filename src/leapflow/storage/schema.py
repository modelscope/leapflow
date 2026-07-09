"""Unified DuckDB schema definition and migration for leap.duckdb.

Single source of truth for all table schemas. Each store registers its
schema here rather than running ad-hoc CREATE TABLE in its own __init__.

Migration is version-tracked via a ``_schema_version`` table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import duckdb

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TableDef:
    """Declarative table definition."""
    name: str
    ddl: str
    indexes: List[str] = field(default_factory=list)


TABLES: List[TableDef] = [
    # ── Memory ──
    TableDef(
        name="mem_entries",
        ddl="""
        CREATE TABLE IF NOT EXISTS mem_entries (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT 'system',
            content TEXT NOT NULL,
            path TEXT,
            metadata TEXT,
            created_at DOUBLE NOT NULL,
            accessed_at DOUBLE NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 1,
            workspace_id TEXT DEFAULT '',
            session_id TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_mem_created ON mem_entries(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_mem_kind ON mem_entries(kind)",
            "CREATE INDEX IF NOT EXISTS idx_mem_domain ON mem_entries(domain)",
        ],
    ),
    # ── Trajectory ──
    TableDef(
        name="traj_headers",
        ddl="""
        CREATE TABLE IF NOT EXISTS traj_headers (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time DOUBLE NOT NULL,
            end_time DOUBLE NOT NULL,
            step_count INTEGER NOT NULL,
            metadata TEXT,
            created_at DOUBLE NOT NULL,
            workspace_id TEXT DEFAULT '',
            session_id TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_traj_time ON traj_headers(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_traj_user ON traj_headers(user_id)",
        ],
    ),
    TableDef(
        name="traj_steps",
        ddl="""
        CREATE TABLE IF NOT EXISTS traj_steps (
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
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_tstep_traj ON traj_steps(trajectory_id)",
            "CREATE INDEX IF NOT EXISTS idx_tstep_action ON traj_steps(action_type)",
        ],
    ),
    TableDef(
        name="traj_episodes",
        ddl="""
        CREATE TABLE IF NOT EXISTS traj_episodes (
            id TEXT PRIMARY KEY,
            trajectory_id TEXT NOT NULL,
            start_idx INTEGER NOT NULL,
            end_idx INTEGER NOT NULL,
            inferred_goal TEXT,
            app_sequence TEXT,
            semantic_actions TEXT,
            confidence DOUBLE,
            created_at DOUBLE NOT NULL,
            procedure_graph TEXT
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_ep_traj ON traj_episodes(trajectory_id)",
            "CREATE INDEX IF NOT EXISTS idx_ep_goal ON traj_episodes(inferred_goal)",
        ],
    ),
    # ── Learning ──
    TableDef(
        name="learn_sessions",
        ddl="""
        CREATE TABLE IF NOT EXISTS learn_sessions (
            session_id TEXT PRIMARY KEY,
            trajectory_id TEXT NOT NULL,
            goal TEXT NOT NULL DEFAULT '',
            start_time DOUBLE NOT NULL,
            end_time DOUBLE,
            status TEXT NOT NULL DEFAULT 'recording',
            annotations TEXT,
            metadata TEXT,
            created_at DOUBLE NOT NULL,
            workspace_id TEXT DEFAULT '',
            session_ref TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_lsess_traj ON learn_sessions(trajectory_id)",
            "CREATE INDEX IF NOT EXISTS idx_lsess_status ON learn_sessions(status)",
        ],
    ),
    # ── Skill Library ──
    TableDef(
        name="skill_library",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_library (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            trigger_phrases TEXT,
            steps TEXT,
            parameters TEXT,
            pre_conditions TEXT,
            post_conditions TEXT,
            app_sequence TEXT,
            action_names TEXT,
            source_trajectory_id TEXT,
            source_episode_id TEXT,
            confidence DOUBLE,
            version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active',
            created_at DOUBLE NOT NULL,
            updated_at DOUBLE NOT NULL,
            workspace_id TEXT DEFAULT '',
            session_id TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_skill_status ON skill_library(status)",
        ],
    ),
    TableDef(
        name="skill_parameterized",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_parameterized (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            parameters TEXT,
            preconditions TEXT,
            postconditions TEXT,
            triggers TEXT,
            source TEXT DEFAULT 'builtin',
            source_trajectory_id TEXT,
            source_episode_id TEXT,
            confidence DOUBLE DEFAULT 1.0,
            version INTEGER DEFAULT 1,
            code TEXT,
            created_at DOUBLE,
            updated_at DOUBLE,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_param_active ON skill_parameterized(is_active)",
        ],
    ),
    TableDef(
        name="skill_suggestions",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_suggestions (
            id TEXT PRIMARY KEY,
            existing_skill_id TEXT NOT NULL,
            existing_skill_title TEXT,
            new_candidate_json TEXT NOT NULL,
            similarity_score DOUBLE NOT NULL,
            similarity_details TEXT,
            suggestion_type TEXT NOT NULL,
            proposed_changes TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            source_trajectory_id TEXT,
            source_episode_id TEXT,
            created_at DOUBLE NOT NULL,
            resolved_at DOUBLE
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_sug_status ON skill_suggestions(status)",
        ],
    ),
    TableDef(
        name="skill_executions",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_executions (
            id TEXT PRIMARY KEY,
            skill_id TEXT NOT NULL,
            trajectory_id TEXT NOT NULL,
            episode_id TEXT NOT NULL,
            similarity_score DOUBLE NOT NULL,
            diff_hash TEXT NOT NULL,
            diff_summary TEXT,
            verdict TEXT NOT NULL,
            created_at DOUBLE NOT NULL
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_sexec_skill ON skill_executions(skill_id)",
            "CREATE INDEX IF NOT EXISTS idx_sexec_hash ON skill_executions(diff_hash)",
        ],
    ),
    # ── Conversation ──
    TableDef(
        name="conv_sessions",
        ddl="""
        CREATE TABLE IF NOT EXISTS conv_sessions (
            session_id VARCHAR PRIMARY KEY,
            title VARCHAR DEFAULT '',
            created_at DOUBLE DEFAULT 0.0,
            updated_at DOUBLE DEFAULT 0.0,
            parent_session_id VARCHAR,
            model VARCHAR DEFAULT '',
            source VARCHAR DEFAULT 'cli',
            cwd VARCHAR DEFAULT '',
            message_count INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE,
            metadata_json VARCHAR DEFAULT '{}',
            workspace_id TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_csess_updated ON conv_sessions(updated_at DESC)",
        ],
    ),
    TableDef(
        name="conv_messages",
        ddl="""
        CREATE TABLE IF NOT EXISTS conv_messages (
            message_id VARCHAR PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            content VARCHAR DEFAULT '',
            created_at DOUBLE DEFAULT 0.0,
            tool_name VARCHAR,
            tool_call_id VARCHAR,
            tool_calls_json VARCHAR,
            active BOOLEAN DEFAULT TRUE,
            compacted BOOLEAN DEFAULT FALSE,
            token_count INTEGER DEFAULT 0,
            metadata_json VARCHAR DEFAULT '{}'
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_cmsg_session ON conv_messages(session_id, created_at)",
        ],
    ),
    # ── Evolution ──
    TableDef(
        name="evo_episodes",
        ddl="""
        CREATE TABLE IF NOT EXISTS evo_episodes (
            episode_id VARCHAR PRIMARY KEY,
            skill_name VARCHAR NOT NULL,
            actions_json VARCHAR DEFAULT '[]',
            outcome VARCHAR DEFAULT '',
            reward DOUBLE DEFAULT 0.0,
            context_json VARCHAR DEFAULT '{}',
            created_at DOUBLE DEFAULT 0.0,
            workspace_id TEXT DEFAULT '',
            session_id TEXT DEFAULT ''
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_evep_skill ON evo_episodes(skill_name, created_at DESC)",
        ],
    ),
    TableDef(
        name="evo_patterns",
        ddl="""
        CREATE TABLE IF NOT EXISTS evo_patterns (
            pattern_id VARCHAR PRIMARY KEY,
            skill_name VARCHAR NOT NULL,
            pattern_json VARCHAR DEFAULT '{}',
            confidence DOUBLE DEFAULT 0.0,
            episode_count INTEGER DEFAULT 0,
            created_at DOUBLE DEFAULT 0.0
        )
        """,
        indexes=[],
    ),
    # ── Scheduler ──
    TableDef(
        name="sched_tasks",
        ddl="""
        CREATE TABLE IF NOT EXISTS sched_tasks (
            task_id TEXT PRIMARY KEY,
            skill_name TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            trigger_config TEXT NOT NULL,
            state TEXT DEFAULT 'armed',
            execution_tier TEXT DEFAULT 'auto',
            context_snapshot TEXT DEFAULT '{}',
            confidence DOUBLE DEFAULT 0.0,
            created_at DOUBLE NOT NULL,
            next_due_at DOUBLE DEFAULT 0.0,
            last_run_at DOUBLE DEFAULT 0.0,
            run_count INTEGER DEFAULT 0,
            max_runs INTEGER DEFAULT -1,
            grace_seconds DOUBLE DEFAULT 120.0,
            parameters TEXT DEFAULT '{}',
            cloud_worker_id TEXT DEFAULT '',
            metadata TEXT DEFAULT '{}'
        )
        """,
        indexes=[],
    ),
    # ── Schema version tracking ──
    TableDef(
        name="_schema_version",
        ddl="""
        CREATE TABLE IF NOT EXISTS _schema_version (
            version INTEGER NOT NULL,
            applied_at DOUBLE NOT NULL
        )
        """,
        indexes=[],
    ),
]


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> int:
    """Create all tables and indexes if they don't exist.

    Returns the current schema version after applying.
    """
    for table_def in TABLES:
        conn.execute(table_def.ddl)
        for idx_sql in table_def.indexes:
            conn.execute(idx_sql)

    row = conn.execute(
        "SELECT MAX(version) FROM _schema_version"
    ).fetchone()
    current = row[0] if row and row[0] is not None else 0

    if current < CURRENT_SCHEMA_VERSION:
        import time
        conn.execute(
            "INSERT INTO _schema_version VALUES (?, ?)",
            [CURRENT_SCHEMA_VERSION, time.time()],
        )
        logger.info("schema: applied version %d", CURRENT_SCHEMA_VERSION)

    return CURRENT_SCHEMA_VERSION

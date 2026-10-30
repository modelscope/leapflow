# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified DuckDB schema definition and migration for leap.duckdb.

Single source of truth for all table schemas. Each store registers its
schema here rather than running ad-hoc CREATE TABLE in its own __init__.

Migration is version-tracked via a ``_schema_version`` table.

**Every migration must be idempotent.** The version integer records how far a
database has been advanced, but it cannot guarantee that a migration's *contents*
match the physical objects present -- an intermediate build may have created an
object under a different version, and a crash between a DDL statement and the
version bump can leave a migration half-applied. So every statement uses
``CREATE ... IF NOT EXISTS`` / ``DROP ... IF EXISTS`` / ``ADD COLUMN IF NOT EXISTS``
and re-running it against a database that already holds the object is a safe no-op.
A non-idempotent ``CREATE`` here is a latent startup crash: it aborts the whole
bootstrap transaction the first time a pre-existing object is met.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List

import duckdb

logger = logging.getLogger(__name__)

BASE_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = 10


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
    # ── Skill evolution memory ──
    TableDef(
        name="skill_episodes",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_episodes (
            episode_id VARCHAR PRIMARY KEY,
            skill_name VARCHAR NOT NULL,
            actions_json VARCHAR DEFAULT '[]',
            outcome VARCHAR DEFAULT '',
            reward DOUBLE DEFAULT 0.0,
            context_json VARCHAR DEFAULT '{}',
            created_at DOUBLE DEFAULT 0.0
        )
        """,
        indexes=[
            "CREATE INDEX IF NOT EXISTS idx_episodes_skill ON skill_episodes(skill_name, created_at DESC)",
        ],
    ),
    TableDef(
        name="skill_patterns",
        ddl="""
        CREATE TABLE IF NOT EXISTS skill_patterns (
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


@dataclass(frozen=True)
class MigrationDef:
    """One ordered, transactional schema migration."""

    version: int
    name: str
    apply: Callable[[duckdb.DuckDBPyConnection], None]


def _apply_evolution_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the append-only event stream and durable cold-path work queues."""
    statements = (
        """
        CREATE TABLE IF NOT EXISTS evolution_events (
            sequence BIGINT NOT NULL,
            event_id VARCHAR PRIMARY KEY,
            event_type VARCHAR NOT NULL,
            profile_id VARCHAR NOT NULL DEFAULT '',
            workspace_id VARCHAR NOT NULL DEFAULT '',
            session_id VARCHAR NOT NULL DEFAULT '',
            session_generation INTEGER NOT NULL DEFAULT 0,
            turn_id VARCHAR NOT NULL DEFAULT '',
            frame_id VARCHAR NOT NULL DEFAULT '',
            action_id VARCHAR NOT NULL DEFAULT '',
            observation_id VARCHAR NOT NULL DEFAULT '',
            requirement_id VARCHAR NOT NULL DEFAULT '',
            decision_id VARCHAR NOT NULL DEFAULT '',
            proposal_id VARCHAR NOT NULL DEFAULT '',
            artifact_id VARCHAR NOT NULL DEFAULT '',
            plugin_id VARCHAR NOT NULL DEFAULT '',
            version_id VARCHAR NOT NULL DEFAULT '',
            correlation_id VARCHAR NOT NULL DEFAULT '',
            causation_id VARCHAR NOT NULL DEFAULT '',
            occurred_at DOUBLE NOT NULL,
            producer VARCHAR NOT NULL,
            producer_version VARCHAR NOT NULL DEFAULT '',
            privacy_class VARCHAR NOT NULL DEFAULT 'system',
            schema_version INTEGER NOT NULL,
            payload_json VARCHAR NOT NULL DEFAULT '{}',
            payload_hash VARCHAR NOT NULL,
            dedup_key VARCHAR NOT NULL,
            UNIQUE(profile_id, dedup_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS evolution_teacher_jobs (
            job_id VARCHAR PRIMARY KEY,
            profile_id VARCHAR NOT NULL,
            workspace_id VARCHAR NOT NULL DEFAULT '',
            session_id VARCHAR NOT NULL DEFAULT '',
            session_generation INTEGER NOT NULL DEFAULT 0,
            episode_id VARCHAR NOT NULL,
            from_sequence BIGINT NOT NULL DEFAULT 0,
            through_sequence BIGINT NOT NULL DEFAULT 0,
            reason VARCHAR NOT NULL DEFAULT '',
            goal VARCHAR NOT NULL DEFAULT '',
            status VARCHAR NOT NULL,
            lease_owner VARCHAR NOT NULL DEFAULT '',
            lease_until DOUBLE NOT NULL DEFAULT 0.0,
            attempts INTEGER NOT NULL DEFAULT 0,
            model VARCHAR NOT NULL DEFAULT '',
            prompt_hash VARCHAR NOT NULL DEFAULT '',
            result_artifact_id VARCHAR NOT NULL DEFAULT '',
            next_attempt_at DOUBLE NOT NULL DEFAULT 0.0,
            created_at DOUBLE NOT NULL,
            updated_at DOUBLE NOT NULL,
            completed_at DOUBLE NOT NULL DEFAULT 0.0,
            error VARCHAR NOT NULL DEFAULT '',
            UNIQUE(profile_id, episode_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_evo_event_session ON evolution_events(profile_id, session_id, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_evo_event_correlation ON evolution_events(profile_id, correlation_id, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_evo_event_type ON evolution_events(profile_id, event_type, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_evo_teacher_status ON evolution_teacher_jobs(profile_id, status, next_attempt_at)",
    )
    for statement in statements:
        conn.execute(statement)


def _apply_evolution_sequence(conn: duckdb.DuckDBPyConnection) -> None:
    """Create a database-global cursor after any pre-sequence event rows."""
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_evo_event_sequence ON evolution_events(sequence)"
    )
    row = conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM evolution_events").fetchone()
    start = max(1, int(row[0] if row else 1))
    conn.execute(f"CREATE SEQUENCE IF NOT EXISTS evolution_event_sequence START {start}")


def _apply_teacher_job_context(conn: duckdb.DuckDBPyConnection) -> None:
    """Make every durable teacher job independently executable and auditable."""
    statements = (
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS workspace_id VARCHAR DEFAULT ''",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS session_id VARCHAR DEFAULT ''",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS session_generation INTEGER DEFAULT 0",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS from_sequence BIGINT DEFAULT 0",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS through_sequence BIGINT DEFAULT 0",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS reason VARCHAR DEFAULT ''",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS goal VARCHAR DEFAULT ''",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS result_artifact_id VARCHAR DEFAULT ''",
        "ALTER TABLE evolution_teacher_jobs ADD COLUMN IF NOT EXISTS completed_at DOUBLE DEFAULT 0.0",
    )
    for statement in statements:
        conn.execute(statement)


def _apply_evolution_projection(conn: duckdb.DuckDBPyConnection) -> None:
    """Create checkpointed read models derived exclusively from the event stream."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS evolution_projections (
            projection_name VARCHAR NOT NULL,
            profile_id VARCHAR NOT NULL,
            scope_key VARCHAR NOT NULL,
            last_sequence BIGINT NOT NULL DEFAULT 0,
            state_json VARCHAR NOT NULL DEFAULT '{}',
            updated_at DOUBLE NOT NULL,
            PRIMARY KEY (projection_name, profile_id, scope_key)
        )
        """
    )


def _apply_proposal_event_index(conn: duckdb.DuckDBPyConnection) -> None:
    """Retire the unused work table and index event-sourced proposal replay."""
    conn.execute("DROP TABLE IF EXISTS evolution_proposal_work")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evo_event_proposal "
        "ON evolution_events(profile_id, proposal_id, sequence)"
    )


def _apply_skill_curation_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the skill curation lifecycle management table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_curation (
            skill_name TEXT PRIMARY KEY,
            state TEXT NOT NULL DEFAULT 'active',
            pinned BOOLEAN NOT NULL DEFAULT FALSE,
            last_activity_at DOUBLE,
            created_at DOUBLE NOT NULL,
            archive_reason TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_skill_curation_state ON skill_curation(state)"
    )


def _apply_session_snapshot_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Add PCD cache-aware session snapshot columns to conv_sessions.

    These columns persist the system prompt, tool schema, and disclosure level
    at the time a session was last active, enabling prefix-cache-friendly
    session resumption.
    """
    statements = (
        "ALTER TABLE conv_sessions ADD COLUMN IF NOT EXISTS system_prompt_snapshot TEXT",
        "ALTER TABLE conv_sessions ADD COLUMN IF NOT EXISTS tool_schema_snapshot TEXT",
        "ALTER TABLE conv_sessions ADD COLUMN IF NOT EXISTS disclosure_level TEXT",
    )
    for statement in statements:
        conn.execute(statement)


def _apply_session_operations_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Add pinned/hidden columns to conv_sessions for session management operations.

    ``pinned`` promotes a session to the top of listings.
    ``hidden`` excludes a session from default listings without deletion.
    """
    statements = (
        "ALTER TABLE conv_sessions ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT FALSE",
        "ALTER TABLE conv_sessions ADD COLUMN IF NOT EXISTS hidden BOOLEAN DEFAULT FALSE",
    )
    for statement in statements:
        conn.execute(statement)


def _apply_approval_decisions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the Guardian LLM approval audit table.

    Records every Guardian decision so the approval pipeline is fully
    auditable even when the LLM auto-approves or auto-denies.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_decisions (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL DEFAULT '',
            timestamp DOUBLE NOT NULL,
            tool_name TEXT NOT NULL DEFAULT '',
            risk_score DOUBLE NOT NULL DEFAULT 0.0,
            recommendation TEXT NOT NULL DEFAULT '',
            decision TEXT NOT NULL DEFAULT '',
            reasoning TEXT NOT NULL DEFAULT '',
            latency_ms DOUBLE NOT NULL DEFAULT 0.0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_approval_decisions_session "
        "ON approval_decisions(session_id, timestamp)"
    )


MIGRATIONS: tuple[MigrationDef, ...] = (
    MigrationDef(2, "evolution event stream", _apply_evolution_tables),
    MigrationDef(3, "database-global evolution cursor", _apply_evolution_sequence),
    MigrationDef(4, "durable teacher job context", _apply_teacher_job_context),
    MigrationDef(5, "checkpointed evolution projections", _apply_evolution_projection),
    MigrationDef(6, "event-sourced proposal index", _apply_proposal_event_index),
    MigrationDef(7, "PCD session snapshot columns", _apply_session_snapshot_columns),
    MigrationDef(8, "skill curation lifecycle", _apply_skill_curation_table),
    MigrationDef(9, "session operations (pin/hide)", _apply_session_operations_columns),
    MigrationDef(10, "guardian approval decisions audit", _apply_approval_decisions_table),
)


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> int:
    """Bootstrap the baseline and apply every pending migration atomically."""
    conn.execute("BEGIN TRANSACTION")
    try:
        for table_def in TABLES:
            conn.execute(table_def.ddl)
            for idx_sql in table_def.indexes:
                conn.execute(idx_sql)

        row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
        current = int(row[0]) if row and row[0] is not None else 0
        if current > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema {current} is newer than supported {CURRENT_SCHEMA_VERSION}"
            )
        if current == 0:
            conn.execute(
                "INSERT INTO _schema_version VALUES (?, ?)",
                [BASE_SCHEMA_VERSION, time.time()],
            )
            current = BASE_SCHEMA_VERSION

        for migration in MIGRATIONS:
            if migration.version <= current:
                continue
            if migration.version != current + 1:
                raise RuntimeError(
                    f"schema migration gap: current={current}, next={migration.version}"
                )
            migration.apply(conn)
            conn.execute(
                "INSERT INTO _schema_version VALUES (?, ?)",
                [migration.version, time.time()],
            )
            current = migration.version
            logger.info("schema: applied version %d (%s)", current, migration.name)

        if current != CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"schema migration incomplete: current={current}, expected={CURRENT_SCHEMA_VERSION}"
            )
        conn.execute("COMMIT")
        return current
    except Exception:
        conn.execute("ROLLBACK")
        raise

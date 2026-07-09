"""Migration: consolidate 6 legacy DuckDB files into single leap.duckdb.

Idempotent — safe to run multiple times. Skips files that don't exist
or have already been migrated. Renames old files to ``.migrated.bak``
after successful migration.

Usage::

    from leapflow.storage.migrate import migrate_to_unified
    migrate_to_unified(data_dir=Path("~/.leapflow").expanduser())
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb

logger = logging.getLogger(__name__)

_TABLE_MAP: List[Tuple[str, str, str, List[str]]] = [
    # (old_db_filename, old_table, new_table, columns_to_copy)
    # columns_to_copy is empty = SELECT * (we handle column mapping in SQL)
    ("memory.duckdb", "leap_memory", "mem_entries", []),
    ("trajectories.duckdb", "leap_trajectory", "traj_headers", []),
    ("trajectories.duckdb", "leap_trajectory_step", "traj_steps", []),
    ("trajectories.duckdb", "leap_episode", "traj_episodes", []),
    ("trajectories.duckdb", "leap_learning_session", "learn_sessions", []),
    ("skill_library.duckdb", "leap_skill_library", "skill_library", []),
    ("skill_library.duckdb", "leap_parameterized_skills", "skill_parameterized", []),
    ("skill_library.duckdb", "leap_skill_suggestion", "skill_suggestions", []),
    ("skill_library.duckdb", "leap_skill_execution", "skill_executions", []),
    ("conversations.duckdb", "conversation_sessions", "conv_sessions", []),
    ("conversations.duckdb", "conversation_messages", "conv_messages", []),
    ("evolution.duckdb", "skill_episodes", "evo_episodes", []),
    ("evolution.duckdb", "skill_patterns", "evo_patterns", []),
    ("scheduler.duckdb", "armed_tasks", "sched_tasks", []),
]

_COLUMN_MAP: Dict[Tuple[str, str], str] = {
    ("leap_memory", "mem_entries"): """
        SELECT id, kind, domain, content, path, metadata,
               created_at, accessed_at, access_count,
               '' AS workspace_id, '' AS session_id
        FROM old_db.{old_table}
    """,
    ("leap_trajectory", "traj_headers"): """
        SELECT id, user_id, start_time, end_time, step_count, metadata,
               created_at,
               '' AS workspace_id, '' AS session_id
        FROM old_db.{old_table}
    """,
    ("leap_learning_session", "learn_sessions"): """
        SELECT session_id, trajectory_id, goal, start_time, end_time,
               status, annotations, metadata, created_at,
               '' AS workspace_id, '' AS session_ref
        FROM old_db.{old_table}
    """,
    ("leap_skill_library", "skill_library"): """
        SELECT id, title, trigger_phrases, steps, parameters,
               pre_conditions, post_conditions, app_sequence, action_names,
               source_trajectory_id, source_episode_id, confidence,
               version, status, created_at, updated_at,
               '' AS workspace_id, '' AS session_id
        FROM old_db.{old_table}
    """,
    ("conversation_sessions", "conv_sessions"): """
        SELECT session_id, title, created_at, updated_at,
               parent_session_id, model, source, cwd,
               message_count, total_tokens, is_active, metadata_json,
               '' AS workspace_id
        FROM old_db.{old_table}
    """,
    ("skill_episodes", "evo_episodes"): """
        SELECT episode_id, skill_name, actions_json, outcome, reward,
               context_json, created_at,
               '' AS workspace_id, '' AS session_id
        FROM old_db.{old_table}
    """,
}


def migrate_to_unified(
    data_dir: Path,
    *,
    profile: str = "default",
    target_db_path: Path | None = None,
) -> Dict[str, int]:
    """Migrate all legacy DuckDB files into the unified ``leap.duckdb``.

    Parameters
    ----------
    data_dir : Path
        The LeapFlow data directory (e.g. ``~/.leapflow``).
    profile : str
        Target profile name (default ``"default"``).
    target_db_path : Path, optional
        Explicit path for the target database. Defaults to
        ``data_dir / "profiles" / profile / "db" / "leap.duckdb"``.

    Returns
    -------
    dict
        Mapping of ``"old_file:old_table"`` to row count migrated.
    """
    target = target_db_path or (data_dir / "profiles" / profile / "db" / "leap.duckdb")
    target.parent.mkdir(parents=True, exist_ok=True)

    from leapflow.storage.duckdb_connect import connect as safe_connect
    from leapflow.storage.schema import ensure_schema

    conn = safe_connect(target)
    ensure_schema(conn)

    results: Dict[str, int] = {}
    attached_dbs: set = set()

    for old_filename, old_table, new_table, _cols in _TABLE_MAP:
        old_path = data_dir / old_filename
        if not old_path.exists():
            logger.debug("migrate: skipping %s (not found)", old_filename)
            continue

        key = f"{old_filename}:{old_table}"

        existing = conn.execute(
            f"SELECT COUNT(*) FROM {new_table}"
        ).fetchone()
        if existing and existing[0] > 0:
            logger.info("migrate: %s already has %d rows, skipping %s",
                        new_table, existing[0], key)
            results[key] = 0
            continue

        alias = old_filename.replace(".", "_").replace("-", "_")
        if alias not in attached_dbs:
            try:
                conn.execute(f"ATTACH '{old_path}' AS old_db (READ_ONLY)")
                attached_dbs.add(alias)
            except Exception as exc:
                logger.warning("migrate: cannot attach %s: %s", old_filename, exc)
                continue

        try:
            tables_in_old = {
                r[0] for r in conn.execute("SHOW TABLES FROM old_db").fetchall()
            }
            if old_table not in tables_in_old:
                logger.debug("migrate: table %s not in %s", old_table, old_filename)
                conn.execute("DETACH old_db")
                attached_dbs.discard(alias)
                continue

            custom_select = _COLUMN_MAP.get((old_table, new_table))
            if custom_select:
                select_sql = custom_select.format(old_table=old_table)
            else:
                select_sql = f"SELECT * FROM old_db.{old_table}"

            conn.execute(f"INSERT INTO {new_table} {select_sql}")

            count_row = conn.execute(
                f"SELECT COUNT(*) FROM {new_table}"
            ).fetchone()
            count = count_row[0] if count_row else 0
            results[key] = count
            logger.info("migrate: %s → %s: %d rows", key, new_table, count)

        except Exception as exc:
            logger.warning("migrate: failed %s → %s: %s", key, new_table, exc)
            results[key] = -1
        finally:
            try:
                conn.execute("DETACH old_db")
                attached_dbs.discard(alias)
            except Exception:
                pass

    conn.close()

    _rename_old_files(data_dir, results)
    return results


def _rename_old_files(data_dir: Path, results: Dict[str, int]) -> None:
    """Rename successfully migrated old .duckdb files to .migrated.bak."""
    migrated_files: set = set()
    for key, count in results.items():
        if count > 0:
            filename = key.split(":")[0]
            migrated_files.add(filename)

    for filename in migrated_files:
        old_path = data_dir / filename
        if not old_path.exists():
            continue

        all_tables_done = all(
            results.get(f"{filename}:{entry[1]}", 0) >= 0
            for entry in _TABLE_MAP
            if entry[0] == filename
        )
        if not all_tables_done:
            continue

        bak_path = old_path.with_suffix(f".duckdb.migrated.bak")
        try:
            old_path.rename(bak_path)
            wal = old_path.with_suffix(old_path.suffix + ".wal")
            if wal.exists():
                wal.rename(bak_path.with_suffix(bak_path.suffix + ".wal"))
            logger.info("migrate: renamed %s → %s", old_path.name, bak_path.name)
        except OSError as exc:
            logger.warning("migrate: rename failed for %s: %s", old_path.name, exc)


def migrate_directory_layout(data_dir: Path, *, profile: str = "default") -> int:
    """Migrate flat ``~/.leapflow/`` to profile-based directory structure.

    Moves ``skills/``, ``cache/``, ``audit.jsonl`` into
    ``profiles/<profile>/``.  Idempotent — skips files that don't exist
    or are already migrated.

    Returns count of items moved.
    """
    import shutil

    profile_dir = data_dir / "profiles" / profile
    moved = 0

    _MOVES: List[Tuple[str, str]] = [
        ("skills", "skills"),
        ("cache/frames", "cache/frames"),
        ("cache/video", "cache/video"),
        ("audit.jsonl", "audit.jsonl"),
    ]

    for src_rel, dst_rel in _MOVES:
        src = data_dir / src_rel
        dst = profile_dir / dst_rel
        if not src.exists():
            continue
        if dst.exists() and (dst.is_dir() and any(dst.iterdir())):
            logger.debug("migrate_layout: %s already populated, skipping", dst_rel)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.is_dir():
                if dst.exists():
                    for item in src.iterdir():
                        target = dst / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))
                            moved += 1
                else:
                    shutil.move(str(src), str(dst))
                    moved += 1
            else:
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
            logger.info("migrate_layout: %s → %s", src_rel, dst_rel)
        except OSError as exc:
            logger.warning("migrate_layout: failed %s: %s", src_rel, exc)

    return moved

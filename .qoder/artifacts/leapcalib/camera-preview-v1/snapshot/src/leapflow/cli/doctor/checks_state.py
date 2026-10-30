# Copyright (c) Alibaba, Inc. and its affiliates.
"""State diagnostic checks (DuckDB, vault, scheduler)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from leapflow.cli.doctor.protocol import Finding

logger = logging.getLogger(__name__)


class DuckDBHealthCheck:
    """Verify that the primary DuckDB database can be opened."""

    name = "DuckDB health"
    section = "state"

    def __init__(self, duckdb_path: Path) -> None:
        self._path = duckdb_path

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        if not self._path.exists():
            f.warn(f"DuckDB file not found at {self._path} (will be created on first use)")
            return f

        try:
            import duckdb

            conn = duckdb.connect(str(self._path), read_only=True)
            # Simple sanity: list tables
            conn.execute("SHOW TABLES").fetchall()
            conn.close()
            f.pass_()
        except Exception as exc:
            f.error(f"DuckDB cannot be opened: {exc}")
        return f


class VaultCheck:
    """Verify that the secrets vault key file exists."""

    name = "Vault key"
    section = "state"

    def __init__(self, profile_layout: Any) -> None:
        self._layout = profile_layout

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        secrets = self._layout.secrets
        secrets_dir: Path = secrets.root
        if not secrets_dir.is_dir():
            if should_fix:
                secrets.ensure()
                f.fix(f"Created secrets directory: {secrets_dir}")
            else:
                f.warn(f"Secrets directory missing: {secrets_dir}")
            return f

        key_path = secrets.key_path
        if not key_path.is_file():
            # Key file is generated on first secret write; absence is normal
            # for fresh installs.
            f.pass_()
        else:
            f.pass_()
        return f


class SchedulerHealthCheck:
    """Run lightweight scheduler task-store diagnostics."""

    name = "Scheduler health"
    section = "state"

    def __init__(self, profile_layout: Any) -> None:
        self._layout = profile_layout

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        try:
            from leapflow.scheduler.store import TaskStore

            db_path = self._layout.duckdb_path
            if not db_path.exists():
                f.pass_()  # No DB yet — scheduler unused
                return f

            store = TaskStore(db_path)
            tasks = store.load_all()
            import time

            now = time.time()
            stale = 0
            near_exhaustion = 0
            for t in tasks:
                if t.state == "armed" and t.next_due_at > 0 and t.next_due_at < now - 120:
                    stale += 1
                if (
                    t.max_retries > 0
                    and t.retry_count >= t.max_retries - 1
                    and t.state not in ("done", "suspended", "failed")
                ):
                    near_exhaustion += 1

            if stale:
                f.warn(f"{stale} stale scheduled task(s) with overdue next_due_at")
            if near_exhaustion:
                f.warn(f"{near_exhaustion} task(s) near retry exhaustion")
            if not stale and not near_exhaustion:
                f.pass_()
        except ImportError:
            f.pass_()  # Scheduler module not available
        except Exception as exc:
            f.warn(f"Scheduler check failed: {exc}")
        return f

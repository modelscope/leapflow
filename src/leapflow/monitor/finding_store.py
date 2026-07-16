"""DuckDB-backed persistence for monitor findings.

Shares the daemon's single ``leap.duckdb`` connection via ``ConnectionHolder``
(same pattern as ``TaskStore``). Findings are append-mostly with a dedup guard
keyed by ``(watch_id, dedup_key)`` so producers can re-observe without spamming.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from leapflow.monitor.types import Finding, Severity
from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
from leapflow.storage.write_buffer import execute_with_retry

logger = logging.getLogger(__name__)


class FindingStore:
    """Atomic CRUD and filtered queries for findings.

    Accepts a shared ``ConnectionHolder`` (daemon-owned) or a legacy ``Path``
    for standalone use in tests.
    """

    def __init__(self, source: Union[ConnectionHolder, Path, str]) -> None:
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder = source
        self._con = self._holder.connection
        self._ensure_table()

    def close(self) -> None:
        """Close the DuckDB connection if owned by this store."""
        if self._owns_holder:
            self._holder.close()

    # ── Schema ───────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS monitor_findings (
                finding_id TEXT PRIMARY KEY,
                watch_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                ts DOUBLE NOT NULL,
                severity TEXT NOT NULL DEFAULT 'info',
                score DOUBLE DEFAULT 0.0,
                title TEXT NOT NULL,
                summary TEXT DEFAULT '',
                evidence TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                suggested_actions TEXT DEFAULT '[]',
                payload TEXT DEFAULT '{}',
                dedup_key TEXT DEFAULT ''
            )
        """)

    # ── Write ────────────────────────────────────────────────────────────

    def save(self, finding: Finding) -> None:
        """Insert or replace a finding by ``finding_id``."""
        execute_with_retry(
            self._con,
            """
            INSERT OR REPLACE INTO monitor_findings (
                finding_id, watch_id, domain, ts, severity, score,
                title, summary, evidence, tags, suggested_actions, payload, dedup_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                finding.finding_id,
                finding.watch_id,
                finding.domain,
                finding.ts,
                finding.severity.value,
                finding.score,
                finding.title,
                finding.summary,
                json.dumps([item.to_dict() for item in finding.evidence], ensure_ascii=False),
                json.dumps(list(finding.tags), ensure_ascii=False),
                json.dumps([a.to_dict() for a in finding.suggested_actions], ensure_ascii=False),
                json.dumps(dict(finding.payload), ensure_ascii=False),
                finding.dedup_key,
            ],
        )

    def exists_dedup(self, watch_id: str, dedup_key: str) -> bool:
        """Return True if a finding with this dedup key already exists."""
        if not dedup_key:
            return False
        row = self._con.execute(
            "SELECT 1 FROM monitor_findings WHERE watch_id = ? AND dedup_key = ? LIMIT 1",
            [watch_id, dedup_key],
        ).fetchone()
        return row is not None

    def delete_for_watch(self, watch_id: str) -> None:
        """Remove all findings belonging to a watch."""
        execute_with_retry(
            self._con,
            "DELETE FROM monitor_findings WHERE watch_id = ?",
            [watch_id],
        )

    # ── Read ─────────────────────────────────────────────────────────────

    def list(
        self,
        *,
        watch_id: Optional[str] = None,
        min_severity: Optional[Severity] = None,
        since: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Finding]:
        """Return findings newest-first with optional filters."""
        clauses: list[str] = []
        params: list[object] = []
        if watch_id:
            clauses.append("watch_id = ?")
            params.append(watch_id)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(since)
        if min_severity is not None:
            allowed = [s.value for s in Severity if s.rank >= min_severity.rank]
            placeholders = ", ".join("?" for _ in allowed)
            clauses.append(f"severity IN ({placeholders})")
            params.extend(allowed)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        params.append(max(0, int(offset)))
        rows = self._con.execute(
            f"""
            SELECT finding_id, watch_id, domain, ts, severity, score,
                   title, summary, evidence, tags, suggested_actions, payload, dedup_key
            FROM monitor_findings
            {where}
            ORDER BY ts DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [self._row_to_finding(row) for row in rows]

    def count(self, *, watch_id: Optional[str] = None, min_severity: Optional[Severity] = None) -> int:
        """Return the number of findings matching the filters."""
        clauses: list[str] = []
        params: list[object] = []
        if watch_id:
            clauses.append("watch_id = ?")
            params.append(watch_id)
        if min_severity is not None:
            allowed = [s.value for s in Severity if s.rank >= min_severity.rank]
            placeholders = ", ".join("?" for _ in allowed)
            clauses.append(f"severity IN ({placeholders})")
            params.extend(allowed)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._con.execute(
            f"SELECT COUNT(*) FROM monitor_findings {where}", params
        ).fetchone()
        return int(row[0]) if row else 0

    # ── Internal ─────────────────────────────────────────────────────────

    def _row_to_finding(self, row: tuple) -> Finding:
        return Finding.from_dict({
            "finding_id": row[0],
            "watch_id": row[1],
            "domain": row[2],
            "ts": row[3],
            "severity": row[4],
            "score": row[5],
            "title": row[6],
            "summary": row[7],
            "evidence": self._safe_json(row[8], []),
            "tags": self._safe_json(row[9], []),
            "suggested_actions": self._safe_json(row[10], []),
            "payload": self._safe_json(row[11], {}),
            "dedup_key": row[12],
        })

    @staticmethod
    def _safe_json(value: object, default: object) -> object:
        if value is None:
            return default
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default


__all__ = ["FindingStore"]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB-backed persistence for monitor findings.

Shares the daemon's single ``leap.duckdb`` connection via ``ConnectionHolder``
(same pattern as ``TaskStore``). Findings are append-mostly with a dedup guard
keyed by ``(watch_id, dedup_key)`` so producers can re-observe without spamming.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

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
        self._ensure_table()

    @property
    def _con(self) -> Any:
        """Resolve per call, so each thread gets its own cursor.

        See ``LocalConnectionHolder.connection``: the value is thread-specific, so
        caching it puts two threads on one connection and the second blocks inside
        DuckDB until the first query ends -- freezing the event loop when it is one
        of them.
        """
        return self._holder.connection

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
        self._con.execute("ALTER TABLE monitor_findings ADD COLUMN IF NOT EXISTS delivery TEXT DEFAULT '{}'")
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS monitor_delivery_state (
                incident_key TEXT PRIMARY KEY, last_enqueued DOUBLE NOT NULL
            )
        """)
        self._con.execute("CREATE TABLE IF NOT EXISTS monitor_delivery_overflow (incident_key TEXT PRIMARY KEY)")
        self._con.execute("ALTER TABLE monitor_delivery_overflow ADD COLUMN IF NOT EXISTS metadata TEXT DEFAULT '{}'")

    def stage_delivery_overflow(self, entries: list[dict[str, Any]], capacity: int) -> int:
        """Bound identities and presentation metadata; complete evidence stays in history."""
        con = self._con
        existing = set(self.delivery_overflow_keys())
        retained = set(sorted(existing)[:max(0, capacity - 3)])
        self.withdraw_delivery_overflow(existing - retained, force=True)
        existing = retained
        for entry in entries:
            key = entry["key"]
            if key in existing or len(existing) < capacity - 3:
                con.execute("INSERT OR REPLACE INTO monitor_delivery_overflow VALUES (?, ?)",
                            [key, json.dumps(entry, ensure_ascii=False)])
                existing.add(key)
        return len(existing)

    def withdraw_delivery_overflow(self, keys: Any, *, force: bool = False) -> None:
        for key in keys:
            condition = "" if force else " AND coalesce(json_extract_string(metadata, '$.actionable'), 'false') != 'true'"
            self._con.execute("DELETE FROM monitor_delivery_overflow WHERE incident_key = ?" + condition, [key])

    def drop_delivery_watch(self, watch_ids: list[str]) -> None:
        """Remove watch ownership, preserving eligible cross-watch incidents."""
        dropped = set(watch_ids)
        for entry in self.delivery_overflow_entries():
            owners = [watch_id for watch_id in entry.get("watch_ids", (entry.get("watch_id"),))
                      if watch_id not in dropped]
            if owners:
                entry.update(watch_ids=owners, watch_id=owners[0])
                self._con.execute("UPDATE monitor_delivery_overflow SET metadata = ? WHERE incident_key = ?",
                                  [json.dumps(entry, ensure_ascii=False), entry["key"]])
            else:
                self.withdraw_delivery_overflow([entry["key"]], force=True)

    def delivery_overflow_entries(self) -> list[dict[str, Any]]:
        return [self._safe_json(row[0], {}) for row in self._con.execute(
            "SELECT metadata FROM monitor_delivery_overflow ORDER BY incident_key").fetchall()]

    def reset_delivery_overflow(self) -> None:
        self._con.execute("DELETE FROM monitor_delivery_overflow")

    def delivery_overflow_keys(self) -> list[str]:
        return [row[0] for row in self._con.execute("SELECT incident_key FROM monitor_delivery_overflow").fetchall()]

    def delivery_timestamps(self, keys: Any) -> dict[str, float]:
        """Read enqueue attempts, not receipt or hardware episode progress."""
        if not keys:
            return {}
        placeholders = ', '.join('?' for _ in keys)
        return dict(self._con.execute(
            f"SELECT incident_key, last_enqueued FROM monitor_delivery_state WHERE incident_key IN ({placeholders})",
            list(keys),
        ).fetchall())

    def record_delivery(self, keys: Any, now: float, retain_after: float, capacity: int) -> bool:
        """Atomically retain unexpired cooldowns and record a bounded attempt.

        Reserved identities ensure overflow can still be reported at capacity.
        New identities fail closed rather than resetting a recent incident's cooldown.
        """
        from leapflow.monitor.delivery import GLOBAL_KEY, OVERFLOW_KEY, SATURATED_KEY

        con = self._con
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM monitor_delivery_state WHERE last_enqueued < ?", [retain_after])
            existing = {row[0] for row in con.execute(
                "SELECT incident_key FROM monitor_delivery_state"
            ).fetchall()}
            reserved = {GLOBAL_KEY, OVERFLOW_KEY, SATURATED_KEY}
            requested = set(keys) | {GLOBAL_KEY}
            if OVERFLOW_KEY in requested:
                spooled = {row[0] for row in con.execute("SELECT incident_key FROM monitor_delivery_overflow").fetchall()}
                available = max(0, capacity - 3 - len((existing | requested) - reserved))
                missing = spooled - existing - requested
                if len(missing) > available:
                    requested.add(SATURATED_KEY)
                requested.update(sorted(missing)[:available])
                requested.update(spooled & existing)
                con.execute("DELETE FROM monitor_delivery_overflow")
            regular = (existing | requested) - reserved
            # A smaller hot-reloaded bound must not erase unexpired cooldowns.
            # Reserved aggregate slots still work; no new regular state may grow it.
            bound = max(capacity - 3, len(existing - reserved))
            if len(regular) > bound:
                con.execute("ROLLBACK")
                return False
            for key in requested:
                con.execute("INSERT OR REPLACE INTO monitor_delivery_state VALUES (?, ?)", [key, now])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        return True

    # ── Write ────────────────────────────────────────────────────────────

    def save(self, finding: Finding) -> None:
        """Insert or replace a finding by ``finding_id``."""
        execute_with_retry(
            self._con,
            """
            INSERT OR REPLACE INTO monitor_findings (
                finding_id, watch_id, domain, ts, severity, score,
                title, summary, evidence, tags, suggested_actions, payload, dedup_key, delivery
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(finding.delivery.to_dict(), ensure_ascii=False),
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

    def find_dedup(self, watch_id: str, dedup_key: str) -> Finding | None:
        """Resolve the persisted reference when a producer re-offers live evidence."""
        rows = self._con.execute("""
            SELECT finding_id, watch_id, domain, ts, severity, score,
                   title, summary, evidence, tags, suggested_actions, payload, dedup_key, delivery
            FROM monitor_findings WHERE watch_id = ? AND dedup_key = ? LIMIT 1
        """, [watch_id, dedup_key]).fetchall()
        return self._row_to_finding(rows[0]) if rows else None

    def latest_notification_checkpoints(self, domain: str) -> dict[str, dict[str, Any]]:
        """Latest persisted condition checkpoint across watch recreation, no new table.

        Ordinary dedup remains all-time per watch. Only producers carrying explicit
        notification metadata participate in this query.
        """
        rows = self._con.execute(
            """
            SELECT payload FROM (
                SELECT payload, row_number() OVER (
                    PARTITION BY json_extract_string(payload, '$.notification.key')
                    ORDER BY ts DESC, rowid DESC
                ) AS position
                FROM monitor_findings
                WHERE domain = ? AND json_extract_string(payload, '$.notification.key') IS NOT NULL
            ) WHERE position = 1
            """, [domain],
        ).fetchall()
        checkpoints = {}
        for (payload,) in rows:
            data = self._safe_json(payload, {})
            checkpoint = data.get("notification", {}) if isinstance(data, dict) else {}
            if isinstance(checkpoint, dict) and checkpoint.get("key"):
                checkpoints[str(checkpoint["key"])] = checkpoint
        return checkpoints

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
                   title, summary, evidence, tags, suggested_actions, payload, dedup_key, delivery
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
            "delivery": self._safe_json(row[13], {}),
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

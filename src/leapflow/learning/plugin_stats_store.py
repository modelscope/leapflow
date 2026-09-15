# Copyright (c) Alibaba, Inc. and its affiliates.
"""DuckDB persistence for plugin trust and usage statistics.

Provides save/load for PluginTrustLedger and PluginUsageTracker state across
process restarts. Uses the existing duckdb_connect() factory from
leapflow.storage.

Trust and usage are stored in separate tables on purpose: trust is a small
decision-bearing ledger, while usage is a bounded rolling sample window. A
corrupt or oversized usage blob must never cost the profile its trust levels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PluginStatsStore:
    """Persists plugin trust ledger state to DuckDB."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._table_created = False
        self._usage_table_created = False

    def _connect(self):
        """Get a DuckDB connection using the centralized factory."""
        try:
            from leapflow.storage.duckdb_connect import connect

            if self._db_path is None:
                return None
            return connect(self._db_path)
        except (ImportError, RuntimeError, OSError):
            return None

    def _ensure_table(self, conn) -> None:
        """Create the trust state table if it does not exist."""
        if self._table_created:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_trust_state (
                key TEXT PRIMARY KEY DEFAULT 'singleton',
                state_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._table_created = True

    def _ensure_usage_table(self, conn) -> None:
        """Create the usage state table if it does not exist."""
        if self._usage_table_created:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_usage_state (
                key TEXT PRIMARY KEY DEFAULT 'singleton',
                state_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._usage_table_created = True

    def save_trust_state(self, state: Dict[str, Any]) -> bool:
        """Persist trust ledger state. Returns True on success."""
        conn = self._connect()
        if conn is None:
            return False
        try:
            self._ensure_table(conn)
            state_json = json.dumps(state, ensure_ascii=False)
            conn.execute(
                "INSERT OR REPLACE INTO plugin_trust_state (key, state_json, updated_at) "
                "VALUES ('singleton', ?, CURRENT_TIMESTAMP)",
                [state_json],
            )
            return True
        except (RuntimeError, OSError) as exc:
            logger.warning("Failed to save plugin trust state: %s", exc)
            return False
        finally:
            conn.close()

    def load_trust_state(self) -> Optional[Dict[str, Any]]:
        """Load trust ledger state. Returns None if not found or on error."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure_table(conn)
            result = conn.execute(
                "SELECT state_json FROM plugin_trust_state WHERE key = 'singleton'"
            ).fetchone()
            if result is None:
                return None
            return json.loads(result[0])
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load plugin trust state: %s", exc)
            return None
        finally:
            conn.close()

    def save_usage_state(self, state: Dict[str, Any]) -> bool:
        """Persist rolling usage samples. Returns True on success."""
        conn = self._connect()
        if conn is None:
            return False
        try:
            self._ensure_usage_table(conn)
            state_json = json.dumps(state, ensure_ascii=False)
            conn.execute(
                "INSERT OR REPLACE INTO plugin_usage_state (key, state_json, updated_at) "
                "VALUES ('singleton', ?, CURRENT_TIMESTAMP)",
                [state_json],
            )
            return True
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save plugin usage state: %s", exc)
            return False
        finally:
            conn.close()

    def load_usage_state(self) -> Optional[Dict[str, Any]]:
        """Load rolling usage samples. Returns None if not found or on error."""
        conn = self._connect()
        if conn is None:
            return None
        try:
            self._ensure_usage_table(conn)
            result = conn.execute(
                "SELECT state_json FROM plugin_usage_state WHERE key = 'singleton'"
            ).fetchone()
            if result is None:
                return None
            return json.loads(result[0])
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load plugin usage state: %s", exc)
            return None
        finally:
            conn.close()

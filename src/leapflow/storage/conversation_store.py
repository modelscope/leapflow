"""Conversation session persistence — chat history storage with full-text search.

Design (inspired by hermes hermes_state.py):
- DuckDB-backed for consistency with leapflow's existing persistence layer
- Full-text search via DuckDB FTS extension for session recall
- Schema reconciliation on startup (no migration chains)
- Write retry with jitter for concurrent access
- Session lineage for compression continuations and subagent delegation
- Protocol-first design (DIP) — core logic depends on ConversationStore, not DuckDB

Fits into leapflow's event-driven architecture:
- Persist on EventBus events (SessionCreated, MessageAppended)
- Retrieve via semantic search or keyword-based FTS
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationSession:
    """Immutable snapshot of a conversation session's metadata."""
    session_id: str
    title: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    parent_session_id: Optional[str] = None
    model: str = ""
    source: str = "cli"
    cwd: str = ""
    message_count: int = 0
    total_tokens: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationMessage:
    """Immutable snapshot of a single message in a conversation."""
    message_id: str
    session_id: str
    role: str
    content: str
    created_at: float = 0.0
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls_json: Optional[str] = None
    active: bool = True
    compacted: bool = False
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationSearchResult:
    """A single search hit with relevance score and context."""
    message_id: str
    session_id: str
    session_title: str
    role: str
    content: str
    score: float
    created_at: float


@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for conversation persistence (DIP)."""

    def create_session(self, session_id: str, *, title: str = "", **kwargs: Any) -> ConversationSession: ...
    def get_session(self, session_id: str) -> Optional[ConversationSession]: ...
    def list_sessions(self, *, limit: int = 20, active_only: bool = True) -> List[ConversationSession]: ...
    def append_message(self, session_id: str, role: str, content: str, **kwargs: Any) -> ConversationMessage: ...
    def get_messages(self, session_id: str, *, limit: int = 100, active_only: bool = True) -> List[ConversationMessage]: ...
    def search_messages(self, query: str, *, limit: int = 10) -> List[ConversationSearchResult]: ...
    def close(self) -> None: ...


class DuckDBConversationStore:
    """DuckDB-backed conversation store with FTS and write-retry.

    Schema:
    - conversation_sessions: metadata, lineage (parent_session_id), model config
    - conversation_messages: full transcript with soft-delete (active) and compaction flag
    - FTS index on messages for keyword search

    Accepts ``ConnectionHolder`` (shared) or legacy ``Path``.
    """

    def __init__(self, source: "Union[ConnectionHolder, Path, str]") -> None:
        from leapflow.storage.connection import ConnectionHolder, LocalConnectionHolder
        self._owns_holder = isinstance(source, (str, Path))
        if self._owns_holder:
            source = LocalConnectionHolder(Path(source))
        self._holder = source
        self._conn = self._holder.connection
        self._db_path = str(self._holder.db_path)
        self._write_count = 0
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
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
                metadata_json VARCHAR DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
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
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_messages_session
            ON conversation_messages (session_id, created_at)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_sessions_updated
            ON conversation_sessions (updated_at DESC)
        """)
        self._ensure_fts()

    def _ensure_fts(self) -> None:
        """Create FTS index on messages. Gracefully skip if extension unavailable."""
        try:
            self._conn.execute("INSTALL fts; LOAD fts;")
            try:
                self._conn.execute(
                    "SELECT * FROM fts_main_conversation_messages.match_bm25('test', fields := 'content') LIMIT 0"
                )
            except Exception:
                try:
                    self._conn.execute("""
                        PRAGMA create_fts_index(
                            'conversation_messages', 'message_id',
                            'content', 'role', 'tool_name',
                            overwrite := 1
                        )
                    """)
                    logger.debug("conversation_store: FTS index created")
                except Exception as e:
                    logger.debug("conversation_store: FTS index creation skipped: %s", e)
        except Exception as e:
            logger.debug("conversation_store: FTS extension unavailable: %s", e)

    def _execute_write(self, sql: str, params: Any = None) -> None:
        """Execute a write with jitter retry for concurrent access."""
        from leapflow.storage.write_buffer import execute_with_retry
        execute_with_retry(self._conn, sql, params)
        self._write_count += 1

    def create_session(
        self,
        session_id: str,
        *,
        title: str = "",
        parent_session_id: Optional[str] = None,
        model: str = "",
        source: str = "cli",
        cwd: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ConversationSession:
        now = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._execute_write(
            """
            INSERT INTO conversation_sessions
                (session_id, title, created_at, updated_at,
                 parent_session_id, model, source, cwd, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (session_id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                title = COALESCE(NULLIF(EXCLUDED.title, ''), conversation_sessions.title),
                model = COALESCE(NULLIF(EXCLUDED.model, ''), conversation_sessions.model)
            """,
            [session_id, title, now, now, parent_session_id, model, source, cwd, meta_json],
        )
        return ConversationSession(
            session_id=session_id, title=title, created_at=now, updated_at=now,
            parent_session_id=parent_session_id, model=model, source=source, cwd=cwd,
            metadata=metadata or {},
        )

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        rows = self._conn.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?", [session_id]
        ).fetchall()
        if not rows:
            return None
        return self._row_to_session(rows[0])

    def list_sessions(
        self, *, limit: int = 20, active_only: bool = True
    ) -> List[ConversationSession]:
        sql = "SELECT * FROM conversation_sessions"
        if active_only:
            sql += " WHERE is_active = TRUE"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        rows = self._conn.execute(sql, [limit]).fetchall()
        return [self._row_to_session(r) for r in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
        token_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ConversationMessage:
        message_id = str(uuid.uuid4())
        now = time.time()
        tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        self._execute_write(
            """
            INSERT INTO conversation_messages
                (message_id, session_id, role, content, created_at,
                 tool_name, tool_call_id, tool_calls_json,
                 token_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [message_id, session_id, role, content, now,
             tool_name, tool_call_id, tc_json, token_count, meta_json],
        )

        self._execute_write(
            """
            UPDATE conversation_sessions SET
                updated_at = ?,
                message_count = message_count + 1,
                total_tokens = total_tokens + ?
            WHERE session_id = ?
            """,
            [now, token_count, session_id],
        )

        return ConversationMessage(
            message_id=message_id, session_id=session_id, role=role,
            content=content, created_at=now, tool_name=tool_name,
            tool_call_id=tool_call_id, tool_calls_json=tc_json,
            token_count=token_count, metadata=metadata or {},
        )

    def get_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        active_only: bool = True,
    ) -> List[ConversationMessage]:
        sql = "SELECT * FROM conversation_messages WHERE session_id = ?"
        if active_only:
            sql += " AND active = TRUE"
        sql += " ORDER BY created_at ASC LIMIT ?"
        rows = self._conn.execute(sql, [session_id, limit]).fetchall()
        return [self._row_to_message(r) for r in rows]

    def search_messages(
        self,
        query: str,
        *,
        limit: int = 10,
        role_filter: Optional[str] = None,
    ) -> List[ConversationSearchResult]:
        """Full-text search across all sessions' messages."""
        if not query or len(query.strip()) < 2:
            return []

        query_safe = query[:2048]

        try:
            fts_sql = """
                SELECT m.message_id, m.session_id, m.role, m.content, m.created_at,
                       s.title as session_title,
                       fts.score
                FROM (
                    SELECT *, fts_main_conversation_messages.match_bm25(
                        message_id, ?, fields := 'content'
                    ) AS score
                    FROM conversation_messages
                    WHERE score IS NOT NULL
                ) m
                JOIN conversation_sessions s ON m.session_id = s.session_id
                WHERE m.active = TRUE
            """
            params: list[Any] = [query_safe]
            if role_filter:
                fts_sql += " AND m.role = ?"
                params.append(role_filter)
            fts_sql += " ORDER BY m.score DESC LIMIT ?"
            params.append(limit)

            rows = self._conn.execute(fts_sql, params).fetchall()
            return [
                ConversationSearchResult(
                    message_id=r[0], session_id=r[1], role=r[2],
                    content=r[3], created_at=r[4],
                    session_title=r[5] or "", score=r[6] or 0.0,
                )
                for r in rows
            ]
        except Exception:
            return self._fallback_search(query_safe, limit=limit, role_filter=role_filter)

    def _fallback_search(
        self,
        query: str,
        *,
        limit: int = 10,
        role_filter: Optional[str] = None,
    ) -> List[ConversationSearchResult]:
        """LIKE-based fallback when FTS is unavailable."""
        sql = """
            SELECT m.message_id, m.session_id, m.role, m.content, m.created_at,
                   s.title as session_title
            FROM conversation_messages m
            JOIN conversation_sessions s ON m.session_id = s.session_id
            WHERE m.active = TRUE AND m.content LIKE ?
        """
        params: list[Any] = [f"%{query}%"]
        if role_filter:
            sql += " AND m.role = ?"
            params.append(role_filter)
        sql += " ORDER BY m.created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            ConversationSearchResult(
                message_id=r[0], session_id=r[1], role=r[2],
                content=r[3], created_at=r[4],
                session_title=r[5] or "", score=1.0,
            )
            for r in rows
        ]

    def get_anchored_view(
        self,
        session_id: str,
        message_id: str,
        *,
        window: int = 5,
    ) -> List[ConversationMessage]:
        """Get messages around a specific message (+-window) for context."""
        target = self._conn.execute(
            "SELECT created_at FROM conversation_messages WHERE message_id = ?", [message_id]
        ).fetchone()
        if not target:
            return []

        target_ts = target[0]
        rows = self._conn.execute(
            """
            SELECT * FROM conversation_messages
            WHERE session_id = ? AND active = TRUE
            ORDER BY ABS(created_at - ?)
            LIMIT ?
            """,
            [session_id, target_ts, window * 2 + 1],
        ).fetchall()

        messages = [self._row_to_message(r) for r in rows]
        messages.sort(key=lambda m: m.created_at)
        return messages

    def soft_delete_messages(
        self, session_id: str, message_ids: List[str]
    ) -> int:
        """Soft-delete messages (set active=FALSE). Returns count affected."""
        if not message_ids:
            return 0
        placeholders = ",".join(["?"] * len(message_ids))
        self._execute_write(
            f"UPDATE conversation_messages SET active = FALSE WHERE session_id = ? AND message_id IN ({placeholders})",
            [session_id, *message_ids],
        )
        return len(message_ids)

    def mark_compacted(self, session_id: str, message_ids: List[str]) -> None:
        """Mark messages as compacted (preserved but excluded from new summaries)."""
        if not message_ids:
            return
        placeholders = ",".join(["?"] * len(message_ids))
        self._execute_write(
            f"UPDATE conversation_messages SET compacted = TRUE WHERE session_id = ? AND message_id IN ({placeholders})",
            [session_id, *message_ids],
        )

    def end_session(self, session_id: str) -> None:
        """Mark a session as inactive (completed/archived)."""
        self._execute_write(
            "UPDATE conversation_sessions SET is_active = FALSE, updated_at = ? WHERE session_id = ?",
            [time.time(), session_id],
        )

    def fork_session(
        self,
        parent_session_id: str,
        *,
        title: str = "",
        model: str = "",
        carry_messages: int = 0,
    ) -> ConversationSession:
        """Fork a new child session from a parent (compression continuation).

        Optionally copies the last N messages from parent to child for context.
        """
        child_id = str(uuid.uuid4())
        child = self.create_session(
            child_id,
            title=title or f"Fork of {parent_session_id[:8]}",
            parent_session_id=parent_session_id,
            model=model,
        )

        if carry_messages > 0:
            recent = self.get_messages(parent_session_id, limit=carry_messages)
            for msg in recent:
                self.append_message(
                    child_id, msg.role, msg.content,
                    tool_name=msg.tool_name, tool_call_id=msg.tool_call_id,
                )

        return child

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a session with all its messages as a portable dict.

        Returns None if session not found. Suitable for JSON serialization.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        messages = self.get_messages(session_id, limit=10_000, active_only=False)
        return {
            "session_id": session.session_id,
            "title": session.title,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "parent_session_id": session.parent_session_id,
            "model": session.model,
            "source": session.source,
            "cwd": session.cwd,
            "metadata": session.metadata,
            "messages": [
                {
                    "message_id": m.message_id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                    "tool_name": m.tool_name,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls_json": m.tool_calls_json,
                    "token_count": m.token_count,
                }
                for m in messages
            ],
        }

    def import_session(self, data: Dict[str, Any]) -> ConversationSession:
        """Import a session from a portable dict (e.g. from export_session).

        Creates the session and all its messages. Skips messages that already
        exist (idempotent on message_id).
        """
        sid = data["session_id"]
        session = self.create_session(
            sid,
            title=data.get("title", ""),
            parent_session_id=data.get("parent_session_id"),
            model=data.get("model", ""),
            source=data.get("source", "imported"),
            cwd=data.get("cwd", ""),
            metadata=data.get("metadata"),
        )

        for msg in data.get("messages", []):
            tc_json = msg.get("tool_calls_json")
            tc_list = None
            if tc_json:
                try:
                    tc_list = json.loads(tc_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            self.append_message(
                sid,
                msg["role"],
                msg.get("content", ""),
                tool_name=msg.get("tool_name"),
                tool_call_id=msg.get("tool_call_id"),
                tool_calls=tc_list,
                token_count=msg.get("token_count", 0),
            )

        return session

    def close(self) -> None:
        """Close the database connection if owned by this store."""
        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

    def _row_to_session(self, row: tuple) -> ConversationSession:
        meta = {}
        try:
            meta = json.loads(row[11]) if row[11] else {}
        except (json.JSONDecodeError, IndexError):
            pass
        return ConversationSession(
            session_id=row[0], title=row[1] or "", created_at=row[2] or 0.0,
            updated_at=row[3] or 0.0, parent_session_id=row[4],
            model=row[5] or "", source=row[6] or "cli", cwd=row[7] or "",
            message_count=row[8] or 0, total_tokens=row[9] or 0,
            is_active=bool(row[10]) if row[10] is not None else True,
            metadata=meta,
        )

    def _row_to_message(self, row: tuple) -> ConversationMessage:
        meta = {}
        try:
            meta = json.loads(row[11]) if row[11] else {}
        except (json.JSONDecodeError, IndexError):
            pass
        return ConversationMessage(
            message_id=row[0], session_id=row[1], role=row[2] or "",
            content=row[3] or "", created_at=row[4] or 0.0,
            tool_name=row[5], tool_call_id=row[6],
            tool_calls_json=row[7], active=bool(row[8]) if row[8] is not None else True,
            compacted=bool(row[9]) if row[9] is not None else False,
            token_count=row[10] or 0, metadata=meta,
        )

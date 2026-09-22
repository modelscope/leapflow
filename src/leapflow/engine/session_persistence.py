# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session persistence and memory-prefetch helpers for :class:`AgentEngine`.

Extracted from ``engine.py`` (Phase 3 refactor). This component owns session
resume/load, conversation-store persistence, prefix-freeze on resume, and the
session-start memory prefetch snapshot. It holds a back-reference to the owning
engine so every access reads the engine's *live* mutable state (stores injected
at runtime via ``set_*`` methods), preserving exact runtime semantics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from leapflow.llm.message_builder import (
    build_assistant_message,
    build_user_message_text,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.engine.engine import AgentEngine
    from leapflow.engine.agent_loop import AgentLoopFrame

logger = logging.getLogger(__name__)


class SessionPersistence:
    """Session load/resume and conversation persistence, held by composition."""

    def __init__(self, engine: "AgentEngine") -> None:
        self._engine = engine

    def load_session(self, session_id: str) -> bool:
        """Resume a previous session by loading messages from DuckDB.

        Returns True if the session was found and messages loaded.
        """
        if not self._engine._conversation_store:
            return False
        try:
            messages = self._engine._conversation_store.get_messages(session_id, limit=500)
            if not messages:
                return False
            self._engine._current_session_id = session_id
            for msg in messages:
                role = msg.role
                content = msg.content
                if role == "user":
                    self._engine._wm.remember_chat(build_user_message_text(content))
                elif role == "assistant":
                    self._engine._wm.remember_chat(build_assistant_message(content))
            logger.info("session.resume loaded %d messages from %s", len(messages), session_id)
            self.apply_resume_cache_snapshot(session_id)
            return True
        except Exception:
            logger.debug("session.resume failed", exc_info=True)
            return False

    def freeze_prefix_for_resume(
        self,
        *,
        system_prompt: Optional[str],
        tool_schema: Optional[str],
        disclosure_level: Optional[str],
    ) -> None:
        """Freeze a persisted prefix so the next turn reproduces it verbatim (5c).

        Sets the resume-freeze fields consumed once by the next
        ``_assemble_unified_prompt`` and force-commits the controller so the
        first resumed turn enters ``COMMITTED`` and the provider prefix cache is
        hit immediately. The commitment is re-applied inside prompt assembly
        because ``_begin_turn_context`` resets the controller at each turn start;
        the frozen fields (independent of commitment state) are what survive to
        drive that re-application.
        """
        self._engine._frozen_system_prompt = system_prompt or None
        self._engine._frozen_tool_schema = tool_schema or None
        self._engine._last_disclosure_level = str(disclosure_level or "")
        self._engine._prefix_commitment.force_commit()

    def apply_resume_cache_snapshot(self, session_id: str) -> bool:
        """Load and apply a persisted prefix snapshot on resume (5c).

        Honors ``session_resume_cache_policy``: ``cache_priority`` (default)
        freezes the persisted system prompt / tool schema so the first resumed
        turn is a cache hit; ``tool_freshness`` skips the freeze and lets normal
        PCD rediscover tools. Best-effort and gated on a conversation store that
        implements ``get_session_snapshot``; any failure or missing snapshot
        degrades to a normal (non-frozen) resume. Returns whether a freeze was
        applied.
        """
        if not session_id or not self._engine._conversation_store:
            return False
        policy = str(
            getattr(self._engine._settings, "session_resume_cache_policy", "cache_priority")
            or "cache_priority"
        )
        if policy != "cache_priority":
            return False
        getter = getattr(self._engine._conversation_store, "get_session_snapshot", None)
        if getter is None:
            return False
        try:
            snapshot = getter(session_id)
        except Exception:  # noqa: BLE001 - resume must never fail on an aux read
            logger.debug("session.resume snapshot load failed", exc_info=True)
            return False
        if snapshot is None:
            return False
        system_prompt = getattr(snapshot, "system_prompt", None)
        if not system_prompt:
            return False
        self.freeze_prefix_for_resume(
            system_prompt=system_prompt,
            tool_schema=getattr(snapshot, "tool_schema", None),
            disclosure_level=getattr(snapshot, "disclosure_level", None),
        )
        logger.info("session.resume applied cache-priority prefix freeze for %s", session_id)
        return True

    def _ensure_session_for_frame(
        self, frame: "AgentLoopFrame", user_text: str
    ) -> Optional[str]:
        """Resolve the persistence session for a loop frame (S4-E isolation).

        Root frames reuse the turn's conversation session; a recursive child
        frame (subagent) gets its *own* isolated ``sub_`` session so its
        transcript is persisted separately and never mixes into the parent
        turn's conversation.
        """
        if frame.is_root:
            return self._ensure_session(user_text)
        if (
            not self._engine._conversation_store
            or not self._engine._settings.session_persistence_enabled
        ):
            return None
        try:
            import uuid as _uuid

            child_session = f"sub_{_uuid.uuid4().hex[:12]}"
            title = user_text[:80].replace("\n", " ").strip() or "subagent"
            self._engine._conversation_store.create_session(
                child_session,
                title=title,
                model=self._engine._settings.llm_model,
                source="subagent",
            )
            return child_session
        except Exception:
            logger.debug("child session creation failed; skipping child persistence", exc_info=True)
            return None

    def _ensure_session(self, user_text: str) -> Optional[str]:
        """Create or reuse a conversation session. Returns session_id or None."""
        if (
            not self._engine._conversation_store
            or not self._engine._settings.session_persistence_enabled
        ):
            return None
        try:
            import uuid as _uuid

            if self._engine._current_session_id is None:
                self._engine._current_session_id = _uuid.uuid4().hex[:16]
            # Create the session row if it does not exist yet. This covers a
            # freshly-minted id and a client-provided id alike (e.g. a distinct
            # per-TUI session bound by the daemon), so persistence works no matter
            # who chose the id.
            if self._engine._conversation_store.get_session(self._engine._current_session_id) is None:
                title = user_text[:80].replace("\n", " ").strip()
                self._engine._conversation_store.create_session(
                    self._engine._current_session_id,
                    title=title,
                    model=self._engine._settings.llm_model,
                    source="cli",
                    cwd=str(getattr(self._engine._settings, "workspace_root", "") or ""),
                )
            self._persist_message(self._engine._current_session_id, "user", user_text)
            return self._engine._current_session_id
        except Exception:
            logger.debug("session.ensure failed", exc_info=True)
            return None

    def _persist_message(
        self,
        session_id: Optional[str],
        role: str,
        content: str,
        *,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a message to conversation store (fire-and-forget)."""
        if not session_id or not self._engine._conversation_store:
            return
        try:
            self._engine._conversation_store.append_message(
                session_id,
                role,
                content[:8000],
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        except Exception:
            logger.debug("session.persist_message failed", exc_info=True)

    def _persist_session_snapshot(self, session_id: Optional[str]) -> None:
        """Persist the current committed prefix for cache-priority resume (5b).

        Records the system prompt, tool schema (JSON), and disclosure level that
        this turn actually assembled so a later ``build_session_engine`` resume
        can reproduce a byte-identical prefix and hit the provider cache on its
        first request. Fire-and-forget and gated on session persistence: an
        auxiliary snapshot must never fail or slow the main turn.
        """
        if not session_id or not self._engine._conversation_store:
            return
        if not self._engine._settings.session_persistence_enabled:
            return
        if not self._engine._last_system_prompt:
            return
        updater = getattr(self._engine._conversation_store, "update_session_snapshot", None)
        if updater is None:
            return
        try:
            updater(
                session_id,
                system_prompt=self._engine._last_system_prompt,
                tool_schema=self._engine._last_tool_definitions_json or None,
                disclosure_level=self._engine._last_disclosure_level or None,
            )
        except Exception:
            logger.debug("session.persist_snapshot failed", exc_info=True)

    async def _prefetch_and_freeze_memory(self, user_text: str) -> str:
        """Prefetch memory context and freeze snapshot for session duration.

        Combines narrative memory (always-on MEMORY.md) with signal-based
        prefetch results into a unified context block.
        """
        if self._engine._memory_context_snapshot is not None:
            return self._engine._memory_context_snapshot

        if not self._engine._memory_manager or not self._engine._settings.memory_integration_enabled:
            self._engine._memory_context_snapshot = ""
            return ""

        parts: list[str] = []

        # Layer 1: Narrative memory (MEMORY.md — always loaded, no timeout)
        narrative = self._engine._memory_manager.get_provider("narrative")
        if narrative is not None and hasattr(narrative, "context_block"):
            try:
                block = narrative.context_block()
                if block:
                    parts.append(block)
            except Exception:
                logger.debug("narrative.context_block failed", exc_info=True)

        # Layer 2: Signal-based prefetch (DuckDB — timeout-bounded)
        try:
            entries = await asyncio.wait_for(
                self._engine._memory_manager.prefetch(
                    user_text,
                    limit=self._engine._settings.memory_prefetch_limit,
                    workspace_root=(
                        self._engine._current_task_contract.workspace_root
                        if self._engine._current_task_contract
                        else ""
                    ),
                    task_id=(
                        self._engine._current_task_contract.task_id
                        if self._engine._current_task_contract
                        else ""
                    ),
                    scope_keywords=self._engine._prompt_assembler._task_scope_keywords(user_text),
                    session_scope="",
                ),
                timeout=self._engine._settings.memory_prefetch_timeout_s,
            )
            if entries:
                parts.append(
                    "## Recent Context\n"
                    + "\n".join(f"- [{e.kind.value}] {e.content[:500]}" for e in entries)
                )
        except asyncio.TimeoutError:
            logger.debug(
                "memory.prefetch timed out (%.1fs)",
                self._engine._settings.memory_prefetch_timeout_s,
            )
        except Exception:
            logger.debug("memory.prefetch failed", exc_info=True)

        # Layer 0: Recent task history (session summaries)
        try:
            semantic = self._engine._memory_manager.get_provider("semantic")
            if semantic is not None and hasattr(semantic, "query_recent_summaries"):
                summaries = semantic.query_recent_summaries(limit=5)
                if summaries:
                    history_lines = []
                    for s in summaries:
                        history_lines.append(f"- {s['content'][:300]}")
                    history_block = "## Recent Task History\n" + "\n".join(history_lines)
                    parts.insert(0, history_block)
        except Exception:
            logger.debug("Layer 0 task history injection failed", exc_info=True)

        self._engine._memory_context_snapshot = "\n\n".join(parts)
        return self._engine._memory_context_snapshot

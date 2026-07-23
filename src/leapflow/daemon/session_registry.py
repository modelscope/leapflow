"""Session-scoped execution registry for the daemon (Stage 3, P3-2a).

Maps a ``session_id`` to a :class:`SessionExecutionContext` — the per-session
engine (built via ``build_session_engine``) plus its own turn lock. Concurrent
turns of *different* sessions therefore run on *different* engine instances
(isolated substrate), while turns *within* a session serialize on the session
lock. A daemon-wide semaphore (wired in P3-2b/P3-4) bounds total concurrency.

The first session to arrive reuses the daemon's existing base engine, so a
single-session daemon (the common case) is byte-for-byte unchanged; only
additional concurrent sessions get isolated per-session engines.

This module is pure infrastructure: it does not import daemon internals and is
unit-tested in isolation. Wiring into ``engine_chat`` (session-id routing) is
P3-2b. See ``temp/plan/concurrent_turns_stage3.md`` §4.1/4.4.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional


class SessionExecutionContext:
    """Per-session execution state: an engine + a turn lock + activity clock."""

    def __init__(self, session_id: str, engine: Any) -> None:
        self.session_id = session_id
        self.engine = engine
        self.lock = asyncio.Lock()  # serialize this session's own turns
        self.last_active = time.monotonic()

    def touch(self) -> None:
        self.last_active = time.monotonic()


class SessionRegistry:
    """Create/reuse per-session execution contexts, bounded and idle-evicted.

    Parameters
    ----------
    base_engine:
        The daemon's existing wired engine. The first session reuses it so a
        single-session daemon is unchanged.
    build_engine:
        ``(base_engine, session_id, working_memory) -> engine`` — normally
        ``leapflow.engine.session_factory.build_session_engine`` (adapted).
    build_working_memory:
        ``() -> WorkingMemoryProvider`` — a fresh per-session working memory.
    max_sessions / idle_ttl_s:
        Registry bound + idle eviction (protect memory).
    """

    def __init__(
        self,
        *,
        base_engine: Any,
        build_engine: Callable[[Any, str, Any], Any],
        build_working_memory: Callable[[], Any],
        max_sessions: int = 16,
        idle_ttl_s: float = 1800.0,
    ) -> None:
        self._base = base_engine
        self._build_engine = build_engine
        self._build_wm = build_working_memory
        self._max_sessions = max(1, int(max_sessions))
        self._idle_ttl_s = float(idle_ttl_s)
        self._contexts: Dict[str, SessionExecutionContext] = {}
        self._primary_session_id: Optional[str] = None
        self._lock = asyncio.Lock()  # guards registry mutation

    async def acquire(self, session_id: str) -> SessionExecutionContext:
        """Return the context for ``session_id``, creating it if needed."""
        sid = str(session_id or "")
        async with self._lock:
            self._evict_idle()
            existing = self._contexts.get(sid)
            if existing is not None:
                existing.touch()
                return existing
            if self._primary_session_id is None:
                # First session reuses the base engine → single-session daemon
                # behaves exactly as before.
                self._primary_session_id = sid
                engine = self._base
            else:
                if len(self._contexts) >= self._max_sessions:
                    self._evict_oldest()
                engine = self._build_engine(self._base, sid, self._build_wm())
            ctx = SessionExecutionContext(sid, engine)
            self._contexts[sid] = ctx
            return ctx

    def _evict_idle(self) -> None:
        if self._idle_ttl_s <= 0:
            return
        now = time.monotonic()
        for sid in [s for s, c in self._contexts.items()
                    if s != self._primary_session_id and (now - c.last_active) > self._idle_ttl_s]:
            del self._contexts[sid]

    def _evict_oldest(self) -> None:
        # Never evict the primary (base-engine) session.
        candidates: List[SessionExecutionContext] = [
            c for s, c in self._contexts.items() if s != self._primary_session_id
        ]
        if not candidates:
            return
        oldest = min(candidates, key=lambda c: c.last_active)
        del self._contexts[oldest.session_id]

    def active_count(self) -> int:
        return len(self._contexts)

    def session_ids(self) -> List[str]:
        return list(self._contexts.keys())

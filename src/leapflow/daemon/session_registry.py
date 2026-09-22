# Copyright (c) Alibaba, Inc. and its affiliates.
"""Session-scoped execution registry for the daemon (Stage 3, P3-2a).

Maps a ``session_id`` to a :class:`SessionExecutionContext` — the per-session
engine (built via ``build_session_engine``) plus its own turn lock. Concurrent
turns of *different* sessions therefore run on *different* engine instances
(isolated substrate), while turns *within* a session serialize on the session
lock. A daemon-wide semaphore (wired in P3-2b/P3-4) bounds total concurrency.

Every session — including the first — gets its own engine built via
``build_session_engine``, so all sessions follow one homogeneous path and the
daemon's base engine is never mutated by conversation state. Consumers that
need a session's live state (e.g. session-analysis watches feeding LeapBoard)
must therefore resolve it through this registry rather than reading the base
engine, which carries no conversation.

This module is pure infrastructure: it does not import daemon internals and is
unit-tested in isolation. Wiring into ``engine_chat`` (session-id routing) is
P3-2b. See ``temp/plan/concurrent_turns_stage3.md`` §4.1/4.4.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class WorkspaceMismatchError(ValueError):
    """Raised when one session id is reused from a different workspace root.

    Reaching this from ordinary use means a client sent a session id it does not
    own — a defect, not something the user can act on. The only legitimate cause
    is an explicit ``--resume`` of a session created in another workspace, which
    is why the guidance names that case instead of telling the user to do what
    they already did.
    """

    def __init__(self, session_id: str, expected: Path, requested: Path) -> None:
        super().__init__(
            f"Session {session_id!r} belongs to workspace {expected}, but this request "
            f"came from {requested}. If you resumed it with --resume, resume it from "
            f"{expected} instead, or omit --resume to start a session for this workspace."
        )
        self.session_id = session_id
        self.expected = expected
        self.requested = requested


class SessionExecutionContext:
    """Per-session execution state: an engine + workspace + turn lock."""

    # Coarse clocks quantize: on Windows, time.monotonic resolves to ~15.6ms,
    # so two activities can share a timestamp. This counter stamps every
    # activity strictly after the previous one, keeping "most recent" and
    # "oldest" orderings deterministic.
    _activity_counter = 0

    def __init__(self, session_id: str, engine: Any, workspace_root: Path) -> None:
        self.session_id = session_id
        self.engine = engine
        self.workspace_root = workspace_root
        self.lock = asyncio.Lock()  # serialize this session's own turns
        self.last_active = time.monotonic()
        SessionExecutionContext._activity_counter += 1
        self.activity_seq = SessionExecutionContext._activity_counter

    def touch(self) -> None:
        self.last_active = time.monotonic()
        SessionExecutionContext._activity_counter += 1
        self.activity_seq = SessionExecutionContext._activity_counter


class SessionRegistry:
    """Create/reuse per-session execution contexts, bounded and idle-evicted.

    Parameters
    ----------
    base_engine:
        The daemon's existing wired engine. The first session reuses it so a
        single-session daemon is unchanged.
    build_engine:
        ``(base_engine, session_id, working_memory) -> engine`` — normally
        ``leapflow.engine.session.session_factory.build_session_engine`` (adapted).
    build_working_memory:
        ``() -> WorkingMemoryProvider`` — a fresh per-session working memory.
    max_sessions / idle_ttl_s:
        Registry bound + idle eviction (protect memory).
    """

    def __init__(
        self,
        *,
        base_engine: Any,
        build_engine: Callable[[Any, str, Any, Path], Any],
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

    def _default_workspace_root(self) -> Path:
        settings = getattr(self._base, "_settings", None)
        root = getattr(settings, "workspace_root", Path.cwd())
        return Path(str(root)).expanduser().resolve()

    async def acquire(
        self,
        session_id: str,
        workspace_root: str | Path | None = None,
    ) -> SessionExecutionContext:
        """Return the context for ``session_id``, creating it if needed.

        A session is bound to the workspace from its first request. Reusing the
        same session id from another workspace is rejected because it would mix
        one conversation's memory, task contract, and tool path boundary across
        projects.
        """
        sid = str(session_id or "")
        requested_root = (
            Path(str(workspace_root)).expanduser().resolve()
            if workspace_root else None
        )
        async with self._lock:
            self._evict_idle()
            existing = self._contexts.get(sid)
            if existing is not None:
                if requested_root is not None and requested_root != existing.workspace_root:
                    raise WorkspaceMismatchError(sid, existing.workspace_root, requested_root)
                existing.touch()
                return existing
            root = requested_root or self._default_workspace_root()
            if self._primary_session_id is None:
                self._primary_session_id = sid
            elif len(self._contexts) >= self._max_sessions:
                self._evict_oldest()
            engine = self._build_engine(self._base, sid, self._build_wm(), root)
            ctx = SessionExecutionContext(sid, engine, root)
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
        oldest = min(candidates, key=lambda c: (c.last_active, c.activity_seq))
        del self._contexts[oldest.session_id]

    def active_count(self) -> int:
        return len(self._contexts)

    def session_ids(self) -> List[str]:
        return list(self._contexts.keys())

    def get(self, session_id: str) -> Optional[SessionExecutionContext]:
        """Return an existing session context without creating one.

        Read-only lookup for consumers that observe a session (e.g. the
        session-analysis watch) and must never materialize an engine.
        """
        return self._contexts.get(str(session_id or ""))

    def most_recent_any_client(self) -> Optional[SessionExecutionContext]:
        """Return the most recently active session of *any* client, if any.

        Named for what it actually does. It ignores workspace and client
        identity, so it is only valid for genuinely cross-session views (an
        aggregate dashboard). Using it to answer "the caller's session" leaks one
        client's session id and context figures to another, which is how a second
        TUI ended up sending a session bound to a different workspace.
        """
        if not self._contexts:
            return None
        return max(
            self._contexts.values(),
            key=lambda ctx: (ctx.last_active, ctx.activity_seq),
        )

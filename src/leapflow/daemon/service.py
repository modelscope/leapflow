"""Runtime-backed LeapService implementation for leapd."""
from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from leapflow.daemon.protocol import StreamChunk
from leapflow.engine import StreamEvent
from leapflow.memory.protocol import MemoryEntry, MemoryQuery

logger = logging.getLogger(__name__)


class RuntimeLeapService:
    """LeapService implementation backed by a single initialized Context."""

    def __init__(self, settings: Any, *, mock_host: bool = False) -> None:
        self._settings = settings
        self._mock_host = mock_host
        self._ctx: Any | None = None
        self._started_at = time.time()
        self._client_count: Callable[[], int] = lambda: 0

    def set_client_count_provider(self, provider: Callable[[], int]) -> None:
        """Set a lightweight callback used by status reporting."""
        self._client_count = provider

    async def start(self) -> None:
        """Initialize the daemon-owned runtime once."""
        if self._ctx is not None:
            return
        from leapflow.cli.context import Context

        ctx = Context(self._settings, self._mock_host)
        await ctx.initialize()
        self._ctx = ctx

    @property
    def context(self) -> Any:
        """Return the initialized Context or raise a clear lifecycle error."""
        if self._ctx is None:
            raise RuntimeError("leapd runtime is not initialized")
        return self._ctx

    async def signal_record(self, signal_data: dict[str, Any]) -> dict[str, Any]:
        ctx = self.context
        event_type = str(signal_data.get("type") or "daemon.signal")
        payload = dict(signal_data.get("payload") or {})
        await ctx.event_bus.handle_event(event_type, payload)
        return {"ok": True}

    async def memory_search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        ctx = self.context
        memory_query = MemoryQuery(keywords=query.split()[:8], limit=limit)
        results = await ctx.memory.search(memory_query)
        return [self._memory_entry_to_dict(item) for item in results]

    async def memory_insert(self, content: str, kind: str = "fact", **kwargs: Any) -> str:
        ctx = self.context
        metadata = dict(kwargs.get("metadata") or {})
        entry_id = ctx.lt.ingest(kind, content, metadata=metadata)
        return str(entry_id)

    async def session_create(self, **kwargs: Any) -> dict[str, Any]:
        ctx = self.context
        session_id = getattr(ctx.session, "session_id", "") if ctx.session else ""
        return {"session_id": str(session_id), "created": bool(session_id), **kwargs}

    async def session_resume(self, session_id: str) -> dict[str, Any]:
        ctx = self.context
        engine = getattr(ctx, "engine", None)
        found = bool(engine and engine.load_session(session_id))
        current = getattr(engine, "_current_session_id", "") if engine else ""
        return {"found": found, "session_id": str(current or session_id)}

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        ctx = self.context
        engine = getattr(ctx, "engine", None)
        if engine is None:
            raise RuntimeError("leapd engine is not initialized")

        enable_thinking = bool(kwargs.get("enable_thinking", False))
        async for event in engine.run_stream(message, enable_thinking=enable_thinking):
            stream_event = self._normalize_event(event)
            yield self._chunk_from_event(stream_event)

    async def engine_cancel(self) -> bool:
        ctx = self.context
        engine = getattr(ctx, "engine", None)
        if engine is not None and hasattr(engine, "cancel"):
            result = engine.cancel()
            if hasattr(result, "__await__"):
                await result
            return True
        return False

    async def skill_execute(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("skill.execute is not available in this daemon phase")

    async def scheduler_arm(self, task_config: dict[str, Any]) -> str:
        raise NotImplementedError("scheduler.arm is not available in this daemon phase")

    async def status(self) -> dict[str, Any]:
        ctx = self._ctx
        engine = getattr(ctx, "engine", None) if ctx is not None else None
        db_holder = getattr(ctx, "_db_holder", None) if ctx is not None else None
        return {
            "pid": os.getpid(),
            "profile": getattr(self._settings, "profile", "default"),
            "profile_dir": str(getattr(self._settings, "profile_dir", "")),
            "db_path": str(getattr(db_holder, "db_path", self._settings.duckdb_path)),
            "volatile": bool(getattr(ctx, "storage_volatile", False)) if ctx is not None else False,
            "uptime_s": max(0.0, time.time() - self._started_at),
            "active_clients": max(0, self._client_count()),
            "model": getattr(self._settings, "llm_model", ""),
            "session_id": str(getattr(engine, "_current_session_id", "") or ""),
        }

    async def shutdown(self) -> None:
        if self._ctx is None:
            return
        ctx = self._ctx
        self._ctx = None
        self._checkpoint_open_connection(ctx)
        await ctx.cleanup()

    async def gateway_connect(
        self,
        platform: str,
        credentials: dict[str, str],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("gateway.connect is not available in this daemon phase")

    async def gateway_disconnect(self, platform: str) -> dict[str, Any]:
        raise NotImplementedError("gateway.disconnect is not available in this daemon phase")

    async def gateway_status(self) -> list[dict[str, Any]]:
        raise NotImplementedError("gateway.status is not available in this daemon phase")

    async def gateway_send(
        self,
        platform: str,
        chat_id: str,
        text: str,
        thread_id: str = "",
    ) -> dict[str, Any]:
        raise NotImplementedError("gateway.send is not available in this daemon phase")

    def _memory_entry_to_dict(self, entry: MemoryEntry) -> dict[str, Any]:
        return {
            "entry_id": entry.entry_id,
            "kind": entry.kind.value,
            "domain": entry.domain.value,
            "content": entry.content,
            "timestamp": entry.timestamp,
            "score": entry.score,
            "metadata": dict(entry.metadata),
        }

    def _checkpoint_open_connection(self, ctx: Any) -> None:
        holder = getattr(ctx, "_db_holder", None)
        conn = getattr(holder, "_conn", None)
        if conn is None:
            return
        try:
            conn.execute("CHECKPOINT")
        except Exception:
            logger.debug("daemon: DuckDB checkpoint skipped", exc_info=True)

    def _normalize_event(self, event: object) -> StreamEvent:
        if isinstance(event, StreamEvent):
            return event
        return StreamEvent(type="chunk", content=str(event), metadata=None)

    def _chunk_from_event(self, event: StreamEvent) -> StreamChunk:
        ctx = self.context
        engine = getattr(ctx, "engine", None)
        metadata = dict(event.metadata or {})
        session_id = getattr(engine, "_current_session_id", "") if engine else ""
        if session_id:
            metadata.setdefault("session_id", str(session_id))
        return StreamChunk(
            request_id="",
            content=event.content,
            done=False,
            event_type=event.type,
            metadata=metadata,
        )

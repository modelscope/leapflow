"""Runtime-backed LeapService implementation for leapd."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
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
        self._engine_lock = asyncio.Lock()
        self._started_at = time.time()
        self._client_count: Callable[[], int] = lambda: 0
        self._approval_pending: dict[str, dict[str, Any]] = {}
        self._approval_event_queue: asyncio.Queue[StreamChunk] | None = None

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
        self._install_daemon_approval(ctx)
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
        async with self._engine_lock:
            ctx = self.context
            if ctx.reload_runtime_config_if_changed():
                self._settings = ctx.settings
                yield StreamChunk(
                    request_id="",
                    content="Configuration reloaded — LLM settings updated in leapd.",
                    event_type="status",
                    metadata={
                        "llm_model": getattr(ctx.settings, "llm_model", ""),
                        "llm_context_length": getattr(ctx.settings, "llm_context_length", 0),
                        "context_used": getattr(getattr(ctx, "engine", None), "context_token_count", 0),
                    },
                )
            engine = getattr(ctx, "engine", None)
            if engine is None:
                raise RuntimeError("leapd engine is not initialized")

            enable_thinking = bool(kwargs.get("enable_thinking", False))
            approval_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
            previous_queue = self._approval_event_queue
            self._approval_event_queue = approval_queue
            try:
                stream = engine.run_stream(message, enable_thinking=enable_thinking)
                async for chunk in self._stream_engine_events(stream, approval_queue):
                    yield chunk
            finally:
                self._approval_event_queue = previous_queue

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
        settings = getattr(ctx, "settings", self._settings) if ctx is not None else self._settings
        engine = getattr(ctx, "engine", None) if ctx is not None else None
        db_holder = getattr(ctx, "_db_holder", None) if ctx is not None else None
        config_path = os.path.join(str(getattr(settings, "data_dir", "")), ".env")
        project_env_path = os.path.join(os.getcwd(), ".env")
        return {
            "pid": os.getpid(),
            "profile": getattr(settings, "profile", "default"),
            "profile_dir": str(getattr(settings, "profile_dir", "")),
            "config_path": config_path,
            "project_env_path": project_env_path,
            "db_path": str(getattr(db_holder, "db_path", settings.duckdb_path)),
            "volatile": bool(getattr(ctx, "storage_volatile", False)) if ctx is not None else False,
            "uptime_s": max(0.0, time.time() - self._started_at),
            "active_clients": max(0, self._client_count()),
            "model": getattr(settings, "llm_model", ""),
            "llm_context_length": getattr(settings, "llm_context_length", 0),
            "context_used": getattr(engine, "context_token_count", 0) if engine is not None else 0,
            "session_id": str(getattr(engine, "_current_session_id", "") or ""),
            "runtime_source": self._runtime_source(),
            "runtime_executable": sys.executable,
            "runtime_version": self._runtime_version(),
            "pending_approvals": len(self._approval_pending),
        }

    async def approval_status(self) -> dict[str, Any]:
        """Return currently pending daemon approval requests."""
        return {"pending": self._pending_payloads()}

    async def approval_resolve(
        self,
        pending_id: str,
        decision: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Resolve a pending approval request from a thin client."""
        pending = self._approval_pending.get(pending_id)
        if pending is None:
            return {"ok": False, "error": f"Unknown approval request: {pending_id}"}
        future = pending.get("future")
        if not isinstance(future, asyncio.Future) or future.done():
            return {"ok": False, "error": f"Approval request is no longer pending: {pending_id}"}
        future.set_result({"decision": self._normalize_approval_decision(decision), "reason": reason})
        return {"ok": True, "pending_id": pending_id, "decision": self._normalize_approval_decision(decision)}

    async def approval_cancel(self, pending_id: str, reason: str = "cancelled") -> dict[str, Any]:
        """Cancel a pending approval request, causing the action to be denied."""
        return await self.approval_resolve(pending_id, "deny", reason=reason)

    @staticmethod
    def _runtime_source() -> str:
        import leapflow

        return str(getattr(leapflow, "__file__", ""))

    @staticmethod
    def _runtime_version() -> str:
        try:
            from leapflow.version import __version__
        except ImportError:
            return "unknown"
        return str(__version__)

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

    def _install_daemon_approval(self, ctx: Any) -> None:
        try:
            from leapflow.security.approval import SessionAwareGate
            from leapflow.security.actions import ActionDescriptor
            from leapflow.security.orchestrator import ApprovalOrchestrator
            from leapflow.tools.gateway_tool import set_gateway_approval_gate
            from leapflow.tools.registry_bootstrap import set_file_write_gate
            from leapflow.tools.shell_tools import set_approval_gate

            existing = getattr(ctx, "_approval_orchestrator", None)
            gate = SessionAwareGate(_DaemonApprovalGate(self))
            orchestrator = ApprovalOrchestrator(
                gate,
                grants=getattr(existing, "grants", None),
                audit=getattr(existing, "audit", None),
            )
            ctx._approval_gate = gate
            ctx._approval_orchestrator = orchestrator
            set_approval_gate(orchestrator)
            set_gateway_approval_gate(orchestrator)

            class _FileWriteGate:
                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(self, path: str, content: str, mode: str = "overwrite") -> bool:
                    result = await orchestrator.evaluate(ActionDescriptor.file_write(path, content, mode=mode))
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            set_file_write_gate(_FileWriteGate())
            logger.debug("daemon approval gate installed")
        except Exception:
            logger.debug("daemon approval gate installation skipped", exc_info=True)

    async def _stream_engine_events(
        self,
        stream: AsyncIterator[object],
        approval_queue: asyncio.Queue[StreamChunk],
    ) -> AsyncIterator[StreamChunk]:
        engine_task: asyncio.Task[Any] | None = asyncio.create_task(anext(stream))
        approval_task: asyncio.Task[StreamChunk] | None = asyncio.create_task(approval_queue.get())
        try:
            while engine_task is not None:
                wait_set = {task for task in (engine_task, approval_task) if task is not None}
                done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                if approval_task is not None and approval_task in done:
                    yield approval_task.result()
                    approval_task = asyncio.create_task(approval_queue.get())
                    continue
                if engine_task in done:
                    try:
                        event = engine_task.result()
                    except StopAsyncIteration:
                        engine_task = None
                        break
                    stream_event = self._normalize_event(event)
                    yield self._chunk_from_event(stream_event)
                    engine_task = asyncio.create_task(anext(stream))
        finally:
            for task in (engine_task, approval_task):
                if task is not None and not task.done():
                    task.cancel()
            self._deny_pending_for_queue(approval_queue, reason="stream_closed")
            if hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("daemon: failed to close engine stream", exc_info=True)

    def _pending_payloads(self) -> list[dict[str, Any]]:
        return [dict(item.get("request") or {}) for item in self._approval_pending.values()]

    async def _request_approval(self, request: Any) -> str:
        queue = self._approval_event_queue
        if queue is None:
            return "deny"
        pending_id = str(getattr(request, "request_id", "") or uuid.uuid4().hex)
        payload = request.to_dict()
        payload["pending_id"] = pending_id
        payload.setdefault("request_id", pending_id)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._approval_pending[pending_id] = {
            "request": payload,
            "future": future,
            "queue": queue,
            "created_at": time.time(),
        }
        await queue.put(StreamChunk(
            request_id="",
            content="Approval required",
            event_type="approval_request",
            metadata={"approval": payload},
        ))
        timeout_s = 120.0
        if getattr(request, "expires_at", None):
            timeout_s = max(1.0, float(request.expires_at) - time.time())
        try:
            result = await asyncio.wait_for(future, timeout=timeout_s)
            return str(result.get("decision") or "deny")
        except TimeoutError:
            return "deny"
        finally:
            self._approval_pending.pop(pending_id, None)

    @staticmethod
    def _normalize_approval_decision(decision: str) -> str:
        allowed = {
            "allow",
            "allow_once",
            "allow_session",
            "allow_always",
            "deny",
            "deny_always",
            "cancel_workflow",
        }
        value = str(decision or "deny").strip().lower()
        return value if value in allowed else "deny"

    def _deny_pending_for_queue(
        self,
        queue: asyncio.Queue[StreamChunk],
        *,
        reason: str,
    ) -> None:
        for pending_id, pending in list(self._approval_pending.items()):
            if pending.get("queue") is not queue:
                continue
            future = pending.get("future")
            if isinstance(future, asyncio.Future) and not future.done():
                future.set_result({"decision": "deny", "reason": reason})
            self._approval_pending.pop(pending_id, None)

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
        if engine is not None:
            metadata.setdefault("context_used", getattr(engine, "context_token_count", 0))
        return StreamChunk(
            request_id="",
            content=event.content,
            done=False,
            event_type=event.type,
            metadata=metadata,
        )


class _DaemonApprovalGate:
    """Approval gate that bridges daemon-side actions to thin clients."""

    def __init__(self, service: RuntimeLeapService) -> None:
        self._service = service

    async def request_approval(self, request: Any) -> Any:
        from leapflow.security.approval import ApprovalDecision

        decision = await self._service._request_approval(request)
        try:
            return ApprovalDecision(decision)
        except ValueError:
            return ApprovalDecision.DENY

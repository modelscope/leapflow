"""Runtime-backed LeapService implementation for leapd.

This module is the lightweight orchestrator: it assembles coordinators, manages
lifecycle, and delegates domain work.  Pure utility logic lives in
``_service_helpers``.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from leapflow.daemon._service_helpers import (
    ProducerServices as _ProducerServices,
    checkpoint_open_connection,
    engine_context_metadata,
    host_backend_status,
    install_learn_notifications,
    memory_entry_to_dict,
    normalize_stream_event,
    persisted_session_workspace,
    runtime_source,
    runtime_version,
)
from leapflow.daemon.approval_coordinator import ApprovalCoordinator
from leapflow.daemon.approval_route import approval_route as _approval_route
from leapflow.daemon.lease import ClientLeaseSnapshot
from leapflow.daemon.monitor_coordinator import MonitorCoordinator
from leapflow.daemon.protocol import StreamChunk
from leapflow.daemon.reentry_coordinator import ReentryCoordinator
from leapflow.daemon.session_coordinator import SessionCoordinator
from leapflow.daemon.turn_admission import TurnAdmission
from leapflow.engine import StreamEvent
from leapflow.memory.protocol import MemoryQuery
from leapflow.utils.build_info import StalenessMonitor, capture_build_info, is_stale

logger = logging.getLogger(__name__)

# Per-turn approval routing ContextVar — imported from shared module to avoid
# circular dependency with approval_coordinator.  See approval_route.py.


class RuntimeLeapService:
    """LeapService implementation backed by a single initialized Context."""

    # Max time a turn waits for deferred init before degrading to
    # critical-only mode (the background init keeps running).
    _DEFERRED_WAIT_TIMEOUT_S: float = 15.0

    # ── Construction ─────────────────────────────────────────────────

    def __init__(self, settings: Any, *, mock_host: bool = False, auto_start_deferred: bool = True) -> None:
        self._settings = settings
        self._mock_host = mock_host
        self._auto_start_deferred = auto_start_deferred
        self._ctx: Any | None = None
        self._monitor_coordinator = MonitorCoordinator()
        self._reentry_coordinator = ReentryCoordinator()
        self._deferred_init_task: "asyncio.Task[Any] | None" = None
        self._turn_admission = TurnAdmission(
            int(getattr(settings, "daemon_max_concurrent_turns", 3) or 3)
        )
        self._session_coordinator = SessionCoordinator()
        self._started_at = time.time()
        # Captured once at daemon startup so status() can tell a developer
        # this process is stale (source changed since it started) instead of
        # a fixed behavior change looking like a code defect. The staleness
        # check itself shells out to git, so it is wrapped in a non-blocking
        # cache: status() must return promptly even while other background
        # work (e.g. a deferred DB worker) is busy (see
        # test_daemon_event_loop_blocking.py).
        self._build_info = capture_build_info()
        self._build_staleness = StalenessMonitor(self._build_info)
        self._client_count: Callable[[], int] = lambda: 0
        self._client_leases: Callable[[], list[ClientLeaseSnapshot]] = lambda: []
        # Pending approvals have no TTL: they are released when their owning
        # turn/command ends, not after an elapsed deadline.
        self._approval_coordinator = ApprovalCoordinator()
        self._active_engine_request_id: str = ""
        self._active_engines: dict[str, Any] = {}
        self._observation: Any | None = None
        self._engine_request_ledger: dict[str, dict[str, Any]] = {}
        self._request_ledger_ttl_s = max(1.0, float(getattr(settings, "daemon_request_ledger_ttl_s", 600.0) or 600.0))
        self._request_ledger_max_entries = max(1, int(getattr(settings, "daemon_request_ledger_max_entries", 128) or 128))

        from leapflow.daemon.notifications import NotificationBus
        self.notification_bus = NotificationBus()

    def set_client_count_provider(self, provider: Callable[[], int]) -> None:
        self._client_count = provider

    def set_client_lease_provider(self, provider: Callable[[], list[ClientLeaseSnapshot]]) -> None:
        self._client_leases = provider

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize the daemon-owned runtime once."""
        if self._ctx is not None:
            return
        from leapflow.cli.context import Context

        ctx = Context(self._settings, self._mock_host)
        await ctx.initialize_critical()
        self._approval_coordinator.install_gate(ctx, self)
        install_learn_notifications(ctx, self.notification_bus)
        self._ctx = ctx
        if self._auto_start_deferred:
            self.start_deferred_init()
        # Monitor: start only when scheduler is enabled (coordinator checks internally)
        settings = getattr(ctx, "settings", self._settings)
        self._monitor_coordinator._build_services_proxy = lambda c, s: _ProducerServices(self)
        if getattr(settings, "scheduler_enabled", True):
            await self._monitor_coordinator.start(ctx, self.notification_bus, settings)

        # Reentry: start only when explicitly enabled
        if getattr(settings, "agent_reentry_enabled", False):
            await self._reentry_coordinator.start(
                ctx, self._settings, self._turn_admission, self.notification_bus,
                request_approval=self._request_approval,
            )

        # Platform observers: start FS/clipboard/focus signal collection
        if getattr(settings, "observation_enabled", True):
            event_bus = getattr(ctx, "event_bus", None)
            if event_bus is not None:
                try:
                    from leapflow.platform.observers.daemon import ObservationDaemon
                    self._observation = ObservationDaemon(bus=event_bus)
                    await self._observation.start()
                except Exception:
                    logger.debug("daemon: observation subsystem start failed", exc_info=True)
                    self._observation = None

    def start_deferred_init(self) -> None:
        """Start background non-critical initialization once."""
        if self._ctx is None:
            raise RuntimeError("leapd runtime is not initialized")
        if self._deferred_init_task is None or self._deferred_init_task.done():
            self._deferred_init_task = asyncio.create_task(self._run_deferred_init(self._ctx))

    async def shutdown(self) -> None:
        self._build_staleness.cancel_pending()
        if self._ctx is None:
            return
        ctx = self._ctx
        self._ctx = None
        # Stop background deferred init first: its yield points allow cleanup
        # to interleave with a half-initialized context otherwise.
        task = self._deferred_init_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._deferred_init_task = None
        # The service-level task above only *waits* (shielded) on the
        # context's runner task; cancel the runner itself so deferred init
        # actually stops before cleanup proceeds.
        runner = getattr(ctx, "_deferred_task", None)
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except (asyncio.CancelledError, Exception):
                pass
        await self._reentry_coordinator.stop()
        if self._observation is not None:
            try:
                await self._observation.stop()
            except Exception:
                logger.debug("daemon: observation stop failed", exc_info=True)
            self._observation = None
        await self._monitor_coordinator.stop()
        checkpoint_open_connection(ctx)
        await ctx.cleanup()

    async def _run_deferred_init(self, ctx: Any) -> None:
        """Background non-critical initialization."""
        try:
            # This task is created before the RPC server task in serve_daemon().
            # Yield once so the control-plane socket can start accepting status
            # and recovery RPCs before the heavier deferred phases begin.
            await asyncio.sleep(0)
            await ctx._ensure_deferred()
        except Exception:
            logger.warning("Deferred initialization failed; components will init on first use", exc_info=True)

    @property
    def context(self) -> Any:
        """Return the initialized Context or raise a clear lifecycle error."""
        if self._ctx is None:
            raise RuntimeError("leapd runtime is not initialized")
        return self._ctx

    # ── Backward-compat properties ───────────────────────────────────

    @property
    def _monitors(self) -> Any | None:
        """Backward-compat: used by test fixtures that inject MonitorManager directly.

        Prefer MonitorCoordinator API for new code.
        """
        return self._monitor_coordinator._monitors

    @_monitors.setter
    def _monitors(self, value: Any) -> None:
        self._monitor_coordinator._monitors = value

    @property
    def _session_registry(self) -> Any:
        """Backward-compat: used by test fixtures that inject SessionRegistry directly.

        Prefer SessionCoordinator API for new code.
        """
        return self._session_coordinator._session_registry

    @_session_registry.setter
    def _session_registry(self, value: Any) -> None:
        self._session_coordinator._session_registry = value

    # ── Core execution: engine_chat ──────────────────────────────────

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        request_id = str(kwargs.get("request_id") or uuid.uuid4().hex[:12])
        workspace_arg = str(kwargs.get("workspace_root") or "").strip()

        # Ensure deferred init completed. Emit a status chunk first so the
        # server dispatch loop receives an immediate first chunk (keepalive
        # heartbeats start right away) before the potentially long wait.
        ctx = self._ctx
        if ctx is not None and not getattr(ctx, '_deferred_initialized', True):
            yield StreamChunk(
                event_type="status",
                content="Warming up runtime components...",
                request_id=request_id,
            )
            try:
                # Bounded wait: _ensure_deferred() shields the background
                # runner task, so a timeout here cancels only this wait —
                # deferred init keeps running in the background.
                await asyncio.wait_for(
                    ctx._ensure_deferred(), timeout=self._DEFERRED_WAIT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.info(
                    "Deferred init still in progress after %.0fs; serving turn "
                    "in critical-only mode", self._DEFERRED_WAIT_TIMEOUT_S,
                )
                yield StreamChunk(
                    event_type="status",
                    content=(
                        "Runtime is still warming up; answering with core "
                        "capabilities only. Full capabilities return shortly."
                    ),
                    request_id=request_id,
                    metadata={"degraded": "warmup"},
                )
            except Exception:
                logger.warning(
                    "Deferred init failed; serving turn in critical-only mode",
                    exc_info=True,
                )
                yield StreamChunk(
                    event_type="status",
                    content=(
                        "Some runtime components failed to initialize; answering "
                        "with core capabilities only. See daemon logs for details."
                    ),
                    request_id=request_id,
                    metadata={"degraded": "deferred_init_failed"},
                )

        # Busy-feedback for clients when all slots occupied
        if self._turn_admission.locked():
            admission = self._turn_admission_status(queued_delta=1)
            active = int(admission.get("active", 0) or 0)
            cap = int(admission.get("max_concurrent", 1) or 1)
            waiting = int(admission.get("waiting", 0) or 0)
            yield StreamChunk(
                request_id=request_id,
                content=(
                    f"leapd turn capacity is full ({active}/{cap}); "
                    f"your request is waiting for a daemon slot (waiting:{waiting}). "
                    "To raise the cap, run `leap config set daemon.max_concurrent_turns N` "
                    "then `leap daemon restart`. Use `/cancel` to interrupt running work."
                ),
                event_type="status",
                metadata={
                    "request_id": request_id, "queued": True,
                    "active_request_id": self._active_engine_request_id or "",
                    "turn_admission": admission,
                },
            )

        async with self._turn_admission.turn_slot():
            self._prune_engine_request_ledger()
            existing = self._engine_request_ledger.get(request_id)
            if existing and existing.get("status") == "completed":
                for chunk in existing.get("chunks", []):
                    if isinstance(chunk, StreamChunk):
                        metadata = dict(chunk.metadata or {})
                        metadata["replayed_request"] = True
                        yield StreamChunk(
                            request_id=request_id, content=chunk.content,
                            done=chunk.done, event_type=chunk.event_type,
                            metadata=metadata,
                        )
                return
            if existing and existing.get("status") == "running":
                yield StreamChunk(
                    request_id=request_id,
                    content="Duplicate engine request is already running.",
                    event_type="status",
                    metadata={"request_id": request_id, "duplicate_request": True},
                )
                return

            request_record: dict[str, Any] = {"status": "running", "chunks": [], "created_at": time.time()}
            self._engine_request_ledger[request_id] = request_record
            ctx = self.context
            try:
                if ctx.reload_runtime_config_if_changed():
                    self._settings = ctx.settings
                    self._monitor_coordinator.update_settings(ctx.settings)
                    self._propagate_config_to_sessions(ctx)
                    chunk = StreamChunk(
                        request_id=request_id,
                        content="Configuration reloaded in leapd.",
                        event_type="status",
                        metadata={
                            **engine_context_metadata(self._active_engine(), ctx.settings),
                            "llm_model": getattr(ctx.settings, "llm_model", ""),
                            "request_id": request_id,
                        },
                    )
                    request_record["chunks"].append(chunk)
                    yield chunk

                engine = getattr(ctx, "engine", None)
                if engine is None:
                    raise RuntimeError("leapd engine is not initialized")

                # Route turn to session engine (isolated per-session engines prevent
                # cross-contamination; primary session reuses base engine).
                session_id = str(kwargs.get("session_id") or "")
                workspace_root = workspace_arg or str(self._workspace_root())
                if workspace_arg and not session_id:
                    session_id = request_id

                session_lock: asyncio.Lock | None = None
                if session_id:
                    from leapflow.daemon.session_registry import WorkspaceMismatchError
                    try:
                        if workspace_arg:
                            persisted_root = persisted_session_workspace(engine, session_id)
                            if persisted_root:
                                expected = Path(persisted_root).expanduser().resolve()
                                requested = Path(workspace_arg).expanduser().resolve()
                                if expected != requested:
                                    raise WorkspaceMismatchError(session_id, expected, requested)
                        exec_ctx = await self._ensure_session_registry(engine).acquire(
                            session_id, workspace_root=workspace_root,
                        )
                    except WorkspaceMismatchError as exc:
                        chunk = StreamChunk(
                            request_id=request_id, content=str(exc), event_type="error",
                            metadata={
                                "request_id": request_id, "workspace_mismatch": True,
                                "session_id": exc.session_id,
                                "expected_workspace_root": str(exc.expected),
                                "requested_workspace_root": str(exc.requested),
                            },
                        )
                        request_record["chunks"].append(chunk)
                        request_record["status"] = "failed"
                        request_record["completed_at"] = time.time()
                        yield chunk
                        return
                    engine = exec_ctx.engine
                    session_lock = exec_ctx.lock
                    if getattr(engine, "_current_session_id", None) != session_id:
                        engine._current_session_id = session_id

                # Serialize turns within one session
                if session_lock is not None:
                    await session_lock.acquire()

                enable_thinking = bool(kwargs.get("enable_thinking", False))
                approval_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
                previous_request_id = self._active_engine_request_id
                route_token = _approval_route.set((approval_queue, request_id))
                self._approval_coordinator.register_route(request_id)
                self._active_engine_request_id = request_id
                self._active_engines[request_id] = engine
                try:
                    sig = inspect.signature(engine.run_stream)
                    if "request_id" in sig.parameters:
                        stream = engine.run_stream(message, enable_thinking=enable_thinking, request_id=request_id)
                    else:
                        stream = engine.run_stream(message, enable_thinking=enable_thinking)
                    async for chunk in self._stream_engine_events(
                        stream, approval_queue, request_id=request_id, engine=engine,
                    ):
                        request_record["chunks"].append(chunk)
                        yield chunk
                    request_record["status"] = "completed"
                    request_record["completed_at"] = time.time()
                    self._prune_engine_request_ledger()
                finally:
                    _approval_route.reset(route_token)
                    self._approval_coordinator.unregister_route(request_id)
                    self._active_engine_request_id = previous_request_id
                    self._active_engines.pop(request_id, None)
                    self._approval_coordinator.deny_for_request(request_id, reason="turn_ended")
                    if session_lock is not None:
                        session_lock.release()
            except Exception:
                request_record["status"] = "failed"
                request_record["completed_at"] = time.time()
                self._prune_engine_request_ledger()
                raise

    async def engine_cancel(self, request_id: str = "") -> bool:
        targets: list[Any] = []
        if request_id:
            eng = self._active_engines.get(request_id)
            if eng is not None:
                targets = [eng]
        else:
            targets = list(self._active_engines.values())
        if not targets:
            ctx = self.context
            eng = getattr(ctx, "engine", None)
            if eng is not None:
                targets = [eng]
        cancelled = False
        for eng in targets:
            if eng is not None and hasattr(eng, "cancel"):
                result = eng.cancel()
                if hasattr(result, "__await__"):
                    await result
                cancelled = True
        return cancelled

    # ── Stream event fusion ──────────────────────────────────────────

    async def _stream_engine_events(
        self,
        stream: AsyncIterator[object],
        approval_queue: asyncio.Queue[StreamChunk],
        *,
        request_id: str = "",
        engine: Any = None,
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
                    stream_event = normalize_stream_event(event)
                    yield self._chunk_from_event(
                        stream_event, request_id=request_id, engine=engine,
                    )
                    engine_task = asyncio.create_task(anext(stream))
        finally:
            for task in (engine_task, approval_task):
                if task is not None and not task.done():
                    task.cancel()
            self._approval_coordinator.deny_for_queue(approval_queue, reason="stream_closed")
            if hasattr(stream, "aclose"):
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("daemon: failed to close engine stream", exc_info=True)

    def _active_engine(self, session_id: str = "") -> Any:
        """Return the engine holding a caller's live conversation state.

        ``ctx.engine`` is only a template used to build per-session engines and
        never accumulates a conversation, so reading it yields zero turns and zero
        context — which has surfaced repeatedly as an empty LeapBoard and a status
        bar stuck at ``0/<limit>``. Anything assembling runtime metadata must come
        through here rather than reaching for ``ctx.engine`` directly.

        Pass the caller's ``session_id``. Omitting it resolves "whichever session
        was most recently active", which on a daemon serving several workspaces is
        somebody else's — acceptable only for genuinely cross-session views.
        """
        ctx = self._ctx
        if ctx is None:
            return None
        engine, _ = self._session_coordinator.resolve_session_engine(ctx, session_id)
        return engine

    def _chunk_from_event(
        self, event: StreamEvent, *, request_id: str = "", engine: Any,
    ) -> StreamChunk:
        """Wrap an engine stream event as an RPC chunk with runtime metadata.

        ``engine`` is required and must be the engine that produced the event —
        the per-session one. There is deliberately no fallback: resolving "some
        active engine" would report the base engine (no conversation, so context
        reads 0) or, on a daemon serving several TUIs, another client's session —
        whose id the client would then adopt as its own.
        """
        ctx = self.context
        metadata = dict(event.metadata or {})
        session_id = getattr(engine, "_current_session_id", "") if engine is not None else ""
        if request_id:
            metadata.setdefault("request_id", request_id)
        if session_id:
            metadata.setdefault("session_id", str(session_id))
        if engine is not None:
            metadata.update(engine_context_metadata(engine, getattr(ctx, "settings", self._settings)))
        return StreamChunk(
            request_id=request_id, content=event.content,
            done=False, event_type=event.type, metadata=metadata,
        )

    # ── Delegate: session ────────────────────────────────────────────

    async def session_create(self, **kwargs: Any) -> dict[str, Any]:
        return await self._session_coordinator.create(self.context, **kwargs)

    async def session_resume(self, session_id: str) -> dict[str, Any]:
        return await self._session_coordinator.resume(self.context, session_id)

    async def session_history(self, limit: int = 200, session_id: str = "") -> dict[str, Any]:
        return await self._session_coordinator.get_history(
            self._ctx, self._settings, limit=limit, session_id=session_id,
        )

    async def session_detail(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
        include_inactive: bool = True,
        workspace_root: str = "",
    ) -> dict[str, Any]:
        return await self._session_coordinator.get_detail(
            self._ctx,
            self._settings,
            session_id,
            limit=limit,
            offset=offset,
            include_inactive=include_inactive,
            workspace_root=workspace_root,
        )

    async def session_analyze(self) -> dict[str, Any]:
        return await self._session_coordinator.analyze(self._monitors, self._ctx, self._settings)

    def _ensure_session_registry(self, base_engine: Any) -> Any:
        return self._session_coordinator.ensure_registry(base_engine, self._settings)

    def _propagate_config_to_sessions(self, ctx: Any) -> None:
        """Propagate updated LLM/config to all active session engines."""
        registry = self._session_coordinator.registry
        if registry is None:
            return
        settings = ctx.settings
        llm = ctx.llm
        vlm = getattr(ctx, "vlm", None)
        classifier = getattr(ctx, "intent_classifier", None)
        for sid in registry.session_ids():
            session_ctx = registry.get(sid)
            if session_ctx is None:
                continue
            engine = session_ctx.engine
            if engine is None or engine is getattr(ctx, "engine", None):
                # Base engine is already reconfigured by reload_runtime_config_if_changed.
                continue
            try:
                engine.reconfigure_runtime(
                    settings=settings,
                    llm=llm,
                    vlm=vlm,
                    classifier=classifier,
                )
            except Exception:
                logger.debug(
                    "Failed to propagate config to session %s", sid, exc_info=True,
                )

    # ── Delegate: watch (monitor subsystem) ──────────────────────────

    def has_active_watches(self) -> bool:
        return self._monitor_coordinator.has_active_watches()

    async def _watch_runtime_summary(self) -> dict[str, Any]:  # backward-compat
        return await self._monitor_coordinator.get_summary()

    async def watch_arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        return await self._monitor_coordinator.arm(spec)

    async def watch_list(self) -> list[dict[str, Any]]:
        return await self._monitor_coordinator.list_watches()

    async def watch_get(self, watch_id: str) -> dict[str, Any]:
        return await self._monitor_coordinator.get_watch(watch_id)

    async def watch_pause(self, watch_id: str) -> dict[str, Any]:
        return await self._monitor_coordinator.pause(watch_id)

    async def watch_resume(self, watch_id: str) -> dict[str, Any]:
        return await self._monitor_coordinator.resume(watch_id)

    async def watch_stop(self, watch_id: str) -> dict[str, Any]:
        return await self._monitor_coordinator.stop_watch(watch_id)

    async def watch_mute(self, watch_id: str, muted: bool = True) -> dict[str, Any]:
        return await self._monitor_coordinator.mute(watch_id, muted)

    async def watch_refresh(self, watch_id: str) -> dict[str, Any]:
        return await self._monitor_coordinator.refresh(watch_id)

    async def watch_findings(self, watch_id: str = "", limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return await self._monitor_coordinator.findings(watch_id, limit, offset)

    # ── Delegate: approval ───────────────────────────────────────────

    async def approval_status(self) -> dict[str, Any]:
        return self._approval_coordinator.get_status()

    async def approval_resolve(self, pending_id: str, decision: str, reason: str = "") -> dict[str, Any]:
        return await self._approval_coordinator.resolve(pending_id, decision, reason)

    async def approval_cancel(self, pending_id: str, reason: str = "cancelled") -> dict[str, Any]:
        return await self._approval_coordinator.cancel(pending_id, reason)

    async def _request_approval(self, request: Any) -> str:
        """Route approval through the ContextVar-based turn routing."""
        route = _approval_route.get()
        return await self._approval_coordinator.request_approval(request, route)

    def _deny_pending_for_request(self, request_id: str, reason: str = "turn_ended") -> None:
        """Backward-compat delegate (used by tests)."""
        self._approval_coordinator.deny_for_request(request_id, reason)

    def _release_orphaned_approvals(self) -> int:
        """Release pendings whose owning turn/command is gone (used by tests).

        Named for what it does: pending approvals have no TTL, so this is a
        liveness sweep, not a staleness sweep.
        """
        return self._approval_coordinator.prune_orphaned()

    # ── Delegate: host backend ───────────────────────────────────────

    async def host_status(self) -> dict[str, Any]:
        ctx = self.context
        status_fn = getattr(ctx, "host_backend_status", None)
        if callable(status_fn):
            return dict(await status_fn())
        return host_backend_status(ctx)

    async def host_start(self) -> dict[str, Any]:
        async with self._turn_admission.exclusive():
            ctx = self.context
            start = getattr(ctx, "host_backend_start", None)
            if not callable(start):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await start())

    async def host_stop(self) -> dict[str, Any]:
        async with self._turn_admission.exclusive():
            ctx = self.context
            stop = getattr(ctx, "host_backend_stop", None)
            if not callable(stop):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await stop())

    async def host_restart(self) -> dict[str, Any]:
        async with self._turn_admission.exclusive():
            ctx = self.context
            restart = getattr(ctx, "host_backend_restart", None)
            if not callable(restart):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await restart())

    # ── Delegate: signal metrics ──────────────────────────────────────

    async def monitor_signal_metrics(self) -> dict[str, Any]:
        """Collect real-time signal flow metrics from runtime components."""
        from leapflow.monitor.signal_metrics import SignalMetricsCollector

        ctx = self._ctx
        collector = SignalMetricsCollector()
        snapshot = collector.collect(
            event_bus=getattr(ctx, "event_bus", None) if ctx else None,
            monitor_manager=self._monitor_coordinator._monitors,
            signal_noise_gate=self._monitor_coordinator.signal_noise_stats,
        )
        stream = self._monitor_coordinator.get_signal_stream()
        return {"ok": True, "metrics": snapshot.to_dict(), "signal_stream": stream}

    # ── Delegate: memory / signal ────────────────────────────────────

    async def signal_record(self, signal_data: dict[str, Any]) -> dict[str, Any]:
        ctx = self.context
        event_type = str(signal_data.get("type") or "daemon.signal")
        payload = dict(signal_data.get("payload") or {})
        await ctx.event_bus.handle_event(event_type, payload)
        return {"ok": True}

    async def memory_search(self, query: str, *, limit: int = 10, workspace_root: str = "") -> list[dict[str, Any]]:
        ctx = self.context
        # Derive session_scope from active engine if available
        engine = getattr(ctx, "engine", None)
        session_scope = str(getattr(engine, "_current_session_id", "") or "")
        memory_query = MemoryQuery(
            keywords=query.split()[:8],
            limit=limit,
            workspace_root=workspace_root,
            session_scope=session_scope,
        )
        results = await ctx.memory.search(memory_query)
        return [memory_entry_to_dict(item) for item in results]

    async def memory_insert(self, content: str, kind: str = "fact", *, workspace_root: str = "", **kwargs: Any) -> str:
        ctx = self.context
        metadata = dict(kwargs.get("metadata") or {})
        if workspace_root and "workspace_root" not in metadata:
            try:
                metadata["workspace_root"] = str(Path(workspace_root).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                metadata["workspace_root"] = workspace_root
        entry_id = ctx.lt.ingest(kind, content, metadata=metadata)
        return str(entry_id)

    # ── Delegate: tools / commands ───────────────────────────────────

    async def tools_list(self) -> dict[str, Any]:
        from leapflow.cli.commands.slash_handlers import build_tool_payload
        return build_tool_payload(self.context)

    async def usage_summary(self) -> dict[str, Any]:
        from leapflow.cli.commands.slash_handlers import build_usage_payload
        return build_usage_payload(self.context)

    async def app_command(self, args: str = "") -> dict[str, Any]:
        from leapflow.cli.commands.slash_handlers import build_app_payload
        return await build_app_payload(self.context, args)

    async def command_execute(self, name: str, args: str = "", session_id: str = "") -> dict[str, Any]:
        from leapflow.cli.commands.slash_handlers import command_execute
        return await command_execute(self.context, name, args, session_id=session_id)

    # ── Delegate: gateway (stubs) ────────────────────────────────────

    async def gateway_connect(self, platform: str, credentials: dict[str, str], options: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError("gateway.connect is not available in this daemon phase")

    async def gateway_disconnect(self, platform: str) -> dict[str, Any]:
        raise NotImplementedError("gateway.disconnect is not available in this daemon phase")

    async def gateway_status(self) -> list[dict[str, Any]]:
        raise NotImplementedError("gateway.status is not available in this daemon phase")

    async def gateway_send(self, platform: str, chat_id: str, text: str, thread_id: str = "") -> dict[str, Any]:
        raise NotImplementedError("gateway.send is not available in this daemon phase")

    # ── Delegate: skill / scheduler (stubs) ──────────────────────────

    async def skill_execute(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("skill.execute is not available in this daemon phase")

    async def scheduler_arm(self, task_config: dict[str, Any]) -> str:
        raise NotImplementedError("scheduler.arm is not available in this daemon phase")

    # ── Notifications ────────────────────────────────────────────────

    async def subscribe_notifications(self) -> AsyncIterator[StreamChunk]:
        """Long-lived streaming RPC: yield notifications until client disconnects."""
        subscriber_id = str(uuid.uuid4())
        queue = self.notification_bus.subscribe(subscriber_id)
        try:
            while True:
                notification = await queue.get()
                if notification is None:
                    break
                yield StreamChunk(
                    request_id="", content="", event_type="status",
                    metadata=notification.to_dict(),
                )
        finally:
            self.notification_bus.unsubscribe(subscriber_id)

    # ── Status ───────────────────────────────────────────────────────

    async def status(self, session_id: str = "") -> dict[str, Any]:
        ctx = self._ctx
        settings = getattr(ctx, "settings", self._settings) if ctx is not None else self._settings
        # Report on the caller's session engine, not the base template (which has
        # no conversation, so context usage would read zero) and not on whichever
        # session happened to run last (which on a multi-workspace daemon belongs
        # to another client, and whose id that client would then adopt).
        engine = self._active_engine(session_id) if session_id else None
        db_holder = getattr(ctx, "_db_holder", None) if ctx is not None else None
        layout = settings.layout
        profile_layout = settings.profile_layout
        workspace_root = Path(str(getattr(settings, "workspace_root", os.getcwd())))
        context_metadata = engine_context_metadata(engine, settings)
        self._approval_coordinator.prune_orphaned()
        clients = await asyncio.to_thread(self._safe_client_lease_summaries)
        host = await asyncio.to_thread(host_backend_status, ctx)
        # Non-blocking: returns the last cached verdict (None on the very
        # first call) and refreshes it in the background, never awaiting the
        # git subprocess on this hot path. `is_stale` is looked up here (not
        # bound at __init__ time) so tests can still
        # `monkeypatch.setattr(service_module, "is_stale", ...)`.
        build_stale = self._build_staleness.current(is_stale)
        return {
            "pid": os.getpid(),
            "profile": getattr(settings, "profile", "default"),
            "profile_dir": str(settings.profile_dir),
            "profile_manifest_path": str(profile_layout.manifest_path),
            "profile_config_dir": str(profile_layout.config_dir),
            "user_config_path": str(layout.user_config_path),
            "mcp_servers_path": str(layout.mcp_servers_path),
            "workspace_config_path": str(layout.workspace_config_path(workspace_root)),
            "workspace_manifest_path": str(layout.workspace_manifest_path(workspace_root)),
            "config_sources": list(getattr(settings, "config_sources", ())),
            "config_warnings": list(getattr(settings, "config_warnings", ())),
            "watched_config_paths": [str(p) for p in getattr(settings, "watched_config_paths", ())],
            "runtime_dir": str(getattr(settings, "runtime_dir", "")),
            "tui_history_path": str(profile_layout.tui_history_path),
            "cache_index_path": str(profile_layout.cache.index_path),
            "secrets_scope": str(getattr(getattr(settings, "profile_manifest", None), "secrets_scope", "profile")),
            "db_path": str(getattr(db_holder, "db_path", settings.duckdb_path)),
            "volatile": bool(getattr(ctx, "storage_volatile", False)) if ctx is not None else False,
            "uptime_s": max(0.0, time.time() - self._started_at),
            "active_clients": max(0, self._client_count()),
            "active_connections": max(0, self._client_count()),
            "connected_clients": len(clients),
            "clients": clients,
            "model": getattr(settings, "llm_model", ""),
            "llm_context_length": context_metadata.get("llm_context_length", getattr(settings, "llm_context_length", 0)),
            "context_used": context_metadata.get("context_used", 0),
            "context_posture": context_metadata.get("context_posture", "baseline"),
            "context_signal": context_metadata.get("context_signal", ""),
            "context_guidance": context_metadata.get("context_guidance", ""),
            "compression_reason": context_metadata.get("compression_reason", ""),
            "compression_savings_ratio": context_metadata.get("compression_savings_ratio", 0.0),
            "context_budget_snapshot": context_metadata.get("context_budget_snapshot", {}),
            "session_id": str(getattr(engine, "_current_session_id", "") or "") if engine is not None else "",
            "runtime_source": runtime_source(),
            "runtime_executable": sys.executable,
            "runtime_version": runtime_version(),
            "pending_approvals": self._approval_coordinator.pending_count(),
            "turn_admission": self._turn_admission_status(),
            "deferred_init": self._deferred_init_status(ctx),
            "watch_summary": await self._monitor_coordinator.get_summary(),
            "host_backend": host,
            # Whether *this* daemon process still matches the source tree on
            # disk (None when outside a git checkout, e.g. a packaged install).
            "build": {**self._build_info.to_dict(), "stale": build_stale},
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _workspace_root(self) -> Path:
        ctx = self._ctx
        settings = getattr(ctx, "settings", self._settings) if ctx is not None else self._settings
        return Path(str(getattr(settings, "workspace_root", os.getcwd()))).expanduser().resolve()

    def _turn_admission_status(self, *, queued_delta: int = 0) -> dict[str, Any]:
        snapshot = dict(self._turn_admission.snapshot())
        if queued_delta:
            snapshot["waiting"] = max(0, int(snapshot.get("waiting", 0) or 0) + queued_delta)
        snapshot["active_request_ids"] = sorted(self._active_engines)
        return snapshot

    def _deferred_init_status(self, ctx: Any | None) -> dict[str, Any]:
        """Return a non-blocking diagnostic snapshot of deferred init state."""
        if ctx is None:
            return {"initialized": False, "running": False, "attempts": 0, "max_attempts": 0}
        runner = getattr(ctx, "_deferred_task", None)
        service_task = self._deferred_init_task
        attempts = int(getattr(ctx, "_deferred_attempts", 0) or 0)
        max_attempts = int(getattr(ctx, "_DEFERRED_MAX_ATTEMPTS", 0) or 0)
        running = bool(
            (runner is not None and not runner.done())
            or (service_task is not None and not service_task.done())
        )
        snapshot: dict[str, Any] = {
            "initialized": bool(getattr(ctx, "_deferred_initialized", False)),
            "running": running,
            "attempts": attempts,
            "max_attempts": max_attempts,
            "done": bool(runner.done()) if runner is not None else False,
            "cancelled": bool(runner.cancelled()) if runner is not None else False,
        }
        if runner is not None and runner.done() and not runner.cancelled():
            exc = runner.exception()
            if exc is not None:
                snapshot["error"] = str(exc)
        if max_attempts and attempts >= max_attempts and not snapshot["initialized"]:
            snapshot["degraded"] = True
        return snapshot

    def _safe_client_lease_summaries(self) -> list[dict[str, Any]]:
        try:
            return self._client_lease_summaries()
        except Exception:
            logger.debug("daemon: client lease status unavailable", exc_info=True)
            return []

    def _client_lease_summaries(self) -> list[dict[str, Any]]:
        """Per-client lease view for status observability."""
        return [
            {
                "client_id": snap.client_id, "pid": snap.pid,
                "kind": snap.kind, "state": snap.state,
                "session_id": snap.session_id, "workspace_root": snap.cwd,
            }
            for snap in self._client_leases()
        ]

    def _prune_engine_request_ledger(self) -> None:
        """Bound completed/failed engine request replay records by TTL and size."""
        now = time.time()
        for rid, record in list(self._engine_request_ledger.items()):
            if str(record.get("status") or "") == "running":
                continue
            completed_at = float(record.get("completed_at") or record.get("created_at") or 0.0)
            if now - completed_at > self._request_ledger_ttl_s:
                self._engine_request_ledger.pop(rid, None)
        overflow = len(self._engine_request_ledger) - self._request_ledger_max_entries
        if overflow <= 0:
            return
        evictable = sorted(
            (
                (float(r.get("completed_at") or r.get("created_at") or 0.0), rid)
                for rid, r in self._engine_request_ledger.items()
                if str(r.get("status") or "") != "running"
            ),
            key=lambda item: item[0],
        )
        for _ts, rid in evictable[:overflow]:
            self._engine_request_ledger.pop(rid, None)

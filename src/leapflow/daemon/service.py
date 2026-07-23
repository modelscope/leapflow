"""Runtime-backed LeapService implementation for leapd."""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from leapflow.daemon.lease import ClientLeaseSnapshot
from leapflow.daemon.protocol import StreamChunk
from leapflow.engine import StreamEvent
from leapflow.memory.protocol import MemoryEntry, MemoryQuery

logger = logging.getLogger(__name__)

_MAX_SESSION_ARTIFACTS = 5
_MAX_SESSION_ARTIFACT_CHARS = 6000
_MAX_SESSION_ARTIFACT_TOTAL_CHARS = 16000
_PATH_RE = re.compile(r'(?P<key>path|file_path)["\'=:\s]+(?P<value>[^"\'\n|]+)')

# Per-turn approval routing. Each running turn sets this to (its approval queue,
# its request_id) inside its own asyncio task; the globally-shared approval gate
# reads it deep inside tool execution to route an approval prompt to the correct
# turn. Because concurrent turns run in separate tasks, the ContextVar isolates
# them without a single shared slot (which would misroute under N>1).
_approval_route: "contextvars.ContextVar[tuple[asyncio.Queue[StreamChunk], str] | None]" = (
    contextvars.ContextVar("leapd_approval_route", default=None)
)


class RuntimeLeapService:
    """LeapService implementation backed by a single initialized Context."""

    def __init__(self, settings: Any, *, mock_host: bool = False) -> None:
        self._settings = settings
        self._mock_host = mock_host
        self._ctx: Any | None = None
        self._monitors: Any | None = None
        self._reentry_task: "asyncio.Task[Any] | None" = None
        self._reentry_stop: "asyncio.Event | None" = None
        self._reentry_service: Any | None = None
        self._engine_lock = asyncio.Lock()
        self._session_registry: Any | None = None  # lazy per-session engine registry (Stage 3)
        self._started_at = time.time()
        self._client_count: Callable[[], int] = lambda: 0
        self._client_leases: Callable[[], list[ClientLeaseSnapshot]] = lambda: []
        self._approval_pending: dict[str, dict[str, Any]] = {}
        self._active_engine_request_id: str = ""
        # request_id -> the engine actually running that turn (may be a
        # per-session engine, not the base), so cancel targets the right one.
        self._active_engines: dict[str, Any] = {}
        self._engine_request_ledger: dict[str, dict[str, Any]] = {}
        self._request_ledger_ttl_s = max(1.0, float(getattr(settings, "daemon_request_ledger_ttl_s", 600.0) or 600.0))
        self._request_ledger_max_entries = max(1, int(getattr(settings, "daemon_request_ledger_max_entries", 128) or 128))

        from leapflow.daemon.notifications import NotificationBus
        self.notification_bus = NotificationBus()

    def set_client_count_provider(self, provider: Callable[[], int]) -> None:
        """Set a lightweight callback used by status reporting."""
        self._client_count = provider

    def set_client_lease_provider(self, provider: Callable[[], list[ClientLeaseSnapshot]]) -> None:
        """Set a callback used to report live client leases."""
        self._client_leases = provider

    async def start(self) -> None:
        """Initialize the daemon-owned runtime once."""
        if self._ctx is not None:
            return
        from leapflow.cli.context import Context

        ctx = Context(self._settings, self._mock_host)
        await ctx.initialize()
        self._install_daemon_approval(ctx)
        self._install_learn_notifications(ctx)
        self._ctx = ctx
        await self._start_monitors(ctx)
        await self._start_reentry_driver(ctx)

    async def _start_monitors(self, ctx: Any) -> None:
        """Build and start the daemon-hosted monitor runtime (watches)."""
        settings = getattr(ctx, "settings", self._settings)
        if not getattr(settings, "scheduler_enabled", True):
            return
        try:
            from leapflow.monitor import MonitorManager, SessionAnalysisProducer

            bus = self.notification_bus
            self._monitors = MonitorManager(
                holder=ctx._db_holder,
                emit=lambda event_type, payload: bus.emit_event(event_type, **payload),
                services=_ProducerServices(self),
                tick_seconds=int(getattr(settings, "scheduler_tick_seconds", 60)),
                grace_seconds=float(getattr(settings, "scheduler_grace_seconds", 120.0)),
            )
            self._monitors.producers.register(SessionAnalysisProducer())
            setattr(ctx, "monitors", self._monitors)
            await self._monitors.start()
            # A fresh daemon lifetime owns no interactive clients yet, so any
            # persisted client-coupled watch (e.g. a session-analysis watch left
            # over from a prior run or an unclean client exit) is stale. Drop it
            # so the status bar and keep-alive only reflect real active monitors.
            try:
                swept = self._monitors.sweep_client_coupled_watches()
                if swept:
                    logger.info("daemon: swept %d stale client-coupled watch(es) on startup", swept)
            except Exception:
                logger.debug("daemon: client-coupled watch sweep failed", exc_info=True)
            logger.debug("daemon: monitor runtime started")
        except Exception:
            logger.debug("daemon: monitor runtime start skipped", exc_info=True)
            self._monitors = None
            setattr(ctx, "monitors", None)

    def has_active_watches(self) -> bool:
        """Return True when any hosted watch is armed/watching (idle keep-alive)."""
        monitors = self._monitors
        if monitors is None:
            return False
        try:
            return bool(monitors.has_active_watches())
        except Exception:
            return False

    def _watch_runtime_summary(self) -> dict[str, Any]:
        monitors = self._monitors
        if monitors is None:
            return {
                "total": 0,
                "active": 0,
                "standalone_active": 0,
                "client_coupled_active": 0,
                "active_samples": [],
            }
        try:
            watches = [view.to_dict() for view in monitors.list_watches()]
        except Exception:
            logger.debug("daemon: watch summary unavailable", exc_info=True)
            watches = []
        active_states = {"armed", "watching", "due", "confirming", "executing"}
        active = [watch for watch in watches if str(watch.get("state", "")) in active_states]
        standalone = [watch for watch in active if not bool(watch.get("client_coupled", False))]
        coupled = [watch for watch in active if bool(watch.get("client_coupled", False))]
        return {
            "total": len(watches),
            "active": len(active),
            "standalone_active": len(standalone),
            "client_coupled_active": len(coupled),
            "active_samples": [
                {
                    "watch_id": str(watch.get("watch_id", "")),
                    "name": str(watch.get("name", "")),
                    "domain": str(watch.get("domain", "")),
                    "state": str(watch.get("state", "")),
                    "client_coupled": bool(watch.get("client_coupled", False)),
                }
                for watch in active[:5]
            ],
        }

    async def _start_reentry_driver(self, ctx: Any) -> None:
        """Start the background re-entry service (S2 N3b + N4 + N5).

        Dispatches due TIME triggers periodically and matches inbound gateway
        EVENT triggers, always as *isolated subagents* (fresh context -> no
        interactive-engine / working-memory / session pollution), serialized via
        ``_engine_lock``. Gated by ``agent_reentry_enabled`` (default off) plus a
        global-budget backstop. Best-effort: never blocks startup.
        """
        try:
            store = getattr(ctx, "_reentry_store", None)
            manager = getattr(ctx, "_subagent_manager", None)
            if store is None or manager is None:
                return
            from leapflow.scheduler.reentry_service import ReentryService

            # SO3: governed proactive delivery (wired only when enabled; default off).
            send_governor = None
            send_fn = None
            request_approval = None
            if getattr(self._settings, "agent_reentry_send_enabled", False):
                from leapflow.scheduler.reentry_send import SendGovernor, SendRateLimiter
                from leapflow.security.send_trust import SendTrustLedger
                send_governor = SendGovernor(
                    trust=SendTrustLedger(
                        verified_at=int(getattr(self._settings, "agent_reentry_send_verified_at", 3)),
                    ),
                    rate=SendRateLimiter(
                        per_hour=int(getattr(self._settings, "agent_reentry_send_rate_per_hour", 4)),
                    ),
                    enabled=True,
                    global_budget=int(getattr(self._settings, "agent_reentry_send_global_budget", 50)),
                )
                gw = getattr(ctx, "gateway_server", None)
                send_fn = getattr(gw, "send_message", None) if gw is not None else None
                request_approval = self._request_approval

            service = ReentryService(
                store=store,
                manager=manager,
                settings=self._settings,
                engine_lock=self._engine_lock,
                notify=lambda event_type, **kw: self.notification_bus.emit_event(event_type, **kw),
                global_budget=int(getattr(self._settings, "agent_reentry_global_budget", 100) or 0),
                send_governor=send_governor,
                send_fn=send_fn,
                request_approval=request_approval,
            )
            self._reentry_service = service
            # N4: observe inbound gateway messages for EVENT-trigger matches.
            try:
                setattr(ctx, "_reentry_event_observer", service.on_gateway_message)
            except Exception:
                logger.debug("daemon: reentry event observer wiring failed", exc_info=True)

            interval = max(5.0, float(getattr(self._settings, "agent_reentry_tick_seconds", 30.0) or 30.0))
            self._reentry_stop = asyncio.Event()

            async def _loop() -> None:
                stop = self._reentry_stop
                assert stop is not None
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=interval)
                        break  # stop signalled
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
                    if stop.is_set():
                        break
                    try:
                        await service.tick()
                    except Exception:
                        logger.debug("reentry service tick failed", exc_info=True)

            self._reentry_task = asyncio.create_task(_loop(), name="leapd-reentry-driver")
            logger.debug("daemon: re-entry service started (interval=%.0fs)", interval)
        except Exception:
            logger.debug("daemon: re-entry service start skipped", exc_info=True)

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

    def _ensure_session_registry(self, base_engine: Any) -> Any:
        """Lazily build the per-session engine registry around the base engine.

        The first session reuses ``base_engine`` (single-session daemon
        unchanged); additional sessions get isolated engines via the P3-1
        factory with a fresh working memory of the same capacity.
        """
        if self._session_registry is None:
            from leapflow.daemon.session_registry import SessionRegistry
            from leapflow.engine.session_factory import build_session_engine
            from leapflow.memory import WorkingMemoryProvider
            base_wm = getattr(base_engine, "_wm", None)
            max_tokens = int(getattr(base_wm, "_max_tokens", 8192) or 8192)
            s = self._settings
            self._session_registry = SessionRegistry(
                base_engine=base_engine,
                build_engine=lambda base, sid, wm: build_session_engine(
                    base, session_id=sid, working_memory=wm
                ),
                build_working_memory=lambda: WorkingMemoryProvider(max_tokens=max_tokens),
                max_sessions=int(getattr(s, "daemon_max_live_sessions", 16) or 16),
                idle_ttl_s=float(getattr(s, "daemon_session_idle_ttl_s", 1800.0) or 1800.0),
            )
        return self._session_registry

    async def engine_chat(self, message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        request_id = str(kwargs.get("request_id") or uuid.uuid4().hex[:12])
        # The single engine serializes turns via _engine_lock. If it is already
        # busy with another turn (a long task can hold it for minutes), tell the
        # waiting client immediately instead of leaving it on a silent "thinking"
        # spinner while it blocks on the lock. The request is then queued and
        # starts when the active turn finishes; /cancel interrupts the running one.
        if self._engine_lock.locked():
            yield StreamChunk(
                request_id=request_id,
                content=(
                    "leapd is busy running another task — your request is queued and will start "
                    "when it finishes. Use /cancel to interrupt the running task, or wait."
                ),
                event_type="status",
                metadata={
                    "request_id": request_id,
                    "queued": True,
                    "active_request_id": self._active_engine_request_id or "",
                },
            )
        async with self._engine_lock:
            self._prune_engine_request_ledger()
            existing = self._engine_request_ledger.get(request_id)
            if existing and existing.get("status") == "completed":
                for chunk in existing.get("chunks", []):
                    if isinstance(chunk, StreamChunk):
                        metadata = dict(chunk.metadata or {})
                        metadata["replayed_request"] = True
                        yield StreamChunk(
                            request_id=request_id,
                            content=chunk.content,
                            done=chunk.done,
                            event_type=chunk.event_type,
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

            request_record: dict[str, Any] = {
                "status": "running",
                "chunks": [],
                "created_at": time.time(),
            }
            self._engine_request_ledger[request_id] = request_record
            ctx = self.context
            try:
                if ctx.reload_runtime_config_if_changed():
                    self._settings = ctx.settings
                    chunk = StreamChunk(
                        request_id=request_id,
                        content="Configuration reloaded in leapd.",
                        event_type="status",
                        metadata={
                            **self._engine_context_metadata(getattr(ctx, "engine", None), ctx.settings),
                            "llm_model": getattr(ctx.settings, "llm_model", ""),
                            "request_id": request_id,
                        },
                    )
                    request_record["chunks"].append(chunk)
                    yield chunk
                engine = getattr(ctx, "engine", None)
                if engine is None:
                    raise RuntimeError("leapd engine is not initialized")
                # Route the turn to its session's engine. The primary session
                # reuses the base engine (single-session daemon unchanged);
                # additional distinct sessions run on isolated per-session
                # engines so their working memory / per-turn state never
                # cross-contaminate. Turns remain globally serialized here
                # (_engine_lock); bounded cross-session concurrency is P3-4.
                # Un-sessioned turns keep using the base engine directly, which
                # is also the primary session's engine, so the two stay
                # consistent through a ""→real session_id transition.
                session_id = str(
                    kwargs.get("session_id")
                    or getattr(engine, "_current_session_id", "")
                    or ""
                )
                if session_id:
                    exec_ctx = await self._ensure_session_registry(engine).acquire(session_id)
                    engine = exec_ctx.engine

                enable_thinking = bool(kwargs.get("enable_thinking", False))
                approval_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
                previous_request_id = self._active_engine_request_id
                route_token = _approval_route.set((approval_queue, request_id))
                self._active_engine_request_id = request_id
                self._active_engines[request_id] = engine
                try:
                    sig = inspect.signature(engine.run_stream)
                    if "request_id" in sig.parameters:
                        stream = engine.run_stream(
                            message,
                            enable_thinking=enable_thinking,
                            request_id=request_id,
                        )
                    else:
                        stream = engine.run_stream(
                            message,
                            enable_thinking=enable_thinking,
                        )
                    async for chunk in self._stream_engine_events(
                        stream, approval_queue, request_id=request_id,
                    ):
                        request_record["chunks"].append(chunk)
                        yield chunk
                    request_record["status"] = "completed"
                    request_record["completed_at"] = time.time()
                    self._prune_engine_request_ledger()
                finally:
                    _approval_route.reset(route_token)
                    self._active_engine_request_id = previous_request_id
                    self._active_engines.pop(request_id, None)
            except Exception:
                request_record["status"] = "failed"
                request_record["completed_at"] = time.time()
                self._prune_engine_request_ledger()
                raise

    async def engine_cancel(self, request_id: str = "") -> bool:
        # Cancel the running turn's own engine (which may be a per-session
        # engine, not the base). With a request_id, target that specific turn;
        # without one, cancel all active turns (at N=1 that is the single one).
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

    async def skill_execute(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("skill.execute is not available in this daemon phase")

    async def scheduler_arm(self, task_config: dict[str, Any]) -> str:
        raise NotImplementedError("scheduler.arm is not available in this daemon phase")

    # ── Watch runtime (monitor subsystem) ───────────────────────────────

    def _require_monitors(self) -> Any:
        if self._monitors is None:
            raise RuntimeError("monitor runtime is not available (scheduler disabled)")
        return self._monitors

    async def watch_arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        from leapflow.monitor import WatchSpec

        view = await self._require_monitors().arm_watch(WatchSpec.from_dict(spec or {}))
        return view.to_dict()

    async def watch_list(self) -> list[dict[str, Any]]:
        if self._monitors is None:
            return []
        return [view.to_dict() for view in self._monitors.list_watches()]

    async def watch_get(self, watch_id: str) -> dict[str, Any]:
        view = self._require_monitors().get_watch(watch_id)
        return view.to_dict() if view else {}

    async def watch_pause(self, watch_id: str) -> dict[str, Any]:
        view = self._require_monitors().pause_watch(watch_id)
        return view.to_dict() if view else {}

    async def watch_resume(self, watch_id: str) -> dict[str, Any]:
        view = self._require_monitors().resume_watch(watch_id)
        return view.to_dict() if view else {}

    async def watch_stop(self, watch_id: str) -> dict[str, Any]:
        view = self._require_monitors().stop_watch(watch_id)
        return view.to_dict() if view else {}

    async def watch_mute(self, watch_id: str, muted: bool = True) -> dict[str, Any]:
        view = self._require_monitors().set_muted(watch_id, bool(muted))
        return view.to_dict() if view else {}

    async def watch_refresh(self, watch_id: str) -> dict[str, Any]:
        return await self._require_monitors().run_watch_once(watch_id)

    async def watch_findings(
        self, watch_id: str = "", limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if self._monitors is None:
            return []
        findings = self._monitors.list_findings(
            watch_id=watch_id or None, limit=int(limit), offset=int(offset)
        )
        return [finding.to_dict() for finding in findings]

    # ── Session analysis (domain=session watch) ───────────────────────

    async def session_history(self, limit: int = 200) -> dict[str, Any]:
        ctx = self._ctx
        if ctx is None:
            return {"session_id": "", "turn_count": 0, "token_count": 0, "messages": [], "artifacts": []}
        engine = getattr(ctx, "engine", None)
        session_id = getattr(engine, "_current_session_id", "") if engine else ""
        messages: list[dict[str, Any]] = []
        if engine is not None:
            wm = getattr(engine, "_wm", None)
            if wm is not None and hasattr(wm, "as_chat_messages"):
                try:
                    messages = [dict(m) for m in wm.as_chat_messages() if isinstance(m, dict)]
                except Exception:
                    messages = []
        store_messages = self._session_store_messages(session_id, limit=int(limit))
        if store_messages:
            if not messages:
                messages = store_messages
            else:
                messages.extend(m for m in store_messages if m.get("role") == "tool")
        normalized = [
            {
                "role": str(m.get("role", "")),
                "content": str(m.get("content", "")),
                "tool_name": str(m.get("tool_name", "") or ""),
                "created_at": float(m.get("created_at", 0.0) or 0.0),
            }
            for m in messages
        ][-int(limit):]
        artifacts = self._collect_session_artifacts(session_id, store_messages or normalized)
        return {
            "session_id": session_id,
            "turn_count": int(getattr(engine, "turn_count", 0)) if engine else 0,
            "token_count": int(getattr(engine, "context_token_count", 0)) if engine else 0,
            "messages": normalized,
            "artifacts": artifacts,
        }

    def _session_store_messages(self, session_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        ctx = self._ctx
        store = getattr(ctx, "_conversation_store", None) if ctx is not None else None
        if store is None or not session_id:
            return []
        try:
            rows = store.get_messages(session_id, limit=int(limit))
        except Exception:
            logger.debug("daemon: session store messages unavailable", exc_info=True)
            return []
        return [self._conversation_message_to_dict(row) for row in rows]

    @staticmethod
    def _conversation_message_to_dict(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            return dict(message)
        return {
            "role": str(getattr(message, "role", "")),
            "content": str(getattr(message, "content", "")),
            "tool_name": str(getattr(message, "tool_name", "") or ""),
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "created_at": float(getattr(message, "created_at", 0.0) or 0.0),
            "metadata": dict(getattr(message, "metadata", {}) or {}),
        }

    def _collect_session_artifacts(self, session_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not session_id:
            return []
        workspace = self._workspace_root()
        candidates: list[tuple[str, dict[str, Any]]] = []
        for message in messages:
            if str(message.get("role", "")) != "tool":
                continue
            tool_name = str(message.get("tool_name", "") or "")
            if tool_name and tool_name not in {"file_write", "write_file"}:
                continue
            for path in self._extract_artifact_paths(message):
                candidates.append((path, message))
        seen: set[str] = set()
        artifacts: list[dict[str, Any]] = []
        total_chars = 0
        for raw_path, message in reversed(candidates):
            if len(artifacts) >= _MAX_SESSION_ARTIFACTS:
                break
            artifact = self._read_session_artifact(raw_path, workspace, message)
            key = str(artifact.get("path") or raw_path)
            if key in seen:
                continue
            seen.add(key)
            if artifact.get("status") == "included":
                content = str(artifact.get("content_excerpt", ""))
                remaining = max(0, _MAX_SESSION_ARTIFACT_TOTAL_CHARS - total_chars)
                if len(content) > remaining:
                    artifact["content_excerpt"] = content[:remaining]
                    artifact["truncated"] = True
                    artifact["reason"] = "artifact context budget reached"
                total_chars += len(str(artifact.get("content_excerpt", "")))
            artifacts.append(artifact)
        artifacts.reverse()
        return artifacts

    @staticmethod
    def _extract_artifact_paths(message: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        payloads = [message.get("content", ""), message.get("metadata", {})]
        for payload in payloads:
            if isinstance(payload, dict):
                for key in ("path", "file_path"):
                    if payload.get(key):
                        paths.append(str(payload[key]))
                continue
            text = str(payload or "")
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ("path", "file_path"):
                        if data.get(key):
                            paths.append(str(data[key]))
            except Exception:
                pass
            for match in _PATH_RE.finditer(text):
                value = match.group("value").strip().strip(",}")
                if value:
                    paths.append(value)
        return paths

    def _workspace_root(self) -> Path:
        ctx = self._ctx
        settings = getattr(ctx, "settings", self._settings) if ctx is not None else self._settings
        return Path(str(getattr(settings, "workspace_root", os.getcwd()))).expanduser().resolve()

    def _read_session_artifact(self, raw_path: str, workspace: Path, message: dict[str, Any]) -> dict[str, Any]:
        target = Path(raw_path).expanduser()
        if not target.is_absolute():
            target = workspace / target
        try:
            target = target.resolve()
        except OSError:
            target = target.absolute()
        base = {
            "path": str(target),
            "name": target.name,
            "source": "file_write",
            "tool_call_id": str(message.get("tool_call_id", "") or ""),
            "status": "skipped",
        }
        try:
            target.relative_to(workspace)
        except ValueError:
            return {**base, "reason": "outside workspace boundary"}
        try:
            from leapflow.security.path_sensitivity import classify_path_sensitivity
            sensitivity = classify_path_sensitivity(target)
        except Exception:
            sensitivity = None
        if sensitivity is not None:
            base.update({"sensitivity": sensitivity.category, "sensitivity_level": sensitivity.level})
            if not sensitivity.readable or sensitivity.requires_approval or sensitivity.redact_on_read:
                return {**base, "reason": f"sensitive path ({sensitivity.category}) not read in background"}
        if not target.exists() or not target.is_file():
            return {**base, "reason": "file no longer exists"}
        try:
            stat = target.stat()
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {**base, "reason": f"read failed: {exc}"}
        truncated = len(content) > _MAX_SESSION_ARTIFACT_CHARS
        excerpt = content[:_MAX_SESSION_ARTIFACT_CHARS]
        try:
            from leapflow.security.redact import redact_sensitive_text
            excerpt = redact_sensitive_text(excerpt, file_read=bool(getattr(sensitivity, "redact_on_read", False)))
        except Exception:
            pass
        return {
            **base,
            "status": "included",
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
            "content_excerpt": excerpt,
            "truncated": truncated,
        }

    async def session_analyze(self) -> dict[str, Any]:
        if self._monitors is None:
            return {"ok": False, "error": "monitor runtime unavailable"}
        watch_id = await self._ensure_session_watch()
        result = await self._monitors.run_watch_once(watch_id, force=True)
        return {"ok": bool(result.get("ok", True)), "watch_id": watch_id, "result": result}

    async def _ensure_session_watch(self) -> str:
        from leapflow.monitor.session_producer import ensure_session_watch, session_watch_params

        monitors = self._require_monitors()
        settings = getattr(self._ctx, "settings", self._settings)
        return await ensure_session_watch(monitors, params=session_watch_params(settings))

    async def _analyze_session_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "story": "", "insights": [], "decisions": [], "action_items": [],
            "open_questions": [], "entities": [], "next_prompts": [],
            "process_notes": [], "series_intents": [], "usage": {},
        }
        ctx = self._ctx
        llm = getattr(ctx, "llm", None) if ctx is not None else None
        if llm is None or not messages:
            return base
        transcript = "\n".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:500]}" for m in messages[-40:]
        )[:12000]
        artifact_block = self._format_artifact_context(artifacts or [])
        user_content = transcript if not artifact_block else f"{transcript}\n\n## Session file artifacts\n{artifact_block}"
        prompt = [
            {"role": "system", "content": _SESSION_ANALYSIS_SYSTEM},
            {"role": "user", "content": user_content[:18000]},
        ]
        try:
            response = await llm.achat(prompt, stream=False)
            data = _parse_session_json(getattr(response, "content", ""))
        except Exception:
            logger.debug("daemon: session analysis LLM call failed", exc_info=True)
            return base
        if isinstance(data, dict):
            for key in base:
                if key != "usage" and key in data:
                    base[key] = data[key]
        return base

    @staticmethod
    def _format_artifact_context(artifacts: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for artifact in artifacts:
            status = str(artifact.get("status", ""))
            path = str(artifact.get("path", ""))
            if status != "included":
                lines.append(f"- SKIPPED {path}: {artifact.get('reason', 'not included')}")
                continue
            excerpt = str(artifact.get("content_excerpt", ""))[:_MAX_SESSION_ARTIFACT_CHARS]
            truncated = " (truncated)" if artifact.get("truncated") else ""
            lines.append(f"- FILE {path}{truncated}\n```text\n{excerpt}\n```")
        return "\n".join(lines)

    async def _session_should_refresh(self, messages: list[dict[str, Any]]) -> bool:
        ctx = self._ctx
        llm = getattr(ctx, "llm", None) if ctx is not None else None
        if llm is None or not messages:
            return False
        tail = "\n".join(
            f"{m.get('role', '')}: {str(m.get('content', ''))[:200]}" for m in messages[-6:]
        )[:2000]
        prompt = [
            {"role": "system", "content": _SESSION_SALIENCE_SYSTEM},
            {"role": "user", "content": tail},
        ]
        try:
            response = await llm.achat(prompt, stream=False)
            return str(getattr(response, "content", "")).strip().upper().startswith("Y")
        except Exception:
            return False

    async def status(self) -> dict[str, Any]:
        ctx = self._ctx
        settings = getattr(ctx, "settings", self._settings) if ctx is not None else self._settings
        engine = getattr(ctx, "engine", None) if ctx is not None else None
        db_holder = getattr(ctx, "_db_holder", None) if ctx is not None else None
        layout = settings.layout
        profile_layout = settings.profile_layout
        workspace_root = Path(str(getattr(settings, "workspace_root", os.getcwd())))
        workspace_config_path = layout.workspace_config_path(workspace_root)
        workspace_manifest_path = layout.workspace_manifest_path(workspace_root)
        context_metadata = self._engine_context_metadata(engine, settings)
        watch_summary = self._watch_runtime_summary()
        return {
            "pid": os.getpid(),
            "profile": getattr(settings, "profile", "default"),
            "profile_dir": str(settings.profile_dir),
            "profile_manifest_path": str(profile_layout.manifest_path),
            "profile_config_dir": str(profile_layout.config_dir),
            "user_config_path": str(layout.user_config_path),
            "mcp_servers_path": str(layout.mcp_servers_path),
            "workspace_config_path": str(workspace_config_path),
            "workspace_manifest_path": str(workspace_manifest_path),
            "config_sources": list(getattr(settings, "config_sources", ())),
            "config_warnings": list(getattr(settings, "config_warnings", ())),
            "watched_config_paths": [str(path) for path in getattr(settings, "watched_config_paths", ())],
            "runtime_dir": str(getattr(settings, "runtime_dir", "")),
            "tui_history_path": str(profile_layout.tui_history_path),
            "cache_index_path": str(profile_layout.cache.index_path),
            "secrets_scope": str(getattr(getattr(settings, "profile_manifest", None), "secrets_scope", "profile")),
            "db_path": str(getattr(db_holder, "db_path", settings.duckdb_path)),
            "volatile": bool(getattr(ctx, "storage_volatile", False)) if ctx is not None else False,
            "uptime_s": max(0.0, time.time() - self._started_at),
            "active_clients": max(0, self._client_count()),
            "active_connections": max(0, self._client_count()),
            "connected_clients": len(self._client_leases()),
            "model": getattr(settings, "llm_model", ""),
            "llm_context_length": context_metadata.get("llm_context_length", getattr(settings, "llm_context_length", 0)),
            "context_used": context_metadata.get("context_used", 0),
            "context_posture": context_metadata.get("context_posture", "baseline"),
            "context_signal": context_metadata.get("context_signal", ""),
            "context_guidance": context_metadata.get("context_guidance", ""),
            "compression_reason": context_metadata.get("compression_reason", ""),
            "compression_savings_ratio": context_metadata.get("compression_savings_ratio", 0.0),
            "context_budget_snapshot": context_metadata.get("context_budget_snapshot", {}),
            "session_id": str(getattr(engine, "_current_session_id", "") or ""),
            "runtime_source": self._runtime_source(),
            "runtime_executable": sys.executable,
            "runtime_version": self._runtime_version(),
            "pending_approvals": len(self._approval_pending),
            "watch_summary": watch_summary,
            "host_backend": self._host_backend_status(ctx),
        }

    async def host_status(self) -> dict[str, Any]:
        """Return daemon-owned host backend status."""
        ctx = self.context
        status = getattr(ctx, "host_backend_status", None)
        if callable(status):
            return dict(await status())
        return self._host_backend_status(ctx)

    async def host_start(self) -> dict[str, Any]:
        """Start daemon-owned CuaDriver without resetting chat state."""
        async with self._engine_lock:
            ctx = self.context
            start = getattr(ctx, "host_backend_start", None)
            if not callable(start):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await start())

    async def host_stop(self) -> dict[str, Any]:
        """Stop daemon-owned CuaDriver without shutting down leapd."""
        async with self._engine_lock:
            ctx = self.context
            stop = getattr(ctx, "host_backend_stop", None)
            if not callable(stop):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await stop())

    async def host_restart(self) -> dict[str, Any]:
        """Restart daemon-owned CuaDriver without resetting chat state."""
        async with self._engine_lock:
            ctx = self.context
            restart = getattr(ctx, "host_backend_restart", None)
            if not callable(restart):
                return {"ok": False, "started": False, "last_error": "host lifecycle is unavailable"}
            return dict(await restart())

    async def tools_list(self) -> dict[str, Any]:
        """Return daemon-owned tool summary for slash-command rendering."""
        from leapflow.cli.commands.slash_handlers import build_tool_payload

        return build_tool_payload(self.context)

    async def usage_summary(self) -> dict[str, Any]:
        """Return token usage for the current daemon-owned session."""
        from leapflow.cli.commands.slash_handlers import build_usage_payload

        return build_usage_payload(self.context)

    async def app_command(self, args: str = "") -> dict[str, Any]:
        """Return daemon-owned App Connector slash-command payload."""
        from leapflow.cli.commands.slash_handlers import build_app_payload

        return await build_app_payload(self.context, args)

    async def command_execute(self, name: str, args: str = "") -> dict[str, Any]:
        """Execute any engine-routed slash command via unified dispatch."""
        from leapflow.cli.commands.slash_handlers import command_execute

        return await command_execute(self.context, name, args)

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

    def _engine_context_metadata(self, engine: Any | None, settings: Any) -> dict[str, Any]:
        """Return safe context-budget metadata for daemon status and stream events."""
        context_length = max(0, int(getattr(settings, "llm_context_length", 0) or 0))
        metadata: dict[str, Any] = {
            "llm_context_length": context_length,
            "context_used": 0,
        }
        if engine is None:
            return metadata
        metadata["context_used"] = max(0, int(getattr(engine, "context_token_count", 0) or 0))
        snapshot = getattr(engine, "context_budget_snapshot", {})
        if callable(snapshot):
            snapshot = snapshot()
        if isinstance(snapshot, dict) and snapshot:
            safe_snapshot = dict(snapshot)
            if safe_snapshot.get("context_length"):
                metadata["llm_context_length"] = max(1, int(safe_snapshot["context_length"]))
            if safe_snapshot.get("total_tokens") is not None:
                metadata["context_used"] = max(0, int(safe_snapshot["total_tokens"]))
            posture = safe_snapshot.get("context_posture")
            if posture:
                metadata["context_posture"] = str(posture)
            signal = safe_snapshot.get("context_signal")
            if signal:
                metadata["context_signal"] = str(signal)
            guidance = safe_snapshot.get("context_guidance")
            if guidance:
                metadata["context_guidance"] = str(guidance)
            for key in (
                "compression_reason",
                "compression_savings_ratio",
                "compression_saved_tokens",
                "disclosure_level",
                "disclosure_reason",
                "disclosure",
            ):
                if safe_snapshot.get(key) is not None:
                    metadata[key] = safe_snapshot[key]
            metadata["context_budget_snapshot"] = safe_snapshot
        return metadata

    def _host_backend_status(self, ctx: Any | None) -> dict[str, Any]:
        if ctx is None:
            return {"backend": "none", "started": False, "reason": "runtime_not_initialized"}
        rpc = getattr(ctx, "rpc", None)
        snapshot = getattr(rpc, "status_snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception as exc:
                return {"backend": type(rpc).__name__, "started": False, "last_error": str(exc)}
        return {
            "backend": type(rpc).__name__ if rpc is not None else "none",
            "started": rpc is not None,
            "pid": None,
            "pid_source": "unavailable",
        }

    async def shutdown(self) -> None:
        if self._ctx is None:
            return
        ctx = self._ctx
        self._ctx = None
        if self._reentry_stop is not None:
            self._reentry_stop.set()
        if self._reentry_task is not None:
            try:
                await asyncio.wait_for(self._reentry_task, timeout=5.0)
            except (asyncio.TimeoutError, TimeoutError):
                self._reentry_task.cancel()
            except Exception:
                logger.debug("daemon: reentry task stop failed", exc_info=True)
            self._reentry_task = None
        self._reentry_stop = None
        if self._monitors is not None:
            try:
                await self._monitors.stop()
            except Exception:
                logger.debug("daemon: monitor stop failed", exc_info=True)
            self._monitors = None
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

    def _prune_engine_request_ledger(self) -> None:
        """Bound completed/failed engine request replay records by TTL and size."""
        now = time.time()
        for request_id, record in list(self._engine_request_ledger.items()):
            status = str(record.get("status") or "")
            if status == "running":
                continue
            completed_at = float(record.get("completed_at") or record.get("created_at") or 0.0)
            if now - completed_at > self._request_ledger_ttl_s:
                self._engine_request_ledger.pop(request_id, None)
        overflow = len(self._engine_request_ledger) - self._request_ledger_max_entries
        if overflow <= 0:
            return
        evictable = sorted(
            (
                (float(record.get("completed_at") or record.get("created_at") or 0.0), request_id)
                for request_id, record in self._engine_request_ledger.items()
                if str(record.get("status") or "") != "running"
            ),
            key=lambda item: item[0],
        )
        for _timestamp, request_id in evictable[:overflow]:
            self._engine_request_ledger.pop(request_id, None)

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
                    request_id="",
                    content="",
                    event_type="status",
                    metadata=notification.to_dict(),
                )
        finally:
            self.notification_bus.unsubscribe(subscriber_id)

    def _install_learn_notifications(self, ctx: Any) -> None:
        """Wire session progress/completion callbacks to the notification bus."""
        bus = self.notification_bus

        def _on_progress(stage: str, current: int, total: int) -> None:
            bus.emit_event(
                "teach.progress",
                phase=stage,
                current=current,
                total=total,
                progress=current / total if total > 0 else 0.0,
            )

        def _on_complete(result: Any) -> None:
            payload: dict[str, Any] = {"phase": "done"}
            if result:
                payload["step_count"] = getattr(result, "step_count", 0)
                payload["duration"] = getattr(result, "duration", 0.0)
                candidates = getattr(result, "candidates", None) or []
                payload["candidate_count"] = len(candidates)
                activated = getattr(result, "activated_skill_names", None) or set()
                payload["activated_skills"] = list(activated)
                new = getattr(result, "new_skills", None) or []
                payload["new_skills"] = list(new)
            bus.emit_event("teach.complete", **payload)

        if ctx.session:
            ctx.session.set_on_learn_progress(_on_progress)
            if hasattr(ctx.session, "set_on_learn_complete"):
                ctx.session.set_on_learn_complete(_on_complete)

            # Monitor mode changes to notify TUI of idle-watchdog stops
            original_on_idle = ctx.session._on_idle_timeout

            def _on_idle_with_notification() -> None:
                bus.emit_event("teach.stopped", reason="idle_timeout")
                original_on_idle()

            ctx.session._on_idle_timeout = _on_idle_with_notification

    def _install_daemon_approval(self, ctx: Any) -> None:
        try:
            from leapflow.security.approval import SessionAwareGate
            from leapflow.security.actions import ActionDescriptor
            from leapflow.security.orchestrator import ApprovalOrchestrator
            from leapflow.tools.gateway_tool import set_gateway_approval_gate
            from leapflow.tools.registry_bootstrap import set_file_read_gate, set_file_write_gate
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

            class _FileReadGate:
                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    mode: str = "raw",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    result = await orchestrator.evaluate(
                        ActionDescriptor.file_read(path, mode=mode, metadata=dict(sensitivity_meta or {}))
                    )
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            class _FileWriteGate:
                def __init__(self) -> None:
                    self.denial_message = ""

                async def check(
                    self,
                    path: str,
                    content: str,
                    mode: str = "overwrite",
                    sensitivity_meta: dict | None = None,
                ) -> bool:
                    result = await orchestrator.evaluate(
                        ActionDescriptor.file_write(path, content, mode=mode, metadata=dict(sensitivity_meta or {}))
                    )
                    self.denial_message = result.denial_message if not result.approved else ""
                    return result.approved

            set_file_read_gate(_FileReadGate())
            set_file_write_gate(_FileWriteGate())
            logger.debug("daemon approval gate installed")
        except Exception:
            logger.debug("daemon approval gate installation skipped", exc_info=True)

    async def _stream_engine_events(
        self,
        stream: AsyncIterator[object],
        approval_queue: asyncio.Queue[StreamChunk],
        *,
        request_id: str = "",
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
                    yield self._chunk_from_event(stream_event, request_id=request_id)
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
        route = _approval_route.get()
        if route is None:
            return "deny"
        queue, active_request_id = route
        pending_id = str(getattr(request, "request_id", "") or uuid.uuid4().hex)
        request_id = active_request_id or pending_id
        payload = request.to_dict()
        payload["pending_id"] = pending_id
        payload["request_id"] = request_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._approval_pending[pending_id] = {
            "request": payload,
            "future": future,
            "queue": queue,
            "created_at": time.time(),
        }
        await queue.put(StreamChunk(
            request_id=request_id,
            content="Approval required",
            event_type="approval_request",
            metadata={"approval": payload, "request_id": request_id},
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

    def _chunk_from_event(self, event: StreamEvent, *, request_id: str = "") -> StreamChunk:
        ctx = self.context
        engine = getattr(ctx, "engine", None)
        metadata = dict(event.metadata or {})
        session_id = getattr(engine, "_current_session_id", "") if engine else ""
        if request_id:
            metadata.setdefault("request_id", request_id)
        if session_id:
            metadata.setdefault("session_id", str(session_id))
        if engine is not None:
            metadata.update(self._engine_context_metadata(engine, getattr(ctx, "settings", self._settings)))
        return StreamChunk(
            request_id=request_id,
            content=event.content,
            done=False,
            event_type=event.type,
            metadata=metadata,
        )


_SESSION_ANALYSIS_SYSTEM = (
    "You are a session analyst writing FOR THE USER (not for the agent). Read the "
    "conversation transcript and return STRICT JSON only, with keys: story (a "
    "user-facing narrative of the user's goals, findings, and outcomes — NOT a "
    "replay of the agent's tool calls), insights (array of {title, summary, "
    "severity in [info,notable,alert], kind in [finding,process]}), decisions "
    "(array of strings), action_items (array of strings), open_questions (array "
    "of strings), entities (array of strings), next_prompts (array of strings), "
    "process_notes (array of strings), series_intents (array of {id, label, unit, "
    "kind in [line,area,ohlc,distribution]}). "
    "DE-WEIGHT the agent's own mechanics: tool usage, failures, retries, auth "
    "errors, and script fixes are LOW-SIGNAL process. Omit them, or fold at most "
    "one into insights with kind='process' and severity='info'; never emit them "
    "as decisions or action_items unless the USER must act (e.g. provide an API "
    "key). Put unavoidable process remarks in process_notes. In series_intents, "
    "only NAME chart-worthy quantitative series that are actually present in the "
    "data (labels/units); do NOT invent numbers — numeric values are extracted "
    "separately. If a Session file artifacts section is present, treat artifact "
    "contents as first-class evidence. Do not wrap the JSON in prose or code fences."
)

_SESSION_SALIENCE_SYSTEM = (
    "Answer with only YES or NO: does the latest conversation contain a new decision, "
    "a topic shift, or a new action item that would justify refreshing an analysis "
    "dashboard?"
)


def _parse_session_json(content: str) -> Any:
    """Best-effort extraction of a JSON object from an LLM response."""
    import json as _json

    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower().startswith("json"):
                text = rest
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return _json.loads(text)
    except Exception:
        return None


class _ProducerServices:
    """Facade exposing daemon capabilities to monitor producers (session, etc.)."""

    def __init__(self, service: "RuntimeLeapService") -> None:
        self._service = service

    async def session_history(self) -> dict[str, Any]:
        return await self._service.session_history()

    async def analyze_session(
        self,
        messages: list[dict[str, Any]],
        *,
        prior: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return await self._service._analyze_session_llm(messages, artifacts=artifacts)

    async def should_refresh(self, messages: list[dict[str, Any]]) -> bool:
        return await self._service._session_should_refresh(messages)


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

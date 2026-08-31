"""Manages the daemon-hosted monitor runtime (watches, findings, tickers).

Extracted from service.py (Phase 2.2) to keep RuntimeLeapService focused on
orchestration while MonitorCoordinator owns all monitor lifecycle and RPC logic.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, Optional

from leapflow.daemon.notifications import Notification
from leapflow.monitor.signal_noise import SignalNoiseConfig, SignalNoiseGate

logger = logging.getLogger(__name__)

# Rate-limit: max pushes per event_type per second.
_SIGNAL_STREAM_MAX_PER_SEC = 2


class MonitorCoordinator:
    """Manages the daemon-hosted monitor runtime (watches, findings, tickers)."""

    def __init__(self) -> None:
        self._monitors: Any | None = None
        self._bridge_subscribed: bool = False
        self._bridge_callback: Any | None = None
        self._event_bus: Any | None = None
        self._signal_stream_subscriber: Optional[Callable[..., None]] = None
        self._notification_bus: Any | None = None
        self._signal_stream_buffer: deque[dict[str, Any]] = deque(maxlen=50)
        self._signal_noise_gate: SignalNoiseGate | None = None
        self._off_loop: Optional[Callable[[Callable[[], Any]], Any]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self, ctx: Any, notification_bus: Any, settings: Any) -> None:
        """Build and start the monitor runtime if scheduler is enabled."""
        if not getattr(settings, "scheduler_enabled", True):
            return
        # The runtime's serialized DB channel, so status and watch reads never block
        # the loop. Captured here because this is the only place ctx is in scope.
        self._off_loop = getattr(ctx, "_run_deferred_db", None)
        try:
            from leapflow.monitor import (
                CapabilityAdaptationProducer,
                MonitorManager,
                PluginHealthProducer,
                SessionAnalysisProducer,
            )
            from leapflow.monitor.signal_producer import SignalObservationProducer

            bus = notification_bus
            self._monitors = MonitorManager(
                holder=ctx._db_holder,
                emit=lambda event_type, payload: bus.emit_event(event_type, **payload),
                services=self._build_services_proxy(ctx, settings),
                tick_seconds=int(getattr(settings, "scheduler_tick_seconds", 120)),
                grace_seconds=float(getattr(settings, "scheduler_grace_seconds", 120.0)),
            )
            self._monitors.producers.register(SessionAnalysisProducer())
            self._monitors.producers.register(SignalObservationProducer())
            self._monitors.producers.register(CapabilityAdaptationProducer())
            self._monitors.producers.register(PluginHealthProducer())
            self._register_hardware_producer(ctx, settings)
            setattr(ctx, "monitors", self._monitors)
            await self._monitors.start()

            # Subscribe one monitor/display boundary callback to EventBus. The
            # callback applies SignalNoiseGate once, then fans accepted events
            # into both EventBridge (watch activation) and LeapBoard stream.
            event_bus = getattr(ctx, "event_bus", None)
            self._signal_noise_gate = SignalNoiseGate(SignalNoiseConfig.from_settings(settings))
            if event_bus is not None and hasattr(event_bus, "subscribe") and not self._bridge_subscribed:
                self._bridge_callback = self._make_monitor_signal_subscriber(
                    self._monitors.event_bridge.on_event,
                    bus,
                )
                event_bus.subscribe(self._bridge_callback)
                self._bridge_subscribed = True
                self._event_bus = event_bus
                self._notification_bus = bus
                logger.debug("daemon: monitor signal subscriber registered")

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

            # Auto-arm default event-driven watches (idempotent).
            await self._arm_default_watches()

            logger.debug("daemon: monitor runtime started")
        except Exception:
            logger.debug("daemon: monitor runtime start skipped", exc_info=True)
            self._monitors = None
            setattr(ctx, "monitors", None)

    def _register_hardware_producer(self, ctx: Any, settings: Any) -> None:
        """Register the physical-bench domain, but only when hardware is enabled.

        Conditional because ``hardware.enabled`` is off by default: with no devices
        declared the producer would run every cycle to conclude there is nothing to
        report. The registry is resolved lazily rather than captured, since its
        reading store and experience store are bound during deferred initialization
        -- after this call.
        """
        if not getattr(settings, "hardware_enabled", False):
            return
        monitors = self._monitors
        if monitors is None:
            return
        try:
            from leapflow.hardware.observability import HardwareObservationProducer

            monitors.producers.register(
                HardwareObservationProducer(lambda: getattr(ctx, "_hardware_registry", None))
            )
        except Exception:
            logger.debug("daemon: hardware observability producer unavailable", exc_info=True)

    def _build_services_proxy(self, ctx: Any, settings: Any) -> Any:
        """Build the _ProducerServices proxy.

        This is deferred to the service layer via a back-reference injected
        before start() is called. When no back-reference is available (e.g.
        tests that set _monitors directly), returns None.
        """
        # The proxy is built by the service layer and passed via
        # _set_service_ref(). This method is a placeholder; the actual
        # _ProducerServices is built in service.py and passed to start().
        return None

    def update_settings(self, settings: Any) -> None:
        """Hot-apply signal noise policy settings to the running gate."""
        gate = self._signal_noise_gate
        if gate is not None:
            gate.update_config(SignalNoiseConfig.from_settings(settings))

    def _make_monitor_signal_subscriber(
        self,
        event_bridge_callback: Callable[[Any], None],
        notification_bus: Any,
    ) -> Callable[..., None]:
        """Apply noise policy once, then route accepted events to watch + stream."""
        stream_callback = self._make_signal_stream_subscriber(notification_bus)

        def _on_event(event: Any) -> None:
            gate = self._signal_noise_gate
            if gate is not None and not gate.should_pass(event):
                return
            event_bridge_callback(event)
            stream_callback(event)

        return _on_event

    def _make_signal_stream_subscriber(self, notification_bus: Any) -> Callable[..., None]:
        """Build a rate-limited callback that pushes signal summaries to NotificationBus.

        Every event (regardless of rate-limit) is appended to the ring buffer so
        the dashboard can display the full recent stream.
        """
        last_push: dict[str, float] = {}
        min_interval = 1.0 / _SIGNAL_STREAM_MAX_PER_SEC

        def _on_event(event: Any) -> None:
            event_type = getattr(event, "event_type", "")
            if not event_type:
                return
            summary = {
                "event_type": event_type,
                "source": getattr(event, "source", ""),
                "ts": getattr(event, "timestamp", time.time()),
            }
            # Always record into ring buffer for dashboard polling.
            self._signal_stream_buffer.append(summary)
            # Rate-limited push to notification bus.
            now = time.monotonic()
            prev = last_push.get(event_type, 0.0)
            if now - prev < min_interval:
                return
            last_push[event_type] = now
            notification_bus.emit(Notification(event_type="signal.stream", payload=summary))

        return _on_event

    def get_signal_stream(self) -> list[dict[str, Any]]:
        """Return recent accepted signal events (up to 50) from the ring buffer."""
        return list(self._signal_stream_buffer)

    @property
    def signal_noise_stats(self) -> dict[str, Any]:
        """Return monitor/display noise-gate counters for metrics."""
        gate = self._signal_noise_gate
        return gate.stats if gate is not None else {}

    # ── Default event-driven watches ──────────────────────────────────────

    # Default watches to arm on daemon startup. Each tuple:
    # (name, domain, trigger_expr)
    _DEFAULT_WATCHES = [
        ("fs-observer", "signal", "event:fs.*"),
        ("gateway-observer", "signal", "event:gateway.*"),
        # Plugin health is polled rather than event-driven: trust degradation and a
        # rising error rate are both trends, visible only by comparing successive
        # observations. Without this watch the producer is registered and never
        # called, which is how it sat unused while its own docstring said otherwise.
        ("plugin-health", "plugin_health", "5m"),
        # Polled for the same reason: an envelope excursion is caught by the event
        # detector, but cadence drift, quality decay and unpersisted windows are all
        # trends that only a comparison between cycles can show. Armed regardless of
        # ``hardware.enabled`` so the board has a watch to report against; with the
        # producer unregistered the cycle is a no-op.
        ("hardware-bench", "hardware", "2m"),
    ]

    async def _arm_default_watches(self) -> None:
        """Arm built-in watches if not already present (idempotent).

        Stale watches (state=done/failed) with the same name are removed and
        re-created so daemon restarts always restore monitoring.
        """
        from leapflow.monitor import WatchSpec

        monitors = self._monitors
        if monitors is None:
            return

        # Keyed by name, not by trigger label. The label is rendered by the manager
        # ("every 5m", "event:fs.*"), so reconstructing it here only worked for event
        # triggers -- an interval watch would never match its own entry and would be
        # re-armed on every start, accumulating duplicates.
        existing: dict[str, tuple[Any, bool]] = {}
        _ACTIVE_STATES = {"armed", "watching", "due", "confirming", "executing"}
        try:
            for view in monitors.list_watches():
                is_active = str(view.state) in _ACTIVE_STATES
                existing[str(view.name)] = (view, is_active)
        except Exception:
            logger.debug("daemon: failed to list watches for default arm", exc_info=True)
            return

        for name, domain, trigger_expr in self._DEFAULT_WATCHES:
            entry = existing.get(name)
            if entry is not None:
                view, is_active = entry
                if is_active:
                    # Already armed and active — nothing to do.
                    continue
                # Stale (done/failed/suspended) — remove so we can re-arm.
                try:
                    monitors.stop_watch(view.watch_id)
                except Exception:
                    pass
                try:
                    monitors._task_store.delete(view.watch_id)
                except Exception:
                    logger.debug("daemon: failed to delete stale watch %s", name, exc_info=True)
            try:
                await monitors.arm_watch(
                    WatchSpec(
                        name=name,
                        domain=domain,
                        trigger_expr=trigger_expr,
                    )
                )
                logger.debug("daemon: armed default watch %s (%s)", name, trigger_expr)
            except Exception:
                logger.debug("daemon: failed to arm default watch %s", name, exc_info=True)

    async def stop(self) -> None:
        """Stop the monitor runtime."""
        if self._monitors is not None:
            # Unsubscribe signal stream subscriber.
            if self._signal_stream_subscriber is not None:
                if self._event_bus is not None and hasattr(self._event_bus, "unsubscribe"):
                    self._event_bus.unsubscribe(self._signal_stream_subscriber)
                self._signal_stream_subscriber = None
                self._notification_bus = None
            # Unsubscribe event bridge before stopping monitors.
            if self._bridge_subscribed and self._bridge_callback is not None:
                if self._event_bus is not None and hasattr(self._event_bus, "unsubscribe"):
                    self._event_bus.unsubscribe(self._bridge_callback)
                self._bridge_subscribed = False
                self._bridge_callback = None
                self._event_bus = None
                self._notification_bus = None
                self._signal_noise_gate = None
            try:
                await self._monitors.stop()
            except Exception:
                logger.debug("daemon: monitor stop failed", exc_info=True)
            self._monitors = None

    # ── Watch RPC operations ──────────────────────────────────────────────

    def _require_monitors(self) -> Any:
        if self._monitors is None:
            raise RuntimeError("monitor runtime is not available (scheduler disabled)")
        return self._monitors

    async def arm(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Register a new watch from a spec dict."""
        from leapflow.monitor import WatchSpec

        view = await self._require_monitors().arm_watch(WatchSpec.from_dict(spec or {}))
        return view.to_dict()

    async def list_watches(self) -> list[dict[str, Any]]:
        """List all registered watches, reading the store off the event loop."""
        if self._monitors is None:
            return []
        return await self._read_watches()

    async def _read_watches(self) -> list[dict[str, Any]]:
        """Load every watch view without blocking the loop.

        The store is DuckDB, so this is a blocking read of unbounded duration -- it
        grows with the number of armed watches. Run inline it stalls every other RPC
        for as long as the query takes, which is how a daemon that is merely busy
        becomes a daemon that looks hung.

        Routed through the runtime's single-thread channel rather than
        ``asyncio.to_thread``: reads here must stay serialized against the deferred
        initialisation work that uses the same database, and one worker is what makes
        that true by construction.
        """
        monitors = self._monitors
        if monitors is None:
            return []

        def _load() -> list[dict[str, Any]]:
            return [view.to_dict() for view in monitors.list_watches()]

        off_loop = self._off_loop
        if off_loop is None:
            # No runtime channel installed (tests, or a coordinator used standalone).
            # Reading inline is worse for latency but still correct, and refusing
            # would turn a slow answer into no answer.
            return _load()
        result = await off_loop(_load)
        return list(result or [])

    async def get_watch(self, watch_id: str) -> dict[str, Any]:
        """Get a single watch by id."""
        view = self._require_monitors().get_watch(watch_id)
        return view.to_dict() if view else {}

    async def pause(self, watch_id: str) -> dict[str, Any]:
        """Pause an active watch."""
        view = self._require_monitors().pause_watch(watch_id)
        return view.to_dict() if view else {}

    async def resume(self, watch_id: str) -> dict[str, Any]:
        """Resume a paused watch."""
        view = self._require_monitors().resume_watch(watch_id)
        return view.to_dict() if view else {}

    async def stop_watch(self, watch_id: str) -> dict[str, Any]:
        """Stop a watch permanently."""
        view = self._require_monitors().stop_watch(watch_id)
        return view.to_dict() if view else {}

    async def mute(self, watch_id: str, muted: bool = True) -> dict[str, Any]:
        """Mute or unmute a watch."""
        view = self._require_monitors().set_muted(watch_id, bool(muted))
        return view.to_dict() if view else {}

    async def refresh(self, watch_id: str) -> dict[str, Any]:
        """Manually trigger a watch run."""
        return await self._require_monitors().run_watch_once(watch_id)

    async def findings(
        self, watch_id: str = "", limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get findings, optionally filtered by watch_id."""
        if self._monitors is None:
            return []
        results = self._monitors.list_findings(
            watch_id=watch_id or None, limit=int(limit), offset=int(offset)
        )
        return [finding.to_dict() for finding in results]

    # ── Status / queries ──────────────────────────────────────────────────

    def has_active_watches(self) -> bool:
        """Return True when any hosted watch is armed/watching (idle keep-alive)."""
        monitors = self._monitors
        if monitors is None:
            return False
        try:
            return bool(monitors.has_active_watches())
        except Exception:
            return False

    async def get_summary(self) -> dict[str, Any]:
        """Runtime summary for daemon.status().

        Async because it reads the watch store, and that read is DuckDB: done inline
        it made every status poll block the loop for the length of the query, which
        gets worse with each armed watch. ``status()`` is the most frequently called
        RPC there is -- the TUI status bar polls it -- so it is the last place that
        can afford a synchronous database read.
        """
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
            watches = await self._read_watches()
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

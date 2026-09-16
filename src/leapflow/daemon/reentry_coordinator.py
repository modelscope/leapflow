# Copyright (c) Alibaba, Inc. and its affiliates.
"""Manages the reentry driver lifecycle: background tick loop, gateway observation.

Extracted from service.py (Phase 2.4) to keep RuntimeLeapService focused on
orchestration while ReentryCoordinator owns the reentry service lifecycle.

Phase 3 enhancements:
- Conditional start: gated by ``agent_reentry_enabled`` setting.
- Idle-aware ticker: escalates to ``idle_interval`` when consecutive ticks
  dispatch zero work; reverts to ``base_interval`` on first dispatch.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Consecutive zero-dispatch ticks before escalating to idle interval.
_IDLE_THRESHOLD: int = 3


class ReentryCoordinator:
    """Manages the reentry driver lifecycle: background tick loop, gateway observation."""

    def __init__(self) -> None:
        self._reentry_task: asyncio.Task[Any] | None = None
        self._reentry_stop: asyncio.Event | None = None
        self._reentry_service: Any | None = None

    async def start(
        self,
        ctx: Any,
        settings: Any,
        turn_admission: Any,
        notification_bus: Any,
        request_approval: Callable[..., Any] | None = None,
    ) -> None:
        """Start the background re-entry service (S2 N3b + N4 + N5).

        Dispatches due TIME triggers periodically and matches inbound gateway
        EVENT triggers, always as *isolated subagents* (fresh context -> no
        interactive-engine / working-memory / session pollution), serialized via
        the turn-admission gate (exclusive with all turns). Gated by
        ``agent_reentry_enabled`` (default off) plus a
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
            resolved_approval = None
            if getattr(settings, "agent_reentry_send_enabled", False):
                from leapflow.scheduler.reentry_send import SendGovernor, SendRateLimiter
                from leapflow.security.send_trust import SendTrustLedger
                send_governor = SendGovernor(
                    trust=SendTrustLedger(
                        verified_at=int(getattr(settings, "agent_reentry_send_verified_at", 3)),
                    ),
                    rate=SendRateLimiter(
                        per_hour=int(getattr(settings, "agent_reentry_send_rate_per_hour", 4)),
                    ),
                    enabled=True,
                    global_budget=int(getattr(settings, "agent_reentry_send_global_budget", 50)),
                )
                gw = getattr(ctx, "gateway_server", None)
                send_fn = getattr(gw, "send_message", None) if gw is not None else None
                resolved_approval = request_approval

            service = ReentryService(
                store=store,
                manager=manager,
                settings=settings,
                engine_lock=turn_admission.exclusive_gate(),
                notify=lambda event_type, **kw: notification_bus.emit_event(event_type, **kw),
                global_budget=int(getattr(settings, "agent_reentry_global_budget", 100) or 0),
                send_governor=send_governor,
                send_fn=send_fn,
                request_approval=resolved_approval,
            )
            self._reentry_service = service
            # N4: observe inbound gateway messages for EVENT-trigger matches.
            try:
                setattr(ctx, "_reentry_event_observer", service.on_gateway_message)
            except Exception:
                logger.debug("daemon: reentry event observer wiring failed", exc_info=True)

            base_interval = max(5.0, float(getattr(settings, "agent_reentry_tick_seconds", 30.0) or 30.0))
            idle_interval = max(
                base_interval,
                float(getattr(settings, "agent_reentry_idle_tick_seconds", base_interval * 4) or base_interval * 4),
            )
            self._reentry_stop = asyncio.Event()

            async def _loop() -> None:
                stop = self._reentry_stop
                assert stop is not None
                idle_count = 0
                while not stop.is_set():
                    # Adaptive interval: use longer sleep when idle
                    interval = idle_interval if idle_count >= _IDLE_THRESHOLD else base_interval
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=interval)
                        break  # stop signalled
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
                    if stop.is_set():
                        break
                    try:
                        dispatched = await service.tick()
                        if dispatched:
                            idle_count = 0
                        else:
                            idle_count += 1
                    except Exception:
                        logger.debug("reentry service tick failed", exc_info=True)
                        idle_count += 1

            self._reentry_task = asyncio.create_task(_loop(), name="leapd-reentry-driver")
            logger.debug(
                "daemon: re-entry service started (base=%.0fs, idle=%.0fs)",
                base_interval, idle_interval,
            )
        except Exception:
            logger.debug("daemon: re-entry service start skipped", exc_info=True)

    async def stop(self) -> None:
        """Stop the reentry driver and wait for task completion."""
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

    def is_running(self) -> bool:
        """Check if reentry driver is active."""
        return self._reentry_task is not None and not self._reentry_task.done()

    @property
    def service(self) -> Any:
        """Access the underlying ReentryService (may be None)."""
        return self._reentry_service

"""Re-entry orchestration service (S2 phases N3b–N5).

Consolidates the two trigger sources (time ticks and gateway events) behind one
dispatch path: build an isolated subagent config from the OrientSnapshot, run it
serialized (optional engine lock), and record the outcome. Safety closure (N5):

- **Audit**: every dispatch emits ``reentry.dispatched`` / ``reentry.completed``
  notifications and a structured log line.
- **Global budget**: a lifetime cap across all triggers backstops runaway loops
  (per-trigger ``max_reentries`` / ``deadline`` still apply in the store).
- **Governed proactive Act (SO3, default-off)**: the re-entry subagent still
  blocks ``send_message``; any outbound delivery of the *result* is decided at
  this service level by ``SendGovernor`` (send-scope Progressive Trust) and,
  below VERIFIED trust, an asynchronous ApprovalGate (deny on timeout / no
  approver). Enabled only via ``agent.reentry_send_enabled``.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from leapflow.scheduler.reentry_driver import (
    ReentryDriver,
    build_reentry_subagent_config,
    event_matches,
)
from leapflow.scheduler.reentry_send import (
    ReentrySendSpec,
    SendAction,
    resolve_reentry_send_target,
)
from leapflow.storage.reentry_store import OrientSnapshot, ReentryStore

logger = logging.getLogger(__name__)

NotifyFn = Callable[..., Any]

# TTL for a queued autonomous-send approval; the daemon denies on timeout.
_SEND_APPROVAL_TTL = 300.0
_ALLOW_DECISIONS = frozenset({"allow", "allow_once", "allow_session", "allow_always"})


class ReentryService:
    """Owns re-entry dispatch for both time and event triggers."""

    def __init__(
        self,
        *,
        store: ReentryStore,
        manager: Any,
        settings: Any,
        engine_lock: Any = None,
        notify: Optional[NotifyFn] = None,
        global_budget: int = 100,
        max_per_tick: int = 4,
        send_governor: Any = None,
        send_fn: Optional[Callable[..., Any]] = None,
        request_approval: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._store = store
        self._manager = manager
        self._settings = settings
        self._engine_lock = engine_lock
        self._notify = notify
        self._global_budget = max(0, int(global_budget))
        self._dispatched_total = 0
        # SO3: governed proactive delivery (all optional; None => never sends).
        self._send_governor = send_governor
        self._send_fn = send_fn
        self._request_approval = request_approval
        self._driver = ReentryDriver(
            store=store,
            runner=self._dispatch,
            enabled=self._enabled,
            max_per_tick=max_per_tick,
        )

    def _enabled(self) -> bool:
        return bool(getattr(self._settings, "agent_reentry_enabled", False))

    def _budget_ok(self) -> bool:
        return self._global_budget <= 0 or self._dispatched_total < self._global_budget

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._notify is None:
            return
        try:
            self._notify(event_type, **payload)
        except Exception:
            logger.debug("reentry notify failed", exc_info=True)

    async def _dispatch(self, orient: OrientSnapshot, trigger: Any = None) -> None:
        """Run one re-entry as an isolated subagent (shared by both sources)."""
        if not self._budget_ok():
            logger.warning(
                "reentry global budget exhausted (%d); skipping task=%s",
                self._global_budget, orient.task_id,
            )
            return
        config = build_reentry_subagent_config(orient)
        self._emit("reentry.dispatched", task_id=orient.task_id)
        if self._engine_lock is not None:
            async with self._engine_lock:
                result = await self._manager.delegate(config)
        else:
            result = await self._manager.delegate(config)
        self._dispatched_total += 1
        status = getattr(result, "status", "")
        summary = str(getattr(result, "summary", "") or "")
        logger.info(
            "reentry.completed task=%s status=%s total=%d",
            orient.task_id, status, self._dispatched_total,
        )
        self._emit(
            "reentry.completed",
            task_id=orient.task_id,
            status=status,
            summary=summary[:2000],
        )
        # SO3: governed proactive delivery of the result (outside the engine lock).
        if trigger is not None:
            await self._maybe_send(trigger, orient, summary)

    async def _maybe_send(self, trigger: Any, orient: OrientSnapshot, summary: str) -> None:
        """SO3: decide + perform governed outbound delivery of a re-entry result.

        Default-off and fail-safe: does nothing unless a ``SendGovernor`` and a
        send function are wired and the governor is enabled. Never raises into
        the dispatch path. Below VERIFIED trust, delivery requires an approval
        (which also accrues trust); autonomous context with no approver denies.
        """
        gov = self._send_governor
        if gov is None or self._send_fn is None or not summary.strip():
            return
        try:
            spec = ReentrySendSpec(
                target=resolve_reentry_send_target(trigger),
                text=summary,
                origin_trigger_id=str(getattr(trigger, "trigger_id", "") or orient.task_id),
            )
            decision = gov.decide(
                spec,
                destructive=False,   # first phase: reply to the originating chat only
                has_approver=self._request_approval is not None,
                now=time.time(),
            )
            if decision.action is SendAction.AUTO_ALLOW:
                sent = await self._do_send(spec)
                self._emit("reentry.send", task_id=orient.task_id,
                           result="auto_allow" if sent else "send_failed")
            elif decision.action is SendAction.NEEDS_APPROVAL:
                await self._approve_and_send(spec, orient)
            else:
                self._emit("reentry.send", task_id=orient.task_id, result=decision.reason)
        except Exception:
            logger.error("reentry send failed for task=%s", orient.task_id, exc_info=True)

    async def _do_send(self, spec: ReentrySendSpec) -> bool:
        """Perform the actual gateway send; record it for idempotency/budget."""
        if spec.target is None or self._send_fn is None:
            return False
        try:
            result = await self._send_fn(spec.target.platform, spec.target.chat, spec.text)
        except Exception:
            logger.error("gateway send failed", exc_info=True)
            return False
        ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
        if ok:
            self._send_governor.record_sent(spec)
        return ok

    async def _approve_and_send(self, spec: ReentrySendSpec, orient: OrientSnapshot) -> None:
        """Queue an asynchronous human approval; send + accrue trust on ALLOW."""
        from leapflow.security.approval import ApprovalRequest

        grant = spec.target.grant_key(spec.kind)
        request = ApprovalRequest(
            category="reentry_send",
            detail=f"Autonomous reply to {spec.target.platform}:{spec.target.chat} — {spec.text[:200]}",
            risk_hint=0.7,
            expires_at=time.time() + _SEND_APPROVAL_TTL,
            metadata={"platform": spec.target.platform, "chat": spec.target.chat, "task_id": orient.task_id},
        )
        try:
            decision = await self._request_approval(request)
        except Exception:
            decision = "deny"
        value = str(getattr(decision, "value", decision)).lower()
        if value in _ALLOW_DECISIONS:
            self._send_governor.record_human_allow(grant)
            sent = await self._do_send(spec)
            self._emit("reentry.send", task_id=orient.task_id,
                       result="approved" if sent else "approved_send_failed")
        else:
            self._send_governor.record_human_deny(grant)
            self._emit("reentry.send", task_id=orient.task_id, result="denied")

    async def tick(self, now: Optional[float] = None) -> int:
        """Dispatch due TIME triggers (called periodically by the daemon)."""
        return await self._driver.tick(now)

    async def on_gateway_message(
        self, *, platform: str = "", chat: str = "", text: str = "",
    ) -> int:
        """Match an inbound gateway message against armed EVENT triggers (N4).

        For each match: CAS-claim (``fire``) and dispatch. Single-shot triggers
        become exhausted; recurring ones stay armed to match future events
        (bounded by ``max_reentries``).
        """
        if not self._enabled():
            return 0
        dispatched = 0
        for trig in self._store.list_armed_events():
            if not event_matches(trig.event_match, platform=platform, chat=chat, text=text):
                continue
            claimed = self._store.fire(trig.trigger_id)
            if claimed is None or claimed.orient is None:
                continue
            try:
                await self._dispatch(claimed.orient, claimed)
                dispatched += 1
            except Exception:
                logger.error("reentry event dispatch failed for %s", claimed.trigger_id, exc_info=True)
        return dispatched

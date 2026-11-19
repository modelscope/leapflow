# Copyright (c) Alibaba, Inc. and its affiliates.
"""Watch lifecycle orchestration for the monitoring subsystem.

``MonitorManager`` wires the domain-neutral contract to the existing scheduler:
watches are ``ArmedTask`` rows (``kind=watch``); each due tick runs the matching
producer, persists findings, and pushes qualifying ones to view clients through
an injected ``emit`` callback (the daemon NotificationBus). It owns no domain
logic and no transport -- both are injected.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Callable, List, Optional, Sequence

from leapflow.monitor.delivery import BoundedFindingDelivery, DeliveryCandidate, DeliveryStore, DeliveryValidator, FindingPresentation
from leapflow.monitor.event_bridge import EventBridge
from leapflow.monitor.finding_store import FindingStore
from leapflow.monitor.producers import ProducerRegistry
from leapflow.monitor.types import (
    EVENT_ERROR,
    EVENT_WATCH_STATE,
    METADATA_CLIENT_COUPLED_KEY,
    METADATA_KIND_KEY,
    METADATA_MUTED_KEY,
    WATCH_KIND,
    Finding,
    FindingDelivery,
    CurrentIncidentSource,
    Severity,
    ProducerContext,
    WatchSpec,
    WatchView,
)
from leapflow.scheduler.coordinator import TaskCoordinator
from leapflow.scheduler.local_scheduler import LocalScheduler
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.triggers.event import EventTrigger
from leapflow.scheduler.types import ArmedTask, TaskSource, TaskState

logger = logging.getLogger(__name__)

# Emit signature: (event_type, payload_dict) -> None. Injected by the host.
EmitFn = Callable[[str, dict], None]

_ACTIVE_STATES = frozenset({
    TaskState.ARMED.value,
    TaskState.WATCHING.value,
    TaskState.DUE.value,
    TaskState.CONFIRMING.value,
    TaskState.EXECUTING.value,
})


def _format_trigger(task: ArmedTask) -> str:
    """Return a compact human-readable trigger label for a task."""
    cfg = task.trigger_config if isinstance(task.trigger_config, dict) else {}
    if task.trigger_type == "interval":
        sec = float(cfg.get("interval_seconds", 0) or 0)
        if sec < 60:
            return f"every {int(sec)}s"
        if sec < 3600:
            return f"every {int(sec / 60)}m"
        if sec < 86400:
            return f"every {int(sec / 3600)}h"
        return f"every {int(sec / 86400)}d"
    if task.trigger_type == "cron":
        return str(cfg.get("expression", "cron"))
    if task.trigger_type == "event":
        return f"event:{cfg.get('event_pattern', '?')}"
    if task.trigger_type == "condition":
        return f"cond:{str(cfg.get('expression', '?'))[:24]}"
    return task.trigger_type


def _is_watch(task: ArmedTask) -> bool:
    """Return True when an ArmedTask row represents a monitor watch."""
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    return meta.get(METADATA_KIND_KEY) == WATCH_KIND


class _MonitorExecutor:
    """SkillExecutor adapter: run a producer, persist + push its findings."""

    def __init__(
        self,
        *,
        task_store: TaskStore,
        finding_store: FindingStore,
        producers: ProducerRegistry,
        emit: Optional[EmitFn],
        services: Any,
        delivery: FindingPresentation,
        run_db: Any = None,
    ) -> None:
        self._delivery = delivery
        self._run_db = run_db
        self._tasks = task_store
        self._findings = finding_store
        self._producers = producers
        self._emit = emit
        self._services = services

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        """Run one observation cycle for a watch (skill_name == domain)."""
        force = bool(parameters.get("_force", False))
        spec = WatchSpec.from_params(parameters)
        task = await self._db(lambda: self._tasks.load(spec.watch_id)) if spec.watch_id else None
        muted = bool((task.metadata or {}).get(METADATA_MUTED_KEY)) if task else False
        run_count = task.run_count if task else 0
        last_run_at = task.last_run_at if task else 0.0

        producer = self._producers.resolve(spec.domain)
        if producer is None:
            logger.debug(
                "monitor: no producer for domain=%s (watch=%s)", spec.domain, spec.watch_id
            )
            return {"ok": False, "error": f"no producer registered for domain {spec.domain!r}"}

        ctx = ProducerContext(
            spec=replace(spec, muted=muted),
            now=time.time(),
            run_count=run_count,
            last_run_at=last_run_at,
            services=self._services,
            force=force,
        )
        try:
            findings = list(await producer.observe(ctx))
        except Exception as exc:  # noqa: BLE001 - producer errors are surfaced, not fatal
            logger.warning("monitor: producer %s failed: %s", spec.domain, exc)
            self._push(EVENT_ERROR, {"watch_id": spec.watch_id, "domain": spec.domain,
                                     "error": str(exc)})
            return {"ok": False, "error": str(exc)}

        try:
            saved, state_updates = await self._db(lambda: self._persist(findings, spec.watch_id))
            current_task = await self._db(lambda: self._tasks.load(spec.watch_id)) if spec.watch_id else None
            muted = (current_task is None or current_task.state not in _ACTIVE_STATES
                     or bool((current_task.metadata or {}).get(METADATA_MUTED_KEY)))
            current_spec = WatchSpec.from_params(current_task.parameters) if current_task else spec
            await self._delivery.offer([*state_updates, *saved], current_spec.push_threshold(), muted)
        except Exception as exc:  # noqa: BLE001 - never present evidence that failed persistence
            logger.warning("monitor: persistence/delivery failed", exc_info=True)
            self._push(EVENT_ERROR, {"watch_id": spec.watch_id, "domain": spec.domain, "error": str(exc)})
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "findings": len(saved), "emitted": 0}

    async def _db(self, fn: Callable[[], Any]) -> Any:
        return await self._run_db(fn) if self._run_db else await asyncio.to_thread(fn)

    def _persist(self, findings: List[Finding], watch_id: str) -> tuple[List[Finding], List[Finding]]:
        saved, state_updates = [], []
        for finding in findings:
            if not finding.watch_id:
                finding = replace(finding, watch_id=watch_id)
            existing = self._findings.find_dedup(finding.watch_id, finding.dedup_key) if finding.dedup_key else None
            if existing is not None:
                finding = replace(finding, finding_id=existing.finding_id)
                delivery = finding.delivery
                withdrawn = delivery.withdrawn_keys
                if delivery.key and not delivery.actionable and (not delivery.active or not delivery.current):
                    withdrawn = (*withdrawn, delivery.key)
                if delivery.key and (delivery.actionable or (delivery.active and delivery.current)):
                    state_updates.append(finding)
                elif withdrawn:
                    state_updates.append(replace(finding, severity=Severity.INFO,
                        delivery=FindingDelivery(active=False, withdrawn_keys=withdrawn)))
                continue
            self._findings.save(finding)
            saved.append(finding)
        return saved, state_updates

    def _push(self, event_type: str, payload: dict) -> None:
        if self._emit is None:
            return
        try:
            self._emit(event_type, payload)
        except Exception:  # noqa: BLE001 - a failing sink must not break the tick
            logger.debug("monitor: emit failed for %s", event_type, exc_info=True)


class MonitorManager:
    """Owns watch lifecycle: arm, list, control, and finding retrieval.

    Local execution only in this phase; cloud dispatch remains available through
    the scheduler for a later phase. Transport (``emit``) and domain producers
    are injected, keeping this class domain- and platform-neutral.
    """

    def __init__(
        self,
        *,
        holder: Any,
        producers: Optional[ProducerRegistry] = None,
        emit: Optional[EmitFn] = None,
        services: Any = None,
        tick_seconds: int = 60,
        grace_seconds: float = 120.0,
        delivery_factory: Callable[[DeliveryStore, Optional[EmitFn], DeliveryValidator], FindingPresentation] | None = None,
        run_db: Any = None,
    ) -> None:
        self._run_db = run_db
        self._task_store = TaskStore(holder)
        self._finding_store = FindingStore(holder)
        self.producers = producers or ProducerRegistry()
        self._emit = emit
        self.delivery = (delivery_factory(self._finding_store, emit, self._validate_delivery) if delivery_factory else
                         BoundedFindingDelivery(self._finding_store, emit, run_db=run_db,
                                                validate=self._validate_delivery))
        self._executor = _MonitorExecutor(
            task_store=self._task_store,
            finding_store=self._finding_store,
            producers=self.producers,
            emit=emit,
            services=services,
            delivery=self.delivery,
            run_db=run_db,
        )
        self._scheduler = LocalScheduler(
            store=self._task_store,
            executor=self._executor,
            tick_seconds=tick_seconds,
            grace_seconds=grace_seconds,
        )
        self._coordinator = TaskCoordinator(
            store=self._task_store,
            local_scheduler=self._scheduler,
            cloud_dispatcher=None,
            default_tier="local",
        )
        self._event_bridge = EventBridge(
            scheduler_wake=self._scheduler.wake,
            mark_due=self._mark_event_due,
        )
        self._started = False
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def _validate_delivery(self, candidates: Sequence[DeliveryCandidate], now: float) -> dict[str, tuple[str, ...]]:
        """Re-read consent off-loop; domain producers alone validate live evidence."""
        started = time.monotonic()
        watch_ids = {watch_id for item in candidates for watch_id in item.watch_ids}
        def load() -> dict[str, Any]:
            return {watch_id: self._task_store.load(watch_id) for watch_id in watch_ids}
        tasks = await self._executor._db(load)
        current: dict[str, set[str]] = {}
        for domain in {item.domain for item in candidates if not item.actionable}:
            producer = self.producers.resolve(domain)
            if isinstance(producer, CurrentIncidentSource):
                keys = [item.incident_key for item in candidates if item.domain == domain and not item.actionable]
                current[domain] = producer.current_incidents(keys, now + time.monotonic() - started)
        eligible = {}
        for item in candidates:
            if item.domain in current and not item.actionable and item.incident_key not in current[item.domain]:
                continue
            owners = []
            for watch_id in item.watch_ids:
                task = tasks.get(watch_id)
                if (task is not None and task.state in _ACTIVE_STATES
                        and not (task.metadata or {}).get(METADATA_MUTED_KEY)
                        and Severity(item.severity).rank >= WatchSpec.from_params(task.parameters).push_threshold().rank):
                    owners.append(watch_id)
            if owners:
                eligible[item.key] = tuple(owners)
        return eligible

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background tick loop (idempotent)."""
        if self._started:
            return
        self.delivery.start()
        tasks = await self._executor._db(self._task_store.load_all)
        for task in tasks:
            if _is_watch(task) and task.trigger_type == "event" and task.state not in {"done", "failed"}:
                self._event_bridge.register(task.task_id, EventTrigger.deserialize(task.trigger_config))
        await self._scheduler.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the background tick loop (idempotent)."""
        self._event_bridge.close()
        if self._started:
            await self._scheduler.stop()
        self._started = False
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.delivery.stop()

    @property
    def event_bridge(self) -> EventBridge:
        """Expose the event bridge for external EventBus subscription."""
        return self._event_bridge

    @property
    def finding_store(self) -> FindingStore:
        """Expose the finding store for read-only queries by hosts."""
        return self._finding_store

    # ── Watch management ───────────────────────────────────────────────────

    async def arm_watch(self, spec: WatchSpec) -> WatchView:
        """Create and register a watch, returning its runtime view."""
        task = await self._coordinator.arm(
            skill_name=spec.domain,
            trigger_expr=spec.trigger_expr,
            execution_tier="local",
            max_runs=spec.max_runs,
            parameters=spec.to_task_parameters(),
            source=TaskSource.SYSTEM.value,
        )
        # Backfill watch_id into parameters and stamp watch metadata so ticks
        # and listings can identify the row without re-deriving it.
        params = dict(task.parameters) if isinstance(task.parameters, dict) else {}
        params["watch_id"] = task.task_id
        task.parameters = params
        task.metadata = {
            METADATA_KIND_KEY: WATCH_KIND,
            METADATA_MUTED_KEY: bool(spec.muted),
            METADATA_CLIENT_COUPLED_KEY: bool(spec.client_coupled),
        }
        self._task_store.save(task)

        # Register event trigger with the bridge for instant activation
        if task.trigger_type == "event":
            trigger_cfg = task.trigger_config if isinstance(task.trigger_config, dict) else {}
            event_trigger = EventTrigger.deserialize(trigger_cfg)
            self._event_bridge.register(task.task_id, event_trigger)

        self._emit_state(task)
        return self._to_view(task)

    def list_watches(self) -> List[WatchView]:
        """Return runtime views for all watches (newest first)."""
        watches = [t for t in self._task_store.load_all() if _is_watch(t)]
        watches.sort(key=lambda t: t.created_at, reverse=True)
        return [self._to_view(t) for t in watches]

    def get_watch(self, watch_id: str) -> Optional[WatchView]:
        """Return a single watch view, or None when absent/not a watch."""
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return None
        return self._to_view(task)

    def has_active_watches(self) -> bool:
        """Return True when a standalone watch is armed/watching (keep-alive signal).

        Client-coupled watches (e.g. session analysis) are excluded: they only
        matter while an interactive client is present and must not, by themselves,
        keep the daemon alive across idle periods.
        """
        for task in self._task_store.load_all():
            if not _is_watch(task) or task.state not in _ACTIVE_STATES:
                continue
            meta = task.metadata if isinstance(task.metadata, dict) else {}
            if meta.get(METADATA_CLIENT_COUPLED_KEY):
                continue
            return True
        return False

    def sweep_client_coupled_watches(self) -> int:
        """Delete leftover client-coupled watches (e.g. session analysis).

        Client-coupled watches only make sense while an interactive client owns
        them. On a fresh daemon lifetime none can, so any persisted ones are
        stale and must not surface as active monitors (status bar, keep-alive,
        ``/board list``). Standalone watches are intentionally durable and are
        left untouched. Returns the number of watches removed.
        """
        removed = 0
        for task in self._task_store.load_all():
            if not _is_watch(task):
                continue
            meta = task.metadata if isinstance(task.metadata, dict) else {}
            if not meta.get(METADATA_CLIENT_COUPLED_KEY):
                continue
            self._task_store.delete(task.task_id)
            self._finding_store.delete_for_watch(task.task_id)
            removed += 1
        return removed

    def pause_watch(self, watch_id: str) -> Optional[WatchView]:
        """Suspend a watch so it stops firing until resumed."""
        self.delivery.drop_watch(watch_id)
        return self._transition(watch_id, TaskState.SUSPENDED.value)

    def resume_watch(self, watch_id: str) -> Optional[WatchView]:
        """Re-arm a suspended watch."""
        return self._transition(watch_id, TaskState.ARMED.value)

    def stop_watch(self, watch_id: str) -> Optional[WatchView]:
        """Terminally stop a watch (kept for history)."""
        self._event_bridge.unregister(watch_id)
        self.delivery.drop_watch(watch_id)
        return self._transition(watch_id, TaskState.DONE.value)

    def set_muted(self, watch_id: str, muted: bool) -> Optional[WatchView]:
        """Toggle whether a watch's findings are pushed to view clients."""
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return None
        meta = dict(task.metadata) if isinstance(task.metadata, dict) else {}
        meta[METADATA_MUTED_KEY] = bool(muted)
        if muted:
            self.delivery.drop_watch(watch_id)
        meta.setdefault(METADATA_KIND_KEY, WATCH_KIND)
        task.metadata = meta
        self._task_store.save(task)
        self._emit_state(task)
        return self._to_view(task)

    # ── Findings ───────────────────────────────────────────────────────────

    def list_findings(
        self,
        *,
        watch_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        since: Optional[float] = None,
    ) -> List[Finding]:
        """Return persisted findings newest-first with optional filters."""
        return self._finding_store.list(
            watch_id=watch_id, limit=limit, offset=offset, since=since
        )

    async def run_watch_once(self, watch_id: str, *, force: bool = False) -> dict:
        """Run one observation cycle immediately (manual refresh/trigger).

        Bypasses the tick timer while reusing the same producer -> persist ->
        push path, so a user-triggered refresh is identical to a scheduled one.
        ``force=True`` signals producers to re-analyze even without new input.

        Run bookkeeping is applied here because it is the scheduler that normally
        does it, and this path deliberately skips the scheduler. Without it a manual
        refresh left no trace at all: findings dedup on content, so a cycle that
        confirmed "nothing changed" wrote nothing and advanced nothing, and the board
        could not tell a refresh that ran from one that never happened -- which is
        precisely what a reader presses refresh to find out.
        """
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return {"ok": False, "error": f"watch not found: {watch_id}"}
        params = dict(task.parameters) if isinstance(task.parameters, dict) else {}
        params.setdefault("watch_id", task.task_id)
        if force:
            params["_force"] = True
        result = await self._executor.execute(task.skill_name, params)
        # After the cycle, not before: ``last_run_at`` means "a cycle completed",
        # and a producer that raised did not complete one.
        if isinstance(result, dict) and result.get("ok"):
            try:
                self._task_store.increment_run_count(task.task_id)
            except Exception:  # noqa: BLE001 - bookkeeping must not fail the refresh
                logger.debug("monitor: run bookkeeping failed for %s", watch_id, exc_info=True)
        return result

    def schedule_watch_once(self, watch_id: str, *, force: bool = False) -> None:
        """Fire one observation cycle in the background (non-blocking).

        Producers may be slow (e.g. an LLM-backed session analysis). Callers
        that only need to *trigger* a refresh -- such as opening a live board --
        must not block on that latency, so the cycle runs as a tracked task and
        its finding reaches clients over the normal push path when it completes.
        """
        task = asyncio.create_task(self._safe_run_once(watch_id, force=force))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _safe_run_once(self, watch_id: str, *, force: bool) -> None:
        try:
            await self.run_watch_once(watch_id, force=force)
        except Exception:  # noqa: BLE001 - background refresh must never crash the loop
            logger.debug("monitor: background run_watch_once failed for %s", watch_id, exc_info=True)

    # ── Internal ───────────────────────────────────────────────────────────

    def _mark_event_due(self, watch_id: str, at: float) -> None:
        task = asyncio.create_task(self._persist_event_due(watch_id, at))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _persist_event_due(self, watch_id: str, at: float) -> None:
        try:
            await self._executor._db(lambda: self._task_store.advance_next_due(watch_id, at))
            self._scheduler.wake()
        except Exception:  # noqa: BLE001 - an event subscriber must never block acquisition
            logger.warning("monitor: event due time could not be persisted", exc_info=True)

    def _transition(self, watch_id: str, state: str) -> Optional[WatchView]:
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return None
        self._task_store.update_state(watch_id, state)
        task.state = state
        self._emit_state(task)
        return self._to_view(task)

    def _to_view(self, task: ArmedTask) -> WatchView:
        meta = task.metadata if isinstance(task.metadata, dict) else {}
        params = task.parameters if isinstance(task.parameters, dict) else {}
        cfg = task.trigger_config if isinstance(task.trigger_config, dict) else {}
        return WatchView(
            watch_id=task.task_id,
            name=str(params.get("name") or task.task_id[:8]),
            domain=str(params.get("domain") or task.skill_name),
            trigger=_format_trigger(task),
            state=task.state,
            muted=bool(meta.get(METADATA_MUTED_KEY, False)),
            run_count=task.run_count,
            next_due_at=task.next_due_at,
            last_run_at=task.last_run_at,
            finding_count=self._finding_store.count(watch_id=task.task_id),
            client_coupled=bool(meta.get(METADATA_CLIENT_COUPLED_KEY, False)),
            # Only an interval trigger has one. Left at 0.0 for event, cron and
            # condition watches, which the board reads as "no cadence to judge
            # against" and so withholds a staleness verdict rather than inventing
            # an interval for a watch that does not have one.
            interval_seconds=(
                float(cfg.get("interval_seconds", 0) or 0)
                if task.trigger_type == "interval"
                else 0.0
            ),
        )

    def _emit_state(self, task: ArmedTask) -> None:
        if self._emit is None:
            return
        try:
            self._emit(EVENT_WATCH_STATE, self._to_view(task).to_dict())
        except Exception:  # noqa: BLE001 - state notification is best-effort
            logger.debug("monitor: watch.state emit failed", exc_info=True)


__all__ = ["MonitorManager", "EmitFn"]

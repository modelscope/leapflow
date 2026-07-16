"""Watch lifecycle orchestration for the monitoring subsystem.

``MonitorManager`` wires the domain-neutral contract to the existing scheduler:
watches are ``ArmedTask`` rows (``kind=watch``); each due tick runs the matching
producer, persists findings, and pushes qualifying ones to view clients through
an injected ``emit`` callback (the daemon NotificationBus). It owns no domain
logic and no transport -- both are injected.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Callable, List, Optional

from leapflow.monitor.finding_store import FindingStore
from leapflow.monitor.producers import ProducerRegistry
from leapflow.monitor.types import (
    EVENT_ERROR,
    EVENT_FINDING,
    EVENT_WATCH_STATE,
    METADATA_CLIENT_COUPLED_KEY,
    METADATA_KIND_KEY,
    METADATA_MUTED_KEY,
    WATCH_KIND,
    Finding,
    ProducerContext,
    WatchSpec,
    WatchView,
)
from leapflow.scheduler.coordinator import TaskCoordinator
from leapflow.scheduler.local_scheduler import LocalScheduler
from leapflow.scheduler.store import TaskStore
from leapflow.scheduler.types import ArmedTask, TaskState

logger = logging.getLogger(__name__)

# Emit signature: (event_type, payload_dict) -> None. Injected by the host.
EmitFn = Callable[[str, dict], None]

_ACTIVE_STATES = frozenset({TaskState.ARMED.value, TaskState.WATCHING.value})


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
    ) -> None:
        self._tasks = task_store
        self._findings = finding_store
        self._producers = producers
        self._emit = emit
        self._services = services

    async def execute(self, skill_name: str, parameters: dict) -> dict:
        """Run one observation cycle for a watch (skill_name == domain)."""
        force = bool(parameters.get("_force", False))
        spec = WatchSpec.from_params(parameters)
        task = self._tasks.load(spec.watch_id) if spec.watch_id else None
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

        threshold = spec.push_threshold()
        persisted = 0
        emitted = 0
        for finding in findings:
            if not finding.watch_id:
                finding = replace(finding, watch_id=spec.watch_id)
            if finding.dedup_key and self._findings.exists_dedup(finding.watch_id, finding.dedup_key):
                continue
            self._findings.save(finding)
            persisted += 1
            if not muted and finding.severity.rank >= threshold.rank:
                self._push(EVENT_FINDING, finding.to_dict())
                emitted += 1
        return {"ok": True, "findings": persisted, "emitted": emitted}

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
    ) -> None:
        self._task_store = TaskStore(holder)
        self._finding_store = FindingStore(holder)
        self.producers = producers or ProducerRegistry()
        self._emit = emit
        self._executor = _MonitorExecutor(
            task_store=self._task_store,
            finding_store=self._finding_store,
            producers=self.producers,
            emit=emit,
            services=services,
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
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background tick loop (idempotent)."""
        if self._started:
            return
        await self._scheduler.start()
        self._started = True

    async def stop(self) -> None:
        """Stop the background tick loop (idempotent)."""
        if not self._started:
            return
        await self._scheduler.stop()
        self._started = False

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

    def pause_watch(self, watch_id: str) -> Optional[WatchView]:
        """Suspend a watch so it stops firing until resumed."""
        return self._transition(watch_id, TaskState.SUSPENDED.value)

    def resume_watch(self, watch_id: str) -> Optional[WatchView]:
        """Re-arm a suspended watch."""
        return self._transition(watch_id, TaskState.ARMED.value)

    def stop_watch(self, watch_id: str) -> Optional[WatchView]:
        """Terminally stop a watch (kept for history)."""
        return self._transition(watch_id, TaskState.DONE.value)

    def set_muted(self, watch_id: str, muted: bool) -> Optional[WatchView]:
        """Toggle whether a watch's findings are pushed to view clients."""
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return None
        meta = dict(task.metadata) if isinstance(task.metadata, dict) else {}
        meta[METADATA_MUTED_KEY] = bool(muted)
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
        """
        task = self._task_store.load(watch_id)
        if task is None or not _is_watch(task):
            return {"ok": False, "error": f"watch not found: {watch_id}"}
        params = dict(task.parameters) if isinstance(task.parameters, dict) else {}
        params.setdefault("watch_id", task.task_id)
        if force:
            params["_force"] = True
        return await self._executor.execute(task.skill_name, params)

    # ── Internal ───────────────────────────────────────────────────────────

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
        )

    def _emit_state(self, task: ArmedTask) -> None:
        if self._emit is None:
            return
        try:
            self._emit(EVENT_WATCH_STATE, self._to_view(task).to_dict())
        except Exception:  # noqa: BLE001 - state notification is best-effort
            logger.debug("monitor: watch.state emit failed", exc_info=True)


__all__ = ["MonitorManager", "EmitFn"]

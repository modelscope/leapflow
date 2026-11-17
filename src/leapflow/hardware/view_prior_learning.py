# Copyright (c) Alibaba, Inc. and its affiliates.
"""Derived survey signals through the shared privacy/noise pipeline, never raw views.

The host observes terminal skill reports; only the EventBus subscriber can persist
trials. Hints reorder proposals, and the node independently checks their live scope.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter, OrderedDict
from typing import Any, Awaitable, Callable, Mapping

from leapflow.domain.events import SystemEvent
from leapflow.hardware.stream import HardwareEvent, HardwareEventClass
from leapflow.hardware.view_priors import ViewPriorSettings
from leapflow.monitor.signal_noise import SignalNoiseGate
from leapflow.storage.view_prior_store import ViewPriorStore

logger = logging.getLogger(__name__)
OBSERVATION_TRIAL = "observation_trial"


class ViewPriorLearning:
    """Profile-owned bounded learner with serialized, thread-local DuckDB access."""

    def __init__(self, store: ViewPriorStore, bus: Any, publish: Callable[[Any], Any],
                 run_db: Callable[[Callable[[], Any]], Awaitable[Any]],
                 noise: SignalNoiseGate | None = None) -> None:
        self.settings: ViewPriorSettings = store.settings
        self._store, self._bus, self._publish, self._run_db = store, bus, publish, run_db
        self._noise = noise or SignalNoiseGate()
        self._scopes: OrderedDict[str, str] = OrderedDict()
        self._pending: set[asyncio.Task[Any]] = set()
        self._closed = False
        self.stats: Counter[str] = Counter()
        self._callback = self._receive
        bus.subscribe(self._callback)

    def expect_scope(self, workcell: str, scope: str) -> None:
        """Accept configuration provenance only from a verified node plan."""
        if not workcell or not _scope(scope):
            return
        self._scopes[workcell] = scope
        self._scopes.move_to_end(workcell)
        while len(self._scopes) > self.settings.max_events:
            self._scopes.popitem(last=False)

    async def hints(self, workcell: str) -> Mapping[str, Any]:
        if self._closed or not self.settings.enabled:
            return {}
        try:
            return await asyncio.wait_for(
                self._run_db(lambda: self._store.hints(workcell, time.time())), self.settings.lookup_timeout_s)
        except Exception:  # Optional telemetry must never fail a physical task.
            logger.warning("View prior lookup failed; using unlearned ordering", exc_info=True)
            return {}

    def record_run(self, node_id: str, run: Mapping[str, Any]) -> None:
        """Project a terminal report into a redacted signal; never write a store here."""
        if self._closed or run.get("skill_id") != "perceive.survey" or not run.get("outcome"):
            return
        try:
            value = _project(node_id, run)
            self._publish_value(value)
        except (TypeError, ValueError, KeyError):
            self.stats["malformed_report"] += 1
            logger.debug("Rejected malformed survey report", exc_info=True)
        except Exception:
            logger.warning("Survey signal publication failed", exc_info=True)

    def record_decision(self, node_id: str, workcell: str, scope: str, phase: str,
                        outcome: Mapping[str, Any], decision_id: str) -> None:
        """Observe a real planning/governance refusal, without inventing an executed run."""
        if self._closed or not _scope(scope) or phase not in {'planning', 'approval', 'admission'}:
            return
        self._publish_value({
            'event_id': f'{node_id}:decision:{decision_id}:{phase}', 'workcell': f'{node_id}/{workcell}',
            'scope': scope, 'observed_at': time.time(), 'trials': [],
            'status': 'rejected', 'failure_code': str(outcome.get('failure_code', '')), 'moves': 0})

    def _publish_value(self, value: Mapping[str, Any]) -> None:
        try:
            self._publish(HardwareEvent(OBSERVATION_TRIAL, value['workcell'], 'survey',
                'observation', 'Bounded survey terminal evidence', value=value,
                observed_at=value['observed_at'], event_class=HardwareEventClass.RECORD))
        except Exception:
            logger.warning('Survey signal publication failed', exc_info=True)

    def _receive(self, event: SystemEvent) -> None:
        if event.event_type != f"hw.{OBSERVATION_TRIAL}" or self._closed or not self.settings.enabled:
            return
        try:
            value = _validate(event, self.settings, time.time())
            if self._scopes.get(value["workcell"]) != value["scope"]:
                self.stats["configuration_mismatch"] += 1
                if any(t['reward'] is not None for t in value['trials']):
                    return
            if not self._noise.should_pass(event):
                self.stats["noise"] += 1
                return
            if len(self._pending) >= self.settings.max_pending:
                self.stats["backpressure"] += 1
                return
            task = asyncio.create_task(self._accept(value), name="view-prior-store")
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        except (TypeError, ValueError, KeyError):
            self.stats["invalid_signal"] += 1

    async def _accept(self, value: Mapping[str, Any]) -> None:
        try:
            # A reconfiguration between enqueue and DB dispatch invalidates evidence.
            if (self._scopes.get(value["workcell"]) != value["scope"]
                    and any(t['reward'] is not None for t in value['trials'])):
                self.stats["configuration_mismatch"] += 1
                return
            def persist() -> bool | None:
                # The serialized executor may have queued this behind another query.
                if (self._scopes.get(value["workcell"]) != value["scope"]
                        and any(t['reward'] is not None for t in value['trials'])):
                    return None
                return self._store.accept(value, time.time())
            accepted = await self._run_db(persist)
            self.stats["configuration_mismatch" if accepted is None else
                       "accepted" if accepted else "duplicate"] += 1
        except Exception:
            self.stats["storage_error"] += 1
            logger.warning("View prior persistence failed", exc_info=True)

    async def close(self) -> None:
        """Unsubscribe then drain before the profile executor/connection shuts down."""
        self._closed = True
        self._bus.unsubscribe(self._callback)
        if self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)


def _scope(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _support(report: Mapping[str, Any]) -> float:
    views = report.get("views", ())
    return max((float(v["support"]) for v in views if isinstance(v, Mapping)
                and _finite(v.get("support")) and 0 <= v["support"] <= 1), default=0.)


def _brief(report: Mapping[str, Any]) -> dict[str, Any]:
    return {'support': _support(report),
            'sufficient': report.get('sufficient', report.get('verdict') == 'sufficient') is True,
            'extent_is_lower_bound': report.get('extent_is_lower_bound', True) is not False,
            'conflict': report.get('conflict', False) is True}


def _project(node_id: str, run: Mapping[str, Any]) -> dict[str, Any]:
    outcome = run["outcome"]
    recorded = outcome.get("evidence", {}).get("survey", {})
    result = {**recorded, **outcome.get("result", {})}
    scope = result.get("observation_scope", result.get("scope"))
    if not _scope(scope):
        # Refusal/cancellation can precede any capture; preserve a nonlearning record
        # only when the runtime attached the approved scope to its terminal report.
        raise ValueError("missing approved observation scope")
    workcell, run_id = run["workcell"], run["run_id"]
    if not all(isinstance(v, str) and 0 < len(v) <= 256 for v in (node_id, workcell, run_id)):
        raise ValueError("invalid run provenance")
    observed = run["finished_at"]
    if not _finite(observed):
        raise ValueError("missing terminal observation time")
    trials = []
    reported = result.get("trials", ())
    if not isinstance(reported, (list, tuple)) or len(reported) > 65:
        raise ValueError("unbounded terminal report")
    for trial in reported:
        before, after = trial.get("before", {}), trial.get("after", {})
        gain = _support(after) - _support(before)
        if after.get("sufficient", after.get("verdict") == "sufficient") is True:
            gain = max(gain, 1. - float(before.get("sufficient", before.get("verdict") == "sufficient") is True))
        trials.append({"feature": trial["feature"], "gain": gain,
                       "duration_s": trial["duration_s"], 'before': _brief(before), 'after': _brief(after),
                       "usable": outcome.get("ok") is True and not trial.get("failure_code")})
    for refused in result.get('refused', ()):
        if isinstance(refused, Mapping) and refused.get('feature') and len(trials) < 65:
            trials.append({'feature': refused['feature'], 'gain': 0., 'duration_s': 0.,
                           'before': _brief({}), 'after': _brief({}), 'usable': False})
    return {"event_id": f"{node_id}:{run_id}", "workcell": f"{node_id}/{workcell}",
            "scope": scope, "observed_at": observed, "trials": trials,
            "status": str(outcome.get("status", "")),
            "failure_code": str(outcome.get("failure_code", "")),
            "moves": result.get("moves", 0)}


def _validate(event: SystemEvent, settings: ViewPriorSettings, now: float) -> dict[str, Any]:
    payload = event.payload
    if payload.get("event_class") != HardwareEventClass.RECORD:
        raise ValueError("not record-only")
    value = payload["value"]
    if not isinstance(value, Mapping) or set(value) != {"event_id", "workcell", "scope", "observed_at", "trials", "status", "failure_code", "moves"}:
        raise ValueError("invalid or sensitive signal fields")
    if not _scope(value["scope"]) or any(not isinstance(value[k], str) or not 0 < len(value[k]) <= 513
                                         for k in ("event_id", "workcell")):
        raise ValueError("invalid provenance")
    if payload.get("device_id") != value["workcell"]:
        raise ValueError("source mismatch")
    observed = value["observed_at"]
    if not _finite(observed) or not 0 <= now-observed <= settings.max_age_s:
        raise ValueError("stale or future trial")
    if (type(value['moves']) is not int or value['moves'] < 0
            or any(not isinstance(value[k], str) or len(value[k]) > 128 for k in ('status', 'failure_code'))):
        raise ValueError('invalid terminal verdict')
    trials = value["trials"]
    if not isinstance(trials, list) or len(trials) > 65:
        raise ValueError("unbounded trials")
    filtered = []
    for trial in trials:
        if not isinstance(trial, Mapping) or set(trial) != {"feature", "gain", "duration_s", "usable", "before", "after"}:
            raise ValueError("invalid trial")
        if not isinstance(trial["feature"], str) or not 0 < len(trial["feature"]) <= 128:
            raise ValueError("invalid feature")
        if not _finite(trial["gain"]) or not -1 <= trial["gain"] <= 1:
            raise ValueError("invalid gain")
        duration = trial["duration_s"]
        if not _finite(duration) or duration < 0 or type(trial["usable"]) is not bool:
            raise ValueError("invalid cost or verdict")
        reward = trial["gain"] / (1 + duration/settings.duration_scale_s) if trial["usable"] else None
        for key in ('before', 'after'):
            brief = trial[key]
            if (not isinstance(brief, Mapping)
                    or set(brief) != {'support', 'sufficient', 'extent_is_lower_bound', 'conflict'}
                    or not _finite(brief['support']) or not 0 <= brief['support'] <= 1
                    or any(type(brief[k]) is not bool for k in ('sufficient', 'extent_is_lower_bound', 'conflict'))):
                raise ValueError('invalid observation summary')
        filtered.append({**trial, "reward": reward})
    return {**value, "trials": filtered}

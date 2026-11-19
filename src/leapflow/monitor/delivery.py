# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded, domain-neutral presentation of already persisted observations.

Cooldowns describe enqueue attempts, never confirmed user receipt. Pending state
is intentionally not replayed after restart: producers must supply current evidence.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from leapflow.monitor.types import EVENT_FINDING, EVENT_UPDATED, Finding, Severity

logger = logging.getLogger(__name__)
GLOBAL_KEY = "delivery:global"
OVERFLOW_KEY = "delivery:overflow"
SATURATED_KEY = "delivery:saturated"


@runtime_checkable
class DeliveryStore(Protocol):
    """Durable enqueue timestamps, independent of producer episode checkpoints."""

    def delivery_timestamps(self, keys: Sequence[str]) -> dict[str, float]: ...
    def stage_delivery_overflow(self, entries: list[dict[str, Any]], capacity: int) -> int: ...
    def withdraw_delivery_overflow(self, keys: Sequence[str], *, force: bool = False) -> None: ...
    def drop_delivery_watch(self, watch_ids: list[str]) -> None: ...
    def delivery_overflow_entries(self) -> list[dict[str, Any]]: ...
    def reset_delivery_overflow(self) -> None: ...
    def delivery_overflow_keys(self) -> list[str]: ...

    def record_delivery(self, keys: Sequence[str], now: float, retain_after: float,
                        capacity: int) -> bool:
        """Record one attempt atomically; refuse if unexpired state cannot fit."""
        ...


@runtime_checkable
class FindingPresentation(Protocol):
    """Injected presentation lifecycle; consumes only typed finding metadata."""

    async def offer(self, findings: Sequence[Finding], threshold: Severity,
                    muted: bool = False) -> int: ...
    def start(self) -> None: ...
    async def stop(self) -> None: ...
    def drop_watch(self, watch_id: str) -> None: ...
    def configure(self, raw: Any) -> bool: ...
    @property
    def stats(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DeliveryConfig:
    minimum_severity: str = "critical"
    global_interval_s: float = 5.0
    repeat_interval_s: float = 60.0
    max_pending: int = 128
    max_items: int = 5
    max_bytes: int = 8192
    updated_interval_s: float = 1.0
    max_state: int = 4096

    def __post_init__(self) -> None:
        if self.minimum_severity not in {s.value for s in Severity}:
            raise ValueError("monitor.delivery.minimum_severity is unknown")
        for name in ("global_interval_s", "repeat_interval_s", "updated_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"monitor.delivery.{name} must be positive and finite")
        for name in ("max_pending", "max_items", "max_bytes", "max_state"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"monitor.delivery.{name} must be a positive integer")
        if self.max_bytes < 1024 or self.max_state < self.max_pending + 3:
            raise ValueError("monitor.delivery byte/state limits cannot fit a bounded notice")
        if self.updated_interval_s < 1:
            raise ValueError("monitor.updated may not be emitted more than once per second")

    @classmethod
    def parse(cls, raw: Any) -> DeliveryConfig:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("monitor.delivery must be valid JSON") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping) or set(raw) - cls.__dataclass_fields__.keys():
            raise ValueError("monitor.delivery contains unknown policy keys")
        return cls(**raw)


def _key(domain: str, identity: str) -> str:
    return hashlib.sha256(json.dumps([domain, identity], ensure_ascii=False).encode()).hexdigest()


def _text(value: str, size: int) -> str:
    return value.encode("utf-8")[:size].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class DeliveryCandidate:
    """Bounded, typed metadata for live watch and domain evidence revalidation."""

    key: str
    domain: str
    incident_key: str
    watch_ids: tuple[str, ...]
    severity: str
    actionable: bool


@runtime_checkable
class DeliveryValidator(Protocol):
    """Resolve current evidence and consenting owners immediately before enqueue."""

    async def __call__(self, candidates: Sequence[DeliveryCandidate], now: float) -> Mapping[str, Sequence[str]]: ...


@dataclass
class _Pending:
    key: str
    watch_id: str
    finding_id: str
    title: str
    summary: str
    severity: str
    actionable: bool
    domain: str = ""
    incident_key: str = ""
    watch_ids: tuple[str, ...] = ()

    def candidate(self) -> DeliveryCandidate:
        return DeliveryCandidate(self.key, self.domain, self.incident_key,
                                 tuple(self.watch_ids or (self.watch_id,)), self.severity, self.actionable)

    def retain_watches(self, watches: Sequence[str]) -> bool:
        self.watch_ids = tuple(watches)
        if not self.watch_ids:
            return False
        self.watch_id = self.watch_ids[0]
        return True

    @classmethod
    def from_finding(cls, key: str, finding: Finding) -> _Pending:
        return cls(key, _text(finding.watch_id, 128), _text(finding.finding_id, 128),
                   _text(finding.title, 256), _text(finding.summary, 512),
                   finding.severity.value, finding.delivery.actionable,
                   _text(finding.domain, 128), _text(finding.delivery.key, 512),
                   (_text(finding.watch_id, 128),))

    def to_dict(self) -> dict[str, str]:
        return {"finding_id": self.finding_id, "watch_id": self.watch_id,
                "title": self.title, "summary": self.summary, "severity": self.severity}


class BoundedFindingDelivery:
    """One global budget and bounded incident coalescing, with no urgent bypass."""

    def __init__(self, store: DeliveryStore, emit: Callable[[str, dict], Any] | None,
                 *, config: DeliveryConfig | None = None,
                 run_db: Callable[[Callable[[], Any]], Any] | None = None,
                 clock: Callable[[], float] = time.time,
                 validate: DeliveryValidator | None = None) -> None:
        self._store, self._emit, self._run_db, self._clock = store, emit, run_db, clock
        self._config = config or DeliveryConfig()
        self._validate = validate
        self._policy_revision = 0
        self._untracked_count = 0
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._overflow: OrderedDict[str, _Pending] = OrderedDict()
        self._overflow_count = 0
        self._dropped_watches: set[str] = set()
        self._reset_overflow = False
        self._counts: Counter[str] = Counter()
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._updated_task: asyncio.Task | None = None
        self._last_updated = float("-inf")
        self._closed = False
        self._initialized = False

    async def _db(self, fn: Callable[[], Any]) -> Any:
        return await self._run_db(fn) if self._run_db else await asyncio.to_thread(fn)

    @property
    def stats(self) -> dict[str, Any]:
        return {**{name: self._counts[name] for name in (
            "filtered", "coalesced", "cooldown_suppressed", "overflowed", "enqueued", "failed")},
            "pending": len(self._pending), "overflow_pending": self._overflow_count,
            "receipt_confirmed": False}

    def configure(self, raw: Any) -> bool:
        try:
            config = DeliveryConfig.parse(raw)
        except (ValueError, TypeError):
            logger.warning("monitor: invalid delivery reload; retaining last valid policy", exc_info=True)
            return False
        if Severity(config.minimum_severity).rank > Severity(self._config.minimum_severity).rank:
            self._discard_untracked()
        self._config = config
        self._policy_revision += 1
        # Apply shrinking bounds under the async admission lock, where moved
        # identities can be spooled before their in-memory details are discarded.
        self._ensure_flush()
        return True

    def start(self) -> None:
        """Resume lifecycle without resetting counters or durable cooldowns."""
        self._closed = False
        self._initialized = False

    def drop_watch(self, watch_id: str) -> None:
        self._policy_revision += 1
        self._discard_untracked()
        for queue in (self._pending, self._overflow):
            for key, item in list(queue.items()):
                watches = [owner for owner in item.candidate().watch_ids if owner != watch_id]
                if not item.retain_watches(watches):
                    del queue[key]
        self._dropped_watches.add(watch_id)
        if len(self._dropped_watches) > self._config.max_state:
            self._dropped_watches.clear()
            self._reset_overflow = True
        self._ensure_flush()

    def _discard_untracked(self) -> None:
        # Saturated identities retain complete evidence in findings, but cannot
        # prove current consent after a state change. Never page stale summaries.
        self._counts["filtered"] += self._untracked_count
        self._overflow_count = max(0, self._overflow_count - self._untracked_count)
        self._untracked_count = 0

    async def _clean_dropped_watches(self) -> None:
        watches = list(self._dropped_watches)
        if self._reset_overflow:
            await self._db(self._store.reset_delivery_overflow)
            self._reset_overflow = False
            self._overflow.clear()
            self._overflow_count = 0
        elif watches:
            await self._db(lambda: self._store.drop_delivery_watch(watches))
        self._dropped_watches.difference_update(watches)

    async def _stage(self, entries: list[dict[str, Any]]) -> None:
        before = set(await self._db(self._store.delivery_overflow_keys))
        attempted = before | {entry["key"] for entry in entries}
        count = await self._db(lambda: self._store.stage_delivery_overflow(entries, self._config.max_state))
        self._untracked_count = min(2**63 - 1, self._untracked_count + max(0, len(attempted) - count))
        self._overflow_count = count + self._untracked_count

    async def _apply_bounds(self) -> None:
        # Also prune an already durable spool when its configured bound shrinks.
        await self._stage([])
        moved = list(self._pending.values())[self._config.max_pending:]
        if moved:
            await self._stage([asdict(item) for item in moved])
            for item in moved:
                self._pending.pop(item.key, None)
                self._counts["overflowed"] += 1
        for item in self._pending.values():
            item.watch_ids = item.candidate().watch_ids[:self._config.max_state]
        while len(self._overflow) > self._config.max_items:
            self._overflow.popitem(last=False)

    def _withdraw(self, key: str) -> None:
        item = self._pending.get(key)
        if item is not None and not item.actionable:
            del self._pending[key]
        item = self._overflow.get(key)
        if item is not None and not item.actionable:
            del self._overflow[key]
            self._overflow_count = max(0, self._overflow_count - 1)

    def _add_overflow(self, key: str, item: _Pending) -> None:
        self._counts["overflowed"] += 1
        self._overflow_count += 1
        self._overflow[key] = item
        while len(self._overflow) > self._config.max_items:
            self._overflow.popitem(last=False)

    async def offer(self, findings: Sequence[Finding], threshold: Severity,
                    muted: bool = False) -> int:
        """Offer a persisted cycle; the first batch is flushed on the next loop turn."""
        if self._closed:
            return 0
        self._invalidate()
        async with self._lock:
            if not self._initialized:
                await self._db(self._store.reset_delivery_overflow)
                self._initialized = True
            await self._clean_dropped_watches()
            await self._apply_bounds()
            overflow_entries, withdrawals = [], []
            spool_items = {entry["key"]: _Pending(**entry)
                           for entry in await self._db(self._store.delivery_overflow_entries)}
            spooled = set(spool_items)
            floor = max(threshold.rank, Severity(self._config.minimum_severity).rank)
            identities = [_key(f.domain, f.delivery.key or f.dedup_key or f.finding_id) for f in findings]
            stamps = await self._db(lambda: self._store.delivery_timestamps([SATURATED_KEY, *identities]))
            for finding, key in zip(findings, identities):
                delivery = finding.delivery
                for withdrawn in delivery.withdrawn_keys:
                    withdrawn_key = _key(finding.domain, withdrawn)
                    self._withdraw(withdrawn_key)
                    withdrawals.append(withdrawn_key)
                eligible = delivery.actionable or (delivery.active and delivery.current and delivery.phase != "recovery")
                if not eligible or finding.severity.rank < floor:
                    self._withdraw(key)
                    withdrawals.append(key)
                if muted or finding.severity.rank < floor or not eligible:
                    self._counts["filtered"] += 1
                    continue
                last = stamps.get(key)
                # A full durable ledger uses the reserved aggregate cooldown for
                # untracked identities, never silently resets a recent incident.
                overflow_last = stamps.get(SATURATED_KEY)
                if ((last is not None and self._clock() - last < self._config.repeat_interval_s)
                        or (last is None and overflow_last is not None
                            and self._clock() - overflow_last < self._config.repeat_interval_s)):
                    self._counts["cooldown_suppressed"] += 1
                    continue
                item = _Pending.from_finding(key, finding)
                prior = self._pending.get(key) or spool_items.get(key)
                if prior is not None:
                    item.watch_ids = tuple(dict.fromkeys((*item.watch_ids, *prior.candidate().watch_ids)))[:self._config.max_state]
                    item.actionable = item.actionable or prior.actionable
                spool_items[key] = item
                if key in self._pending:
                    self._counts["coalesced"] += 1
                    self._pending[key] = item
                elif key not in spooled and len(self._pending) < self._config.max_pending:
                    self._pending[key] = item
                else:
                    if key in spooled:
                        self._counts["coalesced"] += 1
                    else:
                        self._add_overflow(key, item)
                        spooled.add(key)
                    overflow_entries.append(asdict(item))
            if overflow_entries:
                await self._stage(overflow_entries)
            if withdrawals:
                self._discard_untracked()
                await self._db(lambda: self._store.withdraw_delivery_overflow(withdrawals))
            self._ensure_flush()
        return 0  # enqueue counts are reported by stats, not observation counts

    def _ensure_flush(self) -> None:
        if (not self._closed and self._flush_task is None
                and (self._pending or self._overflow_count or self._dropped_watches or self._reset_overflow)
                and self._emit is not None):
            self._flush_task = asyncio.create_task(self._flush_loop(), name="monitor-delivery")

    def _invalidate(self) -> None:
        if self._updated_task is None and self._emit is not None:
            self._updated_task = asyncio.create_task(self._updated(), name="monitor-updated")

    async def _updated(self) -> None:
        try:
            await asyncio.sleep(max(0, self._config.updated_interval_s - (self._clock() - self._last_updated)))
            self._emit(EVENT_UPDATED, {})
            self._last_updated = self._clock()
        except Exception:  # noqa: BLE001 - view freshness must never fail acquisition
            logger.debug("monitor.updated emit failed", exc_info=True)
        finally:
            self._updated_task = None

    async def _flush_loop(self) -> None:
        try:
            # Coalesce observations already queued in this event-loop turn.
            await asyncio.sleep(0)
            while not self._closed and (self._pending or self._overflow_count or self._dropped_watches or self._reset_overflow):
                delay = await self.flush()
                if delay > 0:
                    await asyncio.sleep(delay)
        except Exception:  # noqa: BLE001 - durable evidence survives presentation failure
            self._counts["failed"] += 1
            logger.warning("monitor: delivery failed; findings remain inspectable", exc_info=True)
        finally:
            self._flush_task = None

    async def _revalidate(self, entries: Sequence[_Pending], now: float) -> set[str]:
        if self._validate is None:
            return set()
        owners = await self._validate([item.candidate() for item in entries], now)
        removed = set()
        for item in entries:
            if not item.retain_watches(owners.get(item.key, ())):
                removed.add(item.key)
        self._counts["filtered"] += len(removed)
        return removed

    async def flush(self) -> float:
        """Flush one eligible aggregate, returning seconds until another slot."""
        async with self._lock:
            if self._closed or self._emit is None:
                return 0
            config, now = self._config, self._clock()
            revision = self._policy_revision
            await self._clean_dropped_watches()
            await self._apply_bounds()
            stamps = await self._db(lambda: self._store.delivery_timestamps(
                [GLOBAL_KEY, OVERFLOW_KEY, *self._pending]))
            if GLOBAL_KEY in stamps:
                delay = config.global_interval_s - (now - stamps[GLOBAL_KEY])
                if delay > 0:
                    return delay
            for key in list(self._pending):
                if Severity(self._pending[key].severity).rank < Severity(config.minimum_severity).rank:
                    del self._pending[key]
                    self._counts["filtered"] += 1
                    continue
                if key in stamps and now - stamps[key] < config.repeat_interval_s:
                    del self._pending[key]
                    self._counts["cooldown_suppressed"] += 1
            entries = await self._db(self._store.delivery_overflow_entries)
            spool_items = [_Pending(**entry) for entry in entries]
            removed = await self._revalidate([*self._pending.values(), *spool_items], now)
            for key in removed:
                self._pending.pop(key, None)
            if removed:
                self._discard_untracked()
                await self._db(lambda: self._store.withdraw_delivery_overflow(list(removed), force=True))
            entries = [asdict(item) for item in spool_items if item.key not in removed]
            overflow_stamps = await self._db(lambda: self._store.delivery_timestamps([entry["key"] for entry in entries]))
            cooling = [entry["key"] for entry in entries if entry["key"] in overflow_stamps
                       and now - overflow_stamps[entry["key"]] < config.repeat_interval_s]
            if cooling:
                await self._db(lambda: self._store.withdraw_delivery_overflow(cooling, force=True))
                self._counts["cooldown_suppressed"] += len(cooling)
                entries = [entry for entry in entries if entry["key"] not in cooling]
            below_floor = [entry["key"] for entry in entries
                           if Severity(entry["severity"]).rank < Severity(config.minimum_severity).rank]
            if below_floor:
                # A raised global floor removes actionable notices too.
                for key in below_floor:
                    self._overflow.pop(key, None)
                await self._db(lambda: self._store.withdraw_delivery_overflow(below_floor, force=True))
                entries = [entry for entry in entries if entry["key"] not in below_floor]
            untracked = self._untracked_count
            self._overflow_count = len(entries) + untracked
            self._overflow = OrderedDict((entry["key"], _Pending(**entry))
                                         for entry in entries[:config.max_items])
            # Without bounded identity metadata, live domain revalidation is not
            # possible. Saturation stays queryable but cannot trigger a notice alone.
            overflow_ready = bool(entries or (untracked and self._validate is None))
            if not self._pending and not overflow_ready:
                self._counts["cooldown_suppressed"] += self._overflow_count
                self._overflow.clear()
                self._overflow_count = 0
                self._untracked_count = 0
                await self._db(self._store.reset_delivery_overflow)
                return 0
            items = list(self._pending.values())
            keys = list(self._pending) + ([OVERFLOW_KEY] if overflow_ready else [])
            if overflow_ready and untracked:
                keys.append(SATURATED_KEY)
            recorded = await self._db(lambda: self._store.record_delivery(
                keys, now, now - max(config.global_interval_s, config.repeat_interval_s), config.max_state))
            if not recorded:
                moved = [asdict(item) for item in self._pending.values()]
                for key, item in list(self._pending.items()):
                    self._add_overflow(key, item)
                await self._stage(moved)
                self._pending.clear()
                return 0  # retry once as a reserved aggregate, without waiting another slot
            self._pending.clear()
            try:
                await self._emit_recorded(items, [_Pending(**entry) for entry in entries], untracked, revision)
            finally:
                self._overflow.clear()
                self._overflow_count = 0
                self._untracked_count = 0
            return self._config.global_interval_s

    async def _emit_recorded(self, items: list[_Pending], overflow: list[_Pending],
                             untracked: int, revision: int) -> None:
        """Revalidate after durable I/O; an attempt is not a promise to present."""
        if self._closed:
            return
        removed = await self._revalidate([*items, *overflow], self._clock())
        floor = Severity(self._config.minimum_severity).rank
        def admitted(item: _Pending) -> bool:
            return (item.key not in removed and Severity(item.severity).rank >= floor
                    and item.retain_watches([owner for owner in item.candidate().watch_ids
                                            if owner not in self._dropped_watches]))
        items = [item for item in items if admitted(item)]
        overflow = [item for item in overflow if admitted(item)]
        if revision != self._policy_revision or removed:
            untracked = 0
        if self._closed or not (items or overflow or (untracked and self._validate is None)):
            return
        self._overflow = OrderedDict((item.key, item) for item in overflow[:self._config.max_items])
        self._untracked_count = untracked
        self._overflow_count = len(overflow) + untracked
        self._emit(EVENT_FINDING, self._payload(items, bool(overflow or untracked)))
        self._counts["enqueued"] += 1

    def _payload(self, items: Sequence[_Pending], overflow_ready: bool) -> dict[str, Any]:
        config = self._config
        refs = list(items) + (list(self._overflow.values()) if overflow_ready else [])
        count = len(items) + (self._overflow_count if overflow_ready else 0)
        severity = max((Severity(item.severity) for item in refs),
                       default=Severity(config.minimum_severity), key=lambda s: s.rank).value
        payload: dict[str, Any] = {
            "severity": severity, "title": f"{count} {severity} observation(s) require inspection",
            "summary": "Inspect current state and persisted evidence before continuing affected work.",
            "count": count, "overflow_count": self._overflow_count if overflow_ready else 0,
            "items": [item.to_dict() for item in refs[:config.max_items]],
            "overflow_references": [item.finding_id for item in self._overflow.values()] if overflow_ready else [],
            "inspection": "/board list", "delivery_phase": "enqueue",
        }
        if overflow_ready and self._untracked_count:
            payload["overflow_untracked_observations"] = self._untracked_count
        while len(json.dumps(payload, ensure_ascii=False).encode()) > config.max_bytes and payload["items"]:
            payload["items"].pop()
        while len(json.dumps(payload, ensure_ascii=False).encode()) > config.max_bytes and payload["overflow_references"]:
            payload["overflow_references"].pop()
        return payload

    async def stop(self) -> None:
        """Cancel and await owned tasks; never page pending history on shutdown."""
        self._closed = True
        tasks = [task for task in (self._flush_task, self._updated_task) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
        self._overflow.clear()
        self._overflow_count = 0
        self._dropped_watches.clear()
        self._untracked_count = 0

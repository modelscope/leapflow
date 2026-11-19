# Copyright (c) Alibaba, Inc. and its affiliates.
"""Deterministic presentation budgets with real DuckDB, never physical faults."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from leapflow.daemon.notifications import NotificationBus
from leapflow.monitor.delivery import (
    GLOBAL_KEY, OVERFLOW_KEY, BoundedFindingDelivery, DeliveryConfig, DeliveryStore,
    FindingPresentation,
)
from leapflow.monitor.finding_store import FindingStore
from leapflow.monitor.types import EVENT_FINDING, Finding, FindingDelivery, Severity


async def _inline(fn):
    return fn()


def _finding(key, severity=Severity.CRITICAL, **metadata):
    return Finding(watch_id="w1", domain="test", title=f"Incident {key}",
                   severity=severity, delivery=FindingDelivery(key=key, **metadata))


@pytest.fixture
def store(tmp_path):
    result = FindingStore(tmp_path / "findings.duckdb")
    yield result
    result.close()


def _delivery(store, config=None):
    now = [1000.0]
    events = []
    delivery = BoundedFindingDelivery(store, lambda kind, payload: events.append((kind, payload)),
                                     config=config, run_db=_inline, clock=lambda: now[0])
    assert isinstance(store, DeliveryStore)
    assert isinstance(delivery, FindingPresentation)
    return delivery, now, events


def _notices(events):
    return [payload for kind, payload in events if kind == EVENT_FINDING]


@pytest.mark.parametrize("raw", [
    {"global_interval_s": 0}, {"repeat_interval_s": float("nan")},
    {"max_pending": True}, {"max_bytes": 10}, {"updated_interval_s": 0.1},
    {"minimum_severity": "urgent"}, {"unknown": 3}, "not-json", [],
])
def test_policy_validation(raw):
    with pytest.raises((ValueError, TypeError)):
        DeliveryConfig.parse(raw)


async def test_global_floor_and_first_aggregate(store):
    delivery, now, events = _delivery(store)
    findings = [_finding(str(index), Severity.ALERT) for index in range(3000)]
    for finding in findings:
        store.save(finding)
    await delivery.offer(findings, Severity.INFO)
    await delivery.flush()
    assert _notices(events) == []
    assert store.count() == 3000
    assert delivery.stats["filtered"] == 3000
    await delivery.offer([_finding("a"), _finding("b")], Severity.NOTABLE)
    await delivery.flush()
    assert len(_notices(events)) == 1
    assert _notices(events)[0]["count"] == 2
    await delivery.offer([_finding("c")], Severity.NOTABLE)
    assert await delivery.flush() == 5
    now[0] += 4.99
    assert await delivery.flush() == pytest.approx(0.01)
    assert len(_notices(events)) == 1
    now[0] += 0.01
    await delivery.flush()
    assert len(_notices(events)) == 2
    await delivery.stop()


async def test_cross_watch_identity_and_durable_restart_cooldown(store):
    delivery, now, events = _delivery(store)
    first = _finding("stable")
    await delivery.offer([first, replace(first, watch_id="w2", finding_id="other")], Severity.NOTABLE)
    await delivery.flush()
    assert len(_notices(events)) == 1
    assert _notices(events)[0]["count"] == 1
    assert delivery.stats["coalesced"] == 1
    await delivery.stop()
    restarted, clock, emitted = _delivery(store)
    clock[0] = now[0] + 10
    await restarted.offer([replace(first, finding_id="new-episode", watch_id="new-watch")], Severity.INFO)
    await restarted.flush()
    assert not _notices(emitted)
    assert restarted.stats["cooldown_suppressed"] == 1
    clock[0] = now[0] + 60
    await restarted.offer([first], Severity.INFO)
    await restarted.flush()
    assert len(_notices(emitted)) == 1
    await restarted.stop()


async def test_history_recovery_unknown_and_actionable(store):
    delivery, now, events = _delivery(store)
    await delivery.offer([_finding("historical", current=False),
                          _finding("recovered", phase="recovery", active=False),
                          _finding("stopped", current=False, actionable=True)], Severity.INFO)
    await delivery.flush()
    assert _notices(events)[0]["count"] == 1
    assert _notices(events)[0]["items"][0]["title"] == "Incident stopped"
    await delivery.offer([_finding("pending")], Severity.INFO)
    await delivery.offer([_finding("state", Severity.INFO, active=False,
                                    withdrawn_keys=("pending",))], Severity.INFO)
    now[0] += 5
    await delivery.flush()
    assert len(_notices(events)) == 1
    await delivery.stop()


async def test_bounded_overflow_references_and_repeat(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=8, max_items=3, max_bytes=1024))
    batch = [replace(_finding(str(index)), title="界" * 10000, summary="large" * 10000)
             for index in range(1000)]
    for finding in batch:
        store.save(finding)
    await delivery.offer(batch, Severity.INFO)
    assert delivery.stats["pending"] == 8
    await delivery.flush()
    notice = _notices(events)[0]
    assert notice["count"] == 1000
    assert notice["overflow_count"] == 992
    assert len(notice["items"]) <= 3
    assert len(notice["overflow_references"]) <= 3
    assert len(json.dumps(notice, ensure_ascii=False).encode()) <= 1024
    assert all(store.list(limit=1, watch_id="w1"))
    await delivery.offer(batch, Severity.INFO)
    now[0] += 5
    await delivery.flush()
    assert len(_notices(events)) == 1
    assert delivery.stats["cooldown_suppressed"] == 1000
    assert store.delivery_timestamps([GLOBAL_KEY, OVERFLOW_KEY])
    await delivery.stop()


async def test_overflow_recovery_removes_pending(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1))
    await delivery.offer([_finding("first")], Severity.INFO)
    await delivery.flush()
    await delivery.offer([_finding("second"), _finding("third")], Severity.INFO)
    await delivery.offer([_finding("third", Severity.NOTABLE, active=False, phase="recovery")], Severity.INFO)
    now[0] += 5
    await delivery.flush()
    assert _notices(events)[1]["count"] == 1
    assert not store.delivery_overflow_keys()
    await delivery.stop()


async def test_reload_mutes_and_shutdown_preserve_budget(store):
    delivery, now, events = _delivery(store)
    await delivery.offer([_finding("first")], Severity.INFO)
    await delivery.flush()
    before = delivery.stats.copy()
    assert not delivery.configure({"global_interval_s": -1})
    assert delivery.configure({"max_pending": 4})
    assert delivery.stats == before
    await delivery.offer([_finding("muted")], Severity.INFO, muted=True)
    await delivery.offer([_finding("pending")], Severity.INFO)
    delivery.drop_watch("w1")
    now[0] += 5
    await delivery.flush()
    assert len(_notices(events)) == 1
    await delivery.stop()
    await delivery.offer([_finding("after-stop")], Severity.INFO)
    await asyncio.sleep(0)
    assert len(_notices(events)) == 1


async def test_failed_sink_keeps_durable_attempt_without_claiming_receipt(store):
    now = [1000.0]
    def fail(kind, payload):
        raise RuntimeError("subscriber unavailable")
    delivery = BoundedFindingDelivery(store, fail, run_db=_inline, clock=lambda: now[0])
    finding = _finding("failed")
    store.save(finding)
    await delivery.offer([finding], Severity.INFO)
    with pytest.raises(RuntimeError):
        await delivery.flush()
    assert delivery.stats["enqueued"] == 0
    assert not delivery.stats["receipt_confirmed"]
    assert store.count() == 1
    assert store.delivery_timestamps([GLOBAL_KEY])[GLOBAL_KEY] == 1000
    await delivery.stop()


def test_additive_finding_schema_and_critical_filter(store):
    finding = _finding("serialized", withdrawn_keys=("old",))
    store.save(finding)
    assert store.list(min_severity=Severity.CRITICAL)[0] == finding
    legacy = finding.to_dict()
    legacy.pop("delivery")
    assert Finding.from_dict(legacy).delivery == FindingDelivery()
    assert not FindingDelivery.from_dict({"active": "true", "actionable": "false"}).active
    assert not FindingDelivery.from_dict({"actionable": "false"}).actionable
    assert FindingDelivery.from_dict({"withdrawn_keys": "not-a-list"}).withdrawn_keys == ()


async def test_notification_priority_replacement_and_full_shutdown():
    bus = NotificationBus(maxsize=3)
    slow = bus.subscribe("slow")
    fast = bus.subscribe("fast")
    for index in range(1000):
        bus.emit_event("monitor.updated", sequence=index)
        await fast.get()
    assert slow.qsize() == 1
    assert bus.stats["coalesced"] == 999
    bus.emit_event("signal.stream", sample=1)
    bus.emit_event("ordinary.status", state="running")
    bus.emit_event("monitor.finding", severity="critical", title="Important")
    assert slow.qsize() == 3
    assert (await slow.get()).event_type == EVENT_FINDING
    assert bus.stats["overflowed"] >= 1
    bus.emit_event("control", state="a")
    bus.emit_event("control", state="b")
    await bus.shutdown()
    assert await slow.get() is None
    assert await fast.get() is None
    assert bus.subscriber_count == 0


async def test_replacing_subscriber_terminates_old_stream():
    bus = NotificationBus(maxsize=1)
    old = bus.subscribe("same")
    bus.emit_event("control")
    current = bus.subscribe("same")
    assert await old.get() is None
    assert current is not old
    bus.unsubscribe("same", old)
    assert bus.subscriber_count == 1
    bus.emit_event("control", state="replacement alive")
    assert (await current.get()).payload["state"] == "replacement alive"
    await bus.shutdown()


async def test_queue_join_waits_for_retained_work():
    bus = NotificationBus(maxsize=2)
    queue = bus.subscribe("join")
    bus.emit_event("monitor.updated", sequence=1)
    joined = asyncio.create_task(queue.join())
    await asyncio.sleep(0)
    bus.emit_event("monitor.updated", sequence=2)
    await asyncio.sleep(0)
    assert not joined.done()
    assert (await queue.get()).payload["sequence"] == 2
    queue.task_done()
    await joined
    await bus.shutdown()


@pytest.mark.parametrize("spooled", [False, True])
async def test_cross_watch_mute_preserves_other_owner(store, spooled):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=2))
    first = _finding("shared")
    if spooled:
        await delivery.offer([_finding("filler1"), _finding("filler2")], Severity.INFO)
    await delivery.offer([first, replace(first, watch_id="w2", finding_id="other")], Severity.INFO)
    delivery.drop_watch("w1")
    await delivery.flush()
    assert _notices(events)[0]["count"] == 1
    assert _notices(events)[0]["items"][0]["watch_id"] == "w2"
    await delivery.stop()


async def test_pending_domain_evidence_is_revalidated(store):
    live = {"pending", "initial"}
    async def validate(candidates, now):
        return {item.key: item.watch_ids for item in candidates
                if item.incident_key in live or item.actionable}
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1))
    delivery._validate = validate
    await delivery.offer([_finding("initial")], Severity.INFO)
    await delivery.flush()
    await delivery.offer([_finding("pending"), _finding("cleared"),
                          _finding("interrupted", actionable=True)], Severity.INFO)
    live.clear()
    now[0] += 5
    await delivery.flush()
    notice = _notices(events)[1]
    assert notice["count"] == 1
    assert notice["items"][0]["title"] == "Incident interrupted"
    await delivery.stop()


async def test_saturation_payload_and_counts_survive_retry(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1, max_state=8, max_bytes=1024))
    batch = [replace(_finding(str(i)), title="界" * 1000, summary="界" * 1000) for i in range(100)]
    await delivery.offer(batch, Severity.INFO)
    untracked = delivery._untracked_count
    assert untracked > 0
    await delivery.flush()
    notice = _notices(events)[0]
    assert notice["overflow_untracked_observations"] == untracked
    assert notice["count"] == 100
    assert len(json.dumps(notice, ensure_ascii=False).encode()) <= 1024
    await delivery.stop()


async def test_spooled_watch_mute_and_actionable_recovery(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1, max_items=1))
    await delivery.offer([_finding("initial")], Severity.INFO)
    await delivery.flush()
    await delivery.offer([_finding("kept"), *[_finding(str(i), actionable=True) for i in range(10)]], Severity.INFO)
    await delivery.offer([_finding(str(i), Severity.INFO, active=False, phase="recovery")
                          for i in range(10)], Severity.INFO)
    assert len(store.delivery_overflow_keys()) == 10
    delivery.drop_watch("w1")
    now[0] += 5
    await delivery.flush()
    assert len(_notices(events)) == 1
    assert not store.delivery_overflow_keys()
    await delivery.stop()


async def test_reload_shrink_keeps_all_moved_identities(store):
    delivery, now, events = _delivery(store)
    await delivery.offer([_finding(str(i)) for i in range(100)], Severity.INFO)
    assert delivery.configure({"max_pending": 1, "max_items": 1})
    await delivery.flush()
    assert _notices(events)[0]["count"] == 100
    await delivery.offer([_finding(str(i)) for i in range(100)], Severity.INFO)
    assert delivery.stats["cooldown_suppressed"] == 100
    await delivery.stop()


async def test_distinct_overflow_uses_next_global_slot_not_repeat_interval(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1))
    await delivery.offer([_finding("a"), _finding("b")], Severity.INFO)
    await delivery.flush()
    await delivery.offer([_finding("c"), _finding("d")], Severity.INFO)
    now[0] += 5
    await delivery.flush()
    assert len(_notices(events)) == 2
    assert _notices(events)[1]["count"] == 2
    await delivery.stop()


async def test_durable_state_saturation_never_resets_repeat_cooldown(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1, max_state=8))
    batch = [_finding(str(i)) for i in range(100)]
    for finding in batch:
        store.save(finding)
    await delivery.offer(batch, Severity.INFO)
    await delivery.flush()
    assert len(_notices(events)) == 1
    assert _notices(events)[0]["overflow_untracked_observations"] > 0
    await delivery.stop()
    restarted, clock, output = _delivery(store, DeliveryConfig(max_pending=1, max_state=8))
    clock[0] = now[0] + 5
    await restarted.offer(batch, Severity.INFO)
    await restarted.flush()
    assert not _notices(output)
    assert restarted.stats["cooldown_suppressed"] == 100
    assert store.count() == 100
    assert store._con.execute("SELECT count(*) FROM monitor_delivery_state").fetchone()[0] <= 8
    await restarted.stop()


async def test_global_floor_reload_drops_actionable_spool(store):
    delivery, now, events = _delivery(store, DeliveryConfig(minimum_severity="alert", max_pending=1))
    await delivery.offer([_finding("a", Severity.ALERT), _finding("b", Severity.ALERT, actionable=True)], Severity.INFO)
    assert delivery.configure({"minimum_severity": "critical", "max_pending": 1})
    await delivery.flush()
    assert not _notices(events)
    assert not store.delivery_overflow_keys()
    await delivery.stop()


@pytest.mark.parametrize("change", ["mute", "floor", "recovery"])
async def test_saturated_summary_cannot_bypass_state_change(store, change):
    delivery, now, events = _delivery(store, DeliveryConfig(minimum_severity="alert", max_pending=1, max_state=8))
    await delivery.offer([_finding(str(i), Severity.ALERT) for i in range(100)], Severity.INFO)
    assert delivery._untracked_count > 0
    if change == "mute":
        delivery.drop_watch("w1")
    elif change == "floor":
        delivery.configure({"minimum_severity": "critical", "max_pending": 1, "max_state": 8})
    else:
        await delivery.offer([_finding(str(i), Severity.INFO, active=False, phase="recovery")
                              for i in range(100)], Severity.INFO)
    await delivery.flush()
    assert not _notices(events)
    assert delivery._untracked_count == 0
    await delivery.stop()


async def test_three_spooled_owners_merge_in_one_cycle(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1))
    shared = _finding("shared")
    await delivery.offer([_finding("filler"), shared, replace(shared, watch_id="w2"),
                          replace(shared, watch_id="w3")], Severity.INFO)
    delivery.drop_watch("w1")
    delivery.drop_watch("w3")
    await delivery.flush()
    assert _notices(events)[0]["count"] == 1
    assert _notices(events)[0]["items"][0]["watch_id"] == "w2"
    await delivery.stop()


async def test_cleanup_failure_retains_work_for_retry(store, monkeypatch):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1))
    await delivery.offer([_finding("first"), _finding("spooled")], Severity.INFO)
    delivery.drop_watch("w1")
    original = store.drop_delivery_watch
    def fail(watches):
        raise RuntimeError("temporary database failure")
    monkeypatch.setattr(store, "drop_delivery_watch", fail)
    with pytest.raises(RuntimeError):
        await delivery.flush()
    assert delivery._dropped_watches == {"w1"}
    monkeypatch.setattr(store, "drop_delivery_watch", original)
    await delivery.flush()
    assert not _notices(events)
    assert not store.delivery_overflow_keys()
    await delivery.stop()


async def test_reload_prunes_existing_durable_spool(store):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1, max_state=128))
    await delivery.offer([_finding(str(i)) for i in range(100)], Severity.INFO)
    assert len(store.delivery_overflow_keys()) == 99
    delivery.configure({"max_pending": 1, "max_state": 8})
    await delivery.flush()
    assert len(store.delivery_overflow_keys()) <= 5
    assert _notices(events)[0]["count"] == 100
    await delivery.stop()


@pytest.mark.parametrize("spooled", [False, True])
async def test_final_live_check_after_recording_keeps_unrelated_incident(store, monkeypatch, spooled):
    delivery, now, events = _delivery(store, DeliveryConfig(max_pending=1 if spooled else 8))
    live = {"cleared", "kept"}
    async def validate(candidates, instant):
        return {item.key: item.watch_ids for item in candidates if item.incident_key in live}
    delivery._validate = validate
    await delivery.offer([_finding("cleared"), _finding("kept")], Severity.INFO)
    original = store.record_delivery
    def record(*args):
        result = original(*args)
        live.remove("cleared")
        now[0] += 1
        return result
    monkeypatch.setattr(store, "record_delivery", record)
    await delivery.flush()
    assert _notices(events)[0]["count"] == 1
    assert _notices(events)[0]["items"][0]["title"] == "Incident kept"
    await delivery.stop()
    await delivery.flush()
    assert len(_notices(events)) == 1

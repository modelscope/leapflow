# Copyright (c) Alibaba, Inc. and its affiliates.
"""Rank-only learning through real EventBus and profile-owned DuckDB persistence."""
from __future__ import annotations

import asyncio
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from leapflow.hardware.stream import HardwareEventClass, classify_hardware_event
from leapflow.hardware.view_prior_learning import ViewPriorLearning, _project
from leapflow.hardware.view_priors import ViewPriorPort, ViewPriorSettings
from leapflow.platform.event_bus import EventBus
from leapflow.storage.connection import LocalConnectionHolder
from leapflow.storage.view_prior_store import ViewPriorStore


class Memory:
    def ingest(self, *args, **kwargs):
        pass

    def remember_event(self, *args, **kwargs):
        pass


class Privacy:
    def should_ingest(self, *args):
        return False


def report(run_id="r1", scope="a"*64, workcell="cell", finished=None):
    return {"skill_id": "perceive.survey", "workcell": workcell, "run_id": run_id,
            "finished_at": time.time() if finished is None else finished,
            "outcome": {"ok": True, "status": "succeeded", "result": {
                "observation_scope": scope, "moves": 1, "trials": [{
                    "feature": "camera.mounted:ring:2", "duration_s": 2.,
                    "before": {"views": [{"support": .2}], "verdict": "partial"},
                    "after": {"views": [{"support": .9}], "verdict": "sufficient"}}]}}}


@pytest.fixture
async def learning(tmp_path):
    holder = LocalConnectionHolder(tmp_path / "profile.duckdb")
    bus = EventBus(Memory(), Memory())
    store = ViewPriorStore(holder, ViewPriorSettings(enabled=True))

    async def run_db(fn):
        return fn()

    def publish(event):
        return asyncio.create_task(bus.handle_event(event.event_type, event.to_payload()))

    learner = ViewPriorLearning(store, bus, publish, run_db)
    yield SimpleNamespace(learner=learner, store=store, holder=holder, bus=bus)
    await learner.close()
    await bus.shutdown()
    holder.close()


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def test_pipeline_deduplicates_persists_and_isolates(learning):
    learner = learning.learner
    assert isinstance(learner, ViewPriorPort)
    assert classify_hardware_event("observation_trial") == HardwareEventClass.RECORD
    learner.expect_scope("node/cell", "a"*64)
    learner.record_run("node", report())
    learner.record_run("node", report())
    await settle()
    hints = await learner.hints("node/cell")
    assert hints["prior_scope"] == "a"*64
    assert 0 < hints["priors"]["camera.mounted:ring:2"] <= 1
    assert learner.stats["accepted"] == learner.stats["duplicate"] == 1
    assert await learner.hints("another/cell") == {}
    restarted = ViewPriorStore(learning.holder, learner.settings)
    assert restarted.hints("node/cell", time.time()) == hints
    assert restarted.hints("node/cell", time.time()+learner.settings.ttl_s+1) == {}


@pytest.mark.parametrize("change", ["stale", "future", "scope", "source", "raw", "nan", "cost", "class"])
async def test_invalid_signals_never_train(learning, change):
    learner = learning.learner
    learner.expect_scope("node/cell", "a"*64)
    value = _project("node", report())
    payload = {"event_class": "record", "device_id": "node/cell", "value": value}
    if change == "stale":
        value["observed_at"] -= learner.settings.max_age_s+1
    elif change == "future":
        value["observed_at"] += 100
    elif change == "scope":
        value["scope"] = "b"*64
    elif change == "source":
        payload["device_id"] = "other/cell"
    elif change == "raw":
        value["depth"] = [1, 2]
    elif change == "nan":
        value["trials"][0]["gain"] = float("nan")
    elif change == "cost":
        value["trials"][0]["duration_s"] = -1
    else:
        payload["event_class"] = "reflex"
    await learning.bus.handle_event("hw.observation_trial", payload)
    await settle()
    assert await learner.hints("node/cell") == {}


async def test_privacy_precedes_learning(learning):
    learning.bus.set_privacy_filter(Privacy())
    learning.learner.expect_scope("node/cell", "a"*64)
    learning.learner.record_run("node", report())
    await settle()
    assert await learning.learner.hints("node/cell") == {}


async def test_failed_and_noop_trials_are_durable_without_positive_reward(learning):
    learner = learning.learner
    learner.expect_scope("node/cell", "a"*64)
    failed = report()
    failed["outcome"]["ok"] = False
    learner.record_run("node", failed)
    await settle()
    assert (await learner.hints("node/cell"))["priors"] == {}
    assert learning.holder.connection.execute("SELECT count(*) FROM observation_view_trials").fetchone()[0] == 1


async def test_scope_change_and_close_stop_pending_ingestion(learning):
    learner = learning.learner
    learner.expect_scope("node/cell", "b"*64)
    learner.record_run("node", report())
    await settle()
    assert learner.stats["configuration_mismatch"] == 1
    await learner.close()
    assert await learner.hints("node/cell") == {}
    learner.record_run("node", report(scope="b"*64))
    await settle()
    assert learner.stats["accepted"] == 0


def test_store_is_bounded_and_configuration_specific(tmp_path):
    holder = LocalConnectionHolder(tmp_path / "bounded.duckdb")
    settings = ViewPriorSettings(enabled=True, max_events=2)
    store = ViewPriorStore(holder, settings)
    try:
        for i, scope in enumerate(("a", "a", "b")):
            store.accept({"event_id": str(i), "workcell": "node/cell", "scope": scope*64,
                          "observed_at": 100.+i, "trials": [{"feature": scope, "reward": .5}]}, 102.)
        assert holder.connection.execute("SELECT count(*) FROM observation_view_trials").fetchone()[0] == 2
        assert store.hints("node/cell", 102.) == {"prior_scope": "b"*64, "priors": {"b": .5}}
    finally:
        holder.close()


@pytest.mark.parametrize("values", [{"enabled": 1}, {"ttl_s": 0}, {"max_pending": 0},
                                   {"max_events": 1.5}, {"duration_scale_s": float("inf")}])
def test_settings_reject_unbounded_policy(values):
    with pytest.raises(ValueError):
        replace(ViewPriorSettings(), **values)


async def test_terminal_summaries_and_refused_views_are_redacted_and_durable(learning):
    learner = learning.learner
    learner.expect_scope('node/cell', 'a'*64)
    run = report()
    run['outcome']['result']['trials'][0]['before']['raw_image'] = 'must not persist'
    run['outcome']['result']['refused'] = [{'feature': 'camera:blocked', 'failure_code': 'unreachable'}]
    learner.record_run('node', run)
    await settle()
    trials, terminal = learning.holder.connection.execute(
        'SELECT trials, terminal FROM observation_view_trials').fetchone()
    assert 'raw_image' not in trials
    projected = json.loads(trials)
    assert projected[0]['before']['support'] == .2
    assert projected[0]['after']['sufficient'] is True
    assert projected[1]['reward'] is None
    assert json.loads(terminal) == {'status': 'succeeded', 'failure_code': '', 'moves': 1}


@pytest.mark.parametrize('phase', ['planning', 'approval', 'admission'])
async def test_governance_refusals_persist_without_training_or_invented_runs(learning, phase):
    learner = learning.learner
    learner.record_decision('node', 'cell', 'b'*64, phase, {'failure_code': 'declined'}, 'decision')
    await settle()
    event_id, trials, terminal = learning.holder.connection.execute(
        'SELECT event_id, trials, terminal FROM observation_view_trials').fetchone()
    assert event_id == f'node:decision:decision:{phase}'
    assert json.loads(trials) == []
    assert json.loads(terminal)['failure_code'] == 'declined'
    assert (await learner.hints('node/cell'))['priors'] == {}


async def test_lookup_timeout_degrades_without_failing_task(learning):
    learner = learning.learner
    learner.settings = replace(learner.settings, lookup_timeout_s=.01)
    cancelled = asyncio.Event()
    async def slow(fn):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    learner._run_db = slow
    assert await asyncio.wait_for(learner.hints('node/cell'), .5) == {}
    assert cancelled.is_set()


async def test_backpressure_is_bounded_and_close_drains_accepted_work(learning):
    learner = learning.learner
    learner.settings = replace(learner.settings, max_pending=1)
    learner.expect_scope('node/cell', 'a'*64)
    release = asyncio.Event()
    async def deferred(fn):
        await release.wait()
        return fn()
    learner._run_db = deferred
    learner.record_run('node', report('first'))
    learner.record_run('node', report('second'))
    await settle()
    assert len(learner._pending) == learner.stats['backpressure'] == 1
    closing = asyncio.create_task(learner.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert learner.stats['accepted'] == 1
    assert await learner.hints('node/cell') == {}


async def test_reconfiguration_while_database_dispatch_waits_rejects_positive_learning(learning):
    learner = learning.learner
    learner.expect_scope('node/cell', 'a'*64)
    entered, release = asyncio.Event(), asyncio.Event()
    async def queued(fn):
        entered.set()
        await release.wait()
        return fn()
    learner._run_db = queued
    learner.record_run('node', report())
    await asyncio.wait_for(entered.wait(), 1.)
    learner.expect_scope('node/cell', 'b'*64)
    release.set()
    await settle()
    assert learner.stats['configuration_mismatch'] == 1
    assert learner.stats['accepted'] == 0


async def test_shared_holder_resolves_connection_on_serialized_worker_and_migrates(tmp_path):
    holder = LocalConnectionHolder(tmp_path / 'threaded.duckdb')
    holder.connection.execute('''CREATE TABLE observation_view_trials (
        event_id VARCHAR PRIMARY KEY, workcell VARCHAR, scope VARCHAR, observed_at DOUBLE, trials VARCHAR)''')
    root = holder.connection
    bus = EventBus(Memory(), Memory())
    threads = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        async def run_db(fn):
            def run():
                threads.append(threading.get_ident())
                assert holder.connection is not root
                return fn()
            return await asyncio.get_running_loop().run_in_executor(executor, run)
        def publish(event):
            return asyncio.create_task(bus.handle_event(event.event_type, event.to_payload()))
        learner = ViewPriorLearning(ViewPriorStore(holder, ViewPriorSettings(enabled=True)), bus, publish, run_db)
        try:
            learner.expect_scope('node/cell', 'a'*64)
            learner.record_run('node', report())
            await settle()
            for _ in range(100):
                if learner.stats['accepted']:
                    break
                await asyncio.sleep(.01)
            assert learner.stats['accepted'] == 1
            assert (await learner.hints('node/cell'))['priors']
            assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
            assert root.execute('SELECT terminal FROM observation_view_trials').fetchone()
        finally:
            await learner.close()
            await bus.shutdown()
    holder.close()


async def test_production_binding_replaces_registry_without_stale_publisher(learning, monkeypatch):
    from leapflow.cli.context import Context
    import leapflow.plugins as plugins
    class Registry:
        report = SimpleNamespace(admitted=(), rejected=())
        def __init__(self):
            self.events = []
        def set_event_emitter(self, emit):
            self.emit = emit
        def publish_event(self, event):
            self.events.append(event)
            self.emit(event)
    class Wiring:
        _bind_hardware_plugin = Context._bind_hardware_plugin
        _hardware_event_emitter = Context._hardware_event_emitter
        _publish_hardware_observation = Context._publish_hardware_observation
        def _process_trust_ledger(self):
            return None
    wired = Wiring()
    wired.event_bus = learning.bus
    wired.settings = SimpleNamespace(hardware_view_priors='{}')
    wired._approval_orchestrator = None
    wired._hardware_view_priors = learning.learner
    first, second = Registry(), Registry()
    bindings = []
    monkeypatch.setattr(plugins, 'get_registry', lambda: SimpleNamespace(bind_runtime=lambda **kw: bindings.append(kw)))
    wired._hardware_registry = first
    wired._bind_hardware_plugin()
    learning.learner._publish = wired._publish_hardware_observation
    learning.learner.expect_scope('node/cell', 'a'*64)
    wired._hardware_registry = second
    wired._bind_hardware_plugin()
    learning.learner.record_run('node', report())
    await settle()
    assert not first.events and len(second.events) == 1
    assert learning.learner.stats['accepted'] == 1
    assert bindings[-1]['hardware_view_priors'] is learning.learner

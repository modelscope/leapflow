# Copyright (c) Alibaba, Inc. and its affiliates.
"""Real shared-DuckDB persistence primitives; NOT physical model admission."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from leapflow.cache.manager import CacheManager
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.layout import build_layout

import pytest

from leapflow.hardware.calibration.generations import GenerationStore
from leapflow.hardware.calibration_store import CalibrationRecord, CalibrationStore
from leapflow.hardware.lhp.control_sessions import NodeSessions
from leapflow.hardware.node_gate import NodeWriteGate
from leapflow.hardware.transport import WriteOutcome
from leapflow.storage.connection import LocalConnectionHolder
from tests._harness.hardware_stubs import StubNodeRegistry

pytestmark = pytest.mark.component


def commit(store: GenerationStore, request: str, expected: int | None = None, **changes: object):
    return store.commit_generation(
        **{
            "workcell_id": "cell",
            "request_id": request,
            "expected_generation": expected,
            "dependency_digest": "deps",
            "model": {"synthetic": True, "value": request},
            "accepted_blocks": {"synthetic_block": {"value": request}},
            "causal": {"qualification": "isolated software test only"},
            **changes,
        }
    )


def test_shared_holder_atomic_generation_restart_outbox_and_legacy_records(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        legacy = CalibrationStore(connection_holder=holder)
        record = CalibrationRecord("arm", "tcp", 1, {"old": True})
        assert legacy.record(record)
        store = GenerationStore(holder)
        candidate = store.record_candidate(
            workcell_id="cell", evidence_digest="e", candidate={"synthetic": True}
        )
        assert store.candidate(candidate, workcell_id="cell") == {"synthetic": True}
        store.record_assessment(candidate, {"qualified": False, "reason": "software test"})
        first = commit(store, "one")
        assert first.generation == 1
        assert commit(store, "one") == first
        assert len(store.pending_events()) == 1
        with pytest.raises(ValueError, match="different activation"):
            commit(store, "one", model={"different": True})
        second = commit(store, "two", 1)
        assert second.previous_generation == 1
        restarted = GenerationStore(holder)
        assert restarted.active("cell") == second
        assert restarted.active("cell").model["synthetic"] is True
        restored = restarted.rollback(
            workcell_id="cell",
            request_id="rb",
            expected_generation=2,
            dependency_digest="deps",
            causal={"test": True},
        )
        assert restored == first
        assert (
            restarted.rollback(
                workcell_id="cell",
                request_id="rb",
                expected_generation=2,
                dependency_digest="deps",
                causal={"test": True},
            )
            == first
        )
        events = restarted.pending_events()
        assert len(events) == 3
        for event in events:
            assert event["event_class"] == "record"
            restarted.acknowledge_event(event["event_id"])
        assert restarted.pending_events() == ()
        assert legacy.latest("arm", "tcp").parameters["old"] is True
    finally:
        holder.close()


def test_cas_and_midtransaction_failure_preserve_previous_pointer(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        first = commit(store, "one")
        with pytest.raises(ValueError, match="compare-and-swap"):
            commit(store, "stale")
        with pytest.raises(ValueError, match="identity cannot be empty"):
            commit(store, "fails", 1, accepted_blocks={"good": {}, "": {}})
        assert store.active("cell") == first
        assert store.committed_request("cell", "fails") is None
        assert (
            holder.connection.execute("SELECT COUNT(*) FROM calib_generations").fetchone()[0] == 1
        )
        assert holder.connection.execute("SELECT COUNT(*) FROM calib_blocks").fetchone()[0] == 1
        assert len(store.pending_events()) == 1
    finally:
        holder.close()


def test_incompatible_rollback_deactivates_without_restoring_invalid_geometry(
    tmp_path: Path,
) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        commit(store, "one")
        commit(store, "two", 1)
        assert (
            store.rollback(
                workcell_id="cell",
                request_id="changed",
                expected_generation=2,
                dependency_digest="new-mount",
                causal={"mount": "changed"},
            )
            is None
        )
        assert store.active("cell") is None
        assert any(e["kind"] == "invalidated" for e in store.pending_events())
    finally:
        holder.close()


async def test_reservation_blocks_gate_before_precheck_and_dispatch_blocks_activation() -> None:
    registry = StubNodeRegistry()
    sessions = NodeSessions(registry)
    registry.node_sessions = sessions
    gate = NodeWriteGate(registry)
    device = "conformance_device"
    token = sessions.reserve((device, "consumer:precision"), owner_id="activation")
    assert sessions.busy_device((device,)) == device
    assert not (await gate.write(device, "setpoint", 1)).ok
    sessions.release_reservation(token)
    entered, settle = asyncio.Event(), asyncio.Event()
    transport = await registry.transport(device)
    original_write = transport.write

    async def waiting(channel_id: str, value: object) -> WriteOutcome:
        entered.set()
        await settle.wait()
        return await original_write(channel_id, value)

    transport.write = waiting
    task = asyncio.create_task(gate.write(device, "setpoint", 1))
    await entered.wait()
    with pytest.raises(ValueError, match="unsettled"):
        sessions.reserve((device,), owner_id="activation")
    settle.set()
    assert (await task).ok
    token = sessions.reserve((device,), owner_id="activation")
    sessions.release_reservation(token)


async def test_unknown_dispatch_latches_activation_refusal() -> None:
    registry = StubNodeRegistry()
    sessions = NodeSessions(registry)
    registry.node_sessions = sessions
    transport = await registry.transport("conformance_device")

    async def uncertain(channel_id: str, value: object) -> WriteOutcome:
        return WriteOutcome(ok=False, side_effect_state="unknown", error="uncertain")

    transport.write = uncertain
    gate = NodeWriteGate(registry)
    assert not (await gate.write("conformance_device", "setpoint", 1)).ok
    with pytest.raises(ValueError, match="unsettled"):
        sessions.reserve(("conformance_device",), owner_id="activation")
    assert not (await gate.write("conformance_device", "setpoint", 2)).ok


class Provider:
    kind = "static"

    def discover(self):
        return StubNodeRegistry().contexts()


def real_registry(tmp_path: Path) -> HardwareRegistry:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, instrument_db_path=str(tmp_path / "instrument.duckdb")),
        providers=[Provider()],
    )
    registry.load()
    registry.bind_node_sessions(NodeSessions(registry))
    return registry


async def test_real_registry_lazy_shared_holder_and_private_cache_binding(tmp_path: Path) -> None:
    registry = real_registry(tmp_path)
    try:
        store = registry.calibration_generations
        assert not (tmp_path / "instrument.duckdb").exists()
        assert store._holder is registry.calibration_store._holder
        cache = CacheManager(
            build_layout(tmp_path / "data").ensure(profile_id="default").cache, profile_id="default"
        )
        registry.bind_persistence(cache_manager=cache)
        first = await asyncio.to_thread(commit, store, "private")
        entries = [e for e in cache.list_entries() if e.path == tmp_path / "instrument.duckdb"]
        assert len(entries) == 1 and entries[0].sensitive and not entries[0].syncable
        assert entries[0].owner_component == "hardware" and entries[0].scope == "profile"
        assert await asyncio.to_thread(store.active, "cell") == first
    finally:
        await registry.close_all()


async def test_reserved_commit_repeated_cancellation_and_registry_shutdown_settle(
    tmp_path: Path,
) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions
    entered, release = threading.Event(), threading.Event()

    def delayed_commit():
        entered.set()
        assert release.wait(5), "test transaction settlement never released"
        return commit(store, "cancelled-request")

    projections = []

    def publish(generation):
        assert sessions.reserved_device("conformance_device")
        assert sessions.reserved_device("consumer:precision")
        projections.append(generation)

    task = asyncio.create_task(
        sessions.run_reserved(
            ("conformance_device", "consumer:precision"),
            owner_id="owner",
            operation=delayed_commit,
            on_settle=publish,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert sessions.busy_device(("consumer:precision",)) == "consumer:precision"
        assert not (await NodeWriteGate(registry).write("conformance_device", "setpoint", 1)).ok
        refused = await sessions.open(
            {"device_id": "conformance_device"}, connection_id="other-client"
        )
        assert not refused["ok"]
        with pytest.raises(ValueError, match="busy"):
            await sessions.run_reserved(
                ("conformance_device",), owner_id="other", operation=lambda: commit(store, "race")
            )
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.01)
        assert not closing.done() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert not sessions.reserved_device("conformance_device")
        assert (await asyncio.to_thread(store.active, "cell")).request_id == "cancelled-request"
        assert len(projections) == 1 and projections[0].request_id == "cancelled-request"
        assert len(await asyncio.to_thread(store.pending_events)) == 1
    finally:
        release.set()
        await sessions.settle_transactions()
        await registry.close_all()


@pytest.mark.parametrize("end", ["success", "cancel", "refusal"])
async def test_reserved_preflight_holds_all_consumers_and_refusal_never_dispatches(
    tmp_path: Path, end: str
) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions
    entered, release = asyncio.Event(), asyncio.Event()

    async def preflight() -> None:
        assert sessions.reserved_device("conformance_device")
        assert sessions.reserved_device("consumer:precision")
        entered.set()
        await release.wait()
        if end == "refusal":
            raise ValueError("isolated observational preflight refusal")

    task = asyncio.create_task(
        sessions.run_reserved(
            ("conformance_device", "consumer:precision"),
            owner_id="owner",
            preflight=preflight,
            operation=lambda: commit(store, "preflight-request"),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not (await NodeWriteGate(registry).write("conformance_device", "setpoint", 1)).ok
        assert not (
            await sessions.open({"device_id": "conformance_device"}, connection_id="other")
        )["ok"]
        with pytest.raises(ValueError, match="busy"):
            sessions.reserve(("consumer:precision",), owner_id="other")
        assert await asyncio.to_thread(store.snapshot, "cell") is None
        if end == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if end == "refusal":
                with pytest.raises(ValueError, match="preflight refusal"):
                    await task
            else:
                assert (await task).request_id == "preflight-request"
        assert not sessions.reserved_device("conformance_device")
        assert not sessions.unsettled_device("consumer:precision")
        if end != "success":
            assert await asyncio.to_thread(store.snapshot, "cell") is None
            token = sessions.reserve(("conformance_device", "consumer:precision"), owner_id="later")
            sessions.release_reservation(token)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.close_all()


async def test_owner_shutdown_waits_for_reserved_preflight_before_closing_shared_holder(
    tmp_path: Path,
) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions
    entered, release = asyncio.Event(), asyncio.Event()

    async def preflight() -> None:
        entered.set()
        await release.wait()
        # The actual holder must still be usable after shutdown started.
        assert await asyncio.to_thread(store.snapshot, "cell") is None

    task = asyncio.create_task(
        sessions.run_reserved(
            ("conformance_device",),
            owner_id="owner",
            preflight=preflight,
            operation=lambda: commit(store, "shutdown-preflight"),
        )
    )
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closing = asyncio.create_task(registry.close_all())
        await asyncio.sleep(0.02)
        assert not closing.done() and sessions.reserved_device("conformance_device")
        release.set()
        assert (await task).request_id == "shutdown-preflight"
        await closing
        assert not sessions.reserved_device("conformance_device")
        assert (await asyncio.to_thread(store.snapshot, "cell")).request_id == "shutdown-preflight"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await closing
        await registry.close_all()


async def test_reserved_failure_preserves_generation_and_releases_only_after_settlement(
    tmp_path: Path,
) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions
    try:
        first = await asyncio.to_thread(commit, store, "first")
        with pytest.raises(ValueError):
            await sessions.run_reserved(
                ("conformance_device",),
                owner_id="owner",
                operation=lambda: commit(store, "failure", 1, accepted_blocks={"ok": {}, "": {}}),
            )
        assert not sessions.reserved_device("conformance_device")
        assert await asyncio.to_thread(store.active, "cell") == first
        assert await asyncio.to_thread(store.committed_request, "cell", "failure") is None
        await sessions.shutdown()
        with pytest.raises(ValueError, match="busy"):
            await sessions.run_reserved(
                ("conformance_device",), owner_id="late", operation=lambda: commit(store, "late", 1)
            )
    finally:
        await registry.close_all()


def test_snapshot_does_not_create_generation_schema_and_refuses_corrupt_pointer(
    tmp_path: Path,
) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        assert store.snapshot("cell") is None
        assert (
            holder.connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name LIKE 'calib_%'"
            ).fetchone()[0]
            == 0
        )
        first = commit(store, "one")
        assert store.snapshot("cell") == first
        assert store.snapshot("other-client") is None
        holder.connection.execute("UPDATE calib_active SET generation=999 WHERE workcell_id='cell'")
        with pytest.raises(ValueError, match="missing committed generation"):
            store.snapshot("cell")
        holder.connection.execute("UPDATE calib_active SET generation=1 WHERE workcell_id='cell'")
        holder.connection.execute("UPDATE calib_generations SET model_json='{}'")
        with pytest.raises(ValueError, match="digest mismatch"):
            store.snapshot("cell")
    finally:
        holder.close()


async def test_projection_failure_latches_all_reserved_consumers(tmp_path: Path) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions

    def broken(generation):
        assert sessions.reserved_device("conformance_device")
        raise ValueError("projection cannot install committed geometry")

    try:
        with pytest.raises(ValueError, match="projection"):
            await sessions.run_reserved(
                ("conformance_device", "consumer:precision"),
                owner_id="owner",
                operation=lambda: commit(store, "landed"),
                on_settle=broken,
            )
        assert (await asyncio.to_thread(store.snapshot, "cell")).request_id == "landed"
        assert not sessions.reserved_device("conformance_device")
        assert sessions.unsettled_device("conformance_device")
        assert sessions.unsettled_device("consumer:precision")
        with pytest.raises(ValueError, match="unsettled"):
            sessions.reserve(("consumer:precision",), owner_id="other")
        assert not (await NodeWriteGate(registry).write("conformance_device", "setpoint", 1)).ok
    finally:
        await registry.close_all()


async def test_async_projection_refused_before_transaction(tmp_path: Path) -> None:
    registry = real_registry(tmp_path)
    store, sessions = registry.calibration_generations, registry.node_sessions

    async def forbidden(generation):
        pytest.fail("owner publication must not suspend")

    try:
        with pytest.raises(ValueError, match="synchronous"):
            await sessions.run_reserved(
                ("conformance_device",),
                owner_id="owner",
                operation=lambda: commit(store, "never"),
                on_settle=forbidden,
            )
        assert await asyncio.to_thread(store.snapshot, "cell") is None
        assert not sessions.reserved_device("conformance_device")
    finally:
        await registry.close_all()


@pytest.mark.parametrize("kind", ["candidate", "rejected", "no_op"])
def test_record_only_decisions_survive_restart_deduplicate_and_never_advance_pointer(
    tmp_path, kind
):
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        first = commit(store, "one")
        causal = {
            "candidate_digest": "isolated-candidate",
            "evidence_digest": "isolated-evidence",
            "gate": kind,
        }
        event = store.record_decision(
            workcell_id="cell", request_id="branch", kind=kind, causal=causal
        )
        assert event["event_class"] == "record" and event["kind"] == kind
        assert (
            store.record_decision(workcell_id="cell", request_id="branch", kind=kind, causal=causal)
            == event
        )
        with pytest.raises(ValueError, match="rebound"):
            store.record_decision(
                workcell_id="cell", request_id="branch", kind=kind, causal={"gate": "different"}
            )
        with pytest.raises(ValueError, match="record-only"):
            store.record_decision(workcell_id="cell", request_id="bad", kind="activated", causal={})
        restarted = GenerationStore(holder)
        assert restarted.snapshot("cell") == first and restarted.invalidation("cell") is None
        assert restarted.committed_request("cell", "branch") == event
        assert len(restarted.pending_events()) == 2
        holder.connection.execute(
            "UPDATE calib_outbox SET event_json='{}' WHERE event_id=?", [event["event_id"]]
        )
        with pytest.raises(ValueError, match="schema"):
            restarted.pending_events()
    finally:
        holder.close()


def test_invalidation_cas_retains_stop_use_after_restart_and_record_only_events(tmp_path):
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        commit(store, "one")
        second = commit(store, "two", 1)
        kwargs = {
            "workcell_id": "cell",
            "request_id": "stop-use",
            "expected_generation": 2,
            "dependency_digest": "changed-mount",
            "causal": {"reason": "isolated dependency mismatch"},
        }
        with pytest.raises(ValueError, match="compare-and-swap"):
            store.invalidate(**{**kwargs, "expected_generation": 1})
        assert store.snapshot("cell") == second
        store.invalidate(**kwargs)
        store.invalidate(**kwargs)
        with pytest.raises(ValueError, match="different invalidation"):
            store.invalidate(**{**kwargs, "dependency_digest": "different"})
        assert store.snapshot("cell") is None
        event = store.invalidation("cell")
        assert event["generation"] is None and event["previous_generation"] == 2
        assert event["kind"] == "invalidated" and len(store.pending_events()) == 3
        store.record_decision(
            workcell_id="cell",
            request_id="later-no-op",
            kind="no_op",
            causal={"gate": "nothing changed"},
        )
        restarted = GenerationStore(holder)
        assert restarted.invalidation("cell") == event and restarted.snapshot("cell") is None
        assert restarted.invalidation("other-client") is None
        # A separately admitted generation would replace the stop-use verdict.
        third = commit(restarted, "three")
        assert third.generation == 3 and restarted.invalidation("cell") is None
    finally:
        holder.close()


@pytest.mark.parametrize("damage", ["blocks", "causal", "missing_decision", "pointer", "journal"])
def test_restart_refuses_incomplete_or_corrupted_committed_state(tmp_path, damage):
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        commit(store, "one")
        commit(store, "two", 1)
        statements = {
            "blocks": "UPDATE calib_blocks SET record_json='{}' WHERE generation=2",
            "causal": "UPDATE calib_decisions SET decision_json='{}' WHERE request_id='two'",
            "missing_decision": "DELETE FROM calib_decisions WHERE request_id='two'",
            "pointer": "UPDATE calib_active SET generation=1",
            "journal": "UPDATE calib_pointer_journal SET event_id='missing' WHERE revision=2",
        }
        holder.connection.execute(statements[damage])
        with pytest.raises(ValueError):
            GenerationStore(holder).snapshot("cell")
    finally:
        holder.close()


def test_pointer_journal_uses_commit_order_not_wall_clock(tmp_path, monkeypatch):
    from leapflow.hardware.calibration import generations

    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        clock = iter([900, 100, 100, 50, 1])
        monkeypatch.setattr(generations.time, "time", lambda: next(clock))
        first = commit(store, "one")
        commit(store, "two", 1)
        assert (
            store.rollback(
                workcell_id="cell",
                request_id="back",
                expected_generation=2,
                dependency_digest="deps",
                causal={},
            )
            == first
        )
        assert GenerationStore(holder).snapshot("cell") == first
        store.invalidate(
            workcell_id="cell",
            request_id="invalid",
            expected_generation=1,
            dependency_digest="changed",
            causal={},
        )
        assert GenerationStore(holder).invalidation("cell")["request_id"] == "invalid"
    finally:
        holder.close()


@pytest.mark.parametrize("damage", ["last", "all", "gap", "kind", "request", "schema"])
def test_invalidation_never_becomes_nominal_with_incomplete_journal(
    tmp_path: Path, damage: str
) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        commit(store, "one")
        store.invalidate(
            workcell_id="cell",
            request_id="stop",
            expected_generation=1,
            dependency_digest="changed",
            causal={},
        )
        mutations = {
            "last": "DELETE FROM calib_pointer_journal WHERE revision=2",
            "all": "DELETE FROM calib_pointer_journal",
            "gap": "UPDATE calib_pointer_journal SET revision=4 WHERE revision=2",
            "kind": "UPDATE calib_decisions SET kind='no_op' WHERE request_id='stop'",
            "request": "UPDATE calib_decisions SET request_id='other' WHERE request_id='stop'",
            "schema": "DROP TABLE calib_active",
        }
        holder.connection.execute(mutations[damage])
        with pytest.raises(ValueError):
            GenerationStore(holder).state("cell")
        with pytest.raises(ValueError):
            GenerationStore(holder).states()
    finally:
        holder.close()


def test_state_binds_stop_use_digest_and_removed_generation_without_mutating(
    tmp_path: Path,
) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        assert store.states() == ()
        assert store.state("cell").basis()["generation"] is None
        first = commit(store, "one")
        assert store.state("cell").generation == first
        store.invalidate(
            workcell_id="cell",
            request_id="stop",
            expected_generation=1,
            dependency_digest="changed",
            causal={},
        )
        state = store.state("cell")
        assert state.generation is None and state.stopped_generation == first
        assert state.basis()["invalidation_digest"] == state.invalidation["event_id"]
        decoded = state.invalidation
        decoded["kind"] = "activated"
        assert state.invalidation["kind"] == "invalidated"
        assert store.states() == (state,)
    finally:
        holder.close()


@pytest.mark.parametrize("ambiguous", [False, True])
def test_legacy_pointer_migration_preserves_or_refuses_order(
    tmp_path: Path, ambiguous: bool
) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        commit(store, "one")
        second = commit(store, "two", 1)
        holder.connection.execute("DROP TABLE calib_pointer_journal")
        holder.connection.execute("UPDATE calib_decisions SET created_at=1 WHERE request_id='one'")
        holder.connection.execute(
            "UPDATE calib_decisions SET created_at=? WHERE request_id='two'",
            [1 if ambiguous else 2],
        )
        restarted = GenerationStore(holder)
        if ambiguous:
            with pytest.raises(ValueError, match="ambiguous"):
                restarted.snapshot("cell")
            with pytest.raises(ValueError, match="ambiguous"):
                restarted.active("cell")
            assert (
                holder.connection.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name='calib_pointer_journal'"
                ).fetchone()[0]
                == 0
            )
        else:
            assert restarted.snapshot("cell") == second
            assert restarted.active("cell") == second
            assert holder.connection.execute(
                "SELECT revision FROM calib_pointer_journal ORDER BY revision"
            ).fetchall() == [(1,), (2,)]
    finally:
        holder.close()


def test_identical_assessments_preserve_distinct_candidate_bindings(tmp_path: Path) -> None:
    holder = LocalConnectionHolder(tmp_path / "instrument.duckdb")
    try:
        store = GenerationStore(holder)
        a = store.record_candidate(workcell_id="cell", evidence_digest="ea", candidate={"value": 1})
        b = store.record_candidate(workcell_id="cell", evidence_digest="eb", candidate={"value": 2})
        ra = store.record_assessment(a, {"qualified": False})
        rb = store.record_assessment(b, {"qualified": False})
        assert ra != rb
        assert (
            holder.connection.execute("SELECT COUNT(*) FROM calib_assessments").fetchone()[0] == 2
        )
        with pytest.raises(ValueError, match="keys must be strings"):
            commit(store, "bad-key", model={"nested": {1: "one", "1": "another"}})
        for generation in (0, True, -1, 1.0):
            with pytest.raises(ValueError, match="positive integer"):
                commit(store, "bad-generation", generation)
        assert store.active("cell") is None
    finally:
        holder.close()

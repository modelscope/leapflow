# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for durable plugin trust persistence (Fix D2).

Covers the DuckDB-backed ``PluginStatsStore`` round-trip, the
``_PersistingTrustLedger`` save-on-transition behavior, graceful degradation
when the store is unavailable or corrupt, the durable rolling usage samples that
reliability scoring depends on, and the process-global wiring in
``_wire_plugin_stats_sink`` / ``persist_plugin_trust_state``.

Hermetic: no network, no LLM, DuckDB writes confined to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.engine.session.session_factory import (
    _PersistingTrustLedger,
    _default_stats_db_path,
    _load_or_new_trust_ledger,
    _resolve_stats_store,
    persist_plugin_trust_state,
)
from leapflow.engine.turn_usage import TurnUsageTracker
from leapflow.learning.plugin_stats import PluginUsageTracker
from leapflow.learning.plugin_stats_store import PluginStatsStore
from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel, trust_history


def _db(tmp_path: Path) -> Path:
    return tmp_path / "plugin_stats.duckdb"


class TestStoreRoundTrip:
    """Raw save/load contract of PluginStatsStore + PluginTrustLedger."""

    def test_save_new_ledger_load_round_trips_trust_levels(self, tmp_path: Path) -> None:
        """save → fresh store+ledger → load restores the exact trust levels."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = PluginTrustLedger(candidate_at=2, verified_at=4, production_at=6)
        # Drive one plugin to VERIFIED and another to CANDIDATE.
        for _ in range(4):
            ledger.record_success("alpha")
        for _ in range(2):
            ledger.record_success("beta")
        assert ledger.level("alpha") is PluginTrustLevel.VERIFIED
        assert ledger.level("beta") is PluginTrustLevel.CANDIDATE

        assert store.save_trust_state(ledger.to_state()) is True

        # A brand-new store instance over the same file, and a brand-new ledger.
        reopened = PluginStatsStore(_db(tmp_path))
        state = reopened.load_trust_state()
        assert state is not None
        restored = PluginTrustLedger.load_state(state)

        assert restored.level("alpha") is PluginTrustLevel.VERIFIED
        assert restored.level("beta") is PluginTrustLevel.CANDIDATE

    def test_hard_failure_freeze_survives_round_trip(self, tmp_path: Path) -> None:
        """A frozen (hard-failed) plugin stays frozen after reload."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = PluginTrustLedger(candidate_at=2)
        ledger.record_success("gamma")
        ledger.record_success("gamma")
        ledger.record_failure("gamma", hard=True)
        assert ledger.level("gamma") is PluginTrustLevel.DRAFT

        store.save_trust_state(ledger.to_state())
        restored = PluginTrustLedger.load_state(PluginStatsStore(_db(tmp_path)).load_trust_state())
        # Frozen plugins cannot re-accrue trust.
        for _ in range(5):
            restored.record_success("gamma")
        assert restored.level("gamma") is PluginTrustLevel.DRAFT


class TestGracefulDegradation:
    """Missing / unavailable / corrupt store must never crash callers."""

    def test_no_path_store_is_noop(self) -> None:
        """A store with no db_path returns falsy save / None load."""
        store = PluginStatsStore(None)
        assert store.save_trust_state({"levels": {}}) is False
        assert store.load_trust_state() is None

    def test_missing_state_loads_as_none(self, tmp_path: Path) -> None:
        """An initialized-but-empty store reports no state, not an error."""
        store = PluginStatsStore(_db(tmp_path))
        assert store.load_trust_state() is None

    def test_corrupt_state_degrades_to_fresh_ledger(self, tmp_path: Path) -> None:
        """Invalid JSON in the store is ignored; loader yields a DRAFT ledger."""
        db_path = _db(tmp_path)
        # Write a syntactically invalid state_json directly into the table.
        from leapflow.storage.duckdb_connect import connect

        conn = connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_trust_state (
                    key TEXT PRIMARY KEY DEFAULT 'singleton',
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO plugin_trust_state (key, state_json) "
                "VALUES ('singleton', ?)",
                ["{not valid json"],
            )
        finally:
            conn.close()

        store = PluginStatsStore(db_path)
        # load_trust_state swallows JSONDecodeError and returns None.
        assert store.load_trust_state() is None
        # The higher-level loader then produces a usable, empty ledger.
        ledger = _load_or_new_trust_ledger(store)
        assert isinstance(ledger, _PersistingTrustLedger)
        assert ledger.level("anything") is PluginTrustLevel.DRAFT

    def test_persisting_ledger_without_store_never_raises(self) -> None:
        """Transitions on a store-less persisting ledger are safe no-ops."""
        ledger = _PersistingTrustLedger(candidate_at=1, store=None)
        ledger.record_success("x")  # would trigger a flush if a store existed
        assert ledger.level("x") is PluginTrustLevel.CANDIDATE


class TestPersistingLedgerFlush:
    """_PersistingTrustLedger auto-flushes on level transitions only."""

    def test_flush_on_promotion_transition(self, tmp_path: Path) -> None:
        """Reaching a new trust level writes through to the store immediately."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = _PersistingTrustLedger(candidate_at=2, store=store)
        ledger.record_success("p")  # streak 1 — no transition, no write yet
        assert PluginStatsStore(_db(tmp_path)).load_trust_state() is None
        ledger.record_success("p")  # streak 2 — promote to CANDIDATE → flush

        state = PluginStatsStore(_db(tmp_path)).load_trust_state()
        assert state is not None
        restored = PluginTrustLedger.load_state(state)
        assert restored.level("p") is PluginTrustLevel.CANDIDATE

    def test_flush_on_hard_freeze(self, tmp_path: Path) -> None:
        """A hard failure flushes even though the reported level stays DRAFT."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = _PersistingTrustLedger(store=store)
        ledger.record_failure("q", hard=True)
        state = PluginStatsStore(_db(tmp_path)).load_trust_state()
        assert state is not None
        assert "q" in state.get("frozen", [])

    def test_load_state_returns_persisting_subclass(self, tmp_path: Path) -> None:
        """classmethod load_state on the subclass yields the subclass type."""
        store = PluginStatsStore(_db(tmp_path))
        seed = _PersistingTrustLedger(candidate_at=1, store=store)
        seed.record_success("r")
        restored = _load_or_new_trust_ledger(store)
        assert isinstance(restored, _PersistingTrustLedger)
        assert restored.level("r") is PluginTrustLevel.CANDIDATE


class TestPathDerivation:
    """Profile-scoped path derivation uses existing layout APIs."""

    def test_resolve_store_honors_explicit_path(self, tmp_path: Path) -> None:
        store = _resolve_stats_store(_db(tmp_path))
        assert isinstance(store, PluginStatsStore)

    def test_default_path_is_plugin_stats_beside_profile_dbs(self) -> None:
        """When derivable, the default path is plugin_stats.duckdb in db_dir."""
        path = _default_stats_db_path()
        if path is None:
            pytest.skip("No profile layout reachable in this environment")
        assert path.name == "plugin_stats.duckdb"
        # Sits in the same directory family as the other profile DuckDB stores.
        assert path.parent.name == "db"


class TestSinkWiringPersistence:
    """End-to-end: wiring restores state and persist_plugin_trust_state writes."""

    @pytest.fixture
    def reset_singletons(self, tmp_path: Path):
        """Isolate the process-global advisor + store around each test."""
        import leapflow.engine.session.session_factory as sf
        from leapflow.learning import plugin_advisor as pa

        saved_advisor = pa._default_advisor
        saved_store = sf._DEFAULT_STATS_STORE
        pa._default_advisor = None
        sf._DEFAULT_STATS_STORE = None
        try:
            yield sf, pa
        finally:
            pa._default_advisor = saved_advisor
            sf._DEFAULT_STATS_STORE = saved_store

    def test_wire_persist_reload_round_trip(self, tmp_path: Path, reset_singletons) -> None:
        """Wire a sink with an explicit db, earn trust, persist, reload it back."""
        sf, pa = reset_singletons
        db_path = _db(tmp_path)

        tracker = TurnUsageTracker()
        sf._wire_plugin_stats_sink(tracker, db_path=db_path)

        advisor = pa.get_default_advisor()
        assert advisor is not None
        ledger = advisor._trust_ledger
        # Earn trust directly on the wired ledger, then flush via the public API.
        for _ in range(60):
            ledger.record_success("wired_plugin")
        assert ledger.level("wired_plugin") is PluginTrustLevel.PRODUCTION

        assert persist_plugin_trust_state() is True

        # A fresh store instance over the same file sees the persisted state.
        state = PluginStatsStore(db_path).load_trust_state()
        assert state is not None
        restored = PluginTrustLedger.load_state(state)
        assert restored.level("wired_plugin") is PluginTrustLevel.PRODUCTION

    def test_second_wire_reuses_existing_advisor(self, tmp_path: Path, reset_singletons) -> None:
        """A subsequent wiring reuses the singleton and just attaches the sink."""
        sf, pa = reset_singletons
        first = TurnUsageTracker()
        sf._wire_plugin_stats_sink(first, db_path=_db(tmp_path))
        advisor_before = pa.get_default_advisor()

        second = TurnUsageTracker()
        sf._wire_plugin_stats_sink(second, db_path=_db(tmp_path))
        assert pa.get_default_advisor() is advisor_before

    def test_persist_without_store_returns_false(self, reset_singletons) -> None:
        """With no wired store, persist is a safe no-op returning False."""
        sf, pa = reset_singletons
        assert sf._DEFAULT_STATS_STORE is None
        assert persist_plugin_trust_state() is False


class TestUsageStateRoundTrip:
    """Rolling usage samples must outlive the process.

    Reliability scoring reads error rate and p95 latency to choose between
    competing plugins. Held only in memory, that evidence resets on every
    restart and the scorer silently falls back to "insufficient data".
    """

    def test_usage_samples_round_trip_preserves_stats(self, tmp_path: Path) -> None:
        """save → reopen → load reproduces the same aggregated statistics."""
        store = PluginStatsStore(_db(tmp_path))
        tracker = PluginUsageTracker()
        for _ in range(7):
            tracker.record("probe_tool", True, 10.0)
        for _ in range(3):
            tracker.record("probe_tool", False, 50.0)

        assert store.save_usage_state(tracker.to_state()) is True

        state = PluginStatsStore(_db(tmp_path)).load_usage_state()
        assert state is not None
        restored = PluginUsageTracker.load_state(state)

        samples = restored._samples["probe_tool"]
        assert len(samples) == 10
        assert sum(1 for s in samples if not s.ok) == 3
        # The durations survive, so p95 and average remain computable.
        assert max(s.duration_ms for s in samples) == 50.0

    def test_empty_and_missing_state_degrade_to_fresh_tracker(self, tmp_path: Path) -> None:
        """No stored usage yields an empty tracker rather than an error."""
        assert PluginStatsStore(_db(tmp_path)).load_usage_state() is None
        assert PluginUsageTracker.load_state({})._samples == {}

    def test_malformed_sample_rows_are_skipped(self) -> None:
        """A truncated blob loses samples, never startup."""
        state = {
            "max_samples_per_tool": 500,
            "samples": {
                "good_tool": [[1.0, True, 5.0], [2.0, False, 7.0]],
                "broken_tool": [[1.0, True], "not-a-row", [1.0, True, "abc"]],
                "wrong_shape": "not-a-list",
            },
        }
        restored = PluginUsageTracker.load_state(state)

        assert len(restored._samples["good_tool"]) == 2
        assert len(restored._samples.get("broken_tool", [])) == 0
        assert "wrong_shape" not in restored._samples

    def test_persisted_window_uses_tracker_sample_limit(self) -> None:
        """Persistence uses the tracker-configured sample window, not a new limit."""
        tracker = PluginUsageTracker(max_samples_per_tool=25)
        for i in range(40):
            tracker.record("busy_tool", True, float(i))

        rows = tracker.to_state()["samples"]["busy_tool"]
        assert len(rows) == 25
        # Truncated from the left: the newest sample is retained.
        assert rows[-1][2] == 39.0

    def test_no_path_store_usage_is_noop(self) -> None:
        """A store with no db_path reports failure instead of raising."""
        store = PluginStatsStore(None)
        assert store.save_usage_state({"samples": {}}) is False
        assert store.load_usage_state() is None


class TestUsageSinkWiring:
    """Wiring restores usage history and flushes it beside trust."""

    @pytest.fixture
    def reset_singletons(self, tmp_path: Path):
        import leapflow.engine.session.session_factory as sf
        from leapflow.learning import plugin_advisor as pa

        saved_advisor = pa._default_advisor
        saved_store = sf._DEFAULT_STATS_STORE
        pa._default_advisor = None
        sf._DEFAULT_STATS_STORE = None
        try:
            yield sf, pa
        finally:
            pa._default_advisor = saved_advisor
            sf._DEFAULT_STATS_STORE = saved_store

    def test_persist_flushes_usage_alongside_trust(
        self, tmp_path: Path, reset_singletons
    ) -> None:
        """One persist call writes both tables, so trust keeps its evidence."""
        sf, pa = reset_singletons
        db_path = _db(tmp_path)

        tracker = TurnUsageTracker()
        sf._wire_plugin_stats_sink(tracker, db_path=db_path)
        advisor = pa.get_default_advisor()
        assert advisor is not None

        advisor._usage_tracker.record("sink_tool", True, 12.0)
        advisor._usage_tracker.record("sink_tool", False, 30.0)
        assert persist_plugin_trust_state() is True

        usage_state = PluginStatsStore(db_path).load_usage_state()
        assert usage_state is not None
        assert len(usage_state["samples"]["sink_tool"]) == 2

    def test_wiring_restores_previous_usage_history(
        self, tmp_path: Path, reset_singletons
    ) -> None:
        """A fresh process inherits the reliability history of the last one."""
        sf, pa = reset_singletons
        db_path = _db(tmp_path)
        PluginStatsStore(db_path).save_usage_state(
            {
                "max_samples_per_tool": 500,
                "samples": {"legacy_tool": [[1.0, True, 4.0], [2.0, False, 9.0]]},
            }
        )

        sf._wire_plugin_stats_sink(TurnUsageTracker(), db_path=db_path)
        advisor = pa.get_default_advisor()
        assert advisor is not None

        assert len(advisor._usage_tracker._samples["legacy_tool"]) == 2
        # The restored tracker is still wired to the trust ledger.
        assert advisor._usage_tracker._trust_ledger is advisor._trust_ledger

    def test_corrupt_usage_state_degrades_to_empty_tracker(
        self, tmp_path: Path, reset_singletons
    ) -> None:
        """An unreadable usage blob must not block wiring."""
        sf, pa = reset_singletons
        db_path = _db(tmp_path)
        from leapflow.storage.duckdb_connect import connect

        conn = connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plugin_usage_state (
                    key TEXT PRIMARY KEY DEFAULT 'singleton',
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO plugin_usage_state (key, state_json) "
                "VALUES ('singleton', ?)",
                ["{not valid json"],
            )
        finally:
            conn.close()

        sf._wire_plugin_stats_sink(TurnUsageTracker(), db_path=db_path)
        advisor = pa.get_default_advisor()
        assert advisor is not None
        assert advisor._usage_tracker._samples == {}


class TestTrustTransitionHistory:
    """Trust transition time-series persistence and query."""

    def test_trust_transition_persisted_to_history_table(self, tmp_path: Path) -> None:
        """A promotion writes a row to plugin_trust_transitions with correct levels."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = _PersistingTrustLedger(candidate_at=5, store=store)
        for _ in range(5):
            ledger.record_success("alpha")
        assert ledger.level("alpha") is PluginTrustLevel.CANDIDATE

        records = trust_history(_db(tmp_path))
        assert len(records) == 1
        rec = records[0]
        assert rec.plugin_id == "alpha"
        assert rec.from_level == PluginTrustLevel.DRAFT
        assert rec.to_level == PluginTrustLevel.CANDIDATE
        assert rec.trigger == "success"
        assert rec.consecutive_ok == 5
        assert rec.consecutive_fail == 0
        assert rec.transitioned_at is not None

    def test_trust_history_query_by_plugin_id(self, tmp_path: Path) -> None:
        """Querying by plugin_id returns only matching records."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = _PersistingTrustLedger(candidate_at=2, store=store)
        # Drive two plugins to CANDIDATE.
        for _ in range(2):
            ledger.record_success("plug_a")
        for _ in range(2):
            ledger.record_success("plug_b")
        assert ledger.level("plug_a") is PluginTrustLevel.CANDIDATE
        assert ledger.level("plug_b") is PluginTrustLevel.CANDIDATE

        all_records = trust_history(_db(tmp_path))
        assert len(all_records) == 2

        a_records = trust_history(_db(tmp_path), plugin_id="plug_a")
        assert len(a_records) == 1
        assert a_records[0].plugin_id == "plug_a"

        b_records = trust_history(_db(tmp_path), plugin_id="plug_b")
        assert len(b_records) == 1
        assert b_records[0].plugin_id == "plug_b"

    def test_trust_history_cold_path_only(self, tmp_path: Path) -> None:
        """A single success (no promotion) writes NO row to trust_transitions."""
        store = PluginStatsStore(_db(tmp_path))
        ledger = _PersistingTrustLedger(candidate_at=5, store=store)
        ledger.record_success("beta")  # streak 1, no level change
        assert ledger.level("beta") is PluginTrustLevel.DRAFT

        records = trust_history(_db(tmp_path))
        assert records == []

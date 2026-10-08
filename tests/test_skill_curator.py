# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the Skill Curator — lifecycle management, persistence, and CLI."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from leapflow.skills.curator import (
    CurationState,
    SkillCurationEntry,
    SkillCurator,
)
from leapflow.skills.index import SkillEntry, SkillIndex, _BUILTIN_SKILLS_DIR


# ── In-memory store (implements SkillCurationStore protocol) ──

class InMemoryCurationStore:
    """In-memory implementation for testing without DuckDB."""

    def __init__(self) -> None:
        self._data: Dict[str, SkillCurationEntry] = {}

    def load_all(self) -> list[SkillCurationEntry]:
        return list(self._data.values())

    def load(self, skill_name: str) -> Optional[SkillCurationEntry]:
        return self._data.get(skill_name)

    def save(self, entry: SkillCurationEntry) -> None:
        self._data[entry.skill_name] = entry

    def delete(self, skill_name: str) -> bool:
        return self._data.pop(skill_name, None) is not None


# ── Fake EventBus for verifying event emission ──

class FakeEventBus:
    def __init__(self) -> None:
        self.events: List[tuple[str, Dict[str, Any]]] = []

    async def handle_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append((event_type, payload))


# ── Fixtures ──

@pytest.fixture
def store() -> InMemoryCurationStore:
    return InMemoryCurationStore()


@pytest.fixture
def curator(store: InMemoryCurationStore) -> SkillCurator:
    return SkillCurator(store, stale_after_days=14, archive_after_days=30)


# ── Three-state lifecycle tests ──

class TestLifecycleTransitions:
    """Test ACTIVE → STALE → ARCHIVED automatic transitions."""

    def test_new_skill_starts_active(self, curator: SkillCurator) -> None:
        curator.record_activity("my-skill")
        assert curator.get_state("my-skill") == CurationState.ACTIVE

    def test_unknown_skill_returns_active(self, curator: SkillCurator) -> None:
        assert curator.get_state("nonexistent") == CurationState.ACTIVE

    def test_active_to_stale_transition(self, store: InMemoryCurationStore) -> None:
        # Create a skill with last activity 15 days ago
        fifteen_days_ago = time.time() - (15 * 86400)
        entry = SkillCurationEntry(
            skill_name="old-skill",
            state=CurationState.ACTIVE,
            last_activity_at=fifteen_days_ago,
            created_at=fifteen_days_ago - 86400,
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        # Force sweep (bypass throttle)
        curator._last_sweep_time = 0.0
        report = curator.apply_automatic_transitions()

        assert curator.get_state("old-skill") == CurationState.STALE
        assert len(report.transitions) == 1
        assert report.transitions[0].from_state == CurationState.ACTIVE
        assert report.transitions[0].to_state == CurationState.STALE

    def test_stale_to_archived_transition(self, store: InMemoryCurationStore) -> None:
        # Create a stale skill with last activity 31 days ago
        thirty_one_days_ago = time.time() - (31 * 86400)
        entry = SkillCurationEntry(
            skill_name="ancient-skill",
            state=CurationState.STALE,
            last_activity_at=thirty_one_days_ago,
            created_at=thirty_one_days_ago - 86400,
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        curator._last_sweep_time = 0.0
        report = curator.apply_automatic_transitions()

        assert curator.get_state("ancient-skill") == CurationState.ARCHIVED
        assert len(report.transitions) == 1
        assert report.transitions[0].to_state == CurationState.ARCHIVED

    def test_stale_reactivates_on_activity(self, store: InMemoryCurationStore) -> None:
        entry = SkillCurationEntry(
            skill_name="sleepy-skill",
            state=CurationState.STALE,
            last_activity_at=time.time() - (20 * 86400),
            created_at=time.time() - (30 * 86400),
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        curator.record_activity("sleepy-skill")

        assert curator.get_state("sleepy-skill") == CurationState.ACTIVE

    def test_archived_does_not_auto_reactivate_on_activity(
        self, store: InMemoryCurationStore
    ) -> None:
        """Archived skills do NOT auto-reactivate; only manual reactivate works."""
        entry = SkillCurationEntry(
            skill_name="dead-skill",
            state=CurationState.ARCHIVED,
            last_activity_at=time.time() - (60 * 86400),
            created_at=time.time() - (90 * 86400),
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        # record_activity should only reactivate STALE, not ARCHIVED
        curator.record_activity("dead-skill")
        assert curator.get_state("dead-skill") == CurationState.ARCHIVED

    def test_recently_active_skill_not_transitioned(
        self, store: InMemoryCurationStore
    ) -> None:
        recent = time.time() - (3 * 86400)
        entry = SkillCurationEntry(
            skill_name="fresh-skill",
            state=CurationState.ACTIVE,
            last_activity_at=recent,
            created_at=recent - 86400,
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        curator._last_sweep_time = 0.0
        report = curator.apply_automatic_transitions()

        assert curator.get_state("fresh-skill") == CurationState.ACTIVE
        assert len(report.transitions) == 0


# ── Pin protection tests ──

class TestPinProtection:
    """Pinned skills are exempt from all automatic transitions."""

    def test_pinned_skill_not_staled(self, store: InMemoryCurationStore) -> None:
        old_time = time.time() - (20 * 86400)
        entry = SkillCurationEntry(
            skill_name="pinned-skill",
            state=CurationState.ACTIVE,
            pinned=True,
            last_activity_at=old_time,
            created_at=old_time - 86400,
        )
        store.save(entry)

        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        curator._last_sweep_time = 0.0
        report = curator.apply_automatic_transitions()

        assert curator.get_state("pinned-skill") == CurationState.ACTIVE
        assert len(report.transitions) == 0

    def test_pin_and_unpin(self, curator: SkillCurator) -> None:
        curator.record_activity("my-skill")
        curator.pin("my-skill")
        entry = curator.get_entry("my-skill")
        assert entry is not None and entry.pinned is True

        curator.unpin("my-skill")
        entry = curator.get_entry("my-skill")
        assert entry is not None and entry.pinned is False

    def test_pin_unknown_skill_creates_entry(self, curator: SkillCurator) -> None:
        curator.pin("brand-new")
        entry = curator.get_entry("brand-new")
        assert entry is not None
        assert entry.pinned is True
        assert entry.state == CurationState.ACTIVE


# ── Manual operations tests ──

class TestManualOperations:

    def test_manual_archive(self, curator: SkillCurator) -> None:
        curator.record_activity("target-skill")
        curator.archive("target-skill", "no longer needed")
        assert curator.get_state("target-skill") == CurationState.ARCHIVED
        entry = curator.get_entry("target-skill")
        assert entry is not None
        assert entry.archive_reason == "no longer needed"

    def test_manual_reactivate(self, curator: SkillCurator) -> None:
        curator.record_activity("target-skill")
        curator.archive("target-skill", "cleanup")
        curator.reactivate("target-skill")
        assert curator.get_state("target-skill") == CurationState.ACTIVE
        entry = curator.get_entry("target-skill")
        assert entry is not None
        assert entry.archive_reason is None

    def test_archive_unknown_raises(self, curator: SkillCurator) -> None:
        with pytest.raises(KeyError):
            curator.archive("no-such-skill")

    def test_reactivate_unknown_raises(self, curator: SkillCurator) -> None:
        with pytest.raises(KeyError):
            curator.reactivate("no-such-skill")

    def test_unpin_unknown_raises(self, curator: SkillCurator) -> None:
        with pytest.raises(KeyError):
            curator.unpin("no-such-skill")


# ── Activity recording tests ──

class TestActivityRecording:

    def test_record_activity_creates_entry(self, curator: SkillCurator) -> None:
        curator.record_activity("new-skill")
        entry = curator.get_entry("new-skill")
        assert entry is not None
        assert entry.last_activity_at is not None
        assert entry.state == CurationState.ACTIVE

    def test_record_activity_updates_timestamp(self, curator: SkillCurator) -> None:
        curator.record_activity("my-skill")
        t1 = curator.get_entry("my-skill").last_activity_at

        time.sleep(0.01)
        curator.record_activity("my-skill")
        t2 = curator.get_entry("my-skill").last_activity_at
        assert t2 > t1


# ── Report and query tests ──

class TestCurationReport:

    def test_report_counts(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("a", CurationState.ACTIVE, created_at=now))
        store.save(SkillCurationEntry("b", CurationState.ACTIVE, pinned=True, created_at=now))
        store.save(SkillCurationEntry("c", CurationState.STALE, created_at=now))
        store.save(SkillCurationEntry("d", CurationState.ARCHIVED, created_at=now))

        curator = SkillCurator(store)
        report = curator.get_curation_report()

        assert report.total == 4
        assert report.active == 2
        assert report.stale == 1
        assert report.archived == 1
        assert report.pinned == 1

    def test_list_by_state(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("a", CurationState.ACTIVE, created_at=now))
        store.save(SkillCurationEntry("b", CurationState.STALE, created_at=now))
        store.save(SkillCurationEntry("c", CurationState.ARCHIVED, created_at=now))

        curator = SkillCurator(store)
        assert len(curator.list_by_state(CurationState.ACTIVE)) == 1
        assert len(curator.list_by_state(CurationState.STALE)) == 1
        assert len(curator.list_by_state(CurationState.ARCHIVED)) == 1

    def test_get_archived_names(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("a", CurationState.ACTIVE, created_at=now))
        store.save(SkillCurationEntry("b", CurationState.ARCHIVED, created_at=now))

        curator = SkillCurator(store)
        assert curator.get_archived_names() == {"b"}

    def test_get_stale_names(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("a", CurationState.STALE, created_at=now))
        store.save(SkillCurationEntry("b", CurationState.ACTIVE, created_at=now))

        curator = SkillCurator(store)
        assert curator.get_stale_names() == {"a"}


# ── EventBus integration tests ──

class TestEventBusIntegration:

    def test_transition_emits_event(self, store: InMemoryCurationStore) -> None:
        """Transition events are emitted via EventBus.handle_event."""
        import asyncio

        bus = FakeEventBus()
        curator = SkillCurator(store, event_bus=bus)

        curator.record_activity("test-skill")

        async def run() -> None:
            curator.archive("test-skill", "test reason")

        asyncio.run(run())

        # Check that event was emitted
        assert len(bus.events) >= 1
        event_type, payload = bus.events[-1]
        assert event_type == "skill.curation_changed"
        assert payload["skill_name"] == "test-skill"
        assert payload["to_state"] == "archived"

    def test_no_event_without_bus(self, curator: SkillCurator) -> None:
        """No crash when event_bus is None."""
        curator.record_activity("test-skill")
        curator.archive("test-skill")  # Should not raise


# ── SkillIndex integration tests ──

class TestSkillIndexIntegration:

    def test_archived_skills_excluded_from_index(self, tmp_path) -> None:
        """SkillIndex filters out archived skills."""
        index = SkillIndex(tmp_path)
        # Inject some entries into the cache
        entries = [
            SkillEntry(name="active-skill", description="Active"),
            SkillEntry(name="archived-skill", description="Archived"),
        ]
        index._entries = entries
        index._cache_time = time.monotonic()

        # Filter with archived set
        result = index.get_entries(archived={"archived-skill"})
        names = [e.name for e in result]
        assert "active-skill" in names
        assert "archived-skill" not in names

    def test_include_archived_overrides_filter(self, tmp_path) -> None:
        index = SkillIndex(tmp_path)
        entries = [
            SkillEntry(name="active-skill", description="Active"),
            SkillEntry(name="archived-skill", description="Archived"),
        ]
        index._entries = entries
        index._cache_time = time.monotonic()

        result = index.get_entries(
            archived={"archived-skill"}, include_archived=True
        )
        names = [e.name for e in result]
        assert "archived-skill" in names

    def test_fresh_scan_discovers_all_builtin_skills(self, tmp_path) -> None:
        """A fresh skills_dir must surface every bundled builtin SKILL.md."""
        builtin = sorted(p.parent.name for p in _BUILTIN_SKILLS_DIR.glob("*/SKILL.md"))
        assert len(builtin) >= 1, "expected bundled builtin skills to exist"
        index = SkillIndex(tmp_path)
        names = {e.name for e in index.get_entries()}
        assert set(builtin).issubset(names)

    def test_legacy_empty_snapshot_is_rejected_and_self_heals(self, tmp_path) -> None:
        """Regression: a pre-existing bare-list '[]' snapshot must NOT mask the
        bundled builtin skills — the exact defect that reported only 3 skills."""
        import json

        (tmp_path / ".skills_index.json").write_text("[]", encoding="utf-8")
        index = SkillIndex(tmp_path)
        names = {e.name for e in index.get_entries()}
        assert "web_research" in names and "code_review" in names
        # The snapshot is rewritten in the new signed format.
        rewritten = json.loads((tmp_path / ".skills_index.json").read_text())
        assert isinstance(rewritten, dict)
        assert rewritten.get("schema") == 2 and rewritten.get("signature")
        assert len(rewritten.get("entries", [])) == len(names)

    def test_signed_snapshot_is_reused_then_busted_by_new_skill(self, tmp_path) -> None:
        """A matching signature reuses L2; adding a skill changes the signature."""
        base = {e.name for e in SkillIndex(tmp_path).get_entries()}
        # Second instance reuses the signed snapshot (same source signature).
        assert {e.name for e in SkillIndex(tmp_path).get_entries()} == base
        # Add a user skill → signature drift → rescan picks it up.
        skill = tmp_path / "custom_skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: custom_skill\ndescription: d\n---\n", encoding="utf-8"
        )
        after = {e.name for e in SkillIndex(tmp_path).get_entries()}
        assert "custom_skill" in after
        assert base.issubset(after)


# ── DuckDB persistence tests ──

class TestDuckDBPersistence:

    def test_round_trip(self) -> None:
        """Save and load entries through DuckDB store."""
        import duckdb
        from leapflow.storage.skill_curation_store import DuckDBSkillCurationStore

        conn = duckdb.connect(":memory:")

        class InMemHolder:
            @property
            def connection(self):
                return conn

            @property
            def db_path(self):
                from pathlib import Path
                return Path(":memory:")

            def close(self):
                conn.close()

        holder = InMemHolder()
        store = DuckDBSkillCurationStore(holder)

        now = time.time()
        entry = SkillCurationEntry(
            skill_name="db-skill",
            state=CurationState.STALE,
            pinned=True,
            last_activity_at=now - 86400,
            created_at=now - (10 * 86400),
            archive_reason=None,
        )
        store.save(entry)

        loaded = store.load("db-skill")
        assert loaded is not None
        assert loaded.skill_name == "db-skill"
        assert loaded.state == CurationState.STALE
        assert loaded.pinned is True
        assert loaded.last_activity_at is not None

        # Update state
        entry.state = CurationState.ARCHIVED
        entry.archive_reason = "manual"
        store.save(entry)

        loaded2 = store.load("db-skill")
        assert loaded2 is not None
        assert loaded2.state == CurationState.ARCHIVED
        assert loaded2.archive_reason == "manual"

        # Load all
        all_entries = store.load_all()
        assert len(all_entries) == 1

        # Delete
        assert store.delete("db-skill") is True
        assert store.load("db-skill") is None
        assert store.delete("nonexistent") is False

        conn.close()


# ── Sweep throttling tests ──

class TestSweepThrottling:

    def test_sweep_is_throttled(self, store: InMemoryCurationStore) -> None:
        curator = SkillCurator(store, stale_after_days=14, archive_after_days=30)
        now = time.time()
        old = now - (20 * 86400)
        store.save(SkillCurationEntry(
            "throttle-test", CurationState.ACTIVE,
            last_activity_at=old, created_at=old,
        ))
        # First sweep — should produce transitions
        curator._last_sweep_time = 0.0
        r1 = curator.apply_automatic_transitions()
        assert len(r1.transitions) == 1

        # Reset state for second test
        store.save(SkillCurationEntry(
            "throttle-test2", CurationState.ACTIVE,
            last_activity_at=old, created_at=old,
        ))
        curator._invalidate_cache()

        # Second sweep — should be throttled (no new transitions)
        r2 = curator.apply_automatic_transitions()
        assert len(r2.transitions) == 0


# ── CLI command integration tests ──

class TestCLICommandIntegration:

    def _make_ctx(self, curator: SkillCurator) -> MagicMock:
        ctx = MagicMock()
        ctx.skill_curator = curator
        ctx.registry = None
        ctx.skill_lib = None
        return ctx

    def test_curator_report_command(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("a", CurationState.ACTIVE, created_at=now))
        store.save(SkillCurationEntry("b", CurationState.STALE, created_at=now))

        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "")
        assert result["ok"] is True
        assert "Total: 2" in result["message"]

    def test_curator_archive_command(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry("my-skill", CurationState.ACTIVE, created_at=now))

        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "archive my-skill old and tired")
        assert result["ok"] is True
        assert "archived" in result["message"]
        assert curator.get_state("my-skill") == CurationState.ARCHIVED

    def test_curator_reactivate_command(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry(
            "my-skill", CurationState.ARCHIVED, created_at=now,
            archive_reason="test",
        ))

        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "reactivate my-skill")
        assert result["ok"] is True
        assert curator.get_state("my-skill") == CurationState.ACTIVE

    def test_curator_pin_command(self, store: InMemoryCurationStore) -> None:
        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "pin my-skill")
        assert result["ok"] is True
        entry = curator.get_entry("my-skill")
        assert entry is not None and entry.pinned is True

    def test_curator_unpin_command(self, store: InMemoryCurationStore) -> None:
        now = time.time()
        store.save(SkillCurationEntry(
            "my-skill", CurationState.ACTIVE, pinned=True, created_at=now,
        ))

        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "unpin my-skill")
        assert result["ok"] is True

    def test_curator_sweep_command(self, store: InMemoryCurationStore) -> None:
        old = time.time() - (20 * 86400)
        store.save(SkillCurationEntry(
            "old-skill", CurationState.ACTIVE,
            last_activity_at=old, created_at=old,
        ))

        curator = SkillCurator(store, stale_after_days=14)
        curator._last_sweep_time = 0.0  # bypass throttle
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "sweep")
        assert result["ok"] is True
        assert "Sweep complete" in result["message"]
        assert "1 transition" in result["message"]

    def test_curator_not_initialized(self) -> None:
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = MagicMock()
        ctx.skill_curator = None
        result = _execute_skill(ctx, "skill curator", "")
        assert result["ok"] is False
        assert "not initialized" in result["message"]

    def test_curator_unknown_subcommand(self, store: InMemoryCurationStore) -> None:
        curator = SkillCurator(store)
        from leapflow.cli.commands.slash_handlers import _execute_skill

        ctx = self._make_ctx(curator)
        result = _execute_skill(ctx, "skill curator", "invalid")
        assert result["ok"] is False

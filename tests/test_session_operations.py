# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for session operations: pin/unpin, hide/unhide, archive, list filtering."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def store(tmp_path: Path):
    """Create a DuckDBConversationStore backed by a temp database."""
    from leapflow.storage.conversation_store import DuckDBConversationStore

    db_path = tmp_path / "session_ops.duckdb"
    s = DuckDBConversationStore(db_path)
    yield s
    s.close()


@pytest.fixture()
def seeded_store(store):
    """Store pre-populated with three sessions."""
    store.create_session("s1", title="Session One")
    store.create_session("s2", title="Session Two")
    store.create_session("s3", title="Session Three")
    return store


# ── Pin / Unpin ──────────────────────────────────────────────────────


class TestPinUnpin:
    def test_pin_session(self, seeded_store):
        seeded_store.pin_session("s1")
        session = seeded_store.get_session("s1")
        assert session is not None
        assert session.pinned is True

    def test_unpin_session(self, seeded_store):
        seeded_store.pin_session("s1")
        seeded_store.unpin_session("s1")
        session = seeded_store.get_session("s1")
        assert session is not None
        assert session.pinned is False

    def test_pinned_sessions_sort_first(self, seeded_store):
        """Pinned sessions appear before unpinned in listings."""
        # Pin the oldest session
        seeded_store.pin_session("s1")
        sessions = seeded_store.list_sessions(limit=10, active_only=False)
        assert len(sessions) >= 2
        # First session in list should be the pinned one
        assert sessions[0].session_id == "s1"
        assert sessions[0].pinned is True

    def test_pin_idempotent(self, seeded_store):
        """Pinning an already-pinned session is a no-op."""
        seeded_store.pin_session("s1")
        seeded_store.pin_session("s1")
        session = seeded_store.get_session("s1")
        assert session is not None
        assert session.pinned is True


# ── Hide / Unhide ────────────────────────────────────────────────────


class TestHideUnhide:
    def test_hide_session(self, seeded_store):
        seeded_store.hide_session("s2")
        session = seeded_store.get_session("s2")
        assert session is not None
        assert session.hidden is True

    def test_unhide_session(self, seeded_store):
        seeded_store.hide_session("s2")
        seeded_store.unhide_session("s2")
        session = seeded_store.get_session("s2")
        assert session is not None
        assert session.hidden is False

    def test_hidden_excluded_from_default_list(self, seeded_store):
        """Hidden sessions are excluded from default list_sessions."""
        seeded_store.hide_session("s2")
        sessions = seeded_store.list_sessions(limit=10, active_only=False)
        ids = [s.session_id for s in sessions]
        assert "s2" not in ids

    def test_hidden_included_with_flag(self, seeded_store):
        """Hidden sessions appear when include_hidden=True."""
        seeded_store.hide_session("s2")
        sessions = seeded_store.list_sessions(limit=10, active_only=False, include_hidden=True)
        ids = [s.session_id for s in sessions]
        assert "s2" in ids

    def test_hide_idempotent(self, seeded_store):
        seeded_store.hide_session("s2")
        seeded_store.hide_session("s2")
        session = seeded_store.get_session("s2")
        assert session is not None
        assert session.hidden is True


# ── Archive ──────────────────────────────────────────────────────────


class TestArchive:
    def test_archive_marks_inactive(self, seeded_store):
        seeded_store.archive_session("s3")
        session = seeded_store.get_session("s3")
        assert session is not None
        assert session.is_active is False

    def test_archived_excluded_from_active_list(self, seeded_store):
        """Archived (inactive) sessions excluded by default."""
        seeded_store.archive_session("s3")
        sessions = seeded_store.list_sessions(limit=10, active_only=True)
        ids = [s.session_id for s in sessions]
        assert "s3" not in ids

    def test_archived_included_with_flag(self, seeded_store):
        """Archived sessions appear with include_archived=True."""
        seeded_store.archive_session("s3")
        sessions = seeded_store.list_sessions(
            limit=10, active_only=True, include_archived=True
        )
        ids = [s.session_id for s in sessions]
        assert "s3" in ids


# ── List Filtering ───────────────────────────────────────────────────


class TestListFiltering:
    def test_default_list_excludes_hidden_and_archived(self, seeded_store):
        seeded_store.hide_session("s1")
        seeded_store.archive_session("s2")
        sessions = seeded_store.list_sessions(limit=10)
        ids = [s.session_id for s in sessions]
        assert "s1" not in ids
        assert "s2" not in ids
        assert "s3" in ids

    def test_list_all(self, seeded_store):
        """With both flags, all sessions appear."""
        seeded_store.hide_session("s1")
        seeded_store.archive_session("s2")
        sessions = seeded_store.list_sessions(
            limit=10, active_only=False, include_hidden=True, include_archived=True,
        )
        ids = [s.session_id for s in sessions]
        assert "s1" in ids
        assert "s2" in ids
        assert "s3" in ids


# ── CLI Handler (unit) ───────────────────────────────────────────────


class TestSessionHandler:
    """Unit tests for the CLI handler payload builder."""

    def test_build_payload_list(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "list")
        assert result["ok"] is True
        assert "Sessions:" in result["message"]

    def test_build_payload_pin(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "pin s1")
        assert result["ok"] is True
        assert "pinned" in result["message"]
        session = seeded_store.get_session("s1")
        assert session.pinned is True

    def test_build_payload_archive(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "archive s2")
        assert result["ok"] is True
        assert "archived" in result["message"]

    def test_build_payload_hide(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "hide s3")
        assert result["ok"] is True
        session = seeded_store.get_session("s3")
        assert session.hidden is True

    def test_build_payload_not_found(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "pin nonexistent")
        assert result["ok"] is False
        assert "not found" in result["message"]

    def test_build_payload_no_store(self):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            pass

        result = build_session_payload(FakeCtx(), "list")
        assert result["ok"] is False

    def test_build_payload_unknown_subcommand(self, seeded_store):
        from leapflow.cli.commands.session_handler import build_session_payload

        class FakeCtx:
            _conversation_store = seeded_store

        result = build_session_payload(FakeCtx(), "destroy s1")
        assert result["ok"] is False
        assert "Unknown" in result["message"]


# ── Command Registry ─────────────────────────────────────────────────


class TestCommandRegistration:
    def test_session_command_registered(self):
        from leapflow.cli.commands.registry import resolve_command

        cmd = resolve_command("session")
        assert cmd is not None
        assert cmd.name == "session"

    def test_session_subcommands_registered(self):
        from leapflow.cli.commands.registry import resolve_command

        for sub in ("session archive", "session pin", "session unpin", "session hide", "session unhide"):
            cmd = resolve_command(sub)
            assert cmd is not None, f"/{sub} not registered"
            assert cmd.name == sub

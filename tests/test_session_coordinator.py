# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for SessionCoordinator — session engine resolution, workspace mismatch,
pagination, artifact collection, and JSON parsing.

Uses lightweight fakes: no daemon startup, no LLM, deterministic/offline.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.daemon.session_coordinator import (
    SessionCoordinator,
    _parse_session_json,
)


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeEngine:
    """Minimal engine stub exposing only the attributes SessionCoordinator reads."""

    def __init__(
        self,
        session_id: str = "",
        turn_count: int = 0,
        context_token_count: int = 0,
    ) -> None:
        self._current_session_id = session_id
        self.turn_count = turn_count
        self.context_token_count = context_token_count


class _FakeSessionCtx:
    """Stands in for ``SessionExecutionContext`` returned by the registry."""

    def __init__(self, session_id: str, engine: _FakeEngine | None = None) -> None:
        self.session_id = session_id
        self.engine = engine or _FakeEngine(session_id=session_id)


class _FakeRegistry:
    """Minimal stand-in for ``SessionRegistry`` supporting get/most_recent."""

    def __init__(self, contexts: dict[str, _FakeSessionCtx] | None = None) -> None:
        self._contexts = dict(contexts or {})

    def get(self, session_id: str) -> _FakeSessionCtx | None:
        return self._contexts.get(session_id)

    def most_recent_any_client(self) -> _FakeSessionCtx | None:
        if not self._contexts:
            return None
        return list(self._contexts.values())[-1]


class _FakeSession:
    """Minimal session metadata stub for get_detail tests."""

    def __init__(
        self,
        session_id: str = "sess-1",
        cwd: str = "",
        message_count: int = 0,
        **kwargs: Any,
    ) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.message_count = message_count
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeStore:
    """Minimal conversation store: get_session + get_messages with pagination."""

    def __init__(
        self,
        session: _FakeSession | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._session = session
        self._messages = list(messages or [])

    def get_session(self, session_id: str) -> _FakeSession | None:
        if self._session and self._session.session_id == session_id:
            return self._session
        return None

    def get_messages(
        self,
        session_id: str,
        limit: int = 200,
        offset: int = 0,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        return self._messages[offset : offset + limit]


def _make_ctx(
    engine: _FakeEngine | None = None,
    store: _FakeStore | None = None,
    settings: Any = None,
) -> SimpleNamespace:
    ctx = SimpleNamespace()
    ctx.engine = engine or _FakeEngine()
    ctx._conversation_store = store
    ctx.settings = settings or SimpleNamespace(workspace_root=os.getcwd())
    return ctx


# ── resolve_session_engine ───────────────────────────────────────────────────


def test_resolve_explicit_session() -> None:
    """Branch 1: explicit session_id found in registry → returns that engine."""
    coord = SessionCoordinator()
    eng = _FakeEngine(session_id="s1", turn_count=5)
    sctx = _FakeSessionCtx("s1", engine=eng)
    coord._session_registry = _FakeRegistry({"s1": sctx})

    result_engine, result_sid = coord.resolve_session_engine(
        _make_ctx(), session_id="s1",
    )
    assert result_engine is eng
    assert result_sid == "s1"


def test_resolve_most_recent_any_client_when_no_session_id() -> None:
    """Branch 2: no session_id → falls back to most_recent_any_client (aggregate)."""
    coord = SessionCoordinator()
    eng2 = _FakeEngine(session_id="s2")
    coord._session_registry = _FakeRegistry({
        "s1": _FakeSessionCtx("s1"),
        "s2": _FakeSessionCtx("s2", engine=eng2),
    })

    result_engine, result_sid = coord.resolve_session_engine(_make_ctx(), session_id="")
    assert result_engine is eng2
    assert result_sid == "s2"


def test_resolve_base_engine_fallback_no_registry() -> None:
    """Branch 3: no registry (in-process mode) → returns base engine from ctx."""
    coord = SessionCoordinator()
    base = _FakeEngine(session_id="base")
    ctx = _make_ctx(engine=base)

    result_engine, result_sid = coord.resolve_session_engine(ctx, session_id="")
    assert result_engine is base
    assert result_sid == "base"


def test_resolve_none_ctx_returns_none() -> None:
    """ctx=None → (None, '') without crashing."""
    coord = SessionCoordinator()
    engine, sid = coord.resolve_session_engine(None)
    assert engine is None
    assert sid == ""


def test_resolve_explicit_session_not_found_uses_base() -> None:
    """explicit session_id not found + no fallback → base engine."""
    coord = SessionCoordinator()
    coord._session_registry = _FakeRegistry({})
    base = _FakeEngine(session_id="base")
    ctx = _make_ctx(engine=base)

    engine, sid = coord.resolve_session_engine(ctx, session_id="unknown")
    assert engine is base


# ── get_detail: workspace mismatch ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_detail_workspace_mismatch() -> None:
    """Session cwd ≠ requested workspace → workspace_mismatch error."""
    coord = SessionCoordinator()
    session = _FakeSession(session_id="s1", cwd="/home/alice/project-a")
    store = _FakeStore(session=session, messages=[])
    ctx = _make_ctx(store=store)

    result = await coord.get_detail(
        ctx,
        SimpleNamespace(workspace_root="/home/bob/project-b"),
        "s1",
        workspace_root="/home/bob/project-b",
    )
    assert result["ok"] is False
    assert result["code"] == "workspace_mismatch"
    assert result["workspace_mismatch"] is True


# ── get_detail: pagination ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_detail_pagination_has_more() -> None:
    """When the store returns more rows than limit, has_more is True and
    result is trimmed to limit."""
    coord = SessionCoordinator()
    msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(6)]
    session = _FakeSession(session_id="s1", message_count=10)
    store = _FakeStore(session=session, messages=msgs)
    ctx = _make_ctx(store=store)
    settings = SimpleNamespace(workspace_root=os.getcwd())

    result = await coord.get_detail(ctx, settings, "s1", limit=4, offset=0)
    assert result["ok"] is True
    # query_limit = limit + 1 = 5, store returns 5 out of 6, so has_more
    assert result["has_more"] is True
    assert len(result["messages"]) == 4
    assert result["limit"] == 4
    assert result["offset"] == 0


# ── get_detail: store unavailable ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_detail_store_unavailable() -> None:
    """Missing conversation store returns a structured error, never raises."""
    coord = SessionCoordinator()
    ctx = _make_ctx(store=None)
    settings = SimpleNamespace(workspace_root=os.getcwd())

    result = await coord.get_detail(ctx, settings, "s1")
    assert result["ok"] is False
    assert result["code"] == "store_unavailable"


# ── _collect_session_artifacts ───────────────────────────────────────────────


def test_collect_artifacts_max_five_and_total_char_bound(tmp_path: Path) -> None:
    """At most 5 artifacts, and total character content respects the budget."""
    coord = SessionCoordinator()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Create 7 files, each with 4000 chars of content.
    messages: list[dict[str, Any]] = []
    for i in range(7):
        fp = workspace / f"file_{i}.txt"
        fp.write_text("x" * 4000, encoding="utf-8")
        messages.append({
            "role": "tool",
            "tool_name": "file_write",
            "content": json.dumps({"path": str(fp)}),
        })

    artifacts = coord._collect_session_artifacts("sess-1", messages, workspace)
    assert len(artifacts) <= 5
    total_chars = sum(
        len(str(a.get("content_excerpt", "")))
        for a in artifacts
        if a.get("status") == "included"
    )
    assert total_chars <= 16_000


def test_collect_artifacts_excludes_outside_workspace(tmp_path: Path) -> None:
    """Paths outside the workspace boundary are skipped."""
    coord = SessionCoordinator()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "other" / "secret.txt"
    outside.parent.mkdir(parents=True)
    outside.write_text("secret", encoding="utf-8")

    messages = [
        {
            "role": "tool",
            "tool_name": "file_write",
            "content": json.dumps({"path": str(outside)}),
        }
    ]
    artifacts = coord._collect_session_artifacts("sess-1", messages, workspace)
    assert len(artifacts) == 1
    assert artifacts[0]["status"] == "skipped"
    assert "outside workspace" in artifacts[0].get("reason", "")


# ── _parse_session_json ──────────────────────────────────────────────────────


def test_parse_session_json_fenced() -> None:
    """Fenced ```json ... ``` blocks are unwrapped before parsing."""
    raw = textwrap.dedent("""\
        ```json
        {"story": "test narrative", "insights": []}
        ```
    """)
    result = _parse_session_json(raw)
    assert isinstance(result, dict)
    assert result["story"] == "test narrative"


def test_parse_session_json_outermost_object() -> None:
    """Non-fenced text with an embedded JSON object → outermost {} extracted."""
    raw = 'Some preamble {"key": 42, "nested": {"a": 1}} trailing text'
    result = _parse_session_json(raw)
    assert isinstance(result, dict)
    assert result["key"] == 42


def test_parse_session_json_returns_none_on_garbage() -> None:
    assert _parse_session_json("not json at all") is None


# ── _session_workspace_mismatch: path normalization ──────────────────────────


def test_session_workspace_mismatch_normalized_paths() -> None:
    """Trailing slashes and symlink-equivalent paths resolve to the same value."""
    coord = SessionCoordinator()
    # Both resolve to the same physical path → no mismatch
    session = _FakeSession(cwd="/tmp/./project")
    workspace = Path("/tmp/project").resolve()
    assert coord._session_workspace_mismatch(session, workspace) is None


def test_session_workspace_mismatch_detects_real_diff() -> None:
    """Different resolved paths produce a mismatch dict."""
    coord = SessionCoordinator()
    session = _FakeSession(session_id="s1", cwd="/home/alice/proj-a")
    workspace = Path("/home/bob/proj-b").resolve()
    result = coord._session_workspace_mismatch(session, workspace)
    assert result is not None
    assert result["session_id"] == "s1"

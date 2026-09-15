# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the daemon SessionRegistry (Stage 3, P3-2a).

Pure infrastructure tests with fake engine/working-memory factories: every
session gets a workspace-scoped engine (including the primary session), acquire
is idempotent per session/workspace, and bounds (max sessions, idle TTL) evict
only non-primary contexts.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from leapflow.daemon.session_registry import SessionRegistry, WorkspaceMismatchError


class _FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name


def _make_registry(**kwargs):
    built: list = []

    def build_engine(base, sid, wm, workspace_root):
        built.append((sid, wm, Path(workspace_root)))
        return _FakeEngine(f"engine-{sid}")

    def build_wm():
        return object()  # a fresh, distinct working memory per call

    reg = SessionRegistry(
        base_engine=_FakeEngine("base"),
        build_engine=build_engine,
        build_working_memory=build_wm,
        **kwargs,
    )
    return reg, built


@pytest.mark.asyncio
async def test_primary_session_gets_workspace_scoped_engine() -> None:
    reg, built = _make_registry()
    ctx = await reg.acquire("s1", workspace_root="/tmp/work-a")
    assert ctx.engine.name == "engine-s1"
    assert ctx.workspace_root == Path("/tmp/work-a").resolve()
    assert len(built) == 1
    assert built[0][0] == "s1"
    assert built[0][2] == Path("/tmp/work-a").resolve()


@pytest.mark.asyncio
async def test_second_session_gets_isolated_engine() -> None:
    reg, built = _make_registry()
    a = await reg.acquire("s1", workspace_root="/tmp/work-a")
    b = await reg.acquire("s2", workspace_root="/tmp/work-b")
    assert a.engine.name == "engine-s1" and b.engine.name == "engine-s2"
    assert a.engine is not b.engine
    assert len(built) == 2 and built[1][0] == "s2"


@pytest.mark.asyncio
async def test_acquire_is_idempotent_per_session() -> None:
    reg, built = _make_registry()
    assert await reg.acquire("s1", workspace_root="/tmp/work-a") is await reg.acquire("s1", workspace_root="/tmp/work-a")
    b1 = await reg.acquire("s2", workspace_root="/tmp/work-b")
    b2 = await reg.acquire("s2", workspace_root="/tmp/work-b")
    assert b1 is b2 and len(built) == 2   # each session engine built exactly once


@pytest.mark.asyncio
async def test_acquire_rejects_same_session_from_different_workspace() -> None:
    reg, _ = _make_registry()
    await reg.acquire("s1", workspace_root="/tmp/work-a")

    with pytest.raises(WorkspaceMismatchError) as exc:
        await reg.acquire("s1", workspace_root="/tmp/work-b")

    assert exc.value.session_id == "s1"
    assert exc.value.expected == Path("/tmp/work-a").resolve()
    assert exc.value.requested == Path("/tmp/work-b").resolve()


@pytest.mark.asyncio
async def test_max_sessions_evicts_oldest_non_primary() -> None:
    reg, _ = _make_registry(max_sessions=2)
    await reg.acquire("primary", workspace_root="/tmp/primary")
    await reg.acquire("s2", workspace_root="/tmp/s2")
    await reg.acquire("s3", workspace_root="/tmp/s3")
    ids = reg.session_ids()
    assert "primary" in ids and "s3" in ids and "s2" not in ids


@pytest.mark.asyncio
async def test_idle_ttl_evicts_only_non_primary() -> None:
    reg, _ = _make_registry(idle_ttl_s=0.05)
    await reg.acquire("primary", workspace_root="/tmp/primary")
    await reg.acquire("s2", workspace_root="/tmp/s2")
    await asyncio.sleep(0.08)
    await reg.acquire("primary", workspace_root="/tmp/primary")  # acquire triggers idle eviction
    ids = reg.session_ids()
    assert "primary" in ids and "s2" not in ids

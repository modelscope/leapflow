"""Tests for the daemon SessionRegistry (Stage 3, P3-2a).

Pure infrastructure tests with fake engine/working-memory factories: the primary
session reuses the base engine, additional sessions get isolated engines, acquire
is idempotent per session, and bounds (max sessions, idle TTL) evict only
non-primary contexts.
"""
from __future__ import annotations

import asyncio

import pytest

from leapflow.daemon.session_registry import SessionRegistry


class _FakeEngine:
    def __init__(self, name: str) -> None:
        self.name = name


def _make_registry(**kwargs):
    built: list = []

    def build_engine(base, sid, wm):
        built.append((sid, wm))
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
async def test_primary_session_reuses_base_engine() -> None:
    reg, built = _make_registry()
    ctx = await reg.acquire("s1")
    assert ctx.engine.name == "base"      # first session uses the base engine
    assert built == []                    # no per-session engine built for the primary


@pytest.mark.asyncio
async def test_second_session_gets_isolated_engine() -> None:
    reg, built = _make_registry()
    a = await reg.acquire("s1")           # primary → base
    b = await reg.acquire("s2")           # second → isolated
    assert a.engine.name == "base" and b.engine.name == "engine-s2"
    assert a.engine is not b.engine
    assert len(built) == 1 and built[0][0] == "s2"


@pytest.mark.asyncio
async def test_acquire_is_idempotent_per_session() -> None:
    reg, built = _make_registry()
    assert await reg.acquire("s1") is await reg.acquire("s1")
    b1 = await reg.acquire("s2")
    b2 = await reg.acquire("s2")
    assert b1 is b2 and len(built) == 1   # s2 engine built exactly once


@pytest.mark.asyncio
async def test_max_sessions_evicts_oldest_non_primary() -> None:
    reg, _ = _make_registry(max_sessions=2)
    await reg.acquire("primary")          # base
    await reg.acquire("s2")               # isolated
    await reg.acquire("s3")               # at cap → evict oldest non-primary (s2)
    ids = reg.session_ids()
    assert "primary" in ids and "s3" in ids and "s2" not in ids


@pytest.mark.asyncio
async def test_idle_ttl_evicts_only_non_primary() -> None:
    reg, _ = _make_registry(idle_ttl_s=0.05)
    await reg.acquire("primary")
    await reg.acquire("s2")
    await asyncio.sleep(0.08)
    await reg.acquire("primary")          # acquire triggers idle eviction
    ids = reg.session_ids()
    assert "primary" in ids and "s2" not in ids

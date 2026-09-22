# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for ApprovalCoordinator — future resolution, batch deny, route
management, pruning, decision normalization, and idempotent resolution.

Uses real asyncio.Future instances and the real coordinator class; no daemon
startup or network I/O required.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.daemon.approval_coordinator import ApprovalCoordinator


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_pending(
    coord: ApprovalCoordinator,
    pending_id: str,
    request_id: str = "",
    queue: asyncio.Queue | None = None,
) -> asyncio.Future:
    """Inject a synthetic pending entry and return its future."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    coord._approval_pending[pending_id] = {
        "request": {"pending_id": pending_id, "request_id": request_id or pending_id},
        "future": future,
        "queue": queue or asyncio.Queue(),
        "created_at": 0.0,
    }
    return future


# ── resolve / cancel ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_sets_future_result_and_cleans_state() -> None:
    """resolve() sets the pending future with the normalized decision."""
    coord = ApprovalCoordinator()
    future = _make_pending(coord, "p1")

    result = await coord.resolve("p1", "allow_once", reason="user said ok")
    assert result["ok"] is True
    assert result["decision"] == "allow_once"

    # The future must have been resolved.
    assert future.done()
    decision_payload = future.result()
    assert decision_payload["decision"] == "allow_once"
    assert decision_payload["reason"] == "user said ok"


@pytest.mark.asyncio
async def test_cancel_resolves_as_deny() -> None:
    """cancel() delegates to resolve with decision='deny'."""
    coord = ApprovalCoordinator()
    future = _make_pending(coord, "p1")

    result = await coord.cancel("p1", reason="user cancelled")
    assert result["ok"] is True
    assert result["decision"] == "deny"
    assert future.done()
    assert future.result()["decision"] == "deny"


# ── deny_for_queue ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deny_for_queue_batch() -> None:
    """deny_for_queue denies all pendings bound to a specific queue."""
    coord = ApprovalCoordinator()
    shared_queue: asyncio.Queue = asyncio.Queue()
    other_queue: asyncio.Queue = asyncio.Queue()

    f1 = _make_pending(coord, "p1", queue=shared_queue)
    f2 = _make_pending(coord, "p2", queue=shared_queue)
    f3 = _make_pending(coord, "p3", queue=other_queue)

    coord.deny_for_queue(shared_queue, reason="stream_closed")

    assert f1.done() and f1.result()["decision"] == "deny"
    assert f2.done() and f2.result()["decision"] == "deny"
    assert not f3.done()  # unrelated queue untouched
    # p1/p2 removed, p3 remains
    assert "p1" not in coord._approval_pending
    assert "p2" not in coord._approval_pending
    assert "p3" in coord._approval_pending


# ── deny_for_request ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_deny_for_request_batch() -> None:
    """deny_for_request denies all pendings sharing a request_id."""
    coord = ApprovalCoordinator()
    f1 = _make_pending(coord, "p1", request_id="req-A")
    f2 = _make_pending(coord, "p2", request_id="req-A")
    f3 = _make_pending(coord, "p3", request_id="req-B")

    coord.deny_for_request("req-A", reason="turn_ended")

    assert f1.done() and f1.result()["decision"] == "deny"
    assert f2.done() and f2.result()["reason"] == "turn_ended"
    assert not f3.done()
    assert coord.pending_count() == 1


# ── prune_orphaned ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prune_orphaned_removes_only_without_live_route() -> None:
    """prune_orphaned denies pendings whose request_id has no live route,
    but leaves pendings with a live route or no request_id alone."""
    coord = ApprovalCoordinator()

    # p1: has a live route → kept
    f1 = _make_pending(coord, "p1", request_id="req-alive")
    coord.register_route("req-alive")

    # p2: no live route → pruned
    f2 = _make_pending(coord, "p2", request_id="req-dead")

    # p3: no request_id → conservatively kept (cannot determine owner)
    # Bypass helper default to ensure the payload has an empty request_id.
    loop = asyncio.get_running_loop()
    f3: asyncio.Future[dict[str, Any]] = loop.create_future()
    coord._approval_pending["p3"] = {
        "request": {"pending_id": "p3", "request_id": ""},
        "future": f3,
        "queue": asyncio.Queue(),
        "created_at": 0.0,
    }

    pruned = coord.prune_orphaned()

    assert pruned == 1
    assert not f1.done()  # still alive
    assert f2.done() and f2.result()["decision"] == "deny"
    assert not f3.done()  # deliberately left alone
    assert "p1" in coord._approval_pending
    assert "p2" not in coord._approval_pending
    assert "p3" in coord._approval_pending


# ── normalize_decision ───────────────────────────────────────────────────────


def test_normalize_decision_defaults_unknown_to_deny() -> None:
    """Unknown or empty decision values normalize to 'deny'."""
    coord = ApprovalCoordinator()
    assert coord._normalize_decision("") == "deny"
    assert coord._normalize_decision("UNKNOWN") == "deny"
    assert coord._normalize_decision("  gibberish  ") == "deny"
    # Known values are preserved (case-insensitive).
    assert coord._normalize_decision("Allow_Once") == "allow_once"
    assert coord._normalize_decision("allow_session") == "allow_session"
    assert coord._normalize_decision("cancel_workflow") == "cancel_workflow"


# ── register_route / unregister_route ────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_unregister_route_influences_pruning() -> None:
    """A route registered then unregistered flips a pending from kept to pruned."""
    coord = ApprovalCoordinator()
    f1 = _make_pending(coord, "p1", request_id="req-X")
    coord.register_route("req-X")

    # With the route live, pruning should not touch p1.
    assert coord.prune_orphaned() == 0
    assert not f1.done()

    # After unregistering, the pending becomes an orphan.
    coord.unregister_route("req-X")
    assert coord.prune_orphaned() == 1
    assert f1.done()


# ── repeated resolution ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeated_resolution_is_safe() -> None:
    """Resolving the same pending_id twice does not raise; the second call
    reports 'no longer pending'."""
    coord = ApprovalCoordinator()
    _make_pending(coord, "p1")

    first = await coord.resolve("p1", "allow")
    assert first["ok"] is True

    # The future is done and the pending is cleaned up by resolve → second
    # call finds it either gone or done.
    second = await coord.resolve("p1", "allow")
    assert second["ok"] is False
    assert "no longer pending" in second.get("error", "") or "Unknown" in second.get("error", "")

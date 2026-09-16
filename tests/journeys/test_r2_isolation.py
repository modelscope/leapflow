# Copyright (c) Alibaba, Inc. and its affiliates.
"""R2 — concurrency and session identity across two workspaces on one daemon.

Several TUIs in different workspaces sharing one leapd is a supported way to use
LeapFlow, not an edge case, and it is the scenario a single-session test cannot
observe at all. Every incident in this area shipped with a green suite: a second
client adopted the first's session id, sent it with its own workspace, and was
rejected on every turn.

Phases: two sessions in two workspaces → interleaved turns → cross-read status,
history and usage → an anonymous status reveals nobody → cross-workspace reuse is
refused with an actionable message.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted
from tests._harness.journey import JourneyFactory

# Journey metadata read by tools/impact.py (see test_r1_conversation.py).
SUBJECT_PATHS = (
    "src/leapflow/daemon/",
    "src/leapflow/engine/",
    "src/leapflow/cli/",
)

# Concurrency and identity behaviour can shift with real provider latency, which
# is exactly the condition under which cross-client leakage appeared.
LIVE_SIGNAL = True

SESSION_A = "r2-client-a"
SESSION_B = "r2-client-b"


async def _turn(client: Any, message: str, *, session_id: str, workspace: str) -> list[Any]:
    """Run one turn to completion and return its stream events."""
    events: list[Any] = []
    async for event in client.engine_chat(
        message, session_id=session_id, workspace_root=workspace
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_r2_two_workspaces_stay_isolated(journeys: JourneyFactory) -> None:
    """Two clients on one daemon never see each other's session, usage, or turns."""
    journey = journeys(
        "r2_isolation",
        script=scripted(
            answer("Workspace A acknowledged."),
            answer("Workspace B acknowledged."),
            answer("Still workspace A."),
            answer("Still workspace B."),
        ),
        deadline_s=90.0,
        # Six turns across two clients. Observed 8 calls against a real provider.
        max_llm_calls=14,
        # Observed 72k tokens against a real provider (~8.9k prompt per call).
        max_llm_tokens=160_000,
    )
    workspace_a = journey.workspace("alpha")
    workspace_b = journey.workspace("beta")
    client_a = journey.client()
    client_b = journey.client()

    with journey.phase("cross-process: the daemon really is another process"):
        boot = await client_a.status()
        assert boot["pid"] != os.getpid(), (
            "status() reported this test's pid — the journey is not exercising a "
            "separate daemon process, so no cross-process contract is being tested"
        )
        assert boot["session_id"] == "", (
            "an anonymous caller must receive no session identity; reporting one "
            "is how a fresh client adopts another client's session"
        )

    with journey.phase("first turns: each client opens its own session"):
        events_a = await _turn(
            client_a, "Hello from A.", session_id=SESSION_A, workspace=str(workspace_a)
        )
        events_b = await _turn(
            client_b, "Hello from B.", session_id=SESSION_B, workspace=str(workspace_b)
        )
        for label, events in (("A", events_a), ("B", events_b)):
            errors = [event.content for event in events if event.type == "error"]
            assert not errors, f"client {label} turn failed: {errors}"

    with journey.phase("identity: each status reports only its own caller"):
        status_a = await client_a.status(SESSION_A)
        status_b = await client_b.status(SESSION_B)
        assert status_a["session_id"] == SESSION_A
        assert status_b["session_id"] == SESSION_B
        assert status_a["session_id"] != status_b["session_id"]

    with journey.phase("anonymous status still reveals nobody after both are live"):
        anonymous = await client_a.status()
        assert anonymous["session_id"] == "", (
            "with two live sessions the daemon answered an unscoped status with a "
            "session identity — that value belongs to whichever session ran last"
        )
        assert anonymous.get("context_used", 0) == 0, (
            "unscoped status reported per-session context usage that belongs to "
            "some other client"
        )

    with journey.phase("history: neither client can read the other's conversation"):
        history_a = await client_a.session_history(session_id=SESSION_A)
        history_b = await client_b.session_history(session_id=SESSION_B)
        blob_a = str(history_a.get("messages") or [])
        blob_b = str(history_b.get("messages") or [])
        assert "Hello from A." in blob_a
        assert "Hello from B." in blob_b
        assert "Hello from B." not in blob_a, "client A can read client B's conversation"
        assert "Hello from A." not in blob_b, "client B can read client A's conversation"

    with journey.phase("concurrent turns: interleaving does not cross-contaminate"):
        results = await asyncio.gather(
            _turn(client_a, "Second A turn.", session_id=SESSION_A, workspace=str(workspace_a)),
            _turn(client_b, "Second B turn.", session_id=SESSION_B, workspace=str(workspace_b)),
        )
        for label, events in zip(("A", "B"), results):
            errors = [event.content for event in events if event.type == "error"]
            assert not errors, f"concurrent turn for client {label} failed: {errors}"

        after_a = await client_a.status(SESSION_A)
        after_b = await client_b.status(SESSION_B)
        assert after_a["session_id"] == SESSION_A
        assert after_b["session_id"] == SESSION_B

        final_a = await client_a.session_history(session_id=SESSION_A)
        assert "Second B turn." not in str(final_a.get("messages") or []), (
            "a concurrent turn from client B landed in client A's session"
        )

    with journey.phase("workspace binding: reuse from another workspace is refused"):
        events = await _turn(
            client_b, "Steal A's session.", session_id=SESSION_A, workspace=str(workspace_b)
        )
        errors = [event for event in events if event.type == "error"]
        assert errors, (
            "reusing session A from workspace B was accepted — a session is bound "
            "to the workspace of its first request"
        )
        metadata = errors[0].metadata or {}
        assert metadata.get("workspace_mismatch") is True, (
            f"refusal was not classified as a workspace mismatch: {metadata}"
        )
        assert metadata.get("session_id") == SESSION_A
        assert str(workspace_a) in str(metadata.get("expected_workspace_root", "")), (
            "the refusal must name the workspace the session is bound to, since "
            "an explicit --resume into another workspace is the only legitimate cause"
        )

    journey.finish()

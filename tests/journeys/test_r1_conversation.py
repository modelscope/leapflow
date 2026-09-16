# Copyright (c) Alibaba, Inc. and its affiliates.
"""R1 — the conversation main line, end to end through a real daemon.

Phases: first turn → streamed chunks → native tool call → tool result fed back →
final answer → history and usage persisted → second turn continues the session.

What only this layer can observe:

- turns land on the *per-session* engine, not on the base template, so
  ``status(session_id)`` reports real context usage instead of zero;
- the tool result actually reaches the next model call, across process boundaries;
- history and token accounting survive in DuckDB, written by the daemon process
  and read back over RPC.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted, tool_call
from tests._harness.journey import JourneyFactory
from tests._harness.leapd import await_for

# ── Journey metadata, read by tools/impact.py (AST-parsed, never imported) ──
#
# SUBJECT_PATHS names the source areas this journey actually exercises, so a
# label-triggered live run can pick the journeys a change could plausibly break
# instead of paying for all of them. Declared here rather than in a central table
# because it belongs next to the assertions it describes and moves with them.
SUBJECT_PATHS = (
    "src/leapflow/engine/",
    "src/leapflow/llm/",
    "src/leapflow/tools/",
    "src/leapflow/daemon/",
    "src/leapflow/memory/",
    "src/leapflow/storage/",
)

# Running this against a real provider adds signal: it is the only journey that
# exercises real tool-calling and real streaming end to end.
LIVE_SIGNAL = True

WORKSPACE_FILE = "invoice.txt"
WORKSPACE_CONTENT = "Invoice 42\nTotal: 128.50 USD\n"


async def _drive(client: Any, message: str, *, session_id: str, workspace: str) -> list[Any]:
    """Run one turn and return every stream event it produced."""
    events: list[Any] = []
    async for event in client.engine_chat(
        message, session_id=session_id, workspace_root=workspace
    ):
        events.append(event)
    return events


def _text_of(events: list[Any], *types: str) -> str:
    """Concatenate the content of events of the given types."""
    return "".join(event.content for event in events if event.type in types)


@pytest.mark.asyncio
async def test_r1_conversation_main_line(journeys: JourneyFactory) -> None:
    """A full conversation: stream, call a tool, answer, persist, continue."""
    journey = journeys(
        "r1_conversation",
        script=scripted(
            answer("Hello from LeapFlow."),
            tool_call("file_read", path=WORKSPACE_FILE),
            answer("The invoice total is 128.50 USD."),
            answer("Yes, that is the same invoice."),
        ),
        deadline_s=90.0,
        # Four turns. Observed against a real provider: 4 calls when the model
        # went straight to file_read, 7 when it globbed first. The ceiling leaves
        # room for that swing without leaving room for a runaway loop.
        max_llm_calls=12,
        # Observed 36k-65k tokens across real runs; ~9k prompt tokens per call is
        # the floor set by the system prompt plus tool schemas. A jump past this
        # means prompt assembly grew, which is exactly what would otherwise raise
        # the live lane's bill in silence.
        max_llm_tokens=140_000,
    )
    workspace = journey.workspace("main")
    (workspace / WORKSPACE_FILE).write_text(WORKSPACE_CONTENT, encoding="utf-8")
    session_id = "r1-session"
    client = journey.client()

    with journey.phase("boot: daemon reports itself without inventing a session"):
        status = await client.status()
        assert status["pid"] > 0
        assert status["profile"] == "default"
        assert status["session_id"] == "", (
            "a caller that named no session must not be handed somebody else's identity"
        )

    with journey.phase("first turn: streamed answer"):
        events = await _drive(
            client, "Say hello.", session_id=session_id, workspace=str(workspace)
        )
        assert _text_of(events, "chunk", "final"), f"no answer text in {[e.type for e in events]}"
        assert not [e for e in events if e.type == "error"], (
            f"turn reported errors: {[e.content for e in events if e.type == 'error']}"
        )

    with journey.phase("session state: reported from the session engine, not the template"):
        scoped = await client.status(session_id)
        assert scoped["session_id"] == session_id
        assert scoped["context_used"] > 0, (
            "context usage read as zero — the reporting path resolved the base "
            "engine template instead of the session engine"
        )
        assert scoped["llm_context_length"] > 0

    with journey.phase("tool turn: native call executes and its result feeds back"):
        events = await _drive(
            client,
            f"Use the file_read tool on {WORKSPACE_FILE} and report the total.",
            session_id=session_id,
            workspace=str(workspace),
        )
        assert not [e for e in events if e.type == "error"], (
            f"tool turn failed: {[e.content for e in events if e.type == 'error']}"
        )
        started = [e.content for e in events if e.type == "tool_start"]
        completed = [e.content for e in events if e.type == "tool_complete"]

        if journey.is_live:
            # Which tool a real model reaches for is its own decision, so the
            # live lane asserts only that tool dispatch works when it happens.
            # The strict form below runs on every push, in replay.
            assert set(started) == set(completed), (
                f"a tool started but never completed: started={started} "
                f"completed={completed}"
            )
        else:
            assert "file_read" in started, f"tool was never started: {[e.type for e in events]}"
            assert "file_read" in completed, f"tool never completed: {started}"
            fed_back = journey.proxy.stats.prompts_containing("128.50")
            assert fed_back, (
                "the file content never reached a subsequent model call — the tool "
                "result did not make it back into the conversation"
            )

    with journey.phase("persistence: history and usage survive in the daemon's store"):
        history = await await_for(
            lambda: _history_of(client, session_id),
            timeout_s=10.0,
            what="persisted history",
        )
        roles = [str(message.get("role", "")) for message in history]
        assert roles.count("user") >= 2, f"user turns missing from history: {roles}"
        assert "assistant" in roles, f"assistant turns missing from history: {roles}"

        usage = await client.usage_summary()
        assert usage, "usage summary is empty"

    with journey.phase("continuation: the same session keeps its context"):
        events = await _drive(
            client,
            "Is that the same invoice?",
            session_id=session_id,
            workspace=str(workspace),
        )
        assert not [e for e in events if e.type == "error"]
        final_status = await client.status(session_id)
        assert final_status["session_id"] == session_id
        assert final_status["context_used"] >= scoped["context_used"], (
            "context usage went backwards across turns of one session"
        )

    journey.finish()


async def _history_of(client: Any, session_id: str) -> list[dict[str, Any]]:
    """Fetch persisted messages, returning [] until the write is visible."""
    payload = await client.session_history(session_id=session_id)
    return list(payload.get("messages") or [])

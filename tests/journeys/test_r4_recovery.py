# Copyright (c) Alibaba, Inc. and its affiliates.
"""R4 — failure and recovery, driven by real provider wire semantics.

Every failure here arrives as an actual HTTP response through the real ``openai``
client, so the classifier reads the same bytes a provider would send. That is the
part a mock cannot check: a hand-raised exception proves the handler runs, not
that the *real* error is recognised as the category it belongs to.

Phases: rate limit is retried → server error is retried → context overflow is
transformed and retried → an unrecoverable failure halts with something the user
can act on, and leaves an audit trail.

Deliberately *not* here (they belong to other layers per the duty matrix):
exception-type classification of local defects, per-category budget arithmetic,
and side-effect gating decisions are table-and-branch concerns for the mock
layer; truncated-SSE handling is a provider-parsing concern covered where the
streaming path actually runs (``tests/test_journey_harness.py``), since the
engine's native-tool round is non-streaming.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._harness.cassette import (
    context_overflow_response,
    error_response,
    rate_limited_response,
    server_error_response,
)
from tests._harness.cassette_proxy import answer, scripted
from tests._harness.journey import Journey, JourneyFactory
from tests._harness.leapd import await_for

# Journey metadata read by tools/impact.py (see test_r1_conversation.py).
SUBJECT_PATHS = (
    "src/leapflow/engine/",
    "src/leapflow/llm/",
)

# Cannot run live at all: every response it asserts on is an injected failure,
# and a forwarding mode sends each request upstream instead. The factory also
# enforces this at runtime via requires_scripted_responses.
LIVE_SIGNAL = False

SESSION = "r4-recovery"


async def _turn(journey: Journey, client: Any, message: str, workspace: str) -> list[Any]:
    """Run one turn and return its stream events."""
    events: list[Any] = []
    async for event in client.engine_chat(
        message, session_id=SESSION, workspace_root=workspace
    ):
        events.append(event)
    return events


def _errors(events: list[Any]) -> list[Any]:
    return [event for event in events if event.type == "error"]


def _answer_text(events: list[Any]) -> str:
    return "".join(event.content for event in events if event.type in ("chunk", "final"))


def _audit_entries(path: Path) -> list[dict[str, Any]]:
    """Read the recovery audit trail the daemon wrote to ``path``."""
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


@pytest.mark.asyncio
async def test_r4_failure_and_recovery(journeys: JourneyFactory) -> None:
    """Provider failures are classified, retried where safe, and halt actionably."""
    journey = journeys(
        "r4_recovery",
        script=scripted(
            # Phase: transient rate limit, then the real answer.
            rate_limited_response(),
            answer("Recovered after a rate limit."),
            # Phase: transient server error, then the real answer.
            server_error_response(),
            answer("Recovered after a server error."),
            # Phase: context overflow, then the real answer once transformed.
            context_overflow_response(),
            answer("Recovered after compressing context."),
            # Phase: a permanent, non-retryable failure for every attempt.
            error_response(
                400,
                code="unsupported_value",
                message="The requested configuration is not supported by this model",
            ),
        ),
        # Every response here is an injected failure, so this journey is only
        # meaningful against recordings: a forwarding mode sends each request
        # upstream and the failures never happen.
        requires_scripted_responses=True,
        deadline_s=120.0,
        # Eight scripted responses plus provider-level retries. A higher count
        # means recovery stopped converging.
        max_llm_calls=20,
        # Injected failures carry no usage, so the real spend here is the retried
        # successes; replay-only, so this ceiling guards loop growth, not cost.
        max_llm_tokens=100_000,
    )
    workspace = str(journey.workspace("recover"))
    client = journey.client()

    with journey.phase("rate limit: retried transparently, user still gets an answer"):
        before = journey.proxy.stats.call_count
        events = await _turn(journey, client, "Summarize the situation.", workspace)
        assert not _errors(events), f"a retryable 429 surfaced as an error: {_errors(events)}"
        assert _answer_text(events), "no answer text after recovering from a 429"
        assert journey.proxy.stats.call_count > before + 1, (
            "the 429 was never retried — only one provider call was made"
        )

    with journey.phase("server error: retried transparently"):
        before = journey.proxy.stats.call_count
        events = await _turn(journey, client, "And now?", workspace)
        assert not _errors(events), f"a retryable 500 surfaced as an error: {_errors(events)}"
        assert journey.proxy.stats.call_count > before + 1, "the 500 was never retried"

    with journey.phase("context overflow: transformed and retried, not abandoned"):
        before = journey.proxy.stats.call_count
        events = await _turn(journey, client, "Keep going with more context.", workspace)
        assert not _errors(events), (
            f"a context overflow was treated as terminal: {[e.content for e in _errors(events)]}"
        )
        assert _answer_text(events), "no answer after context-overflow recovery"
        assert journey.proxy.stats.call_count > before + 1, (
            "the overflow produced no follow-up call — nothing was transformed or retried"
        )

    with journey.phase("unrecoverable: halts with something the user can act on"):
        events = await _turn(journey, client, "Do the impossible thing.", workspace)
        errors = _errors(events)
        assert errors, "a permanent provider failure produced no error event"

        terminal = errors[-1]
        assert terminal.content.strip(), "the terminal error carried no message at all"

        interaction = (terminal.metadata or {}).get("interaction") or {}
        if interaction:
            assert interaction.get("title"), f"interaction without a title: {interaction}"
            assert interaction.get("suggested_actions"), (
                "a halt must name the next step, not just the failure: "
                f"{interaction}"
            )
            assert interaction.get("resumption_key"), (
                "a resumable halt needs a resumption key so the client can continue"
            )
        else:
            # No InteractionRequest was attached; the message itself must then be
            # more than internal jargon, since it is all the user gets.
            assert "No applicable recovery strategy" not in terminal.content, (
                "the user was shown a raw internal reason with no guidance: "
                f"{terminal.content!r}"
            )

    with journey.phase("evidence: recovery decisions are recorded for after the fact"):
        audit_path = journey.daemon.audit_log_path
        entries = await await_for(
            lambda: _await_audit(audit_path),
            timeout_s=10.0,
            what=f"recovery audit entries in {audit_path}",
        )
        classified = [
            entry for entry in entries if str(entry.get("failure_category") or "").strip()
        ]
        assert classified, (
            f"audit entries carry no failure classification: {entries[:3]}"
        )
        assert any(str(entry.get("strategy_key") or "").strip() for entry in classified), (
            "no audit entry names the strategy that decided the outcome, so an "
            f"incident could not be reconstructed: {classified[:3]}"
        )
        assert any(str(entry.get("reason") or "").strip() for entry in classified), (
            "audit entries carry no explainable reason"
        )

    with journey.phase("still usable: the session survives the failure sequence"):
        status = await client.status(SESSION)
        assert status["session_id"] == SESSION
        assert status["context_used"] > 0

    journey.finish()


async def _await_audit(path: Path) -> list[dict[str, Any]]:
    """Return audit entries once the daemon has flushed at least one."""
    return _audit_entries(path)

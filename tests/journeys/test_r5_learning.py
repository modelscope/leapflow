# Copyright (c) Alibaba, Inc. and its affiliates.
"""R5 — the learning loop: teach → record → stop → distill → skill visible.

Progressive Trust starts at recording, so the loop only means something if each
stage survives a process boundary: the teaching session is opened inside the
daemon, steps accumulate on the daemon-owned session, distillation runs there,
and the resulting skill must be readable back through a separate RPC call. An
in-process test can hold all of that in one object graph and prove none of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted
from tests._harness.journey import Journey, JourneyFactory
from tests._harness.leapd import await_for

# Journey metadata read by tools/impact.py (see test_r1_conversation.py).
SUBJECT_PATHS = (
    "src/leapflow/recording/",
    "src/leapflow/learning/",
    "src/leapflow/analysis/",
    "src/leapflow/skills/",
    "src/leapflow/engine/session.py",
    "src/leapflow/storage/",
)

# Distillation quality depends on the real model's output, so running it live is
# the only way to see whether a real answer still distils into a usable skill.
LIVE_SIGNAL = True

SESSION = "r5-learning"


async def _cmd(client: Any, name: str, args: str = "") -> dict[str, Any]:
    """Execute one slash command for this journey's session."""
    return await client.command_execute(name, args, session_id=SESSION)


async def _turn(journey: Journey, client: Any, message: str, workspace: str) -> list[Any]:
    """Run one turn so the teaching session has something to record."""
    events: list[Any] = []
    async for event in client.engine_chat(
        message, session_id=SESSION, workspace_root=workspace
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_r5_learning_loop(journeys: JourneyFactory) -> None:
    """Teaching records, stops, distills, and leaves an inspectable skill library."""
    journey = journeys(
        "r5_learning",
        script=scripted(
            answer("Noted the first step."),
            answer("Noted the second step."),
            answer(
                '{"title": "Tidy invoices", '
                '"trigger_phrases": ["tidy invoices", "sort invoices"], '
                '"steps": ["List the invoice folder", "Classify by month", "Move into folders"], '
                '"parameters": [{"name": "path", "description": "invoice folder"}], '
                '"pre_conditions": [], "confidence": 0.7}'
            ),
            # The final entry repeats for every later call, so it must be a benign
            # answer. Leaving the distillation JSON last made the closing turn
            # re-read it as a reply and loop until its iteration budget ran out.
            answer("Done — nothing further needed."),
        ),
        deadline_s=120.0,
        # Three turns plus a possible background distillation call. Observed 3
        # calls against a real provider. A jump here is the signature of the loop
        # that used to re-read the distillation payload as a reply and run to its
        # iteration cap.
        max_llm_calls=12,
        # Observed 27k tokens against a real provider.
        max_llm_tokens=140_000,
    )
    workspace = str(journey.workspace("learn"))
    client = journey.client()

    with journey.phase("baseline: the skill library is inspectable and starts empty"):
        listing = await _cmd(client, "skill", "list")
        assert listing.get("ok") is True, f"/skill list failed: {listing}"
        assert listing.get("view") == "skill_list"
        baseline = {str(item.get("name")) for item in listing.get("skills") or []}

    with journey.phase("open: a turn must exist before teaching can attach to it"):
        events = await _turn(journey, client, "Let me show you something.", workspace)
        assert not [event for event in events if event.type == "error"], (
            "the seeding turn failed, so there is no session to teach against"
        )

    with journey.phase("record: /teach start enters learning mode"):
        started = await _cmd(client, "teach start", "tidy the invoice folder")
        assert started.get("session_mode") == "learning", (
            f"/teach start did not enter learning mode: {started}"
        )

    with journey.phase("status: the daemon reports the recording session honestly"):
        status = await _cmd(client, "teach status")
        assert status.get("ok") is not False, f"/teach status failed: {status}"

    with journey.phase("annotate: the user's own words are captured"):
        annotated = await _cmd(client, "annotate", "invoices go into per-month folders")
        assert annotated.get("ok") is not False, f"/annotate failed: {annotated}"

    with journey.phase("steps: activity accumulates on the daemon-owned session"):
        await _turn(journey, client, "Now sort them by month.", workspace)

    with journey.phase("stop: recording ends and reports what it captured"):
        stopped = await _cmd(client, "teach stop")
        assert stopped.get("ok") is True, f"/teach stop failed: {stopped}"
        assert stopped.get("session_mode") != "learning", (
            f"still in learning mode after /teach stop: {stopped}"
        )
        assert stopped.get("message"), "/teach stop said nothing about what it recorded"

    with journey.phase("persist: the trajectory survives in the daemon's store"):
        trajectories = await await_for(
            lambda: _trajectory_files(journey),
            timeout_s=20.0,
            what="a persisted teaching trajectory",
        )
        assert trajectories, "teaching produced no durable trajectory"

    with journey.phase("library: skills remain listable, and nothing was corrupted"):
        after = await _cmd(client, "skill", "list")
        assert after.get("ok") is True, f"/skill list failed after teaching: {after}"
        names = {str(item.get("name")) for item in after.get("skills") or []}
        assert names >= baseline, (
            f"skills disappeared during the learning loop: {baseline - names}"
        )
        for entry in after.get("skills") or []:
            assert 0.0 <= float(entry.get("confidence", 0.0)) <= 1.0, (
                f"skill confidence is out of range: {entry}"
            )

    with journey.phase("still usable: a normal turn works after teaching"):
        events = await _turn(journey, client, "Thanks, that is all.", workspace)
        assert not [event for event in events if event.type == "error"]

    journey.finish()


async def _trajectory_files(journey: Journey) -> list[str]:
    """Return DuckDB stores the daemon created, once any exist on disk."""
    return [str(path.name) for path in journey.daemon.data_dir.rglob("*.duckdb")]

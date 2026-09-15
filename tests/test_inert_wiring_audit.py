# Copyright (c) Alibaba, Inc. and its affiliates.
"""The audit of capability that was built and never ran, and the two hops it closed.

Three defects of one shape shipped in a single week with a green suite: the lifecycle
governor was never constructed, the durable trust ledger was never handed to it, and the
hardware trust gate was linked to a ledger that was always ``None``. The audit that found
them lives at ``tools/audit_inert_wiring.py``; these are the regressions.

What makes the shape invisible is that every unit test constructs the collaborator itself,
so the logic is exercised and the wiring never is. The tests here go through the production
resolvers on purpose.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.learning.capability_gap_detector import CapabilityGapDetector
from leapflow.learning.degradation_sink import build_proposal_sink
from leapflow.storage.capability_proposal_queue import JsonCapabilityProposalQueue


def _proposal(capability: str, *, risk: str = "read_only") -> Any:
    verdict = AdaptationVerdict.create(
        "acquire", capability, f"nothing installed serves {capability}", max_risk_level=risk
    )
    return CapabilityGapDetector().proposal_from_evolution_intent(verdict.to_intent())


# ── the acquisition chain's last hop ──────────────────────────────────────────


def test_an_acquire_verdict_reaches_the_proposal_queue(tmp_path):
    """Without this the chain stopped at a requirement nothing enqueued.

    Resolution would report the capability unmet forever, so the teacher's most expensive
    verdict -- the only one that leads to code -- had no effect whatsoever.
    """
    queue = JsonCapabilityProposalQueue(tmp_path / "q.json")
    identifier = build_proposal_sink(queue=queue)(_proposal("mail.send"))

    assert identifier
    items = queue.list_items()
    assert len(items) == 1
    assert items[0].status == "PENDING", "queueing is not acting; approval still gates it"
    assert items[0].source == "world_model"
    assert items[0].requirements[0]["capability"] == "mail.send"


def test_the_same_capability_does_not_pile_up_across_sessions(tmp_path):
    """The queue deduplicates on a hash of the requirement payload.

    Minting a fresh ``requirement_id`` on each rebuild defeated that silently: a reviewer
    would face a growing pile of identical items, and the queue's depth would measure how
    long the process had been running rather than how much was outstanding.
    """
    queue = JsonCapabilityProposalQueue(tmp_path / "q.json")
    sink = build_proposal_sink(queue=queue)

    first = sink(_proposal("mail.send"))
    second = sink(_proposal("mail.send"))
    other = sink(_proposal("chat.reply"))

    assert first == second, "the same capability must resolve to the same proposal"
    assert other != first
    assert len(queue.list_items()) == 2


def test_the_queued_requirement_keeps_the_clamped_risk(tmp_path):
    """``CapabilityRequirement`` defaults ``max_risk_level`` to ``external``.

    That is the most permissive value there is, so omitting it would let a proposal clamped
    to ``read_only`` enter the queue asking for everything -- the exact opposite of what
    the clamp exists for.
    """
    queue = JsonCapabilityProposalQueue(tmp_path / "q.json")
    build_proposal_sink(queue=queue)(_proposal("shell.run", risk="external"))

    requirement = queue.list_items()[0].requirements[0]
    assert requirement["max_risk_level"] == "read_only", "the model cannot widen its own ask"


def test_a_proposal_without_a_capability_is_refused(tmp_path):
    """The queue has nothing to deduplicate on and resolution nothing to satisfy."""
    queue = JsonCapabilityProposalQueue(tmp_path / "q.json")
    sink = build_proposal_sink(queue=queue)

    assert sink(SimpleNamespace(evidence=(), proposal_id="p1")) == ""
    assert queue.list_items() == []


def test_a_failing_queue_does_not_fail_the_session(tmp_path):
    """Queueing improves the next session; it must never break this one."""

    class _Broken:
        def enqueue(self, **kwargs):
            raise OSError("disk full")

    assert build_proposal_sink(queue=_Broken())(_proposal("mail.send")) == ""


# ── the audit itself, kept honest ─────────────────────────────────────────────


def test_the_audit_knows_all_three_supply_channels():
    """Two earlier versions of the audit reported wired code as inert.

    They only recognised keyword arguments, so ``EvolutionMemoryProvider`` (supplied by
    direct attribute assignment) and ``_reentry_event_observer`` (supplied by ``setattr``
    from the daemon) both looked dead. An audit that cries wolf gets ignored, which is
    worse than not having one.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "tools"
        / "audit_inert_wiring.py"
    ).read_text(encoding="utf-8")

    for channel in ("keyword", "attribute", "setattr"):
        assert f'"{channel}"' in source, channel
    # And it must admit what it cannot see, or a clean report reads as proof.
    assert "positional" in source


def test_the_audit_runs_and_reports_a_bounded_set():
    """A regression that keeps the audit executable, not just present.

    Run for its exit status and shape rather than an exact count: the count is expected to
    move as wiring changes, and pinning it would turn every legitimate fix into a failure.
    """
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "tools/audit_inert_wiring.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "A1" in result.stdout and "A2" in result.stdout and "A3" in result.stdout


# ── the one-way link the audit flagged, checked rather than assumed ────────────


def test_generalised_patterns_do_reach_the_durable_store():
    """The audit flagged this as write-only; it is not, and the check is the record.

    ``EvolutionMemoryProvider`` is constructed with only ``max_episodes``, so the
    constructor parameter ``persistent_store`` looked unsupplied -- one of the shapes that
    hid three real defects this week. Here the store arrives by post-construction
    assignment instead (``cli/context.py``: ``self._evolution._persistent_store = ...``),
    which is a legitimate channel, and generalisation does write through it.

    Worth a test rather than a note, because the first attempt to verify it by hand
    concluded the opposite: the probe used a different action sequence per episode, so no
    common pattern could be generalised and nothing was written. The link was fine and the
    probe was wrong -- exactly the way an audit finding becomes a phantom fix.
    """
    from leapflow.memory.providers.evolution import EvolutionMemoryProvider

    written: list[dict[str, Any]] = []

    class _Store:
        def save_pattern(self, **kwargs: Any) -> None:
            written.append(kwargs)

    provider = EvolutionMemoryProvider(max_episodes=50)
    provider._persistent_store = _Store()

    # A *repeated* sequence: generalisation looks for what the episodes share, so varying
    # the actions produces no pattern and therefore no write.
    for _ in range(4):
        provider.record_episode(
            skill_name="chat.reply",
            actions=[{"tool": "chat_reply", "step": 1}, {"tool": "verify", "step": 2}],
            outcome="ok",
            reward=1.0,
        )

    assert written, "a generalised pattern must reach the store"
    assert written[-1]["skill_name"] == "chat.reply"
    assert written[-1]["episode_count"] >= 3


def test_varying_actions_generalise_to_nothing():
    """The negative half, so the test above cannot pass for the wrong reason."""
    from leapflow.memory.providers.evolution import EvolutionMemoryProvider

    written: list[dict[str, Any]] = []

    class _Store:
        def save_pattern(self, **kwargs: Any) -> None:
            written.append(kwargs)

    provider = EvolutionMemoryProvider(max_episodes=50)
    provider._persistent_store = _Store()
    for index in range(8):
        provider.record_episode(
            skill_name="chat.reply",
            actions=[{"tool": f"tool_{index}", "step": index}],
            outcome="ok",
            reward=1.0,
        )

    assert written == [], "episodes with nothing in common have no pattern to persist"

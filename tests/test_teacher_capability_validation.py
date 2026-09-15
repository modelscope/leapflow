# Copyright (c) Alibaba, Inc. and its affiliates.
"""The teacher's capability names are validated, because a live model abused them.

S9 ran `qwen3.7-plus` as the teacher against an episode that failed for a
*non-capability* reason. On 3 of 3 trials it returned the episode's own name
(`chat.cosmetic.example`) as the missing capability. Nothing downstream would have
caught it: the string is well-formed, so it would have become a `CapabilityRequirement`
and the governed pipeline would have faithfully tried to build it.

These pin the two guards that resulted. Neither can catch a *plausible but wrong*
capability -- that is what validation, effect verification and quarantine are for. They
catch the degenerate case, which is the one that produces pure noise.
"""

from __future__ import annotations

from leapflow.domain.evolution_intent import is_capability_name
from leapflow.world_model.trajectory_grader import (
    TrajectoryGrader,
    _echoes_goal,
)


# ── shape ─────────────────────────────────────────────────────────────────────



def _verdict_intents(payload, goal=""):
    """The acquisition intents a payload yields, through the real parser.

    Intents are derived from acquire verdicts now, so a test that wants to assert on
    intents has to go through the same derivation production does.
    """
    verdicts = _grader()._parse_verdicts(payload, goal)
    return tuple(i for i in (v.to_intent() for v in verdicts) if i is not None)


def test_real_capability_names_are_accepted():
    for name in ("chat.reply", "ui.view_messages", "app.chat.send", "fs.file.read.bytes"):
        assert is_capability_name(name), name


def test_prose_and_bare_words_are_rejected():
    """A model asked for a capability sometimes answers with a sentence."""
    for name in (
        "",
        "reply",                                    # no dotted structure
        "The agent cannot reply to the message",    # prose
        "chat reply",                               # spaces
        "Chat.Reply",                               # not lowercase
        "chat.",                                    # trailing separator
        ".reply",                                   # leading separator
        "a.b.c.d.e.f",                              # too many segments
        "chat." + "x" * 90,                         # too long
        "/usr/bin/chat",                            # a path
    ):
        assert not is_capability_name(name), name


# ── goal echo ─────────────────────────────────────────────────────────────────


def test_the_measured_failure_mode_is_rejected():
    """The exact string the live model returned, 3 of 3 trials."""
    assert _echoes_goal("chat.cosmetic.example", "chat.cosmetic.example") is True


def test_echo_detection_ignores_separators_and_case():
    assert _echoes_goal("chat_cosmetic_example", "chat.cosmetic.example") is True
    assert _echoes_goal("CHAT.COSMETIC.EXAMPLE", "chat.cosmetic.example") is True


def test_a_genuine_capability_is_not_an_echo():
    assert _echoes_goal("chat.reply", "reply to the latest message in the thread") is False
    assert _echoes_goal("chat.reply", "") is False


# ── the parse path (drive the real method) ────────────────────────────────────


def _grader():
    class _LLM:
        async def achat(self, *a, **k):
            raise AssertionError("not called")

    class _Store:
        def __getattr__(self, name):
            return lambda *a, **k: None

    from leapflow.world_model.budget import LearningBudgetController

    return TrajectoryGrader(_LLM(), _Store(), LearningBudgetController(grading_budget=1))


def test_goal_restatement_never_becomes_a_requirement():
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "chat.cosmetic.example", "knowledge": "the send failed"}
        ]
    }
    assert _grader()._parse_verdicts(payload, "chat.cosmetic.example") == ()


def test_a_sentence_never_becomes_a_requirement():
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "the agent lacks a way to reply", "knowledge": "h"}
        ]
    }
    assert _grader()._parse_verdicts(payload, "goal") == ()


def test_a_well_formed_gap_still_passes():
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "chat.reply",
                "knowledge": "the send control no-ops",
                "confidence": 0.8,
                "expected_effect": "the reply appears in the thread",
            }
        ]
    }
    intents = _verdict_intents(payload, "reply to the latest message")
    assert len(intents) == 1
    assert intents[0].capability == "chat.reply"
    assert intents[0].expected_effect == "the reply appears in the thread"


def test_one_bad_gap_does_not_discard_a_good_one():
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "my.goal", "knowledge": "h"},
            {"action": "acquire", "capability": "chat.reply", "knowledge": "the send control no-ops"},
        ]
    }
    intents = _verdict_intents(payload, "my.goal")
    assert [i.capability for i in intents] == ["chat.reply"]


def test_the_prompt_tells_the_model_both_rules():
    """The guard is defence; the prompt is what should prevent it being needed."""
    from leapflow.world_model.trajectory_grader import _GAP_PROMPT_SECTION

    assert "Do NOT restate the task" in _GAP_PROMPT_SECTION
    assert "Report nothing at all" in _GAP_PROMPT_SECTION
    assert "worse than" in _GAP_PROMPT_SECTION
    # The action space replaced the binary question, so the prompt must also say what
    # the cheap answers are -- otherwise "report nothing" is the only alternative to
    # building something, and building wins by default.
    assert "- absorb:" in _GAP_PROMPT_SECTION
    assert "- rebind:" in _GAP_PROMPT_SECTION


def test_model_authored_risk_is_still_clamped():
    """The guard must not have disturbed the clamp: a model may never widen risk."""
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "chat.reply", "knowledge": "h", "max_risk_level": "external"}
        ]
    }
    intents = _verdict_intents(payload, "goal")
    assert intents[0].effective_risk_ceiling() == "read_only"

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

from leapflow.world_model.trajectory_grader import (
    TrajectoryGrader,
    _echoes_goal,
    _is_capability_name,
)


# ── shape ─────────────────────────────────────────────────────────────────────


def test_real_capability_names_are_accepted():
    for name in ("chat.reply", "ui.view_messages", "app.chat.send", "fs.file.read.bytes"):
        assert _is_capability_name(name), name


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
        assert not _is_capability_name(name), name


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
        "capability_gaps": [
            {"capability": "chat.cosmetic.example", "hypothesis": "the send failed"}
        ]
    }
    assert _grader()._parse_intents(payload, "chat.cosmetic.example") == ()


def test_a_sentence_never_becomes_a_requirement():
    payload = {
        "capability_gaps": [
            {"capability": "the agent lacks a way to reply", "hypothesis": "h"}
        ]
    }
    assert _grader()._parse_intents(payload, "goal") == ()


def test_a_well_formed_gap_still_passes():
    payload = {
        "capability_gaps": [
            {
                "capability": "chat.reply",
                "hypothesis": "the send control no-ops",
                "confidence": 0.8,
                "expected_effect": "the reply appears in the thread",
            }
        ]
    }
    intents = _grader()._parse_intents(payload, "reply to the latest message")
    assert len(intents) == 1
    assert intents[0].capability == "chat.reply"
    assert intents[0].expected_effect == "the reply appears in the thread"


def test_one_bad_gap_does_not_discard_a_good_one():
    payload = {
        "capability_gaps": [
            {"capability": "my.goal", "hypothesis": "h"},
            {"capability": "chat.reply", "hypothesis": "the send control no-ops"},
        ]
    }
    intents = _grader()._parse_intents(payload, "my.goal")
    assert [i.capability for i in intents] == ["chat.reply"]


def test_the_prompt_tells_the_model_both_rules():
    """The guard is defence; the prompt is what should prevent it being needed."""
    from leapflow.world_model.trajectory_grader import _GAP_PROMPT_SECTION

    assert "Do NOT restate the task" in _GAP_PROMPT_SECTION
    assert "return an empty list" in _GAP_PROMPT_SECTION
    assert "worse than" in _GAP_PROMPT_SECTION


def test_model_authored_risk_is_still_clamped():
    """The guard must not have disturbed the clamp: a model may never widen risk."""
    payload = {
        "capability_gaps": [
            {"capability": "chat.reply", "hypothesis": "h", "max_risk_level": "external"}
        ]
    }
    intents = _grader()._parse_intents(payload, "goal")
    assert intents[0].effective_risk_ceiling() == "read_only"

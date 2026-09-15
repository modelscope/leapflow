# Copyright (c) Alibaba, Inc. and its affiliates.
"""Phase B: the teacher answers with an action, and every answer teaches something.

The prompt this replaces asked a binary question -- "is this a capability gap?" -- and
told the teacher to report one only when the implementation was *wrong*. Checked against
the EVO-02 scenario that reading suppresses the case most needing an answer: when the
application steps from v1 to v2 the incumbent was not written wrongly, it was right for
v1, so the honest answer to "is the implementation wrong" is no and nothing happens --
even when a new adapter is the only way forward.

So the answer space is now the set of things the system can do, ordered by cost, and
every answer carries what the acting agent should know. Three of the four actions change
nothing else, which is why the knowledge field is mandatory rather than optional.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from leapflow.domain.adaptation_verdict import (
    ACQUIRE,
    ADAPTATION_ACTIONS,
    AdaptationVerdict,
)
from leapflow.learning.world_model_driver import WorldModelEvolutionDriver
from leapflow.world_model.trajectory_grader import TeacherVerdict, TrajectoryGrader


def _verdict(action: str, capability: str = "chat.reply", **kw: Any) -> AdaptationVerdict:
    return AdaptationVerdict.create(
        action, capability, kw.pop("knowledge", "the app is now v3"), **kw
    )


# ── the action space is closed, and each answer must be actionable ─────────────


def test_the_four_actions_are_the_whole_space():
    """An action outside the set has no consumer, so accepting one would be silent."""
    assert ADAPTATION_ACTIONS == {"absorb", "rebind", "acquire", "escalate"}
    with pytest.raises(ValueError, match="action must be one of"):
        AdaptationVerdict.create("rebuild", "chat.reply", "knowledge")


def test_knowledge_is_mandatory_on_every_verdict():
    """A verdict that teaches nothing leaves the teacher with no effect when it is right.

    Three of the four actions change nothing except what the acting agent knows, so a
    verdict without knowledge is not a cheap answer -- it is no answer.
    """
    for action in sorted(ADAPTATION_ACTIONS):
        with pytest.raises(ValueError, match="knowledge is required"):
            AdaptationVerdict.create(action, "chat.reply", "   ")


def test_a_capability_name_must_look_like_one():
    """Measured: a live model returned the episode's own name on 3 of 3 trials.

    The name was well-formed prose, so nothing downstream would have caught it and the
    governed pipeline would have faithfully tried to build it.
    """
    with pytest.raises(ValueError, match="short dotted name"):
        AdaptationVerdict.create("acquire", "reply to the message in the thread", "k")
    assert AdaptationVerdict.create("acquire", "chat.reply", "k").capability == "chat.reply"


# ── only acquire writes code ───────────────────────────────────────────────────


def test_only_acquire_derives_an_acquisition_intent():
    """A recommendation to rebind must never become a request to write code."""
    assert _verdict(ACQUIRE).to_intent() is not None
    for action in ("absorb", "rebind", "escalate"):
        assert _verdict(action).to_intent() is None, action
        assert _verdict(action).writes_code is False
    assert _verdict(ACQUIRE).writes_code is True


def test_intents_are_derived_from_verdicts_not_carried_beside_them():
    """One source of truth, so the two can never disagree.

    Keeping them independent is how "I recommend doing X" and "I want a new capability"
    get mixed into one object, and then a rebind silently queues an acquisition.
    """
    teacher_verdict = TeacherVerdict(
        grades=(),
        verdicts=(
            _verdict("absorb", "chat.react"),
            _verdict("rebind", "chat.reply", target="chat_reply_v3"),
            _verdict(ACQUIRE, "mail.send"),
        ),
    )

    assert len(teacher_verdict.verdicts) == 3
    assert [i.capability for i in teacher_verdict.intents] == ["mail.send"]
    assert teacher_verdict.by_action("rebind")[0].target == "chat_reply_v3"


def test_a_model_cannot_widen_the_ceiling_and_the_request_stays_auditable():
    """One clamp, one place -- and the original request survives to the approver.

    Clamping inside ``to_intent`` as well looked safer and silently destroyed the audit
    trail: the downstream clamp records the request only when it differs from what was
    granted, so pre-clamping made the two equal and an approver could no longer see that
    the model had asked for more than it got.
    """
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    wide = AdaptationVerdict.create(
        "acquire", "shell.run", "nothing can run shell here", max_risk_level="external"
    )
    intent = wide.to_intent()
    assert intent.max_risk_level == "external", "the request travels unclamped"

    proposal = CapabilityGapDetector().proposal_from_evolution_intent(intent)
    assert proposal.risk_level == "read_only", "the clamp still holds"
    metadata = dict(proposal.evidence[0].metadata)
    assert metadata["requested_max_risk_level"] == "external", "the ask is on the record"


# ── the prompt asks for an action, never for blame ─────────────────────────────


def test_the_prompt_asks_what_to_do_rather_than_whose_fault_it_is():
    """The binary reading suppressed the environment-upgrade case entirely."""
    from leapflow.world_model.trajectory_grader import _GAP_PROMPT_SECTION

    section = _GAP_PROMPT_SECTION
    for action in sorted(ADAPTATION_ACTIONS):
        assert f"- {action}:" in section, action
    assert "adaptation_verdicts" in section
    assert "MUST carry `knowledge`" in section
    # The distinction the old prompt collapsed.
    assert "not for blame" in section
    assert "different questions" in section
    # acquire must be named as the expensive one, so it is not the default answer.
    assert "ONLY action that causes code to be written" in section


def test_the_prompt_still_refuses_invention():
    """The two guards that caught a measured 3/3 false-positive rate must survive."""
    from leapflow.world_model.trajectory_grader import _GAP_PROMPT_SECTION

    assert "Do NOT restate the task" in _GAP_PROMPT_SECTION
    assert "An invented verdict is worse than a missed" in _GAP_PROMPT_SECTION
    # And it must say *why* invention is costly, in terms of what happens next.
    assert "acquire builds code and rebind redirects traffic" in _GAP_PROMPT_SECTION


def test_the_parser_keeps_both_rejections_and_requires_knowledge():
    """Parsing is where a malformed answer must die, not downstream."""
    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    payload = {
        "adaptation_verdicts": [
            {"action": "absorb", "capability": "chat.reply", "knowledge": "v3 now"},
            {"action": "absorb", "capability": "reply to this thread", "knowledge": "k"},
            {"action": "absorb", "capability": "chat.reply"},          # no knowledge
            {"action": "rebuild", "capability": "chat.reply", "knowledge": "k"},
            "not a dict",
        ]
    }
    verdicts = grader._parse_verdicts(payload, goal="reply in the thread")

    assert len(verdicts) == 1
    assert verdicts[0].capability == "chat.reply"


def test_a_goal_restatement_is_still_rejected():
    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    payload = {
        "adaptation_verdicts": [
            {"action": "acquire", "capability": "chat.cosmetic.example",
             "knowledge": "k"},
        ]
    }
    assert grader._parse_verdicts(payload, goal="chat cosmetic example") == ()


# ── the driver dispatches by action ────────────────────────────────────────────


class _Teacher:
    def __init__(self, verdict: TeacherVerdict) -> None:
        self._verdict = verdict

    async def grade_and_propose(self, trajectory, goal="", **kwargs):
        return self._verdict


class _Intake:
    def observe_result(self, result, **kwargs):
        return {"observation_id": "o1"}

    def requirements(self, *, min_count: int = 1, limit: int = 50):
        return ()


def _drive(verdicts, sink=None):
    queued: list[Any] = []
    driver = WorldModelEvolutionDriver(
        teacher=_Teacher(TeacherVerdict(grades=(), verdicts=tuple(verdicts))),
        intake=_Intake(),
        proposal_sink=sink or (lambda p: queued.append(p) or p.proposal_id),
    )
    return asyncio.run(driver.drive([{"action": "a"}], "reply in the thread")), queued


def test_only_the_acquire_verdict_reaches_the_proposal_queue():
    result, queued = _drive(
        [
            _verdict("absorb", "chat.react"),
            _verdict("rebind", "chat.reply", target="chat_reply_v3"),
            _verdict(ACQUIRE, "mail.send"),
            _verdict("escalate", "drive.upload", target="grant drive.file"),
        ]
    )

    assert result.to_dict()["by_action"] == {
        "absorb": 1, "rebind": 1, "acquire": 1, "escalate": 1
    }
    assert len(queued) == 1
    assert dict(queued[0].evidence[0].metadata)["capability"] == "mail.send"


def test_the_cheap_verdicts_survive_on_the_result():
    """Dropping them for not writing code would discard the common correct answer."""
    result, _ = _drive(
        [
            _verdict("absorb", "chat.react"),
            _verdict("rebind", "chat.reply", target="chat_reply_v3"),
            _verdict("escalate", "drive.upload"),
        ]
    )

    cheap = tuple(v for v in result.verdicts if not v.writes_code)
    assert {v.action for v in cheap} == {"absorb", "rebind", "escalate"}
    assert all(v.knowledge for v in cheap)


def test_a_session_that_only_absorbed_is_not_reported_as_idle():
    """The cheapest answer must be visible, or adapting well looks like doing nothing."""
    result, queued = _drive([_verdict("absorb", "chat.react")])

    assert queued == [], "absorb writes no code"
    assert result.to_dict()["by_action"]["absorb"] == 1
    assert len(result.verdicts) == 1
    assert result.proposed == 0, "absorb is not a proposal"


def test_a_teacher_with_nothing_to_say_stays_empty():
    result, queued = _drive([])
    assert result.verdicts == () and queued == []
    assert result.to_dict()["by_action"] == {
        "absorb": 0, "rebind": 0, "acquire": 0, "escalate": 0
    }


# ── review findings: the derivation must be pure and the rule single ───────────


def test_reading_intents_twice_yields_the_same_intents():
    """An immutable object whose derived value changes on every read is a landmine.

    ``EvolutionIntent.create`` mints its own id and timestamp, so deriving through it
    returned a *different* intent on each access -- and ``intent_id`` is the evidence
    identity downstream. No single-read test could notice; only comparing two reads can.
    """
    verdict = _verdict(ACQUIRE, "mail.send")
    teacher_verdict = TeacherVerdict(grades=(), verdicts=(verdict,))

    first, second = teacher_verdict.intents, teacher_verdict.intents
    assert first == second, "derivation must be a pure function of the verdict"
    assert first[0].intent_id == second[0].intent_id
    assert first[0].created_at == second[0].created_at


def test_an_intent_is_traceable_back_to_the_verdict_that_asked_for_it():
    """Identity from the verdict is what makes the derivation pure, and it audits."""
    verdict = _verdict(ACQUIRE, "mail.send")
    intent = verdict.to_intent()
    assert intent.intent_id.endswith(verdict.verdict_id.removeprefix("adv-"))


def test_one_capability_name_rule_governs_both_gates():
    """Two nearly-identical regexes diverged and silently dropped a valid name.

    The parser's gate accepted ``chat.2fa`` and the verdict constructor rejected it, so
    a legitimately named capability died with only a debug log between them.
    """
    from leapflow.domain.evolution_intent import is_capability_name
    from leapflow.world_model.trajectory_grader import TrajectoryGrader

    for name in ("chat.reply", "chat.2fa", "a.b", "x" * 40 + ".y", "reply to the thread"):
        gate = is_capability_name(name)
        try:
            AdaptationVerdict.create("absorb", name, "k")
            constructor = True
        except ValueError:
            constructor = False
        assert gate == constructor, f"{name!r}: gate={gate} constructor={constructor}"

    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    parsed = grader._parse_verdicts(
        {"adaptation_verdicts": [
            {"action": "absorb", "capability": "chat.2fa", "knowledge": "a 2fa prompt appears"}
        ]}
    )
    assert len(parsed) == 1, "the name the two rules disagreed about must survive"


# ── a rebind must be able to point at something real ──────────────────────────


def test_a_rebind_to_a_nonexistent_target_is_rejected():
    """Measured on a real model at 3 of 3 trials, and scored as a correct answer.

    Asked about a chat-app failure while shown the naming catalogue of LeapFlow's own
    tools, the teacher answered ``rebind`` every time -- on a unit whose candidate set had
    exactly one entry, so there was nothing to rebind to. The report read 1.0 accuracy
    because only capability naming was scored. A rebind that cannot point at a real
    provider is not the cheap answer: the failure stays in place and the student is told
    to "Prefer" something that does not exist.
    """
    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    payload = {
        "adaptation_verdicts": [
            {
                "action": "rebind",
                "capability": "chat.reply",
                "knowledge": "the app is now v3",
                "target": "no_such_capability.anywhere",
            }
        ]
    }
    assert grader._parse_verdicts(payload, goal="reply in the thread") == ()


def test_a_rebind_to_a_declared_capability_survives():
    """The guard must not reject the case rebind exists for."""
    from leapflow.plugins import get_registry
    from leapflow.world_model.trajectory_grader import _is_declared_capability

    declared = sorted(
        capability
        for plugin in get_registry().plugins.values()
        for tool in plugin.tools
        for capability in (tool.provides_capabilities or ())
        if capability
    )
    assert declared, "the registry must expose declarations for this to mean anything"
    assert _is_declared_capability(declared[0])

    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    parsed = grader._parse_verdicts(
        {
            "adaptation_verdicts": [
                {
                    "action": "rebind",
                    "capability": "chat.reply",
                    "knowledge": "the app is now v3",
                    "target": declared[0],
                }
            ]
        },
        goal="reply in the thread",
    )
    assert len(parsed) == 1 and parsed[0].target == declared[0]


def test_a_tool_name_is_an_acceptable_rebind_target():
    """Naming a concrete provider is more specific than asked, not less."""
    from leapflow.plugins import get_registry
    from leapflow.world_model.trajectory_grader import _is_declared_capability

    names = [t.name for p in get_registry().plugins.values() for t in p.tools]
    assert names
    assert _is_declared_capability(names[0])


def test_only_rebind_is_gated_on_its_target():
    """The other three actions do not promise a provider, so they must pass through."""
    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    for action in ("absorb", "acquire", "escalate"):
        parsed = grader._parse_verdicts(
            {
                "adaptation_verdicts": [
                    {
                        "action": action,
                        "capability": "chat.reply",
                        "knowledge": "something changed",
                        "target": "no_such_capability.anywhere",
                    }
                ]
            },
            goal="reply in the thread",
        )
        assert len(parsed) == 1, action


def test_the_catalogue_says_what_it_is_for():
    """Listing names invited reading them as "these all fit here"."""
    from leapflow.world_model.trajectory_grader import _declared_capability_section

    section = _declared_capability_section()
    assert "for *naming*" in section
    assert "an environment that is not present" in section
    assert "`rebind` is the wrong action" in section


def test_a_rebind_to_a_provider_we_offered_is_accepted():
    """Rejecting a target we ourselves named is incoherent, and it silently was.

    The guard consulted only the live registry, so in any process whose alternatives come
    from somewhere else -- a replay, a study, a registry that has not caught up -- every
    legitimate rebind was discarded. It measured as 3/3 silence on a rebind unit across two
    unrelated corpus designs, which read as "the model will not answer rebind" and was in
    fact "we threw the answer away".
    """
    grader = TrajectoryGrader.__new__(TrajectoryGrader)
    facts = (
        {
            "capability": "chat.reply",
            "plugin_id": "chat_reply_v1",
            "failure_streak": 3,
            "alternatives": (
                {"plugin_id": "tb", "tool_name": "chat_reply_toolbar",
                 "fits_here": True, "requires": ()},
            ),
        },
    )
    payload = {
        "adaptation_verdicts": [
            {"action": "rebind", "capability": "chat.reply",
             "knowledge": "the toolbar path still delivers",
             "target": "chat_reply_toolbar"}
        ]
    }

    parsed = grader._parse_verdicts(payload, "reply in the thread", facts)
    assert len(parsed) == 1 and parsed[0].target == "chat_reply_toolbar"

    # The registry route remains, and catches a target that was invented rather than
    # selected from what it was shown.
    invented = {
        "adaptation_verdicts": [
            {"action": "rebind", "capability": "chat.reply", "knowledge": "k",
             "target": "no_such_provider_anywhere"}
        ]
    }
    assert grader._parse_verdicts(invented, "reply in the thread", facts) == ()

    # And with no alternatives offered, only the registry can vouch for a target.
    assert grader._parse_verdicts(payload, "reply in the thread", ()) == ()


def test_the_plugin_id_of_an_offered_alternative_also_counts():
    """The teacher may name either identity; both were shown to it."""
    from leapflow.world_model.trajectory_grader import _offered_providers

    offered = _offered_providers(
        ({"alternatives": ({"plugin_id": "tb", "tool_name": "chat_reply_toolbar"},)},)
    )
    assert offered == {"tb", "chat_reply_toolbar"}

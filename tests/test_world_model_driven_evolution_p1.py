"""P1: the world model becomes a driver of capability evolution.

Two halves:

* **WM-2** -- ``TrajectoryGrader.grade_and_propose`` extends the teacher's single
  hindsight call to also emit ``EvolutionIntent``. Proposing must cost no extra
  budget token beyond grading, and a malformed proposal must never break grading.
* **WM-4** -- ``accepted_evidence_kinds`` config admits ``world_model_intent`` into
  the observation layer, so an intent reaches the governed pipeline. Default
  configuration must be unchanged (``unknown_tool`` only).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from leapflow.domain.evolution_intent import WORLD_MODEL_INTENT, EvolutionIntent
from leapflow.learning.capability_observation import (
    CapabilityEvidenceClassifier,
    CapabilityObservationBuffer,
    DEFAULT_ACCEPTED_EVIDENCE,
)
from leapflow.world_model.budget import LearningBudgetController
from leapflow.world_model.trajectory_grader import TeacherVerdict, TrajectoryGrader

# TrajectoryGrader's min_trajectory_length defaults to 3.
_TRAJECTORY = [
    {"experience_id": "e1", "action_description": "click send_button",
     "predicted_effect": "message sent", "actual_effect": "no such element", "delta": "1.0"},
    {"experience_id": "e2", "action_description": "retry click",
     "predicted_effect": "message sent", "actual_effect": "no such element", "delta": "1.0"},
    {"experience_id": "e3", "action_description": "scan for alternatives",
     "predicted_effect": "found a send control", "actual_effect": "only submit_button", "delta": "0.7"},
]


class _FakeLLM:
    """Deterministic teacher response source."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0
        self.prompts: list[str] = []

    async def achat(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(str(messages[-1]))
        return SimpleNamespace(content=self._payload)


class _Store:
    """Minimal ExperienceStore stand-in; grading persistence is not under test."""

    def __init__(self) -> None:
        self.updates: list[tuple] = []

    def update_advantage(self, *args, **kwargs):
        self.updates.append((args, kwargs))

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            self.updates.append((name, args, kwargs))
        return _noop


def _grader(payload: str, *, grading_budget: int = 5):
    llm = _FakeLLM(payload)
    budget = LearningBudgetController(grading_budget=grading_budget)
    return TrajectoryGrader(llm, _Store(), budget), llm, budget


_GRADES = json.dumps({"grades": [
    {"step": 1, "advantage": -0.8, "is_forking": True, "grade_label": "harmful"},
    {"step": 2, "advantage": -0.9, "is_forking": False, "grade_label": "harmful"},
    {"step": 3, "advantage": 0.2, "is_forking": False, "grade_label": "acceptable"},
]})

_GRADES_AND_GAP = json.dumps({
    "grades": [
        {"step": 1, "advantage": -0.8, "is_forking": True, "grade_label": "harmful"},
        {"step": 2, "advantage": -0.9, "is_forking": False, "grade_label": "harmful"},
        {"step": 3, "advantage": 0.2, "is_forking": False, "grade_label": "acceptable"},
    ],
    "capability_gaps": [{
        "capability": "chat.reply",
        "hypothesis": "the send affordance was renamed and no adapter targets it",
        "confidence": 0.82,
        "target_affordance": "app.chat.v2",
        "rationale": "every available tool binds send_button, which no longer exists",
        "expected_effect": "the message appears in the thread",
    }],
})


# ── WM-2: the teacher proposes ────────────────────────────────────────────────


def test_grade_and_propose_returns_grades_and_intents():
    grader, llm, _ = _grader(_GRADES_AND_GAP)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY, goal="reply in chat"))
    assert isinstance(verdict, TeacherVerdict)
    assert len(verdict.grades) == 3
    assert len(verdict.intents) == 1
    intent = verdict.intents[0]
    assert isinstance(intent, EvolutionIntent)
    assert intent.capability == "chat.reply"
    assert intent.target_affordance == "app.chat.v2"
    assert 0.8 < intent.confidence < 0.85
    assert llm.calls == 1          # one hindsight call for both outputs


def test_proposing_costs_no_extra_budget_token():
    """Grading and proposing must consume exactly one `grading` token."""
    grader, _, budget = _grader(_GRADES_AND_GAP, grading_budget=1)
    first = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert first.intents                       # succeeded on the only token
    second = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert second.grades == () and second.intents == ()   # budget exhausted
    assert budget.has_tokens("grading") is False


def test_gap_section_only_appears_when_proposing():
    grader, llm, _ = _grader(_GRADES)
    asyncio.run(grader.grade_trajectory(_TRAJECTORY))
    assert "capability_gaps" not in llm.prompts[0]

    grader2, llm2, _ = _grader(_GRADES_AND_GAP)
    asyncio.run(grader2.grade_and_propose(_TRAJECTORY))
    assert "capability_gaps" in llm2.prompts[0]


def test_teacher_is_not_asked_to_choose_a_risk_level():
    """An intent is a hypothesis: the model must not select its own risk ceiling."""
    grader, llm, _ = _grader(_GRADES_AND_GAP)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert "risk" not in llm.prompts[0].lower()
    # ...and whatever it proposed lands at the clamped ceiling.
    assert verdict.intents[0].to_requirement().max_risk_level == "read_only"


def test_grading_still_works_when_no_gaps_are_reported():
    grader, _, _ = _grader(_GRADES)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert len(verdict.grades) == 3
    assert verdict.intents == ()


def test_malformed_gaps_are_discarded_without_losing_grades():
    payload = json.dumps({
        "grades": [
            {"step": 1, "advantage": 0.1, "is_forking": False, "grade_label": "acceptable"},
            {"step": 2, "advantage": 0.2, "is_forking": False, "grade_label": "acceptable"},
            {"step": 3, "advantage": 0.3, "is_forking": False, "grade_label": "acceptable"},
        ],
        "capability_gaps": [
            {"hypothesis": "no capability field"},        # missing capability
            {"capability": "chat.reply"},                  # missing hypothesis
            "not even an object",
            {"capability": "chat.send", "hypothesis": "valid", "confidence": "NaN-ish"},
        ],
    })
    grader, _, _ = _grader(payload)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert len(verdict.grades) == 3                        # grading unaffected
    assert [i.capability for i in verdict.intents] == ["chat.send"]
    assert verdict.intents[0].confidence == 0.0            # unparseable -> 0.0


def test_unparseable_teacher_response_is_survivable():
    grader, _, _ = _grader("the model rambled instead of emitting JSON")
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY))
    assert verdict.grades == () and verdict.intents == ()


def test_short_trajectory_is_not_graded_or_proposed():
    grader, llm, _ = _grader(_GRADES_AND_GAP)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY[:2]))
    assert verdict.grades == () and verdict.intents == ()
    assert llm.calls == 0          # no LLM spend below the minimum length


def test_grade_trajectory_signature_is_unchanged():
    """The pre-existing public API must keep returning a list of grades."""
    grader, _, _ = _grader(_GRADES)
    grades = asyncio.run(grader.grade_trajectory(_TRAJECTORY, goal="reply"))
    assert isinstance(grades, list)
    assert len(grades) == 3


# ── WM-4: config-gated admission into the governed pipeline ───────────────────


def test_default_settings_do_not_admit_world_model_intents():
    settings = SimpleNamespace()                     # no accepted_evidence_kinds at all
    assert CapabilityEvidenceClassifier.from_settings(settings).accepted == DEFAULT_ACCEPTED_EVIDENCE
    settings_empty = SimpleNamespace(accepted_evidence_kinds=())
    assert CapabilityEvidenceClassifier.from_settings(settings_empty).accepted == DEFAULT_ACCEPTED_EVIDENCE


def test_configured_settings_admit_world_model_intents():
    settings = SimpleNamespace(accepted_evidence_kinds=(WORLD_MODEL_INTENT,))
    classifier = CapabilityEvidenceClassifier.from_settings(settings)
    assert classifier.accepted == frozenset({WORLD_MODEL_INTENT})
    intent = EvolutionIntent.create("chat.reply", "v2 send path unserved")
    assert classifier.accepts(intent.to_observation_result()) is True


def test_teacher_intent_reaches_a_requirement_end_to_end():
    """WM-2 + WM-4 + P0-2: teacher output becomes a governed requirement."""
    grader, _, _ = _grader(_GRADES_AND_GAP)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY, goal="reply in chat"))
    settings = SimpleNamespace(accepted_evidence_kinds=(WORLD_MODEL_INTENT,))
    buffer = CapabilityObservationBuffer(
        classifier=CapabilityEvidenceClassifier.from_settings(settings)
    )
    for intent in verdict.intents:
        assert buffer.add_result(intent.to_observation_result()) is True
    requirements = buffer.requirements()
    assert len(requirements) == 1
    req = requirements[0]
    assert req.origin == "world_model"
    assert req.capability == "chat.reply"
    assert req.max_risk_level == "read_only"          # clamped, not model-chosen
    assert dict(req.metadata)["target_affordance"] == "app.chat.v2"


def test_config_default_is_empty_so_shipped_behaviour_is_unchanged():
    """The new setting must default to opt-out."""
    import dataclasses

    from leapflow.config import Settings

    field = next(
        f for f in dataclasses.fields(Settings) if f.name == "accepted_evidence_kinds"
    )
    assert field.default == ()


# ── WM-A: the intent reaches the surface that leads to governed acquisition ───
#
# The engine's observation hook is documented as observe-only and must stay that
# way. The path that actually leads to acquisition is the self-management tool
# chain: plugin_propose (side-effect-free) -> plugin_generate (validated code, no
# install) -> plugin_install (approval-gated). These tests cover the bridge from a
# world-model intent into that chain's PluginProposal shape.


def _intent(**kw):
    base = dict(
        capability="chat.reply",
        hypothesis="the send affordance was renamed and no adapter targets it",
        confidence=0.8,
        target_affordance="app.chat.v2",
        expected_effect="the message appears in the thread",
        rationale="every tool binds send_button, which no longer exists",
    )
    base.update(kw)
    cap = base.pop("capability")
    hyp = base.pop("hypothesis")
    return EvolutionIntent.create(cap, hyp, **base)


def test_intent_becomes_a_plugin_proposal():
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    proposal = CapabilityGapDetector().proposal_from_evolution_intent(_intent())
    assert proposal.plugin_id == "chat_reply_plugin"
    assert proposal.gap_type == "tool_plugin"
    assert proposal.status == "draft"                  # side-effect-free
    assert [t.name for t in proposal.proposed_tools] == ["chat_reply"]
    assert proposal.proposed_tools[0].mutates_state is False

    meta = dict(proposal.evidence[0].metadata)
    assert proposal.evidence[0].evidence_type == WORLD_MODEL_INTENT
    assert meta["target_affordance"] == "app.chat.v2"
    assert meta["expected_effect"] == "the message appears in the thread"
    assert meta["capability"] == "chat.reply"


def test_proposal_risk_is_clamped_not_model_chosen():
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    greedy = _intent(max_risk_level="external")
    proposal = CapabilityGapDetector().proposal_from_evolution_intent(greedy)
    assert proposal.risk_level == "read_only"
    assert proposal.proposed_tools[0].risk_level == "read_only"
    assert proposal.proposed_tools[0].mutates_state is False
    # the denied request stays visible for audit
    assert dict(proposal.evidence[0].metadata)["requested_max_risk_level"] == "external"


def test_trusted_caller_may_raise_the_proposal_ceiling():
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    intent = _intent(max_risk_level="mutating")
    proposal = CapabilityGapDetector().proposal_from_evolution_intent(
        intent, risk_ceiling="external"
    )
    assert proposal.risk_level == "mutating"
    assert proposal.proposed_tools[0].mutates_state is True


def test_teacher_output_reaches_a_proposal_end_to_end():
    """WM-2 -> WM-A: one hindsight call produces a reviewable proposal."""
    from leapflow.learning.capability_gap_detector import CapabilityGapDetector

    grader, _, _ = _grader(_GRADES_AND_GAP)
    verdict = asyncio.run(grader.grade_and_propose(_TRAJECTORY, goal="reply in chat"))
    detector = CapabilityGapDetector()
    proposals = [detector.proposal_from_evolution_intent(i) for i in verdict.intents]
    assert len(proposals) == 1
    assert proposals[0].capability_summary.startswith("the send affordance")
    assert proposals[0].risk_level == "read_only"
    assert dict(proposals[0].evidence[0].metadata)["target_affordance"] == "app.chat.v2"

"""WM-B: the world model is now the first driver of capability evolution.

`grade_and_propose` could form a capability hypothesis and the observation pipeline
could consume one, but nothing joined them -- so the world model's conclusions
reached no part of the system. `WorldModelEvolutionDriver` is that join.

The tests that matter most here are the negative ones: the driver must not be able
to bypass the opt-in evidence gate, must not widen a risk ceiling, and must never
fail the session that produced the trajectory.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from leapflow.domain.evolution_intent import WORLD_MODEL_INTENT, EvolutionIntent
from leapflow.learning.capability_observation import (
    CapabilityEvidenceClassifier,
    CapabilityObservationService,
)
from leapflow.learning.world_model_driver import (
    CapabilityGapTeacher,
    EvidenceIntake,
    WorldModelDriveResult,
    WorldModelEvolutionDriver,
)
from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
from leapflow.world_model.trajectory_grader import TeacherVerdict

_TRAJECTORY = [
    {"experience_id": "e1", "action_description": "click send_button",
     "predicted_effect": "sent", "actual_effect": "silently no-op", "delta": "1.0"},
    {"experience_id": "e2", "action_description": "retry",
     "predicted_effect": "sent", "actual_effect": "silently no-op", "delta": "1.0"},
]


class _Teacher:
    """Stand-in for TrajectoryGrader with a fixed hindsight verdict."""

    def __init__(self, intents=(), grades=("g1", "g2"), raises=False) -> None:
        self._verdict = TeacherVerdict(tuple(grades), tuple(intents))
        self._raises = raises
        self.calls = 0

    async def grade_and_propose(self, trajectory, goal=""):
        self.calls += 1
        if self._raises:
            raise RuntimeError("teacher exploded")
        return self._verdict


def _intent(**kw):
    base = dict(
        confidence=0.8, target_affordance="app.chat.v2",
        expected_effect="message appears in the thread",
    )
    base.update(kw)
    return EvolutionIntent.create("chat.reply", "the send path silently no-ops", **base)


def _service(tmp_path, *, opted_in: bool):
    store = JsonCapabilityObservationStore(tmp_path / "observations.json")
    kinds = [WORLD_MODEL_INTENT] if opted_in else None
    classifier = CapabilityEvidenceClassifier.from_kinds(kinds) if kinds else None
    return store, CapabilityObservationService(store, classifier=classifier)


def _drive(teacher, service, trajectory=_TRAJECTORY, **kw):
    driver = WorldModelEvolutionDriver(teacher=teacher, intake=service, **kw)
    return asyncio.run(driver.drive(trajectory, "reply in chat"))


# ── the join works ────────────────────────────────────────────────────────────


def test_protocols_are_satisfied_by_the_real_components(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    assert isinstance(service, EvidenceIntake)
    assert isinstance(_Teacher(), CapabilityGapTeacher)


def test_teacher_hypothesis_becomes_a_governed_requirement(tmp_path):
    """The whole point: hindsight -> intent -> admitted evidence -> requirement."""
    store, service = _service(tmp_path, opted_in=True)
    result = _drive(_Teacher(intents=[_intent()]), service)

    assert isinstance(result, WorldModelDriveResult)
    assert result.proposed == 1
    assert result.admitted == 1
    assert len(store.unresolved()) == 1

    requirement = result.requirements[0]
    assert requirement.origin == "world_model"       # the world model drove this
    assert requirement.capability == "chat.reply"
    assert requirement.max_risk_level == "read_only"
    assert dict(requirement.metadata)["target_affordance"] == "app.chat.v2"


def test_grades_are_returned_so_no_second_llm_call_is_needed(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    teacher = _Teacher(intents=[_intent()])
    result = _drive(teacher, service)
    assert len(result.grades) == 2
    assert teacher.calls == 1          # one hindsight call for grading AND proposing


def test_multiple_gaps_all_reach_the_pipeline(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    other = EvolutionIntent.create("chat.attach", "no attachment capability exists")
    result = _drive(_Teacher(intents=[_intent(), other]), service)
    assert result.proposed == 2 and result.admitted == 2
    assert {r.capability for r in result.requirements} == {"chat.reply", "chat.attach"}


# ── the driver must not be able to bypass the gate ────────────────────────────


def test_driver_cannot_bypass_the_opt_in_gate(tmp_path):
    """Default configuration: proposals are formed but change nothing."""
    store, service = _service(tmp_path, opted_in=False)
    result = _drive(_Teacher(intents=[_intent()]), service)

    assert result.proposed == 1        # the world model did form a hypothesis
    assert result.admitted == 0        # ...and the gate refused it
    assert result.requirements == ()
    assert store.unresolved() == []    # nothing durable was written


def test_driver_cannot_widen_the_risk_ceiling(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    greedy = _intent(max_risk_level="external")
    result = _drive(_Teacher(intents=[greedy]), service)
    requirement = result.requirements[0]
    assert requirement.max_risk_level == "read_only"
    assert dict(requirement.metadata)["requested_max_risk_level"] == "external"


def test_caller_may_tighten_the_ceiling_further(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    result = _drive(
        _Teacher(intents=[_intent(max_risk_level="medium")]), service,
        risk_ceiling="read_only",
    )
    assert result.requirements[0].max_risk_level == "read_only"


# ── learning must never break the session ─────────────────────────────────────


def test_teacher_failure_is_contained(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    result = _drive(_Teacher(raises=True), service)
    assert result == WorldModelDriveResult()      # empty, not an exception


def test_intake_failure_is_contained(tmp_path):
    class _BrokenIntake:
        def observe_result(self, result, **kwargs):
            raise OSError("disk on fire")

        def requirements(self, *, min_count=1, limit=50):
            return ()

    driver = WorldModelEvolutionDriver(teacher=_Teacher(intents=[_intent()]), intake=_BrokenIntake())
    result = asyncio.run(driver.drive(_TRAJECTORY, "goal"))
    assert result.proposed == 1 and result.admitted == 0


def test_empty_trajectory_spends_nothing(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    teacher = _Teacher(intents=[_intent()])
    result = _drive(teacher, service, trajectory=[])
    assert result == WorldModelDriveResult()
    assert teacher.calls == 0          # no LLM spend without an episode


def test_no_gaps_still_returns_grades(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    result = _drive(_Teacher(intents=[]), service)
    assert len(result.grades) == 2
    assert result.proposed == 0 and result.requirements == ()


def test_drive_result_is_reportable(tmp_path):
    _, service = _service(tmp_path, opted_in=True)
    payload = _drive(_Teacher(intents=[_intent()]), service).to_dict()
    assert payload["proposed"] == 1
    assert payload["admitted"] == 1
    assert payload["capabilities"] == ["chat.reply"]


# ── the real grader satisfies the teacher contract ────────────────────────────


def test_real_trajectory_grader_can_drive_evolution(tmp_path):
    """End to end with the REAL TrajectoryGrader, only the LLM substituted."""
    import json

    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.trajectory_grader import TrajectoryGrader

    payload = json.dumps({
        "grades": [
            {"step": 1, "advantage": -0.9, "is_forking": True, "grade_label": "harmful"},
            {"step": 2, "advantage": -0.9, "is_forking": False, "grade_label": "harmful"},
            {"step": 3, "advantage": -0.5, "is_forking": False, "grade_label": "suboptimal"},
        ],
        "capability_gaps": [{
            "capability": "chat.reply",
            "hypothesis": "the send control exists but no longer delivers the message",
            "confidence": 0.77,
            "target_affordance": "app.chat.v2",
            "expected_effect": "the message appears in the thread",
        }],
    })

    class _FakeLLM:
        async def achat(self, messages, **kwargs):
            return SimpleNamespace(content=payload)

    class _Store:
        def __getattr__(self, name):
            return lambda *a, **k: None

    grader = TrajectoryGrader(_FakeLLM(), _Store(), LearningBudgetController(grading_budget=2))
    assert isinstance(grader, CapabilityGapTeacher)

    _, service = _service(tmp_path, opted_in=True)
    trajectory = _TRAJECTORY + [
        {"experience_id": "e3", "action_description": "verify thread",
         "predicted_effect": "message present", "actual_effect": "absent", "delta": "1.0"},
    ]
    driver = WorldModelEvolutionDriver(teacher=grader, intake=service)
    result = asyncio.run(driver.drive(trajectory, "reply in chat"))

    assert result.proposed == 1
    assert result.admitted == 1
    assert result.requirements[0].origin == "world_model"
    assert result.requirements[0].capability == "chat.reply"

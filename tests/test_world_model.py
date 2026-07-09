"""Scenario-based integration tests for the world model subsystem."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from conftest import StubLLM
from leapflow.memory.providers.semantic import SemanticMemoryProvider
from leapflow.perception.state_snapshot import SnapshotFidelity, StateSnapshot
from leapflow.world_model._json_utils import extract_json_object
from leapflow.world_model.budget import LearningBudgetController
from leapflow.world_model.curiosity import CuriosityConfig, CuriositySignal
from leapflow.world_model.experience_store import ExperienceStore
from leapflow.world_model.prediction import Prediction, PredictionOutcome
from leapflow.world_model.replay import ExperienceReplayEngine
from leapflow.world_model.trajectory_grader import (
    DEFAULT_GRADE_LABELS,
    TrajectoryGrader,
)


# ── Helpers ────────────────────────────────────────────────────────


def _open_store(tmp_path) -> tuple[SemanticMemoryProvider, ExperienceStore]:
    lt = SemanticMemoryProvider(source=tmp_path / "world_model.duckdb")
    # _ensure_connection() auto-initializes on first legacy method call
    lt._ensure_connection()
    return lt, ExperienceStore(lt)


def _close_store(lt: SemanticMemoryProvider) -> None:
    lt.close()


def _make_snapshot(
    *,
    app: str = "com.apple.finder",
    action_context: str = "finder",
) -> StateSnapshot:
    return StateSnapshot(
        timestamp=time.time(),
        fidelity=SnapshotFidelity.LIGHT,
        app_bundle_id=app,
        window_title=f"{action_context} window",
        clipboard_text="",
        recent_events=("event_a",),
        ax_digest="digest_a",
        ax_summary="summary",
        screenshot_phash="",
    )


def _make_outcome(
    *,
    delta: float = 0.3,
    action: str = "click save",
    expected: str = "file saved",
    actual: str = "dialog appeared",
    app: str = "com.apple.finder",
    experience_id: str = "",
) -> PredictionOutcome:
    return PredictionOutcome(
        prediction=Prediction(
            action_description=action,
            expected_effect=expected,
            confidence=0.8,
        ),
        pre_snapshot=_make_snapshot(app=app),
        post_snapshot=_make_snapshot(app=app, action_context="changed"),
        actual_effect=actual,
        delta=delta,
        delta_source="structural",
        timestamp=time.time(),
        experience_id=experience_id,
    )


@dataclass
class _MockCausalEvent:
    confidence: float = 0.5
    channel: str = "ui"
    app_context: str = "com.apple.finder"


class _MockCausalGraph:
    def __init__(self, event_count: int = 0) -> None:
        self.events = {
            f"ev_{i}": _MockCausalEvent(confidence=0.4 + (i % 5) * 0.1)
            for i in range(event_count)
        }


def _grade_response(steps: int) -> str:
    grades = [
        {
            "step": i + 1,
            "advantage": 0.5 - i * 0.1,
            "is_forking": i == 1,
            "grade_label": DEFAULT_GRADE_LABELS[i % len(DEFAULT_GRADE_LABELS)],
        }
        for i in range(steps)
    ]
    return json.dumps({"grades": grades})


def _replay_insight_response() -> str:
    return json.dumps({
        "insights": [
            {
                "type": "causal_transfer",
                "description": "Save shortcuts behave differently across apps",
                "confidence": 0.85,
                "actionable": True,
            },
            {
                "type": "edge_correction",
                "description": "Modal dialogs block expected save effects",
                "confidence": 0.7,
                "actionable": False,
            },
        ],
    })


def _distill_response() -> str:
    return json.dumps({
        "distilled_rules": [
            {
                "type": "heuristic",
                "description": "Confirm save dialog before expecting file write",
                "confidence": 0.9,
                "actionable": True,
                "source_grade": "optimal",
            },
            {
                "type": "correction",
                "description": "Avoid dismissing dialogs when saving",
                "confidence": 0.75,
                "actionable": True,
                "source_grade": "harmful",
            },
        ],
    })


def _store_experience(
    store: ExperienceStore,
    *,
    action: str = "click save button",
    app: str = "com.apple.finder",
    predicted: str = "file saved",
    actual: str = "save dialog opened",
    delta: float = 0.2,
    advantage: float = 0.0,
    grade_label: str = "",
    is_forking: bool = False,
) -> str:
    return store.store(
        action_description=action,
        app_context=app,
        predicted_effect=predicted,
        actual_effect=actual,
        delta=delta,
        advantage=advantage,
        grade_label=grade_label,
        is_forking=is_forking,
    )


# ── 1. Budget lifecycle ──────────────────────────────────────────


def test_budget_lifecycle() -> None:
    budget = LearningBudgetController(
        prediction_budget=3,
        comparison_budget=2,
        replay_budget=1,
        grading_budget=2,
        distillation_budget=1,
    )

    assert budget.has_tokens("prediction")
    assert budget.spend("prediction") is True
    assert budget.spend("prediction") is True
    assert budget.spend("prediction") is True
    assert budget.spend("prediction") is False
    assert not budget.has_tokens("prediction")

    status = budget.status
    assert status["prediction"]["tokens"] == 0.0
    assert status["prediction"]["max"] == 3.0
    assert status["comparison"]["tokens"] == 2.0

    budget.reset()
    assert budget.has_tokens("prediction")
    assert budget.status["prediction"]["tokens"] == 3.0

    low_accuracy_budget = LearningBudgetController(prediction_budget=3, replay_budget=1)
    low_accuracy_budget.adjust_for_accuracy(0.05)
    assert low_accuracy_budget.status["prediction"]["max"] == pytest.approx(2.4)
    assert low_accuracy_budget.status["replay"]["max"] == pytest.approx(1.2)

    high_error_budget = LearningBudgetController(prediction_budget=3, comparison_budget=2)
    high_error_budget.adjust_for_accuracy(0.7)
    assert high_error_budget.status["prediction"]["max"] == pytest.approx(3.9)
    assert high_error_budget.status["comparison"]["max"] == pytest.approx(2.4)


# ── 2. Experience store and retrieve ─────────────────────────────


def test_experience_store_and_retrieve(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        ids = [
            _store_experience(
                store,
                action="click save document",
                delta=0.8,
                actual="permission denied",
            ),
            _store_experience(
                store,
                action="click save draft",
                delta=0.1,
                actual="file saved",
            ),
            _store_experience(
                store,
                action="open preferences panel",
                app="com.apple.systempreferences",
                delta=0.6,
                actual="settings opened",
            ),
        ]
        assert all(isinstance(eid, str) and eid for eid in ids)
        assert store.count() == 3

        similar = store.retrieve_similar("click save", "com.apple.finder", limit=5)
        assert len(similar) >= 2
        assert all(e.app_context == "com.apple.finder" for e in similar)

        high_delta = store.retrieve_high_delta(delta_min=0.5, limit=20)
        assert len(high_delta) == 2
        assert all(e.delta >= 0.5 for e in high_delta)
        high_actions = {e.action_description for e in high_delta}
        assert "click save document" in high_actions
        assert "open preferences panel" in high_actions
    finally:
        _close_store(lt)


# ── 3. OPD advantage workflow ────────────────────────────────────


def test_experience_opd_advantage_workflow(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        good_id = _store_experience(store, action="click save good", delta=0.3)
        bad_id = _store_experience(store, action="click save bad", delta=0.7)
        neutral_id = _store_experience(store, action="click save neutral", delta=0.4)

        store.update_advantage(good_id, advantage=0.8, grade_label="optimal")
        store.update_advantage(bad_id, advantage=-0.6, is_forking=True, grade_label="harmful")
        store.update_advantage(neutral_id, advantage=0.1, grade_label="acceptable")

        filtered = store.retrieve_similar(
            "click save",
            "com.apple.finder",
            limit=10,
        )
        assert len(filtered) == 3

        non_negative = store.retrieve_similar(
            "click save",
            "com.apple.finder",
            limit=10,
            advantage_floor=0.0,
        )
        assert len(non_negative) == 2
        assert all(e.advantage >= 0.0 for e in non_negative)

        positive_only = store.retrieve_similar(
            "click save",
            "com.apple.finder",
            limit=10,
            advantage_floor=0.2,
        )
        assert len(positive_only) == 1
        assert positive_only[0].experience_id == good_id
        assert positive_only[0].advantage == pytest.approx(0.8)
        assert positive_only[0].grade_label == "optimal"
    finally:
        _close_store(lt)


# ── 4. On-policy boost ranking ───────────────────────────────────


def test_on_policy_boost_prioritizes_session_data(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        off_policy_id = _store_experience(
            store,
            action="click save shared",
            delta=0.4,
            advantage=0.2,
        )
        time.sleep(0.02)
        store.mark_session_start()
        on_policy_id = _store_experience(
            store,
            action="click save shared",
            delta=0.4,
            advantage=0.2,
        )

        ranked = store.retrieve_similar(
            "click save shared",
            "com.apple.finder",
            limit=2,
            on_policy_boost=1.5,
        )
        assert len(ranked) == 2
        assert ranked[0].experience_id == on_policy_id
        assert ranked[1].experience_id == off_policy_id
        assert ranked[0].timestamp >= store.session_start
        assert ranked[1].timestamp < store.session_start
    finally:
        _close_store(lt)


# ── 5. Curiosity signal components ───────────────────────────────


def test_curiosity_signal_components(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        early_config = CuriosityConfig(auto_balance=True)
        early_signal = CuriositySignal(early_config, store)

        high_delta = _make_outcome(delta=0.9, action="rename file")
        first_score = early_signal.compute(high_delta)
        assert first_score.maturity_stage == "early"
        assert first_score.prediction_surprise == pytest.approx(0.9)
        assert first_score.information_gain == pytest.approx(0.0)
        assert first_score.frequency_novelty == pytest.approx(1.0)
        expected_early = 0.2 * 0.9 + 0.3 * 0.0 + 0.5 * 1.0
        assert first_score.total == pytest.approx(expected_early)

        repeat_score = early_signal.compute(high_delta)
        assert repeat_score.frequency_novelty == pytest.approx(1.0 / (2 ** 0.5))
        assert repeat_score.total < first_score.total

        mature_config = CuriosityConfig(
            auto_balance=True,
            early_event_threshold=0,
            early_experience_threshold=0,
            middle_event_threshold=0,
            middle_experience_threshold=0,
        )
        graph = _MockCausalGraph(event_count=5)
        mature_signal = CuriositySignal(mature_config, store, causal_graph=graph)
        graph_score = mature_signal.compute(_make_outcome(delta=0.5, action="copy text"))
        assert graph_score.maturity_stage == "mature"
        assert graph_score.information_gain > 0.0
        expected_mature = 0.6 * 0.5 + 0.3 * graph_score.information_gain + 0.1 * 1.0
        assert graph_score.total == pytest.approx(min(1.0, max(0.0, expected_mature)))
    finally:
        _close_store(lt)


# ── 6. Curiosity advantage modulation ────────────────────────────


def test_curiosity_advantage_modulation(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        config = CuriosityConfig(advantage_modulation=0.3)
        outcome = _make_outcome(delta=0.6)

        base_signal = CuriositySignal(config, store)
        base = base_signal.compute(outcome)

        positive_signal = CuriositySignal(config, store)
        positive = positive_signal.compute_with_trajectory_context(outcome, advantage=0.8)

        negative_signal = CuriositySignal(config, store)
        negative = negative_signal.compute_with_trajectory_context(outcome, advantage=-0.8)

        zero_signal = CuriositySignal(config, store)
        zero = zero_signal.compute_with_trajectory_context(outcome, advantage=0.0)

        assert zero.total == pytest.approx(base.total)
        assert positive.total < base.total
        assert negative.total > base.total
        assert positive.prediction_surprise == base.prediction_surprise
        assert negative.frequency_novelty == base.frequency_novelty

        expected_positive = base.total * (1.0 - 0.3 * 0.8)
        expected_negative = min(1.0, base.total * (1.0 - 0.3 * (-0.8)))
        assert positive.total == pytest.approx(expected_positive)
        assert negative.total == pytest.approx(expected_negative)
    finally:
        _close_store(lt)


# ── 7. Trajectory grading end-to-end ─────────────────────────────


@pytest.mark.asyncio
async def test_trajectory_grading_end_to_end(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        exp_ids = [
            _store_experience(store, action=f"step action {i}", delta=0.2 + i * 0.1)
            for i in range(3)
        ]
        trajectory = [
            {
                "experience_id": exp_ids[i],
                "action_description": f"step action {i}",
                "predicted_effect": "expected",
                "actual_effect": "observed",
                "delta": 0.2 + i * 0.1,
            }
            for i in range(3)
        ]

        llm = StubLLM(replies=[_grade_response(3)])
        budget = LearningBudgetController(grading_budget=5)
        grader = TrajectoryGrader(llm, store, budget)

        grades = await grader.grade_trajectory(trajectory, goal="save the document")
        assert len(grades) == 3
        assert grades[0].experience_id == exp_ids[0]
        assert grades[1].is_forking is True
        assert grades[0].grade_label in DEFAULT_GRADE_LABELS

        similar = store.retrieve_similar("step action", "com.apple.finder", limit=5)
        by_id = {e.experience_id: e for e in similar}
        assert by_id[exp_ids[0]].advantage == pytest.approx(0.5)
        assert by_id[exp_ids[0]].grade_label == DEFAULT_GRADE_LABELS[0]
        assert by_id[exp_ids[1]].is_forking is True
        assert budget.status["grading"]["tokens"] == 4.0
    finally:
        _close_store(lt)


# ── 8. Trajectory grading respects budget ────────────────────────


@pytest.mark.asyncio
async def test_trajectory_grading_respects_budget(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        trajectory = [
            {
                "experience_id": _store_experience(store, action=f"budget action {i}"),
                "action_description": f"budget action {i}",
                "predicted_effect": "p",
                "actual_effect": "a",
                "delta": 0.3,
            }
            for i in range(3)
        ]
        llm = StubLLM(replies=[_grade_response(3), _grade_response(3)])
        budget = LearningBudgetController(grading_budget=1)
        grader = TrajectoryGrader(llm, store, budget)

        first = await grader.grade_trajectory(trajectory)
        second = await grader.grade_trajectory(trajectory)

        assert len(first) == 3
        assert second == []
        assert budget.status["grading"]["tokens"] == 0.0
        assert llm.call_count == 1
    finally:
        _close_store(lt)


# ── 9. Replay session discovers insights ─────────────────────────


@pytest.mark.asyncio
async def test_replay_session_discovers_insights(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        for i in range(3):
            _store_experience(
                store,
                action=f"replay action {i}",
                delta=0.55 + i * 0.05,
                actual=f"unexpected outcome {i}",
            )

        captured: List[Any] = []

        def on_insight(insight) -> None:
            captured.append(insight)

        llm = StubLLM(replies=[_replay_insight_response()])
        budget = LearningBudgetController(replay_budget=2)
        engine = ExperienceReplayEngine(llm, store, budget, on_insight=on_insight)

        insights = await engine.replay_session()
        assert len(insights) == 2
        assert insights[0].insight_type == "causal_transfer"
        assert insights[0].confidence == pytest.approx(0.85)
        assert insights[0].actionable is True
        assert len(insights[0].source_experiences) == 3
        assert len(captured) == 1
        assert budget.status["replay"]["tokens"] == 1.0
    finally:
        _close_store(lt)


# ── 10. Self-distill from graded experiences ───────────────────


@pytest.mark.asyncio
async def test_self_distill_from_graded_experiences(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        _store_experience(
            store,
            action="distill optimal save",
            delta=0.3,
            advantage=0.9,
            grade_label="optimal",
        )
        _store_experience(
            store,
            action="distill harmful dismiss",
            delta=0.8,
            advantage=-0.7,
            grade_label="harmful",
            is_forking=True,
        )
        _store_experience(
            store,
            action="distill ungraded action",
            delta=0.4,
        )

        llm = StubLLM(replies=[_distill_response()])
        budget = LearningBudgetController(distillation_budget=2)
        engine = ExperienceReplayEngine(llm, store, budget)

        rules = await engine.self_distill()
        assert len(rules) == 2
        assert rules[0].insight_type == "heuristic"
        assert rules[1].insight_type == "correction"
        assert rules[0].description.startswith("Confirm save dialog")
        assert len(rules[0].source_experiences) == 2
        assert budget.status["distillation"]["tokens"] == 1.0
    finally:
        _close_store(lt)


# ── 11. Regression detection ─────────────────────────────────────


def test_regression_detection(tmp_path) -> None:
    lt, store = _open_store(tmp_path)
    try:
        for i in range(6):
            _store_experience(
                store,
                action=f"baseline action {i}",
                delta=0.1,
                actual="as predicted",
            )

        budget = LearningBudgetController()
        engine = ExperienceReplayEngine(StubLLM(), store, budget)

        stable_outcomes = [_make_outcome(delta=0.12) for _ in range(5)]
        assert engine.detect_regression(stable_outcomes, window=5) is False

        regressing_outcomes = [_make_outcome(delta=0.55 + i * 0.02) for i in range(5)]
        assert engine.detect_regression(regressing_outcomes, window=5) is True
        assert engine.detect_regression(regressing_outcomes[:3], window=5) is False
    finally:
        _close_store(lt)


# ── 12. JSON extraction robustness ───────────────────────────────


def test_json_extraction_robustness() -> None:
    assert extract_json_object('{"grades": [{"step": 1}]}') == {"grades": [{"step": 1}]}

    wrapped = 'Here is the result:\n```json\n{"insights": [{"type": "pattern"}]}\n```\nDone.'
    assert extract_json_object(wrapped) == {"insights": [{"type": "pattern"}]}

    assert extract_json_object("{not valid json}") == {}
    assert extract_json_object("no braces here") == {}
    assert extract_json_object("") == {}

    nested = 'prefix {"outer": {"inner": 1}, "list": [1, 2]} suffix'
    assert extract_json_object(nested) == {"outer": {"inner": 1}, "list": [1, 2]}

    trailing_comma = '{"grades": [{"step": 1,}],}'
    assert extract_json_object(trailing_comma) == {}

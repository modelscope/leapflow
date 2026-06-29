"""Scenario-based integration tests for the learn-distill lifecycle."""

from __future__ import annotations

import time
from typing import List, Optional
from unittest.mock import patch

import pytest

from conftest import make_action, make_candidate, make_episode, make_event, make_skill

from leapflow.analysis.pipeline import ImitationPipeline
from leapflow.domain.trajectory import (
    ActionType,
    Episode,
    RawAction,
    SemanticAction,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
)
from leapflow.engine.session import (
    LearnResult,
    SessionController,
    SessionMode,
)
from leapflow.learning.active_learning import ActiveLearningObserver
from leapflow.learning.feedback import FeedbackEvaluator
from leapflow.learning.similarity import HeuristicSimilarityScorer
from leapflow.memory.providers.working import WorkingMemoryProvider
from leapflow.skills.registry import SkillRegistry
from leapflow.storage.skill_library import SkillLibraryStore, StoredSkill
from leapflow.storage.trajectory_store import TrajectoryStore


# ── Fixtures ──


@pytest.fixture
def session_controller(
    imitation_pipeline: ImitationPipeline,
    skill_registry: SkillRegistry,
) -> SessionController:
    return SessionController(
        imitation_pipeline,
        skill_registry,
        idle_timeout=300.0,
        auto_learn=False,
    )


@pytest.fixture
def active_observer(
    skill_library: SkillLibraryStore,
    working_memory: WorkingMemoryProvider,
) -> ActiveLearningObserver:
    scorer = HeuristicSimilarityScorer()
    return ActiveLearningObserver(
        skill_library,
        scorer,
        working_memory,
        heuristic_low=0.1,
        heuristic_high=0.99,
    )


@pytest.fixture
def feedback_observer(
    skill_library: SkillLibraryStore,
    working_memory: WorkingMemoryProvider,
) -> ActiveLearningObserver:
    scorer = HeuristicSimilarityScorer()
    evaluator = FeedbackEvaluator(trajectory_store=None)
    return ActiveLearningObserver(
        skill_library,
        scorer,
        working_memory,
        feedback_evaluator=evaluator,
        heuristic_low=0.1,
        heuristic_high=0.5,
    )


# ── Helpers ──


def _make_stored_skill(
    *,
    skill_id: str = "skill_001",
    title: str = "Organize downloads",
    steps: Optional[List[str]] = None,
    triggers: Optional[List[str]] = None,
    action_names: Optional[List[str]] = None,
    version: int = 1,
    confidence: float = 0.7,
    traj_id: str = "t1",
    ep_id: str = "e1",
) -> StoredSkill:
    return StoredSkill(
        skill_id=skill_id,
        title=title,
        trigger_phrases=triggers or ["organize files"],
        steps=steps or ["List", "Classify", "Move"],
        parameters=[],
        pre_conditions=[],
        post_conditions=[],
        app_sequence=["com.apple.finder"],
        action_names=action_names or ["list", "classify", "move"],
        source_trajectory_id=traj_id,
        source_episode_id=ep_id,
        confidence=confidence,
        version=version,
        status="active",
    )


def _build_trajectory(
    tid: str,
    *,
    goal: str = "test",
    step_count: int = 1,
) -> Trajectory:
    traj = Trajectory(trajectory_id=tid, metadata={"goal": goal})
    base_ts = time.time()
    for i in range(step_count):
        ts = base_ts + i
        step = TrajectoryStep(
            state=StateSnapshot(
                timestamp=ts,
                focused_app="com.app",
            ),
            action=RawAction(
                timestamp=ts,
                action_type=ActionType.UI_CLICK,
                target="button",
                app_bundle_id="com.app",
                app_name="App",
                params={"source": "click", "x": 100 + i},
            ),
            post_state=StateSnapshot(
                timestamp=ts + 0.1,
                focused_app="com.app",
            ),
        )
        traj.steps.append(step)
    traj.start_time = base_ts
    traj.end_time = base_ts + step_count
    return traj


def _learning_events() -> list:
    return [
        make_event(
            "app.focus_change",
            {"bundle_id": "com.apple.finder", "app_name": "Finder"},
            ts=1.0,
            source="com.apple.finder",
        ),
        make_event("fs.change", {"path": "/Downloads/a.pdf", "action": "created"}, ts=2.0),
        make_event("fs.change", {"path": "/Downloads/b.jpg", "action": "created"}, ts=3.0),
        make_event("fs.change", {"path": "/Downloads/a.pdf", "action": "modified"}, ts=4.0),
        make_event(
            "app.focus_change",
            {"bundle_id": "com.apple.Terminal", "app_name": "Terminal"},
            ts=5.0,
            source="com.apple.Terminal",
        ),
    ]


# ── Session + pipeline lifecycle ──


@pytest.mark.asyncio
async def test_learn_record_analyze_lifecycle(
    session_controller: SessionController,
    imitation_pipeline: ImitationPipeline,
):
    session = await session_controller.enter_learning(goal="organize downloads")
    assert session_controller.mode == SessionMode.LEARNING

    for event in _learning_events():
        imitation_pipeline.recorder.on_event(event)

    result = await session_controller.exit_learning()
    assert session_controller.mode == SessionMode.IDLE
    assert isinstance(result, LearnResult)
    assert result.step_count == 5

    episodes = await imitation_pipeline.analyze(
        session.trajectory_id,
        goal="organize downloads",
    )
    assert len(episodes) >= 1
    assert all(ep.trajectory_id == session.trajectory_id for ep in episodes)


@pytest.mark.asyncio
async def test_pipeline_record_stop_distill(imitation_pipeline: ImitationPipeline):
    tid = await imitation_pipeline.start_recording()
    for event in _learning_events():
        imitation_pipeline.recorder.on_event(event)

    traj = await imitation_pipeline.stop_recording()
    assert traj is not None
    assert traj.step_count == 5

    candidates = await imitation_pipeline.distill(tid, goal="organize downloads")
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    assert candidates[0].source_trajectory_id == tid


@pytest.mark.asyncio
async def test_session_mode_transitions(session_controller: SessionController):
    assert session_controller.mode == SessionMode.IDLE

    await session_controller.enter_learning(goal="organize downloads")
    assert session_controller.mode == SessionMode.LEARNING

    traj = _build_trajectory(
        session_controller.current_session.trajectory_id,
        goal="organize downloads",
        step_count=3,
    )
    with patch.object(
        session_controller._pipeline,
        "stop_recording",
        return_value=traj,
    ):
        await session_controller.exit_learning()

    assert session_controller.mode == SessionMode.IDLE


@pytest.mark.asyncio
async def test_session_skill_execution(
    session_controller: SessionController,
    skill_registry: SkillRegistry,
    imitation_pipeline: ImitationPipeline,
):
    ran = {"value": False}

    async def _run(**kwargs):
        ran["value"] = True
        return "executed"

    skill_registry.register(
        make_skill("organize_downloads", run_fn=_run, triggers=["organize files"])
    )

    await session_controller.enter_learning(goal="organize downloads")
    assert session_controller.mode == SessionMode.LEARNING

    # SessionController blocks execute_skill during LEARNING; invoke via registry.
    result = await skill_registry.invoke("organize_downloads")
    assert result.ok
    assert ran["value"]

    for event in _learning_events()[:2]:
        imitation_pipeline.recorder.on_event(event)

    traj = _build_trajectory(
        session_controller.current_session.trajectory_id,
        step_count=2,
    )
    with patch.object(imitation_pipeline, "stop_recording", return_value=traj):
        await session_controller.exit_learning()
    assert session_controller.mode == SessionMode.IDLE


# ── Active learning ──


def test_new_candidate_becomes_skill(
    skill_library: SkillLibraryStore,
    active_observer: ActiveLearningObserver,
):
    assert skill_library.load_all_active() == []

    candidate = make_candidate("Organize downloads")
    episode = make_episode()
    active_observer.on_candidates_ready([candidate], [episode])

    active = skill_library.load_all_active()
    assert len(active) == 1
    assert active[0].title == "Organize downloads"
    assert skill_library.count_pending() == 0


def test_similar_candidate_skipped(
    skill_library: SkillLibraryStore,
    working_memory: WorkingMemoryProvider,
):
    skill_library.save_skill(_make_stored_skill())
    scorer = HeuristicSimilarityScorer()
    observer = ActiveLearningObserver(
        skill_library,
        scorer,
        working_memory,
        heuristic_low=0.1,
        heuristic_high=0.99,
    )

    candidate = make_candidate("Organize downloads")
    episode = make_episode()
    observer.on_candidates_ready([candidate], [episode])

    assert len(skill_library.load_all_active()) == 1
    assert skill_library.count_pending() == 0


# ── Feedback loop ──


def test_feedback_unchanged_logs_execution(
    skill_library: SkillLibraryStore,
    feedback_observer: ActiveLearningObserver,
):
    skill_library.save_skill(_make_stored_skill())

    candidate = make_candidate("Organize downloads")
    episode = make_episode()
    feedback_observer.on_candidates_ready([candidate], [episode])

    executions = skill_library.load_executions("skill_001")
    assert len(executions) == 1
    assert executions[0].verdict == "unchanged"
    assert skill_library.load_skill("skill_001").version == 1


def test_feedback_additive_auto_applies(
    skill_library: SkillLibraryStore,
    feedback_observer: ActiveLearningObserver,
):
    skill_library.save_skill(_make_stored_skill())

    candidate = make_candidate(
        "Organize downloads",
        steps=["List directory", "Classify by type", "Rename by date", "Move files"],
        triggers=["organize files", "sort downloads", "rename and move"],
    )
    episode = make_episode(
        actions=[
            make_action("list", description="List dir", raw_range=(0, 1)),
            make_action("classify", description="Classify", raw_range=(1, 2)),
            make_action("rename", description="Rename by date", raw_range=(2, 3)),
            make_action("move", description="Move files", raw_range=(3, 4)),
        ],
    )
    feedback_observer.on_candidates_ready([candidate], [episode])

    executions = skill_library.load_executions("skill_001")
    assert len(executions) == 1
    assert executions[0].verdict == "auto_applied"

    updated = skill_library.load_skill("skill_001")
    assert updated.version == 2
    assert "Rename by date" in updated.steps
    assert "rename and move" in updated.trigger_phrases


def test_feedback_structural_creates_suggestion(
    skill_library: SkillLibraryStore,
    feedback_observer: ActiveLearningObserver,
):
    skill_library.save_skill(_make_stored_skill())

    candidate = make_candidate(
        "Organize downloads",
        steps=["List directory", "Tag files", "Move files"],
        triggers=["organize files", "tag and move"],
    )
    episode = make_episode(
        actions=[
            make_action("list", description="List dir", raw_range=(0, 1)),
            make_action("tag", description="Tag files", raw_range=(1, 2)),
            make_action("move", description="Move files", raw_range=(2, 3)),
        ],
    )
    feedback_observer.on_candidates_ready([candidate], [episode])

    executions = skill_library.load_executions("skill_001")
    assert len(executions) == 1
    assert executions[0].verdict == "suggested"
    assert skill_library.count_pending() == 1

    suggestions = skill_library.load_pending_suggestions()
    assert suggestions[0].suggestion_type == "feedback_improvement"
    assert skill_library.load_skill("skill_001").version == 1


# ── Storage ──


def test_trajectory_store_roundtrip(trajectory_store: TrajectoryStore):
    traj = _build_trajectory("traj_roundtrip", goal="organize downloads", step_count=3)
    trajectory_store.save_trajectory(traj)

    loaded = trajectory_store.load_trajectory("traj_roundtrip")
    assert loaded is not None
    assert loaded.trajectory_id == "traj_roundtrip"
    assert loaded.step_count == 3
    assert loaded.steps[0].action.action_type == ActionType.UI_CLICK
    assert loaded.steps[0].action.params["x"] == 100
    assert loaded.start_time == pytest.approx(traj.start_time)
    assert loaded.end_time == pytest.approx(traj.end_time)


def test_episode_search_by_goal(trajectory_store: TrajectoryStore):
    for goal in ["organize downloads", "edit photos", "organize documents"]:
        ep = Episode(
            trajectory_id="traj_search",
            inferred_goal=goal,
            start_idx=0,
            end_idx=1,
        )
        trajectory_store.save_episode(ep)

    results = trajectory_store.search_episodes_by_goal(["organize"])
    assert len(results) == 2
    goals = {ep.inferred_goal for ep in results}
    assert "organize downloads" in goals
    assert "organize documents" in goals


def test_observer_working_memory_notification(
    skill_library: SkillLibraryStore,
    working_memory: WorkingMemoryProvider,
):
    skill_library.save_skill(_make_stored_skill())
    scorer = HeuristicSimilarityScorer()
    observer = ActiveLearningObserver(
        skill_library,
        scorer,
        working_memory,
        heuristic_low=0.1,
        heuristic_high=0.99,
        final_low=0.1,
        final_high=0.99,
    )

    candidate = make_candidate(
        "Organize downloads v2",
        steps=["List directory", "Classify by type", "Rename", "Move files"],
        triggers=["organize", "sort"],
    )
    episode = make_episode()
    observer.on_candidates_ready([candidate], [episode])

    msgs = working_memory.as_chat_messages()
    assert any("skill_suggestion" in str(m) for m in msgs)
    assert skill_library.count_pending() == 1

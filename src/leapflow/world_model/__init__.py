"""World Model — train-free curiosity-driven predictive learning.

Provides the Predict → Execute → Compare → Learn loop,
off-policy experience replay, and OPD trajectory grading,
all without gradient updates.
"""

from leapflow.world_model.budget import LearningBudgetController
from leapflow.world_model.curiosity import CuriosityConfig, CuriosityScore, CuriositySignal
from leapflow.world_model.experience_store import ExperienceStore, ExperienceTuple
from leapflow.world_model.prediction import Prediction, PredictionLoop, PredictionOutcome
from leapflow.world_model.replay import ExperienceReplayEngine, ReplayInsight
from leapflow.world_model.trajectory_grader import ActionGrade, TrajectoryGrader

__all__ = [
    "LearningBudgetController",
    "CuriosityConfig",
    "CuriosityScore",
    "CuriositySignal",
    "ExperienceStore",
    "ExperienceTuple",
    "Prediction",
    "PredictionLoop",
    "PredictionOutcome",
    "ExperienceReplayEngine",
    "ReplayInsight",
    "ActionGrade",
    "TrajectoryGrader",
]

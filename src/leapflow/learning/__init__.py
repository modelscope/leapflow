"""Skill learning layer — distillation, code generation, feedback, and active learning."""

from leapflow.learning.cold_start import ColdStartConfig, ColdStartManager, ColdStartPhase
from leapflow.learning.distiller import DistillationCandidate, LLMSkillDistiller, SkillDistiller
from leapflow.learning.effectiveness import LearningEffectivenessTracker, LearningMetrics
from leapflow.learning.event_consumer import EventConsumer
from leapflow.learning.pattern_miner import PatternMiner, SkillCandidate

__all__ = [
    "ActiveLearningObserver",
    "ColdStartConfig",
    "ColdStartManager",
    "ColdStartPhase",
    "DistillationCandidate",
    "EventConsumer",
    "LearningEffectivenessTracker",
    "LearningMetrics",
    "LLMSkillDistiller",
    "PatternMiner",
    "SkillCandidate",
    "SkillDistiller",
]


def __getattr__(name: str):
    if name == "ActiveLearningObserver":
        from leapflow.learning.active_learning import ActiveLearningObserver
        return ActiveLearningObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

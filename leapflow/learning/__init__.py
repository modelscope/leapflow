"""Skill learning layer — distillation, code generation, feedback, and active learning."""

from leapflow.learning.distiller import DistillationCandidate, LLMSkillDistiller, SkillDistiller

__all__ = [
    "ActiveLearningObserver",
    "DistillationCandidate",
    "LLMSkillDistiller",
    "SkillDistiller",
]


def __getattr__(name: str):
    if name == "ActiveLearningObserver":
        from leapflow.learning.active_learning import ActiveLearningObserver
        return ActiveLearningObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

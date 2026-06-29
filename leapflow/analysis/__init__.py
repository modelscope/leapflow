"""Offline analysis layer — synthesis, abstraction, segmentation, and distillation pipeline."""

from leapflow.analysis.abstractor import ActionAbstractor
from leapflow.analysis.synthesis import PlatformSynthesisPass

__all__ = [
    "ActionAbstractor",
    "ImitationPipeline",
    "PlatformSynthesisPass",
]


def __getattr__(name: str):
    if name == "ImitationPipeline":
        from leapflow.analysis.pipeline import ImitationPipeline
        return ImitationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

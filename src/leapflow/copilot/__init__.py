"""Workflow Copilot — context-triggered workflow auto-completion engine."""

from leapflow.copilot.config import CopilotConfig
from leapflow.copilot.context import ContextEncoder, CopilotEventSubscriber
from leapflow.copilot.degradation import DegradationLevel, DegradationPolicy
from leapflow.copilot.engine import PredictionEngine
from leapflow.copilot.feedback import EvolutionLoop, FeedbackCollector
from leapflow.copilot.idle import IdleDetector
from leapflow.copilot.pipeline import SpeculativePipeline
from leapflow.copilot.predictors import (
    L0HashPredictor,
    L1MarkovPredictor,
    L2EmbeddingPredictor,
    L3LLMPredictor,
)
from leapflow.copilot.renderer import DisplayGate, LogHintRenderer, SuggestionRenderer
from leapflow.copilot.types import (
    ContextState,
    FeedbackSignal,
    FeedbackType,
    HintRenderer,
    PredictionCandidate,
    PredictorLayer,
    Signal,
    SignalChannel,
)

__all__ = [
    # types & protocols
    "ContextState",
    "FeedbackSignal",
    "FeedbackType",
    "HintRenderer",
    "PredictionCandidate",
    "PredictorLayer",
    "Signal",
    "SignalChannel",
    # config
    "CopilotConfig",
    # context
    "ContextEncoder",
    "CopilotEventSubscriber",
    # engine & pipeline
    "PredictionEngine",
    "SpeculativePipeline",
    # idle
    "IdleDetector",
    # renderer
    "DisplayGate",
    "LogHintRenderer",
    "SuggestionRenderer",
    # feedback
    "EvolutionLoop",
    "FeedbackCollector",
    # degradation
    "DegradationLevel",
    "DegradationPolicy",
    # predictors
    "L0HashPredictor",
    "L1MarkovPredictor",
    "L2EmbeddingPredictor",
    "L3LLMPredictor",
]

"""MHMS-SF: Multi-source Heterogeneous Multi-scale Signal Fusion.

This module is the orchestration layer for fusing visual, event, and
contextual signals across multiple temporal scales (action, segment,
episode). It reuses and extends existing perception and analysis
components rather than reimplementing them.

Public API:
    MHMSFusionPipeline  — main entry point for running fusion
    FusionContext        — input container with all signal sources
    FusionResult         — unified output from the pipeline
    FusionQuality        — quality assessment of fusion results

Types:
    AtomicAction         — fused single-action output
    Segment              — sub-task grouping with wait annotations
    EnrichedEpisode      — episode with workflow graph context
    WorkflowGraph        — DAG of cross-app workflow
    AppTransitionEvent   — first-class app transition event

Supporting:
    CrossAppContextTracker — stateful cross-app context tracking
    WaitPeriodClassifier   — gap classification (AI gen, idle, etc.)
"""

from leapflow.signal_fusion.cross_app import CrossAppContextTracker
from leapflow.signal_fusion.pipeline import MHMSFusionPipeline
from leapflow.signal_fusion.protocol import FusionContext, FusionResult
from leapflow.signal_fusion.quality import FusionQuality
from leapflow.signal_fusion.types import (
    AppTransitionEvent,
    AtomicAction,
    EnrichedEpisode,
    Segment,
    WorkflowGraph,
)
from leapflow.signal_fusion.wait_classifier import WaitPeriodClassifier

__all__ = [
    "MHMSFusionPipeline",
    "FusionContext",
    "FusionResult",
    "FusionQuality",
    "AtomicAction",
    "Segment",
    "EnrichedEpisode",
    "WorkflowGraph",
    "AppTransitionEvent",
    "CrossAppContextTracker",
    "WaitPeriodClassifier",
]

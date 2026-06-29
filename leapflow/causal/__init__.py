"""Causal Propagation Chain — unified causal model for LEAP Agent.

Core data model: CausalEvent → CausalChain → CausalGraph.
Channel abstraction: ChannelSpec + ChannelRegistry.
Inference: 3-tier engine (rules → heuristics → VLM verification).
"""

from leapflow.causal.types import (
    CausalChain,
    CausalEvent,
    CausalGraph,
    EventSource,
    EventType,
    FrameRef,
)
from leapflow.causal.channel import ChannelRegistry, ChannelSpec
from leapflow.causal.pipeline import CausalFusionPipeline, FusionOutput
from leapflow.causal.adapter import graph_to_pair_context, graph_to_semantic_actions
from leapflow.causal.channel import build_default_registry

__all__ = [
    "CausalChain",
    "CausalEvent",
    "CausalGraph",
    "CausalFusionPipeline",
    "ChannelRegistry",
    "ChannelSpec",
    "EventSource",
    "EventType",
    "FrameRef",
    "FusionOutput",
    "build_default_registry",
    "graph_to_pair_context",
    "graph_to_semantic_actions",
]

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declarative action/observation adaptation between a policy and an embodiment.

A served policy declares what its vectors mean (``spec``); the workcell
declares which channels and cameras embody them. ``pipeline`` builds, from
those two declarations alone, the reversible chain of ``steps`` that turns a
model chunk into channel targets -- and channel values back into the model's
space for the observation, for real-time chunking and for training data.
``aggregate`` holds the tick-level stages that follow, before the shield;
``encode`` rebuilds a recorded binding offline to encode training data.
"""

from leapflow.hardware.skills.adapt.aggregate import Action, ChunkAggregator, LowPass
from leapflow.hardware.skills.adapt.encode import EpisodeEncoder, RecordedFrame
from leapflow.hardware.skills.adapt.pipeline import (
    ACTION_STEP_GROUP,
    ActionPipeline,
    BoundGroup,
    build_action_pipeline,
    build_plugins,
    build_state_pipeline,
    discover_action_steps,
)
from leapflow.hardware.skills.adapt.spec import ModelSpaces, PolicyDecl
from leapflow.hardware.skills.adapt.steps import (
    ActionStep,
    AdaptError,
    AdaptViolation,
    StepContext,
    StepFactory,
)

__all__ = [
    "ACTION_STEP_GROUP",
    "Action",
    "ActionPipeline",
    "ActionStep",
    "AdaptError",
    "AdaptViolation",
    "BoundGroup",
    "ChunkAggregator",
    "EpisodeEncoder",
    "LowPass",
    "ModelSpaces",
    "PolicyDecl",
    "RecordedFrame",
    "StepContext",
    "StepFactory",
    "build_action_pipeline",
    "build_plugins",
    "build_state_pipeline",
    "discover_action_steps",
]

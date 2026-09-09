"""Shared domain model — zero-dependency data types used across all layers."""

from leapflow.domain.capability_requirement import (
    ApprovalMode,
    CapabilityRequirement,
    RequirementOrigin,
)
from leapflow.domain.effect_scope import EffectScope, ScopeState
from leapflow.domain.event_types import (
    CLIEventType,
    ImplicitFeedbackType,
    LearningEventType,
    NormalizedEventType,
    UIActionSubType,
    UNDO_SHORTCUTS,
)
from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
from leapflow.domain.events import SystemEvent, UIElement, UISnapshot
from leapflow.domain.evolution_intent import (
    WORLD_MODEL_INTENT,
    WORLD_MODEL_ORIGIN,
    EvolutionIntent,
)
from leapflow.domain.platform import (
    Capability,
    DEFAULT_DARWIN_CAPABILITIES,
    PlatformID,
    PlatformManifest,
    capability_from_str,
)
from leapflow.domain.plugin_fiber import FiberState, IllegalStateTransition, PluginFiber
from leapflow.domain.plugin_proposal import BehaviorTestCase, GapEvidence, PluginProposal, ProposedToolSpec
from leapflow.domain.skill_types import DistillationCandidate, SkillMetadata, SkillParameter
from leapflow.domain.trajectory import (
    ActionType,
    Episode,
    NoiseSignal,
    RawAction,
    RecordingState,
    SemanticAction,
    SnapshotLevel,
    StateSnapshot,
    Trajectory,
    TrajectoryStep,
    action_type_from_event,
)

__all__ = [
    "ActionType",
    "ApprovalMode",
    "BehaviorTestCase",
    "CLIEventType",
    "CapabilityRequirement",
    "EffectScope",
    "EvolutionIntent",
    "FiberState",
    "GapEvidence",
    "ImplicitFeedbackType",
    "LearningEventType",
    "NormalizedEventType",
    "UIActionSubType",
    "UNDO_SHORTCUTS",
    "Capability",
    "DEFAULT_DARWIN_CAPABILITIES",
    "DistillationCandidate",
    "EnvironmentFingerprint",
    "Episode",
    "IllegalStateTransition",
    "NoiseSignal",
    "PlatformID",
    "PlatformManifest",
    "PluginFiber",
    "PluginProposal",
    "ProposedToolSpec",
    "RawAction",
    "RecordingState",
    "RequirementOrigin",
    "ScopeState",
    "SemanticAction",
    "SkillMetadata",
    "SkillParameter",
    "SnapshotLevel",
    "StateSnapshot",
    "SystemEvent",
    "Trajectory",
    "TrajectoryStep",
    "UIElement",
    "UISnapshot",
    "WORLD_MODEL_INTENT",
    "WORLD_MODEL_ORIGIN",
    "action_type_from_event",
    "capability_from_str",
]

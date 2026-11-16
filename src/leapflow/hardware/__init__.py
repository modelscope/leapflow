# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hardware Context Protocol -- safe agent operation of physical devices.

The protocol splits device integration along the axis of what can be known:

``context`` (stable)
    What an agent must know to operate a device safely -- quantities, units,
    operating envelopes, interlocks, settling time, reversibility. Determined by
    physics and by governance requirements, so it is defined here and frozen.

``transport`` (volatile)
    How a command reaches the device. Determined by whichever southbound standard
    or vendor SDK is in play, so it lives behind a six-method Protocol and can be
    swapped without touching anything else.

Two seams follow from that split. ``providers/`` answers where device knowledge
comes from; ``transports/`` answers how an operation executes. Both are
``kind -> factory`` lookup tables, so supporting a new standard is a module plus a
row -- and no upstream concept is permitted to leak into ``context.py``.

Nothing here is enabled by default: without a bound registry the plugin exposes no
tools, and the rest of the system behaves exactly as it did before.
"""

from __future__ import annotations

from leapflow.hardware.context import (
    HC_VERSION,
    SUPPORTED_HC_VERSIONS,
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    PrivacyTier,
    Quality,
    Representation,
    TransportRef,
)
from leapflow.hardware.control_bus import (
    ControlBusConfig,
    ControlBusStats,
    ControlCommand,
    ControlPolicy,
    ControlState,
    HighFrequencyControlBus,
)
from leapflow.hardware.degradation import (
    DegradationAction,
    DegradationCoordinator,
    DegradationOutcome,
)
from leapflow.hardware.evidence import (
    EvidenceStore,
)
from leapflow.hardware.fleet import (
    FleetError,
    FleetManager,
    FleetNode,
    FleetSessionRouter,
    FleetTopology,
)
from leapflow.hardware.fleet_tools import build_fleet_tools
from leapflow.hardware.health_monitor import (
    DeviceHealthRecord,
    HardwareHealthMonitor,
)
from leapflow.hardware.lhp_gateway import (
    DeviceSnapshot,
    HardwareContextDelta,
    HardwareContextSnapshot,
    LHPGateway,
    PCDLevel,
)
from leapflow.hardware.realtime import (
    ControlTelemetryEvent,
    ImpedanceController,
    PIDJointController,
    RealtimeControlLoop,
    TrajectoryTracker,
)
from leapflow.hardware.registry import (
    AdmissionNote,
    HardwareRegistry,
    HardwareSettings,
    LoadReport,
    UnverifiedContextPolicy,
)
from leapflow.hardware.stream import (
    EventKind,
    HardwareDeliberativeObserver,
    HardwareEvent,
    HardwareEventClass,
    classify_hardware_event,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_COMMITTED,
    SIDE_EFFECT_NONE,
    SIDE_EFFECT_PARTIAL,
    SIDE_EFFECT_UNKNOWN,
    FrameReading,
    FrameTransport,
    HardwareTransport,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)
from leapflow.hardware.verification import (
    EvidenceBundle,
    GraspVerifier,
    OperationVerdict,
    OperationVerifier,
    PositionVerifier,
    VerdictStatus,
    VerificationMode,
    collect_evidence,
)
from leapflow.hardware.write_governor import (
    BatchFeasibilityResult,
    ControlFeasibilityResult,
    GovernanceDecision,
    WriteFeasibilityResult,
    WriteGovernanceRequest,
    WriteGovernor,
)
from leapflow.hardware.write_verification import (
    WriteExpectation,
    WriteVerdict,
    WriteVerdictEvent,
    WriteVerdictStatus,
    WriteVerificationCoordinator,
    aggregate_write_verdicts,
)

__all__ = [
    "HC_VERSION",
    "SIDE_EFFECT_COMMITTED",
    "SIDE_EFFECT_NONE",
    "SIDE_EFFECT_PARTIAL",
    "SIDE_EFFECT_UNKNOWN",
    "SUPPORTED_HC_VERSIONS",
    "AdmissionNote",
    "Channel",
    "ContextProvenance",
    "ContextSource",
    "DegradationAction",
    "DegradationCoordinator",
    "DegradationOutcome",
    # control_bus.py
    "ControlBusConfig",
    "ControlBusStats",
    "ControlCommand",
    "ControlPolicy",
    "ControlState",
    "Direction",
    "Envelope",
    "EvidenceBundle",
    "EvidenceStore",
    "FrameReading",
    "FrameTransport",
    "GraspVerifier",
    "HardwareContext",
    "HardwareEffect",
    "HardwareRegistry",
    "HardwareSettings",
    "HardwareTransport",
    "HighFrequencyControlBus",
    "Interlock",
    "LoadReport",
    "OperationVerdict",
    "OperationVerifier",
    "PositionVerifier",
    "PrivacyTier",
    "Quality",
    "Reading",
    "Representation",
    "TransportError",
    "TransportRef",
    "TransportStatus",
    "UnverifiedContextPolicy",
    "VerdictStatus",
    "VerificationMode",
    "WriteExpectation",
    "WriteOutcome",
    "WriteVerdict",
    "WriteVerdictEvent",
    "WriteVerdictStatus",
    "WriteVerificationCoordinator",
    "WriteGovernanceRequest",
    "WriteGovernor",
    "WriteFeasibilityResult",
    "BatchFeasibilityResult",
    "ControlFeasibilityResult",
    "GovernanceDecision",
    "aggregate_write_verdicts",
    "collect_evidence",
    # realtime.py
    "ControlTelemetryEvent",
    "ImpedanceController",
    "PIDJointController",
    "RealtimeControlLoop",
    "TrajectoryTracker",
    # lhp_gateway.py
    "DeviceSnapshot",
    "HardwareContextDelta",
    "HardwareContextSnapshot",
    "LHPGateway",
    "PCDLevel",
    # fleet.py
    "FleetError",
    "FleetManager",
    "FleetNode",
    "FleetSessionRouter",
    "FleetTopology",
    "build_fleet_tools",
    # stream.py -- event classification
    "EventKind",
    "HardwareDeliberativeObserver",
    "HardwareEvent",
    "HardwareEventClass",
    "classify_hardware_event",
    # health_monitor.py
    "DeviceHealthRecord",
    "HardwareHealthMonitor",
]

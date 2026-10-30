# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared LeapRobot benchmark harness utilities.

Constructs test fixtures (HardwareRegistry, SimulatedTransport, Trust/Approval/
Degradation chains) entirely from LeapFlow public APIs, without importing the
private test harness.  Every helper is synchronous and allocation-only — no I/O
at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmarks.evidence import evidence_root
from leapflow.hardware.context import (
    CapabilityDeclaration,
    Channel,
    DegradationPolicy,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    KinematicsDeclaration,
    SafetyPolicy,
    TransportRef,
    TrustConfig,
)
from leapflow.hardware.registry import (
    HardwareRegistry,
    HardwareSettings,
    UnverifiedContextPolicy,
)
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.transports.mock import MockTransport
from leapflow.hardware.transports.simulated import SimulatedTransport
from leapflow.hardware.trust import HardwareTrustGate


# ════════════════════════════════════════════════════════════════
# Lightweight approval gate (no dependency on tests._harness)
# ════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ApprovalOutcome:
    """Minimal public-shape approval result consumed by HardwareTools."""

    approved: bool
    decision: str
    denial_message: str = ""


class ScriptedApprovalGate:
    """A deterministic approval gate that returns pre-scripted decisions."""

    ALLOW = "allow_once"
    DENY = "deny"

    def __init__(
        self,
        decisions: Sequence[str] = (),
        *,
        default: str = "deny",
    ) -> None:
        self._queue: list[str] = list(decisions)
        self._default = default
        self.call_count = 0
        self.last_detail: Any = None

    async def evaluate(self, detail: Any) -> ApprovalOutcome:
        """Return the next scripted decision."""
        self.call_count += 1
        self.last_detail = detail
        decision = self._queue.pop(0) if self._queue else self._default
        approved = decision not in ("deny", "deny_always", "cancel_workflow")
        message = "" if approved else "The operator denied this hardware command."
        return ApprovalOutcome(approved, decision, message)


# ════════════════════════════════════════════════════════════════
# Context / registry factory helpers
# ════════════════════════════════════════════════════════════════


def make_channel(
    channel_id: str,
    *,
    quantity: str = "angular_velocity",
    unit: str = "rad/s",
    writable: bool = True,
    reversible: bool = True,
    streaming: bool = False,
    envelope: Envelope | None = None,
) -> Channel:
    """Build a minimal Channel for benchmark use."""
    selected = envelope or Envelope(
        declared=True,
        min_value=-100.0,
        max_value=100.0,
        reversible=reversible,
    )
    return Channel(
        channel_id=channel_id,
        direction=Direction.READWRITE.value if writable else Direction.READ.value,
        quantity=quantity,
        unit=unit,
        effect=HardwareEffect.ACTUATE.value if writable else HardwareEffect.READ.value,
        envelope=selected,
        sample_rate_hz=10.0 if streaming else 0.0,
    )


def make_context(
    device_id: str = "bench_arm",
    *,
    channels: tuple[Channel, ...] | None = None,
    safety: SafetyPolicy | None = None,
    trust_config: TrustConfig | None = None,
    degradation: DegradationPolicy | None = None,
    capabilities: CapabilityDeclaration | None = None,
    kinematics: KinematicsDeclaration | None = None,
    interlocks: tuple[Interlock, ...] = (),
    transport_kind: str = "mock",
    transport_config: Mapping[str, Any] | None = None,
    halt_supported: bool = True,
    control_bindings: Mapping[str, Any] | None = None,
) -> HardwareContext:
    """Build a HardwareContext with hc.v1 extensions for benchmarking."""
    if channels is None:
        channels = (
            make_channel("joint_0"),
            make_channel("joint_1"),
        )
    return HardwareContext(
        device_id=device_id,
        hc_version="hc.v1",
        display_name=device_id,
        transport=TransportRef(
            kind=transport_kind,
            config=dict(transport_config or {}),
        ),
        channels=channels,
        interlocks=interlocks,
        halt_supported=halt_supported,
        safety=safety,
        trust_config=trust_config,
        degradation=degradation,
        capabilities=capabilities,
        kinematics=kinematics,
        control_bindings=control_bindings,
    )


def make_mock_transport(
    values: Mapping[str, Any] | None = None,
    *,
    halt_supported: bool = True,
    failures: Sequence[Mapping[str, Any]] = (),
) -> MockTransport:
    """Build a MockTransport from benchmark parameters."""
    config: dict[str, Any] = {
        "halt_supported": halt_supported,
    }
    if values:
        config["values"] = dict(values)
    if failures:
        config["failures"] = list(failures)
    return MockTransport(config)


def make_simulated_transport(
    values: Mapping[str, Any] | None = None,
    *,
    halt_supported: bool = True,
    seed: int = 42,
    failures: Sequence[Mapping[str, Any]] = (),
    disconnects: Sequence[Mapping[str, Any]] = (),
    waveforms: Mapping[str, Any] | None = None,
) -> SimulatedTransport:
    """Build a SimulatedTransport from benchmark parameters."""
    config: dict[str, Any] = {
        "halt_supported": halt_supported,
        "seed": seed,
    }
    if values:
        config["values"] = dict(values)
    if failures:
        config["failures"] = list(failures)
    if disconnects:
        config["disconnects"] = list(disconnects)
    if waveforms:
        config["waveforms"] = dict(waveforms)
    return SimulatedTransport(config)


def make_trust_gate(
    *,
    candidate_at: int = 3,
    verified_at: int = 8,
    production_at: int = 20,
    demote_after: int = 2,
) -> HardwareTrustGate:
    """Build a HardwareTrustGate with configurable thresholds."""
    return HardwareTrustGate(
        candidate_at=candidate_at,
        verified_at=verified_at,
        production_at=production_at,
        demote_after=demote_after,
    )


class StaticContextProvider:
    """In-memory provider for one or more benchmark declarations."""

    kind = "native_benchmark"

    def __init__(self, contexts: Sequence[HardwareContext]) -> None:
        self._contexts = tuple(contexts)

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return the configured immutable contexts."""
        return self._contexts


def make_registry(*contexts: HardwareContext) -> HardwareRegistry:
    """Build and load a registry without persistence or background streams."""
    settings = HardwareSettings(
        enabled=True,
        unverified_context_policy=UnverifiedContextPolicy.ALLOW,
        require_describe_before_write=False,
        trust_skip_enabled=True,
        stream_enabled=False,
        persist_readings=False,
    )
    registry = HardwareRegistry(settings, providers=(StaticContextProvider(contexts),))
    registry.load()
    return registry


def make_tools(
    registry: HardwareRegistry,
    *,
    gate: ScriptedApprovalGate | None = None,
    trust_gate: HardwareTrustGate | None = None,
) -> HardwareTools:
    """Build HardwareTools against a benchmark registry."""
    return HardwareTools(
        registry,
        gate=gate,
        session_id="native-benchmark",
        hardware_trust_gate=trust_gate,
    )


@dataclass
class HarnessBundle:
    """Complete public-API fixture ready for benchmark trials."""

    context: HardwareContext
    registry: HardwareRegistry
    tools: HardwareTools
    gate: ScriptedApprovalGate | None = None
    trust_gate: HardwareTrustGate | None = None

    def transport(self) -> MockTransport | SimulatedTransport:
        """Return the opened benchmark transport."""
        opened = self.registry.get_open_transport(self.context.device_id)
        if not isinstance(opened, (MockTransport, SimulatedTransport)):
            raise TypeError(f"unexpected benchmark transport: {type(opened).__name__}")
        return opened

    async def close(self) -> None:
        """Close all transports owned by this fixture."""
        await self.registry.close_all()


def make_bundle(
    context: HardwareContext,
    *,
    decisions: Sequence[str] = ("allow_once",),
    default_decision: str = "deny",
    trust_gate: HardwareTrustGate | None = None,
) -> HarnessBundle:
    """Assemble context, registry, approval gate, trust gate, and tools."""
    registry = make_registry(context)
    gate = ScriptedApprovalGate(decisions, default=default_decision)
    if trust_gate is not None:
        trust_gate.register_device(context.device_id, context.trust_config)
    tools = make_tools(registry, gate=gate, trust_gate=trust_gate)
    return HarnessBundle(context, registry, tools, gate, trust_gate)


__all__ = [
    "ApprovalOutcome",
    "HarnessBundle",
    "ScriptedApprovalGate",
    "StaticContextProvider",
    "evidence_root",
    "make_bundle",
    "make_channel",
    "make_context",
    "make_mock_transport",
    "make_registry",
    "make_simulated_transport",
    "make_tools",
    "make_trust_gate",
]

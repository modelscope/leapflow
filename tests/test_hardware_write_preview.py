"""Dry-run preview for hardware writes (Phase 1.5).

A preview must run the full feasibility chain -- envelope, rate, reachability,
interlocks -- and build the approval descriptor, yet never reach
``transport.write``. The verdict it reports is therefore ``SIDE_EFFECT_NONE``:
nothing was commanded, which is exactly what makes a preview safe to issue
against an irreversible channel.

These cases drive the real ``HardwareRegistry`` and the production ``MockTransport``
so the "was the device touched?" assertion is genuine rather than mocked. The
approval gate is deliberately absent for the dry-run cases, pinning the invariant
that a preview returns before consent is sought and works with no gate installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    Interlock,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.transport import SIDE_EFFECT_NONE

SESSION = "session-preview"


class _StaticProvider:
    """Hands a fixed set of declarations to the registry, no discovery I/O."""

    kind = "static"

    def __init__(self, *contexts: HardwareContext) -> None:
        self._contexts = contexts

    def discover(self) -> tuple[HardwareContext, ...]:
        return self._contexts


class _AllowGate:
    """A gate that always approves, used only to prove real writes still run."""

    async def evaluate(self, descriptor: Any) -> Any:
        return _Approved()


class _Approved:
    approved = True
    denial_message = ""


def _rig_context(*, guard: bool = True) -> HardwareContext:
    """A bench with a plain actuator and an interlocked, irreversible pump.

    ``guard`` sets the interlock source so a test can present either a satisfied
    or an open interlock without restating the declaration.
    """
    return HardwareContext(
        device_id="rig",
        hc_version=HC_VERSION,
        display_name="Preview rig",
        location="bench-1",
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={
                "values": {"guard": guard, "motor": 0.0, "pump": 0.0, "config": 0.0},
                "halt_supported": True,
            },
        ),
        interlocks=(
            Interlock(
                interlock_id="guard_closed",
                channel_id="guard",
                operator="eq",
                value=True,
                description="The guard must be closed before dispensing.",
            ),
        ),
        channels=(
            Channel(
                channel_id="guard",
                direction=Direction.READ.value,
                quantity="state.guard",
                unit="bool",
                envelope=Envelope(declared=True),
            ),
            Channel(
                channel_id="motor",
                direction=Direction.READWRITE.value,
                quantity="ratio.motor",
                unit="percent",
                effect=HardwareEffect.ACTUATE.value,
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=100.0, reversible=True
                ),
            ),
            Channel(
                channel_id="pump",
                direction=Direction.WRITE.value,
                quantity="volume.pump",
                unit="uL_per_s",
                effect=HardwareEffect.DISPENSE.value,
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=50.0,
                    reversible=False,
                    requires_interlocks=("guard_closed",),
                ),
            ),
            Channel(
                channel_id="config",
                direction=Direction.READWRITE.value,
                quantity="setting.mode",
                unit="level",
                effect=HardwareEffect.CONFIGURE.value,
                envelope=Envelope(
                    declared=True, min_value=0.0, max_value=10.0, reversible=True
                ),
            ),
        ),
        provenance=ContextProvenance(verified_by="james"),
    )


def _tools(context: HardwareContext, *, gate: Any = None) -> tuple[HardwareTools, HardwareRegistry]:
    registry = HardwareRegistry(
        HardwareSettings(enabled=True, require_describe_before_write=False),
        providers=[_StaticProvider(context)],
    )
    registry.load()
    return HardwareTools(registry, gate=gate, session_id=SESSION), registry


# ════════════════════════════════════════════════════════════════
# Preview does not touch the device
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_dry_run_actuate_does_not_write_and_reports_none() -> None:
    """A valid dry run reports a passing preview without commanding the device."""
    tools, registry = _tools(_rig_context())

    result = await tools.hw_actuate(
        device_id="rig", channel_id="motor", value=50.0, dry_run=True
    )

    assert result["ok"] is True
    assert result["preview"] is True
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    # The plan describes what *would* be commanded, so intent can be confirmed.
    assert result["plan"]["value"] == 50.0
    assert result["plan"]["value_in_envelope"] is True
    assert result["plan"]["interlocks_satisfied"] is True

    transport = await registry.transport("rig")
    assert transport.write_log == ()
    assert transport.write_attempts("motor") == 0


@pytest.mark.asyncio
async def test_dry_run_configure_reports_preview_without_writing() -> None:
    """A configure dry run previews the setting change without touching the device.

    The third write class (alongside actuate and dispense) must honour the same
    preview contract, so an irreversible reconfiguration can be confirmed before
    it is committed.
    """
    tools, registry = _tools(_rig_context())

    result = await tools.hw_configure(
        device_id="rig", channel_id="config", value=5.0, dry_run=True
    )

    assert result["ok"] is True
    assert result["preview"] is True
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    # The plan names the command that would be issued, so intent can be confirmed.
    assert result["plan"]["summary"]
    assert result["plan"]["value"] == 5.0
    assert result["plan"]["effect"] == HardwareEffect.CONFIGURE.value

    transport = await registry.transport("rig")
    assert transport.write_log == ()
    assert transport.write_attempts("config") == 0


@pytest.mark.asyncio
async def test_dry_run_out_of_envelope_fails_validation_without_writing() -> None:
    """A value outside the envelope yields ok=False, still touching nothing."""
    tools, registry = _tools(_rig_context())

    result = await tools.hw_actuate(
        device_id="rig", channel_id="motor", value=500.0, dry_run=True
    )

    assert result["ok"] is False
    assert result["preview"] is True
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    assert result["failure_code"] == "value_out_of_envelope"
    assert result["plan"]["value_in_envelope"] is False

    transport = await registry.transport("rig")
    assert transport.write_log == ()
    assert transport.write_attempts("motor") == 0


@pytest.mark.asyncio
async def test_dry_run_failed_interlock_fails_validation_without_writing() -> None:
    """An open interlock makes the preview fail, and still nothing is dispensed."""
    tools, registry = _tools(_rig_context(guard=False))

    result = await tools.hw_dispense(
        device_id="rig", channel_id="pump", value=10.0, dry_run=True
    )

    assert result["ok"] is False
    assert result["preview"] is True
    assert result["side_effect_state"] == SIDE_EFFECT_NONE
    assert result["failure_code"] == "interlocks_unsatisfied"
    assert "guard_closed" in result["plan"]["interlocks_failed"]

    transport = await registry.transport("rig")
    assert transport.write_log == ()
    assert transport.write_attempts("pump") == 0


@pytest.mark.asyncio
async def test_dry_run_needs_no_gate() -> None:
    """A preview returns before consent, so an absent gate must not block it."""
    tools, _ = _tools(_rig_context(), gate=None)

    result = await tools.hw_actuate(
        device_id="rig", channel_id="motor", value=25.0, dry_run=True
    )

    assert result["ok"] is True
    assert result["preview"] is True


# ════════════════════════════════════════════════════════════════
# Default behaviour is unchanged
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_default_write_still_reaches_transport() -> None:
    """Without dry_run, an approved command reaches the device as before."""
    tools, registry = _tools(_rig_context(), gate=_AllowGate())

    result = await tools.hw_actuate(device_id="rig", channel_id="motor", value=40.0)

    assert result["ok"] is True
    # A real write carries a committed effect and is not marked as a preview.
    assert result["side_effect_state"] != SIDE_EFFECT_NONE
    assert "preview" not in result

    transport = await registry.transport("rig")
    assert transport.write_log == (("motor", 40.0),)

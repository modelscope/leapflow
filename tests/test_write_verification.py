# Copyright (c) Alibaba, Inc. and its affiliates.
"""Canonical post-write verification across write and observation channels."""

from __future__ import annotations

from pathlib import Path

import pytest

from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.context import (
    HC_VERSION,
    Channel,
    ContextProvenance,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.tools import HardwareTools
from leapflow.hardware.transport import Reading
from leapflow.hardware.trust import HardwareTrustGate, HardwareTrustLevel
from leapflow.hardware.verification import OperationVerdict, VerdictStatus
from leapflow.hardware.write_verification import (
    WriteExpectation,
    WriteVerdict,
    WriteVerdictStatus,
    aggregate_write_verdicts,
)
from leapflow.security.approval import ApprovalDecision, SessionAwareGate
from leapflow.security.orchestrator import ApprovalOrchestrator
from leapflow.security.policy import ApprovalPolicyEngine

from tests._harness.hardware_stubs import ScriptedHuman


class _StaticProvider:
    kind = "static"

    def __init__(self, context: HardwareContext) -> None:
        self._context = context

    def discover(self) -> tuple[HardwareContext, ...]:
        return (self._context,)


def _cross_channel_context(*, actual: float = 5.0, settling: float = 0.0) -> HardwareContext:
    return HardwareContext(
        device_id="heater",
        hc_version=HC_VERSION,
        display_name="Test heater",
        halt_supported=True,
        transport=TransportRef(
            kind="mock",
            config={"values": {"power": 0.0, "temperature": actual}},
        ),
        channels=(
            Channel(
                channel_id="power",
                direction=Direction.READWRITE.value,
                quantity="power",
                unit="W",
                effect=HardwareEffect.ACTUATE.value,
                verify_after_write=True,
                verification_channel_id="temperature",
                envelope=Envelope(
                    declared=True,
                    min_value=0.0,
                    max_value=10.0,
                    tolerance=0.1,
                    settling_time_s=settling,
                    reversible=True,
                ),
            ),
            Channel(
                channel_id="temperature",
                direction=Direction.READ.value,
                quantity="temperature",
                unit="C",
                effect=HardwareEffect.READ.value,
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
        ),
        provenance=ContextProvenance(verified_by="test"),
    )


def _tools(
    context: HardwareContext,
    *,
    audit_path: Path | None = None,
) -> tuple[HardwareTools, HardwareTrustGate, HardwareRegistry]:
    registry = HardwareRegistry(
        HardwareSettings(
            enabled=True,
            require_describe_before_write=False,
            persist_readings=False,
        ),
        providers=[_StaticProvider(context)],
    )
    registry.load()
    gate = ApprovalOrchestrator(
        SessionAwareGate(ScriptedHuman(ApprovalDecision.ALLOW_ALL_SESSION)),
        policy=ApprovalPolicyEngine(),
    )
    trust = HardwareTrustGate(candidate_at=1)
    tools = HardwareTools(
        registry,
        gate=gate,
        session_id="verification-test",
        audit_log=HardwareAuditLog(audit_path),
        hardware_trust_gate=trust,
    )
    return tools, trust, registry


def test_write_expectation_round_trips_cross_channel_declaration() -> None:
    context = _cross_channel_context(settling=0.25)
    channel = context.channel("power")
    assert channel is not None

    expectation = WriteExpectation.from_channel(
        device_id=context.device_id,
        channel=channel,
        commanded_value=5.0,
        now=10.0,
    )

    assert expectation.command_channel_id == "power"
    assert expectation.observation_channel_id == "temperature"
    assert expectation.settle_after == pytest.approx(10.25)
    assert expectation.to_dict()["settle_delay_s"] == pytest.approx(0.25)
    restored = Channel.from_mapping(channel.to_dict())
    assert restored.verification_channel_id == "temperature"


def test_typed_verdict_adapts_readings_and_semantic_verdicts() -> None:
    channel = _cross_channel_context().channel("power")
    assert channel is not None
    expectation = WriteExpectation.from_channel(
        device_id="heater",
        channel=channel,
        commanded_value=5.0,
        now=10.0,
    )

    verified = WriteVerdict.from_reading(
        expectation,
        Reading(device_id="heater", channel_id="temperature", value=5.05),
        now=10.0,
    )
    mismatch = WriteVerdict.from_reading(
        expectation,
        Reading(device_id="heater", channel_id="temperature", value=6.0),
        now=10.0,
    )
    inconclusive = WriteVerdict.from_operation_verdict(
        expectation,
        OperationVerdict(
            status=VerdictStatus.INCONCLUSIVE.value,
            confidence=0.0,
            detail="no camera evidence",
        ),
    )

    assert verified.status == WriteVerdictStatus.VERIFIED.value
    assert verified.to_dict()["match"] is True
    assert mismatch.status == WriteVerdictStatus.MISMATCH.value
    assert inconclusive.status == WriteVerdictStatus.UNOBSERVABLE.value
    assert aggregate_write_verdicts([verified, mismatch]) == "mismatch"


@pytest.mark.asyncio
async def test_batch_cross_channel_verification_updates_command_trust_and_audit(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "hardware.ndjson"
    tools, trust, _registry = _tools(
        _cross_channel_context(actual=5.0), audit_path=audit_path
    )

    result = await tools.batch_actuate(
        {
            "device_id": "heater",
            "commands": [{"channel_id": "power", "value": 5.0}],
            "verify": True,
        }
    )

    verdict = result["channels"][0]["verification"]
    assert verdict["status"] == "verified"
    assert verdict["observation_channel_id"] == "temperature"
    assert trust.level("heater", "power") == HardwareTrustLevel.CANDIDATE
    entries = HardwareAuditLog(audit_path).read_entries()
    assert any(
        entry.action == "write_verification"
        and entry.channel == "power"
        and entry.outcome == "verified"
        for entry in entries
    )


@pytest.mark.asyncio
async def test_deferred_cross_channel_expectation_settles_on_later_read() -> None:
    tools, trust, registry = _tools(_cross_channel_context(actual=5.0))

    write_result = await tools.batch_actuate(
        {
            "device_id": "heater",
            "commands": [{"channel_id": "power", "value": 5.0}],
            "verify": False,
        }
    )
    assert write_result["channels"][0]["verification"]["status"] == "pending"
    assert trust.level("heater", "power") == HardwareTrustLevel.UNTRUSTED

    reader = HardwareTools(registry, hardware_trust_gate=trust)
    read_result = await reader.hw_read(device_id="heater", channel_id="temperature")

    assert read_result["verification"]["status"] == "verified"
    assert read_result["verification"]["channel_id"] == "power"
    assert trust.level("heater", "power") == HardwareTrustLevel.CANDIDATE


# ── WriteVerdictEvent payload and structure ─────────────────────────


class TestWriteVerdictEvent:
    """Verify terminal verdict event payload, identity, and detail."""

    def _expectation(self, *, identity: str = "session-A") -> WriteExpectation:
        return WriteExpectation.from_channel(
            device_id="heater",
            channel=_cross_channel_context().channel("power"),
            commanded_value=5.0,
            now=10.0,
            identity=identity,
        )

    def test_event_type_and_kind(self) -> None:
        from leapflow.hardware.write_verification import WriteVerdictEvent

        expectation = self._expectation()
        verdict = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.05),
            now=10.0,
        )
        event = WriteVerdictEvent(verdict=verdict, identity="session-A")
        assert event.kind == "write_verdict"
        assert event.event_type == "hw.write_verdict"
        assert event.device_id == "heater"
        assert event.channel_id == "power"

    def test_payload_contains_identity_and_trust(self) -> None:
        from leapflow.hardware.write_verification import WriteVerdictEvent

        expectation = self._expectation()
        verdict = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.05),
            now=10.0,
        )
        transition = {"before": {"level": "untrusted"}, "after": {"level": "candidate"}, "changed": True}
        event = WriteVerdictEvent(verdict=verdict, identity="session-A", trust_transition=transition)
        payload = event.to_payload()
        assert payload["session_id"] == "session-A"
        assert payload["trust_transition"] == transition
        assert payload["kind"] == "write_verdict"
        assert payload["device_id"] == "heater"
        assert payload["channel_id"] == "power"
        assert payload["status"] == "verified"

    def test_payload_omits_trust_when_none(self) -> None:
        from leapflow.hardware.write_verification import WriteVerdictEvent

        expectation = self._expectation()
        verdict = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.05),
            now=10.0,
        )
        event = WriteVerdictEvent(verdict=verdict, identity="session-A")
        payload = event.to_payload()
        assert "trust_transition" not in payload

    def test_detail_includes_operation_id(self) -> None:
        from leapflow.hardware.write_verification import WriteVerdictEvent

        expectation = self._expectation()
        verdict = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.05),
            now=10.0,
        )
        event = WriteVerdictEvent(verdict=verdict, identity="session-A")
        detail = event.to_detail()
        assert expectation.operation_id in detail
        assert "heater.power" in detail


# ── Trust transition audit in record_verdict ────────────────────────


class TestRecordVerdictTrustTransition:
    """Verify trust snapshots, audit, and publish in record_verdict()."""

    def test_verified_verdict_records_trust_transition_in_audit(
        self,
        tmp_path: Path,
    ) -> None:
        audit_path = tmp_path / "hw.ndjson"
        tools, trust, registry = _tools(
            _cross_channel_context(actual=5.0), audit_path=audit_path
        )
        # Create a verified verdict and record it
        expectation = WriteExpectation.from_channel(
            device_id="heater",
            channel=_cross_channel_context().channel("power"),
            commanded_value=5.0,
            now=10.0,
            identity="trust-test",
        )
        verdict = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.0),
            now=10.0,
        )
        tools._write_verifier.record_verdict(verdict)

        entries = HardwareAuditLog(audit_path).read_entries()
        verification_entries = [
            e for e in entries if e.action == "write_verification"
        ]
        assert verification_entries
        last = verification_entries[-1]
        assert last.outcome == "verified"
        value = last.value
        assert "trust_transition" in value
        assert value["trust_transition"]["changed"] is True

    def test_terminal_verdict_publishes_event(self) -> None:
        """Terminal verdict must call registry.publish_event with a WriteVerdictEvent."""
        published = []

        class FakeRegistry:
            pending_write_expectations: dict = {}
            outcome_recorder = None

            def publish_event(self, event: object) -> None:
                published.append(event)

        from leapflow.hardware.write_verification import WriteVerificationCoordinator, WriteVerdictEvent

        registry = FakeRegistry()
        coordinator = WriteVerificationCoordinator(
            registry,
            audit_log=HardwareAuditLog(None),
            trust_gate=HardwareTrustGate(candidate_at=1),
            identity_provider=lambda: "pub-test",
        )
        expectation = WriteExpectation.from_channel(
            device_id="heater",
            channel=_cross_channel_context().channel("power"),
            commanded_value=5.0,
            now=10.0,
            identity="pub-test",
        )
        verified = WriteVerdict.from_reading(
            expectation,
            Reading(device_id="heater", channel_id="temperature", value=5.0),
            now=10.0,
        )
        coordinator.record_verdict(verified)
        assert len(published) == 1
        assert isinstance(published[0], WriteVerdictEvent)
        assert published[0].identity == "pub-test"
        assert published[0].verdict.status == "verified"

    def test_pending_verdict_does_not_publish(self) -> None:
        """Pending (non-terminal) verdicts must not trigger an event."""
        published = []

        class FakeRegistry:
            pending_write_expectations: dict = {}
            outcome_recorder = None

            def publish_event(self, event: object) -> None:
                published.append(event)

        from leapflow.hardware.write_verification import WriteVerificationCoordinator

        coordinator = WriteVerificationCoordinator(
            FakeRegistry(),
            audit_log=HardwareAuditLog(None),
            trust_gate=None,
            identity_provider=lambda: "s1",
        )
        expectation = WriteExpectation.from_channel(
            device_id="heater",
            channel=_cross_channel_context(settling=1.0).channel("power"),
            commanded_value=5.0,
            now=10.0,
        )
        pending = WriteVerdict.pending(expectation)
        coordinator.record_verdict(pending)
        assert len(published) == 0


# ── Cross-session verdict visibility ────────────────────────────────


class TestCrossSessionFiltering:
    """settle_pending must only return verdicts visible to the active session."""

    @pytest.mark.asyncio
    async def test_other_session_verdict_hidden(self) -> None:
        """Session B's read must not surface session A's deferred expectation."""
        tools_a, trust, registry = _tools(_cross_channel_context(actual=5.0))
        # Session A writes (no immediate verify → deferred)
        write_result = await tools_a.batch_actuate(
            {
                "device_id": "heater",
                "commands": [{"channel_id": "power", "value": 5.0}],
                "verify": False,
            }
        )
        assert write_result["channels"][0]["verification"]["status"] == "pending"

        # Session B reads the observation channel
        tools_b = HardwareTools(
            registry,
            session_id="session-B",
            hardware_trust_gate=trust,
        )
        read_result = await tools_b.hw_read(device_id="heater", channel_id="temperature")
        # The deferred expectation from session A was settled, trust was updated,
        # but the verdict must NOT be visible to session B.
        verdicts = read_result.get("write_verdicts", [])
        # Either no verdicts returned, or they're filtered
        assert not verdicts or all(
            v.get("expectation", {}).get("session_id") in ("", "session-B")
            for v in verdicts
        )

    @pytest.mark.asyncio
    async def test_same_session_verdict_visible(self) -> None:
        """Session A's own read must return its deferred verdict."""
        tools, trust, _reg = _tools(_cross_channel_context(actual=5.0))
        await tools.batch_actuate(
            {
                "device_id": "heater",
                "commands": [{"channel_id": "power", "value": 5.0}],
                "verify": False,
            }
        )
        read_result = await tools.hw_read(device_id="heater", channel_id="temperature")
        assert read_result.get("verification", {}).get("status") == "verified"

    def test_expectation_preserves_identity(self) -> None:
        """from_channel must retain the originating session identity."""
        expectation = WriteExpectation.from_channel(
            device_id="heater",
            channel=_cross_channel_context().channel("power"),
            commanded_value=5.0,
            identity="my-session",
        )
        assert expectation.identity == "my-session"
        assert expectation.to_dict()["session_id"] == "my-session"

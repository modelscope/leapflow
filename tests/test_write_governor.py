# Copyright (c) Alibaba, Inc. and its affiliates.
"""WriteGovernor unit tests: approval evaluation, trust skip, fail-closed semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.audit import HardwareAuditLog
from leapflow.hardware.write_governor import (
    WriteGovernanceRequest,
    WriteGovernor,
    execute_sequential_batch,
)
from leapflow.hardware.transport import TransportError, WriteOutcome


# ── Helpers ──────────────────────────────────────────────────────────────


def _descriptor(
    *,
    device_id: str = "heater",
    channel_id: str = "power",
    reversible: bool = True,
    kind: str = "hw_actuate",
) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        metadata={
            "device_id": device_id,
            "channel_id": channel_id,
            "reversible": reversible,
        },
    )


def _request(
    *,
    device_id: str = "heater",
    channel_id: str = "power",
    value: float = 5.0,
    identity: str = "session-1",
    reversible: bool = True,
    descriptor: Any = None,
) -> WriteGovernanceRequest:
    desc = descriptor or _descriptor(
        device_id=device_id,
        channel_id=channel_id,
        reversible=reversible,
    )
    return WriteGovernanceRequest.single(
        device_id=device_id,
        channel_id=channel_id,
        value=value,
        descriptor=desc,
        identity=identity,
        reversible=reversible,
    )


# ── Gate behavior ────────────────────────────────────────────────────────


def _async_gate(approved: bool = True, denial: str = "") -> SimpleNamespace:
    async def _evaluate(_d: Any) -> SimpleNamespace:
        return SimpleNamespace(approved=approved, denial_message=denial)

    return SimpleNamespace(evaluate=_evaluate)


def _boom_gate() -> SimpleNamespace:
    async def _evaluate(_d: Any) -> None:
        raise RuntimeError("gate crash")

    return SimpleNamespace(evaluate=_evaluate)


class TestGateEvaluation:
    @pytest.mark.asyncio
    async def test_no_gate_denies(self) -> None:
        gov = WriteGovernor(gate=None)
        decision = await gov.evaluate(_request())
        assert not decision.allowed
        assert "No approval gate" in decision.denial

    @pytest.mark.asyncio
    async def test_approved_returns_allowed(self) -> None:
        gov = WriteGovernor(gate=_async_gate(approved=True))
        decision = await gov.evaluate(_request())
        assert decision.allowed
        assert decision.denial == ""

    @pytest.mark.asyncio
    async def test_denied_returns_denial_message(self) -> None:
        gov = WriteGovernor(gate=_async_gate(approved=False, denial="operator refused"))
        decision = await gov.evaluate(_request())
        assert not decision.allowed
        assert "operator refused" in decision.denial

    @pytest.mark.asyncio
    async def test_gate_exception_denies(self) -> None:
        gov = WriteGovernor(gate=_boom_gate())
        decision = await gov.evaluate(_request())
        assert not decision.allowed
        assert "failed" in decision.denial


# ── Trust-based skip ─────────────────────────────────────────────────────


class _FakeTrust:
    """Minimal trust gate that approves skip above a threshold."""

    def __init__(self, skip: bool = True) -> None:
        self._skip = skip
        self._level = SimpleNamespace(name="VERIFIED")

    def may_skip_approval(
        self, device_id: str, channel_id: str, *, reversible: bool = True
    ) -> bool:
        return self._skip and reversible

    def level(self, device_id: str, channel_id: str) -> Any:
        return self._level


class TestTrustSkip:
    @pytest.mark.asyncio
    async def test_trust_skip_when_enabled_and_reversible(self) -> None:
        gov = WriteGovernor(
            gate=None,  # no gate -- trust skip bypasses it
            trust_gate=_FakeTrust(skip=True),
            trust_skip_enabled=True,
            audit_log=HardwareAuditLog(None),
        )
        decision = await gov.evaluate(_request(reversible=True))
        assert decision.allowed
        assert decision.trust_skipped

    @pytest.mark.asyncio
    async def test_trust_skip_denied_for_irreversible(self) -> None:
        gov = WriteGovernor(
            gate=_async_gate(approved=False, denial="should not reach here"),
            trust_gate=_FakeTrust(skip=True),
            trust_skip_enabled=True,
        )
        decision = await gov.evaluate(_request(reversible=False))
        # Irreversible must always require full approval
        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_trust_skip_disabled_falls_through_to_gate(self) -> None:
        gov = WriteGovernor(
            gate=_async_gate(approved=True),
            trust_gate=_FakeTrust(skip=True),
            trust_skip_enabled=False,
        )
        decision = await gov.evaluate(_request())
        assert decision.allowed
        assert not decision.trust_skipped

    @pytest.mark.asyncio
    async def test_trust_skip_records_audit(self, tmp_path: Any) -> None:
        audit_path = tmp_path / "audit.ndjson"
        gov = WriteGovernor(
            gate=None,
            trust_gate=_FakeTrust(skip=True),
            trust_skip_enabled=True,
            audit_log=HardwareAuditLog(audit_path),
        )
        await gov.evaluate(_request())
        entries = HardwareAuditLog(audit_path).read_entries()
        skip_entries = [e for e in entries if e.action == "trust_skip"]
        assert skip_entries
        assert "trust_skip=True" in skip_entries[0].outcome


# ── WriteGovernanceRequest factories ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["batch", "control_session"])
async def test_primary_trust_never_authorizes_untrusted_secondary(shape: str) -> None:
    descriptor = _descriptor(channel_id="trusted")
    descriptor.metadata.update(
        {
            shape: True,
            "channels": ["trusted", "draft"],
            "commands": [{"channel_id": "trusted"}, {"channel_id": "draft"}],
        }
    )
    checked: list[str] = []

    def may_skip(device_id: str, channel_id: str, **kwargs: Any) -> bool:
        checked.append(channel_id)
        return channel_id == "trusted"

    gov = WriteGovernor(
        gate=_async_gate(False),
        trust_gate=SimpleNamespace(may_skip_approval=may_skip),
        trust_skip_enabled=True,
    )
    decision = await gov.evaluate(_request(channel_id="trusted", descriptor=descriptor))
    assert not decision.allowed
    assert not decision.trust_skipped
    assert "draft" in checked


@pytest.mark.asyncio
@pytest.mark.parametrize("irreversible_secondary", [False, True])
async def test_batch_trust_exemption_requires_every_channel_to_be_reversible(
    irreversible_secondary: bool,
) -> None:
    descriptor = _descriptor(channel_id="first")
    descriptor.metadata["commands"] = [
        {"channel_id": "first", "reversible": True},
        {"channel_id": "second", "reversible": not irreversible_secondary},
    ]
    checked: list[str] = []

    def may_skip(device_id: str, channel_id: str, **kwargs: Any) -> bool:
        checked.append(channel_id)
        return True

    gov = WriteGovernor(
        gate=_async_gate(False),
        trust_gate=SimpleNamespace(may_skip_approval=may_skip),
        trust_skip_enabled=True,
    )
    decision = await gov.evaluate(_request(channel_id="first", descriptor=descriptor))
    assert decision.allowed is not irreversible_secondary
    assert decision.trust_skipped is not irreversible_secondary
    if not irreversible_secondary:
        assert checked == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_succeeds", [False, True])
async def test_sequential_batch_preserves_success_and_empty_noop(first_succeeds: bool) -> None:
    attempted: list[str] = []

    async def write(channel_id: str, value: Any) -> WriteOutcome:
        attempted.append(channel_id)
        return WriteOutcome(ok=True, side_effect_state="committed")

    commands = (("first", 1), ("second", 2)) if first_succeeds else ()
    result = await execute_sequential_batch(SimpleNamespace(write=write), commands)
    assert result.ok
    assert result.side_effect_state == ("committed" if commands else "none")
    assert attempted == [cid for cid, _ in commands]


@pytest.mark.asyncio
async def test_trust_exception_requires_gate() -> None:
    def broken(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("trust unavailable")

    gov = WriteGovernor(
        gate=_async_gate(False),
        trust_gate=SimpleNamespace(may_skip_approval=broken),
        trust_skip_enabled=True,
    )
    assert not (await gov.evaluate(_request())).allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "partial", "unknown", "transport", "defect"])
@pytest.mark.parametrize("first_succeeds", [False, True])
async def test_sequential_batch_stops_and_preserves_effects(
    failure: str,
    first_succeeds: bool,
) -> None:
    attempted: list[str] = []

    async def write(channel_id: str, value: Any) -> WriteOutcome:
        attempted.append(channel_id)
        if first_succeeds and channel_id == "first":
            return WriteOutcome(ok=True, side_effect_state="committed")
        if failure == "transport":
            raise TransportError("refused")
        if failure == "defect":
            raise TypeError("driver defect")
        return WriteOutcome(ok=False, side_effect_state=failure)

    result = await execute_sequential_batch(
        SimpleNamespace(write=write),
        (("first", 1), ("second", 2), ("third", 3)),
    )
    expected = (
        "unknown"
        if failure in ("unknown", "defect")
        else ("partial" if first_succeeds or failure == "partial" else "none")
    )
    assert not result.ok
    assert result.side_effect_state == expected
    assert attempted == (["first", "second"] if first_succeeds else ["first"])
    assert result.outcomes[-1].failure_code == "batch_not_attempted"
    assert result.outcomes[-1].side_effect_state == "none"


class TestRequestFactories:
    def test_single_factory(self) -> None:
        req = WriteGovernanceRequest.single(
            device_id="heater",
            channel_id="power",
            value=5.0,
            descriptor="d",
            identity="s1",
        )
        assert req.channel_ids == ("power",)
        assert req.values == (5.0,)
        assert not req.verify

    def test_batch_factory(self) -> None:
        req = WriteGovernanceRequest.batch(
            device_id="heater",
            channel_ids=("power", "temperature"),
            values=(5.0, 30.0),
            descriptor="d",
            identity="s1",
            verify=True,
            source="batch-test",
        )
        assert req.channel_ids == ("power", "temperature")
        assert req.verify is True
        assert req.source == "batch-test"

    def test_request_is_frozen(self) -> None:
        req = _request()
        with pytest.raises(AttributeError):
            req.device_id = "other"  # type: ignore[misc]

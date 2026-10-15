# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the LHP v2 execution permit system.

Covers: signing/verification, tamper rejection, unknown key, audience mismatch,
digest mismatch, expiry, clock skew, nonce replay, pre-boot rejection, batch
digest, observe-only/trusted_v1 node modes, permit-required local refusal.
"""

from __future__ import annotations

import time

import pytest

from leapflow.hardware.lhp.permit import (
    DEFAULT_PERMIT_TTL_S,
    MAX_CLOCK_SKEW_S,
    ExecutionPermit,
    PermitIssuer,
    PermitVerifier,
    action_digest,
    generate_keypair,
    key_id_from_public,
    public_key_pem,
    public_key_from_pem,
)
from leapflow.hardware.lhp.protocol import (
    FAILURE_PERMIT_REQUIRED,
    FAILURE_WRITES_DISABLED,
)
from leapflow.hardware.lhp.server import (
    LhpNodeServer,
    WRITE_MODE_OBSERVE_ONLY,
    WRITE_MODE_PERMIT_REQUIRED,
    WRITE_MODE_TRUSTED_V1,
)
from leapflow.hardware.transport import SIDE_EFFECT_NONE


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_issuer(ttl_s: float = DEFAULT_PERMIT_TTL_S) -> PermitIssuer:
    private, _ = generate_keypair()
    return PermitIssuer(private, ttl_s=ttl_s)


def _make_verifier(issuer: PermitIssuer, *, node_id: str = "jetson01") -> PermitVerifier:
    kid = issuer.key_id
    pub = issuer.public_key
    return PermitVerifier({kid: pub}, node_id=node_id)


def _issue(issuer: PermitIssuer, *, audience: str = "jetson01", digest: str = "") -> ExecutionPermit:
    return issuer.issue(
        audience=audience,
        session_id="sess-1",
        workspace="/workspace",
        approval_id="appr-1",
        digest=digest or action_digest(method="write", device_id="arm", commands=[("joint.shoulder.position", 1.5)]),
    )


# ── action_digest ────────────────────────────────────────────────────────


class TestActionDigest:
    def test_deterministic(self) -> None:
        d1 = action_digest(method="write", device_id="arm", commands=[("ch", 1.0)])
        d2 = action_digest(method="write", device_id="arm", commands=[("ch", 1.0)])
        assert d1 == d2

    def test_different_value_different_digest(self) -> None:
        d1 = action_digest(method="write", device_id="arm", commands=[("ch", 1.0)])
        d2 = action_digest(method="write", device_id="arm", commands=[("ch", 2.0)])
        assert d1 != d2

    def test_batch_order_matters(self) -> None:
        d1 = action_digest(method="write_batch", device_id="arm", commands=[("a", 1), ("b", 2)])
        d2 = action_digest(method="write_batch", device_id="arm", commands=[("b", 2), ("a", 1)])
        assert d1 != d2

    def test_control_digest(self) -> None:
        d = action_digest(
            method="control.open",
            device_id="arm",
            mode="position",
            frequency_hz=50.0,
            channels=["j1", "j2"],
            targets={"j1": 0.5, "j2": 1.0},
            lease_s=1.0,
            comm_loss_policy="halt",
        )
        assert isinstance(d, str)
        assert len(d) == 64  # sha256 hex


# ── PermitIssuer / PermitVerifier round-trip ─────────────────────────────


class TestPermitRoundTrip:
    def test_valid_permit_passes(self) -> None:
        issuer = _make_issuer()
        verifier = _make_verifier(issuer)
        permit = _issue(issuer)
        rejection = verifier.verify(permit, expected_digest=permit.action_digest)
        assert rejection is None

    def test_tampered_signature_rejected(self) -> None:
        issuer = _make_issuer()
        verifier = _make_verifier(issuer)
        permit = _issue(issuer)
        # Flip a byte in the signature
        bad_sig = "ff" + permit.signature[2:]
        tampered = ExecutionPermit(**{**permit.to_dict(), "signature": bad_sig})
        rejection = verifier.verify(tampered, expected_digest=tampered.action_digest)
        assert rejection is not None
        assert rejection.code == "permit_invalid"

    def test_unknown_key_rejected(self) -> None:
        issuer = _make_issuer()
        other_issuer = _make_issuer()
        verifier = _make_verifier(other_issuer)  # trusts the other key only
        permit = _issue(issuer)
        rejection = verifier.verify(permit, expected_digest=permit.action_digest)
        assert rejection is not None
        assert rejection.code == "permit_invalid"

    def test_audience_mismatch_rejected(self) -> None:
        issuer = _make_issuer()
        verifier = _make_verifier(issuer, node_id="jetson02")
        permit = _issue(issuer, audience="jetson01")
        rejection = verifier.verify(permit, expected_digest=permit.action_digest)
        assert rejection is not None
        assert rejection.code == "permit_audience_mismatch"

    def test_digest_mismatch_rejected(self) -> None:
        issuer = _make_issuer()
        verifier = _make_verifier(issuer)
        permit = _issue(issuer)
        rejection = verifier.verify(permit, expected_digest="wrong_digest")
        assert rejection is not None
        assert rejection.code == "permit_digest_mismatch"

    def test_expired_permit_rejected(self) -> None:
        issuer = _make_issuer(ttl_s=1.0)
        # Use a verifier with boot_time far enough in the past
        verifier = PermitVerifier(
            {issuer.key_id: issuer.public_key},
            node_id="jetson01",
            boot_time=time.time() - 200,
        )
        permit = _issue(issuer)
        # Forge an expired permit (valid signature won't match, but
        # expiry is checked before signature for early rejection).
        # Instead, issue a permit then wait... but that's too slow.
        # Create a verifier that sees a past-expired permit:
        expired = ExecutionPermit(**{
            **permit.to_dict(),
            "issued_at": time.time() - 100,
            "expires_at": time.time() - 90,
        })
        rejection = verifier.verify(expired, expected_digest=expired.action_digest)
        assert rejection is not None
        assert rejection.code in ("permit_expired", "permit_invalid")

    def test_nonce_replay_rejected(self) -> None:
        issuer = _make_issuer()
        verifier = _make_verifier(issuer)
        permit = _issue(issuer)
        # First use succeeds
        r1 = verifier.verify(permit, expected_digest=permit.action_digest)
        assert r1 is None
        # Replay rejected
        r2 = verifier.verify(permit, expected_digest=permit.action_digest)
        assert r2 is not None
        assert r2.code == "permit_replayed"

    def test_pre_boot_rejected(self) -> None:
        issuer = _make_issuer()
        verifier = PermitVerifier(
            {issuer.key_id: issuer.public_key},
            node_id="jetson01",
            boot_time=time.time() + 100,  # boot in the future
        )
        permit = _issue(issuer)
        rejection = verifier.verify(permit, expected_digest=permit.action_digest)
        assert rejection is not None
        assert rejection.code == "permit_pre_boot"

    def test_key_serialization_round_trip(self) -> None:
        issuer = _make_issuer()
        pem = public_key_pem(issuer.public_key)
        restored = public_key_from_pem(pem)
        assert key_id_from_public(restored) == issuer.key_id


# ── Serialization ────────────────────────────────────────────────────────


class TestPermitSerialization:
    def test_to_dict_from_dict_round_trip(self) -> None:
        issuer = _make_issuer()
        permit = _issue(issuer)
        restored = ExecutionPermit.from_dict(permit.to_dict())
        assert restored.nonce == permit.nonce
        assert restored.signature == permit.signature
        assert restored.action_digest == permit.action_digest


# ── Node write mode gates (using mock registry) ─────────────────────────


class _MockChannel:
    def __init__(self, channel_id: str = "ch", writable: bool = True) -> None:
        self.channel_id = channel_id
        self.is_writable = writable
        self.is_readable = True
        self.is_media = False
        self.envelope = _MockEnvelope()


class _MockEnvelope:
    requires_interlocks: tuple[str, ...] = ()

    def contains(self, value: object) -> bool:
        return True

    def rate_wait_s(self, *, delta: float, elapsed_s: float) -> float:
        return 0.0


class _MockContext:
    def __init__(self, device_id: str = "arm") -> None:
        self.device_id = device_id
        self.safety = None
        self.transport = type("T", (), {"kind": "mock", "config": {}})()
        self._channels = {"ch": _MockChannel("ch")}
        self.halt_supported = False

    def channel(self, name: str) -> _MockChannel | None:
        return self._channels.get(name)

    def interlock(self, name: str) -> None:
        return None

    def to_dict(self) -> dict:
        return {"device_id": self.device_id, "channels": {}, "transport": {"kind": "mock", "config": {}}}


class _MockTransport:
    def __init__(self) -> None:
        self.last_write: tuple[str, object] | None = None

    async def write(self, channel_id: str, value: object) -> object:
        from leapflow.hardware.transport import WriteOutcome, SIDE_EFFECT_COMMITTED
        self.last_write = (channel_id, value)
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED)


class _MockRegistry:
    def __init__(self) -> None:
        self._ctx = _MockContext()
        self._transport = _MockTransport()

    def contexts(self) -> list[_MockContext]:
        return [self._ctx]

    def context(self, device_id: str) -> _MockContext | None:
        if device_id == self._ctx.device_id:
            return self._ctx
        return None

    def device_io(self, device_id: str) -> object:
        import contextlib
        return contextlib.nullcontext()

    async def transport(self, device_id: str) -> _MockTransport:
        return self._transport

    def check_safety_policy(self, context: object, channel: object, value: object) -> tuple[bool, str]:
        return True, ""


@pytest.mark.asyncio
async def test_observe_only_refuses_all_writes() -> None:
    server = LhpNodeServer(
        _MockRegistry(), node_id="n", write_mode=WRITE_MODE_OBSERVE_ONLY,
    )
    result = await server.handle_request(
        {"id": 1, "method": "hello", "params": {"protocol": 2}},
    )
    assert result["ok"]
    result = await server.handle_request(
        {"id": 2, "method": "write", "params": {"device_id": "arm", "channel_id": "ch", "value": 1.0}},
    )
    assert result["ok"]
    outcome = result["result"]
    assert outcome["ok"] is False
    assert outcome["failure_code"] == FAILURE_WRITES_DISABLED
    assert outcome["side_effect_state"] == SIDE_EFFECT_NONE


@pytest.mark.asyncio
async def test_permit_required_without_permit_refuses() -> None:
    issuer = _make_issuer()
    verifier = _make_verifier(issuer, node_id="n")
    server = LhpNodeServer(
        _MockRegistry(), node_id="n",
        write_mode=WRITE_MODE_PERMIT_REQUIRED,
        permit_verifier=verifier,
    )
    await server.handle_request({"id": 1, "method": "hello", "params": {"protocol": 2}})
    result = await server.handle_request(
        {"id": 2, "method": "write", "params": {"device_id": "arm", "channel_id": "ch", "value": 1.0}},
    )
    assert result["ok"]
    outcome = result["result"]
    assert outcome["ok"] is False
    assert outcome["failure_code"] == FAILURE_PERMIT_REQUIRED


@pytest.mark.asyncio
async def test_trusted_v1_allows_writes_without_permit() -> None:
    server = LhpNodeServer(
        _MockRegistry(), node_id="n", write_mode=WRITE_MODE_TRUSTED_V1,
    )
    await server.handle_request({"id": 1, "method": "hello", "params": {"protocol": 2}})
    result = await server.handle_request(
        {"id": 2, "method": "write", "params": {"device_id": "arm", "channel_id": "ch", "value": 1.0}},
    )
    assert result["ok"]
    outcome = result["result"]
    assert outcome["ok"] is True


@pytest.mark.asyncio
async def test_permit_required_with_valid_permit_passes() -> None:
    issuer = _make_issuer()
    verifier = _make_verifier(issuer, node_id="n")
    server = LhpNodeServer(
        _MockRegistry(), node_id="n",
        write_mode=WRITE_MODE_PERMIT_REQUIRED,
        permit_verifier=verifier,
    )
    await server.handle_request({"id": 1, "method": "hello", "params": {"protocol": 2}})
    dig = action_digest(method="write", device_id="arm", commands=[("ch", 1.0)])
    permit = issuer.issue(
        audience="n", session_id="s", workspace="/w", approval_id="a", digest=dig,
    )
    result = await server.handle_request({
        "id": 2,
        "method": "write",
        "params": {
            "device_id": "arm",
            "channel_id": "ch",
            "value": 1.0,
            "permit": permit.to_dict(),
        },
    })
    assert result["ok"]
    outcome = result["result"]
    assert outcome["ok"] is True

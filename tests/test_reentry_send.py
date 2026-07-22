"""S2 outbound SO1+SO2+SO4: governance kernel for autonomous re-entry sends.

Hermetic unit tests for the pure decision primitives — contracts + target
resolution (SO1), send-scope Progressive Trust (SO2), and rate/idempotency/
budget guards + the combined decision flow (SO4). No I/O, no network.
"""
from __future__ import annotations

from types import SimpleNamespace

from leapflow.scheduler.reentry_send import (
    ReentrySendSpec,
    SendAction,
    SendGovernor,
    SendRateLimiter,
    SendTarget,
    resolve_reentry_send_target,
)
from leapflow.security.send_trust import SendTrustLedger, SendTrustLevel


# ── SO1: contracts + target resolution ──


def test_resolve_target_from_event_match() -> None:
    trigger = SimpleNamespace(event_match={"platform": "lark", "chat": "c1", "keyword": "x"})
    target = resolve_reentry_send_target(trigger)
    assert target == SendTarget(platform="lark", chat="c1")


def test_resolve_target_none_without_platform_chat() -> None:
    # Time-triggered re-entry (no event_match) or partial match -> no target.
    assert resolve_reentry_send_target(SimpleNamespace(event_match=None)) is None
    assert resolve_reentry_send_target(SimpleNamespace(event_match={"platform": "lark"})) is None
    assert resolve_reentry_send_target(SimpleNamespace(event_match={})) is None


def test_send_spec_idempotency_key_is_stable_and_content_sensitive() -> None:
    a = ReentrySendSpec(target=None, text="done", origin_trigger_id="t1")
    b = ReentrySendSpec(target=None, text="done", origin_trigger_id="t1")
    c = ReentrySendSpec(target=None, text="different", origin_trigger_id="t1")
    assert a.idempotency_key() == b.idempotency_key()
    assert a.idempotency_key() != c.idempotency_key()


# ── SO2: send-scope Progressive Trust ──


def test_send_trust_gradient_accrues_on_allow() -> None:
    led = SendTrustLedger(verified_at=3)
    key = "lark:c1:reply"
    assert led.level(key) == SendTrustLevel.DRAFT
    led.record_allow(key)
    assert led.level(key) == SendTrustLevel.CANDIDATE
    led.record_allow(key)
    led.record_allow(key)
    assert led.level(key) == SendTrustLevel.VERIFIED


def test_send_trust_deny_freezes_to_draft() -> None:
    led = SendTrustLedger(verified_at=2)
    key = "lark:c1:reply"
    led.record_allow(key)
    led.record_allow(key)
    assert led.level(key) == SendTrustLevel.VERIFIED
    led.record_deny(key)
    assert led.level(key) == SendTrustLevel.DRAFT          # frozen
    led.record_allow(key)                                   # unfreezes
    assert led.level(key) != SendTrustLevel.DRAFT


def test_send_trust_auto_approve_only_verified_nondestructive() -> None:
    led = SendTrustLedger(verified_at=1)
    key = "lark:c1:reply"
    assert led.auto_approve_ok(key, destructive=False) is False   # DRAFT
    led.record_allow(key)                                          # -> VERIFIED (verified_at=1)
    assert led.auto_approve_ok(key, destructive=False) is True
    assert led.auto_approve_ok(key, destructive=True) is False     # destructive never auto


def test_send_trust_state_round_trip() -> None:
    led = SendTrustLedger(verified_at=3)
    led.record_allow("lark:c1:reply")
    led.record_deny("lark:c2:reply")
    restored = SendTrustLedger()
    restored.load_state(led.to_state())
    assert restored.level("lark:c1:reply") == SendTrustLevel.CANDIDATE
    assert restored.level("lark:c2:reply") == SendTrustLevel.DRAFT


# ── SO4: rate limiter + governor decision flow ──


def test_rate_limiter_blocks_beyond_quota() -> None:
    rate = SendRateLimiter(per_hour=2)
    assert rate.allow("lark:c1", now=100.0) is True
    assert rate.allow("lark:c1", now=101.0) is True
    assert rate.allow("lark:c1", now=102.0) is False          # 3rd within window
    assert rate.allow("lark:c1", now=100.0 + 3601.0) is True  # window slid


def test_rate_limiter_unlimited_when_zero() -> None:
    rate = SendRateLimiter(per_hour=0)
    for i in range(100):
        assert rate.allow("lark:c1", now=float(i)) is True


def _governor(*, enabled=True, verified_at=3, per_hour=4, budget=50):
    return SendGovernor(
        trust=SendTrustLedger(verified_at=verified_at),
        rate=SendRateLimiter(per_hour=per_hour),
        enabled=enabled,
        global_budget=budget,
    )


def _spec(text="done", tid="t1"):
    return ReentrySendSpec(target=SendTarget("lark", "c1"), text=text, origin_trigger_id=tid)


def test_governor_disabled_blocks() -> None:
    dec = _governor(enabled=False).decide(_spec(), destructive=False, has_approver=True, now=0.0)
    assert dec.action == SendAction.BLOCKED and dec.reason == "disabled"


def test_governor_no_target_blocks() -> None:
    spec = ReentrySendSpec(target=None, text="x", origin_trigger_id="t1")
    dec = _governor().decide(spec, destructive=False, has_approver=True, now=0.0)
    assert dec.action == SendAction.BLOCKED and dec.reason == "no_target"


def test_governor_deny_without_trust_or_approver() -> None:
    dec = _governor().decide(_spec(), destructive=False, has_approver=False, now=0.0)
    assert dec.action == SendAction.DENY and dec.reason == "no_approver_no_trust"


def test_governor_queues_for_human_without_trust() -> None:
    dec = _governor().decide(_spec(), destructive=False, has_approver=True, now=0.0)
    assert dec.action == SendAction.NEEDS_APPROVAL


def test_governor_auto_allows_when_verified_nondestructive() -> None:
    gov = _governor(verified_at=2)
    gov._trust.record_allow(SendTarget("lark", "c1").grant_key("reply"))
    gov._trust.record_allow(SendTarget("lark", "c1").grant_key("reply"))
    dec = gov.decide(_spec(), destructive=False, has_approver=False, now=0.0)
    assert dec.action == SendAction.AUTO_ALLOW and dec.reason == "trust_verified"
    # Destructive stays gated even when trust is high.
    dec2 = gov.decide(_spec(text="x2", tid="t2"), destructive=True, has_approver=True, now=0.0)
    assert dec2.action == SendAction.NEEDS_APPROVAL


def test_governor_blocks_duplicate_after_send() -> None:
    gov = _governor()
    spec = _spec()
    gov.record_sent(spec)
    dec = gov.decide(spec, destructive=False, has_approver=True, now=0.0)
    assert dec.action == SendAction.BLOCKED and dec.reason == "duplicate"


def test_governor_blocks_when_global_budget_exhausted() -> None:
    gov = _governor(budget=1)
    gov.record_sent(_spec(text="a", tid="ta"))
    dec = gov.decide(_spec(text="b", tid="tb"), destructive=False, has_approver=True, now=0.0)
    assert dec.action == SendAction.BLOCKED and dec.reason == "global_budget_exhausted"


def test_governor_blocks_when_rate_limited() -> None:
    gov = _governor(per_hour=1)
    first = gov.decide(_spec(text="a", tid="ta"), destructive=False, has_approver=True, now=0.0)
    assert first.action == SendAction.NEEDS_APPROVAL          # consumed the one token
    second = gov.decide(_spec(text="b", tid="tb"), destructive=False, has_approver=True, now=1.0)
    assert second.action == SendAction.BLOCKED and second.reason == "rate_limited"

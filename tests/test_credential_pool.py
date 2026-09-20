# Copyright (c) Alibaba, Inc. and its affiliates.
"""Credential pool state machine + FailoverChain credential-failover tests.

Covers the P1-4 credential state machine:
- OK -> EXHAUSTED -> OK recovery after cooldown
- OK -> DEAD terminal (stays DEAD across the cooldown window)
- LRU selection (least-recently-used OK key chosen first)
- AllCredentialsExhausted when every key is unusable
- FailoverChain: billing kills a key (DEAD), rate-limit cools it (EXHAUSTED),
  and a fully drained pool triggers provider failover / surfaces exhaustion
- CredentialRotateStrategy bows out when no rotatable credential remains
- AllCredentialsExhausted classified as auth_permanent / ADMIN_REQUIRED
"""
from __future__ import annotations

import pytest

from leapflow.engine.failure_envelope import Recoverability
from leapflow.engine.recovery_strategies.credential_rotate import CredentialRotateStrategy
from leapflow.engine.unified_classifier import UnifiedErrorClassifier
from leapflow.llm import provider_chain as pc
from leapflow.llm.base import LLMChatResponse, LLMProvider
from leapflow.llm.credential_state import AllCredentialsExhausted, CredentialState
from leapflow.llm.provider_chain import CredentialPool, FailoverChain, ProviderConfig


class _Clock:
    """Deterministic monotonic replacement injected into a pool's ``_now``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_pool(keys, cooldown_s: float = 10.0):
    pool = CredentialPool(keys, cooldown_s=cooldown_s, name="primary")
    clock = _Clock()
    pool._now = clock  # instance attribute shadows the staticmethod
    return pool, clock


# ---------------------------------------------------------------------------
# Pool state machine
# ---------------------------------------------------------------------------

def test_rate_limited_recovers_after_cooldown():
    pool, clock = _make_pool(["k1", "k2"], cooldown_s=10.0)

    pool.mark_rate_limited("k1", cooldown_s=10.0)
    assert pool._find("k1").state is CredentialState.EXHAUSTED
    # While cooling, the other OK key is selected instead.
    assert pool.acquire() == "k2"

    clock.advance(11.0)
    assert pool.has_available()
    assert pool._find("k1").state is CredentialState.OK


def test_dead_is_terminal_across_cooldown_window():
    pool, clock = _make_pool(["k1", "k2"])

    pool.mark_dead("k1", reason="billing: account disabled")
    assert pool._find("k1").state is CredentialState.DEAD

    clock.advance(10_000.0)
    pool.has_available()  # trigger the lazy recovery sweep
    assert pool._find("k1").state is CredentialState.DEAD

    # A transient (rate-limit) signal never resurrects a dead key.
    pool.mark_rate_limited("k1")
    assert pool._find("k1").state is CredentialState.DEAD


def test_lru_selection_picks_least_recently_used():
    pool, clock = _make_pool(["k1", "k2", "k3"])

    first = pool.acquire()
    clock.advance(1.0)
    second = pool.acquire()
    clock.advance(1.0)
    third = pool.acquire()

    assert {first, second, third} == {"k1", "k2", "k3"}

    clock.advance(1.0)
    # The first-acquired key is now the least-recently-used and comes back.
    assert pool.acquire() == first


def test_all_credentials_exhausted_carries_context():
    pool, _clock = _make_pool(["k1", "k2"], cooldown_s=30.0)

    pool.mark_dead("k1", reason="revoked")
    pool.mark_rate_limited("k2", cooldown_s=30.0)

    with pytest.raises(AllCredentialsExhausted) as excinfo:
        pool.acquire()

    exc = excinfo.value
    assert exc.provider == "primary"
    assert exc.total == 2
    assert exc.dead == 1
    assert exc.cooling_down == 1


def test_has_recoverable_false_only_when_all_dead():
    pool, _clock = _make_pool(["k1", "k2"])

    pool.mark_dead("k1", reason="x")
    assert pool.has_recoverable()  # k2 still OK

    pool.mark_dead("k2", reason="x")
    assert not pool.has_recoverable()


def test_record_success_resets_exhausted_key():
    pool, _clock = _make_pool(["k1", "k2"])
    pool.mark_rate_limited("k1", cooldown_s=999.0)
    assert pool._find("k1").state is CredentialState.EXHAUSTED

    pool.record_success("k1")
    assert pool._find("k1").state is CredentialState.OK
    assert pool._find("k1").consecutive_failures == 0


# ---------------------------------------------------------------------------
# FailoverChain credential integration
# ---------------------------------------------------------------------------

class _FakeError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ScriptedProvider(LLMProvider):
    """Provider that replays a scripted sequence of exceptions / responses."""

    def __init__(self, api_key: str, script) -> None:
        self.api_key = api_key
        self._script = list(script)
        self.calls = 0

    async def achat(self, messages, *, stream=True, enable_thinking=False,
                    on_chunk=None, **kwargs) -> LLMChatResponse:
        self.calls += 1
        item = self._script.pop(0) if self._script else LLMChatResponse(content="ok")
        if isinstance(item, Exception):
            raise item
        return item

    async def achat_stream(self, messages, *, enable_thinking=False, **kwargs):
        raise NotImplementedError
        yield ""  # pragma: no cover - unreachable, makes this an async generator


def _two_provider_chain(monkeypatch, build):
    configs = [
        ProviderConfig(name="primary", api_key="k1,k2", base_url="u", model="m", priority=0),
        ProviderConfig(name="fallback", api_key="fk", base_url="u", model="m", priority=1),
    ]
    pools = pc.parse_credential_pools(configs, cooldown_s=30.0)
    monkeypatch.setattr(pc, "_build_provider", build)
    return FailoverChain(configs, credential_pools=pools), pools


async def test_billing_kills_keys_then_fails_over(monkeypatch):
    def build(config, api_key):
        if config.name == "fallback":
            return _ScriptedProvider(api_key, [LLMChatResponse(content="from-fallback")])
        return _ScriptedProvider(api_key, [_FakeError("billing: payment required", status_code=402)])

    chain, pools = _two_provider_chain(monkeypatch, build)
    resp = await chain.achat([{"role": "user", "content": "hi"}], stream=False)

    assert resp.content == "from-fallback"
    assert chain.active_provider_name == "fallback"
    # Both primary keys were permanently killed by the billing error.
    assert not pools["primary"].has_recoverable()
    for key in ("k1", "k2"):
        assert pools["primary"]._find(key).state is CredentialState.DEAD
    # Active provider (fallback, single key) is always rotatable-eligible.
    assert chain.has_rotatable_credentials() is True


async def test_rate_limit_rotates_within_provider(monkeypatch):
    def build(config, api_key):
        if api_key == "k1":
            return _ScriptedProvider(api_key, [_FakeError("rate limit 429", status_code=429)])
        return _ScriptedProvider(api_key, [LLMChatResponse(content="via-k2")])

    configs = [ProviderConfig(name="primary", api_key="k1,k2", base_url="u", model="m")]
    pools = pc.parse_credential_pools(configs, cooldown_s=30.0)
    monkeypatch.setattr(pc, "_build_provider", build)
    chain = FailoverChain(configs, credential_pools=pools)

    resp = await chain.achat([{"role": "user", "content": "hi"}], stream=False)

    assert resp.content == "via-k2"
    # k1 was cooled down (EXHAUSTED), not killed; still recoverable.
    assert pools["primary"]._find("k1").state is CredentialState.EXHAUSTED
    assert chain.active_provider_name == "primary"


async def test_all_providers_exhausted_raises(monkeypatch):
    def build(config, api_key):
        return _ScriptedProvider(api_key, [_FakeError("insufficient_quota billing", status_code=402)])

    configs = [ProviderConfig(name="primary", api_key="k1,k2", base_url="u", model="m")]
    pools = pc.parse_credential_pools(configs, cooldown_s=30.0)
    monkeypatch.setattr(pc, "_build_provider", build)
    chain = FailoverChain(configs, credential_pools=pools)

    with pytest.raises(AllCredentialsExhausted):
        await chain.achat([{"role": "user", "content": "hi"}], stream=False)

    assert chain.has_rotatable_credentials() is False


# ---------------------------------------------------------------------------
# Recovery-layer wiring
# ---------------------------------------------------------------------------

class _Inspector:
    def __init__(self, value: bool) -> None:
        self._value = value

    def has_rotatable_credentials(self) -> bool:
        return self._value


def test_credential_rotate_bows_out_when_no_rotatable():
    # can_apply only consults budget + availability, so envelope/state are unused.
    assert CredentialRotateStrategy(_Inspector(True)).can_apply(None, None) is True
    assert CredentialRotateStrategy(_Inspector(False)).can_apply(None, None) is False
    # No inspector wired -> preserves the original always-applicable behavior.
    assert CredentialRotateStrategy().can_apply(None, None) is True


def test_all_credentials_exhausted_maps_to_admin_required():
    classifier = UnifiedErrorClassifier()
    envelope = classifier.classify_llm_error(
        AllCredentialsExhausted("primary", total=2, dead=2, cooling_down=0)
    )
    assert envelope.category == "auth_permanent"
    assert envelope.recoverability is Recoverability.ADMIN_REQUIRED

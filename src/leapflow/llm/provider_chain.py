"""Multi-provider LLM chain — failover, credential rotation, auxiliary client.

Architecture (Protocol-first, inspired by hermes credential_pool + transports):
- ProviderConfig: immutable endpoint descriptor with capability metadata
- CredentialPool: multi-key rotation with rate-limit cooldown (per key)
- FailoverChain: ordered provider list with automatic failover on errors
- AuxiliaryClient: cheap model for compression, approval, title generation

Design choices vs Hermes:
- Async-first (no thread pool / contextvar gymnastics)
- OpenAI-compatible transport only (covers 95% of providers)
- Config-driven via env + YAML overlay (no adapter matrix)
- Credential rotation is per-provider, not global
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, FrozenSet, List, Optional, Protocol, runtime_checkable

from leapflow.config import DEFAULT_LLM_CONTEXT_LENGTH
from leapflow.llm.base import ChunkCallback, LLMChatResponse, LLMProvider

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_LENGTH = DEFAULT_LLM_CONTEXT_LENGTH


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for a single LLM provider endpoint."""

    name: str
    api_key: str
    base_url: str
    model: str
    max_retries: int = 3
    timeout_s: float = 180.0
    context_length: int = _DEFAULT_CONTEXT_LENGTH
    supports_tools: bool = True
    supports_thinking: bool = False
    supports_vision: bool = False
    priority: int = 0


@dataclass(frozen=True)
class ProviderMetadata:
    """Runtime metadata about a provider (populated after first successful call)."""

    name: str
    model: str
    context_length: int = _DEFAULT_CONTEXT_LENGTH
    supports_tools: bool = True
    supports_thinking: bool = False
    supports_vision: bool = False


@runtime_checkable
class FailoverObserver(Protocol):
    """Observer notified on provider failover events."""

    def on_failover(self, from_provider: str, to_provider: str, reason: str) -> None: ...


class CredentialPool:
    """Multi-key rotation with per-key rate-limit cooldown.

    When a key hits a rate limit, it enters cooldown for a configurable period.
    The pool round-robins through non-cooled keys.
    """

    def __init__(
        self,
        keys: List[str],
        *,
        cooldown_s: float = 60.0,
    ) -> None:
        if not keys:
            raise ValueError("CredentialPool requires at least one key")
        self._keys = list(keys)
        self._cooldown_s = cooldown_s
        self._cooldown_until: Dict[int, float] = {}
        self._current_idx = 0

    @property
    def size(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """Return next available key, skipping those in cooldown."""
        now = time.monotonic()
        for _ in range(len(self._keys)):
            idx = self._current_idx % len(self._keys)
            self._current_idx = (self._current_idx + 1) % len(self._keys)
            until = self._cooldown_until.get(idx, 0.0)
            if now >= until:
                return self._keys[idx]
        return self._keys[0]

    def mark_rate_limited(self, key: str) -> None:
        """Put a key into cooldown after a rate-limit error."""
        try:
            idx = self._keys.index(key)
            self._cooldown_until[idx] = time.monotonic() + self._cooldown_s
            logger.info("credential_pool: key %d/%d in cooldown for %.0fs",
                        idx + 1, len(self._keys), self._cooldown_s)
        except ValueError:
            pass

    def rotate(self) -> str:
        """Force rotation to next key (e.g., on billing error)."""
        self._current_idx = (self._current_idx + 1) % len(self._keys)
        return self.get_key()


def _build_provider(config: ProviderConfig, pool: Optional[CredentialPool] = None) -> LLMProvider:
    """Construct an OpenAIChat provider from config, using pool key if available."""
    from leapflow.llm.openai_provider import OpenAIChat

    api_key = pool.get_key() if pool else config.api_key
    return OpenAIChat(
        api_key=api_key,
        base_url=config.base_url,
        model=config.model,
        max_retries=config.max_retries,
        timeout_s=config.timeout_s,
    )


class _CircuitState:
    """Per-provider circuit breaker state.

    States:
    - CLOSED: normal operation, requests flow through
    - OPEN: failures exceeded threshold, requests rejected for cooldown_s
    - HALF_OPEN: cooldown expired, next request is a probe
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, *, failure_threshold: int = 5, cooldown_s: float = 60.0) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._state = self.CLOSED
        self._opened_at = 0.0

    @property
    def is_available(self) -> bool:
        if self._state == self.CLOSED:
            return True
        if self._state == self.OPEN:
            if time.monotonic() - self._opened_at >= self._cooldown_s:
                self._state = self.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN allows one probe

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = self.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._state = self.OPEN
            self._opened_at = time.monotonic()
            logger.info(
                "circuit_breaker: opened after %d failures (cooldown %.0fs)",
                self._consecutive_failures, self._cooldown_s,
            )


class FailoverChain(LLMProvider):
    """Ordered provider chain with automatic failover and circuit breaker.

    Behavior:
    1. Primary provider receives all requests.
    2. On unrecoverable error (billing, auth, persistent 500), failover to next.
    3. Between turns, attempt to restore primary (avoid permanent degradation).
    4. Credential pool rotates keys within a single provider on rate-limit.
    5. Circuit breaker per provider: opens after N consecutive failures.
    6. Observer notified on every failover event.

    Implements LLMProvider so engine code needs zero changes.
    """

    _FAILOVER_ERRORS: FrozenSet[str] = frozenset({
        "billing", "auth", "unauthorized", "forbidden",
        "account", "quota", "insufficient",
    })

    def __init__(
        self,
        providers: List[ProviderConfig],
        *,
        credential_pools: Optional[Dict[str, CredentialPool]] = None,
        observer: Optional[FailoverObserver] = None,
        circuit_failure_threshold: int = 5,
        circuit_cooldown_s: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("FailoverChain requires at least one provider")
        self._configs = sorted(providers, key=lambda p: p.priority)
        self._pools = credential_pools or {}
        self._observer = observer
        self._active_idx = 0
        self._providers: Dict[int, LLMProvider] = {}
        self._failed_indices: set[int] = set()
        self._circuits: Dict[int, _CircuitState] = {
            i: _CircuitState(
                failure_threshold=circuit_failure_threshold,
                cooldown_s=circuit_cooldown_s,
            )
            for i in range(len(self._configs))
        }

    @property
    def active_provider_name(self) -> str:
        return self._configs[self._active_idx].name

    @property
    def active_config(self) -> ProviderConfig:
        return self._configs[self._active_idx]

    @property
    def model(self) -> str:
        return self._configs[self._active_idx].model

    @property
    def context_length(self) -> int:
        return self._configs[self._active_idx].context_length

    def _get_or_create(self, idx: int) -> LLMProvider:
        if idx not in self._providers:
            config = self._configs[idx]
            pool = self._pools.get(config.name)
            self._providers[idx] = _build_provider(config, pool)
        return self._providers[idx]

    def _should_failover(self, exc: BaseException) -> bool:
        """Determine if the error warrants trying the next provider."""
        msg = str(exc).lower()
        if any(keyword in msg for keyword in self._FAILOVER_ERRORS):
            return True
        status = getattr(exc, "status_code", None)
        if status in (401, 402, 403):
            return True
        return False

    def _is_rate_limit(self, exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        return status == 429 or "rate" in str(exc).lower()

    def _failover(self, reason: str) -> bool:
        """Move to next provider. Returns False if no more providers."""
        old_name = self._configs[self._active_idx].name
        self._failed_indices.add(self._active_idx)

        for idx in range(len(self._configs)):
            if idx not in self._failed_indices:
                self._active_idx = idx
                new_name = self._configs[idx].name
                logger.warning("llm_failover: %s → %s (reason: %s)", old_name, new_name, reason)
                if self._observer:
                    try:
                        self._observer.on_failover(old_name, new_name, reason)
                    except Exception as observer_exc:
                        logger.debug("llm_failover observer error: %s", observer_exc)
                return True
        return False

    def try_restore_primary(self) -> None:
        """Between turns, attempt to restore primary provider."""
        if self._active_idx == 0:
            return
        if 0 in self._failed_indices:
            return
        old = self._configs[self._active_idx].name
        self._active_idx = 0
        self._providers.pop(0, None)
        logger.info("llm_failover: restored primary provider %s (was %s)",
                     self._configs[0].name, old)

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        *,
        stream: bool = True,
        enable_thinking: bool = False,
        on_chunk: ChunkCallback = None,
        **kwargs: Any,
    ) -> LLMChatResponse:
        last_exc: Optional[BaseException] = None

        for attempt in range(len(self._configs)):
            provider = self._get_or_create(self._active_idx)
            config = self._configs[self._active_idx]
            pool = self._pools.get(config.name)

            try:
                resp = await provider.achat(
                    messages, stream=stream,
                    enable_thinking=enable_thinking,
                    on_chunk=on_chunk, **kwargs,
                )
                self._circuits[self._active_idx].record_success()
                return resp
            except Exception as exc:
                self._circuits[self._active_idx].record_failure()
                last_exc = exc

                if self._is_rate_limit(exc) and pool and pool.size > 1:
                    current_key = pool.get_key()
                    pool.mark_rate_limited(current_key)
                    self._providers.pop(self._active_idx, None)
                    logger.info("llm_chain: rotated credential for %s", config.name)
                    continue

                if self._should_failover(exc):
                    if not self._failover(str(exc)[:100]):
                        break
                    continue

                raise

        assert last_exc is not None
        raise last_exc

    async def achat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        last_exc: Optional[BaseException] = None

        for _attempt in range(len(self._configs)):
            provider = self._get_or_create(self._active_idx)
            config = self._configs[self._active_idx]
            pool = self._pools.get(config.name)

            try:
                async for chunk in provider.achat_stream(
                    messages, enable_thinking=enable_thinking, **kwargs
                ):
                    yield chunk
                self._circuits[self._active_idx].record_success()
                return
            except Exception as exc:
                self._circuits[self._active_idx].record_failure()
                last_exc = exc

                if self._is_rate_limit(exc) and pool and pool.size > 1:
                    current_key = pool.get_key()
                    pool.mark_rate_limited(current_key)
                    self._providers.pop(self._active_idx, None)
                    logger.info("llm_chain: rotated credential for %s (stream)", config.name)
                    continue

                if self._should_failover(exc):
                    if not self._failover(str(exc)[:100]):
                        break
                    continue

                raise

        assert last_exc is not None
        raise last_exc


class AuxiliaryClient:
    """Lightweight LLM client for cheap operations (summarization, approval, title).

    Uses a separate (often smaller/cheaper) model to avoid burning main model budget.
    Falls back to primary provider if no auxiliary is configured.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_output_tokens: int = 1024,
    ) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    async def summarize(self, text: str, *, max_chars: int = 2000) -> str:
        """Summarize text using the auxiliary model."""
        from leapflow.llm.message_builder import build_user_message_text
        prompt = (
            "Summarize the following text concisely, preserving key facts, "
            f"file paths, and action items. Stay under {max_chars} characters.\n\n"
            f"{text[:8000]}"
        )
        try:
            resp = await self._provider.achat(
                [build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return (resp.content or "").strip()[:max_chars]
        except Exception as exc:
            logger.warning("auxiliary.summarize failed: %s", exc)
            return text[:max_chars]

    async def classify_risk(self, command: str) -> float:
        """Classify command risk level (0.0 = safe, 1.0 = dangerous)."""
        from leapflow.llm.message_builder import build_user_message_text, build_system_message
        messages = [
            build_system_message(
                "You are a security classifier. Given a shell command, "
                "respond with ONLY a number 0.0-1.0 indicating risk level. "
                "0.0=completely safe, 0.5=moderate, 1.0=destructive. "
                "Consider: data loss, privilege escalation, network exposure."
            ),
            build_user_message_text(command),
        ]
        try:
            resp = await self._provider.achat(messages, stream=False, enable_thinking=False)
            text = (resp.content or "").strip()
            import re
            match = re.search(r"(\d+\.?\d*)", text)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
        except Exception:
            pass
        return 0.5

    async def generate_title(self, user_message: str) -> str:
        """Generate a short session title from the first user message."""
        from leapflow.llm.message_builder import build_user_message_text
        prompt = (
            "Generate a concise title (max 6 words) for a conversation "
            f"that starts with: {user_message[:200]}\n"
            "Reply with ONLY the title, no quotes or punctuation."
        )
        try:
            resp = await self._provider.achat(
                [build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return (resp.content or "").strip()[:60]
        except Exception:
            return user_message[:40]


def parse_provider_configs(
    primary_key: str,
    primary_url: str,
    primary_model: str,
    *,
    fallback_json: str = "",
    primary_context_length: int = _DEFAULT_CONTEXT_LENGTH,
) -> List[ProviderConfig]:
    """Parse provider configs from primary settings + optional fallback JSON.

    Fallback JSON format: [{"api_key":"...", "base_url":"...", "model":"...", ...}]
    """
    configs = [
        ProviderConfig(
            name="primary",
            api_key=primary_key,
            base_url=primary_url,
            model=primary_model,
            context_length=primary_context_length,
            priority=0,
        ),
    ]

    if fallback_json:
        try:
            fallbacks = json.loads(fallback_json)
            if isinstance(fallbacks, list):
                for i, fb in enumerate(fallbacks):
                    if not isinstance(fb, dict):
                        continue
                    configs.append(ProviderConfig(
                        name=fb.get("name", f"fallback_{i+1}"),
                        api_key=fb.get("api_key", primary_key),
                        base_url=fb.get("base_url", primary_url),
                        model=fb.get("model", primary_model),
                        max_retries=int(fb.get("max_retries", 2)),
                        timeout_s=float(fb.get("timeout_s", 180.0)),
                        context_length=int(fb.get("context_length", _DEFAULT_CONTEXT_LENGTH)),
                        supports_tools=fb.get("supports_tools", True),
                        priority=i + 1,
                    ))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse LEAPFLOW_LLM_FALLBACK_PROVIDERS: %s", exc)

    return configs


def parse_credential_pools(
    configs: List[ProviderConfig],
    *,
    cooldown_s: float = 60.0,
) -> Dict[str, CredentialPool]:
    """Build credential pools for providers that have comma-separated api_keys."""
    pools: Dict[str, CredentialPool] = {}
    for config in configs:
        if "," in config.api_key:
            keys = [k.strip() for k in config.api_key.split(",") if k.strip()]
            if len(keys) > 1:
                pools[config.name] = CredentialPool(keys, cooldown_s=cooldown_s)
                logger.info("credential_pool: %s has %d keys", config.name, len(keys))
    return pools

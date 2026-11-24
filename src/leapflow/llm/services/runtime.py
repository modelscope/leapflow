# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generation-pinned, stateless specialist calls with explicit resource ownership."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

from leapflow.llm.base import AsyncCloseableProvider, LLMChatResponse, LLMProvider
from leapflow.llm.provider_registry import LLMProviderRegistry
from leapflow.llm.services.codec import JsonCodec
from leapflow.llm.services.http_client import build_client
from leapflow.llm.services.protocol import (
    ServiceCodec, ServiceError, ServiceSpec, StructuredResult, Validator, canonical,
)

logger = logging.getLogger(__name__)
SecretResolver = Callable[[str], str | None]
ClientFactory = Callable[[ServiceSpec], httpx.AsyncClient]


def _provider_failure(exc: Exception, service_id: str) -> ServiceError | None:
    """Classify transport/SDK failures by type, never by model-generated message text."""
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ServiceError):
            return cause
        cause = cause.__cause__
    if isinstance(exc, httpx.TransportError):
        return ServiceError("Model connection failed; check the endpoint", service_id=service_id,
                            code="model_connection_failed", recoverable=True)
    if type(exc).__module__.split(".")[0] in {"openai", "anthropic"}:
        status = getattr(exc, "status_code", None)
        retryable = status in {408, 429, 500, 502, 503, 504} or type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError",
        }
        return ServiceError("Model provider refused or could not complete the request; check configuration",
                            service_id=service_id, code="model_provider_failed", recoverable=retryable)
    return None


@dataclass
class _Generation:
    spec: ServiceSpec
    number: int
    codec: ServiceCodec
    provider: LLMProvider
    client: httpx.AsyncClient
    semaphore: asyncio.Semaphore
    provider_revision: int
    revocation_epoch: int = 0
    references: int = 0
    admitted: int = 0
    revoked: str = ""
    closed: bool = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await self.provider.aclose()  # Managed contract verified at construction.
        finally:
            await self.client.aclose()


class ModelServiceRuntime:
    """One profile owner; catalog reads are cached and never perform network I/O.

    Call state lives on each handle, not in this shared owner. Configuration and
    provider registry changes affect future acquisition only, except disable/removal
    which irreversibly revoke existing admission and result delivery. Releasing the last
    handle closes the generation's clients; the owner closes all remaining handles
    at shutdown, after physical consumers have stopped.
    """

    def __init__(self, registry: LLMProviderRegistry, specs: tuple[ServiceSpec, ...] = (), *,
                 resolve_secret: SecretResolver | None = None,
                 client_factory: ClientFactory = build_client,
                 revision: object = None) -> None:
        self._registry = registry
        self._specs = {spec.service_id: spec for spec in specs}
        self._resolve_secret = resolve_secret
        self._client_factory = client_factory
        self._revision = revision
        self._registry_version = registry.version
        self._number = 1
        self._codecs: dict[str, ServiceCodec] = {"json.v1": JsonCodec()}
        self._generations: dict[tuple[str, int], _Generation] = {}
        # Persist revocation even after the last handle releases its generation.
        self._revocations: dict[str, tuple[int, str]] = {}
        self._handles: set[ModelServiceHandle] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> ModelServiceRuntime:
        """Compose a profile owner without endpoint I/O or credential materialization."""
        from leapflow.llm.provider_registry import get_default_registry, get_scoped_default_registry
        from leapflow.security.secrets import ScopedSecretResolver

        # Discover once at composition and adopt provider fibers. Never rediscover
        # from acquire(): a deliberately removed capability must stay unavailable.
        get_scoped_default_registry()
        resolver = ScopedSecretResolver(settings.layout, settings.profile_layout)
        return cls(get_default_registry(), settings.llm_services, resolve_secret=resolver.get,
                   revision=cls.secret_revision(settings))

    @staticmethod
    def secret_revision(settings: Any) -> object:
        from leapflow.config_loader import config_signature

        profile, shared = settings.profile_layout.secrets, settings.layout.global_secrets
        return config_signature((profile.vault_path, profile.key_path, shared.vault_path, shared.key_path))

    def register_codec(self, name: str, codec: ServiceCodec) -> Callable[[], None]:
        """Return undo for the registering plugin's existing EffectScope."""
        if self._closed or name in self._codecs or not isinstance(codec, ServiceCodec):
            raise ValueError("Codec registration requires a unique name and a live codec")
        self._codecs[name] = codec
        self._number += 1

        def undo() -> None:
            if self._codecs.get(name) is codec:
                del self._codecs[name]
                self._number += 1

        return undo

    def publish(self, specs: tuple[ServiceSpec, ...], *, revision: object = None) -> bool:
        """Atomically publish a previously validated immutable snapshot without probing."""
        if self._closed:
            raise RuntimeError("Model service owner is closed")
        updated = {spec.service_id: spec for spec in specs}
        if updated == self._specs and revision == self._revision:
            return False
        for name in self._specs:
            current = updated.get(name)
            if current is None or not current.enabled:
                epoch, _ = self._revocations.get(name, (0, ""))
                self._revocations[name] = (epoch + 1, "model_service_disabled" if current is not None
                                          else "model_capability_removed")
        for generation in self._generations.values():
            current = updated.get(generation.spec.service_id)
            if not generation.revoked and (current is None or not current.enabled):
                generation.revoked = "model_service_disabled" if current is not None else "model_capability_removed"
        self._specs, self._revision = updated, revision
        self._number += 1
        return True

    def _refresh_registry(self) -> None:
        if self._registry_version != self._registry.version:
            self._registry_version = self._registry.version
            self._number += 1

    def describe(self, service_id: str = "") -> dict[str, Any]:
        """Describe configuration availability, not live endpoint health or session usage."""
        self._refresh_registry()
        catalog = []
        for name, spec in sorted(self._specs.items()):
            if service_id and name != service_id:
                continue
            provider_available = self._registry.get_plugin(spec.provider) is not None
            available = not self._closed and spec.enabled and provider_available and spec.codec in self._codecs
            catalog.append({"id": name, "provider": spec.provider, "model": spec.model,
                            "task": spec.task, "codec": spec.codec, "identity_mode": spec.identity_mode,
                            "processing_location": spec.processing_location,
                            "configured": True, "enabled": spec.enabled,
                            "provider_available": provider_available, "available": available, "usable": available,
                            "unavailable_reason": "disabled" if not spec.enabled else ("" if available else "capability_unavailable"),
                            "generation": self._number,
                            "health": "not_probed"})
        return {"services": catalog, "generation": self._number}

    async def acquire(self, service_id: str) -> ModelServiceHandle:
        """Acquire a pinned provider/codec snapshot; construction does not call the endpoint."""
        self._refresh_registry()
        spec = self._specs.get(service_id)
        if self._closed or spec is None:
            raise ServiceError("Service unavailable; use config llm services show/set", service_id=service_id)
        if not spec.enabled:
            raise ServiceError("Service is disabled; use config llm services enable to admit new calls",
                               service_id=service_id, code="model_service_disabled")
        if self._registry.get_plugin(spec.provider) is None or spec.codec not in self._codecs:
            raise ServiceError("Provider or codec unavailable; install/enable its plugin", service_id=service_id)
        key = (service_id, self._number)
        generation = self._generations.get(key)
        if generation is None:
            generation = await self._construct(spec)
            # Construction has no awaits on success; acquisitions cannot interleave.
            self._generations[key] = generation
        generation.references += 1
        handle = ModelServiceHandle(self, generation)
        self._handles.add(handle)
        return handle

    async def _construct(self, spec: ServiceSpec) -> _Generation:
        api_key = "no-auth"
        if spec.auth_ref:
            api_key = self._resolve_secret(spec.auth_ref) if self._resolve_secret else None
            if not api_key:
                raise ServiceError("Service credential unavailable; repair its vault reference",
                                   service_id=spec.service_id, code="model_auth_unavailable")
        client = self._client_factory(spec)
        try:
            provider = self._registry.create_isolated(spec.provider, {
                "api_key": api_key, "base_url": spec.base_url, "model": spec.model,
                "max_retries": 1, "timeout_s": spec.limits.deadline_s,
                "max_tokens": spec.limits.max_output_tokens, "wire_profile": "generic",
                "async_http_client": client,
            })
            if not isinstance(provider, AsyncCloseableProvider):
                raise ServiceError("Provider does not support owned async clients", service_id=spec.service_id,
                                   code="model_provider_unsupported")
        except (ImportError, LookupError, ValueError) as exc:
            await client.aclose()
            raise ServiceError("Provider cannot be constructed; check SDK and declaration",
                               service_id=spec.service_id) from exc
        except Exception:
            await client.aclose()
            raise
        return _Generation(spec, self._number, self._codecs[spec.codec], provider, client,
                           asyncio.Semaphore(spec.limits.max_inflight),
                           self._registry.provider_revision(spec.provider),
                           self._revocations.get(spec.service_id, (0, ""))[0])

    async def _release(self, handle: ModelServiceHandle) -> None:
        self._handles.discard(handle)
        generation = handle._generation
        generation.references -= 1
        if generation.references == 0:
            self._generations.pop((generation.spec.service_id, generation.number), None)
            await generation.close()

    async def aclose(self) -> None:
        """Cancel outstanding advisory calls, then release every owned connection."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_handles())
        await asyncio.shield(self._close_task)

    async def _close_handles(self) -> None:
        results = await asyncio.gather(*(handle.aclose() for handle in tuple(self._handles)),
                                       return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("Model service cleanup failed", exc_info=(type(result), result, result.__traceback__))


class ModelServiceHandle:
    """A consumer's immutable admission snapshot and cancellation ownership."""

    def __init__(self, owner: ModelServiceRuntime, generation: _Generation) -> None:
        self._owner = owner
        self._generation = generation
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def spec(self) -> ServiceSpec:
        return self._generation.spec

    @property
    def generation(self) -> int:
        return self._generation.number

    async def invoke(self, request: Mapping[str, Any], schema: Mapping[str, Any], *,
                     validate: Validator | None = None) -> StructuredResult:
        start = time.monotonic()
        gen = self._generation
        if self._closed or gen.closed:
            raise ServiceError("Service handle closed; acquire a new snapshot", service_id=self.spec.service_id)
        self.check_admission()
        if gen.admitted >= self.spec.limits.max_inflight + self.spec.limits.max_queue:
            raise ServiceError("Service busy; wait for the admitted request to finish",
                               service_id=self.spec.service_id, code="model_busy", recoverable=True)
        gen.admitted += 1
        task = asyncio.create_task(self._invoke(request, schema, validate, start))
        self._tasks.add(task)
        try:
            result = await asyncio.wait_for(task, timeout=self.spec.limits.deadline_s)
            self.check_admission()
            return result
        except asyncio.TimeoutError as exc:
            raise ServiceError("Model deadline exhausted; check latency or increase the declared budget",
                               service_id=self.spec.service_id, code="model_timeout", recoverable=True) from exc
        finally:
            self._tasks.discard(task)
            gen.admitted -= 1

    def check_admission(self) -> None:
        """Refuse revoked snapshots, including after a later re-enable; never preempt CUDA."""
        if self._closed or self._generation.closed:
            raise ServiceError("Service handle closed; acquire a new snapshot", service_id=self.spec.service_id)
        self.check_result_eligibility()

    def check_result_eligibility(self) -> None:
        """Check delivery after owned cleanup, without reopening a closed handle."""
        epoch, reason = self._owner._revocations.get(self.spec.service_id, (0, ""))
        revoked = self._generation.revoked or (reason if epoch != self._generation.revocation_epoch else "")
        if revoked:
            raise ServiceError("Service admission was revoked; use config llm services show/enable and acquire a fresh handle",
                               service_id=self.spec.service_id, code=revoked)
        if self._owner._closed:
            raise ServiceError("Service owner closed; result delivery refused", service_id=self.spec.service_id)
        if (self._owner._registry.get_plugin(self.spec.provider) is None
                or self._owner._registry.provider_revision(self.spec.provider) != self._generation.provider_revision
                or self._owner._codecs.get(self.spec.codec) is not self._generation.codec):
            raise ServiceError("Pinned service capability was removed or replaced; stop and re-admit the consumer",
                               service_id=self.spec.service_id, code="model_capability_removed")

    async def _invoke(self, request: Mapping[str, Any], schema: Mapping[str, Any],
                      validate: Validator | None, start: float) -> StructuredResult:
        spec, gen = self.spec, self._generation
        async with gen.semaphore:
            self.check_admission()
            try:
                # Bound input before expensive provider shaping/decoding; worker results
                # are discarded on cancellation and can never become action chunks.
                if len(canonical(request).encode()) > spec.limits.max_request_bytes:
                    raise ValueError("request exceeds byte budget")
                messages, options = await asyncio.to_thread(gen.codec.encode, spec, request, schema)
            except ValueError as exc:
                raise ServiceError("Invalid or oversized model request", service_id=spec.service_id,
                                   code="model_invalid_request") from exc
            self.check_admission()
            response = await self._call(messages, options)
            self.check_admission()
            try:
                payload = await asyncio.to_thread(gen.codec.decode, spec, response, schema)
                self.check_admission()
                if validate is not None:
                    await asyncio.to_thread(validate, payload)
                    self.check_admission()
                encoded = canonical(payload)
                if len(encoded.encode()) > spec.limits.max_response_bytes:
                    raise ValueError("decoded result exceeds byte budget")
            except ValueError as exc:
                raise ServiceError("Invalid model result; check the task codec and serving adapter",
                                   service_id=spec.service_id, code="model_invalid_output") from exc
            self.check_admission()
            if time.monotonic() - start >= spec.limits.deadline_s:
                raise asyncio.TimeoutError()
            return StructuredResult(encoded, spec.service_id, gen.number, response.model or spec.model,
                                    spec.identity_mode, (time.monotonic() - start) * 1000,
                                    canonical(response.usage))

    async def _call(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> LLMChatResponse:
        for attempt in range(self.spec.limits.attempts):
            self.check_admission()
            try:
                response = await self._generation.provider.achat(messages, stream=False, **options)
            except Exception as exc:
                self.check_admission()
                failure = _provider_failure(exc, self.spec.service_id)
                if failure is None:
                    raise  # Local defects preserve their traceback and never retry.
                if not failure.failure.recoverable or attempt + 1 == self.spec.limits.attempts:
                    raise failure from exc
                await asyncio.sleep(min(0.1 * 2 ** min(attempt, 6), 1.0))
            else:
                self.check_admission()
                return response
        raise AssertionError("positive attempt budget exhausted without a result")

    async def aclose(self) -> None:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._owner._release(self)

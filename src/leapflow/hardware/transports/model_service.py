# Copyright (c) Alibaba, Inc. and its affiliates.
"""Model-service transport: binds the v1 model-serving contract over local HTTP.

A model server is admitted like any other device -- declared, opened, probed --
and implements ``ModelService`` for the skills that call it. The module is a
mechanism, not a model: the URL, the task, and the identity it must report all
arrive in the declaration, and both tasks share one request path.

Fail-closed at open. The server must answer ``/v1/health`` with a ``model_info``
whose ``task`` matches the declaration and, when the declaration pins one, the
same ``name``/``version``/``checksum``. A swapped checkpoint is a different
device, not a silent upgrade -- and it is caught while open too: every health
check re-verifies the identity (``refresh`` adopts a legitimate change so a
skill can compare it with what its plan was approved against), and every
inference response must echo exactly the identity last verified.

The URL must be loopback unless ``allow_remote: true``: camera frames are sent
to this service, and they must not leave the board by accident. Proxy
environment variables are ignored for the same reason.

Retries cover what is transient for a stateless inference -- connection
failures, timeouts, 409 (busy; the server is single-flight) and 503 (warming
up). A 422 means the request does not match the model and is never retried.
Nothing here commands hardware, so every write refuses with a provable NONE.
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from collections import deque
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.model_contract import (
    BINDING_KEYS,
    MODEL_TASKS,
    InferenceResult,
    ModelInfo,
    check_request,
    parse_response,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

INFO_CHANNEL = "model.info"
HEALTH_CHANNEL = "model.health"
LATENCY_CHANNEL = "model.latency"
_CHANNELS = frozenset({INFO_CHANNEL, HEALTH_CHANNEL, LATENCY_CHANNEL})
_RETRYABLE_STATUS = frozenset({409, 503})
_LATENCY_WINDOW = 64
_LOOPBACK_NAMES = frozenset({"localhost"})


class ModelServiceTransport:
    """One locally served model, verified at open and called per task."""

    kind = "model_service"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._url = str(config.get("url") or "").strip().rstrip("/")
        self._task = str(config.get("task") or "").strip()
        self._allow_remote = bool(config.get("allow_remote", False))
        self._timeout_s = _number(config, "timeout_s", 5.0)
        self._max_retries = int(_number(config, "max_retries", 1))
        self._retry_backoff_s = _number(config, "retry_backoff_s", 0.2)
        expect = config.get("expect")
        self._expect: dict[str, Any] = dict(expect) if isinstance(expect, Mapping) else {}
        self._context: HardwareContext | None = None
        self._client: httpx.AsyncClient | None = None
        self._info: ModelInfo | None = None
        self._infer_lock = asyncio.Lock()
        self._sequence = 0
        self._latencies: deque[tuple[float, float]] = deque(maxlen=_LATENCY_WINDOW)
        self._failures = 0

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Verify the server serves the declared model; idempotent."""
        if self._client is not None and self._info is not None:
            return await self.probe()
        self._context = context
        problem = self._config_problem()
        if problem:
            raise TransportError(problem, failure_code="model_service_misconfigured")
        self._client = httpx.AsyncClient(
            base_url=self._url, timeout=self._timeout_s, trust_env=False
        )
        try:
            self._info = self._verify(await self._call("GET", "/v1/health"))
        except BaseException:
            await self.close()
            raise
        return await self.probe()

    async def close(self) -> TransportStatus:
        client, self._client = self._client, None
        self._info = None
        if client is not None:
            try:
                await client.aclose()
            except (httpx.HTTPError, RuntimeError):
                # Closing must be idempotent even when the server already went away.
                pass
        return TransportStatus(connected=False, halt_supported=False, detail="closed")

    async def probe(self) -> TransportStatus:
        """Report the bound model and recent latency; touches the server only via health."""
        connected = self._client is not None and self._info is not None
        detail = "model service is closed"
        if connected:
            healthy, detail = await self._health()
            connected = healthy
        return TransportStatus(
            connected=connected,
            halt_supported=False,
            detail=detail,
            latency_ms=self._latency_stats().get("last_ms", 0.0),
            metadata={
                "url": self._url,
                "task": self._task,
                "model_info": self._info.to_dict() if self._info else {},
                "latency": self._latency_stats(),
            },
        )

    async def halt(self) -> TransportStatus:
        """A model has no physical effect to stop; answering honestly keeps admission sound."""
        return TransportStatus(
            connected=self._client is not None,
            halt_supported=False,
            detail="a model service commands nothing and has no emergency stop",
        )

    # ── Data plane ──

    async def read(self, channel_id: str) -> Reading:
        info = self._require_open(channel_id)
        if channel_id not in _CHANNELS:
            raise TransportError(f"unknown channel {channel_id!r}", failure_code="unknown_channel")
        quality = Quality.OK.value
        if channel_id == INFO_CHANNEL:
            value: Any = info.to_dict()
        elif channel_id == HEALTH_CHANNEL:
            healthy, detail = await self._health()
            value = {"ok": healthy, "detail": detail}
            quality = Quality.OK.value if healthy else Quality.SUSPECT.value
        else:
            value = self._latency_stats()
        self._sequence += 1
        return Reading(
            device_id=self._device_id,
            channel_id=channel_id,
            value=value,
            quantity="model_metadata",
            sequence=self._sequence,
            quality=quality,
        )

    async def write(self, channel_id: str, value: Any) -> WriteOutcome:
        """Refuse every write; the call never reaches the server, so NONE is provable."""
        return WriteOutcome(
            ok=False,
            side_effect_state=SIDE_EFFECT_NONE,
            error=f"model service channel {channel_id!r} is read-only",
            failure_code="not_writable",
        )

    # ── ModelService ──

    @property
    def model_info(self) -> ModelInfo:
        return self._require_open("model_info")

    async def refresh(self) -> ModelInfo:
        """Re-read ``/v1/health`` and adopt the identity it verifies (pins still apply)."""
        self._require_open("refresh")
        self._info = self._verify(await self._call("GET", "/v1/health", retries=0))
        return self._info

    async def infer(self, request: Mapping[str, Any]) -> InferenceResult:
        """Run one inference for the declared task, serialised per transport."""
        info = self._require_open("infer")
        try:
            body = check_request(self._task, request)
        except ValueError as exc:
            raise TransportError(
                f"invalid {self._task} request: {exc}", failure_code="model_bad_request"
            ) from exc
        async with self._infer_lock:
            started = time.monotonic()
            payload = await self._call("POST", "/v1/infer", json={"task": self._task, **body})
            round_trip_ms = (time.monotonic() - started) * 1000.0
        try:
            result = parse_response(self._task, payload)
        except ValueError as exc:
            raise TransportError(
                f"model service returned an invalid {self._task} response: {exc}",
                failure_code="model_bad_response",
            ) from exc
        changed = info.identity.difference(result.model)
        if changed:
            raise TransportError(
                f"the answering model is not the verified one: {changed}",
                failure_code="model_identity_changed",
            )
        self._latencies.append((round_trip_ms, result.infer_ms))
        return result

    # ── Internals ──

    def _config_problem(self) -> str:
        if self._task not in MODEL_TASKS:
            return f"model_service task must be one of {sorted(MODEL_TASKS)}, got {self._task!r}"
        parts = urlsplit(self._url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return f"model_service url must be http(s)://host:port, got {self._url!r}"
        if not self._allow_remote and not _is_loopback(parts.hostname):
            return (
                f"model_service url {self._url!r} is not loopback; camera frames would leave "
                "the node (set allow_remote: true only for a deliberately remote server)"
            )
        unknown = set(self._expect) - set(BINDING_KEYS)
        if unknown:
            return f"model_service expect accepts {list(BINDING_KEYS)}, got {sorted(unknown)}"
        if self._timeout_s <= 0 or self._max_retries < 0 or self._retry_backoff_s < 0:
            return "model_service needs timeout_s > 0, max_retries >= 0, retry_backoff_s >= 0"
        return ""

    def _verify(self, health: Mapping[str, Any]) -> ModelInfo:
        try:
            info = ModelInfo.from_mapping(health.get("model_info"))
        except ValueError as exc:
            raise TransportError(
                f"model service health has no usable model_info: {exc}",
                failure_code="model_bad_response",
            ) from exc
        if info.task != self._task:
            raise TransportError(
                f"model service serves task {info.task!r}, declaration expects {self._task!r}",
                failure_code="model_task_mismatch",
            )
        problem = info.binding_problem(self._expect)
        if problem:
            raise TransportError(
                f"model service is not the pinned model: {problem}",
                failure_code="model_binding_mismatch",
            )
        return info

    async def _health(self) -> tuple[bool, str]:
        """Reachable *and* still serving the verified identity (a swap is unhealthy)."""
        try:
            current = self._verify(await self._call("GET", "/v1/health", retries=0))
        except TransportError as exc:
            return False, f"{exc.failure_code}: {exc}"
        if self._info is not None:
            changed = self._info.identity.difference(current.identity)
            if changed:
                return False, f"model_identity_changed: {changed}"
        return True, f"serving {current.name} {current.version}"

    async def _call(
        self, method: str, path: str, *, json: Any = None, retries: int | None = None
    ) -> dict[str, Any]:
        """One request with bounded retries on transient failures."""
        client = self._client
        if client is None:
            raise TransportError("model service is not open", failure_code="transport_not_open")
        budget = self._max_retries if retries is None else retries
        attempt = 0
        while True:
            outcome = await self._attempt(client, method, path, json)
            if isinstance(outcome, dict):
                return outcome
            if not isinstance(outcome, _TransientError) or attempt >= budget:
                self._failures += 1
                raise outcome
            attempt += 1
            await asyncio.sleep(self._retry_backoff_s * attempt)

    async def _attempt(
        self, client: httpx.AsyncClient, method: str, path: str, json: Any
    ) -> dict[str, Any] | TransportError:
        try:
            response = await client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            return _TransientError(
                f"model service timed out after {self._timeout_s}s: {exc!r}",
                failure_code="model_timeout",
            )
        except httpx.TransportError as exc:
            return _TransientError(
                f"model service at {self._url} is unreachable: {exc!r}",
                failure_code="model_unreachable",
            )
        if response.status_code in _RETRYABLE_STATUS:
            return _TransientError(
                f"model service answered {response.status_code}: {_detail(response)}",
                failure_code="model_busy" if response.status_code == 409 else "model_unavailable",
            )
        if response.status_code == 422:
            return TransportError(
                f"request does not match the model: {_detail(response)}",
                failure_code="model_schema_mismatch",
            )
        if response.status_code >= 400:
            return TransportError(
                f"model service failed with {response.status_code}: {_detail(response)}",
                failure_code="model_error",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            return TransportError(
                f"model service returned non-JSON: {exc}", failure_code="model_bad_response"
            )
        if not isinstance(payload, dict):
            return TransportError(
                "model service returned a non-object body", failure_code="model_bad_response"
            )
        return payload

    def _latency_stats(self) -> dict[str, Any]:
        if not self._latencies:
            return {"count": 0, "failures": self._failures}
        round_trips = sorted(sample[0] for sample in self._latencies)
        infers = sorted(sample[1] for sample in self._latencies)
        return {
            "count": len(round_trips),
            "failures": self._failures,
            "last_ms": round(self._latencies[-1][0], 2),
            "p50_ms": round(_percentile(round_trips, 0.5), 2),
            "p95_ms": round(_percentile(round_trips, 0.95), 2),
            "infer_p50_ms": round(_percentile(infers, 0.5), 2),
        }

    def _require_open(self, what: str) -> ModelInfo:
        if self._client is None or self._info is None:
            raise TransportError(
                f"model service is not open ({what})", failure_code="transport_not_open"
            )
        return self._info

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context is not None else ""


class _TransientError(TransportError):
    """A failure a stateless inference may retry: unreachable, timed out, busy, warming."""


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, Mapping):
        return str(body.get("error") or body.get("detail") or body)[:200]
    return str(body)[:200]


def _is_loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _number(config: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric setting where 0 is a legitimate value, not "unset"."""
    value = config.get(key)
    return default if value is None else float(value)


def _percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def build_transport(config: Mapping[str, Any] | None = None) -> ModelServiceTransport:
    """Factory registered in the transport table."""
    return ModelServiceTransport(config)


__all__ = [
    "HEALTH_CHANNEL",
    "INFO_CHANNEL",
    "LATENCY_CHANNEL",
    "ModelServiceTransport",
    "build_transport",
]

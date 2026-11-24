# Copyright (c) Alibaba, Inc. and its affiliates.
"""Model-service transport: v1 contract over HTTP or a qualified injected provider.

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
import time
from collections import deque
from typing import Any, Mapping

from leapflow.hardware.model_qualification import QualificationVerifier
from leapflow.llm.services.protocol import StructuredModelClient

from leapflow.hardware.context import HardwareContext, Quality
from leapflow.hardware.model_contract import (
    BINDING_KEYS,
    MODEL_TASKS,
    InferenceResult,
    ModelInfo,
    ModelBackend,
    check_request,
    parse_response,
)
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    STOP_EFFECT_NONE,
    Reading,
    TransportError,
    TransportStatus,
    WriteOutcome,
)

INFO_CHANNEL = "model.info"
HEALTH_CHANNEL = "model.health"
LATENCY_CHANNEL = "model.latency"
_CHANNELS = frozenset({INFO_CHANNEL, HEALTH_CHANNEL, LATENCY_CHANNEL})
_LATENCY_WINDOW = 64


class ModelServiceTransport:
    """One locally served model, verified at open and called per task."""

    kind = "model_service"

    def __init__(self, config: Mapping[str, Any] | None = None, *,
                 model_services: StructuredModelClient | None = None,
                 qualification: QualificationVerifier | None = None) -> None:
        config = dict(config or {})
        self._config = config
        self._url = str(config.get("url") or config.get("service") or "").strip().rstrip("/")
        self._task = str(config.get("task") or "").strip()
        self._backend_name = config.get("backend", "http")
        self._model_services = model_services
        self._qualification = qualification
        expect = config.get("expect")
        self._expect: dict[str, Any] = dict(expect) if isinstance(expect, Mapping) else {}
        self._context: HardwareContext | None = None
        self._backend: ModelBackend | None = None
        self._info: ModelInfo | None = None
        self._infer_lock = asyncio.Lock()
        self._sequence = 0
        self._latencies: deque[tuple[float, float]] = deque(maxlen=_LATENCY_WINDOW)
        self._failures = 0

    # ── Lifecycle ──

    async def open(self, context: HardwareContext) -> TransportStatus:
        """Verify the server serves the declared model; idempotent."""
        if self._backend is not None and self._info is not None:
            return await self.probe()
        self._context = context
        problem = self._config_problem()
        if problem:
            raise TransportError(problem, failure_code="model_service_misconfigured")
        from leapflow.hardware.transports._model_http import HttpModelBackend
        from leapflow.hardware.transports._model_provider import ProviderModelBackend

        try:
            self._backend = (HttpModelBackend(self._config) if self._backend_name == "http" else
                             ProviderModelBackend(self._config, self._model_services, self._qualification))
        except (ValueError, TypeError) as exc:
            raise TransportError("Invalid model backend configuration", failure_code="model_service_misconfigured") from exc
        try:
            await self._backend.open()
            self._info = self._verify(await self._backend.describe())
        except BaseException:
            await self.close()
            raise
        return await self.probe()

    async def close(self) -> TransportStatus:
        backend, self._backend = self._backend, None
        self._info = None
        if backend is not None:
            await backend.close()
        return TransportStatus(connected=False, halt_supported=False, detail="closed")

    async def probe(self) -> TransportStatus:
        """Report the bound model and recent latency; touches the server only via health."""
        connected = self._backend is not None and self._info is not None
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
            connected=self._backend is not None,
            halt_supported=False,
            detail="a model service commands nothing and has no emergency stop",
            stop_effect=STOP_EFFECT_NONE,
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
            sequence_scope="device_read",
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
        self._info = self._verify(await self._backend.describe())
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
        # Hardware consumers never build an implicit queue outside the provider's
        # bounded admission. A busy physical model must stop/refuse, not wait on
        # an observation whose freshness is no longer known.
        if self._backend_name == "provider" and self._infer_lock.locked():
            raise TransportError("Provider model busy; no hardware inference queue", failure_code="model_busy")
        async with self._infer_lock:
            started = time.monotonic()
            try:
                payload = await self._backend.infer({"task": self._task, **body})
            except TransportError:
                self._failures += 1
                raise
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
        if not isinstance(self._backend_name, str) or self._backend_name not in {"http", "provider"}:
            return "model_service backend must be http or provider"
        unknown = set(self._expect) - set(BINDING_KEYS)
        if unknown:
            return f"model_service expect accepts {list(BINDING_KEYS)}, got {sorted(unknown)}"
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
            current = self._verify(await self._backend.describe())
        except TransportError as exc:
            return False, f"{exc.failure_code}: {exc}"
        if self._info is not None:
            changed = self._info.identity.difference(current.identity)
            if changed:
                return False, f"model_identity_changed: {changed}"
        return True, f"serving {current.name} {current.version}"

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
        if self._backend is None or self._info is None:
            raise TransportError(
                f"model service is not open ({what})", failure_code="transport_not_open"
            )
        return self._info

    @property
    def _device_id(self) -> str:
        return self._context.device_id if self._context is not None else ""


def _percentile(ordered: list[float], fraction: float) -> float:
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def build_transport(config: Mapping[str, Any] | None = None, *,
                    model_services: StructuredModelClient | None = None,
                    qualification: QualificationVerifier | None = None) -> ModelServiceTransport:
    """Default table factory; composition roots may bind dependencies per registry."""
    return ModelServiceTransport(config, model_services=model_services, qualification=qualification)


__all__ = [
    "HEALTH_CHANNEL",
    "INFO_CHANNEL",
    "LATENCY_CHANNEL",
    "ModelServiceTransport",
    "build_transport",
]

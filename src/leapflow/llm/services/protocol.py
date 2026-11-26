# Copyright (c) Alibaba, Inc. and its affiliates.
"""Immutable structured-inference values and lifecycle ports, independent of domains."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from leapflow.llm.services.contracts import RegisteredContract
    from leapflow.llm.services.telemetry import InferenceAttempt

from leapflow.domain.model_service import ServiceLimits, ServiceMedia, ServiceSpec  # noqa: F401
from leapflow.llm.base import LLMChatResponse


def canonical(value: Any) -> str:
    """Freeze JSON data without accepting NaN or implementation-specific objects."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ServiceFailure:
    source: str
    code: str
    category: str
    recoverable: bool = False
    side_effect_state: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "code": self.code, "category": self.category,
                "recoverable": self.recoverable, "side_effect_state": self.side_effect_state}


class ServiceError(Exception):
    """Expected invocation refusal; exceptions must remain mutable for tracebacks."""

    def __init__(self, message: str, *, service_id: str = "", code: str = "model_unavailable",
                 category: str = "capability_unavailable", recoverable: bool = False) -> None:
        super().__init__(message)
        self.failure = ServiceFailure(service_id, code, category, recoverable)


@dataclass(frozen=True)
class StructuredResult:
    payload_json: str
    service_id: str
    generation: int
    api_model: str | None
    identity_mode: str
    elapsed_ms: float
    usage_json: str = "{}"
    call_id: str = ""
    attempts: tuple[InferenceAttempt, ...] = ()
    timing_json: str = "{}"

    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)


Validator = Callable[[Mapping[str, Any]], Any]
_Result = TypeVar("_Result")


class ActiveCallBudget:
    """One monotonic work budget across shaping, resolution, transport and validation.

    Human approval waits deliberately occur outside run(): consent never expires
    because an inference deadline elapsed. Cancellation discards worker results.
    """

    def __init__(self, seconds: float) -> None:
        self.remaining = seconds

    async def run(self, operation: Awaitable[_Result]) -> _Result:
        started = time.monotonic()
        try:
            return await asyncio.wait_for(operation, timeout=max(0.0, self.remaining))
        finally:
            self.remaining -= time.monotonic() - started


@runtime_checkable
class ServiceCodec(Protocol):
    def encode(self, spec: ServiceSpec, request: Mapping[str, Any],
               schema: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...
    def decode(self, spec: ServiceSpec, response: LLMChatResponse,
               schema: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class ServiceHandle(Protocol):
    @property
    def spec(self) -> ServiceSpec: ...
    @property
    def generation(self) -> int: ...
    @property
    def contract(self) -> RegisteredContract: ...
    @property
    def codec_revision(self) -> int: ...
    def check_admission(self) -> None: ...
    def check_result_eligibility(self) -> None: ...
    async def invoke(self, request: Mapping[str, Any], schema: Mapping[str, Any],
                     *, validate: Validator | None = None) -> StructuredResult: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class StructuredModelClient(Protocol):
    async def acquire(self, service_id: str) -> ServiceHandle: ...
    def describe(self, service_id: str = "") -> dict[str, Any]: ...
    async def aclose(self) -> None: ...

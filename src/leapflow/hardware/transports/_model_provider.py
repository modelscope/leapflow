# Copyright (c) Alibaba, Inc. and its affiliates.
"""Qualified domain envelopes over an injected, generation-pinned standard provider."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from typing import Any, Mapping

from leapflow.hardware.model_contract import (
    ModelInfo, PROVIDER_CONTRACT, PROVIDER_DESCRIBE_SCHEMA,
    parse_response, provider_request, provider_result_schema,
)
from leapflow.hardware.model_qualification import QualificationVerifier, native_metadata
from leapflow.hardware.transport import TransportError
from leapflow.llm.services.protocol import ServiceError, ServiceHandle, StructuredModelClient

class ProviderModelBackend:
    def __init__(self, config: Mapping[str, Any], client: StructuredModelClient | None,
                 qualification: QualificationVerifier | None) -> None:
        self._config = dict(config)
        self._client = client
        self._qualification = qualification
        self._handle: ServiceHandle | None = None
        self._info: ModelInfo | None = None

    async def open(self) -> None:
        if self._client is None or self._qualification is None:
            raise TransportError("Provider model requires its owning service and managed qualification verifier",
                                 failure_code="model_service_misconfigured")
        if not self._config.get("service"):
            raise TransportError("Provider model requires service ID", failure_code="model_service_misconfigured")
        try:
            handle = await self._client.acquire(self._config["service"])
        except ServiceError as exc:
            raise TransportError(str(exc), failure_code=exc.failure.code) from exc
        self._handle = handle
        if handle.spec.task != self._config.get("task"):
            raise TransportError("Provider service task does not match device", failure_code="model_task_mismatch")
        if handle.spec.identity_mode != "server_reported" or handle.spec.processing_location == "unknown":
            raise TransportError("Stock chat labels cannot qualify a hardware model",
                                 failure_code="model_unqualified")
        if handle.spec.limits.attempts != 1:
            raise TransportError("Hardware provider calls require one attempt; no transparent replay",
                                 failure_code="model_service_misconfigured")
        self._info = await self._bounded(self._describe())

    async def close(self) -> None:
        handle, self._handle = self._handle, None
        self._info = None
        if handle is not None:
            await handle.aclose()

    async def describe(self) -> Mapping[str, Any]:
        info = await self._bounded(self._describe())
        return {"model_info": info.to_dict()}

    async def _describe(self) -> ModelInfo:
        handle = self._require_handle()
        result = await handle.invoke({"contract": PROVIDER_CONTRACT, "operation": "describe", "task": handle.spec.task},
                                     PROVIDER_DESCRIBE_SCHEMA)
        info = ModelInfo.from_mapping(result.payload()["model_info"])
        if info.task != handle.spec.task:
            raise ValueError("Serving task differs from admitted task")
        if self._info is not None:
            if self._info.identity.difference(info.identity) or native_metadata(self._info) != native_metadata(info):
                raise TransportError("Serving instance or native metadata changed; replan and requalify",
                                     failure_code="model_identity_changed")
        provenance = await self._qualification.verify(handle.spec, info)
        basis = {**provenance, "generation": handle.generation, "codec": handle.spec.codec,
                 "service_id": handle.spec.service_id}
        return replace(info, details={**info.details, "service_basis": basis})

    async def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._bounded(self._infer(request))

    async def _infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        handle = self._require_handle()
        # Recheck runtime metadata and revocable qualification before every chunk.
        info = await self._describe()
        body = dict(request)
        if handle.spec.task != "action_chunk":
            body.pop("task", None)
        images = body.pop("images", None)
        single = body.pop("image", None)
        if single is not None:
            images = {"image": single}
        if not isinstance(images, dict) or not images or any(not isinstance(k, str) for k in images):
            raise ValueError("Inference requires named image blocks")
        media = {}
        for label, data in images.items():
            if not isinstance(data, str) or not base64.b64decode(data, validate=True).startswith(b"\xff\xd8"):
                raise ValueError("Hardware image blocks must contain the declared JPEG data")
            media[label] = {"media_type": "image/jpeg", "data": data}
        envelope = provider_request(handle.spec.task, body, media)
        result = await handle.invoke(envelope, provider_result_schema(handle.spec.task),
                                     validate=lambda value: parse_response(handle.spec.task, value))
        payload = result.payload()
        parsed = parse_response(handle.spec.task, payload)
        if info.identity.difference(parsed.model):
            raise TransportError("Answering model differs from qualified serving instance",
                                 failure_code="model_identity_changed")
        # Qualification may expire or be revoked while the GPU call is in flight.
        # Reuse the managed verifier, without another provider request, before any
        # result can reach policy execution. This work shares the total deadline.
        await self._qualification.verify(handle.spec, info)
        return payload

    async def _bounded(self, operation: Any) -> Any:
        handle = self._require_handle()
        try:
            result = await asyncio.wait_for(operation, handle.spec.limits.deadline_s)
            # Qualification and task delivery await after provider completion too.
            handle.check_result_eligibility()
            return result
        except asyncio.TimeoutError as exc:
            raise TransportError("Provider model total deadline exhausted", failure_code="model_timeout") from exc
        except ServiceError as exc:
            raise TransportError(str(exc), failure_code=exc.failure.code) from exc
        except (ValueError, OSError) as exc:
            raise TransportError(str(exc), failure_code="model_unqualified") from exc

    def _require_handle(self) -> ServiceHandle:
        if self._handle is None:
            raise TransportError("Provider model not open", failure_code="transport_not_open")
        return self._handle

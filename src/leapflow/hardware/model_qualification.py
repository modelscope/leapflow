# Copyright (c) Alibaba, Inc. and its affiliates.
"""Verify deployment provenance against existing managed evidence and approval stores.

Receipts are not run consent, trust promotion, or GPU attestation. Their production
creation requires the separate operator-owned native/deployment qualification lane.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit

from leapflow.domain.model_service import ServiceSpec
from leapflow.hardware.model_contract import ModelInfo
from leapflow.security.actions import ActionDescriptor, ActionKind


def digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode()).hexdigest()


def native_metadata(info: ModelInfo) -> dict[str, Any]:
    metadata = info.to_dict()
    metadata.pop("instance", None)
    metadata.pop("service_basis", None)
    return metadata


def qualification_binding(spec: ServiceSpec, info: ModelInfo) -> dict[str, Any]:
    """Bind exact endpoint, weights, native semantics, adapter and serving declaration."""
    declaration = spec.declaration()
    deployment = declaration.get("deployment", {})
    # A receipt cannot hash its own reference. Everything else in the immutable
    # declaration and effective defaults is part of the operator-approved basis.
    declared_basis = {**declaration, "deployment": {k: v for k, v in deployment.items()
                                                  if k != "qualification_ref"}}
    effective = asdict(spec)
    effective.pop("declaration_json")
    url = urlsplit(spec.base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    return {"schema_version": 1, "task": spec.task, "provider": spec.provider,
            "api_model": spec.model, "endpoint_origin": f"{url.scheme}://{url.hostname}:{port}",
            "base_url": spec.base_url, "checkpoint_checksum": info.checksum,
            "native_metadata_digest": digest(native_metadata(info)), "codec": spec.codec,
            "service_spec_digest": digest({"effective": effective, "declaration": declared_basis}),
            "adapter_digest": deployment.get("adapter_digest"),
            "engine_version": deployment.get("engine_version"),
            "processing_location": spec.processing_location}


def deployment_action(binding: Mapping[str, Any]) -> ActionDescriptor:
    """Existing configure action, scoped to the binding digest, never to a chat label."""
    return ActionDescriptor.device(kind=ActionKind.DEVICE_CONFIGURE.value,
                                   device_id="model_qualification", channel_id=digest(binding),
                                   metadata={"platform": "model_services", "reversible": False})


@runtime_checkable
class QualificationVerifier(Protocol):
    async def verify(self, spec: ServiceSpec, info: ModelInfo) -> Mapping[str, Any]: ...


class ManagedModelQualification:
    """Cold-path receipt validation; never accepts evidence or approval content from YAML."""

    def __init__(self, evidence_store: Callable[[], Any], audit: Any) -> None:
        self._evidence_store = evidence_store
        self._audit = audit

    async def verify(self, spec: ServiceSpec, info: ModelInfo) -> Mapping[str, Any]:
        if spec.identity_mode != "server_reported" or spec.processing_location == "unknown":
            raise ValueError("Hardware requires server-owned identity and qualified processing locality")
        declared = ModelInfo.from_mapping(spec.declaration().get("model_info"))
        if not info.checksum or not info.instance or native_metadata(info) != native_metadata(declared):
            raise ValueError("Loaded native metadata does not match the pinned manifest")
        if info.task == "action_chunk":
            from leapflow.hardware.skills.adapt.spec import ModelSpaces
            ModelSpaces.from_details(info.details)
        binding = qualification_binding(spec, info)
        receipt_id = spec.declaration()["deployment"]["qualification_ref"]
        store = await asyncio.to_thread(self._evidence_store)
        if store is None:
            raise ValueError("Managed qualification evidence unavailable")
        record = await store.get(receipt_id)
        if record is None:
            raise ValueError("Managed qualification receipt missing or ambiguous")
        if record.get("device_id") != "model_qualification" or record.get("channel_id") != spec.service_id:
            raise ValueError("Qualification receipt belongs to a different evidence source or service")
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        receipt = metadata.get("model_qualification") if isinstance(metadata, dict) else None
        if not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
            raise ValueError("Unsupported model qualification receipt")
        expiry = receipt.get("expires_at")
        if (isinstance(expiry, bool) or not isinstance(expiry, (int, float))
                or not math.isfinite(expiry) or expiry <= time.time() or receipt.get("revoked") is not False):
            raise ValueError("Qualification receipt expired, revoked or not explicitly valid")
        if (receipt.get("binding") != binding or receipt.get("identity_insertion") != "runtime"
                or receipt.get("native_parity") is not True):
            raise ValueError("Qualification evidence does not bind this native deployment")
        approval = await asyncio.to_thread(self._audit.get, receipt.get("approval_action_id", ""))
        action = deployment_action(binding)
        if (not approval or approval.get("decision") != "allow" or approval.get("actor") != "user"
                or approval.get("action_kind") != action.kind or approval.get("effect") != action.effect
                or approval.get("resource") != action.resource):
            raise ValueError("Deployment lacks a binding-specific operator approval record")
        if expiry <= time.time():
            raise ValueError("Qualification receipt expired during verification")
        return {"qualification_ref": receipt_id, "binding_digest": digest(binding),
                "service_spec_digest": binding["service_spec_digest"],
                "native_metadata_digest": binding["native_metadata_digest"],
                "adapter_digest": binding["adapter_digest"], "engine_version": binding["engine_version"]}

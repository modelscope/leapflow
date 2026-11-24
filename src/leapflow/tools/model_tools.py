# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed read-only collaboration with configured specialist services."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import asdict, replace
from functools import partial
from typing import Any, Callable, Mapping

from leapflow.llm.services.codec import validate_schema
from leapflow.llm.services.protocol import ActiveCallBudget, ServiceError, ServiceSpec, StructuredModelClient, canonical
from leapflow.security.actions import ActionDescriptor
from leapflow.security.network import classify_url
from leapflow.tools.execution_context import current_tool_context
from leapflow.world_model.specialist_contract import (
    INPUT_SCHEMAS, OUTPUT_SCHEMAS, array, eligible_ids, obj, text, validate_result,
)

logger = logging.getLogger(__name__)

GEOMETRY_INPUTS = {
    "grounding": obj({"queries": array(text(256), minimum=1, maximum=16)}, ("queries",)),
    "grasp_proposal": obj({"depth": obj({"scale_m": {"type": "number", "minimum": 0.000000001}},
                                        ("scale_m",)),
                           "intrinsics": obj({
                               "width": {"type": "integer", "minimum": 1},
                               "height": {"type": "integer", "minimum": 1},
                               "fx": {"type": "number", "minimum": 0.000000001},
                               "fy": {"type": "number", "minimum": 0.000000001},
                               "ppx": {"type": "number"}, "ppy": {"type": "number"},
                               "model": text(128), "coeffs": array({"type": "number"}, maximum=5),
                           }, ("width", "height", "fx", "fy", "ppx", "ppy")),
                           "object": obj({"label": text(256),
                               "center": array({"type": "number"}, minimum=3, maximum=3),
                               "pixel": array({"type": "number", "minimum": 0}, minimum=2, maximum=2),
                               "size": obj({key: {"type": "number", "minimum": 0} for key in
                                            ("length", "width", "height")}, ("length", "width", "height")),
                           }, ("center",)),
                           "gripper": obj({"max_width_mm": {"type": "number", "minimum": 0.000000001}},
                                          ("max_width_mm",)),
                           "top_k": {"type": "integer", "minimum": 1, "maximum": 64}},
                          ("depth", "intrinsics", "object", "gripper", "top_k")),
}
GEOMETRY_OUTPUTS = {
    "grounding": obj({"detections": array({"type": "object"}, maximum=256)}, ("detections",)),
    "grasp_proposal": obj({"grasps": array({"type": "object"}, maximum=64)}, ("grasps",)),
}


def task_schemas(task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if task in INPUT_SCHEMAS:
        return INPUT_SCHEMAS[task], OUTPUT_SCHEMAS[task]
    if task in GEOMETRY_INPUTS:
        return GEOMETRY_INPUTS[task], GEOMETRY_OUTPUTS[task]
    raise ValueError("action_chunk is not a read-only query; use admitted hw_skill_* / policy.run")


def _validate_geometry(task: str, request: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    from leapflow.hardware.model_contract import Detection, ProposedGrasp

    parser, field = (Detection, "detections") if task == "grounding" else (ProposedGrasp, "grasps")
    for item in result[field]:
        parser.from_mapping(item)


async def authorize(action: ActionDescriptor, gate: Any = None) -> None:
    """Use the active turn's orchestrator; no check-only fallback or shared session."""
    context = current_tool_context()
    active = getattr(context, "orchestrator", None) or gate
    if not callable(getattr(active, "evaluate", None)):
        raise ServiceError("Approval unavailable; retry from an interactive governed session",
                           code="permission_denied", category="security")
    if context is not None:
        action = replace(action, session_id=context.session_id, turn_id=context.task_id)
    try:
        decision = await active.evaluate(action)
    except Exception as exc:
        logger.warning("Model service approval evaluation failed", exc_info=True)
        raise ServiceError("Approval could not be consulted; request refused",
                           code="permission_denied", category="security") from exc
    if not getattr(decision, "approved", False):
        raise ServiceError(str(getattr(decision, "denial_message", "") or "Request denied by approval policy"),
                           code="permission_denied", category="security")


def failure_payload(exc: ServiceError) -> dict[str, Any]:
    failure = exc.failure
    return {"ok": False, "error": str(exc), "failure": asdict(failure),
            "failure_code": failure.code,
            "failure_class": "authorization" if failure.category == "security" else failure.category,
            "retryable": failure.recoverable, "side_effect_state": failure.side_effect_state}


async def authorize_endpoint(spec: ServiceSpec, gate: Any, *, sensitive: bool,
                             budget: ActiveCallBudget | None = None,
                             check_admission: Callable[[], None] | None = None) -> None:
    if check_admission is not None:
        check_admission()
    if sensitive and spec.processing_location == "unknown":
        raise ServiceError("Processing locality is unknown; qualify the deployment first",
                           code="model_privacy_unknown", category="security")
    try:
        target = await (budget or ActiveCallBudget(spec.limits.deadline_s)).run(classify_url(spec.base_url))
    except asyncio.TimeoutError as exc:
        raise ServiceError("Endpoint resolution timed out; check the configured host",
                           code="model_timeout", recoverable=True) from exc
    if check_admission is not None:
        check_admission()
    await authorize(ActionDescriptor.network_fetch(
        spec.base_url, origin=target.origin, method="POST",
        metadata={**target.to_metadata(), "platform": "model_services", "service_id": spec.service_id,
                  "processing_location": spec.processing_location, "sensitive_payload": sensitive}), gate)
    if check_admission is not None:
        check_admission()


def validate_depth(request: Mapping[str, Any], content: bytes) -> None:
    """Depth is aligned uint16 little-endian, never an opaque or guessed tensor."""
    from leapflow.hardware.rgbd import CameraIntrinsics

    intrinsics = CameraIntrinsics(**request["intrinsics"])
    if len(content) != intrinsics.width * intrinsics.height * 2:
        raise ValueError("Depth bytes do not match the declared aligned image dimensions")


def _media_type(content: bytes) -> str:
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("artifact is not an admitted JPEG, PNG or WebP")


class ModelTools:
    """Dependency-bound tool behavior; each invocation owns only its own handle."""

    def __init__(self) -> None:
        self.client: StructuredModelClient | None = None
        self.settings: Any = None
        self.gate: Any = None
        self._calls: set[asyncio.Task[Any]] = set()

    async def describe(self, service_id: str = "", **_: Any) -> dict[str, Any]:
        if self.client is None:
            return {"ok": False, "error": "Model services unavailable", "retryable": False}
        catalog = self.client.describe(service_id)
        if service_id and not catalog["services"]:
            return {"ok": False, "error": "Unknown service ID; use model_describe without an ID", "retryable": False}
        if service_id:
            task = catalog["services"][0]["task"]
            if task != "action_chunk":
                input_schema, output_schema = task_schemas(task)
                catalog = {**catalog, "input_schema": input_schema, "output_schema": output_schema}
        return {"ok": True, **catalog, "read_only": True, "physical_authority": False}

    async def query(self, service_id: str, input: dict[str, Any],
                    artifacts: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        task = asyncio.current_task()
        if task is not None:
            self._calls.add(task)
        try:
            return await self._query(service_id, input, artifacts or {})
        except ServiceError as exc:
            return failure_payload(exc)
        except asyncio.TimeoutError:
            return failure_payload(ServiceError("Model query work deadline exhausted", service_id=service_id,
                                                code="model_timeout", recoverable=True))
        except (ValueError, OSError) as exc:
            return failure_payload(ServiceError(str(exc), service_id=service_id, code="model_invalid_request",
                                                category="validation"))
        finally:
            if task is not None:
                self._calls.discard(task)

    async def _query(self, service_id: str, data: dict[str, Any], artifacts: dict[str, str]) -> dict[str, Any]:
        context = current_tool_context()
        if self.client is None or context is None or not context.session_id:
            raise ServiceError("Model query requires a bound service and active session", code="model_unavailable")
        catalog = self.client.describe(service_id)["services"]
        if not catalog:
            raise ServiceError("Unknown service; use model_describe", service_id=service_id)
        task = catalog[0]["task"]
        input_schema, output_schema = task_schemas(task)
        validate_schema(data, input_schema, "input")
        if task in INPUT_SCHEMAS:
            eligible_ids(data, task)
        validate_schema(artifacts, {"type": "object", "properties": {}, "additionalProperties": True})
        if len(artifacts) > 16 or any(not isinstance(k, str) or not isinstance(v, str) for k, v in artifacts.items()):
            raise ValueError("artifacts must map at most 16 labels to session artifact IDs")
        if task == "grounding" and set(artifacts) != {"image"}:
            raise ValueError("grounding requires exactly one image artifact")
        if task == "grasp_proposal" and set(artifacts) != {"image", "depth"}:
            raise ValueError("grasp_proposal requires image and depth artifacts")
        handle = await self.client.acquire(service_id)
        budget = ActiveCallBudget(handle.spec.limits.deadline_s)
        try:
            handle.check_admission()
            request = await budget.run(asyncio.to_thread(lambda: json.loads(canonical(data))))
            handle.check_admission()
            if (artifacts or data.get("observation_refs") or data.get("episode_ref")) and handle.spec.processing_location == "unknown":
                raise ServiceError("Image processing locality is unknown; qualify the deployment first",
                                   code="model_privacy_unknown", category="security")
            # Obtain egress consent before reading snapshots so a long consent
            # wait cannot turn admitted artifacts into expired evidence at upload.
            sensitive = bool(artifacts or data.get("observation_refs") or data.get("episode_ref"))
            await authorize_endpoint(handle.spec, self.gate, sensitive=sensitive, budget=budget,
                                     check_admission=handle.check_admission)
            media, evidence = await self._artifacts(artifacts, data, task, handle.spec.limits.max_request_bytes,
                                                   budget, check_admission=handle.check_admission)
            handle.check_admission()
            if task == "grasp_proposal":
                depth = evidence.pop("depth")
                await budget.run(asyncio.to_thread(validate_depth, request, depth))
                handle.check_admission()
                request["depth"]["data"] = base64.b64encode(depth).decode("ascii")
                from leapflow.hardware.model_contract import check_request
                check_request(task, {**request, "image": media[0]["data"]})
            request.update({"_images": media, "artifact_order": sorted(k for k in artifacts if k != "depth"),
                            "context": {"session_id": context.session_id,
                                        "run_id": context.task_id}})
            if evidence:
                request["artifact_evidence"] = {k: json.loads(v) for k, v in evidence.items()}
            validator = (lambda value: validate_result(task, data, value)) if task in INPUT_SCHEMAS else (
                lambda value: _validate_geometry(task, data, value))
            if handle.spec.identity_mode == "server_reported" and task in GEOMETRY_INPUTS:
                from leapflow.hardware.model_contract import parse_response, provider_request, provider_result_schema
                # The same domain envelope is used by the qualified hardware
                # adapter, but this tool's result remains advisory only.
                labels = sorted(k for k in artifacts if k != "depth")
                request.pop("_images")
                request.pop("artifact_order")
                request.pop("context")
                request = provider_request(task, request, dict(zip(labels, media)))
                output_schema = provider_result_schema(task)
                validator = partial(parse_response, task)
            handle.check_admission()
            result = await budget.run(handle.invoke(request, output_schema, validate=validator))
            handle.check_admission()
            payload = {"ok": True, "service_id": service_id, "task": task, "result": result.payload(),
                    "provenance": {"generation": result.generation, "api_model": result.api_model,
                                   "identity_mode": result.identity_mode, "advisory": True,
                                   "hypothetical": task == "prediction", "teacher_assessment": task == "evaluation",
                                   "scores_calibrated": False, "physical_authority": False},
                    "elapsed_ms": result.elapsed_ms, "usage": json.loads(result.usage_json)}
        finally:
            await handle.aclose()
        handle.check_result_eligibility()
        return payload

    async def _artifacts(self, artifacts: dict[str, str], data: dict[str, Any], task: str,
                         budget: int, deadline: ActiveCallBudget, *,
                         check_admission: Callable[[], None]) -> tuple[list[dict[str, str]], dict[str, bytes]]:
        from leapflow.cache.manager import CacheManager
        from leapflow.layout import workspace_id_for_path

        context = current_tool_context()
        references = dict(artifacts)
        observed = {ref: ref for ref in data.get("observation_refs", [])}
        if task == "evaluation":
            observed["episode"] = data["episode_ref"]
        if set(references) & set(observed):
            raise ValueError("Artifact labels collide with observation or episode references")
        references.update(observed)
        if not references:
            return [], {}
        if self.settings is None:
            raise ValueError("Session artifact storage is unavailable")
        check_admission()
        cache = await deadline.run(asyncio.to_thread(CacheManager, self.settings.profile_layout.cache,
                                                     profile_id=self.settings.profile))
        check_admission()
        media, evidence, snapshots = [], {}, []
        for label, entry_id in sorted(references.items()):
            check_admission()
            entry = await deadline.run(asyncio.to_thread(cache.resolve_session_entry, entry_id,
                                            workspace_id=workspace_id_for_path(context.workspace_root),
                                            session_id=context.session_id, max_bytes=budget))
            check_admission()
            await authorize(ActionDescriptor.file_read(str(entry.path), mode="model_evidence",
                                                       metadata={"sensitive": True, "sensitivity_category": "cache_sensitive",
                                                                 "platform": "model_services"}), self.gate)
            check_admission()

            def read_entry():
                check_admission()
                return cache.read_session_entry(entry, max_bytes=budget)

            content = await deadline.run(asyncio.to_thread(read_entry))
            check_admission()
            snapshots.append(entry)
            budget -= len(content)
            if label in artifacts and label != "depth":
                media.append(await deadline.run(asyncio.to_thread(
                    lambda value=content: {"media_type": _media_type(value),
                                           "data": base64.b64encode(value).decode("ascii")})))
            else:
                evidence[label] = content
            check_admission()
        # A later file consent may outlive an earlier artifact. Validate the
        # complete snapshot again before it can be submitted to the provider.
        for entry in snapshots:
            check_admission()
            current = await deadline.run(asyncio.to_thread(cache.resolve_session_entry, entry.entry_id,
                workspace_id=entry.workspace_id, session_id=entry.session_id, max_bytes=entry.size_bytes))
            check_admission()
            if current != entry:
                raise ValueError("artifact changed while assembling the request")
        return media, evidence

    async def aclose(self) -> None:
        calls = tuple(self._calls)
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)

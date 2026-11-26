# Copyright (c) Alibaba, Inc. and its affiliates.
"""Governed read-only collaboration with configured specialist services."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, replace
from functools import partial
from typing import Any, Callable, Mapping

from leapflow.llm.services.codec import media_kind, validate_schema
from leapflow.llm.services.protocol import ActiveCallBudget, ServiceError, ServiceSpec, StructuredModelClient, canonical
from leapflow.llm.services.telemetry import (
    ConsultationRecord, InferenceCapture, capture_inference, current_capture, measure_phase,
)
from leapflow.domain.events import RecordScope
from leapflow.security.actions import ActionDescriptor
from leapflow.security.network import _category_for_address, classify_url
from leapflow.tools.execution_context import current_tool_context
from leapflow.hardware.service_contracts import GEOMETRY_INPUTS

logger = logging.getLogger(__name__)

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
    categories = sorted({_category_for_address(address) for address in target.addresses})
    qualified_local = spec.processing_location == "local" and categories == ["loopback"] and not spec.allow_remote
    scope = "text_and_evidence" if qualified_local else ("evidence" if sensitive else "text")
    binding = {"platform": "model_services", "service_id": spec.service_id,
               "declaration_digest": hashlib.sha256(canonical(asdict(spec)).encode()).hexdigest(),
               "origin": target.origin, "target_category": target.category,
               "resolved_addresses": sorted(set(target.addresses)),
               "processing_location": spec.processing_location, "evidence_scope": scope}
    action = ActionDescriptor.network_fetch(
        spec.base_url, origin=target.origin, method="POST",
        metadata={**target.to_metadata(), "platform": "model_services", "service_id": spec.service_id,
                  "allow_remote": spec.allow_remote, "approval_binding": binding, "resolved_categories": categories,
                  "processing_location": spec.processing_location, "sensitive_payload": sensitive})
    with measure_phase("approval_wait_ms"):
        await authorize(replace(action, detail="<model-service>",
                                metadata={key: value for key, value in action.metadata.items() if key != "url"},
                                summary=f"Query model service {spec.service_id} "
                                        f"({spec.processing_location}; {scope.replace('_', ' ')})"), gate)
    if check_admission is not None:
        check_admission()
    current = await (budget or ActiveCallBudget(spec.limits.deadline_s)).run(classify_url(spec.base_url))
    if current.origin != target.origin or current.category != target.category or set(current.addresses) != set(target.addresses):
        raise ServiceError("Model destination changed during consent; retry to authorize the new destination",
                           code="permission_denied", category="security")
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
    if content[4:8] == b"ftyp":
        return "video/mp4"
    if content.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    raise ValueError("artifact is not an admitted JPEG, PNG, WebP, MP4 or WebM")


def check_media_capacity(spec: ServiceSpec, media: list[dict[str, str]]) -> dict[str, int]:
    """Refuse media the service did not declare, before any byte leaves the host."""
    counts = {kind: sum(media_kind(item["media_type"]) == kind for item in media) for kind in ("image", "video")}
    for kind, count in counts.items():
        if count > spec.media.capacity(kind):
            raise ServiceError(f"Service {spec.service_id} accepts at most {spec.media.capacity(kind)} {kind}(s) "
                               f"per request; got {count}. See model_describe media capacity",
                               service_id=spec.service_id, code="media_unsupported", category="validation")
    return counts


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
            selected = catalog["services"][0]
            catalog = {**catalog, "input_schema": selected.get("input_schema"),
                       "output_schema": selected.get("output_schema")}
        catalog["services"] = [{k: v for k, v in entry.items() if k not in {"input_schema", "output_schema"}}
                               for entry in catalog["services"]]
        feedback = {}
        try:
            capture = current_capture()
            sink = capture.usage_sink() if capture else None
            if callable(getattr(sink, "combined_usage", None)):
                feedback = {"window": "current_turn", "inference_usage": sink.combined_usage(),
                            "measured_task_outcome": "unverified"}
            if capture and callable(getattr(sink, "collaboration_feedback", None)):
                feedback["history"] = sink.collaboration_feedback(capture.scope, service_id=service_id)
        except Exception:  # noqa: BLE001 - feedback must never fail a catalog read
            logger.warning("Specialist usage feedback unavailable", exc_info=True)
        return {"ok": True, **catalog, "read_only": True, "physical_authority": False,
                "usage_feedback": feedback}

    async def query(self, service_id: str, input: dict[str, Any],
                    artifacts: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        from leapflow.layout import workspace_id_for_path

        context = current_tool_context()
        scope = RecordScope(str(getattr(self.settings, "profile", "") or ""),
            workspace_id_for_path(context.workspace_root) if context else "",
            context.session_id if context else "")
        consultation_id = uuid.uuid4().hex
        started = time.monotonic()
        payload: dict[str, Any] = {}
        status = "failed"
        with capture_inference(scope=scope, task_id=context.task_id if context else "") as capture:
            try:
                payload = await self._query_safe(service_id, input, artifacts or {})
                status = "eligible" if payload.get("ok") else "refused"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            finally:
                self._finish_consultation(capture, consultation_id, service_id, status, started, payload)
        if payload.get("ok") is True:
            # The tool owns what constitutes sufficient advisory context. Budget
            # shaping selects these existing fields atomically; audit stays full.
            payload["_context_fields"] = {
                "ok": True, "service_id": True, "task": True, "result": True,
                "provenance": {key: True for key in (
                    "api_model", "contract_revision", "advisory", "eligible_candidate_ids",
                    "hypothetical", "teacher_assessment", "scores_calibrated", "physical_authority")},
                "consultation": {"call_id": True, "status": True},
            }
        return payload

    @staticmethod
    def _finish_consultation(capture: InferenceCapture, identity: str, service_id: str,
                             status: Any, started: float, payload: dict[str, Any]) -> None:
        """Retain cancellation/refusal as well as eligible advice; telemetry never raises."""
        try:
            record = ConsultationRecord(identity, capture.scope, capture.task_id, capture.tool_call_id,
                service_id, status, (time.monotonic() - started) * 1000,
                str(payload.get("failure_code") or ""), capture.provenance_json, tuple(capture.attempts),
                hashlib.sha256(canonical(payload["result"]).encode()).hexdigest() if "result" in payload else "",
                timing_json=canonical(capture.phase_timings))
            capture.record_consultation(record)
            payload["consultation"] = record.to_dict()
            provenance = json.loads(record.provenance_json)
            if provenance:
                payload["provenance"] = {**provenance, **payload.get("provenance", {})}
        except Exception:  # noqa: BLE001 - bookkeeping must never change a tool result
            logger.warning("Consultation provenance recording failed", exc_info=True)

    async def _query_safe(self, service_id: str, data: dict[str, Any], artifacts: dict[str, str]) -> dict[str, Any]:
        task = asyncio.current_task()
        if task is not None:
            self._calls.add(task)
        try:
            return await self._query(service_id, data, artifacts)
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
        validate_schema(artifacts, {"type": "object", "properties": {}, "additionalProperties": True})
        if len(artifacts) > 16 or any(not isinstance(k, str) or not isinstance(v, str) for k, v in artifacts.items()):
            raise ValueError("artifacts must map at most 16 labels to session artifact IDs")
        if task == "grounding" and set(artifacts) != {"image"}:
            raise ValueError("grounding requires exactly one image artifact")
        if task == "grasp_proposal" and set(artifacts) != {"image", "depth"}:
            raise ValueError("grasp_proposal requires image and depth artifacts")
        with measure_phase("acquisition_ms"):
            handle = await self.client.acquire(service_id)
        budget = ActiveCallBudget(handle.spec.limits.deadline_s)
        try:
            handle.check_admission()
            contract = handle.contract
            self._capture_provenance(handle, data, [])
            if contract.classification != "read_only":
                raise ValueError("This contract is not a read-only query; use admitted hw_skill_* / policy.run")
            _, output_schema = contract.schemas()
            with measure_phase("input_validation_ms"):
                await budget.run(asyncio.to_thread(contract.validate_input, data))
            observed = contract.evidence_references(data)
            handle.check_admission()
            request = await budget.run(asyncio.to_thread(lambda: json.loads(canonical(data))))
            handle.check_admission()
            if (artifacts or observed) and handle.spec.processing_location == "unknown":
                raise ServiceError("Image processing locality is unknown; qualify the deployment first",
                                   code="model_privacy_unknown", category="security")
            # Admit metadata before consent; no evidence bytes are read until it succeeds.
            sensitive = bool(artifacts or observed)
            cache, snapshots = await self._resolve_artifacts(artifacts, observed,
                handle.spec.limits.max_request_bytes, budget, check_admission=handle.check_admission)
            self._capture_provenance(handle, data, snapshots)
            await authorize_endpoint(handle.spec, self.gate, sensitive=sensitive, budget=budget,
                                     check_admission=handle.check_admission)
            with measure_phase("artifact_encoding_ms"):
                media, evidence = await self._artifacts(artifacts, cache, snapshots,
                    handle.spec.limits.max_request_bytes, budget, check_admission=handle.check_admission)
            handle.check_admission()
            labels = sorted(k for k in artifacts if k != "depth")
            media_counts = check_media_capacity(handle.spec, media)
            if task in GEOMETRY_INPUTS and media_counts["video"]:
                raise ValueError(f"{task} requires image evidence, not video")
            if task == "grasp_proposal":
                depth = evidence.pop("depth")
                await budget.run(asyncio.to_thread(validate_depth, request, depth))
                handle.check_admission()
                request["depth"]["data"] = base64.b64encode(depth).decode("ascii")
                from leapflow.hardware.model_contract import check_request
                check_request(task, {**request, "image": media[0]["data"]})
            # Block order equals ``artifact_order`` so the service can bind each view to its label.
            request.update({"_media": media, "artifact_order": labels,
                            "context": {"session_id": context.session_id,
                                        "run_id": context.task_id}})
            if evidence:
                request["artifact_evidence"] = {k: json.loads(v) for k, v in evidence.items()}
            validator = partial(contract.validate_result, data)
            if handle.spec.identity_mode == "server_reported" and task in GEOMETRY_INPUTS:
                from leapflow.hardware.model_contract import parse_response, provider_request, provider_result_schema
                # The same domain envelope is used by the qualified hardware
                # adapter, but this tool's result remains advisory only.
                request.pop("_media")
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
                                   "contract_revision": contract.revision, "contract_generation": contract.number,
                                   "codec_revision": handle.codec_revision,
                                   "inference_call_id": result.call_id,
                                   "declaration_digest": hashlib.sha256(canonical(asdict(handle.spec)).encode()).hexdigest(),
                                   "input_digest": hashlib.sha256(canonical(data).encode()).hexdigest(),
                                   "eligible_candidate_ids": [candidate["id"] for candidate in data.get("candidates", [])
                                       if isinstance(candidate, dict) and isinstance(candidate.get("id"), str)],
                                   "evidence": self._evidence_provenance(snapshots),
                                   "media": {"images": media_counts["image"], "videos": media_counts["video"],
                                             "video_s": self._video_seconds(snapshots), "labels": labels},
                                   "hypothetical": task == "prediction", "teacher_assessment": task == "evaluation",
                                   "scores_calibrated": False, "physical_authority": False},
                    "elapsed_ms": result.elapsed_ms, "usage": json.loads(result.usage_json),
                    "timing": json.loads(result.timing_json)}
        finally:
            with measure_phase("cleanup_ms"):
                await handle.aclose()
        handle.check_result_eligibility()
        return payload

    @staticmethod
    def _capture_provenance(handle: Any, data: Any, snapshots: list[tuple[str, Any]]) -> None:
        """Detach pinned identity before consent/inference so refusals retain lineage."""
        try:
            capture = current_capture()
            if capture is None:
                return
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            candidates = candidates if isinstance(candidates, list) else []
            capture.provenance_json = canonical({
                "generation": handle.generation, "api_model": handle.spec.model,
                "identity_mode": handle.spec.identity_mode,
                "contract_revision": handle.contract.revision,
                "contract_generation": handle.contract.number, "codec_revision": handle.codec_revision,
                "declaration_digest": hashlib.sha256(canonical(asdict(handle.spec)).encode()).hexdigest(),
                "input_digest": hashlib.sha256(canonical(data).encode()).hexdigest(),
                "eligible_candidate_ids": [item["id"] for item in candidates
                    if isinstance(item, dict) and isinstance(item.get("id"), str)],
                "evidence": ModelTools._evidence_provenance(snapshots),
                "media": {"images": sum(entry.metadata.get("kind") == "image" for _, entry in snapshots),
                          "videos": sum(entry.metadata.get("kind") == "video" for _, entry in snapshots),
                          "video_s": ModelTools._video_seconds(snapshots),
                          "labels": [label for label, _ in snapshots if label != "depth"]},
                "physical_authority": False, "advisory": True,
            })
        except Exception:  # noqa: BLE001 - provenance failure is not provider failure
            logger.warning("Pinned consultation provenance unavailable", exc_info=True)

    @staticmethod
    def _video_seconds(snapshots: list[tuple[str, Any]]) -> float:
        return sum(float(entry.metadata.get("duration_s") or 0) for _, entry in snapshots
                   if entry.metadata.get("kind") == "video")

    @staticmethod
    def _evidence_provenance(snapshots: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for label, entry in snapshots:
            record = entry.metadata.get("record", {})
            value = record.get("value", {}) if isinstance(record, dict) else {}
            value = value if isinstance(value, dict) else {}
            records.append({"label": label, "artifact_id": entry.entry_id,
                "content_hash": entry.content_hash, "source": entry.source,
                "created_at": entry.created_at, "expires_at": entry.expires_at,
                **{key: value[key] for key in ("observation_id", "frame", "capture_id", "camera",
                    "observed_at", "calibration_ref", "kind", "duration_s", "media_source", "run_id", "historical") if key in value}})
        return records

    async def _resolve_artifacts(self, artifacts: dict[str, str], observed: Mapping[str, str],
                                 budget: int, deadline: ActiveCallBudget, *,
                                 check_admission: Callable[[], None]) -> tuple[Any, list[tuple[str, Any]]]:
        from leapflow.cache.manager import CacheManager
        from leapflow.layout import workspace_id_for_path

        context = current_tool_context()
        references = dict(artifacts)
        if set(references) & set(observed):
            raise ValueError("Artifact labels collide with observation or episode references")
        references.update(observed)
        if not references:
            return None, []
        if self.settings is None:
            raise ValueError("Session artifact storage is unavailable")
        check_admission()
        cache = await deadline.run(asyncio.to_thread(CacheManager, self.settings.profile_layout.cache,
                                                     profile_id=self.settings.profile))
        check_admission()
        snapshots = []
        for label, entry_id in sorted(references.items()):
            check_admission()
            entry = await deadline.run(asyncio.to_thread(cache.resolve_session_entry, entry_id,
                                            workspace_id=workspace_id_for_path(context.workspace_root),
                                            session_id=context.session_id, max_bytes=budget))
            check_admission()
            snapshots.append((label, entry))
            budget -= entry.size_bytes
        return cache, snapshots

    async def _artifacts(self, artifacts: dict[str, str], cache: Any, snapshots: list[tuple[str, Any]],
                         budget: int, deadline: ActiveCallBudget, *,
                         check_admission: Callable[[], None]) -> tuple[list[dict[str, str]], dict[str, bytes]]:
        async def verify() -> None:
            for _, entry in snapshots:
                check_admission()
                current = await deadline.run(asyncio.to_thread(cache.resolve_session_entry, entry.entry_id,
                    workspace_id=entry.workspace_id, session_id=entry.session_id, max_bytes=entry.size_bytes))
                check_admission()
                if current != entry:
                    raise ValueError("artifact changed while assembling the request")

        # Check the entire selection after consent before reading even the first entry.
        await verify()
        media, evidence = [], {}
        for label, entry in snapshots:
            check_admission()

            def read_entry():
                check_admission()
                return cache.read_session_entry(entry, max_bytes=budget)

            content = await deadline.run(asyncio.to_thread(read_entry))
            check_admission()
            budget -= len(content)
            if label in artifacts and label != "depth":
                media.append(await deadline.run(asyncio.to_thread(
                    lambda value=content: {"media_type": _media_type(value),
                                           "data": base64.b64encode(value).decode("ascii")})))
            else:
                evidence[label] = content
            check_admission()
        # Assembly may outlive an earlier artifact; preserve the final freshness check.
        await verify()
        return media, evidence

    async def aclose(self) -> None:
        calls = tuple(self._calls)
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Explicit operator fixture checks; no capture, persistent results, or action execution."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from leapflow.hardware.model_contract import (
    check_request, parse_response, provider_request, provider_result_schema,
)
from leapflow.llm.services.codec import _pairs, validate_schema, validate_schema_definition
from leapflow.llm.services.protocol import ActiveCallBudget, ServiceError, ServiceHandle, StructuredModelClient
from leapflow.security.actions import ActionDescriptor
from leapflow.tools.execution_context import require_workspace_access
from leapflow.tools.model_tools import (
    _media_type, _validate_geometry, authorize, authorize_endpoint, failure_payload, task_schemas, validate_depth,
)
from leapflow.world_model.specialist_contract import INPUT_SCHEMAS, eligible_ids, validate_result

def _open_fixture(path: Path) -> int:
    """Walk the approved canonical path without following replacement symlinks."""
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)


def _read_fixture(path: Path, budget: int) -> dict[str, Any]:
    fd = _open_fixture(path)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > budget:
            raise ValueError("Fixture must be a bounded regular JSON file")
        content = stream.read(budget + 1)
    if len(content) > budget:
        raise ValueError("Fixture exceeds its byte budget")
    return _validate_fixture(json.loads(content, object_pairs_hook=_pairs))


def _validate_fixture(value: Any) -> dict[str, Any]:
    """One schema validator for approved file cases and bounded inline cases."""
    try:
        validate_schema(value, {"type": "object", "properties": {
            "schema_version": {"type": "integer", "enum": [1]},
            "artifact_scope": {"type": "string", "enum": ["synthetic", "recorded"]},
            "input": {"type": "object"}, "artifacts": {"type": "object"},
            "expected": {"type": "object"}},
            "required": ["schema_version", "artifact_scope", "input", "expected"],
            "additionalProperties": False})
        validate_schema_definition(value["expected"])
    except (ValueError, UnicodeError, RecursionError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid fixture JSON/schema; use schema_version 1, input, artifact_scope and expected") from exc
    return value


def _prepare(handle: ServiceHandle, fixture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Any]:
    task, data = handle.spec.task, dict(fixture["input"])
    artifacts = fixture.get("artifacts", {})
    if len(artifacts) > 16:
        raise ValueError("A fixture may contain at most 16 embedded artifacts")
    media, evidence = {}, {}
    for label, item in artifacts.items():
        if not isinstance(label, str) or not isinstance(item, dict):
            raise ValueError("Fixture artifacts must map labels to embedded data")
        if set(item) == {"media_type", "data"}:
            raw = base64.b64decode(item["data"], validate=True)
            if _media_type(raw) != item["media_type"]:
                raise ValueError("Fixture image media_type does not match its bytes")
            media[label] = dict(item)
        elif set(item) == {"json"} and isinstance(item["json"], dict):
            evidence[label] = item["json"]
        else:
            raise ValueError("Fixtures accept embedded image data or JSON evidence, never paths or URLs")
    if task == "action_chunk":
        if not media or evidence or set(data) - {"state", "task", "pending"}:
            raise ValueError("Action fixture requires state, task, optional pending, and named embedded camera images")
        if any(item["media_type"] != "image/jpeg" for item in media.values()):
            raise ValueError("Native action fixtures require JPEG camera images")
        if "pending" in data and (not isinstance(data["pending"], dict)
                                  or set(data["pending"]) != {"actions", "latency_steps"}):
            raise ValueError("pending must contain exactly actions and latency_steps")
        if not isinstance(data.get("task"), str):
            raise ValueError("task must be an instruction string")
        data = check_request(task, {**data, "images": {k: v["data"] for k, v in media.items()}})
        data.pop("images")
        if not data["state"]:
            raise ValueError("state must be a nonempty native vector")
        schema = provider_result_schema(task)
        validator = partial(parse_response, task)
    else:
        input_schema, schema = task_schemas(task)
        depth = None
        if task == "grasp_proposal" and isinstance(data.get("depth"), dict):
            data["depth"] = dict(data["depth"])
            depth = data["depth"].pop("data", None)
        validate_schema(data, input_schema, "input")
        if task in INPUT_SCHEMAS:
            eligible_ids(data, task)
            refs = set(data.get("observation_refs", []))
            if task == "evaluation":
                refs.add(data["episode_ref"])
            if refs != set(evidence):
                raise ValueError("Fixture evidence must exactly match its declared observation/episode references")
            validator = partial(validate_result, task, data)
        else:
            if set(media) != {"image"} or evidence:
                raise ValueError("Geometry fixture requires exactly one image and no unrelated evidence")
            if task == "grasp_proposal":
                if not isinstance(depth, str):
                    raise ValueError("Grasp fixture depth.data must be embedded uint16 little-endian base64")
                validate_depth(data, base64.b64decode(depth, validate=True))
                data["depth"]["data"] = depth
                data = check_request(task, {**data, "image": media["image"]["data"]})
                data.pop("image")
            validator = partial(_validate_geometry, task, data)
    if handle.spec.identity_mode == "server_reported" and task not in INPUT_SCHEMAS:
        request = provider_request(task, data, media)
        schema = provider_result_schema(task)
        validator = partial(parse_response, task)
    else:
        request = {**data, "_images": [media[k] for k in sorted(media)], "artifact_order": sorted(media)}
        if evidence:
            request["artifact_evidence"] = evidence
    return request, schema, validator


async def check_fixture(client: StructuredModelClient, service_id: str, path: Path,
                        gate: Any = None) -> dict[str, Any]:
    """Check one explicitly supplied fixture with ordinary file and egress approvals."""
    return await _check_fixture(client, service_id, path=path, gate=gate)


async def check_inline_fixture(client: StructuredModelClient, service_id: str, case: Mapping[str, Any],
                               gate: Any = None) -> dict[str, Any]:
    """Check an explicitly supplied text-only case, never model-authored image data."""
    return await _check_fixture(client, service_id, case=case, gate=gate)


async def _check_fixture(client: StructuredModelClient, service_id: str, *, path: Path | None = None,
                         case: Mapping[str, Any] | None = None, gate: Any = None) -> dict[str, Any]:
    handle = None
    stage = "configuration"
    stages = {"configuration": "not_checked", "provider_construction": "not_checked",
              "fixture_input": "not_checked", "egress_approval": "not_checked",
              "api_request_subset": "not_checked", "task_result_schema": "not_checked",
              "fixture_assertions": "not_checked", "native_semantics": "not_checked",
              "identity_provenance": "not_qualified", "embodiment": "not_checked",
              "execution_consent": "not_requested"}
    try:
        if not service_id:
            raise ValueError("check --case requires exactly one service ID")
        stage = "provider_construction"
        handle = await client.acquire(service_id)
        handle.check_admission()
        stages["configuration"] = stages["provider_construction"] = "pass"
        budget = ActiveCallBudget(handle.spec.limits.deadline_s)
        stage = "fixture_input"
        if path is not None:
            path = await budget.run(asyncio.to_thread(path.resolve, strict=True))
            handle.check_admission()
            refused = await require_workspace_access(path, operation="model_fixture")
            handle.check_admission()
            if refused is not None:
                raise ServiceError(refused["error"], code="permission_denied", category="security")
            await authorize(ActionDescriptor.file_read(str(path), mode="model_fixture",
                                                       metadata={"platform": "model_services"}), gate)
            handle.check_admission()

            def read_fixture():
                handle.check_admission()
                return _read_fixture(path, handle.spec.limits.max_request_bytes)

            fixture = await budget.run(asyncio.to_thread(read_fixture))
            handle.check_admission()
        else:
            from leapflow.llm.services.protocol import canonical

            encoded = canonical(case).encode("utf-8")
            if len(encoded) > min(65536, handle.spec.limits.max_request_bytes):
                raise ValueError("Inline fixture exceeds its text budget")
            fixture = _validate_fixture(json.loads(encoded))
            if fixture.get("artifacts") or handle.spec.task not in INPUT_SCHEMAS:
                raise ValueError("Inline cases are text-only specialist cases; use model_query for owned images")
        handle.check_admission()
        request, schema, validator = await budget.run(asyncio.to_thread(_prepare, handle, fixture))
        handle.check_admission()
        stages["fixture_input"] = "pass"
        stage = "egress_approval"
        sensitive = (fixture["artifact_scope"] == "recorded" or bool(fixture.get("artifacts"))
                     or handle.spec.task in {"grasp_proposal", "action_chunk"})
        await authorize_endpoint(handle.spec, gate, sensitive=sensitive, budget=budget,
                                 check_admission=handle.check_admission)
        stages["egress_approval"] = "pass"
        stage = "api_request_subset"

        def validate(value: Mapping[str, Any]) -> None:
            validator(value)
            try:
                validate_schema(value, fixture["expected"], "expected")
            except ValueError as exc:
                raise ServiceError("Model result did not satisfy the supplied fixture assertions",
                                   service_id=service_id, code="model_fixture_mismatch", category="validation") from exc

        result = await budget.run(handle.invoke(request, schema, validate=validate))
        handle.check_admission()
        stages.update(api_request_subset="pass", task_result_schema="pass", fixture_assertions="pass")
        payload = {"ok": True, "mode": "fixture", "service_id": service_id,
                "endpoint_probed": True, "result": result.payload(),
                "generation": result.generation, "elapsed_ms": result.elapsed_ms,
                "stages": stages,
                "physical_authority": False}
        await handle.aclose()
        handle.check_result_eligibility()
        handle = None
        return payload
    except ServiceError as exc:
        failure = failure_payload(exc)
    except asyncio.TimeoutError:
        failure = failure_payload(ServiceError("Fixture work deadline exhausted", service_id=service_id,
                                               code="model_timeout", recoverable=True))
    except (ValueError, OSError, TypeError, RecursionError):
        failure = failure_payload(ServiceError("Invalid or unreadable fixture; check input, embedded artifacts and expected schema",
                                               service_id=service_id, code="model_invalid_fixture", category="validation"))
    finally:
        if handle is not None:
            await handle.aclose()
    if failure["failure_code"] == "model_fixture_mismatch":
        stages.update(api_request_subset="pass", task_result_schema="pass")
        stage = "fixture_assertions"
    elif failure["failure_code"] == "model_invalid_output":
        stage = "task_result_schema"
    stages[stage] = "fail"
    return {**failure, "mode": "fixture", "service_id": service_id, "failed_stage": stage,
            "stages": stages, "physical_authority": False}

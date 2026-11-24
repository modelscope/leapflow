# Copyright (c) Alibaba, Inc. and its affiliates.
"""Strict JSON result codec and portable nonstreaming request shaping."""
from __future__ import annotations

import base64
import json
import math
from typing import Any, Mapping

from leapflow.llm.base import LLMChatResponse
from leapflow.llm.services.protocol import ServiceError, ServiceSpec, canonical


def validate_schema(value: Any, schema: Mapping[str, Any], path: str = "result", depth: int = 0) -> None:
    """Validate the bounded JSON-schema subset used by domain result contracts."""
    if depth > 32:
        raise ValueError(f"{path} exceeds the JSON nesting limit")
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    if kind is not None and (not isinstance(kind, str) or kind not in types or not isinstance(value, types[kind]) or
                 (kind in {"number", "integer"} and isinstance(value, bool))):
        raise ValueError(f"{path} must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} has an unsupported value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path} is outside the declared range")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", ())) - set(value):
            raise ValueError(f"{path} is missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path} contains unexpected fields")
        for key, item in value.items():
            validate_schema(item, properties.get(key, {}), f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path} has an invalid item count")
        for index, item in enumerate(value):
            validate_schema(item, schema.get("items", {}), f"{path}[{index}]", depth + 1)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ValueError(f"{path} has an invalid text length")


def validate_schema_definition(schema: Any, depth: int = 0) -> None:
    """Reject malformed operator assertions before any provider call.

    The portable subset has a protocol nesting limit of 32; it does not accept
    arbitrary JSON Schema extensions or executable validators.
    """
    keys = {"type", "properties", "required", "additionalProperties", "items", "enum",
            "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum"}
    if depth > 32 or not isinstance(schema, dict) or set(schema) - keys:
        raise ValueError("Use the supported bounded JSON Schema subset")
    if "type" in schema and (not isinstance(schema["type"], str) or schema["type"] not in {
        "object", "array", "string", "number", "integer", "boolean", "null"
    }):
        raise ValueError("Unsupported schema type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Schema properties must be an object")
    for child in properties.values():
        validate_schema_definition(child, depth + 1)
    if "items" in schema:
        validate_schema_definition(schema["items"], depth + 1)
    required = schema.get("required", [])
    if (not isinstance(required, list) or any(not isinstance(key, str) for key in required)
            or len(required) != len(set(required)) or set(required) - set(properties)):
        raise ValueError("Schema required must name unique declared properties")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise ValueError("Schema additionalProperties must be boolean")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError("Schema enum must be a nonempty array")
        encoded = [canonical(value) for value in values]
        if len(set(encoded)) != len(encoded):
            raise ValueError("Schema enum must contain distinct JSON values")
    for lower, upper in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
        for key in (lower, upper):
            if key not in schema:
                continue
            value = schema[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (lower != "minimum" and (type(value) is not int or value < 0))):
                raise ValueError("Schema bounds must be finite and correctly typed")
        if lower in schema and upper in schema and schema[lower] > schema[upper]:
            raise ValueError("Schema bounds are reversed")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_result(content: str) -> Any:
    try:
        return json.loads(content, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError("model result exceeds the JSON nesting limit") from exc


def image_block(image: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """Only embedded, declared images are portable; never dereference image URLs."""
    mime, data = image.get("media_type"), image.get("data")
    if mime not in {"image/jpeg", "image/png", "image/webp"} or not isinstance(data, str):
        raise ValueError("image requires an embedded JPEG, PNG or WebP")
    if not base64.b64decode(data, validate=True):
        raise ValueError("image is empty")
    if provider == "anthropic":
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


class JsonCodec:
    """Portable JSON/text/schema/tool encoding; no generated call ever executes."""

    def encode(self, spec: ServiceSpec, request: Mapping[str, Any],
               schema: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        body = dict(request)
        images = body.pop("_images", [])
        if not isinstance(images, list):
            raise ValueError("images must be an ordered list")
        if images and spec.processing_location == "unknown":
            raise ServiceError("Declare and authorize the actual image processing location first",
                               service_id=spec.service_id, code="model_privacy_unknown", category="security")
        content = [{"type": "text", "text": canonical(body)}]
        content.extend(image_block(image, spec.provider) for image in images)
        messages: list[dict[str, Any]] = []
        declaration = spec.declaration()
        if spec.identity_mode == "unverified":
            messages.append({"role": "system", "content": (
                declaration.get("prompt_template") or "Return only the declared JSON result, not executable instructions."
            ) + "\nResult schema: " + canonical(schema)})
        messages.append({"role": "user", "content": content})
        options = dict(declaration.get("request_options", {}))
        options["max_tokens"] = spec.limits.max_output_tokens
        if spec.output == "json_schema":
            options["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "result", "schema": dict(schema), "strict": True}}
        elif spec.output == "tool_call":
            options["tools"] = [{"type": "function", "function": {
                "name": "submit_result", "description": "Submit one structured inference result",
                "parameters": dict(schema)}}]
        return messages, options

    def decode(self, spec: ServiceSpec, response: LLMChatResponse,
               schema: Mapping[str, Any]) -> Mapping[str, Any]:
        if response.finish_reason in {"length", "max_tokens", "content_filter"}:
            raise ValueError("model result was truncated or filtered")
        if spec.output == "tool_call":
            if response.content.strip() or len(response.tool_calls) != 1:
                raise ValueError("expected exactly one result call, without conflicting text")
            call = response.tool_calls[0]
            if call.name != "submit_result" or not call.arguments:
                raise ValueError("invalid result call or malformed arguments")
            result = (_load_result(call.raw_arguments)
                      if call.raw_arguments is not None else call.arguments)
        else:
            if response.tool_calls:
                raise ValueError("unexpected result call")
            result = _load_result(response.content)
        if not isinstance(result, dict):
            raise ValueError("model result must be one JSON object")
        validate_schema(result, schema)
        canonical(result)
        return result

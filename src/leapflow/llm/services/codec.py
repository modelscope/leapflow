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


def strict_wire_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Compile closed domain schemas without changing their optional-field semantics.

    Open objects and untyped values cannot be represented by OpenAI strict output.
    Nullable unions exist only on the wire, not in the domain schema language.
    """
    validate_schema_definition(dict(schema))
    if schema.get("type") != "object":
        raise ValueError("Strict output requires an object root")

    def compile_node(node: Mapping[str, Any]) -> dict[str, Any]:
        kind = node.get("type")
        if kind is None:
            raise ValueError("Strict output cannot represent untyped values")
        allowed = {"type", "enum"} | {
            "object": {"properties", "required", "additionalProperties"},
            "array": {"items", "minItems", "maxItems"},
            "string": {"minLength", "maxLength"},
            "number": {"minimum", "maximum"}, "integer": {"minimum", "maximum"},
        }.get(kind, set())
        if set(node) - allowed or kind in {"object", "array"} and "enum" in node:
            raise ValueError("Strict output cannot represent this schema shape")
        for item in node.get("enum", []):
            validate_schema(item, {k: v for k, v in node.items() if k != "enum"})
        result = dict(node)
        if kind == "object":
            if node.get("additionalProperties") is not False:
                raise ValueError("Strict output cannot represent open objects; select an explicit compatible codec/output mode")
            required = set(node.get("required", ()))
            properties = {}
            for name, child in node.get("properties", {}).items():
                compiled = compile_node(child)
                if name not in required:
                    if child.get("type") == "null":
                        raise ValueError("Strict output cannot distinguish optional null from absence")
                    compiled = {"anyOf": [compiled, {"type": "null"}]}
                properties[name] = compiled
            result.update(properties=properties, required=list(properties), additionalProperties=False)
        elif kind == "array":
            if "items" not in node:
                raise ValueError("Strict output requires typed array items")
            result["items"] = compile_node(node["items"])
        return result

    return compile_node(schema)


def normalize_strict_result(value: Any, schema: Mapping[str, Any]) -> Any:
    """Remove only compiled optional nulls, enforcing the complete wire shape first."""
    if schema.get("type") == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(value) != set(properties):
            raise ValueError("Strict result must contain exactly all declared wire fields")
        required = set(schema.get("required", ()))
        return {name: normalize_strict_result(item, properties[name])
                for name, item in value.items() if item is not None or name in required}
    if schema.get("type") == "array" and isinstance(value, list):
        return [normalize_strict_result(item, schema["items"]) for item in value]
    return value


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


IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
VIDEO_TYPES = frozenset({"video/mp4", "video/webm"})


def media_kind(mime: Any) -> str:
    """Classify a declared embedded media type; unknown types have no kind."""
    return "image" if mime in IMAGE_TYPES else "video" if mime in VIDEO_TYPES else ""


def media_block(item: Mapping[str, Any], provider: str, *, service_id: str = "") -> dict[str, Any]:
    """Only embedded, declared media are portable; never dereference media URLs."""
    mime, data = item.get("media_type"), item.get("data")
    kind = media_kind(mime)
    if not kind or not isinstance(data, str):
        raise ValueError("media requires an embedded JPEG, PNG, WebP, MP4 or WebM")
    if not base64.b64decode(data, validate=True):
        raise ValueError(f"{kind} is empty")
    if kind == "video":
        if provider == "anthropic":
            raise ServiceError("This provider cannot carry video; send image evidence instead",
                               service_id=service_id, code="media_unsupported", category="validation")
        return {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}}
    if provider == "anthropic":
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def image_block(image: Mapping[str, Any], provider: str) -> dict[str, Any]:
    """Image-only shaping retained for callers that must never carry video."""
    if media_kind(image.get("media_type")) != "image":
        raise ValueError("image requires an embedded JPEG, PNG or WebP")
    return media_block(image, provider)


class JsonCodec:
    """Portable JSON/text/schema/tool encoding; no generated call ever executes."""

    def encode(self, spec: ServiceSpec, request: Mapping[str, Any],
               schema: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        body = dict(request)
        # ``_media`` is the single ordered, kind-neutral carrier set by the tool layer.
        media = body.pop("_media", [])
        if not isinstance(media, list):
            raise ValueError("media must be an ordered list")
        if media and spec.processing_location == "unknown":
            raise ServiceError("Declare and authorize the actual media processing location first",
                               service_id=spec.service_id, code="model_privacy_unknown", category="security")
        kinds = [media_kind(item.get("media_type")) if isinstance(item, Mapping) else "" for item in media]
        for kind in ("image", "video"):
            if kinds.count(kind) > spec.media.capacity(kind):
                raise ServiceError(f"Service {spec.service_id} accepts at most {spec.media.capacity(kind)} "
                                   f"{kind}(s) per request", service_id=spec.service_id,
                                   code="media_unsupported", category="validation")
        content = [{"type": "text", "text": canonical(body)}]
        content.extend(media_block(item, spec.provider, service_id=spec.service_id) for item in media)
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
                "name": "result", "schema": strict_wire_schema(schema), "strict": True}}
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
        if spec.output == "json_schema":
            strict_wire_schema(schema)
            result = normalize_strict_result(result, schema)
        validate_schema(result, schema)
        canonical(result)
        return result

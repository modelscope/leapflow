# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fail-closed v1 service admission, retaining references rather than credentials."""
from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import fields
from typing import Any, Mapping
from urllib.parse import urlsplit

from leapflow.llm.services.protocol import ServiceLimits, ServiceSpec, canonical

TASKS = frozenset({"grounding", "grasp_proposal", "action_chunk", "decision", "prediction", "evaluation"})
_KEYS = frozenset({"schema_version", "provider", "base_url", "model", "task", "codec", "output",
                   "auth", "allow_remote", "processing_location", "limits", "identity",
                   "model_info", "deployment", "request_options", "prompt_template", "enabled"})
_OPTIONS = frozenset({"temperature", "top_p", "seed", "stop"})


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _only(value: Mapping[str, Any], allowed: set[str] | frozenset[str], name: str) -> None:
    if set(value) - allowed:
        raise ValueError(f"{name} contains unsupported fields: {sorted(set(value) - allowed)}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_services(value: Any) -> tuple[ServiceSpec, ...]:
    """Validate a complete mapping and freeze each declaration without mutating it."""
    entries = _object(value, "llm.services")
    if any(not isinstance(name, str) for name in entries):
        raise ValueError("service IDs must be strings")
    return tuple(_parse(name, entries[name]) for name in sorted(entries))


def _parse(name: str, value: Any) -> ServiceSpec:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("service IDs must contain 1-64 letters, digits, underscores or hyphens")
    data = _object(value, name)
    _only(data, _KEYS, name)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"{name}: schema_version must be 1")
    enabled = data.get("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    provider = _text(data.get("provider"), "provider")
    model = _text(data.get("model"), "model")
    url = _text(data.get("base_url"), "base_url").rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("base_url must be an explicit http(s) URL without credentials/query/fragment")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    remote = data.get("allow_remote", False)
    if type(remote) is not bool:
        raise ValueError("allow_remote must be boolean")
    if not remote and not is_loopback(parts.hostname):
        raise ValueError("non-loopback endpoints require explicit allow_remote and egress approval")
    task = _text(data.get("task"), "task")
    if task not in TASKS:
        raise ValueError(f"unsupported task {task!r}")
    auth = _object(data.get("auth"), "auth")
    _only(auth, {"mode", "ref"}, "auth")
    mode = auth.get("mode")
    ref = auth.get("ref", "")
    if mode == "secret":
        from leapflow.security.secrets import secret_scope
        secret_scope(_text(ref, "auth.ref"))
        if not re.fullmatch(r"secret://(?:profile|global)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", ref):
            raise ValueError("auth.ref must name a vault entry")
    elif mode != "none" or ref:
        raise ValueError("auth must explicitly declare none, or secret with a vault reference")
    identity = _object(data.get("identity", {"mode": "unverified"}), "identity")
    _only(identity, {"mode"}, "identity")
    identity_mode = identity.get("mode", "unverified")
    if identity_mode not in {"unverified", "server_reported"}:
        raise ValueError("identity.mode must be unverified or server_reported")
    location = data.get("processing_location", "unknown")
    if location not in {"local", "remote", "unknown"}:
        raise ValueError("processing_location must be local, remote or unknown")
    output = data.get("output", "json_text")
    if output not in {"json_text", "json_schema", "tool_call"}:
        raise ValueError("unsupported structured output mode")
    if output == "json_schema" and provider != "openai":
        raise ValueError("json_schema mode requires a qualified OpenAI-compatible endpoint")
    options = _object(data.get("request_options", {}), "request_options")
    _only(options, _OPTIONS, "request_options")
    for key in ("temperature", "top_p"):
        number = options.get(key)
        if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float))
                                   or not math.isfinite(number) or not 0 <= number <= (2 if key == "temperature" else 1)):
            raise ValueError(f"request_options.{key} is outside its supported range")
    if "seed" in options and type(options["seed"]) is not int:
        raise ValueError("request_options.seed must be an integer")
    if "stop" in options:
        stop = options["stop"]
        if not (isinstance(stop, str) or isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
            raise ValueError("request_options.stop must be text or a list of text")
    template = data.get("prompt_template", "")
    if not isinstance(template, str):
        raise ValueError("prompt_template must be text")
    limits = _limits(data.get("limits", {}))
    if len(canonical(data).encode()) > limits.max_request_bytes:
        raise ValueError("service declaration exceeds its request size budget")
    if identity_mode == "server_reported":
        info = _object(data.get("model_info"), "model_info")
        if not all(isinstance(info.get(key), str) and info[key] for key in ("name", "version", "checksum")):
            raise ValueError("server_reported requires pinned model_info name/version/checksum")
        if info.get("task") != task:
            raise ValueError("model_info task must match the service task")
        deployment = _object(data.get("deployment"), "deployment")
        _only(deployment, {"engine_version", "adapter_digest", "qualification_ref"}, "deployment")
        for key in ("engine_version", "adapter_digest", "qualification_ref"):
            _text(deployment.get(key), f"deployment.{key}")
    return ServiceSpec(name, provider, url, model, task,
                       codec=_text(data.get("codec", "json.v1"), "codec"), output=output,
                       auth_ref=ref, processing_location=location, allow_remote=remote,
                       identity_mode=identity_mode, limits=limits, declaration_json=canonical(data),
                       enabled=enabled)


def _limits(value: Any) -> ServiceLimits:
    data = _object(value, "limits")
    _only(data, {f.name for f in fields(ServiceLimits)}, "limits")
    for key, number in data.items():
        if key == "deadline_s":
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ValueError("deadline_s must be finite and positive")
        elif type(number) is not int or number < (0 if key == "max_queue" else 1):
            raise ValueError(f"{key} must be a valid integer budget")
    return ServiceLimits(**data)


def declarations(specs: tuple[ServiceSpec, ...]) -> dict[str, Any]:
    return {spec.service_id: spec.declaration() for spec in specs}

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Per-ID specialist onboarding through the existing configuration control plane."""
from __future__ import annotations

import asyncio
import re
from typing import Any

from leapflow.llm.services.config import declarations, parse_services
from leapflow.llm.services.protocol import ServiceError, canonical
from leapflow.tools import config_tools
from leapflow.tools.execution_context import current_tool_context
from leapflow.tools.model_fixture import check_inline_fixture
from leapflow.tools.model_tools import failure_payload


_LIMITATIONS = [
    "Configuration and API checks do not qualify physical execution or native model semantics.",
    "Primary chat model and credentials are unchanged; serving processes are operator-owned.",
    "Inline checks are text-only. Use model_query with owned session artifact IDs for images.",
]


def _validate(args: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    if not isinstance(args, dict) or set(args) - {"operation", "service_id", "declaration", "case"}:
        raise ValueError("Use operation, service_id, declaration or case; paths and scopes are not accepted")
    if len(canonical(args).encode("utf-8")) > 65536:
        raise ValueError("Model service configuration input exceeds 64 KiB")
    operation, service_id = args.get("operation"), args.get("service_id", "")
    if operation not in {"show", "set", "unset", "check", "enable", "disable"}:
        raise ValueError("operation must be show, set, unset, check, enable or disable")
    if not isinstance(service_id, str) or (service_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", service_id)):
        raise ValueError("Use a catalog service ID, not a path")
    if operation in {"set", "unset", "enable", "disable"} or "case" in args:
        if not service_id:
            raise ValueError("This operation requires exactly one service_id")
    if ("declaration" in args) != (operation == "set"):
        raise ValueError("Only set accepts and requires a declaration")
    if "case" in args and operation != "check":
        raise ValueError("Only check accepts an explicitly supplied inline case")
    validated = None
    if operation == "set":
        specs = parse_services({service_id: args["declaration"]})
        from leapflow.llm.provider_registry import get_default_registry, get_scoped_default_registry

        get_scoped_default_registry()
        if get_default_registry().get_plugin(specs[0].provider) is None:
            raise ServiceError("Provider is unavailable; choose a discovered provider before saving",
                               service_id=service_id, code="model_provider_unavailable")
        # The existing v1 onboarding contract supports the built-in JSON codec.
        if specs[0].codec != "json.v1":
            raise ValueError("Chat onboarding requires the supported json.v1 codec")
        validated = declarations(specs)[service_id]
    return operation, service_id, validated


async def _check_inline(service_id: str, case: Any) -> dict[str, Any]:
    context = current_tool_context()
    if context is None or not context.session_id or not callable(getattr(context.orchestrator, "evaluate", None)):
        raise ServiceError("Inline API check requires an active governed session", code="permission_denied", category="security")
    owner = getattr(config_tools._context_ref, "model_services", None)
    owned = owner is None
    if owned:
        from dataclasses import replace
        from leapflow.llm.services.runtime import ModelServiceRuntime

        settings = config_tools._active_settings()
        specs = parse_services(await asyncio.to_thread(config_tools._service().model_services))
        owner = ModelServiceRuntime.from_settings(replace(settings, llm_services=specs))
    try:
        return await check_inline_fixture(owner, service_id, case, gate=context.orchestrator)
    finally:
        if owned:
            await owner.aclose()


async def _publish_saved(service: Any, mutation: Any, operation: str, service_id: str) -> dict[str, Any]:
    """Report durable commit separately from fallible runtime/read-back publication."""
    result = {"ok": mutation.ok, "saved": bool(mutation.ok), "operation": operation,
              "service_id": service_id, "changed_keys": list(mutation.changed_keys),
              "scope": "profile", "side_effect_state": "committed" if mutation.ok and mutation.changed_keys else "none",
              "endpoint_probed": False, "physical_authority": False, "limitations": _LIMITATIONS}
    reloaded = config_tools._reload_after_write()
    try:
        effective = await asyncio.to_thread(service.model_services)
        owner = getattr(config_tools._context_ref, "model_services", None)
        runtime = owner.describe(service_id) if owner is not None else None
        generation = runtime.get("generation") if runtime is not None else None
        runtime_service = runtime["services"][0] if runtime and runtime["services"] else None
    except (ValueError, OSError, RuntimeError):
        return {**result, "session_reloaded": reloaded, "read_back_status": "unavailable",
                "generation": None, "effective_enabled": None,
                "runtime_publication_confirmed": False, "retryable": False,
                "next_step": "The change was saved. Verify durable configuration and runtime status before making another change."}
    declaration = effective.get(service_id)
    matches = ((declaration is None and runtime_service is None) or
               (declaration is not None and runtime_service is not None
                and declaration.get("enabled", True) == runtime_service["enabled"]))
    published = bool(reloaded and runtime is not None and matches)
    return {**result, "declaration": declaration, "session_reloaded": reloaded,
            "read_back_status": "pass", "generation": generation,
            "effective_enabled": runtime_service["enabled"] if published and runtime_service else None,
            "runtime_publication_confirmed": published,
            "next_step": ("Read back and run a local check; API checks require a supplied case and consent."
                          if published else "The change was saved; runtime reload is not confirmed. Reload at idle before querying the specialist.")}


async def config_model_service_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Discover, approve and mutate one specialist; never edit the primary model."""
    try:
        operation, service_id, declaration = await asyncio.to_thread(_validate, args)
        service = config_tools._service()
        if operation == "check":
            if "case" in args:
                return await _check_inline(service_id, args["case"])
            return {**await asyncio.to_thread(service.check_model_services, service_id), "limitations": _LIMITATIONS}
        catalog = await asyncio.to_thread(service.model_services)
        if service_id and operation != "set" and service_id not in catalog:
            raise ValueError("Unknown model service; use show to discover catalog IDs")
        if operation == "show":
            return {"ok": True, "scope": "profile", "services": {service_id: catalog[service_id]} if service_id else catalog,
                    "endpoint_probed": False, "limitations": _LIMITATIONS}
        denial = await config_tools._approve_write(
            f"llm.services.{service_id}", scope="profile", secret=False, hot_reload="yes",
            tool="config_model_service", service_id=service_id,
            declaration_keys=tuple(sorted(declaration or {})), require_route=True, operation=operation)
        if denial:
            result = failure_payload(ServiceError(denial, service_id=service_id, code="permission_denied", category="security"))
            return {**result, "blocks_approval": True, "llm_instruction":
                    "STOP: Do not retry the denied configuration change or pursue it through another tool."}
        if operation in {"enable", "disable"}:
            mutation = await asyncio.to_thread(service.set_model_service_enabled, service_id, operation == "enable")
        elif operation == "set":
            mutation = await asyncio.to_thread(service.set_model_service, service_id, declaration)
        else:
            mutation = await asyncio.to_thread(service.unset_model_service, service_id)
        return await _publish_saved(service, mutation, operation, service_id)
    except ServiceError as exc:
        return failure_payload(exc)
    except (ValueError, TypeError, RecursionError):
        # Avoid echoing potentially credential-bearing malformed declarations.
        return failure_payload(ServiceError("Invalid model service input; use show and the declared bounded schema",
                                            code="model_invalid_configuration", category="validation")) | {"retryable": True}
    except OSError:
        return failure_payload(ServiceError("Configuration was not saved; inspect storage permissions or contention",
                                            code="config_storage_unavailable", recoverable=True))

# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for ToolMetadata handler invocation compatibility."""
from __future__ import annotations

from typing import Any

import pytest

from leapflow.plugins.handler_invocation import ToolHandlerInvocationError, invoke_tool_handler


@pytest.mark.asyncio
async def test_invokes_kwargs_handler_with_empty_arguments() -> None:
    async def handler(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "kwargs": kwargs}

    result = await invoke_tool_handler(handler, {})

    assert result == {"ok": True, "kwargs": {}}


@pytest.mark.asyncio
async def test_invokes_explicit_kwargs_handler() -> None:
    async def handler(message: str = "", **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "message": message, "extra": kwargs}

    result = await invoke_tool_handler(handler, {"message": "hi", "unused": 1})

    assert result == {"ok": True, "message": "hi", "extra": {"unused": 1}}


@pytest.mark.asyncio
async def test_invokes_legacy_params_handler_with_argument_object() -> None:
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "params": params}

    result = await invoke_tool_handler(handler, {"category": "system"})

    assert result == {"ok": True, "params": {"category": "system"}}


@pytest.mark.asyncio
async def test_invokes_legacy_params_handler_with_optional_runner() -> None:
    async def handler(params: dict[str, Any], runner: Any = None) -> dict[str, Any]:
        return {"ok": True, "params": params, "runner": runner}

    result = await invoke_tool_handler(handler, {"query": "status"})

    assert result == {"ok": True, "params": {"query": "status"}, "runner": None}


@pytest.mark.asyncio
async def test_invokes_no_argument_handler_when_payload_is_empty() -> None:
    async def handler() -> dict[str, Any]:
        return {"ok": True}

    result = await invoke_tool_handler(handler, {})

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_rejects_non_mapping_arguments() -> None:
    async def handler(**kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    with pytest.raises(ToolHandlerInvocationError, match="JSON object"):
        await invoke_tool_handler(handler, ["not", "object"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_does_not_mask_internal_type_errors() -> None:
    async def handler(**kwargs: Any) -> dict[str, Any]:
        raise TypeError("real handler bug")

    with pytest.raises(TypeError, match="real handler bug"):
        await invoke_tool_handler(handler, {})


@pytest.mark.asyncio
async def test_invokes_sync_legacy_params_handler() -> None:
    def handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "params": params}

    result = await invoke_tool_handler(handler, {"mode": "sync"})

    assert result == {"ok": True, "params": {"mode": "sync"}}


@pytest.mark.asyncio
async def test_rejects_incompatible_required_signature() -> None:
    async def handler(params: dict[str, Any], runner: Any) -> dict[str, Any]:
        return {"ok": True, "runner": runner}

    with pytest.raises(ToolHandlerInvocationError, match="incompatible"):
        await invoke_tool_handler(handler, {"query": "status"})


class _UsageTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, float]] = []

    def record_tool_call(self, name: str, ok: bool, duration_ms: float) -> None:
        self.calls.append((name, ok, duration_ms))


class _MarkerInterceptor:
    @property
    def name(self) -> str:
        return "test-marker"

    @property
    def priority(self) -> int:
        return 100

    async def before(self, context: Any) -> None:
        return None

    async def after(self, context: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {**result, "intercepted": True}


@pytest.mark.asyncio
async def test_engine_executes_plugin_list_with_empty_native_arguments() -> None:
    import leapflow.engine.engine as engine_module
    import leapflow.plugins as plugins_module
    import leapflow.plugins.tool_plugins as tool_plugins_module
    from leapflow.engine.engine import AgentEngine
    from leapflow.plugins import get_registry

    plugins_module._registry = None
    plugins_module._scoped_registry = None
    tool_plugins_module._all_plugins = None
    engine_module._registry_cache = None
    registry = get_registry()
    registry.assemble()
    engine = AgentEngine.__new__(AgentEngine)
    engine._tool_timeouts = {}
    engine._default_tool_timeout_s = 2.0
    engine._usage_tracker = _UsageTracker()

    result = await engine._execute_general_tool(
        {"name": "plugin_list", "arguments": {}}, registry.tool_handlers
    )

    assert result["ok"] is True
    assert result["capability_report"]["registry"]["tool_count"] >= 1
    assert any(plugin["plugin_id"] == "self_management" for plugin in result["plugins"])


@pytest.mark.asyncio
async def test_engine_executes_handlers_through_interceptor_pipeline() -> None:
    import leapflow.engine.engine as engine_module
    import leapflow.plugins as plugins_module
    import leapflow.plugins.tool_plugins as tool_plugins_module
    from leapflow.engine.engine import AgentEngine
    from leapflow.plugins import get_registry

    plugins_module._registry = None
    plugins_module._scoped_registry = None
    tool_plugins_module._all_plugins = None
    engine_module._registry_cache = None
    registry = get_registry()
    registry.assemble()
    registry.tool_pipeline.register(_MarkerInterceptor())
    engine = AgentEngine.__new__(AgentEngine)
    engine._tool_timeouts = {}
    engine._default_tool_timeout_s = 2.0
    engine._usage_tracker = _UsageTracker()
    try:
        result = await engine._execute_general_tool(
            {"name": "plugin_status", "arguments": {"plugin_id": "self_management"}},
            registry.tool_handlers,
        )
    finally:
        registry.tool_pipeline.unregister("test-marker")

    assert result["ok"] is True
    assert result["plugin_id"] == "self_management"
    assert result["intercepted"] is True
